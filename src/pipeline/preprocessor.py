from __future__ import annotations

"""Mesh preprocessing — cleaning, normalization, and point sampling."""

from pathlib import Path

import torch
import trimesh

from ..data.mesh_io import load_mesh, normalize_mesh, mesh_to_tensors, sample_points


class MeshPreprocessor:
    """Preprocess meshes for ML UV unwrapping.

    Handles mesh cleaning, normalization, point sampling, and
    conversion to tensor format suitable for the neural network.
    """

    def __init__(
        self,
        num_points: int = 10000,
        sample_method: str = "face_area",
        normalize: bool = True,
        device: str = "cpu",
    ):
        self.num_points = num_points
        self.sample_method = sample_method
        self.normalize = normalize
        self.device = device

    def process(self, mesh_input: str | Path | trimesh.Trimesh) -> dict:
        """Full preprocessing pipeline.

        Returns:
            Dictionary with:
                - points: (1, N, 3) tensor
                - face_idx: (N,) tensor of face indices per sampled point
                - normals: (1, N, 3) tensor (if available)
                - vertices: (V, 3) tensor
                - faces: (F, 3) tensor
                - edges: (E, 2) tensor
                - mesh: original trimesh object
        """
        # Load
        if isinstance(mesh_input, (str, Path)):
            mesh = load_mesh(mesh_input)
        else:
            mesh = mesh_input

        # Clean
        mesh = self._clean(mesh)

        # Normalize
        if self.normalize:
            mesh = normalize_mesh(mesh)

        # Sample points
        points, normals, face_idx = sample_points(mesh, self.num_points, self.sample_method)
        points = points.unsqueeze(0).to(self.device)  # (1, N, 3)
        face_idx = face_idx.to(self.device)  # (N,)

        # Convert to tensors
        tensors = mesh_to_tensors(mesh)

        result = {
            "points": points,
            "face_idx": face_idx,
            "vertices": tensors["vertices"].unsqueeze(0).to(self.device),
            "faces": tensors["faces"].to(self.device),
            "edges": tensors["edges"].to(self.device),
            "mesh": mesh,
        }
        if normals is not None:
            result["normals"] = normals.unsqueeze(0).to(self.device)

        return result

    def _clean(self, mesh: trimesh.Trimesh) -> trimesh.Trimesh:
        """Basic mesh cleaning."""
        # Remove degenerate faces (zero area)
        mesh.update_faces(mesh.area_faces > 1e-10)

        # Remove unreferenced vertices
        mesh.remove_unreferenced_vertices()

        # Fix normals
        mesh.fix_normals()

        return mesh
