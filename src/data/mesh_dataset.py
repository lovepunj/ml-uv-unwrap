from __future__ import annotations

"""Dataset for loading 3D meshes for UV unwrapping training."""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .mesh_io import load_mesh, normalize_mesh, sample_points


class MeshDataset(Dataset):
    """Dataset of 3D meshes for unsupervised UV unwrapping training.

    Loads meshes from a directory, samples surface points,
    and optionally applies augmentations.
    """

    SUPPORTED_EXTS = {".obj", ".ply", ".stl", ".off", ".glb", ".gltf", ".fbx"}

    def __init__(
        self,
        root_dir: str | Path,
        num_points: int = 10000,
        sample_method: str = "face_area",
        normalize: bool = True,
        max_faces: int | None = 50000,
        augment: bool = False,
    ):
        self.root_dir = Path(root_dir)
        self.num_points = num_points
        self.sample_method = sample_method
        self.normalize = normalize
        self.max_faces = max_faces
        self.augment = augment

        self.mesh_paths = self._scan_meshes()

    def _scan_meshes(self) -> list[Path]:
        """Recursively find all mesh files."""
        paths = []
        for ext in self.SUPPORTED_EXTS:
            paths.extend(self.root_dir.rglob(f"*{ext}"))
        paths.sort()
        return paths

    def __len__(self) -> int:
        return len(self.mesh_paths)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        mesh_path = self.mesh_paths[idx]

        try:
            mesh = load_mesh(mesh_path)
        except Exception as e:
            # Return a dummy mesh on failure
            return self._dummy_sample()

        # Skip meshes with too many faces (memory)
        if self.max_faces and len(mesh.faces) > self.max_faces:
            return self._dummy_sample()

        # Normalize
        if self.normalize:
            mesh = normalize_mesh(mesh)

        # Sample points
        points, normals, face_idx = sample_points(mesh, self.num_points, self.sample_method)

        # Augment
        if self.augment:
            points, normals = self._augment(points, normals)

        sample = {
            "points": points,
            "path": str(mesh_path),
        }
        if normals is not None:
            sample["normals"] = normals

        return sample

    def _augment(
        self,
        points: torch.Tensor,
        normals: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Apply random augmentations."""
        # Random rotation
        if np.random.random() > 0.5:
            points, normals = _random_rotate(points, normals)

        # Small random noise
        if np.random.random() > 0.7:
            noise = torch.randn_like(points) * 0.01
            points = points + noise

        return points, normals

    def _dummy_sample(self) -> dict[str, torch.Tensor]:
        """Return a unit sphere as fallback."""
        points = _unit_sphere_points(self.num_points)
        return {
            "points": points,
            "path": "dummy",
            "normals": points,  # sphere normals = points
        }


def _unit_sphere_points(n: int) -> torch.Tensor:
    """Generate n approximately uniform points on a unit sphere."""
    indices = torch.arange(0, n, dtype=torch.float32)
    phi = torch.acos(1 - 2 * (indices + 0.5) / n)
    theta = torch.pi * (1 + 5**0.5) * indices
    x = torch.sin(phi) * torch.cos(theta)
    y = torch.sin(phi) * torch.sin(theta)
    z = torch.cos(phi)
    return torch.stack([x, y, z], dim=-1)


def _random_rotate(
    points: torch.Tensor,
    normals: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Apply a random rotation to points and normals."""
    # Random rotation matrix via QR decomposition
    A = torch.randn(3, 3)
    Q, R = torch.linalg.qr(A)
    # Ensure proper rotation (det = +1)
    if torch.det(Q) < 0:
        Q[:, 0] = -Q[:, 0]
    Q = Q.to(points.device, points.dtype)

    points = points @ Q.T
    if normals is not None:
        normals = normals @ Q.T
    return points, normals
