"""GaussianWrapping-inspired normal-aware seam placement.

Uses oriented Gaussian normal fields for intelligent seam placement.
Key insight from "From Blobs to Spokes" (ECCV 2026): seams should be
placed along high-curvature regions where the surface normal changes
rapidly, making them less visible.

This module computes:
1. Per-vertex curvature from normal field variation
2. Normal-aligned seam scores (high curvature = good seam location)
3. Edge-based seam candidate selection using normal discontinuity
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_vertex_curvature(
    vertices: np.ndarray,
    faces: np.ndarray,
) -> np.ndarray:
    """Compute per-vertex curvature from normal field variation.

    Measures how much the surface normal changes across neighboring
    faces — high variation = crease/edge = good seam location.

    Args:
        vertices: (V, 3) mesh vertices
        faces: (F, 3) face indices

    Returns:
        (V,) curvature values in [0, inf)
    """
    V = len(vertices)
    num_faces = len(faces)

    # Compute face normals
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    e1 = v1 - v0
    e2 = v2 - v0
    cross = np.cross(e1, e2)
    norms = np.linalg.norm(cross, axis=1, keepdims=True) + 1e-10
    face_normals = cross / norms

    # Compute face areas
    face_areas = norms.squeeze() / 2.0

    # Accumulate area-weighted normals per vertex
    vertex_normals = np.zeros((V, 3), dtype=np.float64)
    vertex_areas = np.zeros(V, dtype=np.float64)

    for i in range(3):
        np.add.at(vertex_normals, faces[:, i], face_normals * face_areas[:, None])
        np.add.at(vertex_areas, faces[:, i], face_areas)

    vertex_areas = np.maximum(vertex_areas, 1e-10)
    vertex_normals = vertex_normals / vertex_areas[:, None]

    # Compute curvature as normal variation across faces
    # For each vertex, compute mean angular difference with adjacent face normals
    curvature = np.zeros(V, dtype=np.float64)

    for i in range(3):
        face_idx = faces[:, i]
        # Dot product between vertex normal and face normal
        dot = np.sum(vertex_normals[face_idx] * face_normals, axis=1)
        dot = np.clip(dot, -1.0, 1.0)
        angular_diff = np.arccos(dot)
        np.add.at(curvature, face_idx, angular_diff * face_areas)

    curvature = curvature / np.maximum(vertex_areas, 1e-10)

    # Normalize to [0, 1]
    if curvature.max() > 0:
        curvature = curvature / curvature.max()

    return curvature.astype(np.float32)


def compute_normal_discontinuity(
    vertices: np.ndarray,
    faces: np.ndarray,
) -> np.ndarray:
    """Compute per-edge normal discontinuity.

    High discontinuity across an edge = sharp feature = good seam.

    Args:
        vertices: (V, 3) mesh vertices
        faces: (F, 3) face indices

    Returns:
        dict with 'edge_pairs' (E, 2) and 'discontinuity' (E,) scores
    """
    import trimesh

    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)

    if not hasattr(mesh, 'face_adjacency') or len(mesh.face_adjacency) == 0:
        return {"edge_pairs": np.array([], dtype=np.int64).reshape(0, 2),
                "discontinuity": np.array([], dtype=np.float32)}

    adj = mesh.face_adjacency
    edges = mesh.face_adjacency_edges

    # Compute face normals
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    e1 = v1 - v0
    e2 = v2 - v0
    cross = np.cross(e1, e2)
    norms = np.linalg.norm(cross, axis=1, keepdims=True) + 1e-10
    face_normals = cross / norms

    # Normal discontinuity across adjacent faces
    n1 = face_normals[adj[:, 0]]
    n2 = face_normals[adj[:, 1]]
    dot = np.sum(n1 * n2, axis=1)
    dot = np.clip(dot, -1.0, 1.0)
    discontinuity = 1.0 - dot  # 0 = smooth, 1 = sharp

    return {
        "edge_pairs": edges,
        "discontinuity": discontinuity.astype(np.float32),
    }


class NormalAwareSeamScorer(nn.Module):
    """Learn seam scores from normal field and curvature.

    Combines geometric cues (curvature, normal discontinuity) with
    a learned scoring network to predict per-edge seam likelihood.
    """

    def __init__(self, hidden_dim: int = 64):
        super().__init__()
        # Input: edge features (4D: 2 endpoint curvatures + normal discontinuity + edge length)
        self.scorer = nn.Sequential(
            nn.Linear(4, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        edge_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            edge_features: (E, 4) per-edge features

        Returns:
            (E, 1) seam probability scores
        """
        return self.scorer(edge_features)


def compute_seam_candidates(
    vertices: np.ndarray,
    faces: np.ndarray,
    top_k: int = 50,
) -> list[tuple[int, int]]:
    """Compute best seam candidates using normal-aware scoring.

    Combines:
    1. High curvature vertices (creases)
    2. Normal discontinuity across edges
    3. Edge length (prefer shorter seams)

    Args:
        vertices: (V, 3) mesh vertices
        faces: (F, 3) face indices
        top_k: number of top seam candidates to return

    Returns:
        List of (v0, v1) vertex index pairs
    """
    curvature = compute_vertex_curvature(vertices, faces)
    nd = compute_normal_discontinuity(vertices, faces)

    if len(nd["edge_pairs"]) == 0:
        return []

    edges = nd["edge_pairs"]
    disc = nd["discontinuity"]

    # Per-edge score: high normal discontinuity + high endpoint curvature
    score = disc.copy()
    for i in range(2):
        endpoint_curv = curvature[edges[:, i]]
        score = score + endpoint_curv * 0.3

    # Prefer shorter edges (less visible seams)
    edge_lengths = np.linalg.norm(
        vertices[edges[:, 0]] - vertices[edges[:, 1]], axis=1
    )
    max_len = edge_lengths.max() + 1e-10
    score = score + (1.0 - edge_lengths / max_len) * 0.2

    # Top-k by score
    top_indices = np.argsort(score)[::-1][:top_k]
    candidates = [tuple(edges[i]) for i in top_indices]

    return candidates


class GaussianNormalField(nn.Module):
    """Oriented Gaussian normal field for mesh analysis.

    Wraps the normal field computation from GaussianWrapping
    for use in UV seam placement. Computes:
    - Per-vertex Gaussian curvature
    - Normal field variation
    - Feature vectors for seam scoring
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        vertices: torch.Tensor,
        faces: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Compute normal field features.

        Args:
            vertices: (V, 3) mesh vertices
            faces: (F, 3) face indices

        Returns:
            Dictionary with curvature, normal_features, edge_scores
        """
        V = vertices.shape[0]

        # Compute face normals
        v0 = vertices[faces[:, 0]]
        v1 = vertices[faces[:, 1]]
        v2 = vertices[faces[:, 2]]
        e1 = v1 - v0
        e2 = v2 - v0
        face_normals = F.normalize(torch.cross(e1, e2, dim=-1), dim=-1)
        face_areas = torch.norm(torch.cross(e1, e2, dim=-1), dim=-1) * 0.5

        # Accumulate per-vertex normals (area-weighted)
        vertex_normals = torch.zeros(V, 3, device=vertices.device)
        vertex_areas = torch.zeros(V, device=vertices.device)

        for i in range(3):
            idx = faces[:, i]
            vertex_normals.index_add_(0, idx, face_normals * face_areas.unsqueeze(-1))
            vertex_areas.index_add_(0, idx, face_areas)

        vertex_areas = vertex_areas.clamp(min=1e-10)
        vertex_normals = vertex_normals / vertex_areas.unsqueeze(-1)

        # Gaussian curvature: measure normal variation per vertex
        curvature = torch.zeros(V, device=vertices.device)
        for i in range(3):
            idx = faces[:, i]
            # Angular difference between vertex normal and face normal
            dot = (vertex_normals[idx] * face_normals).sum(dim=-1)
            dot = dot.clamp(-1, 1)
            angular = torch.acos(dot)
            curvature.index_add_(0, idx, angular * face_areas)

        curvature = curvature / vertex_areas
        curvature = curvature / (curvature.max() + 1e-10)

        return {
            "curvature": curvature,
            "vertex_normals": vertex_normals,
            "face_normals": face_normals,
        }
