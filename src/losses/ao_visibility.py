"""Ambient occlusion-based visibility loss for UV seam placement.

Key insight: seams should be placed in occluded (crevice) regions where
they are less visible to the eye. This module computes per-vertex AO
and uses it to weight seam placement decisions.

Based on: Semantic-Visibility-UV-Param (ICLR 2026)
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


def compute_ambient_occlusion(
    vertices: np.ndarray,
    faces: np.ndarray,
    num_rays: int = 32,
    max_dist: float | None = None,
) -> np.ndarray:
    """Compute per-vertex ambient occlusion via ray casting.

    Args:
        vertices: (V, 3) mesh vertices
        faces: (F, 3) face indices
        num_rays: number of random rays per vertex
        max_dist: maximum ray distance (None = auto from bbox)

    Returns:
        (V,) AO values in [0, 1] where 1 = fully occluded (crevice)
    """
    import trimesh

    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)

    if max_dist is None:
        bbox = mesh.bounding_box.extents
        max_dist = float(np.max(bbox)) * 0.5

    V = len(vertices)
    ao = np.zeros(V, dtype=np.float32)

    # Compute face normals and areas for importance sampling
    face_verts = vertices[faces]  # (F, 3, 3)
    e1 = face_verts[:, 1] - face_verts[:, 0]
    e2 = face_verts[:, 2] - face_verts[:, 0]
    cross = np.cross(e1, e2)
    face_areas = np.linalg.norm(cross, axis=1) * 0.5
    face_normals = cross / (np.linalg.norm(cross, axis=1, keepdims=True) + 1e-10)

    total_area = face_areas.sum() + 1e-10
    face_probs = face_areas / total_area

    # For each vertex, cast rays toward random directions weighted by face areas
    for vi in range(V):
        v = vertices[vi]
        # Generate random directions on hemisphere
        theta = np.random.uniform(0, 2 * np.pi, num_rays)
        phi = np.random.uniform(0, np.pi * 0.5, num_rays)
        dirs = np.column_stack([
            np.sin(phi) * np.cos(theta),
            np.sin(phi) * np.sin(theta),
            np.cos(phi),
        ])

        # Shoot rays
        hits = 0
        for d in dirs:
            origin = v + d * 1e-4
            loc, idx, _ = mesh.ray.intersects_location(
                rays_origin=origin.reshape(1, 3),
                rays_direction=d.reshape(1, 3),
            )
            if len(loc) > 0:
                dist = np.linalg.norm(loc[0] - origin)
                if dist < max_dist:
                    hits += 1

        ao[vi] = hits / num_rays

    return ao


def compute_ao_loss_torch(
    points: torch.Tensor,
    ao_values: torch.Tensor,
    seam_logits: torch.Tensor,
) -> torch.Tensor:
    """Compute AO-weighted seam visibility loss.

    Encourages high seam probabilities in high-AO (crevice) regions
    and low seam probabilities in exposed regions.

    Args:
        points: (B, N, 3) surface points
        ao_values: (N,) precomputed AO values [0,1] where 1=occluded
        seam_logits: (B, N, 1) predicted seam probabilities

    Returns:
        Scalar loss tensor
    """
    # seam_probs should be high where AO is high (crevices)
    seam_probs = torch.sigmoid(seam_logits.squeeze(-1))  # (B, N)
    ao = ao_values.to(seam_probs.device)  # (N,)

    # Visibility loss: seam probability should correlate with AO
    # High AO = good seam location = high seam probability
    target = ao.unsqueeze(0).expand_as(seam_probs)  # (B, N)

    # Binary cross-entropy between seam probability and AO target
    loss = nn.functional.binary_cross_entropy(seam_probs, target)

    return loss


class AOVisibilityModule(nn.Module):
    """Compute and cache ambient occlusion for a mesh.

    Precomputes AO values once, then provides differentiable
    visibility loss during training.
    """

    def __init__(self, num_rays: int = 32):
        super().__init__()
        self.num_rays = num_rays
        self._ao_cache: dict[int, torch.Tensor] = {}

    def compute_ao(self, vertices: np.ndarray, faces: np.ndarray) -> torch.Tensor:
        """Compute and cache AO values for a mesh."""
        V = len(vertices)
        cache_key = V  # Simple cache by vertex count

        if cache_key not in self._ao_cache:
            ao_np = compute_ambient_occlusion(
                vertices, faces, num_rays=self.num_rays
            )
            self._ao_cache[cache_key] = torch.tensor(ao_np, dtype=torch.float32)

        return self._ao_cache[cache_key]

    def forward(
        self,
        points: torch.Tensor,
        seam_logits: torch.Tensor,
        ao_values: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute AO visibility loss.

        Args:
            points: (B, N, 3) surface points
            seam_logits: (B, N, 1) predicted seam logits
            ao_values: (N,) precomputed AO, or None to skip

        Returns:
            Scalar loss (0 if ao_values is None)
        """
        if ao_values is None:
            return torch.tensor(0.0, device=points.device)

        return compute_ao_loss_torch(points, ao_values, seam_logits)
