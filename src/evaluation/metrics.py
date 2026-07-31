from __future__ import annotations

"""UV unwrapping quality evaluation metrics."""

import numpy as np
import torch


def angular_distortion(
    vertices: np.ndarray,
    uv_coords: np.ndarray,
    faces: np.ndarray,
) -> dict[str, float]:
    """Compute angular distortion metrics.

    Measures how well angles are preserved between 3D and UV space.

    Returns:
        Dictionary with:
            - mean: mean absolute angular distortion (degrees)
            - max: 95th percentile distortion
            - conformal: conformal quality metric (1.0 = perfect)
    """
    distortions = []

    for face in faces:
        # 3D triangle angles
        v0, v1, v2 = vertices[face[0]], vertices[face[1]], vertices[face[2]]
        angles_3d = _triangle_angles(v0, v1, v2)

        # UV triangle angles
        u0, u1, u2 = uv_coords[face[0]], uv_coords[face[1]], uv_coords[face[2]]
        angles_2d = _triangle_angles(
            np.append(u0, 0), np.append(u1, 0), np.append(u2, 0)
        )

        # Angular distortion
        diff = np.abs(angles_3d - angles_2d)
        distortions.extend(np.degrees(diff))

    distortions = np.array(distortions)
    return {
        "mean": float(np.mean(distortions)),
        "max": float(np.percentile(distortions, 95)),
        "conformal": float(1.0 - np.mean(distortions) / 180.0),
    }


def area_distortion(
    vertices: np.ndarray,
    uv_coords: np.ndarray,
    faces: np.ndarray,
) -> dict[str, float]:
    """Compute area distortion metrics.

    Measures how well relative areas are preserved.

    Returns:
        Dictionary with:
            - mean_log: mean log area ratio
            - max: 95th percentile area ratio
            - uniformity: area uniformity metric (1.0 = perfect)
    """
    ratios = []

    for face in faces:
        v0, v1, v2 = vertices[face[0]], vertices[face[1]], vertices[face[2]]
        area_3d = _triangle_area(v0, v1, v2)

        u0, u1, u2 = uv_coords[face[0]], uv_coords[face[1]], uv_coords[face[2]]
        area_2d = _triangle_area(
            np.append(u0, 0), np.append(u1, 0), np.append(u2, 0)
        )

        if area_3d > 1e-10:
            ratios.append(area_2d / area_3d)

    ratios = np.array(ratios)
    log_ratios = np.log(ratios + 1e-8)

    return {
        "mean_log": float(np.mean(np.abs(log_ratios))),
        "max": float(np.percentile(ratios, 95)),
        "uniformity": float(1.0 - np.std(log_ratios)),
    }


def seam_length(
    uv_coords: np.ndarray,
    edges: np.ndarray,
    threshold: float = 0.05,
) -> dict[str, float]:
    """Measure UV seam length.

    Detects seams (edges where UV coordinates are discontinuous)
    and computes their total length.

    Returns:
        Dictionary with:
            - total: total seam length
            - ratio: seam length / total edge length
            - num_seams: number of seam edges
    """
    uv0 = uv_coords[edges[:, 0]]
    uv1 = uv_coords[edges[:, 1]]
    uv_dist = np.linalg.norm(uv1 - uv0, axis=-1)

    # Also compute 3D edge lengths for ratio (need vertices too)
    seam_mask = uv_dist > threshold
    num_seams = int(seam_mask.sum())

    return {
        "total": float(uv_dist[seam_mask].sum()) if num_seams > 0 else 0.0,
        "ratio": float(seam_mask.mean()),
        "num_seams": num_seams,
    }


def chart_count(
    uv_coords: np.ndarray,
    faces: np.ndarray,
    threshold: float = 0.05,
) -> int:
    """Count number of disconnected UV charts."""
    # Build adjacency from UV discontinuities
    n_verts = len(uv_coords)
    parent = list(range(n_verts))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for face in faces:
        for i in range(3):
            for j in range(i + 1, 3):
                vi, vj = face[i], face[j]
                uv_dist = np.linalg.norm(uv_coords[vi] - uv_coords[vj])
                if uv_dist < threshold:
                    union(vi, vj)

    return len(set(find(i) for i in range(n_verts)))


def _triangle_angles(v0, v1, v2):
    """Compute angles of a triangle (in radians)."""
    e0 = v1 - v0
    e1 = v2 - v0
    e2 = v2 - v1

    def angle(a, b):
        cos_a = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10)
        return np.arccos(np.clip(cos_a, -1, 1))

    return np.array([angle(e0, e1), angle(-e0, e2), angle(-e1, -e2)])


def _triangle_area(v0, v1, v2):
    """Compute area of a triangle."""
    return 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0))
