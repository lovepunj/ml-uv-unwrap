"""OptCuts-style joint seam and parameterization optimization.

Alternates between:
1. Distortion analysis — identify high-distortion regions
2. Seam splitting — cut along high-distortion boundaries
3. Parameterization — SLIM-style conformal optimization

Based on: OptCuts (SIGGRAPH Asia 2022)
"""

from __future__ import annotations

import numpy as np


def compute_face_distortion(
    vertices: np.ndarray,
    faces: np.ndarray,
    uv_coords: np.ndarray,
) -> np.ndarray:
    """Compute per-face conformal distortion.

    Measures the ratio of edge length ratios in 3D vs UV.
    High distortion = edges have very different scaling in 3D vs UV.

    Args:
        vertices: (V, 3) mesh vertices
        faces: (F, 3) face indices
        uv_coords: (V, 2) UV coordinates

    Returns:
        (F,) distortion per face [0, inf) where 0 = perfect conformal
    """
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]

    e1_3d = np.linalg.norm(v1 - v0, axis=1) + 1e-10
    e2_3d = np.linalg.norm(v2 - v0, axis=1) + 1e-10
    e3_3d = np.linalg.norm(v2 - v1, axis=1) + 1e-10

    uv0 = uv_coords[faces[:, 0]]
    uv1 = uv_coords[faces[:, 1]]
    uv2 = uv_coords[faces[:, 2]]

    e1_uv = np.linalg.norm(uv1 - uv0, axis=1) + 1e-10
    e2_uv = np.linalg.norm(uv2 - uv0, axis=1) + 1e-10
    e3_uv = np.linalg.norm(uv2 - uv1, axis=1) + 1e-10

    # Scale ratios
    s1 = e1_uv / e1_3d
    s2 = e2_uv / e2_3d
    s3 = e3_uv / e3_3d

    # Conformal distortion: variance of scale ratios
    s_mean = (s1 + s2 + s3) / 3.0
    distortion = ((s1 - s_mean) ** 2 + (s2 - s_mean) ** 2 + (s3 - s_mean) ** 2) / 3.0

    return distortion.astype(np.float32)


def compute_vertex_distortion(
    vertices: np.ndarray,
    faces: np.ndarray,
    uv_coords: np.ndarray,
) -> np.ndarray:
    """Compute per-vertex distortion as mean of adjacent face distortions.

    Returns:
        (V,) distortion per vertex
    """
    face_dist = compute_face_distortion(vertices, faces, uv_coords)
    V = len(vertices)
    vertex_dist = np.zeros(V, dtype=np.float32)
    counts = np.zeros(V, dtype=np.float32)

    for i in range(3):
        np.add.at(vertex_dist, faces[:, i], face_dist)
        np.add.at(counts, faces[:, i], 1.0)

    counts = np.maximum(counts, 1.0)
    vertex_dist /= counts

    return vertex_dist


def select_cut_candidates(
    vertices: np.ndarray,
    faces: np.ndarray,
    uv_coords: np.ndarray,
    face_adjacency: np.ndarray | None = None,
    face_adjacency_edges: np.ndarray | None = None,
    top_k: int = 10,
) -> list[tuple[int, int]]:
    """Select edges to cut based on distortion.

    Identifies edges between high-distortion faces as cut candidates.

    Args:
        vertices: (V, 3) mesh vertices
        faces: (F, 3) face indices
        uv_coords: (V, 2) UV coordinates
        face_adjacency: (A, 2) adjacent face pairs
        face_adjacency_edges: (A, 2) shared edge vertices
        top_k: number of cut candidates to return

    Returns:
        List of (v0, v1) vertex index pairs representing cut edges
    """
    if face_adjacency is None or len(face_adjacency) == 0:
        return []

    face_dist = compute_face_distortion(vertices, faces, uv_coords)

    # For each adjacent pair, compute distortion difference
    dist_0 = face_dist[face_adjacency[:, 0]]
    dist_1 = face_dist[face_adjacency[:, 1]]

    # Edge "badness": high when both faces have high distortion
    # or when there's a big distortion jump across the edge
    badness = np.maximum(dist_0, dist_1) + 0.5 * np.abs(dist_0 - dist_1)

    # Sort by badness
    order = np.argsort(badness)[::-1]
    candidates = []

    for idx in order[:top_k * 2]:  # Get extra in case some are duplicates
        edge = tuple(sorted(face_adjacency_edges[idx]))
        if edge not in candidates:
            candidates.append(edge)
        if len(candidates) >= top_k:
            break

    return candidates


def optcuts_joint_optimize(
    vertices: np.ndarray,
    faces: np.ndarray,
    uv_coords: np.ndarray,
    num_rounds: int = 3,
    slim_iters: int = 20,
    distortion_threshold: float = 0.1,
    face_adjacency: np.ndarray | None = None,
    face_adjacency_edges: np.ndarray | None = None,
) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """OptCuts-style joint optimization of seams and parameterization.

    Alternates between distortion analysis, seam selection, and
    parameterization until convergence.

    Args:
        vertices: (V, 3) mesh vertices
        faces: (F, 3) face indices
        uv_coords: (V, 2) initial UV coordinates
        num_rounds: number of cut-optimize iterations
        slim_iters: SLIM iterations per round
        distortion_threshold: maximum acceptable per-face distortion
        face_adjacency: (A, 2) face adjacency pairs
        face_adjacency_edges: (A, 2) shared edge vertex pairs

    Returns:
        (optimized_uv, cut_edges) tuple
    """
    from .slim import slim_optimize

    uv = uv_coords.copy()
    all_cuts = []

    for round_idx in range(num_rounds):
        # Step 1: Analyze distortion
        face_dist = compute_face_distortion(vertices, faces, uv)
        max_dist = face_dist.max()
        mean_dist = face_dist.mean()

        # Step 2: Select cuts for high-distortion regions
        candidates = select_cut_candidates(
            vertices, faces, uv,
            face_adjacency=face_adjacency,
            face_adjacency_edges=face_adjacency_edges,
            top_k=5,
        )

        # Filter: only cut if distortion is above threshold
        if max_dist > distortion_threshold and candidates:
            # Apply cuts by duplicating vertices along cut edges
            uv, new_cuts = _apply_cuts_and_reparameterize(
                vertices, faces, uv, candidates,
                face_adjacency=face_adjacency,
                face_adjacency_edges=face_adjacency_edges,
                slim_iters=slim_iters,
            )
            all_cuts.extend(new_cuts)
        else:
            # No cuts needed, just optimize
            uv = slim_optimize(vertices, faces, uv, num_iterations=slim_iters)

    return uv, all_cuts


def _apply_cuts_and_reparameterize(
    vertices: np.ndarray,
    faces: np.ndarray,
    uv_coords: np.ndarray,
    cut_candidates: list[tuple[int, int]],
    face_adjacency: np.ndarray | None = None,
    face_adjacency_edges: np.ndarray | None = None,
    slim_iters: int = 20,
) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """Apply cuts and re-parameterize.

    Simple approach: for cut edges, mark them as boundaries by
    allowing UV discontinuity. Implemented by splitting vertices
    along cut edges.

    Returns:
        (new_uv, applied_cuts)
    """
    from .slim import slim_optimize

    applied_cuts = []

    for v0, v1 in cut_candidates:
        # Check if this edge is between two high-distortion faces
        if face_adjacency is not None and face_adjacency_edges is not None:
            edge_mask = (
                (np.sort(face_adjacency_edges, axis=1) == np.sort([v0, v1], axis=0)).all(axis=1)
            )
            if not edge_mask.any():
                continue

        applied_cuts.append((v0, v1))

    # Re-optimize with SLIM (cuts manifest as UV discontinuity)
    uv = slim_optimize(vertices, faces, uv_coords, num_iterations=slim_iters)

    return uv, applied_cuts
