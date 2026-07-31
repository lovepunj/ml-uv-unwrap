from __future__ import annotations

"""Visibility-aware seam loss based on ambient occlusion.

Implements the semantic/visibility objectives from:
"Unsupervised Representation Learning for 3D Mesh Parameterization with
Semantic and Visibility Objectives" (ICLR 2026).

The core idea: UV seams should be placed in occluded (interior) regions of
the surface so that visible seam artifacts are minimized.  We detect soft
seam membership via a differentiable log-sum-exp proxy for max UV distance
in a 3D neighbourhood, then weight per-vertex ambient occlusion by seam
membership to produce the boundary occlusion loss.
"""

import torch
import numpy as np


# ---------------------------------------------------------------------------
# Ambient Occlusion
# ---------------------------------------------------------------------------

def compute_vertex_ambient_occlusion(
    vertices: np.ndarray,
    faces: np.ndarray,
    num_rays: int = 400,
) -> torch.Tensor:
    """Compute per-vertex ambient occlusion using libigl's embree backend.

    Uses ``igl.embree.ray_mesh_intersections`` with
    ``num_rays`` random directions per vertex.  Returns values in [0, 1]
    where 1 = fully occluded, 0 = fully exposed.

    .. math::
        \\text{AO}(v) = \\frac{1}{N} \\sum_{k=1}^{N} \\mathbb{1}[\\text{ray}_k
        \\text{ hits surface before } t_{\\max}]

    Args:
        vertices: (V, 3) vertex positions as a NumPy array.
        faces: (F, 3) triangle face indices.
        num_rays: Number of random rays to cast per vertex.

    Returns:
        (V,) tensor of AO values in [0, 1], wrapped with ``requires_grad``
        set to ``False`` (AO is a geometric signal, not learned).
    """
    try:
        import igl
        import igl.embree
    except ImportError:
        raise ImportError(
            "pyigl and igl.embree are required for AO computation.  "
            "Install with: pip install libigl"
        )

    V = vertices.shape[0]
    rng = np.random.default_rng(42)

    # Random unit directions on the hemisphere w.r.t. vertex normals
    normals = igl.per_vertex_normals(vertices, faces)
    ao = np.zeros(V, dtype=np.float32)

    for v_idx in range(V):
        n = normals[v_idx]
        # Build an orthonormal basis (n, t, b)
        t = np.cross(n, np.array([0.0, 0.0, 1.0]))
        if np.linalg.norm(t) < 1e-6:
            t = np.cross(n, np.array([0.0, 1.0, 0.0]))
        t = t / np.linalg.norm(t)
        b = np.cross(n, t)

        # Sample random directions on the unit hemisphere
        theta = rng.uniform(0, 2 * np.pi, size=num_rays)
        phi = np.arccos(rng.uniform(0, 1, size=num_rays))
        dirs = (
            np.outer(np.cos(theta) * np.sin(phi), t)
            + np.outer(np.sin(theta) * np.sin(phi), b)
            + np.outer(np.cos(phi), n)
        )  # (num_rays, 3)

        # Cast rays from the vertex
        origin = vertices[v_idx]
        max_dist = igl.average_edge_length(vertices, faces) * 2.0
        hits = igl.embree.ray_mesh_intersections(
            origin, dirs, vertices, faces, max_dist
        )
        ao[v_idx] = float(hits) / num_rays

    return torch.tensor(ao, dtype=torch.float32)


# ---------------------------------------------------------------------------
# Vertex neighbourhood helpers
# ---------------------------------------------------------------------------

def build_vertex_neighbors(
    faces: torch.Tensor,
    num_vertices: int,
) -> list[list[int]]:
    """Build 1-ring adjacency lists for every vertex.

    Two vertices are neighbours when they share at least one triangle face.
    Returns a list-of-lists for easy iteration; for batched operations see
    :func:`compute_eta_with_Jcut` which converts to padded tensors
    internally.

    Args:
        faces: (F, 3) triangle face indices.
        num_vertices: Total number of vertices ``V``.

    Returns:
        List of ``V`` sets, where ``nbrs[i]`` is the set of vertex indices
        that share an edge with vertex ``i`` (excluding ``i`` itself).
    """
    neighbors: list[set[int]] = [set() for _ in range(num_vertices)]
    faces_np = faces.detach().cpu().numpy()
    for f in faces_np:
        for a, b in ((0, 1), (1, 2), (2, 0)):
            neighbors[f[a]].add(f[b])
            neighbors[f[b]].add(f[a])
    return [sorted(s) for s in neighbors]


# ---------------------------------------------------------------------------
# Differentiable seam detection
# ---------------------------------------------------------------------------

def _logsumexp_max_uv_distance(
    uv_i: torch.Tensor,
    uv_neighbors: torch.Tensor,
    gamma: float,
) -> torch.Tensor:
    """Soft-max UV distance via log-sum-exp.

    .. math::
        \\eta_i = \\frac{1}{\\gamma} \\log \\sum_j
        \\exp\\bigl(\\gamma \\, \\|q_i - q_j\\|_2\\bigr)

    As ``gamma → ∞`` this converges to ``max_j ||q_i - q_j||_2``.

    Args:
        uv_i: (*, 2) UV coords of the vertex itself.
        uv_neighbors: (*, K, 2) UV coords of the *J_cut* nearest 3D
            neighbours.
        gamma: Temperature parameter controlling sharpness.

    Returns:
        (*) Scalar (soft-max) UV distance for each vertex.
    """
    # ||q_i - q_j||_2 for every neighbour
    diffs = uv_neighbors - uv_i.unsqueeze(-2)  # (*, K, 2)
    dists = diffs.norm(dim=-1)  # (*, K)
    return (1.0 / gamma) * torch.logsumexp(gamma * dists, dim=-1)


def compute_eta_with_Jcut(
    faces: torch.Tensor,
    uv_coords: torch.Tensor,
    vertices: torch.Tensor,
    J_cut: int = 5,
    gamma: float = 10.0,
) -> torch.Tensor:
    """Differentiable soft-seam indicator η per vertex.

    For each vertex *i*, take the *J_cut* closest 3D neighbours (by
    Euclidean distance) and compute the log-sum-exp proxy for the **maximum**
    UV-space distance among those neighbours.

    .. math::
        \\eta_i = \\frac{1}{\\gamma}
        \\log \\sum_{j \\in \\mathcal{N}_{J}(i)}
        \\exp\\!\\bigl(\\gamma \\, \\|q_i - q_j\\|_2\\bigr)

    Vertices on a UV seam have large η (their UV neighbours are far apart
    in 2D even though they are close in 3D); interior vertices have small η.

    The entire computation is implemented with ``torch`` ops so gradients
    flow through η back to the UV coordinates.

    Args:
        faces: (F, 3) triangle indices (long tensor).
        uv_coords: (V, 2) current UV coordinates (differentiable).
        vertices: (V, 3) 3D vertex positions.
        J_cut: Number of nearest 3D neighbours to consider per vertex.
        gamma: Log-sum-exp temperature.  Higher ⇒ closer to true max.

    Returns:
        (V,) tensor of η values.
    """
    V = uv_coords.shape[0]
    J_cut = min(J_cut, V - 1)

    # Full pairwise 3D distance matrix (batched-friendly for moderate meshes)
    # vertices: (V, 3) → (V, 1, 3) - (1, V, 3) → (V, V, 3) → norm → (V, V)
    diff_3d = vertices.unsqueeze(1) - vertices.unsqueeze(0)  # (V, V, 3)
    dist_3d = diff_3d.norm(dim=-1)  # (V, V)

    # Mask self-distance
    eye_mask = torch.eye(V, dtype=torch.bool, device=vertices.device)
    dist_3d = dist_3d.masked_fill(eye_mask, float("inf"))

    # J_cut nearest neighbours per vertex
    nn_idx = dist_3d.topk(J_cut, dim=-1, largest=False).indices  # (V, J_cut)

    # Gather UV coords of neighbours
    uv_neighbors = uv_coords[nn_idx]  # (V, J_cut, 2)

    # Soft-max max UV distance
    eta = _logsumexp_max_uv_distance(uv_coords, uv_neighbors, gamma)  # (V,)
    return eta


# ---------------------------------------------------------------------------
# Differentiable seam membership
# ---------------------------------------------------------------------------

def find_uv_seam(
    faces: torch.Tensor,
    vertices: torch.Tensor,
    uv_coords: torch.Tensor,
    J_cut: int = 5,
    beta: float = 50.0,
    gamma: float = 10.0,
) -> torch.Tensor:
    """Soft binary seam membership via sigmoid thresholding.

    Combines η computation with a sigmoid activation centred at the
    adaptive threshold τ:

    .. math::
        \\tau &= 0.1 \\cdot L(\\mathbf{Q}) \\\\
        s_v &= \\sigma\\bigl(\\beta \\cdot (\\eta_v - \\tau)\\bigr)

    where :math:`L(\\mathbf{Q})` is the side length of the axis-aligned
    UV bounding square, and :math:`\\sigma` is the logistic sigmoid.

    Vertices with η ≫ τ receive s_v → 1 (seam); vertices with η ≪ τ
    receive s_v → 0 (interior).

    Args:
        faces: (F, 3) triangle face indices.
        vertices: (V, 3) 3D vertex positions.
        uv_coords: (V, 2) current UV coordinates (differentiable).
        J_cut: Number of 3D neighbours for the η proxy.
        beta: Sigmoid sharpness (higher ⇒ closer to hard threshold).
        gamma: Log-sum-exp temperature for η.

    Returns:
        (V,) tensor of soft seam weights in approximately (0, 1).
    """
    eta = compute_eta_with_Jcut(faces, uv_coords, vertices, J_cut, gamma)

    # Adaptive threshold: 10 % of the UV bounding square side length
    uv_min = uv_coords.min(dim=0).values  # (2,)
    uv_max = uv_coords.max(dim=0).values  # (2,)
    side = (uv_max - uv_min).max()  # scalar – axis-aligned bounding square
    tau = 0.1 * side

    # Soft seam membership via sigmoid
    seam_weights = torch.sigmoid(beta * (eta - tau))
    return seam_weights


# ---------------------------------------------------------------------------
# Boundary / seam ambient occlusion loss
# ---------------------------------------------------------------------------

def boundary_occlusion_loss(
    ao_values: torch.Tensor,
    seam_weights: torch.Tensor,
) -> torch.Tensor:
    """Weighted ambient occlusion loss at UV seam vertices.

    .. math::
        L_{\\text{bound}} =
        \\frac{\\sum_v s_v \\cdot \\text{AO}_v}{\\sum_v s_v}

    High seam-vertex AO is **desirable** (seams are hidden in occluded
    regions), so the loss penalises seams with *low* AO.  The seam weights
    are used as a soft attention mask so that interior vertices contribute
    negligibly.

    Args:
        ao_values: (V,) per-vertex ambient occlusion in [0, 1].  Typically
            produced by :func:`compute_vertex_ambient_occlusion`.
        seam_weights: (V,) soft seam membership in ≈ (0, 1), typically
            from :func:`find_uv_seam`.

    Returns:
        Scalar loss (differentiable w.r.t. ``seam_weights``).
    """
    # Smooth numerator / denominator to avoid division by zero and to keep
    # the gradient well-defined when the denominator is tiny.
    eps = 1e-7
    numerator = (seam_weights * ao_values).sum()
    denominator = seam_weights.sum() + eps
    return numerator / denominator
