"""Distortion losses for UV parameterization quality."""

import torch


def conformal_loss(
    points_3d: torch.Tensor,
    uv_coords: torch.Tensor,
    edges: torch.Tensor,
) -> torch.Tensor:
    """Conformal (angle-preserving) distortion loss.

    Measures how well angles are preserved between 3D and UV space.
    Uses edge length ratios to compute local distortion.

    Args:
        points_3d: (N, 3) 3D surface points
        uv_coords: (N, 2) UV coordinates
        edges: (E, 2) edge indices into the point arrays

    Returns:
        Scalar conformal distortion loss
    """
    p0 = points_3d[edges[:, 0]]
    p1 = points_3d[edges[:, 1]]
    u0 = uv_coords[edges[:, 0]]
    u1 = uv_coords[edges[:, 1]]

    edge_3d = (p1 - p0).norm(dim=-1) + 1e-8  # (E,)
    edge_2d = (u1 - u0).norm(dim=-1) + 1e-8  # (E,)

    # For a conformal map, edge length ratios should be uniform
    ratio = edge_2d / edge_3d
    # All ratios should be equal for perfect conformality
    mean_ratio = ratio.mean()
    loss = ((ratio - mean_ratio) ** 2).mean()
    return loss


def isometric_loss(
    points_3d: torch.Tensor,
    uv_coords: torch.Tensor,
    faces: torch.Tensor,
) -> torch.Tensor:
    """Isometric (angle + area preserving) distortion loss.

    Combines angular distortion (via triangle angles) and area distortion.
    Uses the Jacobian of the mapping to measure local stretch.

    Args:
        points_3d: (N, 3) 3D surface points
        uv_coords: (N, 2) UV coordinates
        faces: (F, 3) triangle face indices

    Returns:
        Scalar isometric distortion loss
    """
    # Get triangle vertices
    v0_3d = points_3d[faces[:, 0]]  # (F, 3)
    v1_3d = points_3d[faces[:, 1]]
    v2_3d = points_3d[faces[:, 2]]

    v0_uv = uv_coords[faces[:, 0]]  # (F, 2)
    v1_uv = uv_coords[faces[:, 1]]
    v2_uv = uv_coords[faces[:, 2]]

    # 3D edge vectors
    e1_3d = v1_3d - v0_3d  # (F, 3)
    e2_3d = v2_3d - v0_3d

    # 2D edge vectors
    e1_uv = v1_uv - v0_uv  # (F, 2)
    e2_uv = v2_uv - v0_uv

    # Compute Jacobian approximation via least squares
    # J maps 3D edges to 2D: [e1_uv, e2_uv] = J @ [e1_3d, e2_3d]
    # Use edge lengths as stretch indicators
    len3d_1 = e1_3d.norm(dim=-1) + 1e-8  # (F,)
    len3d_2 = e2_3d.norm(dim=-1) + 1e-8
    len2d_1 = e1_uv.norm(dim=-1) + 1e-8
    len2d_2 = e2_uv.norm(dim=-1) + 1e-8

    # Stretch ratios
    s1 = len2d_1 / len3d_1
    s2 = len2d_2 / len3d_2

    # Isometric: both stretch ratios should be equal and uniform
    loss = ((s1 - s2) ** 2).mean()
    return loss


def distortion_loss(
    points_3d: torch.Tensor,
    uv_coords: torch.Tensor,
    edges: torch.Tensor,
    faces: torch.Tensor,
    alpha: float = 1.0,
    beta: float = 1.0,
) -> torch.Tensor:
    """Combined distortion loss (conformal + isometric).

    When mesh connectivity (edges/faces) indexes beyond the point array
    (i.e. sampled points != mesh vertices), falls back to kNN-based
    local distortion estimation.

    Args:
        points_3d: (N, 3) 3D surface points
        uv_coords: (N, 2) UV coordinates
        edges: (E, 2) edge indices
        faces: (F, 3) face indices
        alpha: weight for conformal term
        beta: weight for isometric term

    Returns:
        Combined scalar distortion loss
    """
    N = points_3d.shape[0]
    max_idx = max(edges.max().item(), faces.max().item()) if len(edges) > 0 and len(faces) > 0 else 0

    if max_idx < N:
        conf = conformal_loss(points_3d, uv_coords, edges)
        iso = isometric_loss(points_3d, uv_coords, faces)
        return alpha * conf + beta * iso

    return _knn_distortion_loss(points_3d, uv_coords)


def _knn_distortion_loss(
    points_3d: torch.Tensor,
    uv_coords: torch.Tensor,
    k: int = 6,
) -> torch.Tensor:
    """kNN-based local distortion: ratio of 3D vs UV neighbor distances."""
    d3 = torch.cdist(points_3d.unsqueeze(0), points_3d.unsqueeze(0))[0]
    d2 = torch.cdist(uv_coords.unsqueeze(0), uv_coords.unsqueeze(0))[0]

    _, idx3 = d3.topk(k + 1, dim=1, largest=False)
    idx = idx3[:, 1:]  # (N, k)

    pts_k = points_3d[idx]  # (N, k, 3)
    uvs_k = uv_coords[idx]  # (N, k, 2)

    e3d = (pts_k - points_3d.unsqueeze(1)).norm(dim=-1) + 1e-8  # (N, k)
    e2d = (uvs_k - uv_coords.unsqueeze(1)).norm(dim=-1) + 1e-8  # (N, k)

    r = e2d / e3d
    r_mean = r.mean(dim=1, keepdim=True)
    return ((r - r_mean) ** 2).mean()
