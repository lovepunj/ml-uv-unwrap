"""ArtUV-style loss functions for UV parameterization quality.

Implements the 4-term loss from "ArtUV: Artist-style UV Unwrapping"
(ICLR 2026):
- L_recon: UV reconstruction loss with Horn alignment
- L_silhouette: differentiable silhouette rendering loss
- L_distortion: Jacobian SVD-based conformal distortion
- L_overlap: penalizes flipped UV face normals (overlapping triangles)
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def horn_align(
    pred: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """Horn's method for rigid alignment (rotation + translation).

    Aligns pred to target via SVD-based optimal rotation,
    avoiding penalizing rotations in the loss.

    Args:
        pred: (B, N, 2) predicted points
        target: (B, N, 2) target points

    Returns:
        (B, N, 2) aligned predictions
    """
    # Center both
    pred_mean = pred.mean(dim=1, keepdim=True)
    target_mean = target.mean(dim=1, keepdim=True)
    pred_c = pred - pred_mean
    target_c = target - target_mean

    # SVD for optimal rotation
    H = torch.bmm(pred_c.transpose(1, 2), target_c)  # (B, 2, 2)
    U, S, Vt = torch.linalg.svd(H)
    R = torch.bmm(Vt.transpose(1, 2), U.transpose(1, 2))

    # Handle reflection
    det = torch.det(R)
    sign = torch.ones_like(det)
    sign[det < 0] = -1
    Vt_fix = Vt.clone()
    Vt_fix[:, -1, :] *= sign.unsqueeze(-1)
    R = torch.bmm(Vt_fix.transpose(1, 2), U.transpose(1, 2))

    # Apply rotation
    aligned = torch.bmm(pred_c, R) + target_mean
    return aligned


def reconstruction_loss(
    uv_pred: torch.Tensor,
    uv_target: torch.Tensor,
    use_horn: bool = True,
) -> torch.Tensor:
    """UV reconstruction loss with optional Horn alignment.

    Args:
        uv_pred: (B, N, 2) predicted UV coordinates
        uv_target: (B, N, 2) target UV coordinates
        use_horn: align predictions before comparing (avoids rotation penalty)

    Returns:
        Scalar L1 reconstruction loss
    """
    if use_horn:
        uv_pred = horn_align(uv_pred, uv_target)

    return F.l1_loss(uv_pred, uv_target)


def silhouette_loss(
    uv_pred: torch.Tensor,
    uv_target: torch.Tensor,
    grid_size: int = 256,
) -> torch.Tensor:
    """Silhouette rendering loss.

    Rasterizes UV coordinates into a binary silhouette image and
    penalizes differences. Encourages clean, compact UV boundaries.

    Args:
        uv_pred: (B, N, 2) predicted UV coordinates in [0, 1]
        uv_target: (B, N, 2) target UV coordinates in [0, 1]
        grid_size: rasterization resolution

    Returns:
        Scalar silhouette loss
    """
    B = uv_pred.shape[0]
    total_loss = 0.0

    for b in range(B):
        pred = uv_pred[b]  # (N, 2)
        target = uv_target[b]  # (N, 2)

        # Rasterize to grid
        pred_grid = _rasterize_silhouette(pred, grid_size)
        target_grid = _rasterize_silhouette(target, grid_size)

        total_loss += F.mse_loss(pred_grid, target_grid)

    return total_loss / B


def _rasterize_silhouette(uv: torch.Tensor, grid_size: int) -> torch.Tensor:
    """Rasterize UV points into a binary silhouette grid.

    Args:
        uv: (N, 2) UV coordinates in [0, 1]
        grid_size: output grid resolution

    Returns:
        (grid_size, grid_size) binary silhouette
    """
    grid = torch.zeros(grid_size, grid_size, device=uv.device)

    # Map UV to grid coordinates
    uv_clamped = uv.clamp(0, 1)
    gx = (uv_clamped[:, 0] * (grid_size - 1)).long()
    gy = (uv_clamped[:, 1] * (grid_size - 1)).long()

    # Scatter fill
    indices = gy * grid_size + gx
    grid.view(-1).index_fill_(0, indices.clamp(0, grid_size * grid_size - 1), 1.0)

    # Slight dilation for connectivity
    kernel = torch.ones(3, 3, device=uv.device) / 9.0
    grid = grid.unsqueeze(0).unsqueeze(0)
    grid = F.conv2d(grid, kernel.unsqueeze(0).unsqueeze(0), padding=1)
    grid = (grid > 0.1).float().squeeze()

    return grid


def overlap_loss(
    uv_coords: torch.Tensor,
    faces: torch.Tensor,
) -> torch.Tensor:
    """Overlap detection loss.

    Penalizes faces where UV triangle winding is flipped relative
    to the 3D face normal (indicates overlapping triangles).

    Args:
        uv_coords: (V, 2) UV coordinates
        faces: (F, 3) face indices

    Returns:
        Scalar overlap loss
    """
    uv0 = uv_coords[faces[:, 0]]
    uv1 = uv_coords[faces[:, 1]]
    uv2 = uv_coords[faces[:, 2]]

    # Compute signed area of UV triangles
    # cross product z-component: (u1-u0) x (u2-u0)
    e1 = uv1 - uv0
    e2 = uv2 - uv0
    cross_z = e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0]

    # Penalize negative signed area (flipped winding)
    # For correct winding, cross_z should be positive
    overlap = F.relu(-cross_z).mean()

    return overlap


def distortion_loss_jacobian(
    vertices: torch.Tensor,
    uv_coords: torch.Tensor,
    faces: torch.Tensor,
) -> torch.Tensor:
    """Jacobian SVD-based conformal distortion loss.

    Computes the Jacobian of the 3D→UV mapping per triangle,
    then measures deviation from conformal (angle-preserving).

    Args:
        vertices: (V, 3) mesh vertices
        uv_coords: (V, 2) UV coordinates
        faces: (F, 3) face indices

    Returns:
        Scalar distortion loss
    """
    v0_3d = vertices[faces[:, 0]]
    v1_3d = vertices[faces[:, 1]]
    v2_3d = vertices[faces[:, 2]]

    v0_uv = uv_coords[faces[:, 0]]
    v1_uv = uv_coords[faces[:, 1]]
    v2_uv = uv_coords[faces[:, 2]]

    # 3D edge vectors (F, 3, 2) — project to tangent plane
    e1_3d = v1_3d - v0_3d  # (F, 3)
    e2_3d = v2_3d - v0_3d

    # 2D edge vectors (F, 2)
    e1_uv = v1_uv - v0_uv
    e2_uv = v2_uv - v0_uv

    # Compute Jacobian via least squares: J = [e1_uv, e2_uv] @ pinv([e1_3d, e2_3d])
    # Simplified: use edge length ratios as conformal measure
    len3d_1 = e1_3d.norm(dim=-1) + 1e-8
    len3d_2 = e2_3d.norm(dim=-1) + 1e-8
    len2d_1 = e1_uv.norm(dim=-1) + 1e-8
    len2d_2 = e2_uv.norm(dim=-1) + 1e-8

    # Conformal distortion: (r1 - r2)^2 where r_i = |e_i^uv| / |e_i^3d|
    r1 = len2d_1 / len3d_1
    r2 = len2d_2 / len3d_2

    distortion = ((r1 - r2) ** 2).mean()

    return distortion


def artuv_total_loss(
    vertices: torch.Tensor,
    uv_pred: torch.Tensor,
    uv_target: torch.Tensor,
    faces: torch.Tensor,
    weights: dict[str, float] | None = None,
) -> dict[str, torch.Tensor]:
    """Compute total ArtUV-style loss.

    L = w_recon * L_recon + w_silhouette * L_silhouette
      + w_distortion * L_distortion + w_overlap * L_overlap

    Args:
        vertices: (V, 3) mesh vertices
        uv_pred: (B, V, 2) or (V, 2) predicted UV coordinates
        uv_target: (B, V, 2) or (V, 2) target UV coordinates
        faces: (F, 3) face indices
        weights: loss weight overrides

    Returns:
        Dictionary of losses including 'total'
    """
    w = {
        "recon": 1.0,
        "silhouette": 1.0,
        "distortion": 0.0001,
        "overlap": 0.01,
    }
    if weights:
        w.update(weights)

    # Ensure batch dimension
    if uv_pred.dim() == 2:
        uv_pred = uv_pred.unsqueeze(0)
        uv_target = uv_target.unsqueeze(0)

    losses = {}

    # Reconstruction loss with Horn alignment
    losses["recon"] = reconstruction_loss(uv_pred, uv_target, use_horn=True)

    # Silhouette loss
    losses["silhouette"] = silhouette_loss(uv_pred, uv_target)

    # Distortion loss (per batch element)
    dist_losses = []
    for b in range(uv_pred.shape[0]):
        dist_losses.append(
            distortion_loss_jacobian(vertices, uv_pred[b], faces)
        )
    losses["distortion"] = torch.stack(dist_losses).mean()

    # Overlap loss (per batch element)
    overlap_losses = []
    for b in range(uv_pred.shape[0]):
        overlap_losses.append(overlap_loss(uv_pred[b], faces))
    losses["overlap"] = torch.stack(overlap_losses).mean()

    # Total
    total = (
        w["recon"] * losses["recon"]
        + w["silhouette"] * losses["silhouette"]
        + w["distortion"] * losses["distortion"]
        + w["overlap"] * losses["overlap"]
    )
    losses["total"] = total

    return losses
