from __future__ import annotations

"""Cycle consistency loss for bi-directional mapping."""

import torch


def cycle_consistency_loss(
    points_3d: torch.Tensor,
    reconstructed_3d: torch.Tensor,
    uv_coords: torch.Tensor,
    reconstructed_uv: torch.Tensor,
) -> torch.Tensor:
    """Bi-directional cycle consistency loss.

    Ensures that the round-trip mappings are identity:
    - 3D → UV → 3D should reconstruct original 3D points
    - UV → 3D → UV should reconstruct original UV coordinates

    Args:
        points_3d: (B, N, 3) original 3D points
        reconstructed_3d: (B, N, 3) 3D points after 3D→UV→3D cycle
        uv_coords: (B, N, 2) predicted UV coordinates
        reconstructed_uv: (B, N, 2) UV coords after UV→3D→UV cycle

    Returns:
        Scalar loss combining both cycle directions
    """
    loss_3d_cycle = torch.nn.functional.l1_loss(reconstructed_3d, points_3d)
    loss_2d_cycle = torch.nn.functional.l1_loss(reconstructed_uv, uv_coords)
    return loss_3d_cycle + loss_2d_cycle


def direction_loss(
    points_3d: torch.Tensor,
    reconstructed_3d: torch.Tensor,
    normals_3d: torch.Tensor | None = None,
) -> torch.Tensor:
    """Direction-aware cycle loss using normals.

    Penalizes reconstruction error more strongly along the normal direction,
    since errors along the surface normal are more perceptually visible.
    """
    diff = reconstructed_3d - points_3d
    if normals_3d is not None:
        normal_component = (diff * normals_3d).sum(dim=-1, keepdim=True)
        tangent_component = diff - normal_component * normals_3d
        return (normal_component.abs().mean() + 0.1 * tangent_component.norm(dim=-1).mean())
    return diff.norm(dim=-1).mean()
