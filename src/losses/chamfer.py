"""Chamfer distance for point cloud reconstruction quality."""

import torch


def chamfer_distance(
    pred: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """Symmetric Chamfer distance between two point sets.

    Computes the sum of nearest-neighbor distances in both directions:
    for each point in pred, find nearest in target (and vice versa).

    Args:
        pred: (B, N, 3) predicted points
        target: (B, M, 3) target points

    Returns:
        Scalar Chamfer distance
    """
    # pred: (B, N, 3), target: (B, M, 3)
    # Compute pairwise distances
    diff = pred.unsqueeze(2) - target.unsqueeze(1)  # (B, N, M, 3)
    dist = diff.pow(2).sum(dim=-1)  # (B, N, M)

    # pred → target: for each pred point, nearest target
    min_dist_pred, _ = dist.min(dim=2)  # (B, N)
    # target → pred: for each target point, nearest pred
    min_dist_target, _ = dist.min(dim=1)  # (B, M)

    return min_dist_pred.mean() + min_dist_target.mean()


def chamfer_distance_unidirectional(
    pred: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """One-way Chamfer distance: pred → target only."""
    diff = pred.unsqueeze(2) - target.unsqueeze(1)  # (B, N, M, 3)
    dist = diff.pow(2).sum(dim=-1)  # (B, N, M)
    min_dist, _ = dist.min(dim=2)  # (B, N)
    return min_dist.mean()


def repulsion_loss(points: torch.Tensor, k: int = 8) -> torch.Tensor:
    """Repulsion loss to prevent point collapse.

    Encourages uniform point distribution by penalizing points
    that are too close to their k nearest neighbors.

    Args:
        points: (B, N, 3) or (N, 3) point cloud
        k: number of neighbors to consider

    Returns:
        Scalar repulsion loss
    """
    if points.dim() == 2:
        points = points.unsqueeze(0)

    B, N, _ = points.shape

    # Compute pairwise distances
    dist = torch.cdist(points, points)  # (B, N, N)

    # Find k nearest neighbors (excluding self)
    knn_dist, _ = dist.topk(k + 1, dim=-1, largest=True, sorted=True)
    knn_dist = knn_dist[:, :, 1:]  # (B, N, k) — exclude self-distance

    # Mean nearest-neighbor distance
    mean_dist = knn_dist.mean(dim=-1)  # (B, N)

    # Repulsion: encourage minimum distance
    target_dist = 0.01
    loss = torch.relu(target_dist - mean_dist).mean()
    return loss
