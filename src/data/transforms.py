"""Data transforms for mesh preprocessing."""

import torch
import torch.nn as nn


class NormalizePoints(nn.Module):
    """Normalize point cloud to fit in unit sphere."""

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        center = points.mean(dim=-2, keepdim=True)
        points = points - center
        max_dist = points.norm(dim=-1).max()
        if max_dist > 0:
            points = points / max_dist
        return points


class RandomRotation(nn.Module):
    """Apply random 3D rotation."""

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        A = torch.randn(3, 3, device=points.device, dtype=points.dtype)
        Q, _ = torch.linalg.qr(A)
        if torch.det(Q) < 0:
            Q[:, 0] = -Q[:, 0]
        return points @ Q.T


class AddNoise(nn.Module):
    """Add Gaussian noise for robustness."""

    def __init__(self, std: float = 0.001):
        super().__init__()
        self.std = std

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        return points + torch.randn_like(points) * self.std


class CenterOnMean(nn.Module):
    """Center points on their mean."""

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        return points - points.mean(dim=-2, keepdim=True)


class Subsample(nn.Module):
    """Random subsampling to fixed number of points."""

    def __init__(self, num_points: int):
        super().__init__()
        self.num_points = num_points

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        N = points.shape[-2]
        if N <= self.num_points:
            return points
        idx = torch.randperm(N, device=points.device)[: self.num_points]
        return points[..., idx, :]
