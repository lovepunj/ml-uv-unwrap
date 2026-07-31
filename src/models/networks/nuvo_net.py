from __future__ import annotations

"""Nuvo-style multi-chart UV architecture.

Reimplementation based on the SIGGRAPH paper:
"Nuvo: Neural UV Parametrization for 3D Shapes"

Key components:
1. Chart assignment network (semantic part segmentation)
2. Per-chart UV parameterization MLPs
3. Chart overlap prevention loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .positional_encoding import PositionalEncoding


class ChartAssignmentNet(nn.Module):
    """Predict per-vertex chart assignments using learned features.

    Uses a PointNet-style architecture with positional encoding
    to predict soft chart assignments for each vertex.
    """

    def __init__(
        self,
        input_dim: int = 3,
        hidden_dim: int = 128,
        num_charts: int = 4,
        num_freqs: int = 6,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_charts = num_charts
        self.pos_enc = PositionalEncoding(num_freqs=num_freqs)

        enc_dim = input_dim * (num_freqs * 2 + 1)

        self.mlp = nn.Sequential(
            nn.Linear(enc_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
        )

        # Soft assignment head
        self.assignment_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, num_charts),
        )

        # Boundary detection head
        self.boundary_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, points: torch.Tensor) -> dict[str, torch.Tensor]:
        """Predict chart assignments and boundary probabilities.

        Args:
            points: (B, N, 3) point cloud

        Returns:
            Dictionary with chart_logits, chart_probs, boundary_logits
        """
        # Encode with positional encoding
        encoded = self.pos_enc(points)  # (B, N, enc_dim)

        B, N, _ = encoded.shape
        flat = encoded.reshape(B * N, -1)
        feat = self.mlp(flat)
        feat = feat.reshape(B, N, -1)

        # Chart assignment
        chart_logits = self.assignment_head(feat)  # (B, N, num_charts)
        chart_probs = F.softmax(chart_logits, dim=-1)

        # Boundary detection
        boundary_logits = self.boundary_head(feat).squeeze(-1)  # (B, N)
        boundary_probs = torch.sigmoid(boundary_logits)

        return {
            "chart_logits": chart_logits,
            "chart_probs": chart_probs,
            "boundary_logits": boundary_logits,
            "boundary_probs": boundary_probs,
        }


class PerChartUVNet(nn.Module):
    """Per-chart UV parameterization network.

    Each chart has its own MLP that maps local 3D coordinates
    to 2D UV coordinates within that chart's parameterization space.
    """

    def __init__(
        self,
        input_dim: int = 3,
        hidden_dim: int = 64,
        num_layers: int = 4,
    ):
        super().__init__()

        layers = []
        in_dim = input_dim
        for i in range(num_layers):
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.GELU(),
            ])
            in_dim = hidden_dim

        layers.append(nn.Linear(hidden_dim, 2))
        layers.append(nn.Tanh())  # Output in [-1, 1]

        self.mlp = nn.Sequential(*layers)

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        """Map 3D points to 2D UV coordinates.

        Args:
            points: (B, N, 3) local 3D coordinates

        Returns:
            (B, N, 2) UV coordinates in [-1, 1]
        """
        return self.mlp(points)


class NuvoNet(nn.Module):
    """Full Nuvo-style multi-chart UV parameterization network.

    Combines chart assignment with per-chart UV generation,
    producing a complete UV map with multiple charts.
    """

    def __init__(
        self,
        input_dim: int = 3,
        hidden_dim: int = 128,
        num_charts: int = 4,
        num_freqs: int = 6,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_charts = num_charts

        # Chart assignment
        self.chart_net = ChartAssignmentNet(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_charts=num_charts,
            num_freqs=num_freqs,
            dropout=dropout,
        )

        # Per-chart UV networks
        self.chart_uv_nets = nn.ModuleList([
            PerChartUVNet(input_dim=input_dim, hidden_dim=hidden_dim // 2)
            for _ in range(num_charts)
        ])

        # Global feature extractor for chart conditioning
        self.global_encoder = nn.Sequential(
            nn.Linear(3 * (num_freqs * 2 + 1), hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, points: torch.Tensor) -> dict[str, torch.Tensor]:
        """Full forward pass.

        Args:
            points: (B, N, 3) point cloud

        Returns:
            Dictionary with UV coords, chart assignments, boundary info
        """
        B, N, _ = points.shape

        # Get chart assignments
        chart_info = self.chart_net(points)

        # Generate UV coordinates per chart and composite
        all_uvs = []
        for chart_idx in range(self.num_charts):
            # Get this chart's UV prediction
            chart_uv = self.chart_uv_nets[chart_idx](points)  # (B, N, 2)

            # Scale UVs to [0, 1] and offset to chart's region
            chart_uv = (chart_uv + 1) / 2  # [0, 1]

            # Offset to chart's tile in UV space
            # Charts are arranged in a grid
            grid_size = int(self.num_charts ** 0.5) + 1
            row = chart_idx // grid_size
            col = chart_idx % grid_size
            offset = torch.tensor([col / grid_size, row / grid_size], device=points.device)
            scale = 1.0 / grid_size
            chart_uv = chart_uv * scale + offset

            all_uvs.append(chart_uv)

        # Composite UVs using chart assignment weights
        stacked_uvs = torch.stack(all_uvs, dim=2)  # (B, N, num_charts, 2)
        chart_weights = chart_info["chart_probs"].unsqueeze(-1)  # (B, N, num_charts, 1)
        final_uv = (stacked_uvs * chart_weights).sum(dim=2)  # (B, N, 2)

        return {
            "uv_coords": final_uv,
            "chart_probs": chart_info["chart_probs"],
            "chart_logits": chart_info["chart_logits"],
            "boundary_probs": chart_info["boundary_probs"],
            "per_chart_uvs": stacked_uvs,
        }

    def compute_nuvo_losses(
        self,
        points: torch.Tensor,
        outputs: dict[str, torch.Tensor],
        edges: torch.Tensor | None = None,
        faces: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Compute Nuvo-specific losses.

        Args:
            points: (B, N, 3) input points
            outputs: dict from forward()
            edges: (E, 2) edge indices
            faces: (F, 3) face indices

        Returns:
            Dictionary of losses
        """
        losses = {}

        # Chart balance loss: encourage uniform chart usage
        chart_probs = outputs["chart_probs"]  # (B, N, C)
        mean_usage = chart_probs.mean(dim=(0, 1))  # (C,)
        ideal = 1.0 / self.num_charts
        losses["chart_balance"] = ((mean_usage - ideal) ** 2).mean()

        # Boundary smoothness loss: penalize high boundary probability
        # (we want clean, not fuzzy boundaries)
        boundary = outputs["boundary_probs"]  # (B, N)
        losses["boundary_reg"] = boundary.mean()

        # Chart coherence loss: encourage points in same chart to be close in UV
        if edges is not None:
            chart_assignments = outputs["chart_probs"].argmax(dim=-1)  # (B, N)
            uv = outputs["uv_coords"]

            # Only penalize UV distortion within same chart
            same_chart = (chart_assignments[:, edges[:, 0]] == chart_assignments[:, edges[:, 1]])
            uv_diff = (uv[:, edges[:, 0]] - uv[:, edges[:, 1]]).norm(dim=-1)
            pt_diff = (points[:, edges[:, 0]] - points[:, edges[:, 1]]).norm(dim=-1)

            # Conformal loss within charts
            ratio = uv_diff / (pt_diff + 1e-8)
            expected_ratio = 1.0  # Equal scale
            coherence = ((ratio - expected_ratio) ** 2 * same_chart.float()).mean()
            losses["chart_coherence"] = coherence

        return losses
