"""Unwrapping network — maps 3D surface points to 2D UV coordinates."""

import torch
import torch.nn as nn

from .positional_encoding import PositionalEncoding


class UnwrapNet(nn.Module):
    """Maps 3D points to 2D UV coordinates.

    The core network that learns the forward parameterization:
    f: R^3 -> R^2, mapping surface points to UV space.
    """

    def __init__(
        self,
        point_dim: int = 3,
        uv_dim: int = 2,
        hidden_dim: int = 256,
        num_layers: int = 8,
        num_freqs: int = 6,
    ):
        super().__init__()
        self.pos_enc = PositionalEncoding(num_freqs=num_freqs, include_input=True)
        input_dim = point_dim * (num_freqs * 2 + 1)

        layers = []
        for i in range(num_layers):
            in_d = input_dim if i == 0 else hidden_dim
            layers.extend([
                nn.Linear(in_d, hidden_dim),
                nn.ReLU(inplace=True),
            ])
        self.mlp = nn.Sequential(*layers)
        self.head = nn.Linear(hidden_dim, uv_dim)

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        """
        Args:
            points: (B, N, 3) or (N, 3) 3D surface points
        Returns:
            uv: (B, N, 2) or (N, 2) UV coordinates
        """
        encoded = self.pos_enc(points)
        features = self.mlp(encoded)
        return self.head(features)
