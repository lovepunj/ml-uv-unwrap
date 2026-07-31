"""Wrapping network — maps 2D UV coordinates back to 3D surface points."""

import torch
import torch.nn as nn

from .positional_encoding import PositionalEncoding


class WrapNet(nn.Module):
    """Maps 2D UV coordinates back to 3D surface points.

    The inverse mapping: g: R^2 -> R^3, reconstructing 3D positions
    from UV coordinates. Used in the bi-directional cycle consistency
    framework.
    """

    def __init__(
        self,
        uv_dim: int = 2,
        point_dim: int = 3,
        hidden_dim: int = 256,
        num_layers: int = 8,
        num_freqs: int = 6,
    ):
        super().__init__()
        self.pos_enc = PositionalEncoding(num_freqs=num_freqs, include_input=True)
        input_dim = uv_dim * (num_freqs * 2 + 1)

        layers = []
        for i in range(num_layers):
            in_d = input_dim if i == 0 else hidden_dim
            layers.extend([
                nn.Linear(in_d, hidden_dim),
                nn.ReLU(inplace=True),
            ])
        self.mlp = nn.Sequential(*layers)
        self.head = nn.Linear(hidden_dim, point_dim)

    def forward(self, uv_coords: torch.Tensor) -> torch.Tensor:
        """
        Args:
            uv_coords: (B, N, 2) or (N, 2) UV coordinates
        Returns:
            points: (B, N, 3) or (N, 3) reconstructed 3D points
        """
        encoded = self.pos_enc(uv_coords)
        features = self.mlp(encoded)
        return self.head(features)
