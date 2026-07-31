"""UV deformation network — deforms initial 2D grid to optimal UV coordinates."""

import torch
import torch.nn as nn

from .positional_encoding import PositionalEncoding


class DeformNet(nn.Module):
    """Deforms 2D UV coordinates using residual learning.

    Takes initial UV coordinates (e.g., from PCA projection) and
    outputs a residual deformation to optimize the parameterization.
    """

    def __init__(
        self,
        in_dim: int = 2,
        hidden_dim: int = 256,
        num_layers: int = 6,
        num_freqs: int = 6,
    ):
        super().__init__()
        self.pos_enc = PositionalEncoding(num_freqs=num_freqs, include_input=True)
        input_dim = in_dim * (num_freqs * 2 + 1)

        layers = []
        for i in range(num_layers):
            in_d = input_dim if i == 0 else hidden_dim
            layers.extend([
                nn.Linear(in_d, hidden_dim),
                nn.ReLU(inplace=True),
            ])
        self.mlp = nn.Sequential(*layers)

        mlp_out_dim = hidden_dim if num_layers > 0 else input_dim
        self.head = nn.Sequential(
            nn.Linear(mlp_out_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, in_dim),
        )

    def forward(self, uv_coords: torch.Tensor) -> torch.Tensor:
        """
        Args:
            uv_coords: (B, N, 2) or (N, 2) initial UV coordinates
        Returns:
            deformed_uv: (B, N, 2) or (N, 2) deformed UV coordinates (residual added)
        """
        encoded = self.pos_enc(uv_coords)
        features = self.mlp(encoded)
        residual = self.head(features)
        return uv_coords + residual
