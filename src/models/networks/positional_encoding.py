"""Positional encoding for 3D coordinates."""

import math

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """Fourier feature positional encoding.

    Maps input coordinates to higher-dimensional space using sinusoidal
    functions at multiple frequencies, enabling MLPs to learn
    high-frequency geometric detail.
    """

    def __init__(self, num_freqs: int = 6, include_input: bool = True):
        super().__init__()
        self.num_freqs = num_freqs
        self.include_input = include_input
        freqs = 2.0 ** torch.arange(num_freqs).float()  # [1, 2, 4, ..., 2^(n-1)]
        self.register_buffer("freqs", freqs)

    @property
    def output_dim(self) -> int:
        d = self.num_freqs * 2  # sin + cos per frequency
        if self.include_input:
            d += 1  # raw input
        return d

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (..., D) input coordinates
        Returns:
            encoded: (..., D * output_dim) or (..., D + D * num_freqs * 2)
        """
        # x: (..., D)
        args = x.unsqueeze(-1) * self.freqs * math.pi  # (..., D, F)
        encoded = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)  # (..., D, F*2)
        encoded = encoded.reshape(*x.shape[:-1], -1)  # (..., D*F*2)

        if self.include_input:
            encoded = torch.cat([x, encoded], dim=-1)

        return encoded
