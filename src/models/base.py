from __future__ import annotations

"""Base UV unwrap model interface."""

from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class UVUnwrapModel(ABC, nn.Module):
    """Abstract base class for UV unwrapping models."""

    @abstractmethod
    def forward(self, points: torch.Tensor, **kwargs) -> dict[str, torch.Tensor]:
        """Run the unwrapping pipeline.

        Args:
            points: (B, N, 3) surface points

        Returns:
            Dictionary with keys:
                - uv_coords: (B, N, 2) UV coordinates
                - seam_logits: (B, N, 1) seam probabilities (optional)
        """
        ...

    @abstractmethod
    def unwrap(self, points: torch.Tensor, **kwargs) -> torch.Tensor:
        """Convenience method that returns only UV coordinates."""
        ...

    def compute_losses(
        self,
        points: torch.Tensor,
        outputs: dict[str, torch.Tensor],
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        """Compute all training losses. Override for custom loss weighting."""
        return {}
