from __future__ import annotations

"""ArtUV-style UV refinement module.

Reimplementation based on the SIGGRAPH paper:
"ArtUV: Learning In-the-wild UV Completion for Automatic 3D Shape Unwrapping"

This module refines initial UV parameterizations using a learned
post-optimization step that improves seam placement and reduces distortion.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .positional_encoding import PositionalEncoding


class UVRefinementNet(nn.Module):
    """Refine UV coordinates using learned post-processing.

    Takes initial UV coordinates and 3D point features,
    then outputs refined UV coordinates with better properties:
    - Reduced distortion
    - Better seam placement
    - Improved chart packing
    """

    def __init__(
        self,
        input_dim: int = 5,  # 3D coords + 2D UV
        hidden_dim: int = 128,
        num_layers: int = 4,
        num_freqs: int = 6,
    ):
        super().__init__()
        self.pos_enc = PositionalEncoding(num_freqs=num_freqs)
        enc_dim = input_dim * (num_freqs * 2 + 1)

        # Feature encoder
        self.encoder = nn.Sequential(
            nn.Linear(enc_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # UV refinement heads
        self.delta_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 2),
            nn.Tanh(),  # Small delta in [-0.1, 0.1]
        )

        self.scale_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),  # Scale factor in [0, 1]
        )

    def forward(
        self,
        points_3d: torch.Tensor,
        uv_init: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Refine initial UV coordinates.

        Args:
            points_3d: (B, N, 3) 3D coordinates
            uv_init: (B, N, 2) initial UV coordinates

        Returns:
            Dictionary with refined UVs and refinement details
        """
        # Concatenate features
        features = torch.cat([points_3d, uv_init], dim=-1)  # (B, N, 5)
        encoded = self.pos_enc(features)

        # Encode
        feat = self.encoder(encoded)  # (B, N, hidden)

        # Predict UV delta and scale
        delta = self.delta_head(feat) * 0.1  # Scale down for stability
        scale = self.scale_head(feat)

        # Apply refinement
        # Center the UV coordinates
        uv_centered = uv_init - uv_init.mean(dim=1, keepdim=True)

        # Scale and add delta
        uv_refined = uv_centered * scale + delta

        # Renormalize to [0, 1]
        uv_min = uv_refined.min(dim=1, keepdim=True).values
        uv_max = uv_refined.max(dim=1, keepdim=True).values
        uv_refined = (uv_refined - uv_min) / (uv_max - uv_min + 1e-8)

        return {
            "uv_refined": uv_refined,
            "delta": delta,
            "scale": scale,
        }


class DistortionAwareRefiner(nn.Module):
    """UV refiner that explicitly minimizes distortion.

    Uses a learned distortion metric to guide the refinement process,
    similar to ArtUV's approach of using a discriminator to evaluate
    UV quality.
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        num_freqs: int = 6,
    ):
        super().__init__()
        self.refiner = UVRefinementNet(
            input_dim=5,
            hidden_dim=hidden_dim,
            num_freqs=num_freqs,
        )

        # Distortion evaluator
        self.distortion_eval = nn.Sequential(
            nn.Linear(2 + 3 + 1, hidden_dim),  # UV + 3D + edge_length
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),  # Distortion score in [0, 1]
        )

    def forward(
        self,
        points_3d: torch.Tensor,
        uv_init: torch.Tensor,
        edges: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Refine UVs with distortion awareness.

        Args:
            points_3d: (B, N, 3) 3D coordinates
            uv_init: (B, N, 2) initial UV coordinates
            edges: (E, 2) edge indices for distortion computation

        Returns:
            Dictionary with refined UVs and distortion scores
        """
        # Refine UVs
        refined = self.refiner(points_3d, uv_init)

        # Evaluate distortion if edges provided
        distortion_score = None
        if edges is not None:
            # Compute edge-based distortion
            uv_a = refined["uv_refined"][:, edges[:, 0]]
            uv_b = refined["uv_refined"][:, edges[:, 1]]
            uv_edge_len = (uv_a - uv_b).norm(dim=-1)

            pt_a = points_3d[:, edges[:, 0]]
            pt_b = points_3d[:, edges[:, 1]]
            pt_edge_len = (pt_a - pt_b).norm(dim=-1)

            # Distortion ratio
            ratio = uv_edge_len / (pt_edge_len + 1e-8)

            # Evaluate with distortion network
            edge_features = torch.cat([
                uv_edge_len.unsqueeze(-1),
                pt_edge_len.unsqueeze(-1),
                ratio.unsqueeze(-1),
            ], dim=-1)

            distortion_score = self.distortion_eval(edge_features).mean()

        return {
            "uv_refined": refined["uv_refined"],
            "delta": refined["delta"],
            "scale": refined["scale"],
            "distortion_score": distortion_score,
        }
