"""Surface cutting network — learns where to place UV seams.

Supports optional conditioning on PartField semantic features for
part-aware seam prediction.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .positional_encoding import PositionalEncoding


class CutNet(nn.Module):
    """Predicts per-point seam probability for surface cutting.

    Two modes:
    - Geometry-only: uses 3D coordinates + positional encoding
    - Part-aware: concatenates PartField features for semantic seam prediction
    """

    def __init__(
        self,
        point_dim: int = 3,
        hidden_dim: int = 256,
        num_layers: int = 8,
        out_dim: int = 1,
        num_freqs: int = 6,
        partfield_dim: int = 0,
    ):
        """
        Args:
            point_dim: Input point dimension (3 for xyz)
            hidden_dim: Hidden layer width
            num_layers: Number of MLP layers
            out_dim: Output dimension (1 for seam logit)
            num_freqs: Positional encoding frequencies
            partfield_dim: If > 0, condition on PartField features of this dim
        """
        super().__init__()
        self.partfield_dim = partfield_dim
        self.pos_enc = PositionalEncoding(num_freqs=num_freqs, include_input=True)
        geom_input_dim = point_dim * (num_freqs * 2 + 1)

        # Total input = geometry features + PartField features
        total_input_dim = geom_input_dim + partfield_dim

        # Feature fusion: project PartField features to hidden space
        if partfield_dim > 0:
            self.partfield_proj = nn.Sequential(
                nn.Linear(partfield_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, hidden_dim),
            )
            # Fusion layer combines geometry MLP output + part features
            self.fusion = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.ReLU(inplace=True),
            )

        layers = []
        for i in range(num_layers):
            in_d = geom_input_dim if i == 0 else hidden_dim
            layers.extend([
                nn.Linear(in_d, hidden_dim),
                nn.ReLU(inplace=True),
            ])
        self.mlp = nn.Sequential(*layers)
        self.head = nn.Linear(hidden_dim, out_dim)

    def forward(
        self,
        points: torch.Tensor,
        partfield_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            points: (B, N, 3) or (N, 3) surface points
            partfield_features: (B, N, 448) or (N, 448) optional PartField features

        Returns:
            seam_logits: (B, N, 1) or (N, 1) raw seam logits
        """
        encoded = self.pos_enc(points)  # (..., geom_dim)

        if self.partfield_dim > 0 and partfield_features is not None:
            # Ensure partfield_features matches point count
            if partfield_features.shape[-2] != encoded.shape[-2]:
                # Interpolate: each point gets nearest feature
                pf_flat = partfield_features.reshape(-1, partfield_features.shape[-1])
                # Use nearest-neighbor: expand points to match feature count if needed
                N = encoded.shape[-2]
                M = pf_flat.shape[0]
                if M < N:
                    # More points than features — repeat features
                    idx = torch.arange(N, device=encoded.device) % M
                    partfield_features = pf_flat[idx].reshape(*encoded.shape[:-1], -1)
                else:
                    # Fewer points than features — subsample features
                    idx = torch.arange(0, M, M // N, device=encoded.device)[:N]
                    partfield_features = pf_flat[idx].reshape(*encoded.shape[:-1], -1)

            geom_feat = self.mlp(encoded)  # Full MLP on geometry -> (..., hidden)
            part_feat = self.partfield_proj(partfield_features)  # (..., hidden)
            fused = self.fusion(torch.cat([geom_feat, part_feat], dim=-1))
            features = fused
        else:
            features = self.mlp(encoded)

        return self.head(features)


class PartFieldAwareCutNet(nn.Module):
    """CutNet with explicit PartField feature fusion at multiple scales.

    Uses a cross-attention mechanism to let geometry features attend to
    PartField semantic features, producing better seam predictions.
    """

    def __init__(
        self,
        point_dim: int = 3,
        hidden_dim: int = 256,
        num_layers: int = 8,
        out_dim: int = 1,
        num_freqs: int = 6,
        partfield_dim: int = 448,
        num_heads: int = 4,
    ):
        super().__init__()
        self.pos_enc = PositionalEncoding(num_freqs=num_freqs, include_input=True)
        geom_input_dim = point_dim * (num_freqs * 2 + 1)

        # Geometry encoder
        self.geom_encoder = nn.Sequential(
            nn.Linear(geom_input_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )

        # PartField encoder
        self.part_encoder = nn.Sequential(
            nn.Linear(partfield_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )

        # Cross-attention: geometry attends to part features
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True,
        )

        # MLP layers
        layers = []
        for _ in range(num_layers):
            layers.extend([
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(inplace=True),
            ])
        self.mlp = nn.Sequential(*layers)
        self.head = nn.Linear(hidden_dim, out_dim)

    def forward(
        self,
        points: torch.Tensor,
        partfield_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            points: (B, N, 3) surface points
            partfield_features: (B, N, 448) PartField features

        Returns:
            seam_logits: (B, N, 1) seam probabilities
        """
        encoded = self.pos_enc(points)
        geom_feat = self.geom_encoder(encoded)  # (B, N, hidden)

        if partfield_features is not None:
            part_feat = self.part_encoder(partfield_features)  # (B, N, hidden)

            # Cross-attention: geometry queries attend to part features
            attn_out, _ = self.cross_attn(
                query=geom_feat,
                key=part_feat,
                value=part_feat,
            )
            features = geom_feat + attn_out  # Residual connection
        else:
            features = geom_feat

        features = self.mlp(features)
        return self.head(features)
