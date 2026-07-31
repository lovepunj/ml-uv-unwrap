"""ArtUV-style offset prediction module for UV unwrapping.

Implements the key architecture from "ArtUV: Artist-style UV Unwrapping"
(ICLR 2026): learn residual adjustments Q_o to an initial UV Q_i,
producing final UV = Q_i + Q_o.

Architecture:
- Res-M MLP: adaptive dimension mapping for multi-modal features
- GraphConv (SAGEConv-style): per-vertex neighborhood aggregation
- Pyramid Attention: coarse-to-fine refinement at multiple scales

The model predicts per-vertex UV offsets given:
- Initial coarse UV coordinates (from xatlas/LSCM/PCA)
- 3D vertex positions
- Per-vertex features (normals, curvature, etc.)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .positional_encoding import PositionalEncoding


class ResMLP(nn.Module):
    """Residual MLP with adaptive dimension mapping.

    Maps multi-modal inputs (UV coords, positions, features) to
    a shared hidden dimension via separate linear projections,
    then processes through residual blocks.
    """

    def __init__(
        self,
        uv_dim: int = 2,
        pos_dim: int = 3,
        feat_dim: int = 32,
        hidden_dim: int = 128,
        num_freqs: int = 4,
    ):
        super().__init__()
        self.pos_enc = PositionalEncoding(num_freqs=num_freqs, include_input=True)

        # Adaptive projections for each modality
        self.uv_proj = nn.Linear(uv_dim * self.pos_enc.output_dim, hidden_dim)
        self.pos_proj = nn.Linear(pos_dim * self.pos_enc.output_dim, hidden_dim)
        self.feat_proj = nn.Linear(feat_dim, hidden_dim)

        # Residual blocks
        self.res_blocks = nn.ModuleList([
            ResBlock(hidden_dim) for _ in range(3)
        ])

        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        uv: torch.Tensor,
        positions: torch.Tensor,
        features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            uv: (B, N, 2) initial UV coordinates
            positions: (B, N, 3) 3D vertex positions
            features: (B, N, feat_dim) per-vertex features

        Returns:
            (B, N, hidden_dim) aggregated features
        """
        uv_enc = self.pos_enc(uv)
        pos_enc = self.pos_enc(positions)

        uv_h = self.uv_proj(uv_enc)
        pos_h = self.pos_proj(pos_enc)
        feat_h = self.feat_proj(features)

        # Combine: sum of modality-specific projections
        h = uv_h + pos_h + feat_h

        for block in self.res_blocks:
            h = block(h)

        return self.norm(h)


class ResBlock(nn.Module):
    """Simple residual block with LayerNorm."""

    def __init__(self, dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class GraphConvLayer(nn.Module):
    """SAGEConv-style graph convolution for vertex neighborhood aggregation.

    Aggregates features from neighboring vertices on the mesh graph,
    then applies a learned transformation.
    """

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.self_proj = nn.Linear(in_dim, out_dim)
        self.neigh_proj = nn.Linear(in_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x: (N, in_dim) vertex features
            edge_index: (2, E) edge indices [source, target]

        Returns:
            (N, out_dim) aggregated features
        """
        N = x.shape[0]
        src, dst = edge_index

        # Self projection
        h_self = self.self_proj(x)

        # Neighbor aggregation (mean)
        msgs = self.neigh_proj(x[src])
        # Scatter mean
        agg = torch.zeros(N, msgs.shape[-1], device=x.device)
        count = torch.zeros(N, device=x.device)
        agg.index_add_(0, dst, msgs)
        count.index_add_(0, dst, torch.ones(dst.shape[0], device=x.device))
        agg = agg / (count.unsqueeze(-1) + 1e-6)

        h = h_self + agg
        return self.norm(F.gelu(h))


class PyramidAttention(nn.Module):
    """Coarse-to-fine attention at multiple scales.

    Downsamples features progressively, applies self-attention
    at each scale, then upsamples and combines.
    """

    def __init__(self, dim: int = 128, num_heads: int = 4, num_scales: int = 3):
        super().__init__()
        self.scales = nn.ModuleList()
        self.down_projs = nn.ModuleList()
        self.up_projs = nn.ModuleList()

        for i in range(num_scales):
            scale_dim = dim // (2 ** i)
            self.scales.append(
                nn.MultiheadAttention(scale_dim, num_heads, batch_first=True)
            )
            if i > 0:
                self.down_projs.append(nn.Linear(dim // (2 ** (i - 1)), scale_dim))
            self.up_projs.append(nn.Linear(scale_dim, dim))

        self.output_proj = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, N, dim) vertex features

        Returns:
            (B, N, dim) refined features
        """
        B, N, _ = x.shape
        residuals = []

        h = x
        for i, (attn, up_proj) in enumerate(zip(self.scales, self.up_projs)):
            if i > 0:
                # Downsample: pool pairs
                h = self.down_projs[i - 1](h)
                h = h.reshape(B, -1, h.shape[-1]).mean(dim=1, keepdim=True).expand(B, N, -1)

            # Self-attention
            h_attn, _ = attn(h, h, h)

            # Upsample and store residual
            h_up = up_proj(h_attn)
            residuals.append(h_up)

        # Combine all scales
        out = sum(residuals)
        return self.norm(self.output_proj(out))


class ArtUVModel(nn.Module):
    """ArtUV-style UV offset prediction model.

    Given an initial coarse UV Q_i, predicts a residual offset Q_o
    such that final UV = Q_i + Q_o.

    Architecture:
    1. Res-M MLP: aggregate UV + position + feature inputs
    2. GraphConv layers: mesh-aware neighborhood aggregation
    3. Pyramid Attention: multi-scale refinement
    4. Offset head: predict 2D UV offset per vertex
    """

    def __init__(
        self,
        uv_dim: int = 2,
        pos_dim: int = 3,
        feat_dim: int = 32,
        hidden_dim: int = 128,
        num_graph_layers: int = 5,
        num_freqs: int = 4,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Res-M MLP: multi-modal input aggregation
        self.res_mlp = ResMLP(
            uv_dim=uv_dim,
            pos_dim=pos_dim,
            feat_dim=feat_dim,
            hidden_dim=hidden_dim,
            num_freqs=num_freqs,
        )

        # Graph convolution layers (SAGEConv-style)
        self.graph_convs = nn.ModuleList([
            GraphConvLayer(hidden_dim, hidden_dim) for _ in range(num_graph_layers)
        ])

        # Pyramid attention refinement
        self.pyramid = PyramidAttention(dim=hidden_dim, num_heads=4, num_scales=3)

        # Offset prediction head
        self.offset_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, uv_dim),
            nn.Tanh(),  # Offset in [-1, 1] range
        )

        # Scale parameter (learnable)
        self.offset_scale = nn.Parameter(torch.tensor(0.1))

    def compute_vertex_features(
        self,
        vertices: torch.Tensor,
        faces: torch.Tensor,
    ) -> torch.Tensor:
        """Compute per-vertex geometric features.

        Args:
            vertices: (V, 3) mesh vertices
            faces: (F, 3) face indices

        Returns:
            (V, feat_dim) per-vertex features
        """
        V = vertices.shape[0]
        F_count = faces.shape[0]
        feat_dim = 32

        features = torch.zeros(V, feat_dim, device=vertices.device)

        # Compute face normals
        v0 = vertices[faces[:, 0]]
        v1 = vertices[faces[:, 1]]
        v2 = vertices[faces[:, 2]]
        e1 = v1 - v0
        e2 = v2 - v0
        face_normals = F.normalize(torch.cross(e1, e2, dim=-1), dim=-1)

        # Compute face areas
        face_areas = torch.norm(torch.cross(e1, e2, dim=-1), dim=-1) * 0.5

        # Accumulate area-weighted normals per vertex
        # Expand face_normals to feat_dim: pad with zeros beyond dim 3
        face_feat = torch.zeros(F_count, feat_dim, device=vertices.device)
        face_feat[:, :3] = face_normals
        areas_expanded = face_areas.unsqueeze(-1).expand(-1, feat_dim)
        face_feat = face_feat * areas_expanded

        for i in range(3):
            idx = faces[:, i]
            features.index_add_(0, idx, face_feat)

        # Normalize by vertex valence
        valence = torch.zeros(V, device=vertices.device)
        ones = torch.ones(F_count, device=vertices.device)
        for i in range(3):
            valence.index_add_(0, faces[:, i], ones)
        valence = valence.clamp(min=1)

        features = features / valence.unsqueeze(-1)

        # Add position features (normalized)
        pos_features = F.normalize(vertices, dim=-1)
        features[:, :3] = pos_features

        return features

    def forward(
        self,
        initial_uv: torch.Tensor,
        vertices: torch.Tensor,
        faces: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Predict UV offset from initial coarse UV.

        Args:
            initial_uv: (B, V, 2) or (V, 2) initial UV coordinates
            vertices: (B, V, 3) or (V, 3) mesh vertices
            faces: (F, 3) face indices
            edge_index: (2, E) graph edges

        Returns:
            Dictionary with:
                - uv_pred: (B, V, 2) predicted UV coordinates
                - offset: (B, V, 2) predicted offset
                - features: (B, V, hidden_dim) intermediate features
        """
        single = initial_uv.dim() == 2
        if single:
            initial_uv = initial_uv.unsqueeze(0)
            vertices = vertices.unsqueeze(0)

        B, V, _ = initial_uv.shape

        # Compute vertex features (process each batch element)
        feat_list = []
        for b in range(B):
            feat_list.append(self.compute_vertex_features(vertices[b], faces))
        vertex_feat = torch.stack(feat_list)  # (B, V, feat_dim)

        # Truncate to feat_dim if needed
        if vertex_feat.shape[-1] > 32:
            vertex_feat = vertex_feat[:, :, :32]
        elif vertex_feat.shape[-1] < 32:
            pad = torch.zeros(B, V, 32 - vertex_feat.shape[-1], device=vertex_feat.device)
            vertex_feat = torch.cat([vertex_feat, pad], dim=-1)

        # Res-M MLP: aggregate multi-modal inputs
        h = self.res_mlp(initial_uv, vertices, vertex_feat)

        # Graph convolutions (reshape to flat vertex layout)
        B, N, D = h.shape
        E = edge_index.shape[1]
        h_flat = h.reshape(B * N, D)
        # Adjust edge_index for batch
        offsets = torch.arange(B, device=h.device).unsqueeze(1) * N
        edge_index_batched = edge_index.unsqueeze(0) + offsets  # (B, 2, E)
        edge_index_batched = edge_index_batched.reshape(2, B * E)

        for conv in self.graph_convs:
            h_flat = conv(h_flat, edge_index_batched)
        h = h_flat.reshape(B, N, -1)

        # Pyramid attention
        h = self.pyramid(h)

        # Predict offset
        offset = self.offset_head(h) * self.offset_scale

        # Final UV = initial + offset
        uv_pred = initial_uv + offset

        if single:
            uv_pred = uv_pred.squeeze(0)
            offset = offset.squeeze(0)
            h = h.squeeze(0)

        return {
            "uv_pred": uv_pred,
            "offset": offset,
            "features": h,
        }

    def unwrap(
        self,
        initial_uv: torch.Tensor,
        vertices: torch.Tensor,
        faces: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        """Return predicted UV coordinates."""
        return self.forward(initial_uv, vertices, faces, edge_index)["uv_pred"]
