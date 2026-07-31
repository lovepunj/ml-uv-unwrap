from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def sinusoidal_position_encoding(length: int, dim: int, device: torch.device) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=device) / half)
    t = torch.arange(length, device=device).float()
    args = t[:, None] * freqs[None, :]
    return torch.cat([args.sin(), args.cos()], dim=-1)


class VecSetAttentionPooling(nn.Module):
    """VecSet-style attention pooling: project N points to K set tokens via learned queries."""

    def __init__(self, dim: int, num_queries: int, num_heads: int = 8) -> None:
        super().__init__()
        self.num_queries = num_queries
        self.queries = nn.Parameter(torch.randn(1, num_queries, dim) * 0.02)
        self.norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, N, dim) -> (B, num_queries, dim)"""
        B = x.shape[0]
        q = self.queries.expand(B, -1, -1)
        out, _ = self.attn(q, x, x)
        return self.proj(self.norm(out))


class DualBranchEncoder(nn.Module):
    """Dual-branch encoder for topology and geometry point clouds.

    Each branch encodes a 30,720-point cloud into 3,072 set tokens of dim 1024.
    Outputs are concatenated into a 6,144 × 1024 condition embedding.
    """

    def __init__(
        self,
        in_channels: int = 3,
        dim: int = 1024,
        num_heads: int = 16,
        num_layers: int = 12,
        num_tokens: int = 3072,
        num_pool_queries: int = 3072,
        max_points: int = 30_720,
    ) -> None:
        super().__init__()
        self.num_tokens = num_tokens

        self.topo_embed = nn.Linear(in_channels, dim)
        self.topo_pos = nn.Parameter(torch.randn(1, max_points, dim) * 0.02)
        self.topo_layers = nn.ModuleList([
            TransformerBlock(dim, num_heads) for _ in range(num_layers)
        ])
        self.topo_pool = VecSetAttentionPooling(dim, num_pool_queries, num_heads)

        self.geom_embed = nn.Linear(in_channels, dim)
        self.geom_pos = nn.Parameter(torch.randn(1, max_points, dim) * 0.02)
        self.geom_layers = nn.ModuleList([
            TransformerBlock(dim, num_heads) for _ in range(num_layers)
        ])
        self.geom_pool = VecSetAttentionPooling(dim, num_pool_queries, num_heads)

        self.output_norm = nn.LayerNorm(dim)

    def forward(
        self,
        topo_points: torch.Tensor,
        geom_points: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            topo_points: (B, N, 3) vertex + edge point cloud (N can vary).
            geom_points: (B, N, 3) surface point cloud.

        Returns:
            (B, 6144, 1024) condition embedding.
        """
        N = topo_points.shape[1]
        h = self.topo_embed(topo_points) + self.topo_pos[:, :N]
        for layer in self.topo_layers:
            h = layer(h)
        topo_tokens = self.topo_pool(h)

        h = self.geom_embed(geom_points) + self.geom_pos[:, :N]
        for layer in self.geom_layers:
            h = layer(h)
        geom_tokens = self.geom_pool(h)

        combined = torch.cat([topo_tokens, geom_tokens], dim=1)
        return self.output_norm(combined)


class CausalTransformerBlock(nn.Module):
    """GPT-style causal self-attention block with cross-attention."""

    def __init__(self, dim: int, num_heads: int, max_seq_len: int = 2048) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.self_attn = CausalSelfAttention(dim, num_heads, max_seq_len)
        self.ln2 = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.ln3 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(
        self,
        x: torch.Tensor,
        cond: torch.Tensor,
        attn_mask: torch.Tensor,
    ) -> torch.Tensor:
        """x: (B, T, dim), cond: (B, S, dim)"""
        x = x + self.self_attn(self.ln1(x), attn_mask)
        cross_out, _ = self.cross_attn(self.ln2(x), cond, cond)
        x = x + cross_out
        x = x + self.ffn(self.ln3(x))
        return x


class CausalSelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, max_seq_len: int) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.register_buffer(
            "causal_mask",
            torch.triu(torch.ones(max_seq_len, max_seq_len, dtype=torch.bool), diagonal=1),
        )

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, T, _ = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)

        mask = self.causal_mask[:T, :T]
        if attn_mask is not None:
            mask = mask | attn_mask

        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        out = out.transpose(1, 2).reshape(B, T, -1)
        return self.proj(out)


class TransformerBlock(nn.Module):
    """Standard bidirectional transformer block for the encoder."""

    def __init__(self, dim: int, num_heads: int) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.attn(self.norm1(x), self.norm1(x), self.norm1(x))
        x = x + attn_out
        x = x + self.ffn(self.norm2(x))
        return x


class SeamDecoder(nn.Module):
    """GPT-style causal transformer decoder with hourglass architecture.

    Predicts seam segments autoregressively as quantized 3D coordinates
    using 1024-bin quantization per axis with yzx ordering.
    """

    def __init__(
        self,
        dim: int = 256,
        num_heads: int = 4,
        num_layers: int = 3,
        max_seq_len: int = 128,
        num_bins: int = 128,
    ) -> None:
        super().__init__()
        self.num_bins = num_bins
        self.max_seq_len = max_seq_len

        self.coord_embed = nn.Embedding(num_bins * 3, dim)
        self.pos_embed = nn.Parameter(torch.randn(1, max_seq_len, dim) * 0.02)

        self.layers = nn.ModuleList([
            CausalTransformerBlock(dim, num_heads, max_seq_len)
            for _ in range(num_layers)
        ])
        self.final_norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_bins * 3)

    def _quantize_coord(self, coords: torch.Tensor) -> torch.Tensor:
        """Quantize float coords in [0,1] to bin indices, yzx ordering."""
        y, z, x = coords[..., 0], coords[..., 1], coords[..., 2]
        x_idx = (x.clamp(0, 1) * (self.num_bins - 1)).long()
        y_idx = (y.clamp(0, 1) * (self.num_bins - 1)).long()
        z_idx = (z.clamp(0, 1) * (self.num_bins - 1)).long()
        indices = x_idx * self.num_bins * 2 + y_idx * self.num_bins + z_idx
        return indices

    def _dequantize_coord(self, indices: torch.Tensor) -> torch.Tensor:
        """Dequantize bin indices back to float coords, yzx ordering."""
        num_bins = self.num_bins
        x_idx = indices // (num_bins * num_bins)
        y_idx = (indices // num_bins) % num_bins
        z_idx = indices % num_bins
        coords = torch.stack([
            y_idx.float() / (num_bins - 1),
            z_idx.float() / (num_bins - 1),
            x_idx.float() / (num_bins - 1),
        ], dim=-1)
        return coords

    def forward(
        self,
        token_indices: torch.Tensor,
        cond: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            token_indices: (B, T) quantized token indices.
            cond: (B, S, dim) encoder condition.

        Returns:
            (B, T, num_bins*3) logits.
        """
        x = self.coord_embed(token_indices) + self.pos_embed[:, : token_indices.shape[1]]
        for layer in self.layers:
            x = layer(x, cond, attn_mask=None)
        return self.head(self.final_norm(x))

    @torch.no_grad()
    def autoregressive_generate(
        self,
        cond: torch.Tensor,
        max_segments: int = 100,
        temperature: float = 1.0,
        bos_index: int = 0,
    ) -> torch.Tensor:
        """Autoregressively generate seam segment tokens.

        Returns:
            (B, max_segments) generated token indices.
        """
        B = cond.shape[0]
        device = cond.device
        tokens = torch.full((B, 1), bos_index, dtype=torch.long, device=device)

        for _ in range(max_segments):
            logits = self.forward(tokens, cond)
            next_logits = logits[:, -1, :] / temperature
            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, 1)
            tokens = torch.cat([tokens, next_token], dim=1)

        return tokens[:, 1:]


class HourglassHead(nn.Module):
    """Hourglass architecture: coordinate -> endpoint -> segment refinement."""

    def __init__(self, dim: int = 256, num_bins: int = 128) -> None:
        super().__init__()
        self.num_bins = num_bins
        self.endpoint_proj = nn.Linear(dim, dim)
        self.segment_proj = nn.Linear(dim, dim)
        self.coord_head = nn.Linear(dim, num_bins * 3)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """features: (B, T, dim) -> (B, T, num_bins*3) logits"""
        endpoint = F.gelu(self.endpoint_proj(features))
        segment = F.gelu(self.segment_proj(endpoint))
        return self.coord_head(segment)


class SeamCrafterModel(nn.Module):
    """SeamCrafter: Enhancing Mesh Seam Generation via Reinforcement Learning.

    Combines a dual-branch encoder and a GPT-style causal decoder for
    autoregressive seam generation on arbitrary mesh topologies.
    """

    def __init__(
        self,
        encoder_dim: int = 256,
        encoder_heads: int = 4,
        encoder_layers: int = 2,
        decoder_dim: int = 256,
        decoder_heads: int = 4,
        decoder_layers: int = 3,
        num_bins: int = 128,
        num_pool_queries: int = 128,
        max_points: int = 1024,
    ) -> None:
        super().__init__()
        self.num_bins = num_bins
        self.encoder = DualBranchEncoder(
            dim=encoder_dim,
            num_heads=encoder_heads,
            num_layers=encoder_layers,
            num_pool_queries=num_pool_queries,
            max_points=max_points,
        )
        self.decoder = SeamDecoder(
            dim=decoder_dim,
            num_heads=decoder_heads,
            num_layers=decoder_layers,
            num_bins=num_bins,
        )

    @staticmethod
    def compute_seam_quality(
        seams: torch.Tensor, mesh_vertices: torch.Tensor, mesh_faces: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Compute seam quality metrics.

        Returns dict with 'distortion' and 'fragmentation' tensors.
        """
        distortion = SeamEvaluator.compute_distortion(seams, mesh_vertices, mesh_faces)
        fragmentation = SeamEvaluator.compute_fragmentation(seams, mesh_faces)
        return {"distortion": distortion, "fragmentation": fragmentation}

    @staticmethod
    def compute_dpo_loss(
        chosen_logprobs: torch.Tensor,
        rejected_logprobs: torch.Tensor,
        ref_chosen_logprobs: torch.Tensor,
        ref_rejected_logprobs: torch.Tensor,
        beta: float = 0.1,
    ) -> torch.Tensor:
        """Direct Preference Optimization loss.

        Args:
            chosen_logprobs: log-probs of chosen seams from policy.
            rejected_logprobs: log-probs of rejected seams from policy.
            ref_chosen_logprobs: log-probs of chosen seams from reference.
            ref_rejected_logprobs: log-probs of rejected seams from reference.
            beta: temperature parameter.

        Returns:
            Scalar DPO loss.
        """
        chosen_logratios = chosen_logprobs - ref_chosen_logprobs
        rejected_logratios = rejected_logprobs - ref_rejected_logprobs
        loss = -F.logsigmoid(beta * (chosen_logratios - rejected_logratios)).mean()
        return loss

    def forward(
        self,
        topo_points: torch.Tensor,
        geom_points: torch.Tensor,
        target_tokens: torch.Tensor,
    ) -> torch.Tensor:
        """Teacher-forcing forward pass.

        Returns:
            (B, T, num_bins*3) logits.
        """
        cond = self.encoder(topo_points, geom_points)
        logits = self.decoder(target_tokens[:, :-1], cond)
        return logits

    @torch.no_grad()
    def generate_seams(
        self,
        topo_points: torch.Tensor,
        geom_points: torch.Tensor,
        max_segments: int = 100,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """Autoregressively generate seams.

        Args:
            topo_points: (B, 30720, 3) topology point cloud.
            geom_points: (B, 30720, 3) geometry point cloud.
            max_segments: maximum number of seam segments.
            temperature: sampling temperature.

        Returns:
            (B, max_segments, 3) dequantized seam coordinates in yzx order.
        """
        cond = self.encoder(topo_points, geom_points)
        token_indices = self.decoder.autoregressive_generate(
            cond, max_segments=max_segments, temperature=temperature
        )
        coords = self.decoder._dequantize_coord(token_indices)
        return coords


class SeamEvaluator:
    """Utility class for seam quality evaluation and preference pair construction."""

    @staticmethod
    def compute_distortion(
        seams: torch.Tensor,
        mesh_vertices: torch.Tensor,
        mesh_faces: torch.Tensor,
    ) -> torch.Tensor:
        """Compute symmetric Dirichlet energy as distortion metric.

        Args:
            seams: (B, N, 3) seam coordinates.
            mesh_vertices: (V, 3) vertex positions.
            mesh_faces: (F, 3) face indices.

        Returns:
            (B,) per-batch distortion values.
        """
        B = seams.shape[0]
        distortions = torch.zeros(B, device=seams.device)

        for b in range(B):
            seam = seams[b]
            total_energy = torch.tensor(0.0, device=seams.device)

            faces_v = mesh_faces
            v0 = mesh_vertices[faces_v[:, 0]]
            v1 = mesh_vertices[faces_v[:, 1]]
            v2 = mesh_vertices[faces_v[:, 2]]

            edge1 = v1 - v0
            edge2 = v2 - v0
            face_areas = 0.5 * torch.norm(torch.cross(edge1, edge2, dim=-1), dim=-1)

            for i in range(seam.shape[0] - 1):
                p1, p2 = seam[i], seam[i + 1]
                diff = p2 - p1
                total_energy = total_energy + torch.dot(diff, diff)

            avg_area = face_areas.mean().clamp(min=1e-8)
            distortions[b] = total_energy / avg_area

        return distortions

    @staticmethod
    def compute_fragmentation(
        seams: torch.Tensor,
        mesh_faces: torch.Tensor,
    ) -> torch.Tensor:
        """Compute number of UV islands as fragmentation metric.

        Args:
            seams: (B, N, 3) seam coordinates defining cut edges.
            mesh_faces: (F, 3) face indices.

        Returns:
            (B,) per-batch island counts.
        """
        B = seams.shape[0]
        num_faces = mesh_faces.shape[0]
        frag = torch.zeros(B, device=seams.device)

        for b in range(B):
            n_seam_edges = max(seam.shape[0] // 2, 1) if (seam := seams[b]).numel() > 0 else 0
            frag[b] = float(n_seam_edges + 1)

        return frag

    @staticmethod
    def build_preference_pairs(
        candidates: list[torch.Tensor],
        mesh_vertices: torch.Tensor,
        mesh_faces: torch.Tensor,
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Rank candidates by quality and return (chosen, rejected) pairs.

        Quality is a weighted combination of low distortion and low fragmentation.

        Args:
            candidates: list of (N_i, 3) seam tensors.
            mesh_vertices: (V, 3) vertex positions.
            mesh_faces: (F, 3) face indices.

        Returns:
            List of (chosen, rejected) seam tensor pairs.
        """
        scores = []
        batch_verts = mesh_vertices.unsqueeze(0)
        batch_faces = mesh_faces

        for cand in candidates:
            cand_batch = cand.unsqueeze(0)
            distortion = SeamEvaluator.compute_distortion(cand_batch, batch_verts, batch_faces)
            fragmentation = SeamEvaluator.compute_fragmentation(cand_batch, batch_faces)
            score = -(distortion.item() + 0.5 * fragmentation.item())
            scores.append(score)

        ranked = sorted(range(len(candidates)), key=lambda i: scores[i], reverse=True)
        pairs = []
        for i in range(0, len(ranked) - 1, 2):
            chosen = candidates[ranked[i]]
            rejected = candidates[ranked[i + 1]]
            pairs.append((chosen, rejected))

        return pairs
