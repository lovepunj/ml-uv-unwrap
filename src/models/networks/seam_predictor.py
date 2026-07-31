from __future__ import annotations

"""SeamGPT-inspired auto-regressive seam predictor.

Reimplementation based on the SIGGRAPH 2025 paper:
"SeamGPT: High-quality Seam Generation for 3D Surface Parameterization"

Architecture:
- Point cloud encoder (sample points on vertices and edges)
- GPT-style causal transformer decoder
- Auto-regressive generation of seam segments as quantized 3D coordinates
- Beam search for high-quality seam line selection
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class PointCloudEncoder(nn.Module):
    """Encode point cloud with local + global features.

    Follows the paper's approach:
    1. Sample points on vertices (original) and edges (augmented)
    2. Multi-scale PointNet encoding
    3. Global feature aggregation
    """

    def __init__(self, input_dim: int = 3, hidden_dim: int = 256, num_freqs: int = 6):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Positional encoding
        freqs = torch.arange(num_freqs).float()
        self.register_buffer("freqs", freqs)

        enc_dim = input_dim * (num_freqs * 2 + 1)

        # Local feature extractor (shared MLP)
        self.local_mlp = nn.Sequential(
            nn.Linear(enc_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Global feature aggregation
        self.global_attn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.Tanh(),
            nn.Linear(hidden_dim // 4, 1),
        )

        # Final projection
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, points: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode point cloud.

        Args:
            points: (B, N, 3) point cloud

        Returns:
            local_features: (B, N, hidden_dim) per-point features
            global_feature: (B, hidden_dim) global feature
        """
        # Positional encoding
        pts_expanded = points.unsqueeze(-1) * (2 ** self.freqs)  # (B, N, 3, F)
        pts_sin = torch.sin(pts_expanded)
        pts_cos = torch.cos(pts_expanded)
        enc = torch.cat([
            points,
            pts_sin.reshape(*points.shape[:-1], -1),
            pts_cos.reshape(*points.shape[:-1], -1),
        ], dim=-1)  # (B, N, 3*(2F+1))

        # Local features
        local_feat = self.local_mlp(enc)  # (B, N, D)

        # Global feature via attention pooling
        attn_weights = self.global_attn(local_feat)  # (B, N, 1)
        attn_weights = F.softmax(attn_weights, dim=1)
        global_feat = (local_feat * attn_weights).sum(dim=1)  # (B, D)

        # Broadcast global feature
        global_broadcast = global_feat.unsqueeze(1).expand_as(local_feat)
        combined = local_feat + global_broadcast

        return self.output_proj(combined), global_feat


class CausalTransformerBlock(nn.Module):
    """Single transformer decoder block with causal masking."""

    def __init__(self, hidden_dim: int = 256, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.self_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.cross_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm3 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        causal_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Self-attention with causal mask
        residual = x
        x = self.norm1(x)
        x, _ = self.self_attn(x, x, x, attn_mask=causal_mask)
        x = x + residual

        # Cross-attention to encoder memory
        residual = x
        x = self.norm2(x)
        x, _ = self.cross_attn(x, memory, memory)
        x = x + residual

        # Feed-forward
        residual = x
        x = self.norm3(x)
        x = self.ffn(x) + residual

        return x


class SeamGPTPredictor(nn.Module):
    """Auto-regressive seam predictor inspired by SeamGPT.

    Predicts seam line segments as a sequence of 3D coordinates on the
    mesh surface, using a GPT-style transformer decoder.
    """

    def __init__(
        self,
        vocab_size: int = 1024,
        hidden_dim: int = 256,
        num_layers: int = 6,
        num_heads: int = 8,
        max_seq_len: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len

        # Point cloud encoder
        self.encoder = PointCloudEncoder(input_dim=3, hidden_dim=hidden_dim)

        # Token embedding (quantized 3D coordinates)
        self.token_embed = nn.Embedding(vocab_size, hidden_dim)

        # Positional embedding
        self.pos_embed = nn.Embedding(max_seq_len, hidden_dim)

        # Transformer decoder blocks
        self.blocks = nn.ModuleList([
            CausalTransformerBlock(hidden_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(hidden_dim)

        # Output heads
        self.token_head = nn.Linear(hidden_dim, vocab_size)
        self.stop_head = nn.Linear(hidden_dim, 1)

        # Coordinate quantization
        self.coord_min = -1.0
        self.coord_max = 1.0

    def _create_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Create causal attention mask."""
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
        mask = mask.masked_fill(mask == 1, float("-inf"))
        return mask

    def _quantize_coords(self, coords: torch.Tensor) -> torch.Tensor:
        """Quantize 3D coordinates to discrete tokens."""
        # Normalize to [0, 1]
        normalized = (coords - self.coord_min) / (self.coord_max - self.coord_min)
        # Quantize
        tokens = (normalized * (self.vocab_size - 1)).long()
        return tokens.clamp(0, self.vocab_size - 1)

    def _dequantize_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        """Convert tokens back to 3D coordinates."""
        normalized = tokens.float() / (self.vocab_size - 1)
        coords = normalized * (self.coord_max - self.coord_min) + self.coord_min
        return coords

    def forward(
        self,
        points: torch.Tensor,
        seam_tokens: torch.Tensor | None = None,
        temperature: float = 1.0,
    ) -> dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            points: (B, N, 3) point cloud
            seam_tokens: (B, T) existing seam tokens (for teacher forcing)
            temperature: sampling temperature

        Returns:
            Dictionary with predictions and generated sequences
        """
        B = points.shape[0]

        # Encode point cloud
        memory, global_feat = self.encoder(points)  # (B, N, D), (B, D)

        if seam_tokens is not None:
            # Teacher forcing mode
            T = seam_tokens.shape[1]
            token_emb = self.token_embed(seam_tokens)
            positions = torch.arange(T, device=points.device).unsqueeze(0)
            pos_emb = self.pos_embed(positions)
            x = token_emb + pos_emb

            causal_mask = self._create_causal_mask(T, points.device)

            for block in self.blocks:
                x = block(x, memory, causal_mask)

            x = self.norm(x)
            logits = self.token_head(x)
            stop_logits = self.stop_head(x).squeeze(-1)

            return {
                "logits": logits,
                "stop_logits": stop_logits,
                "predictions": logits.argmax(dim=-1),
            }
        else:
            # Auto-regressive generation
            return self.generate(points, temperature=temperature)

    def generate(
        self,
        points: torch.Tensor,
        max_length: int | None = None,
        temperature: float = 1.0,
        stop_threshold: float = 0.5,
    ) -> dict[str, torch.Tensor]:
        """Auto-regressively generate seam tokens.

        Args:
            points: (B, N, 3) point cloud
            max_length: maximum sequence length
            temperature: sampling temperature
            stop_threshold: threshold for stop token

        Returns:
            Dictionary with generated tokens and stop probabilities
        """
        B = points.shape[0]
        max_len = max_length or self.max_seq_len

        # Encode point cloud
        memory, global_feat = self.encoder(points)

        # Start with a learnable start token
        start_token = torch.zeros(B, 1, dtype=torch.long, device=points.device)
        generated = start_token

        all_logits = []
        all_stop_logits = []

        for step in range(max_len):
            T = generated.shape[1]
            token_emb = self.token_embed(generated)
            positions = torch.arange(T, device=points.device).unsqueeze(0)
            pos_emb = self.pos_embed(positions)
            x = token_emb + pos_emb

            causal_mask = self._create_causal_mask(T, points.device)

            for block in self.blocks:
                x = block(x, memory, causal_mask)

            x = self.norm(x)

            # Get predictions for last position
            logits = self.token_head(x[:, -1:, :])  # (B, 1, vocab)
            stop_logits = self.stop_head(x[:, -1:, :]).squeeze(-1)  # (B, 1)

            all_logits.append(logits)
            all_stop_logits.append(stop_logits)

            # Sample next token
            probs = F.softmax(logits[:, 0, :] / temperature, dim=-1)
            next_token = torch.multinomial(probs, 1)  # (B, 1)
            generated = torch.cat([generated, next_token], dim=1)

            # Check early stopping
            if (torch.sigmoid(stop_logits[:, 0]) > stop_threshold).all():
                break

        return {
            "generated_tokens": generated[:, 1:],  # Remove start token
            "logits": torch.cat(all_logits, dim=1),
            "stop_logits": torch.cat(all_stop_logits, dim=1),
            "generated_coords": self._dequantize_tokens(generated[:, 1:]),
        }

    def compute_seam_loss(
        self,
        points: torch.Tensor,
        target_seams: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Compute training loss for seam prediction.

        Args:
            points: (B, N, 3) point cloud
            target_seams: (B, T) target seam token sequences

        Returns:
            Dictionary of losses
        """
        outputs = self.forward(points, seam_tokens=target_seams)

        logits = outputs["logits"]
        stop_logits = outputs["stop_logits"]

        # Token prediction loss (cross-entropy)
        targets = target_seams[:, 1:]  # Shift by one
        if logits.shape[1] > targets.shape[1]:
            logits = logits[:, :targets.shape[1], :]

        token_loss = F.cross_entropy(
            logits.reshape(-1, self.vocab_size),
            targets.reshape(-1),
            ignore_index=-1,
        )

        # Stop token loss
        # Binary cross-entropy: 0 for non-last, 1 for last
        stop_targets = torch.zeros_like(stop_logits)
        for b in range(stop_logits.shape[0]):
            # Find the actual length (non-zero tokens)
            lengths = (target_seams[b] != 0).sum().item()
            if lengths > 0 and lengths <= stop_logits.shape[1]:
                stop_targets[b, lengths - 1] = 1.0

        stop_loss = F.binary_cross_entropy_with_logits(stop_logits, stop_targets)

        return {
            "seam_token_loss": token_loss,
            "seam_stop_loss": stop_loss,
            "seam_total_loss": token_loss + 0.5 * stop_loss,
        }
