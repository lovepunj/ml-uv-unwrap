"""Flatten Anything Model (FAM) — bi-directional cycle mapping for UV unwrapping.

Implements the framework from "Flatten Anything: Unwrapping 3D Surfaces
with Point-Wise Embeddings" (NeurIPS 2024).

Key design principles:
- Point-wise operation (no mesh connectivity needed)
- All sub-networks use Point-Wise Embeddings (PWE) = Conv1d(1×1)
- Bi-directional cycle ensures consistent 3D↔2D mapping
- Free-boundary parameterization (no fixed boundary constraints)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .positional_encoding import PositionalEncoding


class DeformNet2D(nn.Module):
    """Deforms a 2D UV grid using residual learning.

    Takes initial UV coordinates and outputs a residual deformation
    to optimize the parameterization. Uses Fourier encoding followed
    by an MLP with skip connections.

    Architecture: Fourier encoding → MLP with skip connections
    MLP: in_dim → 512 → 512 → 512 → 66 → 66 → 512 → 512 → 512 → in_dim
    """

    def __init__(
        self,
        in_dim: int = 2,
        hidden_dim: int = 512,
        bottleneck_dim: int = 66,
        num_freqs: int = 6,
    ):
        """
        Args:
            in_dim: Input/output dimension (2 for UV coordinates)
            hidden_dim: Hidden layer width
            bottleneck_dim: Bottleneck layer width
            num_freqs: Number of Fourier encoding frequencies
        """
        super().__init__()
        self.pos_enc = PositionalEncoding(num_freqs=num_freqs, include_input=True)
        input_dim = in_dim * self.pos_enc.output_dim

        # Encoder
        self.enc1 = nn.Conv1d(input_dim, hidden_dim, 1)
        self.enc2 = nn.Conv1d(hidden_dim, hidden_dim, 1)
        self.enc3 = nn.Conv1d(hidden_dim, hidden_dim, 1)

        # Bottleneck
        self.bottleneck1 = nn.Conv1d(hidden_dim, bottleneck_dim, 1)
        self.bottleneck2 = nn.Conv1d(bottleneck_dim, bottleneck_dim, 1)

        # Decoder
        self.dec1 = nn.Conv1d(bottleneck_dim, hidden_dim, 1)
        self.dec2 = nn.Conv1d(hidden_dim, hidden_dim, 1)
        self.dec3 = nn.Conv1d(hidden_dim, hidden_dim, 1)

        # Output projection
        self.out_proj = nn.Conv1d(hidden_dim, in_dim, 1)

        self.act = nn.ReLU(inplace=True)

    def forward(self, uv_coords: torch.Tensor) -> torch.Tensor:
        """
        Args:
            uv_coords: (B, N, 2) or (N, 2) UV coordinates
        Returns:
            deformed_uv: (B, N, 2) or (N, 2) deformed UV coordinates (residual added)
        """
        need_squeeze = uv_coords.dim() == 2
        if need_squeeze:
            uv_coords = uv_coords.unsqueeze(0)

        # Fourier encoding → (B, N, input_dim)
        encoded = self.pos_enc(uv_coords)
        x = encoded.transpose(1, 2)  # (B, input_dim, N)

        # Encoder with skip connections
        e1 = self.act(self.enc1(x))
        e2 = self.act(self.enc2(e1))
        e3 = self.act(self.enc3(e2))

        # Bottleneck
        b1 = self.act(self.bottleneck1(e3))
        b2 = self.act(self.bottleneck2(b1))

        # Decoder with skip connections
        d1 = self.act(self.dec1(b2) + e3)
        d2 = self.act(self.dec2(d1) + e2)
        d3 = self.act(self.dec3(d2) + e1)

        residual = self.out_proj(d3).transpose(1, 2)  # (B, N, 2)

        result = uv_coords + residual

        if need_squeeze:
            result = result.squeeze(0)
        return result


class WrapNet2D(nn.Module):
    """Maps 2D UV coordinates to 3D surface points with normals.

    The wrapping function g: R^2 → R^6, reconstructing 3D positions
    and normals from UV coordinates. Uses Fourier encoding followed
    by an MLP with skip connections.
    """

    def __init__(
        self,
        uv_dim: int = 2,
        out_dim: int = 6,
        hidden_dim: int = 512,
        bottleneck_dim: int = 66,
        num_freqs: int = 6,
    ):
        """
        Args:
            uv_dim: Input UV dimension (2)
            out_dim: Output dimension (6 = 3D coords + 3D normals)
            hidden_dim: Hidden layer width
            bottleneck_dim: Bottleneck layer width
            num_freqs: Number of Fourier encoding frequencies
        """
        super().__init__()
        self.pos_enc = PositionalEncoding(num_freqs=num_freqs, include_input=True)
        input_dim = uv_dim * self.pos_enc.output_dim

        # Encoder
        self.enc1 = nn.Conv1d(input_dim, hidden_dim, 1)
        self.enc2 = nn.Conv1d(hidden_dim, hidden_dim, 1)
        self.enc3 = nn.Conv1d(hidden_dim, hidden_dim, 1)

        # Bottleneck
        self.bottleneck1 = nn.Conv1d(hidden_dim, bottleneck_dim, 1)
        self.bottleneck2 = nn.Conv1d(bottleneck_dim, bottleneck_dim, 1)

        # Decoder
        self.dec1 = nn.Conv1d(bottleneck_dim, hidden_dim, 1)
        self.dec2 = nn.Conv1d(hidden_dim, hidden_dim, 1)
        self.dec3 = nn.Conv1d(hidden_dim, hidden_dim, 1)

        # Output projection
        self.out_proj = nn.Conv1d(hidden_dim, out_dim, 1)

        self.act = nn.ReLU(inplace=True)

    def forward(self, uv_coords: torch.Tensor) -> torch.Tensor:
        """
        Args:
            uv_coords: (B, N, 2) or (N, 2) UV coordinates
        Returns:
            points_normals: (B, N, 6) or (N, 6) 3D points concatenated with normals
        """
        need_squeeze = uv_coords.dim() == 2
        if need_squeeze:
            uv_coords = uv_coords.unsqueeze(0)

        encoded = self.pos_enc(uv_coords)
        x = encoded.transpose(1, 2)

        e1 = self.act(self.enc1(x))
        e2 = self.act(self.enc2(e1))
        e3 = self.act(self.enc3(e2))

        b1 = self.act(self.bottleneck1(e3))
        b2 = self.act(self.bottleneck2(b1))

        d1 = self.act(self.dec1(b2) + e3)
        d2 = self.act(self.dec2(d1) + e2)
        d3 = self.act(self.dec3(d2) + e1)

        out = self.out_proj(d3).transpose(1, 2)

        if need_squeeze:
            out = out.squeeze(0)
        return out


class UnwrapNet2D(nn.Module):
    """Maps 3D surface points to 2D UV coordinates.

    The forward parameterization function f: R^3 → R^2, mapping
    surface points to UV space. Uses Fourier encoding followed
    by an MLP with skip connections.
    """

    def __init__(
        self,
        point_dim: int = 3,
        uv_dim: int = 2,
        hidden_dim: int = 512,
        bottleneck_dim: int = 66,
        num_freqs: int = 6,
    ):
        """
        Args:
            point_dim: Input point dimension (3 for xyz)
            uv_dim: Output UV dimension (2)
            hidden_dim: Hidden layer width
            bottleneck_dim: Bottleneck layer width
            num_freqs: Number of Fourier encoding frequencies
        """
        super().__init__()
        self.pos_enc = PositionalEncoding(num_freqs=num_freqs, include_input=True)
        input_dim = point_dim * self.pos_enc.output_dim

        # Encoder
        self.enc1 = nn.Conv1d(input_dim, hidden_dim, 1)
        self.enc2 = nn.Conv1d(hidden_dim, hidden_dim, 1)
        self.enc3 = nn.Conv1d(hidden_dim, hidden_dim, 1)

        # Bottleneck
        self.bottleneck1 = nn.Conv1d(hidden_dim, bottleneck_dim, 1)
        self.bottleneck2 = nn.Conv1d(bottleneck_dim, bottleneck_dim, 1)

        # Decoder
        self.dec1 = nn.Conv1d(bottleneck_dim, hidden_dim, 1)
        self.dec2 = nn.Conv1d(hidden_dim, hidden_dim, 1)
        self.dec3 = nn.Conv1d(hidden_dim, hidden_dim, 1)

        # Output projection
        self.out_proj = nn.Conv1d(hidden_dim, uv_dim, 1)

        self.act = nn.ReLU(inplace=True)

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        """
        Args:
            points: (B, N, 3) or (N, 3) 3D surface points
        Returns:
            uv: (B, N, 2) or (N, 2) UV coordinates
        """
        need_squeeze = points.dim() == 2
        if need_squeeze:
            points = points.unsqueeze(0)

        encoded = self.pos_enc(points)
        x = encoded.transpose(1, 2)

        e1 = self.act(self.enc1(x))
        e2 = self.act(self.enc2(e1))
        e3 = self.act(self.enc3(e2))

        b1 = self.act(self.bottleneck1(e3))
        b2 = self.act(self.bottleneck2(b1))

        d1 = self.act(self.dec1(b2) + e3)
        d2 = self.act(self.dec2(d1) + e2)
        d3 = self.act(self.dec3(d2) + e1)

        uv = self.out_proj(d3).transpose(1, 2)

        if need_squeeze:
            uv = uv.squeeze(0)
        return uv


class CutNet2D(nn.Module):
    """Predicts seam probabilities on the 3D surface.

    The cutting function predicts per-point seam logits to determine
    where to place UV seams. Uses Fourier encoding followed by an MLP
    with skip connections and sigmoid activation.
    """

    def __init__(
        self,
        point_dim: int = 3,
        hidden_dim: int = 512,
        num_layers: int = 6,
        num_freqs: int = 6,
    ):
        """
        Args:
            point_dim: Input point dimension (3 for xyz)
            hidden_dim: Hidden layer width
            num_layers: Number of MLP layers
            num_freqs: Number of Fourier encoding frequencies
        """
        super().__init__()
        self.pos_enc = PositionalEncoding(num_freqs=num_freqs, include_input=True)
        input_dim = point_dim * self.pos_enc.output_dim

        # PWE layers
        self.pwe_in = nn.Conv1d(input_dim, hidden_dim, 1)

        # MLP layers with skip connections
        self.mlp_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.mlp_layers.append(nn.Conv1d(hidden_dim, hidden_dim, 1))

        # Output projection
        self.out_proj = nn.Conv1d(hidden_dim, 1, 1)

        self.act = nn.ReLU(inplace=True)

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        """
        Args:
            points: (B, N, 3) or (N, 3) 3D surface points
        Returns:
            seam_logits: (B, N, 1) or (N, 1) raw seam logits (apply sigmoid for probabilities)
        """
        need_squeeze = points.dim() == 2
        if need_squeeze:
            points = points.unsqueeze(0)

        encoded = self.pos_enc(points)
        x = encoded.transpose(1, 2)

        h = self.act(self.pwe_in(x))

        # MLP with skip connections
        for i, layer in enumerate(self.mlp_layers):
            h = self.act(layer(h) + h if i > 0 else layer(h))

        seam_logits = self.out_proj(h).transpose(1, 2)

        if need_squeeze:
            seam_logits = seam_logits.squeeze(0)
        return seam_logits


class FlattenAnythingModel(nn.Module):
    """Full Flatten Anything Model combining all sub-networks.

    Implements bi-directional cycle mapping between 3D surfaces and 2D UV space.

    Forward cycle: 3D points → Cut → Unwrap → 2D UV
    Backward cycle: 2D grid → Deform → Wrap → 3D points → Unwrap → 2D UV cycle

    Shared parameters between matching forward/backward paths ensure consistency.
    """

    def __init__(
        self,
        hidden_dim: int = 512,
        bottleneck_dim: int = 66,
        num_freqs: int = 6,
    ):
        """
        Args:
            hidden_dim: Hidden layer width for all sub-networks
            bottleneck_dim: Bottleneck layer width for encoder-decoder networks
            num_freqs: Number of Fourier encoding frequencies
        """
        super().__init__()

        # Sub-networks
        self.deform_net = DeformNet2D(
            in_dim=2,
            hidden_dim=hidden_dim,
            bottleneck_dim=bottleneck_dim,
            num_freqs=num_freqs,
        )
        self.wrap_net = WrapNet2D(
            uv_dim=2,
            out_dim=6,
            hidden_dim=hidden_dim,
            bottleneck_dim=bottleneck_dim,
            num_freqs=num_freqs,
        )
        self.unwrap_net = UnwrapNet2D(
            point_dim=3,
            uv_dim=2,
            hidden_dim=hidden_dim,
            bottleneck_dim=bottleneck_dim,
            num_freqs=num_freqs,
        )
        self.cut_net = CutNet2D(
            point_dim=3,
            hidden_dim=hidden_dim,
            num_layers=6,
            num_freqs=num_freqs,
        )

    def forward(
        self,
        points: torch.Tensor,
        init_uv: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            points: (B, N, 3) 3D surface points
            init_uv: (B, N, 2) initial UV coordinates (e.g., from PCA).
                     If None, uses zeros.

        Returns:
            Dictionary containing:
                - 'seam_logits': (B, N, 1) seam predictions
                - 'uv_forward': (B, N, 2) UV from forward cycle (3D → 2D)
                - 'uv_backward': (B, N, 2) UV from backward cycle (2D → 3D → 2D)
                - 'points_reconstructed': (B, N, 3) points from backward cycle
                - 'points_normals': (B, N, 6) wrapped points with normals
        """
        B, N, _ = points.shape

        if init_uv is None:
            init_uv = torch.zeros(B, N, 2, device=points.device, dtype=points.dtype)

        # Forward cycle: 3D points → Cut → Unwrap → 2D UV
        seam_logits = self.cut_net(points)
        uv_forward = self.unwrap_net(points)

        # Backward cycle: 2D grid → Deform → Wrap → 3D points → Unwrap → 2D UV
        uv_deformed = self.deform_net(init_uv)
        points_normals = self.wrap_net(uv_deformed)
        points_reconstructed = points_normals[..., :3]
        uv_cycle = self.unwrap_net(points_reconstructed)

        return {
            "seam_logits": seam_logits,
            "uv_forward": uv_forward,
            "uv_backward": uv_deformed,
            "points_reconstructed": points_reconstructed,
            "points_normals": points_normals,
            "uv_cycle": uv_cycle,
        }

    def compute_losses(
        self,
        points: torch.Tensor,
        outputs: dict[str, torch.Tensor],
        lambda_cycle: float = 1.0,
        lambda_chamfer: float = 1.0,
        lambda_distortion: float = 1.0,
        lambda_repulsion: float = 1.0,
        lambda_wrap: float = 1.0,
    ) -> dict[str, torch.Tensor]:
        """Compute all losses for training.

        Args:
            points: (B, N, 3) input 3D points
            outputs: Dictionary from forward() method
            lambda_cycle: Weight for cycle consistency loss
            lambda_chamfer: Weight for chamfer distance loss
            lambda_distortion: Weight for distortion loss
            lambda_repulsion: Weight for repulsion loss
            lambda_wrap: Weight for wrapping loss

        Returns:
            Dictionary of named losses and total loss
        """
        B, N, _ = points.shape
        losses: dict[str, torch.Tensor] = {}

        # 1. Cycle consistency loss
        # Forward UV should be close to backward UV cycle
        losses["cycle"] = lambda_cycle * F.mse_loss(outputs["uv_forward"], outputs["uv_cycle"])

        # 2. Chamfer distance loss
        # Reconstructed points should match input points
        pts_pred = outputs["points_reconstructed"]
        # Bidirectional chamfer
        dist_fwd = torch.cdist(points, pts_pred)  # (B, N, N)
        losses["chamfer"] = lambda_chamfer * (
            dist_fwd.min(dim=2)[0].mean() + dist_fwd.min(dim=1)[0].mean()
        )

        # 3. Distortion loss
        # Preserve local distances in UV space
        # Use k-nearest neighbors for local distortion instead of full N×N quantile
        with torch.no_grad():
            k_local = min(16, N - 1)
            dist_3d = torch.cdist(points, points)
            knn_idx_3d = torch.topk(dist_3d, k_local + 1, dim=-1, largest=False).indices[..., 1:]  # (B, N, k_local)
        dist_uv = torch.cdist(outputs["uv_forward"], outputs["uv_forward"])
        local_uv_dist = torch.gather(dist_uv, 2, knn_idx_3d)  # (B, N, k_local)
        with torch.no_grad():
            local_3d_dist = torch.gather(dist_3d, 2, knn_idx_3d)  # (B, N, k_local)
        losses["distortion"] = lambda_distortion * ((local_uv_dist - local_3d_dist) ** 2).mean()

        # 4. Repulsion loss
        # Encourage uniform sampling in UV space
        with torch.no_grad():
            k = min(6, N - 1)
            knn_dist = torch.topk(dist_uv, k + 1, dim=-1, largest=False)[0][..., 1:]
        losses["repulsion"] = lambda_repulsion * (-knn_dist.mean())

        # 5. Wrapping loss
        # Normals from wrap_net should be consistent
        normals = F.normalize(outputs["points_normals"][..., 3:6], dim=-1)
        losses["wrapping"] = lambda_wrap * (1.0 - torch.abs(normals).sum(dim=-1).mean())

        # Total loss
        losses["total"] = sum(v for k, v in losses.items() if k != "total")

        return losses
