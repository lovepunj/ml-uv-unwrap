from __future__ import annotations

"""FlexPara-based UV unwrapping model."""

import torch
import torch.nn as nn

from ..losses.ao_visibility import AOVisibilityModule
from ..losses.chamfer import chamfer_distance, repulsion_loss
from ..losses.cycle import cycle_consistency_loss
from ..losses.distortion import distortion_loss
from .base import UVUnwrapModel
from .networks import CutNet, DeformNet, UnwrapNet, WrapNet


class FlexParaUnwrapper(UVUnwrapModel):
    """FlexPara-inspired neural UV unwrapping model.

    Uses a bi-directional cycle mapping framework with four
    geometrically-interpretable sub-networks:
    - CutNet: learns seam placement
    - DeformNet: deforms UV grid
    - UnwrapNet: 3D → 2D mapping
    - WrapNet: 2D → 3D reconstruction

    Supports both single-chart (global) and multi-chart parameterization.
    """

    def __init__(
        self,
        num_charts: int = 1,
        point_dim: int = 3,
        hidden_dim: int = 128,
        num_layers: int = 5,
        num_freqs: int = 6,
        partfield_dim: int = 0,
    ):
        super().__init__()
        self.num_charts = num_charts
        self.point_dim = point_dim
        self.partfield_dim = partfield_dim

        # Core sub-networks (ensure minimum layer counts)
        self.cut_net = CutNet(
            point_dim=point_dim,
            hidden_dim=hidden_dim,
            num_layers=max(num_layers, 2),
            num_freqs=num_freqs,
            partfield_dim=partfield_dim,
        )
        self.deform_net = DeformNet(
            in_dim=2,
            hidden_dim=hidden_dim,
            num_layers=max(num_layers - 2, 1),
            num_freqs=num_freqs,
        )
        self.unwrap_net = UnwrapNet(
            point_dim=point_dim,
            uv_dim=2,
            hidden_dim=hidden_dim,
            num_layers=max(num_layers, 2),
            num_freqs=num_freqs,
        )
        self.wrap_net = WrapNet(
            uv_dim=2,
            point_dim=point_dim,
            hidden_dim=hidden_dim,
            num_layers=max(num_layers, 2),
            num_freqs=num_freqs,
        )

        # Chart assignment network (for multi-chart mode)
        if num_charts > 1:
            self.chart_net = nn.Sequential(
                nn.Linear(point_dim * (num_freqs * 2 + 1), hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, num_charts),
            )
            self.chart_pos_enc = nn.Identity()  # reuse cut_net's
        else:
            self.chart_net = None

        # AO visibility module (for seam placement in crevices)
        self.ao_module = AOVisibilityModule(num_rays=16)

    def _initial_uv(self, points: torch.Tensor) -> torch.Tensor:
        """Compute initial UV coordinates via PCA projection."""
        # points: (B, N, 3)
        centered = points - points.mean(dim=1, keepdim=True)
        # Simple PCA: project onto first two principal axes
        cov = torch.bmm(centered.transpose(1, 2), centered)  # (B, 3, 3)
        eigvals, eigvecs = torch.linalg.eigh(cov)  # ascending order
        # Take top 2 eigenvectors
        axes = eigvecs[:, :, -2:]  # (B, 3, 2)
        uv = torch.bmm(centered, axes)  # (B, N, 2)
        # Normalize to [0, 1]
        uv_min = uv.min(dim=1, keepdim=True).values
        uv_max = uv.max(dim=1, keepdim=True).values
        uv = (uv - uv_min) / (uv_max - uv_min + 1e-8)
        return uv

    def forward(
        self,
        points: torch.Tensor,
        partfield_features: torch.Tensor | None = None,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        """Full forward pass.

        Args:
            points: (B, N, 3) surface points
            partfield_features: (B, N, 448) optional PartField features

        Returns:
            Dictionary containing UV coords, seam logits, chart assignments, etc.
        """
        B, N, _ = points.shape

        # 1. Predict seam probabilities (with optional PartField conditioning)
        seam_logits = self.cut_net(points, partfield_features=partfield_features)  # (B, N, 1)

        # 2. Get initial UV via PCA
        uv_init = self._initial_uv(points)

        # 3. Deform initial UV grid
        uv_deformed = self.deform_net(uv_init)  # (B, N, 2)

        # 4. Learn the unwrap mapping
        uv_pred = self.unwrap_net(points)  # (B, N, 2)

        # 5. Reconstruct 3D from predicted UV (cycle)
        recon_3d = self.wrap_net(uv_pred)  # (B, N, 3)

        # 6. Reconstruct UV from reconstructed 3D (cycle)
        recon_uv = self.unwrap_net(recon_3d)  # (B, N, 2)

        outputs = {
            "uv_coords": uv_pred,
            "uv_deformed": uv_deformed,
            "seam_logits": seam_logits,
            "reconstructed_3d": recon_3d,
            "reconstructed_uv": recon_uv,
        }

        # 7. Chart assignment for multi-chart
        if self.chart_net is not None:
            from .networks.positional_encoding import PositionalEncoding
            pos_enc = PositionalEncoding(num_freqs=6)
            encoded = pos_enc(points)
            chart_logits = self.chart_net(encoded)  # (B, N, num_charts)
            chart_probs = torch.softmax(chart_logits, dim=-1)
            outputs["chart_logits"] = chart_logits
            outputs["chart_probs"] = chart_probs

        return outputs

    def unwrap(self, points: torch.Tensor, **kwargs) -> torch.Tensor:
        """Return predicted UV coordinates."""
        return self.forward(points, **kwargs)["uv_coords"]

    def compute_losses(
        self,
        points: torch.Tensor,
        outputs: dict[str, torch.Tensor],
        edges: torch.Tensor | None = None,
        faces: torch.Tensor | None = None,
        weights: dict[str, float] | None = None,
        ao_values: torch.Tensor | None = None,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        """Compute all training losses."""
        w = {
            "cycle_3d": 1.0,
            "cycle_2d": 1.0,
            "distortion": 1.0,
            "chamfer": 0.5,
            "repulsion": 0.1,
            "laplacian": 0.5,
            "ao_visibility": 0.3,
        }
        if weights:
            w.update(weights)

        losses = {}

        # Cycle consistency loss
        cycle_loss = cycle_consistency_loss(
            points_3d=points,
            reconstructed_3d=outputs["reconstructed_3d"],
            uv_coords=outputs["uv_coords"],
            reconstructed_uv=outputs["reconstructed_uv"],
        )
        losses["cycle"] = cycle_loss

        # Chamfer distance (reconstruction quality)
        cd = chamfer_distance(points, outputs["reconstructed_3d"])
        losses["chamfer"] = cd

        # Distortion loss (if mesh connectivity available)
        if edges is not None and faces is not None:
            batch_dist = []
            for i in range(points.shape[0]):
                batch_dist.append(
                    distortion_loss(points[i], outputs["uv_coords"][i], edges, faces)
                )
            dist_loss = torch.stack(batch_dist).mean()
            losses["distortion"] = dist_loss

        # Repulsion loss (prevent point collapse)
        rep = repulsion_loss(outputs["uv_coords"])
        losses["repulsion"] = rep

        # Laplacian smoothing loss on UV coordinates
        if edges is not None:
            uv = outputs["uv_coords"]
            max_edge_idx = edges.max().item()
            if max_edge_idx < uv.shape[1]:
                v0 = uv[:, edges[:, 0]]
                v1 = uv[:, edges[:, 1]]
                edge_len = (v1 - v0).norm(dim=-1)  # (B, E)
                losses["laplacian"] = edge_len.mean()
            else:
                losses["laplacian"] = torch.tensor(0.0, device=points.device)
        else:
            losses["laplacian"] = torch.tensor(0.0, device=points.device)

        # AO visibility loss (encourage seams in crevices)
        losses["ao_visibility"] = self.ao_module(
            points, outputs["seam_logits"], ao_values
        )

        # Chart assignment regularization (multi-chart)
        if "chart_probs" in outputs:
            chart_usage = outputs["chart_probs"].mean(dim=1)
            ideal_usage = 1.0 / self.num_charts
            chart_balance = ((chart_usage - ideal_usage) ** 2).mean()
            losses["chart_balance"] = chart_balance

        # Total weighted loss
        total = (
            w.get("cycle_3d", 1.0) * losses.get("cycle", 0.0)
            + w.get("chamfer", 0.5) * losses.get("chamfer", 0.0)
            + w.get("distortion", 1.0) * losses.get("distortion", 0.0)
            + w.get("repulsion", 0.1) * losses.get("repulsion", 0.0)
            + w.get("laplacian", 0.5) * losses.get("laplacian", 0.0)
            + w.get("ao_visibility", 0.3) * losses.get("ao_visibility", 0.0)
            + 0.01 * losses.get("chart_balance", 0.0)
        )
        losses["total"] = total

        return losses
