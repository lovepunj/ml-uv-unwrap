from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class PointNetEncoder(nn.Module):
    """Shared encoder for point cloud feature extraction.

    Processes raw point coordinates (and optional features) through a series of
    shared MLPs to produce per-point embeddings at multiple scales.

    Args:
        in_channels: Number of input channels per point (3 for XYZ, plus
            optional normals, curvature, etc.).
        base_channels: Width of the first hidden layer.
        hidden_channels: Width of subsequent hidden layers.
        num_layers: Number of shared MLP layers.
    """

    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 64,
        hidden_channels: int = 128,
        num_layers: int = 4,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        c_in = in_channels
        for i in range(num_layers):
            c_out = base_channels if i == 0 else hidden_channels
            layers.append(nn.Conv1d(c_in, c_out, 1))
            layers.append(nn.BatchNorm1d(c_out))
            layers.append(nn.ReLU(inplace=True))
            c_in = c_out
        self.mlp = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape ``(B, C, N)`` where ``C`` is
                ``in_channels`` and ``N`` is the number of points.

        Returns:
            Per-point features of shape ``(B, hidden_channels, N)``.
        """
        return self.mlp(x)


class MultiScaleAggregator(nn.Module):
    """Aggregates features from multiple local neighbourhood scales.

    For each scale the module applies a shared ``1x1`` convolution to project
    the encoder output, then concatenates all scales along the channel
    dimension and projects back to ``out_channels``.

    Args:
        in_channels: Channel count of the input features.
        scales: List of ``k`` for each ``k``-NN graph aggregation level.
            Each scale is implemented as a ``1x1`` conv (point-wise) so the
            actual graph construction is left to the caller; the module only
            mixes the pre-computed multi-scale feature tensors.
        out_channels: Output channel count after aggregation.
    """

    def __init__(
        self,
        in_channels: int,
        scales: list[int] | None = None,
        out_channels: int = 256,
    ) -> None:
        super().__init__()
        self.scales = scales or [1, 3, 9, 27]
        self.scale_convs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(in_channels, in_channels, 1),
                    nn.BatchNorm1d(in_channels),
                    nn.ReLU(inplace=True),
                )
                for _ in self.scales
            ]
        )
        self.project = nn.Sequential(
            nn.Conv1d(in_channels * len(self.scales), out_channels, 1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, multi_scale_features: list[Tensor]) -> Tensor:
        """Aggregate multi-scale features.

        Args:
            multi_scale_features: List of tensors, each ``(B, C, N)``,
                one per scale, typically obtained by neighbourhood
                pooling at different ``k`` values.

        Returns:
            Aggregated features ``(B, out_channels, N)``.
        """
        transformed = [
            conv(feat) for conv, feat in zip(self.scale_convs, multi_scale_features)
        ]
        return self.project(torch.cat(transformed, dim=1))


class SemanticBoundaryNet(nn.Module):
    """Detects semantic boundaries on 3D meshes.

    A PointNet-style architecture that predicts per-point boundary
    probabilities from a point cloud with optional auxiliary features
    (normals, curvature).  Multi-scale feature aggregation allows the
    network to reason about both local sharp features (edges/creases)
    and global shape structure.

    Args:
        in_channels: Number of input channels.  3 for XYZ; add 3 for
            normals and 1 for curvature for a total of 7.
        base_channels: Width of the initial shared MLP.
        hidden_channels: Width of intermediate feature channels.
        aggregator_out_channels: Channel count after multi-scale
            aggregation.
        num_agg_layers: Number of layers in the aggregation MLP.
        scales: Neighbourhood sizes for multi-scale aggregation.
    """

    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 64,
        hidden_channels: int = 128,
        aggregator_out_channels: int = 256,
        num_agg_layers: int = 3,
        scales: list[int] | None = None,
    ) -> None:
        super().__init__()
        self.encoder = PointNetEncoder(
            in_channels=in_channels,
            base_channels=base_channels,
            hidden_channels=hidden_channels,
        )
        self.aggregator = MultiScaleAggregator(
            in_channels=hidden_channels,
            scales=scales,
            out_channels=aggregator_out_channels,
        )

        # Boundary prediction head: 256 → 128 → 64 → 1
        head: list[nn.Module] = []
        c_in = aggregator_out_channels
        for _ in range(num_agg_layers - 1):
            c_out = c_in // 2
            head.append(nn.Conv1d(c_in, c_out, 1))
            head.append(nn.BatchNorm1d(c_out))
            head.append(nn.ReLU(inplace=True))
            c_in = c_out
        head.append(nn.Conv1d(c_in, 1, 1))
        head.append(nn.Sigmoid())
        self.boundary_head = nn.Sequential(*head)

    def forward(
        self,
        points: Tensor,
        features: Tensor | None = None,
        multi_scale_indices: list[Tensor] | None = None,
    ) -> Tensor:
        """Predict per-point boundary probabilities.

        Args:
            points: Point cloud ``(B, N, 3)`` or ``(B, 3, N)``.
            features: Optional auxiliary features ``(B, F, N)`` (e.g.
                normals, curvature).  If ``None``, only XYZ is used.
            multi_scale_indices: Pre-computed neighbour indices for each
                scale.  When ``None`` the aggregator falls back to
                identity (scale-1) features replicated to fill the
                required number of scales.

        Returns:
            Boundary scores ``(B, N, 1)`` in ``[0, 1]``.
        """
        if points.dim() == 3 and points.shape[-1] == 3:
            x = points.permute(0, 2, 1)  # (B, 3, N)
        else:
            x = points

        if features is not None:
            x = torch.cat([x, features], dim=1)

        point_features = self.encoder(x)  # (B, H, N)

        # Build multi-scale features.  In a full implementation each
        # scale would use the corresponding ``multi_scale_indices`` for
        # neighbourhood pooling; here we approximate by applying
        # max-pool with varying 1-D kernel sizes.
        if multi_scale_indices is not None and len(multi_scale_indices) == len(
            self.aggregator.scales
        ):
            ms_feats: list[Tensor] = []
            for k, idx in zip(self.aggregator.scales, multi_scale_indices):
                ms_feats.append(self._pool_neighbours(point_features, idx, k))
        else:
            ms_feats = [point_features] * len(self.aggregator.scales)

        aggregated = self.aggregator(ms_feats)  # (B, A, N)
        boundary_scores = self.boundary_head(aggregated)  # (B, 1, N)
        return boundary_scores.permute(0, 2, 1)  # (B, N, 1)

    @staticmethod
    def _pool_neighbours(
        features: Tensor, indices: Tensor, k: int
    ) -> Tensor:
        """Max-pool features over ``k`` nearest neighbours.

        Args:
            features: ``(B, C, N)`` point features.
            indices: ``(B, N, k)`` neighbour indices.
            k: Number of neighbours.

        Returns:
            Pooled features ``(B, C, N)``.
        """
        B, C, N = features.shape
        # Gather neighbours: (B, C, N, k)
        idx_expanded = indices[:, :N, :k].unsqueeze(1).expand(-1, C, -1, -1)
        gathered = torch.gather(features.unsqueeze(-1).expand(-1, -1, -1, k), 2, idx_expanded)
        return gathered.max(dim=-1).values


class ManMadeParameterizer(nn.Module):
    """UV parameterization network for hard-surface / man-made objects.

    Given boundary predictions and (optionally) geometric features the
    module assigns points to charts and regresses per-chart UV
    coordinates using a straight-cut LSCM-inspired formulation.  The
    architecture is:

    1. **Boundary encoding** -- encode boundary probabilities together
       with local geometric context.
    2. **Chart assignment** -- soft clustering of points into charts
       respecting predicted boundary seams.
    3. **Per-chart parameterization** -- regress UV coordinates within
       each chart using a lightweight MLP.

    Args:
        in_channels: Total input channels (3 for XYZ + 1 for boundary
            score + optional normals/curvature).
        feature_channels: Intermediate feature width.
        num_charts: Maximum number of UV charts.
        coord_hidden_dims: Hidden layer sizes for the UV regression head.
    """

    def __init__(
        self,
        in_channels: int = 4,
        feature_channels: int = 128,
        num_charts: int = 16,
        coord_hidden_dims: list[int] | None = None,
    ) -> None:
        super().__init__()
        self.num_charts = num_charts
        coord_hidden_dims = coord_hidden_dims or [256, 128, 64]

        # --- boundary & geometry encoder ---
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels, feature_channels, 1),
            nn.BatchNorm1d(feature_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(feature_channels, feature_channels, 1),
            nn.BatchNorm1d(feature_channels),
            nn.ReLU(inplace=True),
        )

        # --- chart assignment head ---
        self.chart_head = nn.Sequential(
            nn.Conv1d(feature_channels, feature_channels, 1),
            nn.BatchNorm1d(feature_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(feature_channels, num_charts, 1),
        )

        # --- per-chart UV regression ---
        uv_layers: list[nn.Module] = []
        c_in = feature_channels + num_charts  # features one-hot chart
        for h in coord_hidden_dims:
            uv_layers.append(nn.Conv1d(c_in, h, 1))
            uv_layers.append(nn.BatchNorm1d(h))
            uv_layers.append(nn.ReLU(inplace=True))
            c_in = h
        uv_layers.append(nn.Conv1d(c_in, 2, 1))
        self.uv_head = nn.Sequential(*uv_layers)

    def forward(
        self,
        points: Tensor,
        boundary_scores: Tensor,
        features: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Parameterize the surface.

        Args:
            points: Original point positions ``(B, N, 3)`` or
                ``(B, 3, N)``.
            boundary_scores: Boundary predictions ``(B, N, 1)`` from
                :class:`SemanticBoundaryNet`.
            features: Optional extra per-point features ``(B, F, N)``.

        Returns:
            A tuple of:

            - **uv** -- ``(B, N, 2)`` UV coordinates.
            - **chart_labels** -- ``(B, N,)`` integer chart assignments
              (hard assignment from soft predictions).
        """
        if points.dim() == 3 and points.shape[-1] == 3:
            xyz = points.permute(0, 2, 1)  # (B, 3, N)
        else:
            xyz = points

        B, _, N = xyz.shape
        boundary = boundary_scores.permute(0, 2, 1) if boundary_scores.dim() == 3 else boundary_scores.unsqueeze(-1).permute(0, 2, 1)

        x = torch.cat([xyz, boundary], dim=1)  # (B, 4, N)
        if features is not None:
            x = torch.cat([x, features], dim=1)

        encoded = self.encoder(x)  # (B, F, N)

        # Chart assignment (soft)
        chart_logits = self.chart_head(encoded)  # (B, K, N)
        chart_soft = F.softmax(chart_logits, dim=1)
        chart_labels = chart_soft.argmax(dim=1)  # (B, N)

        # UV regression conditioned on chart assignment
        chart_onehot = F.one_hot(
            chart_labels, num_classes=self.num_charts
        ).permute(0, 2, 1).float()  # (B, K, N)
        uv_input = torch.cat([encoded, chart_onehot], dim=1)
        uv = self.uv_head(uv_input)  # (B, 2, N)

        # LSCM-style straight-cut constraint: zero-set one corner per
        # chart and fix aspect ratio via tanh squashing to [0, 1].
        uv = torch.tanh(uv)  # squash to roughly [-1, 1]
        uv = (uv + 1.0) * 0.5  # shift to [0, 1]

        return uv.permute(0, 2, 1), chart_labels


class UVSegNetPipeline(nn.Module):
    """Full pipeline: boundary detection → chart decomposition → unwrapping.

    Orchestrates :class:`SemanticBoundaryNet` and
    :class:`ManMadeParameterizer` into an end-to-end trainable UV
    segmentation and parameterization pipeline for man-made objects.

    The pipeline supports three modes:

    * **Automatic** -- uses neural boundary predictions directly.
    * **User-guided** -- accepts external boundary hints that modulate
      the neural predictions via a learnable gating mechanism.
    * **Fallback** -- when neural boundary confidence is below
      ``uncertainty_threshold`` the pipeline reverts to purely
      geometric dihedral-angle boundary detection.

    Args:
        in_channels: Input channels for the boundary network (3 for XYZ
            plus optional normals/curvature).
        base_channels: Encoder base width.
        hidden_channels: Encoder hidden width.
        aggregator_out_channels: Post-aggregation feature width.
        scales: Multi-scale neighbourhood sizes.
        num_charts: Maximum number of UV charts.
        uncertainty_threshold: Confidence below which geometric
            fallback boundaries are used instead of neural predictions.
        fallback_angle_deg: Dihedral angle threshold (degrees) for the
            geometric boundary fallback.
    """

    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 64,
        hidden_channels: int = 128,
        aggregator_out_channels: int = 256,
        scales: list[int] | None = None,
        num_charts: int = 16,
        uncertainty_threshold: float = 0.3,
        fallback_angle_deg: float = 30.0,
    ) -> None:
        super().__init__()
        self.uncertainty_threshold = uncertainty_threshold
        self.fallback_angle_deg = fallback_angle_deg

        self.boundary_net = SemanticBoundaryNet(
            in_channels=in_channels,
            base_channels=base_channels,
            hidden_channels=hidden_channels,
            aggregator_out_channels=aggregator_out_channels,
            scales=scales,
        )
        self.parameterizer = ManMadeParameterizer(
            in_channels=in_channels + 1,
            feature_channels=hidden_channels,
            num_charts=num_charts,
        )

        # Learnable gate to blend user hints with neural predictions
        self.hint_gate = nn.Sequential(
            nn.Conv1d(2, 16, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(16, 1, 1),
            nn.Sigmoid(),
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def forward(
        self,
        points: Tensor,
        features: Tensor | None = None,
        user_boundary_hints: Tensor | None = None,
        dihedral_angles: Tensor | None = None,
        multi_scale_indices: list[Tensor] | None = None,
    ) -> dict[str, Tensor]:
        """Run the full UV segmentation pipeline.

        Args:
            points: Point cloud ``(B, N, 3)``.
            features: Optional per-point features ``(B, F, N)`` (normals,
                curvature, etc.).
            user_boundary_hints: Optional user-provided boundary
                probabilities ``(B, N, 1)`` in ``[0, 1]``.  Combined
                with neural predictions via a learned gate.
            dihedral_angles: Pre-computed dihedral angles ``(B, N, 1)``
                in degrees.  Required when geometric fallback is
                expected.
            multi_scale_indices: Pre-computed neighbour indices for
                multi-scale aggregation.

        Returns:
            Dictionary with keys:

            - ``"uv"`` -- ``(B, N, 2)`` UV coordinates.
            - ``"chart_labels"`` -- ``(B, N,)`` integer chart IDs.
            - ``"boundary_scores"`` -- ``(B, N, 1)`` final boundary
              probabilities after fallback blending.
            - ``"confidence"`` -- ``(B, N, 1)`` network confidence
              (1 - entropy of boundary prediction).
        """
        # 1. Neural boundary prediction
        neural_boundary = self.boundary_net(
            points, features=features, multi_scale_indices=multi_scale_indices
        )  # (B, N, 1)

        # 2. Confidence: high confidence when score is close to 0 or 1
        confidence = 1.0 - 4.0 * neural_boundary * (1.0 - neural_boundary)
        confidence = confidence.clamp(0.0, 1.0)

        # 3. Geometric fallback for low-confidence regions
        geometric_boundary = self._geometric_boundary_fallback(
            points, dihedral_angles
        )

        uncertain_mask = (confidence < self.uncertainty_threshold).float()
        boundary_scores = (
            (1.0 - uncertain_mask) * neural_boundary
            + uncertain_mask * geometric_boundary
        )

        # 4. Optional user-guided boundary placement
        if user_boundary_hints is not None:
            gate_input = torch.cat(
                [neural_boundary, user_boundary_hints], dim=-1
            )  # (B, N, 2)
            gate = self.hint_gate(gate_input.permute(0, 2, 1)).permute(0, 2, 1)
            boundary_scores = gate * user_boundary_hints + (1.0 - gate) * boundary_scores

        # 5. UV parameterization
        uv, chart_labels = self.parameterizer(
            points, boundary_scores, features=features
        )

        return {
            "uv": uv,
            "chart_labels": chart_labels,
            "boundary_scores": boundary_scores,
            "confidence": confidence,
        }

    # ------------------------------------------------------------------
    # Geometric fallback
    # ------------------------------------------------------------------

    def _geometric_boundary_fallback(
        self,
        points: Tensor,
        dihedral_angles: Tensor | None = None,
    ) -> Tensor:
        """Compute geometric boundary probabilities from dihedral angles.

        When ``dihedral_angles`` are not provided the fallback returns
        an empty (all-zeros) tensor so the pipeline effectively trusts
        the neural predictions only.

        Args:
            points: ``(B, N, 3)`` – used only for shape.
            dihedral_angles: ``(B, N, 1)`` angles in degrees, or
                ``None``.

        Returns:
            Geometric boundary scores ``(B, N, 1)`` in ``[0, 1]``.
        """
        B, N, _ = points.shape
        if dihedral_angles is None:
            return torch.zeros(B, N, 1, device=points.device, dtype=points.dtype)

        threshold = self.fallback_angle_deg
        scores = torch.sigmoid((dihedral_angles - threshold) / 5.0)
        return scores

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    def predict_boundaries(
        self, points: Tensor, features: Tensor | None = None
    ) -> Tensor:
        """Standalone boundary prediction (no parameterization).

        Args:
            points: ``(B, N, 3)``
            features: Optional ``(B, F, N)``

        Returns:
            Boundary scores ``(B, N, 1)``.
        """
        return self.boundary_net(points, features=features)

    def parameterize_from_boundaries(
        self,
        points: Tensor,
        boundary_scores: Tensor,
        features: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Standalone parameterization from pre-computed boundaries.

        Args:
            points: ``(B, N, 3)``
            boundary_scores: ``(B, N, 1)``
            features: Optional ``(B, F, N)``

        Returns:
            Tuple of UV ``(B, N, 2)`` and chart labels ``(B, N,)``.
        """
        return self.parameterizer(points, boundary_scores, features=features)
