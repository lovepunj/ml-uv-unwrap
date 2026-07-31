from __future__ import annotations

"""DA-Wand style quality selector for UV parameterizations.

Inspired by the DA-Wand approach of using a learned distortion-aware
network to evaluate and select among competing UV unwraps.  The module
provides both a lightweight neural quality scorer and a heuristic
analytic metric that together enable automatic selection of the best
unwrap from a pool of candidates produced by different backends.

Architecture:
    Per-vertex feature vector (UV coords, 3D coords, normals, curvature)
    → Fourier positional encoding → shared MLP → per-vertex quality →
    attention-weighted global pooling → scalar quality score in [0, 1].
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .positional_encoding import PositionalEncoding


class QualitySelectorNet(nn.Module):
    """Lightweight quality scorer for UV parameterizations.

    Evaluates how well a 2D UV parameterization represents the input
    3D surface by jointly considering conformality, area distortion,
    and local geometric context.  The network operates in a point-wise
    fashion so it generalises to arbitrary mesh sizes.

    Args:
        hidden_dim: Width of the shared MLP layers.
        num_layers: Number of hidden layers in the encoder.
        num_freqs: Number of Fourier frequencies for positional encoding.
        dropout: Dropout rate inside the MLP.
    """

    def __init__(
        self,
        hidden_dim: int = 96,
        num_layers: int = 4,
        num_freqs: int = 6,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Per-vertex input: uv(2) + pos(3) + normal(3) + curvature(1) = 9
        raw_input_dim = 9
        self.pos_enc = PositionalEncoding(num_freqs=num_freqs, include_input=True)
        enc_dim = raw_input_dim * self.pos_enc.output_dim

        # Shared point-wise encoder
        layers: list[nn.Module] = []
        for i in range(num_layers):
            in_d = enc_dim if i == 0 else hidden_dim
            layers.extend([
                nn.Linear(in_d, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
        self.encoder = nn.Sequential(*layers)

        # Per-vertex quality logit
        self.vertex_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )

        # Attention-weighted global pooling
        self.attn_pool = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.Tanh(),
            nn.Linear(hidden_dim // 4, 1),
        )

        # Global refinement after pooling
        self.global_head = nn.Sequential(
            nn.Linear(hidden_dim + 1, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(
        self,
        uv: torch.Tensor,
        verts: torch.Tensor,
        normals: torch.Tensor,
        curvature: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Score a UV parameterization.

        Args:
            uv: (B, N, 2) UV coordinates.
            verts: (B, N, 3) corresponding 3D vertex positions.
            normals: (B, N, 3) per-vertex surface normals.
            curvature: (B, N, 1) per-vertex curvature magnitude.

        Returns:
            Dictionary with:
                - ``score``: (B,) global quality in [0, 1].
                - ``vertex_scores``: (B, N) per-vertex quality in [0, 1].
        """
        # Assemble per-vertex feature vector
        features = torch.cat([uv, verts, normals, curvature], dim=-1)  # (B, N, 9)
        encoded = self.pos_enc(features)  # (B, N, enc_dim)

        # Point-wise encoding
        h = self.encoder(encoded)  # (B, N, D)

        # Per-vertex quality logit → score
        v_logit = self.vertex_head(h).squeeze(-1)  # (B, N)
        vertex_scores = torch.sigmoid(v_logit)  # (B, N)

        # Attention-weighted global pooling
        attn_weights = self.attn_pool(h)  # (B, N, 1)
        attn_weights = F.softmax(attn_weights, dim=1)
        global_feat = (h * attn_weights).sum(dim=1)  # (B, D)

        # Combine global feature with mean vertex score for final score
        mean_v = vertex_scores.mean(dim=1, keepdim=True)  # (B, 1)
        combined = torch.cat([global_feat, mean_v], dim=-1)  # (B, D+1)
        score = torch.sigmoid(self.global_head(combined).squeeze(-1))  # (B,)

        return {"score": score, "vertex_scores": vertex_scores}


# ---------------------------------------------------------------------------
# Analytic quality metrics
# ---------------------------------------------------------------------------

def _face_areas(verts: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
    """Compute per-face areas via cross product.

    Args:
        verts: (V, 3) vertex positions.
        faces: (F, 3) face indices.

    Returns:
        (F,) face areas.
    """
    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]
    return 0.5 * torch.cross(v1 - v0, v2 - v0, dim=-1).norm(dim=-1)


def _uv_face_areas(uv: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
    """Compute per-face areas in UV space.

    Args:
        uv: (V, 2) UV coordinates.
        faces: (F, 3) face indices.

    Returns:
        (F,) UV face areas.
    """
    u0 = uv[faces[:, 0]]
    u1 = uv[faces[:, 1]]
    u2 = uv[faces[:, 2]]
    # 2D cross product magnitude
    return 0.5 * ((u1[:, 0] - u0[:, 0]) * (u2[:, 1] - u0[:, 1])
                  - (u2[:, 0] - u0[:, 0]) * (u1[:, 1] - u0[:, 1])).abs()


def _edge_lengths(verts: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
    """Compute per-face edge lengths in 3D.

    Args:
        verts: (V, 3) vertex positions.
        faces: (F, 3) face indices.

    Returns:
        (F, 3) edge lengths for each face.
    """
    e0 = (verts[faces[:, 1]] - verts[faces[:, 0]]).norm(dim=-1)
    e1 = (verts[faces[:, 2]] - verts[faces[:, 1]]).norm(dim=-1)
    e2 = (verts[faces[:, 0]] - verts[faces[:, 2]]).norm(dim=-1)
    return torch.stack([e0, e1, e2], dim=-1)


def _uv_edge_lengths(uv: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
    """Compute per-face edge lengths in UV space.

    Args:
        uv: (V, 2) UV coordinates.
        faces: (F, 3) face indices.

    Returns:
        (F, 3) UV edge lengths for each face.
    """
    e0 = (uv[faces[:, 1]] - uv[faces[:, 0]]).norm(dim=-1)
    e1 = (uv[faces[:, 2]] - uv[faces[:, 1]]).norm(dim=-1)
    e2 = (uv[faces[:, 0]] - uv[faces[:, 2]]).norm(dim=-1)
    return torch.stack([e0, e1, e2], dim=-1)


def compute_conformality(
    verts: torch.Tensor,
    uv: torch.Tensor,
    faces: torch.Tensor,
) -> torch.Tensor:
    """Angle-based conformality metric (L2 angular distortion).

    Measures how well angles are preserved between 3D and UV.  A value
    of 1 means perfect angle preservation.

    Uses the symmetric angular difference approach: for each face the
    three angles in 3D and UV are computed and the sum of squared
    differences is normalised.

    Args:
        verts: (V, 3) mesh vertices.
        uv: (V, 2) UV coordinates.
        faces: (F, 3) face indices.

    Returns:
        Scalar conformality score in [0, 1].
    """
    def _face_angles(pts: torch.Tensor) -> torch.Tensor:
        """Compute angles at each vertex of every face.

        Args:
            pts: (F, 3, 3) – the three vertices of each face.

        Returns:
            (F, 3) angles in radians.
        """
        v01 = pts[:, 1] - pts[:, 0]
        v02 = pts[:, 2] - pts[:, 0]
        v12 = pts[:, 2] - pts[:, 1]
        v10 = -v01
        v20 = -v02
        v21 = -v12

        def _angle(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            cos = F.cosine_similarity(a, b, dim=-1).clamp(-1, 1)
            return torch.acos(cos)

        a0 = _angle(v01, v02)
        a1 = _angle(v10, v12)
        a2 = _angle(v20, v21)
        return torch.stack([a0, a1, a2], dim=-1)

    # 3D angles
    pts_3d = verts[faces]  # (F, 3, 3)
    ang_3d = _face_angles(pts_3d)

    # UV angles (extend to 3D with z=0)
    uv_exp = torch.cat([uv[faces], torch.zeros(*faces.shape, 1, device=uv.device)], dim=-1)
    ang_uv = _face_angles(uv_exp)

    # L2 angular distortion per face, normalised by π²
    diff = ((ang_3d - ang_uv) ** 2).sum(dim=-1)  # (F,)
    # Max possible sum of squared diffs is 3 * π²
    score_per_face = 1.0 - diff / (3.0 * math.pi ** 2)
    return score_per_face.clamp(0, 1).mean()


def compute_equiareality(
    verts: torch.Tensor,
    uv: torch.Tensor,
    faces: torch.Tensor,
) -> torch.Tensor:
    """Area-based equiareality metric.

    Measures how uniformly areas are mapped.  Perfect equiareality
    means the ratio UV_area / 3D_area is constant across all faces.

    Returns:
        Scalar score in [0, 1] where 1 is perfect area preservation.
    """
    area_3d = _face_areas(verts, faces)  # (F,)
    area_uv = _uv_face_areas(uv, faces)  # (F,)

    # Per-face ratio (normalised so ideal ratio ≈ 1)
    ratio = area_uv / (area_3d + 1e-12)
    # Use coefficient of variation: low CV → uniform
    mean_r = ratio.mean()
    std_r = ratio.std()
    cv = std_r / (mean_r + 1e-12)
    # Map to [0, 1] via exponential decay
    return torch.exp(-cv)


def compute_seam_length(
    uv: torch.Tensor,
    faces: torch.Tensor,
    seam_edges: torch.Tensor | None = None,
) -> float:
    """Compute total seam length in UV space.

    Seam edges are edges shared by two faces whose UV counterparts
    are disconnected (the two half-edges in the unfolded mesh).  When
    an explicit ``seam_edges`` tensor is not provided, an approximation
    is used: any edge where the two endpoint UVs differ by more than
    half the bounding-box diagonal is treated as a seam.

    Args:
        uv: (V, 2) UV coordinates.
        faces: (F, 3) face indices.
        seam_edges: Optional (E, 2) explicit seam edge vertex indices.

    Returns:
        Normalised seam length (ratio to bounding-box diagonal).
    """
    device = uv.device

    if seam_edges is not None and seam_edges.shape[0] > 0:
        edge_uv_len = (uv[seam_edges[:, 0]] - uv[seam_edges[:, 1]]).norm(dim=-1)
    else:
        # Build edge→face adjacency to find boundary / UV-discontinuous edges
        edge_map: dict[tuple[int, int], list[int]] = {}
        faces_cpu = faces.cpu()
        for fi in range(faces_cpu.shape[0]):
            for j in range(3):
                a = int(faces_cpu[fi, j])
                b = int(faces_cpu[fi, (j + 1) % 3])
                key = (min(a, b), max(a, b))
                edge_map.setdefault(key, []).append(fi)

        seam_pairs: list[list[int]] = []
        for (a, b), face_list in edge_map.items():
            if len(face_list) == 2:
                # Check UV discontinuity
                uv_a0, uv_b0 = uv[a], uv[b]
                if (uv_a0 - uv_b0).norm() > 1e-6:
                    seam_pairs.append([a, b])
            elif len(face_list) == 1:
                # Boundary edge is always a seam
                seam_pairs.append([a, b])

        if len(seam_pairs) == 0:
            return 0.0

        seam_idx = torch.tensor(seam_pairs, dtype=torch.long, device=device)
        edge_uv_len = (uv[seam_idx[:, 0]] - uv[seam_idx[:, 1]]).norm(dim=-1)

    total_seam = edge_uv_len.sum().item()

    # Normalise by bounding-box diagonal
    uv_min = uv.min(dim=0).values
    uv_max = uv.max(dim=0).values
    diag = (uv_max - uv_min).norm().item() + 1e-12
    return total_seam / diag


def compute_chart_compactness(uv: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
    """Measure how compact UV charts are (convex-hull fill ratio).

    A compact chart has a high area ratio between the actual chart area
    and its axis-aligned bounding box.  This is approximated per-face
    by comparing triangle area to its bounding-box area.

    Returns:
        Scalar compactness score in [0, 1].
    """
    uv_f = uv[faces]  # (F, 3, 2)

    # Triangle area
    tri_area = 0.5 * ((uv_f[:, 1, 0] - uv_f[:, 0, 0]) * (uv_f[:, 2, 1] - uv_f[:, 0, 1])
                      - (uv_f[:, 2, 0] - uv_f[:, 0, 0]) * (uv_f[:, 1, 1] - uv_f[:, 0, 1])).abs()

    # Bounding box area per triangle
    bb_min = uv_f.min(dim=1).values  # (F, 2)
    bb_max = uv_f.max(dim=1).values  # (F, 2)
    bb_area = ((bb_max - bb_min).prod(dim=-1)).clamp(min=1e-12)  # (F,)

    ratio = tri_area / bb_area  # (F,)
    # Average ratio; ideal equilateral fills ~0.5 of its AABB
    return ratio.mean()


def compute_heuristic_quality(
    verts: torch.Tensor,
    uv: torch.Tensor,
    faces: torch.Tensor,
    seam_edges: torch.Tensor | None = None,
    weights: tuple[float, float, float, float] = (0.3, 0.3, 0.2, 0.2),
) -> dict[str, float]:
    """Compute a combined heuristic quality score.

    Args:
        verts: (V, 3) mesh vertices.
        uv: (V, 2) UV coordinates.
        faces: (F, 3) face indices.
        seam_edges: Optional explicit seam edge indices.
        weights: (w_conform, w_area, w_seam, w_compact) weighting.

    Returns:
        Dictionary with individual metrics and combined ``quality`` in [0, 1].
    """
    conf = compute_conformality(verts, uv, faces).item()
    area = compute_equiareality(verts, uv, faces).item()
    seam = compute_seam_length(uv, faces, seam_edges)
    comp = compute_chart_compactness(uv, faces).item()

    # Seam contribution: lower seam length → higher score
    seam_score = max(0.0, 1.0 - seam)

    w_conf, w_area, w_seam, w_comp = weights
    quality = (w_conf * conf + w_area * area + w_seam * seam_score + w_comp * comp)

    return {
        "conformality": conf,
        "equiareality": area,
        "seam_length": seam,
        "compactness": comp,
        "quality": quality,
    }


# ---------------------------------------------------------------------------
# Selection API
# ---------------------------------------------------------------------------

def _build_features_from_mesh(
    uv: torch.Tensor,
    verts: torch.Tensor,
    faces: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Derive normals and curvature from mesh geometry.

    Args:
        uv: (V, 2) UV coordinates.
        verts: (V, 3) vertices.
        faces: (F, 3) face indices.

    Returns:
        Dictionary with ``uv``, ``verts``, ``normals``, ``curvature``
        all shaped (1, V, …) ready for batching.
    """
    device = verts.device
    V = verts.shape[0]

    # Per-vertex normals via face normal averaging
    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]
    face_normals = torch.cross(v1 - v0, v2 - v0, dim=-1)
    face_normals = F.normalize(face_normals, dim=-1)

    normals = torch.zeros(V, 3, device=device)
    counts = torch.zeros(V, 1, device=device)
    for ch in range(3):
        normals.index_add_(0, faces[:, 0], face_normals[:, ch : ch + 1])
        normals.index_add_(0, faces[:, 1], face_normals[:, ch : ch + 1])
        normals.index_add_(0, faces[:, 2], face_normals[:, ch : ch + 1])
    counts.index_add_(0, faces[:, 0], torch.ones(faces.shape[0], 1, device=device))
    counts.index_add_(0, faces[:, 1], torch.ones(faces.shape[0], 1, device=device))
    counts.index_add_(0, faces[:, 2], torch.ones(faces.shape[0], 1, device=device))
    normals = normals / counts.clamp(min=1)
    normals = F.normalize(normals, dim=-1)

    # Approximate curvature via normal variation across faces
    curvature = torch.zeros(V, 1, device=device)
    for fi in range(faces.shape[0]):
        for j in range(3):
            idx = faces[fi, j]
            neighbours = faces[fi].tolist()
            diffs = (face_normals[fi].unsqueeze(0) - face_normals[faces[fi]].abs()).norm(dim=-1)
            curvature[idx] += diffs.mean()
    counts_flat = counts.squeeze(-1).clamp(min=1)
    curvature = curvature / counts_flat.unsqueeze(-1)
    # Normalise to roughly [0, 1]
    c_max = curvature.max() + 1e-8
    curvature = curvature / c_max

    return {
        "uv": uv.unsqueeze(0),
        "verts": verts.unsqueeze(0),
        "normals": normals.unsqueeze(0),
        "curvature": curvature,
    }


def select_best_unwrap(
    candidates: list[dict],
    mesh: object,
    net: QualitySelectorNet | None = None,
    device: str | torch.device = "cpu",
) -> dict:
    """Pick the best UV unwrap from a list of candidates.

    Each candidate is a dictionary with at least a ``"uv"`` key holding
    ``(V, 2)`` UV coordinates.  Optionally candidates may carry a
    ``"name"`` key for identification.

    The selection combines the neural quality score (when ``net`` is
    provided) with the analytic heuristic score.  If ``net`` is ``None``
    only the heuristic is used.

    Args:
        candidates: List of candidate dicts, each with ``"uv"`` tensor.
        mesh: A mesh object exposing ``vertices`` (V, 3) and ``faces`` (F, 3)
            attributes (e.g. a ``trimesh.Trimesh`` or similar).
        net: Optional pre-trained ``QualitySelectorNet``.
        device: Device for inference.

    Returns:
        Dictionary with:
            - ``index``: index of the selected candidate.
            - ``uv``: the winning UV tensor.
            - ``score``: combined quality score.
            - ``details``: per-candidate heuristic breakdowns.
    """
    if len(candidates) == 0:
        raise ValueError("At least one candidate is required.")

    verts = torch.as_tensor(mesh.vertices, dtype=torch.float32, device=device)
    faces = torch.as_tensor(mesh.faces, dtype=torch.long, device=device)

    best_idx = 0
    best_score = -1.0
    all_details: list[dict[str, float]] = []

    net.eval() if net is not None else None

    for i, cand in enumerate(candidates):
        uv = torch.as_tensor(cand["uv"], dtype=torch.float32, device=device)

        # Heuristic score
        h = compute_heuristic_quality(verts, uv, faces)
        heuristic_score = h["quality"]

        # Neural score
        if net is not None:
            feats = _build_features_from_mesh(uv, verts, faces)
            with torch.no_grad():
                out = net(feats["uv"], feats["verts"], feats["normals"], feats["curvature"])
            neural_score = out["score"].item()
            # Combine: 60 % neural + 40 % heuristic
            combined = 0.6 * neural_score + 0.4 * heuristic_score
        else:
            combined = heuristic_score

        h["combined_score"] = combined
        all_details.append(h)

        if combined > best_score:
            best_score = combined
            best_idx = i

    return {
        "index": best_idx,
        "uv": candidates[best_idx]["uv"],
        "score": best_score,
        "details": all_details,
    }
