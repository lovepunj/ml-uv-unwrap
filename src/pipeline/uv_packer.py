from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rotate_rect(
    w: float, h: float, angle: int
) -> Tuple[float, float]:
    """Return (w', h') after rotating a rectangle by *angle* degrees."""
    if angle % 180 == 0:
        return w, h
    return h, w


def _chart_bounds(pts: np.ndarray) -> Tuple[float, float, float, float]:
    """Return (min_x, min_y, max_x, max_y) of a set of 2-D points."""
    mins = pts.min(axis=0)
    maxs = pts.max(axis=0)
    return float(mins[0]), float(mins[1]), float(maxs[0]), float(maxs[1])


def _area_of_chart(uv_coords: np.ndarray, faces: np.ndarray) -> float:
    """Signed area of a triangulated chart (absolute value)."""
    total = 0.0
    for tri in faces:
        p0, p1, p2 = uv_coords[tri[0]], uv_coords[tri[1]], uv_coords[tri[2]]
        total += (p0[0] * (p1[1] - p2[1]) +
                  p1[0] * (p2[1] - p0[1]) +
                  p2[0] * (p0[1] - p1[1]))
    return abs(total) * 0.5


# ---------------------------------------------------------------------------
# Normalization / transformation helpers
# ---------------------------------------------------------------------------

def _normalize_chart(
    uv_coords: np.ndarray, faces: np.ndarray
) -> Tuple[np.ndarray, float, float, float, float]:
    """Scale and translate a chart so that its bounding box fits [0,1]×[0,1].

    Returns (normalised_uvs, scale, offset_x, offset_y, area).
    """
    mn_x, mn_y, mx_x, mx_y = _chart_bounds(uv_coords)
    w = mx_x - mn_x
    h = mx_y - mn_y
    span = max(w, h, 1e-12)
    norm = (uv_coords - np.array([mn_x, mn_y])) / span
    area = _area_of_chart(norm, faces)
    return norm, span, mn_x, mn_y, area


def _denormalize_chart(
    norm_uv: np.ndarray, scale: float, ox: float, oy: float,
    new_offset: np.ndarray, new_scale: float,
) -> np.ndarray:
    """Bring a normalised chart back to atlas space."""
    return norm_uv * new_scale + new_offset


# ---------------------------------------------------------------------------
# Packing strategies
# ---------------------------------------------------------------------------

class _Rect:
    """Lightweight axis-aligned rectangle used by the bin-packers."""
    __slots__ = ("w", "h", "x", "y", "chart_idx")

    def __init__(self, w: float, h: float, chart_idx: int = 0):
        self.w = w
        self.h = h
        self.x = 0.0
        self.y = 0.0
        self.chart_idx = chart_idx


def _pack_simple(rects: List[_Rect], atlas_size: float) -> List[_Rect]:
    """Grid / simple row packing – each chart gets its own grid cell."""
    n = len(rects)
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)
    cell_w = atlas_size / cols
    cell_h = atlas_size / rows
    for i, r in enumerate(rects):
        col = i % cols
        row = i // cols
        r.x = col * cell_w + (cell_w - r.w) * 0.5
        r.y = row * cell_h + (cell_h - r.h) * 0.5
    return rects


def _pack_shelf(rects: List[_Rect], atlas_size: float) -> List[_Rect]:
    """Shelf (row) packing – items are placed left-to-right, top-to-bottom."""
    rects.sort(key=lambda r: -r.h)
    shelf_y = 0.0
    shelf_h = 0.0
    cursor_x = 0.0
    for r in rects:
        if cursor_x + r.w > atlas_size + 1e-9:
            shelf_y += shelf_h
            cursor_x = 0.0
            shelf_h = 0.0
        r.x = cursor_x
        r.y = shelf_y
        cursor_x += r.w
        shelf_h = max(shelf_h, r.h)
    return rects


def _pack_guillotine(rects: List[_Rect], atlas_size: float) -> List[_Rect]:
    """Guillotine bin packing – recursively splits free rectangles."""
    rects.sort(key=lambda r: -(r.w * r.h))
    free: List[Tuple[float, float, float, float]] = [
        (0.0, 0.0, atlas_size, atlas_size)
    ]

    def _find_best(r: _Rect) -> Optional[int]:
        best_idx = None
        best_short = float("inf")
        for i, (fx, fy, fw, fh) in enumerate(free):
            if r.w <= fw + 1e-9 and r.h <= fh + 1e-9:
                short_side = min(fw - r.w, fh - r.h)
                if short_side < best_short:
                    best_short = short_side
                    best_idx = i
        return best_idx

    for r in rects:
        idx = _find_best(r)
        if idx is None:
            continue
        fx, fy, fw, fh = free[idx]
        r.x = fx
        r.y = fy
        free.pop(idx)
        # Horizontal split
        if r.w < fw - 1e-9:
            free.append((fx + r.w, fy, fw - r.w, r.h))
        # Vertical split
        if r.h < fh - 1e-9:
            free.append((fx, fy + r.h, fw, fh - r.h))
    return rects


class _Skyline:
    """Skyline bin-packing helper."""

    def __init__(self, width: float):
        self.width = width
        self.skyline: List[Tuple[float, float, float]] = []  # (x, w, y)

    def _insert(self, x: float, w: float, h: float) -> int:
        seg = (x, w, h)
        idx = len(self.skyline)
        self.skyline.append(seg)
        return idx

    def _merge(self) -> None:
        merged: List[Tuple[float, float, float]] = []
        for seg in self.skyline:
            if merged and abs(merged[-1][2] - seg[2]) < 1e-9 and \
               abs(merged[-1][0] + merged[-1][1] - seg[0]) < 1e-9:
                merged[-1] = (merged[-1][0], merged[-1][1] + seg[1], seg[2])
            else:
                merged.append(seg)
        self.skyline = merged

    def insert(self, w: float, h: float) -> Optional[Tuple[float, float]]:
        best_y = float("inf")
        best_idx = -1
        best_x = 0.0
        for i, (sx, sw, sy) in enumerate(self.skyline):
            if sx + w > self.width + 1e-9:
                continue
            # Check that the gap [sx, sx+w) does not exceed skyline height
            y = sy
            ok = True
            rem = w
            j = i
            while rem > 1e-9:
                if j >= len(self.skyline):
                    ok = False
                    break
                seg_x, seg_w, seg_y = self.skyline[j]
                if j == i:
                    start = sx
                else:
                    start = seg_x
                usable_w = min(seg_w - (start - seg_x), rem)
                y = max(y, seg_y)
                rem -= usable_w
                j += 1
            if not ok:
                continue
            if y < best_y:
                best_y = y
                best_idx = i
                best_x = sx
        if best_idx == -1:
            return None
        # Place the rectangle
        new_seg = (best_x, w, best_y + h)
        # Remove/reduce skyline segments under the placed rect
        new_skyline: List[Tuple[float, float, float]] = []
        placed_end = best_x + w
        for sx, sw, sy in self.skyline:
            seg_end = sx + sw
            if seg_end <= best_x + 1e-9 or sx >= placed_end - 1e-9:
                new_skyline.append((sx, sw, sy))
            else:
                if sx < best_x - 1e-9:
                    new_skyline.append((sx, best_x - sx, sy))
                if seg_end > placed_end + 1e-9:
                    new_skyline.append((placed_end, seg_end - placed_end, sy))
        new_skyline.append(new_seg)
        self.skyline = new_skyline
        self._merge()
        return (best_x, best_y)


def _pack_skyline(rects: List[_Rect], atlas_size: float) -> List[_Rect]:
    """Skyline-based bin packing."""
    rects.sort(key=lambda r: -r.h)
    sl = _Skyline(atlas_size)
    for r in rects:
        pos = sl.insert(r.w, r.h)
        if pos is None:
            # Try swapped orientation is not meaningful here; skip
            continue
        r.x, r.y = pos
    return rects


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def pack_uv_charts_advanced(
    chart_uvs: List[np.ndarray],
    chart_faces: List[np.ndarray],
    all_faces: np.ndarray,
    all_vertices: np.ndarray,
    method: str = "shelf",
    margin: float = 0.01,
) -> Dict[str, object]:
    """Pack multiple UV charts into a single atlas texture space.

    Parameters
    ----------
    chart_uvs:
        Per-chart UV coordinate arrays (list of ``(N, 2)`` float arrays).
    chart_faces:
        Per-chart face index arrays (list of ``(M, 3)`` int arrays) that
        index into the corresponding ``chart_uvs`` entry.
    all_faces:
        Global face index array ``(K, 3)`` indexing ``all_vertices``.
    all_vertices:
        Global vertex array ``(V, 3)``.
    method:
        Packing strategy – one of ``'simple'``, ``'shelf'``,
        ``'guillotine'``, or ``'skyline'``.
    margin:
        Minimum normalised margin between charts (0–1).

    Returns
    -------
    dict
        ``'packed_uvs'`` – ``(V, 2)`` array of re-mapped UV coordinates
        for every vertex in ``all_vertices``.
        ``'chart_transforms'`` – list of per-chart dicts with normalisation
        metadata needed to reverse the transform.
        ``'atlas_size'`` – ``(width, height)`` (normalised square 1×1).
    """
    n_charts = len(chart_uvs)
    if n_charts == 0:
        return {
            "packed_uvs": np.zeros((len(all_vertices), 2), dtype=np.float64),
            "chart_transforms": [],
            "atlas_size": (1.0, 1.0),
        }

    # 1. Normalise each chart and compute bounding rects + areas
    norm_charts: List[np.ndarray] = []
    rects: List[_Rect] = []
    areas: List[float] = []
    raw_bboxes: List[Tuple[float, float, float, float]] = []

    for ci, (uvs, faces) in enumerate(zip(chart_uvs, chart_faces)):
        norm_uv, span, ox, oy, area = _normalize_chart(uvs, faces)
        norm_charts.append(norm_uv)
        areas.append(area)
        raw_bboxes.append((ox, oy, span, 0.0))  # span stored for denorm

        # Without rotation: width & height in normalised space
        mn = norm_uv.min(axis=0)
        mx = norm_uv.max(axis=0)
        w = mx[0] - mn[0]
        h = mx[1] - mn[1]
        rects.append(_Rect(max(w, 1e-6), max(h, 1e-6), ci))

    # Total chart area (normalised) – needed to compute atlas scale
    total_area = sum(areas)
    # Estimate atlas size: sqrt(total_area / 0.9) gives ~90% util target
    atlas_size = math.sqrt(total_area / 0.90)
    atlas_size = max(atlas_size, 1.0)

    # 2. Try every rotation (0, 90, 180, 270) per chart, keep best overall.
    #    For simplicity we only try 0° and 90° rotations per chart since
    #    180°/270° are equivalent for axis-aligned rectangles.
    best_rects: List[_Rect] = list(rects)
    best_waste = float("inf")

    for combo in range(1 << n_charts):
        trial: List[_Rect] = []
        for ci in range(n_charts):
            r = rects[ci]
            if combo & (1 << ci):
                trial.append(_Rect(r.h, r.w, ci))
            else:
                trial.append(_Rect(r.w, r.h, ci))

        # Run the chosen packing algorithm
        packed = _dispatch_pack(method, trial, atlas_size)
        waste = sum(
            atlas_size * atlas_size for _ in []
        )  # placeholder – we measure below
        # Compute occupied bounding box height
        max_y = max(r.y + r.h for r in packed)
        max_x = max(r.x + r.w for r in packed)
        used = max_y * atlas_size  # approximate waste metric
        if max_y < best_waste:
            best_waste = max_y
            best_rects = packed

    # Normalise packed positions into [0, 1]
    max_y = max(r.y + r.h for r in best_rects)
    max_x = max(r.x + r.w for r in best_rects)
    scale_x = 1.0 / max(max_x, 1e-9)
    scale_y = 1.0 / max(max_y, 1e-9)
    scale = min(scale_x, scale_y)

    for r in best_rects:
        r.x *= scale
        r.y *= scale
        r.w *= scale
        r.h *= scale

    # 3. Build the output UV array (all_vertices)
    packed_uvs = np.zeros((len(all_vertices), 2), dtype=np.float64)
    chart_transforms: List[Dict[str, object]] = []

    for ci, (uvs, faces, norm_uv, raw_span, raw_ox, raw_oy, area) in enumerate(
        zip(
            chart_uvs,
            chart_faces,
            norm_charts,
            [b[2] for b in raw_bboxes],
            [b[0] for b in raw_bboxes],
            [b[1] for b in raw_bboxes],
            areas,
        )
    ):
        r = best_rects[ci]
        chart_transforms.append(
            {
                "chart_index": ci,
                "offset": np.array([r.x, r.y]),
                "scale": r.w,  # uniform scale since normalised to square
                "orig_span": raw_span,
                "orig_offset": np.array([raw_ox, raw_oy]),
                "area": area,
            }
        )

        # Determine which global vertices belong to this chart
        vert_set = set(faces.flatten().tolist())
        for vi in vert_set:
            if vi < len(all_vertices):
                local_uv = norm_uv[vi] if vi < len(norm_uv) else np.zeros(2)
                packed_uvs[vi] = local_uv * r.w + np.array([r.x, r.y])

    return {
        "packed_uvs": packed_uvs,
        "chart_transforms": chart_transforms,
        "atlas_size": (1.0, 1.0),
    }


def _dispatch_pack(
    method: str, rects: List[_Rect], atlas_size: float
) -> List[_Rect]:
    """Route to the correct packing algorithm."""
    if method == "simple":
        return _pack_simple(rects, atlas_size)
    if method == "shelf":
        return _pack_shelf(rects, atlas_size)
    if method == "guillotine":
        return _pack_guillotine(rects, atlas_size)
    if method == "skyline":
        return _pack_skyline(rects, atlas_size)
    raise ValueError(f"Unknown packing method {method!r}")


# ---------------------------------------------------------------------------
# Utility metrics
# ---------------------------------------------------------------------------

def compute_uv_utilization(
    uv_coords: np.ndarray, faces: np.ndarray
) -> float:
    """Measure packing efficiency as fraction of bounding box area used.

    Returns a value in ``[0, 1]`` where 1.0 means the UVs perfectly fill
    the bounding box with zero wasted space.
    """
    if faces.size == 0 or uv_coords.size == 0:
        return 0.0
    mn = uv_coords.min(axis=0)
    mx = uv_coords.max(axis=0)
    bbox_area = max((mx[0] - mn[0]) * (mx[1] - mn[1]), 1e-12)
    used_area = _area_of_chart(uv_coords, faces)
    return min(used_area / bbox_area, 1.0)


# ---------------------------------------------------------------------------
# Margin insertion
# ---------------------------------------------------------------------------

def add_uv_margins_advanced(
    uv_coords: np.ndarray,
    faces: np.ndarray,
    margin: float = 0.01,
) -> np.ndarray:
    """Add per-chart margins by shrinking each chart towards its centroid.

    The margin is applied as a fraction of the chart's own bounding box,
    so larger charts receive proportionally larger absolute margins while
    maintaining consistent *relative* padding.

    Parameters
    ----------
    uv_coords:
        ``(N, 2)`` UV coordinates (modified in-place is **not** done;
        a new array is returned).
    faces:
        ``(M, 3)`` face index array.
    margin:
        Fractional margin (0–0.5). A value of 0.01 shrinks the chart by
        0.5 % on each side (total 1 % in each axis).

    Returns
    -------
    np.ndarray
        New ``(N, 2)`` array with margins applied.
    """
    out = uv_coords.copy().astype(np.float64)
    if faces.size == 0 or uv_coords.size == 0:
        return out

    mn = out.min(axis=0)
    mx = out.max(axis=0)
    span = mx - mn
    span = np.maximum(span, 1e-12)

    centre = (mn + mx) * 0.5
    shrink = 1.0 - margin  # e.g. 0.99 for margin=0.01

    out = centre + (out - centre) * shrink
    return out
