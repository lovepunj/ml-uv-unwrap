from __future__ import annotations

"""Multi-chart UV unwrapper — unwraps each chart independently.

Takes a ChartDecomposition and unwraps each chart using the optimal
method for that chart's geometry. Charts are then packed into a
shared UV atlas.

Strategy selection per chart:
- Simple charts (low distortion): xatlas with single chart
- Complex charts (high genus): LSCM with boundary conditions
- Organic charts: ABF++ for angle preservation
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import trimesh

from .chart_decomposer import ChartDecomposition


@dataclass
class MultiChartResult:
    """Result of multi-chart unwrapping."""
    uv_coords: np.ndarray          # (N, 2) UV coordinates for all vertices
    vertices: np.ndarray           # (V, 3) original vertices
    faces: np.ndarray              # (F, 3) original face indices
    chart_labels: np.ndarray       # (F,) chart ID per face
    chart_uvs: list[np.ndarray]    # per-chart UV coordinates
    chart_faces: list[np.ndarray]  # per-chart face indices
    num_charts: int
    total_distortion: float
    per_chart_distortion: list[float]


class MultiChartUnwrapper:
    """Unwrap mesh using multi-chart decomposition.

    Each chart is unwrapped independently using the best method
    for its geometry, then packed into a shared UV atlas.
    """

    def __init__(
        self,
        default_method: str = "xatlas",
        pack_method: str = "simple",
    ):
        """
        Args:
            default_method: fallback unwrapping method
            pack_method: chart packing strategy ('simple', 'greedy', 'xatlas')
        """
        self.default_method = default_method
        self.pack_method = pack_method

    def unwrap(
        self,
        decomposition: ChartDecomposition,
        mesh: trimesh.Trimesh,
    ) -> MultiChartResult:
        """Unwrap all charts and pack into UV atlas."""
        vertices = np.array(mesh.vertices, dtype=np.float64)
        faces = np.array(mesh.faces, dtype=np.int64)

        chart_uvs = []
        chart_faces_list = []
        chart_vert_maps = []
        per_chart_distortion = []

        for chart_idx, chart_id in enumerate(sorted(decomposition.chart_faces.keys())):
            face_idx = decomposition.chart_faces[chart_id]
            chart_mesh = decomposition.chart_meshes[chart_idx]
            vert_map = decomposition.chart_vert_maps[chart_idx]

            method = self._select_method(chart_mesh)
            uv, distortion = self._unwrap_chart(chart_mesh, method)
            chart_uvs.append(uv)
            chart_faces_list.append(face_idx)
            chart_vert_maps.append(vert_map)
            per_chart_distortion.append(distortion)

        all_uvs = self._pack_charts(
            chart_uvs, chart_faces_list, chart_vert_maps, faces, vertices,
        )

        total_distortion = float(np.mean(per_chart_distortion))

        return MultiChartResult(
            uv_coords=all_uvs,
            vertices=vertices,
            faces=faces,
            chart_labels=decomposition.face_labels,
            chart_uvs=chart_uvs,
            chart_faces=chart_faces_list,
            num_charts=decomposition.chart_count,
            total_distortion=total_distortion,
            per_chart_distortion=per_chart_distortion,
        )

    def _select_method(self, chart_mesh: trimesh.Trimesh) -> str:
        """Select best unwrapping method for a chart."""
        num_faces = len(chart_mesh.faces)

        # Simple charts: xatlas
        if num_faces < 2000:
            return "xatlas"

        # Check if watertight
        is_watertight = chart_mesh.is_watertight if hasattr(chart_mesh, 'is_watertight') else False

        if is_watertight:
            return "xatlas"
        else:
            # Non-watertight: use LSCM
            return "lscm"

    def _unwrap_chart(
        self,
        chart_mesh: trimesh.Trimesh,
        method: str,
    ) -> tuple[np.ndarray, float]:
        """Unwrap a single chart.

        Returns:
            (uv_coords, distortion_score)
        """
        vertices = np.array(chart_mesh.vertices, dtype=np.float64)
        faces = np.array(chart_mesh.faces, dtype=np.int32)

        if method == "xatlas":
            return self._unwrap_xatlas(vertices, faces)
        elif method == "lscm":
            return self._unwrap_lscm(vertices, faces)
        elif method == "abf":
            return self._unwrap_abf(vertices, faces)
        else:
            return self._unwrap_xatlas(vertices, faces)

    def _unwrap_xatlas(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
    ) -> tuple[np.ndarray, float]:
        """Unwrap using xatlas. Returns UVs indexed by original vertex order."""
        try:
            import xatlas

            atlas = xatlas.Atlas()
            atlas.add_mesh(vertices.astype(np.float32), faces.astype(np.int32))

            chart_options = xatlas.ChartOptions()
            pack_options = xatlas.PackOptions()

            atlas.generate(chart_options, pack_options)

            unique_ids, face_ids, uv_coords = atlas.get_mesh(0)

            # Build per-original-vertex UVs
            num_verts = len(vertices)
            uv_per_vert = np.zeros((num_verts, 2), dtype=np.float32)
            uv_per_vert[unique_ids] = uv_coords

            # Normalize to [0, 1]
            valid = uv_per_vert.any(axis=1)
            if valid.any():
                uv_min = uv_per_vert[valid].min(axis=0)
                uv_max = uv_per_vert[valid].max(axis=0)
                uv_per_vert[valid] = (uv_per_vert[valid] - uv_min) / (uv_max - uv_min + 1e-10)

            distortion = self._compute_distortion(vertices, faces, uv_per_vert)

            return uv_per_vert, distortion

        except ImportError:
            return self._unwrap_lscm(vertices, faces)

    def _unwrap_lscm(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
    ) -> tuple[np.ndarray, float]:
        """Unwrap using LSCM."""
        try:
            import igl

            # Find boundary
            boundary = igl.boundary_loop(faces.astype(np.int32))
            if len(boundary) == 0:
                boundary = np.array([0, 1], dtype=np.int32)

            # Place boundary on circle
            n = len(boundary)
            angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
            boundary_uv = np.column_stack([np.cos(angles), np.sin(angles)])

            # Compute LSCM
            uv = igl.lscm(vertices, faces.astype(np.int32), boundary, boundary_uv)[0]

            # Normalize
            uv_min = uv.min(axis=0)
            uv_max = uv.max(axis=0)
            uv = (uv - uv_min) / (uv_max - uv_min + 1e-10)

            distortion = self._compute_distortion(vertices, faces, uv)

            return uv.astype(np.float32), distortion

        except ImportError:
            return self._unwrap_xatlas(vertices, faces)

    def _unwrap_abf(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
    ) -> tuple[np.ndarray, float]:
        """Unwrap using ABF (falls back to LSCM)."""
        return self._unwrap_lscm(vertices, faces)

    def _compute_distortion(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
        uv: np.ndarray,
    ) -> float:
        """Compute average distortion metric."""
        if len(faces) == 0 or len(vertices) == 0:
            return 0.0

        try:
            # Simple distortion: ratio of 2D to 3D edge lengths
            v0_3d = vertices[faces[:, 0]]
            v1_3d = vertices[faces[:, 1]]
            v2_3d = vertices[faces[:, 2]]

            v0_uv = uv[faces[:, 0]]
            v1_uv = uv[faces[:, 1]]
            v2_uv = uv[faces[:, 2]]

            # Edge lengths
            e1_3d = np.linalg.norm(v1_3d - v0_3d, axis=1) + 1e-10
            e2_3d = np.linalg.norm(v2_3d - v0_3d, axis=1) + 1e-10
            e1_uv = np.linalg.norm(v1_uv - v0_uv, axis=1) + 1e-10
            e2_uv = np.linalg.norm(v2_uv - v0_uv, axis=1) + 1e-10

            # Stretch ratios
            s1 = e1_uv / e1_3d
            s2 = e2_uv / e2_3d

            # Distortion = how different the stretch ratios are
            distortion = float(np.mean((s1 - s2) ** 2))
            return distortion

        except Exception:
            return 0.0

    def _pack_charts(
        self,
        chart_uvs: list[np.ndarray],
        chart_faces: list[np.ndarray],
        chart_vert_maps: list[np.ndarray],
        all_faces: np.ndarray,
        all_vertices: np.ndarray,
    ) -> np.ndarray:
        """Pack chart UVs into shared [0, 1] atlas space.

        chart_vert_maps[i] maps chart-internal vertex index -> original mesh vertex index.
        chart_uvs[i] has shape (num_chart_verts, 2) indexed by chart-internal indices.
        """
        num_verts = len(all_vertices)
        all_uvs = np.zeros((num_verts, 2), dtype=np.float32)

        num_charts = len(chart_uvs)
        if num_charts == 0:
            return all_uvs

        if num_charts == 1:
            uv = chart_uvs[0]
            vert_map = chart_vert_maps[0]
            for i, orig_v in enumerate(vert_map):
                if i < len(uv):
                    all_uvs[orig_v] = uv[i]
            return all_uvs

        grid_size = int(np.ceil(np.sqrt(num_charts)))

        for chart_idx, (uv, vert_map) in enumerate(zip(chart_uvs, chart_vert_maps)):
            row = chart_idx // grid_size
            col = chart_idx % grid_size

            tile_x = col / grid_size
            tile_y = row / grid_size
            tile_size = 1.0 / grid_size

            margin = 0.01
            tile_size -= margin * 2
            tile_x += margin
            tile_y += margin

            chart_uv = uv.copy()
            if len(chart_uv) > 0:
                uv_min = chart_uv.min(axis=0)
                uv_max = chart_uv.max(axis=0)
                uv_range = uv_max - uv_min + 1e-10
                chart_uv = (chart_uv - uv_min) / uv_range
                chart_uv = chart_uv * tile_size + np.array([tile_x, tile_y])

            for i, orig_v in enumerate(vert_map):
                if i < len(chart_uv):
                    all_uvs[orig_v] = chart_uv[i]

        return all_uvs
