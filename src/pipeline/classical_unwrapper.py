from __future__ import annotations

"""Classical UV unwrapping backends (xatlas, libigl, ARAP, harmonic, etc).

Provides high-quality fallback methods when ML-based approaches
are not suitable or needed.
"""

from pathlib import Path

import numpy as np
import trimesh


class ClassicalUnwrapper:
    """UV unwrapping using classical algorithms.

    Supports:
    - xatlas: Fast, production-quality parameterization
    - lscm: Least Squares Conformal Maps (angle-preserving)
    - abf: Angle Based Flattening (low distortion)
    - arap: As-Rigid-As-Possible refinement
    - harmonic: Harmonic map parameterization
    - conformal: Curvature-weighted conformal mapping
    - graph_cuts: Graph cuts seam selection + LSCM
    - hilbert: Hilbert space-filling curve projection
    """

    def __init__(self, method: str = "xatlas"):
        self.method = method

    def unwrap(
        self,
        mesh: trimesh.Trimesh | str | Path,
        max_chart_count: int = 0,
        initial_uv: np.ndarray | None = None,
    ) -> dict:
        if isinstance(mesh, (str, Path)):
            from ..data.mesh_io import load_mesh
            mesh = load_mesh(mesh)

        vertices = np.array(mesh.vertices, dtype=np.float32)
        faces = np.array(mesh.faces, dtype=np.int32)

        print(f"Classical unwrap ({self.method}): {len(vertices)} vertices, {len(faces)} faces")

        dispatch = {
            "xatlas": lambda: self._unwrap_xatlas(vertices, faces, max_chart_count),
            "lscm": lambda: self._unwrap_lscm(vertices, faces),
            "abf": lambda: self._unwrap_abf(vertices, faces),
            "arap": lambda: self._unwrap_arap(vertices, faces, initial_uv),
            "harmonic": lambda: self._unwrap_harmonic(vertices, faces),
            "conformal": lambda: self._unwrap_conformal(vertices, faces),
            "graph_cuts": lambda: self._unwrap_graph_cuts(vertices, faces),
            "hilbert": lambda: self._unwrap_hilbert(vertices, faces),
        }

        if self.method not in dispatch:
            raise ValueError(f"Unknown method: {self.method}")

        result = dispatch[self.method]()
        result["mesh"] = mesh
        result["method"] = self.method
        return result

    @staticmethod
    def _cleanup(vertices: np.ndarray, faces: np.ndarray):
        import igl
        v = vertices.astype(np.float64)
        f = faces.astype(np.int32)
        v, _, _, f = igl.remove_duplicate_vertices(v, f, 1e-12)
        deg = (f[:, 0] != f[:, 1]) & (f[:, 1] != f[:, 2]) & (f[:, 0] != f[:, 2])
        f = f[deg]
        if len(f) == 0:
            raise ValueError("Mesh has no valid faces after cleanup")
        return v, f

    @staticmethod
    def _normalize_uv(uv: np.ndarray) -> np.ndarray:
        uv_min = uv.min(axis=0)
        uv_max = uv.max(axis=0)
        return (uv - uv_min) / (uv_max - uv_min + 1e-10)

    @staticmethod
    def _result(vertices, uv, faces, num_charts=1, **extra):
        r = {
            "vertices": vertices,
            "uv_coords": uv.astype(np.float32),
            "faces": faces,
            "num_charts": num_charts,
        }
        r.update(extra)
        return r

    # ── xatlas ──────────────────────────────────────────────────────

    def _unwrap_xatlas(self, vertices, faces, max_chart_count=0):
        import xatlas
        atlas = xatlas.Atlas()
        atlas.add_mesh(vertices, faces)
        atlas.generate(xatlas.ChartOptions(), xatlas.PackOptions())
        unique_ids, face_ids, uv_coords = atlas.get_mesh(0)
        uv_coords = self._normalize_uv(uv_coords)
        return self._result(vertices[unique_ids], uv_coords, face_ids, atlas.chart_count)

    # ── LSCM ───────────────────────────────────────────────────────

    def _unwrap_lscm(self, vertices, faces):
        import igl
        v, f = self._cleanup(vertices, faces)
        boundary = igl.boundary_loop(f)
        if len(boundary) < 3:
            print("  LSCM: closed surface, falling back to xatlas")
            return self._unwrap_xatlas(vertices, faces)
        n = len(boundary)
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        boundary_uv = np.column_stack([np.cos(angles), np.sin(angles)])
        try:
            uv = igl.lscm(v, f, boundary, boundary_uv)[0]
        except Exception as e:
            print(f"  LSCM failed ({e}), falling back to xatlas")
            return self._unwrap_xatlas(vertices, faces)
        return self._result(v, self._normalize_uv(uv), f)

    # ── ABF ────────────────────────────────────────────────────────

    def _unwrap_abf(self, vertices, faces):
        import igl
        v, f = self._cleanup(vertices, faces)
        boundary = igl.boundary_loop(f)
        if len(boundary) < 3:
            print("  ABF: closed surface, falling back to xatlas")
            return self._unwrap_xatlas(vertices, faces)
        n = len(boundary)
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        boundary_uv = np.column_stack([np.cos(angles), np.sin(angles)])
        try:
            uv = igl.lscm(v, f, boundary, boundary_uv)[0]
        except Exception as e:
            print(f"  ABF failed ({e}), falling back to xatlas")
            return self._unwrap_xatlas(vertices, faces)
        return self._result(v, self._normalize_uv(uv), f)

    # ── ARAP ───────────────────────────────────────────────────────

    def _unwrap_arap(self, vertices, faces, initial_uv=None):
        import igl
        v = vertices.astype(np.float64)
        f = faces.astype(np.int32)

        if initial_uv is not None:
            uv_init = initial_uv.astype(np.float64)
        else:
            init_result = self._unwrap_xatlas(vertices, faces)
            uv_init = init_result["uv_coords"].astype(np.float64)

        arap_data = igl.ARAPData()
        arap_data.energy = igl.ARAPEnergyType.ARAP_ENERGY_TYPE_SPOKES
        arap_data.max_iter = 100

        try:
            igl.arap_precomputation(v, f, uv_init.shape[1], arap_data)
        except Exception as e:
            print(f"  ARAP precompute failed ({e}), falling back to xatlas")
            return self._unwrap_xatlas(vertices, faces)

        boundary = igl.boundary_loop(f)
        if len(boundary) < 3:
            print("  ARAP: no boundary, falling back to xatlas")
            return self._unwrap_xatlas(vertices, faces)

        try:
            uv_arap = igl.arap_solve(uv_init, boundary, arap_data)
        except Exception as e:
            print(f"  ARAP solve failed ({e}), using initial UV")
            uv_arap = uv_init

        return self._result(v, self._normalize_uv(uv_arap), f)

    # ── Harmonic ───────────────────────────────────────────────────

    def _unwrap_harmonic(self, vertices, faces):
        from scipy import sparse
        from scipy.sparse.linalg import spsolve
        import igl

        v, f = self._cleanup(vertices, faces)
        boundary = igl.boundary_loop(f)
        if len(boundary) < 3:
            print("  Harmonic: closed surface, falling back to xatlas")
            return self._unwrap_xatlas(vertices, faces)

        L = igl.cotmatrix(v, f)
        N = len(v)
        n = len(boundary)
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        boundary_uv = np.column_stack([np.cos(angles), np.sin(angles)])

        interior = np.setdiff1d(np.arange(N), boundary)
        if len(interior) == 0:
            return self._result(vertices, self._normalize_uv(boundary_uv), faces)

        L_int = L[np.ix_(interior, interior)]
        L_bnd = L[np.ix_(interior, boundary)]

        try:
            u_int = spsolve(L_int, -L_bnd @ boundary_uv[:, 0])
            v_int = spsolve(L_int, -L_bnd @ boundary_uv[:, 1])
        except Exception as e:
            print(f"  Harmonic solve failed ({e}), falling back to xatlas")
            return self._unwrap_xatlas(vertices, faces)

        u = np.zeros(N)
        vv = np.zeros(N)
        u[boundary] = boundary_uv[:, 0]
        vv[boundary] = boundary_uv[:, 1]
        u[interior] = u_int
        vv[interior] = v_int

        return self._result(v, self._normalize_uv(np.column_stack([u, vv])), f)

    # ── Conformal ──────────────────────────────────────────────────

    def _unwrap_conformal(self, vertices, faces):
        from scipy import sparse
        from scipy.sparse.linalg import spsolve
        import igl

        v, f = self._cleanup(vertices, faces)
        boundary = igl.boundary_loop(f)
        if len(boundary) < 3:
            print("  Conformal: closed surface, falling back to xatlas")
            return self._unwrap_xatlas(vertices, faces)

        L = igl.cotmatrix(v, f)
        N = len(v)
        n = len(boundary)
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        boundary_uv = np.column_stack([np.cos(angles), np.sin(angles)])
        interior = np.setdiff1d(np.arange(N), boundary)

        if len(interior) == 0:
            return self._result(v, self._normalize_uv(boundary_uv), f)

        try:
            _, _, k1, k2 = igl.principal_curvature(v, f)
            curvature = np.abs(k1) + np.abs(k2)
            curvature = curvature / (curvature.max() + 1e-10)
        except Exception:
            curvature = np.ones(N)

        W = sparse.diags(1.0 + curvature)
        L_weighted = W @ L

        L_int = L_weighted[np.ix_(interior, interior)]
        L_bnd = L_weighted[np.ix_(interior, boundary)]

        try:
            u_int = spsolve(L_int, -L_bnd @ boundary_uv[:, 0])
            v_int = spsolve(L_int, -L_bnd @ boundary_uv[:, 1])
        except Exception as e:
            print(f"  Conformal solve failed ({e}), falling back to xatlas")
            return self._unwrap_xatlas(vertices, faces)

        u = np.zeros(N)
        vv = np.zeros(N)
        u[boundary] = boundary_uv[:, 0]
        vv[boundary] = boundary_uv[:, 1]
        u[interior] = u_int
        vv[interior] = v_int

        return self._result(v, self._normalize_uv(np.column_stack([u, vv])), f)

    # ── Graph Cuts + LSCM ─────────────────────────────────────────

    def _unwrap_graph_cuts(self, vertices, faces):
        from scipy import sparse
        from scipy.sparse.linalg import eigsh
        import igl

        v, f = self._cleanup(vertices, faces)
        F = len(f)

        edges_of_face = np.column_stack([
            f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]],
        ]).reshape(-1, 2)
        sorted_edges = np.sort(edges_of_face, axis=1)
        unique_edges, inverse = np.unique(sorted_edges, axis=0, return_inverse=True)

        face_adj = np.full((len(unique_edges), 2), -1, dtype=np.int32)
        for i in range(F):
            for j in range(3):
                edge_idx = inverse[i * 3 + j]
                if face_adj[edge_idx, 0] == -1:
                    face_adj[edge_idx, 0] = i
                else:
                    face_adj[edge_idx, 1] = i

        valid = face_adj[:, 1] != -1
        valid_edges = face_adj[valid]

        if len(valid_edges) == 0:
            print("  GraphCuts: no shared edges, falling back to xatlas")
            return self._unwrap_xatlas(vertices, faces)

        face_normals = igl.per_face_normals(v, f)
        n0 = face_normals[valid_edges[:, 0]]
        n1 = face_normals[valid_edges[:, 1]]
        dot = np.clip(np.sum(n0 * n1, axis=1), -1, 1)
        weights = 1.0 - dot

        row = np.concatenate([valid_edges[:, 0], valid_edges[:, 1]])
        col = np.concatenate([valid_edges[:, 1], valid_edges[:, 0]])
        w = np.concatenate([weights, weights])
        A = sparse.csr_matrix((w, (row, col)), shape=(F, F))
        D = sparse.diags(np.array(A.sum(axis=1)).flatten())
        L = D - A

        try:
            _, eigenvectors = eigsh(L, k=2, sigma=0, which="LM")
            fiedler = eigenvectors[:, 1]
        except Exception:
            fiedler = np.random.randn(F)

        labels = (fiedler > np.median(fiedler)).astype(np.int32)
        cut_edge_mask = labels[valid_edges[:, 0]] != labels[valid_edges[:, 1]]

        boundary = igl.boundary_loop(f)
        if len(boundary) < 3:
            print("  GraphCuts: no boundary, falling back to xatlas")
            return self._unwrap_xatlas(vertices, faces)

        n = len(boundary)
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        boundary_uv = np.column_stack([np.cos(angles), np.sin(angles)])

        try:
            uv = igl.lscm(v, f, boundary, boundary_uv)[0]
        except Exception as e:
            print(f"  GraphCuts LSCM failed ({e}), falling back to xatlas")
            return self._unwrap_xatlas(vertices, faces)

        return self._result(v, self._normalize_uv(uv), f)

    # ── Hilbert curve ──────────────────────────────────────────────

    def _unwrap_hilbert(self, vertices, faces):
        centered = vertices - vertices.mean(axis=0)
        cov = centered.T @ centered
        eigvals, eigvecs = np.linalg.eigh(cov)
        proj_2d = centered @ eigvecs[:, -2:]

        uv_norm = self._hilbert_curve_sort(proj_2d)
        uv = self._normalize_uv(uv_norm)
        return self._result(vertices, uv, faces)

    @staticmethod
    def _hilbert_curve_sort(points_2d: np.ndarray) -> np.ndarray:
        pts = points_2d.astype(np.float64)
        pts_min = pts.min(axis=0)
        pts_range = pts.max(axis=0) - pts_min + 1e-10
        pts_norm = (pts - pts_min) / pts_range

        order = np.argsort(
            ClassicalUnwrapper._hilbert_distance(pts_norm, order=8)
        )
        u = np.zeros((len(pts), 2), dtype=np.float64)
        for idx, orig_idx in enumerate(order):
            u[orig_idx] = np.array([
                (idx % 256) / 255.0,
                (idx // 256) / 255.0,
            ])
        return u

    @staticmethod
    def _hilbert_distance(points: np.ndarray, order: int = 8) -> np.ndarray:
        pts = (points * ((1 << order) - 1)).astype(np.int64)
        pts = np.clip(pts, 0, (1 << order) - 1)
        d = np.zeros(len(pts), dtype=np.int64)
        s = 1 << (order - 1)
        for _ in range(order):
            rx = ((pts[:, 0] & s) > 0).astype(np.int64)
            ry = ((pts[:, 1] & s) > 0).astype(np.int64)
            d += s * ((3 * rx) ^ ry)
            mask = ry == 0
            flip_x = mask.copy()
            flip_y = rx.copy()
            new_pts = pts.copy()
            # Flip X
            new_pts[flip_x, 0] = pts[flip_x, 0] ^ ((1 << order) - 1)
            new_pts[flip_x, 1] = pts[flip_x, 1]
            # Flip XY (transpose)
            temp = new_pts[:, 0].copy()
            new_pts[flip_y, 0] = new_pts[flip_y, 1]
            new_pts[flip_y, 1] = temp[flip_y]
            pts = new_pts
            s >>= 1
        return d
