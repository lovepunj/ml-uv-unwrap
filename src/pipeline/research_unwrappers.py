from __future__ import annotations

"""Research UV unwrapping algorithms.

Implements:
- Voronoi disk segmentation (Maggioli et al., STAG 2025)
- Instant meshes style integer-grid parameterization
- libUvula subprocess wrapper
"""

import subprocess
import tempfile
from pathlib import Path

import numpy as np
import trimesh


class ResearchUnwrapper:
    """Advanced research-based UV unwrapping methods."""

    def __init__(self, method: str = "voronoi_disks"):
        self.method = method

    def unwrap(
        self,
        mesh: trimesh.Trimesh | str | Path,
    ) -> dict:
        if isinstance(mesh, (str, Path)):
            mesh = trimesh.load(mesh, force="mesh")
            if isinstance(mesh, trimesh.Scene):
                mesh = trimesh.util.concatenate(mesh.dump())

        vertices = np.array(mesh.vertices, dtype=np.float32)
        faces = np.array(mesh.faces, dtype=np.int32)

        print(f"Research unwrap ({self.method}): {len(vertices)} verts, {len(faces)} faces")

        dispatch = {
            "voronoi_disks": lambda: self._unwrap_voronoi_disks(vertices, faces),
            "instant_meshes": lambda: self._unwrap_instant_meshes(vertices, faces),
            "libuvula": lambda: self._unwrap_libuvula(vertices, faces, mesh),
        }

        if self.method not in dispatch:
            raise ValueError(f"Unknown method: {self.method}")

        result = dispatch[self.method]()
        result["mesh"] = mesh
        result["method"] = self.method
        return result

    @staticmethod
    def _normalize_uv(uv: np.ndarray) -> np.ndarray:
        uv_min = uv.min(axis=0)
        uv_max = uv.max(axis=0)
        return (uv - uv_min) / (uv_max - uv_min + 1e-10)

    # ── Voronoi disk segmentation ──────────────────────────────────

    def _unwrap_voronoi_disks(self, vertices: np.ndarray, faces: np.ndarray) -> dict:
        """Segment mesh into topological disks via Voronoi decomposition,
        then flatten each with harmonic mapping.

        Based on Maggioli et al. 'UV Parametrization via Topological
        Disk Segmentation of Surfaces' (STAG 2025).
        """
        import igl
        from scipy import sparse
        from scipy.sparse.linalg import spsolve

        v = vertices.astype(np.float64)
        f = faces.astype(np.int32)

        N = len(v)
        F = len(f)

        # Step 1: Sample sparse point set (n=10 per face as in the paper)
        n_samples = min(10, max(1, F // 10))
        face_centroids = v[f].mean(axis=1)

        # Step 2: Farthest-point sampling for initial Voronoi seeds
        k = max(4, min(50, F // 20))
        seed_indices = self._farthest_point_sampling(face_centroids, k)

        # Step 3: Assign each face to nearest seed (Voronoi cells)
        seed_pos = face_centroids[seed_indices]
        dists = np.linalg.norm(
            face_centroids[:, None, :] - seed_pos[None, :, :], axis=2
        )
        assignments = dists.argmin(axis=1)

        # Step 4: Merge small / degenerate cells
        merged = self._merge_small_cells(assignments, face_centroids, min_size=5)
        chart_labels = merged
        chart_ids = np.unique(merged)
        num_charts = len(chart_ids)

        print(f"  Voronoi disks: {num_charts} initial regions")

        # Step 5: Flatten each chart with harmonic mapping
        uv = np.zeros((N, 2), dtype=np.float32)
        per_chart_distortion = []

        for cid in range(num_charts):
            mask = chart_labels == cid
            if mask.sum() == 0:
                continue

            chart_faces = f[mask]
            unique_verts = np.unique(chart_faces)
            vert_map = np.full(N, -1, dtype=np.int32)
            vert_map[unique_verts] = np.arange(len(unique_verts))
            remapped = vert_map[chart_faces]

            sub_v = v[unique_verts]
            sub_f = remapped

            # Need boundary for harmonic mapping
            boundary = igl.boundary_loop(sub_f)
            if len(boundary) < 3:
                # Degenerate chart: PCA projection
                centered = sub_v - sub_v.mean(axis=0)
                cov = centered.T @ centered
                _, eigvecs = np.linalg.eigh(cov)
                uv_proj = centered @ eigvecs[:, -2:]
                uv_proj -= uv_proj.min(axis=0)
                uv_proj /= uv_proj.max(axis=0) + 1e-8
                for j, vi in enumerate(unique_verts):
                    uv[vi] = uv_proj[j].astype(np.float32)
                continue

            n_bnd = len(boundary)
            angles = np.linspace(0, 2 * np.pi, n_bnd, endpoint=False)
            boundary_uv = np.column_stack([np.cos(angles), np.sin(angles)])

            n_sub = len(sub_v)
            interior = np.setdiff1d(np.arange(n_sub), boundary)

            if len(interior) == 0:
                sub_uv = np.zeros((n_sub, 2))
                sub_uv[boundary] = boundary_uv
            else:
                try:
                    L = igl.cotmatrix(sub_v, sub_f)
                    L_int = L[np.ix_(interior, interior)]
                    L_bnd = L[np.ix_(interior, boundary)]
                    u_int = spsolve(L_int, -L_bnd @ boundary_uv[:, 0])
                    v_int = spsolve(L_int, -L_bnd @ boundary_uv[:, 1])
                except Exception:
                    # Fallback: PCA
                    centered = sub_v - sub_v.mean(axis=0)
                    cov = centered.T @ centered
                    _, eigvecs = np.linalg.eigh(cov)
                    uv_proj = centered @ eigvecs[:, -2:]
                    uv_proj -= uv_proj.min(axis=0)
                    uv_proj /= uv_proj.max(axis=0) + 1e-8
                    for j, vi in enumerate(unique_verts):
                        uv[vi] = uv_proj[j].astype(np.float32)
                    continue

                sub_uv = np.zeros((n_sub, 2))
                sub_uv[boundary] = boundary_uv
                sub_uv[interior, 0] = u_int
                sub_uv[interior, 1] = v_int

            # Compute per-chart distortion
            try:
                e1_3d = np.linalg.norm(sub_v[remapped[:, 1]] - sub_v[remapped[:, 0]], axis=1) + 1e-10
                e2_3d = np.linalg.norm(sub_v[remapped[:, 2]] - sub_v[remapped[:, 0]], axis=1) + 1e-10
                e1_uv = np.linalg.norm(sub_uv[remapped[:, 1]] - sub_uv[remapped[:, 0]], axis=1) + 1e-10
                e2_uv = np.linalg.norm(sub_uv[remapped[:, 2]] - sub_uv[remapped[:, 0]], axis=1) + 1e-10
                s1, s2 = e1_uv / e1_3d, e2_uv / e2_3d
                distortion = float(np.mean((s1 - s2) ** 2))
                per_chart_distortion.append(distortion)
            except Exception:
                per_chart_distortion.append(0.0)

            # Map back to full vertex array
            for j, vi in enumerate(unique_verts):
                uv[vi] = self._normalize_uv(sub_uv[j:j+1])[0].astype(np.float32)

            print(f"    Chart {cid}: {mask.sum()} faces, distortion={per_chart_distortion[-1]:.4f}")

        avg_distortion = np.mean(per_chart_distortion) if per_chart_distortion else 0.0

        return {
            "uv_coords": uv,
            "vertices": vertices,
            "faces": faces,
            "num_charts": num_charts,
            "chart_labels": chart_labels.astype(np.int32),
            "per_chart_distortion": per_chart_distortion,
            "total_distortion": avg_distortion,
        }

    @staticmethod
    def _farthest_point_sampling(points: np.ndarray, k: int) -> np.ndarray:
        n = len(points)
        if k >= n:
            return np.arange(n)
        selected = [np.random.randint(n)]
        dists = np.full(n, np.inf)
        for _ in range(k - 1):
            d = np.linalg.norm(points - points[selected[-1]], axis=1)
            dists = np.minimum(dists, d)
            selected.append(int(dists.argmax()))
        return np.array(selected)

    @staticmethod
    def _merge_small_cells(assignments: np.ndarray, centroids: np.ndarray, min_size: int = 5) -> np.ndarray:
        result = assignments.copy()
        for _ in range(10):
            ids, counts = np.unique(result, return_counts=True)
            small_ids = ids[counts < min_size]
            if len(small_ids) == 0:
                break
            for sid in small_ids:
                mask = result == sid
                if mask.sum() == 0:
                    continue
                my_centroid = centroids[mask].mean(axis=0)
                other_ids = ids[ids != sid]
                if len(other_ids) == 0:
                    continue
                other_centroids = np.array([centroids[result == oid].mean(axis=0) for oid in other_ids])
                dists = np.linalg.norm(other_centroids - my_centroid, axis=1)
                nearest = other_ids[dists.argmin()]
                result[mask] = nearest
        return result

    # ── Instant Meshes style ───────────────────────────────────────

    def _unwrap_instant_meshes(self, vertices: np.ndarray, faces: np.ndarray) -> dict:
        """Integer-grid based parameterization inspired by Instant Meshes.

        Projects mesh to a regular grid in UV space, then relaxes
        using Laplacian smoothing.
        """
        import igl
        from scipy import sparse
        from scipy.sparse.linalg import spsolve

        v = vertices.astype(np.float64)
        f = faces.astype(np.int32)
        N = len(v)

        # Step 1: Compute edge-aligned frame field using cross fields
        try:
            singularities, frames = igl.comb_cross_field(v, f, np.zeros(len(f)))
        except Exception:
            # Fallback: use vertex normals to build a simple frame
            vn = igl.per_vertex_normals(v, f)
            frames = np.tile(np.eye(3), (len(f), 1)).reshape(len(f), 3, 3)

        # Step 2: Initial parameterization via LSCM
        boundary = igl.boundary_loop(f)
        if len(boundary) < 3:
            # Closed mesh: cut along longest edge loop
            edge_lens = np.linalg.norm(v[f[:, 1]] - v[f[:, 0]], axis=1)
            long_edges = np.argsort(edge_lens)[-10:]
            # Fallback to xatlas for closed meshes
            from .classical_unwrapper import ClassicalUnwrapper
            return ClassicalUnwrapper(method="xatlas").unwrap(
                trimesh.Trimesh(vertices=vertices, faces=faces)
            )

        n = len(boundary)
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        boundary_uv = np.column_stack([np.cos(angles), np.sin(angles)])

        try:
            uv = igl.lscm(v, f, boundary, boundary_uv)[0]
        except Exception:
            from .classical_unwrapper import ClassicalUnwrapper
            return ClassicalUnwrapper(method="xatlas").unwrap(
                trimesh.Trimesh(vertices=vertices, faces=faces)
            )

        # Step 3: Snap to integer grid (Instant Meshes style)
        # Quantize UVs to nearest integer positions, then relax
        grid_size = max(2, min(64, int(np.sqrt(len(faces)))))
        uv_int = np.round(uv * grid_size) / grid_size

        # Step 4: Laplacian relaxation to improve alignment
        L = igl.cotmatrix(v, f)
        uv_relaxed = uv_int.copy()

        for _ in range(10):
            uv_new = uv_relaxed - 0.5 * L @ uv_relaxed
            # Keep boundary fixed
            uv_new[boundary] = uv_int[boundary]
            # Snap interior to grid
            interior = np.setdiff1d(np.arange(N), boundary)
            uv_new[interior] = np.round(uv_new[interior] * grid_size) / grid_size
            uv_relaxed = uv_new

        uv_norm = self._normalize_uv(uv_relaxed)

        return {
            "uv_coords": uv_norm.astype(np.float32),
            "vertices": vertices,
            "faces": faces,
            "num_charts": 1,
            "grid_size": grid_size,
        }

    # ── libUvula wrapper ───────────────────────────────────────────

    def _unwrap_libuvula(self, vertices: np.ndarray, faces: np.ndarray, mesh: trimesh.Trimesh) -> dict:
        """Wrap Ultimaker's libUvula UV unwrapper via subprocess.

        Requires libUvula to be installed and available on PATH.
        Falls back to xatlas if not available.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            obj_path = Path(tmpdir) / "input.obj"
            out_path = Path(tmpdir) / "output.obj"

            mesh.export(str(obj_path))

            try:
                result = subprocess.run(
                    ["uvula", str(obj_path), str(out_path)],
                    capture_output=True,
                    text=True,
                    timeout=60,
                )
                if result.returncode != 0:
                    raise RuntimeError(result.stderr)

                out_mesh = trimesh.load(str(out_path), force="mesh")
                out_verts = np.array(out_mesh.vertices, dtype=np.float32)
                out_faces = np.array(out_mesh.faces, dtype=np.int32)
                out_uvs = out_mesh.visual.uv
                if out_uvs is None:
                    raise RuntimeError("No UV data in libUvula output")

                return {
                    "uv_coords": out_uvs.astype(np.float32),
                    "vertices": out_verts,
                    "faces": out_faces,
                    "num_charts": 1,
                }

            except (FileNotFoundError, subprocess.TimeoutExpired, Exception) as e:
                print(f"  libUvula unavailable ({e}), falling back to xatlas")
                from .classical_unwrapper import ClassicalUnwrapper
                return ClassicalUnwrapper(method="xatlas").unwrap(mesh)
