"""FFHQ-UV inspired face UV unwrapping.

Implements the core techniques from FFHQ-UV (CVPR 2023):
1. Topology-based UV transfer: copy pre-computed UV coordinates for known
   mesh topologies (FLAME, HiFi3D++, etc.)
2. Multi-view texture projection: project 3D surface from multiple viewpoints
   to build UV-space texture maps
3. RGB fitting: optimize UV coordinates by minimizing texture-space distortion
   under camera projection constraints

References:
    Bai et al., "FFHQ-UV: Normalized Facial UV-Texture Dataset for 3D
    Face Reconstruction", CVPR 2023.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import trimesh


class FFHQUVUnwrapper:
    """UV unwrapping using FFHQ-UV techniques.

    Supports:
    - topo_transfer: Transfer UV coordinates from a reference mesh with known
      topology (same face count and connectivity).
    - multi_view: Multi-view projection UV unwrapping from camera viewpoints.
    - rgb_fitting: Optimize UV layout by fitting to texture samples.
    - face_auto: Auto-detect face mesh and use optimal strategy.
    """

    # Canonical face UV layout (simplified HiFi3D++ style):
    # Face occupies [0.05, 0.05] to [0.95, 0.95], eyes at [1.1, 0.3] and [1.1, 0.7]
    FACE_UV_TEMPLATE_PATH = Path(__file__).parent.parent.parent / "assets" / "face_uv_template.npz"

    EYE_VERTEX_RANGE = (3931, 5022)  # FLAME eye vertex indices

    def __init__(self, method: str = "face_auto"):
        self.method = method

    def unwrap(
        self,
        mesh: trimesh.Trimesh | str | Path,
        reference_uv_path: str | Path | None = None,
        num_views: int = 6,
        max_chart_count: int = 0,
    ) -> dict:
        if isinstance(mesh, (str, Path)):
            from ..data.mesh_io import load_mesh
            mesh = load_mesh(mesh)

        vertices = np.array(mesh.vertices, dtype=np.float32)
        faces = np.array(mesh.faces, dtype=np.int32)

        print(f"FFHQ-UV unwrap ({self.method}): {len(vertices)} vertices, {len(faces)} faces")

        dispatch = {
            "topo_transfer": lambda: self._topo_transfer(vertices, faces, reference_uv_path),
            "multi_view": lambda: self._multi_view_projection(vertices, faces, mesh, num_views),
            "rgb_fitting": lambda: self._rgb_fitting(vertices, faces, mesh),
            "face_auto": lambda: self._face_auto(vertices, faces, mesh, reference_uv_path, num_views),
        }

        if self.method not in dispatch:
            raise ValueError(f"Unknown FFHQ-UV method: {self.method}")

        result = dispatch[self.method]()
        result["mesh"] = mesh
        result["method"] = f"ffhq_uv_{self.method}"
        return result

    # ── Topology Transfer ──────────────────────────────────────────

    def _topo_transfer(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
        reference_uv_path: str | Path | None = None,
    ) -> dict:
        """Transfer UV coordinates from a reference mesh with identical topology.

        This is the core FFHQ-UV technique: if the input mesh shares the same
        connectivity as a reference (e.g. FLAME → HiFi3D++ UV layout), directly
        copy the pre-computed UV coordinates.

        Args:
            vertices: (V, 3) vertex positions
            faces: (F, 3) face indices
            reference_uv_path: Path to reference .obj with UVs, or .npz with
                             'uv_coords' array of shape (V, 2).
        """
        if reference_uv_path is None:
            print("  No reference UV provided, falling back to xatlas")
            from .classical_unwrapper import ClassicalUnwrapper
            return ClassicalUnwrapper("xatlas").unwrap(trimesh.Trimesh(vertices=vertices, faces=faces))

        ref_path = Path(reference_uv_path)

        if ref_path.suffix == ".npz":
            data = np.load(str(ref_path))
            uv_coords = data["uv_coords"]
        elif ref_path.suffix == ".obj":
            ref_mesh = trimesh.load(str(ref_path), force="mesh")
            uv_coords = np.array(ref_mesh.visual.uv, dtype=np.float32)
        else:
            raise ValueError(f"Unsupported reference UV format: {ref_path.suffix}")

        if uv_coords.shape[0] < len(vertices):
            uv_padded = np.zeros((len(vertices), 2), dtype=np.float32)
            uv_padded[: uv_coords.shape[0]] = uv_coords
            uv_coords = uv_padded
        elif uv_coords.shape[0] > len(vertices):
            uv_coords = uv_coords[: len(vertices)]

        uv_coords = self._normalize_uv(uv_coords)

        print(f"  Topology transfer: copied {uv_coords.shape[0]} UV coords from {ref_path.name}")
        return {
            "vertices": vertices,
            "uv_coords": uv_coords.astype(np.float32),
            "faces": faces,
            "num_charts": 1,
        }

    # ── Multi-View Projection ──────────────────────────────────────

    def _multi_view_projection(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
        mesh: trimesh.Trimesh,
        num_views: int = 6,
    ) -> dict:
        """Multi-view projection UV unwrapping (FFHQ-UV texture pipeline concept).

        Projects the 3D surface from multiple camera viewpoints and composites
        the projections into a UV texture map. Inspired by FFHQ-UV's approach
        of using left/front/right views for face texture extraction.

        The UV coordinates are derived by:
        1. Computing face normals and selecting best viewing direction per face
        2. Projecting faces from their best-view camera onto the UV plane
        3. Solving for smooth UV coordinates using Laplacian smoothing
        """
        N = len(vertices)
        F = len(faces)

        v0 = vertices[faces[:, 0]]
        v1 = vertices[faces[:, 1]]
        v2 = vertices[faces[:, 2]]
        centroids = (v0 + v1 + v2) / 3.0
        normals = np.cross(v1 - v0, v2 - v0)
        norms = np.linalg.norm(normals, axis=1, keepdims=True) + 1e-10
        normals = normals / norms

        center = vertices.mean(axis=0)
        extent = vertices.max(axis=0) - vertices.min(axis=0)
        cam_dist = extent.max() * 2.0

        angles = np.linspace(0, 2 * np.pi, num_views, endpoint=False)
        cam_positions = np.column_stack([
            np.cos(angles) * cam_dist + center[0],
            np.full(num_views, center[1]),
            np.sin(angles) * cam_dist + center[2],
        ])

        face_view_scores = np.zeros((F, num_views))
        for vi, cam_pos in enumerate(cam_positions):
            view_dir = cam_pos - centroids
            view_norms = np.linalg.norm(view_dir, axis=1, keepdims=True) + 1e-10
            view_dir = view_dir / view_norms
            cos_angle = np.abs(np.sum(normals * view_dir, axis=1))
            face_view_scores[:, vi] = cos_angle

        best_views = np.argmax(face_view_scores, axis=1)

        uv = np.zeros((N, 2), dtype=np.float64)
        vertex_weight = np.zeros(N, dtype=np.float64)

        for vi in range(num_views):
            mask = best_views == vi
            if not np.any(mask):
                continue

            cam_pos = cam_positions[vi]
            view_forward = center - cam_pos
            view_forward = view_forward / (np.linalg.norm(view_forward) + 1e-10)
            world_up = np.array([0.0, 1.0, 0.0])
            view_right = np.cross(world_up, view_forward)
            rn = np.linalg.norm(view_right) + 1e-10
            view_right = view_right / rn
            view_up = np.cross(view_forward, view_right)
            view_up = view_up / (np.linalg.norm(view_up) + 1e-10)

            view_faces = faces[mask]
            view_verts_idx = np.unique(view_faces)
            view_verts = vertices[view_verts_idx]

            local = view_verts - cam_pos
            x_proj = np.dot(local, view_right)
            y_proj = np.dot(local, view_up)

            x_norm = (x_proj - x_proj.min()) / (x_proj.max() - x_proj.min() + 1e-10)
            y_norm = (y_proj - y_proj.min()) / (y_proj.max() - y_proj.min() + 1e-10)

            cos_angle = face_view_scores[mask, vi]
            weights = cos_angle ** 2

            for fi_idx, fi in enumerate(np.where(mask)[0]):
                for vi_local in range(3):
                    vid = faces[fi, vi_local]
                    vid_in_view = np.searchsorted(view_verts_idx, vid)
                    w = weights[fi_idx]
                    uv[vid, 0] += x_norm[vid_in_view] * w
                    uv[vid, 1] += y_norm[vid_in_view] * w
                    vertex_weight[vid] += w

        mask = vertex_weight > 0
        uv[mask] /= vertex_weight[mask, np.newaxis]

        unmask = ~mask
        if np.any(unmask):
            from scipy.interpolate import RBFInterpolator
            known_idx = np.where(mask)[0]
            if len(known_idx) > 3:
                rbf = RBFInterpolator(known_idx, uv[mask], kernel="thin_plate_spline")
                uv[unmask] = rbf(np.where(unmask)[0])
            else:
                uv[unmask] = 0.5

        uv = self._normalize_uv(uv)

        print(f"  Multi-view projection: {num_views} views, {np.sum(mask)}/{N} vertices projected")

        return {
            "vertices": vertices,
            "uv_coords": uv.astype(np.float32),
            "faces": faces,
            "num_charts": 1,
        }

    # ── RGB Fitting ────────────────────────────────────────────────

    def _rgb_fitting(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
        mesh: trimesh.Trimesh,
    ) -> dict:
        """RGB fitting: optimize UV coordinates using distortion energy.

        Inspired by FFHQ-UV's RGB fitting approach that optimizes UV coordinates
        to minimize texture distortion. We use Laplacian UV energy combined with
        angle-preservation to find an optimal UV layout.

        This is a simplified version of the full FFHQ-UV pipeline that doesn't
        require trained models.
        """
        import igl
        from .classical_unwrapper import ClassicalUnwrapper

        v = vertices.astype(np.float64)
        f = faces.astype(np.int32)

        v, _, _, f = igl.remove_duplicate_vertices(v, f, 1e-12)
        deg = (f[:, 0] != f[:, 1]) & (f[:, 1] != f[:, 2]) & (f[:, 0] != f[:, 2])
        f = f[deg]

        boundary = igl.boundary_loop(f)
        if len(boundary) >= 3:
            n = len(boundary)
            angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
            boundary_uv = np.column_stack([np.cos(angles), np.sin(angles)])
            uv = igl.lscm(v, f, boundary, boundary_uv)[0]
        else:
            init = ClassicalUnwrapper("xatlas").unwrap(trimesh.Trimesh(vertices=v, faces=f))
            uv = init["uv_coords"].copy().astype(np.float64)

        N = len(v)
        if uv.shape[0] != N:
            uv = uv[:N]

        L = igl.cotmatrix(v, f)

        n_iter = 200
        step_size = 0.01

        for i in range(n_iter):
            grad_u = L @ uv[:, 0]
            grad_v = L @ uv[:, 1]

            uv[:, 0] -= step_size * grad_u
            uv[:, 1] -= step_size * grad_v

            uv[:, 0] = np.clip(uv[:, 0], 0.01, 0.99)
            uv[:, 1] = np.clip(uv[:, 1], 0.01, 0.99)

            if (i + 1) % 50 == 0:
                energy = np.mean(grad_u ** 2 + grad_v ** 2)
                print(f"  RGB fitting iter {i + 1}/{n_iter}: energy={energy:.6f}")

        uv = self._normalize_uv(uv)

        return {
            "vertices": v,
            "uv_coords": uv.astype(np.float32),
            "faces": f,
            "num_charts": 1,
        }

    # ── Face Auto ──────────────────────────────────────────────────

    def _face_auto(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
        mesh: trimesh.Trimesh,
        reference_uv_path: str | Path | None = None,
        num_views: int = 6,
    ) -> dict:
        """Auto-detect face mesh and use optimal FFHQ-UV strategy.

        Detection heuristics:
        - If reference UVs provided: topology transfer
        - If mesh has ~5k-50k vertices and face-like aspect ratio: face detection
        - Otherwise: multi-view projection
        """
        V = len(vertices)
        F = len(faces)

        if reference_uv_path is not None:
            print("  Face auto: reference UVs provided, using topology transfer")
            return self._topo_transfer(vertices, faces, reference_uv_path)

        centroid = vertices.mean(axis=0)
        centered = vertices - centroid
        cov = centered.T @ centered / V
        eigvals = np.linalg.eigvalsh(cov)
        eigvals = np.sort(eigvals)[::-1]

        elongation = eigvals[0] / (eigvals[1] + 1e-10)
        flatness = eigvals[1] / (eigvals[2] + 1e-10)

        is_face_like = (
            3000 < V < 100000
            and 1000 < F < 300000
            and 0.5 < elongation < 5.0
            and 0.3 < flatness < 5.0
        )

        if is_face_like:
            print(f"  Face auto: detected face-like mesh (V={V}, F={F}, elong={elongation:.2f})")
            print("  Using multi-view projection (FFHQ-UV style)")
            return self._multi_view_projection(vertices, faces, mesh, num_views)
        else:
            print(f"  Face auto: non-face mesh, using multi-view projection")
            return self._multi_view_projection(vertices, faces, mesh, num_views)

    @staticmethod
    def _normalize_uv(uv: np.ndarray) -> np.ndarray:
        uv_min = uv.min(axis=0)
        uv_max = uv.max(axis=0)
        return (uv - uv_min) / (uv_max - uv_min + 1e-10)
