from __future__ import annotations

"""Classical UV parameterization backends using xatlas and libigl.

These provide high-quality fallback methods when ML-based approaches
are not suitable or as comparison baselines.
"""

from pathlib import Path

import numpy as np


class XAtlasBackend:
    """UV parameterization using xatlas.

    xatlas is a fast, high-quality mesh parameterization library
    used in production game engines and 3D tools.
    """

    def __init__(self):
        try:
            import xatlas
            self.xatlas = xatlas
        except ImportError:
            raise ImportError("xatlas not installed. Run: pip install xatlas")

    def unwrap(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
        max_chart_count: int = 0,
        max_iterations: int = 1,
    ) -> dict[str, np.ndarray]:
        """Unwrap mesh using xatlas.

        Args:
            vertices: (V, 3) vertex positions
            faces: (F, 3) face indices
            max_chart_count: max charts (0 = auto)
            max_iterations: parameterization iterations

        Returns:
            Dictionary with UV coords and updated face indices
        """
        # xatlas expects float32 vertices
        vertices = vertices.astype(np.float32)
        faces = faces.astype(np.uint32)

        # Parameterize
        atlas = self.xatlas.Atlas()

        # Add mesh
        atlas.add_mesh(vertices, faces)

        # Generate parameterization
        atlas.generate(
            max_chart_number=max_chart_count if max_chart_count > 0 else 0,
            max_iterations=max_iterations,
        )

        # Get results
        new_vertices = atlas.get_vertices()
        new_uv = atlas.get_uvs()
        new_faces = atlas.get_faces()

        return {
            "vertices": new_vertices,
            "uv_coords": new_uv[:, :2],  # xatlas returns 3D UVs, take first 2
            "faces": new_faces,
            "num_charts": len(atlas.get_chart_ids()) if hasattr(atlas, 'get_chart_ids') else 0,
        }

    def unwrap_with_charts(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
    ) -> dict[str, np.ndarray]:
        """Unwrap with chart detection.

        Returns:
            Dictionary with UVs, faces, and chart assignments
        """
        result = self.unwrap(vertices, faces)

        # xatlas provides chart IDs per face
        try:
            chart_ids = self.xatlas.get_chart_ids()
            result["face_chart_ids"] = chart_ids
            result["num_charts"] = len(np.unique(chart_ids))
        except AttributeError:
            result["num_charts"] = 1

        return result


class LibIGLBackend:
    """UV parameterization using libigl.

    Provides access to industry-standard parameterization methods:
    - LSCM (Least Squares Conformal Maps)
    - ABF++ (Angle Based Flattening)
    - Harmonic maps
    """

    def __init__(self):
        try:
            import igl
            self.igl = igl
        except ImportError:
            raise ImportError("libigl not installed. Run: pip install libigl")

    def lscm(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
        boundary_uv: tuple[np.ndarray, np.ndarray] | None = None,
    ) -> dict[str, np.ndarray]:
        """Least Squares Conformal Map parameterization.

        Args:
            vertices: (V, 3) vertex positions
            faces: (F, 3) face indices
            boundary_uv: optional (boundary_indices, boundary_uv_coords)

        Returns:
            Dictionary with UV coordinates
        """
        vertices = vertices.astype(np.float64)
        faces = faces.astype(np.int32)

        # Find boundary
        boundary = self.igl.boundary_loop(faces)

        if len(boundary) == 0:
            # No boundary found, use first two vertices
            boundary = np.array([0, 1], dtype=np.int32)

        # Create boundary UVs
        if boundary_uv is None:
            # Place boundary on a circle
            n = len(boundary)
            angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
            boundary_uv_coords = np.column_stack([np.cos(angles), np.sin(angles)])
        else:
            boundary_uv_coords = boundary_uv[1].astype(np.float64)

        # Compute LSCM
        uv = self.igl.lscm(vertices, faces, boundary, boundary_uv_coords)[0]

        # Normalize to [0, 1]
        uv_min = uv.min(axis=0)
        uv_max = uv.max(axis=0)
        uv = (uv - uv_min) / (uv_max - uv_min + 1e-10)

        return {
            "uv_coords": uv.astype(np.float32),
            "vertices": vertices.astype(np.float32),
            "faces": faces,
        }

    def abf(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
    ) -> dict[str, np.ndarray]:
        """ABF++ (Angle Based Flattening) parameterization.

        Produces angle-preserving UV maps with low distortion.
        """
        vertices = vertices.astype(np.float64)
        faces = faces.astype(np.int32)

        try:
            # Try libigl's ABF
            uv = self.igl.abf(vertices, faces)
        except AttributeError:
            # Fall back to harmonic map if ABF not available
            boundary = self.igl.boundary_loop(faces)
            if len(boundary) < 3:
                boundary = np.array([0, 1, 2], dtype=np.int32)
            n = len(boundary)
            angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
            boundary_uv = np.column_stack([np.cos(angles), np.sin(angles)])
            uv = self.igl.harmonic(vertices, faces, boundary, boundary_uv, 1)[0]

        # Normalize
        uv_min = uv.min(axis=0)
        uv_max = uv.max(axis=0)
        uv = (uv - uv_min) / (uv_max - uv_min + 1e-10)

        return {
            "uv_coords": uv.astype(np.float32),
            "vertices": vertices.astype(np.float32),
            "faces": faces,
        }

    def harmonic(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
        boundary_indices: np.ndarray,
        boundary_uv: np.ndarray,
        power: int = 1,
    ) -> dict[str, np.ndarray]:
        """Harmonic map parameterization.

        Args:
            vertices: (V, 3) vertex positions
            faces: (F, 3) face indices
            boundary_indices: (B,) boundary vertex indices
            boundary_uv: (B, 2) boundary UV coordinates
            power: harmonic power (1 = harmonic, 2 = biharmonic)

        Returns:
            Dictionary with UV coordinates
        """
        vertices = vertices.astype(np.float64)
        faces = faces.astype(np.int32)
        boundary_indices = boundary_indices.astype(np.int32)
        boundary_uv = boundary_uv.astype(np.float64)

        uv = self.igl.harmonic(
            vertices, faces, boundary_indices, boundary_uv, power
        )[0]

        # Normalize
        uv_min = uv.min(axis=0)
        uv_max = uv.max(axis=0)
        uv = (uv - uv_min) / (uv_max - uv_min + 1e-10)

        return {
            "uv_coords": uv.astype(np.float32),
            "vertices": vertices.astype(np.float32),
            "faces": faces,
        }


def unwrap_mesh(
    vertices: np.ndarray,
    faces: np.ndarray,
    method: str = "xatlas",
    **kwargs,
) -> dict[str, np.ndarray]:
    """High-level function to unwrap a mesh using classical methods.

    Args:
        vertices: (V, 3) vertex positions
        faces: (F, 3) face indices
        method: 'xatlas', 'lscm', 'abf', or 'harmonic'
        **kwargs: additional arguments for the method

    Returns:
        Dictionary with UV coordinates and metadata
    """
    if method == "xatlas":
        backend = XAtlasBackend()
        return backend.unwrap(vertices, faces, **kwargs)
    elif method == "lscm":
        backend = LibIGLBackend()
        return backend.lscm(vertices, faces, **kwargs)
    elif method == "abf":
        backend = LibIGLBackend()
        return backend.abf(vertices, faces, **kwargs)
    else:
        raise ValueError(f"Unknown method: {method}. Use 'xatlas', 'lscm', or 'abf'.")
