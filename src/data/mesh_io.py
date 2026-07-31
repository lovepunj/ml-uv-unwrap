from __future__ import annotations

"""Mesh I/O utilities using trimesh (plus FBX via assimp, USD/USDZ via usd-core)."""

from pathlib import Path

import numpy as np
import torch
import trimesh

#: File extensions accepted by :func:`load_mesh`.
SUPPORTED_FORMATS = {
    ".obj",
    ".ply",
    ".stl",
    ".off",
    ".glb",
    ".gltf",
    ".fbx",
    ".usd",
    ".usda",
    ".usdc",
    ".usdz",
}

_USD_FORMATS = {".usd", ".usda", ".usdc", ".usdz"}


def load_mesh(path: str | Path) -> trimesh.Trimesh:
    """Load a mesh from file, normalizing scenes to a single Trimesh.

    Supported formats: OBJ, PLY, STL, OFF, GLB, GLTF, FBX (assimp), and
    USD/USDA/USDC/USDZ (Pixar usd-core).

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the format is unsupported, unreadable, or the file
            contains no mesh geometry.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_FORMATS:
        raise ValueError(
            f"Unsupported file format '{suffix}'. "
            f"Supported: {', '.join(sorted(SUPPORTED_FORMATS))}"
        )

    if suffix in _USD_FORMATS:
        mesh = _load_usd(path)
    elif suffix == ".fbx":
        mesh = _load_fbx(path)
    else:
        mesh = trimesh.load(path, force="mesh")
        if isinstance(mesh, trimesh.Scene):
            if not mesh.geometry:
                raise ValueError(f"No geometry found in '{path.name}'")
            mesh = trimesh.util.concatenate(mesh.dump())

    if mesh is None or len(mesh.vertices) == 0 or len(mesh.faces) == 0:
        raise ValueError(
            f"'{path.name}' contains no mesh geometry (0 vertices or 0 faces)"
        )

    return mesh


def _load_usd(path: Path) -> trimesh.Trimesh:
    """Load a USD/USDA/USDC/USDZ file (via Pixar usd-core) into a Trimesh."""
    try:
        from pxr import Usd, UsdGeom
    except ImportError:
        raise ValueError(
            "USD/USDZ support requires the 'usd-core' package "
            "(pip install usd-core)"
        ) from None

    try:
        stage = Usd.Stage.Open(str(path))
    except Exception as e:
        raise ValueError(f"Failed to open USD file '{path.name}': {e}") from e
    if stage is None:
        raise ValueError(f"Failed to open USD file '{path.name}'")

    meshes = []
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Mesh):
            continue
        mesh_prim = UsdGeom.Mesh(prim)
        points = mesh_prim.GetPointsAttr().Get()
        counts = mesh_prim.GetFaceVertexCountsAttr().Get() or []
        indices = mesh_prim.GetFaceVertexIndicesAttr().Get() or []
        if not points or not counts or not indices:
            continue

        # Build faces, fan-triangulating any n-gons so trimesh only sees triangles.
        faces = []
        i = 0
        for count in counts:
            if i + count > len(indices):
                break
            face = [int(x) for x in indices[i:i + count]]
            if len(face) == 3:
                faces.append(face)
            elif len(face) > 3:
                for j in range(1, len(face) - 1):
                    faces.append([face[0], face[j], face[j + 1]])
            i += count

        if faces:
            tri = trimesh.Trimesh(
                vertices=np.asarray(points, dtype=np.float64),
                faces=np.asarray(faces, dtype=np.int64),
                process=True,
            )
            if hasattr(tri, "remove_degenerate_faces"):
                tri.remove_degenerate_faces()
            if len(tri.faces):
                meshes.append(tri)

    if not meshes:
        raise ValueError(f"No meshes (UsdGeom.Mesh) found in '{path.name}'")
    if len(meshes) == 1:
        return meshes[0]
    return trimesh.util.concatenate(meshes)


def _load_fbx(path: Path) -> trimesh.Trimesh:
    """Load an FBX file (via assimp) into a Trimesh."""
    try:
        import assimp_py
    except ImportError:
        raise ValueError(
            "FBX support requires the 'assimp-py' package (pip install assimp-py)"
        ) from None

    flags = (
        assimp_py.Process_Triangulate
        | assimp_py.Process_JoinIdenticalVertices
        | assimp_py.Process_GenNormals
        | assimp_py.Process_ValidateDataStructure
    )
    try:
        scene = assimp_py.import_file(str(path), flags)
    except Exception as e:
        raise ValueError(f"Failed to parse FBX file '{path.name}': {e}") from e

    meshes = []
    for m in scene.meshes:
        verts = np.asarray(m.vertices, dtype=np.float64).reshape(-1, 3)
        indices = np.asarray(m.indices, dtype=np.int64).reshape(-1, 3)
        if len(verts) == 0 or len(indices) == 0:
            continue
        tri = trimesh.Trimesh(vertices=verts, faces=indices, process=True)
        if hasattr(tri, "remove_degenerate_faces"):
            tri.remove_degenerate_faces()
        if len(tri.faces):
            meshes.append(tri)

    if not meshes:
        raise ValueError(f"No meshes found in '{path.name}'")
    if len(meshes) == 1:
        return meshes[0]
    return trimesh.util.concatenate(meshes)


def mesh_to_tensors(
    mesh: trimesh.Trimesh,
) -> dict[str, torch.Tensor]:
    """Convert trimesh to PyTorch tensors.

    Returns:
        vertices: (V, 3) float tensor
        faces: (F, 3) long tensor
        edges: (E, 2) long tensor (unique edges from faces)
        normals: (V, 3) vertex normals
    """
    vertices = torch.tensor(mesh.vertices, dtype=torch.float32)
    faces = torch.tensor(mesh.faces, dtype=torch.long)

    # Compute edges from faces
    edges = _compute_edges(faces)

    # Vertex normals
    if mesh.vertex_normals is not None and len(mesh.vertex_normals) == len(mesh.vertices):
        normals = torch.tensor(mesh.vertex_normals, dtype=torch.float32)
    else:
        normals = _compute_vertex_normals(vertices, faces)

    return {
        "vertices": vertices,
        "faces": faces,
        "edges": edges,
        "normals": normals,
    }


def sample_points(
    mesh: trimesh.Trimesh,
    num_points: int,
    method: str = "face_area",
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
    """Sample points uniformly from mesh surface.

    Args:
        mesh: trimesh mesh
        num_points: number of points to sample
        method: 'uniform' | 'face_area' | 'poisson'

    Returns:
        points: (N, 3) sampled surface points
        normals: (N, 3) corresponding normals (if available)
        face_idx: (N,) index of the face each point was sampled from
    """
    if method == "uniform":
        points, face_idx = trimesh.sample.sample_surface(mesh, num_points)
    elif method == "face_area":
        points, face_idx = trimesh.sample.sample_surface(
            mesh, num_points, sample_color=False
        )
    elif method == "poisson":
        points, face_idx = trimesh.sample.sample_surface(
            mesh, num_points, sample_color=False
        )
    else:
        raise ValueError(f"Unknown sampling method: {method}")

    # Get normals at sampled points
    normals = None
    if mesh.vertex_normals is not None and len(mesh.vertex_normals) > 0:
        face_normals = mesh.face_normals[face_idx]
        normals = torch.tensor(face_normals, dtype=torch.float32)

    points = torch.tensor(points, dtype=torch.float32)
    face_idx = torch.tensor(face_idx, dtype=torch.long)
    return points, normals, face_idx


def interpolate_uv_barycentric(
    uv_points: np.ndarray,
    points: np.ndarray,
    vertices: np.ndarray,
    faces: np.ndarray,
    face_idx: np.ndarray,
) -> np.ndarray:
    """Interpolate per-point UVs to per-vertex UVs using barycentric coordinates.

    Args:
        uv_points: (N, 2) UV coordinates for sampled points
        points: (N, 3) sampled point positions
        vertices: (V, 3) mesh vertex positions
        faces: (F, 3) face indices
        face_idx: (N,) which face each sampled point belongs to

    Returns:
        (V, 2) UV coordinates for each vertex
    """
    num_verts = len(vertices)
    uv_per_vertex = np.zeros((num_verts, 2), dtype=np.float32)
    vert_count = np.zeros(num_verts, dtype=np.float32)

    for i in range(len(points)):
        fi = face_idx[i]
        v0, v1, v2 = faces[fi]
        p0, p1, p2 = vertices[v0], vertices[v1], vertices[v2]
        pt = points[i]

        # Barycentric coordinates
        v0v1 = p1 - p0
        v0v2 = p2 - p0
        v0pt = pt - p0

        dot00 = np.dot(v0v1, v0v1)
        dot01 = np.dot(v0v1, v0v2)
        dot02 = np.dot(v0v1, v0pt)
        dot11 = np.dot(v0v2, v0v2)
        dot12 = np.dot(v0v2, v0pt)

        denom = dot00 * dot11 - dot01 * dot01 + 1e-12
        b1 = (dot11 * dot02 - dot01 * dot12) / denom
        b2 = (dot00 * dot12 - dot01 * dot02) / denom
        b0 = 1.0 - b1 - b2

        uv = uv_points[i]
        for b, v in [(b0, v0), (b1, v1), (b2, v2)]:
            if b > 0:
                uv_per_vertex[v] += b * uv
                vert_count[v] += b

    # Average by accumulated barycentric weights
    valid = vert_count > 1e-8
    uv_per_vertex[valid] /= vert_count[valid, None]

    # Fallback for vertices with no contributions: nearest sampled point
    if not valid.all():
        missing = np.where(~valid)[0]
        verts_t = torch.tensor(vertices[missing], dtype=torch.float32)
        pts_t = torch.tensor(points, dtype=torch.float32)
        dists = torch.cdist(verts_t, pts_t)
        nearest = dists.argmin(dim=1)
        uv_per_vertex[missing] = uv_points[nearest.numpy()]

    return uv_per_vertex


def normalize_mesh(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """Normalize mesh to fit in unit sphere."""
    vertices = mesh.vertices.copy()
    center = vertices.mean(axis=0)
    vertices -= center
    max_extent = np.abs(vertices).max()
    if max_extent > 0:
        vertices /= max_extent
    mesh.vertices = vertices
    return mesh


def save_obj(
    path: str | Path,
    vertices: np.ndarray,
    uv_coords: np.ndarray,
    faces: np.ndarray | None = None,
    uv_faces: np.ndarray | None = None,
):
    """Save mesh with UV coordinates to OBJ format."""
    with open(path, "w") as f:
        f.write("# ML UV Unwrap output\n")
        for v in vertices:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for uv in uv_coords:
            f.write(f"vt {uv[0]:.6f} {uv[1]:.6f}\n")
        if faces is not None:
            for i, face in enumerate(faces):
                # OBJ is 1-indexed
                v_ids = face + 1
                if uv_faces is not None:
                    uv_ids = uv_faces[i] + 1
                else:
                    uv_ids = v_ids
                parts = []
                for vi, uvi in zip(v_ids, uv_ids):
                    parts.append(f"{vi}/{uvi}")
                f.write(f"f {' '.join(parts)}\n")


def _compute_edges(faces: torch.Tensor) -> torch.Tensor:
    """Extract unique edges from face array."""
    edges = set()
    for face in faces:
        for i in range(len(face)):
            e = tuple(sorted([face[i].item(), face[(i + 1) % len(face)].item()]))
            edges.add(e)
    return torch.tensor(list(edges), dtype=torch.long)


def _compute_vertex_normals(
    vertices: torch.Tensor,
    faces: torch.Tensor,
) -> torch.Tensor:
    """Compute per-vertex normals by averaging face normals."""
    normals = torch.zeros_like(vertices)
    for face in faces:
        v0, v1, v2 = vertices[face[0]], vertices[face[1]], vertices[face[2]]
        face_normal = torch.cross(v1 - v0, v2 - v0)
        face_normal = face_normal / (face_normal.norm() + 1e-8)
        for idx in face:
            normals[idx] += face_normal
    normals = normals / (normals.norm(dim=-1, keepdim=True) + 1e-8)
    return normals
