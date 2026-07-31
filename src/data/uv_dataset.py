from __future__ import annotations

"""Dataset for loading 3D meshes with ground-truth UV coordinates.

Supports loading from:
- OBJ files (with vt UV coordinates)
- GLB/glTF files (with UV attributes)
- Preprocessed .pt files (vertices, faces, uv_coords)

Used for supervised training of UV unwrapping models.
"""

import json
from pathlib import Path

import numpy as np
import torch
import trimesh
from torch.utils.data import Dataset

from .mesh_io import load_mesh, normalize_mesh, sample_points


class UVDataset(Dataset):
    """Dataset of 3D meshes with ground-truth UV maps for supervised training.

    Loads meshes that have per-vertex UV coordinates and returns
    (vertices, faces, uv_coords) triples for supervised UV prediction.
    """

    SUPPORTED_EXTS = {".obj", ".glb", ".gltf", ".pt", ".npz"}

    def __init__(
        self,
        root_dir: str | Path,
        num_points: int = 3000,
        max_verts: int = 10000,
        max_faces: int = 20000,
        normalize: bool = True,
        augment: bool = False,
        split: str | None = None,
        min_uv_verts: int = 4,
    ):
        """
        Args:
            root_dir: Directory containing meshes with UV maps, or a single .pt/.npz file.
            num_points: Number of surface points to sample per mesh.
            max_verts: Skip meshes with more vertices than this.
            max_faces: Skip meshes with more faces than this.
            normalize: Whether to normalize meshes to unit sphere.
            augment: Whether to apply data augmentation.
            split: Optional 'train'/'val'/'test' split (requires splits.json in root_dir).
            min_uv_verts: Minimum vertices with valid UVs to keep a mesh.
        """
        self.root_dir = Path(root_dir)
        self.num_points = num_points
        self.max_verts = max_verts
        self.max_faces = max_faces
        self.normalize = normalize
        self.augment = augment
        self.min_uv_verts = min_uv_verts

        self.samples = self._scan_samples(split)
        print(f"UVDataset: found {len(self.samples)} valid samples in {root_dir}")

    def _scan_samples(self, split: str | None) -> list[dict]:
        """Scan directory for valid UV-mapped meshes."""
        samples = []

        # Check for preprocessed .pt/.npz files
        pt_files = list(self.root_dir.rglob("*.pt"))
        npz_files = list(self.root_dir.rglob("*.npz"))
        if pt_files or npz_files:
            for f in sorted(pt_files + npz_files):
                samples.append({"type": "preprocessed", "path": str(f)})
            return samples

        # Load split filter if available
        split_set = None
        if split:
            splits_file = self.root_dir / "splits.json"
            if splits_file.exists():
                with open(splits_file) as fp:
                    splits = json.load(fp)
                split_set = set(splits.get(split, []))

        # Scan for mesh files with UV data
        for ext in self.SUPPORTED_EXTS - {".pt", ".npz"}:
            for f in sorted(self.root_dir.rglob(f"*{ext}")):
                if split_set is not None:
                    if f.stem not in split_set:
                        continue
                samples.append({"type": "mesh", "path": str(f)})

        # Filter out meshes without UV data (try loading first few)
        valid_samples = []
        for s in samples[:500]:  # Check first 500 to avoid slow startup
            try:
                uv_data = self._try_extract_uv(s["path"])
                if uv_data is not None:
                    verts, faces, uvs = uv_data
                    if len(verts) <= self.max_verts and len(faces) <= self.max_faces:
                        uv_count = np.count_nonzero(np.any(uvs != 0, axis=1))
                        if uv_count >= self.min_uv_verts:
                            valid_samples.append(s)
            except Exception:
                continue

        # If we scanned a subset, include remaining unscanned files
        if len(samples) > 500:
            valid_samples.extend(samples[500:])

        return valid_samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        sample_info = self.samples[idx]

        try:
            if sample_info["type"] == "preprocessed":
                return self._load_preprocessed(sample_info["path"])
            else:
                return self._load_mesh(sample_info["path"])
        except Exception as e:
            return self._dummy_sample()

    def _load_mesh(self, path: str) -> dict[str, torch.Tensor]:
        """Load a mesh file and extract UV coordinates."""
        uv_data = self._try_extract_uv(path)
        if uv_data is None:
            return self._dummy_sample()

        vertices, faces, uv_coords = uv_data

        if self.normalize:
            # Normalize vertices
            center = vertices.mean(axis=0)
            vertices = vertices - center
            max_ext = np.abs(vertices).max()
            if max_ext > 0:
                vertices = vertices / max_ext

            # Normalize UVs to [0, 1] range
            uv_min = uv_coords.min(axis=0)
            uv_max = uv_coords.max(axis=0)
            uv_range = uv_max - uv_min
            uv_range[uv_range < 1e-8] = 1.0
            uv_coords = (uv_coords - uv_min) / uv_range

        # Create trimesh for sampling
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)

        # Sample surface points
        points, normals, face_idx = sample_points(mesh, self.num_points)

        # Compute edges
        edges = _compute_edges_fast(faces)

        # Augmentation
        if self.augment:
            vertices, points, normals, uv_coords = self._augment(
                vertices, points, normals, uv_coords
            )

        result = {
            "vertices": torch.tensor(vertices, dtype=torch.float32),
            "faces": torch.tensor(faces, dtype=torch.long),
            "uv_coords": torch.tensor(uv_coords, dtype=torch.float32),
            "points": points,
            "face_idx": face_idx,
            "edges": edges,
            "path": path,
        }
        if normals is not None:
            result["normals"] = normals
        return result

    def _load_preprocessed(self, path: str) -> dict[str, torch.Tensor]:
        """Load preprocessed .pt or .npz file."""
        if path.endswith(".pt"):
            data = torch.load(path, map_location="cpu", weights_only=False)
        elif path.endswith(".npz"):
            data_np = np.load(path, allow_pickle=True)
            data = {k: torch.tensor(v) for k, v in data_np.items()}
        else:
            return self._dummy_sample()

        pts = data.get("points", data["vertices"]).float()
        num_pts = pts.shape[0]
        return {
            "vertices": data["vertices"].float(),
            "faces": data["faces"].long(),
            "uv_coords": data["uv_coords"].float(),
            "points": pts,
            "face_idx": data.get("face_idx", torch.zeros(num_pts, dtype=torch.long)),
            "edges": _compute_edges_fast(data["faces"].long()),
            "path": path,
        }

    def _try_extract_uv(self, path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        """Try to extract vertices, faces, and UV coordinates from a mesh file.

        Returns (vertices, faces, uv_coords) or None if no UV data found.
        """
        path_lower = path.lower()

        if path_lower.endswith(".obj"):
            return self._extract_obj_uv(path)
        elif path_lower.endswith((".glb", ".gltf")):
            return self._extract_glb_uv(path)
        return None

    def _extract_obj_uv(self, path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        """Extract UV from OBJ file by parsing vt and f directives."""
        verts = []
        uvs_raw = []
        uv_faces = []
        vert_faces = []

        with open(path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue
                if parts[0] == "v":
                    verts.append([float(x) for x in parts[1:4]])
                elif parts[0] == "vt":
                    uvs_raw.append([float(x) for x in parts[1:3]])
                elif parts[0] == "f":
                    face_verts = []
                    face_uvs = []
                    for p in parts[1:]:
                        ids = p.split("/")
                        face_verts.append(int(ids[0]) - 1)
                        if len(ids) >= 2 and ids[1]:
                            face_uvs.append(int(ids[1]) - 1)
                        else:
                            face_uvs.append(int(ids[0]) - 1)
                    vert_faces.append(face_verts)
                    uv_faces.append(face_uvs)

        if not verts or not uvs_raw or not vert_faces:
            return None

        vertices = np.array(verts, dtype=np.float32)
        uvs_all = np.array(uvs_raw, dtype=np.float32)
        vert_faces = np.array(vert_faces, dtype=np.int64)
        uv_faces = np.array(uv_faces, dtype=np.int64)

        # Build per-vertex UV by averaging UVs at shared vertices
        uv_coords = np.zeros((len(vertices), 2), dtype=np.float32)
        uv_count = np.zeros(len(vertices), dtype=np.float32)

        for fi in range(len(vert_faces)):
            for vi_local in range(len(vert_faces[fi])):
                vi = vert_faces[fi, vi_local]
                ui = uv_faces[fi, vi_local]
                if 0 <= ui < len(uvs_all):
                    uv_coords[vi] += uvs_all[ui]
                    uv_count[vi] += 1.0

        valid = uv_count > 0
        uv_coords[valid] /= uv_count[valid, None]

        return vertices, vert_faces, uv_coords

    def _extract_glb_uv(self, path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        """Extract UV from GLB/glTF file via trimesh."""
        scene = trimesh.load(path, process=False)
        if isinstance(scene, trimesh.Scene):
            meshes = [g for g in scene.geometry.values() if isinstance(g, trimesh.Trimesh)]
            if not meshes:
                return None
            mesh = trimesh.util.concatenate(meshes)
        else:
            mesh = scene

        vertices = np.array(mesh.vertices, dtype=np.float32)
        faces = np.array(mesh.faces, dtype=np.int64)

        # Try to get UV coordinates from visual
        uv_coords = None
        if hasattr(mesh, "visual") and hasattr(mesh.visual, "uv"):
            uv_raw = mesh.visual.uv
            if uv_raw is not None and len(uv_raw) > 0:
                # UV may be per-face-vertex; expand to per-vertex
                if len(uv_raw) == len(faces) * 3:
                    # Per-face-vertex UV layout: reshape to (F, 3, 2)
                    face_uvs = uv_raw.reshape(-1, 3, 2)
                    uv_coords = np.zeros((len(vertices), 2), dtype=np.float32)
                    uv_count = np.zeros(len(vertices), dtype=np.float32)
                    for fi in range(len(faces)):
                        for vi_local in range(3):
                            vi = faces[fi, vi_local]
                            uv_coords[vi] += face_uvs[fi, vi_local]
                            uv_count[vi] += 1.0
                    valid = uv_count > 0
                    uv_coords[valid] /= uv_count[valid, None]
                elif len(uv_raw) >= len(vertices):
                    uv_coords = np.array(uv_raw[:len(vertices)], dtype=np.float32)

        if uv_coords is None or len(uv_coords) != len(vertices):
            return None

        return vertices, faces, uv_coords

    def _augment(
        self,
        vertices: np.ndarray,
        points: torch.Tensor,
        normals: torch.Tensor | None,
        uv_coords: np.ndarray,
    ) -> tuple[np.ndarray, torch.Tensor, torch.Tensor | None, np.ndarray]:
        """Apply data augmentation: random rotation, scaling, jitter."""
        # Random rotation
        if np.random.random() > 0.5:
            A = np.random.randn(3, 3).astype(np.float32)
            Q, _ = np.linalg.qr(A)
            if np.linalg.det(Q) < 0:
                Q[:, 0] = -Q[:, 0]
            vertices = vertices @ Q.T
            points = points @ torch.tensor(Q.T, dtype=torch.float32)
            if normals is not None:
                normals = normals @ torch.tensor(Q.T, dtype=torch.float32)
            # UV coords stay the same (rotation shouldn't affect UVs)

        # Random scale [0.95, 1.05]
        if np.random.random() > 0.5:
            scale = np.random.uniform(0.95, 1.05)
            vertices *= scale
            points *= scale

        # Random vertex jitter
        if np.random.random() > 0.7:
            jitter = np.random.normal(0, 0.01, vertices.shape).astype(np.float32)
            vertices += jitter

        return vertices, points, normals, uv_coords

    def _dummy_sample(self) -> dict[str, torch.Tensor]:
        """Return a unit cube with default UVs as fallback."""
        v = torch.tensor([
            [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
            [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1],
        ], dtype=torch.float32)
        f = torch.tensor([
            [0, 1, 2], [0, 2, 3], [4, 6, 5], [4, 7, 6],
            [0, 4, 5], [0, 5, 1], [1, 5, 6], [1, 6, 2],
            [2, 6, 7], [2, 7, 3], [3, 7, 4], [3, 4, 0],
        ], dtype=torch.long)
        uv = torch.tensor([
            [0, 0], [1, 0], [1, 1], [0, 1],
            [0, 0], [1, 0], [1, 1], [0, 1],
        ], dtype=torch.float32)
        edges = _compute_edges_fast(f)
        return {
            "vertices": v,
            "faces": f,
            "uv_coords": uv,
            "points": v.unsqueeze(0).expand(self.num_points, -1, -1).reshape(-1, 3)[:self.num_points],
            "face_idx": torch.zeros(self.num_points, dtype=torch.long),
            "edges": edges,
            "path": "dummy",
        }


def _compute_edges_fast(faces: torch.Tensor) -> torch.Tensor:
    """Fast edge extraction from faces."""
    edges = set()
    f_np = faces.numpy()
    for face in f_np:
        for i in range(len(face)):
            e = (int(min(face[i], face[(i + 1) % len(face)])),
                 int(max(face[i], face[(i + 1) % len(face)])))
            edges.add(e)
    if not edges:
        return torch.zeros((0, 2), dtype=torch.long)
    return torch.tensor(sorted(edges), dtype=torch.long)
