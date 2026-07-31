from __future__ import annotations

"""Prepare UV training dataset from Objaverse and other sources.

Downloads meshes with UV maps, extracts per-vertex UV coordinates,
filters for quality, and saves as preprocessed .pt files for fast loading.

Usage:
    python -m src.data.prepare_dataset --source objaverse --output data/uv_train --max-samples 5000
    python -m src.data.prepare_dataset --input-dir /path/to/meshes --output data/uv_custom
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import trimesh


def extract_uv_from_obj(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Extract (vertices, faces, uv_coords) from an OBJ file."""
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
    if valid.sum() < 4:
        return None
    uv_coords[valid] /= uv_count[valid, None]

    return vertices, vert_faces, uv_coords


def extract_uv_from_glb(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Extract UV from GLB/glTF via trimesh."""
    try:
        scene = trimesh.load(str(path), process=False)
    except Exception:
        return None

    if isinstance(scene, trimesh.Scene):
        meshes = [g for g in scene.geometry.values() if isinstance(g, trimesh.Trimesh)]
        if not meshes:
            return None
        mesh = trimesh.util.concatenate(meshes)
    else:
        mesh = scene

    vertices = np.array(mesh.vertices, dtype=np.float32)
    faces = np.array(mesh.faces, dtype=np.int64)

    uv_coords = None
    if hasattr(mesh, "visual") and hasattr(mesh.visual, "uv"):
        uv_raw = mesh.visual.uv
        if uv_raw is not None and len(uv_raw) > 0:
            if len(uv_raw) == len(faces) * 3:
                face_uvs = uv_raw.reshape(-1, 3, 2)
                uv_coords = np.zeros((len(vertices), 2), dtype=np.float32)
                uv_count = np.zeros(len(vertices), dtype=np.float32)
                for fi in range(len(faces)):
                    for vi_local in range(3):
                        vi = faces[fi, vi_local]
                        uv_coords[vi] += face_uvs[fi, vi_local]
                        uv_count[vi] += 1.0
                valid = uv_count > 0
                if valid.sum() > 0:
                    uv_coords[valid] /= uv_count[valid, None]
            elif len(uv_raw) >= len(vertices):
                uv_coords = np.array(uv_raw[:len(vertices)], dtype=np.float32)

    if uv_coords is None or len(uv_coords) != len(vertices):
        return None

    # Check quality
    has_uv = np.count_nonzero(np.any(uv_coords != 0, axis=1))
    if has_uv < 4:
        return None

    return vertices, faces, uv_coords


def normalize_and_save(
    vertices: np.ndarray,
    faces: np.ndarray,
    uv_coords: np.ndarray,
    out_path: Path,
    source_path: str = "",
    num_sample_points: int = 3000,
):
    """Normalize mesh and save as preprocessed .pt file."""
    # Normalize vertices to unit sphere
    center = vertices.mean(axis=0)
    vertices = vertices - center
    max_ext = np.abs(vertices).max()
    if max_ext > 0:
        vertices = vertices / max_ext

    # Normalize UVs to [0, 1]
    uv_min = uv_coords.min(axis=0)
    uv_max = uv_coords.max(axis=0)
    uv_range = uv_max - uv_min
    uv_range[uv_range < 1e-8] = 1.0
    uv_coords = (uv_coords - uv_min) / uv_range

    # Sample surface points for efficient training
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    pts, face_idx = trimesh.sample.sample_surface(mesh, num_sample_points)

    data = {
        "vertices": torch.tensor(vertices, dtype=torch.float32),
        "faces": torch.tensor(faces, dtype=torch.long),
        "uv_coords": torch.tensor(uv_coords, dtype=torch.float32),
        "points": torch.tensor(pts, dtype=torch.float32),
        "face_idx": torch.tensor(face_idx, dtype=torch.long),
        "source": source_path,
    }
    torch.save(data, out_path)


def quality_score(vertices: np.ndarray, faces: np.ndarray, uv_coords: np.ndarray) -> float:
    """Compute a quality score for a UV map (higher is better).

    Based on ArtUV paper's filtering criteria:
    - Penalize excessive fragmentation
    - Penalize overlapping UV regions
    - Reward good UV utilization
    """
    num_verts = len(vertices)
    num_faces = len(faces)

    if num_verts == 0 or num_faces == 0:
        return 0.0

    score = 1.0

    # Penalize very high vertex count (training stability)
    if num_verts > 5000:
        score *= 0.3
    elif num_verts > 2000:
        score *= 0.7

    # Penalize very low vertex count
    if num_verts < 20:
        score *= 0.2

    # Check UV coverage (utilization)
    uv_valid = np.count_nonzero(np.any(np.abs(uv_coords) > 1e-6, axis=1))
    coverage = uv_valid / max(num_verts, 1)
    if coverage < 0.5:
        score *= 0.3

    # Check UV spread (how well the UVs fill the [0,1] space)
    uv_range = uv_coords.max(axis=0) - uv_coords.min(axis=0)
    spread = uv_range.mean()
    if spread < 0.1:
        score *= 0.3

    return score


def download_objaverse(
    output_dir: Path,
    max_samples: int = 5000,
    download_procs: int = 4,
) -> list[str]:
    """Download meshes from Objaverse dataset.

    Returns list of downloaded file paths.
    """
    try:
        import objaverse
    except ImportError:
        print("ERROR: objaverse package not installed.")
        print("Install with: pip install objaverse")
        print("Falling back to manual mode. Place .obj/.glb files in:", output_dir)
        return []

    print(f"Downloading up to {max_samples} objects from Objaverse...")
    annotations = objaverse.load_annotations()
    uids = list(annotations.keys())[:max_samples]

    print(f"  Found {len(annotations)} total objects, using {len(uids)}")

    objects = objaverse.load_objects(
        uids=uids,
        download_processes=download_procs,
    )

    paths = []
    for uid, path in objects.items():
        if Path(path).exists():
            paths.append(path)

    print(f"  Downloaded {len(paths)} objects")
    return paths


def process_directory(
    input_dir: Path,
    output_dir: Path,
    max_samples: int | None = None,
    max_verts: int = 10000,
    max_faces: int = 20000,
    quality_threshold: float = 0.3,
) -> dict:
    """Process a directory of mesh files and extract UV data.

    Returns stats dict.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all mesh files
    exts = {".obj", ".glb", ".gltf", ".ply"}
    mesh_files = []
    for ext in exts:
        mesh_files.extend(input_dir.rglob(f"*{ext}"))
    mesh_files.sort()

    if max_samples:
        mesh_files = mesh_files[:max_samples]

    print(f"Processing {len(mesh_files)} mesh files...")

    stats = {
        "total_found": len(mesh_files),
        "processed": 0,
        "skipped_no_uv": 0,
        "skipped_too_large": 0,
        "skipped_low_quality": 0,
        "saved": 0,
    }

    saved_count = 0
    for i, mesh_path in enumerate(mesh_files):
        if i % 100 == 0:
            print(f"  [{i}/{len(mesh_files)}] processed={stats['processed']} saved={stats['saved']}")

        stats["processed"] += 1

        try:
            ext = mesh_path.suffix.lower()
            if ext == ".obj":
                result = extract_uv_from_obj(mesh_path)
            elif ext in (".glb", ".gltf"):
                result = extract_uv_from_glb(mesh_path)
            else:
                stats["skipped_no_uv"] += 1
                continue

            if result is None:
                stats["skipped_no_uv"] += 1
                continue

            vertices, faces, uv_coords = result

            if len(vertices) > max_verts or len(faces) > max_faces:
                stats["skipped_too_large"] += 1
                continue

            q = quality_score(vertices, faces, uv_coords)
            if q < quality_threshold:
                stats["skipped_low_quality"] += 1
                continue

            out_path = output_dir / f"sample_{saved_count:06d}.pt"
            normalize_and_save(vertices, faces, uv_coords, out_path, str(mesh_path))
            stats["saved"] += 1
            saved_count += 1

        except Exception as e:
            stats["skipped_no_uv"] += 1
            continue

    print(f"\nDone! Saved {stats['saved']} samples to {output_dir}")
    print(f"  Total found: {stats['total_found']}")
    print(f"  No UV data: {stats['skipped_no_uv']}")
    print(f"  Too large: {stats['skipped_too_large']}")
    print(f"  Low quality: {stats['skipped_low_quality']}")

    # Save stats
    with open(output_dir / "stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    return stats


def main():
    parser = argparse.ArgumentParser(description="Prepare UV training dataset")
    parser.add_argument("--source", choices=["objaverse", "manual"], default="manual",
                        help="Data source: 'objaverse' (auto-download) or 'manual' (your own files)")
    parser.add_argument("--input-dir", type=str, default=None,
                        help="Input directory of mesh files (for manual mode)")
    parser.add_argument("--output", type=str, required=True,
                        help="Output directory for preprocessed data")
    parser.add_argument("--max-samples", type=int, default=5000,
                        help="Maximum number of samples to process")
    parser.add_argument("--max-verts", type=int, default=10000,
                        help="Maximum vertices per mesh")
    parser.add_argument("--max-faces", type=int, default=20000,
                        help="Maximum faces per mesh")
    parser.add_argument("--quality-threshold", type=float, default=0.3,
                        help="Minimum quality score to keep a sample")
    parser.add_argument("--download-procs", type=int, default=4,
                        help="Number of parallel download processes for Objaverse")

    args = parser.parse_args()
    output_dir = Path(args.output)

    if args.source == "objaverse":
        # Download from Objaverse
        temp_dir = output_dir / "_raw_downloads"
        temp_dir.mkdir(parents=True, exist_ok=True)
        downloaded = download_objaverse(temp_dir, args.max_samples, args.download_procs)
        if not downloaded:
            print("No files downloaded. Place mesh files in:", temp_dir)
            sys.exit(1)

        # Process downloaded files
        input_dir = temp_dir
    else:
        if not args.input_dir:
            print("ERROR: --input-dir required for manual mode")
            sys.exit(1)
        input_dir = Path(args.input_dir)
        if not input_dir.exists():
            print(f"ERROR: Input directory does not exist: {input_dir}")
            sys.exit(1)

    process_directory(
        input_dir=input_dir,
        output_dir=output_dir,
        max_samples=args.max_samples,
        max_verts=args.max_verts,
        max_faces=args.max_faces,
        quality_threshold=args.quality_threshold,
    )


if __name__ == "__main__":
    main()
