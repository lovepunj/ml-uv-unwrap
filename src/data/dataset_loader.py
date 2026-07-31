from __future__ import annotations

"""Open source 3D model dataset loaders for mesh classification training.

Supports:
- Thingi10K: 10K 3D printing models with rich semantic tags (via thingi10k package)
- ShapeNetCore: 55 categories, 51K models (via HuggingFace datasets)
- Synthetic: built-in test meshes for quick validation

Usage:
    loader = DatasetLoader(cache_dir="./datasets")
    for mesh, label in loader.iter_thingi10k(max_samples=500):
        ...
    for mesh, label in loader.iter_shapenet(max_samples=1000):
        ...
"""

import json
import os
from pathlib import Path
from typing import Generator

import numpy as np
import trimesh

# Thingi10K tag -> ProjectType mapping (grouped by semantic similarity)
THINGI10K_TAG_MAP = {
    # Characters / Animals
    "animal": "organic", "dragon": "organic", "monster": "character",
    "human": "character", "skull": "character", "face": "character",
    "head": "character", "hand": "character", "figurine": "character",
    "toy": "character", "doll": "character", "robot": "character",
    "alien": "character", "zombie": "character", "pirate": "character",
    "fairy": "character", "goblin": "character", "orc": "character",
    "skeleton": "character", "warrior": "character", "superhero": "character",
    "cat": "character", "dog": "character", "horse": "character",
    "bird": "character", "fish": "character", "butterfly": "character",
    "insect": "organic", "spider": "organic", "frog": "character",
    "dinosaur": "character", "creature": "character", "person": "character",
    "statue": "character", "sculpture": "organic", "bust": "character",
    # Vehicles
    "car": "vehicle", "airplane": "vehicle", "aircraft": "vehicle",
    "jet": "vehicle", "helicopter": "vehicle", "drone": "vehicle",
    "boat": "vehicle", "ship": "vehicle", "submarine": "vehicle",
    "train": "vehicle", "locomotive": "vehicle", "bus": "vehicle",
    "truck": "vehicle", "motorcycle": "vehicle", "bicycle": "vehicle",
    "bike": "vehicle", "scooter": "vehicle", "skateboard": "vehicle",
    "rocket": "vehicle", "spaceship": "vehicle", "ufo": "vehicle",
    "tank": "vehicle", "jeep": "vehicle", "van": "vehicle",
    # Architecture / Buildings
    "building": "architecture", "house": "architecture", "castle": "architecture",
    "tower": "architecture", "bridge": "architecture", "church": "architecture",
    "temple": "architecture", "pyramid": "architecture", "wall": "architecture",
    "door": "architecture", "window": "architecture", "column": "architecture",
    "stairs": "architecture", "fence": "architecture", "gate": "architecture",
    "roof": "architecture", "chimney": "architecture", "dome": "architecture",
    # Furniture
    "chair": "furniture", "table": "furniture", "desk": "furniture",
    "sofa": "furniture", "couch": "furniture", "bed": "furniture",
    "shelf": "furniture", "cabinet": "furniture", "drawer": "furniture",
    "bookcase": "furniture", "wardrobe": "furniture", "stool": "furniture",
    "bench": "furniture", "ottoman": "furniture", "cabinet": "furniture",
    "lamp": "furniture", "light": "furniture", "chandelier": "furniture",
    # Weapons
    "gun": "weapon", "pistol": "weapon", "rifle": "weapon",
    "sword": "weapon", "knife": "weapon", "blade": "weapon",
    "axe": "weapon", "bow": "weapon", "arrow": "weapon",
    "cannon": "weapon", "weapon": "weapon", "blade": "weapon",
    "dagger": "weapon", "spear": "weapon", "shield": "weapon",
    # Hard surface / Mechanical
    "gear": "hard_surface", "cog": "hard_surface", "spring": "hard_surface",
    "bolt": "hard_surface", "nut": "hard_surface", "screw": "hard_surface",
    "pipe": "hard_surface", "valve": "hard_surface", "engine": "hard_surface",
    "motor": "hard_surface", "machine": "hard_surface", "tool": "hard_surface",
    "wrench": "hard_surface", "hammer": "hard_surface", "drill": "hard_surface",
    "key": "hard_surface", "lock": "hard_surface", "hinge": "hard_surface",
    # Plants
    "plant": "plant", "tree": "plant", "flower": "plant",
    "leaf": "plant", "mushroom": "plant", "cactus": "plant",
    "vine": "plant", "bush": "plant", "grass": "plant",
    "pot": "plant", "vase": "plant",
    # Terrain
    "terrain": "terrain", "landscape": "terrain", "mountain": "terrain",
    "hill": "terrain", "valley": "terrain", "cave": "terrain",
    "rock": "terrain", "stone": "terrain", "ground": "terrain",
    # Tech / Misc
    "phone": "hard_surface", "computer": "hard_surface",
    "keyboard": "hard_surface", "mouse": "hard_surface",
    "camera": "hard_surface", "speaker": "hard_surface",
    "headphones": "hard_surface", "microphone": "hard_surface",
    "tv": "hard_surface", "monitor": "hard_surface",
    "printer": "hard_surface", "scanner": "hard_surface",
    "keyboard": "hard_surface", "remote": "hard_surface",
    # Jewelry
    "ring": "hard_surface", "necklace": "hard_surface",
    "bracelet": "hard_surface", "earring": "hard_surface",
    "pendant": "hard_surface", "gem": "hard_surface",
    "diamond": "hard_surface", "crown": "hard_surface",
    # Food
    "food": "organic", "fruit": "organic", "cake": "organic",
    "pizza": "organic", "cupcake": "organic", "cookie": "organic",
    "bread": "organic", "apple": "organic", "banana": "organic",
    # Containers
    "bottle": "hard_surface", "cup": "hard_surface", "mug": "hard_surface",
    "glass": "hard_surface", "bowl": "hard_surface", "plate": "hard_surface",
    "jar": "hard_surface", "can": "hard_surface", "box": "hard_surface",
    "basket": "hard_surface", "bag": "hard_surface",
}


class DatasetLoader:
    """Load 3D models from open source datasets."""

    def __init__(self, cache_dir: str | Path = "./datasets"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def iter_thingi10k(
        self,
        max_samples: int = 500,
        min_faces: int = 100,
        max_faces: int = 100000,
    ) -> Generator[tuple[trimesh.Trimesh, str], None, None]:
        """Iterate over Thingi10K dataset.

        Yields (mesh, project_type_label) tuples.
        Requires: pip install thingi10k
        """
        try:
            import thingi10k
        except ImportError:
            print("thingi10k not installed. Run: pip install thingi10k")
            return

        thingi10k.init(cache_dir=str(self.cache_dir / "thingi10k"))

        count = 0
        for entry in thingi10k.dataset():
            if count >= max_samples:
                break

            try:
                # Load mesh
                vertices, faces = thingi10k.load_file(entry["file_path"])
                if vertices is None or faces is None:
                    continue

                # Filter by face count
                if len(faces) < min_faces or len(faces) > max_faces:
                    continue

                mesh = trimesh.Trimesh(
                    vertices=np.array(vertices, dtype=np.float64),
                    faces=np.array(faces, dtype=np.int64),
                )

                # Get project type from tags
                tags = entry.get("tags", [])
                if isinstance(tags, str):
                    tags = tags.split(",")
                tags = [t.strip().lower() for t in tags]

                project_type = self._tags_to_type(tags)
                if project_type is None:
                    continue

                count += 1
                yield mesh, project_type

            except Exception:
                continue

    def iter_shapenet(
        self,
        max_samples: int = 1000,
        min_faces: int = 100,
        max_faces: int = 100000,
    ) -> Generator[tuple[trimesh.Trimesh, str], None, None]:
        """Iterate over ShapeNetCore dataset.

        Yields (mesh, project_type_label) tuples.
        Requires ShapeNetCore downloaded to cache_dir/shapenet/
        """
        shapenet_dir = self.cache_dir / "shapenet"
        if not shapenet_dir.exists():
            print(f"ShapeNet not found at {shapenet_dir}")
            print("Download from https://huggingface.co/ShapeNet/ShapeNetCore")
            return

        count = 0
        for category_dir in sorted(shapenet_dir.iterdir()):
            if not category_dir.is_dir():
                continue

            # Map ShapeNet category to project type
            category_name = category_dir.name.lower()
            project_type = self._shapenet_category_to_type(category_name)
            if project_type is None:
                continue

            for model_dir in sorted(category_dir.iterdir()):
                if count >= max_samples:
                    return

                if not model_dir.is_dir():
                    continue

                try:
                    # Find OBJ file
                    obj_files = list(model_dir.glob("*.obj"))
                    if not obj_files:
                        continue

                    mesh = trimesh.load(obj_files[0], force="mesh")
                    if isinstance(mesh, trimesh.Scene):
                        mesh = trimesh.util.concatenate(mesh.dump())

                    if len(mesh.faces) < min_faces or len(mesh.faces) > max_faces:
                        continue

                    count += 1
                    yield mesh, project_type

                except Exception:
                    continue

    def _tags_to_type(self, tags: list[str]) -> str | None:
        """Convert Thingi10K tags to project type."""
        # Check tags in reverse priority order (more specific first)
        for tag in tags:
            if tag in THINGI10K_TAG_MAP:
                return THINGI10K_TAG_MAP[tag]

        # Try partial matching
        for tag in tags:
            for key, value in THINGI10K_TAG_MAP.items():
                if key in tag or tag in key:
                    return value

        return None

    def _shapenet_category_to_type(self, category: str) -> str | None:
        """Map ShapeNet category name to project type."""
        # Simple keyword-based mapping
        vehicle_kw = ["car", "airplane", "aircraft", "boat", "ship", "train",
                       "bus", "truck", "motorcycle", "bicycle", "bike",
                       "helicopter", "jet", "rocket", "vehicle", "tank"]
        char_kw = ["person", "human", "animal", "cat", "dog", "horse",
                    "bird", "fish", "figure", "figurine", "toy"]
        arch_kw = ["building", "house", "tower", "bridge", "church",
                    "structure", "wall", "column", "architecture"]
        furn_kw = ["chair", "table", "desk", "sofa", "bed", "shelf",
                    "cabinet", "lamp", "furniture", "stool", "bench"]
        weapon_kw = ["gun", "rifle", "pistol", "sword", "knife", "weapon"]
        organic_kw = ["plant", "flower", "tree", "mushroom", "organic"]
        hard_kw = ["gear", "engine", "motor", "tool", "mechanical",
                    "hardware", "electronic"]

        for kw in vehicle_kw:
            if kw in category:
                return "vehicle"
        for kw in char_kw:
            if kw in category:
                return "character"
        for kw in arch_kw:
            if kw in category:
                return "architecture"
        for kw in furn_kw:
            if kw in category:
                return "furniture"
        for kw in weapon_kw:
            if kw in category:
                return "weapon"
        for kw in organic_kw:
            if kw in category:
                return "organic"
        for kw in hard_kw:
            if kw in category:
                return "hard_surface"

        return None


class SyntheticDataset:
    """Built-in test meshes for quick validation."""

    TEST_MESHES = {
        "character": trimesh.creation.icosphere(subdivisions=3),
        "vehicle": trimesh.creation.box(extents=[4, 2, 1.5]),
        "architecture": trimesh.creation.box(extents=[10, 8, 12]),
        "furniture": trimesh.creation.cylinder(radius=2, height=4),
        "weapon": trimesh.creation.cylinder(radius=0.3, height=8),
        "organic": trimesh.creation.cone(radius=2, height=4),
        "hard_surface": trimesh.creation.icosphere(subdivisions=2),
        "terrain": trimesh.creation.box(extents=[20, 20, 1]),
        "plant": trimesh.creation.cylinder(radius=1, height=6),
    }

    @classmethod
    def get_test_meshes(cls) -> Generator[tuple[trimesh.Trimesh, str], None, None]:
        """Yield test meshes with known labels."""
        for label, mesh in cls.TEST_MESHES.items():
            yield mesh.copy(), label

    @classmethod
    def get_mesh(cls, label: str) -> trimesh.Trimesh:
        """Get a single test mesh by label."""
        if label not in cls.TEST_MESHES:
            raise ValueError(f"Unknown label: {label}. Choose from: {list(cls.TEST_MESHES.keys())}")
        return cls.TEST_MESHES[label].copy()
