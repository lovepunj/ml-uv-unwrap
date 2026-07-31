from __future__ import annotations

"""Mesh type detector — classifies 3D project type from geometric features.

Analyzes mesh topology, geometry, and PartField semantic features to
determine the type of 3D project (character, architecture, vehicle, etc.)
and recommend optimal unwrapping strategies.

Usage:
    detector = MeshTypeDetector()
    result = detector.detect(mesh)
    print(result["project_type"])  # "character", "architecture", etc.
    print(result["recommended_strategy"])  # "multi_chart", "single_chart", etc.
"""

from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import numpy as np
import torch
import trimesh


class ProjectType(str, Enum):
    """Detected 3D project types."""
    CHARACTER = "character"
    ARCHITECTURE = "architecture"
    VEHICLE = "vehicle"
    ORGANIC = "organic"
    HARD_SURFACE = "hard_surface"
    TERRAIN = "terrain"
    FURNITURE = "furniture"
    WEAPON = "weapon"
    PLANT = "plant"
    MYSTERY = "unknown"


@dataclass
class MeshFeatures:
    """Extracted geometric features for classification."""
    num_vertices: int
    num_faces: int
    num_edges: int
    surface_area: float
    volume: float
    bounding_box_diag: float
    aspect_ratio: float
    compactness: float
    convexity: float
    genus: int
    euler_number: int
    num_boundaries: int
    edge_length_mean: float
    edge_length_std: float
    face_area_mean: float
    face_area_std: float
    dihedral_angle_mean: float
    dihedral_angle_std: float
    normal_consistency: float
    symmetry_score: float
    elongation: float
    planarity: float
    sphericity: float


@dataclass
class DetectionResult:
    """Mesh type detection result."""
    project_type: ProjectType
    confidence: float
    features: MeshFeatures
    recommended_strategy: str
    recommended_charts: int
    recommended_method: str
    complexity_score: float
    part_count_estimate: int
    analysis_notes: list[str]


class MeshTypeDetector:
    """Detect 3D project type and recommend unwrapping strategy.

    Uses geometric analysis and optional PartField features to classify
    the mesh and suggest optimal unwrapping parameters.
    """

    def __init__(self, partfield_extractor=None):
        """
        Args:
            partfield_extractor: Optional PartFieldFeatureExtractor for
                               semantic feature analysis.
        """
        self.partfield_extractor = partfield_extractor

    def detect(
        self,
        mesh: trimesh.Trimesh | str | Path,
        use_partfield: bool = False,
    ) -> DetectionResult:
        """Detect mesh type and recommend unwrapping strategy.

        Args:
            mesh: Input mesh (trimesh object or file path)
            use_partfield: Whether to extract PartField features for analysis

        Returns:
            DetectionResult with project type, confidence, and recommendations
        """
        # Load mesh if needed
        if isinstance(mesh, (str, Path)):
            mesh = trimesh.load(mesh, force="mesh")
            if isinstance(mesh, trimesh.Scene):
                mesh = trimesh.util.concatenate(mesh.dump())

        # Extract geometric features
        features = self._extract_features(mesh)

        # Classify based on features
        project_type, confidence, notes = self._classify(features)

        # Get PartField analysis if available
        part_count = 1
        if use_partfield and self.partfield_extractor is not None:
            part_count, partfield_notes = self._analyze_partfield(mesh)
            notes.extend(partfield_notes)

        # Recommend strategy
        strategy, charts, method = self._recommend_strategy(
            project_type, features, part_count
        )

        # Compute complexity score
        complexity = self._compute_complexity(features, part_count)

        return DetectionResult(
            project_type=project_type,
            confidence=confidence,
            features=features,
            recommended_strategy=strategy,
            recommended_charts=charts,
            recommended_method=method,
            complexity_score=complexity,
            part_count_estimate=part_count,
            analysis_notes=notes,
        )

    def _extract_features(self, mesh: trimesh.Trimesh) -> MeshFeatures:
        """Extract geometric features from mesh."""
        vertices = np.array(mesh.vertices, dtype=np.float64)
        faces = np.array(mesh.faces, dtype=np.int64)
        edges = np.array(mesh.edges_unique, dtype=np.int64) if hasattr(mesh, 'edges_unique') else np.array([], dtype=np.int64)

        # Basic counts
        num_vertices = len(vertices)
        num_faces = len(faces)
        num_edges = len(edges) if len(edges.shape) > 1 else 0

        # Surface area and volume
        surface_area = mesh.area if hasattr(mesh, 'area') else 0.0
        volume = mesh.volume if hasattr(mesh, 'is_watertight') and mesh.is_watertight else 0.0

        # Bounding box
        bbox_min = vertices.min(axis=0)
        bbox_max = vertices.max(axis=0)
        bbox_diag = np.linalg.norm(bbox_max - bbox_min)
        bbox_size = bbox_max - bbox_min
        aspect_ratio = bbox_max.max() / (bbox_min.min() + 1e-10)

        # Compactness (surface area relative to volume)
        compactness = surface_area / (volume ** (2/3) + 1e-10) if volume > 0 else 1.0

        # Convexity (volume ratio to convex hull)
        try:
            convex_hull = mesh.convex_hull
            convexity = volume / (convex_hull.volume + 1e-10) if convex_hull.volume > 0 else 1.0
        except:
            convexity = 1.0

        # Euler characteristic and genus
        euler_number = num_vertices - num_edges + num_faces
        genus = max(0, (2 - euler_number) // 2)

        # Boundary count
        num_boundaries = 0
        if hasattr(mesh, 'edges_unique'):
            try:
                boundary_edges = mesh.edges_unique[mesh.edges_unique_length == 1] if hasattr(mesh, 'edges_unique_length') else np.array([])
                num_boundaries = len(np.unique(boundary_edges)) if len(boundary_edges) > 0 else 0
            except:
                num_boundaries = 0

        # Edge lengths
        if num_edges > 0:
            edge_vectors = vertices[edges[:, 1]] - vertices[edges[:, 0]]
            edge_lengths = np.linalg.norm(edge_vectors, axis=1)
            edge_length_mean = float(np.mean(edge_lengths))
            edge_length_std = float(np.std(edge_lengths))
        else:
            edge_length_mean = 0.0
            edge_length_std = 0.0

        # Face areas
        if num_faces > 0:
            face_areas = mesh.area_faces if hasattr(mesh, 'area_faces') else np.zeros(num_faces)
            face_area_mean = float(np.mean(face_areas))
            face_area_std = float(np.std(face_areas))
        else:
            face_area_mean = 0.0
            face_area_std = 0.0

        # Dihedral angles
        dihedral_angles = self._compute_dihedral_angles(mesh)
        dihedral_angle_mean = float(np.mean(dihedral_angles)) if len(dihedral_angles) > 0 else 0.0
        dihedral_angle_std = float(np.std(dihedral_angles)) if len(dihedral_angles) > 0 else 0.0

        # Normal consistency
        normal_consistency = self._compute_normal_consistency(mesh)

        # Symmetry score
        symmetry_score = self._compute_symmetry_score(mesh)

        # Shape descriptors
        elongation = bbox_size.max() / (bbox_size.min() + 1e-10)
        planarity = self._compute_planarity(vertices)
        sphericity = self._compute_sphericity(surface_area, volume)

        return MeshFeatures(
            num_vertices=num_vertices,
            num_faces=num_faces,
            num_edges=num_edges,
            surface_area=surface_area,
            volume=volume,
            bounding_box_diag=bbox_diag,
            aspect_ratio=aspect_ratio,
            compactness=compactness,
            convexity=convexity,
            genus=genus,
            euler_number=euler_number,
            num_boundaries=num_boundaries,
            edge_length_mean=edge_length_mean,
            edge_length_std=edge_length_std,
            face_area_mean=face_area_mean,
            face_area_std=face_area_std,
            dihedral_angle_mean=dihedral_angle_mean,
            dihedral_angle_std=dihedral_angle_std,
            normal_consistency=normal_consistency,
            symmetry_score=symmetry_score,
            elongation=elongation,
            planarity=planarity,
            sphericity=sphericity,
        )

    def _compute_dihedral_angles(self, mesh: trimesh.Trimesh) -> np.ndarray:
        """Compute dihedral angles at edges."""
        if not hasattr(mesh, 'edges_unique') or not hasattr(mesh, 'face_adjacency'):
            return np.array([])

        try:
            # Get adjacent faces
            adjacent = mesh.face_adjacency
            if len(adjacent) == 0:
                return np.array([])

            # Compute face normals
            normals = mesh.face_normals

            # Get normal pairs for adjacent faces
            n1 = normals[adjacent[:, 0]]
            n2 = normals[adjacent[:, 1]]

            # Compute angle between normals
            dot = np.sum(n1 * n2, axis=1)
            dot = np.clip(dot, -1.0, 1.0)
            angles = np.arccos(dot)

            return angles
        except:
            return np.array([])

    def _compute_normal_consistency(self, mesh: trimesh.Trimesh) -> float:
        """Compute consistency of normals across the mesh."""
        if not hasattr(mesh, 'face_normals'):
            return 0.0

        try:
            normals = mesh.face_normals
            # Compute mean normal
            mean_normal = np.mean(normals, axis=0)
            mean_normal = mean_normal / (np.linalg.norm(mean_normal) + 1e-10)

            # Compute consistency as mean dot product with mean normal
            dot_products = np.sum(normals * mean_normal, axis=1)
            consistency = float(np.mean(np.abs(dot_products)))
            return consistency
        except:
            return 0.0

    def _compute_symmetry_score(self, mesh: trimesh.Trimesh) -> float:
        """Compute symmetry score of the mesh."""
        if not hasattr(mesh, 'vertices'):
            return 0.0

        try:
            vertices = np.array(mesh.vertices, dtype=np.float64)

            # Center the mesh
            centroid = np.mean(vertices, axis=0)
            centered = vertices - centroid

            # Compute principal axes
            cov = np.cov(centered.T)
            eigenvalues, eigenvectors = np.linalg.eigh(cov)

            # Check symmetry along each principal axis
            symmetry_scores = []
            for axis in eigenvectors:
                # Project vertices onto axis
                projections = np.dot(centered, axis)

                # Check if distribution is symmetric around 0
                pos_projections = projections[projections > 0]
                neg_projections = projections[projections < 0]

                if len(pos_projections) > 0 and len(neg_projections) > 0:
                    # Compare histograms
                    max_proj = max(np.max(pos_projections), -np.min(neg_projections))
                    bins = np.linspace(-max_proj, max_proj, 20)

                    pos_hist, _ = np.histogram(pos_projections, bins=bins, density=True)
                    neg_hist, _ = np.histogram(-neg_projections, bins=bins, density=True)

                    # Symmetry score based on histogram similarity
                    score = 1.0 - np.mean(np.abs(pos_hist - neg_hist))
                    symmetry_scores.append(score)

            return float(np.mean(symmetry_scores)) if symmetry_scores else 0.0
        except:
            return 0.0

    def _compute_planarity(self, vertices: np.ndarray) -> float:
        """Compute planarity of the mesh."""
        if len(vertices) < 3:
            return 0.0

        try:
            # Center vertices
            centroid = np.mean(vertices, axis=0)
            centered = vertices - centroid

            # Compute covariance matrix
            cov = np.cov(centered.T)
            eigenvalues, _ = np.linalg.eigh(cov)

            # Planarity: ratio of middle to largest eigenvalue
            eigenvalues = np.sort(eigenvalues)[::-1]
            if eigenvalues[0] > 0:
                planarity = eigenvalues[1] / eigenvalues[0]
                return float(planarity)
            return 0.0
        except:
            return 0.0

    def _compute_sphericity(self, surface_area: float, volume: float) -> float:
        """Compute sphericity of the mesh."""
        if volume <= 0 or surface_area <= 0:
            return 0.0

        # Sphericity: ratio of sphere surface area to mesh surface area
        # for the same volume
        sphere_radius = (3 * volume / (4 * np.pi)) ** (1/3)
        sphere_surface_area = 4 * np.pi * sphere_radius ** 2

        sphericity = sphere_surface_area / surface_area
        return float(min(1.0, sphericity))

    def _classify(
        self, features: MeshFeatures
    ) -> tuple[ProjectType, float, list[str]]:
        """Classify mesh type based on features."""
        notes = []
        scores = {t: 0.0 for t in ProjectType}

        # Character detection
        if features.compactness > 1.5 and features.convexity > 0.7:
            scores[ProjectType.CHARACTER] += 0.3
        if features.genus == 0 and features.num_boundaries == 0:
            scores[ProjectType.CHARACTER] += 0.2
        if features.symmetry_score > 0.6:
            scores[ProjectType.CHARACTER] += 0.2
        if features.elongation < 2.0 and features.sphericity > 0.5:
            scores[ProjectType.CHARACTER] += 0.1

        # Architecture detection
        if features.elongation > 3.0:
            scores[ProjectType.ARCHITECTURE] += 0.3
        if features.planarity > 0.8:
            scores[ProjectType.ARCHITECTURE] += 0.2
        if features.dihedral_angle_std < 0.1:
            scores[ProjectType.ARCHITECTURE] += 0.2
        if features.compactness < 1.0:
            scores[ProjectType.ARCHITECTURE] += 0.1

        # Vehicle detection
        if features.elongation > 2.0 and features.elongation < 5.0:
            scores[ProjectType.VEHICLE] += 0.2
        if features.convexity > 0.6 and features.compactness < 1.5:
            scores[ProjectType.VEHICLE] += 0.2
        if features.symmetry_score > 0.7:
            scores[ProjectType.VEHICLE] += 0.2

        # Organic detection
        if features.compactness > 2.0 and features.dihedral_angle_std > 0.3:
            scores[ProjectType.ORGANIC] += 0.3
        if features.normal_consistency < 0.8:
            scores[ProjectType.ORGANIC] += 0.2
        if features.face_area_std / (features.face_area_mean + 1e-10) > 0.5:
            scores[ProjectType.ORGANIC] += 0.2

        # Hard surface detection
        if features.dihedral_angle_std < 0.2 and features.normal_consistency > 0.9:
            scores[ProjectType.HARD_SURFACE] += 0.3
        if features.edge_length_std / (features.edge_length_mean + 1e-10) < 0.3:
            scores[ProjectType.HARD_SURFACE] += 0.2
        if features.planarity > 0.7:
            scores[ProjectType.HARD_SURFACE] += 0.2

        # Terrain detection
        if features.elongation > 1.5 and features.planarity > 0.6:
            scores[ProjectType.TERRAIN] += 0.3
        if features.num_boundaries > 0:
            scores[ProjectType.TERRAIN] += 0.2
        if features.compactness < 0.8:
            scores[ProjectType.TERRAIN] += 0.2

        # Furniture detection
        if features.elongation < 3.0 and features.compactness < 1.5:
            scores[ProjectType.FURNITURE] += 0.2
        if features.dihedral_angle_mean > 0.5:
            scores[ProjectType.FURNITURE] += 0.2
        if features.convexity > 0.5:
            scores[ProjectType.FURNITURE] += 0.2

        # Weapon detection
        if features.elongation > 3.0 and features.compactness < 1.0:
            scores[ProjectType.WEAPON] += 0.3
        if features.dihedral_angle_std < 0.15:
            scores[ProjectType.WEAPON] += 0.2

        # Plant detection
        if features.genus > 2:
            scores[ProjectType.PLANT] += 0.3
        if features.compactness > 2.5:
            scores[ProjectType.PLANT] += 0.2
        if features.normal_consistency < 0.7:
            scores[ProjectType.PLANT] += 0.2

        # Find best match
        best_type = max(scores, key=scores.get)
        confidence = scores[best_type]

        # Normalize confidence
        total_score = sum(scores.values())
        if total_score > 0:
            confidence = confidence / total_score
        else:
            confidence = 0.0

        # Add notes
        notes.append(f"Geometric analysis: compactness={features.compactness:.2f}, "
                     f"convexity={features.convexity:.2f}, "
                     f"elongation={features.elongation:.2f}")
        notes.append(f"Topology: genus={features.genus}, "
                     f"boundaries={features.num_boundaries}, "
                     f"euler={features.euler_number}")
        notes.append(f"Symmetry: {features.symmetry_score:.2f}, "
                     f"planarity: {features.planarity:.2f}")

        return best_type, confidence, notes

    def _analyze_partfield(
        self, mesh: trimesh.Trimesh
    ) -> tuple[int, list[str]]:
        """Analyze mesh using PartField features."""
        notes = []
        part_count = 1

        if self.partfield_extractor is None:
            return part_count, notes

        try:
            import copy
            mesh_copy = copy.deepcopy(mesh)
            features = self.partfield_extractor.extract(mesh_copy, sample_on_faces=10)

            # Cluster into parts
            labels = self.partfield_extractor.cluster_parts(features, max_parts=15)
            part_count = len(np.unique(labels))

            # Analyze part distribution
            unique, counts = np.unique(labels, return_counts=True)
            mean_part_size = np.mean(counts)
            std_part_size = np.std(counts)
            size_ratio = std_part_size / (mean_part_size + 1e-10)

            notes.append(f"PartField analysis: {part_count} parts detected, "
                        f"size ratio: {size_ratio:.2f}")

            # Detect if parts are well-separated
            if size_ratio > 0.5:
                notes.append("Parts have varied sizes - likely semantic decomposition")
            else:
                notes.append("Parts are similar in size - likely geometric decomposition")

        except Exception as e:
            notes.append(f"PartField analysis failed: {str(e)}")

        return part_count, notes

    def _recommend_strategy(
        self,
        project_type: ProjectType,
        features: MeshFeatures,
        part_count: int,
    ) -> tuple[str, int, str]:
        """Recommend unwrapping strategy based on analysis."""
        # Base recommendations by project type
        strategy_map = {
            ProjectType.CHARACTER: ("multi_chart", 6, "xatlas"),
            ProjectType.ARCHITECTURE: ("single_chart", 1, "xatlas"),
            ProjectType.VEHICLE: ("multi_chart", 4, "xatlas"),
            ProjectType.ORGANIC: ("multi_chart", 5, "xatlas"),
            ProjectType.HARD_SURFACE: ("single_chart", 1, "xatlas"),
            ProjectType.TERRAIN: ("single_chart", 1, "xatlas"),
            ProjectType.FURNITURE: ("multi_chart", 3, "xatlas"),
            ProjectType.WEAPON: ("multi_chart", 3, "xatlas"),
            ProjectType.PLANT: ("multi_chart", 4, "xatlas"),
            ProjectType.MYSTERY: ("multi_chart", 4, "xatlas"),
        }

        strategy, charts, method = strategy_map[project_type]

        # Adjust based on mesh complexity
        if features.num_faces > 10000:
            charts = min(charts + 2, 10)
            strategy = "multi_chart"
        elif features.num_faces < 500:
            charts = 1
            strategy = "single_chart"

        # Adjust based on PartField part count
        if part_count > 1:
            charts = max(charts, part_count)
            if part_count > 5:
                strategy = "multi_chart"

        # Adjust based on geometry
        if features.genus > 3:
            charts = max(charts, features.genus)
            strategy = "multi_chart"

        if features.num_boundaries > 0:
            # Has holes - may need more charts
            charts = max(charts, features.num_boundaries + 1)

        return strategy, charts, method

    def _compute_complexity(
        self, features: MeshFeatures, part_count: int
    ) -> float:
        """Compute overall complexity score (0-1)."""
        # Normalize features to 0-1 range
        vertex_score = min(1.0, features.num_vertices / 100000)
        face_score = min(1.0, features.num_faces / 200000)
        genus_score = min(1.0, features.genus / 10)
        part_score = min(1.0, part_count / 15)
        symmetry_penalty = features.symmetry_score * 0.2

        complexity = (
            vertex_score * 0.2 +
            face_score * 0.2 +
            genus_score * 0.2 +
            part_score * 0.2 +
            (1 - symmetry_penalty) * 0.2
        )

        return float(np.clip(complexity, 0.0, 1.0))
