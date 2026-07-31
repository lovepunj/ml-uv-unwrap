from __future__ import annotations

"""PartField-guided chart decomposition for multi-chart UV unwrapping.

Uses PartField semantic features to decompose a mesh into meaningful
chart regions, then unwraps each chart independently for lower distortion.

Key insight: PartField's 448-dim per-face features encode semantic part
structure. Faces belonging to the same semantic part should be in the
same UV chart.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import trimesh


@dataclass
class ChartDecomposition:
    """Result of chart decomposition."""
    face_labels: np.ndarray          # (num_faces,) chart ID per face
    chart_count: int                 # number of charts
    chart_faces: dict[int, np.ndarray]  # chart_id -> face indices
    chart_meshes: list[trimesh.Trimesh]  # sub-mesh per chart
    chart_vert_maps: list[np.ndarray] # chart_id -> (chart_vert_idx -> original_vert_idx)
    boundary_edges: np.ndarray       # (E, 2) edges at chart boundaries
    face_features: torch.Tensor      # (num_faces, 448) PartField features


class PartFieldChartDecomposer:
    """Decompose mesh into UV charts using PartField semantic features.

    Algorithm (PartUV-inspired recursive splitting):
    1. Extract PartField features per face (448-dim)
    2. Start with one chart covering all faces
    3. Recursively split chart if distortion > threshold
    4. Split using PartField feature bisection along dominant axis
    5. Stop when all charts below distortion threshold or max_charts reached

    This produces fewer, cleaner charts than fixed-count clustering.
    """

    def __init__(
        self,
        partfield_extractor=None,
        device: str | None = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._extractor = partfield_extractor

    def decompose(
        self,
        mesh: trimesh.Trimesh,
        num_charts: int | None = None,
        max_charts: int = 10,
        min_chart_faces: int = 50,
        distortion_threshold: float = 0.15,
        progress_callback=None,
    ) -> ChartDecomposition:
        """Decompose mesh into charts using recursive splitting.

        Args:
            mesh: Input trimesh mesh
            num_charts: Fixed number of charts (None = auto-detect via distortion)
            max_charts: Maximum charts allowed
            min_chart_faces: Minimum faces per chart
            distortion_threshold: max conformal distortion before splitting

        Returns:
            ChartDecomposition with per-face labels and sub-meshes
        """
        # 1. Extract PartField features
        if progress_callback:
            progress_callback(10, 100, {"stage": "extracting_features"})
        face_features = self._extract_features(mesh)
        num_faces = len(mesh.faces)

        # 2. Cluster into charts via recursive distortion-based splitting
        if progress_callback:
            progress_callback(30, 100, {"stage": "splitting_charts"})
        face_labels = self._recursive_split(
            mesh, face_features, num_faces,
            target_charts=num_charts,
            max_charts=max_charts,
            min_chart_faces=min_chart_faces,
            distortion_threshold=distortion_threshold,
        )

        # 3. Build per-chart face index maps
        if progress_callback:
            progress_callback(70, 100, {"stage": "building_charts"})
        unique_labels = np.unique(face_labels)
        chart_faces = {}
        for label in unique_labels:
            chart_faces[int(label)] = np.where(face_labels == label)[0]

        # 4. Build per-chart sub-meshes and vertex maps
        chart_meshes, chart_vert_maps = self._build_chart_meshes(mesh, chart_faces)

        # 5. Detect boundary edges
        if progress_callback:
            progress_callback(85, 100, {"stage": "detecting_boundaries"})
        boundary_edges = self._find_boundary_edges(mesh, face_labels)

        return ChartDecomposition(
            face_labels=face_labels,
            chart_count=len(unique_labels),
            chart_faces=chart_faces,
            chart_meshes=chart_meshes,
            chart_vert_maps=chart_vert_maps,
            boundary_edges=boundary_edges,
            face_features=face_features,
        )

    def _extract_features(self, mesh: trimesh.Trimesh) -> torch.Tensor:
        """Extract PartField features per face."""
        if self._extractor is not None:
            import copy
            mesh_copy = copy.deepcopy(mesh)
            features = self._extractor.extract(mesh_copy, sample_on_faces=10)
            return features.cpu()
        else:
            # Fallback: use geometric features per face
            return self._geometric_features(mesh)

    def _geometric_features(self, mesh: trimesh.Trimesh) -> torch.Tensor:
        """Compute simple geometric features per face (fallback when no PartField)."""
        vertices = np.array(mesh.vertices, dtype=np.float64)
        faces = np.array(mesh.faces, dtype=np.int64)
        num_faces = len(faces)

        # Face centroids
        v0 = vertices[faces[:, 0]]
        v1 = vertices[faces[:, 1]]
        v2 = vertices[faces[:, 2]]
        centroids = (v0 + v1 + v2) / 3.0

        # Face normals
        e1 = v1 - v0
        e2 = v2 - v0
        normals = np.cross(e1, e2)
        norms = np.linalg.norm(normals, axis=1, keepdims=True) + 1e-10
        normals = normals / norms

        # Face areas
        areas = norms.squeeze() / 2.0

        # Edge lengths
        edges_01 = np.linalg.norm(v1 - v0, axis=1)
        edges_02 = np.linalg.norm(v2 - v0, axis=1)
        edges_12 = np.linalg.norm(v2 - v1, axis=1)

        # Dihedral angles with adjacent faces
        dihedral = self._compute_face_dihedrals(mesh)

        # Combine into feature vector
        features = np.column_stack([
            centroids,           # 3
            normals,             # 3
            areas.reshape(-1, 1),  # 1
            edges_01.reshape(-1, 1),  # 1
            edges_02.reshape(-1, 1),  # 1
            edges_12.reshape(-1, 1),  # 1
            dihedral.reshape(-1, 1),  # 1
        ])  # Total: 10 features

        # Pad to 448 with zeros (to match PartField dim)
        padded = np.zeros((num_faces, 448), dtype=np.float32)
        padded[:, :features.shape[1]] = features.astype(np.float32)

        return torch.tensor(padded, dtype=torch.float32)

    def _compute_face_dihedrals(self, mesh: trimesh.Trimesh) -> np.ndarray:
        """Compute mean dihedral angle per face."""
        num_faces = len(mesh.faces)
        dihedral = np.zeros(num_faces, dtype=np.float64)

        if not hasattr(mesh, 'face_adjacency') or len(mesh.face_adjacency) == 0:
            return dihedral

        try:
            adj = mesh.face_adjacency
            normals = mesh.face_normals

            n1 = normals[adj[:, 0]]
            n2 = normals[adj[:, 1]]
            dot = np.sum(n1 * n2, axis=1)
            dot = np.clip(dot, -1.0, 1.0)
            angles = np.arccos(dot)

            # Accumulate per face
            np.add.at(dihedral, adj[:, 0], angles)
            np.add.at(dihedral, adj[:, 1], angles)

            # Normalize by adjacency count
            counts = np.zeros(num_faces, dtype=np.float64)
            np.add.at(counts, adj[:, 0], 1)
            np.add.at(counts, adj[:, 1], 1)
            counts = np.maximum(counts, 1)
            dihedral /= counts
        except Exception:
            pass

        return dihedral

    def _estimate_chart_count(
        self,
        mesh: trimesh.Trimesh,
        features: torch.Tensor,
        max_charts: int,
    ) -> int:
        """Automatically estimate optimal number of charts."""
        num_faces = len(mesh.faces)

        # Base estimate from mesh size
        if num_faces < 500:
            return 1
        elif num_faces < 2000:
            base = 2
        elif num_faces < 10000:
            base = 4
        elif num_faces < 50000:
            base = 6
        else:
            base = 8

        # Adjust based on feature variance
        feat_np = features.numpy()
        feat_var = np.var(feat_np, axis=0).mean()
        if feat_var > 0.1:
            base = min(base + 2, max_charts)

        # Adjust based on topology
        genus = 0
        try:
            euler = len(mesh.vertices) - len(mesh.edges_unique) + len(mesh.faces)
            genus = max(0, (2 - euler) // 2)
        except Exception:
            pass

        if genus > 2:
            base = min(base + genus, max_charts)

        return min(base, max_charts)

    def _cluster_faces(
        self,
        features: torch.Tensor,
        num_charts: int,
        min_chart_faces: int,
    ) -> np.ndarray:
        """Cluster faces into charts using agglomerative clustering."""
        from sklearn.cluster import AgglomerativeClustering
        from sklearn.preprocessing import StandardScaler

        feat_np = features.numpy()

        # Normalize features
        scaler = StandardScaler()
        feat_scaled = scaler.fit_transform(feat_np)

        # Cluster
        n_clusters = min(num_charts, len(feat_np) // min_chart_faces)
        n_clusters = max(1, n_clusters)

        clustering = AgglomerativeClustering(
            n_clusters=n_clusters,
            linkage="ward",
        )
        labels = clustering.fit_predict(feat_scaled)

        # Merge small charts
        labels = self._merge_small_charts(labels, min_chart_faces)

        return labels

    def _recursive_split(
        self,
        mesh: trimesh.Trimesh,
        features: torch.Tensor,
        num_faces: int,
        target_charts: int | None = None,
        max_charts: int = 10,
        min_chart_faces: int = 50,
        distortion_threshold: float = 0.15,
    ) -> np.ndarray:
        """Recursively split charts based on distortion threshold.

        PartUV-inspired: start with one chart, split along PartField
        feature axis if distortion exceeds threshold.
        """
        faces = np.array(mesh.faces, dtype=np.int64)
        vertices = np.array(mesh.vertices, dtype=np.float64)
        feat_np = features.numpy()

        labels = np.zeros(num_faces, dtype=np.int64)
        next_label = 1

        # Queue of (face_indices, chart_label) to process
        all_face_idx = np.arange(num_faces)
        queue = [(all_face_idx, 0)]

        while queue and (target_charts is None or next_label < target_charts):
            if target_charts is not None and len(queue) + next_label >= target_charts:
                break
            if next_label >= max_charts:
                break

            face_idx, current_label = queue.pop(0)

            # Check minimum size
            if len(face_idx) < min_chart_faces * 2:
                labels[face_idx] = current_label
                continue

            # Compute distortion for this subset
            chart_faces_sub = faces[face_idx]
            distortion = self._compute_chart_distortion(vertices, chart_faces_sub)

            if distortion < distortion_threshold and target_charts is None:
                labels[face_idx] = current_label
                continue

            # Split along dominant PartField feature axis
            sub_features = feat_np[face_idx]
            split_mask = self._find_best_split(sub_features)

            if split_mask.sum() < min_chart_faces or (~split_mask).sum() < min_chart_faces:
                labels[face_idx] = current_label
                continue

            # Assign current label to first half
            labels[face_idx[split_mask]] = current_label

            # Queue second half with new label
            labels[face_idx[~split_mask]] = next_label
            queue.append((face_idx[~split_mask], next_label))
            next_label += 1

        # Assign remaining queued items
        for face_idx, lbl in queue:
            labels[face_idx] = lbl

        # Merge small charts
        labels = self._merge_small_charts(labels, min_chart_faces)

        return labels

    def _compute_chart_distortion(
        self, vertices: np.ndarray, faces: np.ndarray
    ) -> float:
        """Compute mean conformal distortion for a set of faces."""
        if len(faces) == 0:
            return 0.0

        v0 = vertices[faces[:, 0]]
        v1 = vertices[faces[:, 1]]
        v2 = vertices[faces[:, 2]]

        e1 = np.linalg.norm(v1 - v0, axis=1) + 1e-10
        e2 = np.linalg.norm(v2 - v0, axis=1) + 1e-10
        e3 = np.linalg.norm(v2 - v1, axis=1) + 1e-10

        # Use edge ratios as distortion proxy
        s1 = e1 / (e1.mean() + 1e-10)
        s2 = e2 / (e2.mean() + 1e-10)
        s3 = e3 / (e3.mean() + 1e-10)

        distortion = np.mean((s1 - 1.0) ** 2 + (s2 - 1.0) ** 2 + (s3 - 1.0) ** 2)
        return float(distortion)

    def _find_best_split(self, features: np.ndarray) -> np.ndarray:
        """Find best axis to split features along (median split).

        Returns boolean mask of True for first half, False for second half.
        """
        # Use PCA to find dominant axis
        feat_centered = features - features.mean(axis=0)
        cov = feat_centered.T @ feat_centered
        try:
            eigvals, eigvecs = np.linalg.eigh(cov)
            dominant_axis = eigvecs[:, -1]  # Largest eigenvalue
        except Exception:
            dominant_axis = np.ones(features.shape[1]) / np.sqrt(features.shape[1])

        # Project onto dominant axis
        projections = feat_centered @ dominant_axis

        # Split at median
        median = np.median(projections)
        mask = projections >= median

        return mask

    def _merge_small_charts(
        self,
        labels: np.ndarray,
        min_faces: int,
    ) -> np.ndarray:
        """Merge charts with too few faces into nearest neighbor."""
        unique, counts = np.unique(labels, return_counts=True)

        small_charts = unique[counts < min_faces]
        large_charts = unique[counts >= min_faces]

        if len(large_charts) == 0:
            return labels

        for small in small_charts:
            # Find nearest large chart
            small_mask = labels == small
            best_chart = large_charts[0]
            best_score = -1

            for large in large_charts:
                large_mask = labels == large
                # Simple adjacency heuristic: count shared edges
                score = np.sum(small_mask & np.roll(large_mask, 1))
                if score > best_score:
                    best_score = score
                    best_chart = large

            labels[small_mask] = best_chart

        # Re-label to consecutive integers
        unique_labels = np.unique(labels)
        remap = {old: new for new, old in enumerate(unique_labels)}
        labels = np.array([remap[l] for l in labels])

        return labels

    def _build_chart_meshes(
        self,
        mesh: trimesh.Trimesh,
        chart_faces: dict[int, np.ndarray],
    ) -> tuple[list[trimesh.Trimesh], list[np.ndarray]]:
        """Build sub-mesh for each chart. Returns (meshes, vertex_maps).

        vertex_maps[i] maps chart-internal vertex index -> original mesh vertex index.
        """
        vertices = np.array(mesh.vertices)
        faces = np.array(mesh.faces)
        charts = []
        vert_maps = []

        for chart_id in sorted(chart_faces.keys()):
            face_idx = chart_faces[chart_id]
            chart_triangles = faces[face_idx]

            # Get unique vertices in this chart
            unique_verts = np.unique(chart_triangles)
            vert_map = np.zeros(len(vertices), dtype=np.int64)
            vert_map[unique_verts] = np.arange(len(unique_verts))

            # Remap face indices
            remapped_faces = vert_map[chart_triangles]

            chart_mesh = trimesh.Trimesh(
                vertices=vertices[unique_verts],
                faces=remapped_faces,
                process=False,
            )
            charts.append(chart_mesh)
            vert_maps.append(unique_verts)

        return charts, vert_maps

    def _find_boundary_edges(
        self,
        mesh: trimesh.Trimesh,
        face_labels: np.ndarray,
    ) -> np.ndarray:
        """Find edges that lie between different charts."""
        if not hasattr(mesh, 'face_adjacency') or len(mesh.face_adjacency) == 0:
            return np.array([], dtype=np.int64).reshape(0, 2)

        try:
            adj = mesh.face_adjacency
            edges = mesh.face_adjacency_edges

            # Boundary = edges where adjacent faces have different labels
            label_0 = face_labels[adj[:, 0]]
            label_1 = face_labels[adj[:, 1]]
            boundary_mask = label_0 != label_1

            return edges[boundary_mask]
        except Exception:
            return np.array([], dtype=np.int64).reshape(0, 2)
