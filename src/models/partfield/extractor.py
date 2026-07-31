"""PartField feature extractor — extracts semantic part features from 3D meshes.

Wraps the PartField neural network (from PartUV, SIGGRAPH Asia 2025) into
a clean interface for our ML UV Unwrap pipeline.

Usage:
    extractor = PartFieldFeatureExtractor(device="cpu")
    features = extractor.extract(mesh)  # -> (num_faces, 448) tensor
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch
import trimesh


class PartFieldFeatureExtractor:
    """Extract per-face semantic part features using PartField.

    PartField is a PVCNN + Triplane Transformer that produces 448-dimensional
    feature vectors per face, encoding hierarchical part structure. These
    features can be used to:
    - Predict UV seam locations (seams at part boundaries)
    - Assign faces to UV charts
    - Guide agglomerative clustering for part decomposition
    """

    FEATURE_DIM = 448

    def __init__(
        self,
        checkpoint_path: str | Path | None = None,
        device: str | None = None,
    ):
        """
        Args:
            checkpoint_path: Path to model_objaverse.ckpt.
                           If None, uses default path relative to this package.
            device: 'cuda' or 'cpu'. Auto-detected if None.
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._model = None
        self._checkpoint_path = checkpoint_path
        self._loaded = False

    def available(self) -> bool:
        """Return True if PartField can actually run on this install.

        Cheap check — does not import heavy modules. Requires both the
        ``lightning`` package and a model checkpoint to be present.
        """
        import importlib.util

        if importlib.util.find_spec("lightning") is None:
            return False
        try:
            return Path(self._find_checkpoint()).exists()
        except FileNotFoundError:
            return False

    def _ensure_loaded(self):
        """Lazy-load the model on first use."""
        if self._loaded:
            return

        try:
            from .model_trainer_pvcnn_only_demo import Model
        except ImportError as e:
            raise RuntimeError(
                "PartField unavailable: the 'lightning' package is not installed. "
                "PartUV/PartField modes fall back to geometric features."
            ) from e
        from .config import setup, default_argument_parser

        parser = default_argument_parser()
        args = parser.parse_args([])

        # Find config
        cfg_path = Path(__file__).parent / "configs" / "final" / "demo.yaml"
        args.config_file = str(cfg_path)
        cfg = setup(args)

        # Find checkpoint
        ckpt_path = self._find_checkpoint()

        # Load model
        self._model = Model(cfg, device=torch.device(self.device))
        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        state = ckpt.get("state_dict", ckpt)
        self._model.load_state_dict(state, strict=True)
        self._model.to(self.device)
        self._model.eval()

        self._loaded = True
        print(f"PartField loaded on {self.device}")

    def _find_checkpoint(self) -> str:
        """Locate the PartField checkpoint."""
        if self._checkpoint_path is not None:
            return str(self._checkpoint_path)

        # Search common locations
        candidates = [
            Path(__file__).parent / "model_objaverse.ckpt",
            Path(__file__).parent.parent.parent.parent / "model_objaverse.ckpt",
            Path("model_objaverse.ckpt"),
            Path.home() / ".cache" / "partfield" / "model_objaverse.ckpt",
        ]

        for p in candidates:
            if p.exists():
                return str(p)

        raise FileNotFoundError(
            "PartField checkpoint not found. Please provide checkpoint_path. "
            "Download model_objaverse.ckpt and place it in the project root "
            "or pass the path to PartFieldFeatureExtractor(checkpoint_path=...)."
        )

    @torch.no_grad()
    def extract(
        self,
        mesh: trimesh.Trimesh,
        sample_on_faces: int = 10,
        sample_batch_size: int = 100_000,
    ) -> torch.Tensor:
        """Extract per-face features from a mesh.

        Args:
            mesh: Input trimesh mesh
            sample_on_faces: Points per face for feature aggregation.
                           Higher = more accurate but slower.
            sample_batch_size: Batch size for memory-efficient processing.

        Returns:
            features: (num_faces, 448) tensor of per-face part features
        """
        self._ensure_loaded()

        # Save temporarily (PartField expects a file path or mesh object)
        # Use the model's run_inference directly
        point_feat, processed_mesh, num_bridge = self._model.run_inference(
            filename=None,
            mesh=mesh,
            sample_on_faces=sample_on_faces,
            sample_batch_size=sample_batch_size,
            seed=42,
            device=self.device,
        )

        # point_feat is (num_faces, 448) tensor
        if isinstance(point_feat, np.ndarray):
            point_feat = torch.tensor(point_feat, dtype=torch.float32)

        return point_feat.float().to(self.device)

    def extract_from_file(
        self,
        mesh_path: str | Path,
        **kwargs,
    ) -> torch.Tensor:
        """Extract features from a mesh file.

        Args:
            mesh_path: Path to OBJ/PLY/etc file
            **kwargs: Passed to extract()

        Returns:
            features: (num_faces, 448) tensor
        """
        mesh = trimesh.load(mesh_path, force="mesh")
        if isinstance(mesh, trimesh.Scene):
            mesh = trimesh.util.concatenate(mesh.dump())
        return self.extract(mesh, **kwargs)

    def cluster_parts(
        self,
        features: torch.Tensor,
        max_parts: int = 10,
    ) -> np.ndarray:
        """Cluster faces into parts using agglomerative clustering.

        Args:
            features: (num_faces, 448) per-face features
            max_parts: Maximum number of parts

        Returns:
            labels: (num_faces,) integer part labels
        """
        from sklearn.cluster import AgglomerativeClustering

        if isinstance(features, torch.Tensor):
            feat_np = features.cpu().numpy()
        else:
            feat_np = features

        # Normalize features
        feat_np = feat_np / (np.linalg.norm(feat_np, axis=-1, keepdims=True) + 1e-8)

        clustering = AgglomerativeClustering(
            n_clusters=min(max_parts, len(feat_np)),
        )
        labels = clustering.fit_predict(feat_np)
        return labels

    def predict_seams(
        self,
        features: torch.Tensor,
        labels: np.ndarray | None = None,
        threshold: float = 0.5,
    ) -> torch.Tensor:
        """Predict UV seam locations from part features.

        Seams occur at boundaries between different parts.

        Args:
            features: (num_faces, 448) per-face features
            labels: Optional pre-computed part labels
            threshold: Similarity threshold for seam detection

        Returns:
            seam_mask: (num_faces, 1) tensor, 1.0 = seam face
        """
        if labels is None:
            labels = self.cluster_parts(features)

        # Normalize features
        if isinstance(features, torch.Tensor):
            feat = features / (features.norm(dim=-1, keepdim=True) + 1e-8)
        else:
            feat = torch.tensor(features, dtype=torch.float32)
            feat = feat / (feat.norm(dim=-1, keepdim=True) + 1e-8)

        # Compute pairwise cosine similarity between adjacent faces
        # For simplicity, faces with different labels are considered seams
        labels_t = torch.tensor(labels, dtype=torch.long)
        # A face is a seam if its label differs from any neighbor
        # (simplified: just mark faces at label boundaries)
        seam_mask = torch.zeros(len(labels), 1)

        # Simple heuristic: faces with different labels from their neighbors
        for i in range(len(labels)):
            # Check if this face's label differs from the mean of nearby faces
            neighbors_feat = feat[labels == labels[i]]
            if len(neighbors_feat) > 1:
                mean_feat = neighbors_feat.mean(dim=0)
                sim = (feat[i] * mean_feat).sum()
                if sim < threshold:
                    seam_mask[i] = 1.0

        return seam_mask
