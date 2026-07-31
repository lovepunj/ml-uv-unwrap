from __future__ import annotations

"""Full UV unwrapping pipeline — from mesh input to UV output.

Supports multiple modes:
- ml: FlexPara unsupervised optimization
- partuv: PartField-guided part-aware unwrapping
- classical: xatlas, LSCM, or ABF++ parameterization
- hybrid: ML refinement of classical initial UVs
- multi_chart: PartField-guided chart decomposition + per-chart unwrapping
- detect: Analyze mesh type and recommend strategy
- flatten_anything: Flatten Anything Model (FAM) global parameterization
- mesh_tailor: Graph-native seam generation via MeshTailor
- seam_crafter: DPO-trained seam prediction
- uv_segnet: Semantic boundary detection for man-made objects
- quality_select: Run multiple backends and pick best automatically
- artuv: ArtUV-style offset prediction (ICLR 2026)
- arap: As-Rigid-As-Possible parameterization
- harmonic: Harmonic map parameterization
- conformal: Curvature-weighted conformal mapping
- graph_cuts: Graph cuts seam selection + LSCM
- hilbert: Hilbert space-filling curve projection
- voronoi_disks: Voronoi decomposition into topological disks
- instant_meshes: Integer-grid based parameterization
- libuvula: Ultimaker libUvula wrapper
"""

from pathlib import Path

import numpy as np
import torch

from ..models import FlexParaUnwrapper
from .postprocessor import add_uv_margins, export_uv_mesh, pack_uv_charts
from .preprocessor import MeshPreprocessor


class UVUnwrapPipeline:
    """End-to-end UV unwrapping pipeline.

    Supports multiple modes:
    - ml: FlexPara unsupervised optimization
    - partuv: PartField-guided part-aware unwrapping
    - classical: xatlas, LSCM, or ABF++ parameterization
    - hybrid: ML refinement of classical initial UVs
    - multi_chart: PartField-guided chart decomposition + per-chart unwrapping
    - detect: Analyze mesh type and recommend strategy
    - flatten_anything: Flatten Anything Model (FAM) global parameterization
    - mesh_tailor: Graph-native seam generation via MeshTailor
    - seam_crafter: DPO-trained seam prediction
    - uv_segnet: Semantic boundary detection for man-made objects
    - quality_select: Run multiple backends and pick best automatically
    - artuv: ArtUV-style offset prediction (ICLR 2026)
    - arap: As-Rigid-As-Possible parameterization
    - harmonic: Harmonic map parameterization
    - conformal: Curvature-weighted conformal mapping
    - graph_cuts: Graph cuts seam selection + LSCM
    - hilbert: Hilbert space-filling curve projection
    - voronoi_disks: Voronoi decomposition into topological disks
    - instant_meshes: Integer-grid based parameterization
    - libuvula: Ultimaker libUvula wrapper
    - ffhq_uv: FFHQ-UV face UV unwrapping (topology transfer, multi-view, rgb fitting)

    Usage:
        pipeline = UVUnwrapPipeline()
        result = pipeline.unwrap("model.obj")
        pipeline.export(result, "model_unwrapped.obj")

        # Multi-chart mode (recommended for complex meshes)
        pipeline = UVUnwrapPipeline(mode="multi_chart")
        result = pipeline.unwrap("model.obj")

        # Detect mesh type and get recommendations
        pipeline = UVUnwrapPipeline()
        analysis = pipeline.detect("model.obj")
        print(analysis["project_type"])
        print(analysis["recommended_strategy"])
    """

    def __init__(
        self,
        model_path: str | Path | None = None,
        num_points: int = 3000,
        num_charts: int = 1,
        num_iterations: int = 500,
        device: str | None = None,
        use_partuv: bool = False,
        partfield_checkpoint: str | Path | None = None,
        mode: str = "ml",
        classical_method: str = "xatlas",
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.num_iterations = num_iterations
        self.num_charts = num_charts
        self.use_partuv = use_partuv
        self.mode = mode
        self.classical_method = classical_method

        # PartField feature extractor (lazy loaded)
        self._partfield_extractor = None
        if use_partuv or mode in ("partuv", "multi_chart", "detect"):
            self._init_partfield(partfield_checkpoint)

        # Classical unwrapper (lazy loaded)
        self._classical_unwrapper = None
        if mode in ("classical", "hybrid", "quality_select"):
            self._init_classical()

        # Chart decomposer (lazy loaded)
        self._chart_decomposer = None
        if mode == "multi_chart":
            self._init_chart_decomposer()

        # Multi-chart unwrapper (lazy loaded)
        self._multi_chart_unwrapper = None
        if mode == "multi_chart":
            from .multi_chart_unwrapper import MultiChartUnwrapper
            self._multi_chart_unwrapper = MultiChartUnwrapper()

        # Mesh type detector (lazy loaded)
        self._mesh_detector = None

        # Initialize ML model if needed
        self.model = None
        if mode in ("ml", "hybrid"):
            self.model = FlexParaUnwrapper(
                num_charts=num_charts,
                hidden_dim=256,
                num_layers=8,
                partfield_dim=0,
            ).to(self.device)

            if model_path is not None:
                self._load_model(model_path)

        # Flatten Anything Model (lazy loaded)
        self._fam_model = None
        if mode == "flatten_anything":
            from ..models.networks.flatten_anything import FlattenAnythingModel
            self._fam_model = FlattenAnythingModel(hidden_dim=256).to(self.device)

        # MeshTailor model (lazy loaded)
        self._mesh_tailor_model = None
        if mode == "mesh_tailor":
            from ..models.networks.mesh_tailor import MeshTailorModel
            self._mesh_tailor_model = MeshTailorModel().to(self.device)

        # SeamCrafter model (lazy loaded)
        self._seam_crafter_model = None
        if mode == "seam_crafter":
            from ..models.networks.seam_crafter import SeamCrafterModel
            self._seam_crafter_model = SeamCrafterModel().to(self.device)

        # UVSegNet pipeline (lazy loaded)
        self._uv_segnet = None
        if mode == "uv_segnet":
            from ..models.networks.uv_segnet import UVSegNetPipeline
            self._uv_segnet = UVSegNetPipeline()

        # Quality selector (lazy loaded)
        self._quality_selector = None
        if mode == "quality_select":
            from ..models.networks.quality_selector import QualitySelectorNet
            self._quality_selector = QualitySelectorNet().to(self.device)

        # ArtUV model (lazy loaded)
        self._artuv_model = None
        if mode == "artuv":
            from ..models.networks.artuv import ArtUVModel
            self._artuv_model = ArtUVModel(
                hidden_dim=128,
                num_graph_layers=5,
            ).to(self.device)

        # UV refinement module for hybrid mode
        self._refiner = None
        if mode == "hybrid":
            from ..models.networks import DistortionAwareRefiner
            self._refiner = DistortionAwareRefiner(hidden_dim=128).to(self.device)

        # Preprocessor
        self.preprocessor = MeshPreprocessor(
            num_points=num_points,
            device=self.device,
        )

    def _init_partfield(self, checkpoint_path: str | Path | None):
        """Initialize PartField feature extractor.

        Degrades to ``None`` (geometric features) when PartField cannot
        actually run here — e.g. ``lightning`` is not installed or the
        checkpoint is missing.
        """
        from ..models.partfield.extractor import PartFieldFeatureExtractor
        extractor = PartFieldFeatureExtractor(
            checkpoint_path=checkpoint_path,
            device=self.device,
        )
        if extractor.available():
            self._partfield_extractor = extractor
        else:
            print(
                "PartField unavailable (lightning/checkpoint missing); "
                "falling back to geometric features."
            )
            self._partfield_extractor = None

    def _init_classical(self):
        """Initialize classical unwrapper."""
        from .classical_unwrapper import ClassicalUnwrapper
        self._classical_unwrapper = ClassicalUnwrapper(method=self.classical_method)

    def _init_chart_decomposer(self):
        """Initialize PartField chart decomposer."""
        from .chart_decomposer import PartFieldChartDecomposer
        self._chart_decomposer = PartFieldChartDecomposer(
            partfield_extractor=self._partfield_extractor,
            device=self.device,
        )

    def _init_mesh_detector(self):
        """Initialize mesh type detector."""
        from ..models.mesh_classifier import MeshTypeDetector
        self._mesh_detector = MeshTypeDetector(
            partfield_extractor=self._partfield_extractor,
        )

    def detect(self, mesh_input: str | Path) -> dict:
        """Analyze mesh type and recommend unwrapping strategy.

        Args:
            mesh_input: file path or trimesh mesh

        Returns:
            Dictionary with project_type, recommended_strategy, etc.
        """
        import trimesh

        # Load mesh if needed
        if isinstance(mesh_input, (str, Path)):
            mesh = trimesh.load(mesh_input, force="mesh")
            if isinstance(mesh, trimesh.Scene):
                mesh = trimesh.util.concatenate(mesh.dump())
        else:
            mesh = mesh_input

        # Initialize detector if needed
        if self._mesh_detector is None:
            self._init_mesh_detector()

        # Run detection
        result = self._mesh_detector.detect(
            mesh,
            use_partfield=self._partfield_extractor is not None,
        )

        return {
            "project_type": result.project_type.value,
            "confidence": result.confidence,
            "recommended_strategy": result.recommended_strategy,
            "recommended_charts": result.recommended_charts,
            "recommended_method": result.recommended_method,
            "complexity_score": result.complexity_score,
            "part_count_estimate": result.part_count_estimate,
            "analysis_notes": result.analysis_notes,
            "features": {
                "num_vertices": result.features.num_vertices,
                "num_faces": result.features.num_faces,
                "surface_area": result.features.surface_area,
                "compactness": result.features.compactness,
                "convexity": result.features.convexity,
                "elongation": result.features.elongation,
                "symmetry_score": result.features.symmetry_score,
                "genus": result.features.genus,
            },
        }

    def unwrap(
        self,
        mesh_input: str | Path,
        num_iterations: int | None = None,
        log_every: int = 100,
        progress_callback=None,
    ) -> dict:
        """Unwrap a single mesh.

        Args:
            mesh_input: file path or trimesh mesh
            num_iterations: override default optimization steps
            log_every: print loss every N steps
            progress_callback: optional callback(step, total_steps, losses_dict)

        Returns:
            Dictionary with results including UV coordinates
        """
        import trimesh

        # Load mesh if needed
        if isinstance(mesh_input, (str, Path)):
            mesh = trimesh.load(mesh_input, force="mesh")
            if isinstance(mesh, trimesh.Scene):
                mesh = trimesh.util.concatenate(mesh.dump())
        else:
            mesh = mesh_input

        print(f"Unwrapping mesh: {mesh_input}")
        print(f"  Mode: {self.mode}")

        # Classical mode (all classical methods: xatlas, lscm, abf, arap, harmonic, conformal, graph_cuts, hilbert)
        if self.mode == "classical":
            return self._unwrap_classical(mesh)

        # Individual classical method shortcuts
        if self.mode in ("arap", "harmonic", "conformal", "graph_cuts", "hilbert"):
            return self._unwrap_classical_method(mesh, self.mode)

        # Research methods (voronoi_disks, instant_meshes, libuvula)
        if self.mode in ("voronoi_disks", "instant_meshes", "libuvula"):
            return self._unwrap_research(mesh)

        # Hybrid mode: classical first, then ML refinement
        if self.mode == "hybrid":
            return self._unwrap_hybrid(mesh, num_iterations, log_every, progress_callback)

        # Multi-chart mode: PartField decomposition + per-chart unwrapping
        if self.mode == "multi_chart":
            return self._unwrap_multi_chart(mesh)

        # Detect mode: analyze and recommend
        if self.mode == "detect":
            analysis = self.detect(mesh)
            return self._auto_unwrap(mesh, analysis, num_iterations, log_every, progress_callback)

        # PartUV mode: PartField-guided chart decomposition + per-chart unwrapping
        if self.mode == "partuv":
            return self._unwrap_partuv(mesh, num_iterations, log_every, progress_callback)

        # Flatten Anything mode: global free-boundary parameterization
        if self.mode == "flatten_anything":
            return self._unwrap_flatten_anything(mesh, num_iterations, log_every, progress_callback)

        # MeshTailor mode: graph-native seam generation
        if self.mode == "mesh_tailor":
            return self._unwrap_mesh_tailor(mesh)

        # SeamCrafter mode: DPO-trained seam prediction
        if self.mode == "seam_crafter":
            return self._unwrap_seam_crafter(mesh, progress_callback)

        # UVSegNet mode: semantic boundary detection for man-made objects
        if self.mode == "uv_segnet":
            return self._unwrap_uv_segnet(mesh)

        # Quality select mode: run multiple backends and pick best
        if self.mode == "quality_select":
            return self._unwrap_quality_select(mesh)

        # ArtUV mode: offset prediction from coarse initial UV
        if self.mode == "artuv":
            return self._unwrap_artuv(mesh, num_iterations, log_every, progress_callback)

        # FFHQ-UV mode: face-specific UV unwrapping (topology transfer, multi-view, rgb fitting)
        if self.mode == "ffhq_uv":
            return self._unwrap_ffhq_uv(mesh)

    def _unwrap_classical(self, mesh: trimesh.Trimesh) -> dict:
        """Unwrap using classical methods (xatlas/LSCM/ABF)."""
        vertices = np.array(mesh.vertices, dtype=np.float32)
        faces = np.array(mesh.faces, dtype=np.int32)

        result = self._classical_unwrapper.unwrap(mesh)
        result["history"] = {}
        result["edges"] = np.array(mesh.edges_unique, dtype=np.int32) if hasattr(mesh, 'edges_unique') else np.array([], dtype=np.int32)
        result["mode"] = "classical"

        return result

    def _unwrap_classical_method(self, mesh: trimesh.Trimesh, method: str) -> dict:
        """Unwrap using a specific classical method (arap/harmonic/conformal/graph_cuts/hilbert)."""
        from .classical_unwrapper import ClassicalUnwrapper
        unwrapper = ClassicalUnwrapper(method=method)
        result = unwrapper.unwrap(mesh)
        result["history"] = {}
        result["edges"] = np.array(mesh.edges_unique, dtype=np.int32) if hasattr(mesh, 'edges_unique') else np.array([], dtype=np.int32)
        result["mode"] = method
        return result

    def _unwrap_research(self, mesh: trimesh.Trimesh) -> dict:
        """Unwrap using research methods (voronoi_disks/instant_meshes/libuvula)."""
        from .research_unwrappers import ResearchUnwrapper
        unwrapper = ResearchUnwrapper(method=self.mode)
        result = unwrapper.unwrap(mesh)
        result["history"] = {}
        result["edges"] = np.array(mesh.edges_unique, dtype=np.int32) if hasattr(mesh, 'edges_unique') else np.array([], dtype=np.int32)
        result["mode"] = self.mode
        return result

    def _unwrap_ffhq_uv(self, mesh: trimesh.Trimesh) -> dict:
        """FFHQ-UV inspired face UV unwrapping."""
        from .ffhq_uv_unwrapper import FFHQUVUnwrapper
        unwrapper = FFHQUVUnwrapper(method="face_auto")
        result = unwrapper.unwrap(mesh)
        result["history"] = {}
        result["edges"] = np.array(mesh.edges_unique, dtype=np.int32) if hasattr(mesh, 'edges_unique') else np.array([], dtype=np.int32)
        result["mode"] = "ffhq_uv"
        return result

    def _unwrap_multi_chart(self, mesh: trimesh.Trimesh) -> dict:
        """Multi-chart unwrapping using PartField decomposition."""
        print("  Decomposing mesh into charts...")

        # Initialize decomposer if needed
        if self._chart_decomposer is None:
            self._init_chart_decomposer()

        # Decompose mesh into charts
        decomposition = self._chart_decomposer.decompose(mesh)
        print(f"  Found {decomposition.chart_count} charts")

        # Unwrap each chart
        print("  Unwrapping charts...")
        multi_result = self._multi_chart_unwrapper.unwrap(decomposition, mesh)

        return {
            "uv_coords": multi_result.uv_coords,
            "vertices": multi_result.vertices.astype(np.float32),
            "faces": multi_result.faces,
            "edges": np.array(mesh.edges_unique, dtype=np.int32) if hasattr(mesh, 'edges_unique') else np.array([], dtype=np.int32),
            "history": {"multi_chart": True, "num_charts": multi_result.num_charts},
            "mesh": mesh,
            "mode": "multi_chart",
            "chart_labels": multi_result.chart_labels,
            "num_charts": multi_result.num_charts,
            "per_chart_distortion": multi_result.per_chart_distortion,
            "total_distortion": multi_result.total_distortion,
        }

    def _unwrap_partuv(
        self,
        mesh: trimesh.Trimesh,
        num_iterations: int | None = None,
        log_every: int = 100,
        progress_callback=None,
    ) -> dict:
        """PartUV: PartField-guided chart decomposition + per-chart unwrapping."""
        import copy
        import trimesh as trimesh_mod
        import xatlas as xatlas_mod

        print("  PartUV: extracting PartField features...")
        mesh_copy = copy.deepcopy(mesh)
        with torch.no_grad():
            face_features = self._partfield_extractor.extract(mesh_copy, sample_on_faces=10)
        print(f"  PartField features: {face_features.shape}")

        print("  Clustering faces into semantic parts...")
        num_charts = self.num_charts
        if num_charts <= 1:
            num_charts = max(2, min(10, len(mesh.faces) // 200))

        part_labels = self._partfield_extractor.cluster_parts(face_features, max_parts=num_charts)
        num_parts = len(set(part_labels))
        print(f"  Found {num_parts} semantic parts")

        vertices = np.array(mesh.vertices, dtype=np.float32)
        faces = np.array(mesh.faces, dtype=np.int32)
        uv_result = np.zeros((len(vertices), 2), dtype=np.float32)
        chart_labels = np.zeros(len(faces), dtype=np.int32)
        per_chart_distortion = []

        for part_id in range(num_parts):
            mask = part_labels == part_id
            if mask.sum() == 0:
                continue

            chart_labels[mask] = part_id
            chart_faces = faces[mask]
            unique_verts = np.unique(chart_faces)
            vert_map = np.full(len(vertices), -1, dtype=np.int32)
            vert_map[unique_verts] = np.arange(len(unique_verts))
            remapped = vert_map[chart_faces]

            submesh = trimesh_mod.Trimesh(
                vertices=vertices[unique_verts],
                faces=remapped,
            )
            if len(submesh.faces) == 0:
                continue

            print(f"    Part {part_id}: {len(submesh.faces)} faces, {len(submesh.vertices)} vertices")

            try:
                atlas = xatlas_mod.Atlas()
                atlas.add_mesh(submesh.vertices, submesh.faces)
                atlas.generate()
                unique_ids, face_ids, chart_uv = atlas.get_mesh(0)

                uv_min = chart_uv.min(axis=0)
                uv_max = chart_uv.max(axis=0)
                chart_uv = (chart_uv - uv_min) / (uv_max - uv_min + 1e-10)

                # unique_ids[i] = original submesh vertex for output vertex i
                # chart_uv[i] = UV for output vertex i
                # So: sub_vert_uvs[unique_ids[i]] = chart_uv[i]
                sub_vert_uvs = np.zeros((len(submesh.vertices), 2), dtype=np.float32)
                for i in range(len(unique_ids)):
                    sub_vert_uvs[unique_ids[i]] = chart_uv[i]

                for j, vi in enumerate(unique_verts):
                    uv_result[vi] = sub_vert_uvs[j]

                distortion = 0.0
                try:
                    e1_3d = np.linalg.norm(submesh.vertices[remapped[:, 1]] - submesh.vertices[remapped[:, 0]], axis=1) + 1e-10
                    e2_3d = np.linalg.norm(submesh.vertices[remapped[:, 2]] - submesh.vertices[remapped[:, 0]], axis=1) + 1e-10
                    v_uv_0 = sub_vert_uvs[remapped[:, 0]]
                    v_uv_1 = sub_vert_uvs[remapped[:, 1]]
                    v_uv_2 = sub_vert_uvs[remapped[:, 2]]
                    e1_uv = np.linalg.norm(v_uv_1 - v_uv_0, axis=1) + 1e-10
                    e2_uv = np.linalg.norm(v_uv_2 - v_uv_0, axis=1) + 1e-10
                    s1, s2 = e1_uv / e1_3d, e2_uv / e2_3d
                    distortion = float(np.mean((s1 - s2) ** 2))
                except Exception:
                    pass

                per_chart_distortion.append(distortion)
                print(f"      distortion: {distortion:.4f}")
            except Exception as e:
                print(f"      Failed: {e}, falling back to PCA projection")
                chart_verts = vertices[unique_verts]
                centered = chart_verts - chart_verts.mean(axis=0)
                cov = centered.T @ centered
                eigvals, eigvecs = np.linalg.eigh(cov)
                uv_proj = centered @ eigvecs[:, -2:]
                uv_proj -= uv_proj.min(axis=0)
                uv_proj /= uv_proj.max(axis=0) + 1e-8
                for j, vi in enumerate(unique_verts):
                    uv_result[vi] = uv_proj[j]

        avg_distortion = np.mean(per_chart_distortion) if per_chart_distortion else 0.0
        print(f"  Total: avg distortion: {avg_distortion:.4f}")

        return {
            "uv_coords": uv_result,
            "vertices": vertices,
            "faces": faces,
            "edges": np.array(mesh.edges_unique, dtype=np.int32) if hasattr(mesh, 'edges_unique') else np.array([], dtype=np.int32),
            "history": {"partuv": True, "num_parts": num_parts},
            "mesh": mesh,
            "mode": "partuv",
            "chart_labels": chart_labels,
            "num_charts": num_parts,
            "per_chart_distortion": per_chart_distortion,
            "total_distortion": avg_distortion,
        }

    def _unwrap_flatten_anything(
        self,
        mesh: trimesh.Trimesh,
        num_iterations: int | None = None,
        log_every: int = 100,
        progress_callback=None,
    ) -> dict:
        """Flatten Anything Model (FAM) — global free-boundary parameterization."""
        vertices = np.array(mesh.vertices, dtype=np.float32)
        faces = np.array(mesh.faces, dtype=np.int32)
        num_verts = len(vertices)

        from ..models.networks.flatten_anything import FlattenAnythingModel

        if self._fam_model is None:
            self._fam_model = FlattenAnythingModel(hidden_dim=256).to(self.device)

        # Prepare tensors — FAM operates on sampled points (B, N, 3)
        self.preprocessor = MeshPreprocessor(num_points=num_verts, device=self.device)
        data = self.preprocessor.process(mesh)
        points = data["points"]  # (1, N, 3)

        self._fam_model.train()
        optimizer = torch.optim.Adam(self._fam_model.parameters(), lr=1e-3)
        num_steps = num_iterations or 500

        print(f"  FAM: {num_steps} steps on {points.shape[1]} points...")

        for step in range(1, num_steps + 1):
            optimizer.zero_grad()
            outputs = self._fam_model(points)
            losses = self._fam_model.compute_losses(points, outputs)
            loss = losses["total"]
            loss.backward()
            optimizer.step()

            if progress_callback:
                progress_callback(step, num_steps, {k: v.item() for k, v in losses.items() if isinstance(v, torch.Tensor)})

            if step % 50 == 0:
                loss_dict = {k: v.item() for k, v in losses.items() if isinstance(v, torch.Tensor)}
                print(f"  FAM step {step}/{num_steps}: loss={loss.item():.4f} metrics={loss_dict}")

        # Extract final UVs
        self._fam_model.eval()
        with torch.no_grad():
            outputs = self._fam_model(points)
            uv_coords = outputs["uv_forward"][0].cpu().numpy()

        print(f"  FAM: final UVs shape {uv_coords.shape}, range [{uv_coords.min():.3f}, {uv_coords.max():.3f}]")

        return {
            "uv_coords": uv_coords,
            "vertices": vertices,
            "faces": faces,
            "edges": np.array(mesh.edges_unique, dtype=np.int32) if hasattr(mesh, 'edges_unique') else np.array([], dtype=np.int32),
            "history": {"fam_steps": num_steps},
            "mesh": mesh,
            "mode": "flatten_anything",
        }

    def _unwrap_mesh_tailor(self, mesh: trimesh.Trimesh) -> dict:
        """MeshTailor — graph-native seam generation + per-chart unwrapping."""
        from ..models.networks.mesh_tailor import MeshTailorModel, SeamTokenizer
        from ..pipeline.chart_decomposer import PartFieldChartDecomposer
        from ..pipeline.multi_chart_unwrapper import MultiChartUnwrapper

        vertices = np.array(mesh.vertices, dtype=np.float32)
        faces = np.array(mesh.faces, dtype=np.int32)

        if self._mesh_tailor_model is None:
            self._mesh_tailor_model = MeshTailorModel().to(self.device)

        # Step 1: MeshTailor generates seam proposals (heuristic + learned scoring)
        print("  MeshTailor: generating seams...")
        self._mesh_tailor_model.eval()
        verts_t = torch.tensor(vertices, dtype=torch.float32, device=self.device)
        faces_t = torch.tensor(faces, dtype=torch.long, device=self.device)

        with torch.no_grad():
            seam_sets = self._mesh_tailor_model.generate_seams(verts_t, faces_t, beam_width=8)

        # seam_sets is list[set[tuple[int, int]]] — pick best beam
        best_seams = seam_sets[0] if seam_sets else set()
        num_seam_edges = len(best_seams)
        print(f"  MeshTailor: {num_seam_edges} seam edges proposed")

        # Step 2: Use PartField chart decomposition for robustness
        if self._chart_decomposer is None:
            self._init_chart_decomposer()

        decomposition = self._chart_decomposer.decompose(mesh)
        print(f"  PartField decomposition: {decomposition.chart_count} charts")

        # Step 3: If MeshTailor found many seams, suggest more charts
        if num_seam_edges > len(mesh.edges_unique) * 0.1 and decomposition.chart_count < 4:
            target_charts = min(decomposition.chart_count * 2, 8)
            print(f"  MeshTailor suggests more charts: {decomposition.chart_count} → {target_charts}")
            decomposition = self._chart_decomposer.decompose(mesh, num_charts=target_charts)

        # Step 4: Unwrap each chart
        if self._multi_chart_unwrapper is None:
            self._multi_chart_unwrapper = MultiChartUnwrapper()

        multi_result = self._multi_chart_unwrapper.unwrap(decomposition, mesh)

        print(f"  MeshTailor pipeline complete: {multi_result.num_charts} charts")

        return {
            "uv_coords": multi_result.uv_coords,
            "vertices": multi_result.vertices,
            "faces": multi_result.faces,
            "edges": np.array(mesh.edges_unique, dtype=np.int32) if hasattr(mesh, 'edges_unique') else np.array([], dtype=np.int32),
            "history": {
                "seam_edges": num_seam_edges,
                "total_edges": len(mesh.edges_unique) if hasattr(mesh, 'edges_unique') else 0,
            },
            "mesh": mesh,
            "mode": "mesh_tailor",
            "num_charts": multi_result.num_charts,
        }

    def _unwrap_seam_crafter(self, mesh: trimesh.Trimesh, progress_callback=None) -> dict:
        """SeamCrafter — DPO-trained seam prediction + PartField chart decomposition."""
        from ..models.networks.seam_crafter import SeamCrafterModel, SeamEvaluator
        from ..pipeline.chart_decomposer import PartFieldChartDecomposer
        from ..pipeline.multi_chart_unwrapper import MultiChartUnwrapper

        vertices = np.array(mesh.vertices, dtype=np.float32)
        faces = np.array(mesh.faces, dtype=np.int32)

        if self._seam_crafter_model is None:
            self._seam_crafter_model = SeamCrafterModel().to(self.device)

        if progress_callback:
            progress_callback(1, 100, {"stage": "seam_prediction"})

        # Step 1: SeamCrafter predicts seam coordinates
        print("  SeamCrafter: predicting seams...")
        self._seam_crafter_model.eval()
        verts_t = torch.tensor(vertices, dtype=torch.float32, device=self.device)

        # Sample/pad to 1024 points (CPU-friendly)
        target_n = 1024
        n = verts_t.shape[0]
        if n >= target_n:
            idx = torch.randperm(n)[:target_n]
            sampled = verts_t[idx].unsqueeze(0)
        else:
            repeats = (target_n // n) + 1
            sampled = verts_t.repeat(repeats, 1)[:target_n].unsqueeze(0)

        with torch.no_grad():
            seam_coords = self._seam_crafter_model.generate_seams(
                sampled, sampled, max_segments=20, temperature=1.2,
            )

        print(f"  SeamCrafter: generated {seam_coords.shape[1]} seam segments")

        if progress_callback:
            progress_callback(5, 100, {"stage": "partfield_decomposition"})

        # Step 2: Use PartField decomposition for actual chart assignment
        if self._chart_decomposer is None:
            self._init_chart_decomposer()

        def _decomp_progress(step, total, losses=None):
            if progress_callback:
                progress_callback(5 + int(step / total * 75), 100, losses or {})

        decomposition = self._chart_decomposer.decompose(mesh, progress_callback=_decomp_progress)
        print(f"  PartField decomposition: {decomposition.chart_count} charts")

        if progress_callback:
            progress_callback(82, 100, {"stage": "multi_chart_unwrap"})

        # Step 3: Unwrap each chart
        if self._multi_chart_unwrapper is None:
            self._multi_chart_unwrapper = MultiChartUnwrapper()

        multi_result = self._multi_chart_unwrapper.unwrap(decomposition, mesh)

        if progress_callback:
            progress_callback(90, 100, {"stage": "evaluation"})

        # Step 4: Evaluate quality
        evaluator = SeamEvaluator()
        faces_t = torch.tensor(faces, dtype=torch.long, device=self.device)
        distortion = evaluator.compute_distortion(seam_coords, verts_t, faces_t)
        frag = evaluator.compute_fragmentation(seam_coords, faces_t)
        print(f"  SeamCrafter quality: distortion={distortion.item():.4f}, fragmentation={frag.item():.4f}")

        return {
            "uv_coords": multi_result.uv_coords,
            "vertices": multi_result.vertices,
            "faces": multi_result.faces,
            "edges": np.array(mesh.edges_unique, dtype=np.int32) if hasattr(mesh, 'edges_unique') else np.array([], dtype=np.int32),
            "history": {
                "seam_segments": seam_coords.shape[1],
                "distortion": distortion.item(),
                "fragmentation": frag.item(),
            },
            "mesh": mesh,
            "mode": "seam_crafter",
            "num_charts": multi_result.num_charts,
        }

    def _unwrap_uv_segnet(self, mesh: trimesh.Trimesh) -> dict:
        """UVSegNet — semantic boundary detection for man-made objects."""
        from ..models.networks.uv_segnet import UVSegNetPipeline

        vertices = np.array(mesh.vertices, dtype=np.float32)
        faces = np.array(mesh.faces, dtype=np.int32)

        if self._uv_segnet is None:
            self._uv_segnet = UVSegNetPipeline()

        self._uv_segnet.eval()
        points_t = torch.tensor(vertices, dtype=torch.float32, device=self.device).unsqueeze(0)

        print("  UVSegNet: detecting semantic boundaries...")
        with torch.no_grad():
            out = self._uv_segnet(points_t)

        uv_coords = out["uv"][0].cpu().numpy()  # (N, 2)
        chart_labels = out["chart_labels"][0].cpu().numpy()  # (N,)
        num_charts = len(np.unique(chart_labels))

        print(f"  UVSegNet: {num_charts} charts detected")

        return {
            "uv_coords": uv_coords,
            "vertices": vertices,
            "faces": faces,
            "edges": np.array(mesh.edges_unique, dtype=np.int32) if hasattr(mesh, 'edges_unique') else np.array([], dtype=np.int32),
            "history": {"uv_segnet_charts": num_charts},
            "mesh": mesh,
            "mode": "uv_segnet",
            "num_charts": num_charts,
        }

    def _unwrap_quality_select(self, mesh: trimesh.Trimesh) -> dict:
        """Quality select — run multiple backends and pick best via QualitySelectorNet."""
        from ..models.networks.quality_selector import select_best_unwrap
        from ..pipeline.classical_unwrapper import ClassicalUnwrapper

        candidates = []

        # Candidate 1: xatlas
        try:
            print("  Quality select: running xatlas...")
            classical = ClassicalUnwrapper(method="xatlas")
            xatlas_result = classical.unwrap(mesh)
            candidates.append(xatlas_result)
        except Exception as e:
            print(f"  Quality select: xatlas failed: {e}")

        # Candidate 2: LSCM
        try:
            print("  Quality select: running LSCM...")
            classical_lscm = ClassicalUnwrapper(method="lscm")
            lscm_result = classical_lscm.unwrap(mesh)
            candidates.append(lscm_result)
        except Exception as e:
            print(f"  Quality select: LSCM failed: {e}")

        # Candidate 3: ABF++
        try:
            print("  Quality select: running ABF++...")
            classical_abf = ClassicalUnwrapper(method="abf")
            abf_result = classical_abf.unwrap(mesh)
            candidates.append(abf_result)
        except Exception as e:
            print(f"  Quality select: ABF++ failed: {e}")

        # Candidate 4: ARAP
        try:
            print("  Quality select: running ARAP...")
            classical_arap = ClassicalUnwrapper(method="arap")
            arap_result = classical_arap.unwrap(mesh)
            candidates.append(arap_result)
        except Exception as e:
            print(f"  Quality select: ARAP failed: {e}")

        # Candidate 5: harmonic
        try:
            print("  Quality select: running harmonic...")
            classical_harmonic = ClassicalUnwrapper(method="harmonic")
            harmonic_result = classical_harmonic.unwrap(mesh)
            candidates.append(harmonic_result)
        except Exception as e:
            print(f"  Quality select: harmonic failed: {e}")

        # Candidate 6: conformal
        try:
            print("  Quality select: running conformal...")
            classical_conformal = ClassicalUnwrapper(method="conformal")
            conformal_result = classical_conformal.unwrap(mesh)
            candidates.append(conformal_result)
        except Exception as e:
            print(f"  Quality select: conformal failed: {e}")

        # Candidate 7: multi-chart (if chart decomposer available)
        try:
            if self._chart_decomposer is None:
                self._init_chart_decomposer()
            if self._multi_chart_unwrapper is None:
                from .multi_chart_unwrapper import MultiChartUnwrapper
                self._multi_chart_unwrapper = MultiChartUnwrapper()

            decomposition = self._chart_decomposer.decompose(mesh)
            multi_result = self._multi_chart_unwrapper.unwrap(decomposition, mesh)
            # Convert MultiChartResult dataclass to dict with "uv" key
            multi_dict = {
                "uv": multi_result.uv_coords,
                "uv_coords": multi_result.uv_coords,
                "vertices": multi_result.vertices,
                "faces": multi_result.faces,
                "num_charts": multi_result.num_charts,
            }
            candidates.append(multi_dict)
            print(f"  Quality select: multi-chart ({multi_result.num_charts} charts)")
        except Exception as e:
            print(f"  Quality select: multi-chart failed: {e}")

        # Format candidates for select_best_unwrap (needs "uv" key)
        formatted_candidates = []
        for c in candidates:
            if isinstance(c, dict):
                fc = dict(c)
                if "uv" not in fc:
                    fc["uv"] = fc["uv_coords"]
                formatted_candidates.append(fc)

        if not formatted_candidates:
            print("  Quality select: all candidates failed, falling back to xatlas...")
            from ..pipeline.classical_unwrapper import ClassicalUnwrapper
            fallback = ClassicalUnwrapper(method="xatlas").unwrap(mesh)
            fallback["mode"] = "quality_select"
            fallback["history"] = {"num_candidates": 0, "fallback": "xatlas"}
            return fallback

        # Select best
        print(f"  Quality select: evaluating {len(formatted_candidates)} candidates...")
        sel = select_best_unwrap(formatted_candidates, mesh)
        best_idx = sel["index"]
        best = formatted_candidates[best_idx]

        print(f"  Quality select: candidate {best_idx} selected, score={sel['score']:.4f}")

        return {
            "uv_coords": best["uv_coords"],
            "vertices": best["vertices"],
            "faces": best["faces"],
            "edges": best.get("edges", np.array([], dtype=np.int32)),
            "history": {
                "num_candidates": len(formatted_candidates),
                "selected_idx": best_idx,
                "quality_score": sel["score"],
                "candidate_modes": [c.get("mode", "classical") for c in formatted_candidates],
            },
            "mesh": mesh,
            "mode": "quality_select",
            "num_charts": best.get("num_charts", 1),
        }

    def _unwrap_artuv(
        self,
        mesh: trimesh.Trimesh,
        num_iterations: int | None = None,
        log_every: int = 100,
        progress_callback=None,
    ) -> dict:
        """ArtUV: offset prediction from coarse initial UV.

        Uses GaussianWrapping-inspired normal field for seam guidance,
        then ArtUV-style residual offset learning for UV refinement.
        """
        from ..models.networks.artuv import ArtUVModel
        from ..losses.artuv import artuv_total_loss
        from ..losses.gaussian_normal import compute_vertex_curvature, compute_seam_candidates
        from ..data.mesh_io import interpolate_uv_barycentric

        vertices = np.array(mesh.vertices, dtype=np.float32)
        faces = np.array(mesh.faces, dtype=np.int32)
        V = len(vertices)

        # Step 1: Compute normal-aware seam candidates (GaussianWrapping-inspired)
        print("  ArtUV: computing normal-aware seam scores...")
        curvature = compute_vertex_curvature(vertices, faces)
        seam_candidates = compute_seam_candidates(vertices, faces, top_k=30)
        print(f"  ArtUV: {len(seam_candidates)} seam candidates from normal field")

        # Step 2: Get initial coarse UV from xatlas
        print("  ArtUV: generating initial coarse UV via xatlas...")
        import xatlas as xatlas_mod
        atlas = xatlas_mod.Atlas()
        atlas.add_mesh(vertices, faces)
        atlas.generate()
        unique_ids, face_ids, initial_uv = atlas.get_mesh(0)

        # Remap to original vertices
        uv_full = np.zeros((V, 2), dtype=np.float32)
        for i in range(len(unique_ids)):
            uv_full[unique_ids[i]] = initial_uv[i]

        # Normalize initial UV to [0, 1]
        uv_min = uv_full.min(axis=0)
        uv_max = uv_full.max(axis=0)
        uv_full = (uv_full - uv_min) / (uv_max - uv_min + 1e-10)

        # Step 3: Build edge index for graph convolutions
        edges_unique = np.array(mesh.edges_unique, dtype=np.int64)
        edge_index = torch.tensor(
            np.concatenate([edges_unique, edges_unique[:, ::-1]], axis=0).T,
            dtype=torch.long,
        )

        # Step 4: Optimize ArtUV offset prediction
        if self._artuv_model is None:
            from ..models.networks.artuv import ArtUVModel
            self._artuv_model = ArtUVModel(hidden_dim=128, num_graph_layers=5).to(self.device)

        vertices_t = torch.tensor(vertices, dtype=torch.float32, device=self.device)
        faces_t = torch.tensor(faces, dtype=torch.long, device=self.device)
        uv_init_t = torch.tensor(uv_full, dtype=torch.float32, device=self.device)
        edge_index = edge_index.to(self.device)

        self._artuv_model.train()
        optimizer = torch.optim.Adam(self._artuv_model.parameters(), lr=1e-3)
        iters = num_iterations or 500

        print(f"  ArtUV: optimizing {iters} steps...")
        for step in range(1, iters + 1):
            optimizer.zero_grad()

            outputs = self._artuv_model(uv_init_t, vertices_t, faces_t, edge_index)
            uv_pred = outputs["uv_pred"]

            losses = artuv_total_loss(vertices_t, uv_pred, uv_init_t, faces_t)
            losses["total"].backward()
            optimizer.step()

            if progress_callback and step % max(1, iters // 20) == 0:
                progress_callback(step, iters, {k: v.item() for k, v in losses.items() if isinstance(v, torch.Tensor)})

            if step % 50 == 0:
                loss_dict = {k: v.item() for k, v in losses.items() if isinstance(v, torch.Tensor)}
                print(f"    step {step}/{iters}: {loss_dict}")

        # Extract final UVs
        self._artuv_model.eval()
        with torch.no_grad():
            outputs = self._artuv_model(uv_init_t, vertices_t, faces_t, edge_index)
            uv_final = outputs["uv_pred"].cpu().numpy()

        print(f"  ArtUV: final UVs range [{uv_final.min():.3f}, {uv_final.max():.3f}]")

        return {
            "uv_coords": uv_final,
            "vertices": vertices,
            "faces": faces,
            "edges": np.array(mesh.edges_unique, dtype=np.int32) if hasattr(mesh, 'edges_unique') else np.array([], dtype=np.int32),
            "history": {"artuv_steps": iters},
            "mesh": mesh,
            "mode": "artuv",
            "seam_candidates": len(seam_candidates),
            "curvature_range": [float(curvature.min()), float(curvature.max())],
        }

    def _auto_unwrap(
        self,
        mesh: trimesh.Trimesh,
        analysis: dict,
        num_iterations: int | None = None,
        log_every: int = 100,
        progress_callback=None,
    ) -> dict:
        """Automatically select best unwrapping strategy based on analysis."""
        strategy = analysis["recommended_strategy"]
        charts = analysis["recommended_charts"]
        complexity = analysis["complexity_score"]

        print(f"  Auto-selected strategy: {strategy} ({charts} charts)")
        print(f"  Complexity score: {complexity:.2f}")

        if strategy == "single_chart" and charts <= 1:
            # Simple mesh: use classical xatlas
            print("  Using single-chart xatlas...")
            if self._classical_unwrapper is None:
                self._init_classical()
            return self._unwrap_classical(mesh)
        else:
            # Complex mesh: use multi-chart
            print("  Using multi-chart decomposition...")
            if self._chart_decomposer is None:
                self._init_chart_decomposer()
            if self._multi_chart_unwrapper is None:
                from .multi_chart_unwrapper import MultiChartUnwrapper
                self._multi_chart_unwrapper = MultiChartUnwrapper()
            return self._unwrap_multi_chart(mesh)

    def _unwrap_hybrid(
        self,
        mesh: trimesh.Trimesh,
        num_iterations: int | None = None,
        log_every: int = 100,
        progress_callback=None,
    ) -> dict:
        """Hybrid: classical UV init + ML refinement."""
        # Step 1: Classical unwrap
        print("  Step 1: Classical unwrap...")
        classical_result = self._classical_unwrapper.unwrap(mesh)

        # Preprocess for ML
        data = self.preprocessor.process(mesh)
        points = data["points"]

        # Get initial UV from classical method
        # Map classical UVs to our point cloud
        vertices = np.array(mesh.vertices, dtype=np.float32)
        faces = np.array(mesh.faces, dtype=np.int32)
        classical_uv = classical_result["uv_coords"]

        # Build vertex-to-UV mapping and interpolate to points
        uv_tensor = torch.tensor(classical_uv, dtype=torch.float32, device=self.device)
        points_tensor = points[0]  # sampled points (N, 3), NOT mesh vertices

        # Simple nearest-vertex interpolation
        vert_tensor = torch.tensor(vertices, dtype=torch.float32, device=self.device)
        dists = torch.cdist(points_tensor, vert_tensor)
        nearest_vert = dists.argmin(dim=1)
        uv_init = uv_tensor[nearest_vert].unsqueeze(0)  # (1, N, 2)

        # Step 2: ML refinement
        print("  Step 2: ML refinement...")
        iters = num_iterations or self.num_iterations

        if self._refiner is not None:
            # Use learned refiner
            uv_init_tensor = uv_init.to(self.device)
            with torch.no_grad():
                refined = self._refiner(points, uv_init_tensor)
                uv_per_point = refined["uv_refined"][0].cpu().numpy()  # (N, 2)
            # Barycentric interpolation: per-point UVs → per-vertex UVs
            from ..data.mesh_io import interpolate_uv_barycentric
            uv_coords = interpolate_uv_barycentric(
                uv_per_point, points[0].numpy(),
                np.array(mesh.vertices, dtype=np.float32),
                faces, data["face_idx"].numpy(),
            )
            history = {"refiner": True}
        else:
            # Use standard training
            from ..training.trainer import train_unsupervised
            history = train_unsupervised(
                model=self.model,
                points=points,
                num_iterations=iters,
                log_every=log_every,
                device=self.device,
                progress_callback=progress_callback,
                edges=data["edges"],
                faces=data["faces"],
            )
            with torch.no_grad():
                outputs = self.model(points)
            uv_per_point = outputs["uv_coords"][0].cpu().numpy()  # (N, 2)
            # Barycentric interpolation: per-point UVs → per-vertex UVs
            from ..data.mesh_io import interpolate_uv_barycentric
            uv_coords = interpolate_uv_barycentric(
                uv_per_point, points[0].numpy(),
                np.array(mesh.vertices, dtype=np.float32),
                faces, data["face_idx"].numpy(),
            )

        return {
            "uv_coords": uv_coords,
            "vertices": np.array(mesh.vertices, dtype=np.float32),
            "faces": data["faces"].cpu().numpy(),
            "edges": data["edges"].cpu().numpy(),
            "history": history,
            "mesh": data["mesh"],
            "mode": "hybrid",
            "classical_num_charts": classical_result.get("num_charts", 1),
        }

    def _interpolate_face_to_point(
        self,
        face_features: torch.Tensor,
        points: torch.Tensor,
        mesh,
    ) -> torch.Tensor:
        """Interpolate per-face features to per-point features.

        Args:
            face_features: (num_faces, feature_dim) per-face features
            points: (1, N, 3) point cloud
            mesh: trimesh mesh

        Returns:
            point_features: (1, N, feature_dim) per-point features
        """
        # Simple approach: assign each point the feature of its closest face
        # For production, could use barycentric interpolation
        vertices = torch.tensor(mesh.vertices, dtype=torch.float32, device=self.device)
        faces = torch.tensor(mesh.faces, dtype=torch.long, device=self.device)

        # Compute face centroids
        face_verts = vertices[faces]  # (F, 3, 3)
        centroids = face_verts.mean(dim=1)  # (F, 3)

        # For each point, find closest face centroid
        pts = points[0]  # (N, 3)
        dists = torch.cdist(pts, centroids)  # (N, F)
        nearest_face = dists.argmin(dim=1)  # (N,)

        # Gather features
        point_features = face_features[nearest_face]  # (N, feature_dim)
        return point_features.unsqueeze(0)  # (1, N, feature_dim)

    def export(
        self,
        result: dict,
        output_path: str | Path,
        add_margins: bool = True,
    ):
        """Export unwrapped mesh to OBJ.

        Args:
            result: dict from unwrap()
            output_path: output OBJ file path
            add_margins: add UV margins to prevent texture bleeding
        """
        uv = result["uv_coords"]
        vertices = result["vertices"]
        faces = result["faces"]

        # Pack UVs into [0, 1]
        uv_packed = pack_uv_charts(uv)

        # Add margins
        if add_margins:
            uv_packed = add_uv_margins(uv_packed, faces)

        export_uv_mesh(output_path, vertices, uv_packed, faces)

    def _load_model(self, path: str | Path):
        """Load model weights from checkpoint."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        if "model_state_dict" in checkpoint:
            self.model.load_state_dict(checkpoint["model_state_dict"])
        else:
            self.model.load_state_dict(checkpoint)
        print(f"Loaded model from {path}")
