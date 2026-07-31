"""FastAPI web server for ML UV Unwrap."""

from __future__ import annotations

import asyncio
import shutil
import time
import uuid
from pathlib import Path

import numpy as np
import torch
import trimesh
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from src.models import FlexParaUnwrapper
from src.pipeline.preprocessor import MeshPreprocessor
from src.pipeline.postprocessor import add_uv_margins, export_uv_mesh, pack_uv_charts
from src.training.trainer import train_unsupervised
from src.models.partfield.extractor import PartFieldFeatureExtractor
from src.optimization.slim import slim_optimize

app = FastAPI(title="ML UV Unwrap", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).parent
UPLOAD_DIR = BASE_DIR / "uploads"
RESULT_DIR = BASE_DIR / "results"
UPLOAD_DIR.mkdir(exist_ok=True)
RESULT_DIR.mkdir(exist_ok=True)

# In-memory job store
jobs: dict[str, dict] = {}

# Shared model (loaded once)
_model: FlexParaUnwrapper | None = None
_partfield_extractor: PartFieldFeatureExtractor | None = None


def get_model(num_charts: int = 1, use_partuv: bool = False) -> FlexParaUnwrapper:
    global _model
    partfield_dim = 448 if use_partuv else 0
    if _model is None or _model.num_charts != num_charts or _model.partfield_dim != partfield_dim:
        _model = FlexParaUnwrapper(
            num_charts=num_charts,
            hidden_dim=128,
            num_layers=5,
            partfield_dim=partfield_dim,
        )
    return _model


def get_partfield_extractor() -> PartFieldFeatureExtractor | None:
    global _partfield_extractor
    try:
        if _partfield_extractor is None:
            _partfield_extractor = PartFieldFeatureExtractor(device="cpu")
        return _partfield_extractor
    except Exception:
        return None


def _postprocess_uvs(
    uv_coords: np.ndarray,
    vertices: np.ndarray,
    faces: np.ndarray,
    slim_iters: int = 20,
    use_optcuts: bool = True,
    chart_labels=None,
) -> np.ndarray:
    """Apply OptCuts-style joint optimization + packing + margins.

    This is the universal postprocessing step for all UV methods.
    Alternates between distortion analysis, seam selection, and SLIM.
    """
    if use_optcuts:
        from src.optimization.optcuts import optcuts_joint_optimize
        import trimesh as _tm
        try:
            mesh = _tm.Trimesh(vertices=vertices, faces=faces, process=False)
            face_adj = mesh.face_adjacency if hasattr(mesh, 'face_adjacency') else None
            face_adj_edges = mesh.face_adjacency_edges if hasattr(mesh, 'face_adjacency_edges') else None
        except Exception:
            face_adj = None
            face_adj_edges = None

        uv_optimized, _ = optcuts_joint_optimize(
            vertices, faces, uv_coords,
            num_rounds=3,
            slim_iters=slim_iters,
            distortion_threshold=0.15,
            face_adjacency=face_adj,
            face_adjacency_edges=face_adj_edges,
        )
    else:
        from src.optimization.slim import slim_optimize
        uv_optimized = slim_optimize(
            vertices, faces, uv_coords,
            num_iterations=slim_iters,
        )

    # Multi-chart packing with advanced bin-packer
    if chart_labels is not None and len(set(chart_labels)) > 1:
        try:
            from src.pipeline.uv_packer import pack_uv_charts_advanced
            faces_arr = np.array(faces, dtype=np.int32)
            unique_charts = sorted(set(chart_labels))
            chart_uvs_list = []
            chart_faces_list = []
            for ci in unique_charts:
                vert_mask = np.array(chart_labels) == ci
                vert_indices = np.where(vert_mask)[0]
                vert_set = set(vert_indices.tolist())
                chart_face_mask = np.all(np.isin(faces_arr, list(vert_set)), axis=1)
                chart_f = faces_arr[chart_face_mask]
                if len(chart_f) == 0:
                    continue
                local_idx = {int(v): i for i, v in enumerate(sorted(vert_set))}
                local_faces = np.array([[local_idx[int(v)] for v in f[:3]] for f in chart_f], dtype=np.int32)
                local_uvs = uv_optimized[sorted(vert_set)]
                chart_uvs_list.append(local_uvs)
                chart_faces_list.append(local_faces)
            if chart_uvs_list:
                result = pack_uv_charts_advanced(
                    chart_uvs_list, chart_faces_list,
                    faces_arr, uv_optimized,
                    method="skyline", margin=0.01,
                )
                uv_packed = result["packed_uvs"]
                uv_packed = add_uv_margins(uv_packed, faces_arr)
                return uv_packed
        except Exception:
            pass

    # Single chart or fallback: normalize + margins
    uv_packed = pack_uv_charts(uv_optimized)
    uv_packed = add_uv_margins(uv_packed, faces)
    return uv_packed


def _run_unwrap_common(input_path, job, params):
    """Shared unwrap logic: runs pipeline, postprocesses with chart-aware packing, exports."""
    from src.pipeline.unwrapper import UVUnwrapPipeline

    mode = params.get("method", "detect")
    pipeline = UVUnwrapPipeline(
        mode=mode,
        num_points=params.get("num_points", 5000),
        num_charts=params.get("num_charts", 4),
        num_iterations=params.get("num_iterations", 800),
        device="cpu",
        classical_method=params.get("classical_method", "xatlas"),
    )

    job["progress"] = 5
    result = pipeline.unwrap(
        input_path,
        num_iterations=params.get("num_iterations", 800),
        progress_callback=lambda step, total, losses=None: job.update({"progress": 5 + int(step / total * 85)}),
    )

    uv_coords = result["uv_coords"]
    chart_labels = result.get("chart_labels", None)
    if chart_labels is not None and hasattr(chart_labels, 'tolist'):
        chart_labels = chart_labels.tolist()

    uv_packed = _postprocess_uvs(
        uv_coords, result["vertices"], result["faces"],
        chart_labels=chart_labels,
    )

    result_dir = RESULT_DIR / job["id"]
    result_dir.mkdir(parents=True, exist_ok=True)
    output_path = result_dir / "unwrapped.obj"
    export_uv_mesh(output_path, vertices=result["vertices"], uv_coords=uv_packed, faces=result["faces"])

    if "mesh" in result:
        try:
            result["mesh"].export(str(result_dir / "preview.glb"), file_type="glb")
        except Exception:
            pass

    return result, uv_packed, output_path, result_dir


@app.get("/api/health")
async def health():
    return {"status": "ok"}


@app.post("/api/upload")
async def upload_mesh(file: UploadFile = File(...)):
    """Upload a mesh file and return a job ID."""
    allowed = {".obj", ".ply", ".stl", ".glb", ".gltf", ".off", ".fbx"}
    suffix = Path(file.filename or "mesh.obj").suffix.lower()
    if suffix not in allowed:
        raise HTTPException(400, f"Unsupported file type: {suffix}")

    job_id = uuid.uuid4().hex[:12]
    job_dir = UPLOAD_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    save_path = job_dir / f"input{suffix}"
    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # Quick mesh info
    try:
        mesh = trimesh.load(save_path, force="mesh")
        if isinstance(mesh, trimesh.Scene):
            mesh = trimesh.util.concatenate(mesh.dump())
        bbox = mesh.bounding_box.extents
        info = {
            "vertices": len(mesh.vertices),
            "faces": len(mesh.faces),
            "filename": file.filename,
            "bounds": f"{bbox[0]:.2f} x {bbox[1]:.2f} x {bbox[2]:.2f}",
        }
    except Exception as e:
        raise HTTPException(400, f"Failed to parse mesh: {e}")

    jobs[job_id] = {
        "id": job_id,
        "status": "uploaded",
        "input_path": str(save_path),
        "filename": file.filename,
        "mesh_info": info,
        "progress": 0,
        "created_at": time.time(),
    }

    return {"job_id": job_id, "mesh_info": info}


@app.post("/api/unwrap/{job_id}")
async def start_unwrap(
    job_id: str,
    num_iterations: int = Form(500),
    num_points: int = Form(3000),
    num_charts: int = Form(1),
    lr: float = Form(1e-3),
    mode: str = Form("ml"),
    classical_method: str = Form("xatlas"),
):
    """Start UV unwrapping as a background task.

    Modes:
    - ml: Standard FlexPara unsupervised optimization
    - classical: xatlas/LSCM/ABF++ parameterization
    - hybrid: Classical init + ML refinement
    """
    if job_id not in jobs:
        raise HTTPException(404, "Job not found")

    job = jobs[job_id]
    if job["status"] == "processing":
        raise HTTPException(409, "Already processing")

    job["status"] = "processing"
    job["progress"] = 0
    job["params"] = {
        "num_iterations": num_iterations,
        "num_points": num_points,
        "num_charts": num_charts,
        "lr": lr,
        "mode": mode,
        "classical_method": classical_method,
    }

    asyncio.create_task(_run_unwrap(job_id))
    return {"job_id": job_id, "status": "processing", "mode": mode}


@app.post("/api/unwrap-partuv/{job_id}")
async def start_unwrap_partuv(
    job_id: str,
    num_iterations: int = Form(500),
    num_points: int = Form(3000),
    num_charts: int = Form(1),
    lr: float = Form(1e-3),
):
    """Start PartUV-guided UV unwrapping."""
    if job_id not in jobs:
        raise HTTPException(404, "Job not found")

    job = jobs[job_id]
    if job["status"] == "processing":
        raise HTTPException(409, "Already processing")

    job["status"] = "processing"
    job["progress"] = 0
    job["use_partuv"] = True
    job["params"] = {
        "num_iterations": num_iterations,
        "num_points": num_points,
        "num_charts": num_charts,
        "lr": lr,
        "mode": "partuv",
    }

    asyncio.create_task(_run_unwrap(job_id))
    return {"job_id": job_id, "status": "processing", "mode": "partuv"}


@app.post("/api/unwrap-classical/{job_id}")
async def start_unwrap_classical(
    job_id: str,
    method: str = Form("xatlas"),
    max_charts: int = Form(0),
):
    """Start classical UV unwrapping using xatlas/LSCM/ABF++."""
    if job_id not in jobs:
        raise HTTPException(404, "Job not found")

    job = jobs[job_id]
    if job["status"] == "processing":
        raise HTTPException(409, "Already processing")

    job["status"] = "processing"
    job["progress"] = 0
    job["params"] = {
        "mode": "classical",
        "classical_method": method,
        "max_charts": max_charts,
        "num_points": 5000,
    }

    asyncio.create_task(_run_unwrap(job_id))
    return {"job_id": job_id, "status": "processing", "mode": "classical", "method": method}


async def _run_unwrap(job_id: str):
    """Background unwrap task."""
    job = jobs[job_id]
    try:
        input_path = Path(job["input_path"])
        params = job["params"]
        mode = params.get("mode", "ml")
        use_partuv = job.get("use_partuv", False) or mode == "partuv"

        print(f"[unwrap {job_id}] Starting background task: mode={mode}")

        loop = asyncio.get_running_loop()

        if mode == "classical":
            result = await loop.run_in_executor(None, _run_classical_unwrap, input_path, job, params)
        elif mode == "hybrid":
            result = await loop.run_in_executor(None, _run_hybrid_unwrap, input_path, job, params)
        elif mode == "multi_chart":
            result = await loop.run_in_executor(None, _run_multi_chart_unwrap, input_path, job, params)
        elif mode == "detect":
            result = await loop.run_in_executor(None, _run_detect_unwrap, input_path, job, params)
        elif mode == "flatten_anything":
            result = await loop.run_in_executor(None, _run_pipeline_unwrap, input_path, job, params, "flatten_anything")
        elif mode == "mesh_tailor":
            result = await loop.run_in_executor(None, _run_pipeline_unwrap, input_path, job, params, "mesh_tailor")
        elif mode == "seam_crafter":
            result = await loop.run_in_executor(None, _run_pipeline_unwrap, input_path, job, params, "seam_crafter")
        elif mode == "uv_segnet":
            result = await loop.run_in_executor(None, _run_pipeline_unwrap, input_path, job, params, "uv_segnet")
        elif mode == "quality_select":
            result = await loop.run_in_executor(None, _run_pipeline_unwrap, input_path, job, params, "quality_select")
        elif mode == "artuv":
            result = await loop.run_in_executor(None, _run_pipeline_unwrap, input_path, job, params, "artuv")
        elif mode == "partuv":
            result = await loop.run_in_executor(None, _run_pipeline_unwrap, input_path, job, params, "partuv")
        elif mode in ("voronoi_disks", "instant_meshes", "libuvula", "ffhq_uv"):
            result = await loop.run_in_executor(None, _run_pipeline_unwrap, input_path, job, params, mode)
        elif mode in ("arap", "harmonic", "conformal", "graph_cuts", "hilbert"):
            job["params"]["classical_method"] = mode
            result = await loop.run_in_executor(None, _run_classical_unwrap, input_path, job, params)
        else:
            result = await loop.run_in_executor(None, _run_ml_unwrap, input_path, job, params, use_partuv)

        job.update(result)

    except Exception as e:
        job["status"] = "failed"
        job["error"] = str(e)
        import traceback
        traceback.print_exc()


def _run_classical_unwrap(input_path: Path, job: dict, params: dict) -> dict:
    """Run classical unwrapping (xatlas/LSCM/ABF)."""
    from src.pipeline.classical_unwrapper import ClassicalUnwrapper

    method = params.get("classical_method", "xatlas")
    unwrapper = ClassicalUnwrapper(method=method)

    job["progress"] = 10
    result = unwrapper.unwrap(input_path)

    job["progress"] = 80

    # Post-process with SLIM
    uv_coords = result["uv_coords"]
    uv_packed = _postprocess_uvs(uv_coords, result["vertices"], result["faces"])

    # Export
    result_dir = RESULT_DIR / job["id"]
    result_dir.mkdir(parents=True, exist_ok=True)
    output_path = result_dir / "unwrapped.obj"
    export_uv_mesh(
        output_path,
        vertices=result["vertices"],
        uv_coords=uv_packed,
        faces=result["faces"],
    )

    # Export preview
    mesh = result["mesh"]
    mesh.export(str(result_dir / "preview.glb"), file_type="glb")

    return {
        "status": "completed",
        "progress": 100,
        "result_path": str(output_path),
        "preview_path": str(result_dir / "preview.glb"),
        "uv_stats": {
            "uv_min": uv_packed.min().tolist(),
            "uv_max": uv_packed.max().tolist(),
            "num_verts": len(uv_packed),
        },
        "mode": "classical",
        "method": method,
    }


def _run_hybrid_unwrap(input_path: Path, job: dict, params: dict) -> dict:
    """Run hybrid unwrapping: classical init + ML refinement."""
    from src.pipeline.unwrapper import UVUnwrapPipeline

    pipeline = UVUnwrapPipeline(
        mode="hybrid",
        num_points=params.get("num_points", 5000),
        num_iterations=params.get("num_iterations", 800),
        device="cpu",
        classical_method=params.get("classical_method", "xatlas"),
    )

    job["progress"] = 5
    result = pipeline.unwrap(
        input_path,
        num_iterations=params.get("num_iterations", 800),
        progress_callback=lambda step, total, losses=None: job.update({"progress": 5 + int(step / total * 85)}),
    )

    # Post-process with SLIM
    uv_coords = result["uv_coords"]
    uv_packed = _postprocess_uvs(uv_coords, result["vertices"], result["faces"])

    # Export
    result_dir = RESULT_DIR / job["id"]
    result_dir.mkdir(parents=True, exist_ok=True)
    output_path = result_dir / "unwrapped.obj"
    export_uv_mesh(
        output_path,
        vertices=result["vertices"],
        uv_coords=uv_packed,
        faces=result["faces"],
    )

    # Export preview
    result["mesh"].export(str(result_dir / "preview.glb"), file_type="glb")

    return {
        "status": "completed",
        "progress": 100,
        "result_path": str(output_path),
        "preview_path": str(result_dir / "preview.glb"),
        "uv_stats": {
            "uv_min": uv_packed.min().tolist(),
            "uv_max": uv_packed.max().tolist(),
            "num_verts": len(uv_packed),
        },
        "mode": "hybrid",
    }


def _run_ml_unwrap(input_path: Path, job: dict, params: dict, use_partuv: bool) -> dict:
    """Run ML-based unwrapping (standard or PartUV)."""
    preprocessor = MeshPreprocessor(
        num_points=params["num_points"], device="cpu"
    )
    data = preprocessor.process(input_path)
    points = data["points"]

    model = get_model(num_charts=params["num_charts"], use_partuv=use_partuv)

    # PartUV: extract PartField features
    partfield_features = None
    if use_partuv:
        extractor = get_partfield_extractor()
        if extractor is not None:
            job["progress"] = 5
            try:
                import copy
                import trimesh as _tm
                mesh = _tm.load(input_path, force="mesh")
                if isinstance(mesh, _tm.Scene):
                    mesh = _tm.util.concatenate(mesh.dump())

                # PartField modifies mesh.vertices in place, so use a copy
                mesh_copy = copy.deepcopy(mesh)

                with torch.no_grad():
                    face_features = extractor.extract(mesh_copy, sample_on_faces=10)
                    # Interpolate per-face features to per-point features
                    vertices = torch.tensor(mesh.vertices, dtype=torch.float32)
                    faces_t = torch.tensor(mesh.faces, dtype=torch.long)
                    centroids = vertices[faces_t].mean(dim=1)
                    pts = points[0]
                    dists = torch.cdist(pts, centroids)
                    nearest_face = dists.argmin(dim=1)
                    partfield_features = face_features[nearest_face].unsqueeze(0)
                    # Verify shape matches point count
                    assert partfield_features.shape[1] == points.shape[1], \
                        f"PartField features {partfield_features.shape} != points {points.shape}"
                    job["num_parts"] = len(set(extractor.cluster_parts(face_features)))
            except Exception as e:
                print(f"PartField extraction failed, falling back: {e}")
                use_partuv = False

    num_iters = params["num_iterations"]

    # Precompute ambient occlusion for visibility loss
    ao_values = None
    try:
        from src.losses.ao_visibility import compute_ambient_occlusion
        ao_np = compute_ambient_occlusion(
            data["vertices"][0].numpy(),
            data["faces"].numpy(),
            num_rays=16,
        )
        ao_values = torch.tensor(ao_np, dtype=torch.float32)
    except Exception as e:
        print(f"AO computation failed, skipping: {e}")

    def _progress_cb(step, total, losses=None):
        base = 10 if use_partuv else 0
        job["progress"] = base + int(step / total * 80)
        if losses:
            job["current_loss"] = {k: float(v) for k, v in losses.items() if isinstance(v, (int, float))}

    train_unsupervised(
        model=model,
        points=points,
        num_iterations=num_iters,
        lr=params["lr"],
        log_every=max(1, num_iters // 20),
        progress_callback=_progress_cb,
        partfield_features=partfield_features,
        edges=data["edges"],
        faces=data["faces"],
        ao_values=ao_values,
    )

    job["progress"] = 90

    # Extract UVs — model outputs per-point UVs, need per-vertex UVs via barycentric interpolation
    with torch.no_grad():
        outputs = model(points, partfield_features=partfield_features)
    uv_per_point = outputs["uv_coords"][0].numpy()  # (N, 2)
    vertices_np = data["vertices"][0].numpy()  # (V, 3)
    points_np = points[0].numpy()  # (N, 3)
    faces_np = data["faces"].numpy()  # (F, 3)
    face_idx_np = data["face_idx"].numpy()  # (N,)

    from src.data.mesh_io import interpolate_uv_barycentric
    uv_coords = interpolate_uv_barycentric(
        uv_per_point, points_np, vertices_np, faces_np, face_idx_np,
    )

    # Post-process with SLIM distortion optimization
    uv_packed = _postprocess_uvs(uv_coords, vertices_np, faces_np)

    # Export
    result_dir = RESULT_DIR / job["id"]
    result_dir.mkdir(parents=True, exist_ok=True)
    output_path = result_dir / "unwrapped.obj"
    export_uv_mesh(
        output_path,
        vertices=vertices_np,
        uv_coords=uv_packed,
        faces=data["faces"].numpy(),
    )

    mesh = data["mesh"]
    mesh.export(str(result_dir / "preview.glb"), file_type="glb")

    return {
        "status": "completed",
        "progress": 100,
        "result_path": str(output_path),
        "preview_path": str(result_dir / "preview.glb"),
        "uv_stats": {
            "uv_min": uv_packed.min().tolist(),
            "uv_max": uv_packed.max().tolist(),
            "num_verts": len(uv_packed),
        },
        "mode": "partuv" if use_partuv else "ml",
    }


def _run_multi_chart_unwrap(input_path: Path, job: dict, params: dict) -> dict:
    """Run multi-chart unwrapping: PartField decomposition + per-chart unwrapping."""
    from src.pipeline.unwrapper import UVUnwrapPipeline

    pipeline = UVUnwrapPipeline(
        mode="multi_chart",
        num_points=params.get("num_points", 5000),
        device="cpu",
    )

    job["progress"] = 5
    result = pipeline.unwrap(input_path)

    # Post-process with SLIM
    uv_coords = result["uv_coords"]
    uv_packed = _postprocess_uvs(uv_coords, result["vertices"], result["faces"])

    # Export
    result_dir = RESULT_DIR / job["id"]
    result_dir.mkdir(parents=True, exist_ok=True)
    output_path = result_dir / "unwrapped.obj"
    export_uv_mesh(
        output_path,
        vertices=result["vertices"],
        uv_coords=uv_packed,
        faces=result["faces"],
    )

    # Export preview
    result["mesh"].export(str(result_dir / "preview.glb"), file_type="glb")

    return {
        "status": "completed",
        "progress": 100,
        "result_path": str(output_path),
        "preview_path": str(result_dir / "preview.glb"),
        "uv_stats": {
            "uv_min": uv_packed.min().tolist(),
            "uv_max": uv_packed.max().tolist(),
            "num_verts": len(uv_packed),
        },
        "mode": "multi_chart",
        "num_charts": result.get("num_charts", 1),
        "total_distortion": result.get("total_distortion", 0.0),
        "chart_labels": result.get("chart_labels", []).tolist() if hasattr(result.get("chart_labels", []), 'tolist') else [],
    }


def _run_pipeline_unwrap(input_path: Path, job: dict, params: dict, mode: str) -> dict:
    """Generic pipeline-based unwrapping for new modes."""
    from src.pipeline.unwrapper import UVUnwrapPipeline

    pipeline = UVUnwrapPipeline(
        mode=mode,
        num_points=params.get("num_points", 5000),
        num_charts=params.get("num_charts", 4),
        num_iterations=params.get("num_iterations", 800),
        device="cpu",
    )

    job["progress"] = 5

    def _progress_cb(step, total, losses=None):
        job["progress"] = 5 + int(step / total * 85)

    result = pipeline.unwrap(
        input_path,
        num_iterations=params.get("num_iterations", 800),
        progress_callback=_progress_cb,
    )

    job["progress"] = 90

    # Post-process with SLIM
    uv_coords = result["uv_coords"]
    uv_packed = _postprocess_uvs(uv_coords, result["vertices"], result["faces"])

    # Export
    result_dir = RESULT_DIR / job["id"]
    result_dir.mkdir(parents=True, exist_ok=True)
    output_path = result_dir / "unwrapped.obj"
    export_uv_mesh(
        output_path,
        vertices=result["vertices"],
        uv_coords=uv_packed,
        faces=result["faces"],
    )

    # Export preview
    result["mesh"].export(str(result_dir / "preview.glb"), file_type="glb")

    return {
        "status": "completed",
        "progress": 100,
        "result_path": str(output_path),
        "preview_path": str(result_dir / "preview.glb"),
        "uv_stats": {
            "uv_min": uv_packed.min().tolist(),
            "uv_max": uv_packed.max().tolist(),
            "num_verts": len(uv_packed),
        },
        "mode": mode,
        "num_charts": result.get("num_charts", 1),
    }


def _run_detect_unwrap(input_path: Path, job: dict, params: dict) -> dict:
    """Auto-detect mesh type and unwrap with best strategy."""
    from src.pipeline.unwrapper import UVUnwrapPipeline

    pipeline = UVUnwrapPipeline(
        mode="detect",
        num_points=params.get("num_points", 5000),
        device="cpu",
    )

    job["progress"] = 5
    result = pipeline.unwrap(input_path)

    # Post-process with SLIM
    uv_coords = result["uv_coords"]
    uv_packed = _postprocess_uvs(uv_coords, result["vertices"], result["faces"])

    # Export
    result_dir = RESULT_DIR / job["id"]
    result_dir.mkdir(parents=True, exist_ok=True)
    output_path = result_dir / "unwrapped.obj"
    export_uv_mesh(
        output_path,
        vertices=result["vertices"],
        uv_coords=uv_packed,
        faces=result["faces"],
    )

    # Export preview
    result["mesh"].export(str(result_dir / "preview.glb"), file_type="glb")

    return {
        "status": "completed",
        "progress": 100,
        "result_path": str(output_path),
        "preview_path": str(result_dir / "preview.glb"),
        "uv_stats": {
            "uv_min": uv_packed.min().tolist(),
            "uv_max": uv_packed.max().tolist(),
            "num_verts": len(uv_packed),
        },
        "mode": "detect",
        "detected_mode": result.get("mode", "unknown"),
    }


@app.get("/api/status/{job_id}")
async def get_status(job_id: str):
    """Poll job status."""
    if job_id not in jobs:
        raise HTTPException(404, "Job not found")
    job = jobs[job_id]
    return {
        "id": job["id"],
        "status": job["status"],
        "progress": job["progress"],
        "filename": job.get("filename"),
        "mesh_info": job.get("mesh_info"),
        "uv_stats": job.get("uv_stats"),
        "use_partuv": job.get("use_partuv", False),
        "num_parts": job.get("num_parts"),
        "num_charts": job.get("num_charts"),
        "total_distortion": job.get("total_distortion"),
        "chart_labels": job.get("chart_labels"),
        "detected_mode": job.get("detected_mode"),
        "error": job.get("error"),
    }


@app.post("/api/detect/{job_id}")
async def start_detect(job_id: str):
    """Auto-detect mesh type and unwrap with best strategy."""
    if job_id not in jobs:
        raise HTTPException(404, "Job not found")

    job = jobs[job_id]
    if job["status"] == "processing":
        raise HTTPException(409, "Already processing")

    job["status"] = "processing"
    job["progress"] = 0
    job["params"] = {
        "mode": "detect",
        "num_points": 5000,
    }

    asyncio.create_task(_run_unwrap(job_id))
    return {"job_id": job_id, "status": "processing", "mode": "detect"}


@app.get("/api/download/{job_id}")
async def download_result(job_id: str):
    """Download the unwrapped OBJ file."""
    if job_id not in jobs:
        raise HTTPException(404, "Job not found")
    job = jobs[job_id]
    if job["status"] != "completed":
        raise HTTPException(400, "Job not completed yet")

    result_path = Path(job["result_path"])
    if not result_path.exists():
        raise HTTPException(404, "Result file not found")

    return FileResponse(
        result_path,
        media_type="model/obj",
        filename=f"{Path(job['filename']).stem}_unwrapped.obj",
    )


@app.get("/api/preview/{job_id}")
async def get_preview(job_id: str):
    """Get preview GLB of the original mesh."""
    if job_id not in jobs:
        raise HTTPException(404, "Job not found")
    job = jobs[job_id]

    preview = job.get("preview_path")
    if preview and Path(preview).exists():
        return FileResponse(preview, media_type="model/gltf-binary")

    # Fall back to uploaded file
    input_path = Path(job["input_path"])
    if input_path.exists():
        suffix = input_path.suffix.lower()
        media_map = {
            ".glb": "model/gltf-binary",
            ".gltf": "model/gltf+json",
            ".obj": "model/obj",
            ".ply": "model/ply",
            ".stl": "model/stl",
        }
        return FileResponse(input_path, media_type=media_map.get(suffix, "application/octet-stream"))

    raise HTTPException(404, "No preview available")


@app.get("/api/uv-image/{job_id}")
async def get_uv_image(job_id: str, size: int = 2048, download: bool = False):
    """Generate and return a checker UV visualization image."""
    if job_id not in jobs:
        raise HTTPException(404, "Job not found")
    job = jobs[job_id]
    if job["status"] != "completed":
        raise HTTPException(400, "Job not completed yet")

    result_path = Path(job["result_path"])
    if not result_path.exists():
        raise HTTPException(404, "Result not found")

    size = max(256, min(4096, size))

    try:
        from PIL import Image, ImageDraw

        vertices, uv_coords, faces = _parse_obj(result_path)

        img = Image.new("RGB", (size, size), (30, 30, 40))
        draw = ImageDraw.Draw(img)

        # Checker background
        check_size = max(8, size // 32)
        for y in range(0, size, check_size):
            for x in range(0, size, check_size):
                if ((x // check_size) + (y // check_size)) % 2 == 0:
                    draw.rectangle([x, y, x + check_size, y + check_size], fill=(45, 45, 60))

        # Filled UV triangles with semi-transparent colors
        import colorsys
        num_faces = len(face_colors := faces)
        for i, face in enumerate(faces):
            if len(face) < 3:
                continue
            pts = [(uv_coords[vi][0] * size, (1 - uv_coords[vi][1]) * size) for vi in face[:3]]
            # Unique color per face
            hue = (i * 0.618033988749895) % 1.0
            r, g, b = [int(c * 255) for c in colorsys.hls_to_rgb(hue, 0.45, 0.6)]
            draw.polygon(pts, fill=(r, g, b), outline=None)

        # Wireframe on top
        for face in faces:
            if len(face) < 3:
                continue
            pts = [(uv_coords[vi][0] * size, (1 - uv_coords[vi][1]) * size) for vi in face[:3]]
            for j in range(3):
                draw.line([pts[j], pts[(j + 1) % 3]], fill=(200, 200, 220), width=1)

        img_path = result_path.parent / f"uv_map_{size}.png"
        img.save(img_path)

        if download:
            return FileResponse(
                img_path,
                media_type="image/png",
                filename=f"uv_map_{size}.png",
            )
        return FileResponse(img_path, media_type="image/png")

    except Exception as e:
        raise HTTPException(500, f"Failed to generate UV image: {e}")


@app.get("/api/mesh-analysis/{job_id}")
async def get_mesh_analysis(job_id: str):
    """Compute per-face distortion, AO, curvature, chart labels, and seam edges."""
    if job_id not in jobs:
        raise HTTPException(404, "Job not found")
    job = jobs[job_id]
    if job["status"] != "completed":
        raise HTTPException(400, "Job not completed yet")

    result_path = Path(job["result_path"])
    if not result_path.exists():
        raise HTTPException(404, "Result not found")

    try:
        vertices, uv_coords, faces = _parse_obj(result_path)
        num_faces = len(faces)

        # Build face array (N, 3) of vertex indices
        face_arr = np.array([f[:3] for f in faces if len(f) >= 3], dtype=np.int32)
        if len(face_arr) == 0:
            raise HTTPException(400, "No valid faces")

        # --- Distortion (per-face UV stretch) ---
        distortion = np.zeros(num_faces, dtype=np.float32)
        for i, face in enumerate(faces):
            if len(face) < 3:
                continue
            v0, v1, v2 = vertices[face[0]], vertices[face[1]], vertices[face[2]]
            u0, u1, u2 = uv_coords[face[0]], uv_coords[face[1]], uv_coords[face[2]]

            e3d_0 = np.linalg.norm(v1 - v0) + 1e-8
            e3d_1 = np.linalg.norm(v2 - v1) + 1e-8
            e3d_2 = np.linalg.norm(v0 - v2) + 1e-8
            euv_0 = np.linalg.norm(u1 - u0) + 1e-8
            euv_1 = np.linalg.norm(u2 - u1) + 1e-8
            euv_2 = np.linalg.norm(u0 - u2) + 1e-8

            r0, r1, r2 = euv_0 / e3d_0, euv_1 / e3d_1, euv_2 / e3d_2
            mean_r = (r0 + r1 + r2) / 3.0
            distortion[i] = np.sqrt(((r0 - mean_r)**2 + (r1 - mean_r)**2 + (r2 - mean_r)**2) / 3.0)

        # --- AO (per-face ambient occlusion) ---
        ao = np.ones(num_faces, dtype=np.float32)
        try:
            from src.losses.ao_visibility import compute_ambient_occlusion
            ao_vert = compute_ambient_occlusion(vertices, face_arr, num_rays=16)
            for i, face in enumerate(faces):
                if len(face) >= 3:
                    ao[i] = (ao_vert[face[0]] + ao_vert[face[1]] + ao_vert[face[2]]) / 3.0
        except Exception:
            pass

        # --- Curvature (per-vertex Gaussian curvature) ---
        curvature = np.zeros(len(vertices), dtype=np.float32)
        try:
            import trimesh as _tm
            mesh = _tm.Trimesh(vertices=vertices, faces=face_arr, process=False)
            # Angle defect approximation for Gaussian curvature
            for vi in range(len(vertices)):
                incident = np.where(face_arr == vi)[0]
                if len(incident) == 0:
                    continue
                total_angle = 0.0
                for fi in incident:
                    f = face_arr[fi]
                    p = vertices[f]
                    edges = [np.linalg.norm(p[(j+1)%3] - p[j]) for j in range(3)]
                    # Law of cosines for angle at vertex vi
                    idx_in_face = list(f).index(vi)
                    a = edges[(idx_in_face + 1) % 3]
                    b = edges[(idx_in_face + 2) % 3]
                    c = edges[idx_in_face]
                    cos_angle = (a*a + b*b - c*c) / (2*a*b + 1e-10)
                    cos_angle = np.clip(cos_angle, -1, 1)
                    total_angle += np.arccos(cos_angle)
                curvature[vi] = 2 * np.pi - total_angle
        except Exception:
            pass

        # Per-face curvature from vertex values
        face_curvature = np.zeros(num_faces, dtype=np.float32)
        for i, face in enumerate(faces):
            if len(face) >= 3:
                face_curvature[i] = np.mean([curvature[face[j]] for j in range(3)])

        # --- Chart labels ---
        chart_labels = job.get("chart_labels", [])
        if isinstance(chart_labels, list) and len(chart_labels) > 0:
            chart_labels = np.array(chart_labels, dtype=np.int32)
        else:
            chart_labels = np.zeros(len(vertices), dtype=np.int32)

        # --- Seam edges ---
        edge_count = {}
        for face in faces:
            for j in range(min(3, len(face))):
                a, b = face[j], face[(j+1) % len(face)]
                key = (min(a, b), max(a, b))
                edge_count[key] = edge_count.get(key, 0) + 1
        seam_edges = [[list(e)] for e, c in edge_count.items() if c == 1]

        return {
            "distortion": distortion.tolist(),
            "ao": ao.tolist(),
            "curvature": face_curvature.tolist(),
            "chart_labels": chart_labels.tolist() if hasattr(chart_labels, 'tolist') else list(chart_labels),
            "seam_edges": seam_edges,
            "vertices": vertices.tolist(),
            "faces": [f[:3] for f in faces],
            "uv_coords": uv_coords.tolist() if len(uv_coords) > 0 else [],
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Analysis failed: {e}")


def _parse_obj(path: Path):
    """Simple OBJ parser for vertices, UVs, and faces."""
    vertices = []
    uvs = []
    faces = []

    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            if parts[0] == "v":
                vertices.append([float(x) for x in parts[1:4]])
            elif parts[0] == "vt":
                uvs.append([float(x) for x in parts[1:3]])
            elif parts[0] == "f":
                face = []
                for p in parts[1:]:
                    ids = p.split("/")
                    face.append(int(ids[0]) - 1)  # 0-indexed vertex
                faces.append(face)

    return np.array(vertices), np.array(uvs), faces


# ── UV Cutting & Joining ──────────────────────────────────────────


@app.get("/api/cut-edges/{job_id}")
async def get_cut_edges(job_id: str):
    """Get the edge graph of the mesh for seam cutting UI."""
    if job_id not in jobs:
        raise HTTPException(404, "Job not found")
    job = jobs[job_id]

    input_path = Path(job["input_path"])
    mesh = trimesh.load(str(input_path), force="mesh")
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(mesh.dump())

    vertices = np.array(mesh.vertices, dtype=np.float32)
    faces = np.array(mesh.faces, dtype=np.int32)

    edges_unique = np.array(mesh.edges_unique, dtype=np.int32)
    edge_verts = vertices[edges_unique]

    # Compute per-edge dihedral angle for highlighting
    face_adj = mesh.face_adjacency if hasattr(mesh, 'face_adjacency') else None
    face_normals = np.array(mesh.face_normals, dtype=np.float32)

    edge_angles = np.zeros(len(edges_unique), dtype=np.float32)
    if face_adj is not None and len(face_adj) > 0:
        try:
            face_adj_edges = mesh.face_adjacency_edges
            for i, (f0, f1) in enumerate(face_adj):
                if i >= len(face_adj_edges):
                    break
                edge_key = tuple(sorted(face_adj_edges[i]))
                matches = np.where(
                    (edges_unique[:, 0] == edge_key[0]) & (edges_unique[:, 1] == edge_key[1])
                )[0]
                if len(matches) > 0:
                    n0 = face_normals[f0]
                    n1 = face_normals[f1]
                    cos_angle = np.clip(np.dot(n0, n1), -1, 1)
                    edge_angles[matches[0]] = np.degrees(np.arccos(cos_angle))
        except Exception:
            pass

    # Get current UVs if available
    current_uvs = None
    result_dir = RESULT_DIR / job_id
    result_path = result_dir / "unwrapped.obj"
    if result_path.exists():
        try:
            current_uvs = _parse_obj(result_path)
        except Exception:
            pass

    return {
        "vertices": vertices.tolist(),
        "faces": faces.tolist(),
        "edges": edges_unique.tolist(),
        "edge_verts": edge_verts.tolist(),
        "edge_angles": edge_angles.tolist(),
        "current_uvs": current_uvs[1].tolist() if current_uvs is not None else None,
    }


@app.post("/api/cut-edges/{job_id}")
async def apply_cut_edges(
    job_id: str,
    edge_indices: str = Form(""),
    method: str = Form("lscm"),
):
    """Apply seam cuts along selected edges and re-unwrap.

    edge_indices: comma-separated list of edge indices to cut
    method: re-unwrap method (lscm, xatlas, harmonic, arap, conformal)
    """
    if job_id not in jobs:
        raise HTTPException(404, "Job not found")
    job = jobs[job_id]

    input_path = Path(job["input_path"])
    mesh = trimesh.load(str(input_path), force="mesh")
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(mesh.dump())

    vertices = np.array(mesh.vertices, dtype=np.float32)
    faces = np.array(mesh.faces, dtype=np.int32)
    edges_unique = np.array(mesh.edges_unique, dtype=np.int32)

    # Parse cut edge indices
    if edge_indices.strip():
        cut_idx = np.array([int(x) for x in edge_indices.split(",") if x.strip()], dtype=np.int32)
    else:
        cut_idx = np.array([], dtype=np.int32)

    print(f"  Cut: {len(cut_idx)} edges selected, method={method}")

    # Split mesh along cut edges
    if len(cut_idx) > 0:
        cut_edges = edges_unique[cut_idx]
        new_mesh = _split_mesh_along_edges(vertices, faces, cut_edges)
    else:
        new_mesh = mesh

    # Re-unwrap with selected method
    from src.pipeline.classical_unwrapper import ClassicalUnwrapper
    unwrapper = ClassicalUnwrapper(method=method)
    result = unwrapper.unwrap(new_mesh)

    uv_coords = result["uv_coords"]
    uv_packed = _postprocess_uvs(uv_coords, result["vertices"], result["faces"])

    # Export
    result_dir = RESULT_DIR / job_id
    result_dir.mkdir(parents=True, exist_ok=True)
    output_path = result_dir / "unwrapped.obj"
    export_uv_mesh(output_path, vertices=result["vertices"], uv_coords=uv_packed, faces=result["faces"])
    new_mesh.export(str(result_dir / "preview.glb"), file_type="glb")

    # Re-extract for frontend
    _, new_uv, _ = _parse_obj(output_path)

    job.update({
        "status": "completed",
        "progress": 100,
        "result_path": str(output_path),
        "preview_path": str(result_dir / "preview.glb"),
        "uv_stats": {
            "uv_min": uv_packed.min().tolist(),
            "uv_max": uv_packed.max().tolist(),
            "num_verts": len(uv_packed),
        },
        "mode": "cut",
        "method": method,
    })

    return {
        "status": "completed",
        "uv_coords": new_uv.tolist(),
        "vertices": result["vertices"].tolist(),
        "faces": result["faces"].tolist(),
        "num_charts": result.get("num_charts", 1),
    }


def _split_mesh_along_edges(
    vertices: np.ndarray,
    faces: np.ndarray,
    cut_edges: np.ndarray,
) -> trimesh.Trimesh:
    """Split mesh by duplicating vertices along cut edges.

    For each cut edge, duplicate the shared vertices so the two adjacent
    faces no longer share them — creating a boundary seam.
    """
    import trimesh

    cut_set = set(map(tuple, np.sort(cut_edges, axis=1).tolist()))

    # Build face-edge adjacency
    face_edges = np.stack([
        np.sort(faces[:, [0, 1]], axis=1),
        np.sort(faces[:, [1, 2]], axis=1),
        np.sort(faces[:, [0, 2]], axis=1),
    ], axis=1)  # (F, 3, 2)

    # Determine which face-corner to split at each cut edge
    # For simplicity: duplicate vertices for faces on one side of the cut
    new_verts = list(vertices)
    vert_map = {i: i for i in range(len(vertices))}

    # Find connected components before cutting
    import igl
    _, components, _ = igl.connected_components(
        igl.adjacency_matrix(faces.astype(np.int32))
    )

    # Group faces by component
    num_components = components.max() + 1
    component_faces = [np.where(components == c)[0] for c in range(num_components)]

    # For each cut edge, duplicate vertices for one side
    new_faces = faces.copy()
    next_vert = len(vertices)

    for cut_e in cut_edges:
        v0, v1 = int(cut_e[0]), int(cut_e[1])
        # Find faces containing this edge
        has_edge = (
            ((faces[:, 0] == v0) & (faces[:, 1] == v1)) |
            ((faces[:, 1] == v0) & (faces[:, 0] == v1)) |
            ((faces[:, 0] == v0) & (faces[:, 2] == v1)) |
            ((faces[:, 2] == v0) & (faces[:, 0] == v1)) |
            ((faces[:, 1] == v0) & (faces[:, 2] == v1)) |
            ((faces[:, 2] == v0) & (faces[:, 1] == v1))
        )
        face_ids = np.where(has_edge)[0]
        if len(face_ids) < 2:
            continue

        # Pick one face to duplicate vertices for
        f_idx = face_ids[0]
        f_verts = list(faces[f_idx])

        # Duplicate v0 and v1
        dup_v0 = next_vert
        new_verts.append(vertices[v0].tolist())
        next_vert += 1
        dup_v1 = next_vert
        new_verts.append(vertices[v1].tolist())
        next_vert += 1

        # Replace in face
        for j in range(3):
            if f_verts[j] == v0:
                new_faces[f_idx, j] = dup_v0
            elif f_verts[j] == v1:
                new_faces[f_idx, j] = dup_v1

    return trimesh.Trimesh(
        vertices=np.array(new_verts, dtype=np.float32),
        faces=new_faces.astype(np.int32),
    )


@app.get("/api/uv-islands/{job_id}")
async def get_uv_islands(job_id: str):
    """Get UV island connectivity from the current UV result."""
    if job_id not in jobs:
        raise HTTPException(404, "Job not found")

    result_dir = RESULT_DIR / job_id
    result_path = result_dir / "unwrapped.obj"
    if not result_path.exists():
        raise HTTPException(404, "No UV result found — unwrap first")

    vertices, uvs, faces = _parse_obj(result_path)
    faces_arr = np.array(faces, dtype=np.int32)

    # Build UV-space adjacency: edges that are close in UV space
    uv_edges = np.column_stack([
        faces_arr[:, [0, 1]],
        faces_arr[:, [1, 2]],
        faces_arr[:, [0, 2]],
    ]).reshape(-1, 2)

    # UV-space edge distances
    if uvs is not None and len(uvs) > 0:
        uv_arr = np.array(uvs, dtype=np.float32)
        # Only consider edges where UVs are close (same island)
        dists = np.linalg.norm(uv_arr[uv_edges[:, 0]] - uv_arr[uv_edges[:, 1]], axis=1)
        same_island = dists < 0.1  # threshold for "same island"

        # Build adjacency with same-island edges only
        from scipy import sparse
        row = uv_edges[same_island, 0]
        col = uv_edges[same_island, 1]
        data = np.ones(len(row))
        A = sparse.csr_matrix((data, (row, col)), shape=(len(vertices), len(vertices)))
        A = A + A.T

        import igl
        _, components, _ = igl.connected_components(A)
    else:
        components = np.zeros(len(vertices), dtype=np.int32)

    # Group faces by island (based on first vertex's component)
    face_island = components[faces_arr[:, 0]]
    island_ids = np.unique(face_island)
    islands = []
    for iid in island_ids:
        mask = face_island == iid
        island_faces = faces_arr[mask]
        island_verts = np.unique(island_faces)
        islands.append({
            "id": int(iid),
            "face_count": int(mask.sum()),
            "vertex_count": len(island_verts),
            "faces": np.where(mask)[0].tolist(),
            "vertices": island_verts.tolist(),
        })

    return {
        "islands": islands,
        "num_islands": len(islands),
        "uv_coords": uvs.tolist() if uvs is not None else None,
    }


@app.post("/api/join-islands/{job_id}")
async def join_uv_islands(
    job_id: str,
    island_ids: str = Form(""),
    method: str = Form("xatlas"),
):
    """Join selected UV islands into a single chart.

    island_ids: comma-separated island IDs to join
    method: re-unwrap method for the joined chart
    """
    if job_id not in jobs:
        raise HTTPException(404, "Job not found")
    job = jobs[job_id]

    result_dir = RESULT_DIR / job_id
    result_path = result_dir / "unwrapped.obj"
    if not result_path.exists():
        raise HTTPException(404, "No UV result found")

    vertices, uvs, faces = _parse_obj(result_path)
    faces_arr = np.array(faces, dtype=np.int32)

    # Get island info
    island_info = await get_uv_islands(job_id)
    islands = island_info["islands"]

    # Parse island IDs to join
    if island_ids.strip():
        join_ids = [int(x) for x in island_ids.split(",") if x.strip()]
    else:
        raise HTTPException(400, "No island IDs provided")

    # Find faces belonging to selected islands
    selected_faces = set()
    for island in islands:
        if island["id"] in join_ids:
            selected_faces.update(island["faces"])

    if not selected_faces:
        raise HTTPException(400, "No faces found for selected islands")

    selected_face_idx = np.array(sorted(selected_faces), dtype=np.int32)
    selected_faces_arr = faces_arr[selected_face_idx]

    # Extract sub-mesh for the selected islands
    unique_verts = np.unique(selected_faces_arr)
    vert_map = np.full(len(vertices), -1, dtype=np.int32)
    vert_map[unique_verts] = np.arange(len(unique_verts))
    sub_verts = vertices[unique_verts]
    sub_faces = vert_map[selected_faces_arr]

    # Re-unwrap as single chart
    import trimesh as _tm
    sub_mesh = _tm.Trimesh(vertices=sub_verts, faces=sub_faces, process=False)

    from src.pipeline.classical_unwrapper import ClassicalUnwrapper
    unwrapper = ClassicalUnwrapper(method=method)
    result = unwrapper.unwrap(sub_mesh)

    # Map UVs back to full vertex array
    uv_full = np.array(uvs, dtype=np.float32) if uvs is not None else np.zeros((len(vertices), 2), dtype=np.float32)
    sub_uv = result["uv_coords"]

    # sub_uv is indexed by sub_faces vertices, which map to unique_verts
    for i, vi in enumerate(unique_verts):
        if i < len(sub_uv):
            uv_full[vi] = sub_uv[i]

    # Export
    output_path = result_dir / "unwrapped.obj"
    export_uv_mesh(output_path, vertices=vertices, uv_coords=uv_full, faces=faces_arr)

    # Reload for preview
    mesh = trimesh.load(str(result_dir / "preview.glb")) if (result_dir / "preview.glb").exists() else None
    if mesh is not None:
        mesh.export(str(result_dir / "preview.glb"), file_type="glb")

    job.update({
        "status": "completed",
        "progress": 100,
        "result_path": str(output_path),
        "mode": "join",
    })

    _, new_uv, _ = _parse_obj(output_path)

    return {
        "status": "completed",
        "uv_coords": new_uv.tolist(),
        "vertices": vertices.tolist(),
        "faces": faces_arr.tolist(),
    }


# Serve frontend
static_dir = BASE_DIR / "static"

from starlette.responses import Response as _Response


@app.middleware("http")
async def add_no_cache_headers(request, call_next):
    response = await call_next(request)
    path = request.url.path
    if path.endswith(('.js', '.css', '.html')) or path == '/' or path == '':
        response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate'
        response.headers['Pragma'] = 'no-cache'
    return response


app.mount("/", StaticFiles(directory=str(static_dir), html=True), name="static")
