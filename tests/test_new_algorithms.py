"""Tests for new UV unwrapping algorithms."""

import numpy as np
import pytest


def _make_sphere():
    import trimesh
    return trimesh.creation.icosphere(subdivisions=2, radius=1.0)


def _make_cube():
    import trimesh
    return trimesh.creation.box(extents=[1, 1, 1])


def _check_result(result, mesh):
    assert "uv_coords" in result
    assert "vertices" in result
    assert "faces" in result
    V = len(mesh.vertices)
    assert result["uv_coords"].shape[1] == 2
    assert result["uv_coords"].shape[0] >= V
    assert result["faces"].shape[1] == 3
    assert not np.any(np.isnan(result["uv_coords"]))
    assert not np.any(np.isinf(result["uv_coords"]))


# ── Classical new methods ──────────────────────────────────────────


def test_arap():
    from src.pipeline.classical_unwrapper import ClassicalUnwrapper
    mesh = _make_sphere()
    result = ClassicalUnwrapper(method="arap").unwrap(mesh)
    _check_result(result, mesh)


def test_harmonic():
    from src.pipeline.classical_unwrapper import ClassicalUnwrapper
    mesh = _make_sphere()
    result = ClassicalUnwrapper(method="harmonic").unwrap(mesh)
    _check_result(result, mesh)


def test_conformal():
    from src.pipeline.classical_unwrapper import ClassicalUnwrapper
    mesh = _make_sphere()
    result = ClassicalUnwrapper(method="conformal").unwrap(mesh)
    _check_result(result, mesh)


def test_graph_cuts():
    from src.pipeline.classical_unwrapper import ClassicalUnwrapper
    mesh = _make_sphere()
    result = ClassicalUnwrapper(method="graph_cuts").unwrap(mesh)
    _check_result(result, mesh)


def test_hilbert():
    from src.pipeline.classical_unwrapper import ClassicalUnwrapper
    mesh = _make_sphere()
    result = ClassicalUnwrapper(method="hilbert").unwrap(mesh)
    _check_result(result, mesh)


def test_hilbert_cube():
    from src.pipeline.classical_unwrapper import ClassicalUnwrapper
    mesh = _make_cube()
    result = ClassicalUnwrapper(method="hilbert").unwrap(mesh)
    _check_result(result, mesh)


# ── Research methods ───────────────────────────────────────────────


def test_voronoi_disks():
    from src.pipeline.research_unwrappers import ResearchUnwrapper
    mesh = _make_sphere()
    result = ResearchUnwrapper(method="voronoi_disks").unwrap(mesh)
    _check_result(result, mesh)
    assert result["num_charts"] >= 1


def test_instant_meshes():
    from src.pipeline.research_unwrappers import ResearchUnwrapper
    mesh = _make_sphere()
    result = ResearchUnwrapper(method="instant_meshes").unwrap(mesh)
    _check_result(result, mesh)


# ── Pipeline integration ───────────────────────────────────────────


def test_pipeline_arap():
    from src.pipeline.unwrapper import UVUnwrapPipeline
    mesh = _make_sphere()
    pipeline = UVUnwrapPipeline(mode="arap")
    result = pipeline.unwrap(mesh)
    _check_result(result, mesh)


def test_pipeline_harmonic():
    from src.pipeline.unwrapper import UVUnwrapPipeline
    mesh = _make_sphere()
    pipeline = UVUnwrapPipeline(mode="harmonic")
    result = pipeline.unwrap(mesh)
    _check_result(result, mesh)


def test_pipeline_conformal():
    from src.pipeline.unwrapper import UVUnwrapPipeline
    mesh = _make_sphere()
    pipeline = UVUnwrapPipeline(mode="conformal")
    result = pipeline.unwrap(mesh)
    _check_result(result, mesh)


def test_pipeline_graph_cuts():
    from src.pipeline.unwrapper import UVUnwrapPipeline
    mesh = _make_sphere()
    pipeline = UVUnwrapPipeline(mode="graph_cuts")
    result = pipeline.unwrap(mesh)
    _check_result(result, mesh)


def test_pipeline_hilbert():
    from src.pipeline.unwrapper import UVUnwrapPipeline
    mesh = _make_sphere()
    pipeline = UVUnwrapPipeline(mode="hilbert")
    result = pipeline.unwrap(mesh)
    _check_result(result, mesh)


def test_pipeline_voronoi_disks():
    from src.pipeline.unwrapper import UVUnwrapPipeline
    mesh = _make_sphere()
    pipeline = UVUnwrapPipeline(mode="voronoi_disks")
    result = pipeline.unwrap(mesh)
    _check_result(result, mesh)


def test_pipeline_instant_meshes():
    from src.pipeline.unwrapper import UVUnwrapPipeline
    mesh = _make_sphere()
    pipeline = UVUnwrapPipeline(mode="instant_meshes")
    result = pipeline.unwrap(mesh)
    _check_result(result, mesh)


# ── Existing methods still work ────────────────────────────────────


def test_pipeline_classical_still_works():
    from src.pipeline.unwrapper import UVUnwrapPipeline
    mesh = _make_sphere()
    pipeline = UVUnwrapPipeline(mode="classical", classical_method="xatlas")
    result = pipeline.unwrap(mesh)
    _check_result(result, mesh)


def test_pipeline_xatlas():
    from src.pipeline.classical_unwrapper import ClassicalUnwrapper
    mesh = _make_sphere()
    result = ClassicalUnwrapper(method="xatlas").unwrap(mesh)
    _check_result(result, mesh)


def test_pipeline_lscm():
    from src.pipeline.classical_unwrapper import ClassicalUnwrapper
    mesh = _make_sphere()
    result = ClassicalUnwrapper(method="lscm").unwrap(mesh)
    _check_result(result, mesh)


# ── FFHQ-UV methods ──────────────────────────────────────────────


def test_ffhq_uv_face_auto():
    from src.pipeline.ffhq_uv_unwrapper import FFHQUVUnwrapper
    mesh = _make_cube()
    result = FFHQUVUnwrapper(method="face_auto").unwrap(mesh)
    _check_result(result, mesh)


def test_ffhq_uv_multi_view():
    from src.pipeline.ffhq_uv_unwrapper import FFHQUVUnwrapper
    mesh = _make_cube()
    result = FFHQUVUnwrapper(method="multi_view").unwrap(mesh, num_views=6)
    _check_result(result, mesh)


def test_ffhq_uv_rgb_fitting():
    from src.pipeline.ffhq_uv_unwrapper import FFHQUVUnwrapper
    mesh = _make_cube()
    result = FFHQUVUnwrapper(method="rgb_fitting").unwrap(mesh)
    _check_result(result, mesh)


def test_ffhq_uv_topo_transfer_fallback():
    from src.pipeline.ffhq_uv_unwrapper import FFHQUVUnwrapper
    mesh = _make_cube()
    result = FFHQUVUnwrapper(method="topo_transfer").unwrap(mesh)
    _check_result(result, mesh)
