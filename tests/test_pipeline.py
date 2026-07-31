"""Tests for the unwrapping pipeline."""

import tempfile
from pathlib import Path

import numpy as np
import torch

from src.data.mesh_io import normalize_mesh, sample_points
from src.models import FlexParaUnwrapper
from src.pipeline.postprocessor import (
    add_uv_margins,
    export_uv_mesh,
    pack_uv_charts,
)
from src.pipeline.preprocessor import MeshPreprocessor
from src.training.trainer import train_unsupervised


def _make_sphere():
    import trimesh
    return trimesh.creation.icosphere(subdivisions=2, radius=1.0)


def test_preprocessor():
    mesh = _make_sphere()
    preprocessor = MeshPreprocessor(num_points=500)
    data = preprocessor.process(mesh)

    assert data["points"].shape == (1, 500, 3)
    assert data["faces"].dim() == 2
    assert data["faces"].shape[1] == 3


def test_sample_points():
    mesh = _make_sphere()
    points, normals, face_idx = sample_points(mesh, 200)
    assert points.shape == (200, 3)
    assert normals is not None
    assert normals.shape == (200, 3)
    assert face_idx.shape == (200,)


def test_normalize_mesh():
    mesh = _make_sphere()
    mesh.vertices *= 10
    mesh = normalize_mesh(mesh)
    max_val = np.abs(mesh.vertices).max()
    assert max_val <= 1.01


def test_pack_uv_charts():
    uv = np.random.randn(100, 2) * 2  # wide range
    packed = pack_uv_charts(uv)
    assert packed.min() >= 0
    assert packed.max() <= 1


def test_add_uv_margins():
    uv = np.array([[0, 0], [1, 0], [0.5, 1]])
    faces = np.array([[0, 1, 2]])
    padded = add_uv_margins(uv, faces, margin=0.05)
    assert padded.shape == uv.shape
    # Points should have moved outward from centroid
    assert not np.allclose(padded, uv)


def test_export_uv_mesh():
    vertices = np.random.randn(10, 3)
    uv = np.random.rand(10, 2)
    faces = np.array([[0, 1, 2], [3, 4, 5]])

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test.obj"
        export_uv_mesh(path, vertices, uv, faces)
        assert path.exists()
        content = path.read_text()
        assert "v " in content
        assert "vt " in content
        assert "f " in content


def test_train_unsupervised_smoke():
    """Smoke test: run a few optimization steps."""
    model = FlexParaUnwrapper(num_charts=1, hidden_dim=32, num_layers=2)
    points = torch.randn(1, 200, 3)

    history = train_unsupervised(
        model=model,
        points=points,
        num_iterations=10,
        lr=1e-3,
        log_every=5,
    )

    assert len(history["total"]) == 10
    assert all(isinstance(v, float) for v in history["total"])


def test_full_pipeline_smoke():
    """End-to-end smoke test: mesh → preprocess → train → export."""
    mesh = _make_sphere()

    # Preprocess
    preprocessor = MeshPreprocessor(num_points=300)
    data = preprocessor.process(mesh)

    # Model
    model = FlexParaUnwrapper(num_charts=1, hidden_dim=32, num_layers=2)

    # Train (short)
    history = train_unsupervised(
        model=model,
        points=data["points"],
        num_iterations=20,
        lr=1e-3,
        log_every=10,
    )

    # Extract UVs
    with torch.no_grad():
        outputs = model(data["points"])
    uv = outputs["uv_coords"][0].numpy()

    # Export
    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = Path(tmpdir) / "output.obj"
        uv_packed = pack_uv_charts(uv)
        export_uv_mesh(out_path, data["vertices"][0].numpy(), uv_packed, data["faces"].numpy())
        assert out_path.exists()


if __name__ == "__main__":
    test_preprocessor()
    test_sample_points()
    test_normalize_mesh()
    test_pack_uv_charts()
    test_add_uv_margins()
    test_export_uv_mesh()
    test_train_unsupervised_smoke()
    test_full_pipeline_smoke()
    print("All pipeline tests passed!")
