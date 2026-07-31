"""Tests for loss functions."""

import torch

from src.losses.chamfer import chamfer_distance, repulsion_loss
from src.losses.cycle import cycle_consistency_loss
from src.losses.distortion import conformal_loss, distortion_loss, isometric_loss


def test_cycle_consistency_loss():
    points = torch.randn(1, 100, 3)
    recon = points + torch.randn_like(points) * 0.01
    uv = torch.rand(1, 100, 2)
    recon_uv = uv + torch.randn_like(uv) * 0.01

    loss = cycle_consistency_loss(points, recon, uv, recon_uv)
    assert loss.shape == ()
    assert loss.item() > 0


def test_chamfer_distance():
    pred = torch.randn(1, 50, 3)
    target = torch.randn(1, 60, 3)
    dist = chamfer_distance(pred, target)
    assert dist.shape == ()
    assert dist.item() > 0


def test_chamfer_self_distance():
    points = torch.randn(1, 100, 3)
    dist = chamfer_distance(points, points)
    assert dist.item() < 0.01


def test_repulsion_loss():
    # Uniform points should have low repulsion loss
    points = torch.randn(1, 100, 3)
    loss = repulsion_loss(points)
    assert loss.shape == ()
    assert loss.item() >= 0


def test_conformal_loss():
    # Simple square in 3D mapped to square in 2D (conformal)
    vertices = torch.tensor([
        [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
    ], dtype=torch.float32)
    uv = torch.tensor([
        [0, 0], [1, 0], [1, 1], [0, 1],
    ], dtype=torch.float32)
    edges = torch.tensor([[0, 1], [1, 2], [2, 3], [3, 0]])

    loss = conformal_loss(vertices, uv, edges)
    assert loss.shape == ()
    # Should be very low for a conformal mapping
    assert loss.item() < 0.01


def test_isometric_loss():
    vertices = torch.tensor([
        [0, 0, 0], [1, 0, 0], [0.5, 0.866, 0],
    ], dtype=torch.float32)
    uv = torch.tensor([
        [0, 0], [1, 0], [0.5, 0.866],
    ], dtype=torch.float32)
    faces = torch.tensor([[0, 1, 2]])

    loss = isometric_loss(vertices, uv, faces)
    assert loss.shape == ()
    assert loss.item() < 0.01


def test_distortion_loss_combined():
    vertices = torch.tensor([
        [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
    ], dtype=torch.float32)
    uv = torch.tensor([
        [0, 0], [1, 0], [1, 1], [0, 1],
    ], dtype=torch.float32)
    edges = torch.tensor([[0, 1], [1, 2], [2, 3], [3, 0]])
    faces = torch.tensor([[0, 1, 2], [0, 2, 3]])

    loss = distortion_loss(vertices, uv, edges, faces)
    assert loss.shape == ()
    assert loss.item() >= 0


if __name__ == "__main__":
    test_cycle_consistency_loss()
    test_chamfer_distance()
    test_chamfer_self_distance()
    test_repulsion_loss()
    test_conformal_loss()
    test_isometric_loss()
    test_distortion_loss_combined()
    print("All loss tests passed!")
