"""Tests for the UV unwrapping model."""

import torch

from src.models import FlexParaUnwrapper
from src.models.networks import CutNet, DeformNet, UnwrapNet, WrapNet
from src.models.networks.positional_encoding import PositionalEncoding


def test_positional_encoding():
    pe = PositionalEncoding(num_freqs=6, include_input=True)
    x = torch.randn(10, 3)
    out = pe(x)
    # 3 input dims + 3*6*2 encoded dims = 39
    assert out.shape == (10, 39)


def test_cut_net():
    net = CutNet(point_dim=3, hidden_dim=64, num_layers=4)
    x = torch.randn(2, 100, 3)
    out = net(x)
    assert out.shape == (2, 100, 1)


def test_deform_net():
    net = DeformNet(in_dim=2, hidden_dim=64, num_layers=4)
    x = torch.randn(2, 100, 2)
    out = net(x)
    assert out.shape == (2, 100, 2)


def test_unwrap_net():
    net = UnwrapNet(point_dim=3, uv_dim=2, hidden_dim=64, num_layers=4)
    x = torch.randn(2, 100, 3)
    out = net(x)
    assert out.shape == (2, 100, 2)


def test_wrap_net():
    net = WrapNet(uv_dim=2, point_dim=3, hidden_dim=64, num_layers=4)
    x = torch.randn(2, 100, 2)
    out = net(x)
    assert out.shape == (2, 100, 3)


def test_flexpara_single_chart():
    model = FlexParaUnwrapper(num_charts=1, hidden_dim=64, num_layers=4)
    points = torch.randn(1, 500, 3)
    outputs = model(points)

    assert outputs["uv_coords"].shape == (1, 500, 2)
    assert outputs["seam_logits"].shape == (1, 500, 1)
    assert outputs["reconstructed_3d"].shape == (1, 500, 3)
    assert outputs["reconstructed_uv"].shape == (1, 500, 2)
    assert "chart_probs" not in outputs  # single chart


def test_flexpara_multi_chart():
    model = FlexParaUnwrapper(num_charts=4, hidden_dim=64, num_layers=4)
    points = torch.randn(1, 500, 3)
    outputs = model(points)

    assert outputs["uv_coords"].shape == (1, 500, 2)
    assert outputs["chart_probs"].shape == (1, 500, 4)


def test_flexpara_losses():
    model = FlexParaUnwrapper(num_charts=1, hidden_dim=64, num_layers=4)
    points = torch.randn(1, 200, 3)
    outputs = model(points)
    losses = model.compute_losses(points, outputs)

    assert "total" in losses
    assert "cycle" in losses
    assert "chamfer" in losses
    assert losses["total"].requires_grad


def test_flexpara_unwrap():
    model = FlexParaUnwrapper(num_charts=1, hidden_dim=64, num_layers=4)
    points = torch.randn(1, 100, 3)
    uv = model.unwrap(points)
    assert uv.shape == (1, 100, 2)


if __name__ == "__main__":
    test_positional_encoding()
    test_cut_net()
    test_deform_net()
    test_unwrap_net()
    test_wrap_net()
    test_flexpara_single_chart()
    test_flexpara_multi_chart()
    test_flexpara_losses()
    test_flexpara_unwrap()
    print("All tests passed!")
