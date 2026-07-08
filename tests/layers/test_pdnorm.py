import torch
import torch.nn as nn

from torch_pointcloud.layers.pdnorm import PDNorm


def test_pdnorm_forward_shape() -> None:
    norm = PDNorm(16, conditions=["A", "B"], norm=nn.BatchNorm1d)
    x = torch.randn(32, 16)
    out = norm(x, condition="A")
    assert out.shape == (32, 16)


def test_pdnorm_decoupled_key_layout() -> None:
    norm = PDNorm(16, conditions=["A", "B", "C"], norm=nn.BatchNorm1d)
    assert isinstance(norm.norm, nn.ModuleList)
    assert len(norm.norm) == 3
    keys = dict(norm.named_parameters())
    assert "norm.0.weight" in keys
    assert "norm.1.weight" in keys
    assert "norm.2.weight" in keys


def test_pdnorm_condition_selects_inner_norm() -> None:
    norm = PDNorm(16, conditions=["A", "B"], norm=nn.BatchNorm1d)
    assert norm.norm[0] is norm.norm[norm.conditions.index("A")]
    assert norm.norm[1] is norm.norm[norm.conditions.index("B")]

    norm.train()
    x_a = torch.randn(64, 16) * 5.0 + 3.0
    x_b = torch.randn(64, 16) * 0.1 - 2.0
    norm(x_a, condition="A")
    norm(x_b, condition="B")

    mean_a = norm.norm[0].running_mean
    mean_b = norm.norm[1].running_mean
    assert not torch.allclose(mean_a, mean_b)


def test_pdnorm_shared_key_layout() -> None:
    norm = PDNorm(16, conditions=["A", "B"], norm=nn.BatchNorm1d, decouple=False)
    assert not isinstance(norm.norm, nn.ModuleList)
    keys = dict(norm.named_parameters())
    assert "norm.weight" in keys
    assert "norm.0.weight" not in keys


def test_pdnorm_shared_ignores_condition() -> None:
    norm = PDNorm(16, conditions=["A", "B"], norm=nn.BatchNorm1d, decouple=False)
    norm.eval()
    x = torch.randn(8, 16)
    out_a = norm(x, condition="A")
    out_b = norm(x, condition="B")
    assert torch.equal(out_a, out_b)
