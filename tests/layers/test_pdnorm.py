import pytest
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
    inner = norm.norm
    assert isinstance(inner, nn.ModuleList)
    assert inner[0] is inner[norm.conditions.index("A")]
    assert inner[1] is inner[norm.conditions.index("B")]

    norm.train()
    x_a = torch.randn(64, 16) * 5.0 + 3.0
    x_b = torch.randn(64, 16) * 0.1 - 2.0
    norm(x_a, condition="A")
    norm(x_b, condition="B")

    norm_a, norm_b = inner[0], inner[1]
    assert isinstance(norm_a, nn.BatchNorm1d) and isinstance(norm_b, nn.BatchNorm1d)
    assert norm_a.running_mean is not None and norm_b.running_mean is not None
    assert not torch.allclose(norm_a.running_mean, norm_b.running_mean)


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


def test_pdnorm_shared_accepts_none_condition() -> None:
    norm = PDNorm(16, conditions=["A", "B"], norm=nn.BatchNorm1d, decouple=False)
    norm.eval()
    x = torch.randn(8, 16)
    assert torch.equal(norm(x), norm(x, condition="A"))


def test_pdnorm_decoupled_missing_condition_raises() -> None:
    norm = PDNorm(16, conditions=["A", "B"], norm=nn.BatchNorm1d)
    x = torch.randn(8, 16)
    with pytest.raises(ValueError, match="PDNorm requires a condition when decoupled. Valid conditions are: 'A', 'B'."):
        norm(x)


def test_pdnorm_unknown_condition_raises() -> None:
    norm = PDNorm(16, conditions=["A", "B"], norm=nn.BatchNorm1d)
    x = torch.randn(8, 16)
    with pytest.raises(ValueError, match="Unknown condition 'C'. Valid conditions are: 'A', 'B'."):
        norm(x, condition="C")
