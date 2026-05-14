import torch

from torch_pointcloud.layers.affine import Affine, affine


def test_affine_function_with_bias() -> None:
    x = torch.randn(4, 8)
    w = torch.randn(8)
    b = torch.randn(8)
    out = affine(x, w, b)
    torch.testing.assert_close(out, x * w + b)


def test_affine_function_without_bias() -> None:
    x = torch.randn(4, 8)
    w = torch.randn(8)
    out = affine(x, w, None)
    torch.testing.assert_close(out, x * w)


def test_affine_module_with_bias() -> None:
    layer = Affine(num_features=8, bias=True)
    assert layer.bias is not None
    x = torch.randn(4, 8)
    out = layer(x)
    assert out.shape == x.shape
    torch.testing.assert_close(out, x)


def test_affine_module_without_bias() -> None:
    layer = Affine(num_features=8, bias=False)
    assert layer.bias is None
    x = torch.randn(4, 8)
    out = layer(x)
    assert out.shape == x.shape


def test_affine_reset_parameters() -> None:
    layer = Affine(num_features=8, bias=True)
    layer.weight.data.fill_(2.0)
    layer.bias.data.fill_(3.0)
    layer.reset_parameters()
    torch.testing.assert_close(layer.weight, torch.ones(8))
    torch.testing.assert_close(layer.bias, torch.zeros(8))
