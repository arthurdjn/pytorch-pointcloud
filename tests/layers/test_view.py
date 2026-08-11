import torch

from torch_pointcloud.layers.view import View


def test_view_contiguous() -> None:
    layer = View(4, 8)
    x = torch.randn(32).contiguous()
    out = layer(x)
    assert out.shape == (4, 8)


def test_view_repr() -> None:
    layer = View(2, 3)
    assert "(2, 3)" in repr(layer)
