import pytest
import torch

from torch_pointcloud.layers.radius_max_pool import RadiusMaxPool
from torch_pointcloud.utils.imports import _PYG_LIB_AVAILABLE


@pytest.mark.skipif(not _PYG_LIB_AVAILABLE, reason="pyg-lib is not installed")
def test_radius_max_pool_takes_ball_maximum() -> None:
    torch.manual_seed(0)
    pos = torch.rand(80, 3)
    x = torch.randn(80, 4)
    batch = torch.arange(2).repeat_interleave(40)
    pool = RadiusMaxPool(num_points=8, radius=0.3, max_num_neighbors=40)
    out, pos_out, batch_out = pool(x, pos, batch)
    assert out.shape == (16, 4) and pos_out.shape == (16, 3)
    assert torch.equal(batch_out, torch.arange(2).repeat_interleave(8))
    for i in range(16):
        inside = (batch == batch_out[i]) & ((pos - pos_out[i]).norm(dim=1) <= 0.3)
        assert torch.allclose(out[i], x[inside].amax(dim=0))


@pytest.mark.skipif(not _PYG_LIB_AVAILABLE, reason="pyg-lib is not installed")
def test_radius_max_pool_caps_ball_to_smallest_indices() -> None:
    pos = torch.zeros(6, 3)
    x = torch.arange(6, dtype=torch.float).unsqueeze(1)
    batch = torch.tensor([0, 0, 0, 0, 1, 1])
    pool = RadiusMaxPool(num_points=1, radius=1.0, max_num_neighbors=2)
    out, pos_out, batch_out = pool(x, pos, batch)
    assert out.tolist() == [[1.0], [5.0]]
    assert pos_out.shape == (2, 3) and batch_out.tolist() == [0, 1]
