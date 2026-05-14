import pytest
import torch

from torch_pointcloud.layers.grid_pool import GridPool
from torch_pointcloud.utils.imports import _TORCH_SCATTER_AVAILABLE

# See: https://docs.pytest.org/en/stable/how-to/skipping.html#summary
pytestmark = pytest.mark.skipif(
    not _TORCH_SCATTER_AVAILABLE,
    reason="torch-scatter is not installed",
)


def test_grid_pool_basic() -> None:
    pool = GridPool(
        in_channels=16,
        out_channels=32,
        stride=2,
        bias=True,
        act="relu",
        act_kwargs=None,
        act_first=False,
        norm="batch_norm",
        norm_kwargs=None,
        reduce="max",
    )
    torch.manual_seed(0)
    x = torch.randn(100, 16)
    pos_grid = torch.randint(0, 16, (100, 3))
    batch = torch.cat([torch.zeros(40), torch.ones(60)]).long()
    x_pooled, pos_grid_pooled, batch_pooled, cluster, pos_pooled = pool(x, pos_grid, batch)

    assert x_pooled.shape[1] == 32
    assert x_pooled.shape[0] == pos_grid_pooled.shape[0] == batch_pooled.shape[0]
    assert pos_grid_pooled.shape[1] == 3
    assert cluster.shape == (100,)
    assert pos_pooled is None


def test_grid_pool_with_pos() -> None:
    pool = GridPool(in_channels=16, out_channels=32, stride=2, reduce="mean")
    torch.manual_seed(0)
    x = torch.randn(100, 16)
    pos = torch.randn(100, 3)
    pos_grid = torch.randint(0, 16, (100, 3))
    batch = torch.zeros(100, dtype=torch.long)
    _, _, _, _, pos_pooled = pool(x, pos_grid, batch, pos=pos)
    assert pos_pooled is not None
    assert pos_pooled.shape[1] == 3


def test_grid_pool_invalid_reduce_raises() -> None:
    with pytest.raises(ValueError, match="Invalid reduce"):
        GridPool(in_channels=4, out_channels=8, reduce="foo")
