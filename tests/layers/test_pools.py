import pytest
import torch
import torch.nn as nn

from torch_pointcloud.layers.pools import (
    CatPool,
    MaxPool,
    MeanPool,
    MinPool,
    MulPool,
    SumPool,
    create_adaptive_pool,
    create_pool,
)
from torch_pointcloud.utils.imports import _TORCH_SCATTER_AVAILABLE

# See: https://docs.pytest.org/en/stable/how-to/skipping.html#summary
pytestmark = pytest.mark.skipif(
    not _TORCH_SCATTER_AVAILABLE,
    reason="torch-scatter is not installed",
)


# softmax / log_softmax pools are registered but not supported by torch_scatter at runtime.
_POOL_CLASSES = [MaxPool, MinPool, MeanPool, MulPool, SumPool]
_POOL_NAMES = ["max", "min", "mean", "mul", "sum"]


@pytest.mark.parametrize("cls", _POOL_CLASSES)
def test_pool_forward(cls: type) -> None:
    pool = cls(dim=0, dim_size=None)
    x = torch.randn(10, 4)
    batch = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
    out = pool(x, batch)
    assert out.shape[0] == 2
    assert out.shape[1] == 4


@pytest.mark.parametrize("name", _POOL_NAMES)
def test_create_pool_by_name(name: str) -> None:
    pool = create_pool(name)
    assert isinstance(pool, nn.Module)


def test_cat_pool() -> None:
    pool = CatPool(pools=("max", "mean"), dim=0, dim_size=None)
    assert pool.num_pools == 2
    x = torch.randn(10, 4)
    batch = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
    out = pool(x, batch)
    assert out.shape == (2, 4 * 2)


@pytest.mark.parametrize("name,expected_cls", [("mean", nn.AdaptiveAvgPool1d), ("max", nn.AdaptiveMaxPool1d)])
def test_create_adaptive_pool(name: str, expected_cls: type) -> None:
    pool = create_adaptive_pool(name)
    assert isinstance(pool, expected_cls)
