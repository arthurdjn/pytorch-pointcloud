import pytest
import torch
import torch.nn as nn
from torch import Tensor

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


@pytest.mark.parametrize(
    "cls,expected",
    [
        pytest.param(MaxPool, torch.tensor([[3.0, 2.0], [2.0, 5.0]]), id="max"),
        pytest.param(MinPool, torch.tensor([[1.0, 0.0], [2.0, 5.0]]), id="min"),
        pytest.param(MeanPool, torch.tensor([[2.0, 1.0], [2.0, 5.0]]), id="mean"),
        pytest.param(MulPool, torch.tensor([[3.0, 0.0], [2.0, 5.0]]), id="mul"),
        pytest.param(SumPool, torch.tensor([[4.0, 2.0], [2.0, 5.0]]), id="sum"),
    ],
)
def test_pool_forward_values(cls: type, expected: Tensor) -> None:
    """Hand-computed reductions over segments {rows 0, 1} and {row 2}."""
    pool = cls(dim=0, dim_size=None)
    x = torch.tensor([[1.0, 2.0], [3.0, 0.0], [2.0, 5.0]])
    batch = torch.tensor([0, 0, 1])
    torch.testing.assert_close(pool(x, batch), expected)


@pytest.mark.parametrize("name", _POOL_NAMES)
def test_create_pool_by_name(name: str) -> None:
    pool = create_pool(name)  # type: ignore[arg-type]
    assert isinstance(pool, nn.Module)


def test_create_pool_unknown_name_raises() -> None:
    with pytest.raises(ValueError, match="Available modules"):
        create_pool("median")  # type: ignore[arg-type]


def test_cat_pool() -> None:
    pool = CatPool(pools=("max", "mean"), dim=0, dim_size=None)
    assert pool.num_pools == 2
    x = torch.randn(10, 4)
    batch = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
    out = pool(x, batch)
    assert out.shape == (2, 4 * 2)


@pytest.mark.parametrize("name,expected_cls", [("mean", nn.AdaptiveAvgPool1d), ("max", nn.AdaptiveMaxPool1d)])
def test_create_adaptive_pool(name: str, expected_cls: type) -> None:
    pool = create_adaptive_pool(name)  # type: ignore[arg-type]
    assert isinstance(pool, expected_cls)


def test_create_adaptive_pool_defaults_to_global_output() -> None:
    pool = create_adaptive_pool("mean")
    assert isinstance(pool, nn.AdaptiveAvgPool1d)
    assert pool.output_size == 1
