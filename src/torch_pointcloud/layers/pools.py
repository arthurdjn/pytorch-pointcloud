"""Per-segment pooling modules over packed batches and the `create_pool` / `create_adaptive_pool` factories."""

from typing import TYPE_CHECKING, Any, Dict, Literal, Optional, Sequence

import torch
import torch.nn as nn
from torch import Tensor

from torch_pointcloud.utils.imports import _TORCH_SCATTER_GITHUB_URL, optional_import

from ._modules import ModuleLike, RegisteredModuleLike, create_module

if TYPE_CHECKING:
    from torch_scatter import scatter

scatter, _ = optional_import("torch_scatter", "scatter", url=_TORCH_SCATTER_GITHUB_URL)


class MaxPool(nn.Module):
    r"""Per-segment max pooling over a packed batch, via `torch_scatter.scatter(reduce="max")`.

    Args:
        dim: Dimension along which to pool.
        dim_size: Number of output segments $B$. `None` infers it from the segment index.

    Shape:
        Input: $(N, C)$ features `x` and a $(N,)$ segment index `batch`.
        Output: $(B, C)$ pooled features.
    """

    def __init__(self, dim: int = 0, dim_size: Optional[int] = None):
        super().__init__()
        self.dim = dim
        self.dim_size = dim_size

    def forward(self, x: Tensor, batch: Tensor) -> Tensor:
        return scatter(x, batch, dim=self.dim, dim_size=self.dim_size, reduce="max")


class MinPool(nn.Module):
    r"""Per-segment min pooling over a packed batch, via `torch_scatter.scatter(reduce="min")`.

    Args:
        dim: Dimension along which to pool.
        dim_size: Number of output segments $B$. `None` infers it from the segment index.

    Shape:
        Input: $(N, C)$ features `x` and a $(N,)$ segment index `batch`.
        Output: $(B, C)$ pooled features.
    """

    def __init__(self, dim: int = 0, dim_size: Optional[int] = None):
        super().__init__()
        self.dim = dim
        self.dim_size = dim_size

    def forward(self, x: Tensor, batch: Tensor) -> Tensor:
        return scatter(x, batch, dim=self.dim, dim_size=self.dim_size, reduce="min")


class MeanPool(nn.Module):
    r"""Per-segment mean pooling over a packed batch, via `torch_scatter.scatter(reduce="mean")`.

    Args:
        dim: Dimension along which to pool.
        dim_size: Number of output segments $B$. `None` infers it from the segment index.

    Shape:
        Input: $(N, C)$ features `x` and a $(N,)$ segment index `batch`.
        Output: $(B, C)$ pooled features.
    """

    def __init__(self, dim: int = 0, dim_size: Optional[int] = None):
        super().__init__()
        self.dim = dim
        self.dim_size = dim_size

    def forward(self, x: Tensor, batch: Tensor) -> Tensor:
        return scatter(x, batch, dim=self.dim, dim_size=self.dim_size, reduce="mean")


class MulPool(nn.Module):
    r"""Per-segment product pooling over a packed batch, via `torch_scatter.scatter(reduce="mul")`.

    Args:
        dim: Dimension along which to pool.
        dim_size: Number of output segments $B$. `None` infers it from the segment index.

    Shape:
        Input: $(N, C)$ features `x` and a $(N,)$ segment index `batch`.
        Output: $(B, C)$ pooled features.
    """

    def __init__(self, dim: int = 0, dim_size: Optional[int] = None):
        super().__init__()
        self.dim = dim
        self.dim_size = dim_size

    def forward(self, x: Tensor, batch: Tensor) -> Tensor:
        return scatter(x, batch, dim=self.dim, dim_size=self.dim_size, reduce="mul")


class SumPool(nn.Module):
    r"""Per-segment sum pooling over a packed batch, via `torch_scatter.scatter(reduce="sum")`.

    Args:
        dim: Dimension along which to pool.
        dim_size: Number of output segments $B$. `None` infers it from the segment index.

    Shape:
        Input: $(N, C)$ features `x` and a $(N,)$ segment index `batch`.
        Output: $(B, C)$ pooled features.
    """

    def __init__(self, dim: int = 0, dim_size: Optional[int] = None):
        super().__init__()
        self.dim = dim
        self.dim_size = dim_size

    def forward(self, x: Tensor, batch: Tensor) -> Tensor:
        return scatter(x, batch, dim=self.dim, dim_size=self.dim_size, reduce="sum")


class SoftmaxPool(nn.Module):
    r"""Per-segment softmax pooling, delegating `reduce="softmax"` to `torch_scatter.scatter`.

    !!! warning
        `torch_scatter.scatter` only accepts `sum` / `mean` / `min` / `max` / `mul` reductions,
        so calling this module raises `ValueError` with current `torch_scatter` releases.

    Args:
        dim: Dimension along which to pool.
        dim_size: Number of output segments $B$. `None` infers it from the segment index.
    """

    def __init__(self, dim: int = 0, dim_size: Optional[int] = None):
        super().__init__()
        self.dim = dim
        self.dim_size = dim_size

    def forward(self, x: Tensor, batch: Tensor) -> Tensor:
        return scatter(x, batch, dim=self.dim, dim_size=self.dim_size, reduce="softmax")


class LogSoftmaxPool(nn.Module):
    r"""Per-segment log-softmax pooling, delegating `reduce="log_softmax"` to `torch_scatter.scatter`.

    !!! warning
        `torch_scatter.scatter` only accepts `sum` / `mean` / `min` / `max` / `mul` reductions,
        so calling this module raises `ValueError` with current `torch_scatter` releases.

    Args:
        dim: Dimension along which to pool.
        dim_size: Number of output segments $B$. `None` infers it from the segment index.
    """

    def __init__(self, dim: int = 0, dim_size: Optional[int] = None):
        super().__init__()
        self.dim = dim
        self.dim_size = dim_size

    def forward(self, x: Tensor, batch: Tensor) -> Tensor:
        return scatter(x, batch, dim=self.dim, dim_size=self.dim_size, reduce="log_softmax")


class CatPool(nn.Module):
    r"""Runs several pools on the same input and concatenates their outputs along the feature dim.

    Args:
        pools: Pools to combine, each resolved by `create_pool` (a name, class, or instance).
        dim: Dimension along which each pool reduces.
        dim_size: Number of output segments $B$. `None` infers it from the segment index.

    Shape:
        Input: $(N, C)$ features `x` and a $(N,)$ segment index `batch`.
        Output: $(B, C \cdot P)$ where $P$ is the number of pools.

    Example:
        ```{.python notest}
        import torch
        from torch_pointcloud.layers import CatPool

        pool = CatPool(pools=("max", "mean"))
        x = torch.randn(6, 4)
        batch = torch.tensor([0, 0, 0, 1, 1, 1])
        out = pool(x, batch)  # (2, 8)
        ```
    """

    def __init__(self, pools: Sequence["PoolLike"] = ("max", "mean"), dim: int = 0, dim_size: Optional[int] = None):
        super().__init__()
        self.pools = nn.ModuleList([create_pool(p, dim=dim, dim_size=dim_size) for p in pools])

    @property
    def num_pools(self) -> int:
        r"""Number of pools $P$ concatenated, i.e. the feature multiplier."""
        return len(self.pools)

    def forward(self, x: Tensor, batch: Tensor) -> Tensor:
        return torch.cat([pool(x, batch) for pool in self.pools], dim=-1)


PoolName = Literal[
    "max",
    "min",
    "mean",
    "mul",
    "sum",
    "softmax",
    "log_softmax",
]

AdaptivePoolName = Literal[
    "mean",
    "max",
]

PoolLike = ModuleLike[PoolName]
AdaptivePoolLike = ModuleLike[AdaptivePoolName]

_POOL_REGISTRY: Dict[PoolName, RegisteredModuleLike] = dict(
    max=MaxPool,
    min=MinPool,
    mean=MeanPool,
    mul=MulPool,
    sum=SumPool,
    softmax=SoftmaxPool,
    log_softmax=LogSoftmaxPool,
)

_ADAPTIVE_POOL_REGISTRY: Dict[AdaptivePoolName, RegisteredModuleLike] = dict(
    mean=nn.AdaptiveAvgPool1d,
    max=nn.AdaptiveMaxPool1d,
)


def create_pool(name: PoolLike, *args: Any, **kwargs: Any) -> nn.Module:
    """Resolve a packed-batch pooling module from a name, class, or instance.

    Args:
        name: Pool name (`"max"`, `"min"`, `"mean"`, `"mul"`, `"sum"`, `"softmax"`,
            `"log_softmax"`), a module class, or an existing instance (returned as-is).
        *args: Positional arguments forwarded to the pool constructor.
        **kwargs: Keyword arguments forwarded to the pool constructor (`dim`, `dim_size`).

    Returns:
        The instantiated pooling module.
    """
    return create_module(name, *args, registry=_POOL_REGISTRY, **kwargs)


def create_adaptive_pool(name: AdaptivePoolLike, *args: Any, **kwargs: Any) -> nn.Module:
    """Resolve a dense adaptive pooling module (`nn.AdaptiveAvgPool1d` / `nn.AdaptiveMaxPool1d`).

    Args:
        name: Pool name (`"mean"`, `"max"`), a module class, or an existing instance
            (returned as-is).
        *args: Positional arguments forwarded to the pool constructor.
        **kwargs: Keyword arguments forwarded to the pool constructor. `output_size`
            defaults to `1` (global pooling).

    Returns:
        The instantiated pooling module.
    """
    kwargs.setdefault("output_size", 1)
    return create_module(name, *args, registry=_ADAPTIVE_POOL_REGISTRY, **kwargs)
