from typing import TYPE_CHECKING, Any, Dict, Literal, Optional

import torch.nn as nn
from torch import Tensor

from torch_pointcloud.utils.imports import optional_import

from ._modules import ModuleLike, RegisteredModuleLike, create_module

if TYPE_CHECKING:
    from torch_scatter import scatter

scatter, _ = optional_import("torch_scatter", "scatter")


class MaxPool(nn.Module):
    def __init__(self, dim: int = 0, dim_size: Optional[int] = None):
        super().__init__()
        self.dim = dim
        self.dim_size = dim_size

    def forward(self, x: Tensor, batch: Tensor) -> Tensor:
        return scatter(x, batch, dim=self.dim, dim_size=self.dim_size, reduce="max")


class MinPool(nn.Module):
    def __init__(self, dim: int = 0, dim_size: Optional[int] = None):
        super().__init__()
        self.dim = dim
        self.dim_size = dim_size

    def forward(self, x: Tensor, batch: Tensor) -> Tensor:
        return scatter(x, batch, dim=self.dim, dim_size=self.dim_size, reduce="min")


class MeanPool(nn.Module):
    def __init__(self, dim: int = 0, dim_size: Optional[int] = None):
        super().__init__()
        self.dim = dim
        self.dim_size = dim_size

    def forward(self, x: Tensor, batch: Tensor) -> Tensor:
        return scatter(x, batch, dim=self.dim, dim_size=self.dim_size, reduce="mean")


class MulPool(nn.Module):
    def __init__(self, dim: int = 0, dim_size: Optional[int] = None):
        super().__init__()
        self.dim = dim
        self.dim_size = dim_size

    def forward(self, x: Tensor, batch: Tensor) -> Tensor:
        return scatter(x, batch, dim=self.dim, dim_size=self.dim_size, reduce="mul")


class SumPool(nn.Module):
    def __init__(self, dim: int = 0, dim_size: Optional[int] = None):
        super().__init__()
        self.dim = dim
        self.dim_size = dim_size

    def forward(self, x: Tensor, batch: Tensor) -> Tensor:
        return scatter(x, batch, dim=self.dim, dim_size=self.dim_size, reduce="sum")


class SoftmaxPool(nn.Module):
    def __init__(self, dim: int = 0, dim_size: Optional[int] = None):
        super().__init__()
        self.dim = dim
        self.dim_size = dim_size

    def forward(self, x: Tensor, batch: Tensor) -> Tensor:
        return scatter(x, batch, dim=self.dim, dim_size=self.dim_size, reduce="softmax")


class LogSoftmaxPool(nn.Module):
    def __init__(self, dim: int = 0, dim_size: Optional[int] = None):
        super().__init__()
        self.dim = dim
        self.dim_size = dim_size

    def forward(self, x: Tensor, batch: Tensor) -> Tensor:
        return scatter(x, batch, dim=self.dim, dim_size=self.dim_size, reduce="log_softmax")


PoolName = Literal[
    "max",
    "min",
    "mean",
    "mul",
    "sum",
    "softmax",
    "log_softmax",
]

PoolLike = ModuleLike[PoolName]

_POOL_REGISTRY: Dict[PoolName, RegisteredModuleLike] = dict(
    max=MaxPool,
    min=MinPool,
    mean=MeanPool,
    mul=MulPool,
    sum=SumPool,
    softmax=SoftmaxPool,
    log_softmax=LogSoftmaxPool,
)


def create_pool(name: PoolLike, *args: Any, **kwargs: Any) -> nn.Module:
    return create_module(name, *args, registry=_POOL_REGISTRY, **kwargs)
