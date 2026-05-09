import math
from typing import TYPE_CHECKING, Any, Callable, Dict, Literal, Optional, Tuple, Union, overload

import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.nn.resolver import activation_resolver, normalization_resolver

from torch_pointcloud.utils.imports import optional_import

if TYPE_CHECKING:
    import torch_scatter


torch_scatter, _ = optional_import("torch_scatter")


class SerializedPool(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 2,
        bias: bool = True,
        act: Union[str, Callable, None] = None,
        norm: Union[str, Callable, None] = None,
        act_kwargs: Optional[Dict[str, Any]] = None,
        norm_kwargs: Optional[Dict[str, Any]] = None,
        reduce: Literal["sum", "mean", "min", "max"] = "max",
    ):
        super().__init__()
        act_kwargs = act_kwargs or {}
        norm_kwargs = norm_kwargs or {}
        if reduce not in ["sum", "mean", "min", "max"]:
            raise ValueError(f"Invalid reduce operation: {reduce!r}. Must be one of: 'sum', 'mean', 'min', 'max'.")
        if stride != 2 ** (math.ceil(stride) - 1).bit_length():
            raise ValueError(f"Invalid stride: {stride}. Must be a power of 2.")

        self.stride = stride
        self.reduce = reduce
        self.proj = nn.Linear(in_channels, out_channels, bias=bias)
        self.norm = normalization_resolver(norm, out_channels, **norm_kwargs) if norm is not None else None
        self.act = activation_resolver(act, **act_kwargs) if act is not None else None

    @overload
    def forward(
        self,
        x: Tensor,
        pos_grid: Tensor,
        batch: Tensor,
        serialized_code: Tensor,
        return_inverse: Literal[True] = True,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]: ...

    @overload
    def forward(
        self,
        x: Tensor,
        pos_grid: Tensor,
        batch: Tensor,
        serialized_code: Tensor,
        return_inverse: Literal[False] = False,
    ) -> Tuple[Tensor, Tensor, Tensor]: ...

    def forward(
        self,
        x: Tensor,
        pos_grid: Tensor,
        batch: Tensor,
        serialized_code: Tensor,
        return_inverse: bool = False,
    ) -> Tuple[Tensor, ...]:
        pooling_depth = (math.ceil(self.stride) - 1).bit_length()
        pooled_code = serialized_code >> (pooling_depth * 3)
        _, cluster, counts = torch.unique(pooled_code[0], sorted=True, return_inverse=True, return_counts=True)

        # Sort by cluster for segment_csr
        _, indices = torch.sort(cluster)
        idx_ptr = torch.cat([counts.new_zeros(1), torch.cumsum(counts, dim=0)])
        head_indices = indices[idx_ptr[:-1]]

        # Pool features, positions and batch indices
        x = torch_scatter.segment_csr(self.proj(x)[indices], idx_ptr, reduce="max")
        pos_grid = pos_grid[head_indices] >> pooling_depth
        batch = batch[head_indices]
        pooled_code = pooled_code[:, head_indices]

        if self.norm:
            x = self.norm(x)
        if self.act:
            x = self.act(x)

        if return_inverse:
            return x, pos_grid, batch, pooled_code, cluster
        return x, pos_grid, batch, pooled_code


class SerializedUpsample(nn.Module):
    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        norm: Union[str, Callable, None] = None,
        act: Union[str, Callable, None] = None,
        act_kwargs: Optional[Dict[str, Any]] = None,
        norm_kwargs: Optional[Dict[str, Any]] = None,
        bias: bool = True,
    ):
        super().__init__()
        act_kwargs = act_kwargs or {}
        norm_kwargs = norm_kwargs or {}

        self.in_channels = in_channels
        self.skip_channels = skip_channels
        self.out_channels = out_channels
        self.bias = bias

        self.proj = nn.Linear(self.in_channels, self.out_channels, bias=self.bias)
        self.proj_skip = nn.Linear(self.skip_channels, self.out_channels, bias=self.bias)

        self.norm = normalization_resolver(norm, self.out_channels, **norm_kwargs) if norm is not None else None
        self.norm_skip = normalization_resolver(norm, self.out_channels, **norm_kwargs) if norm is not None else None

        self.act = activation_resolver(act, **act_kwargs) if act is not None else None
        self.act_skip = activation_resolver(act, **act_kwargs) if act is not None else None

    def forward(self, x: Tensor, x_skip: Tensor, inverse: Tensor) -> Tensor:
        x = self.proj(x)
        if self.norm is not None:
            x = self.norm(x)
        if self.act is not None:
            x = self.act(x)

        x_skip = self.proj_skip(x_skip)
        if self.norm_skip is not None:
            x_skip = self.norm_skip(x_skip)
        if self.act_skip is not None:
            x_skip = self.act_skip(x_skip)

        return x_skip + x[inverse]
