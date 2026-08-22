"""Pooling and upsampling driven by point cloud serialization codes."""

import math
from typing import TYPE_CHECKING, Any, Callable, Dict, Literal, Optional, Tuple, Union, overload

import torch
import torch.nn as nn
from torch import Tensor

from torch_pointcloud.utils.imports import _TORCH_SCATTER_GITHUB_URL, optional_import

from .act import create_act
from .norms import create_norm

if TYPE_CHECKING:
    import torch_scatter


torch_scatter, _ = optional_import("torch_scatter", url=_TORCH_SCATTER_GITHUB_URL)


class SerializedPool(nn.Module):
    r"""Grid pooling driven by the serialization code: points sharing a coarser code are reduced to one.

    Truncating the serialization code by $3 \cdot \log_2(\text{stride})$ bits is exactly a grid subsampling,
    so no neighbor search is needed. Features are projected then reduced, positions are averaged, and the
    coarsened code is returned so the next stage can pool again.

    Args:
        in_channels: Input channel count.
        out_channels: Output channel count.
        stride: Downsampling factor along each axis. Must be a power of $2$.
        bias: Whether the projection has a bias term.
        act: Activation, name resolved by `create_act`. `None` disables.
        norm: Normalization, name resolved by `create_norm`. `None` disables.
        act_kwargs: Extra kwargs for the activation.
        norm_kwargs: Extra kwargs for the normalization.
        reduce: Reduction applied within a cell: `"sum"`, `"mean"`, `"min"`, or `"max"`.
    """

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
        self.norm: Optional[nn.Module] = create_norm(norm, out_channels, **norm_kwargs)
        self.act = create_act(act, **act_kwargs)

    @overload
    def forward(
        self,
        x: Tensor,
        pos_grid: Tensor,
        batch: Tensor,
        serialized_code: Tensor,
        return_inverse: Literal[True],
        pos: Optional[Tensor] = None,
        condition: Optional[str] = None,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Optional[Tensor]]: ...

    @overload
    def forward(
        self,
        x: Tensor,
        pos_grid: Tensor,
        batch: Tensor,
        serialized_code: Tensor,
        return_inverse: Literal[False] = False,
        pos: Optional[Tensor] = None,
        condition: Optional[str] = None,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Optional[Tensor]]: ...

    def forward(
        self,
        x: Tensor,
        pos_grid: Tensor,
        batch: Tensor,
        serialized_code: Tensor,
        return_inverse: bool = False,
        pos: Optional[Tensor] = None,
        condition: Optional[str] = None,
    ) -> Tuple[Any, ...]:
        pooling_depth = (math.ceil(self.stride) - 1).bit_length()
        pooled_code = serialized_code >> (pooling_depth * 3)
        _, cluster, counts = torch.unique(pooled_code[0], sorted=True, return_inverse=True, return_counts=True)

        # Sort by cluster for segment_csr
        _, indices = torch.sort(cluster)
        idx_ptr = torch.cat([counts.new_zeros(1), torch.cumsum(counts, dim=0)])
        head_indices = indices[idx_ptr[:-1]]

        # Pool features, positions and batch indices
        x = torch_scatter.segment_csr(self.proj(x)[indices], idx_ptr, reduce=self.reduce)
        pos_grid = pos_grid[head_indices] >> pooling_depth
        batch = batch[head_indices]
        pooled_code = pooled_code[:, head_indices]
        pos = torch_scatter.segment_csr(pos[indices], idx_ptr, reduce="mean") if pos is not None else None

        norm_kwargs = {} if condition is None else {"condition": condition}
        if self.norm is not None:
            x = self.norm(x, **norm_kwargs)
        if self.act:
            x = self.act(x)

        if return_inverse:
            return x, pos_grid, batch, pooled_code, cluster, pos
        return x, pos_grid, batch, pooled_code, pos


class SerializedUpsample(nn.Module):
    r"""Undoes a `SerializedPool` step: scatters the pooled features back and adds the projected skip.

    Both branches get their own projection, normalization and activation before being summed.

    Args:
        in_channels: Channel count of the pooled features.
        skip_channels: Channel count of the skip connection.
        out_channels: Output channel count.
        norm: Normalization, name resolved by `create_norm`. `None` disables.
        act: Activation, name resolved by `create_act`. `None` disables.
        act_kwargs: Extra kwargs for the activation.
        norm_kwargs: Extra kwargs for the normalization.
        bias: Whether the projections have a bias term.
    """

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

        self.norm: Optional[nn.Module] = create_norm(norm, self.out_channels, **norm_kwargs)
        self.norm_skip: Optional[nn.Module] = create_norm(norm, self.out_channels, **norm_kwargs)

        self.act = create_act(act, **act_kwargs)
        self.act_skip = create_act(act, **act_kwargs)

    @overload
    def forward(
        self,
        x: Tensor,
        x_skip: Tensor,
        inverse: Tensor,
        return_intermediate: Literal[False] = False,
        condition: Optional[str] = None,
    ) -> Tensor: ...

    @overload
    def forward(
        self,
        x: Tensor,
        x_skip: Tensor,
        inverse: Tensor,
        return_intermediate: Literal[True],
        condition: Optional[str] = None,
    ) -> Tuple[Tensor, Tensor]: ...

    def forward(
        self,
        x: Tensor,
        x_skip: Tensor,
        inverse: Tensor,
        return_intermediate: bool = False,
        condition: Optional[str] = None,
    ) -> Any:
        norm_kwargs = {} if condition is None else {"condition": condition}
        x = self.proj(x)
        if self.norm is not None:
            x = self.norm(x, **norm_kwargs)
        if self.act is not None:
            x = self.act(x)

        x_skip = self.proj_skip(x_skip)
        if self.norm_skip is not None:
            x_skip = self.norm_skip(x_skip, **norm_kwargs)
        if self.act_skip is not None:
            x_skip = self.act_skip(x_skip)

        out = x_skip + x[inverse]

        # Expose the projected skip branch so a decoder block can seed its xCPE from it
        if return_intermediate:
            return out, x_skip
        return out
