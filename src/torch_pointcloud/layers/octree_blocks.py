from typing import TYPE_CHECKING, Any, Callable, Dict, Optional, Sequence, Union

import torch.nn as nn
from torch import Tensor
from torch_geometric.nn.resolver import activation_resolver, normalization_resolver

from torch_pointcloud.utils.conversion import ensure_list
from torch_pointcloud.utils.imports import optional_import

if TYPE_CHECKING:
    import ocnn
    from ocnn.octree import Octree

ocnn, _ = optional_import("ocnn")
Octree, _ = optional_import("ocnn.octree", "Octree")

MAX_BUFFER = int(2e8)


class OctreeConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Sequence[int]],
        stride: int = 1,
        nempty: bool = False,
        act: Union[str, Callable, None] = "relu",
        act_kwargs: Optional[Dict[str, Any]] = None,
        act_first: bool = False,
        norm: Union[str, Callable, None] = "batch_norm",
        norm_kwargs: Optional[Dict[str, Any]] = None,
        bias: bool = True,
        direct_method: bool = False,
        max_buffer: int = MAX_BUFFER,
    ):
        super().__init__()
        # NOTE: OCNN expects the kernel size to be a list, otherwise assertion error will be raised.
        kernel_size = ensure_list(kernel_size)
        act_kwargs = act_kwargs or {}
        norm_kwargs = norm_kwargs or {}

        self.conv = ocnn.nn.OctreeConv(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            nempty=nempty,
            use_bias=bias,
            direct_method=direct_method,
            max_buffer=max_buffer,
        )
        self.norm = normalization_resolver(norm, out_channels, **norm_kwargs)
        self.act = activation_resolver(act, **act_kwargs)
        self.act_first = act_first

    def forward(self, x: Tensor, octree: Octree, depth: int) -> Tensor:
        x = self.conv(x, octree, depth)
        if self.act is not None and self.act_first:
            x = self.act(x)
        if self.norm is not None:
            x = self.norm(x)
        if self.act is not None and not self.act_first:
            x = self.act(x)
        return x
