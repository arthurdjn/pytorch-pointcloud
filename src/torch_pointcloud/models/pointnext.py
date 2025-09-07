from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import torch.nn as nn
from torch import Tensor

from torch_pointcloud.layers.pointnext_blocks import PointNeXtResidualBlock, PointNeXtSetAbstraction
from torch_pointcloud.utils.conversion import ensure_tuple, ensure_tuple_size
from torch_pointcloud.utils.types import AggrType, OptTensor


class PointNeXtEncoderBlock(nn.Module):
    def __init__(
        self,
        spatial_dim: int,
        channels: int,
        depth: int,
        expansion: int,
        ratio: float,
        radius: float,
        num_neighbors: int,
        act: Union[str, Callable, None] = "relu",
        act_kwargs: Optional[Dict[str, Any]] = None,
        act_first: bool = False,
        norm: Union[str, Callable, None] = "batch_norm",
        norm_kwargs: Optional[Dict[str, Any]] = None,
        bias: Union[bool, List[bool]] = True,
        add_self_loops: bool = False,
        aggr: AggrType = "max",
        downsample: Optional[PointNeXtSetAbstraction] = None,
    ):
        super().__init__()
        act_kwargs = act_kwargs or {}
        norm_kwargs = norm_kwargs or {}

        self.downsample = downsample
        self.blocks = nn.ModuleList()
        for _ in range(depth):
            block = PointNeXtResidualBlock(
                spatial_dim=spatial_dim,
                channels=channels,
                expansion=expansion,
                ratio=ratio,
                radius=radius,
                num_neighbors=num_neighbors,
                act=act,
                act_kwargs=act_kwargs,
                act_first=act_first,
                norm=norm,
                norm_kwargs=norm_kwargs,
                bias=bias,
                add_self_loops=add_self_loops,
                aggr=aggr,
            )
            self.blocks.append(block)

    def forward(self, x: Tensor, pos: Tensor, batch: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        if self.downsample is not None:
            x, pos, batch = self.downsample(x, pos, batch)

        for block in self.blocks:
            x = block(x, pos, batch)

        return x, pos, batch


class PointNeXtDecoderBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        depth: int,
        expansion: int,
        ratio: float,
    ):
        super().__init__()
        self.blocks = nn.ModuleList()
        for _ in range(depth):
            pass

    def forward(
        self,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        raise NotImplementedError


class PointNeXtEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        channels: Sequence[int],
        *,
        spatial_dim: int = 3,
        depths: Sequence[int],
        expansion: Union[int, Sequence[int]],
        ratios: Sequence[float],
        radiuses: Sequence[Union[float, Sequence[float]]],
        num_neighbors: Sequence[Union[int, Sequence[int]]],
        act: Union[str, Callable, None] = "relu",
        act_kwargs: Optional[Dict[str, Any]] = None,
        act_first: bool = False,
        norm: Union[str, Callable, None] = "batch_norm",
        norm_kwargs: Optional[Dict[str, Any]] = None,
        bias: Union[bool, List[bool]] = True,
        add_self_loops: bool = False,
        aggr: AggrType = "max",
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.depths = ensure_tuple(depths)

        size = len(self.depths)
        extra_msg = (
            f"Invalid {self.__class__.__name__} parameter: "
            f"expected `{{param}}` to have the same length as the number of channels ({size})."
        )
        self.channels = ensure_tuple_size(channels, size, extra_msg=extra_msg.format(param="channels"))
        self.ratios = ensure_tuple_size(ratios, size, extra_msg=extra_msg.format(param="ratios"))
        self.radiuses = ensure_tuple_size(radiuses, size, extra_msg=extra_msg.format(param="radiuses"))
        self.num_neighbors = ensure_tuple_size(num_neighbors, size, extra_msg=extra_msg.format(param="num_neighbors"))
        self.expansion = ensure_tuple_size(expansion, size, extra_msg=extra_msg.format(param="expansion"))

        channels = [self.in_channels] + list(self.channels)
        self.blocks = nn.ModuleList()
        for i in range(size):
            downsample = PointNeXtSetAbstraction(
                spatial_dim=spatial_dim,
                in_channels=channels[i],
                channels=[channels[i + 1]],
                ratio=self.ratios[i],
                radius=self.radiuses[i],
                num_neighbors=self.num_neighbors[i],
                act=act,
                act_kwargs=act_kwargs,
                act_first=act_first,
                norm=norm,
                norm_kwargs=norm_kwargs,
                bias=bias,
                add_self_loops=add_self_loops,
                aggr=aggr,
            )
            block = PointNeXtEncoderBlock(
                spatial_dim=spatial_dim,
                channels=channels[i + 1],
                depth=self.depths[i],
                expansion=self.expansion[i],
                ratio=self.ratios[i],
                radius=self.radiuses[i],
                num_neighbors=self.num_neighbors[i],
                act=act,
                act_kwargs=act_kwargs,
                act_first=act_first,
                norm=norm,
                norm_kwargs=norm_kwargs,
                bias=bias,
                add_self_loops=add_self_loops,
                aggr=aggr,
                downsample=downsample,
            )
            self.blocks.append(block)

    def forward(self, x: OptTensor, pos: Tensor, batch: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        for block in self.blocks:
            x, pos, batch = block(x, pos, batch)
        return x, pos, batch
