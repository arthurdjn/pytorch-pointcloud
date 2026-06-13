r"""SSD-style 2D BEV backbones shared by the voxel detectors (PointPillars, SECOND, Voxel Mamba).

A packed-format port of the OpenPCDet `BaseBEVBackbone` / `BaseBEVResBackbone`:
:github: [open-mmlab/OpenPCDet](https://github.com/open-mmlab/OpenPCDet).
"""

from typing import Any, Callable, Dict, List, Optional, Sequence, Union

import torch
import torch.nn as nn
from torch import Tensor

from torch_pointcloud.layers.act import create_act
from torch_pointcloud.layers.conv2d_blocks import Conv2dBlock


class BasicBlock2d(nn.Module):
    r"""Residual 2D conv block (the reference's `BasicBlock`) of the BEV residual backbone.

    Args:
        in_channels: Input channels.
        out_channels: Output channels.
        stride: Stride of the first conv (and the optional projection shortcut).
        downsample: Add a $1\times1$ projection shortcut to match channels / stride.
        norm: Normalization type or callable.
        norm_kwargs: Extra normalization arguments.
        act: Activation type or callable.
        act_kwargs: Extra activation arguments.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        stride: int = 1,
        downsample: bool = False,
        norm: Union[str, Callable, None] = "batch_norm",
        norm_kwargs: Optional[Dict[str, Any]] = None,
        act: Union[str, Callable, None] = "relu",
        act_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()
        block_kwargs: Dict[str, Any] = dict(norm=norm, norm_kwargs=norm_kwargs, act=act, act_kwargs=act_kwargs)
        self.conv1 = Conv2dBlock(in_channels, out_channels, 3, stride=stride, padding=1, **block_kwargs)
        self.conv2 = Conv2dBlock(out_channels, out_channels, 3, padding=1, act=None, norm=norm, norm_kwargs=norm_kwargs)

        self.act = create_act(act, **(act_kwargs or {}))
        self.downsample = (
            Conv2dBlock(in_channels, out_channels, 1, stride=stride, act=None, norm=norm, norm_kwargs=norm_kwargs)
            if downsample
            else None
        )

    def forward(self, x: Tensor) -> Tensor:
        identity = x if self.downsample is None else self.downsample(x)
        out = self.conv2(self.conv1(x)) + identity
        return out if self.act is None else self.act(out)


class BaseBEVBackbone(nn.Module):
    r"""SSD-style multi-scale 2D BEV backbone (`BaseBEVBackbone`).

    Each level downsamples the BEV pseudo-image with a strided $3\times3$ conv followed by
    `layer_nums` residual-free $3\times3$ convs, then upsamples back to a common stride; the level
    outputs are concatenated along the channel dim. An upsample factor $\geq 1$ uses a transposed
    conv, a factor $< 1$ (e.g. $0.5$) a strided down-conv (nuScenes configs use both).

    Args:
        input_channels: Channels of the input BEV feature map.
        layer_nums: Number of $3\times3$ convs after the strided conv, per level.
        layer_strides: Downsample stride of the leading conv, per level.
        num_filters: Channel width, per level.
        upsample_strides: Upsample factor per level.
        num_upsample_filters: Channels of each upsampled level.
        act: Activation type or callable for every conv block.
        act_kwargs: Extra activation arguments.
        norm: Normalization type or callable for every conv block.
        norm_kwargs: Extra normalization arguments.
    """

    def __init__(
        self,
        input_channels: int,
        layer_nums: Sequence[int],
        layer_strides: Sequence[int],
        num_filters: Sequence[int],
        upsample_strides: Sequence[float],
        num_upsample_filters: Sequence[int],
        *,
        act: Union[str, Callable, None] = "relu",
        act_kwargs: Optional[Dict[str, Any]] = None,
        norm: Union[str, Callable, None] = "batch_norm",
        norm_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()
        block_kwargs: Dict[str, Any] = dict(act=act, act_kwargs=act_kwargs, norm=norm, norm_kwargs=norm_kwargs)
        c_in_list = [input_channels, *num_filters[:-1]]

        self.blocks = nn.ModuleList()
        self.deblocks = nn.ModuleList()
        for idx in range(len(layer_nums)):
            level: List[nn.Module] = [
                nn.ZeroPad2d(1),
                Conv2dBlock(c_in_list[idx], num_filters[idx], 3, stride=layer_strides[idx], padding=0, **block_kwargs),
            ]
            for _ in range(layer_nums[idx]):
                level.append(Conv2dBlock(num_filters[idx], num_filters[idx], 3, padding=1, **block_kwargs))
            self.blocks.append(nn.Sequential(*level))

            stride = upsample_strides[idx]
            if stride >= 1:
                deblock: nn.Module = Conv2dBlock(
                    num_filters[idx],
                    num_upsample_filters[idx],
                    int(stride),
                    stride=int(stride),
                    transposed=True,
                    **block_kwargs,
                )
            else:
                down = int(round(1 / stride))
                deblock = Conv2dBlock(num_filters[idx], num_upsample_filters[idx], down, stride=down, **block_kwargs)
            self.deblocks.append(deblock)

        self.num_bev_features = sum(num_upsample_filters)

    def forward(self, spatial_features: Tensor) -> Tensor:
        ups = []
        x = spatial_features
        for block, deblock in zip(self.blocks, self.deblocks):
            x = block(x)
            ups.append(deblock(x))
        return torch.cat(ups, dim=1) if len(ups) > 1 else ups[0]


class BaseBEVResBackbone(nn.Module):
    r"""Residual SSD-style 2D BEV backbone (`BaseBEVResBackbone`) used by Voxel Mamba.

    Same scaffolding as [`BaseBEVBackbone`][torch_pointcloud.layers.bev_backbone.BaseBEVBackbone] (per-level
    block then upsample, concatenated), but each level is a stack of residual
    [`BasicBlock2d`][torch_pointcloud.layers.bev_backbone.BasicBlock2d]s instead of plain $3\times3$ convs.

    Args:
        input_channels: Channels of the input BEV feature map.
        layer_nums: Number of residual blocks after the strided block, per level.
        layer_strides: Downsample stride of the leading block, per level.
        num_filters: Channel width, per level.
        upsample_strides: Upsample factor per level.
        num_upsample_filters: Channels of each upsampled level.
        act: Activation type or callable for every conv block.
        act_kwargs: Extra activation arguments.
        norm: Normalization type or callable for every conv block.
        norm_kwargs: Extra normalization arguments.
    """

    def __init__(
        self,
        input_channels: int,
        layer_nums: Sequence[int],
        layer_strides: Sequence[int],
        num_filters: Sequence[int],
        upsample_strides: Sequence[float],
        num_upsample_filters: Sequence[int],
        *,
        act: Union[str, Callable, None] = "relu",
        act_kwargs: Optional[Dict[str, Any]] = None,
        norm: Union[str, Callable, None] = "batch_norm",
        norm_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()
        block_kwargs: Dict[str, Any] = dict(act=act, act_kwargs=act_kwargs, norm=norm, norm_kwargs=norm_kwargs)
        c_in_list = [input_channels, *num_filters[:-1]]

        self.blocks = nn.ModuleList()
        self.deblocks = nn.ModuleList()
        for idx in range(len(layer_nums)):
            level: List[nn.Module] = [
                BasicBlock2d(
                    c_in_list[idx], num_filters[idx], stride=layer_strides[idx], downsample=True, **block_kwargs
                )
            ]
            for _ in range(layer_nums[idx]):
                level.append(BasicBlock2d(num_filters[idx], num_filters[idx], **block_kwargs))
            self.blocks.append(nn.Sequential(*level))

            stride = upsample_strides[idx]
            if stride >= 1:
                deblock: nn.Module = Conv2dBlock(
                    num_filters[idx],
                    num_upsample_filters[idx],
                    int(stride),
                    stride=int(stride),
                    transposed=True,
                    **block_kwargs,
                )
            else:
                down = int(round(1 / stride))
                deblock = Conv2dBlock(num_filters[idx], num_upsample_filters[idx], down, stride=down, **block_kwargs)
            self.deblocks.append(deblock)

        self.num_bev_features = sum(num_upsample_filters)

    def forward(self, spatial_features: Tensor) -> Tensor:
        ups = []
        x = spatial_features
        for block, deblock in zip(self.blocks, self.deblocks):
            x = block(x)
            ups.append(deblock(x))
        return torch.cat(ups, dim=1) if len(ups) > 1 else ups[0]
