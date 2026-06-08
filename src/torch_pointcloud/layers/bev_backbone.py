r"""SSD-style 2D BEV backbone shared by the anchor-based voxel detectors (PointPillars, SECOND).

A packed-format port of the OpenPCDet `BaseBEVBackbone`:
:github: [open-mmlab/OpenPCDet](https://github.com/open-mmlab/OpenPCDet).
"""

from typing import Any, Callable, Dict, List, Optional, Sequence, Union

import torch
import torch.nn as nn
from torch import Tensor

from torch_pointcloud.layers.conv2d_blocks import Conv2dBlock


class BaseBEVBackbone(nn.Module):
    r"""SSD-style multi-scale 2D BEV backbone (`BaseBEVBackbone`).

    Each level downsamples the BEV pseudo-image with a strided $3\times3$ conv followed by
    `layer_nums` residual-free $3\times3$ convs, then upsamples back to a common stride with a
    transposed conv; the level outputs are concatenated along the channel dim.

    Args:
        input_channels: Channels of the input BEV feature map.
        layer_nums: Number of $3\times3$ convs after the strided conv, per level.
        layer_strides: Downsample stride of the leading conv, per level.
        num_filters: Channel width, per level.
        upsample_strides: Upsample factor per level. A factor $\geq 1$ uses a transposed conv; a
            factor $< 1$ (e.g. $0.5$) downsamples with a strided conv (nuScenes configs use both).
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
        num_levels = len(layer_nums)
        c_in_list = [input_channels, *num_filters[:-1]]

        block_kwargs: Dict[str, Any] = dict(act=act, act_kwargs=act_kwargs, norm=norm, norm_kwargs=norm_kwargs)

        self.blocks = nn.ModuleList()
        self.deblocks = nn.ModuleList()
        for idx in range(num_levels):
            cur_layers: List[nn.Module] = [
                nn.ZeroPad2d(1),
                Conv2dBlock(
                    c_in_list[idx],
                    num_filters[idx],
                    3,
                    stride=layer_strides[idx],
                    padding=0,
                    **block_kwargs,
                ),
            ]
            for _ in range(layer_nums[idx]):
                cur_layers.append(Conv2dBlock(num_filters[idx], num_filters[idx], 3, padding=1, **block_kwargs))
            self.blocks.append(nn.Sequential(*cur_layers))

            stride = upsample_strides[idx]
            if stride >= 1:
                deblock = Conv2dBlock(
                    num_filters[idx],
                    num_upsample_filters[idx],
                    int(stride),
                    stride=int(stride),
                    padding=0,
                    transposed=True,
                    **block_kwargs,
                )
            else:
                down = int(round(1 / stride))
                deblock = Conv2dBlock(
                    num_filters[idx], num_upsample_filters[idx], down, stride=down, padding=0, **block_kwargs
                )
            self.deblocks.append(deblock)

        self.num_bev_features = sum(num_upsample_filters)

    def forward(self, spatial_features: Tensor) -> Tensor:
        ups = []
        x = spatial_features
        for block, deblock in zip(self.blocks, self.deblocks):
            x = block(x)
            ups.append(deblock(x))
        return torch.cat(ups, dim=1) if len(ups) > 1 else ups[0]
