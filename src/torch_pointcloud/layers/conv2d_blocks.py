"""2D convolution block with optional normalization and activation."""

from typing import Any, Callable, Dict, Optional, Union

import torch.nn as nn
from torch import Tensor

from torch_pointcloud.layers.act import create_act
from torch_pointcloud.layers.norms import create_norm


class Conv2dBlock(nn.Module):
    r"""Single `nn.Conv2d` (or `nn.ConvTranspose2d`) + optional norm + optional activation.

    The 2D analogue of [`Conv3dBlock`][torch_pointcloud.layers.conv3d_blocks.Conv3dBlock], with
    `stride`, `padding` and `transposed` exposed so it can express the strided down-convs and
    transposed up-convs of an SSD-style BEV backbone.

    Shape:
        Input: $(B, C_\text{in}, H, W)$
        Output: $(B, C_\text{out}, H', W')$

    Args:
        in_channels: Input channel count.
        out_channels: Output channel count.
        kernel_size: Conv kernel size.
        stride: Conv stride.
        padding: Conv padding.
        transposed: Use `nn.ConvTranspose2d` instead of `nn.Conv2d` (for upsampling).
        act: Activation, name resolved by `create_act`. `None` disables.
        act_first: If `True`, run activation before normalization.
        act_kwargs: Extra kwargs for the activation.
        norm: Normalization, name resolved by `create_norm` (with `dim=2`). `None` disables.
        norm_kwargs: Extra kwargs for the normalization.
        bias: Whether the conv has a bias term.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        *,
        stride: int = 1,
        padding: int = 0,
        transposed: bool = False,
        act: Union[str, Callable, None] = "relu",
        act_first: bool = False,
        act_kwargs: Optional[Dict[str, Any]] = None,
        norm: Union[str, Callable, None] = "batch_norm",
        norm_kwargs: Optional[Dict[str, Any]] = None,
        bias: bool = False,
    ) -> None:
        super().__init__()
        self.act_first = act_first
        conv_cls = nn.ConvTranspose2d if transposed else nn.Conv2d
        self.conv = conv_cls(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias)
        self.norm = create_norm(norm, out_channels, dim=2, **(norm_kwargs or {}))
        self.act = create_act(act, **(act_kwargs or {}))

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        if self.act_first and self.act is not None:
            x = self.act(x)
        if self.norm is not None:
            x = self.norm(x)
        if not self.act_first and self.act is not None:
            x = self.act(x)
        return x
