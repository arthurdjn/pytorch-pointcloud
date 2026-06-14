from typing import Any, Callable, Dict, Optional, Union

import torch.nn as nn
from torch import Tensor
from .act import create_act

from torch_pointcloud.layers.norms import create_norm


class Conv3dBlock(nn.Module):
    r"""Single `nn.Conv3d` + optional `nn.BatchNorm3d` + optional activation.

    Shape:
        Input: $(B, C_\text{in}, R, R, R)$
        Output: $(B, C_\text{out}, R, R, R)$

    Args:
        in_channels: Input channel count.
        out_channels: Output channel count.
        kernel_size: Conv3d kernel size.
        act: Activation, name resolved by `create_act`. `None` disables.
        act_first: If `True`, run activation before normalization.
        act_kwargs: Extra kwargs for the activation.
        norm: Normalization, name resolved by `create_norm`. `None` disables.
        norm_kwargs: Extra kwargs for the normalization.
        bias: Whether the Conv3d has a bias term.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        *,
        act: Union[str, Callable, None] = "relu",
        act_first: bool = False,
        act_kwargs: Optional[Dict[str, Any]] = None,
        norm: Union[str, Callable, None] = "batch_norm",
        norm_kwargs: Optional[Dict[str, Any]] = None,
        bias: bool = True,
    ):
        super().__init__()
        self.act_first = act_first
        self.conv = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
            bias=bias,
        )
        self.norm = create_norm(norm, out_channels, dim=3, **(norm_kwargs or {})) if norm is not None else None
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
