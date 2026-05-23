from typing import Any, Callable, Dict, Optional, Union

import torch.nn as nn
from torch import Tensor
from torch_geometric.nn.resolver import activation_resolver, normalization_resolver

from torch_pointcloud.utils.types import OptTensor


class LinearBlock(nn.Module):
    """Linear block consisting of a linear layer, normalization and activation.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        bias: Whether to use a bias for the linear layer.
        act: Activation function to use. If `None`, no activation is applied.
        act_kwargs: Extra arguments for the activation function.
        act_first: Whether to apply the activation function before the normalization.
        norm: Normalization layer to use. If `None`, no normalization is applied.
        norm_kwargs: Extra arguments for the normalization layer.

    Example:
        ```python
        from torch_pointcloud.layers import LinearBlock

        block = LinearBlock(64, 128, act="relu", norm="batch_norm", bias=False)
        x = torch.randn(32, 64)
        y = block(x)
        print(y.shape)
        ```
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        act: Union[str, Callable, None] = None,
        act_kwargs: Optional[Dict[str, Any]] = None,
        act_first: bool = False,
        bias: bool = True,
        norm: Union[str, Callable, None] = None,
        norm_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        self.act_first = act_first
        self.stem = nn.Linear(in_channels, out_channels, bias=bias)
        self.norm = normalization_resolver(norm, out_channels, **(norm_kwargs or {})) if norm is not None else None
        self.act = activation_resolver(act, **(act_kwargs or {})) if act is not None else None

    # TODO: Rename stem to lin, and remove pos and batch arguments as they are not used.
    def forward(self, x: Tensor, pos: OptTensor = None, batch: OptTensor = None) -> Tensor:
        x = self.stem(x)
        if self.act_first:
            if self.act is not None:
                x = self.act(x)
            if self.norm is not None:
                x = self.norm(x)
        else:
            if self.norm is not None:
                x = self.norm(x)
            if self.act is not None:
                x = self.act(x)
        return x
