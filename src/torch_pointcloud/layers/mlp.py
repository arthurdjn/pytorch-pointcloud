from typing import Optional, Sequence, Union

import torch.nn as nn
from torch import Tensor

from torch_pointcloud.utils.conversion import ensure_tuple_size

from .activations import ActLike
from .blocks import linear_block
from .norms import NormLike


class MLP(nn.Module):
    def __init__(
        self,
        channels: Optional[Sequence[int]] = None,
        *,
        in_channels: Optional[int] = None,
        out_channels: Optional[int] = None,
        act: ActLike = "relu",
        norm: NormLike = "batch_norm1d",
        bias: bool = True,
        dropout: Union[float, Sequence[float]] = 0.0,
        order: str = "land",
        plain_last: bool = False,
    ) -> None:
        super().__init__()
        self.plain_last = plain_last
        self.bias = bias
        self.act = act
        self.norm = norm
        self.order = order

        channels = list(channels) if channels is not None else []
        if in_channels is not None:
            channels = [in_channels] + channels
        if out_channels is not None:
            channels = channels + [out_channels]

        if len(channels) < 2:
            raise ValueError(f"MLP must have at least 2 channels, got {len(channels)} channel.")

        num_layers = len(channels) - 1
        self.dropout = ensure_tuple_size(dropout, size=num_layers)
        if plain_last:
            self.dropout = self.dropout[:-1] + (0.0,)

        self.fcs = nn.ModuleList()
        for i in range(num_layers):
            if plain_last and i == num_layers - 1:
                layer = linear_block(channels[i], channels[i + 1], bias=bias, act=None, norm=None, order="ld")
            else:
                layer = linear_block(
                    channels[i],
                    channels[i + 1],
                    bias=bias,
                    act=act,
                    norm=norm,
                    dropout=self.dropout[i],
                    order=order,
                )
            self.fcs.append(layer)

    def forward(self, x: Tensor) -> Tensor:
        for fc in self.fcs:
            x = fc(x)
        return x

    def extra_repr(self) -> str:
        def find_linear_layer(module: nn.Module) -> nn.Linear:
            for layer in module.children():
                if isinstance(layer, nn.Linear):
                    return layer
            raise ValueError("No linear layer found in the module")

        channels = []
        for layer in self.fcs:
            lin = find_linear_layer(layer)
            channels.append(lin.in_features)

        channels.append(lin.out_features)
        inner_repr = f"{', '.join(map(str, channels))}"
        if self.plain_last:
            inner_repr += ", plain_last=True"
        if self.act is not None and "a" in self.order:
            inner_repr += f", act={self.act}"
        if self.norm is not None and "n" in self.order:
            inner_repr += f", norm={self.norm}"
        if self.bias:
            inner_repr += ", bias=True"
        if self.dropout and "d" in self.order:
            dropouts = list(self.dropout)
            if self.plain_last:
                dropouts.pop(-1)
            same = all(dropouts[0] == dropout for dropout in dropouts) if len(dropouts) > 0 else True
            inner_repr += f", dropout={self.dropout[0] if same else dropouts}"
        return inner_repr

    def __repr__(self) -> str:
        inner_repr = self.extra_repr()
        return f"{self.__class__.__name__}({inner_repr})"
