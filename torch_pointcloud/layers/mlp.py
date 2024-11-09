from typing import Any, Dict, List, Optional, Union

import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ._modules import REGISTERED_MODULE_TYPE, ModuleLike, get_module
from .activations import get_act
from .norms import get_norm


class MLP(nn.Module):
    def __init__(
        self,
        dims: List[int],
        *,
        act: Optional[ModuleLike] = "relu",
        act_first: bool = False,
        norm: Optional[ModuleLike] = "batch_norm1d",
        dropout: float = 0.0,
        bias: Union[bool] = True,
        plain_last: bool = True,
    ) -> None:
        super().__init__()
        N = len(dims)
        N1 = N - 1
        N2 = N - 2 if plain_last else N1
        if N < 2:
            raise ValueError(f"The MLP must have at least 2 dimensions. Got {N}.")

        # Format activations (one for each layer except the last one)
        act = get_act(act) if act is not None else None
        # Format batch norms (one for each layer except the last one)
        norms = [get_norm(norm, d) for d in dims[1 : N2 + 1]] if norm is not None else [nn.Identity()] * N2
        # Format dropouts (one for each layer except the last one)
        dropouts = [dropout] * N2
        # Format biases (one for each layer)
        biases = [bias] * N1

        # Sanity check
        if len(norms) != N2 or len(dropouts) != N2:
            raise ValueError(
                "The number of batch norm layers and dropouts must be equal to the number of layers (-1 if plain_last is `True`) in the MLP. "
                f"Got {len(norms)}, {len(dropouts)}, and {N1} (with {plain_last=}) respectively."
            )

        self.lins = nn.ModuleList(
            [nn.Linear(in_dim, out_dim, bias=bias) for in_dim, out_dim, bias in zip(dims[:-1], dims[1:], biases)]
        )
        self.norms = nn.ModuleList(norms)
        self.dropouts = dropouts
        self.act = act
        self.act_first = act_first
        self.plain_last = plain_last

    def forward(self, x: Tensor) -> Tensor:
        # If `plain_last=True`, then `len(norms) = len(acts) = len(dropouts) = len(linear_layers) - 1,
        # thus skipping the execution of the last layer inside the for-loop.
        for lin, norm, dropout in zip(self.lins, self.norms, self.dropouts):
            x = lin(x)
            if self.act and self.act_first:
                x = self.act(x)
            x = norm(x)
            if self.act and not self.act_first:
                x = self.act(x)
            x = F.dropout(x, p=dropout, training=self.training)

        # If `plain_last=True`, then the last layer is executed here.
        if self.plain_last:
            x = self.lins[-1](x)

        return x


_CONV_LAYERS: Dict[str, REGISTERED_MODULE_TYPE] = {
    "conv1d": nn.Conv1d,
    "conv2d": nn.Conv2d,
    "conv3d": nn.Conv3d,
}


def _get_conv(name: ModuleLike, *args: Any, **kwargs: Any) -> nn.Module:
    return get_module(name, *args, registry=_CONV_LAYERS, **kwargs)


class SharedMLP(nn.Module):
    def __init__(
        self,
        channels: List[int],
        *,
        act_first: bool = False,
        act: Optional[ModuleLike] = "relu",
        norm: Optional[ModuleLike] = "batch_norm1d",
        conv: ModuleLike = "conv1d",
        dropout: float = 0.0,
        bias: Union[bool] = True,
        plain_last: bool = True,
    ):
        super().__init__()
        N = len(channels)
        N1 = N - 1
        N2 = N - 2 if plain_last else N1
        if N < 2:
            raise ValueError(f"The SharedMLP must have at least 2 channels. Got {N}.")

        # Format activations (one for each layer except the last one)
        act = get_act(act) if act is not None else None
        # Format batch norms (one for each layer except the last one)
        norms = [get_norm(norm, c) for c in channels[1 : N2 + 1]] if norm is not None else [nn.Identity()] * N2
        # Format dropouts (one for each layer except the last one)
        dropouts = [dropout] * N2
        # Format biases (one for each layer)
        biases = [bias] * N1

        # Sanity check
        if len(norms) != N2 or len(dropouts) != N2 or len(biases) != N1:
            raise ValueError(
                "The number of batch norm layers, and dropouts must be equal to the number of layers (-1 if plain_last is `True`) in the SharedMLP. "
                f"Got {len(norms)}, {len(dropouts)}, and {N1} (with {plain_last=}) respectively."
            )

        convs = [
            _get_conv(conv, in_channels, out_channels, bias=bias, kernel_size=1, stride=1)
            for in_channels, out_channels, bias in zip(channels[:-1], channels[1:], biases)
        ]
        self.convs = nn.ModuleList(convs)
        self.norms = nn.ModuleList(norms)
        self.dropouts = dropouts
        self.act = act
        self.act_first = act_first
        self.plain_last = plain_last

    def forward(self, x: Tensor) -> Tensor:
        # If `plain_last=True`, then `len(norms) = len(dropouts) = len(convs) - 1,
        # thus skipping the execution of the last layer inside the for-loop.
        for conv, norm, dropout in zip(self.convs, self.norms, self.dropouts):
            x = conv(x)
            if self.act and self.act_first:
                x = self.act(x)
            x = norm(x)
            if self.act and not self.act_first:
                x = self.act(x)
            x = F.dropout(x, p=dropout, training=self.training)

        # If `plain_last=True`, then the last layer is executed here.
        if self.plain_last:
            x = self.convs[-1](x)

        return x


def shared_mlp1d(
    channels: List[int],
    *,
    act_first: bool = False,
    act: ModuleLike = "relu",
    bn: bool = True,
    dropout: float = 0,
    bias: bool = True,
    plain_last: bool = True,
) -> SharedMLP:
    return SharedMLP(
        channels,
        act_first=act_first,
        act=act,
        norm="batch_norm1d" if bn else None,
        conv="conv1d",
        dropout=dropout,
        bias=bias,
        plain_last=plain_last,
    )


def shared_mlp2d(
    channels: List[int],
    *,
    act_first: bool = False,
    act: ModuleLike = "relu",
    bn: bool = True,
    dropout: float = 0,
    bias: bool = True,
    plain_last: bool = True,
) -> SharedMLP:
    return SharedMLP(
        channels,
        act_first=act_first,
        act=act,
        norm="batch_norm2d" if bn else None,
        conv="conv2d",
        dropout=dropout,
        bias=bias,
        plain_last=plain_last,
    )
