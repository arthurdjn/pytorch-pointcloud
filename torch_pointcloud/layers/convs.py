from typing import Literal, Optional, Sequence, Union

import torch
import torch.nn as nn

from .activations import ActLike, get_act
from .dropouts import get_dropout
from .norms import NormLike, get_norm


def _check_block_order(order: str, **layers: bool) -> None:
    layers_ids = [layer_name[0] for layer_name in layers.keys()]
    if not all(o in layers_ids for o in order):
        expected_layers = ", ".join(f"'{layer_name[0]}' ({layer_name})" for layer_name in layers.keys())
        raise ValueError(f"The 'order' sequence must contain only {expected_layers} elements. Got '{order}'.")

    if len(order) != len(set(order)):
        raise ValueError("The 'order' sequence must not contain duplicate elements.")

    for layer_name, has_layer in layers.items():
        layer_id = layer_name[0]
        if has_layer and layer_id not in order:
            raise ValueError(
                f"The {layer_name} layer is not present in the 'order' sequence. "
                f"Make sure to include '{layer_id}' in the 'order' sequence."
            )


def _check_conv_block_order(
    order: str,
    act: Optional[ActLike],
    norm: Optional[NormLike],
    dropout: Optional[float],
) -> None:
    _has_act = act is not None
    _has_norm = norm is not None
    _has_dropout = dropout is not None

    layers = {
        "conv": True,
        "act": _has_act,
        "norm": _has_norm,
        "dropout": _has_dropout,
    }

    _check_block_order(order, **layers)


def _check_linear_block_order(
    order: str,
    act: Optional[ActLike],
    norm: Optional[NormLike],
    dropout: Optional[float],
) -> None:
    _has_act = act is not None
    _has_norm = norm is not None
    _has_dropout = dropout is not None

    layers = {
        "linear": True,
        "act": _has_act,
        "norm": _has_norm,
        "dropout": _has_dropout,
    }

    _check_block_order(order, **layers)


class LinearBlock(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        act: Optional[ActLike] = "relu",
        norm: Optional[NormLike] = "batch_norm1d",
        dropout: Optional[float] = 0.0,
        order: Union[str, Sequence[Literal["a", "l", "n", "d"]]] = "land",
    ) -> None:
        super().__init__()
        order = order if isinstance(order, str) else "".join(order)
        _check_linear_block_order(order, act, norm, dropout)

        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.act = get_act(act) if act is not None else None
        self.norm = get_norm(norm, out_features) if norm is not None else None
        self.dropout = get_dropout("dropout", dropout) if dropout is not None else None
        self.order = order

    @property
    def layer_id_to_name(self) -> dict[str, str]:
        return {"a": "act", "l": "linear", "n": "norm", "d": "dropout"}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer_id in self.order:
            layer_name = self.layer_id_to_name[layer_id]
            layer = getattr(self, layer_name)
            if layer is not None:
                x = layer(x)

        return x

    def extra_repr(self) -> str:
        layer_names = [self.layer_id_to_name[o] for o in self.order]
        layer_reprs = [layer if getattr(self, layer) is not None else f"({layer})" for layer in layer_names]
        return "(order): " + " -> ".join(layer_reprs)


class Conv1dBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 1,
        stride: int = 1,
        padding: int = 0,
        groups: int = 1,
        bias: Union[bool, Literal["auto"]] = "auto",
        padding_mode: str = "zeros",
        act: Optional[ActLike] = "relu",
        norm: Optional[NormLike] = "batch_norm1d",
        dropout: Optional[float] = 0.0,
        order: Union[str, Sequence[Literal["a", "c", "n", "d"]]] = "cand",
    ) -> None:
        super().__init__()
        order = order if isinstance(order, str) else "".join(order)
        _check_conv_block_order(order, act, norm, dropout)

        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=bias if bias != "auto" else norm is None,
            padding_mode=padding_mode,
        )
        self.act = get_act(act) if act is not None else None
        self.norm = get_norm(norm, out_channels) if norm is not None else None
        self.dropout = get_dropout("dropout", dropout) if dropout is not None else None
        self.order = order

    @property
    def layer_id_to_name(self) -> dict[str, str]:
        return {"a": "act", "c": "conv", "n": "norm", "d": "dropout"}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer_id in self.order:
            layer_name = self.layer_id_to_name[layer_id]
            layer = getattr(self, layer_name)
            if layer is not None:
                x = layer(x)

        return x

    def extra_repr(self) -> str:
        layer_names = [self.layer_id_to_name[o] for o in self.order]
        layer_reprs = [layer if getattr(self, layer) is not None else f"({layer})" for layer in layer_names]
        return "(order): " + " -> ".join(layer_reprs)
