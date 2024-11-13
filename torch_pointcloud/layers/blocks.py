from typing import Any, Dict, Literal, Optional, Sequence, Union

import torch.nn as nn

from .activations import ActLike, get_act
from .dropouts import get_dropout
from .norms import NormLike, get_norm


def _validate_block_order(order: str, layers: Dict[str, Any]) -> None:
    if not all(o in layers for o in order):
        valid_layer_ids = ", ".join([f"{k!r}" for k in layers.keys()])
        raise ValueError(f"Invalid order sequence. Got order {order!r}, but valid layer IDs are {valid_layer_ids}.")

    if len(order) != len(set(order)):
        raise ValueError("The order sequence must not contain duplicate elements.")

    for layer_id, layer in layers.items():
        if layer is not None and layer_id not in order:
            raise ValueError(f"Layer {layer_id!r} must be in the order sequence. Got order {order!r}.")


def linear_block(
    in_features: int,
    out_features: int,
    bias: bool = True,
    act: Optional[ActLike] = "relu",
    norm: Optional[NormLike] = "batch_norm1d",
    dropout: Optional[float] = 0.0,
    order: Union[str, Sequence[Literal["a", "l", "n", "d"]]] = "land",
) -> nn.Sequential:
    order = order if isinstance(order, str) else "".join(order)
    layers = {
        "l": nn.Linear(in_features, out_features, bias=bias),
        "a": get_act(act) if act is not None else None,
        "n": get_norm(norm, out_features) if norm is not None else None,
        "d": get_dropout(dropout) if dropout is not None else None,
    }

    _validate_block_order(order, layers)

    # NOTE: Explicit assignment for type checking
    layers_ordered = [layer for layer_id in order if (layer := layers[layer_id]) is not None]
    return nn.Sequential(*layers_ordered)


def conv1d_block(
    in_channels: int,
    out_channels: int,
    kernel_size: int,
    stride: int = 1,
    padding: Union[str, int] = 0,
    dilation: int = 1,
    groups: int = 1,
    bias: bool = True,
    act: Optional[ActLike] = "relu",
    norm: Optional[NormLike] = "batch_norm1d",
    dropout: Optional[float] = 0.0,
    order: Union[str, Sequence[Literal["a", "c", "n", "d"]]] = "cand",
) -> nn.Sequential:
    order = order if isinstance(order, str) else "".join(order)
    layers = {
        "c": nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias=bias),
        "a": get_act(act) if act is not None else None,
        "n": get_norm(norm, out_channels) if norm is not None else None,
        "d": get_dropout(dropout) if dropout is not None else None,
    }

    _validate_block_order(order, layers)

    # NOTE: Explicit assignment for type checking
    layers_ordered = [layer for layer_id in order if (layer := layers[layer_id]) is not None]
    return nn.Sequential(*layers_ordered)
