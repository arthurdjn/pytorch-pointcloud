from typing import Literal, Optional, Sequence, Union

import torch
import torch.nn as nn
from typing_extensions import TypeAlias

from .activations import ActLike, get_act
from .dropouts import get_dropout
from .norms import NormLike, get_norm


def _validate_block_order(order: str, layer_id: str, has_act: bool, has_norm: bool, has_dropout: bool) -> None:
    valid_chars = {layer_id, "a", "n", "d"}

    if not all(o in valid_chars for o in order):
        raise ValueError(f"Invalid characters in order string. Valid characters are: {valid_chars}")

    if len(order) != len(set(order)):
        raise ValueError("The 'order' sequence must not contain duplicate elements.")

    if layer_id not in order:
        raise ValueError(f"The main layer '{layer_id}' must be present in the order sequence.")

    if has_act and "a" not in order:
        raise ValueError("Activation layer 'a' must be in order when activation is specified.")

    if has_norm and "n" not in order:
        raise ValueError("Normalization layer 'n' must be in order when normalization is specified.")

    if has_dropout and "d" not in order:
        raise ValueError("Dropout layer 'd' must be in order when dropout is specified.")


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

    has_act = act is not None
    has_norm = norm is not None
    has_dropout = dropout is not None

    _validate_block_order(order, "l", has_act, has_norm, has_dropout)

    # Create layer instances
    layer_instances = {
        "l": nn.Linear(in_features, out_features, bias=bias),
        "a": get_act(act) if has_act else None,
        "n": get_norm(norm, out_features) if has_norm else None,
        "d": get_dropout(dropout) if has_dropout else None,
    }

    # Create sequential container with layers in specified order
    layers_ordered = [layer_instances[layer_id] for layer_id in order if layer_instances[layer_id] is not None]

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

    has_act = act is not None
    has_norm = norm is not None
    has_dropout = dropout is not None

    _validate_block_order(order, "c", has_act, has_norm, has_dropout)

    # Create layer instances
    layer_instances = {
        "c": nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        ),
        "a": get_act(act) if has_act else None,
        "n": get_norm(norm, out_channels) if has_norm else None,
        "d": get_dropout("dropout", dropout) if has_dropout else None,
    }

    # Create sequential container with layers in specified order
    layers_ordered = [layer_instances[layer_id] for layer_id in order if layer_instances[layer_id] is not None]

    return nn.Sequential(*layers_ordered)
