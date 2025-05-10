from typing import Any, Dict, Literal, Optional, Sequence, Union

import torch.nn as nn

from .activations import ActLike, create_act
from .dropouts import create_dropout
from .norms import NormLike, create_norm


def _validate_block_order(layers: Dict[str, Any], order: str) -> None:
    if not all(o in layers for o in order):
        valid_layer_ids = ", ".join([f"{k!r}" for k in layers.keys()])
        raise ValueError(f"Invalid order sequence. Got order {order!r}, but valid layer IDs are {valid_layer_ids}.")

    if len(order) != len(set(order)):
        raise ValueError("The order sequence must not contain duplicate elements.")

    for layer_id, layer in layers.items():
        if layer is not None and layer_id not in order:
            raise ValueError(f"Layer {layer_id!r} must be in the order sequence. Got order {order!r}.")


LinearBlockOrderLike = Union[str, Sequence[Literal["l", "a", "n", "d"]]]


def linear_block(
    in_features: int,
    out_features: int,
    bias: bool = True,
    act: Optional[ActLike] = "relu",
    norm: Optional[NormLike] = "batch_norm1d",
    dropout: Optional[float] = 0.0,
    order: LinearBlockOrderLike = "lnad",
) -> nn.Sequential:
    r"""Creates a customizable linear block consisting of a linear layer, activation, normalization and dropout.
    The order of the layers can be customized using the `order` argument.

    Input Shape:
        - $x$: $(N, *, \text{in\_features})$ where $*$ means any number of additional dimensions.

    Output Shape:
        - $x$: $(N, *, \text{out\_features})$ where $*$ means any number of additional dimensions.

    Args:
        in_features: The number of input features.
        out_features: The number of output features.
        bias: If `True`, adds a learnable bias to the linear layer.
        act: The activation function to use. If `None`, no activation is applied.
        norm: The normalization layer to use. If `None`, no normalization is applied.
        dropout: The dropout rate to use. If `None`, no dropout is applied.
        order: The order of the layers. The order must contain the desired layers as a string of the following characters:
            `"l"` (linear), `"a"` (activation), `"n"` (normalization), `"d"` (dropout).

    Returns:
        A `torch.nn.Sequential` block containing the layers in the specified order.

    Examples:
        >>> block = linear_block(64, 128, act="relu", norm="batch_norm1d", dropout=0.1)
        >>> x = torch.randn(32, 64)
        >>> y = block(x)
    """

    order = order if isinstance(order, str) else "".join(order)
    layers = {
        "l": nn.Linear(in_features, out_features, bias=bias),
        "n": create_norm(norm, out_features) if norm is not None else None,
        "a": create_act(act) if act is not None else None,
        "d": create_dropout(dropout) if dropout is not None else None,
    }

    _validate_block_order(layers, order)

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
    order: Union[str, Sequence[Literal["a", "c", "n", "d"]]] = "cnad",
) -> nn.Sequential:
    r"""Creates a customizable 1D convolutional block consisting of a convolutional layer, activation, normalization and dropout.
    The order of the layers can be customized using the `order` argument.

    Input Shape:
        - $x$: $(N, \text{in\_channels}, L)$ where $L$ is the length of the input.

    Output Shape:
        - $x$: $(N, \text{out\_channels}, L_{\text{out}})$ where $L_{\text{out}}$ is the length of the output.

    Args:
        in_channels: The number of input channels.
        out_channels: The number of output channels.
        kernel_size: The size of the convolving kernel.
        stride: The stride of the convolution.
        padding: The padding to add to the input.
        dilation: The spacing between kernel elements.
        groups: The number of blocked connections from input channels to output channels.
        bias: If `True`, adds a learnable bias to the convolutional layer.
        act: The activation function to use. If `None`, no activation is applied.
        norm: The normalization layer to use. If `None`, no normalization is applied.
        dropout: The dropout rate to use. If `None`, no dropout is applied.
        order: The order of the layers. The order must contain the desired layers as a string of the following characters:
            `"c"` (convolution), `"a"` (activation), `"n"` (normalization), `"d"` (dropout).

    Returns:
        A `torch.nn.Sequential` block containing the layers in the specified order.

    Examples:
        >>> block = conv1d_block(64, 128, 3, act="relu", norm="batch_norm1d", dropout=0.1)
        >>> x = torch.randn(32, 64, 128)
        >>> y = block(x)
    """
    order = order if isinstance(order, str) else "".join(order)
    layers = {
        "c": nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias=bias),
        "n": create_norm(norm, out_channels) if norm is not None else None,
        "a": create_act(act) if act is not None else None,
        "d": create_dropout(dropout) if dropout is not None else None,
    }

    _validate_block_order(layers, order)

    # NOTE: Explicit assignment for type checking
    layers_ordered = [layer for layer_id in order if (layer := layers[layer_id]) is not None]
    return nn.Sequential(*layers_ordered)
