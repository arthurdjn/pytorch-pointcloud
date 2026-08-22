"""Per-channel affine transformation as a function and a learnable module."""

from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from torch import Tensor


def affine(x: Tensor, weight: Tensor, bias: Optional[Tensor] = None) -> Tensor:
    r"""Apply a per-channel affine transformation $y = x \cdot \text{weight} + \text{bias}$.

    Args:
        x: Input tensor.
        weight: Per-channel scale, broadcastable against `x`.
        bias: Optional per-channel offset, broadcastable against `x`. `None` skips the addition.

    Returns:
        The transformed tensor, same shape as `x`.

    Shape:
        Input: $(N, *, C)$ where $*$ means any number of additional dimensions.
        Output: $(N, *, C)$, same as the input.

    Example:
        ```python
        import torch
        from torch_pointcloud.layers import affine

        x = torch.randn(4, 8)
        y = affine(x, weight=torch.ones(8), bias=torch.zeros(8))
        assert torch.equal(y, x)
        ```
    """
    if bias is None:
        return x * weight
    return x * weight + bias


class Affine(nn.Module):
    r"""Applies an affine transformation to the input.
    This layer will apply the following transformation to the input tensor $x$:

    $$
    y = x \cdot \text{weight} + \text{bias}
    $$

    Args:
        num_features: The number of features in the input.
        bias: Whether to use bias.
        device: The device to use.
        dtype: The dtype to use.

    Shape:
        - Input: $(N, *, C)$ where $*$ means any number of additional dimensions.
        - Output: $(N, *, C)$ where $*$ means any number of additional dimensions.
    """

    def __init__(
        self,
        num_features: int,
        bias: bool = True,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        factory_kwargs: Dict[str, Any] = {"device": device, "dtype": dtype}
        self.num_features = num_features
        self.weight = nn.Parameter(torch.empty(num_features, **factory_kwargs))
        if bias:
            self.bias = nn.Parameter(torch.empty(num_features, **factory_kwargs))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        torch.nn.init.ones_(self.weight)
        if self.bias is not None:
            torch.nn.init.zeros_(self.bias)

    def forward(self, x: Tensor) -> Tensor:
        return affine(x, self.weight, self.bias)

    def extra_repr(self) -> str:
        return f"num_features={self.num_features}, bias={self.bias is not None}"
