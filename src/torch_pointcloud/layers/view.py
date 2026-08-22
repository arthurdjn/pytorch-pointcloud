"""Tensor reshaping as a module wrapping `Tensor.view`."""

import torch.nn as nn
from torch import Tensor


class View(nn.Module):
    r"""Returns a new view of the tensor with the same data.
    This is a wrapper around the `torch.view` function.

    Note:
        This will not create a copy of the tensor and is generally more efficient than a `torch.reshape` call.
        However, it requires the input tensor to have a contiguous memory layout.

    Args:
        *shape: The new shape of the tensor.
    """

    def __init__(self, *shape: int):
        super().__init__()
        self.shape = shape

    def forward(self, x: Tensor) -> Tensor:
        return x.view(*self.shape)

    def extra_repr(self) -> str:
        return f"{self.shape}"
