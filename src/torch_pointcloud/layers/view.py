import torch.nn as nn
from torch import Tensor


class View(nn.Module):
    r"""
    Views a tensor to a new shape.
    This is a wrapper around the `torch.view` function.

    Args:
        *size: The new shape of the tensor.
    """

    def __init__(self, *size: int):
        super().__init__()
        self.size = size

    def forward(self, x: Tensor) -> Tensor:
        x = x.view(*self.size)
        return x

    def extra_repr(self) -> str:
        return f"{self.size}"
