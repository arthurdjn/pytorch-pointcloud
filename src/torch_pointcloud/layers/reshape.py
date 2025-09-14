import torch.nn as nn
from torch import Tensor


class Reshape(nn.Module):
    """Reshapes a tensor to a new shape.
    This is a wrapper around the `torch.reshape` function.

    Use it instead of `torch.view` to avoid potential issues with in-place operations.

    Note:
        It is generally recommended to use `torch.view` instead of `torch.reshape`
        to avoid potential issues with in-place operations.
        Also, using a `torch.view` call is more efficient than a `torch.reshape` call.

    Args:
        *shape: The new shape of the tensor.
    """

    def __init__(self, *shape: int):
        super().__init__()
        self.shape = shape

    def forward(self, x: Tensor) -> Tensor:
        return x.reshape(*self.shape)

    def extra_repr(self) -> str:
        return f"{self.shape}"
