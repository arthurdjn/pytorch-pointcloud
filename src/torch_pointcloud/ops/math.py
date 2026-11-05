"""Tensor math: division with a default for zero denominators and a straight-through hard softmax."""

from typing import Union

import torch
from torch import Tensor


def safe_divide(a: Tensor, b: Tensor, /, default: Union[float, Tensor] = float("nan")) -> Tensor:
    """Safely divide two tensors, returning a default value if the denominator is zero.

    !!! note
        If the inputs are not floating point numbers,
        they will be converted to floating point numbers (float32).

    Args:
        a: The numerator tensor.
        b: The denominator tensor.
        default: The default value to return if the denominator is zero.

    Returns:
        The result of the division.

    Example:
        ```pycon
        >>> safe_divide(torch.tensor([1.0, 2.0, 3.0]), torch.tensor([1.0, 0.0, 1.0]))
        tensor([1., nan, 3.])
        >>> safe_divide(torch.tensor([1.0, 2.0, 3.0]), torch.tensor([1.0, 0.0, 1.0]), default=0.0)
        tensor([1., 0., 3.])
        >>> safe_divide(torch.tensor([1, 2, 3]), torch.tensor([1, 0, 1]), default=torch.tensor([0, 0, 0]))
        tensor([1., 0., 3.])

        ```
    """
    if not isinstance(default, Tensor):
        default = torch.full(a.shape, default, device=a.device)

    a = a if torch.is_floating_point(a) else a.float()
    b = b if torch.is_floating_point(b) else b.float()
    default = default if torch.is_floating_point(default) else default.float()
    return torch.where(b != 0, a / b, default)


def hard_softmax(logits: Tensor, dim: int = -1) -> Tensor:
    r"""One-hot of the largest logit in the forward pass, with the softmax gradient (straight-through estimator).

    Args:
        logits: Unnormalized scores.
        dim: Dimension to normalize.

    Returns:
        A one-hot tensor shaped like `logits`.

    Shape:
        Input: $(*, k)$
        Output: $(*, k)$

    Example:
        ```pycon
        >>> import torch
        >>> from torch_pointcloud.ops.math import hard_softmax
        >>> hard_softmax(torch.tensor([[0.1, 2.0, -1.0]])).tolist()
        [[0.0, 1.0, 0.0]]

        ```
    """
    soft = logits.softmax(dim=dim)
    hard = torch.zeros_like(soft).scatter_(dim, soft.argmax(dim=dim, keepdim=True), 1.0)
    return (hard - soft).detach() + soft
