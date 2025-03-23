from collections.abc import Iterable
from typing import Any, Tuple

import numpy as np
import torch
from torch import Tensor


def ensure_tuple(value: Any) -> Tuple[Any, ...]:
    if isinstance(value, np.ndarray):
        if value.ndim > 0:
            return tuple(value.tolist())
        return tuple([value.item()])
    elif isinstance(value, torch.Tensor):
        value = value.detach().cpu()
        if value.ndim > 0:
            return tuple(value.tolist())
        return tuple([value.item()])
    elif isinstance(value, (str, bytes)):
        return tuple([value])
    elif isinstance(value, Iterable):
        return tuple(value)
    return tuple([value])


def ensure_tuple_size(value: Any, size: int) -> Tuple[Any, ...]:
    value = ensure_tuple(value)
    if len(value) == 1:
        return tuple([value[0]] * size)
    elif len(value) == size:
        return value
    else:
        raise ValueError(f"Expected a tuple of size {size}, got {len(value)}")


@torch.no_grad()
def offset_to_bincount(offset: Tensor) -> Tensor:
    """Convert an offset tensor to a bincount tensor.

    Args:
        offset: The offset tensor.

    Returns:
        The bincount tensor.

    Examples:
        >>> import torch
        >>> offset = torch.tensor([4, 7, 12])
        >>> offset_to_bincount(offset)
        tensor([4, 3, 5])
    """
    return torch.diff(offset, prepend=torch.tensor([0], device=offset.device, dtype=torch.long))


@torch.no_grad()
def bincount_to_offset(bincount: Tensor) -> Tensor:
    """Convert a bincount tensor to an offset tensor.

    Args:
        bincount: The bincount tensor.

    Returns:
        The offset tensor.

    Examples:
        >>> import torch
        >>> bincount = torch.tensor([4, 3, 5])
        >>> bincount_to_offset(bincount)
        tensor([4, 7, 12])
    """
    return torch.cumsum(bincount, dim=0)


@torch.no_grad()
def batch_to_offset(batch_idx: Tensor) -> Tensor:
    """Convert a batch index tensor to an offset tensor.

    Args:
        batch_idx: The batch indices of the points.

    Returns:
        The offset tensor.

    Examples:
        >>> import torch
        >>> batch_idxs = torch.tensor([0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 2, 2])
        >>> batch_to_offset(batch_idxs)
        tensor([4, 7, 12])
    """
    bincount = torch.bincount(batch_idx)
    return torch.cumsum(bincount, dim=0)


@torch.no_grad()
def offset_to_batch(offset: Tensor) -> Tensor:
    """Convert an offset tensor to a batch index tensor.

    Args:
        offset: The offset tensor.

    Returns:
        The batch index tensor.

    Examples:
        >>> import torch
        >>> offset = torch.tensor([4, 7, 12])
        >>> offset_to_batch(offset)
        tensor([0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 2, 2])
    """
    device, dtype = offset.device, offset.dtype
    batch_sizes = offset_to_bincount(offset)
    return torch.repeat_interleave(torch.arange(len(batch_sizes), device=device, dtype=dtype), batch_sizes)


@torch.no_grad()
def batch_to_cu_seqlens(batch_idx: Tensor) -> Tensor:
    """Convert a batch index tensor to a cumulative sequence lengths tensor.

    A cumulative sequence length is a tensor of shape $ (N + 2,)$ where $N$ is the size of the batch.
    The first element is 0 and the last element is the size of the batch.
    The other elements are the cumulative sum of the batch sizes.

    See Also:
        This function is related to `batch_to_offset`.

    Args:
        batch_idx: The batch indices of the points.

    Returns:
        The cumulative sequence lengths tensor.

    Examples:
        >>> import torch
        >>> batch_idxs = torch.tensor([0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 2, 2])
        >>> batch_to_cu_seqlens(batch_idxs)
        tensor([0, 4, 7, 12])
    """
    device, dtype = batch_idx.device, batch_idx.dtype
    first = torch.tensor([0], device=device, dtype=dtype)
    last = torch.tensor([len(batch_idx)], device=device, dtype=dtype)
    offset = batch_to_offset(batch_idx)
    return torch.cat([first, offset, last])
