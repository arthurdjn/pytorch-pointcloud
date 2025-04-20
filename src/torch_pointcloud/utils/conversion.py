from collections.abc import Iterable
from typing import Any, List, Tuple

import numpy as np
import torch
from torch import Tensor


def is_iterable(value: Any) -> bool:
    """Check if a value is iterable.
    A value is considered iterable if it is an instance of `Iterable` (i.e. `list`, `tuple`, `set`
    or any iterable defining the `__iter__` method) and not an instance of `str` or `bytes`.

    Args:
        value: The value to check.

    Returns:
        True if the value is iterable, False otherwise.
    """
    return isinstance(value, Iterable) and not isinstance(value, (str, bytes))


def ensure_list(value: Any, recursive: bool = False) -> List[Any]:
    """Convert a value to a list. If the value is a numpy array or torch tensor,
    it will be converted to a list. If the value is a scalar, it will be wrapped in a list.

    Note:
        If the value is None, it will be wrapped in a list.

    Args:
        value: The value to convert.
        recursive: If True, the function will recursively apply itself to the elements of the list.

    Returns:
        The value as a list.

    Examples:
        >>> ensure_list(1)
        [1]
        >>> ensure_list([1, 2, 3])
        [1, 2, 3]
        >>> ensure_list(np.array([1, 2, 3]))
        [1, 2, 3]
        >>> ensure_list(torch.tensor([1, 2, 3]))
        [1, 2, 3]
        >>> ensure_list(None)
        [None]
    """
    if isinstance(value, np.ndarray):
        value = value.tolist() if value.ndim > 0 else value.item()
    elif isinstance(value, torch.Tensor):
        value = value.detach().cpu()
        value = value.tolist() if value.ndim > 0 else value.item()

    if is_iterable(value):
        if recursive:
            return list(ensure_list(v, recursive=True) if is_iterable(v) else v for v in value)
        else:
            return list(value)

    return list([value])


def ensure_tuple(value: Any, recursive: bool = False) -> Tuple[Any, ...]:
    """Convert a value to a tuple. If the value is a scalar or single element,
    it will be wrapped in a tuple.

    Args:
        value: The value to convert.

    Returns:
        The value as a tuple.

    Examples:
        >>> ensure_tuple(1)
        (1,)
        >>> ensure_tuple([1, 2, 3])
        (1, 2, 3)
        >>> ensure_tuple("test")
        ('test',)
        >>> ensure_tuple(np.array([1, 2, 3]))
        (1, 2, 3)
        >>> ensure_tuple(torch.tensor([1, 2, 3], device="cuda"))
        (1, 2, 3)
    """

    value = ensure_list(value, recursive=recursive)
    if recursive:
        return tuple(ensure_tuple(v, recursive=True) if is_iterable(v) else v for v in value)
    return tuple(value)


def ensure_tuple_size(value: Any, size: int, recursive: bool = False, extra_msg: str = "") -> Tuple[Any, ...]:
    """Convert a value to a tuple of a given size.
    If the value is a scalar, it will be repeated to match the size.
    If the value's length does not match the size, an error will be raised.

    Args:
        value: The value to convert.
        size: The size of the tuple.
        recursive: If True, the function will recursively apply itself to the elements of the tuple.
        extra_msg: An additional message to include in the error.

    Returns:
        The value as a tuple of the given size.

    Examples:
        >>> ensure_tuple_size(1, 3)
        (1, 1, 1)
        >>> ensure_tuple_size([1, 2, 3], 3)
        (1, 2, 3)
        >>> ensure_tuple_size(torch.tensor([1, 2, 3]), 3)
        (1, 2, 3)
        >>> ensure_tuple_size(np.array([1, 2, 3]), 3)
        (1, 2, 3)
    """
    value = ensure_tuple(value, recursive=recursive)
    if len(value) == 1:
        return tuple([value[0]] * size)
    elif len(value) == size:
        return value
    else:
        raise ValueError(f"Expected a tuple of size {size}, got {len(value)}. {extra_msg}")


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
def offset_to_cu_seqlens(offset: Tensor) -> Tensor:
    """Convert an offset tensor to a cumulative sequence lengths tensor.
    A cumulative sequence length is a tensor of shape $ (N + 2,)$ where $N$ is the size of the batch.
    The first element is 0 and the last element is the size of the batch.
    The other elements are the cumulative sum of the batch sizes.

    Note:
        This function was provided to ease the conversion between `offset`
        tensor and `cu_seqlens` tensor format required by `spconv`.

    See Also:
        This function is related to `offset_to_batch`.

    Args:
        offset: The offset tensor.

    Returns:
        The cumulative sequence lengths tensor.

    Examples:
        >>> import torch
        >>> offset = torch.tensor([4, 7, 12])
        >>> offset_to_cu_seqlens(offset)
        tensor([0, 4, 7, 12])
    """
    device, dtype = offset.device, offset.dtype
    return torch.cat([torch.tensor([0], device=device, dtype=dtype), offset])


@torch.no_grad()
def batch_to_offset(batch: Tensor) -> Tensor:
    """Convert a batch index tensor to an offset tensor.

    Args:
        batch: The batch indices of the points.

    Returns:
        The offset tensor.

    Examples:
        >>> import torch
        >>> batch = torch.tensor([0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 2, 2])
        >>> batch_to_offset(batch)
        tensor([4, 7, 12])
    """
    bincount = torch.bincount(batch)
    return torch.cumsum(bincount, dim=0)


@torch.no_grad()
def batch_to_bincount(batch: Tensor) -> Tensor:
    """Convert a batch index tensor to a bincount tensor.

    Args:
        batch: The batch indices of the points.

    Returns:
        The bincount tensor.

    Examples:
        >>> import torch
        >>> batch = torch.tensor([0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 2, 2])
        >>> batch_to_bincount(batch)
        tensor([4, 3, 5])
    """
    return torch.bincount(batch)


@torch.no_grad()
def batch_to_cu_seqlens(batch: Tensor) -> Tensor:
    """Convert a batch index tensor to a cumulative sequence lengths tensor.
    A cumulative sequence length is a tensor of shape $ (N + 2,)$ where $N$ is the size of the batch.
    The first element is 0 and the last element is the size of the batch.
    The other elements are the cumulative sum of the batch sizes.

    Note:
        This function was provided to ease the conversion between `batch`
        tensor and `cu_seqlens` tensor format required by `spconv`.

    See Also:
        This function is related to `batch_to_offset`.

    Args:
        batch: The batch indices of the points.

    Returns:
        The cumulative sequence lengths tensor.

    Examples:
        >>> import torch
        >>> batch = torch.tensor([0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 2, 2])
        >>> batch_to_cu_seqlens(batch)
        tensor([0, 4, 7, 12])
    """
    offset = batch_to_offset(batch)
    return offset_to_cu_seqlens(offset)


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
def bincount_to_batch(bincount: Tensor) -> Tensor:
    """Convert a bincount tensor to a batch index tensor.

    Args:
        bincount: The bincount tensor.

    Returns:
        The batch index tensor.

    Examples:
        >>> import torch
        >>> bincount = torch.tensor([4, 3, 5])
        >>> bincount_to_batch(bincount)
        tensor([0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 2, 2])
    """
    device, dtype = bincount.device, bincount.dtype
    return torch.repeat_interleave(torch.arange(len(bincount), device=device, dtype=dtype), bincount)


@torch.no_grad()
def bincount_to_cu_seqlens(bincount: Tensor) -> Tensor:
    """Convert a bincount tensor to a cumulative sequence lengths tensor.

    Args:
        bincount: The bincount tensor.

    Returns:
        The cumulative sequence lengths tensor.

    Examples:
        >>> import torch
        >>> bincount = torch.tensor([4, 3, 5])
        >>> bincount_to_cu_seqlens(bincount)
        tensor([0, 4, 7, 12])
    """
    offset = bincount_to_offset(bincount)
    return offset_to_cu_seqlens(offset)


@torch.no_grad()
def cu_seqlens_to_offset(cu_seqlens: Tensor) -> Tensor:
    """Convert a cumulative sequence lengths tensor to an offset tensor.

    Args:
        cu_seqlens: The cumulative sequence lengths tensor.

    Returns:
        The offset tensor.

    Examples:
        >>> import torch
        >>> cu_seqlens = torch.tensor([0, 4, 7, 12])
        >>> cu_seqlens_to_offset(cu_seqlens)
        tensor([4, 7, 12])
    """
    return cu_seqlens[1:]


@torch.no_grad()
def cu_seqlens_to_bincount(cu_seqlens: Tensor) -> Tensor:
    """Convert a cumulative sequence lengths tensor to a bincount tensor.

    Args:
        cu_seqlens: The cumulative sequence lengths tensor.

    Returns:
        The bincount tensor.

    Examples:
        >>> import torch
        >>> cu_seqlens = torch.tensor([0, 4, 7, 12])
        >>> cu_seqlens_to_bincount(cu_seqlens)
        tensor([4, 3, 5])
    """
    offset = cu_seqlens_to_offset(cu_seqlens)
    return offset_to_bincount(offset)


@torch.no_grad()
def cu_seqlens_to_batch(cu_seqlens: Tensor) -> Tensor:
    """Convert a cumulative sequence lengths tensor to a batch index tensor.

    Args:
        cu_seqlens: The cumulative sequence lengths tensor.

    Returns:
        The batch index tensor.

    Examples:
        >>> import torch
        >>> cu_seqlens = torch.tensor([0, 4, 7, 12])
        >>> cu_seqlens_to_batch(cu_seqlens)
        tensor([0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 2, 2])
    """
    offset = cu_seqlens_to_offset(cu_seqlens)
    return offset_to_batch(offset)
