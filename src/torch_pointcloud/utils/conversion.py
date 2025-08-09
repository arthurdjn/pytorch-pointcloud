from collections.abc import Iterable
from typing import TYPE_CHECKING, Any, List, Optional, Sequence, Tuple, Type, TypeVar

import numpy as np
import torch
from torch import IntTensor, LongTensor, Tensor

from torch_pointcloud.utils.imports import optional_import

if TYPE_CHECKING:
    import spconv.pytorch as spconv
    from spconv.pytorc import SparseConvTensor

spconv, _ = optional_import("spconv.pytorch")
SparseConvTensor, _ = optional_import("spconv.pytorch", "SparseConvTensor")


def is_iterable(value: Any) -> bool:
    """Check if a value is iterable.
    A value is considered iterable if it is an instance of `Iterable` (i.e. `list`, `tuple`, `set`
    or any iterable defining the `__iter__` method) and not an instance of `str` or `bytes`.

    It is worth noting that `torch.Tensor` and `np.ndarray` are considered iterable (here)
    only if they have a dimension greater than 0 (i.e. not scalars).

    Args:
        value: The value to check.

    Returns:
        True if the value is iterable, False otherwise.
    """
    if isinstance(value, (str, bytes)):
        return False
    elif isinstance(value, (torch.Tensor, np.ndarray)) and value.ndim == 0:
        return False
    elif not isinstance(value, Iterable):
        return False
    return True


T = TypeVar("T", bound=Any)


def ensure_iterable(value: Any, type: Type[T], recursive: bool = False, none_as_empty: bool = False) -> T:
    if none_as_empty and value is None:
        return type([])
    if isinstance(value, np.ndarray):
        value = value.tolist() if value.ndim > 0 else value.item()
    elif isinstance(value, torch.Tensor):
        value = value.detach().cpu()
        value = value.tolist() if value.ndim > 0 else value.item()

    if is_iterable(value):
        if recursive:
            return type(ensure_iterable(v, type, recursive=True) if is_iterable(v) else v for v in value)
        return type(value)

    return type([value])


def ensure_iterable_size(
    value: Any,
    type: Type[T],
    size: int,
    recursive: bool = False,
    none_as_empty: bool = False,
    extra_msg: str = "",
) -> T:
    """Convert a value to a list of a given size.
    If the value is a scalar, it will be repeated to match the size.
    If the value's length does not match the size, an error will be raised.
    """
    value = ensure_iterable(value, type, recursive=recursive, none_as_empty=none_as_empty)
    if len(value) == 1:
        return type([value[0]] * size)
    elif len(value) == size:
        return value
    raise ValueError(f"Expected a {type.__name__} of size {size}, got {len(value)}. {extra_msg}")


def ensure_list(value: Any, recursive: bool = False, none_as_empty: bool = False) -> List[Any]:
    """Convert a value to a list. If the value is a numpy array or torch tensor,
    it will be converted to a list. If the value is a scalar, it will be wrapped in a list.

    Note:
        If the value is None, it will be wrapped in a list.

    Args:
        value: The value to convert.
        recursive: If True, the function will recursively apply itself to the elements of the list.
        none_as_empty: If True, and the value is None, an empty list will be returned.
        extra_msg: An additional message to include in the error.

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
        >>> ensure_list(None, none_as_empty=True)
        []
    """
    return ensure_iterable(value, list, recursive=recursive, none_as_empty=none_as_empty)


def ensure_list_size(
    value: Any, size: int, recursive: bool = False, none_as_empty: bool = False, extra_msg: str = ""
) -> List[Any]:
    """Convert a value to a list of a given size.
    If the value is a scalar, it will be repeated to match the size.
    If the value's length does not match the size, an error will be raised.
    """
    return ensure_iterable_size(
        value,
        list,
        size=size,
        recursive=recursive,
        none_as_empty=none_as_empty,
        extra_msg=extra_msg,
    )


def ensure_tuple(value: Any, recursive: bool = False, none_as_empty: bool = False) -> Tuple[Any, ...]:
    """Convert a value to a tuple. If the value is a scalar or single element,
    it will be wrapped in a tuple.

    Args:
        value: The value to convert.
        recursive: If True, the function will recursively apply itself to the elements of the tuple.
        none_as_empty: If True, and the value is None, an empty tuple will be returned.

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
    return ensure_iterable(value, tuple, recursive=recursive, none_as_empty=none_as_empty)


def ensure_tuple_size(
    value: Any,
    size: int,
    recursive: bool = False,
    none_as_empty: bool = False,
    extra_msg: str = "",
) -> Tuple[Any, ...]:
    """Convert a value to a tuple of a given size.
    If the value is a scalar, it will be repeated to match the size.
    If the value's length does not match the size, an error will be raised.

    Args:
        value: The value to convert.
        size: The size of the tuple.
        recursive: If True, the function will recursively apply itself to the elements of the tuple.
        none_as_empty: If True, and the value is None, an empty tuple will be returned.
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
    return ensure_iterable_size(
        value,
        tuple,
        size=size,
        recursive=recursive,
        none_as_empty=none_as_empty,
        extra_msg=extra_msg,
    )


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


def packed_to_spconv_tensor(
    x: Tensor,
    pos: IntTensor,
    batch: Tensor,
    spatial_shape: Optional[Sequence[int]] = None,
    padding: int = 96,
) -> SparseConvTensor:
    """Convert point features and coordinates to `spconv.SparseConvTensor` sparse tensor.

    Args:
        x: The point features.
        pos: The point coordinates.
        batch: The batch indices of the points.
        spatial_shape: The spatial shape of the sparse tensor.

    Returns:
        The `spconv.SparseConvTensor` sparse tensor.
    """
    if spatial_shape is None:
        spatial_shape = torch.add(torch.max(pos, dim=0).values, padding).tolist()

    return spconv.SparseConvTensor(
        features=x,
        indices=torch.cat([batch.unsqueeze(-1).int(), pos.int()], dim=1).contiguous(),
        spatial_shape=spatial_shape,
        batch_size=batch[-1].item() + 1,
    )


def spconv_tensor_to_packed(spconv_tensor: SparseConvTensor) -> Tuple[Tensor, IntTensor, LongTensor]:
    x = spconv_tensor.features
    indices = spconv_tensor.indices

    batch = indices[:, 0]
    pos = indices[:, 1:]

    pos = pos.int()
    batch = batch.long()
    return x, pos, batch
