from typing import Any, Iterable, Tuple, Union

import numpy as np
import pytest
import torch
from torch import Tensor

from torch_pointcloud.utils.conversion import (
    batch_to_bincount,
    batch_to_cu_seqlens,
    batch_to_offset,
    bincount_to_batch,
    bincount_to_cu_seqlens,
    bincount_to_offset,
    cu_seqlens_to_batch,
    cu_seqlens_to_bincount,
    cu_seqlens_to_offset,
    ensure_tuple,
    offset_to_batch,
    offset_to_bincount,
    offset_to_cu_seqlens,
)


@pytest.fixture
def batch_tensor() -> Tensor:
    return torch.tensor([0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 2, 2])


@pytest.fixture
def offset_tensor() -> Tensor:
    return torch.tensor([4, 7, 12])


@pytest.fixture
def bincount_tensor() -> Tensor:
    return torch.tensor([4, 3, 5])


@pytest.fixture
def cu_seqlens_tensor() -> Tensor:
    return torch.tensor([0, 4, 7, 12])


def test_ensure_tuple_numpy_scalar() -> None:
    """Test conversion of numpy scalar to tuple"""
    value = np.array(5)
    result = ensure_tuple(value)
    assert result == (5,)


def test_ensure_tuple_numpy_array() -> None:
    """Test conversion of numpy array to tuple"""
    value = np.array([1, 2, 3])
    result = ensure_tuple(value)
    assert result == (1, 2, 3)


def test_ensure_tuple_multidimensional_numpy_array() -> None:
    """Test conversion of multidimensional numpy array"""
    value = np.array([[1, 2], [3, 4]])
    result = ensure_tuple(value)
    assert result == ([1, 2], [3, 4])


def test_ensure_tuple_torch_scalar() -> None:
    """Test conversion of torch scalar to tuple"""
    value = torch.tensor(5)
    result = ensure_tuple(value)
    assert result == (5,)


def test_ensure_tuple_torch_tensor() -> None:
    """Test conversion of torch tensor to tuple"""
    value = torch.tensor([1, 2, 3])
    result = ensure_tuple(value)
    assert result == (1, 2, 3)


def test_ensure_tuple_multidimensional_torch_tensor() -> None:
    """Test conversion of multidimensional torch tensor"""
    value = torch.tensor([[1, 2], [3, 4]])
    result = ensure_tuple(value)
    assert result == ([1, 2], [3, 4])


@pytest.mark.parametrize("value", ["test", b"test_ensure_tuple_bytes"])
def test_ensure_tuple_string_types(value: Union[str, bytes]) -> None:
    """Test conversion of string and bytes to tuple"""
    result = ensure_tuple(value)
    assert result == (value,)


@pytest.mark.parametrize(
    "value, expected_value",
    [
        ([1, 2, 3], (1, 2, 3)),
        ({1, 2, 3}, (1, 2, 3)),
        (range(3), (0, 1, 2)),
    ],
)
def test_ensure_tuple_iterables(value: Iterable, expected_value: Tuple) -> None:
    """Test conversion of various iterables to tuple"""
    result = ensure_tuple(value)
    assert result == expected_value


@pytest.mark.parametrize("value", [42, 3.14, True, None])
def test_ensure_tuple_scalar_values(value: Any) -> None:
    """Test conversion of non-iterable scalar values to tuple"""
    result = ensure_tuple(value)
    assert result == (value,)


def test_ensure_tuple_empty_iterable() -> None:
    """Test conversion of empty iterables"""
    result = ensure_tuple([])
    assert result == ()


def test_ensure_tuple_none_as_empty() -> None:
    """Test conversion of None to empty tuple"""
    result = ensure_tuple(None, none_as_empty=False)
    assert result == (None,)

    result = ensure_tuple(None, none_as_empty=True)
    assert result == ()

    # Make sure that other values that can evaluate to False
    # are indeed not considered as empty
    result = ensure_tuple(0, none_as_empty=True)
    assert result == (0,)

    result = ensure_tuple("", none_as_empty=True)
    assert result == ("",)


def test_ensure_tuple_custom_type() -> None:
    """Test conversion of custom type"""

    class CustomType:
        pass

    value = CustomType()
    result = ensure_tuple(value)
    assert result == (value,)
    assert result == (value,)


def test_offset_conversions(
    cu_seqlens_tensor: Tensor,
    bincount_tensor: Tensor,
    offset_tensor: Tensor,
    batch_tensor: Tensor,
) -> None:
    """Test all conversions starting from offset tensor."""
    result = offset_to_bincount(offset_tensor)
    assert torch.equal(result, bincount_tensor)

    result = offset_to_batch(offset_tensor)
    assert torch.equal(result, batch_tensor)

    result = offset_to_cu_seqlens(offset_tensor)
    assert torch.equal(result, cu_seqlens_tensor)


def test_bincount_conversions(
    cu_seqlens_tensor: Tensor,
    bincount_tensor: Tensor,
    offset_tensor: Tensor,
    batch_tensor: Tensor,
) -> None:
    """Test all conversions starting from bincount tensor."""
    result = bincount_to_offset(bincount_tensor)
    assert torch.equal(result, offset_tensor)

    result = bincount_to_batch(bincount_tensor)
    assert torch.equal(result, batch_tensor)

    result = bincount_to_cu_seqlens(bincount_tensor)
    assert torch.equal(result, cu_seqlens_tensor)


def test_cu_seqlens_conversions(
    cu_seqlens_tensor: Tensor,
    bincount_tensor: Tensor,
    offset_tensor: Tensor,
    batch_tensor: Tensor,
) -> None:
    """Test all conversions starting from cu_seqlens tensor."""
    result = cu_seqlens_to_offset(cu_seqlens_tensor)
    assert torch.equal(result, offset_tensor)

    result = cu_seqlens_to_bincount(cu_seqlens_tensor)
    assert torch.equal(result, bincount_tensor)

    result = cu_seqlens_to_batch(cu_seqlens_tensor)
    assert torch.equal(result, batch_tensor)


def test_batch_conversions(
    cu_seqlens_tensor: Tensor,
    bincount_tensor: Tensor,
    offset_tensor: Tensor,
    batch_tensor: Tensor,
) -> None:
    """Test all conversions starting from batch tensor."""
    result = batch_to_offset(batch_tensor)
    assert torch.equal(result, offset_tensor)

    result = batch_to_bincount(batch_tensor)
    assert torch.equal(result, bincount_tensor)

    result = batch_to_cu_seqlens(batch_tensor)
    assert torch.equal(result, cu_seqlens_tensor)
