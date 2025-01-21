from typing import Any, Iterable, Tuple, Union

import numpy as np
import pytest
import torch

from torch_pointcloud.utils.conversion import ensure_tuple


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


def test_ensure_tuple_custom_type() -> None:
    """Test conversion of custom type"""

    class CustomType:
        pass

    value = CustomType()
    result = ensure_tuple(value)
    assert result == (value,)
