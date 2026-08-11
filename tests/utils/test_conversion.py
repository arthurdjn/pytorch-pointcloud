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
    convert_from_spconv_tensor,
    convert_to_numpy,
    convert_to_spconv_tensor,
    convert_to_tensor,
    cu_seqlens_to_batch,
    cu_seqlens_to_bincount,
    cu_seqlens_to_offset,
    ensure_tuple,
    offset_to_batch,
    offset_to_bincount,
    offset_to_cu_seqlens,
)
from torch_pointcloud.utils.imports import _SPCONV_AVAILABLE


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


def test_convert_to_tensor_supported_types() -> None:
    assert torch.equal(convert_to_tensor([1, 2, 3]), torch.tensor([1, 2, 3]))
    assert torch.equal(convert_to_tensor(np.array([1, 2, 3])), torch.tensor([1, 2, 3]))
    assert torch.equal(convert_to_tensor(4), torch.tensor(4))
    tensor = torch.tensor([1.0])
    assert convert_to_tensor(tensor) is tensor


def test_convert_to_tensor_dict_recursion() -> None:
    out = convert_to_tensor({"a": [1, 2], "b": {"c": 3}})
    assert torch.equal(out["a"], torch.tensor([1, 2]))
    assert torch.equal(out["b"]["c"], torch.tensor(3))


def test_convert_to_tensor_strict() -> None:
    with pytest.raises(TypeError, match="Unsupported data type"):
        convert_to_tensor("value")
    assert convert_to_tensor("value", strict=False) == "value"


def test_convert_to_numpy_supported_types() -> None:
    assert np.array_equal(convert_to_numpy([1, 2, 3]), np.array([1, 2, 3]))
    assert np.array_equal(convert_to_numpy(torch.tensor([1, 2, 3])), np.array([1, 2, 3]))
    assert np.array_equal(convert_to_numpy(4), np.array(4))
    array = np.array([1.0])
    assert convert_to_numpy(array) is array


def test_convert_to_numpy_dict_recursion() -> None:
    out = convert_to_numpy({"a": torch.tensor([1, 2]), "b": {"c": 3}})
    assert np.array_equal(out["a"], np.array([1, 2]))
    assert np.array_equal(out["b"]["c"], np.array(3))


def test_convert_to_numpy_strict() -> None:
    with pytest.raises(TypeError, match="Unsupported data type"):
        convert_to_numpy("value")
    assert convert_to_numpy("value", strict=False) == "value"


@pytest.mark.skipif(not _SPCONV_AVAILABLE, reason="spconv is not installed")
def test_convert_to_spconv_tensor_round_trip() -> None:
    x = torch.randn(5, 4)
    pos = torch.tensor([[0, 0, 0], [1, 2, 3], [4, 5, 6], [0, 1, 0], [2, 2, 2]])
    batch = torch.tensor([0, 0, 1, 1, 2])
    sparse = convert_to_spconv_tensor(x, pos, batch, spatial_shape=[8, 8, 8])
    assert sparse.batch_size == 3
    assert sparse.spatial_shape == [8, 8, 8]
    x_out, pos_out, batch_out = convert_from_spconv_tensor(sparse)
    assert torch.equal(x_out, x)
    assert torch.equal(pos_out, pos.int())
    assert torch.equal(batch_out, batch)


@pytest.mark.skipif(not _SPCONV_AVAILABLE, reason="spconv is not installed")
def test_convert_to_spconv_tensor_unsorted_batch() -> None:
    x = torch.randn(4, 2)
    pos = torch.randint(0, 8, (4, 3))
    batch = torch.tensor([1, 0, 1, 0])
    sparse = convert_to_spconv_tensor(x, pos, batch)
    assert sparse.batch_size == 2


@pytest.mark.skipif(not _SPCONV_AVAILABLE, reason="spconv is not installed")
def test_convert_to_spconv_tensor_explicit_batch_size_keeps_trailing_empty_scene() -> None:
    x = torch.randn(4, 2)
    pos = torch.randint(0, 8, (4, 3))
    batch = torch.zeros(4, dtype=torch.long)
    assert convert_to_spconv_tensor(x, pos, batch).batch_size == 1
    assert convert_to_spconv_tensor(x, pos, batch, batch_size=3).batch_size == 3


@pytest.mark.skipif(not _SPCONV_AVAILABLE, reason="spconv is not installed")
def test_convert_to_spconv_tensor_padding() -> None:
    x = torch.randn(2, 2)
    pos = torch.tensor([[1, 2, 3], [4, 5, 6]])
    batch = torch.zeros(2, dtype=torch.long)
    sparse = convert_to_spconv_tensor(x, pos, batch, padding=10)
    assert sparse.spatial_shape == [14, 15, 16]
