from typing import Any, Tuple
from unittest.mock import ANY, Mock, patch, sentinel

import pytest
import torch
from torch import Tensor

from torch_pointcloud.utils.serialization import serialize_coords


class TensorArg:
    def __init__(self, tensor: Tensor) -> None:
        self.tensor = tensor

    def __eq__(self, other: Any) -> bool:
        return torch.equal(self.tensor, other)


@pytest.fixture
def grid_coords() -> Tensor:
    return torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.long)


@pytest.fixture
def batch() -> Tensor:
    return torch.tensor([0, 1], dtype=torch.long)


@pytest.fixture
def depth() -> int:
    return 5


@patch("torch_pointcloud.utils.serialization.octree_encode")
def test_z_order_encoding(mock_octree: Mock, grid_coords: Tensor, batch: Tensor, depth: int) -> None:
    serialize_coords(grid_coords, batch, depth, order="z")
    mock_octree.assert_called_once_with(ANY, ANY, ANY, b=None, depth=depth)

    args = mock_octree.call_args[0]
    assert torch.equal(args[0], grid_coords[:, 0].long())
    assert torch.equal(args[1], grid_coords[:, 1].long())
    assert torch.equal(args[2], grid_coords[:, 2].long())


@patch("torch_pointcloud.utils.serialization.octree_encode")
def test_z_order_trans_encoding(mock_octree: Mock, grid_coords: Tensor, batch: Tensor, depth: int) -> None:
    serialize_coords(grid_coords, batch, depth, order="z-trans")
    mock_octree.assert_called_once_with(ANY, ANY, ANY, b=None, depth=depth)

    args = mock_octree.call_args[0]
    assert torch.equal(args[0], grid_coords[:, 1].long())
    assert torch.equal(args[1], grid_coords[:, 0].long())
    assert torch.equal(args[2], grid_coords[:, 2].long())


@patch("torch_pointcloud.utils.serialization.hilbert_encode")
def test_hilbert_encoding(mock_hilbert: Mock, grid_coords: Tensor, batch: Tensor, depth: int) -> None:
    serialize_coords(grid_coords, batch, depth, order="hilbert")
    mock_hilbert.assert_called_once_with(ANY, num_dims=3, num_bits=depth)

    args = mock_hilbert.call_args[0]
    assert len(args) == 1
    assert torch.equal(args[0], grid_coords)


@patch("torch_pointcloud.utils.serialization.hilbert_encode")
def test_hilbert_trans_encoding(mock_hilbert: Mock, grid_coords: Tensor, batch: Tensor, depth: int) -> None:
    serialize_coords(grid_coords, batch, depth, order="hilbert-trans")
    mock_hilbert.assert_called_once_with(ANY, num_dims=3, num_bits=depth)

    args = mock_hilbert.call_args[0]
    assert len(args) == 1
    assert torch.equal(args[0], grid_coords[:, [1, 0, 2]])


def test_invalid_order(grid_coords: Tensor, batch: Tensor, depth: int) -> None:
    with pytest.raises(ValueError, match="Unsupported serialization order"):
        serialize_coords(grid_coords, batch, depth, order="invalid")
