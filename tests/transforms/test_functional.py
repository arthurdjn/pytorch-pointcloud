from typing import Tuple
from unittest.mock import MagicMock, Mock, patch

import pytest
import torch
from torch import Tensor

from torch_pointcloud.transforms.functional import normalize_scale, random_sample, random_sample_vertices


@pytest.fixture
def sample_points() -> Tensor:
    return torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )


@pytest.fixture
def sample_mesh() -> Tuple[Tensor, Tensor]:
    vertices = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    faces = torch.tensor(
        [
            [0, 1, 2],
            [0, 2, 3],
        ]
    )
    return vertices, faces


@patch("torch_pointcloud.transforms.functional.torch.randint")
def test_random_sample(mock_randint: Mock) -> None:
    """Test that the random sample uses torch.randint correctly to sample indices."""
    tensor = MagicMock()
    num_samples = 10

    result = random_sample(tensor, num_samples)
    mock_randint.assert_called_once_with(0, tensor.size(0), (num_samples,), generator=None)
    assert tensor[mock_randint.return_value] is result


@patch("torch_pointcloud.transforms.functional.torch.Generator")
@patch("torch_pointcloud.transforms.functional.torch.randint")
def test_random_sample_with_seed(mock_randint: Mock, mock_generator: Mock) -> None:
    """Test that the random sample uses torch.randint correctly to sample indices with seed."""
    tensor = MagicMock()
    num_samples = 10
    seed = 42
    generator = MagicMock()
    mock_generator.return_value = generator

    result = random_sample(tensor, num_samples, seed=seed)

    mock_generator.assert_called_once()
    generator.manual_seed.assert_called_once_with(seed)
    mock_randint.assert_called_once_with(0, tensor.size(0), (num_samples,), generator=generator)
    assert tensor[mock_randint.return_value] is result


def test_random_sample_vertices(sample_mesh: Tuple[Tensor, Tensor]) -> None:
    """Test that the random sample vertices function returns the correct shape."""
    vertices, faces = sample_mesh
    num_samples = 10

    sampled = random_sample_vertices(vertices, faces, num_samples)
    assert sampled.shape == (num_samples, 3)


def test_random_sample_vertices_with_normals(sample_mesh: Tuple[Tensor, Tensor]) -> None:
    """Test that the random sample vertices function returns the correct shape with normals."""
    vertices, faces = sample_mesh
    num_samples = 10

    sampled, normals = random_sample_vertices(vertices, faces, num_samples, return_normals=True)
    assert sampled.shape == (num_samples, 3)
    assert normals.shape == (num_samples, 3)
    # Check normals are normalized
    assert torch.allclose(torch.norm(normals, dim=1), torch.ones(num_samples))


def test_random_sample_vertices_with_indices(sample_mesh: Tuple[Tensor, Tensor]) -> None:
    """Test that the random sample vertices function returns the correct shape with indices."""
    vertices, faces = sample_mesh
    num_samples = 10

    sampled, indices = random_sample_vertices(vertices, faces, num_samples, return_indices=True)
    assert sampled.shape == (num_samples, 3)
    assert indices.shape == (num_samples,)


def test_random_sample_vertices_with_all_returns(sample_mesh: Tuple[Tensor, Tensor]) -> None:
    """Test that the random sample vertices function returns the correct shape with normals and indices."""
    vertices, faces = sample_mesh
    num_samples = 10

    sampled, normals, indices = random_sample_vertices(
        vertices, faces, num_samples, return_normals=True, return_indices=True
    )
    assert sampled.shape == (num_samples, 3)
    assert normals.shape == (num_samples, 3)
    assert indices.shape == (num_samples,)


def test_normalize_scale(sample_points: Tensor) -> None:
    """Test that the normalize scale function returns the correct shape."""
    normalized = normalize_scale(sample_points)
    expected = torch.tensor(
        [
            [-0.3333, -0.3333, -0.3333],
            [1.0000, -0.3333, -0.3333],
            [-0.3333, 1.0000, -0.3333],
            [-0.3333, -0.3333, 1.0000],
        ]
    )

    assert torch.allclose(normalized, expected, atol=1e-4)
