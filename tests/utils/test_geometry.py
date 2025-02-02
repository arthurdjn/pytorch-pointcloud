import math

import pytest
import torch
from torch import Tensor

from torch_pointcloud.utils.geometry import axis_aligned_bounding_box, cross_product_matrix, rodrigues_rotation_matrix


@pytest.mark.parametrize(
    "points, expected",
    [
        (  # Unit cube
            torch.tensor(
                [
                    [0.0, 0.0, 0.0],  # front bottom left
                    [0.0, 0.0, 1.0],  # front top left
                    [0.0, 1.0, 0.0],  # back bottom left
                    [0.0, 1.0, 1.0],  # back top left
                    [1.0, 0.0, 0.0],  # front bottom right
                    [1.0, 0.0, 1.0],  # front top right
                    [1.0, 1.0, 0.0],  # back bottom right
                    [1.0, 1.0, 1.0],  # back top right
                ]
            ),
            torch.tensor([0.5, 0.5, 0.5, 1.0, 1.0, 1.0]),
        ),
        (  # Translated cube
            torch.tensor(
                [
                    [1.0, 1.0, 1.0],
                    [1.0, 1.0, 2.0],
                    [1.0, 2.0, 1.0],
                    [1.0, 2.0, 2.0],
                    [2.0, 1.0, 1.0],
                    [2.0, 1.0, 2.0],
                    [2.0, 2.0, 1.0],
                    [2.0, 2.0, 2.0],
                ]
            ),
            torch.tensor([1.5, 1.5, 1.5, 1.0, 1.0, 1.0]),
        ),
    ],
)
def test_axis_aligned_bounding_box(points: Tensor, expected: Tensor) -> None:
    """Test that the bounding box of a point cloud is correct."""
    bbox = axis_aligned_bounding_box(points)
    assert torch.allclose(bbox, expected)


def test_axis_aligned_bounding_box_single_point() -> None:
    """Test that the bounding box of a single point is the point itself."""
    points = torch.tensor([[1.0, 2.0, 3.0]])
    expected = torch.tensor([1.0, 2.0, 3.0, 0.0, 0.0, 0.0])

    bbox = axis_aligned_bounding_box(points)
    assert torch.allclose(bbox, expected)


@pytest.mark.parametrize(
    "k, expected",
    [
        (
            torch.tensor([1.0, 0.0, 0.0]),
            torch.tensor(
                [
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, -1.0],
                    [0.0, 1.0, 0.0],
                ]
            ),
        ),
        (
            torch.tensor([0.0, 1.0, 0.0]),
            torch.tensor(
                [
                    [0.0, 0.0, 1.0],
                    [0.0, 0.0, 0.0],
                    [-1.0, 0.0, 0.0],
                ]
            ),
        ),
        (
            torch.tensor([0.0, 0.0, 1.0]),
            torch.tensor(
                [
                    [0.0, -1.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                ]
            ),
        ),
    ],
)
def test_cross_product_matrix_unit_vector(k: Tensor, expected: Tensor) -> None:
    """Test cross product matrix with unit vector."""
    matrix = cross_product_matrix(k)
    assert torch.allclose(matrix, expected)


def test_cross_product_matrix_product_equivalence() -> None:
    """Test that the cross product matrix and the cross product are equivalent."""
    v1 = torch.tensor([1.0, 2.0, 3.0])
    v2 = torch.tensor([4.0, 5.0, 6.0])
    cross_matrix = torch.matmul(cross_product_matrix(v1), v2)
    cross_direct = torch.cross(v1, v2, dim=0)
    assert torch.allclose(cross_matrix, cross_direct)


@pytest.mark.parametrize(
    "axis, theta, expected",
    [
        (
            torch.tensor([1.0, 0.0, 0.0]),  # x-axis
            math.pi / 2,
            torch.tensor(
                [
                    [1.0, 0.0, 0.0],
                    [0.0, 0.0, -1.0],
                    [0.0, 1.0, 0.0],
                ]
            ),
        ),
        (
            torch.tensor([0.0, 1.0, 0.0]),  # y-axis
            math.pi / 2,
            torch.tensor(
                [
                    [0.0, 0.0, 1.0],
                    [0.0, 1.0, 0.0],
                    [-1.0, 0.0, 0.0],
                ]
            ),
        ),
        (
            torch.tensor([0.0, 0.0, 1.0]),  # z-axis
            math.pi / 2,
            torch.tensor(
                [
                    [0.0, -1.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0],
                ]
            ),
        ),
    ],
)
def test_rodrigues_rotation_matrix_principal_axes(axis: Tensor, theta: float, expected: Tensor) -> None:
    """Test that the rotation matrix is correct for the principal axes."""
    R = rodrigues_rotation_matrix(axis, theta)
    assert torch.allclose(R, expected, atol=1e-6), f"Failed for rotation around {axis}"

    # Test rotation matrix properties
    # 1. Determinant should be 1 (proper rotation)
    assert torch.allclose(torch.det(R), torch.tensor(1.0), atol=1e-6)

    # 2. Should be orthogonal (R * R^T = I)
    I = torch.eye(3)  # noqa: E741
    assert torch.allclose(R @ R.T, I, atol=1e-6)

    # 3. Inverse rotation should be transpose
    assert torch.allclose(R @ R.T, I, atol=1e-6)


def test_rodrigues_rotation_matrix_preserves_length() -> None:
    """Test that rotation preserves vector length."""
    axis = torch.tensor([1.0, 0.0, 0.0])
    theta = math.pi / 2
    R = rodrigues_rotation_matrix(axis, theta)

    v = torch.tensor([1.0, 2.0, 3.0])
    rotated = torch.matmul(R, v)
    assert torch.allclose(torch.norm(v), torch.norm(rotated))
