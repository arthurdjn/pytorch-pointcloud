from typing import Literal, Tuple

import pytest
import torch

from torch_pointcloud.utils.geometry import (
    axis_aligned_bounding_box,
    cross_product_matrices,
    cross_product_matrix,
    rodrigues_rotation_matrices,
    rodrigues_rotation_matrix,
    spherical_lloyd,
)


@pytest.mark.parametrize(
    "k, expected_result",
    [
        (torch.tensor([1.0, 2.0, 3.0]), torch.tensor([[0.0, -3.0, 2.0], [3.0, 0.0, -1.0], [-2.0, 1.0, 0.0]])),
        (torch.tensor([0.0, 1.0, 0.0]), torch.tensor([[0.0, 0.0, 1.0], [0.0, 0.0, 0.0], [-1.0, 0.0, 0.0]])),
        (torch.tensor([0.0, 0.0, 1.0]), torch.tensor([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])),
    ],
)
def test_cross_product_matrix(k: torch.Tensor, expected_result: torch.Tensor) -> None:
    result = cross_product_matrix(k)

    assert result.shape == (3, 3)
    assert torch.allclose(result, expected_result, atol=1e-6)


@pytest.mark.parametrize(
    "k, expected_result",
    [
        (
            torch.tensor([[1.0, 2.0, 3.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
            torch.tensor(
                [
                    [[0.0, -3.0, 2.0], [3.0, 0.0, -1.0], [-2.0, 1.0, 0.0]],
                    [[0.0, 0.0, 1.0], [0.0, 0.0, 0.0], [-1.0, 0.0, 0.0]],
                    [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                ]
            ),
        ),
    ],
)
def test_cross_product_matrices(k: torch.Tensor, expected_result: torch.Tensor) -> None:
    result = cross_product_matrices(k)

    assert result.shape == (3, 3, 3)
    assert torch.allclose(result, expected_result, atol=1e-6)


@pytest.mark.parametrize(
    "axis, theta_degrees, expected_result",
    [
        # Test case for 90 degrees rotation around x-axis
        (torch.tensor([1.0, 0.0, 0.0]), 90.0, torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])),
        # Test case for 180 degrees rotation around y-axis
        (
            torch.tensor([0.0, 1.0, 0.0]),
            180.0,
            torch.tensor([[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]]),
        ),
        # Test case for 45 degrees rotation around z-axis
        (
            torch.tensor([0.0, 0.0, 1.0]),
            45.0,
            torch.tensor([[0.7071, -0.7071, 0.0], [0.7071, 0.7071, 0.0], [0.0, 0.0, 1.0]]),
        ),
        # Test case for 0 degrees rotation (identity matrix)
        (torch.tensor([1.0, 0.0, 0.0]), 0.0, torch.eye(3)),
    ],
)
def test_rodrigues_rotation_matrix(axis: torch.Tensor, theta_degrees: float, expected_result: torch.Tensor) -> None:
    R = rodrigues_rotation_matrix(axis, theta_degrees)
    assert R.shape == (3, 3)
    assert torch.allclose(R, expected_result, atol=1e-6)


@pytest.mark.parametrize(
    "axis, theta_degrees",
    [
        (torch.tensor([1.0, 1.0, 0.0]), 90.0),
        (torch.tensor([1.0, 2.0, 3.0]), 45.0),
        (torch.tensor([0.5, 0.5, 0.5]), 120.0),
    ],
)
def test_rodrigues_rotation_matrix_custom_axis(axis: torch.Tensor, theta_degrees: float) -> None:
    axis = axis / axis.norm()  # Normalize axis
    R = rodrigues_rotation_matrix(axis, theta_degrees)

    # Check if the matrix is orthogonal: R * R^T should be identity
    assert torch.allclose(R @ R.T, torch.eye(3), atol=1e-6)
    # Check if the determinant is 1 (valid rotation matrix)
    assert torch.allclose(torch.det(R), torch.tensor(1.0), atol=1e-6)


@pytest.mark.parametrize(
    "axes, theta_degrees, expected_result",
    [
        # Test for batch of rotation matrices with 90 degrees rotation around x, y, z axes
        (
            torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
            torch.tensor([90.0, 90.0, 90.0]),
            torch.tensor(
                [
                    [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]],  # 90 degrees around x-axis
                    [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]],  # 90 degrees around y-axis
                    [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],  # 90 degrees around z-axis
                ]
            ),
        ),
        # Test for 180 degrees rotation around x, y, z axes
        (
            torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
            torch.tensor([180.0, 180.0, 180.0]),
            torch.tensor(
                [
                    [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]],  # 180 degrees around x-axis
                    [[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]],  # 180 degrees around y-axis
                    [[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]],  # 180 degrees around z-axis
                ]
            ),
        ),
        # Test for 0 degrees rotation around x, y, z axes (identity matrices)
        (
            torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
            torch.tensor([0.0, 0.0, 0.0]),
            torch.eye(3).unsqueeze(0).repeat(3, 1, 1),
        ),
    ],
)
def test_rodrigues_rotation_matrices(
    axes: torch.Tensor, theta_degrees: torch.Tensor, expected_result: torch.Tensor
) -> None:
    R = rodrigues_rotation_matrices(axes, theta_degrees)
    assert torch.allclose(R, expected_result, atol=1e-6)


@pytest.mark.parametrize(
    "axes, theta_degrees",
    [
        (torch.tensor([[1.0, 1.0, 0.0], [1.0, 2.0, 3.0], [0.5, 0.5, 0.5]]), torch.tensor([90.0, 45.0, 120.0])),
        (torch.tensor([[1.0, 1.0, 1.0], [2.0, 3.0, 4.0], [0.0, 1.0, 1.0]]), torch.tensor([60.0, 30.0, 150.0])),
    ],
)
def test_rodrigues_rotation_matrices_custom_axes(axes: torch.Tensor, theta_degrees: torch.Tensor) -> None:
    axes = axes / axes.norm(dim=1, keepdim=True)  # Normalize axes
    R = rodrigues_rotation_matrices(axes, theta_degrees)

    # Check if the matrices are orthogonal: R * R^T should be identity
    eyes = torch.eye(3).unsqueeze(0).repeat(axes.size(0), 1, 1)
    assert torch.allclose(torch.matmul(R, R.transpose(1, 2)), eyes, atol=1e-6)

    # Check if the determinant is 1 (valid rotation matrices)
    assert torch.allclose(torch.det(R), torch.ones(axes.size(0)), atol=1e-6)


@pytest.mark.parametrize(
    "radius, num_cells, dim, position, approximation, expected_shape",
    [
        # Test for default parameters with no fixed positions and discretization
        (1.0, 10, 3, "none", "discretization", (10, 3)),
        # Test for center fixed position and discretization
        (1.0, 10, 3, "center", "discretization", (10, 3)),
        # Test for vertical position and discretization
        (1.0, 10, 3, "vertical", "discretization", (10, 3)),
        # Test for 2D space and Monte Carlo approximation
        (1.0, 10, 2, "none", "monte-carlo", (10, 2)),
        # Test for 4D space and Monte Carlo approximation
        (1.0, 10, 4, "none", "monte-carlo", (10, 4)),
    ],
)
def test_spherical_lloyd_output_shape(
    radius: float,
    num_cells: int,
    dim: int,
    position: Literal["none", "center", "vertical"],
    approximation: Literal["discretization", "monte-carlo"],
    expected_shape: Tuple[int, int],
) -> None:
    """Test that spherical_lloyd returns a tensor with the expected shape."""
    kernel_points = spherical_lloyd(radius, num_cells, dim, position, approximation)
    assert kernel_points.shape == expected_shape, f"Expected shape {expected_shape}, but got {kernel_points.shape}"


@pytest.mark.parametrize(
    "radius, num_cells, dim, position, approximation",
    [
        (1.0, 10, 3, "none", "discretization"),
        (1.0, 10, 3, "center", "discretization"),
        (1.0, 10, 3, "vertical", "discretization"),
        (1.0, 10, 2, "none", "monte-carlo"),
        (1.0, 10, 4, "none", "monte-carlo"),
    ],
)
def test_spherical_lloyd_kernel_points_in_sphere(
    radius: float,
    num_cells: int,
    dim: int,
    position: Literal["none", "center", "vertical"],
    approximation: Literal["discretization", "monte-carlo"],
) -> None:
    kernel_points = spherical_lloyd(radius, num_cells, dim, position, approximation)
    norms = kernel_points.norm(dim=1)
    assert torch.all(norms <= radius), "All kernel points should be inside the sphere."


@pytest.mark.parametrize("radius, num_cells, dim", [(1.0, 10, 3), (1.0, 10, 3)])
def test_spherical_lloyd_position_vertical(radius: float, num_cells: int, dim: int) -> None:
    # Test that spherical_lloyd fixes the kernel points as specified by the position parameter.
    kernel_points = spherical_lloyd(radius=radius, num_cells=num_cells, dim=dim, position="vertical")
    # The first three points should be vertically aligned
    assert torch.allclose(kernel_points[0], torch.zeros(dim)), "The first kernel point should be at the origin"
    assert kernel_points[1, -1] > 0, "The second point should be above the origin"
    assert kernel_points[2, -1] < 0, "The third point should be below the origin"


@pytest.mark.parametrize("radius, num_cells, dim", [(1.0, 10, 3), (1.0, 10, 3)])
def test_spherical_lloyd_position_center(radius: float, num_cells: int, dim: int) -> None:
    kernel_points = spherical_lloyd(radius=radius, num_cells=num_cells, dim=dim, position="center")
    assert torch.allclose(kernel_points[0], torch.zeros(dim)), "The first kernel point should be at the origin"


@pytest.mark.parametrize(
    "points, expected_bbox",
    [
        # Test for a perfect cube with min (0,0,0) and max (1,1,1)
        (
            torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 1.0]]),
            torch.tensor([0.5, 0.5, 0.5, 1.0, 1.0, 1.0]),
        ),
        # Test for a rectangle with min (0,0,0) and max (2,4,6)
        (
            torch.tensor([[0.0, 0.0, 0.0], [2.0, 4.0, 6.0], [1.0, 0.0, 0.0], [0.0, 4.0, 6.0]]),
            torch.tensor([1.0, 2.0, 3.0, 2.0, 4.0, 6.0]),
        ),
        # Test for a single point
        (torch.tensor([[1.0, 2.0, 3.0]]), torch.tensor([1.0, 2.0, 3.0, 0.0, 0.0, 0.0])),
        # Test for degenerate case where all points lie along a line in one axis
        (
            torch.tensor([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.5, 0.0, 0.0]]),
            torch.tensor([1.0, 0.0, 0.0, 2.0, 0.0, 0.0]),
        ),
        # Test case: 3D rectangular prism
        (
            torch.tensor(
                [
                    [0.0, 0.0, 0.0],
                    [4.0, 0.0, 0.0],
                    [0.0, 3.0, 0.0],
                    [4.0, 3.0, 0.0],
                    [0.0, 0.0, 2.0],
                    [4.0, 0.0, 2.0],
                    [0.0, 3.0, 2.0],
                    [4.0, 3.0, 2.0],
                ]
            ),
            torch.tensor([2.0, 1.5, 1.0, 4.0, 3.0, 2.0]),
        ),
    ],
)
def test_axis_aligned_bounding_box(points: torch.Tensor, expected_bbox: torch.Tensor) -> None:
    bbox = axis_aligned_bounding_box(points)
    assert torch.allclose(bbox, expected_bbox, atol=1e-6)
