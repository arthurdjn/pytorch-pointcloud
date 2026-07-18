import math
from typing import Literal, Tuple, Union

import pytest
import torch
from torch import Tensor

from torch_pointcloud.utils.geometry import (
    axis_aligned_bounding_box,
    cross_product_matrix,
    random_spherical_points,
    rodrigues_rotation_matrix,
    spherical_points_gradient,
    spherical_points_lloyd,
)


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


def test_random_spherical_points_shape() -> None:
    """Test that random_spherical_points generates the correct number of points."""
    points = random_spherical_points(1.0, 100)
    assert points.shape == (100, 3)

    # Check that points are within the unit sphere
    distances = torch.norm(points, dim=1)
    assert torch.all(distances <= 1.0)
    assert torch.all(distances >= 0.0)


@pytest.mark.parametrize("radius, bounds", [(1.0, 1.0), (2.0, (0.5, 1.0)), (1.0, (0.0, 1.0))])
def test_random_spherical_points_bounds(radius: float, bounds: Union[float, Tuple[float, float]]) -> None:
    """Test that points are generated within the specified bounds."""
    num_points = 100
    points = random_spherical_points(radius, num_points, bounds)

    distances = torch.norm(points, dim=1)
    inner_bound, outer_bound = (0.0, bounds) if isinstance(bounds, float) else bounds

    # Check that all points are within bounds
    assert torch.all(distances <= outer_bound * radius)
    assert torch.all(distances >= inner_bound * radius)


def test_random_spherical_points_distribution() -> None:
    """Test that points are roughly uniformly distributed in the sphere."""
    radius = 1.0
    num_points = 100
    points = random_spherical_points(radius, num_points)

    # Check that the mean is close to the origin
    mean = torch.mean(points, dim=0)
    assert torch.allclose(mean, torch.zeros(3), atol=0.2)

    # For a uniform distribution in a sphere of radius R,
    # the standard deviation along any axis is: σ = R * sqrt(1/5)
    std = torch.std(points, dim=0)
    expected_std = 1.0 * math.sqrt(1 / 5)
    assert torch.allclose(std, torch.full_like(std, expected_std), atol=0.2)


def test_spherical_points_gradient_basic() -> None:
    """Test basic functionality of spherical_points_gradient."""
    radius = 1.0
    num_points = 100
    points, grad_norms = spherical_points_gradient(radius, num_points, return_grad_norms=True)

    # Check output shapes
    assert points.shape == (num_points, 3)
    assert isinstance(grad_norms, torch.Tensor)
    assert grad_norms.ndim == 1

    # Check points are within expected radius (accounting for ratio=0.66)
    distances = torch.norm(points, dim=1)
    assert torch.all(distances <= radius)
    assert torch.allclose(torch.mean(distances[1:]), torch.tensor(0.66), atol=1e-4)


def test_spherical_points_gradient_distribution() -> None:
    """Test that points are roughly uniformly distributed in the sphere."""
    radius = 1.0
    num_points = 100
    points = spherical_points_gradient(radius, num_points)

    # Check that the mean is close to the origin
    mean = torch.mean(points, dim=0)
    assert torch.allclose(mean, torch.zeros(3), atol=0.2)

    # For a uniform distribution in a sphere of radius R,
    # the standard deviation along any axis is: σ = R * sqrt(1/5)
    std = torch.std(points, dim=0)
    expected_std = radius * math.sqrt(1 / 5)
    assert torch.allclose(std, torch.full_like(std, expected_std), atol=0.2)


def test_spherical_points_gradient_fixed_position_none() -> None:
    """Test that the points are not fixed when fixed_position is 'none'."""
    radius = 1.0
    num_points = 100
    points = spherical_points_gradient(radius, num_points, fixed_position="none")
    assert points.shape == (num_points, 3)


def test_spherical_points_gradient_fixed_position_vertical() -> None:
    """Test that the points are vertically aligned when fixed_position is 'vertical'."""
    radius = 1.0
    num_points = 100
    points = spherical_points_gradient(radius, num_points, fixed_position="vertical")

    # First point at center
    assert torch.allclose(points[0], torch.zeros(3))
    # Second and third points only have z component
    assert torch.allclose(points[1, :2], torch.zeros(2))
    assert torch.allclose(points[2, :2], torch.zeros(2))
    # Second point above center, third point below
    assert points[1, 2] > 0
    assert points[2, 2] < 0


def test_spherical_points_gradient_fixed_position_center() -> None:
    """Test that the points are centered when fixed_position is 'center'."""
    radius = 1.0
    num_points = 100
    points = spherical_points_gradient(radius, num_points, fixed_position="center")

    # First point should be at center
    assert torch.allclose(points[0], torch.zeros(3))


@pytest.mark.parametrize("fixed_position", ["none", "center", "vertical"])
def test_spherical_points_gradient_fixed_position_convergence(
    fixed_position: Literal["none", "center", "vertical"],
) -> None:
    """Test that optimization converges with small number of steps."""
    radius = 1.0
    num_points = 100
    _, grad_norms = spherical_points_gradient(
        radius,
        num_points,
        fixed_position=fixed_position,
        max_steps=1000,
        step_size=0.01,
        step_decay=0.9995,
        convergence_threshold=0.00001,
        max_step_size=None,
        return_grad_norms=True,
    )

    # Verify convergence criteria
    diff = torch.abs(grad_norms[-2] - grad_norms[-1])
    assert torch.allclose(diff, torch.tensor(0.0), atol=1e-2)


def test_spherical_points_lloyd_basic() -> None:
    """Test basic functionality of spherical_points_lloyd."""
    radius = 1.0
    num_points = 100
    points = spherical_points_lloyd(radius, num_points)

    # Check output shape
    assert points.shape == (num_points, 3)

    # Check points are within expected radius
    distances = torch.norm(points, dim=1)
    assert torch.all(distances <= radius)


def test_spherical_points_lloyd_distribution() -> None:
    """Test that points are roughly uniformly distributed in the sphere."""
    radius = 1.0
    num_points = 100
    points = spherical_points_lloyd(radius, num_points)

    # Check that the mean is close to the origin
    mean = torch.mean(points, dim=0)
    assert torch.allclose(mean, torch.zeros(3), atol=0.2)

    # For a uniform distribution in a sphere of radius R,
    # the standard deviation along any axis is: σ = R * sqrt(1/5)
    std = torch.std(points, dim=0)
    expected_std = radius * math.sqrt(1 / 5)
    assert torch.allclose(std, torch.full_like(std, expected_std), atol=0.2)


def test_spherical_points_lloyd_fixed_position_center() -> None:
    """Test that the points are centered when fixed_position is 'center'."""
    radius = 1.0
    num_points = 100
    points = spherical_points_lloyd(radius, num_points, fixed_position="center")

    # First point should be at center
    assert torch.allclose(points[0], torch.zeros(3))


def test_spherical_points_lloyd_fixed_position_vertical() -> None:
    """Test that the points are vertically aligned when fixed_position is 'vertical'."""
    radius = 1.0
    num_points = 100
    points = spherical_points_lloyd(radius, num_points, fixed_position="vertical")

    # First point at center
    assert torch.allclose(points[0], torch.zeros(3))
    # Second and third points only have z component
    assert torch.allclose(points[1, :2], torch.zeros(2))
    assert torch.allclose(points[2, :2], torch.zeros(2))
    # Free points keep a nonzero XY spread; only the fixed points are constrained to the axis
    assert points[3:, :2].abs().max() > 0.1
    # Second point above center, third point below
    assert points[1, 2] > 0
    assert points[2, 2] < 0


@pytest.mark.parametrize("approximation", ["discretization", "monte-carlo"])
def test_spherical_points_lloyd_approximation(approximation: Literal["discretization", "monte-carlo"]) -> None:
    """Test different approximation methods."""
    radius = 1.0
    num_points = 100
    points = spherical_points_lloyd(
        radius,
        num_points,
        approximation=approximation,
        approx_n=1000,
        max_iter=100,
        momentum=0.9,
    )

    # Check points are within expected radius
    distances = torch.norm(points, dim=1)
    assert torch.all(distances <= radius)
