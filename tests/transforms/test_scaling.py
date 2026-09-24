import math
from unittest.mock import sentinel

import pytest
import torch
from torch import Tensor

import torch_pointcloud.transforms as T
import torch_pointcloud.transforms.functional as F


def test_rescale_centroid_default() -> None:
    pos = torch.tensor([[0.0, 0.0, 0.0], [4.0, 0.0, 0.0]])
    data = {"pos": pos, "other": sentinel.other}

    result = T.Rescale(keys=["pos"])(data)

    # Centroid: subtract mean (2, 0, 0); divide by max-radius (2). Output spans [-1, 1].
    assert torch.allclose(result["pos"], torch.tensor([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]), atol=1e-5)
    assert result["other"] is sentinel.other


def test_rescale_bbox_method() -> None:
    pos = torch.tensor([[0.0, 0.0, 0.0], [4.0, 2.0, 1.0]])
    result = T.Rescale(keys=["pos"], method="bbox")({"pos": pos})
    # bbox center (2,1,0.5); half-diagonal = max((4,2,1))/2 = 2; output range [-1, 1] on longest axis
    assert result["pos"].abs().max().item() == pytest.approx(1.0, abs=1e-5)


def test_normalize() -> None:
    data = {"x": torch.tensor([[1.0, 2.0, 3.0], [4.0, 6.0, 8.0]])}
    transform = T.Normalize(keys=["x"], mean=[1.0, 2.0, 3.0], std=[1.0, 2.0, 5.0])
    result = transform(data)
    expected = torch.tensor([[0.0, 0.0, 0.0], [3.0, 2.0, 1.0]])
    assert torch.allclose(result["x"], expected)


def test_rescale_empty_passthrough(empty_scene: dict) -> None:
    out = T.Rescale(keys=["pos"])(empty_scene)
    assert out["pos"].shape == (0, 3)


def test_rescale_single_point(single_point_scene: dict) -> None:
    # Single point: radius is 0 → eps prevents NaN. Result should be all zeros.
    out = T.Rescale(keys=["pos"])(single_point_scene)
    assert torch.allclose(out["pos"], torch.zeros_like(out["pos"]))


def test_normalize_zero_std_does_not_divide_by_zero() -> None:
    data = {"x": torch.tensor([[1.0, 2.0]])}
    out = T.Normalize(keys=["x"], mean=[1.0, 2.0], std=[0.0, 0.0], eps=1e-5)(data)
    # (x - mean) is zero, divided by clamped eps; result is zero (not NaN)
    assert torch.all(torch.isfinite(out["x"]))
    assert torch.allclose(out["x"], torch.zeros_like(out["x"]))


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


def test_rescale(sample_points: Tensor) -> None:
    """Test that the normalize scale function returns the correct shape."""
    normalized = F.rescale(sample_points)
    expected = torch.tensor(
        [
            [-0.3015, -0.3015, -0.3015],
            [0.9045, -0.3015, -0.3015],
            [-0.3015, 0.9045, -0.3015],
            [-0.3015, -0.3015, 0.9045],
        ]
    )

    assert torch.allclose(normalized, expected, atol=1e-4)


def test_functional_rescale_single_point() -> None:
    """Test rescale with a single point — centroid subtraction should yield zero, eps prevents div-by-zero."""
    points = torch.tensor([[5.0, 3.0, 1.0]])
    normalized = F.rescale(points)
    assert torch.allclose(normalized, torch.zeros(1, 3), atol=1e-4)


def test_rescale_all_zeros() -> None:
    """Test rescale with all-zero points — eps prevents division by zero."""
    points = torch.zeros(4, 3)
    normalized = F.rescale(points)
    assert torch.allclose(normalized, torch.zeros(4, 3))
    assert not torch.isnan(normalized).any()
    assert not torch.isinf(normalized).any()


def test_rescale_already_centered() -> None:
    """Test rescale with already-centered data."""
    points = torch.tensor([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    normalized = F.rescale(points)
    # Centroid is (0,0,0), max norm is 1.0
    expected = torch.tensor([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    assert torch.allclose(normalized, expected, atol=1e-6)


def test_rescale_output_unit_scale() -> None:
    """Test that normalized points have a max norm of at most 1."""
    points = torch.randn(50, 3) * 100
    normalized = F.rescale(points)
    norms = torch.norm(normalized, dim=-1)
    assert norms.max() <= 1.0 + 1e-6


def test_rescale_bbox_matches_midrange_scale() -> None:
    """Axis-aligned bbox: center (4,5,6), longest edge 6, radius 3 + eps."""
    points = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
    eps = 1e-6
    out = F.rescale(points, eps=eps, method="bbox")
    radius = 3.0 + eps
    expected = (points - torch.tensor([4.0, 5.0, 6.0])) / radius
    assert torch.allclose(out, expected)


def test_rescale_bbox_all_zeros() -> None:
    """Degenerate bbox: radius is ``eps`` only."""
    points = torch.zeros(4, 3)
    eps = 1e-6
    out = F.rescale(points, eps=eps, method="bbox")
    assert torch.allclose(out, torch.zeros_like(points))
    assert not torch.isnan(out).any()


def test_minimal_enclosing_ball_support_points() -> None:
    """Two antipodal points define the ball; the third point lies inside it."""
    points = torch.tensor([[0.0, 0.5, 0.0], [1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]])
    center, radius = F.minimal_enclosing_ball(points)
    assert torch.allclose(center, torch.zeros(3), atol=1e-6)
    assert math.isclose(radius.item(), 1.0, abs_tol=1e-6)


def test_minimal_enclosing_ball_tetrahedron() -> None:
    """A regular tetrahedron inscribed in the unit sphere is bounded by that sphere."""
    points = torch.tensor([[1.0, 1.0, 1.0], [1.0, -1.0, -1.0], [-1.0, 1.0, -1.0], [-1.0, -1.0, 1.0]]) / math.sqrt(3)
    center, radius = F.minimal_enclosing_ball(points + 2.0)
    assert torch.allclose(center, torch.full((3,), 2.0), atol=1e-6)
    assert math.isclose(radius.item(), 1.0, abs_tol=1e-6)


def test_minimal_enclosing_ball_encloses_random_points() -> None:
    torch.manual_seed(0)
    points = torch.randn(500, 3) * torch.tensor([3.0, 1.0, 0.5])
    center, radius = F.minimal_enclosing_ball(points)
    distances = torch.norm(points - center, dim=1)
    assert distances.max() <= radius + 1e-5
    assert radius < torch.norm(points - points.mean(0), dim=1).max() + 1e-6


def test_rescale_min_sphere_unit_radius() -> None:
    torch.manual_seed(0)
    points = torch.rand(200, 3) * 5.0
    out = F.rescale(points, eps=0.0, method="min_sphere")
    assert math.isclose(torch.norm(out, dim=1).max().item(), 1.0, abs_tol=1e-5)


def test_rescale_invalid_method_raises() -> None:
    points = torch.randn(3, 3)
    with pytest.raises(ValueError, match="Invalid method"):
        F.rescale(points, method="typo")  # type: ignore[arg-type]


def test_normalize_standardizes_per_channel() -> None:
    x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 6.0, 8.0]])
    out = F.normalize(x, mean=[1.0, 2.0, 3.0], std=[1.0, 2.0, 5.0])
    expected = torch.tensor([[0.0, 0.0, 0.0], [3.0, 2.0, 1.0]])
    assert torch.allclose(out, expected)


def test_normalize_clamps_zero_std() -> None:
    x = torch.tensor([[1.0, 2.0]])
    out = F.normalize(x, mean=[1.0, 2.0], std=[0.0, 0.0], eps=1e-5)
    assert torch.all(torch.isfinite(out))
    assert torch.allclose(out, torch.zeros_like(out))


def test_normalize_accepts_tensor_inputs() -> None:
    x = torch.tensor([[2.0, 4.0]])
    mean = torch.tensor([2.0, 4.0])
    std = torch.tensor([1.0, 2.0])
    out = F.normalize(x, mean, std)
    assert torch.allclose(out, torch.zeros_like(out))
