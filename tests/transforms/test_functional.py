from typing import Tuple
from unittest.mock import MagicMock, Mock, patch

import pytest
import torch
from torch import Tensor

import torch_pointcloud.transforms.functional as F


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


@pytest.fixture
def batch() -> Tensor:
    return torch.tensor([0, 0, 0, 1, 1, 2, 3, 3, 3, 3, 4, 4, 4, 4, 4])


@patch("torch_pointcloud.transforms.functional.torch.randint")
def test_random_sample(mock_randint: Mock) -> None:
    """Test that the random sample uses torch.randint correctly to sample indices."""
    tensor = MagicMock()
    num_samples = 10

    result = F.random_sample(tensor, num_samples)
    mock_randint.assert_called_once_with(0, tensor.size(0), (num_samples,), generator=None)
    assert tensor[mock_randint.return_value] is result


@patch("torch_pointcloud.transforms.functional.torch.Generator")
@patch("torch_pointcloud.transforms.functional.torch.randint")
def test_random_sample_with_seed(mock_randint: Mock, mock_generator: Mock) -> None:
    """Test that the random sample uses torch.randint correctly to sample indices with seed."""
    tensor = MagicMock()
    num_samples = 10
    generator = MagicMock()
    mock_generator.return_value = generator

    result = F.random_sample(tensor, num_samples, generator=generator)
    mock_randint.assert_called_once_with(0, tensor.size(0), (num_samples,), generator=generator)
    assert tensor[mock_randint.return_value] is result


def test_random_sample_return_indices() -> None:
    """Test that random_sample returns both the sampled tensor and indices when return_indices=True."""
    tensor = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
    sampled, indices = F.random_sample(tensor, num_samples=3, return_indices=True)

    assert sampled.shape == (3, 2)
    assert indices.shape == (3,)
    assert torch.equal(sampled, tensor[indices])


def test_random_sample_oversampling() -> None:
    """Test that random_sample allows num_samples > tensor size (sampling with replacement)."""
    tensor = torch.tensor([[1.0], [2.0]])
    result = F.random_sample(tensor, num_samples=10)
    assert result.shape == (10, 1)


def test_random_sample_seed_reproducibility() -> None:
    """Test that random_sample produces identical results with the same seed."""
    tensor = torch.randn(100, 3)
    generator = torch.Generator()

    generator.manual_seed(42)
    a = F.random_sample(tensor, num_samples=20, generator=generator)
    generator.manual_seed(42)
    b = F.random_sample(tensor, num_samples=20, generator=generator)
    assert torch.equal(a, b)


def test_random_sample_face_vertices(sample_mesh: Tuple[Tensor, Tensor]) -> None:
    """Test that the random sample vertices function returns the correct shape."""
    vertices, faces = sample_mesh
    num_samples = 10

    sampled = F.random_sample_face_vertices(vertices, faces, num_samples)
    assert sampled.shape == (num_samples, 3)


def test_random_sample_face_vertices_with_normals(sample_mesh: Tuple[Tensor, Tensor]) -> None:
    """Test that the random sample vertices function returns the correct shape with normals."""
    vertices, faces = sample_mesh
    num_samples = 10

    sampled, normals = F.random_sample_face_vertices(vertices, faces, num_samples, return_normals=True)
    assert sampled.shape == (num_samples, 3)
    assert normals.shape == (num_samples, 3)
    # Check normals are normalized
    assert torch.allclose(torch.norm(normals, dim=1), torch.ones(num_samples))


def test_random_sample_face_vertices_seed_reproducibility(sample_mesh: Tuple[Tensor, Tensor]) -> None:
    """Test that random_sample_face_vertices produces identical results with the same seed."""
    vertices, faces = sample_mesh
    generator = torch.Generator()

    generator.manual_seed(42)
    a = F.random_sample_face_vertices(vertices, faces, num_samples=10, generator=generator)
    generator.manual_seed(42)
    b = F.random_sample_face_vertices(vertices, faces, num_samples=10, generator=generator)
    assert torch.equal(a, b)


def test_random_sample_face_vertices_single_face() -> None:
    """Test random_sample_face_vertices with a single-face mesh."""
    vertices = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    faces = torch.tensor([[0, 1, 2]])
    sampled = F.random_sample_face_vertices(vertices, faces, num_samples=5)
    assert sampled.shape == (5, 3)


def test_normalize_scale(sample_points: Tensor) -> None:
    """Test that the normalize scale function returns the correct shape."""
    normalized = F.normalize_scale(sample_points)
    expected = torch.tensor(
        [
            [-0.3015, -0.3015, -0.3015],
            [0.9045, -0.3015, -0.3015],
            [-0.3015, 0.9045, -0.3015],
            [-0.3015, -0.3015, 0.9045],
        ]
    )

    assert torch.allclose(normalized, expected, atol=1e-4)


def test_normalize_scale_single_point() -> None:
    """Test normalize_scale with a single point — centroid subtraction should yield zero, eps prevents div-by-zero."""
    points = torch.tensor([[5.0, 3.0, 1.0]])
    normalized = F.normalize_scale(points)
    assert torch.allclose(normalized, torch.zeros(1, 3), atol=1e-4)


def test_normalize_scale_all_zeros() -> None:
    """Test normalize_scale with all-zero points — eps prevents division by zero."""
    points = torch.zeros(4, 3)
    normalized = F.normalize_scale(points)
    assert torch.allclose(normalized, torch.zeros(4, 3))
    assert not torch.isnan(normalized).any()
    assert not torch.isinf(normalized).any()


def test_normalize_scale_already_centered() -> None:
    """Test normalize_scale with already-centered data."""
    points = torch.tensor([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    normalized = F.normalize_scale(points)
    # Centroid is (0,0,0), max norm is 1.0
    expected = torch.tensor([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    assert torch.allclose(normalized, expected, atol=1e-6)


def test_normalize_scale_output_unit_scale() -> None:
    """Test that normalized points have a max norm of at most 1."""
    points = torch.randn(50, 3) * 100
    normalized = F.normalize_scale(points)
    norms = torch.norm(normalized, dim=-1)
    assert norms.max() <= 1.0 + 1e-6


def test_normalize_scale_bbox_matches_midrange_scale() -> None:
    """Axis-aligned bbox: center (4,5,6), longest edge 6, radius 3 + eps."""
    points = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
    eps = 1e-6
    out = F.normalize_scale(points, eps=eps, method="bbox")
    radius = 3.0 + eps
    expected = (points - torch.tensor([4.0, 5.0, 6.0])) / radius
    assert torch.allclose(out, expected)


def test_normalize_scale_bbox_all_zeros() -> None:
    """Degenerate bbox: radius is ``eps`` only."""
    points = torch.zeros(4, 3)
    eps = 1e-6
    out = F.normalize_scale(points, eps=eps, method="bbox")
    assert torch.allclose(out, torch.zeros_like(points))
    assert not torch.isnan(out).any()


def test_normalize_scale_invalid_method_raises() -> None:
    points = torch.randn(3, 3)
    with pytest.raises(ValueError, match="Invalid method"):
        F.normalize_scale(points, method="typo")  # type: ignore[arg-type]


def test_divisible_pad() -> None:
    """Test that the divisible pad functions pads the batch correctly and returns the correct inverse indices."""
    batch = torch.tensor([0, 0, 0, 1, 1, 2, 3, 3, 3, 3])
    k = 3

    expected_padded_idxs = torch.tensor([0, 1, 2, 3, 4, 3, 5, 5, 5, 6, 7, 8, 9, 6, 7])
    expected_inverse_idxs = torch.tensor([0, 1, 2, 3, 4, 6, 9, 10, 11, 12])
    expected_padded_batch = torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 3, 3, 3])

    padded_idxs, padded_inverse, padded_batch = F.divisible_pad(batch, k, return_inverse=True)
    assert torch.equal(padded_idxs, expected_padded_idxs)
    assert torch.equal(padded_inverse, expected_inverse_idxs)
    assert torch.equal(padded_batch, expected_padded_batch)
    assert torch.equal(batch, padded_batch[padded_inverse])


def test_divisible_pad_no_inverse() -> None:
    """Test that the divisible pad functions pads the batch correctly without returning inverse indices."""
    batch = torch.tensor([0, 0, 0, 1, 1, 2, 3, 3, 3, 3])
    k = 3

    expected_padded_idxs = torch.tensor([0, 1, 2, 3, 4, 3, 5, 5, 5, 6, 7, 8, 9, 6, 7])
    expected_padded_batch = torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 3, 3, 3])

    padded_idxs, padded_batch = F.divisible_pad(batch, k, return_inverse=False)
    assert torch.equal(padded_idxs, expected_padded_idxs)
    assert torch.equal(padded_batch, expected_padded_batch)


def test_divisible_pad_with_mode_below() -> None:
    """Test that the divisible pad functions pads the batch correctly for batches with less than k points."""
    batch = torch.tensor([0, 0, 0, 1, 1, 2, 3, 3, 3, 3])
    k = 3

    expected_padded_idxs = torch.tensor([0, 1, 2, 3, 4, 3, 5, 5, 5, 6, 7, 8, 9])
    expected_inverse_idxs = torch.tensor([0, 1, 2, 3, 4, 6, 9, 10, 11, 12])
    expected_padded_batch = torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 3])

    padded_idxs, padded_inverse, padded_batch = F.divisible_pad(batch, k, mode="below", return_inverse=True)
    assert torch.equal(padded_idxs, expected_padded_idxs)
    assert torch.equal(padded_inverse, expected_inverse_idxs)
    assert torch.equal(padded_batch, expected_padded_batch)
    assert torch.equal(batch, padded_batch[padded_inverse])


def test_divisible_pad_with_mode_above() -> None:
    """Test that the divisible pad functions pads the batch correctly for batches with more than k points."""
    batch = torch.tensor([0, 0, 0, 1, 1, 2, 3, 3, 3, 3])
    k = 3

    expected_padded_idxs = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 6, 7])
    expected_inverse_idxs = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    expected_padded_batch = torch.tensor([0, 0, 0, 1, 1, 2, 3, 3, 3, 3, 3, 3])

    padded_idxs, padded_inverse, padded_batch = F.divisible_pad(batch, k, mode="above", return_inverse=True)
    assert torch.equal(padded_idxs, expected_padded_idxs)
    assert torch.equal(padded_inverse, expected_inverse_idxs)
    assert torch.equal(padded_batch, expected_padded_batch)


def test_divisible_pad_invalid_mode() -> None:
    """Test that divisible_pad raises ValueError for an unknown mode."""
    batch = torch.tensor([0, 0, 0])
    with pytest.raises(ValueError, match="Unknown mode"):
        F.divisible_pad(batch, k=2, mode="invalid")  # type: ignore[call-overload]


def test_divisible_pad_already_divisible() -> None:
    """Test that divisible_pad is a no-op when all batches are already divisible by k."""
    batch = torch.tensor([0, 0, 0, 1, 1, 1])
    k = 3

    padded_idxs, padded_batch = F.divisible_pad(batch, k)
    assert torch.equal(padded_idxs, torch.tensor([0, 1, 2, 3, 4, 5]))
    assert torch.equal(padded_batch, batch)


def test_divisible_pad_k_equals_1() -> None:
    """Test that divisible_pad with k=1 is always a no-op (everything is divisible by 1)."""
    batch = torch.tensor([0, 0, 1, 1, 1])
    padded_idxs, padded_batch = F.divisible_pad(batch, k=1)
    assert torch.equal(padded_idxs, torch.arange(len(batch)))
    assert torch.equal(padded_batch, batch)


def test_split_batch() -> None:
    """Test that the split batch function splits the batch into smaller chunks below a maximum size."""
    batch = torch.tensor([0, 0, 0, 1, 1, 2, 3, 3, 3, 3])
    max_size = 3

    expected_split_batch = torch.tensor([0, 0, 0, 1, 1, 2, 3, 3, 3, 4])

    splitted_batch = F.split_batch(batch, max_size)
    assert torch.equal(splitted_batch, expected_split_batch)


def test_remove_near_origin_removes_close_points() -> None:
    """Test that points within the given radius of the origin are removed."""
    pos = torch.tensor(
        [
            [0.0001, 0.0001, 0.0001],  # near origin
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0],  # exactly at origin
        ]
    )
    result = F.remove_near_origin(pos, radius=1e-3)
    assert result.shape == (2, 3)
    assert torch.allclose(result, pos[1:3])


def test_remove_near_origin_keeps_all_when_none_near() -> None:
    """Test that all points are kept when none are near the origin."""
    pos = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    result = F.remove_near_origin(pos, radius=1e-3)
    assert torch.equal(result, pos)


def test_remove_near_origin_return_mask() -> None:
    """Test that the mask is correctly returned when return_mask=True."""
    pos = torch.tensor(
        [
            [0.0, 0.0, 0.0],  # at origin
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0001],  # near origin
            [0.0, 2.0, 0.0],
        ]
    )
    result, mask = F.remove_near_origin(pos, radius=1e-3, return_mask=True)
    expected_mask = torch.tensor([False, True, False, True])
    assert torch.equal(mask, expected_mask)
    assert result.shape == (2, 3)
    assert torch.allclose(result, pos[mask])


def test_remove_near_origin_custom_radius() -> None:
    """Test remove_near_origin with a custom radius."""
    pos = torch.tensor(
        [
            [0.5, 0.0, 0.0],
            [1.5, 0.0, 0.0],
            [2.5, 0.0, 0.0],
        ]
    )
    result = F.remove_near_origin(pos, radius=1.0)
    assert result.shape == (2, 3)
    assert torch.allclose(result, pos[1:])


def test_remove_near_origin_all_removed() -> None:
    """Test remove_near_origin when all points are near the origin."""
    pos = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [0.0001, 0.0001, 0.0],
        ]
    )
    result = F.remove_near_origin(pos, radius=1.0)
    assert result.shape == (0, 3)


def test_remove_near_origin_empty_tensor() -> None:
    """Test remove_near_origin with an empty input tensor."""
    pos = torch.zeros(0, 3)
    result = F.remove_near_origin(pos, radius=1e-3)
    assert result.shape == (0, 3)


def test_abs_basic() -> None:
    """Test that abs returns the absolute values."""
    x = torch.tensor([-1.0, 2.0, -3.0, 0.0])
    result = abs(x)
    expected = torch.tensor([1.0, 2.0, 3.0, 0.0])
    assert torch.equal(result, expected)


def test_abs_already_positive() -> None:
    """Test abs on already positive values returns unchanged tensor."""
    x = torch.tensor([1.0, 2.0, 3.0])
    result = abs(x)
    assert torch.equal(result, x)


def test_abs_not_inplace_by_default() -> None:
    """Test that abs is not in-place by default."""
    x = torch.tensor([-1.0, -2.0])
    original = x.clone()
    result = abs(x)
    # Original should be unchanged
    assert torch.equal(x, original)
    assert torch.equal(result, torch.tensor([1.0, 2.0]))


def test_abs_inplace() -> None:
    """Test that abs modifies tensor in-place when inplace=True."""
    x = torch.tensor([-1.0, -2.0, 3.0])
    result = F.abs(x, inplace=True)
    expected = torch.tensor([1.0, 2.0, 3.0])
    assert torch.equal(x, expected)
    assert result is x


def test_abs_multidimensional() -> None:
    """Test abs on a multi-dimensional tensor."""
    x = torch.tensor([[-1.0, 2.0], [-3.0, 4.0]])
    result = abs(x)
    expected = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    assert torch.equal(result, expected)


def test_bounding_box_basic() -> None:
    """Test bounding_box returns correct min and max values."""
    x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
    result = F.bounding_box(x, dim=0)
    assert result == (1.0, 2.0, 3.0, 7.0, 8.0, 9.0)


def test_bounding_box_default_dim() -> None:
    """Test bounding_box with default dim=0."""
    x = torch.tensor([[0.0, 10.0], [-5.0, 5.0]])
    result = F.bounding_box(x)
    assert result == (-5.0, 5.0, 0.0, 10.0)


def test_bounding_box_dim0() -> None:
    """Test bounding_box along dimension 0."""
    x = torch.tensor(
        [
            [1.0, 2.0, 3.0],
            [4.0, 0.0, 6.0],
        ]
    )
    result = F.bounding_box(x, dim=0)
    assert result == (1.0, 0.0, 3.0, 4.0, 2.0, 6.0)


def test_bounding_box_single_point() -> None:
    """Test bounding_box with a single point returns that point as min and max."""
    x = torch.tensor([[3.0, 5.0, 7.0]])
    result = F.bounding_box(x, dim=0)
    assert result == (3.0, 5.0, 7.0, 3.0, 5.0, 7.0)


def test_bounding_box_negative_coords() -> None:
    """Test bounding_box with all-negative coordinates."""
    x = torch.tensor([[-5.0, -3.0, -1.0], [-10.0, -7.0, -2.0]])
    result = F.bounding_box(x, dim=0)
    assert result == (-10.0, -7.0, -2.0, -5.0, -3.0, -1.0)


def test_bounding_box_composable_with_inbox_mask() -> None:
    """Test that bounding_box output can be directly fed into inbox_mask."""
    x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [0.0, -1.0, 7.0]])
    bbox = F.bounding_box(x, dim=0)
    mask = F.inbox_mask(x, bbox, dim=-1)
    assert mask.shape == (3,)


def test_inbox_mask_all_inside() -> None:
    """Test that all points inside the bounding box produce True mask."""
    x = torch.tensor([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]])
    bbox = (0.0, 0.0, 0.0, 3.0, 3.0, 3.0)  # min x,y,z=0, max x,y,z=3
    result = F.inbox_mask(x, bbox, dim=-1)
    assert torch.equal(result, torch.tensor([True, True]))


def test_inbox_mask_some_outside() -> None:
    """Test that points outside the bounding box produce False mask."""
    x = torch.tensor(
        [
            [1.0, 1.0, 1.0],
            [5.0, 5.0, 5.0],
            [2.0, 2.0, 2.0],
        ]
    )
    bbox = (0.0, 0.0, 0.0, 3.0, 3.0, 3.0)
    result = F.inbox_mask(x, bbox, dim=-1)
    assert torch.equal(result, torch.tensor([True, False, True]))


def test_inbox_mask_boundary_exclusive() -> None:
    """Test that points exactly on the boundary are excluded (strict inequality)."""
    x = torch.tensor(
        [
            [0.0, 0.0, 0.0],  # on min boundary
            [3.0, 3.0, 3.0],  # on max boundary
            [1.5, 1.5, 1.5],  # inside
        ]
    )
    bbox = (0.0, 0.0, 0.0, 3.0, 3.0, 3.0)
    result = F.inbox_mask(x, bbox, dim=-1)
    assert torch.equal(result, torch.tensor([False, False, True]))


def test_inbox_mask_2d() -> None:
    """Test inbox_mask with 2D data."""
    x = torch.tensor(
        [
            [1.0, 1.0],
            [5.0, 1.0],
            [1.0, 5.0],
        ]
    )
    bbox = (0.0, 0.0, 3.0, 3.0)  # min x,y=0, max x,y=3
    result = F.inbox_mask(x, bbox, dim=-1)
    assert torch.equal(result, torch.tensor([True, False, False]))


def test_inbox_mask_invalid_bbox_size() -> None:
    """Test that inbox_mask raises ValueError for mismatched bbox size."""
    x = torch.tensor([[1.0, 2.0, 3.0]])
    bbox = (0.0, 0.0, 3.0, 3.0)  # size 4, but dim size is 3 -> expects 6
    with pytest.raises(ValueError, match="Bounding box size mismatch"):
        F.inbox_mask(x, bbox, dim=-1)


def test_apply_mask_basic() -> None:
    """Test that apply_mask filters elements correctly."""
    x = torch.tensor([1.0, 2.0, 3.0, 4.0])
    mask = torch.tensor([True, False, True, False])
    result = F.apply_mask(x, mask)
    expected = torch.tensor([1.0, 3.0])
    assert torch.equal(result, expected)


def test_apply_mask_all_true() -> None:
    """Test apply_mask with all True mask returns original tensor."""
    x = torch.tensor([1.0, 2.0, 3.0])
    mask = torch.tensor([True, True, True])
    result = F.apply_mask(x, mask)
    assert torch.equal(result, x)


def test_apply_mask_all_false() -> None:
    """Test apply_mask with all False mask returns empty tensor."""
    x = torch.tensor([1.0, 2.0, 3.0])
    mask = torch.tensor([False, False, False])
    result = F.apply_mask(x, mask)
    assert result.shape == (0,)


def test_apply_mask_2d() -> None:
    """Test apply_mask on a 2D tensor (row selection)."""
    x = torch.tensor(
        [
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
        ]
    )
    mask = torch.tensor([True, False, True])
    result = F.apply_mask(x, mask)
    expected = torch.tensor([[1.0, 2.0], [5.0, 6.0]])
    assert torch.equal(result, expected)


@patch("torch_pointcloud.transforms.functional.fps")
def test_sample_farthest_points_with_num_samples(mock_fps: Mock) -> None:
    """Test that sample_farthest_points delegates to fps with num_samples."""
    pos = MagicMock()
    num_samples = 10

    result = F.sample_farthest_points(pos, num_samples=num_samples)

    mock_fps.assert_called_once_with(pos, num_nodes=num_samples, ratio=None, random_start=False)
    assert result is mock_fps.return_value


@patch("torch_pointcloud.transforms.functional.fps")
def test_sample_farthest_points_with_ratio(mock_fps: Mock) -> None:
    """Test that sample_farthest_points delegates to fps with ratio."""
    pos = MagicMock()
    ratio = 0.5

    result = F.sample_farthest_points(pos, ratio=ratio)

    mock_fps.assert_called_once_with(pos, num_nodes=None, ratio=ratio, random_start=False)
    assert result is mock_fps.return_value


@patch("torch_pointcloud.transforms.functional.fps")
def test_sample_farthest_points_random_start(mock_fps: Mock) -> None:
    """Test that sample_farthest_points delegates to fps with random_start."""
    pos = MagicMock()

    result = F.sample_farthest_points(pos, num_samples=5, random_start=True)

    mock_fps.assert_called_once_with(pos, num_nodes=5, ratio=None, random_start=True)
    assert result is mock_fps.return_value
