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
    face = torch.tensor(
        [
            [0, 1, 2],
            [0, 2, 3],
        ]
    )
    return vertices, face


@pytest.fixture
def batch() -> Tensor:
    return torch.tensor([0, 0, 0, 1, 1, 2, 3, 3, 3, 3, 4, 4, 4, 4, 4])


def test_random_sample_default_without_replacement() -> None:
    """Default `replace=False` samples without duplicates."""
    tensor = torch.arange(10, dtype=torch.float32).reshape(10, 1)
    result = F.random_sample(tensor, num_samples=5)
    assert result.shape == (5, 1)
    # Without replacement, all sampled values are unique.
    assert result.unique().numel() == 5


def test_random_sample_return_indices() -> None:
    """random_sample returns both the sampled tensor and indices when return_indices=True."""
    tensor = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
    sampled, indices = F.random_sample(tensor, num_samples=3, return_indices=True)

    assert sampled.shape == (3, 2)
    assert indices.shape == (3,)
    assert torch.equal(sampled, tensor[indices])


def test_random_sample_oversample_without_replace_upsamples() -> None:
    tensor = torch.tensor([[1.0], [2.0]])
    result = F.random_sample(tensor, num_samples=10)
    assert result.shape == (10, 1)


def test_random_sample_oversample_with_replace_ok() -> None:
    tensor = torch.tensor([[1.0], [2.0]])
    result = F.random_sample(tensor, num_samples=10, replace=True)
    assert result.shape == (10, 1)


def test_random_sample_empty_raises() -> None:
    tensor = torch.empty(0, 3)
    with pytest.raises(ValueError, match="empty tensor"):
        F.random_sample(tensor, num_samples=4)


def test_random_sample_empty_zero_samples_ok() -> None:
    tensor = torch.empty(0, 3)
    result = F.random_sample(tensor, num_samples=0)
    assert result.shape == (0, 3)


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
    vertices, face = sample_mesh
    num_samples = 10

    sampled = F.random_sample_face_vertices(vertices, face, num_samples)
    assert sampled.shape == (num_samples, 3)


def test_random_sample_face_vertices_with_normals(sample_mesh: Tuple[Tensor, Tensor]) -> None:
    """Test that the random sample vertices function returns the correct shape with normal."""
    vertices, face = sample_mesh
    num_samples = 10

    sampled, normal = F.random_sample_face_vertices(vertices, face, num_samples, return_normals=True)
    assert sampled.shape == (num_samples, 3)
    assert normal.shape == (num_samples, 3)
    assert torch.allclose(torch.norm(normal, dim=1), torch.ones(num_samples))


def test_random_sample_face_vertices_seed_reproducibility(sample_mesh: Tuple[Tensor, Tensor]) -> None:
    """Test that random_sample_face_vertices produces identical results with the same seed."""
    vertices, face = sample_mesh
    generator = torch.Generator()

    generator.manual_seed(42)
    a = F.random_sample_face_vertices(vertices, face, num_samples=10, generator=generator)
    generator.manual_seed(42)
    b = F.random_sample_face_vertices(vertices, face, num_samples=10, generator=generator)
    assert torch.equal(a, b)


def test_random_sample_face_vertices_single_face() -> None:
    """Test random_sample_face_vertices with a single-face mesh."""
    vertices = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    face = torch.tensor([[0, 1, 2]])
    sampled = F.random_sample_face_vertices(vertices, face, num_samples=5)
    assert sampled.shape == (5, 3)


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


def test_rescale_single_point() -> None:
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


def test_rescale_invalid_method_raises() -> None:
    points = torch.randn(3, 3)
    with pytest.raises(ValueError, match="Invalid method"):
        F.rescale(points, method="typo")  # type: ignore[arg-type]


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


def test_divisible_pad_invalid_pad_fill() -> None:
    """Test that divisible_pad raises ValueError for an unknown pad_fill."""
    batch = torch.tensor([0, 0, 0])
    with pytest.raises(ValueError, match="Unknown pad_fill"):
        F.divisible_pad(batch, k=2, pad_fill="invalid")  # type: ignore[call-overload]


def test_divisible_pad_replicate_basic() -> None:
    """Test replicate fill copies from the previous patch at the same offsets."""
    # 7 points in one batch, k=3: 3 full patches (9), needs 2 padding slots
    # Patches: [0,1,2] [3,4,5] [6, pad, pad]
    # Replicate: copy offsets 1,2 from previous patch [3,4,5] -> pad = [4, 5]
    batch = torch.tensor([0, 0, 0, 0, 0, 0, 0])
    k = 3
    padded_idxs, padded_batch = F.divisible_pad(batch, k, pad_fill="replicate")
    assert len(padded_idxs) == 9
    # Real portion is identity
    assert torch.equal(padded_idxs[:7], torch.arange(7))
    # Padding: previous patch is [3,4,5]; remainder=1, so offsets 1,2 -> indices 4,5
    assert torch.equal(padded_idxs[7:], torch.tensor([4, 5]))


def test_divisible_pad_replicate_falls_back_to_cycle_for_small_batch() -> None:
    """When batch_size <= k, replicate has no previous patch and falls back to cycle."""
    batch = torch.tensor([0, 0])
    k = 3
    padded_idxs_rep, padded_batch_rep = F.divisible_pad(batch, k, pad_fill="replicate")
    padded_idxs_cyc, padded_batch_cyc = F.divisible_pad(batch, k, pad_fill="cycle")
    assert torch.equal(padded_idxs_rep, padded_idxs_cyc)
    assert torch.equal(padded_batch_rep, padded_batch_cyc)


def test_divisible_pad_replicate_no_padding_needed() -> None:
    """When already divisible, replicate produces the same result as cycle (no padding)."""
    batch = torch.tensor([0, 0, 0, 0, 0, 0])
    k = 3
    padded_idxs, padded_batch = F.divisible_pad(batch, k, pad_fill="replicate")
    assert torch.equal(padded_idxs, torch.arange(6))
    assert torch.equal(padded_batch, batch)


def test_divisible_pad_replicate_multi_batch() -> None:
    """Test replicate fill with multiple batches."""
    # batch 0: 5 points (k=3 -> pad 1), batch 1: 4 points (k=3 -> pad 2)
    batch = torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1])
    k = 3
    padded_idxs, padded_batch = F.divisible_pad(batch, k, pad_fill="replicate")
    # batch 0: 5 pts -> 6 padded. Patches: [0,1,2] [3,4,pad]
    # rem=2, prev patch=[0,1,2], copy offset 2 -> index 2
    assert padded_idxs[5] == 2
    # batch 1: 4 pts (orig indices 5,6,7,8) -> 6 padded. Patches: [5,6,7] [8,pad,pad]
    # rem=1, prev patch=[5,6,7], copy offsets 1,2 -> indices 6,7
    b1_start = 6  # new_start for batch 1 (6 padded for batch 0)
    assert padded_idxs[b1_start + 4] == 6
    assert padded_idxs[b1_start + 5] == 7


def test_divisible_pad_replicate_with_mode_above() -> None:
    """Test replicate fill with mode='above' (only pad batches >= k)."""
    # batch 0: 2 points (< k=3, skip), batch 1: 7 points (>= k=3, pad to 9)
    batch = torch.tensor([0, 0, 1, 1, 1, 1, 1, 1, 1])
    k = 3
    padded_idxs, padded_batch = F.divisible_pad(batch, k, mode="above", pad_fill="replicate")
    # batch 0: 2 points, not padded
    assert torch.equal(padded_idxs[:2], torch.tensor([0, 1]))
    # batch 1: 7 points (orig 2..8) -> 9 padded
    # Patches: [2,3,4] [5,6,7] [8, pad, pad]
    # rem=1, prev patch=[5,6,7], copy offsets 1,2 -> indices 6,7
    assert padded_idxs[9] == 6
    assert padded_idxs[10] == 7
    assert len(padded_idxs) == 11  # 2 + 9


def test_divisible_pad_replicate_inverse() -> None:
    """Test that inverse indices correctly recover original batch with replicate fill."""
    batch = torch.tensor([0, 0, 0, 0, 0, 0, 0])
    k = 3
    padded_idxs, inverse, padded_batch = F.divisible_pad(batch, k, pad_fill="replicate", return_inverse=True)
    assert torch.equal(batch, padded_batch[inverse])


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


def test_bounding_box_negative_positions() -> None:
    """Test bounding_box with all-negative positions."""
    x = torch.tensor([[-5.0, -3.0, -1.0], [-10.0, -7.0, -2.0]])
    result = F.bounding_box(x, dim=0)
    assert result == (-10.0, -7.0, -2.0, -5.0, -3.0, -1.0)


def test_bounding_box_composable_with_box_mask() -> None:
    """Test that bounding_box output can be directly fed into box_mask."""
    x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [0.0, -1.0, 7.0]])
    bbox = F.bounding_box(x, dim=0)
    mask = F.box_mask(x, bbox, dim=-1)
    assert mask.shape == (3,)


def test_box_mask_all_inside() -> None:
    """Test that all points inside the bounding box produce True mask."""
    x = torch.tensor([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]])
    bbox = (0.0, 0.0, 0.0, 3.0, 3.0, 3.0)  # min x,y,z=0, max x,y,z=3
    result = F.box_mask(x, bbox, dim=-1)
    assert torch.equal(result, torch.tensor([True, True]))


def test_box_mask_some_outside() -> None:
    """Test that points outside the bounding box produce False mask."""
    x = torch.tensor(
        [
            [1.0, 1.0, 1.0],
            [5.0, 5.0, 5.0],
            [2.0, 2.0, 2.0],
        ]
    )
    bbox = (0.0, 0.0, 0.0, 3.0, 3.0, 3.0)
    result = F.box_mask(x, bbox, dim=-1)
    assert torch.equal(result, torch.tensor([True, False, True]))


def test_box_mask_boundary_exclusive() -> None:
    """Test that points exactly on the boundary are excluded (strict inequality)."""
    x = torch.tensor(
        [
            [0.0, 0.0, 0.0],  # on min boundary
            [3.0, 3.0, 3.0],  # on max boundary
            [1.5, 1.5, 1.5],  # inside
        ]
    )
    bbox = (0.0, 0.0, 0.0, 3.0, 3.0, 3.0)
    result = F.box_mask(x, bbox, dim=-1)
    assert torch.equal(result, torch.tensor([False, False, True]))


def test_box_mask_2d() -> None:
    """Test box_mask with 2D data."""
    x = torch.tensor(
        [
            [1.0, 1.0],
            [5.0, 1.0],
            [1.0, 5.0],
        ]
    )
    bbox = (0.0, 0.0, 3.0, 3.0)  # min x,y=0, max x,y=3
    result = F.box_mask(x, bbox, dim=-1)
    assert torch.equal(result, torch.tensor([True, False, False]))


def test_box_mask_invalid_bbox_size() -> None:
    """Test that box_mask raises ValueError for mismatched bbox size."""
    x = torch.tensor([[1.0, 2.0, 3.0]])
    bbox = (0.0, 0.0, 3.0, 3.0)  # size 4, but dim size is 3 -> expects 6
    with pytest.raises(ValueError, match="Bounding box size mismatch"):
        F.box_mask(x, bbox, dim=-1)


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
def test_farthest_point_sample_with_num_samples(mock_fps: Mock) -> None:
    """Test that farthest_point_sample delegates to fps with num_samples."""
    pos = MagicMock()
    num_samples = 10

    result = F.farthest_point_sample(pos, num_samples=num_samples)

    mock_fps.assert_called_once_with(pos, num_nodes=num_samples, ratio=None, random_start=False)
    assert result is mock_fps.return_value


@patch("torch_pointcloud.transforms.functional.fps")
def test_farthest_point_sample_with_ratio(mock_fps: Mock) -> None:
    """Test that farthest_point_sample delegates to fps with ratio."""
    pos = MagicMock()
    ratio = 0.5

    result = F.farthest_point_sample(pos, ratio=ratio)

    mock_fps.assert_called_once_with(pos, num_nodes=None, ratio=ratio, random_start=False)
    assert result is mock_fps.return_value


@patch("torch_pointcloud.transforms.functional.fps")
def test_farthest_point_sample_random_start(mock_fps: Mock) -> None:
    """Test that farthest_point_sample delegates to fps with random_start."""
    pos = MagicMock()

    result = F.farthest_point_sample(pos, num_samples=5, random_start=True)

    mock_fps.assert_called_once_with(pos, num_nodes=5, ratio=None, random_start=True)
    assert result is mock_fps.return_value


def test_estimate_normals_planar_patch() -> None:
    pytest.importorskip("torch_cluster")
    grid = torch.linspace(-1.0, 1.0, 20)
    xx, yy = torch.meshgrid(grid, grid, indexing="ij")
    plane = torch.stack([xx.reshape(-1), yy.reshape(-1), torch.zeros(400)], dim=1)

    normals = F.estimate_normals(plane, k=16)

    assert normals.shape == (400, 3)
    assert torch.allclose(normals.norm(dim=1), torch.ones(400), atol=1e-5)
    # The z=0 plane's normal is the z axis.
    assert torch.allclose(normals[:, 2].abs(), torch.ones(400), atol=1e-5)


def test_estimate_normals_respects_batch() -> None:
    pytest.importorskip("torch_cluster")
    grid = torch.linspace(-1.0, 1.0, 20)
    xx, yy = torch.meshgrid(grid, grid, indexing="ij")
    plane = torch.stack([xx.reshape(-1), yy.reshape(-1), torch.zeros(400)], dim=1)
    # Second cloud is the same plane translated far along x; neighbours must not cross.
    pos = torch.cat([plane, plane + torch.tensor([100.0, 0.0, 0.0])])
    batch = torch.cat([torch.zeros(400, dtype=torch.long), torch.ones(400, dtype=torch.long)])

    normals = F.estimate_normals(pos, k=16, batch=batch)

    assert normals.shape == (800, 3)
    assert torch.allclose(normals[:, 2].abs(), torch.ones(800), atol=1e-5)


def test_cube_mask_keeps_points_inside_chebyshev_ball() -> None:
    x = torch.tensor(
        [
            [0.0, 0.0, 0.0],  # at center
            [0.5, 0.5, 0.5],  # inside (Linf 0.5 <= 1)
            [1.0, 1.0, 1.0],  # on boundary (inclusive)
            [1.5, 0.0, 0.0],  # outside on axis X
        ]
    )
    mask = F.cube_mask(x, center=[0.0, 0.0, 0.0], radius=1.0, dim=-1)
    assert mask.dtype == torch.bool
    assert mask.tolist() == [True, True, True, False]


def test_cube_mask_off_center() -> None:
    x = torch.tensor([[5.0, 5.0, 5.0], [4.0, 4.0, 4.0]])
    mask = F.cube_mask(x, center=[5.0, 5.0, 5.0], radius=0.5, dim=-1)
    assert mask.tolist() == [True, False]


def test_cube_mask_empty() -> None:
    x = torch.empty(0, 3)
    mask = F.cube_mask(x, center=[0.0, 0.0, 0.0], radius=1.0)
    assert mask.shape == (0,)
    assert mask.dtype == torch.bool


def test_sphere_mask_keeps_points_inside_euclidean_ball() -> None:
    x = torch.tensor(
        [
            [0.0, 0.0, 0.0],  # at center
            [0.5, 0.5, 0.5],  # inside (L2 ≈ 0.87 <= 1)
            [1.0, 0.0, 0.0],  # on boundary
            [1.0, 1.0, 1.0],  # outside (L2 ≈ 1.73)
        ]
    )
    mask = F.sphere_mask(x, center=[0.0, 0.0, 0.0], radius=1.0, dim=-1)
    assert mask.dtype == torch.bool
    assert mask.tolist() == [True, True, True, False]


def test_sphere_mask_differs_from_cube_mask_in_corners() -> None:
    """A corner of the unit cube (L∞=1) is outside the unit sphere (L2≈√3)."""
    x = torch.tensor([[1.0, 1.0, 1.0]])  # L∞=1, L2=√3
    assert F.cube_mask(x, [0.0, 0.0, 0.0], 1.0).item() is True
    assert F.sphere_mask(x, [0.0, 0.0, 0.0], 1.0).item() is False


def test_sphere_mask_empty() -> None:
    x = torch.empty(0, 3)
    mask = F.sphere_mask(x, center=[0.0, 0.0, 0.0], radius=1.0)
    assert mask.shape == (0,)
    assert mask.dtype == torch.bool


def test_remove_near_origin_uses_sphere_mask_semantics() -> None:
    """remove_near_origin now delegates to sphere_mask; verify L2 semantics preserved."""
    pos = torch.tensor(
        [
            [0.5, 0.0, 0.0],  # close (L2=0.5)
            [2.0, 0.0, 0.0],  # far (L2=2.0)
            [0.6, 0.6, 0.6],  # L2 ≈ 1.04 — borderline
        ]
    )
    filtered = F.remove_near_origin(pos, radius=1.0)
    # Points with L2 > 1.0 survive: the (2, 0, 0) and (0.6, 0.6, 0.6)
    assert filtered.shape == (2, 3)


def test_shift_bbox_centers_on_midrange() -> None:
    x = torch.tensor([[0.0, 0.0, 0.0], [2.0, 4.0, 6.0]])
    out = F.shift(x, method="bbox")
    expected = x - torch.tensor([1.0, 2.0, 3.0])  # bbox midrange
    assert torch.allclose(out, expected)


def test_shift_centroid_subtracts_mean() -> None:
    x = torch.tensor([[0.0, 0.0, 0.0], [2.0, 4.0, 6.0]])
    out = F.shift(x, method="centroid")
    assert torch.allclose(out, x - x.mean(dim=0))


def test_shift_min_aligns_positive_octant() -> None:
    x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    out = F.shift(x, method="min")
    assert torch.allclose(out, x - x.min(dim=0).values)
    assert out.min().item() == pytest.approx(0.0)


def test_shift_axes_subset_leaves_other_axes_untouched() -> None:
    x = torch.tensor([[0.0, 0.0, 0.0], [2.0, 4.0, 6.0]])
    out = F.shift(x, method="bbox", axes=[0, 1])  # XY only
    # XY shifted by their midranges (1, 2); Z unchanged.
    assert torch.allclose(out[:, :2], x[:, :2] - torch.tensor([1.0, 2.0]))
    assert torch.allclose(out[:, 2], x[:, 2])


def test_shift_chained_disjoint_axes_match_pointcept_centering() -> None:
    """F.shift composes the way the Pointcept-style centering recipe expects."""
    x = torch.tensor([[0.0, 0.0, 0.0], [2.0, 4.0, 6.0]])
    out = F.shift(x, method="bbox", axes=[0, 1])
    out = F.shift(out, method="min", axes=[2])
    expected = torch.tensor([[-1.0, -2.0, 0.0], [1.0, 2.0, 6.0]])
    assert torch.allclose(out, expected)


def test_shift_empty_passthrough() -> None:
    x = torch.empty(0, 3)
    out = F.shift(x, method="bbox")
    assert out.shape == (0, 3)


def test_shift_invalid_method_raises() -> None:
    x = torch.zeros(5, 3)
    with pytest.raises(ValueError, match="Invalid method"):
        F.shift(x, method="typo")  # type: ignore[arg-type]


def test_axis_min_offset_height_feature() -> None:
    x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [0.0, 0.0, 9.0]])
    out = F.axis_min_offset(x, axis=2)
    assert out.shape == (3, 1)
    # Z column [3, 6, 9], min=3; offsets are [0, 3, 6]
    assert torch.allclose(out, torch.tensor([[0.0], [3.0], [6.0]]))


def test_axis_min_offset_quantile_floor() -> None:
    # Z column 0..100; the q=0.25 quantile is 25, so offsets are z - 25 (VoteNet-style robust floor).
    z = torch.arange(101, dtype=torch.float32)
    x = torch.stack([torch.zeros_like(z), torch.zeros_like(z), z], dim=1)
    out = F.axis_min_offset(x, axis=2, quantile=0.25)
    assert out.shape == (101, 1)
    assert torch.allclose(out[:, 0], z - 25.0)


def test_axis_min_offset_preserves_dtype() -> None:
    x = torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float64)
    out = F.axis_min_offset(x, axis=0)
    assert out.dtype == torch.float64


def test_axis_min_offset_empty() -> None:
    x = torch.empty(0, 3)
    out = F.axis_min_offset(x, axis=2)
    assert out.shape == (0, 1)


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


def test_relabel_one_to_one_mapping() -> None:
    labels = torch.tensor([1, 2, 5, 255])
    out = F.relabel(labels, mapping=[1, 2, 5], default=255)
    assert out.tolist() == [0, 1, 2, 255]


def test_relabel_n_to_one_mapping_via_dict() -> None:
    # SemanticKITTI-style merge: moving-car (252) and car (10) both → 0
    labels = torch.tensor([10, 252, 11, 9999])
    out = F.relabel(labels, mapping={10: 0, 252: 0, 11: 1}, default=255)
    assert out.tolist() == [0, 0, 1, 255]


def test_relabel_preserves_dtype() -> None:
    labels = torch.tensor([0, 1, 2, 7], dtype=torch.int32)
    out = F.relabel(labels, mapping=[0, 1, 2], default=99)
    assert out.dtype == torch.int32


def test_relabel_sparse_sources_no_oom() -> None:
    labels = torch.tensor([2**20, 5, 2**18, 0])
    out = F.relabel(labels, mapping={2**20: 0, 5: 1, 2**18: 2}, default=255)
    assert out.tolist() == [0, 1, 2, 255]


def test_relabel_empty_mapping_raises() -> None:
    labels = torch.tensor([1, 2, 3])
    with pytest.raises(ValueError, match="at least one source"):
        F.relabel(labels, mapping=[])


def test_rotation_matrix_z_90deg() -> None:
    """Rotation matrix around z by 90deg maps (1, 0, 0) to (0, 1, 0)."""
    import math

    R = F.rotation_matrix(math.pi / 2, axis=2)
    v = torch.tensor([1.0, 0.0, 0.0])
    rotated = F.rotate(v, R)
    assert torch.allclose(rotated, torch.tensor([0.0, 1.0, 0.0]), atol=1e-5)


def test_rotation_matrix_is_orthonormal_for_every_axis() -> None:
    """Rotation matrices are orthonormal: R @ R.T = I with det = 1."""
    for axis in (0, 1, 2):
        R = F.rotation_matrix(1.234, axis=axis)
        assert torch.allclose(R @ R.T, torch.eye(3), atol=1e-5)
        assert torch.allclose(torch.det(R), torch.tensor(1.0), atol=1e-5)


def test_rotation_matrix_invalid_axis_raises() -> None:
    with pytest.raises(ValueError, match="axis"):
        F.rotation_matrix(0.0, axis=3)
    with pytest.raises(ValueError, match="axis"):
        F.rotation_matrix(0.0, axis=-1)


def test_random_rotate_in_range_and_deterministic() -> None:
    pos = torch.tensor([[1.0, 0.0, 0.0]])
    g1 = torch.Generator().manual_seed(0)
    g2 = torch.Generator().manual_seed(0)
    a = F.random_rotate(pos, angle_range=(-30, 30), axis=2, generator=g1)
    b = F.random_rotate(pos, angle_range=(-30, 30), axis=2, generator=g2)
    assert torch.equal(a, b)
    # Z-rotation preserves the Z coordinate.
    assert a[0, 2].item() == 0.0


def test_random_scale_uniform_factor() -> None:
    pos = torch.tensor([[1.0, 1.0, 1.0]])
    g = torch.Generator().manual_seed(0)
    out = F.random_scale(pos, scale_range=(2.0, 2.0), generator=g)
    assert torch.allclose(out, torch.tensor([[2.0, 2.0, 2.0]]))


def test_random_scale_anisotropic_per_axis() -> None:
    pos = torch.ones(1, 3)
    g = torch.Generator().manual_seed(0)
    out = F.random_scale(pos, scale_range=(0.5, 2.0), anisotropic=True, generator=g)
    # Each axis scaled independently within range.
    assert (out >= 0.5).all() and (out <= 2.0).all()


def test_random_flip_p_one_flips_all_listed_axes() -> None:
    pos = torch.tensor([[1.0, 2.0, 3.0]])
    out = F.random_flip(pos, axes=(0, 1), p=1.0)
    assert torch.allclose(out, torch.tensor([[-1.0, -2.0, 3.0]]))


def test_random_flip_p_zero_is_noop() -> None:
    pos = torch.tensor([[1.0, 2.0, 3.0]])
    out = F.random_flip(pos, axes=(0, 1, 2), p=0.0)
    assert torch.equal(out, pos)


def test_random_jitter_clipped() -> None:
    pos = torch.zeros(100, 3)
    g = torch.Generator().manual_seed(0)
    out = F.random_jitter(pos, sigma=1.0, clip=0.1, generator=g)
    assert out.abs().max().item() <= 0.1 + 1e-6


def test_random_jitter_no_clip() -> None:
    pos = torch.zeros(1000, 3)
    g = torch.Generator().manual_seed(0)
    out = F.random_jitter(pos, sigma=0.1, clip=None, generator=g)
    # Without clip some samples should exceed 0.1.
    assert out.abs().max().item() > 0.1


def test_random_shift_within_range() -> None:
    pos = torch.zeros(5, 3)
    g = torch.Generator().manual_seed(0)
    out = F.random_shift(pos, shift_range=(-1.0, 1.0), generator=g)
    # Every point in the cloud is shifted by the SAME vector (so all rows equal).
    assert torch.allclose(out, out[0:1].expand_as(out))
    assert (out.abs() <= 1.0 + 1e-6).all()


def test_random_dropout_mask_keep_rate() -> None:
    g = torch.Generator().manual_seed(0)
    mask = F.random_dropout_mask(10000, p_drop=0.3, generator=g)
    rate = mask.float().mean().item()
    assert abs(rate - 0.7) < 0.05  # within statistical noise


def test_random_dropout_mask_invalid_p_drop() -> None:
    with pytest.raises(ValueError, match=r"\[0, 1\)"):
        F.random_dropout_mask(10, p_drop=1.0)


def test_shuffle_indices_is_permutation() -> None:
    g = torch.Generator().manual_seed(0)
    perm = F.shuffle_indices(20, generator=g)
    assert perm.dtype == torch.long
    assert sorted(perm.tolist()) == list(range(20))


def test_random_color_jitter_preserves_range() -> None:
    color = torch.rand(50, 3)
    g = torch.Generator().manual_seed(0)
    out = F.random_color_jitter(color, brightness=0.5, contrast=0.5, saturation=0.3, generator=g)
    assert out.min().item() >= 0.0
    assert out.max().item() <= 1.0
    assert out.shape == color.shape


def test_random_color_jitter_int_dtype_preserved() -> None:
    color = (torch.rand(10, 3) * 255).to(torch.uint8)
    g = torch.Generator().manual_seed(0)
    out = F.random_color_jitter(color, brightness=0.2, int_color=True, generator=g)
    assert out.dtype == torch.uint8


def test_random_color_drop_returns_constant() -> None:
    color = torch.rand(10, 3)
    out = F.random_color_drop(color, fill=0.5)
    assert torch.allclose(out, torch.full_like(color, 0.5))


def test_color_grayscale_makes_channels_equal() -> None:
    color = torch.rand(10, 3)
    out = F.color_grayscale(color)
    assert torch.allclose(out[:, 0], out[:, 1])
    assert torch.allclose(out[:, 1], out[:, 2])


def test_color_grayscale_uses_bt601_weights() -> None:
    # Pure red (1, 0, 0) gives luminance 0.299.
    color = torch.tensor([[1.0, 0.0, 0.0]])
    out = F.color_grayscale(color)
    assert torch.allclose(out, torch.full_like(color, 0.299), atol=1e-5)


def test_color_auto_contrast_full_blend_stretches_range() -> None:
    color = torch.tensor([[0.25, 0.25, 0.25], [0.75, 0.75, 0.75]])
    out = F.color_auto_contrast(color, blend=1.0)
    assert torch.allclose(out.min(dim=0).values, torch.zeros(3), atol=1e-5)
    assert torch.allclose(out.max(dim=0).values, torch.ones(3), atol=1e-5)


def test_color_auto_contrast_zero_blend_is_identity() -> None:
    color = torch.tensor([[0.25, 0.25, 0.25], [0.75, 0.75, 0.75]])
    out = F.color_auto_contrast(color, blend=0.0)
    assert torch.allclose(out, color, atol=1e-5)


def test_random_rotate_choice_picks_from_list() -> None:
    """The rotation is drawn from the given list (so output should land on one of those poses)."""
    pos = torch.tensor([[1.0, 0.0, 0.0]])
    g = torch.Generator().manual_seed(42)
    # angles=[0, 90, 180, 270] around z map (1,0,0) to one of (1,0,0), (0,1,0), (-1,0,0), (0,-1,0)
    candidates = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
        ]
    )
    out = F.random_rotate_choice(pos, angles=[0, 90, 180, 270], axis=2, generator=g)
    diffs = (candidates - out).norm(dim=-1)
    assert diffs.min().item() < 1e-4


def test_random_rotate_choice_empty_raises() -> None:
    pos = torch.zeros(3, 3)
    with pytest.raises(ValueError, match="non-empty"):
        F.random_rotate_choice(pos, angles=[])


def test_random_rotate_choice_determinism() -> None:
    pos = torch.tensor([[1.0, 0.0, 0.0]])
    g1 = torch.Generator().manual_seed(7)
    g2 = torch.Generator().manual_seed(7)
    a = F.random_rotate_choice(pos, angles=[0, 90, 180, 270], generator=g1)
    b = F.random_rotate_choice(pos, angles=[0, 90, 180, 270], generator=g2)
    assert torch.equal(a, b)


def test_random_color_shift_constant_offset_within_range() -> None:
    color = torch.full((5, 3), 0.5)
    g = torch.Generator().manual_seed(0)
    out = F.random_color_shift(color, shift_range=(0.1, 0.1), generator=g)
    assert torch.allclose(out, torch.full_like(color, 0.6))


def test_random_color_shift_clamps_to_valid_range() -> None:
    color = torch.full((5, 3), 0.95)
    g = torch.Generator().manual_seed(0)
    out = F.random_color_shift(color, shift_range=(0.5, 0.5), generator=g)
    # Would go to 1.45 but clamped to 1.0.
    assert torch.all(out <= 1.0)


def test_random_color_shift_int_dtype_preserved() -> None:
    color = torch.full((5, 3), 128, dtype=torch.uint8)
    g = torch.Generator().manual_seed(0)
    out = F.random_color_shift(color, shift_range=(10, 10), int_color=True, generator=g)
    assert out.dtype == torch.uint8


def test_random_elastic_distortion_changes_positions() -> None:
    pos = torch.randn(200, 3)
    g = torch.Generator().manual_seed(0)
    out = F.random_elastic_distortion(pos, granularity=0.5, magnitude=0.1, generator=g)
    assert out.shape == pos.shape
    # Should not be identity at any reasonable magnitude.
    assert (out - pos).abs().max().item() > 0.0


def test_random_elastic_distortion_preserves_local_structure() -> None:
    """Nearby points should still be nearby after distortion (low-frequency field)."""
    pos = torch.tensor([[0.0, 0.0, 0.0], [0.001, 0.0, 0.0]])
    g = torch.Generator().manual_seed(0)
    out = F.random_elastic_distortion(pos, granularity=0.5, magnitude=0.5, generator=g)
    # Displacement at two very close points should also be very close.
    delta_in = (pos[0] - pos[1]).norm().item()
    delta_out = (out[0] - out[1]).norm().item()
    assert abs(delta_out - delta_in) < 0.01


def test_random_elastic_distortion_empty_passthrough() -> None:
    pos = torch.empty(0, 3)
    out = F.random_elastic_distortion(pos, granularity=0.2, magnitude=0.4)
    assert out.shape == (0, 3)


def test_random_elastic_distortion_wrong_shape_raises() -> None:
    pos = torch.randn(10, 2)
    with pytest.raises(ValueError, match=r"\(N, 3\)"):
        F.random_elastic_distortion(pos, granularity=0.2, magnitude=0.4)
