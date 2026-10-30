import math

import pytest
import torch

from torch_pointcloud.ops.geometry import transform_points
from torch_pointcloud.ops.registration import (
    local_to_global_registration,
    mutual_topk_correspondences,
    node_correspondences,
    pairwise_squared_distance,
    point_to_node_partition,
    rotation_angle,
    weighted_procrustes,
)


def _rotation(axis: str, degrees: float) -> torch.Tensor:
    c, s = math.cos(math.radians(degrees)), math.sin(math.radians(degrees))
    if axis == "z":
        return torch.tensor([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    return torch.tensor([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def _transform(rotation: torch.Tensor, translation: torch.Tensor) -> torch.Tensor:
    transform = torch.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = translation
    return transform


def test_pairwise_squared_distance_matches_cdist() -> None:
    torch.manual_seed(0)
    x, y = torch.randn(5, 3), torch.randn(7, 3)
    expected = torch.cdist(x, y) ** 2
    assert torch.allclose(pairwise_squared_distance(x, y), expected, atol=1e-5)


def test_pairwise_squared_distance_normalized() -> None:
    torch.manual_seed(0)
    x = torch.nn.functional.normalize(torch.randn(4, 8), dim=1)
    y = torch.nn.functional.normalize(torch.randn(6, 8), dim=1)
    assert torch.allclose(pairwise_squared_distance(x, y, normalized=True), pairwise_squared_distance(x, y), atol=1e-5)


def test_weighted_procrustes_recovers_rigid_transform() -> None:
    torch.manual_seed(0)
    src = torch.randn(50, 3)
    transform = _transform(_rotation("z", 40.0) @ _rotation("x", -25.0), torch.tensor([0.5, -1.0, 2.0]))
    ref = transform_points(src, transform)
    estimated = weighted_procrustes(src, ref)
    assert torch.allclose(estimated, transform, atol=1e-5)


def test_weighted_procrustes_ignores_zero_weight_outliers() -> None:
    torch.manual_seed(0)
    src = torch.randn(40, 3)
    transform = _transform(_rotation("z", 90.0), torch.tensor([1.0, 0.0, 0.0]))
    ref = transform_points(src, transform)
    ref[:5] += 10.0  # outliers
    weights = torch.ones(40)
    weights[:5] = 0.0
    estimated = weighted_procrustes(src, ref, weights)
    assert torch.allclose(estimated, transform, atol=1e-4)


def test_weighted_procrustes_batched_matches_single() -> None:
    torch.manual_seed(0)
    src = torch.randn(2, 30, 3)
    transforms = torch.stack(
        [
            _transform(_rotation("z", 10.0), torch.tensor([0.1, 0.2, 0.3])),
            _transform(_rotation("x", 70.0), torch.tensor([-1.0, 0.0, 1.0])),
        ]
    )
    ref = transform_points(src, transforms)
    batched = weighted_procrustes(src, ref)
    assert batched.shape == (2, 4, 4)
    assert torch.allclose(batched, transforms, atol=1e-5)
    assert torch.allclose(batched[1], weighted_procrustes(src[1], ref[1]), atol=1e-6)


def test_weighted_procrustes_broadcasts_leading_dimensions() -> None:
    torch.manual_seed(0)
    src = torch.rand(2, 3, 10, 3)
    ref = src + torch.tensor([0.1, 0.2, 0.3])
    weights = torch.rand(2, 3, 10)
    transforms = weighted_procrustes(src, ref, weights)
    assert transforms.shape == (2, 3, 4, 4)
    assert torch.allclose(transforms[1, 2], weighted_procrustes(src[1, 2], ref[1, 2], weights[1, 2]), atol=1e-6)


def test_point_to_node_partition_pads_small_patches() -> None:
    points = torch.tensor([[0.0, 0, 0], [0.1, 0, 0], [0.2, 0, 0], [5.0, 0, 0]])
    nodes = torch.tensor([[0.0, 0, 0], [5.0, 0, 0], [20.0, 0, 0]])
    point_to_node, node_masks, knn_indices, knn_masks = point_to_node_partition(points, nodes, point_limit=2)
    assert point_to_node.tolist() == [0, 0, 0, 1]
    assert node_masks.tolist() == [True, True, False]
    # Node 0 keeps its two nearest points, node 1 its single point padded with the shadow index, node 2 is empty.
    assert knn_indices.tolist() == [[0, 1], [3, 4], [4, 4]]
    assert knn_masks.tolist() == [[True, True], [True, False], [False, False]]


def test_node_correspondences_overlap_under_transform() -> None:
    """The pairs come as `(source node, reference node)`, ordered by reference node."""
    ref_nodes = torch.tensor([[0.0, 0, 0], [1.0, 0, 0], [10.0, 0, 0]])
    offsets = torch.tensor([[[0.0, 0, 0], [0.02, 0, 0], [0.0, 0.02, 0]]])
    ref_patches = ref_nodes.unsqueeze(1) + offsets
    transform = _transform(torch.eye(3), torch.tensor([0.0, 0.0, 0.3]))
    order = torch.tensor([2, 0, 1])  # source node i is reference node order[i]
    src_nodes = transform_points(ref_nodes[order], torch.linalg.inv(transform))
    src_patches = transform_points(ref_patches[order], torch.linalg.inv(transform))
    corr, overlaps = node_correspondences(
        src_nodes, ref_nodes, src_patches, ref_patches, transform.unsqueeze(0), radius=0.05
    )
    assert corr.tolist() == [[1, 0], [2, 1], [0, 2]]
    assert torch.allclose(overlaps, torch.ones(3))


def test_node_correspondences_masks_empty_nodes() -> None:
    nodes = torch.tensor([[0.0, 0, 0], [1.0, 0, 0]])
    patches = nodes.unsqueeze(1).repeat(1, 2, 1)
    knn_masks = torch.tensor([[True, True], [True, False]])
    node_masks = torch.tensor([True, False])
    corr, _ = node_correspondences(
        nodes, nodes, patches, patches, torch.eye(4).unsqueeze(0), 0.05, node_masks, node_masks, knn_masks, knn_masks
    )
    assert corr.tolist() == [[0, 0]]


@pytest.mark.parametrize("degrees", [0.0, 30.0, 120.0, 180.0])
def test_rotation_angle(degrees: float) -> None:
    rotation = _rotation("z", degrees)
    assert rotation_angle(rotation).item() == pytest.approx(degrees, abs=1e-3)
    assert rotation_angle(rotation, rotation).item() == pytest.approx(0.0, abs=1e-3)


def test_point_to_node_partition_pads_to_point_limit_with_few_points() -> None:
    points = torch.tensor([[0.0, 0, 0], [5.0, 0, 0]])
    nodes = torch.tensor([[0.0, 0, 0], [5.0, 0, 0]])
    _, _, knn_indices, knn_masks = point_to_node_partition(points, nodes, point_limit=4)
    assert knn_indices.shape == (2, 4) and knn_masks.shape == (2, 4)
    assert knn_indices.tolist() == [[0, 2, 2, 2], [1, 2, 2, 2]]
    assert knn_masks.tolist() == [[True, False, False, False], [True, False, False, False]]


def test_mutual_topk_correspondences_keeps_mutual_entries_above_threshold() -> None:
    scores = torch.tensor([[[0.9, 0.0, 0.0], [0.7, 0.1, 0.03], [0.0, 0.6, 0.02]]])
    selected = mutual_topk_correspondences(scores, k=1, threshold=0.05)
    # (1, 0) is the best of its row but not of its column.
    assert selected.tolist() == [[[True, False, False], [False, False, False], [False, True, False]]]
    assert not mutual_topk_correspondences(scores, k=1, threshold=0.95).any()
    masked = mutual_topk_correspondences(scores, k=2, threshold=0.05, row_mask=torch.tensor([[True, False, True]]))
    assert not masked[0, 1].any()
    assert mutual_topk_correspondences(scores, k=10, threshold=0.0).shape == (1, 3, 3)


def test_local_to_global_registration_selects_the_consistent_hypothesis() -> None:
    torch.manual_seed(0)
    transform = _transform(_rotation("z", 30.0), torch.tensor([0.5, -0.2, 0.1]))
    src = torch.rand(30, 3)
    ref = transform_points(src, transform)
    ref[20:] += torch.rand(10, 3) + 1.0  # the third hypothesis is all outliers
    group = torch.arange(3).repeat_interleave(10)
    estimate = local_to_global_registration(src, ref, torch.ones(30), group, 0.05, 3, 5)
    assert estimate.shape == (1, 4, 4)
    assert torch.allclose(estimate[0], transform, atol=1e-4)


def test_local_to_global_registration_without_large_groups_fits_all_correspondences() -> None:
    src = torch.rand(6, 3)
    transform = _transform(torch.eye(3), torch.tensor([0.3, 0.0, 0.0]))
    estimate = local_to_global_registration(
        src, transform_points(src, transform), torch.ones(6), torch.arange(6), 0.05, 3, 2
    )
    assert torch.allclose(estimate[0], transform, atol=1e-5)


def test_local_to_global_registration_batches_pairs() -> None:
    """Every pair is registered as alone, and a pair without correspondences gets the identity."""
    torch.manual_seed(0)
    first = _transform(_rotation("z", 30.0), torch.tensor([0.5, -0.2, 0.1]))
    second = _transform(_rotation("x", -50.0), torch.tensor([0.0, 1.0, -0.4]))
    src = torch.rand(50, 3)
    ref = torch.cat([transform_points(src[:30], first), transform_points(src[30:], second)])
    ref[20:30] += torch.rand(10, 3) + 1.0  # the third hypothesis of the first pair is all outliers
    group = torch.cat([torch.arange(3).repeat_interleave(10), torch.tensor([3] * 4 + [4] * 16)])
    batch = torch.tensor([0] * 30 + [2] * 20)
    scores = torch.rand(50)
    estimate = local_to_global_registration(src, ref, scores, group, 0.05, 3, 5, batch=batch, batch_size=4)
    assert estimate.shape == (4, 4, 4)
    alone = local_to_global_registration(src[:30], ref[:30], scores[:30], group[:30], 0.05, 3, 5)
    assert torch.allclose(estimate[0], alone[0], atol=1e-6)
    alone = local_to_global_registration(src[30:], ref[30:], scores[30:], group[30:] - 3, 0.05, 3, 5)
    assert torch.allclose(estimate[2], alone[0], atol=1e-6)
    assert torch.allclose(estimate[2], second, atol=1e-4)
    assert torch.equal(estimate[1], torch.eye(4)) and torch.equal(estimate[3], torch.eye(4))


def test_local_to_global_registration_without_correspondences_is_identity() -> None:
    empty = torch.zeros(0, 3)
    group = torch.zeros(0, dtype=torch.long)
    estimate = local_to_global_registration(empty, empty, torch.zeros(0), group, 0.05, 3, 5)
    assert torch.equal(estimate, torch.eye(4).unsqueeze(0))


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_weighted_procrustes_fits_half_precision_in_float32(dtype: torch.dtype) -> None:
    torch.manual_seed(0)
    src = torch.randn(40, 3)
    transform = _transform(_rotation("z", 40.0), torch.tensor([0.5, -1.0, 0.2]))
    ref = transform_points(src, transform)
    estimate = weighted_procrustes(src.to(dtype), ref.to(dtype), torch.ones(40, dtype=dtype))
    assert estimate.dtype == dtype
    assert torch.allclose(estimate.float(), transform, atol=5e-2)


def test_point_to_node_partition_batches_clouds() -> None:
    torch.manual_seed(0)
    points, nodes = [torch.rand(30, 3), torch.rand(20, 3)], [torch.rand(4, 3), torch.rand(6, 3)]
    batch = torch.tensor([0] * 30 + [1] * 20)
    node_batch = torch.tensor([0] * 4 + [1] * 6)
    point_to_node, node_masks, knn_indices, knn_masks = point_to_node_partition(
        torch.cat(points), torch.cat(nodes), 8, batch=batch, node_batch=node_batch
    )
    first = point_to_node_partition(points[0], nodes[0], 8)
    second = point_to_node_partition(points[1], nodes[1], 8)
    assert torch.equal(point_to_node, torch.cat([first[0], second[0] + 4]))
    assert torch.equal(node_masks, torch.cat([first[1], second[1]]))
    # Patch indices are global, padded with the total number of points.
    first_indices = first[2].masked_fill(~first[3], 50)
    second_indices = (second[2] + 30).masked_fill(~second[3], 50)
    assert torch.equal(knn_indices, torch.cat([first_indices, second_indices]))
    assert torch.equal(knn_masks, torch.cat([first[3], second[3]]))


def test_node_correspondences_batches_pairs() -> None:
    nodes = torch.tensor([[0.0, 0, 0], [1.0, 0, 0], [10.0, 0, 0]])
    patches = nodes.unsqueeze(1) + torch.tensor([[[0.0, 0, 0], [0.02, 0, 0]]])
    shift = _transform(torch.eye(3), torch.tensor([0.0, 0.0, 0.3]))
    src_nodes = torch.cat([nodes, transform_points(nodes[:2], torch.linalg.inv(shift))])
    src_patches = torch.cat([patches, transform_points(patches[:2], torch.linalg.inv(shift))])
    transform = torch.stack([torch.eye(4), shift])
    corr, overlaps = node_correspondences(
        src_nodes,
        torch.cat([nodes, nodes]),
        src_patches,
        torch.cat([patches, patches]),
        transform,
        radius=0.05,
        src_batch=torch.tensor([0, 0, 0, 1, 1]),
        ref_batch=torch.tensor([0, 0, 0, 1, 1, 1]),
    )
    assert corr.tolist() == [[0, 0], [1, 1], [2, 2], [3, 3], [4, 4]]
    assert torch.allclose(overlaps, torch.ones(5))
