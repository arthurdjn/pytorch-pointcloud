import math

import pytest
import torch

from torch_pointcloud.ops.geometry import transform_points
from torch_pointcloud.ops.registration import (
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
    nodes = torch.tensor([[0.0, 0, 0], [1.0, 0, 0], [10.0, 0, 0]])
    offsets = torch.tensor([[[0.0, 0, 0], [0.02, 0, 0], [0.0, 0.02, 0]]])
    ref_patches = nodes.unsqueeze(1) + offsets
    transform = _transform(torch.eye(3), torch.tensor([0.0, 0.0, 0.3]))
    src_nodes = transform_points(nodes, torch.linalg.inv(transform))
    src_patches = transform_points(ref_patches, torch.linalg.inv(transform))
    corr, overlaps = node_correspondences(nodes, src_nodes, ref_patches, src_patches, transform, radius=0.05)
    assert corr.tolist() == [[0, 0], [1, 1], [2, 2]]
    assert torch.allclose(overlaps, torch.ones(3))


def test_node_correspondences_masks_empty_nodes() -> None:
    nodes = torch.tensor([[0.0, 0, 0], [1.0, 0, 0]])
    patches = nodes.unsqueeze(1).repeat(1, 2, 1)
    knn_masks = torch.tensor([[True, True], [True, False]])
    node_masks = torch.tensor([True, False])
    corr, _ = node_correspondences(
        nodes, nodes, patches, patches, torch.eye(4), 0.05, node_masks, node_masks, knn_masks, knn_masks
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
