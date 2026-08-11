import math
from typing import List

import pytest
import torch
from torch import Tensor

from torch_pointcloud.utils.box3d import (
    box3d_overlap,
    box_corners,
    boxes_iou3d,
    boxes_iou_bev,
    count_points_in_boxes,
    nms3d,
)


def _boxes(rows: List[List[float]]) -> Tensor:
    return torch.tensor(rows, dtype=torch.float32)


def test_boxes_iou_bev_half_offset_unit_boxes() -> None:
    a = _boxes([[0.0, 0, 0, 1, 1, 1, 0]])
    b = _boxes([[0.5, 0, 0, 1, 1, 1, 0]])
    assert torch.allclose(boxes_iou_bev(a, b), torch.tensor([[1.0 / 3.0]]), atol=1e-6)


def test_boxes_iou3d_half_offset_unit_boxes() -> None:
    a = _boxes([[0.0, 0, 0, 1, 1, 1, 0]])
    b = _boxes([[0.5, 0, 0, 1, 1, 1, 0]])
    assert torch.allclose(boxes_iou3d(a, b), torch.tensor([[1.0 / 3.0]]), atol=1e-6)


def test_boxes_iou_self_diag_is_one() -> None:
    boxes = _boxes([[1.0, 2, 3, 2, 3, 1, 0.5], [-4.0, 0, 1, 1, 1, 2, -1.2], [0.0, 0, 0, 3, 2, 2, 0.7853981633974483]])
    diag_bev = torch.diag(boxes_iou_bev(boxes, boxes))
    diag_3d = torch.diag(boxes_iou3d(boxes, boxes))
    assert torch.allclose(diag_bev, torch.ones(3), atol=1e-5)
    assert torch.allclose(diag_3d, torch.ones(3), atol=1e-5)


def test_box3d_overlap_degenerate_boxes_finite_zero_iou() -> None:
    # Zero-volume boxes must give IoU 0, not NaN or a division error.
    corners = box_corners(_boxes([[0.0, 0, 0, 0, 0, 0, 0]]))
    inter, iou = box3d_overlap(corners, corners)
    assert torch.isfinite(iou).all()
    assert iou.item() == 0.0
    assert inter.item() == 0.0


def test_nms3d_class_aware_suppresses_same_class_overlap() -> None:
    boxes = _boxes([[0.0, 0, 0, 2, 2, 2, 0], [0.1, 0, 0, 2, 2, 2, 0], [10.0, 10, 10, 2, 2, 2, 0]])
    scores = torch.tensor([0.9, 0.5, 0.8])
    labels = torch.tensor([0, 0, 1])
    assert nms3d(boxes, scores, 0.3, labels=labels).tolist() == [0, 2]


def test_nms3d_class_agnostic_when_labels_omitted() -> None:
    boxes = _boxes([[0.0, 0, 0, 2, 2, 2, 0], [0.0, 0, 0, 2, 2, 2, 0]])
    scores = torch.tensor([0.9, 0.7])
    assert nms3d(boxes, scores, 0.3).tolist() == [0]


def test_nms3d_per_class_keeps_different_class_overlap() -> None:
    boxes = _boxes([[0.0, 0, 0, 2, 2, 2, 0], [0.0, 0, 0, 2, 2, 2, 0]])
    scores = torch.tensor([0.9, 0.7])
    labels = torch.tensor([0, 1])
    assert nms3d(boxes, scores, 0.3, labels=labels).tolist() == [0, 1]


def test_nms3d_batched_runs_per_scene() -> None:
    boxes = _boxes([[0.0, 0, 0, 2, 2, 2, 0], [0.1, 0, 0, 2, 2, 2, 0]])
    scores = torch.tensor([0.9, 0.8])
    batch = torch.tensor([0, 1])
    assert nms3d(boxes, scores, 0.3, batch=batch).tolist() == [0, 1]


def test_nms3d_overlapping_zero_height_boxes_reduce_to_one() -> None:
    boxes = _boxes([[0.0, 0, 0, 2, 2, 0, 0], [0.1, 0, 0, 2, 2, 0, 0], [0.0, 0.1, 0, 2, 2, 0, 0]])
    scores = torch.tensor([0.9, 0.8, 0.7])
    assert nms3d(boxes, scores, 0.3).tolist() == [0]


def test_count_points_in_boxes() -> None:
    pos = torch.tensor([[0.0, 0, 0], [0.5, 0, 0], [5.0, 5, 5]])
    boxes = _boxes([[0.0, 0, 0, 2, 2, 2, 0], [5.0, 5, 5, 0.1, 0.1, 0.1, 0]])
    assert count_points_in_boxes(pos, boxes).tolist() == [2, 1]


def test_count_points_in_boxes_respects_scene_batch() -> None:
    # Two scenes with identical local coordinates; a box only counts points from its own scene.
    pos = torch.tensor([[0.0, 0, 0], [0.0, 0, 0]])
    pos_batch = torch.tensor([0, 1])
    boxes = _boxes([[0.0, 0, 0, 2, 2, 2, 0], [0.0, 0, 0, 2, 2, 2, 0]])
    box_batch = torch.tensor([0, 1])
    assert count_points_in_boxes(pos, boxes, pos_batch=pos_batch, box_batch=box_batch).tolist() == [1, 1]


def test_count_points_in_boxes_rejects_lone_batch_argument() -> None:
    pos = torch.tensor([[0.0, 0, 0]])
    boxes = _boxes([[0.0, 0, 0, 2, 2, 2, 0]])
    with pytest.raises(ValueError, match="`pos_batch` and `box_batch` must be given together"):
        count_points_in_boxes(pos, boxes, pos_batch=torch.tensor([0]))
    with pytest.raises(ValueError, match="`pos_batch` and `box_batch` must be given together"):
        count_points_in_boxes(pos, boxes, box_batch=torch.tensor([0]))


@pytest.mark.parametrize(
    "heading",
    [
        pytest.param(0.0, id="axis-aligned"),
        pytest.param(0.3222, id="0.3222"),
        pytest.param(0.5639, id="0.5639"),
        pytest.param(0.7, id="0.7"),
        pytest.param(0.8055, id="0.8055"),
        pytest.param(math.pi / 4, id="pi/4"),
        pytest.param(math.pi / 2, id="pi/2"),
        pytest.param(2.1749, id="2.1749"),
        pytest.param(3.0, id="3.0"),
    ],
)
def test_box3d_overlap_self_iou_is_one_at_any_heading(heading: float) -> None:
    box = _boxes([[0.0, 0, 0, 2, 4, 2, heading]])
    corners = box_corners(box)
    inter, iou = box3d_overlap(corners, corners)
    assert iou.item() == pytest.approx(1.0, abs=1e-4)
    assert inter.item() == pytest.approx(16.0, abs=1e-3)
    assert boxes_iou3d(box, box).item() == pytest.approx(1.0, abs=1e-4)


def test_box3d_overlap_disjoint_rotated_boxes_zero() -> None:
    a = _boxes([[0.0, 0, 0, 2, 4, 2, 0.7]])
    b = _boxes([[20.0, 0, 0, 2, 4, 2, 1.3]])
    inter, iou = box3d_overlap(box_corners(a), box_corners(b))
    assert inter.item() == 0.0
    assert iou.item() == 0.0
    assert boxes_iou3d(a, b).item() == 0.0


def test_box3d_overlap_45deg_partial_overlap_matches_hand_value() -> None:
    # Concentric 2x2 squares, one rotated 45 degrees: the BEV intersection is the regular octagon of
    # area 8(sqrt(2) - 1); with full z overlap the 3D IoU reduces to 1/sqrt(2).
    a = _boxes([[0.0, 0, 0, 2, 2, 2, 0]])
    b = _boxes([[0.0, 0, 0, 2, 2, 2, math.pi / 4]])
    inter, iou = box3d_overlap(box_corners(a), box_corners(b))
    assert inter.item() == pytest.approx(16.0 * (math.sqrt(2.0) - 1.0), abs=1e-3)
    assert iou.item() == pytest.approx(1.0 / math.sqrt(2.0), abs=1e-3)
    assert iou.item() == pytest.approx(boxes_iou3d(a, b).item(), abs=1e-5)


def test_box3d_overlap_axis_aligned_half_offset_known_value() -> None:
    # Axis-aligned 2x2x2 cubes offset by half an edge: intersection 1x2x2 = 4, IoU 4 / 12 = 1/3.
    a = _boxes([[0.0, 0, 0, 2, 2, 2, 0]])
    b = _boxes([[1.0, 0, 0, 2, 2, 2, 0]])
    inter, iou = box3d_overlap(box_corners(a), box_corners(b))
    assert inter.item() == pytest.approx(4.0, abs=1e-5)
    assert iou.item() == pytest.approx(1.0 / 3.0, abs=1e-5)


def test_box3d_overlap_returns_tensors_on_input_device() -> None:
    corners = box_corners(_boxes([[0.0, 0, 0, 2, 4, 2, 0.7]]))
    inter, iou = box3d_overlap(corners, corners)
    assert inter.device == corners.device
    assert iou.device == corners.device


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_box3d_overlap_returns_tensors_on_cuda_device() -> None:
    corners = box_corners(_boxes([[0.0, 0, 0, 2, 4, 2, 0.7]])).cuda()
    inter, iou = box3d_overlap(corners, corners)
    assert inter.device.type == "cuda"
    assert iou.device.type == "cuda"
    assert iou.item() == pytest.approx(1.0, abs=1e-4)


def test_box3d_overlap_empty_inputs_return_empty_matrices() -> None:
    empty = box_corners(torch.empty(0, 7))
    one = box_corners(_boxes([[0.0, 0, 0, 2, 2, 2, 0.3]]))
    inter, iou = box3d_overlap(empty, one)
    assert inter.shape == (0, 1)
    assert iou.shape == (0, 1)
    inter, iou = box3d_overlap(one, empty)
    assert inter.shape == (1, 0)
    assert iou.shape == (1, 0)


def test_boxes_iou3d_empty_inputs_return_empty_matrices() -> None:
    empty = torch.empty(0, 7)
    one = _boxes([[0.0, 0, 0, 2, 2, 2, 0.3]])
    assert boxes_iou3d(empty, one).shape == (0, 1)
    assert boxes_iou3d(one, empty).shape == (1, 0)
    assert boxes_iou3d(empty, empty).shape == (0, 0)


def test_boxes_iou_bev_empty_inputs_return_empty_matrices() -> None:
    empty = torch.empty(0, 7)
    one = _boxes([[0.0, 0, 0, 2, 2, 2, 0.3]])
    assert boxes_iou_bev(empty, one).shape == (0, 1)
    assert boxes_iou_bev(one, empty).shape == (1, 0)


def test_nms3d_empty_boxes_returns_empty_long_indices() -> None:
    keep = nms3d(torch.empty(0, 7), torch.empty(0), 0.5)
    assert keep.shape == (0,)
    assert keep.dtype == torch.long


def test_count_points_in_boxes_empty_boxes_returns_empty_counts() -> None:
    counts = count_points_in_boxes(torch.randn(5, 3), torch.empty(0, 7))
    assert counts.shape == (0,)
    assert counts.dtype == torch.long
