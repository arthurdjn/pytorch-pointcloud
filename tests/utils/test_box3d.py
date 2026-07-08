from typing import List

import torch
from torch import Tensor

from torch_pointcloud.utils.box3d import boxes_iou3d, boxes_iou_bev, count_points_in_boxes, nms3d


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
