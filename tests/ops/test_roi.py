import math
from typing import List

import pytest
import torch
from torch import Tensor

from torch_pointcloud.ops.box3d import boxes_iou3d
from torch_pointcloud.ops.roi import (
    RoISamples,
    decode_roi_residuals,
    encode_roi_residuals,
    roi_aware_pool3d,
    roi_grid_points,
    sample_rois,
)


def _boxes(rows: List[List[float]]) -> Tensor:
    return torch.tensor(rows, dtype=torch.float32)


def test_roi_aware_pool3d_reduces_every_cell() -> None:
    x = torch.tensor([[1.0], [3.0], [5.0]])
    pos = torch.tensor([[-0.5, 0.0, 0.0], [-0.4, 0.0, 0.0], [0.5, 0.0, 0.0]])
    boxes = _boxes([[0.0, 0.0, 0.0, 2.0, 2.0, 2.0, 0.0]])
    out_max = roi_aware_pool3d(x, pos, boxes, 2, reduce="max")
    out_mean = roi_aware_pool3d(x, pos, boxes, 2, reduce="mean")
    assert out_max.shape == (1, 2, 2, 2, 1)
    assert out_max[0, :, 1, 1, 0].tolist() == [3.0, 5.0]
    assert out_mean[0, :, 1, 1, 0].tolist() == [2.0, 5.0]
    assert out_max.sum().item() == 8.0


def test_roi_aware_pool3d_cells_follow_the_box_heading() -> None:
    # A quarter turn maps the box's +x axis onto the scene's +y axis.
    x = torch.tensor([[7.0]])
    pos = torch.tensor([[0.0, 0.5, 0.0]])
    boxes = _boxes([[0.0, 0.0, 0.0, 2.0, 2.0, 2.0, math.pi / 2]])
    out = roi_aware_pool3d(x, pos, boxes, 2)
    assert out[0, 1, 1, 1, 0].item() == 7.0
    assert out.sum().item() == 7.0


def test_roi_aware_pool3d_keeps_the_first_points_of_a_cell() -> None:
    x = torch.tensor([[1.0], [3.0], [100.0]])
    pos = torch.full((3, 3), 0.5)
    boxes = _boxes([[0.0, 0.0, 0.0, 2.0, 2.0, 2.0, 0.0]])
    out = roi_aware_pool3d(x, pos, boxes, 2, max_points_per_cell=2, reduce="mean")
    assert out[0, 1, 1, 1, 0].item() == 2.0


def test_roi_aware_pool3d_excludes_points_on_the_side_faces() -> None:
    x = torch.ones(2, 1)
    pos = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    boxes = _boxes([[0.0, 0.0, 0.0, 2.0, 2.0, 2.0, 0.0]])
    out = roi_aware_pool3d(x, pos, boxes, 2)
    # The x face is open, the top face is closed and clamps into the last cell.
    assert out.sum().item() == 1.0
    assert out[0, 1, 1, 1, 0].item() == 1.0


def test_roi_aware_pool3d_respects_scene_batch() -> None:
    x = torch.tensor([[1.0], [2.0]])
    pos = torch.zeros(2, 3)
    boxes = _boxes([[0.0, 0.0, 0.0, 2.0, 2.0, 2.0, 0.0], [0.0, 0.0, 0.0, 2.0, 2.0, 2.0, 0.0]])
    out = roi_aware_pool3d(x, pos, boxes, 1, pos_batch=torch.tensor([0, 1]), box_batch=torch.tensor([0, 1]))
    assert out.view(-1).tolist() == [1.0, 2.0]


def test_decode_roi_residuals_in_the_roi_frame() -> None:
    rois = _boxes([[10.0, 5.0, 1.0, 4.0, 2.0, 2.0, math.pi / 2]])
    diagonal = math.sqrt(4.0**2 + 2.0**2)
    encodings = torch.tensor([[1.0 / diagonal, 0.0, 0.5, math.log(2.0), 0.0, 0.0, 0.1]])
    boxes = decode_roi_residuals(encodings, rois)
    # One length unit ahead of a quarter-turned RoI is one unit along +y; z moves by half the RoI height.
    expected = torch.tensor([[10.0, 6.0, 2.0, 8.0, 2.0, 2.0, math.pi / 2 + 0.1]])
    assert torch.allclose(boxes, expected, atol=1e-5)


def test_roi_grid_points_fill_the_box_cells() -> None:
    rois = torch.tensor([[1.0, 2.0, 3.0, 2.0, 4.0, 6.0, 0.0]])
    grid = roi_grid_points(rois, 2)
    assert grid.shape == (1, 8, 3)
    # x varies slowest, z fastest.
    assert grid[0, 0].tolist() == [0.5, 1.0, 1.5]
    assert grid[0, 1].tolist() == [0.5, 1.0, 4.5]
    assert grid[0, 7].tolist() == [1.5, 3.0, 4.5]


def test_roi_grid_points_rotate_with_the_heading() -> None:
    rois = torch.tensor([[0.0, 0.0, 0.0, 2.0, 2.0, 2.0, math.pi / 2]])
    grid = roi_grid_points(rois, 2)
    # The local (+0.5, -0.5) corner of the cells lands at (+0.5, +0.5) after a quarter turn.
    assert torch.allclose(grid[0, 4], torch.tensor([0.5, 0.5, -0.5]), atol=1e-6)


def test_encode_roi_residuals_inverts_decode_roi_residuals() -> None:
    torch.manual_seed(0)
    rois = torch.cat([torch.randn(64, 3) * 10, torch.rand(64, 3) * 3 + 0.5, torch.rand(64, 1) * 12 - 6], dim=1)
    boxes = rois + torch.cat([torch.randn(64, 3) * 0.3, torch.randn(64, 3) * 0.1, torch.randn(64, 1) * 0.3], dim=1)
    boxes[:, 3:6] = boxes[:, 3:6].abs() + 0.1
    decoded = decode_roi_residuals(encode_roi_residuals(boxes, rois), rois)
    assert torch.allclose(decoded[:, :6], boxes[:, :6], atol=1e-4)
    assert torch.allclose(torch.sin(decoded[:, 6] - boxes[:, 6]), torch.zeros(64), atol=1e-4)


def test_encode_roi_residuals_folds_an_opposite_heading_by_a_half_turn() -> None:
    rois = _boxes([[0.0, 0.0, 0.0, 4.0, 2.0, 1.5, 0.0]])
    facing_away = _boxes([[0.0, 0.0, 0.0, 4.0, 2.0, 1.5, math.pi - 0.1]])
    assert encode_roi_residuals(facing_away, rois)[0, 6].item() == pytest.approx(-0.1, abs=1e-5)


def _sample(
    rois: Tensor, roi_labels: Tensor, roi_batch: Tensor, boxes: Tensor, labels: Tensor, batch: Tensor
) -> RoISamples:
    return sample_rois(
        rois,
        roi_labels,
        roi_batch,
        boxes,
        labels,
        batch,
        num_samples=8,
        foreground_ratio=0.5,
        foreground_threshold=0.55,
        hard_background_threshold=0.1,
        hard_background_ratio=0.8,
    )


def test_sample_rois_draws_a_fixed_number_per_scene_with_capped_foreground() -> None:
    torch.manual_seed(0)
    boxes = _boxes([[0.0, 0.0, 0.0, 4.0, 2.0, 1.5, 0.0], [30.0, 0.0, 0.0, 4.0, 2.0, 1.5, 0.0]])
    offsets = torch.tensor([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 1.5, 2.5, 3.0, 20.0, 40.0, 60.0])
    rois = boxes[0].repeat(12, 1)
    rois[:, 0] += offsets
    rois = torch.cat([rois, rois + torch.tensor([30.0, 0, 0, 0, 0, 0, 0])])
    roi_batch = torch.arange(2).repeat_interleave(12)
    zeros = torch.zeros(24, dtype=torch.long)
    samples = _sample(rois, zeros, roi_batch, boxes, zeros[:2], torch.tensor([0, 1]))

    assert samples["index"].shape == samples["target_index"].shape == samples["iou"].shape == (16,)
    assert roi_batch[samples["index"]].tolist() == [0] * 8 + [1] * 8
    foreground = samples["iou"] >= 0.55
    assert foreground.view(2, 8).sum(dim=1).tolist() == [4, 4]
    matched = samples["target_index"][foreground]
    assert torch.equal(matched, roi_batch[samples["index"]][foreground])
    expected = boxes_iou3d(rois[samples["index"][foreground]], boxes[matched], aligned=True)
    assert torch.allclose(samples["iou"][foreground], expected)


def test_sample_rois_matches_only_boxes_of_the_roi_class() -> None:
    boxes = _boxes([[0.0, 0.0, 0.0, 4.0, 2.0, 1.5, 0.0]])
    zero = torch.zeros(1, dtype=torch.long)
    samples = _sample(boxes, torch.ones(1, dtype=torch.long), zero, boxes, zero, zero)
    assert samples["iou"].tolist() == [0.0] * 8
    assert samples["target_index"].tolist() == [-1] * 8


def test_sample_rois_without_background_fills_the_samples_with_foreground() -> None:
    boxes = _boxes([[0.0, 0.0, 0.0, 4.0, 2.0, 1.5, 0.0]])
    zero = torch.zeros(1, dtype=torch.long)
    samples = _sample(boxes, zero, zero, boxes, zero, zero)
    assert samples["index"].tolist() == [0] * 8
    assert samples["target_index"].tolist() == [0] * 8


def test_sample_rois_scene_without_boxes_is_all_background() -> None:
    rois = _boxes([[0.0, 0.0, 0.0, 4.0, 2.0, 1.5, 0.0], [10.0, 0.0, 0.0, 4.0, 2.0, 1.5, 0.0]])
    zero = torch.zeros(2, dtype=torch.long)
    samples = _sample(rois, zero, zero, rois[:0], zero[:0], zero[:0])
    assert samples["iou"].tolist() == [0.0] * 8
    assert samples["target_index"].tolist() == [-1] * 8
