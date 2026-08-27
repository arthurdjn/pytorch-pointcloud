import math
from typing import Any, Dict, Tuple

import pytest
import torch
from torch import Tensor

from torch_pointcloud.losses import PointRCNNLoss
from torch_pointcloud.losses.pointrcnn import _encode_point_residuals
from torch_pointcloud.utils.data import DataKeys

_MEAN_SIZES = [[3.9, 1.6, 1.56], [0.8, 0.6, 1.73], [1.76, 0.6, 1.73]]
_BOX = [10.0, 0.0, -1.0, 3.9, 1.6, 1.56, 0.3]
_BETA = 1.0 / 9.0


def _loss() -> PointRCNNLoss:
    return PointRCNNLoss(num_classes=3, mean_sizes=_MEAN_SIZES)


def _perfect_data(gt_shift_x: float = 0.0) -> Tuple[Dict[str, Tensor], Dict[str, Any]]:
    """A single foreground point / ROI scene where every prediction equals its target.

    The single ROI equals the GT box shifted by `gt_shift_x` (0 keeps the ROI exactly on the box), so the
    stage-2 residual and corner targets are hand-computable.
    """
    gt_box = torch.tensor([_BOX])
    gt_box[0, 0] += gt_shift_x
    roi = torch.tensor([_BOX])
    roi[0, 6] = 0.0
    gt_box[0, 6] = 0.0

    point_pos = torch.rand(20, 3) * 2 + torch.tensor([50.0, 20.0, 0.0])
    point_pos[0] = gt_box[0, :3]
    point_cls_preds = torch.full((20, 3), -10.0)
    point_cls_preds[0] = torch.tensor([10.0, -10.0, -10.0])
    point_box_preds = torch.zeros(20, 8)
    point_box_preds[0] = _encode_point_residuals(gt_box, point_pos[0:1], torch.tensor([1]), torch.tensor(_MEAN_SIZES))

    gt_of_rois = gt_box.clone()
    gt_of_rois[0, :3] = gt_box[0, :3] - roi[0, :3]
    output = {
        "point_cls_preds": point_cls_preds,
        "point_box_preds": point_box_preds,
        "point_pos": point_pos,
        "point_batch": torch.zeros(20, dtype=torch.long),
        "rcnn_cls": torch.tensor([[10.0]]),
        "rcnn_reg": torch.zeros(1, 7),
        "rcnn_boxes": roi.clone(),
        "rois": roi,
        "gt_of_rois": gt_of_rois,
        "gt_of_rois_src": gt_box,
        "roi_ious": torch.tensor([0.9]),
    }
    batch: Dict[str, Any] = {
        DataKeys.BOX: gt_box,
        DataKeys.LABEL: torch.tensor([0]),
        DataKeys.BATCH_BOX: torch.tensor([0]),
    }
    return output, batch


def test_pointrcnn_loss_perfect_predictions_near_zero() -> None:
    output, batch = _perfect_data()
    out = _loss()(output, batch)
    assert out["point_box_loss"] == 0.0
    assert out["rcnn_box_loss"] == 0.0
    assert out["point_cls_loss"] < 1e-3
    assert out["rcnn_cls_loss"] < 1e-3
    assert out["loss"] < 1e-2


def test_pointrcnn_loss_perturbed_rcnn_regression_value() -> None:
    """A 0.2 residual error on the single foreground ROI costs exactly `smooth_l1(0.2)`."""
    output, batch = _perfect_data()
    perfect = _loss()(output, batch)
    output["rcnn_reg"] = output["rcnn_reg"].clone()
    output["rcnn_reg"][0, 0] = 0.2
    out = _loss()(output, batch)
    expected = 0.2 - _BETA / 2
    assert torch.isclose(out["rcnn_box_loss"], torch.tensor(expected), atol=1e-6)
    assert out["loss"] > perfect["loss"]


def test_pointrcnn_loss_shifted_gt_box_value() -> None:
    """A GT box 0.5 m off the ROI: residual `0.5 / diagonal` plus a 0.125 corner term, both hand-computed."""
    output, batch = _perfect_data(gt_shift_x=0.5)
    out = _loss()(output, batch)
    diagonal = math.sqrt(3.9**2 + 1.6**2)
    residual_term = 0.5 / diagonal - _BETA / 2
    corner_term = 0.5 * 0.5**2  # all 8 corners are 0.5 m off, inside the smooth-l1 quadratic zone
    assert torch.isclose(out["rcnn_box_loss"], torch.tensor(residual_term + corner_term), atol=1e-4)


def test_pointrcnn_loss_wrong_point_class_is_penalized() -> None:
    output, batch = _perfect_data()
    perfect = _loss()(output, batch)
    output["point_cls_preds"] = output["point_cls_preds"].clone()
    output["point_cls_preds"][0] = torch.tensor([-10.0, -10.0, 10.0])  # foreground point claims class 3
    out = _loss()(output, batch)
    assert out["point_cls_loss"] > perfect["point_cls_loss"] + 1.0


def test_pointrcnn_loss_ignored_shell_points_do_not_contribute() -> None:
    """A point between a box and its enlarged copy is ignored: flipping its logits changes nothing."""
    output, batch = _perfect_data()
    output["point_pos"] = output["point_pos"].clone()
    output["point_pos"][1] = torch.tensor(_BOX[:3]) + torch.tensor([3.9 / 2 + 0.05, 0.0, 0.0])
    base = _loss()(output, batch)
    output["point_cls_preds"] = output["point_cls_preds"].clone()
    output["point_cls_preds"][1] = 10.0
    flipped = _loss()(output, batch)
    assert torch.isclose(base["point_cls_loss"], flipped["point_cls_loss"])


def test_pointrcnn_loss_background_roi_has_no_box_loss() -> None:
    output, batch = _perfect_data()
    output["roi_ious"] = torch.tensor([0.1])
    output["rcnn_reg"] = torch.full((1, 7), 5.0)
    out = _loss()(output, batch)
    assert out["rcnn_box_loss"] == 0.0


def test_pointrcnn_loss_no_boxes_is_finite() -> None:
    output, batch = _perfect_data()
    batch[DataKeys.BOX] = batch[DataKeys.BOX][:0]
    batch[DataKeys.LABEL] = batch[DataKeys.LABEL][:0]
    batch[DataKeys.BATCH_BOX] = batch[DataKeys.BATCH_BOX][:0]
    output["roi_ious"] = torch.tensor([0.0])
    assert torch.isfinite(_loss()(output, batch)["loss"])


def test_pointrcnn_loss_backward() -> None:
    output, batch = _perfect_data(gt_shift_x=0.5)
    for key in ("point_cls_preds", "point_box_preds", "rcnn_cls", "rcnn_reg", "rcnn_boxes"):
        output[key].requires_grad_(True)
    _loss()(output, batch)["loss"].backward()
    assert output["rcnn_reg"].grad is not None
    grad = output["point_cls_preds"].grad
    assert grad is not None and torch.isfinite(grad).all()


def test_pointrcnn_loss_bad_mean_sizes_shape() -> None:
    with pytest.raises(ValueError, match="mean_sizes"):
        PointRCNNLoss(num_classes=3, mean_sizes=[[1.0, 1.0, 1.0]])
