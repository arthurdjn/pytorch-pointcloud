import pytest
import torch
import torch.nn.functional as F
from torch import Tensor

from torch_pointcloud.losses import RoIHeadLoss, roi_head_loss
from torch_pointcloud.ops.box3d import encode_roi_residuals
from torch_pointcloud.utils.data import DataKeys

CLS_FOREGROUND, CLS_BACKGROUND, REG_FOREGROUND = 0.75, 0.25, 0.55


def _rois() -> Tensor:
    return torch.tensor(
        [
            [0.0, 0.0, 0.0, 4.0, 2.0, 1.5, 0.0],
            [10.0, 5.0, 0.0, 4.0, 2.0, 1.5, 1.0],
            [20.0, 0.0, 0.0, 4.0, 2.0, 1.5, -0.5],
        ]
    )


def test_roi_head_loss_is_zero_on_exact_residuals() -> None:
    rois = _rois()
    targets = rois + torch.tensor([0.3, -0.2, 0.1, 0.2, 0.1, 0.0, 0.2])
    _, box_loss, corner = roi_head_loss(
        torch.zeros(3),
        encode_roi_residuals(targets, rois),
        rois,
        targets,
        torch.ones(3),
        cls_foreground_threshold=CLS_FOREGROUND,
        cls_background_threshold=CLS_BACKGROUND,
        reg_foreground_threshold=REG_FOREGROUND,
    )
    assert box_loss.item() == pytest.approx(0.0, abs=1e-6)
    assert corner.item() == pytest.approx(0.0, abs=1e-5)


def test_roi_head_loss_interpolates_the_confidence_target_between_the_thresholds() -> None:
    logits = torch.tensor([0.3, -0.2, 1.0])
    iou = torch.tensor([0.9, 0.5, 0.1])
    cls_loss, _, _ = roi_head_loss(
        logits,
        torch.zeros(3, 7),
        _rois(),
        _rois(),
        iou,
        cls_foreground_threshold=CLS_FOREGROUND,
        cls_background_threshold=CLS_BACKGROUND,
        reg_foreground_threshold=REG_FOREGROUND,
    )
    expected = F.binary_cross_entropy_with_logits(logits, torch.tensor([1.0, 0.5, 0.0]))
    assert torch.allclose(cls_loss, expected)


def test_roi_head_loss_regresses_only_the_foreground() -> None:
    rois = _rois()
    iou = torch.tensor([0.9, 0.5, 0.1])
    codes = torch.zeros(3, 7)
    reference = roi_head_loss(
        torch.zeros(3),
        codes,
        rois,
        rois,
        iou,
        cls_foreground_threshold=CLS_FOREGROUND,
        cls_background_threshold=CLS_BACKGROUND,
        reg_foreground_threshold=REG_FOREGROUND,
    )
    codes[1:] = 5.0
    background_changed = roi_head_loss(
        torch.zeros(3),
        codes,
        rois,
        rois,
        iou,
        cls_foreground_threshold=CLS_FOREGROUND,
        cls_background_threshold=CLS_BACKGROUND,
        reg_foreground_threshold=REG_FOREGROUND,
    )
    assert torch.equal(reference[1], background_changed[1]) and torch.equal(reference[2], background_changed[2])


def test_roi_head_loss_module_gathers_the_matched_boxes() -> None:
    criterion = RoIHeadLoss(
        cls_foreground_threshold=CLS_FOREGROUND,
        cls_background_threshold=CLS_BACKGROUND,
        reg_foreground_threshold=REG_FOREGROUND,
        corner_weight=0.5,
    )
    rois = _rois()
    gt = rois[[2, 0]] + 0.1
    output = {
        "rcnn_cls": torch.zeros(3, 1, requires_grad=True),
        "rcnn_reg": torch.zeros(3, 7, requires_grad=True),
        "rois": rois,
        "roi_iou": torch.tensor([0.8, 0.0, 0.7]),
        "roi_target_index": torch.tensor([1, -1, 0]),
    }
    losses = criterion(output, {DataKeys.BOX: gt})
    cls_loss, box_loss, corner = roi_head_loss(
        output["rcnn_cls"].view(-1),
        output["rcnn_reg"],
        rois,
        gt[[1, 0, 0]],
        output["roi_iou"],
        cls_foreground_threshold=CLS_FOREGROUND,
        cls_background_threshold=CLS_BACKGROUND,
        reg_foreground_threshold=REG_FOREGROUND,
    )
    assert torch.allclose(losses["loss"], cls_loss + box_loss + 0.5 * corner)
    assert set(losses) == {"loss", "rcnn_cls_loss", "rcnn_box_loss", "rcnn_corner_loss"}
    losses["loss"].backward()
    assert output["rcnn_reg"].grad is not None and output["rcnn_reg"].grad[1].abs().sum() == 0


def test_roi_head_loss_module_without_ground_truth() -> None:
    criterion = RoIHeadLoss(
        cls_foreground_threshold=CLS_FOREGROUND,
        cls_background_threshold=CLS_BACKGROUND,
        reg_foreground_threshold=REG_FOREGROUND,
    )
    output = {
        "rcnn_cls": torch.zeros(3, 1),
        "rcnn_reg": torch.zeros(3, 7),
        "rois": _rois(),
        "roi_iou": torch.zeros(3),
        "roi_target_index": torch.full((3,), -1),
    }
    losses = criterion(output, {DataKeys.BOX: torch.zeros(0, 7)})
    assert losses["rcnn_box_loss"].item() == 0.0 and losses["rcnn_corner_loss"].item() == 0.0
    assert torch.isfinite(losses["loss"])


def test_roi_head_loss_module_rejects_inverted_thresholds() -> None:
    with pytest.raises(ValueError, match="below"):
        RoIHeadLoss(cls_foreground_threshold=0.25, cls_background_threshold=0.75, reg_foreground_threshold=0.55)
