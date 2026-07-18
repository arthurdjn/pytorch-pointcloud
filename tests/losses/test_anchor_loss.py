from typing import Any, Dict, Tuple

import pytest
import torch
from torch import Tensor

from torch_pointcloud.layers.anchors import AnchorHeadMultiOutput
from torch_pointcloud.losses import AnchorLoss, MultiHeadAnchorLoss
from torch_pointcloud.utils.data import DataKeys

_POINT_CLOUD_RANGE = (0.0, -4.0, -2.0, 8.0, 4.0, 2.0)
_VOXEL_SIZE = (1.0, 1.0, 4.0)


def _single_loss() -> AnchorLoss:
    return AnchorLoss(
        1,
        voxel_size=_VOXEL_SIZE,
        point_cloud_range=_POINT_CLOUD_RANGE,
        anchor_sizes=[[3.0, 1.5, 1.5]],
        anchor_bottom_heights=[-1.0],
        feature_map_stride=1,
        matched_thresholds=[0.6],
        unmatched_thresholds=[0.45],
    )


def _perfect_single(loss_fn: AnchorLoss) -> Tuple[Dict[str, Tensor], Dict[str, Any], int]:
    """One GT box equal to a rotation-0 anchor, with confident logits and zero-residual box predictions."""
    anchors = loss_fn.anchors
    idx = int((anchors[:, 6] == 0).nonzero(as_tuple=False)[10])
    gt_box = anchors[idx : idx + 1].clone()

    num_anchors = anchors.shape[0]
    cls = torch.full((1, num_anchors, 1), -10.0)
    cls[0, idx, 0] = 10.0
    box = torch.zeros(1, num_anchors, 7)
    dir_cls = torch.zeros(1, num_anchors, 2)
    dir_cls[..., 0] = -10.0
    dir_cls[..., 1] = 10.0  # heading 0 with dir_offset 0.78539 falls in bin 1

    output = {"cls": cls, "box": box, "dir_cls": dir_cls}
    batch: Dict[str, Any] = {
        DataKeys.BOX: gt_box,
        DataKeys.LABEL: torch.tensor([0]),
        DataKeys.BATCH_BOX: torch.tensor([0]),
    }
    return output, batch, idx


def test_anchor_loss_perfect_predictions_near_zero() -> None:
    loss_fn = _single_loss()
    output, batch, _ = _perfect_single(loss_fn)
    out = loss_fn(output, batch)
    assert out["box_loss"] == 0.0
    assert out["cls_loss"] < 1e-3
    assert out["dir_loss"] < 1e-3
    assert out["loss"] < 1e-2


def test_anchor_loss_perturbed_box_value() -> None:
    """A 0.2 residual error on the positive anchor costs `loc_weight * smooth_l1(0.2)` exactly."""
    loss_fn = _single_loss()
    output, batch, idx = _perfect_single(loss_fn)
    perfect = loss_fn(output, batch)
    output["box"][0, idx, 0] = 0.2
    out = loss_fn(output, batch)
    beta = 1.0 / 9.0
    expected = 2.0 * (0.2 - beta / 2)
    assert torch.isclose(out["box_loss"], torch.tensor(expected), atol=1e-6)
    assert out["loss"] > perfect["loss"]


def test_anchor_loss_wrong_class_logits_are_penalized() -> None:
    loss_fn = _single_loss()
    output, batch, idx = _perfect_single(loss_fn)
    perfect = loss_fn(output, batch)
    output["cls"][0, idx, 0] = -10.0  # the positive anchor now denies the object
    out = loss_fn(output, batch)
    assert out["cls_loss"] > perfect["cls_loss"] + 1.0


def test_anchor_loss_no_boxes_is_finite() -> None:
    loss_fn = _single_loss()
    output, batch, _ = _perfect_single(loss_fn)
    batch[DataKeys.BOX] = batch[DataKeys.BOX][:0]
    batch[DataKeys.LABEL] = batch[DataKeys.LABEL][:0]
    batch[DataKeys.BATCH_BOX] = batch[DataKeys.BATCH_BOX][:0]
    out = loss_fn(output, batch)
    assert torch.isfinite(out["loss"])
    assert out["box_loss"] == 0.0


def test_anchor_loss_backward() -> None:
    loss_fn = _single_loss()
    output, batch, _ = _perfect_single(loss_fn)
    for value in output.values():
        value.requires_grad_(True)
    loss_fn(output, batch)["loss"].backward()
    assert output["cls"].grad is not None
    assert output["box"].grad is not None


def test_anchor_loss_validates_per_class_geometry() -> None:
    with pytest.raises(ValueError, match="anchor_sizes"):
        AnchorLoss(
            2,
            voxel_size=_VOXEL_SIZE,
            point_cloud_range=_POINT_CLOUD_RANGE,
            anchor_sizes=[[3.0, 1.5, 1.5]],
            anchor_bottom_heights=[-1.0],
            feature_map_stride=1,
            matched_thresholds=[0.6, 0.6],
            unmatched_thresholds=[0.45, 0.45],
        )


def _multi_loss() -> MultiHeadAnchorLoss:
    return MultiHeadAnchorLoss(
        2,
        class_groups=[[0], [1]],
        voxel_size=_VOXEL_SIZE,
        point_cloud_range=_POINT_CLOUD_RANGE,
        anchor_sizes=[[3.0, 1.5, 1.5], [1.0, 1.0, 2.0]],
        anchor_bottom_heights=[-1.0, -1.0],
        feature_map_stride=1,
        matched_thresholds=[0.6, 0.6],
        unmatched_thresholds=[0.45, 0.45],
    )


def _perfect_multi(loss_fn: MultiHeadAnchorLoss) -> Tuple[AnchorHeadMultiOutput, Dict[str, Any], int]:
    """One GT box equal to a class-0 rotation-0 anchor; per-head confident logits and zero box residuals."""
    count0, count1 = loss_fn.class_counts
    anchors0 = loss_fn.anchors[:count0]
    idx = int((anchors0[:, 6] == 0).nonzero(as_tuple=False)[10])
    gt_box = anchors0[idx : idx + 1].clone()

    cls0 = torch.full((1, count0, 1), -10.0)
    cls0[0, idx, 0] = 10.0
    cls1 = torch.full((1, count1, 1), -10.0)
    output: AnchorHeadMultiOutput = {
        "cls": [cls0, cls1],
        "box": [torch.zeros(1, count0, 10), torch.zeros(1, count1, 10)],
        "batch_box": torch.zeros(1, count0 + count1, 10),
        "multihead_label_mapping": [torch.tensor([1]), torch.tensor([2])],
    }
    batch: Dict[str, Any] = {
        DataKeys.BOX: gt_box,
        DataKeys.LABEL: torch.tensor([0]),
        DataKeys.BATCH_BOX: torch.tensor([0]),
    }
    return output, batch, idx


def test_multihead_anchor_loss_perfect_predictions_near_zero() -> None:
    loss_fn = _multi_loss()
    output, batch, _ = _perfect_multi(loss_fn)
    out = loss_fn(output, batch)
    assert out["box_loss"] == 0.0
    assert out["cls_loss"] < 1e-2
    assert out["dir_loss"] == 0.0
    assert out["loss"] < 1e-2


def test_multihead_anchor_loss_perturbed_box_value() -> None:
    """A 0.2 residual error on the positive anchor costs `loc_weight * |0.2|` exactly (plain L1)."""
    loss_fn = _multi_loss()
    output, batch, idx = _perfect_multi(loss_fn)
    output["box"][0][0, idx, 0] = 0.2
    out = loss_fn(output, batch)
    assert torch.isclose(out["box_loss"], torch.tensor(0.25 * 0.2), atol=1e-6)


def test_multihead_anchor_loss_velocity_codes_unsupervised_by_default() -> None:
    loss_fn = _multi_loss()
    output, batch, idx = _perfect_multi(loss_fn)
    output["box"][0][0, idx, 8:10] = 5.0  # velocity codes carry zero default weight
    out = loss_fn(output, batch)
    assert out["box_loss"] == 0.0


def test_multihead_anchor_loss_rejects_plain_angle_encoding() -> None:
    with pytest.raises(ValueError, match="sincos"):
        MultiHeadAnchorLoss(
            2,
            class_groups=[[0], [1]],
            voxel_size=_VOXEL_SIZE,
            point_cloud_range=_POINT_CLOUD_RANGE,
            anchor_sizes=[[3.0, 1.5, 1.5], [1.0, 1.0, 2.0]],
            anchor_bottom_heights=[-1.0, -1.0],
            feature_map_stride=1,
            matched_thresholds=[0.6, 0.6],
            unmatched_thresholds=[0.45, 0.45],
            encode_angle_by_sincos=False,
        )
