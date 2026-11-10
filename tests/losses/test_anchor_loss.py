from typing import Any, Dict, Optional, Tuple

import pytest
import torch
from torch import Tensor

from torch_pointcloud.layers.anchors import AnchorHead, AnchorHeadMultiOutput
from torch_pointcloud.losses import AnchorHeadLoss, MultiGroupAnchorHeadLoss
from torch_pointcloud.models import create_model
from torch_pointcloud.models.parta2 import PartA2Detection
from torch_pointcloud.models.pointpillars import PointPillarsDetection
from torch_pointcloud.models.pv_rcnn import PVRCNNDetection
from torch_pointcloud.models.second import SECONDDetection, SECONDIoUDetection
from torch_pointcloud.utils.data import DataKeys
from torch_pointcloud.utils.imports import _SPCONV_AVAILABLE

_POINT_CLOUD_RANGE = (0.0, -4.0, -2.0, 8.0, 4.0, 2.0)
_VOXEL_SIZE = (1.0, 1.0, 4.0)
SPCONV = pytest.mark.skipif(not _SPCONV_AVAILABLE, reason="spconv is not installed")


def _single_loss() -> AnchorHeadLoss:
    return AnchorHeadLoss(
        1,
        voxel_size=_VOXEL_SIZE,
        point_cloud_range=_POINT_CLOUD_RANGE,
        anchor_sizes=[[3.0, 1.5, 1.5]],
        anchor_bottom_heights=[-1.0],
        feature_map_stride=1,
        matched_thresholds=[0.6],
        dir_offset=0.78539,
        unmatched_thresholds=[0.45],
    )


def _perfect_single(loss_fn: AnchorHeadLoss) -> Tuple[Dict[str, Tensor], Dict[str, Any], int]:
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
    data: Dict[str, Any] = {
        DataKeys.BOX: gt_box,
        DataKeys.LABEL: torch.tensor([0]),
        DataKeys.BATCH_BOX: torch.tensor([0]),
    }
    return output, data, idx


@pytest.mark.parametrize(
    "velocity",
    [
        pytest.param(None, id="7-column"),
        pytest.param(torch.tensor([[1.0, -2.0]]), id="9-column"),
    ],
)
def test_anchor_loss_perfect_predictions_near_zero(velocity: Optional[Tensor]) -> None:
    """A (K, 9) box is matched on its geometry columns; the single-group loss does not regress the velocity."""
    loss_fn = _single_loss()
    output, data, _ = _perfect_single(loss_fn)
    if velocity is not None:
        data[DataKeys.BOX] = torch.cat([data[DataKeys.BOX], velocity], dim=1)
    out = loss_fn(output, data)
    assert out["box_loss"] == 0.0
    assert out["cls_loss"] < 1e-3
    assert out["dir_loss"] < 1e-3
    assert out["loss"] < 1e-2


def test_anchor_loss_perturbed_box_value() -> None:
    """A 0.2 residual error on the positive anchor costs `loc_weight * smooth_l1(0.2)` exactly."""
    loss_fn = _single_loss()
    output, data, idx = _perfect_single(loss_fn)
    perfect = loss_fn(output, data)
    output["box"][0, idx, 0] = 0.2
    out = loss_fn(output, data)
    beta = 1.0 / 9.0
    expected = 2.0 * (0.2 - beta / 2)
    assert torch.isclose(out["box_loss"], torch.tensor(expected), atol=1e-6)
    assert out["loss"] > perfect["loss"]


def test_anchor_loss_wrong_class_logits_are_penalized() -> None:
    loss_fn = _single_loss()
    output, data, idx = _perfect_single(loss_fn)
    perfect = loss_fn(output, data)
    output["cls"][0, idx, 0] = -10.0  # the positive anchor now denies the object
    out = loss_fn(output, data)
    assert out["cls_loss"] > perfect["cls_loss"] + 1.0


def test_anchor_loss_no_boxes_is_finite() -> None:
    loss_fn = _single_loss()
    output, data, _ = _perfect_single(loss_fn)
    data[DataKeys.BOX] = data[DataKeys.BOX][:0]
    data[DataKeys.LABEL] = data[DataKeys.LABEL][:0]
    data[DataKeys.BATCH_BOX] = data[DataKeys.BATCH_BOX][:0]
    out = loss_fn(output, data)
    assert torch.isfinite(out["loss"])
    assert out["box_loss"] == 0.0


def test_anchor_loss_forward_returns_dict() -> None:
    loss_fn = _single_loss()
    output, data, _ = _perfect_single(loss_fn)
    out = loss_fn(output, data)
    assert isinstance(out, dict) and out["loss"].ndim == 0


def test_anchor_loss_backward() -> None:
    loss_fn = _single_loss()
    output, data, _ = _perfect_single(loss_fn)
    for value in output.values():
        value.requires_grad_(True)
    loss_fn(output, data)["loss"].backward()
    assert output["cls"].grad is not None
    assert output["box"].grad is not None


def test_anchor_loss_validates_per_class_geometry() -> None:
    with pytest.raises(ValueError, match="anchor_sizes"):
        AnchorHeadLoss(
            2,
            voxel_size=_VOXEL_SIZE,
            point_cloud_range=_POINT_CLOUD_RANGE,
            anchor_sizes=[[3.0, 1.5, 1.5]],
            anchor_bottom_heights=[-1.0],
            feature_map_stride=1,
            matched_thresholds=[0.6, 0.6],
            dir_offset=0.78539,
            unmatched_thresholds=[0.45, 0.45],
        )


def test_anchor_loss_without_dir_offset_warns_and_keeps_the_former_default() -> None:
    with pytest.warns(FutureWarning, match="dir_offset"):
        loss_fn = AnchorHeadLoss(
            1,
            voxel_size=_VOXEL_SIZE,
            point_cloud_range=_POINT_CLOUD_RANGE,
            anchor_sizes=[[3.0, 1.5, 1.5]],
            anchor_bottom_heights=[-1.0],
            feature_map_stride=1,
            matched_thresholds=[0.6],
            unmatched_thresholds=[0.45],
        )
    assert loss_fn.dir_offset == 0.78539


@pytest.mark.parametrize(
    "name",
    [
        pytest.param("pointpillars.kitti.openpcdet", id="pointpillars.kitti.openpcdet"),
        pytest.param("pointpillars.kitti.mmdet3d", id="pointpillars.kitti.mmdet3d"),
        pytest.param("pointpillars.kitti-car.mmdet3d", id="pointpillars.kitti-car.mmdet3d"),
        pytest.param("second.kitti.openpcdet", marks=SPCONV, id="second.kitti.openpcdet"),
        pytest.param("second.kitti.mmdet3d", marks=SPCONV, id="second.kitti.mmdet3d"),
        pytest.param("second.kitti-car.mmdet3d", marks=SPCONV, id="second.kitti-car.mmdet3d"),
        pytest.param("second-dynamic-voxel.kitti.mmdet3d", marks=SPCONV, id="second-dynamic-voxel.kitti.mmdet3d"),
        pytest.param("second-iou.kitti.openpcdet", marks=SPCONV, id="second-iou.kitti.openpcdet"),
        pytest.param("part-a2.kitti.mmdet3d", marks=SPCONV, id="part-a2.kitti.mmdet3d"),
        pytest.param("pv-rcnn.kitti.mmdet3d", marks=SPCONV, id="pv-rcnn.kitti.mmdet3d"),
    ],
)
def test_anchor_loss_rebuilds_the_anchors_of_the_registered_head(name: str) -> None:
    model = create_model(name, task="detection")
    assert isinstance(
        model, (PointPillarsDetection, SECONDDetection, SECONDIoUDetection, PartA2Detection, PVRCNNDetection)
    )
    head = model.head
    assert isinstance(head, AnchorHead)
    loss_fn = AnchorHeadLoss(
        model.num_classes,
        point_cloud_range=model.point_cloud_range,
        voxel_size=model.voxel_size,
        anchor_sizes=model.anchor_sizes,
        anchor_bottom_heights=model.anchor_bottom_heights,
        feature_map_stride=model.feature_map_stride,
        matched_thresholds=[0.6] * model.num_classes,
        unmatched_thresholds=[0.45] * model.num_classes,
        anchor_rotations=model.anchor_rotations,
        # Only PointPillars moves its anchor grid; the other detectors span the point-cloud range.
        anchor_range=model.anchor_range if isinstance(model, PointPillarsDetection) else None,
        align_center=model.align_center if isinstance(model, PointPillarsDetection) else False,
        dir_offset=head.dir_offset,
    )
    assert torch.equal(loss_fn.anchors, head.anchors)


def _multi_loss(**kwargs: Any) -> MultiGroupAnchorHeadLoss:
    return MultiGroupAnchorHeadLoss(
        2,
        class_groups=[[0], [1]],
        voxel_size=_VOXEL_SIZE,
        point_cloud_range=_POINT_CLOUD_RANGE,
        anchor_sizes=[[3.0, 1.5, 1.5], [1.0, 1.0, 2.0]],
        anchor_bottom_heights=[-1.0, -1.0],
        feature_map_stride=1,
        matched_thresholds=[0.6, 0.6],
        unmatched_thresholds=[0.45, 0.45],
        **kwargs,
    )


def _perfect_multi(loss_fn: MultiGroupAnchorHeadLoss) -> Tuple[AnchorHeadMultiOutput, Dict[str, Any], int]:
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
    data: Dict[str, Any] = {
        DataKeys.BOX: gt_box,
        DataKeys.LABEL: torch.tensor([0]),
        DataKeys.BATCH_BOX: torch.tensor([0]),
    }
    return output, data, idx


def test_multihead_anchor_loss_perfect_predictions_near_zero() -> None:
    loss_fn = _multi_loss()
    output, data, _ = _perfect_multi(loss_fn)
    out = loss_fn(output, data)
    assert out["box_loss"] == 0.0
    assert out["cls_loss"] < 1e-2
    assert "dir_loss" not in out
    assert out["loss"] < 1e-2


def test_multihead_anchor_loss_perturbed_box_value() -> None:
    """A 0.2 residual error on the positive anchor costs `loc_weight * |0.2|` exactly (plain L1)."""
    loss_fn = _multi_loss()
    output, data, idx = _perfect_multi(loss_fn)
    output["box"][0][0, idx, 0] = 0.2
    out = loss_fn(output, data)
    assert torch.isclose(out["box_loss"], torch.tensor(0.25 * 0.2), atol=1e-6)


def test_multihead_anchor_loss_velocity_codes_unsupervised_by_default() -> None:
    loss_fn = _multi_loss()
    output, data, idx = _perfect_multi(loss_fn)
    output["box"][0][0, idx, 8:10] = 5.0  # velocity codes carry zero default weight
    out = loss_fn(output, data)
    assert out["box_loss"] == 0.0


def test_multihead_anchor_loss_velocity_targets_from_the_box_columns() -> None:
    """A (K, 9) box supervises the velocity codes: a zero prediction costs `loc_weight * weight * |v|` per code."""
    loss_fn = _multi_loss(code_weights=(1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2))
    output, data, idx = _perfect_multi(loss_fn)
    data[DataKeys.BOX] = torch.cat([data[DataKeys.BOX], torch.tensor([[1.0, -2.0]])], dim=1)
    out = loss_fn(output, data)
    assert torch.isclose(out["box_loss"], torch.tensor(0.25 * 0.2 * 3.0), atol=1e-6)

    output["box"][0][0, idx, 8:10] = torch.tensor([1.0, -2.0])
    assert loss_fn(output, data)["box_loss"] == 0.0


def test_multihead_anchor_loss_rejects_misordered_class_groups() -> None:
    """The anchors and head outputs are laid out class 0..C-1, so any other grouping would silently misalign."""
    with pytest.raises(ValueError, match="class_groups"):
        MultiGroupAnchorHeadLoss(
            2,
            class_groups=[[1], [0]],
            voxel_size=_VOXEL_SIZE,
            point_cloud_range=_POINT_CLOUD_RANGE,
            anchor_sizes=[[3.0, 1.5, 1.5], [1.0, 1.0, 2.0]],
            anchor_bottom_heights=[-1.0, -1.0],
            feature_map_stride=1,
            matched_thresholds=[0.6, 0.6],
            unmatched_thresholds=[0.45, 0.45],
        )


def test_multihead_anchor_loss_rejects_plain_angle_encoding() -> None:
    with pytest.raises(ValueError, match="sincos"):
        MultiGroupAnchorHeadLoss(
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
