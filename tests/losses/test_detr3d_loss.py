import math
from typing import Any, Dict

import torch
from torch import Tensor

from torch_pointcloud.losses import DETR3DLoss
from torch_pointcloud.transforms.functional import angle_to_class
from torch_pointcloud.utils.data import DataKeys

_NUM_CLASSES = 3
_SCENE = 8.0


def _perfect_layer(
    centers: Tensor, sizes: Tensor, angles: Tensor, labels: Tensor, num_queries: int, num_angle_bin: int
) -> Dict[str, Tensor]:
    """One decoder layer whose first `len(centers)` queries predict the given boxes exactly."""
    k = centers.shape[0]
    center = torch.full((1, num_queries, 3), 7.5)
    size = torch.full((1, num_queries, 3), 0.2)
    angle = torch.zeros(1, num_queries)
    cls_logits = torch.full((1, num_queries, _NUM_CLASSES + 1), -10.0)
    cls_logits[0, :, -1] = 10.0  # background by default
    angle_logits = torch.full((1, num_queries, num_angle_bin), -10.0)
    angle_logits[..., 0] = 10.0
    angle_residual_normalized = torch.zeros(1, num_queries, num_angle_bin)

    angle_class, angle_residual = angle_to_class(angles % (2 * math.pi), num_angle_bin)
    for i in range(k):
        center[0, i] = centers[i]
        size[0, i] = sizes[i]
        angle[0, i] = angles[i]
        cls_logits[0, i] = -10.0
        cls_logits[0, i, labels[i]] = 10.0
        angle_logits[0, i] = -10.0
        angle_logits[0, i, angle_class[i]] = 10.0
        angle_residual_normalized[0, i, angle_class[i]] = angle_residual[i] / (math.pi / num_angle_bin)

    cls_prob = cls_logits.softmax(dim=-1)
    return {
        "sem_cls_logits": cls_logits,
        "sem_cls_prob": cls_prob[..., :-1],
        "objectness_prob": 1 - cls_prob[..., -1],
        "center_normalized": center / _SCENE,
        "center_unnormalized": center,
        "size_normalized": size / _SCENE,
        "size_unnormalized": size,
        "angle_logits": angle_logits,
        "angle_residual_normalized": angle_residual_normalized,
        "angle_residual": angle_residual_normalized * (math.pi / num_angle_bin),
        "angle_continuous": angle,
    }


def _batch(centers: Tensor, sizes: Tensor, headings: Tensor, labels: Tensor) -> Dict[str, Any]:
    boxes = torch.cat([centers, sizes, headings.unsqueeze(-1)], dim=-1)
    return {
        DataKeys.BOX: boxes,
        DataKeys.LABEL: labels,
        DataKeys.BATCH_BOX: torch.zeros(centers.shape[0], dtype=torch.long),
    }


def _output(layer: Dict[str, Tensor]) -> Dict[str, Any]:
    dims = (torch.zeros(1, 3), torch.full((1, 3), _SCENE))
    return {"aux_outputs": [layer], "point_cloud_dims": dims}


_CENTERS = torch.tensor([[2.0, 2.0, 1.0], [5.0, 6.0, 2.0]])
_SIZES = torch.tensor([[1.0, 1.5, 2.0], [2.0, 1.0, 1.0]])
_LABELS = torch.tensor([0, 2])


def test_detr3d_loss_giou_weight_defaults_to_zero() -> None:
    """The reference recipe trains with the GIoU term disabled; the GIoU still drives the matcher cost."""
    loss_fn = DETR3DLoss(num_classes=_NUM_CLASSES, num_angle_bin=1)
    assert loss_fn.loss_giou_weight == 0.0
    assert loss_fn.matcher_giou_cost == 2.0


def test_detr3d_loss_perfect_axis_aligned_predictions_near_zero() -> None:
    angles = torch.zeros(2)
    loss_fn = DETR3DLoss(num_classes=_NUM_CLASSES, num_angle_bin=1, loss_giou_weight=1.0)
    layer = _perfect_layer(_CENTERS, _SIZES, angles, _LABELS, num_queries=4, num_angle_bin=1)
    out = loss_fn(_output(layer), _batch(_CENTERS, _SIZES, -angles, _LABELS))
    for key in ("loss_center", "loss_size", "loss_giou", "loss_angle_cls", "loss_angle_reg"):
        assert out[key] < 1e-4, key
    assert out["loss_sem_cls"] < 1e-3
    assert out["loss"] < 1e-2
    assert out["loss_cardinality"] == 0.0


def test_detr3d_loss_perturbed_predictions_are_larger() -> None:
    angles = torch.zeros(2)
    loss_fn = DETR3DLoss(num_classes=_NUM_CLASSES, num_angle_bin=1, loss_giou_weight=1.0)
    layer = _perfect_layer(_CENTERS, _SIZES, angles, _LABELS, num_queries=4, num_angle_bin=1)
    batch = _batch(_CENTERS, _SIZES, -angles, _LABELS)
    perfect = loss_fn(_output(layer), batch)

    layer["center_unnormalized"][0, 0] += 0.2
    layer["center_normalized"] = layer["center_unnormalized"] / _SCENE
    out = loss_fn(_output(layer), batch)
    assert out["loss_center"] > perfect["loss_center"]
    assert out["loss_giou"] > perfect["loss_giou"]
    assert out["loss"] > perfect["loss"]


def test_detr3d_loss_ccw_gt_matches_native_heading_predictions() -> None:
    """GT headings arrive counter-clockwise; the loss must supervise the negated (native) heading bins."""
    native = torch.tensor([0.4, -1.2])
    loss_fn = DETR3DLoss(num_classes=_NUM_CLASSES, num_angle_bin=12)
    layer = _perfect_layer(_CENTERS, _SIZES, native, _LABELS, num_queries=4, num_angle_bin=12)

    ccw = loss_fn(_output(layer), _batch(_CENTERS, _SIZES, -native, _LABELS))
    wrong = loss_fn(_output(layer), _batch(_CENTERS, _SIZES, native, _LABELS))
    assert ccw["loss_angle_cls"] < 1e-4
    assert ccw["loss_angle_reg"] < 1e-4
    assert wrong["loss_angle_cls"] > 0.1
    assert wrong["loss"] > ccw["loss"]


def test_detr3d_loss_densified_targets_follow_box_contract() -> None:
    """Densify negates the CCW heading, keeps full extents, and reads classes from `DataKeys.LABEL`."""
    headings = torch.tensor([0.4, -1.2])
    loss_fn = DETR3DLoss(num_classes=_NUM_CLASSES, num_angle_bin=12)
    dims = (torch.zeros(1, 3), torch.full((1, 3), _SCENE))
    targets = loss_fn._densify(_batch(_CENTERS, _SIZES, headings, _LABELS), dims)
    assert torch.equal(targets.center_unnormalized[0], _CENTERS)
    assert torch.equal(targets.size_unnormalized[0], _SIZES)
    assert torch.equal(targets.angle[0], -headings)
    assert torch.equal(targets.label[0], _LABELS)
    assert torch.equal(targets.present[0], torch.ones(2))


def test_detr3d_loss_degenerate_gt_box_stays_finite() -> None:
    """A zero-size GT box under a collapsed query must not produce NaN costs (the matcher raises on them)."""
    torch.manual_seed(0)
    loss_fn = DETR3DLoss(num_classes=_NUM_CLASSES, num_angle_bin=12)
    centers = torch.tensor([[4.0, 4.0, 1.0], [2.0, 2.0, 1.0]])
    sizes = torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    layer = _perfect_layer(centers, sizes, torch.zeros(2), _LABELS, num_queries=4, num_angle_bin=12)
    out = loss_fn(_output(layer), _batch(centers, sizes, torch.zeros(2), _LABELS))
    assert torch.isfinite(out["loss"])


def test_detr3d_loss_no_boxes_is_finite() -> None:
    loss_fn = DETR3DLoss(num_classes=_NUM_CLASSES, num_angle_bin=1)
    layer = _perfect_layer(_CENTERS, _SIZES, torch.zeros(2), _LABELS, num_queries=4, num_angle_bin=1)
    empty = _batch(_CENTERS[:0], _SIZES[:0], torch.zeros(0), _LABELS[:0])
    out = loss_fn(_output(layer), empty)
    assert torch.isfinite(out["loss"])


def test_detr3d_loss_backward() -> None:
    loss_fn = DETR3DLoss(num_classes=_NUM_CLASSES, num_angle_bin=12)
    layer = _perfect_layer(_CENTERS, _SIZES, torch.tensor([0.4, -1.2]), _LABELS, num_queries=4, num_angle_bin=12)
    for key in ("sem_cls_logits", "center_normalized", "size_normalized", "angle_residual_normalized"):
        layer[key].requires_grad_(True)
    out = loss_fn(_output(layer), _batch(_CENTERS, _SIZES, torch.tensor([-0.4, 1.2]), _LABELS))
    out["loss"].backward()
    assert layer["sem_cls_logits"].grad is not None
    grad = layer["center_normalized"].grad
    assert grad is not None and torch.isfinite(grad).all()
