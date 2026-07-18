from typing import Any, Dict

import pytest
import torch

from torch_pointcloud.datasets.shapenetpart import ShapeNetPart
from torch_pointcloud.lightning.metrics import (
    AveragePrecision3D,
    InstancePartMeanIoU,
    MeanAveragePrecision3D,
)
from torch_pointcloud.utils.imports import _LIGHTNING_AVAILABLE
from torch_pointcloud.utils.metrics import part_mean_iou
from torch_pointcloud.utils.types import Boxes3D, Detection3D

pytestmark = pytest.mark.skipif(not _LIGHTNING_AVAILABLE, reason="lightning is not installed")


def test_mean_average_precision3d_perfect_match_scores_one() -> None:
    """A prediction identical to the ground truth yields `mAP@t = 1.0` at every threshold."""
    metric = MeanAveragePrecision3D(iou_thresholds=(0.25, 0.5))
    preds: Detection3D = {
        "boxes": torch.tensor([[0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0]]),
        "scores": torch.tensor([0.9]),
        "labels": torch.tensor([0]),
        "batch": torch.tensor([0]),
    }
    target: Boxes3D = {
        "boxes": torch.tensor([[0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0]]),
        "labels": torch.tensor([0]),
        "batch": torch.tensor([0]),
    }
    metric.update(preds, target)
    out = metric.compute()
    assert out["mAP@0.25"] == pytest.approx(1.0)
    assert out["mAP@0.5"] == pytest.approx(1.0)


def test_mean_average_precision3d_reset_clears_state() -> None:
    """`reset` empties the accumulated per-batch lists."""
    metric = MeanAveragePrecision3D()
    pred: Detection3D = {
        "boxes": torch.empty(0, 7),
        "scores": torch.empty(0),
        "labels": torch.empty(0),
        "batch": torch.empty(0),
    }
    target: Boxes3D = {"boxes": torch.empty(0, 7), "labels": torch.empty(0), "batch": torch.empty(0)}
    metric.update(pred, target)
    assert len(metric.preds) == 1
    metric.reset()
    assert metric.preds == []


def test_average_precision3d_matches_each_class_at_its_own_iou() -> None:
    """Both predictions overlap their GT at IoU 2/3: below car's 0.7 (AP 0), above pedestrian's 0.5 (AP 1)."""
    metric = AveragePrecision3D(iou_per_class={0: 0.7, 1: 0.5}, class_names=["car", "pedestrian"])
    preds: Detection3D = {
        "boxes": torch.tensor([[0.2, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0], [5.2, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0]]),
        "scores": torch.tensor([0.9, 0.8]),
        "labels": torch.tensor([0, 1]),
        "batch": torch.tensor([0, 0]),
    }
    target: Boxes3D = {
        "boxes": torch.tensor([[0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0], [5.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0]]),
        "labels": torch.tensor([0, 1]),
        "batch": torch.tensor([0, 0]),
    }
    metric.update(preds, target)
    out = metric.compute()
    assert out["AP/car"] == pytest.approx(0.0)
    assert out["AP/pedestrian"] == pytest.approx(1.0)
    assert out["mAP"] == pytest.approx(0.5)


def test_instance_part_mean_iou_matches_functional_across_updates() -> None:
    """Updating per packed multi-shape batch accumulates to the functional's result on the whole split."""
    # Airplane (parts 0-3), Table (47-49), Airplane, Knife (22-23), as two packed batches.
    targets = [torch.tensor([0, 1, 2, 47, 47, 48]), torch.tensor([2, 3, 22, 23])]
    preds = [torch.tensor([0, 1, 3, 47, 48, 48]), torch.tensor([2, 2, 22, 23])]
    batches = [torch.tensor([0, 0, 0, 1, 1, 1]), torch.tensor([0, 0, 1, 1])]

    metric = InstancePartMeanIoU()
    for p, t, b in zip(preds, targets, batches):
        metric.update(p, t, b)
    out = metric.compute()

    part_ids = list(ShapeNetPart.seg_ids.values())
    category = torch.tensor([0, 15, 0, 7])
    batch = torch.cat([batches[0], batches[1] + 2])
    expected = part_mean_iou(torch.cat(preds), torch.cat(targets), part_ids, category, batch)
    assert out["ins_mIoU"].item() == pytest.approx(expected["ins_mIoU"], abs=1e-6)
    assert out["cls_mIoU"].item() == pytest.approx(expected["cls_mIoU"], abs=1e-6)


def test_instance_part_mean_iou_derives_category_from_target() -> None:
    """A shape whose target is a Table part is scored over Table's 3 parts: iou 0 + 0 + 1 (absent) -> 1/3.
    A wrongly derived category (e.g. Airplane's untouched parts) would score 1.0."""
    metric = InstancePartMeanIoU()
    metric.update(torch.tensor([48]), torch.tensor([47]), torch.tensor([0]))
    out = metric.compute()
    assert out["ins_mIoU"].item() == pytest.approx(1.0 / 3.0)
    assert out["cls_mIoU"].item() == pytest.approx(1.0 / 3.0)


def test_instance_part_mean_iou_accepts_logits() -> None:
    """A `(N, num_classes)` prediction is argmaxed, matching the per-point label input."""
    metric = InstancePartMeanIoU(part_ids=[[0, 1], [2, 3]])
    target = torch.tensor([0, 1, 2, 3])
    logits = torch.nn.functional.one_hot(torch.tensor([0, 1, 2, 2]), num_classes=4).float()
    metric.update(logits, target, torch.tensor([0, 0, 1, 1]))
    out = metric.compute()
    # Shape 0 perfect -> 1; shape 1: class 2 iou 1/2, class 3 iou 0 -> 1/4.
    assert out["ins_mIoU"].item() == pytest.approx(0.625)
    assert out["cls_mIoU"].item() == pytest.approx(0.625)


def test_average_precision3d_coerces_string_class_keys() -> None:
    """YAML / OmegaConf mappings may carry string keys; they are coerced to `int` class indices."""
    iou_per_class: Dict[Any, float] = {"0": 0.5}
    metric = AveragePrecision3D(iou_per_class=iou_per_class)
    assert metric.iou_per_class == {0: 0.5}
    preds: Detection3D = {
        "boxes": torch.tensor([[0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0]]),
        "scores": torch.tensor([0.9]),
        "labels": torch.tensor([0]),
        "batch": torch.tensor([0]),
    }
    target: Boxes3D = {
        "boxes": torch.tensor([[0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0]]),
        "labels": torch.tensor([0]),
        "batch": torch.tensor([0]),
    }
    metric.update(preds, target)
    out = metric.compute()
    assert out["AP/0"] == pytest.approx(1.0)
    assert out["mAP"] == pytest.approx(1.0)
