from typing import Any, Dict, List

import pytest
import torch

from torch_pointcloud.datasets.shapenetpart import ShapeNetPart
from torch_pointcloud.lightning.metrics import (
    AveragePrecision3D,
    InstanceAveragePrecision,
    InstancePartMeanIoU,
    MeanAveragePrecision3D,
    NuScenesDetection,
)
from torch_pointcloud.utils.imports import _LIGHTNING_AVAILABLE
from torch_pointcloud.utils.metrics import (
    average_precision3d,
    instance_average_precision,
    instance_matches,
    mean_average_precision3d,
    nuscenes_detection_metrics,
    nuscenes_velocity_attributes,
    part_mean_iou,
)
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


def test_instance_part_mean_iou_restrict_to_category_masks_foreign_parts() -> None:
    """With `restrict_to_category`, a logit row whose global argmax is another category's part is argmaxed
    over the shape's own parts instead: shape 1 (parts 2, 3) predicting part 0 falls back to its best own part."""
    target = torch.tensor([0, 1, 2, 3])
    logits = torch.tensor(
        [
            [5.0, 0.0, 0.0, 0.0],
            [0.0, 5.0, 0.0, 0.0],
            [9.0, 0.0, 4.0, 0.0],  # global argmax = part 0 (foreign), own best = part 2 (correct)
            [9.0, 0.0, 0.0, 4.0],
        ]
    )
    batch = torch.tensor([0, 0, 1, 1])
    plain = InstancePartMeanIoU(part_ids=[[0, 1], [2, 3]])
    plain.update(logits, target, batch)
    restricted = InstancePartMeanIoU(part_ids=[[0, 1], [2, 3]], restrict_to_category=True)
    restricted.update(logits, target, batch)
    # Plain: shape 1 predicts foreign parts only -> iou 0 + 0 -> (1 + 0) / 2; restricted: both shapes perfect.
    assert plain.compute()["ins_mIoU"].item() == pytest.approx(0.5)
    assert restricted.compute()["ins_mIoU"].item() == pytest.approx(1.0)


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


def test_mean_average_precision3d_r11_matches_functional_across_updates() -> None:
    """`interpolation` threads through verbatim: multi-update state equals the functional on the same batches."""
    preds: List[Detection3D] = [
        {
            "boxes": torch.tensor([[0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0]]),
            "scores": torch.tensor([0.9]),
            "labels": torch.tensor([0]),
            "batch": torch.tensor([0]),
        },
        {
            "boxes": torch.tensor([[20.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0]]),
            "scores": torch.tensor([0.8]),
            "labels": torch.tensor([0]),
            "batch": torch.tensor([0]),
        },
    ]
    targets: List[Boxes3D] = [
        {
            "boxes": torch.tensor([[0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0]]),
            "labels": torch.tensor([0]),
            "batch": torch.tensor([0]),
        },
        {
            "boxes": torch.tensor([[5.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0]]),
            "labels": torch.tensor([0]),
            "batch": torch.tensor([0]),
        },
    ]
    metric = MeanAveragePrecision3D(iou_thresholds=(0.5,), interpolation="r11")
    for pred, target in zip(preds, targets):
        metric.update(pred, target)
    out = metric.compute()
    assert out == mean_average_precision3d(preds, targets, iou_thresholds=(0.5,), interpolation="r11")
    assert out != mean_average_precision3d(preds, targets, iou_thresholds=(0.5,))


def test_mean_average_precision3d_update_ignore_mask_matches_functional() -> None:
    """A mask passed to `update` lands as the predictions' `ignore_mask` entry, excluding flagged boxes."""
    preds: Detection3D = {
        "boxes": torch.tensor([[5.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0]]),
        "scores": torch.tensor([0.9, 0.8]),
        "labels": torch.tensor([0, 0]),
        "batch": torch.tensor([0, 0]),
    }
    target: Boxes3D = {
        "boxes": torch.tensor([[0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0]]),
        "labels": torch.tensor([0]),
        "batch": torch.tensor([0]),
    }
    ignore_mask = torch.tensor([True, False])
    metric = MeanAveragePrecision3D(iou_thresholds=(0.5,))
    metric.update(preds, target, ignore_mask=ignore_mask)
    out = metric.compute()
    masked: Detection3D = {**preds, "ignore_mask": ignore_mask}
    assert out == mean_average_precision3d([masked], [target], iou_thresholds=(0.5,))
    assert out["mAP@0.5"] == pytest.approx(1.0)
    assert mean_average_precision3d([preds], [target], iou_thresholds=(0.5,))["mAP@0.5"] == pytest.approx(0.5)


def test_average_precision3d_r11_matches_functional() -> None:
    """`interpolation="r11"` reaches `average_precision3d` (the KITTI 11-point grid, not the all-points AP)."""
    preds: Detection3D = {
        "boxes": torch.tensor([[0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0], [20.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0]]),
        "scores": torch.tensor([0.9, 0.8]),
        "labels": torch.tensor([0, 0]),
        "batch": torch.tensor([0, 1]),
    }
    target: Boxes3D = {
        "boxes": torch.tensor([[0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0], [5.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0]]),
        "labels": torch.tensor([0, 0]),
        "batch": torch.tensor([0, 1]),
    }
    metric = AveragePrecision3D(iou_per_class={0: 0.5}, interpolation="r11")
    metric.update(preds, target)
    out = metric.compute()
    assert out == average_precision3d([preds], [target], iou_per_class={0: 0.5}, interpolation="r11")
    assert out != average_precision3d([preds], [target], iou_per_class={0: 0.5})


def test_average_precision3d_update_ignore_mask_excludes_predictions() -> None:
    """The flagged high-scoring stray box is excluded from scoring instead of counting as a false positive."""
    preds: Detection3D = {
        "boxes": torch.tensor([[5.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0]]),
        "scores": torch.tensor([0.9, 0.8]),
        "labels": torch.tensor([0, 0]),
        "batch": torch.tensor([0, 0]),
    }
    target: Boxes3D = {
        "boxes": torch.tensor([[0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0]]),
        "labels": torch.tensor([0]),
        "batch": torch.tensor([0]),
    }
    ignore_mask = torch.tensor([True, False])
    metric = AveragePrecision3D(iou_per_class={0: 0.5})
    metric.update(preds, target, ignore_mask=ignore_mask)
    out = metric.compute()
    masked: Detection3D = {**preds, "ignore_mask": ignore_mask}
    assert out == average_precision3d([masked], [target], iou_per_class={0: 0.5})
    assert out["AP/0"] == pytest.approx(1.0)
    assert average_precision3d([preds], [target], iou_per_class={0: 0.5})["AP/0"] == pytest.approx(0.5)


def test_nuscenes_detection_matches_functional_across_updates() -> None:
    """Two accumulated batches equal one functional call on the offset-concatenated packed tensors."""
    class_names = ("car", "pedestrian")
    preds: List[Detection3D] = [
        {
            "boxes": torch.tensor([[1.0, 2.0, 0.0, 4.0, 2.0, 1.5, 0.0], [5.0, 5.0, 0.0, 0.8, 0.8, 1.7, 0.0]]),
            "scores": torch.tensor([0.9, 0.8]),
            "labels": torch.tensor([0, 1]),
            "batch": torch.tensor([0, 1]),
            "velocity": torch.tensor([[1.0, 0.0], [0.1, 0.0]]),
        },
        {
            "boxes": torch.tensor([[10.0, 0.7, 0.0, 4.0, 2.0, 1.5, 0.3]]),
            "scores": torch.tensor([0.7]),
            "labels": torch.tensor([0]),
            "batch": torch.tensor([0]),
            "velocity": torch.tensor([[3.0, 0.5]]),
        },
    ]
    targets: List[Boxes3D] = [
        {
            "boxes": torch.tensor([[1.2, 2.0, 0.0, 4.2, 2.0, 1.5, 0.1], [5.0, 5.3, 0.0, 0.8, 0.8, 1.7, 0.0]]),
            "labels": torch.tensor([0, 1]),
            "batch": torch.tensor([0, 1]),
        },
        {
            "boxes": torch.tensor([[10.0, 0.0, 0.0, 4.0, 2.0, 1.5, 0.2], [30.0, 30.0, 0.0, 4.0, 2.0, 1.5, 0.0]]),
            "labels": torch.tensor([0, 0]),
            "batch": torch.tensor([0, 0]),
        },
    ]
    gt_velocities = [torch.tensor([[1.0, 0.1], [0.0, 0.0]]), torch.tensor([[2.5, 0.5], [0.0, 0.0]])]
    gt_num_points = [torch.tensor([12, 4]), torch.tensor([25, 0])]
    gt_attributes = [torch.tensor([3, 5]), torch.tensor([3, -1])]

    metric = NuScenesDetection(class_names=class_names, dist_thresholds=(0.5, 1.0), tp_threshold=1.0)
    for i in range(2):
        metric.update(
            preds[i],
            targets[i],
            velocity=gt_velocities[i],
            num_points=gt_num_points[i],
            attribute=gt_attributes[i],
        )
    out = metric.compute()

    offsets = [0, 2]
    pred_labels = torch.cat([p["labels"] for p in preds])
    pred_velocity = torch.cat([p["velocity"] for p in preds])
    expected = nuscenes_detection_metrics(
        torch.cat([torch.cat([p["boxes"], p["velocity"]], dim=1) for p in preds]),
        torch.cat([p["scores"] for p in preds]),
        pred_labels,
        torch.cat([p["batch"] + offset for p, offset in zip(preds, offsets)]),
        torch.cat([torch.cat([t["boxes"], v], dim=1) for t, v in zip(targets, gt_velocities)]),
        torch.cat([t["labels"] for t in targets]),
        torch.cat([t["batch"] + offset for t, offset in zip(targets, offsets)]),
        class_names=class_names,
        gt_num_points=torch.cat(gt_num_points),
        pred_attributes=nuscenes_velocity_attributes(pred_labels, pred_velocity, class_names=class_names),
        gt_attributes=torch.cat(gt_attributes),
        dist_thresholds=(0.5, 1.0),
        tp_threshold=1.0,
    )
    assert out == expected


def test_nuscenes_detection_reset_restarts_sample_offsets() -> None:
    """`reset` clears the accumulated batches and the sample counter, so scene indices restart at zero."""
    preds: Detection3D = {
        "boxes": torch.tensor([[1.0, 2.0, 0.0, 4.0, 2.0, 1.5, 0.0]]),
        "scores": torch.tensor([0.9]),
        "labels": torch.tensor([0]),
        "batch": torch.tensor([0]),
    }
    target: Boxes3D = {
        "boxes": torch.tensor([[1.0, 2.0, 0.0, 4.0, 2.0, 1.5, 0.0]]),
        "labels": torch.tensor([0]),
        "batch": torch.tensor([0]),
    }
    metric = NuScenesDetection(class_names=("car",))
    metric.update(preds, target)
    metric.reset()
    assert metric.pred_boxes == []
    metric.update(preds, target)
    assert metric.pred_batch[0].tolist() == [0]
    expected = nuscenes_detection_metrics(
        preds["boxes"],
        preds["scores"],
        preds["labels"],
        preds["batch"],
        target["boxes"],
        target["labels"],
        target["batch"],
        class_names=("car",),
    )
    assert metric.compute() == expected


def test_nuscenes_detection_derives_pred_attributes_from_velocity() -> None:
    """`compute` fills the prediction attributes from the accumulated pred velocities (the speed
    heuristic), matching the functional called with `nuscenes_velocity_attributes`; without them the
    attribute error would fall back to the full penalty."""
    preds: Detection3D = {
        "boxes": torch.tensor([[1.0, 2.0, 0.0, 4.0, 2.0, 1.5, 0.0]]),
        "scores": torch.tensor([0.9]),
        "labels": torch.tensor([0]),
        "batch": torch.tensor([0]),
        "velocity": torch.tensor([[3.0, 0.0]]),
    }
    target: Boxes3D = {
        "boxes": torch.tensor([[1.0, 2.0, 0.0, 4.0, 2.0, 1.5, 0.0]]),
        "labels": torch.tensor([0]),
        "batch": torch.tensor([0]),
    }
    gt_velocity = torch.tensor([[3.0, 0.0]])
    gt_attribute = torch.tensor([0])  # vehicle.moving
    metric = NuScenesDetection(class_names=("car",))
    metric.update(preds, target, velocity=gt_velocity, attribute=gt_attribute)
    out = metric.compute()

    args = (
        torch.cat([preds["boxes"], preds["velocity"]], dim=1),
        preds["scores"],
        preds["labels"],
        preds["batch"],
        torch.cat([target["boxes"], gt_velocity], dim=1),
        target["labels"],
        target["batch"],
    )
    expected = nuscenes_detection_metrics(
        *args,
        class_names=("car",),
        pred_attributes=nuscenes_velocity_attributes(preds["labels"], preds["velocity"], class_names=("car",)),
        gt_attributes=gt_attribute,
    )
    assert out == expected
    assert out["mAAE"] == pytest.approx(0.0)
    assert nuscenes_detection_metrics(*args, class_names=("car",), gt_attributes=gt_attribute)["mAAE"] == 1.0


def test_instance_average_precision_matches_functional_across_updates() -> None:
    """Per-scene records accumulated by `update` equal the functional on the same record list."""
    scene1 = instance_matches(
        torch.tensor([[True, True, True, False], [False, False, False, True]]),
        torch.tensor([0, 1]),
        torch.tensor([0.9, 0.8]),
        torch.tensor([0, 0, 0, 1]),
        torch.tensor([0, 0, 0, 1]),
    )
    scene2 = instance_matches(
        torch.tensor([[True, True, False]]),
        torch.tensor([0]),
        torch.tensor([0.7]),
        torch.tensor([0, 0, 1]),
        torch.tensor([0, 0, 1]),
    )
    metric = InstanceAveragePrecision(num_classes=2, class_names=["chair", "table"], min_points=1)
    metric.update(scene1)
    metric.update(scene2)
    out = metric.compute()
    expected = instance_average_precision([scene1, scene2], num_classes=2, class_names=["chair", "table"], min_points=1)
    assert out == expected
    assert "AP/chair" in out
    assert "AP/table" in out
