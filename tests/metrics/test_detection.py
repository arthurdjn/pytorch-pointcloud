import math

import pytest
import torch

from torch_pointcloud.metrics import box_average_precision, box_matches
from torch_pointcloud.utils.types import Boxes3D, Detection3D


def _box(x: float, y: float, yaw: float = 0.0) -> list[float]:
    return [x, y, 0.0, 4.0, 2.0, 1.5, yaw]


def test_box_average_precision_per_class_iou() -> None:
    """A prediction at 3D IoU 0.6 with the GT passes class IoU 0.5 but fails 0.7."""
    gt: Boxes3D = {
        "boxes": torch.tensor([[0.0, 0, 0, 4, 2, 1.5, 0]]),
        "labels": torch.tensor([0]),
        "batch": torch.tensor([0]),
    }
    # shifted 1.0 along x -> axis-aligned IoU = 9 / 15 = 0.6
    pred: Detection3D = {
        "boxes": torch.tensor([[1.0, 0, 0, 4, 2, 1.5, 0]]),
        "scores": torch.tensor([0.9]),
        "labels": torch.tensor([0]),
        "batch": torch.tensor([0]),
    }
    assert box_average_precision([box_matches(pred, gt)], iou_threshold=0.5, average="none")[0].item() == pytest.approx(
        1.0
    )
    assert box_average_precision([box_matches(pred, gt)], iou_threshold=0.7, average="none")[0].item() == pytest.approx(
        0.0
    )


def test_box_average_precision_ignore_mask() -> None:
    """A prediction overlapping an ignore region attributed to its class is dropped, not a false positive."""
    boxes = torch.tensor([[0.0, 0, 0, 4, 2, 1.5, 0], [50, 0, 0, 4, 2, 1.5, 0]])
    batch = torch.tensor([0, 0])
    pred: Detection3D = {
        "boxes": torch.tensor([[50.0, 0, 0, 4, 2, 1.5, 0], [0, 0, 0, 4, 2, 1.5, 0]]),
        "scores": torch.tensor([0.9, 0.5]),
        "labels": torch.tensor([0, 0]),
        "batch": torch.tensor([0, 0]),
    }
    no_ignore: Boxes3D = {"boxes": boxes, "labels": torch.tensor([0, -1]), "batch": batch}
    with_ignore: Boxes3D = {
        "boxes": boxes,
        "labels": torch.tensor([0, 0]),
        "batch": batch,
        "ignore_mask": torch.tensor([False, True]),
    }
    penalized = box_average_precision([box_matches(pred, no_ignore)], iou_threshold=0.5, average="none")
    excused = box_average_precision([box_matches(pred, with_ignore)], iou_threshold=0.5, average="none")
    assert penalized[0].item() == pytest.approx(0.5)
    assert excused[0].item() == pytest.approx(1.0)


def test_box_average_precision_ignore_attribution_per_class() -> None:
    """An ignored Van (attributed to Car) excuses a Car prediction but not a Pedestrian prediction."""
    gt: Boxes3D = {
        # Pedestrian GT at x=0, ignored Van at x=50 attributed to Car, Car GT at x=-50.
        "boxes": torch.tensor([[0.0, 0, 0, 4, 2, 1.5, 0], [50, 0, 0, 4, 2, 1.5, 0], [-50, 0, 0, 4, 2, 1.5, 0]]),
        "labels": torch.tensor([1, 0, 0]),
        "batch": torch.tensor([0, 0, 0]),
        "ignore_mask": torch.tensor([False, True, False]),
    }
    pred: Detection3D = {
        # Pedestrian and Car predictions exactly on the ignored Van, plus one true positive per class.
        "boxes": torch.tensor(
            [[50.0, 0, 0, 4, 2, 1.5, 0], [0, 0, 0, 4, 2, 1.5, 0], [50, 0, 0, 4, 2, 1.5, 0], [-50, 0, 0, 4, 2, 1.5, 0]]
        ),
        "scores": torch.tensor([0.9, 0.5, 0.9, 0.8]),
        "labels": torch.tensor([1, 1, 0, 0]),
        "batch": torch.tensor([0, 0, 0, 0]),
    }
    out = box_average_precision([box_matches(pred, gt)], iou_threshold=0.5, average="none")
    # The Pedestrian prediction on the Van is a false positive (the Van only excuses Car predictions).
    assert out[1].item() == pytest.approx(0.5)
    # The Car prediction on the Van stays excused.
    assert out[0].item() == pytest.approx(1.0)


def test_box_average_precision_prediction_ignore_mask() -> None:
    """A prediction flagged by the prediction-side ignore mask is neither a false positive nor a match."""
    gt: Boxes3D = {
        "boxes": torch.tensor([[0.0, 0, 0, 4, 2, 1.5, 0]]),
        "labels": torch.tensor([0]),
        "batch": torch.tensor([0]),
    }
    pred: Detection3D = {
        "boxes": torch.tensor([[50.0, 0, 0, 4, 2, 1.5, 0], [0, 0, 0, 4, 2, 1.5, 0]]),
        "scores": torch.tensor([0.9, 0.5]),
        "labels": torch.tensor([0, 0]),
        "batch": torch.tensor([0, 0]),
    }
    assert box_average_precision([box_matches(pred, gt)], iou_threshold=0.5, average="none")[0].item() == pytest.approx(
        0.5
    )
    flagged: Detection3D = {**pred, "ignore_mask": torch.tensor([True, False])}
    assert box_average_precision([box_matches(flagged, gt)], iou_threshold=0.5, average="none")[
        0
    ].item() == pytest.approx(1.0)
    # A flagged prediction on the GT cannot consume it: the unflagged lower-score prediction still matches.
    on_gt: Detection3D = {
        "boxes": torch.tensor([[0.0, 0, 0, 4, 2, 1.5, 0], [0, 0, 0, 4, 2, 1.5, 0]]),
        "scores": torch.tensor([0.9, 0.5]),
        "labels": torch.tensor([0, 0]),
        "batch": torch.tensor([0, 0]),
        "ignore_mask": torch.tensor([True, False]),
    }
    assert box_average_precision([box_matches(on_gt, gt)], iou_threshold=0.5, average="none")[
        0
    ].item() == pytest.approx(1.0)


def test_box_average_precision_interpolation_modes() -> None:
    """A hand-computed curve (TP 0.9, FP 0.8, TP 0.7 over 2 GT) where all / r11 / r40 disagree.

    The cumulative curve is recall [0.5, 0.5, 1.0], precision [1.0, 0.5, 2/3]. The all-points integral is
    0.5 * 1.0 + 0.5 * 2/3 = 5/6. The two recall-crossing thresholds fill grid slots 0 and 1 with
    right-max precisions 1.0 and 2/3, so r11 (slots 0, 4, ..., 40) averages 1.0 / 11 and r40
    (slots 1..40) averages (2/3) / 40.
    """
    gt: Boxes3D = {
        "boxes": torch.tensor([[0.0, 0, 0, 4, 2, 1.5, 0], [50, 0, 0, 4, 2, 1.5, 0]]),
        "labels": torch.tensor([0, 0]),
        "batch": torch.tensor([0, 0]),
    }
    pred: Detection3D = {
        "boxes": torch.tensor([[0.0, 0, 0, 4, 2, 1.5, 0], [100, 0, 0, 4, 2, 1.5, 0], [50, 0, 0, 4, 2, 1.5, 0]]),
        "scores": torch.tensor([0.9, 0.8, 0.7]),
        "labels": torch.tensor([0, 0, 0]),
        "batch": torch.tensor([0, 0, 0]),
    }
    ap_all = box_average_precision([box_matches(pred, gt)], iou_threshold=0.5, average="none")[0].item()
    ap_r11 = box_average_precision([box_matches(pred, gt)], iou_threshold=0.5, average="none", interpolation="r11")[
        0
    ].item()
    ap_r40 = box_average_precision([box_matches(pred, gt)], iou_threshold=0.5, average="none", interpolation="r40")[
        0
    ].item()
    assert ap_all == pytest.approx(5.0 / 6.0)
    assert ap_r11 == pytest.approx(1.0 / 11.0)
    assert ap_r40 == pytest.approx((2.0 / 3.0) / 40.0)
    mean_ap = box_average_precision([box_matches(pred, gt)], iou_threshold=0.5, interpolation="r11")
    assert mean_ap == pytest.approx(1.0 / 11.0)


def test_box_average_precision_perfect_rotated_match_is_one() -> None:
    """Predictions identical to the GT at a non-zero heading score mAP 1.0 at every threshold."""
    gt: Boxes3D = {
        "boxes": torch.tensor([[0.0, 0, 0, 4, 2, 1.5, 0.7], [3.0, 3, 0, 2, 2, 1.0, 0.7]]),
        "labels": torch.tensor([0, 1]),
        "batch": torch.tensor([0, 0]),
    }
    pred: Detection3D = {
        "boxes": gt["boxes"].clone(),
        "scores": torch.tensor([0.9, 0.8]),
        "labels": gt["labels"].clone(),
        "batch": gt["batch"].clone(),
    }
    matches = [box_matches(pred, gt)]
    assert box_average_precision(matches, iou_threshold=0.25) == pytest.approx(1.0)
    assert box_average_precision(matches, iou_threshold=0.5) == pytest.approx(1.0)


def test_box_average_precision_no_predictions_is_zero() -> None:
    """A batch with GT but zero predicted boxes scores 0.0 at every threshold, without NaN."""
    gt: Boxes3D = {
        "boxes": torch.tensor([[0.0, 0, 0, 4, 2, 1.5, 0]]),
        "labels": torch.tensor([0]),
        "batch": torch.tensor([0]),
    }
    pred: Detection3D = {
        "boxes": torch.empty(0, 7),
        "scores": torch.empty(0),
        "labels": torch.empty(0, dtype=torch.long),
        "batch": torch.empty(0, dtype=torch.long),
    }
    matches = [box_matches(pred, gt)]
    assert box_average_precision(matches, iou_threshold=0.25) == 0.0
    assert box_average_precision(matches, iou_threshold=0.5) == 0.0


def test_box_average_precision_empty_targets_is_zero() -> None:
    """With no GT boxes there is no class to average over, so every threshold reports 0.0."""
    gt: Boxes3D = {
        "boxes": torch.empty(0, 7),
        "labels": torch.empty(0, dtype=torch.long),
        "batch": torch.empty(0, dtype=torch.long),
    }
    pred: Detection3D = {
        "boxes": torch.tensor([[0.0, 0, 0, 4, 2, 1.5, 0]]),
        "scores": torch.tensor([0.9]),
        "labels": torch.tensor([0]),
        "batch": torch.tensor([0]),
    }
    matches = [box_matches(pred, gt)]
    assert box_average_precision(matches, iou_threshold=0.25) == 0.0
    assert box_average_precision(matches, iou_threshold=0.5) == 0.0


def test_box_average_precision_scene_without_predictions_counts_misses() -> None:
    """One of two scenes has zero predicted boxes: its GT stays unmatched and halves the recall."""
    gt: Boxes3D = {
        "boxes": torch.tensor([[0.0, 0, 0, 4, 2, 1.5, 0], [0.0, 0, 0, 4, 2, 1.5, 0]]),
        "labels": torch.tensor([0, 0]),
        "batch": torch.tensor([0, 1]),
    }
    pred: Detection3D = {
        "boxes": torch.tensor([[0.0, 0, 0, 4, 2, 1.5, 0]]),
        "scores": torch.tensor([0.9]),
        "labels": torch.tensor([0]),
        "batch": torch.tensor([1]),
    }
    matches = [box_matches(pred, gt)]
    assert box_average_precision(matches, iou_threshold=0.25) == 0.5
    assert box_average_precision(matches, iou_threshold=0.5) == 0.5


def test_box_average_precision_class_without_gt_boxes_is_zero() -> None:
    """Predictions for a class with no GT boxes are all false positives: AP 0.0, not NaN."""
    gt: Boxes3D = {
        "boxes": torch.tensor([[0.0, 0, 0, 4, 2, 1.5, 0]]),
        "labels": torch.tensor([0]),
        "batch": torch.tensor([0]),
    }
    pred: Detection3D = {
        "boxes": torch.tensor([[0.0, 0, 0, 4, 2, 1.5, 0], [10.0, 0, 0, 4, 2, 1.5, 0]]),
        "scores": torch.tensor([0.9, 0.8]),
        "labels": torch.tensor([0, 7]),
        "batch": torch.tensor([0, 0]),
    }
    matches = [box_matches(pred, gt)]
    out = box_average_precision(matches, iou_threshold={0: 0.5, 7: 0.5}, average="none")
    assert out[0].item() == pytest.approx(1.0)
    assert out[7].item() == pytest.approx(0.0)
    assert box_average_precision(matches, iou_threshold={0: 0.5, 7: 0.5}) == pytest.approx(0.5)


def test_box_matches_record() -> None:
    """Each kept prediction records its best same-class box; ignored predictions and regions stay out of the ground truth."""
    pred: Detection3D = {
        "boxes": torch.tensor(
            [
                [0.0, 0, 0, 4, 2, 1.5, 0],
                [20.0, 0, 0, 4, 2, 1.5, 0],
                [0.0, 0, 0, 4, 2, 1.5, 0],
                [40.0, 0, 0, 4, 2, 1.5, 0],
                [0.0, 0, 0, 4, 2, 1.5, 0],
            ]
        ),
        "scores": torch.tensor([0.9, 0.8, 0.7, 0.6, 0.5]),
        "labels": torch.tensor([0, 0, 1, 0, 0]),
        "batch": torch.tensor([0, 0, 0, 1, 0]),
        "ignore_mask": torch.tensor([False, False, False, False, True]),
    }
    gt: Boxes3D = {
        "boxes": torch.tensor([[20.0, 0, 0, 4, 2, 1.5, 0], [0.0, 0, 0, 4, 2, 1.5, 0], [40.0, 0, 0, 4, 2, 1.5, 0]]),
        "labels": torch.tensor([0, 0, 0]),
        "batch": torch.tensor([0, 0, 0]),
        "ignore_mask": torch.tensor([True, False, False]),
    }
    match = box_matches(pred, gt)

    assert match["pred_iou"].device.type == "cpu"
    assert match["gt_labels"].device.type == "cpu"
    assert match["gt_labels"].tolist() == [0, 0]
    # sample 0 first, then sample 1; the flagged prediction is gone
    assert match["pred_scores"].tolist() == pytest.approx([0.9, 0.8, 0.7, 0.6])
    assert match["pred_labels"].tolist() == [0, 0, 1, 0]
    # class 1 has no box, and sample 1 holds no box at all: both carry the -1 sentinel
    assert match["pred_iou"].tolist() == pytest.approx([1.0, 0.0, -1.0, -1.0])
    assert match["pred_gt"].tolist() == [0, 0, -1, -1]
    assert match["pred_ignore_iou"].tolist() == pytest.approx([0.0, 1.0, -1.0, -1.0])


def test_box_matches_empty_batch() -> None:
    pred: Detection3D = {
        "boxes": torch.empty(0, 7),
        "scores": torch.empty(0),
        "labels": torch.empty(0, dtype=torch.long),
        "batch": torch.empty(0, dtype=torch.long),
    }
    gt: Boxes3D = {"boxes": torch.tensor([_box(0, 0)]), "labels": torch.tensor([2]), "batch": torch.tensor([0])}
    match = box_matches(pred, gt)

    assert match["gt_labels"].tolist() == [2]
    assert match["pred_scores"].numel() == 0
    assert match["pred_iou"].numel() == 0
    assert match["pred_gt"].numel() == 0


def test_box_average_precision_is_independent_of_batching() -> None:
    """Records gathered batch by batch score exactly like the record of the same samples packed as one batch."""
    generator = torch.Generator().manual_seed(0)
    scale = torch.tensor([6.0, 6.0, 1.0, 3.0, 3.0, 1.5, 3.14])
    preds: list[Detection3D] = []
    targets: list[Boxes3D] = []
    for _ in range(3):
        preds.append(
            {
                "boxes": torch.rand(24, 7, generator=generator) * scale + 0.5,
                "scores": torch.rand(24, generator=generator),
                "labels": torch.randint(0, 3, (24,), generator=generator),
                "batch": torch.randint(0, 2, (24,), generator=generator),
            }
        )
        targets.append(
            {
                "boxes": torch.rand(8, 7, generator=generator) * scale + 0.5,
                "labels": torch.randint(0, 3, (8,), generator=generator),
                "batch": torch.randint(0, 2, (8,), generator=generator),
                "ignore_mask": torch.rand(8, generator=generator) < 0.25,
            }
        )
    matches = [box_matches(pred, target) for pred, target in zip(preds, targets)]
    packed_pred: Detection3D = {
        "boxes": torch.cat([pred["boxes"] for pred in preds]),
        "scores": torch.cat([pred["scores"] for pred in preds]),
        "labels": torch.cat([pred["labels"] for pred in preds]),
        "batch": torch.cat([pred["batch"] + 2 * index for index, pred in enumerate(preds)]),
    }
    packed_target: Boxes3D = {
        "boxes": torch.cat([target["boxes"] for target in targets]),
        "labels": torch.cat([target["labels"] for target in targets]),
        "batch": torch.cat([target["batch"] + 2 * index for index, target in enumerate(targets)]),
        "ignore_mask": torch.cat([target["ignore_mask"] for target in targets]),
    }
    packed = [box_matches(packed_pred, packed_target)]

    for iou_threshold in (0.25, {0: 0.25, 1: 0.5, 2: 0.25}):
        per_class = box_average_precision(matches, iou_threshold=iou_threshold, average="none")
        assert torch.equal(per_class, box_average_precision(packed, iou_threshold=iou_threshold, average="none"))
        assert box_average_precision(matches, iou_threshold=iou_threshold) == pytest.approx(per_class.nanmean().item())
    assert box_average_precision(matches, iou_threshold=0.25) > 0.0


def test_box_average_precision_without_records() -> None:
    assert box_average_precision([]) == 0.0
    assert box_average_precision([], average="none").numel() == 0
    assert box_average_precision([], iou_threshold={0: 0.5}, average="none").tolist() == [0.0]


def test_box_average_precision_class_names() -> None:
    """`class_names` names the per-class output and fixes its length; a class without ground truth is NaN."""
    pred: Detection3D = {
        "boxes": torch.tensor([_box(0, 0), _box(20, 0)]),
        "scores": torch.tensor([0.9, 0.8]),
        "labels": torch.tensor([0, 1]),
        "batch": torch.tensor([0, 0]),
    }
    gt: Boxes3D = {"boxes": torch.tensor([_box(0, 0)]), "labels": torch.tensor([0]), "batch": torch.tensor([0])}
    matches = [box_matches(pred, gt)]

    out = box_average_precision(matches, average="none", class_names=["Car", "Pedestrian", "Cyclist"])
    assert list(out) == ["Car", "Pedestrian", "Cyclist"]
    assert out["Car"] == 1.0
    assert math.isnan(out["Pedestrian"]) and math.isnan(out["Cyclist"])
    assert box_average_precision(matches, average="none", num_classes=4).shape == (4,)
    with pytest.raises(ValueError, match="class_names"):
        box_average_precision(matches, average="none", num_classes=2, class_names=["Car", "Pedestrian", "Cyclist"])
