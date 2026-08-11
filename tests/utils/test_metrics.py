import math

import pytest
import torch
from torch import Tensor

from torch_pointcloud.utils.metrics import (
    average_precision3d,
    compute_intersection_union,
    compute_iou,
    compute_mean_iou,
    confusion_matrix,
    filter_boxes_by_range,
    instance_average_precision,
    instance_matches,
    mean_average_precision3d,
    nuscenes_detection_metrics,
    overall_accuracy,
    part_iou,
    part_mean_iou,
    per_class_accuracy,
)
from torch_pointcloud.utils.types import Boxes3D, Detection3D


@pytest.fixture
def perfect_preds() -> tuple[Tensor, Tensor]:
    target = torch.tensor([0, 0, 1, 1, 1, 2, 2])
    preds = target.clone()
    return preds, target


@pytest.fixture
def mixed_preds() -> tuple[Tensor, Tensor]:
    # 3 classes; target/pred designed so each (i, j) cell is hand-checked below.
    target = torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 2])
    preds = torch.tensor([0, 0, 1, 1, 2, 2, 2, 2, 0])
    return preds, target


def test_confusion_matrix_perfect(perfect_preds: tuple[Tensor, Tensor]) -> None:
    preds, target = perfect_preds
    cm = confusion_matrix(preds, target, num_classes=3)
    assert torch.equal(cm.diag(), torch.tensor([2, 3, 2]))
    assert cm.sum().item() == target.numel()
    # All off-diagonal is zero.
    assert (cm - torch.diag(cm.diag())).sum().item() == 0


def test_confusion_matrix_mixed(mixed_preds: tuple[Tensor, Tensor]) -> None:
    preds, target = mixed_preds
    cm = confusion_matrix(preds, target, num_classes=3)
    # Row i = true class i, column j = predicted class j.
    expected = torch.tensor(
        [
            [2, 1, 0],  # true 0: 2 -> 0, 1 -> 1
            [0, 1, 2],  # true 1: 1 -> 1, 2 -> 2
            [1, 0, 2],  # true 2: 1 -> 0, 2 -> 2
        ],
        dtype=torch.long,
    )
    assert torch.equal(cm, expected)


def test_confusion_matrix_ignore_index() -> None:
    target = torch.tensor([0, 1, 2, 255])
    preds = torch.tensor([0, 1, 2, 0])
    cm = confusion_matrix(preds, target, num_classes=3, ignore_index=255)
    assert torch.equal(cm, torch.eye(3, dtype=torch.long))


def test_confusion_matrix_shape_and_dtype() -> None:
    preds = torch.tensor([0, 1, 2])
    target = torch.tensor([0, 1, 2])
    cm = confusion_matrix(preds, target, num_classes=5)
    assert cm.shape == (5, 5)
    assert cm.dtype == torch.long


def test_intersection_union_perfect(perfect_preds: tuple[Tensor, Tensor]) -> None:
    preds, target = perfect_preds
    inter, union = compute_intersection_union(preds, target, num_classes=3)
    counts = torch.tensor([2.0, 3.0, 2.0])
    assert torch.equal(inter.float(), counts)
    assert torch.equal(union.float(), counts)


def test_intersection_union_mixed(mixed_preds: tuple[Tensor, Tensor]) -> None:
    preds, target = mixed_preds
    inter, union = compute_intersection_union(preds, target, num_classes=3)
    # Diagonal of the confusion matrix from test_confusion_matrix_mixed.
    assert torch.equal(inter, torch.tensor([2, 1, 2]))
    # union[c] = |pred==c| + |target==c| - inter[c]
    area_pred = torch.bincount(preds, minlength=3)
    area_target = torch.bincount(target, minlength=3)
    assert torch.equal(union, area_pred + area_target - inter)


def test_intersection_union_ignore_index_zeros_class() -> None:
    target = torch.tensor([0, 0, 1, 1, 2, 2])
    preds = torch.tensor([0, 0, 1, 1, 2, 1])
    inter, union = compute_intersection_union(preds, target, num_classes=3, ignore_index=2)
    assert inter[2].item() == 0
    assert union[2].item() == 0
    # The remaining classes are unaffected.
    assert inter[0].item() == 2 and union[0].item() == 2
    assert inter[1].item() == 2


def test_intersection_union_per_batch_shape() -> None:
    target = torch.tensor([0, 1, 2, 0, 1, 2])
    preds = torch.tensor([0, 1, 2, 0, 2, 2])
    batch = torch.tensor([0, 0, 0, 1, 1, 1])
    inter, union = compute_intersection_union(preds, target, num_classes=3, batch=batch)
    assert inter.shape == (2, 3)
    assert union.shape == (2, 3)
    # Sample 0 is perfect.
    assert torch.equal(inter[0], torch.tensor([1, 1, 1]))
    # Sample 1: target=[0,1,2], pred=[0,2,2] -> class 0 hit, class 1 missed, class 2 hit.
    assert torch.equal(inter[1], torch.tensor([1, 0, 1]))


def test_intersection_union_per_batch_trailing_ignored_sample_keeps_row() -> None:
    # Sample 1 is fully ignored: it must still get a row of zeros, like an empty or all-wrong sample.
    target = torch.tensor([0, 1, 255, 255])
    preds = torch.tensor([0, 1, 0, 0])
    batch = torch.tensor([0, 0, 1, 1])
    inter, union = compute_intersection_union(preds, target, num_classes=2, batch=batch, ignore_index=255)
    assert inter.shape == (2, 2)
    assert union.shape == (2, 2)
    assert torch.equal(inter[0], torch.tensor([1, 1]))
    assert torch.equal(union[0], torch.tensor([1, 1]))
    assert torch.equal(inter[1], torch.zeros(2, dtype=torch.long))
    assert torch.equal(union[1], torch.zeros(2, dtype=torch.long))


def test_intersection_union_per_batch_all_ignored_returns_zeros() -> None:
    target = torch.tensor([255, 255])
    preds = torch.tensor([0, 1])
    batch = torch.tensor([0, 0])
    inter, union = compute_intersection_union(preds, target, num_classes=2, batch=batch, ignore_index=255)
    assert inter.shape == (1, 2)
    assert union.shape == (1, 2)
    assert inter.sum().item() == 0
    assert union.sum().item() == 0


def test_iou_perfect_is_one(perfect_preds: tuple[Tensor, Tensor]) -> None:
    preds, target = perfect_preds
    iou = compute_iou(preds, target, num_classes=3)
    assert torch.allclose(iou, torch.ones(3))


def test_iou_absent_class_default_zero() -> None:
    # Class 2 is absent from both preds and target -> union is 0 -> safe_divide returns default.
    preds = torch.tensor([0, 1, 1])
    target = torch.tensor([0, 1, 1])
    iou = compute_iou(preds, target, num_classes=3)
    assert iou[0].item() == 1.0
    assert iou[1].item() == 1.0
    assert iou[2].item() == 0.0


def test_iou_absent_class_nan_default() -> None:
    preds = torch.tensor([0, 1, 1])
    target = torch.tensor([0, 1, 1])
    iou = compute_iou(preds, target, num_classes=3, default=float("nan"))
    assert math.isnan(iou[2].item())


def test_iou_partial_overlap() -> None:
    # Class 0: tgt={0,1,2}, pred at those={0,0,1} -> inter=2, area_pred=2, area_tgt=3, union=3 -> 2/3.
    # Class 1: tgt={3,4}, pred at those={1,1} -> inter=2, area_pred=3, area_tgt=2, union=3 -> 2/3.
    preds = torch.tensor([0, 0, 1, 1, 1])
    target = torch.tensor([0, 0, 0, 1, 1])
    iou = compute_iou(preds, target, num_classes=2)
    assert torch.allclose(iou, torch.tensor([2.0 / 3.0, 2.0 / 3.0]))


def test_iou_ignore_index_zeros_class() -> None:
    target = torch.tensor([0, 1, 1, 255, 255])
    preds = torch.tensor([0, 1, 1, 0, 1])
    # ignore_index outside [0, num_classes) just drops points, doesn't zero a class slot.
    iou = compute_iou(preds, target, num_classes=2, ignore_index=255)
    assert torch.allclose(iou, torch.tensor([1.0, 1.0]))


def test_mean_iou_perfect(perfect_preds: tuple[Tensor, Tensor]) -> None:
    preds, target = perfect_preds
    miou = compute_mean_iou(preds, target, num_classes=3)
    assert miou.item() == pytest.approx(1.0)


def test_mean_iou_ignore_index_excluded_from_mean() -> None:
    # Class 2 is absent (iou=0 with default=0.0). Without ignore_index, that pulls the mean down.
    preds = torch.tensor([0, 1, 1])
    target = torch.tensor([0, 1, 1])
    miou_all = compute_mean_iou(preds, target, num_classes=3)
    miou_ignored = compute_mean_iou(preds, target, num_classes=3, ignore_index=2)
    assert miou_all.item() == pytest.approx(2.0 / 3.0)
    assert miou_ignored.item() == pytest.approx(1.0)


def test_mean_iou_per_batch() -> None:
    target = torch.tensor([0, 1, 0, 1])
    preds = torch.tensor([0, 1, 0, 0])
    batch = torch.tensor([0, 0, 1, 1])
    miou = compute_mean_iou(preds, target, num_classes=2, batch=batch)
    # Sample 0 is perfect -> mean(1, 1) = 1.
    # Sample 1: class 0 iou=1/2 (pred extra), class 1 iou=0/1 -> mean(0.5, 0) = 0.25.
    assert miou.shape == (2,)
    assert miou[0].item() == pytest.approx(1.0)
    assert miou[1].item() == pytest.approx(0.25)


def test_mean_iou_per_batch_absent_classes_excluded_from_sample_mean() -> None:
    # Perfect predictions: sample 0 contains 2 of 20 classes, sample 1 contains 1 of 20.
    # Each sample averages only over its present classes, so both score 1.
    target = torch.tensor([0, 0, 1, 1, 2, 2])
    preds = target.clone()
    batch = torch.tensor([0, 0, 0, 0, 1, 1])
    miou = compute_mean_iou(preds, target, num_classes=20, batch=batch)
    assert miou.shape == (2,)
    assert miou[0].item() == pytest.approx(1.0)
    assert miou[1].item() == pytest.approx(1.0)


def test_mean_iou_per_batch_fully_ignored_sample_scores_zero() -> None:
    # Sample 1 has no present classes after masking, so its mIoU is defined as 0.
    target = torch.tensor([0, 1, 255, 255])
    preds = torch.tensor([0, 1, 0, 1])
    batch = torch.tensor([0, 0, 1, 1])
    miou = compute_mean_iou(preds, target, num_classes=2, batch=batch, ignore_index=255)
    assert miou.shape == (2,)
    assert miou[0].item() == pytest.approx(1.0)
    assert miou[1].item() == pytest.approx(0.0)


def test_part_iou_two_shapes_hand_checked() -> None:
    # Shape 0 (cat 0, parts [0, 1]): class 0 iou 1/2, class 1 iou 2/3 -> mean 7/12.
    # Shape 1 (cat 1, parts [2, 3]): class 2 iou 1, class 3 absent from preds and target -> 1 -> mean 1.
    part_ids = [[0, 1], [2, 3]]
    preds = torch.tensor([0, 1, 1, 1, 2, 2])
    target = torch.tensor([0, 0, 1, 1, 2, 2])
    category = torch.tensor([0, 1])
    batch = torch.tensor([0, 0, 0, 0, 1, 1])
    ious = part_iou(preds, target, part_ids, category, batch)
    assert torch.allclose(ious, torch.tensor([7.0 / 12.0, 1.0]))


def test_part_iou_absent_parts_count_as_one() -> None:
    # Only part 0 of the 3-part category appears; the two absent parts each contribute IoU 1.
    part_ids = [[0, 1, 2]]
    preds = torch.tensor([0, 0])
    target = torch.tensor([0, 0])
    ious = part_iou(preds, target, part_ids, torch.tensor([0]), torch.tensor([0, 0]))
    assert ious.item() == pytest.approx(1.0)


def test_part_iou_scores_only_the_category_parts() -> None:
    # A point predicted as another category's part (4) only costs intersection on the true part;
    # part 4 itself is outside the shape's category and is never scored.
    part_ids = [[0, 1], [2, 3], [4]]
    preds = torch.tensor([0, 4])
    target = torch.tensor([0, 0])
    ious = part_iou(preds, target, part_ids, torch.tensor([0]), torch.tensor([0, 0]))
    # class 0 iou 1/2, class 1 absent -> 1 -> mean 3/4.
    assert ious.item() == pytest.approx(0.75)


def test_part_mean_iou_instance_vs_class_averaging() -> None:
    # Shape 0 (cat 0): 7/12. Shapes 1 and 2 (cat 1): 1 and 0. Category 2 has no shape.
    # ins = mean(7/12, 1, 0) = 19/36; cls = mean(7/12, (1 + 0) / 2) = 13/24 (absent category excluded).
    part_ids = [[0, 1], [2, 3], [4]]
    preds = torch.tensor([0, 1, 1, 1, 2, 2, 2])
    target = torch.tensor([0, 0, 1, 1, 2, 2, 3])
    category = torch.tensor([0, 1, 1])
    batch = torch.tensor([0, 0, 0, 0, 1, 1, 2])
    out = part_mean_iou(preds, target, part_ids, category, batch)
    assert out["ins_mIoU"] == pytest.approx(19.0 / 36.0)
    assert out["cls_mIoU"] == pytest.approx(13.0 / 24.0)


def test_overall_accuracy_perfect(perfect_preds: tuple[Tensor, Tensor]) -> None:
    preds, target = perfect_preds
    assert overall_accuracy(preds, target) == pytest.approx(1.0)


def test_overall_accuracy_partial() -> None:
    preds = torch.tensor([0, 1, 1, 0])
    target = torch.tensor([0, 1, 0, 0])
    assert overall_accuracy(preds, target) == pytest.approx(0.75)


def test_overall_accuracy_ignore_index() -> None:
    preds = torch.tensor([0, 1, 0])
    target = torch.tensor([0, 1, 255])
    assert overall_accuracy(preds, target, ignore_index=255) == pytest.approx(1.0)


def test_overall_accuracy_fully_ignored_returns_zero() -> None:
    preds = torch.tensor([0, 1])
    target = torch.tensor([255, 255])
    assert overall_accuracy(preds, target, ignore_index=255) == 0.0


def test_per_class_accuracy_perfect(perfect_preds: tuple[Tensor, Tensor]) -> None:
    preds, target = perfect_preds
    acc = per_class_accuracy(preds, target, num_classes=3)
    assert torch.allclose(acc, torch.ones(3), atol=1e-6)


def test_per_class_accuracy_partial() -> None:
    # class 0: 2/3 correct, class 1: 2/2 correct.
    preds = torch.tensor([0, 0, 1, 1, 1])
    target = torch.tensor([0, 0, 0, 1, 1])
    acc = per_class_accuracy(preds, target, num_classes=2)
    assert torch.allclose(acc, torch.tensor([2.0 / 3.0, 1.0]), atol=1e-6)


def test_per_class_accuracy_ignore_index_zeros_class() -> None:
    preds = torch.tensor([0, 1, 1])
    target = torch.tensor([0, 1, 1])
    acc = per_class_accuracy(preds, target, num_classes=3, ignore_index=2)
    assert acc[2].item() == 0.0
    assert acc[0].item() == pytest.approx(1.0)
    assert acc[1].item() == pytest.approx(1.0)


def test_average_precision3d_per_class_iou() -> None:
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
    assert average_precision3d([pred], [gt], iou_per_class={0: 0.5})["AP/0"] == pytest.approx(1.0)
    assert average_precision3d([pred], [gt], iou_per_class={0: 0.7})["AP/0"] == pytest.approx(0.0)


def test_average_precision3d_ignore_mask() -> None:
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
    assert average_precision3d([pred], [no_ignore], iou_per_class={0: 0.5})["AP/0"] == pytest.approx(0.5)
    assert average_precision3d([pred], [with_ignore], iou_per_class={0: 0.5})["AP/0"] == pytest.approx(1.0)


def test_average_precision3d_ignore_attribution_per_class() -> None:
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
    out = average_precision3d([pred], [gt], iou_per_class={0: 0.5, 1: 0.5})
    # The Pedestrian prediction on the Van is a false positive (the Van only excuses Car predictions).
    assert out["AP/1"] == pytest.approx(0.5)
    # The Car prediction on the Van stays excused.
    assert out["AP/0"] == pytest.approx(1.0)


def test_average_precision3d_prediction_ignore_mask() -> None:
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
    assert average_precision3d([pred], [gt], iou_per_class={0: 0.5})["AP/0"] == pytest.approx(0.5)
    flagged: Detection3D = {**pred, "ignore_mask": torch.tensor([True, False])}
    assert average_precision3d([flagged], [gt], iou_per_class={0: 0.5})["AP/0"] == pytest.approx(1.0)
    # A flagged prediction on the GT cannot consume it: the unflagged lower-score prediction still matches.
    on_gt: Detection3D = {
        "boxes": torch.tensor([[0.0, 0, 0, 4, 2, 1.5, 0], [0, 0, 0, 4, 2, 1.5, 0]]),
        "scores": torch.tensor([0.9, 0.5]),
        "labels": torch.tensor([0, 0]),
        "batch": torch.tensor([0, 0]),
        "ignore_mask": torch.tensor([True, False]),
    }
    assert average_precision3d([on_gt], [gt], iou_per_class={0: 0.5})["AP/0"] == pytest.approx(1.0)


def test_average_precision3d_interpolation_modes() -> None:
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
    ap_all = average_precision3d([pred], [gt], iou_per_class={0: 0.5})["AP/0"]
    ap_r11 = average_precision3d([pred], [gt], iou_per_class={0: 0.5}, interpolation="r11")["AP/0"]
    ap_r40 = average_precision3d([pred], [gt], iou_per_class={0: 0.5}, interpolation="r40")["AP/0"]
    assert ap_all == pytest.approx(5.0 / 6.0)
    assert ap_r11 == pytest.approx(1.0 / 11.0)
    assert ap_r40 == pytest.approx((2.0 / 3.0) / 40.0)
    out = mean_average_precision3d([pred], [gt], iou_thresholds=(0.5,), interpolation="r11")
    assert out["mAP@0.5"] == pytest.approx(1.0 / 11.0)


def test_mean_average_precision3d_perfect_rotated_match_is_one() -> None:
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
    out = mean_average_precision3d([pred], [gt])
    assert out["mAP@0.25"] == pytest.approx(1.0)
    assert out["mAP@0.5"] == pytest.approx(1.0)


def test_mean_average_precision3d_no_predictions_is_zero() -> None:
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
    assert mean_average_precision3d([pred], [gt]) == {"mAP@0.25": 0.0, "mAP@0.5": 0.0}


def test_mean_average_precision3d_empty_targets_is_zero() -> None:
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
    assert mean_average_precision3d([pred], [gt]) == {"mAP@0.25": 0.0, "mAP@0.5": 0.0}


def test_mean_average_precision3d_scene_without_predictions_counts_misses() -> None:
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
    assert mean_average_precision3d([pred], [gt]) == {"mAP@0.25": 0.5, "mAP@0.5": 0.5}


def test_average_precision3d_class_without_gt_boxes_is_zero() -> None:
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
    out = average_precision3d([pred], [gt], iou_per_class={0: 0.5, 7: 0.5})
    assert out["AP/0"] == pytest.approx(1.0)
    assert out["AP/7"] == pytest.approx(0.0)
    assert out["mAP"] == pytest.approx(0.5)


def _box(x: float, y: float, yaw: float = 0.0) -> list[float]:
    return [x, y, 0.0, 4.0, 2.0, 1.5, yaw]


def test_nuscenes_detection_metrics_ap_distinct_per_threshold() -> None:
    r"""Offsets 0.7 / 1.5 / 3.0 m pass 1, 2 and 3 of the four matching thresholds (strict `<`).

    Car (3 GT): at $d{=}0.5$ nothing matches (AP 0). At $d{=}1$ the cumulative recall is $[1/3, 1/3, 1/3]$,
    so the interpolated precision is 1 up to grid index 33 and 0 after: AP $= 23 \cdot 0.9 / (90 \cdot 0.9)
    = 23/90$. At $d{=}2$ recall reaches $2/3$ (ones up to index 66, AP $56/90$); at $d{=}4$ all three match
    (AP 1). Pedestrian: one exact prediction, AP 1 at every threshold.
    """
    pred_boxes = torch.tensor([_box(0.7, 0.0), _box(10.0, 1.5), _box(23.0, 0.0), _box(0.0, 30.0)])
    pred_scores = torch.tensor([0.9, 0.8, 0.7, 0.95])
    pred_labels = torch.tensor([0, 0, 0, 1])
    gt_boxes = torch.tensor([_box(0.0, 0.0), _box(10.0, 0.0), _box(20.0, 0.0), _box(0.0, 30.0)])
    gt_labels = torch.tensor([0, 0, 0, 1])
    batch = torch.zeros(4, dtype=torch.long)
    args = (pred_boxes, pred_scores, pred_labels, batch, gt_boxes, gt_labels, batch)
    names = ("car", "pedestrian")
    for threshold, ap in {0.5: 0.0, 1.0: 23.0 / 90.0, 2.0: 56.0 / 90.0, 4.0: 1.0}.items():
        out = nuscenes_detection_metrics(*args, class_names=names, dist_thresholds=(threshold,))
        assert out["AP/car"] == pytest.approx(ap)
        assert out["AP/pedestrian"] == pytest.approx(1.0)
    out = nuscenes_detection_metrics(*args, class_names=names)
    assert out["AP/car"] == pytest.approx(169.0 / 360.0)
    assert out["mAP"] == pytest.approx(529.0 / 720.0)


def test_nuscenes_detection_metrics_101_point_interpolation_hand_derived() -> None:
    r"""AP from the 101-point clipping formula on a curve with an interpolated precision ramp.

    Matches in score order are TP, FP, TP, TP over 3 GT: recall $[1/3, 1/3, 2/3, 1]$, precision
    $[1, 1/2, 2/3, 3/4]$. Interpolated at $r_i = i/100$: 1 for $i \le 33$, $1/2 + (r_i - 1/3)/2$ for
    $34 \le i \le 66$ and $2/3 + (r_i - 2/3)/4$ for $67 \le i \le 100$. Dropping $i \le 10$, subtracting
    $0.1$ and clamping at 0 sums to $23 \cdot 0.9 + \sum_{34}^{66} (7/30 + i/200) + \sum_{67}^{100}
    (2/5 + i/400) = 20.7 + 15.95 + 20.6975$, so AP $= 57.3475 / (90 \cdot 0.9) = 22939/32400$.
    """
    pred_boxes = torch.tensor([_box(0.0, 0.0), _box(35.0, 0.0), _box(10.5, 0.0), _box(20.0, 1.0)])
    pred_scores = torch.tensor([0.9, 0.8, 0.7, 0.6])
    gt_boxes = torch.tensor([_box(0.0, 0.0), _box(10.0, 0.0), _box(20.0, 0.0)])
    out = nuscenes_detection_metrics(
        pred_boxes,
        pred_scores,
        torch.zeros(4, dtype=torch.long),
        torch.zeros(4, dtype=torch.long),
        gt_boxes,
        torch.zeros(3, dtype=torch.long),
        torch.zeros(3, dtype=torch.long),
        class_names=("car",),
        dist_thresholds=(2.0,),
    )
    assert out["AP/car"] == pytest.approx(22939.0 / 32400.0)


def test_nuscenes_detection_metrics_greedy_closest_consumes_gt() -> None:
    """The top-scoring prediction takes its closest GT even when a globally better assignment exists.

    The 0.9 prediction lies 1.2 m from GT A and 1.8 m from GT B, both under the 2 m threshold; the 0.8
    prediction is 0.5 m from A but 3.5 m from B. Pairing the first with B would match both predictions,
    but the official greedy rule matches it to the closest GT (A), leaving the second unmatched: one TP
    out of three GT (AP 23/90) with ATE 1.2 from the consumed closest match, not 1.8.
    """
    pred_boxes = torch.tensor([_box(1.2, 0.0), _box(-0.5, 0.0)])
    gt_boxes = torch.tensor([_box(0.0, 0.0), _box(3.0, 0.0), _box(40.0, 0.0)])
    out = nuscenes_detection_metrics(
        pred_boxes,
        torch.tensor([0.9, 0.8]),
        torch.zeros(2, dtype=torch.long),
        torch.zeros(2, dtype=torch.long),
        gt_boxes,
        torch.zeros(3, dtype=torch.long),
        torch.zeros(3, dtype=torch.long),
        class_names=("car",),
        dist_thresholds=(2.0,),
    )
    assert out["AP/car"] == pytest.approx(23.0 / 90.0)
    assert out["mATE"] == pytest.approx(1.2)


def test_nuscenes_detection_metrics_tp_errors_hand_values() -> None:
    """ATE is the BEV distance (z ignored), ASE the size-aligned 1 - IoU, AOE the absolute yaw difference.

    The prediction sits exactly 1.0 m from the GT center: it fails the 0.5 m and (strictly) the 1.0 m
    thresholds and matches at 2 m and 4 m, so AP/car = 0.5. On the 2 m match: ATE = 1.0; the size-aligned
    IoU of (4, 2, 1.5) vs (2, 2, 1.5) is 6/12, so ASE = 0.5; AOE = |0.3 - (-0.2)| = 0.5. Velocity and
    attributes are absent, so AVE and AAE take the full 1.0 penalty, and
    NDS = (5 * 0.5 + 0 + 0.5 + 0.5 + 0 + 0) / 10 = 0.35.
    """
    gt_boxes = torch.tensor([[0.0, 0.0, 0.0, 4.0, 2.0, 1.5, 0.3]])
    pred_boxes = torch.tensor([[1.0, 0.0, 0.5, 2.0, 2.0, 1.5, -0.2]])
    zero = torch.tensor([0])
    out = nuscenes_detection_metrics(
        pred_boxes, torch.tensor([0.9]), zero, zero, gt_boxes, zero, zero, class_names=("car",)
    )
    assert out["AP/car"] == pytest.approx(0.5)
    assert out["mATE"] == pytest.approx(1.0)
    assert out["mASE"] == pytest.approx(0.5)
    assert out["mAOE"] == pytest.approx(0.5)
    assert out["mAVE"] == 1.0
    assert out["mAAE"] == 1.0
    assert out["NDS"] == pytest.approx(0.35)


def test_nuscenes_detection_metrics_barrier_orientation_modulo_pi() -> None:
    """A barrier rotated by pi - 0.3 scores AOE 0.3 (period pi); any other class scores pi - 0.3.

    The same matched pair carries equal velocities and attributes: the car measures AVE and AAE of 0,
    while the barrier excludes both, and with no contributing class they report the full 1.0 penalty.
    """
    gt_boxes = torch.tensor([_box(0.0, 0.0, yaw=0.0) + [1.0, 1.0]])
    pred_boxes = torch.tensor([_box(0.0, 0.0, yaw=math.pi - 0.3) + [1.0, 1.0]])
    zero = torch.tensor([0])
    attributes = torch.tensor([2])
    for name, aoe in (("barrier", 0.3), ("car", math.pi - 0.3)):
        out = nuscenes_detection_metrics(
            pred_boxes,
            torch.tensor([0.9]),
            zero,
            zero,
            gt_boxes,
            zero,
            zero,
            class_names=(name,),
            pred_attributes=attributes,
            gt_attributes=attributes,
        )
        assert out["mAOE"] == pytest.approx(aoe, abs=1e-5)
        expected_penalty = 1.0 if name == "barrier" else 0.0
        assert out["mAVE"] == pytest.approx(expected_penalty)
        assert out["mAAE"] == pytest.approx(expected_penalty)


def test_filter_boxes_by_range_strict_bev_distance() -> None:
    """The class-range filter uses the BEV distance from the origin with a strict inequality."""
    boxes = torch.tensor([_box(30.0, 40.0), _box(3.0, 4.0), _box(0.0, -39.0), _box(0.0, 41.0)])
    labels = torch.tensor([0, 0, 1, 1])
    mask = filter_boxes_by_range(boxes, labels, ranges=[50.0, 40.0])
    assert torch.equal(mask, torch.tensor([False, True, True, False]))


def test_nuscenes_detection_metrics_range_and_num_points_filters() -> None:
    """Out-of-range boxes (both sides) and zero-point GT boxes are dropped before matching.

    After filtering, a single in-range TP remains against a single GT (AP 1). Without `gt_num_points`
    the zero-point GT stays: recall stops at 1/2, so the interpolated precision is 1 up to grid index 50
    and AP drops to 40/90 = 4/9.
    """
    pred_boxes = torch.tensor([_box(10.0, 0.0), _box(60.0, 0.0)])
    pred_scores = torch.tensor([0.9, 0.95])
    gt_boxes = torch.tensor([_box(10.0, 0.0), _box(60.0, 0.0), _box(20.0, 0.0)])
    args = (
        pred_boxes,
        pred_scores,
        torch.zeros(2, dtype=torch.long),
        torch.zeros(2, dtype=torch.long),
        gt_boxes,
        torch.zeros(3, dtype=torch.long),
        torch.zeros(3, dtype=torch.long),
    )
    out = nuscenes_detection_metrics(*args, class_names=("car",), gt_num_points=torch.tensor([5, 7, 0]))
    assert out["AP/car"] == pytest.approx(1.0)
    assert out["mATE"] == pytest.approx(0.0, abs=1e-7)
    out = nuscenes_detection_metrics(*args, class_names=("car",))
    assert out["AP/car"] == pytest.approx(4.0 / 9.0)


def test_nuscenes_detection_metrics_max_boxes_per_sample_cap() -> None:
    """The prediction cap keeps the highest-scoring boxes and applies per sample, not globally."""
    # Sample 0: an FP outscores the TP; a cap of 1 leaves only the FP, so nothing matches.
    pred_boxes = torch.tensor([_box(30.0, 0.0), _box(0.0, 0.0)])
    gt_boxes = torch.tensor([_box(0.0, 0.0)])
    zero = torch.tensor([0])
    out = nuscenes_detection_metrics(
        pred_boxes,
        torch.tensor([0.9, 0.8]),
        torch.zeros(2, dtype=torch.long),
        torch.zeros(2, dtype=torch.long),
        gt_boxes,
        zero,
        zero,
        class_names=("car",),
        max_boxes_per_sample=1,
    )
    assert out["AP/car"] == pytest.approx(0.0)
    # Two samples with one exact TP each survive a per-sample cap of 1 untouched.
    pred_boxes = torch.tensor([_box(0.0, 0.0), _box(5.0, 5.0)])
    gt_boxes = pred_boxes.clone()
    batch = torch.tensor([0, 1])
    out = nuscenes_detection_metrics(
        pred_boxes,
        torch.tensor([0.9, 0.8]),
        torch.zeros(2, dtype=torch.long),
        batch,
        gt_boxes,
        torch.zeros(2, dtype=torch.long),
        batch,
        class_names=("car",),
        max_boxes_per_sample=1,
    )
    assert out["AP/car"] == pytest.approx(1.0)


def test_nuscenes_detection_metrics_nds_identity() -> None:
    """NDS = (5 * mAP + sum of clipped TP scores) / 10; a perfect prediction reaches exactly 1.0.

    With an exact match carrying equal velocity and attribute, every error is 0 and NDS = (5 + 5) / 10.
    A 2 m/s velocity gap and a wrong attribute keep mAP = 1 but zero the clipped AVE and AAE scores:
    NDS = (5 + 3) / 10 = 0.8.
    """
    gt_boxes = torch.tensor([[0.0, 0.0, 0.0, 4.0, 2.0, 1.5, 0.5, 1.0, 0.0]])
    zero = torch.tensor([0])
    out = nuscenes_detection_metrics(
        gt_boxes.clone(),
        torch.tensor([0.9]),
        zero,
        zero,
        gt_boxes,
        zero,
        zero,
        class_names=("car",),
        pred_attributes=torch.tensor([1]),
        gt_attributes=torch.tensor([1]),
    )
    assert out["mAVE"] == pytest.approx(0.0)
    assert out["mAAE"] == pytest.approx(0.0)
    assert out["NDS"] == pytest.approx(1.0)
    pred_boxes = torch.tensor([[0.0, 0.0, 0.0, 4.0, 2.0, 1.5, 0.5, 1.0, 2.0]])
    out = nuscenes_detection_metrics(
        pred_boxes,
        torch.tensor([0.9]),
        zero,
        zero,
        gt_boxes,
        zero,
        zero,
        class_names=("car",),
        pred_attributes=torch.tensor([2]),
        gt_attributes=torch.tensor([1]),
    )
    assert out["mAVE"] == pytest.approx(2.0)
    assert out["mAAE"] == pytest.approx(1.0)
    assert out["NDS"] == pytest.approx(0.8)


def test_nuscenes_detection_metrics_void_attribute_and_missing_velocity_penalty() -> None:
    """A negative GT attribute id and 7-column boxes leave nothing measured: AVE and AAE fall back to 1."""
    gt_boxes = torch.tensor([_box(0.0, 0.0)])
    zero = torch.tensor([0])
    out = nuscenes_detection_metrics(
        gt_boxes.clone(),
        torch.tensor([0.9]),
        zero,
        zero,
        gt_boxes,
        zero,
        zero,
        class_names=("car",),
        pred_attributes=torch.tensor([3]),
        gt_attributes=torch.tensor([-1]),
    )
    assert out["mAVE"] == 1.0
    assert out["mAAE"] == 1.0
    assert out["NDS"] == pytest.approx(0.8)


def _mask(num_points: int, indices: list[int]) -> Tensor:
    out = torch.zeros(num_points, dtype=torch.bool)
    out[torch.tensor(indices)] = True
    return out


def test_instance_matches_record_hand_checked() -> None:
    """Counts, void intersections and same-class pairs of a small scene, checked by hand.

    Instance 2 carries the ignore label, so it is excluded from the ground truth and its points are
    void. The class-1 prediction overlapping the class-0 instance produces no pair (cross-class).
    """
    gt_instance = torch.tensor([0, 0, 0, 1, 1, -1, -1, 2])
    gt_label = torch.tensor([0, 0, 0, 1, 1, -1, -1, -1])
    masks = torch.stack([_mask(8, [0, 1, 5]), _mask(8, [3, 4, 7]), _mask(8, [0, 1])])
    match = instance_matches(masks, torch.tensor([0, 1, 1]), torch.tensor([0.9, 0.8, 0.7]), gt_instance, gt_label)
    assert match["gt_counts"].tolist() == [3, 2]
    assert match["gt_labels"].tolist() == [0, 1]
    assert match["pred_counts"].tolist() == [3, 3, 2]
    assert match["pred_void"].tolist() == [1, 1, 0]
    assert match["pair_pred"].tolist() == [0, 1]
    assert match["pair_gt"].tolist() == [0, 1]
    assert match["pair_inter"].tolist() == [2, 2]


def test_instance_average_precision_perfect_two_scenes() -> None:
    """Exact predictions over two scenes score 1.0 at every threshold, with named per-class keys."""
    matches = []
    for _ in range(2):
        gt_instance = torch.tensor([0, 0, 0, 1, 1, 1])
        gt_label = torch.tensor([0, 0, 0, 1, 1, 1])
        masks = torch.stack([_mask(6, [0, 1, 2]), _mask(6, [3, 4, 5])])
        matches.append(instance_matches(masks, torch.tensor([0, 1]), torch.tensor([0.9, 0.8]), gt_instance, gt_label))
    out = instance_average_precision(matches, num_classes=2, class_names=("chair", "table"), min_points=1)
    assert out["AP/chair"] == pytest.approx(1.0)
    assert out["AP/table"] == pytest.approx(1.0)
    assert out["mAP"] == pytest.approx(1.0)
    assert out["mAP@0.5"] == pytest.approx(1.0)
    assert out["mAP@0.25"] == pytest.approx(1.0)


def test_instance_average_precision_hand_scenario() -> None:
    r"""Two classes, one ignored GT instance and one duplicate prediction, fully hand-computed.

    Class 0 has two exactly-predicted instances (scores 0.9 and 0.3) plus a duplicate of the first
    (0.8): the duplicate becomes a false positive between the two true positives, so every threshold
    scores the centered-step integral $19/24$. Class 1 has one instance covered at IoU $0.8$ (score
    0.7): a true positive below threshold $0.8$ (AP $1$) and a false positive with a missed instance
    at $0.8$ and above (AP $0$), averaging $6/9$. The prediction covering the ignored instance (0.95)
    has ignore fraction $1 >$ every threshold and is never a false positive.
    """
    gt_instance = torch.tensor([0] * 10 + [1] * 10 + [2] * 10 + [3] * 10)
    gt_label = torch.tensor([0] * 10 + [0] * 10 + [1] * 10 + [-1] * 10)
    masks = torch.stack(
        [
            _mask(40, list(range(0, 10))),
            _mask(40, list(range(0, 10))),
            _mask(40, list(range(10, 20))),
            _mask(40, list(range(20, 28))),
            _mask(40, list(range(30, 40))),
        ]
    )
    labels = torch.tensor([0, 0, 0, 1, 1])
    scores = torch.tensor([0.9, 0.8, 0.3, 0.7, 0.95])
    match = instance_matches(masks, labels, scores, gt_instance, gt_label)
    out = instance_average_precision([match], num_classes=2, min_points=1)
    assert out["AP/0"] == pytest.approx(19.0 / 24.0)
    assert out["AP/1"] == pytest.approx(2.0 / 3.0)
    assert out["mAP"] == pytest.approx(35.0 / 48.0)
    assert out["mAP@0.5"] == pytest.approx(43.0 / 48.0)
    assert out["mAP@0.25"] == pytest.approx(43.0 / 48.0)


def test_instance_average_precision_void_overlap_excused() -> None:
    """An unmatched prediction on void points is excused; on valid non-instance points it is a false positive.

    With the second mask fully on void points the metric stays 1.0. Relabeling that region to a valid
    class (still without an instance) turns the same mask into a false positive above the true
    positive: precision at full recall is 0.5 and the centered-step integral drops to 0.25.
    """
    gt_instance = torch.tensor([0] * 10 + [-1] * 10)
    masks = torch.stack([_mask(20, list(range(0, 10))), _mask(20, list(range(10, 20)))])
    labels = torch.tensor([0, 0])
    scores = torch.tensor([0.9, 0.95])

    gt_void = torch.tensor([0] * 10 + [-1] * 10)
    match = instance_matches(masks, labels, scores, gt_instance, gt_void)
    out = instance_average_precision([match], num_classes=1, min_points=1)
    assert out["mAP"] == pytest.approx(1.0)

    gt_valid = torch.tensor([0] * 20)
    match = instance_matches(masks, labels, scores, gt_instance, gt_valid)
    out = instance_average_precision([match], num_classes=1, min_points=1)
    assert out["mAP"] == pytest.approx(0.25)
    assert out["mAP@0.25"] == pytest.approx(0.25)


def test_instance_average_precision_min_points_gates() -> None:
    """Instances and predictions under `min_points` are excluded on both sides.

    The 3-point class-0 instance is dropped, so class 0 has no ground truth and no `AP/0` key; the
    prediction overlapping it is excused as ignore, not a false positive. The 3-point class-1
    prediction (score 0.99, above the true positive) is dropped by the prediction gate, keeping
    class 1 at 1.0.
    """
    gt_instance = torch.tensor([0] * 3 + [1] * 10 + [-1] * 7)
    gt_label = torch.tensor([0] * 3 + [1] * 10 + [0] * 2 + [-1] * 5)
    masks = torch.stack([_mask(20, [0, 1, 2, 13, 14]), _mask(20, list(range(3, 13))), _mask(20, [3, 4, 5])])
    labels = torch.tensor([0, 1, 1])
    scores = torch.tensor([0.9, 0.8, 0.99])
    match = instance_matches(masks, labels, scores, gt_instance, gt_label)
    out = instance_average_precision([match], num_classes=2, min_points=5)
    assert "AP/0" not in out
    assert out["AP/1"] == pytest.approx(1.0)
    assert out["mAP"] == pytest.approx(1.0)
    assert out["mAP@0.5"] == pytest.approx(1.0)


def test_instance_average_precision_empty_edges() -> None:
    """No predictions with ground truth scores 0.0; no ground truth at all reports only zero mAPs."""
    gt_instance = torch.tensor([0] * 10)
    gt_label = torch.tensor([0] * 10)
    empty = instance_matches(
        torch.zeros(0, 10, dtype=torch.bool), torch.zeros(0, dtype=torch.long), torch.zeros(0), gt_instance, gt_label
    )
    out = instance_average_precision([empty], num_classes=1, min_points=1)
    assert out["AP/0"] == 0.0
    assert out["mAP"] == 0.0

    no_gt = instance_matches(
        torch.stack([_mask(10, [0, 1])]),
        torch.tensor([0]),
        torch.tensor([0.9]),
        torch.full((10,), -1),
        torch.full((10,), -1),
    )
    out = instance_average_precision([no_gt], num_classes=1, min_points=1)
    assert out == {"mAP": 0.0, "mAP@0.5": 0.0, "mAP@0.25": 0.0}
