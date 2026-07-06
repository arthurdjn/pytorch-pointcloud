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
    """A high-score prediction overlapping an ignore region is dropped, not a false positive."""
    boxes = torch.tensor([[0.0, 0, 0, 4, 2, 1.5, 0], [50, 0, 0, 4, 2, 1.5, 0]])
    labels, batch = torch.tensor([0, -1]), torch.tensor([0, 0])
    pred: Detection3D = {
        "boxes": torch.tensor([[50.0, 0, 0, 4, 2, 1.5, 0], [0, 0, 0, 4, 2, 1.5, 0]]),
        "scores": torch.tensor([0.9, 0.5]),
        "labels": torch.tensor([0, 0]),
        "batch": torch.tensor([0, 0]),
    }
    no_ignore: Boxes3D = {"boxes": boxes, "labels": labels, "batch": batch}
    with_ignore: Boxes3D = {
        "boxes": boxes,
        "labels": labels,
        "batch": batch,
        "ignore_mask": torch.tensor([False, True]),
    }
    assert average_precision3d([pred], [no_ignore], iou_per_class={0: 0.5})["AP/0"] == pytest.approx(0.5)
    assert average_precision3d([pred], [with_ignore], iou_per_class={0: 0.5})["AP/0"] == pytest.approx(1.0)
