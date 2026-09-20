import math

import pytest
import torch
from torch import Tensor

from torch_pointcloud.metrics import (
    confusion_matrix,
    intersection_over_union,
    part_intersection_over_union,
    part_mean_intersection_over_union,
)
from torch_pointcloud.metrics.segmentation import compute_intersection_union


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
    cm = confusion_matrix(preds, target, num_classes=3)
    assert torch.allclose(intersection_over_union(cm, average="none"), torch.ones(3))


def test_iou_absent_class_default_zero() -> None:
    # Class 2 is absent from both preds and target -> union is 0 -> safe_divide returns default.
    preds = torch.tensor([0, 1, 1])
    target = torch.tensor([0, 1, 1])
    cm = confusion_matrix(preds, target, num_classes=3)
    iou = intersection_over_union(cm, average="none")
    assert iou[0].item() == 1.0
    assert iou[1].item() == 1.0
    assert iou[2].item() == 0.0


def test_iou_absent_class_nan_default() -> None:
    preds = torch.tensor([0, 1, 1])
    target = torch.tensor([0, 1, 1])
    cm = confusion_matrix(preds, target, num_classes=3)
    iou = intersection_over_union(cm, average="none", zero_division=float("nan"))
    assert math.isnan(iou[2].item())


def test_iou_partial_overlap() -> None:
    # Class 0: tgt={0,1,2}, pred at those={0,0,1} -> inter=2, area_pred=2, area_tgt=3, union=3 -> 2/3.
    # Class 1: tgt={3,4}, pred at those={1,1} -> inter=2, area_pred=3, area_tgt=2, union=3 -> 2/3.
    preds = torch.tensor([0, 0, 1, 1, 1])
    target = torch.tensor([0, 0, 0, 1, 1])
    cm = confusion_matrix(preds, target, num_classes=2)
    assert torch.allclose(intersection_over_union(cm, average="none"), torch.tensor([2.0 / 3.0, 2.0 / 3.0]))


def test_iou_ignore_index_zeros_class() -> None:
    target = torch.tensor([0, 1, 1, 255, 255])
    preds = torch.tensor([0, 1, 1, 0, 1])
    # ignore_index outside [0, num_classes) just drops points, doesn't zero a class slot.
    cm = confusion_matrix(preds, target, num_classes=2, ignore_index=255)
    assert torch.allclose(intersection_over_union(cm, average="none"), torch.tensor([1.0, 1.0]))


def test_mean_iou_perfect(perfect_preds: tuple[Tensor, Tensor]) -> None:
    preds, target = perfect_preds
    cm = confusion_matrix(preds, target, num_classes=3)
    assert intersection_over_union(cm) == pytest.approx(1.0)


def test_mean_iou_ignore_index_excluded_from_mean() -> None:
    # Class 2 is absent (iou=0 with default=0.0). Without ignore_index, that pulls the mean down.
    preds = torch.tensor([0, 1, 1])
    target = torch.tensor([0, 1, 1])
    cm = confusion_matrix(preds, target, num_classes=3)
    assert intersection_over_union(cm) == pytest.approx(2.0 / 3.0)
    assert intersection_over_union(cm, ignore_index=2) == pytest.approx(1.0)


def test_part_iou_two_shapes_hand_checked() -> None:
    # Shape 0 (cat 0, parts [0, 1]): class 0 iou 1/2, class 1 iou 2/3 -> mean 7/12.
    # Shape 1 (cat 1, parts [2, 3]): class 2 iou 1, class 3 absent from preds and target -> 1 -> mean 1.
    part_ids = [[0, 1], [2, 3]]
    preds = torch.tensor([0, 1, 1, 1, 2, 2])
    target = torch.tensor([0, 0, 1, 1, 2, 2])
    category = torch.tensor([0, 1])
    batch = torch.tensor([0, 0, 0, 0, 1, 1])
    ious = part_intersection_over_union(preds, target, part_ids, category, batch)
    assert torch.allclose(ious, torch.tensor([7.0 / 12.0, 1.0]))


def test_part_iou_absent_parts_count_as_one() -> None:
    # Only part 0 of the 3-part category appears; the two absent parts each contribute IoU 1.
    part_ids = [[0, 1, 2]]
    preds = torch.tensor([0, 0])
    target = torch.tensor([0, 0])
    ious = part_intersection_over_union(preds, target, part_ids, torch.tensor([0]), torch.tensor([0, 0]))
    assert ious.item() == pytest.approx(1.0)


def test_part_iou_scores_only_the_category_parts() -> None:
    # A point predicted as another category's part (4) only costs intersection on the true part;
    # part 4 itself is outside the shape's category and is never scored.
    part_ids = [[0, 1], [2, 3], [4]]
    preds = torch.tensor([0, 4])
    target = torch.tensor([0, 0])
    ious = part_intersection_over_union(preds, target, part_ids, torch.tensor([0]), torch.tensor([0, 0]))
    # class 0 iou 1/2, class 1 absent -> 1 -> mean 3/4.
    assert ious.item() == pytest.approx(0.75)


def test_part_intersection_over_union_without_batch_is_one_shape() -> None:
    part_ids = [[0, 1], [2, 3]]
    preds = torch.tensor([0, 1, 1, 1])
    target = torch.tensor([0, 1, 0, 1])
    batch = torch.zeros(4, dtype=torch.long)

    ious = part_intersection_over_union(preds, target, part_ids, torch.tensor([0]))
    assert ious.shape == (1,)
    assert torch.equal(ious, part_intersection_over_union(preds, target, part_ids, torch.tensor([0]), batch))
    # part 0: 1 / 2, part 1: 2 / 3
    assert ious.item() == pytest.approx((1 / 2 + 2 / 3) / 2)
    assert torch.equal(ious, part_intersection_over_union(preds, target, part_ids, torch.tensor(0)))


def test_part_mean_iou_instance_vs_class_averaging() -> None:
    # Shape 0 (cat 0): 7/12. Shapes 1 and 2 (cat 1): 1 and 0. Category 2 has no shape.
    # ins = mean(7/12, 1, 0) = 19/36; cls = mean(7/12, (1 + 0) / 2) = 13/24 (absent category excluded).
    part_ids = [[0, 1], [2, 3], [4]]
    preds = torch.tensor([0, 1, 1, 1, 2, 2, 2])
    target = torch.tensor([0, 0, 1, 1, 2, 2, 3])
    category = torch.tensor([0, 1, 1])
    batch = torch.tensor([0, 0, 0, 0, 1, 1, 2])
    ious = part_intersection_over_union(preds, target, part_ids, category, batch)
    assert part_mean_intersection_over_union(ious, category) == pytest.approx(19.0 / 36.0)
    assert part_mean_intersection_over_union(ious, category, average="macro") == pytest.approx(13.0 / 24.0)

    per_class = part_mean_intersection_over_union(ious, category, average="none", num_classes=3)
    assert per_class[:2].tolist() == pytest.approx([7.0 / 12.0, 0.5])
    assert math.isnan(per_class[2].item())
    named = part_mean_intersection_over_union(ious, category, average="none", class_names=["Airplane", "Bag", "Cap"])
    assert list(named) == ["Airplane", "Bag", "Cap"]
    assert named["Bag"] == pytest.approx(0.5)
    with pytest.raises(ValueError, match="class_names"):
        part_mean_intersection_over_union(ious, category, average="none", num_classes=2, class_names=["Airplane"])


def test_mean_iou_excludes_ignore_index() -> None:
    cm = torch.tensor([[0, 0, 0], [5, 3, 1], [0, 2, 4]])

    assert intersection_over_union(cm, average="none", ignore_index=0).tolist() == pytest.approx([0.0, 3 / 11, 4 / 7])
    assert intersection_over_union(cm, ignore_index=0) == pytest.approx((3 / 11 + 4 / 7) / 2)
    assert intersection_over_union(cm) == pytest.approx((0.0 + 3 / 11 + 4 / 7) / 3)


def test_part_mean_iou_accumulates_batches() -> None:
    """A shape's IoU ignores the rest of its batch, so IoUs gathered batch by batch equal those of one big batch."""
    part_ids = [[0, 1], [2, 3]]
    first = (torch.tensor([0, 1, 1, 1]), torch.tensor([0, 1, 0, 1]), torch.tensor([0, 0]), torch.tensor([0, 0, 1, 1]))
    second = (torch.tensor([2, 2]), torch.tensor([2, 3]), torch.tensor([1]), torch.tensor([0, 0]))
    ious = torch.cat(
        [
            part_intersection_over_union(preds, target, part_ids, category, batch)
            for preds, target, category, batch in (first, second)
        ]
    )
    category = torch.cat([first[2], second[2]])

    whole = part_intersection_over_union(
        torch.cat([first[0], second[0]]),
        torch.cat([first[1], second[1]]),
        part_ids,
        category,
        torch.cat([first[3], second[3] + 2]),
    )
    assert torch.equal(ious, whole)
    instance_miou = part_mean_intersection_over_union(ious, category)
    assert part_mean_intersection_over_union(ious, category, average="macro") != instance_miou
