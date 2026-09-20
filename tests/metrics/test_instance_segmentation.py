import math

import pytest
import torch
from torch import Tensor

from torch_pointcloud.metrics import instance_average_precision, instance_matches


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
    per_class = instance_average_precision(matches, average="none", class_names=("chair", "table"), min_points=1)
    assert per_class == pytest.approx({"chair": 1.0, "table": 1.0})
    assert instance_average_precision(matches, num_classes=2, min_points=1) == pytest.approx(1.0)
    assert instance_average_precision(matches, iou_threshold=0.5, num_classes=2, min_points=1) == pytest.approx(1.0)
    assert instance_average_precision(matches, iou_threshold=0.25, num_classes=2, min_points=1) == pytest.approx(1.0)


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
    per_class = instance_average_precision([match], average="none", num_classes=2, min_points=1)
    assert per_class.tolist() == pytest.approx([19.0 / 24.0, 2.0 / 3.0])
    assert instance_average_precision([match], num_classes=2, min_points=1) == pytest.approx(35.0 / 48.0)
    assert instance_average_precision([match], iou_threshold=0.5, min_points=1) == pytest.approx(43.0 / 48.0)
    assert instance_average_precision([match], iou_threshold=0.25, min_points=1) == pytest.approx(43.0 / 48.0)


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
    assert instance_average_precision([match], num_classes=1, min_points=1) == pytest.approx(1.0)

    gt_valid = torch.tensor([0] * 20)
    match = instance_matches(masks, labels, scores, gt_instance, gt_valid)
    assert instance_average_precision([match], num_classes=1, min_points=1) == pytest.approx(0.25)
    assert instance_average_precision([match], iou_threshold=0.25, min_points=1) == pytest.approx(0.25)


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
    per_class = instance_average_precision([match], average="none", num_classes=2, min_points=5)
    assert math.isnan(per_class[0].item())
    assert per_class[1].item() == pytest.approx(1.0)
    assert instance_average_precision([match], num_classes=2, min_points=5) == pytest.approx(1.0)
    assert instance_average_precision([match], iou_threshold=0.5, num_classes=2, min_points=5) == pytest.approx(1.0)


def test_instance_average_precision_empty_edges() -> None:
    """No predictions with ground truth scores 0.0; no ground truth at all reports only zero mAPs."""
    gt_instance = torch.tensor([0] * 10)
    gt_label = torch.tensor([0] * 10)
    empty = instance_matches(
        torch.zeros(0, 10, dtype=torch.bool), torch.zeros(0, dtype=torch.long), torch.zeros(0), gt_instance, gt_label
    )
    assert instance_average_precision([empty], average="none", num_classes=1, min_points=1).tolist() == [0.0]
    assert instance_average_precision([empty], num_classes=1, min_points=1) == 0.0

    no_gt = instance_matches(
        torch.stack([_mask(10, [0, 1])]),
        torch.tensor([0]),
        torch.tensor([0.9]),
        torch.full((10,), -1),
        torch.full((10,), -1),
    )
    assert math.isnan(instance_average_precision([no_gt], average="none", num_classes=1, min_points=1)[0].item())
    assert instance_average_precision([no_gt], num_classes=1, min_points=1) == 0.0
    assert instance_average_precision([no_gt], iou_threshold=0.25, num_classes=1, min_points=1) == 0.0
    with pytest.raises(ValueError, match="class_names"):
        instance_average_precision([no_gt], num_classes=1, class_names=["chair", "table"])
