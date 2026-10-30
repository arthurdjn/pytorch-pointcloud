import math
from typing import Any, Optional

import pytest
import torch
from torch import Tensor

from torch_pointcloud.metrics import (
    instance_average_precision,
    instance_coverage,
    instance_f1_score,
    instance_matches,
    instance_precision,
    instance_recall,
)
from torch_pointcloud.metrics.instance_segmentation import InstanceMatches
from torch_pointcloud.utils.types import InstanceSegmentation3D


def _mask(num_points: int, indices: list[int]) -> Tensor:
    out = torch.zeros(num_points, dtype=torch.bool)
    out[torch.tensor(indices)] = True
    return out


def _preds(masks: Tensor, labels: Tensor, scores: Tensor, batch: Optional[Tensor] = None) -> InstanceSegmentation3D:
    """The `decode` form of dense $(K, N)$ masks."""
    return {
        "mask_index": masks.nonzero().t(),
        "labels": labels,
        "scores": scores,
        "batch": torch.zeros_like(labels) if batch is None else batch,
    }


def _random_inputs(
    sizes: list[int],
    num_instances: int,
    num_preds: int,
    *,
    ignore_index: int = -1,
    tie_scores: bool = False,
    score_dtype: torch.dtype = torch.float32,
    shuffle: bool = False,
    with_batch: bool = True,
    with_ignore_mask: bool = False,
) -> dict[str, Any]:
    generator = torch.Generator().manual_seed(0)
    masks, gt_instance, gt_label = [], [], []
    for size in sizes:
        owner = torch.randint(-1, num_instances, (size,), generator=generator)
        classes = torch.randint(0, 3, (max(num_instances, 1),), generator=generator)
        classes[torch.rand(classes.shape, generator=generator) < 0.2] = ignore_index
        void_label = torch.tensor([ignore_index, 0, 1, 2])[torch.randint(0, 4, (size,), generator=generator)]
        gt_instance.append(torch.where(owner >= 0, owner * 1000 + 7, -1))
        gt_label.append(torch.where(owner >= 0, classes[owner.clamp(min=0)], void_label))
        picked = torch.randint(0, max(num_instances, 1), (num_preds,), generator=generator)
        noise = torch.rand(num_preds, size, generator=generator)
        masks.append(((owner[None] == picked[:, None]) & (noise > 0.2)) | (noise < 0.1))
    if tie_scores:
        scores = torch.tensor([0.25, 0.5])[torch.randint(0, 2, (num_preds * len(sizes),), generator=generator)]
    else:
        scores = torch.rand(num_preds * len(sizes), generator=generator)
    inputs = {
        "pred_masks": torch.block_diag(*(mask.int() for mask in masks)).bool(),
        "pred_labels": torch.randint(0, 3, (num_preds * len(sizes),), generator=generator),
        "pred_scores": scores.to(score_dtype),
        "gt_instance": torch.cat(gt_instance),
        "gt_label": torch.cat(gt_label),
        "pred_batch": torch.arange(len(sizes)).repeat_interleave(num_preds),
        "pred_ignore": torch.rand(num_preds * len(sizes), generator=generator) < (0.3 if with_ignore_mask else 0.0),
        "batch": torch.repeat_interleave(torch.arange(len(sizes)), torch.tensor(sizes)) if with_batch else None,
        "ignore_index": ignore_index,
    }
    if shuffle:
        perm = torch.randperm(sum(sizes), generator=generator)
        inputs["pred_masks"] = inputs["pred_masks"][:, perm]
        for key in ("gt_instance", "gt_label", "batch"):
            inputs[key] = inputs[key][perm]
    return inputs


def _brute_force_matches(
    pred_masks: Tensor,
    pred_labels: Tensor,
    pred_scores: Tensor,
    gt_instance: Tensor,
    gt_label: Tensor,
    batch: Optional[Tensor],
    ignore_index: int,
) -> dict[str, list]:
    scenes = [0] * gt_instance.numel() if batch is None else batch.tolist()
    points: dict[tuple[int, int], set[int]] = {}
    for point, (scene, instance, label) in enumerate(zip(scenes, gt_instance.tolist(), gt_label.tolist())):
        if instance >= 0 and label != ignore_index:
            points.setdefault((scene, instance), set()).add(point)
    keys = sorted(points)
    gt_labels = [int(gt_label[min(points[key])]) for key in keys]
    valid = set().union(*points.values())
    record: dict[str, list] = {"pred_counts": [], "pred_void": [], "pair_pred": [], "pair_gt": [], "pair_inter": []}
    for pred, (mask, label) in enumerate(zip(pred_masks, pred_labels.tolist())):
        inside = set(mask.nonzero().flatten().tolist())
        record["pred_counts"].append(len(inside))
        record["pred_void"].append(len(inside - valid))
        for gt, key in enumerate(keys):
            inter = len(inside & points[key])
            if inter > 0 and gt_labels[gt] == label:
                record["pair_pred"].append(pred)
                record["pair_gt"].append(gt)
                record["pair_inter"].append(inter)
    record["pred_labels"] = pred_labels.tolist()
    record["pred_scores"] = pred_scores.tolist()
    record["gt_labels"] = gt_labels
    record["gt_counts"] = [len(points[key]) for key in keys]
    return record


def _instance_matches(inputs: dict[str, Any]) -> InstanceMatches:
    preds = _preds(inputs["pred_masks"], inputs["pred_labels"], inputs["pred_scores"], inputs["pred_batch"])
    preds["ignore_mask"] = inputs["pred_ignore"]
    return instance_matches(
        preds,
        inputs["gt_instance"],
        inputs["gt_label"],
        batch=inputs["batch"],
        ignore_index=inputs["ignore_index"],
    )


def _assert_matches_brute_force(inputs: dict[str, Any]) -> None:
    match = {key: value for key, value in _instance_matches(inputs).items() if isinstance(value, Tensor)}
    keep = ~inputs["pred_ignore"]
    expected = _brute_force_matches(
        inputs["pred_masks"][keep],
        inputs["pred_labels"][keep],
        inputs["pred_scores"][keep],
        inputs["gt_instance"],
        inputs["gt_label"],
        inputs["batch"],
        inputs["ignore_index"],
    )
    assert {key: value.tolist() for key, value in match.items()} == expected
    assert all(value.device.type == "cpu" for value in match.values())
    assert match["pred_scores"].dtype == inputs["pred_scores"].dtype
    assert all(value.dtype == torch.long for key, value in match.items() if key != "pred_scores")


MATCH_CASES = [
    pytest.param({"sizes": [200], "num_instances": 6, "num_preds": 10}, id="one-scene"),
    pytest.param({"sizes": [200], "num_instances": 6, "num_preds": 10, "with_batch": False}, id="no-batch"),
    pytest.param({"sizes": [200], "num_instances": 6, "num_preds": 12, "tie_scores": True}, id="score-ties"),
    pytest.param({"sizes": [200], "num_instances": 6, "num_preds": 0}, id="no-predictions"),
    pytest.param({"sizes": [200], "num_instances": 0, "num_preds": 8}, id="no-instances"),
    pytest.param({"sizes": [200], "num_instances": 6, "num_preds": 8, "ignore_index": 255}, id="ignored-class"),
    pytest.param({"sizes": [1], "num_instances": 2, "num_preds": 4}, id="one-point"),
    pytest.param({"sizes": [400], "num_instances": 150, "num_preds": 30}, id="many-small-instances"),
    pytest.param({"sizes": [200], "num_instances": 3, "num_preds": 20}, id="overlapping-predictions"),
    pytest.param({"sizes": [120, 80, 50], "num_instances": 5, "num_preds": 6}, id="packed-batch"),
    pytest.param({"sizes": [120, 80, 50], "num_instances": 5, "num_preds": 6, "shuffle": True}, id="unsorted-batch"),
    pytest.param({"sizes": [200], "num_instances": 6, "num_preds": 10, "score_dtype": torch.float16}, id="float16"),
    pytest.param(
        {"sizes": [120, 80, 50], "num_instances": 5, "num_preds": 6, "with_ignore_mask": True}, id="ignore-mask"
    ),
]


@pytest.mark.parametrize("case", MATCH_CASES)
def test_instance_matches_equals_brute_force(case: dict[str, Any]) -> None:
    """Every field of the record equals a per-prediction, per-instance count over Python sets.

    The predictions are random dense masks, block-diagonal over the scenes, in their `decode` form.
    """
    _assert_matches_brute_force(_random_inputs(**case))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
@pytest.mark.parametrize("case", MATCH_CASES)
def test_instance_matches_equals_brute_force_cuda(case: dict[str, Any]) -> None:
    inputs = _random_inputs(**case)
    _assert_matches_brute_force(
        {key: value.cuda() if torch.is_tensor(value) else value for key, value in inputs.items()}
    )


def test_instance_matches_record_hand_checked() -> None:
    """Counts, void intersections and same-class pairs of a small scene, checked by hand.

    Instance 2 carries the ignore label, so it is excluded from the ground truth and its points are
    void. The class-1 prediction overlapping the class-0 instance produces no pair (cross-class).
    """
    gt_instance = torch.tensor([0, 0, 0, 1, 1, -1, -1, 2])
    gt_label = torch.tensor([0, 0, 0, 1, 1, -1, -1, -1])
    masks = torch.stack([_mask(8, [0, 1, 5]), _mask(8, [3, 4, 7]), _mask(8, [0, 1])])
    preds = _preds(masks, torch.tensor([0, 1, 1]), torch.tensor([0.9, 0.8, 0.7]))
    match = instance_matches(preds, gt_instance, gt_label)
    assert match["gt_counts"].tolist() == [3, 2]
    assert match["gt_labels"].tolist() == [0, 1]
    assert match["pred_counts"].tolist() == [3, 3, 2]
    assert match["pred_void"].tolist() == [1, 1, 0]
    assert match["pair_pred"].tolist() == [0, 1]
    assert match["pair_gt"].tolist() == [0, 1]
    assert match["pair_inter"].tolist() == [2, 2]


def test_instance_matches_packed_batch_equals_per_scene_records() -> None:
    """One call over a packed batch with `batch` scores like one call per scene, with ids repeating across scenes."""
    generator = torch.Generator().manual_seed(0)
    sizes = [60, 40]
    scenes = []
    for size in sizes:
        gt_instance = torch.randint(-1, 4, (size,), generator=generator)
        gt_label = torch.where(gt_instance >= 0, gt_instance % 2, -1)
        masks = torch.rand(6, size, generator=generator) > 0.6
        labels = torch.randint(0, 2, (6,), generator=generator)
        scores = torch.rand(6, generator=generator)
        scenes.append((masks, labels, scores, gt_instance, gt_label))
    per_scene = [instance_matches(_preds(*scene[:3]), *scene[3:]) for scene in scenes]

    masks = torch.block_diag(*(scene[0].int() for scene in scenes)).bool()
    labels, scores, gt_instance, gt_label = (torch.cat([scene[i] for scene in scenes]) for i in range(1, 5))
    batch = torch.repeat_interleave(torch.arange(len(sizes)), torch.tensor(sizes))
    preds = _preds(masks, labels, scores, torch.arange(len(sizes)).repeat_interleave(6))
    packed = instance_matches(preds, gt_instance, gt_label, batch=batch)

    assert packed["gt_counts"].tolist() == torch.cat([match["gt_counts"] for match in per_scene]).tolist()
    preds["batch"] = torch.zeros_like(labels)
    assert instance_matches(preds, gt_instance, gt_label)["gt_counts"].numel() < packed["gt_counts"].numel()
    for iou_threshold in ((0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95), 0.5, 0.25):
        expected = instance_average_precision(per_scene, iou_threshold=iou_threshold, num_classes=2, min_points=1)
        assert instance_average_precision(
            [packed],
            iou_threshold=iou_threshold,
            num_classes=2,
            min_points=1,
        ) == pytest.approx(expected)


@pytest.mark.parametrize(
    "mask_index,pred_batch,batch",
    [
        pytest.param([[0, 0], [1, 4]], [0], [0, 0, 0, 1, 1, 1], id="mask-spanning-two-scenes"),
        pytest.param([[0, 0], [4, 5]], [0], [0, 0, 0, 1, 1, 1], id="mask-in-another-scene"),
        pytest.param([[0, 0], [4, 5]], [1], None, id="scene-without-batch"),
    ],
)
def test_instance_matches_rejects_a_mask_outside_its_scene(
    mask_index: list[list[int]],
    pred_batch: list[int],
    batch: Optional[list[int]],
) -> None:
    preds: InstanceSegmentation3D = {
        "mask_index": torch.tensor(mask_index),
        "labels": torch.tensor([0]),
        "scores": torch.tensor([0.9]),
        "batch": torch.tensor(pred_batch),
    }
    gt_instance = torch.tensor([0, 0, 0, 0, 0, 0])
    with pytest.raises(ValueError, match="outside the scene of its instance"):
        instance_matches(preds, gt_instance, gt_instance, batch=None if batch is None else torch.tensor(batch))


def test_instance_matches_dense_masks_are_deprecated() -> None:
    """The 0.0.18 call with dense masks warns and gives the record of the `decode` form."""
    inputs = _random_inputs([200], 6, 10, with_batch=False)
    masks, labels, scores = inputs["pred_masks"], inputs["pred_labels"], inputs["pred_scores"]
    with pytest.warns(FutureWarning, match="removed in 0.0.20"):
        legacy = instance_matches(masks.int(), labels, scores, inputs["gt_instance"], inputs["gt_label"])
    expected = instance_matches(_preds(masks, labels, scores), inputs["gt_instance"], inputs["gt_label"])
    for key in expected:
        assert torch.equal(legacy[key], expected[key]), key  # type: ignore[literal-required]


def test_instance_average_precision_perfect_two_scenes() -> None:
    """Exact predictions over two scenes score 1.0 at every threshold, with named per-class keys."""
    matches = []
    for _ in range(2):
        gt_instance = torch.tensor([0, 0, 0, 1, 1, 1])
        gt_label = torch.tensor([0, 0, 0, 1, 1, 1])
        masks = torch.stack([_mask(6, [0, 1, 2]), _mask(6, [3, 4, 5])])
        matches.append(
            instance_matches(_preds(masks, torch.tensor([0, 1]), torch.tensor([0.9, 0.8])), gt_instance, gt_label)
        )
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
    match = instance_matches(_preds(masks, labels, scores), gt_instance, gt_label)
    per_class = instance_average_precision([match], average="none", num_classes=2, min_points=1)
    assert per_class.tolist() == pytest.approx([19.0 / 24.0, 2.0 / 3.0])
    assert instance_average_precision([match], num_classes=2, min_points=1) == pytest.approx(35.0 / 48.0)
    assert instance_average_precision([match], iou_threshold=0.5, min_points=1) == pytest.approx(43.0 / 48.0)
    assert instance_average_precision([match], iou_threshold=0.25, min_points=1) == pytest.approx(43.0 / 48.0)


def test_instance_average_precision_void_overlap_excused() -> None:
    """An unmatched prediction on void points is excused; on another class's instance it is a false positive.

    Void points carry the ignore label or no instance (a valid label without an instance is void too, as in the
    ScanNet evaluation). With the second mask fully on void points the metric stays 1.0. Moving that region onto an
    instance of another class turns the same mask into a false positive above the true positive: precision at full
    recall is 0.5 and the centered-step integral drops to 0.25.
    """
    masks = torch.stack([_mask(20, list(range(0, 10))), _mask(20, list(range(10, 20)))])
    labels = torch.tensor([0, 0])
    scores = torch.tensor([0.9, 0.95])

    gt_instance = torch.tensor([0] * 10 + [-1] * 10)
    for gt_label in (torch.tensor([0] * 10 + [-1] * 10), torch.tensor([0] * 20)):
        match = instance_matches(_preds(masks, labels, scores), gt_instance, gt_label)
        assert match["pred_void"].tolist() == [0, 10]
        assert instance_average_precision([match], num_classes=1, min_points=1) == pytest.approx(1.0)

    gt_instance = torch.tensor([0] * 10 + [1] * 10)
    gt_label = torch.tensor([0] * 10 + [1] * 10)
    match = instance_matches(_preds(masks, labels, scores), gt_instance, gt_label)
    assert instance_average_precision([match], average="none", num_classes=2, min_points=1)[0] == pytest.approx(0.25)
    per_class = instance_average_precision([match], iou_threshold=0.25, average="none", num_classes=2, min_points=1)
    assert per_class[0] == pytest.approx(0.25)


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
    match = instance_matches(_preds(masks, labels, scores), gt_instance, gt_label)
    per_class = instance_average_precision([match], average="none", num_classes=2, min_points=5)
    assert math.isnan(per_class[0].item())
    assert per_class[1].item() == pytest.approx(1.0)
    assert instance_average_precision([match], num_classes=2, min_points=5) == pytest.approx(1.0)
    assert instance_average_precision([match], iou_threshold=0.5, num_classes=2, min_points=5) == pytest.approx(1.0)


def test_instance_average_precision_empty_edges() -> None:
    """No predictions with ground truth scores 0.0; no ground truth at all reports only zero mAPs."""
    gt_instance = torch.tensor([0] * 10)
    gt_label = torch.tensor([0] * 10)
    no_preds = _preds(torch.zeros(0, 10, dtype=torch.bool), torch.zeros(0, dtype=torch.long), torch.zeros(0))
    empty = instance_matches(no_preds, gt_instance, gt_label)
    assert instance_average_precision([empty], average="none", num_classes=1, min_points=1).tolist() == [0.0]
    assert instance_average_precision([empty], num_classes=1, min_points=1) == 0.0

    no_gt = instance_matches(
        _preds(torch.stack([_mask(10, [0, 1])]), torch.tensor([0]), torch.tensor([0.9])),
        torch.full((10,), -1),
        torch.full((10,), -1),
    )
    assert math.isnan(instance_average_precision([no_gt], average="none", num_classes=1, min_points=1)[0].item())
    assert instance_average_precision([no_gt], num_classes=1, min_points=1) == 0.0
    assert instance_average_precision([no_gt], iou_threshold=0.25, num_classes=1, min_points=1) == 0.0
    with pytest.raises(ValueError, match="class_names"):
        instance_average_precision([no_gt], num_classes=1, class_names=["chair", "table"])


def _tree_scenes() -> list[InstanceMatches]:
    """Two scenes of trees: in the first a tree is matched (IoU 3/4), one is split in two (IoU 2/3 and 1/3),
    and one prediction covers no tree; the second scene has one tree and no prediction."""
    gt_instance = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1, 1, 1, -1, -1])
    masks = torch.stack([_mask(12, [0, 1, 2]), _mask(12, [4, 5, 6, 7]), _mask(12, [8, 9]), _mask(12, [10, 11])])
    first = instance_matches(
        _preds(masks, torch.zeros(4, dtype=torch.long), torch.ones(4)), gt_instance, torch.zeros(12)
    )
    empty = _preds(torch.zeros(0, 3, dtype=torch.bool), torch.zeros(0, dtype=torch.long), torch.zeros(0))
    second = instance_matches(empty, torch.zeros(3), torch.zeros(3))
    return [first, second]


def test_instance_precision_recall_and_f1_count_matches_at_the_threshold() -> None:
    matches = _tree_scenes()
    # true positives: the 3/4 match and the 4/6 half, out of 4 predictions and 3 trees
    assert instance_precision(matches) == 0.5
    assert instance_recall(matches) == pytest.approx(2 / 3)
    assert instance_f1_score(matches) == pytest.approx(4 / 7)
    assert instance_precision(matches, iou_threshold=0.7) == 0.25


def test_instance_recall_counts_each_instance_once() -> None:
    """Three copies of one of two trees: every copy is a true positive, the tree is matched once."""
    gt_instance = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
    masks = torch.stack([_mask(8, [0, 1, 2, 3])] * 3)
    match = instance_matches(
        _preds(masks, torch.zeros(3, dtype=torch.long), torch.ones(3)), gt_instance, torch.zeros(8)
    )
    assert instance_precision([match]) == 1.0
    assert instance_recall([match]) == 0.5
    assert instance_f1_score([match]) == pytest.approx(2 / 3)


def test_instance_coverage_averages_the_best_iou_per_scene() -> None:
    matches = _tree_scenes()
    assert instance_coverage(matches) == pytest.approx(((3 / 4 + 4 / 6) / 2 + 0.0) / 2)
    assert instance_coverage(matches, weighted=True) == pytest.approx(((3 / 4 * 4 + 4 / 6 * 6) / 10 + 0.0) / 2)


def test_instance_detection_metrics_without_instances() -> None:
    empty = _preds(torch.zeros(0, 3, dtype=torch.bool), torch.zeros(0, dtype=torch.long), torch.zeros(0))
    match = instance_matches(empty, -torch.ones(3), torch.zeros(3))
    assert instance_precision([match]) == instance_recall([match]) == instance_f1_score([match]) == 0.0
    assert instance_coverage([match]) == 0.0
