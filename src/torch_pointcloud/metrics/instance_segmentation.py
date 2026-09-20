"""Point-mask instance-segmentation metrics: per-scene instance matches and their average precision."""

import math
from typing import Dict, List, Literal, Mapping, Optional, Sequence, TypedDict, Union, overload

import numpy as np
import torch
from torch import Tensor


class InstanceMatches(TypedDict):
    r"""One scene of predicted instance masks reduced against its ground truth (the output of `instance_matches`).

    Holds per-instance counts and pairwise intersections only, nothing mask-sized: $K$ predictions, $I$
    ground-truth instances (ordered by ascending id) and $M$ same-class (prediction, instance) pairs with a
    nonzero intersection.

    Attributes:
        pred_labels: Per-prediction class, shape $(K,)$.
        pred_scores: Per-prediction confidence score, shape $(K,)$.
        pred_counts: Per-prediction point count, shape $(K,)$.
        pred_void: Per-prediction count of void points (ignored label or no instance), shape $(K,)$.
        gt_labels: Per-instance class, shape $(I,)$.
        gt_counts: Per-instance point count, shape $(I,)$.
        pair_pred: Prediction index of each overlapping pair, shape $(M,)$.
        pair_gt: Instance index of each overlapping pair, shape $(M,)$.
        pair_inter: Number of points shared by each overlapping pair, shape $(M,)$.
    """

    pred_labels: Tensor
    pred_scores: Tensor
    pred_counts: Tensor
    pred_void: Tensor
    gt_labels: Tensor
    gt_counts: Tensor
    pair_pred: Tensor
    pair_gt: Tensor
    pair_inter: Tensor


def instance_matches(
    pred_masks: Tensor,
    pred_labels: Tensor,
    pred_scores: Tensor,
    gt_instance: Tensor,
    gt_label: Tensor,
    *,
    ignore_index: int = -1,
) -> InstanceMatches:
    r"""Reduce one scene's instance predictions to the compact match record scored by `instance_average_precision`.

    Predicted masks are dense $(K, N)$ booleans: a few hundred masks over a $\sim 50\text{k}$-point scene
    is only tens of MB and intersections reduce to bincounts, while index lists would be ragged and no
    smaller. The returned record holds per-instance counts and pairwise intersections only, so nothing
    mask-sized outlives the call and a whole validation split can be accumulated scene by scene.

    Ground truth instances are the unique `gt_instance` ids among points with a non-negative id and a
    valid semantic label; points whose `gt_label` equals `ignore_index` are void, and predictions
    overlapping them are excused accordingly during scoring. Each instance must carry a single semantic
    label. Intersections are recorded for same-class (prediction, instance) pairs only.

    Args:
        pred_masks: Per-instance point masks, shape $(K, N)$ bool.
        pred_labels: Per-instance class indices, shape $(K,)$.
        pred_scores: Per-instance confidences, shape $(K,)$.
        gt_instance: Per-point ground-truth instance ids, shape $(N,)$; negative marks no instance.
        gt_label: Per-point semantic labels in the instance-class space, shape $(N,)$.
        ignore_index: Semantic label marking void points.

    Returns:
        The `InstanceMatches` record of the scene, as CPU tensors.

    Example:
        ```pycon
        >>> masks = torch.tensor([[True, True, True, True]])
        >>> match = instance_matches(
        ...     masks, torch.tensor([0]), torch.tensor([0.9]), torch.tensor([0, 0, 1, -1]), torch.tensor([0, 0, 0, -1])
        ... )
        >>> match["gt_counts"].tolist(), match["pair_inter"].tolist(), match["pred_void"].tolist()
        ([2, 1], [2, 1], [1])

        ```
    """
    pred_masks = pred_masks.bool()
    pred_labels = pred_labels.long()
    gt_instance = gt_instance.long()
    gt_label = gt_label.long()

    valid = (gt_label != ignore_index) & (gt_instance >= 0)
    void = gt_label == ignore_index
    inverse = torch.unique(gt_instance[valid], return_inverse=True)[1]
    num_instances = int(inverse.max().item()) + 1 if inverse.numel() else 0
    gt_counts = torch.bincount(inverse, minlength=num_instances)
    gt_labels = gt_label.new_zeros(num_instances).scatter_(0, inverse, gt_label[valid])
    point_instance = torch.full_like(gt_instance, -1)
    point_instance[valid] = inverse

    empty = gt_instance.new_zeros(0)
    pair_pred, pair_gt, pair_inter = [empty], [empty], [empty]
    pred_counts = gt_instance.new_zeros(pred_masks.shape[0])
    pred_void = gt_instance.new_zeros(pred_masks.shape[0])
    for index in range(pred_masks.shape[0]):
        pred_counts[index] = pred_masks[index].sum()
        pred_void[index] = (pred_masks[index] & void).sum()
        hits = point_instance[pred_masks[index]]
        inter = torch.bincount(hits[hits >= 0], minlength=num_instances)
        gt_index = ((inter > 0) & (gt_labels == pred_labels[index])).nonzero(as_tuple=True)[0]
        pair_pred.append(torch.full_like(gt_index, index))
        pair_gt.append(gt_index)
        pair_inter.append(inter[gt_index])

    return {
        "pred_labels": pred_labels.detach().cpu(),
        "pred_scores": pred_scores.detach().cpu(),
        "pred_counts": pred_counts.cpu(),
        "pred_void": pred_void.cpu(),
        "gt_labels": gt_labels.cpu(),
        "gt_counts": gt_counts.cpu(),
        "pair_pred": torch.cat(pair_pred).cpu(),
        "pair_gt": torch.cat(pair_gt).cpu(),
        "pair_inter": torch.cat(pair_inter).cpu(),
    }


def _instance_ap(y_true: np.ndarray, y_score: np.ndarray, num_missed: int) -> float:
    r"""AP of one class at one threshold from its TP/FP entries and unmatched ground-truth count.

    Precision and recall are sampled at each unique score (ascending) plus an artificial
    $(\text{recall}, \text{precision}) = (0, 1)$ point, and integrated with centered recall steps
    (half the distance between the neighboring samples), the indoor-benchmark convention.
    """
    order = np.argsort(y_score)
    y_score = y_score[order]
    y_true = y_true[order]
    cumsum = np.cumsum(y_true)
    num_true = int(cumsum[-1]) if len(cumsum) else 0
    cumsum = np.append(cumsum, 0)  # index -1 stands for "no lower-scored entries"
    unique_indices = np.unique(y_score, return_index=True)[1]

    precision = np.ones(len(unique_indices) + 1)
    recall = np.zeros(len(unique_indices) + 1)
    for out_index, score_index in enumerate(unique_indices):
        below = int(cumsum[score_index - 1])
        tp = num_true - below
        fp = len(y_score) - score_index - tp
        precision[out_index] = tp / (tp + fp)
        recall[out_index] = tp / (tp + below + num_missed)

    padded = np.concatenate(([recall[0]], recall, [0.0]))
    widths = np.convolve(padded, [-0.5, 0.0, 0.5], "valid")
    return float(np.dot(precision, widths))


def _instance_class_ap(
    scenes: Sequence[Mapping[str, np.ndarray]],
    label: int,
    iou_threshold: float,
    min_points: int,
) -> float:
    """Greedy mask-IoU AP for one class at one threshold over per-scene `instance_matches` records.

    Ground-truth instances (in ascending id order) greedily consume overlapping predictions above the
    threshold in prediction order; extra predictions on an already-matched instance become false
    positives carrying the lower of the two scores. An unmatched prediction is dropped, neither a true
    nor a false positive, when it overlaps any same-class instance above the threshold or when the void
    and small-instance fraction of its points exceeds the threshold. Predictions and ground-truth
    instances below `min_points` are excluded, small instances counting as ignore regions.
    """
    y_true: List[float] = []
    y_score: List[float] = []
    num_missed = 0
    has_gt = False
    has_pred = False
    for scene in scenes:
        gt_counts, pred_counts = scene["gt_counts"], scene["pred_counts"]
        pred_scores = scene["pred_scores"]
        pred_keep = pred_counts >= min_points
        pred_indices = np.flatnonzero((scene["pred_labels"] == label) & pred_keep)
        gt_indices = np.flatnonzero(scene["gt_labels"] == label)
        valid_gt = gt_indices[gt_counts[gt_indices] >= min_points]
        has_gt = has_gt or len(valid_gt) > 0
        has_pred = has_pred or len(pred_indices) > 0

        keep = pred_keep[scene["pair_pred"]] & (scene["gt_labels"][scene["pair_gt"]] == label)
        pair_pred, pair_gt, pair_inter = (scene[k][keep] for k in ("pair_pred", "pair_gt", "pair_inter"))
        above = pair_inter / (gt_counts[pair_gt] + pred_counts[pair_pred] - pair_inter) > iou_threshold

        visited = np.zeros(len(pred_counts), dtype=bool)
        for gt_index in valid_gt:
            matched = False
            score = 0.0
            for row in np.flatnonzero(pair_gt == gt_index):
                pred_index = pair_pred[row]
                if not above[row] or visited[pred_index]:
                    continue
                confidence = float(pred_scores[pred_index])
                if matched:
                    y_true.append(0.0)
                    y_score.append(min(score, confidence))
                    score = max(score, confidence)
                else:
                    matched = True
                    score = confidence
                    visited[pred_index] = True
            if matched:
                y_true.append(1.0)
                y_score.append(score)
            else:
                num_missed += 1

        for pred_index in pred_indices:
            rows = np.flatnonzero(pair_pred == pred_index)
            if above[rows].any():
                continue
            ignored = scene["pred_void"][pred_index] + pair_inter[rows][gt_counts[pair_gt[rows]] < min_points].sum()
            if ignored / pred_counts[pred_index] <= iou_threshold:
                y_true.append(0.0)
                y_score.append(float(pred_scores[pred_index]))

    if not has_gt:
        return float("nan")
    if not has_pred:
        return 0.0
    return _instance_ap(np.array(y_true), np.array(y_score), num_missed)


@overload
def instance_average_precision(
    matches: Sequence[InstanceMatches],
    *,
    iou_threshold: Union[float, Sequence[float]] = ...,
    average: Literal["macro"] = ...,
    num_classes: Optional[int] = ...,
    class_names: Optional[Sequence[str]] = ...,
    min_points: int = ...,
) -> float: ...


@overload
def instance_average_precision(
    matches: Sequence[InstanceMatches],
    *,
    iou_threshold: Union[float, Sequence[float]] = ...,
    average: Literal["none"],
    num_classes: Optional[int] = ...,
    class_names: None = ...,
    min_points: int = ...,
) -> Tensor: ...


@overload
def instance_average_precision(
    matches: Sequence[InstanceMatches],
    *,
    iou_threshold: Union[float, Sequence[float]] = ...,
    average: Literal["none"],
    num_classes: Optional[int] = ...,
    class_names: Sequence[str],
    min_points: int = ...,
) -> Dict[str, float]: ...


def instance_average_precision(
    matches: Sequence[InstanceMatches],
    *,
    iou_threshold: Union[float, Sequence[float]] = (0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9),
    average: Literal["macro", "none"] = "macro",
    num_classes: Optional[int] = None,
    class_names: Optional[Sequence[str]] = None,
    min_points: int = 100,
) -> Union[float, Tensor, Dict[str, float]]:
    r"""Point-mask instance-segmentation average precision (AP) of matched instance predictions.

    Follows the standard indoor instance-segmentation protocol (the ScanNet benchmark): per class and
    IoU threshold, ground-truth instances greedily consume overlapping predicted masks above the
    threshold, duplicates on a matched instance count as false positives with the lower score, and an
    unmatched prediction whose void / small-instance point fraction exceeds the threshold is excused.
    The AP integrates the score-swept precision-recall curve with centered recall steps, and is averaged
    over the IoU thresholds: the default sweep $0.5, 0.55, \ldots, 0.9$ is the benchmark's headline AP,
    `iou_threshold=0.5` and `0.25` its AP50 and AP25. A class is scored when it has ground truth.

    Args:
        matches: The `instance_matches` record of every evaluated scene.
        iou_threshold: IoU a match must exceed, or a sequence of them to average the AP over.
        average: `"macro"` returns the mean AP (mAP) over the scored classes; `"none"` returns the per-class AP.
        num_classes: Number of instance classes, i.e. the length of the `average="none"` output; defaults to the
            number of `class_names`, else to the largest class index met plus one.
        class_names: Name of each class index; with `average="none"` the per-class AP comes back as a
            `{name: ap}` dict instead of a tensor.
        min_points: Minimum point count for a prediction or ground-truth instance to be scored;
            smaller ground-truth instances count as ignore regions.

    Returns:
        The mAP as a float with `average="macro"` ($0$ when no class is scored), or the per-class AP, shape
        $(C,)$ float64, with `average="none"` (a `{name: ap}` dict when `class_names` is given), holding NaN
        for the classes without ground truth.

    Shape:
        - output: scalar, or $(C,)$ with `average="none"`

    Example:
        ```pycon
        >>> masks = torch.tensor([[True, True, True, False], [False, False, False, True]])
        >>> match = instance_matches(
        ...     masks,
        ...     torch.tensor([0, 1]),
        ...     torch.tensor([0.9, 0.8]),
        ...     torch.tensor([0, 0, 0, 1]),
        ...     torch.tensor([0, 0, 0, 1]),
        ... )
        >>> instance_average_precision([match], min_points=1)
        1.0
        >>> instance_average_precision([match], iou_threshold=0.5, average="none", class_names=["chair", "table"], min_points=1)
        {'chair': 1.0, 'table': 1.0}

        ```
    """
    if class_names is not None and num_classes not in (None, len(class_names)):
        raise ValueError(f"Got {len(class_names)} `class_names` for `num_classes={num_classes}`.")
    if num_classes is None and class_names is not None:
        num_classes = len(class_names)
    if num_classes is None:
        indices = [
            int(labels.max())
            for match in matches
            for labels in (match["pred_labels"], match["gt_labels"])
            if labels.numel()
        ]
        num_classes = max(indices, default=-1) + 1

    scenes = [
        {
            "pred_labels": match["pred_labels"].numpy(),
            "pred_scores": match["pred_scores"].numpy(),
            "pred_counts": match["pred_counts"].numpy(),
            "pred_void": match["pred_void"].numpy(),
            "gt_labels": match["gt_labels"].numpy(),
            "gt_counts": match["gt_counts"].numpy(),
            "pair_pred": match["pair_pred"].numpy(),
            "pair_gt": match["pair_gt"].numpy(),
            "pair_inter": match["pair_inter"].numpy(),
        }
        for match in matches
    ]
    thresholds = (
        [float(iou_threshold)] if isinstance(iou_threshold, (int, float)) else [float(t) for t in iou_threshold]
    )
    ap = np.full((num_classes, len(thresholds)), np.nan)
    for class_index in range(num_classes):
        for threshold_index, threshold in enumerate(thresholds):
            ap[class_index, threshold_index] = _instance_class_ap(scenes, class_index, threshold, min_points)

    per_class = [float(np.mean(row)) for row in ap]
    if average == "none":
        return (
            torch.tensor(per_class, dtype=torch.float64) if class_names is None else dict(zip(class_names, per_class))
        )
    scored = [value for value in per_class if not math.isnan(value)]
    return float(np.mean(scored)) if scored else 0.0
