"""Point-mask instance-segmentation metrics: instance matches of a scene or packed batch and their average precision."""

import math
import warnings
from typing import Dict, List, Literal, Mapping, Optional, Sequence, TypedDict, Union, overload

import numpy as np
import torch
from torch import Tensor

from torch_pointcloud.utils.types import InstanceSegmentation3D


class InstanceMatches(TypedDict):
    r"""Predicted instance masks of a scene or a packed batch reduced against the ground truth (`instance_matches`).

    Holds per-instance counts and pairwise intersections only, nothing mask-sized: $K$ predictions, $I$
    ground-truth instances (ordered by scene, then ascending id) and $M$ same-class (prediction, instance) pairs
    with a nonzero intersection.

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
    preds: Union[InstanceSegmentation3D, Tensor],
    gt_instance: Tensor,
    gt_label: Tensor,
    *args: Tensor,
    batch: Optional[Tensor] = None,
    ignore_index: int = -1,
) -> InstanceMatches:
    r"""Reduce the predictions of a scene or a packed batch to the record scored by `instance_average_precision`.

    The predictions are a model's `decode` output: sparse masks given by their (instance, point) pairs, with a class,
    a score and a scene per instance. Instances flagged by `ignore_mask` are left out of the record, so they neither
    match an instance nor count as a false positive (e.g. the masks an evaluation protocol drops for their size). The
    record holds per-instance counts and pairwise intersections only, so nothing mask-sized outlives the call and a
    whole validation split can be accumulated batch by batch.

    Ground truth instances are the unique `gt_instance` ids among points with a non-negative id and a
    valid semantic label; points whose `gt_label` equals `ignore_index` or that belong to no instance are void,
    and predictions overlapping them are excused accordingly during scoring. Each instance must carry a single
    semantic label. Intersections are recorded for same-class (prediction, instance) pairs only.

    With `batch`, the points are a packed batch of scenes and instances are keyed by (scene, id), so ids may repeat
    across scenes; without it, every point and prediction belongs to scene $0$. Each predicted mask covers points of
    its own scene `preds["batch"]` only.

    Args:
        preds: Packed predictions (one `decode` output), `{"mask_index", "labels", "scores", "batch"}` with an
            optional `ignore_mask` (see `InstanceSegmentation3D`).
        gt_instance: Per-point ground-truth instance ids, shape $(N,)$; negative marks no instance.
        gt_label: Per-point semantic labels in the instance-class space, shape $(N,)$.
        *args: Ground truth of the deprecated call `instance_matches(pred_masks, pred_labels, pred_scores,
            gt_instance, gt_label)` with dense $(K, N)$ boolean masks of one scene, accepted until 0.0.20 with a
            `FutureWarning`.
        batch: Per-point scene index of a packed batch, shape $(N,)$. `None` scores the points as one scene.
        ignore_index: Semantic label marking void points.

    Returns:
        The `InstanceMatches` record of the predictions left after `ignore_mask`, as CPU tensors.

    Raises:
        ValueError: If a mask covers a point outside the scene of its instance.

    Shape:
        - preds["mask_index"]: $(2, M)$
        - preds["labels"], preds["scores"], preds["batch"]: $(K,)$
        - gt_instance, gt_label, batch: $(N,)$

    Example:
        ```pycon
        >>> index = torch.zeros(1, dtype=torch.long)
        >>> mask_index = torch.tensor([[0, 0, 0, 0], [0, 1, 2, 3]])
        >>> preds = {"mask_index": mask_index, "labels": index, "scores": torch.tensor([0.9]), "batch": index}
        >>> match = instance_matches(preds, torch.tensor([0, 0, 1, -1]), torch.tensor([0, 0, 0, -1]))
        >>> match["gt_counts"].tolist(), match["pair_inter"].tolist(), match["pred_void"].tolist()
        ([2, 1], [2, 1], [1])

        ```
    """
    if isinstance(preds, Tensor):
        warnings.warn(
            "`instance_matches(pred_masks, pred_labels, pred_scores, gt_instance, gt_label)` is deprecated since "
            "0.0.19 and will be removed in 0.0.20; pass the `decode` output instead, e.g. `instance_matches("
            "{'mask_index': pred_masks.nonzero().t(), 'labels': pred_labels, 'scores': pred_scores, 'batch': "
            "torch.zeros_like(pred_labels)}, gt_instance, gt_label)`.",
            FutureWarning,
            stacklevel=2,
        )
        pred_labels, pred_scores = gt_instance, gt_label
        gt_instance, gt_label = args
        preds = {
            "mask_index": preds.bool().nonzero().t(),
            "labels": pred_labels,
            "scores": pred_scores,
            "batch": torch.zeros_like(pred_labels),
        }

    gt_instance = gt_instance.long()
    gt_label = gt_label.long()
    point_batch = torch.zeros_like(gt_instance) if batch is None else batch.long()
    pred, point = preds["mask_index"].long()
    if (preds["batch"].long()[pred] != point_batch[point]).any():
        raise ValueError("A predicted mask covers points outside the scene of its instance (`preds['batch']`).")

    pred_labels, pred_scores = preds["labels"].long(), preds["scores"]
    ignore_mask = preds.get("ignore_mask")
    if ignore_mask is not None:
        keep = ~ignore_mask.bool()
        covered = keep[pred]
        pred, point = (keep.cumsum(0) - 1)[pred[covered]], point[covered]
        pred_labels, pred_scores = pred_labels[keep], pred_scores[keep]

    valid = (gt_label != ignore_index) & (gt_instance >= 0)
    ids = gt_instance[valid]
    if batch is not None:
        # One integer key per (scene, id rank), ordered like the (scene, id) pairs.
        ids = point_batch[valid] * ids.numel() + torch.unique(ids, return_inverse=True)[1]
    keys, inverse = torch.unique(ids, return_inverse=True)
    num_instances = keys.numel()
    gt_counts = torch.bincount(inverse, minlength=num_instances)
    gt_labels = gt_label.new_zeros(num_instances).scatter_(0, inverse, gt_label[valid])
    point_instance = torch.full_like(gt_instance, -1)
    point_instance[valid] = inverse

    # Column 0 counts the void points of each prediction, column i + 1 its intersection with instance i.
    num_preds = pred_labels.shape[0]
    counts = torch.bincount(
        pred * (num_instances + 1) + point_instance[point] + 1,
        minlength=num_preds * (num_instances + 1),
    ).view(num_preds, num_instances + 1)
    inter = counts[:, 1:]
    pair_pred, pair_gt = ((inter > 0) & (gt_labels == pred_labels[:, None])).nonzero(as_tuple=True)

    return {
        "pred_labels": pred_labels.detach().cpu(),
        "pred_scores": pred_scores.detach().cpu(),
        "pred_counts": counts.sum(dim=1).cpu(),
        "pred_void": counts[:, 0].contiguous().cpu(),
        "gt_labels": gt_labels.cpu(),
        "gt_counts": gt_counts.cpu(),
        "pair_pred": pair_pred.cpu(),
        "pair_gt": pair_gt.cpu(),
        "pair_inter": inter[pair_pred, pair_gt].cpu(),
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
    """Greedy mask-IoU AP for one class at one threshold over `instance_matches` records.

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
        matches: The `instance_matches` records of the evaluated scenes or batches.
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
        >>> preds = {
        ...     "mask_index": torch.tensor([[0, 0, 0, 1], [0, 1, 2, 3]]),
        ...     "labels": torch.tensor([0, 1]),
        ...     "scores": torch.tensor([0.9, 0.8]),
        ...     "batch": torch.tensor([0, 0]),
        ... }
        >>> match = instance_matches(preds, torch.tensor([0, 0, 0, 1]), torch.tensor([0, 0, 0, 1]))
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
