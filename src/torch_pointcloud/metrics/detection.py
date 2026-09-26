"""3D box detection metrics: per-batch box matches and their average precision."""

import math
from typing import Dict, List, Literal, Mapping, Optional, Sequence, Tuple, TypedDict, Union, overload

import numpy as np
import torch
from torch import Tensor

from torch_pointcloud.ops.box3d import box3d_overlap, box_corners
from torch_pointcloud.utils.types import Boxes3D, Detection3D

Interpolation = Literal["all", "r11", "r40"]


def _voc_ap(recall: np.ndarray, precision: np.ndarray, interpolation: Interpolation = "all") -> float:
    r"""VOC average precision over a cumulative precision-recall curve.

    `"all"` is the all-points variant (exact area under the right-max interpolated curve). `"r11"` and
    `"r40"` follow the KITTI protocol: up to $41$ score thresholds are picked where the recall curve
    crosses an even $1/40$ grid, precision is right-max interpolated over those samples, and the AP is
    the mean of every 4th sample (`"r11"`, includes the recall $\approx 0$ sample) or of samples
    $1..40$ (`"r40"`); grid slots past the achieved recall stay $0$.
    """
    if interpolation == "all":
        mrec = np.concatenate(([0.0], recall, [1.0]))
        mpre = np.concatenate(([0.0], precision, [0.0]))
        mpre = np.maximum.accumulate(mpre[::-1])[::-1]
        idx = np.where(mrec[1:] != mrec[:-1])[0]
        return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))

    num_samples = 41
    sampled = np.zeros(num_samples)
    tp_idx = np.where(np.diff(recall, prepend=0.0) > 0)[0]

    current_recall = 0.0
    slot = 0
    for j, d in enumerate(tp_idx):
        left = float(recall[d])
        right = float(recall[tp_idx[j + 1]]) if j + 1 < len(tp_idx) else left
        if (right - current_recall) < (current_recall - left) and j < len(tp_idx) - 1:
            continue
        if slot == num_samples:
            break
        sampled[slot] = precision[d]
        slot += 1
        current_recall += 1.0 / (num_samples - 1)

    interpolated = np.maximum.accumulate(sampled[::-1])[::-1]
    return float(interpolated[::4].mean() if interpolation == "r11" else interpolated[1:].mean())


def _best_same_class_iou(
    corners: Tensor,
    labels: Tensor,
    batch: Tensor,
    other_corners: Tensor,
    other_labels: Tensor,
    other_batch: Tensor,
) -> Tuple[Tensor, Tensor]:
    """Largest IoU of each box with a same-class box of `other` in its sample and that box's row, both $-1$ if none."""
    overlap = box3d_overlap(corners, other_corners)[1]
    mismatch = (labels[:, None] != other_labels[None, :]) | (batch[:, None] != other_batch[None, :])
    # A trailing -1 column keeps the row max defined when `other` holds no box at all.
    overlap = torch.cat([overlap.masked_fill(mismatch, -1.0), overlap.new_full((overlap.shape[0], 1), -1.0)], dim=1)
    best_iou, best_row = overlap.max(dim=1)
    return best_iou, best_row.masked_fill(best_iou < 0, -1)


class BoxMatches(TypedDict):
    r"""One batch of box predictions reduced against its ground truth (the output of `box_matches`).

    Holds what scoring needs and nothing box-sized: $P$ predictions are left after the prediction `ignore_mask`,
    $G$ ground-truth boxes after the target `ignore_mask`. A prediction with no same-class box (or ignore region)
    in its sample carries $-1$ in the matching fields.

    Attributes:
        pred_scores: Per-prediction confidence score, shape $(P,)$.
        pred_labels: Per-prediction class, shape $(P,)$.
        pred_iou: Largest IoU with a same-class ground-truth box of the same sample, shape $(P,)$.
        pred_gt: Row of that box in `gt_labels`, shape $(P,)$.
        pred_ignore_iou: Largest IoU with a same-class ignore region of the same sample, shape $(P,)$.
        gt_labels: Per-box class of the ground truth, shape $(G,)$.
    """

    pred_scores: Tensor
    pred_labels: Tensor
    pred_iou: Tensor
    pred_gt: Tensor
    pred_ignore_iou: Tensor
    gt_labels: Tensor


def box_matches(preds: Detection3D, target: Boxes3D) -> BoxMatches:
    r"""Match one batch of box predictions to its ground truth, the record scored by `box_average_precision`.

    Which ground-truth box a prediction overlaps most, and by how much, depends on neither the IoU threshold
    nor on what the other predictions matched. The oriented IoU is therefore paid here, once per batch and on
    the device of the inputs, and nothing box-sized outlives the call: a whole validation split accumulates
    as a few flat tensors per batch, so keep one record per batch in a list and score the list.

    Target boxes flagged by `ignore_mask` (see `Boxes3D`) are ignore regions rather than ground truth; their
    `labels` entry names the class they excuse. Predictions flagged by their own `ignore_mask` are left out
    of the record, so they can neither match a box nor count as a false positive (the KITTI min-height rule).

    Args:
        preds: Packed predictions (one `decode` output), `{"boxes", "scores", "labels", "batch"}` with an
            optional `ignore_mask`.
        target: Packed ground truth aligned to `preds`, `{"boxes", "labels", "batch"}` with an optional
            `ignore_mask`.

    Returns:
        The `BoxMatches` record of the batch, as CPU tensors.

    Shape:
        - preds["boxes"]: $(P, 7)$
        - target["boxes"]: $(G, 7)$
        - output: six tensors of shape $(P',)$ or $(G',)$, the predictions and boxes left after the ignore masks

    Example:
        ```pycon
        >>> boxes = torch.tensor([[0.0, 0.0, 0.0, 2.0, 2.0, 2.0, 0.0], [9.0, 9.0, 9.0, 1.0, 1.0, 1.0, 0.0]])
        >>> index = torch.zeros(2, dtype=torch.long)
        >>> preds = {"boxes": boxes, "scores": torch.tensor([0.9, 0.4]), "labels": index, "batch": index}
        >>> match = box_matches(preds, {"boxes": boxes[:1], "labels": index[:1], "batch": index[:1]})
        >>> match["pred_iou"].tolist(), match["pred_gt"].tolist()
        ([1.0, 0.0], [0, 0])
        >>> box_average_precision([match], iou_threshold=0.5)
        1.0

        ```
    """
    pred_ignore, region_mask = preds.get("ignore_mask"), target.get("ignore_mask")
    pred_keep = torch.ones_like(preds["labels"], dtype=torch.bool) if pred_ignore is None else ~pred_ignore.bool()
    is_region = torch.zeros_like(target["labels"], dtype=torch.bool) if region_mask is None else region_mask.bool()

    pred_corners = box_corners(preds["boxes"].detach().reshape(-1, 7)[pred_keep])
    pred_scores, pred_labels, pred_batch = (
        preds["scores"][pred_keep],
        preds["labels"][pred_keep],
        preds["batch"][pred_keep],
    )
    target_corners = box_corners(target["boxes"].detach().reshape(-1, 7))
    gt_corners, gt_labels, gt_batch = (
        target_corners[~is_region],
        target["labels"][~is_region],
        target["batch"][~is_region],
    )
    region_corners, region_labels = target_corners[is_region], target["labels"][is_region]
    region_batch = target["batch"][is_region]

    # Records list the predictions sample by sample. On GPU the launch overhead of one IoU call per sample outweighs
    # the masked cross-sample pairs of a single call, as long as the batch stays within 2^16 pairs.
    order = torch.argsort(pred_batch, stable=True)
    single_call = pred_corners.is_cuda and pred_corners.shape[0] * target_corners.shape[0] <= 1 << 16
    sample_sizes = pred_batch.unique(return_counts=True)[1].tolist()
    groups = [order] if single_call else order.split(sample_sizes)

    pred_iou: List[Tensor] = []
    pred_gt: List[Tensor] = []
    pred_ignore_iou: List[Tensor] = []
    for rows in groups if order.numel() > 0 else []:
        gt_rows = torch.isin(gt_batch, pred_batch[rows]).nonzero(as_tuple=False).squeeze(-1)
        region_rows = torch.isin(region_batch, pred_batch[rows]).nonzero(as_tuple=False).squeeze(-1)
        best_iou, best_row = _best_same_class_iou(
            pred_corners[rows],
            pred_labels[rows],
            pred_batch[rows],
            gt_corners[gt_rows],
            gt_labels[gt_rows],
            gt_batch[gt_rows],
        )
        ignore_iou, _ = _best_same_class_iou(
            pred_corners[rows],
            pred_labels[rows],
            pred_batch[rows],
            region_corners[region_rows],
            region_labels[region_rows],
            region_batch[region_rows],
        )
        pred_iou.append(best_iou)
        pred_gt.append(torch.cat([gt_rows, gt_rows.new_full((1,), -1)])[best_row])
        pred_ignore_iou.append(ignore_iou)

    def cat(tensors: List[Tensor], dtype: torch.dtype) -> Tensor:
        return torch.cat(tensors).detach().cpu() if tensors else torch.zeros(0, dtype=dtype)

    return {
        "pred_scores": pred_scores[order].detach().cpu(),
        "pred_labels": pred_labels[order].detach().cpu(),
        "pred_iou": cat(pred_iou, pred_corners.dtype),
        "pred_gt": cat(pred_gt, torch.long),
        "pred_ignore_iou": cat(pred_ignore_iou, pred_corners.dtype),
        "gt_labels": gt_labels.detach().cpu(),
    }


def _box_outcomes(match: BoxMatches, iou_threshold: Tensor) -> Tuple[Tensor, Tensor]:
    """True- and false-positive flags of one `box_matches` record, given the IoU threshold of each prediction.

    A prediction only ever claims its best box, so a box goes to the highest-scored prediction whose best box it
    is above the threshold; every other prediction is a false positive, unless it overlaps a same-class ignore
    region above the threshold, which drops it (neither flag set).
    """
    pred_iou, pred_gt = match["pred_iou"], match["pred_gt"]
    claims = pred_iou > iou_threshold
    order = torch.argsort(match["pred_scores"], descending=True, stable=True)
    rank = torch.empty_like(order)
    rank[order] = torch.arange(order.numel())
    best_rank = torch.full((match["gt_labels"].numel(),), order.numel(), dtype=torch.long)
    best_rank.scatter_reduce_(0, pred_gt[claims], rank[claims], reduce="amin")
    true_positive = claims.clone()
    true_positive[claims] = rank[claims] == best_rank[pred_gt[claims]]
    false_positive = ~true_positive & (match["pred_ignore_iou"] <= iou_threshold)
    return true_positive, false_positive


def _ranked_ap(
    scores: Tensor,
    true_positive: Tensor,
    false_positive: Tensor,
    npos: int,
    interpolation: Interpolation = "all",
) -> float:
    """VOC AP of one class from its predictions' scores and outcome flags, against `npos` ground-truth boxes."""
    if scores.numel() == 0:
        return 0.0

    order = torch.argsort(scores, descending=True, stable=True)
    tp_cum = np.cumsum(true_positive[order].numpy().astype(np.float64))
    fp_cum = np.cumsum(false_positive[order].numpy().astype(np.float64))
    recall = tp_cum / max(npos, 1)
    precision = tp_cum / np.maximum(tp_cum + fp_cum, np.finfo(np.float64).eps)
    return _voc_ap(recall, precision, interpolation)


@overload
def box_average_precision(
    matches: Sequence[BoxMatches],
    *,
    iou_threshold: Union[float, Mapping[int, float]] = ...,
    average: Literal["macro"] = ...,
    num_classes: Optional[int] = ...,
    class_names: Optional[Sequence[str]] = ...,
    interpolation: Interpolation = ...,
) -> float: ...


@overload
def box_average_precision(
    matches: Sequence[BoxMatches],
    *,
    iou_threshold: Union[float, Mapping[int, float]] = ...,
    average: Literal["none"],
    num_classes: Optional[int] = ...,
    class_names: None = ...,
    interpolation: Interpolation = ...,
) -> Tensor: ...


@overload
def box_average_precision(
    matches: Sequence[BoxMatches],
    *,
    iou_threshold: Union[float, Mapping[int, float]] = ...,
    average: Literal["none"],
    num_classes: Optional[int] = ...,
    class_names: Sequence[str],
    interpolation: Interpolation = ...,
) -> Dict[str, float]: ...


def box_average_precision(
    matches: Sequence[BoxMatches],
    *,
    iou_threshold: Union[float, Mapping[int, float]] = 0.5,
    average: Literal["macro", "none"] = "macro",
    num_classes: Optional[int] = None,
    class_names: Optional[Sequence[str]] = None,
    interpolation: Interpolation = "all",
) -> Union[float, Tensor, Dict[str, float]]:
    r"""3D detection average precision (AP) of matched box predictions.

    Dataset- and model-agnostic: `box_matches` reduces any detector's packed `(boxes, scores, labels, batch)`
    output to the same record, one per batch, and the records of a whole split are scored together. Within a
    class, predictions claim their best ground-truth box in descending score order: the first one above the IoU
    threshold is a true positive, the others are false positives, and the AP is the area under the resulting
    precision-recall curve.

    With one `iou_threshold` for every class, the classes that have ground truth are scored. With one per class
    index, exactly those classes are scored, and one without ground truth scores $0$ (its predictions are all
    false positives).

    Args:
        matches: The `box_matches` record of every evaluated batch.
        iou_threshold: IoU a match must exceed: one value for every class (e.g. `0.25`), or one per class index
            (e.g. KITTI's `{0: 0.7, 1: 0.5, 2: 0.5}`).
        average: `"macro"` returns the mean AP (mAP) over the scored classes; `"none"` returns the per-class AP.
        num_classes: Number of classes, i.e. the length of the `average="none"` output; defaults to the number of
            `class_names`, else to the largest class index met plus one.
        class_names: Name of each class index; with `average="none"` the per-class AP comes back as a
            `{name: ap}` dict instead of a tensor.
        interpolation: AP interpolation: `"all"` integrates the full precision-recall curve; `"r11"` /
            `"r40"` sample the KITTI 11- / 40-point recall grids.

    Returns:
        The mAP as a float with `average="macro"` ($0$ when no class is scored), or the per-class AP, shape
        $(C,)$ float64, with `average="none"` (a `{name: ap}` dict when `class_names` is given), holding NaN for
        the classes that are not scored.

    Shape:
        - output: scalar, or $(C,)$ with `average="none"`

    Example:
        ```pycon
        >>> boxes = torch.tensor([[0.0, 0.0, 0.0, 2.0, 2.0, 2.0, 0.0], [9.0, 9.0, 9.0, 1.0, 1.0, 1.0, 0.0]])
        >>> labels, batch = torch.tensor([0, 1]), torch.tensor([0, 0])
        >>> preds = {"boxes": boxes, "scores": torch.tensor([0.9, 0.4]), "labels": labels, "batch": batch}
        >>> matches = [box_matches(preds, {"boxes": boxes[:1], "labels": labels[:1], "batch": batch[:1]})]
        >>> box_average_precision(matches, iou_threshold=0.25)
        1.0
        >>> box_average_precision(matches, iou_threshold={0: 0.7, 1: 0.5}, average="none")
        tensor([1., 0.], dtype=torch.float64)
        >>> box_average_precision(matches, average="none", class_names=["Car", "Cyclist"])
        {'Car': 1.0, 'Cyclist': nan}

        ```
    """
    if class_names is not None and num_classes not in (None, len(class_names)):
        raise ValueError(f"Got {len(class_names)} `class_names` for `num_classes={num_classes}`.")

    if num_classes is None and class_names is not None:
        num_classes = len(class_names)
    if num_classes is None:
        indices = [int(index) for index in iou_threshold] if isinstance(iou_threshold, Mapping) else []
        for match in matches:
            indices += [int(labels.max()) for labels in (match["pred_labels"], match["gt_labels"]) if labels.numel()]
        num_classes = max(indices, default=-1) + 1

    thresholds = torch.full((num_classes,), math.nan, dtype=torch.float64)
    if isinstance(iou_threshold, Mapping):
        for index, value in iou_threshold.items():
            thresholds[int(index)] = float(value)
    else:
        thresholds[:] = float(iou_threshold)

    def cat(tensors: List[Tensor]) -> Tensor:
        return torch.cat(tensors) if tensors else torch.zeros(0)

    # Labels outside [0, C), e.g. a negative "don't care" class, read the trailing NaN and are never scored.
    lookup = torch.cat([thresholds, thresholds.new_full((1,), math.nan)])
    outcomes = []
    for match in matches:
        pred_labels = match["pred_labels"].long()
        in_range = (pred_labels >= 0) & (pred_labels < num_classes)
        pred_thresholds = lookup[torch.where(in_range, pred_labels, num_classes)].to(match["pred_iou"].dtype)
        outcomes.append(_box_outcomes(match, pred_thresholds))

    scores = cat([match["pred_scores"] for match in matches])
    labels = cat([match["pred_labels"] for match in matches]).long()
    gt_labels = cat([match["gt_labels"] for match in matches]).long()
    true_positive = cat([tp for tp, _ in outcomes]).bool()
    false_positive = cat([fp for _, fp in outcomes]).bool()
    npos = torch.bincount(gt_labels[(gt_labels >= 0) & (gt_labels < num_classes)], minlength=num_classes)
    scored = ~thresholds.isnan() if isinstance(iou_threshold, Mapping) else npos > 0

    per_class = torch.full((num_classes,), math.nan, dtype=torch.float64)
    aps: List[float] = []
    for label in range(num_classes):
        if not bool(scored[label]):
            continue

        keep = labels == label
        ap = _ranked_ap(scores[keep], true_positive[keep], false_positive[keep], int(npos[label]), interpolation)
        per_class[label] = ap
        aps.append(ap)

    if average == "none":
        return per_class if class_names is None else dict(zip(class_names, per_class.tolist()))
    return float(np.mean(aps)) if aps else 0.0
