import math
from typing import Dict, List, Literal, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import Tensor

from .box3d import box3d_overlap, box_corners
from .ops import safe_divide
from .types import Boxes3D, Detection3D

Interpolation = Literal["all", "r11", "r40"]


def confusion_matrix(
    preds: Tensor,
    target: Tensor,
    num_classes: int,
    ignore_index: Optional[int] = None,
) -> Tensor:
    """Compute the confusion matrix.

    Args:
        preds: Predicted class indices, shape `(N,)`.
        target: Ground truth class indices, shape `(N,)`.
        num_classes: Total number of classes.
        ignore_index: Class index to exclude from computation.

    Returns:
        Confusion matrix of shape `(num_classes, num_classes)` where
        `cm[i, j]` is the number of points with true class `i`
        predicted as class `j`.
    """
    if ignore_index is not None:
        mask = target != ignore_index
        preds = preds[mask]
        target = target[mask]

    indices = (target.long() * num_classes + preds.long()).view(-1)
    flat = torch.bincount(indices, minlength=num_classes * num_classes)
    return flat.view(num_classes, num_classes)


def compute_intersection_union(
    preds: Tensor,
    target: Tensor,
    num_classes: int,
    batch: Optional[Tensor] = None,
    ignore_index: Optional[int] = None,
) -> tuple[Tensor, Tensor]:
    r"""Compute per-class intersection and union counts.

    Args:
        preds: Predicted class indices, shape $(N,)$.
        target: Ground truth class indices, shape $(N,)$.
        num_classes: Total number of classes.
        batch: Optional per-point batch index for per-sample counts. One row is emitted per sample
            (even for samples whose points are all ignored, which count as zero).
        ignore_index: Class index to exclude. Points where
            `target == ignore_index` are dropped, and the returned
            intersection/union at this index are $0$.

    Returns:
        Tuple $(\text{intersection}, \text{union})$, each of shape $(\text{num_classes},)$
        or $(\text{batch_size}, \text{num_classes})$ if `batch` is provided.
    """
    if batch is not None:
        batch = batch.long()
        batch_size = int(batch.max().item()) + 1 if batch.numel() else 0

    if ignore_index is not None:
        mask = target != ignore_index
        preds = preds[mask]
        target = target[mask]
        batch = batch[mask] if batch is not None else None

    preds = preds.long()
    target = target.long()
    correct = preds == target

    if batch is None:
        # Compute per-class intersection and union counts as (num_classes,) tensors
        inter = torch.bincount(target[correct], minlength=num_classes)
        area_pred = torch.bincount(preds, minlength=num_classes)
        area_target = torch.bincount(target, minlength=num_classes)
    else:
        # Compute per-class intersection and union counts as (batch_size, num_classes) tensors
        # such that it can be used to compute the mean IoU per batch (micro/macro IoU)
        flat = batch_size * num_classes
        preds_key = batch * num_classes + preds
        target_key = batch * num_classes + target
        inter = torch.bincount(target_key[correct], minlength=flat).view(batch_size, num_classes)
        area_pred = torch.bincount(preds_key, minlength=flat).view(batch_size, num_classes)
        area_target = torch.bincount(target_key, minlength=flat).view(batch_size, num_classes)

    union = area_pred + area_target - inter
    if ignore_index is not None and 0 <= ignore_index < num_classes:
        # NOTE: using ellipsis (...) to index the last dimension of the tensor, works for both 1D and 2D tensors
        inter[..., ignore_index] = 0
        union[..., ignore_index] = 0

    return inter, union


def compute_iou(
    preds: Tensor,
    target: Tensor,
    num_classes: int,
    batch: Optional[Tensor] = None,
    ignore_index: Optional[int] = None,
    default: float | Tensor = 0.0,
) -> Tensor:
    r"""Compute the Intersection over Union (IoU) for each class.

    Args:
        preds: Predicted class indices, shape $(N,)$.
        target: Ground truth class indices, shape $(N,)$.
        num_classes: Total number of classes.
        batch: Optional per-point batch index for per-sample IoU.
        ignore_index: Class index to exclude from computation.
            The returned IoU at this index will be $0$.
        default: Value returned for classes with zero union (avoids division by zero).

    Returns:
        Per-class IoU tensor of shape $(\text{num_classes},)$
        or $(\text{batch_size}, \text{num_classes})$ if `batch` is provided.
    """
    inter, union = compute_intersection_union(
        preds=preds,
        target=target,
        num_classes=num_classes,
        batch=batch,
        ignore_index=ignore_index,
    )

    return safe_divide(inter.float(), union.float(), default=default)


def compute_mean_iou(
    preds: Tensor,
    target: Tensor,
    num_classes: int,
    batch: Optional[Tensor] = None,
    ignore_index: Optional[int] = None,
) -> Tensor:
    r"""Compute the mean Intersection over Union (mIoU).

    Averages IoU over all classes except `ignore_index`; a class absent from the whole input
    (zero union) counts as IoU $0$, matching sklearn's `jaccard_score(zero_division=0)`. Toolboxes
    that average only over present classes (a nanmean over nonzero unions) report a higher value
    on splits missing a class, so compare published numbers accordingly. With `batch`, each sample
    is averaged only over the classes present in it (nonzero union), so a perfect prediction
    scores $1$ regardless of how many of the dataset's classes the sample contains.

    Args:
        preds: Predicted class indices, shape $(N,)$.
        target: Ground truth class indices, shape $(N,)$.
        num_classes: Total number of classes.
        batch: Optional per-point batch index for per-sample mIoU, averaged over each
            sample's present classes. A sample whose points are all ignored scores $0$.
        ignore_index: Class index to exclude from the mean.

    Returns:
        Scalar mIoU value or per-batch mIoU value if `batch` is provided.
    """
    inter, union = compute_intersection_union(
        preds=preds,
        target=target,
        num_classes=num_classes,
        batch=batch,
        ignore_index=ignore_index,
    )

    if ignore_index is not None and 0 <= ignore_index < num_classes:
        mask = torch.ones(num_classes, dtype=torch.bool, device=union.device)
        mask[ignore_index] = False
        inter = inter[..., mask]
        union = union[..., mask]

    iou = safe_divide(inter.float(), union.float(), default=0.0)
    if batch is None:
        return iou.mean(dim=-1)
    present = (union > 0).float()
    return safe_divide((iou * present).sum(dim=-1), present.sum(dim=-1), default=0.0)


def part_iou(
    preds: Tensor,
    target: Tensor,
    part_ids: Sequence[Sequence[int]],
    category: Tensor,
    batch: Tensor,
) -> Tensor:
    r"""Per-shape IoU averaged over the parts of the shape's category (the ShapeNetPart protocol).

    Each shape is scored only over the part labels its category owns (e.g. ShapeNetPart's `Airplane`
    owns parts $[0, 1, 2, 3]$); a part absent from both the prediction and the target counts as IoU $1$.

    Args:
        preds: Predicted part indices, shape $(N,)$.
        target: Ground truth part indices, shape $(N,)$.
        part_ids: Part labels owned by each category, e.g. `ShapeNetPart.seg_ids.values()`.
        category: Per-shape category index into `part_ids`, shape $(B,)$.
        batch: Per-point shape index, shape $(N,)$.

    Returns:
        Per-shape IoU tensor of shape $(B,)$.
    """
    parts = [list(ids) for ids in part_ids]
    num_classes = max(max(ids) for ids in parts) + 1
    mask = torch.zeros(len(parts), num_classes, dtype=torch.float, device=preds.device)
    for c, ids in enumerate(parts):
        mask[c, ids] = 1.0

    inter, union = compute_intersection_union(preds, target, num_classes, batch=batch)
    iou = safe_divide(inter.float(), union.float(), default=1.0)
    shape_mask = mask[category.long()]
    return (iou * shape_mask).sum(dim=1) / shape_mask.sum(dim=1)


def part_mean_iou(
    preds: Tensor,
    target: Tensor,
    part_ids: Sequence[Sequence[int]],
    category: Tensor,
    batch: Tensor,
) -> Dict[str, float]:
    r"""ShapeNetPart instance and class mean IoU.

    `part_iou` scores each shape over its category's parts; the instance mIoU averages these per-shape
    IoUs over all shapes, and the class mIoU averages them per category first, then over the categories
    present in `category`.

    Args:
        preds: Predicted part indices, shape $(N,)$.
        target: Ground truth part indices, shape $(N,)$.
        part_ids: Part labels owned by each category, e.g. `ShapeNetPart.seg_ids.values()`.
        category: Per-shape category index into `part_ids`, shape $(B,)$.
        batch: Per-point shape index, shape $(N,)$.

    Returns:
        A dict `{"ins_mIoU": ..., "cls_mIoU": ...}`.

    Example:
        >>> part_ids = [[0, 1], [2, 3]]
        >>> preds = torch.tensor([0, 1, 2, 2])
        >>> target = torch.tensor([0, 1, 2, 3])
        >>> category = torch.tensor([0, 1])
        >>> batch = torch.tensor([0, 0, 1, 1])
        >>> part_mean_iou(preds, target, part_ids, category, batch)
        {'ins_mIoU': 0.625, 'cls_mIoU': 0.625}
    """
    ious = part_iou(preds, target, part_ids, category, batch)
    category = category.long()
    count = torch.bincount(category, minlength=len(part_ids))
    iou_sum = torch.zeros(len(part_ids), device=ious.device).index_add_(0, category, ious)
    present = count > 0
    cls_miou = (iou_sum[present] / count[present]).mean()
    return {"ins_mIoU": float(ious.mean()), "cls_mIoU": float(cls_miou)}


def overall_accuracy(
    preds: Tensor,
    target: Tensor,
    ignore_index: Optional[int] = None,
) -> float:
    """Compute the overall prediction accuracy.

    Args:
        preds: Predicted class indices, shape `(N,)`.
        target: Ground truth class indices, shape `(N,)`.
        ignore_index: Class index to exclude from computation.

    Returns:
        Scalar accuracy value, or `0.0` when no points remain after `ignore_index` masking.
    """
    if ignore_index is not None:
        mask = target != ignore_index
        preds = preds[mask]
        target = target[mask]

    if target.numel() == 0:
        return 0.0
    return preds.eq(target).float().mean().item()


def per_class_accuracy(
    preds: Tensor,
    target: Tensor,
    num_classes: int,
    ignore_index: Optional[int] = None,
    eps: float = 1e-10,
) -> Tensor:
    """Compute the accuracy for each class.

    Args:
        preds: Predicted class indices, shape `(N,)`.
        target: Ground truth class indices, shape `(N,)`.
        num_classes: Total number of classes.
        ignore_index: Class index to exclude. The returned accuracy
            at this index will be `0`.
        eps: Small constant to avoid division by zero.

    Returns:
        Per-class accuracy tensor of shape `(num_classes,)`.
    """
    cm = confusion_matrix(preds, target, num_classes, ignore_index=ignore_index)
    per_class = cm.diag().float() / (cm.sum(dim=1).float() + eps)

    if ignore_index is not None and 0 <= ignore_index < num_classes:
        per_class[ignore_index] = 0.0

    return per_class


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
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = max(mpre[i - 1], mpre[i])
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
    for i in range(num_samples - 1, 0, -1):
        sampled[i - 1] = max(sampled[i - 1], sampled[i])
    return float(sampled[::4].mean() if interpolation == "r11" else sampled[1:].mean())


def _average_precision3d(
    scene_preds: List[Tuple[np.ndarray, np.ndarray, np.ndarray]],
    scene_gts: List[Tuple[np.ndarray, np.ndarray]],
    label: int,
    iou_threshold: float,
    scene_ignore: Optional[List[Tuple[np.ndarray, np.ndarray]]] = None,
    scene_pred_ignore: Optional[List[np.ndarray]] = None,
    interpolation: Interpolation = "all",
) -> float:
    """Greedy VOC AP for one class: rank predictions by score, match each to an unused GT box by IoU.

    A prediction flagged in `scene_pred_ignore` is skipped outright: it can neither match a GT box nor
    count as a false positive. An unmatched prediction that overlaps an ignore region attributed to the
    evaluated class (`scene_ignore` rows whose label equals `label`) above the threshold is dropped
    (counted as neither a true nor a false positive).
    """
    gt_corners = [corners[labels == label] for corners, labels in scene_gts]
    matched = [np.zeros(len(c), dtype=bool) for c in gt_corners]
    npos = sum(len(c) for c in gt_corners)

    entries: List[Tuple[float, int, np.ndarray]] = []
    for scene, (corners, scores, labels) in enumerate(scene_preds):
        keep = labels == label
        if scene_pred_ignore is not None:
            keep = keep & ~scene_pred_ignore[scene]
        entries.extend((float(score), scene, corner) for corner, score in zip(corners[keep], scores[keep]))
    if not entries:
        return 0.0
    entries.sort(key=lambda e: -e[0])

    tp = np.zeros(len(entries))
    fp = np.zeros(len(entries))
    for d, (_, scene, corner) in enumerate(entries):
        gts = gt_corners[scene]
        if len(gts) > 0:
            iou = box3d_overlap(torch.from_numpy(corner)[None], torch.from_numpy(gts))[1][0].numpy()
            jmax = int(iou.argmax())
            if iou[jmax] > iou_threshold and not matched[scene][jmax]:
                tp[d] = 1.0
                matched[scene][jmax] = True
                continue

        if scene_ignore is not None:
            ignore_corners, ignore_labels = scene_ignore[scene]
            ignore_corners = ignore_corners[ignore_labels == label]
            if len(ignore_corners) > 0:
                iou_ignore = box3d_overlap(torch.from_numpy(corner)[None], torch.from_numpy(ignore_corners))[1][0]
                if float(iou_ignore.max()) > iou_threshold:
                    continue

        fp[d] = 1.0

    tp_cum, fp_cum = np.cumsum(tp), np.cumsum(fp)
    recall = tp_cum / max(npos, 1)
    precision = tp_cum / np.maximum(tp_cum + fp_cum, np.finfo(np.float64).eps)
    return _voc_ap(recall, precision, interpolation)


def _split_scenes(
    preds: Sequence[Detection3D], targets: Sequence[Boxes3D]
) -> Tuple[
    List[Tuple[np.ndarray, np.ndarray, np.ndarray]],
    List[Tuple[np.ndarray, np.ndarray]],
    List[Tuple[np.ndarray, np.ndarray]],
    List[np.ndarray],
]:
    """Flatten packed preds/targets into per-scene prediction, ground-truth, ignore-region and pred-ignore arrays.

    Target boxes flagged via the optional `ignore_mask` are split out as per-scene `(corners, labels)`
    ignore regions (excluded from the ground truth, used only to suppress false positives of the class
    their label attributes them to). The optional prediction-side `ignore_mask` becomes a per-scene
    boolean mask of predictions excluded from scoring.
    """

    def to_corners(boxes: Tensor) -> np.ndarray:
        return box_corners(boxes).detach().cpu().numpy() if boxes.numel() else np.zeros((0, 8, 3))

    def num_scenes(batch: Tensor) -> int:
        return int(batch.max().item()) + 1 if batch.numel() else 0

    def to_mask(mask: Optional[Tensor], length: int) -> np.ndarray:
        return mask.detach().cpu().numpy().astype(bool) if mask is not None else np.zeros(length, dtype=bool)

    scene_preds: List[Tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    scene_gts: List[Tuple[np.ndarray, np.ndarray]] = []
    scene_ignore: List[Tuple[np.ndarray, np.ndarray]] = []
    scene_pred_ignore: List[np.ndarray] = []
    for pred, target in zip(preds, targets):
        pred_corners, pred_batch = to_corners(pred["boxes"]), pred["batch"].detach().cpu().numpy()
        pred_scores, pred_labels = pred["scores"].detach().cpu().numpy(), pred["labels"].detach().cpu().numpy()
        pred_ignore = to_mask(pred.get("ignore_mask"), len(pred_labels))
        gt_corners, gt_batch = to_corners(target["boxes"]), target["batch"].detach().cpu().numpy()
        gt_labels = target["labels"].detach().cpu().numpy()
        ignore_mask = to_mask(target.get("ignore_mask"), len(gt_labels))
        for s in range(max(num_scenes(pred["batch"]), num_scenes(target["batch"]))):
            p = pred_batch == s
            g = (gt_batch == s) & ~ignore_mask
            ig = (gt_batch == s) & ignore_mask
            scene_preds.append((pred_corners[p], pred_scores[p], pred_labels[p]))
            scene_gts.append((gt_corners[g], gt_labels[g]))
            scene_ignore.append((gt_corners[ig], gt_labels[ig]))
            scene_pred_ignore.append(pred_ignore[p])
    return scene_preds, scene_gts, scene_ignore, scene_pred_ignore


def mean_average_precision3d(
    preds: Sequence[Detection3D],
    targets: Sequence[Boxes3D],
    *,
    iou_thresholds: Sequence[float] = (0.25, 0.5),
    interpolation: Interpolation = "all",
) -> Dict[str, float]:
    r"""3D detection mean average precision over one or more IoU thresholds (same IoU for every class).

    Dataset- and model-agnostic: predictions and targets are packed dicts of parameterized boxes (see
    `box_corners`) carrying a per-box scene index, so any detector emitting `(boxes, scores, labels, batch)`
    is scored the same way. `mAP@t` averages the per-class AP over the classes present in the targets.
    Targets may carry an `ignore_mask` (see `Boxes3D`); an ignore box's `labels` entry names the class it
    excuses, and unmatched predictions of that class overlapping it are not penalized. Predictions may
    carry an `ignore_mask` of their own; flagged predictions are excluded from scoring entirely (the
    KITTI min-height rule). Use `average_precision3d` for per-class IoU thresholds.

    Args:
        preds: Packed predictions (one `decode` output per batch), each
            `{"boxes": (N, 7), "scores": (N,), "labels": (N,), "batch": (N,)}`.
        targets: Packed ground truth aligned to `preds` batch-for-batch, each `{"boxes", "labels", "batch"}`.
        iou_thresholds: IoU thresholds at which `mAP@t` is reported.
        interpolation: AP interpolation: `"all"` integrates the full precision-recall curve; `"r11"` /
            `"r40"` sample the KITTI 11- / 40-point recall grids.

    Returns:
        A dict `{"mAP@0.25": ..., "mAP@0.5": ...}` keyed by threshold.
    """
    scene_preds, scene_gts, scene_ignore, scene_pred_ignore = _split_scenes(preds, targets)
    classes = sorted({int(c) for _, labels in scene_gts for c in labels.tolist()})
    out: Dict[str, float] = {}
    for threshold in iou_thresholds:
        aps = [
            _average_precision3d(scene_preds, scene_gts, c, threshold, scene_ignore, scene_pred_ignore, interpolation)
            for c in classes
        ]
        out[f"mAP@{threshold:g}"] = float(np.mean(aps)) if aps else 0.0
    return out


def average_precision3d(
    preds: Sequence[Detection3D],
    targets: Sequence[Boxes3D],
    *,
    iou_per_class: Mapping[int, float],
    class_names: Optional[Sequence[str]] = None,
    interpolation: Interpolation = "all",
) -> Dict[str, float]:
    r"""Per-class 3D AP, each class scored at its own IoU threshold (e.g. KITTI Car@0.7, Ped/Cyc@0.5).

    Like `mean_average_precision3d` but reports one AP per class at a class-specific IoU, the convention
    of the KITTI / nuScenes detection metrics. Targets may carry an `ignore_mask` (see `Boxes3D`): an
    ignore box's `labels` entry names the class it excuses (e.g. an ignored KITTI `Van` attributed to
    `Car`), and unmatched predictions of that class overlapping it are not counted as false positives.
    Predictions may carry an `ignore_mask` of their own; flagged predictions are excluded from scoring
    entirely (the KITTI min-height rule).

    Args:
        preds: Packed predictions aligned to `targets` batch-for-batch.
        targets: Packed ground truth, each `{"boxes", "labels", "batch"}` with an optional `ignore_mask`.
        iou_per_class: IoU threshold per class index, e.g. `{0: 0.7, 1: 0.5, 2: 0.5}`.
        class_names: Optional names for the output keys (indexed by class); falls back to the index.
        interpolation: AP interpolation: `"all"` integrates the full precision-recall curve; `"r11"` /
            `"r40"` sample the KITTI 11- / 40-point recall grids.

    Returns:
        A dict `{"AP/<class>": ap, ..., "mAP": mean}` (the mean is over `iou_per_class`).
    """
    scene_preds, scene_gts, scene_ignore, scene_pred_ignore = _split_scenes(preds, targets)
    out: Dict[str, float] = {}
    aps: List[float] = []
    for label, iou in iou_per_class.items():
        ap = _average_precision3d(
            scene_preds, scene_gts, int(label), float(iou), scene_ignore, scene_pred_ignore, interpolation
        )
        name = class_names[label] if class_names is not None else str(label)
        out[f"AP/{name}"] = ap
        aps.append(ap)
    out["mAP"] = float(np.mean(aps)) if aps else 0.0
    return out


_TP_KEYS = ("trans", "scale", "orient", "vel", "attr")


def _cummean(values: np.ndarray) -> np.ndarray:
    """Cumulative mean over the non-NaN entries; an all-NaN input yields the full-error sentinel of ones."""
    valid = ~np.isnan(values)
    if not valid.any():
        return np.ones(len(values))
    count = np.cumsum(valid)
    total = np.nancumsum(values)
    return np.divide(total, count, out=np.zeros_like(total), where=count > 0)


def _top_score_mask(scores: Tensor, batch: Tensor, max_boxes: int) -> Tensor:
    """Boolean mask keeping each sample's `max_boxes` highest-scoring entries."""
    order = torch.argsort(scores, descending=True, stable=True)
    grouped = order[torch.argsort(batch[order], stable=True)]
    counts = torch.bincount(batch)
    starts = torch.cumsum(counts, dim=0) - counts
    rank = torch.arange(batch.numel(), device=batch.device) - starts.repeat_interleave(counts)
    keep = torch.zeros(batch.numel(), dtype=torch.bool, device=batch.device)
    keep[grouped[rank < max_boxes]] = True
    return keep


def _nuscenes_accumulate(
    pred_boxes: np.ndarray,
    pred_scores: np.ndarray,
    pred_batch: np.ndarray,
    gt_boxes: np.ndarray,
    gt_batch: np.ndarray,
    pred_attributes: Optional[np.ndarray],
    gt_attributes: Optional[np.ndarray],
    dist_threshold: float,
    period: float,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    r"""Greedy center-distance matching for one class: 101-point precision, confidence and TP-error curves.

    Predictions in descending score order each take the closest not-yet-matched ground-truth box of their
    sample by BEV center distance, a match requiring a distance strictly below `dist_threshold`. The
    cumulative precision, confidence and per-match error curves are interpolated at the 101 recall points
    $0.00, 0.01, \ldots, 1.00$; without any match the sentinel curves are returned (zero precision and
    confidence, all errors $1$).
    """
    sentinel: Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]] = (
        np.zeros(101),
        np.zeros(101),
        {key: np.ones(101) for key in _TP_KEYS},
    )
    if len(gt_boxes) == 0 or len(pred_boxes) == 0:
        return sentinel
    has_velocity = pred_boxes.shape[1] >= 9 and gt_boxes.shape[1] >= 9

    order = np.argsort(-pred_scores, kind="stable")
    matched = np.zeros(len(gt_boxes), dtype=bool)
    tp = np.zeros(len(order))
    errors: Dict[str, List[float]] = {key: [] for key in _TP_KEYS}
    match_conf: List[float] = []
    for position, index in enumerate(order):
        candidates = np.flatnonzero((gt_batch == pred_batch[index]) & ~matched)
        if len(candidates) == 0:
            continue
        dist = np.linalg.norm(gt_boxes[candidates, :2] - pred_boxes[index, :2], axis=1)
        best = int(np.argmin(dist))
        if float(dist[best]) >= dist_threshold:
            continue
        gt, pred = gt_boxes[candidates[best]], pred_boxes[index]
        matched[candidates[best]] = True
        tp[position] = 1.0
        errors["trans"].append(float(dist[best]))
        intersection = float(np.prod(np.minimum(gt[3:6], pred[3:6])))
        union = float(np.prod(gt[3:6]) + np.prod(pred[3:6])) - intersection
        errors["scale"].append(1.0 - intersection / union)
        yaw = float(gt[6] - pred[6])
        errors["orient"].append(abs((yaw + period / 2) % period - period / 2))
        errors["vel"].append(float(np.linalg.norm(gt[7:9] - pred[7:9])) if has_velocity else float("nan"))
        if pred_attributes is None or gt_attributes is None or int(gt_attributes[candidates[best]]) < 0:
            errors["attr"].append(float("nan"))
        else:
            errors["attr"].append(float(int(gt_attributes[candidates[best]]) != int(pred_attributes[index])))
        match_conf.append(float(pred_scores[index]))
    if not match_conf:
        return sentinel

    grid = np.linspace(0.0, 1.0, 101)
    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(1.0 - tp)
    recall = tp_cum / len(gt_boxes)
    precision = np.interp(grid, recall, tp_cum / (tp_cum + fp_cum), right=0)
    confidence = np.interp(grid, recall, pred_scores[order], right=0)
    conf_match = np.array(match_conf)
    curves: Dict[str, np.ndarray] = {}
    for key, values in errors.items():
        cumulative = _cummean(np.array(values, dtype=np.float64))
        curves[key] = np.interp(confidence[::-1], conf_match[::-1], cumulative[::-1])[::-1]
    return precision, confidence, curves


def _nuscenes_ap(precision: np.ndarray, min_recall: float, min_precision: float) -> float:
    r"""AP over a 101-point interpolated precision curve: drop recalls $\le$ `min_recall`, subtract
    `min_precision`, clamp at $0$, average and rescale by the remaining precision span."""
    clipped = np.clip(precision[round(100 * min_recall) + 1 :] - min_precision, 0.0, None)
    return float(np.mean(clipped) / (1.0 - min_precision))


def _nuscenes_tp_error(curve: np.ndarray, confidence: np.ndarray, min_recall: float) -> float:
    """Mean of a 101-point TP-error curve from just above `min_recall` to the highest achieved recall
    (the last nonzero confidence); `1.0` when that recall does not exceed `min_recall`."""
    first = round(100 * min_recall) + 1
    nonzero = np.flatnonzero(confidence)
    last = int(nonzero[-1]) if len(nonzero) else 0
    if last < first:
        return 1.0
    return float(np.mean(curve[first : last + 1]))


def filter_boxes_by_range(boxes: Tensor, labels: Tensor, ranges: Sequence[float]) -> Tensor:
    r"""Mask of boxes whose BEV center distance from the sensor origin is strictly below their class range.

    Args:
        boxes: Boxes $(N, 7)$ or $(N, 9)$ of $(c_x, c_y, c_z, d_x, d_y, d_z, \theta[, v_x, v_y])$.
        labels: Per-box class index into `ranges`, shape $(N,)$.
        ranges: Maximum BEV range per class index, in the coordinate unit.

    Returns:
        Boolean keep mask of shape $(N,)$.

    Example:
        >>> boxes = torch.tensor([[3.0, 4, 0, 4, 2, 1.5, 0], [0, 41, 0, 0.5, 0.5, 1, 0]])
        >>> filter_boxes_by_range(boxes, torch.tensor([0, 1]), ranges=[50.0, 40.0])
        tensor([ True, False])
    """
    limits = boxes.new_tensor(list(ranges))[labels.long()]
    return torch.linalg.norm(boxes[:, :2], dim=1) < limits


def nuscenes_detection_metrics(
    pred_boxes: Tensor,
    pred_scores: Tensor,
    pred_labels: Tensor,
    pred_batch: Tensor,
    gt_boxes: Tensor,
    gt_labels: Tensor,
    gt_batch: Tensor,
    *,
    class_names: Sequence[str],
    gt_num_points: Optional[Tensor] = None,
    pred_attributes: Optional[Tensor] = None,
    gt_attributes: Optional[Tensor] = None,
    class_ranges: Optional[Mapping[str, float]] = None,
    dist_thresholds: Sequence[float] = (0.5, 1.0, 2.0, 4.0),
    tp_threshold: float = 2.0,
    max_boxes_per_sample: int = 500,
    min_recall: float = 0.1,
    min_precision: float = 0.1,
) -> Dict[str, float]:
    r"""The nuScenes detection metrics: per-class AP, mAP, the five TP errors and the NDS.

    Follows the official protocol of the nuScenes benchmark
    ([nuScenes: A Multimodal Dataset for Autonomous Driving](https://arxiv.org/abs/1903.11027)).
    Predictions are matched per sample and class by BEV center distance: in descending score order each
    prediction greedily takes the closest still-unmatched ground-truth box strictly below the threshold.
    AP interpolates precision at 101 recall points $0.00, 0.01, \ldots, 1.00$, drops recalls up to
    `min_recall`, subtracts `min_precision`, clamps at $0$, averages and rescales by the remaining
    precision span; `mAP` averages over `class_names` and `dist_thresholds`. The TP errors ATE (BEV
    center distance), ASE ($1 - $ IoU of center- and yaw-aligned boxes), AOE (absolute yaw difference,
    modulo $\pi$ for `barrier`), AVE (L2 xy-velocity difference) and AAE ($1 - $ attribute accuracy)
    average the cumulative-mean error curve of the `tp_threshold` matches from `min_recall` to the
    highest achieved recall; a class without matches scores the full error of $1$. The officially
    excluded pairs (`traffic_cone`: AOE/AVE/AAE, `barrier`: AVE/AAE) are left out of the per-metric
    means, and $\text{NDS} = (5 \cdot \text{mAP} + \sum_\text{tp} (1 - \min(1, \text{err}))) / 10$.

    Boxes are filtered before scoring: each sample keeps its `max_boxes_per_sample` highest-scoring
    predictions, boxes farther from the sensor origin (BEV) than their class range are dropped on both
    sides, and ground-truth boxes with `gt_num_points == 0` are removed. When velocity columns or
    attributes are absent (on either side), AVE / AAE fall back to the full penalty of $1.0$ per class.

    Args:
        pred_boxes: Predicted boxes $(M, 7)$ of $(c_x, c_y, c_z, d_x, d_y, d_z, \theta)$, or $(M, 9)$
            with $(v_x, v_y)$ velocity columns appended.
        pred_scores: Per-box confidence, shape $(M,)$.
        pred_labels: Per-box class index into `class_names`, shape $(M,)$.
        pred_batch: Per-box sample index, shape $(M,)$.
        gt_boxes: Ground-truth boxes $(K, 7)$ or $(K, 9)$, like `pred_boxes`.
        gt_labels: Per-box class index into `class_names`, shape $(K,)$.
        gt_batch: Per-box sample index, shape $(K,)$.
        class_names: Class name per label index; `barrier` and `traffic_cone` get their official special
            handling by name.
        gt_num_points: Optional per-box point count, shape $(K,)$; boxes with exactly $0$ points are
            removed (unknown counts of $-1$ are kept).
        pred_attributes: Optional per-box attribute id, shape $(M,)$. Without it AAE is $1.0$.
        gt_attributes: Optional per-box attribute id, shape $(K,)$; a negative id marks a box without an
            attribute, which is skipped in the AAE mean. Without it AAE is $1.0$.
        class_ranges: Maximum BEV evaluation range per class name; defaults to the official ranges (50 m
            car/truck/bus/trailer/construction_vehicle, 40 m pedestrian/motorcycle/bicycle, 30 m
            traffic_cone/barrier). A name missing from the mapping is not range-filtered.
        dist_thresholds: Matching thresholds in meters the AP is averaged over.
        tp_threshold: Matching threshold in meters of the TP-error metrics.
        max_boxes_per_sample: Per-sample cap on scored predictions (highest scores kept).
        min_recall: Recall up to which the AP and TP-error curves are clipped.
        min_precision: Precision subtracted before the AP mean.

    Returns:
        A flat dict with `AP/<class>` (averaged over `dist_thresholds`), `mAP`, `mATE`, `mASE`, `mAOE`,
        `mAVE`, `mAAE` and `NDS`.

    Example:
        >>> zero = torch.tensor([0])
        >>> pred_boxes = torch.tensor([[0.25, 0.0, 0.0, 4.0, 2.0, 1.5, 0.0]])
        >>> gt_boxes = torch.tensor([[0.0, 0.0, 0.0, 4.0, 2.0, 1.5, 0.0]])
        >>> metrics = nuscenes_detection_metrics(
        ...     pred_boxes, torch.tensor([0.9]), zero, zero, gt_boxes, zero, zero, class_names=["car"]
        ... )
        >>> f"{metrics['AP/car']:.2f} {metrics['mATE']:.2f} {metrics['NDS']:.3f}"
        '1.00 0.25 0.775'
    """
    if class_ranges is None:
        class_ranges = {
            "car": 50.0,
            "truck": 50.0,
            "bus": 50.0,
            "trailer": 50.0,
            "construction_vehicle": 50.0,
            "pedestrian": 40.0,
            "motorcycle": 40.0,
            "bicycle": 40.0,
            "traffic_cone": 30.0,
            "barrier": 30.0,
        }
    limits = [float(class_ranges.get(name, math.inf)) for name in class_names]

    pred_keep = _top_score_mask(pred_scores, pred_batch, max_boxes_per_sample)
    pred_keep &= filter_boxes_by_range(pred_boxes, pred_labels, limits)
    gt_keep = filter_boxes_by_range(gt_boxes, gt_labels, limits)
    if gt_num_points is not None:
        gt_keep &= gt_num_points != 0

    def to_numpy(tensor: Tensor) -> np.ndarray:
        return tensor.detach().cpu().numpy()

    pred_box, pred_score = to_numpy(pred_boxes[pred_keep]), to_numpy(pred_scores[pred_keep])
    pred_label, pred_sample = to_numpy(pred_labels[pred_keep]), to_numpy(pred_batch[pred_keep])
    gt_box, gt_label, gt_sample = to_numpy(gt_boxes[gt_keep]), to_numpy(gt_labels[gt_keep]), to_numpy(gt_batch[gt_keep])
    pred_attribute = to_numpy(pred_attributes[pred_keep]) if pred_attributes is not None else None
    gt_attribute = to_numpy(gt_attributes[gt_keep]) if gt_attributes is not None else None

    out: Dict[str, float] = {}
    aps: List[float] = []
    tp_errors: Dict[str, List[float]] = {key: [] for key in _TP_KEYS}
    for index, name in enumerate(class_names):
        pred_mask = pred_label == index
        gt_mask = gt_label == index
        period = math.pi if name == "barrier" else 2 * math.pi
        pred_attribute_cls = pred_attribute[pred_mask] if pred_attribute is not None else None
        gt_attribute_cls = gt_attribute[gt_mask] if gt_attribute is not None else None

        class_aps: List[float] = []
        for threshold in dist_thresholds:
            precision, _, _ = _nuscenes_accumulate(
                pred_box[pred_mask],
                pred_score[pred_mask],
                pred_sample[pred_mask],
                gt_box[gt_mask],
                gt_sample[gt_mask],
                pred_attribute_cls,
                gt_attribute_cls,
                float(threshold),
                period,
            )
            class_aps.append(_nuscenes_ap(precision, min_recall, min_precision))
        out[f"AP/{name}"] = float(np.mean(class_aps)) if class_aps else 0.0
        aps.extend(class_aps)

        _, confidence, error_curves = _nuscenes_accumulate(
            pred_box[pred_mask],
            pred_score[pred_mask],
            pred_sample[pred_mask],
            gt_box[gt_mask],
            gt_sample[gt_mask],
            pred_attribute_cls,
            gt_attribute_cls,
            tp_threshold,
            period,
        )
        excluded: Tuple[str, ...]
        if name == "traffic_cone":
            excluded = ("orient", "vel", "attr")
        elif name == "barrier":
            excluded = ("vel", "attr")
        else:
            excluded = ()
        for key in _TP_KEYS:
            if key not in excluded:
                tp_errors[key].append(_nuscenes_tp_error(error_curves[key], confidence, min_recall))

    out["mAP"] = float(np.mean(aps)) if aps else 0.0
    score_sum = 0.0
    for key, metric_name in zip(_TP_KEYS, ("mATE", "mASE", "mAOE", "mAVE", "mAAE")):
        error = float(np.mean(tp_errors[key])) if tp_errors[key] else 1.0
        out[metric_name] = error
        score_sum += max(0.0, 1.0 - error)
    out["NDS"] = (5.0 * out["mAP"] + score_sum) / 10.0
    return out


def nuscenes_velocity_attributes(
    labels: Tensor,
    velocity: Tensor,
    *,
    class_names: Sequence[str],
    speed_threshold: float = 1.0,
) -> Tensor:
    r"""Derive per-box nuScenes attribute ids from predicted velocities (the standard speed heuristic).

    A box moving faster than `speed_threshold` (BEV speed, m/s) gets its class's moving attribute, a
    slower box the parked / stopped / standing default; `barrier` and `traffic_cone` carry no attribute
    (id $-1$). The returned ids index the official 8-entry attribute table (`attribute.json` order), the
    id space of the `pred_attributes` / `gt_attributes` arguments of `nuscenes_detection_metrics`.

    Args:
        labels: Per-box class index into `class_names`, shape $(M,)$ long.
        velocity: Per-box BEV velocity $(v_x, v_y)$, shape $(M, 2)$.
        class_names: Class name per label index (the official 10 detection class names).
        speed_threshold: BEV speed in m/s above which a box counts as moving.

    Returns:
        Per-box attribute id, shape $(M,)$ long, $-1$ for classes without attributes.

    Shape:
        - labels: $(M,)$
        - velocity: $(M, 2)$
        - output: $(M,)$

    Example:
        >>> labels = torch.tensor([0, 0, 1])
        >>> velocity = torch.tensor([[3.0, 0.0], [0.5, 0.0], [2.0, 0.0]])
        >>> nuscenes_velocity_attributes(labels, velocity, class_names=("car", "barrier")).tolist()
        [0, 2, -1]
    """
    attribute_names = (
        "vehicle.moving",
        "vehicle.stopped",
        "vehicle.parked",
        "cycle.with_rider",
        "cycle.without_rider",
        "pedestrian.sitting_lying_down",
        "pedestrian.standing",
        "pedestrian.moving",
    )
    moving_attribute = {
        "car": "vehicle.moving",
        "truck": "vehicle.moving",
        "construction_vehicle": "vehicle.moving",
        "bus": "vehicle.moving",
        "trailer": "vehicle.moving",
        "motorcycle": "cycle.with_rider",
        "bicycle": "cycle.with_rider",
        "pedestrian": "pedestrian.moving",
    }
    stopped_attribute = {
        "car": "vehicle.parked",
        "truck": "vehicle.parked",
        "construction_vehicle": "vehicle.parked",
        "bus": "vehicle.stopped",
        "trailer": "vehicle.parked",
        "motorcycle": "cycle.without_rider",
        "bicycle": "cycle.without_rider",
        "pedestrian": "pedestrian.standing",
    }
    moving = torch.linalg.norm(velocity, dim=1) > speed_threshold
    attributes = torch.full_like(labels, -1)
    for index, name in enumerate(class_names):
        if name not in moving_attribute:
            continue

        mask = labels == index
        attributes[mask & moving] = attribute_names.index(moving_attribute[name])
        attributes[mask & ~moving] = attribute_names.index(stopped_attribute[name])

    return attributes


def instance_matches(
    pred_masks: Tensor,
    pred_labels: Tensor,
    pred_scores: Tensor,
    gt_instance: Tensor,
    gt_label: Tensor,
    *,
    ignore_index: int = -1,
) -> Dict[str, Tensor]:
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
        A dict of CPU tensors: `pred_labels`, `pred_scores`, `pred_counts`, `pred_void` (per prediction),
        `gt_labels`, `gt_counts` (per ground-truth instance, ordered by ascending id), and the nonzero
        same-class intersections as `pair_pred`, `pair_gt`, `pair_inter`.

    Example:
        >>> masks = torch.tensor([[True, True, True, True]])
        >>> match = instance_matches(
        ...     masks, torch.tensor([0]), torch.tensor([0.9]), torch.tensor([0, 0, 1, -1]), torch.tensor([0, 0, 0, -1])
        ... )
        >>> match["gt_counts"].tolist(), match["pair_inter"].tolist(), match["pred_void"].tolist()
        ([2, 1], [2, 1], [1])
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
    for index in range(pred_masks.shape[0]):
        hits = point_instance[pred_masks[index]]
        inter = torch.bincount(hits[hits >= 0], minlength=num_instances)
        gt_index = ((inter > 0) & (gt_labels == pred_labels[index])).nonzero(as_tuple=True)[0]
        pair_pred.append(torch.full_like(gt_index, index))
        pair_gt.append(gt_index)
        pair_inter.append(inter[gt_index])

    return {
        "pred_labels": pred_labels.detach().cpu(),
        "pred_scores": pred_scores.detach().cpu(),
        "pred_counts": pred_masks.sum(dim=1).cpu(),
        "pred_void": (pred_masks & void).sum(dim=1).cpu(),
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


def instance_average_precision(
    matches: Sequence[Mapping[str, Tensor]],
    *,
    num_classes: int,
    class_names: Optional[Sequence[str]] = None,
    min_points: int = 100,
) -> Dict[str, float]:
    r"""Point-mask instance-segmentation AP over per-scene `instance_matches` records.

    Follows the standard indoor instance-segmentation protocol (the ScanNet benchmark): per class and
    IoU threshold, ground-truth instances greedily consume overlapping predicted masks above the
    threshold, duplicates on a matched instance count as false positives with the lower score, and an
    unmatched prediction whose void / small-instance point fraction exceeds the threshold is excused.
    The AP integrates the score-swept precision-recall curve with centered recall steps. `mAP` averages
    per-class APs over the thresholds $0.5, 0.55, \ldots, 0.9$; `mAP@0.5` and `mAP@0.25` report the
    single-threshold values. Classes without any ground-truth instance are excluded from the means and
    get no `AP/<class>` entry.

    Args:
        matches: One `instance_matches` record per scene.
        num_classes: Number of instance classes.
        class_names: Optional names for the `AP/<class>` keys; falls back to the class index.
        min_points: Minimum point count for a prediction or ground-truth instance to be scored;
            smaller ground-truth instances count as ignore regions.

    Returns:
        A dict `{"AP/<class>": ap, ..., "mAP": ..., "mAP@0.5": ..., "mAP@0.25": ...}` where each
        `AP/<class>` is that class's AP averaged over the $0.5{:}0.05{:}0.9$ thresholds.

    Example:
        >>> masks = torch.tensor([[True, True, True, False], [False, False, False, True]])
        >>> match = instance_matches(
        ...     masks,
        ...     torch.tensor([0, 1]),
        ...     torch.tensor([0.9, 0.8]),
        ...     torch.tensor([0, 0, 0, 1]),
        ...     torch.tensor([0, 0, 0, 1]),
        ... )
        >>> out = instance_average_precision([match], num_classes=2, min_points=1)
        >>> out["mAP"], out["mAP@0.5"], out["mAP@0.25"]
        (1.0, 1.0, 1.0)
    """
    scenes = [{key: value.numpy() for key, value in match.items()} for match in matches]
    overlaps = np.append(np.arange(0.5, 0.95, 0.05), 0.25)
    ap = np.zeros((num_classes, len(overlaps)))
    for class_index in range(num_classes):
        for overlap_index, threshold in enumerate(overlaps):
            ap[class_index, overlap_index] = _instance_class_ap(scenes, class_index, float(threshold), min_points)

    strict = ap[:, ~np.isclose(overlaps, 0.25)]
    out: Dict[str, float] = {}
    for class_index in np.flatnonzero(~np.isnan(strict).any(axis=1)):
        name = class_names[class_index] if class_names is not None else str(class_index)
        out[f"AP/{name}"] = float(np.mean(strict[class_index]))
    if np.isnan(strict).all():
        return {"mAP": 0.0, "mAP@0.5": 0.0, "mAP@0.25": 0.0}
    out["mAP"] = float(np.nanmean(strict))
    out["mAP@0.5"] = float(np.nanmean(ap[:, np.isclose(overlaps, 0.5)]))
    out["mAP@0.25"] = float(np.nanmean(ap[:, np.isclose(overlaps, 0.25)]))
    return out
