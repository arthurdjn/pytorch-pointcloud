from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import Tensor

from .box3d import box3d_overlap, box_corners
from .ops import safe_divide
from .types import Boxes3D, Detection3D


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
        ignore_index: Class index to exclude. Points where
            `target == ignore_index` are dropped, and the returned
            intersection/union at this index are $0$.

    Returns:
        Tuple $(\text{intersection}, \text{union})$, each of shape $(\text{num_classes},)$.
    """

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
        batch = batch.long()
        batch_size = int(batch.max().item()) + 1
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
    """Compute the mean Intersection over Union (mIoU).

    Averages IoU over all classes except `ignore_index`.

    Args:
        preds: Predicted class indices, shape $(N,)$.
        target: Ground truth class indices, shape $(N,)$.
        num_classes: Total number of classes.
        batch: Optional per-point batch index for per-sample mIoU.
        ignore_index: Class index to exclude from the mean.

    Returns:
        Scalar mIoU value or per-batch mIoU value if `batch` is provided.
    """
    iou = compute_iou(
        preds=preds,
        target=target,
        num_classes=num_classes,
        batch=batch,
        ignore_index=ignore_index,
    )

    if ignore_index is not None and 0 <= ignore_index < num_classes:
        mask = torch.ones(num_classes, dtype=torch.bool, device=iou.device)
        mask[ignore_index] = False
        iou = iou[..., mask]

    return iou.mean(dim=-1)


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
        Scalar accuracy value.
    """
    if ignore_index is not None:
        mask = target != ignore_index
        preds = preds[mask]
        target = target[mask]

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


def _voc_ap(recall: np.ndarray, precision: np.ndarray) -> float:
    """VOC average precision (the all-points / area-under-precision-recall variant)."""
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def _average_precision3d(
    scene_preds: List[Tuple[np.ndarray, np.ndarray, np.ndarray]],
    scene_gts: List[Tuple[np.ndarray, np.ndarray]],
    label: int,
    iou_threshold: float,
    scene_ignore: Optional[List[np.ndarray]] = None,
) -> float:
    """Greedy VOC AP for one class: rank predictions by score, match each to an unused GT box by IoU.

    A prediction that matches no GT but overlaps an ignore region (`scene_ignore`) above the
    threshold is dropped (counted as neither a true nor a false positive).
    """
    gt_corners = [corners[labels == label] for corners, labels in scene_gts]
    matched = [np.zeros(len(c), dtype=bool) for c in gt_corners]
    npos = sum(len(c) for c in gt_corners)

    entries = [
        (float(score), scene, corner)
        for scene, (corners, scores, labels) in enumerate(scene_preds)
        for corner, score in zip(corners[labels == label], scores[labels == label])
    ]
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

        ignore = scene_ignore[scene] if scene_ignore is not None else None
        if ignore is not None and len(ignore) > 0:
            iou_ignore = box3d_overlap(torch.from_numpy(corner)[None], torch.from_numpy(ignore))[1][0].numpy()
            if iou_ignore.max() > iou_threshold:
                continue

        fp[d] = 1.0

    tp_cum, fp_cum = np.cumsum(tp), np.cumsum(fp)
    recall = tp_cum / max(npos, 1)
    precision = tp_cum / np.maximum(tp_cum + fp_cum, np.finfo(np.float64).eps)
    return _voc_ap(recall, precision)


def _split_scenes(
    preds: Sequence[Detection3D], targets: Sequence[Boxes3D]
) -> Tuple[List[Tuple[np.ndarray, np.ndarray, np.ndarray]], List[Tuple[np.ndarray, np.ndarray]], List[np.ndarray]]:
    """Flatten packed preds/targets into per-scene `(corners, scores, labels)`, `(corners, labels)` and ignore corners.

    Target boxes flagged via the optional `ignore_mask` are split out as per-scene ignore regions
    (excluded from the ground truth, used only to suppress false positives).
    """

    def to_corners(boxes: Tensor) -> np.ndarray:
        return box_corners(boxes).detach().cpu().numpy() if boxes.numel() else np.zeros((0, 8, 3))

    def num_scenes(batch: Tensor) -> int:
        return int(batch.max().item()) + 1 if batch.numel() else 0

    scene_preds: List[Tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    scene_gts: List[Tuple[np.ndarray, np.ndarray]] = []
    scene_ignore: List[np.ndarray] = []
    for pred, target in zip(preds, targets):
        pred_corners, pred_batch = to_corners(pred["boxes"]), pred["batch"].detach().cpu().numpy()
        pred_scores, pred_labels = pred["scores"].detach().cpu().numpy(), pred["labels"].detach().cpu().numpy()
        gt_corners, gt_batch = to_corners(target["boxes"]), target["batch"].detach().cpu().numpy()
        gt_labels = target["labels"].detach().cpu().numpy()
        mask = target.get("ignore_mask")
        ignore_mask = (
            mask.detach().cpu().numpy().astype(bool) if mask is not None else np.zeros(len(gt_labels), dtype=bool)
        )
        for s in range(max(num_scenes(pred["batch"]), num_scenes(target["batch"]))):
            p = pred_batch == s
            g = (gt_batch == s) & ~ignore_mask
            ig = (gt_batch == s) & ignore_mask
            scene_preds.append((pred_corners[p], pred_scores[p], pred_labels[p]))
            scene_gts.append((gt_corners[g], gt_labels[g]))
            scene_ignore.append(gt_corners[ig])
    return scene_preds, scene_gts, scene_ignore


def mean_average_precision3d(
    preds: Sequence[Detection3D],
    targets: Sequence[Boxes3D],
    *,
    iou_thresholds: Sequence[float] = (0.25, 0.5),
) -> Dict[str, float]:
    r"""3D detection mean average precision over one or more IoU thresholds (same IoU for every class).

    Dataset- and model-agnostic: predictions and targets are packed dicts of parameterized boxes (see
    `box_corners`) carrying a per-box scene index, so any detector emitting `(boxes, scores, labels, batch)`
    is scored the same way. AP per class is the VOC all-points integral; `mAP@t` averages it over the
    classes present in the targets. Targets may carry an `ignore_mask` (see `Boxes3D`); predictions that
    only overlap ignore regions are not penalized. Use `average_precision3d` for per-class IoU thresholds.

    Args:
        preds: Packed predictions (one `decode` output per batch), each
            `{"boxes": (N, 7), "scores": (N,), "labels": (N,), "batch": (N,)}`.
        targets: Packed ground truth aligned to `preds` batch-for-batch, each `{"boxes", "labels", "batch"}`.
        iou_thresholds: IoU thresholds at which `mAP@t` is reported.

    Returns:
        A dict `{"mAP@0.25": ..., "mAP@0.5": ...}` keyed by threshold.
    """
    scene_preds, scene_gts, scene_ignore = _split_scenes(preds, targets)
    classes = sorted({int(c) for _, labels in scene_gts for c in labels.tolist()})
    out: Dict[str, float] = {}
    for threshold in iou_thresholds:
        aps = [_average_precision3d(scene_preds, scene_gts, c, threshold, scene_ignore) for c in classes]
        out[f"mAP@{threshold:g}"] = float(np.mean(aps)) if aps else 0.0
    return out


def average_precision3d(
    preds: Sequence[Detection3D],
    targets: Sequence[Boxes3D],
    *,
    iou_per_class: Mapping[int, float],
    class_names: Optional[Sequence[str]] = None,
) -> Dict[str, float]:
    r"""Per-class 3D AP, each class scored at its own IoU threshold (e.g. KITTI Car@0.7, Ped/Cyc@0.5).

    Like `mean_average_precision3d` but reports one AP per class at a class-specific IoU, the convention
    of the KITTI / nuScenes detection metrics. Targets may carry an `ignore_mask` (see `Boxes3D`):
    predictions overlapping an ignore region are not counted as false positives.

    Args:
        preds: Packed predictions aligned to `targets` batch-for-batch.
        targets: Packed ground truth, each `{"boxes", "labels", "batch"}` with an optional `ignore_mask`.
        iou_per_class: IoU threshold per class index, e.g. `{0: 0.7, 1: 0.5, 2: 0.5}`.
        class_names: Optional names for the output keys (indexed by class); falls back to the index.

    Returns:
        A dict `{"AP/<class>": ap, ..., "mAP": mean}` (the mean is over `iou_per_class`).
    """
    scene_preds, scene_gts, scene_ignore = _split_scenes(preds, targets)
    out: Dict[str, float] = {}
    aps: List[float] = []
    for label, iou in iou_per_class.items():
        ap = _average_precision3d(scene_preds, scene_gts, int(label), float(iou), scene_ignore)
        name = class_names[label] if class_names is not None else str(label)
        out[f"AP/{name}"] = ap
        aps.append(ap)
    out["mAP"] = float(np.mean(aps)) if aps else 0.0
    return out
