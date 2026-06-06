from typing import Dict, List, Optional, Sequence, Tuple

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
) -> float:
    """Greedy VOC AP for one class: rank predictions by score, match each to an unused GT box by IoU."""
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
        if len(gts) == 0:
            fp[d] = 1.0
            continue
        iou = box3d_overlap(torch.from_numpy(corner)[None], torch.from_numpy(gts))[1][0].numpy()
        jmax = int(iou.argmax())
        if iou[jmax] > iou_threshold and not matched[scene][jmax]:
            tp[d] = 1.0
            matched[scene][jmax] = True
        else:
            fp[d] = 1.0

    tp_cum, fp_cum = np.cumsum(tp), np.cumsum(fp)
    recall = tp_cum / max(npos, 1)
    precision = tp_cum / np.maximum(tp_cum + fp_cum, np.finfo(np.float64).eps)
    return _voc_ap(recall, precision)


def mean_average_precision3d(
    preds: Sequence[Detection3D],
    targets: Sequence[Boxes3D],
    *,
    iou_thresholds: Sequence[float] = (0.25, 0.5),
) -> Dict[str, float]:
    r"""3D detection mean average precision over one or more IoU thresholds.

    Dataset- and model-agnostic: predictions and targets are packed dicts of parameterized boxes (see
    `box_corners`) carrying a per-box scene index, so any detector emitting `(boxes, scores, labels, batch)`
    is scored the same way. AP per class is the VOC all-points integral; `mAP@t` averages it over the classes present in the targets.

    Args:
        preds: Packed predictions (one `decode` output per batch), each
            `{"boxes": (N, 7), "scores": (N,), "labels": (N,), "batch": (N,)}`.
        targets: Packed ground truth aligned to `preds` batch-for-batch, each `{"boxes", "labels", "batch"}`.
        iou_thresholds: IoU thresholds at which `mAP@t` is reported.

    Returns:
        A dict `{"mAP@0.25": ..., "mAP@0.5": ...}` keyed by threshold.
    """

    def to_corners(boxes: Tensor) -> np.ndarray:
        return box_corners(boxes).detach().cpu().numpy() if boxes.numel() else np.zeros((0, 8, 3))

    def num_scenes(batch: Tensor) -> int:
        return int(batch.max().item()) + 1 if batch.numel() else 0

    scene_preds: List[Tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    scene_gts: List[Tuple[np.ndarray, np.ndarray]] = []
    for pred, target in zip(preds, targets):
        pred_corners, pred_batch = to_corners(pred["boxes"]), pred["batch"].detach().cpu().numpy()
        pred_scores, pred_labels = pred["scores"].detach().cpu().numpy(), pred["labels"].detach().cpu().numpy()
        gt_corners, gt_batch = to_corners(target["boxes"]), target["batch"].detach().cpu().numpy()
        gt_labels = target["labels"].detach().cpu().numpy()
        for s in range(max(num_scenes(pred["batch"]), num_scenes(target["batch"]))):
            p, g = pred_batch == s, gt_batch == s
            scene_preds.append((pred_corners[p], pred_scores[p], pred_labels[p]))
            scene_gts.append((gt_corners[g], gt_labels[g]))

    classes = sorted({int(c) for _, labels in scene_gts for c in labels.tolist()})
    out: Dict[str, float] = {}
    for threshold in iou_thresholds:
        aps = [_average_precision3d(scene_preds, scene_gts, c, threshold) for c in classes]
        out[f"mAP@{threshold:g}"] = float(np.mean(aps)) if aps else 0.0
    return out
