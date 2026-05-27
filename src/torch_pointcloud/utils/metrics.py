from typing import Optional

import torch
from torch import Tensor

from .ops import safe_divide


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
