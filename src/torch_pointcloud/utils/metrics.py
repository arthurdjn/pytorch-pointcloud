from typing import Optional

import torch
from torch import Tensor


def confusion_matrix(
    preds: Tensor,
    target: Tensor,
    num_classes: int,
    ignore_index: Optional[int] = None,
) -> Tensor:
    """Compute the confusion matrix.

    Args:
        preds: Predicted class indices, shape ``(N,)``.
        target: Ground truth class indices, shape ``(N,)``.
        num_classes: Total number of classes.
        ignore_index: Class index to exclude from computation.

    Returns:
        Confusion matrix of shape ``(num_classes, num_classes)`` where
        ``cm[i, j]`` is the number of points with true class ``i``
        predicted as class ``j``.
    """
    if ignore_index is not None:
        mask = target != ignore_index
        preds = preds[mask]
        target = target[mask]

    cm = torch.zeros(num_classes, num_classes, dtype=torch.long, device=preds.device)
    indices = target * num_classes + preds
    cm.view(-1).scatter_add_(0, indices.long(), torch.ones_like(indices, dtype=torch.long))
    return cm


def iou_per_class(
    preds: Tensor,
    target: Tensor,
    num_classes: int,
    ignore_index: Optional[int] = None,
    eps: float = 1e-10,
) -> Tensor:
    """Compute the Intersection over Union (IoU) for each class.

    Args:
        preds: Predicted class indices, shape ``(N,)``.
        target: Ground truth class indices, shape ``(N,)``.
        num_classes: Total number of classes.
        ignore_index: Class index to exclude from computation.
            The returned IoU at this index will be ``0``.
        eps: Small constant to avoid division by zero.

    Returns:
        Per-class IoU tensor of shape ``(num_classes,)``.
    """
    cm = confusion_matrix(preds, target, num_classes, ignore_index=ignore_index)

    intersection = cm.diag().float()
    union = cm.sum(dim=1).float() + cm.sum(dim=0).float() - intersection
    iou = intersection / (union + eps)

    if ignore_index is not None and 0 <= ignore_index < num_classes:
        iou[ignore_index] = 0.0

    return iou


def mean_iou(
    preds: Tensor,
    target: Tensor,
    num_classes: int,
    ignore_index: Optional[int] = None,
    eps: float = 1e-10,
) -> float:
    """Compute the mean Intersection over Union (mIoU).

    Averages IoU over all classes except ``ignore_index``.

    Args:
        preds: Predicted class indices, shape ``(N,)``.
        target: Ground truth class indices, shape ``(N,)``.
        num_classes: Total number of classes.
        ignore_index: Class index to exclude from the mean.
        eps: Small constant to avoid division by zero.

    Returns:
        Scalar mIoU value.
    """
    iou = iou_per_class(preds, target, num_classes, ignore_index, eps)

    if ignore_index is not None and 0 <= ignore_index < num_classes:
        # Average over valid classes only
        valid = torch.ones(num_classes, dtype=torch.bool, device=iou.device)
        valid[ignore_index] = False
        return iou[valid].mean().item()

    return iou.mean().item()


def overall_accuracy(
    preds: Tensor,
    target: Tensor,
    ignore_index: Optional[int] = None,
) -> float:
    """Compute the overall prediction accuracy.

    Args:
        preds: Predicted class indices, shape ``(N,)``.
        target: Ground truth class indices, shape ``(N,)``.
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
        preds: Predicted class indices, shape ``(N,)``.
        target: Ground truth class indices, shape ``(N,)``.
        num_classes: Total number of classes.
        ignore_index: Class index to exclude. The returned accuracy
            at this index will be ``0``.
        eps: Small constant to avoid division by zero.

    Returns:
        Per-class accuracy tensor of shape ``(num_classes,)``.
    """
    cm = confusion_matrix(preds, target, num_classes, ignore_index=ignore_index)
    per_class = cm.diag().float() / (cm.sum(dim=1).float() + eps)

    if ignore_index is not None and 0 <= ignore_index < num_classes:
        per_class[ignore_index] = 0.0

    return per_class
