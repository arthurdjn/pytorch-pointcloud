"""Semantic- and part-segmentation metrics: intersection over union of a confusion matrix, or per sample."""

import math
from typing import Dict, Literal, Optional, Sequence, Union, overload

import torch
from torch import Tensor

from torch_pointcloud.metrics.classification import _ignore_classes
from torch_pointcloud.ops.math import safe_divide


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
        Tuple $(\text{intersection}, \text{union})$, each of shape $(\text{num\_classes},)$
        or $(\text{batch\_size}, \text{num\_classes})$ if `batch` is provided.
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


@overload
def intersection_over_union(
    cm: Tensor,
    *,
    average: Literal["macro"] = ...,
    ignore_index: Union[int, Sequence[int], None] = ...,
    zero_division: float = ...,
    class_names: Optional[Sequence[str]] = ...,
) -> float: ...


@overload
def intersection_over_union(
    cm: Tensor,
    *,
    average: Literal["none"],
    ignore_index: Union[int, Sequence[int], None] = ...,
    zero_division: float = ...,
    class_names: None = ...,
) -> Tensor: ...


@overload
def intersection_over_union(
    cm: Tensor,
    *,
    average: Literal["none"],
    ignore_index: Union[int, Sequence[int], None] = ...,
    zero_division: float = ...,
    class_names: Sequence[str],
) -> Dict[str, float]: ...


def intersection_over_union(
    cm: Tensor,
    *,
    average: Literal["macro", "none"] = "macro",
    ignore_index: Union[int, Sequence[int], None] = None,
    zero_division: float = 0.0,
    class_names: Optional[Sequence[str]] = None,
) -> Union[float, Tensor, Dict[str, float]]:
    r"""Intersection over Union (IoU, the Jaccard index) of a confusion matrix.

    Confusion matrices add up, so the matrix may describe one batch or the sum of `confusion_matrix` over a
    whole split. With `average="macro"` a class absent from the whole matrix (zero union) counts as
    `zero_division`, matching sklearn's `jaccard_score(zero_division=0)`; toolboxes that average only over
    present classes report a higher value on splits missing a class, so compare published numbers accordingly.

    Args:
        cm: Confusion matrix with true classes as rows, shape $(C, C)$ (see `confusion_matrix`).
        average: `"macro"` returns the mean IoU (mIoU) over the classes; `"none"` returns the per-class IoU.
        ignore_index: Class index, or indices, to ignore: points whose true class is ignored are dropped, and
            the ignored classes are left out of the mean. Indices outside $[0, C)$ have no effect.
        zero_division: IoU given to a class with zero union (and to the ignored classes with `average="none"`).
        class_names: Name of each class index; with `average="none"` the per-class IoU comes back as a
            `{name: iou}` dict instead of a tensor.

    Returns:
        The mean IoU as a float with `average="macro"`, or the per-class IoU, shape $(C,)$, with `average="none"`
        (a `{name: iou}` dict when `class_names` is given).

    Shape:
        - cm: $(C, C)$
        - output: scalar, or $(C,)$ with `average="none"`

    Example:
        ```pycon
        >>> cm = torch.tensor([[1, 1], [0, 1]])
        >>> intersection_over_union(cm)
        0.5
        >>> intersection_over_union(cm, average="none")
        tensor([0.5000, 0.5000])
        >>> intersection_over_union(cm, average="none", class_names=["wall", "floor"])
        {'wall': 0.5, 'floor': 0.5}

        ```
    """
    if class_names is not None and len(class_names) != cm.shape[0]:
        raise ValueError(f"Got {len(class_names)} `class_names` for a confusion matrix of {cm.shape[0]} classes.")

    cm, keep = _ignore_classes(cm, ignore_index)
    intersection = cm.diag()
    union = (cm.sum(dim=0) + cm.sum(dim=1) - intersection) * keep
    per_class = safe_divide(intersection.float(), union.float(), default=zero_division)

    if average == "none":
        return per_class if class_names is None else dict(zip(class_names, per_class.tolist()))
    return per_class[keep].mean().item()


def part_intersection_over_union(
    preds: Tensor,
    target: Tensor,
    part_ids: Sequence[Sequence[int]],
    category: Tensor,
    batch: Optional[Tensor] = None,
) -> Tensor:
    r"""Per-shape IoU averaged over the parts of the shape's category (the ShapeNetPart protocol).

    Each shape is scored only over the part labels its category owns (e.g. ShapeNetPart's `Airplane`
    owns parts $[0, 1, 2, 3]$); a part absent from both the prediction and the target counts as IoU $1$.

    Args:
        preds: Predicted part indices, shape $(N,)$.
        target: Ground truth part indices, shape $(N,)$.
        part_ids: Part labels owned by each category, e.g. `ShapeNetPart.seg_ids.values()`.
        category: Per-shape category index into `part_ids`, shape $(B,)$.
        batch: Optional per-point shape index, shape $(N,)$; when omitted, all the points belong to one shape.

    Returns:
        Per-shape IoU tensor of shape $(B,)$.

    Shape:
        - preds, target, batch: $(N,)$
        - category: $(B,)$
        - output: $(B,)$

    Example:
        ```pycon
        >>> preds, target = torch.tensor([0, 1, 1, 1]), torch.tensor([0, 1, 0, 1])
        >>> part_intersection_over_union(preds, target, part_ids=[[0, 1], [2, 3]], category=torch.tensor([0]))
        tensor([0.5833])

        ```
    """
    parts = [list(ids) for ids in part_ids]
    num_classes = max(max(ids) for ids in parts) + 1
    mask = torch.zeros(len(parts), num_classes, dtype=torch.float, device=preds.device)
    for c, ids in enumerate(parts):
        mask[c, ids] = 1.0

    if batch is None:
        batch = torch.zeros_like(target, dtype=torch.long)

    inter, union = compute_intersection_union(preds, target, num_classes, batch=batch)
    iou = safe_divide(inter.float(), union.float(), default=1.0)
    shape_mask = mask[category.long().reshape(-1)]
    return (iou * shape_mask).sum(dim=1) / shape_mask.sum(dim=1)


@overload
def part_mean_intersection_over_union(
    ious: Tensor,
    category: Tensor,
    *,
    average: Literal["micro", "macro"] = ...,
    num_classes: Optional[int] = ...,
    class_names: Optional[Sequence[str]] = ...,
) -> float: ...


@overload
def part_mean_intersection_over_union(
    ious: Tensor,
    category: Tensor,
    *,
    average: Literal["none"],
    num_classes: Optional[int] = ...,
    class_names: None = ...,
) -> Tensor: ...


@overload
def part_mean_intersection_over_union(
    ious: Tensor,
    category: Tensor,
    *,
    average: Literal["none"],
    num_classes: Optional[int] = ...,
    class_names: Sequence[str],
) -> Dict[str, float]: ...


def part_mean_intersection_over_union(
    ious: Tensor,
    category: Tensor,
    *,
    average: Literal["micro", "macro", "none"] = "micro",
    num_classes: Optional[int] = None,
    class_names: Optional[Sequence[str]] = None,
) -> Union[float, Tensor, Dict[str, float]]:
    r"""Mean IoU of per-shape IoUs (the ShapeNetPart instance and class mIoU).

    A shape's IoU does not depend on the other shapes of its batch, so the IoUs may come from one batch or be the
    `part_intersection_over_union` of every batch of a split, concatenated.

    Args:
        ious: Per-shape IoUs (see `part_intersection_over_union`), shape $(B,)$.
        category: Per-shape category index, shape $(B,)$.
        average: `"micro"` returns the instance mIoU (the mean over all shapes); `"macro"` returns the class mIoU
            (the mean, over the categories that have a shape, of their per-category means); `"none"` returns the
            per-category mean IoU.
        num_classes: Number of categories, i.e. the length of the `average="none"` output; defaults to the number of
            `class_names`, else to the largest category index met plus one.
        class_names: Name of each category index; with `average="none"` the per-category mean IoU comes back as a
            `{name: iou}` dict instead of a tensor.

    Returns:
        The mIoU as a float with `average="micro"` or `"macro"` (NaN without any shape), or the per-category mean
        IoU, shape $(C,)$, with `average="none"` (a `{name: iou}` dict when `class_names` is given), holding NaN
        for the categories without any shape.

    Shape:
        - ious: $(B,)$
        - category: $(B,)$
        - output: scalar, or $(C,)$ with `average="none"`

    Example:
        ```pycon
        >>> ious, category = torch.tensor([1.0, 0.5, 0.0]), torch.tensor([0, 0, 1])
        >>> part_mean_intersection_over_union(ious, category)
        0.5
        >>> part_mean_intersection_over_union(ious, category, average="macro")
        0.375
        >>> part_mean_intersection_over_union(ious, category, average="none", class_names=["Airplane", "Bag"])
        {'Airplane': 0.75, 'Bag': 0.0}

        ```
    """
    if class_names is not None and num_classes not in (None, len(class_names)):
        raise ValueError(f"Got {len(class_names)} `class_names` for `num_classes={num_classes}`.")
    if num_classes is None and class_names is not None:
        num_classes = len(class_names)

    if average == "micro":
        return float(ious.mean())

    category = category.long()
    count = torch.bincount(category, minlength=num_classes or 0)
    iou_sum = torch.zeros(count.numel(), device=ious.device).index_add_(0, category, ious)
    present = count > 0
    if average == "macro":
        return float((iou_sum[present] / count[present]).mean())

    per_class = torch.full_like(iou_sum, math.nan)
    per_class[present] = iou_sum[present] / count[present]
    return per_class if class_names is None else dict(zip(class_names, per_class.tolist()))
