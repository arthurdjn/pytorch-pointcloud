"""Classification metrics of a confusion matrix: the matrix itself and the accuracy."""

from typing import Dict, Literal, Optional, Sequence, Tuple, Union, overload

import torch
from torch import Tensor

from torch_pointcloud.utils.ops import safe_divide


def confusion_matrix(
    preds: Tensor,
    target: Tensor,
    num_classes: int,
    ignore_index: Optional[int] = None,
) -> Tensor:
    r"""Compute the confusion matrix.

    Args:
        preds: Predicted class indices, shape $(N,)$.
        target: Ground truth class indices, shape $(N,)$.
        num_classes: Total number of classes.
        ignore_index: Class index to exclude from computation.

    Returns:
        Confusion matrix of shape $(\text{num\_classes}, \text{num\_classes})$ where
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


def _ignore_classes(cm: Tensor, ignore_index: Union[int, Sequence[int], None]) -> Tuple[Tensor, Tensor]:
    """Zero the confusion-matrix rows of the ignored true classes and return the mask of the classes kept."""
    keep = torch.ones(cm.shape[0], dtype=torch.bool, device=cm.device)
    indices = [] if ignore_index is None else [ignore_index] if isinstance(ignore_index, int) else list(ignore_index)
    for index in indices:
        if 0 <= index < cm.shape[0]:
            keep[index] = False
    return cm * keep[:, None], keep


@overload
def accuracy(
    cm: Tensor,
    *,
    average: Literal["micro", "macro"] = ...,
    ignore_index: Union[int, Sequence[int], None] = ...,
    zero_division: float = ...,
    class_names: Optional[Sequence[str]] = ...,
) -> float: ...


@overload
def accuracy(
    cm: Tensor,
    *,
    average: Literal["none"],
    ignore_index: Union[int, Sequence[int], None] = ...,
    zero_division: float = ...,
    class_names: None = ...,
) -> Tensor: ...


@overload
def accuracy(
    cm: Tensor,
    *,
    average: Literal["none"],
    ignore_index: Union[int, Sequence[int], None] = ...,
    zero_division: float = ...,
    class_names: Sequence[str],
) -> Dict[str, float]: ...


def accuracy(
    cm: Tensor,
    *,
    average: Literal["micro", "macro", "none"] = "micro",
    ignore_index: Union[int, Sequence[int], None] = None,
    zero_division: float = 0.0,
    class_names: Optional[Sequence[str]] = None,
) -> Union[float, Tensor, Dict[str, float]]:
    r"""Accuracy of a confusion matrix.

    Confusion matrices add up, so the matrix may describe one batch or the sum of `confusion_matrix` over a
    whole split.

    Args:
        cm: Confusion matrix with true classes as rows, shape $(C, C)$ (see `confusion_matrix`).
        average: `"micro"` returns the overall accuracy (the fraction of points on the diagonal); `"macro"`
            returns the mean class accuracy (the mean of the per-class recalls); `"none"` returns the per-class
            accuracy.
        ignore_index: Class index, or indices, to ignore: points whose true class is ignored are dropped, and
            the ignored classes are left out of the mean. Indices outside $[0, C)$ have no effect.
        zero_division: Accuracy given to a class without any point (and to an empty matrix with `"micro"`).
        class_names: Name of each class index; with `average="none"` the per-class accuracy comes back as a
            `{name: accuracy}` dict instead of a tensor.

    Returns:
        The accuracy as a float with `average="micro"` or `"macro"`, or the per-class accuracy, shape $(C,)$,
        with `average="none"` (a `{name: accuracy}` dict when `class_names` is given).

    Shape:
        - cm: $(C, C)$
        - output: scalar, or $(C,)$ with `average="none"`

    Example:
        ```pycon
        >>> cm = confusion_matrix(torch.tensor([0, 1, 1, 1]), torch.tensor([0, 1, 0, 1]), num_classes=2)
        >>> accuracy(cm), accuracy(cm, average="macro")
        (0.75, 0.75)
        >>> accuracy(cm, average="none")
        tensor([0.5000, 1.0000])
        >>> accuracy(cm, average="none", class_names=["wall", "floor"])
        {'wall': 0.5, 'floor': 1.0}

        ```
    """
    if class_names is not None and len(class_names) != cm.shape[0]:
        raise ValueError(f"Got {len(class_names)} `class_names` for a confusion matrix of {cm.shape[0]} classes.")

    cm, keep = _ignore_classes(cm, ignore_index)
    if average == "micro":
        overall = safe_divide(cm.diag().sum().float(), cm.sum().float(), default=zero_division)

        return overall.item()
    per_class = safe_divide(cm.diag().float(), cm.sum(dim=1).float(), default=zero_division)

    if average == "none":
        return per_class if class_names is None else dict(zip(class_names, per_class.tolist()))
    return per_class[keep].mean().item()
