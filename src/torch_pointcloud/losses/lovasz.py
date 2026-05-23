"""Lovász-Softmax loss."""

import torch
from torch import Tensor, nn


def _lovasz_grad(gt_sorted: Tensor) -> Tensor:
    """Gradient of the Lovász extension w.r.t. the error-sorted ground truth."""
    n = gt_sorted.numel()
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.cumsum(0)
    union = gts + (1.0 - gt_sorted).cumsum(0)
    jaccard = 1.0 - intersection / union
    if n > 1:
        jaccard[1:n] = jaccard[1:n] - jaccard[0 : n - 1]
    return jaccard


def _lovasz_softmax(probas: Tensor, labels: Tensor, classes: str, ignore_index: int) -> Tensor:
    r"""Multi-class Lovász-Softmax loss over packed (flat) per-point predictions.

    Args:
        probas: Per-point class probabilities, shape $(N, C)$.
        labels: Per-point ground-truth labels, shape $(N,)$.
        classes: `"present"` averages only classes present in `labels`; `"all"` averages every class.
        ignore_index: Label value excluded from the loss.

    Returns:
        The scalar Lovász-Softmax loss.
    """
    valid = labels != ignore_index
    probas, labels = probas[valid], labels[valid]
    if probas.numel() == 0:
        return probas.sum()
    losses = []
    for c in range(probas.size(1)):
        fg = (labels == c).to(probas.dtype)
        if classes == "present" and fg.sum() == 0:
            continue
        errors = (fg - probas[:, c]).abs()
        errors_sorted, perm = torch.sort(errors, dim=0, descending=True)
        losses.append(torch.dot(errors_sorted, _lovasz_grad(fg[perm])))
    return torch.stack(losses).mean()


class LovaszLoss(nn.Module):
    """Lovász-Softmax loss: a smooth surrogate for the mean-IoU objective.

    Optimizes segmentation overlap directly, and is typically summed with
    cross-entropy. See :arxiv: [The Lovász-Softmax loss](https://arxiv.org/abs/1705.08790).

    Args:
        ignore_index: Label value excluded from the loss.
        classes: `"present"` averages only classes present in the targets; `"all"` averages every class.
        loss_weight: Scalar multiplier applied to the loss.
    """

    def __init__(self, ignore_index: int = -1, classes: str = "present", loss_weight: float = 1.0) -> None:
        super().__init__()
        self.ignore_index = ignore_index
        self.classes = classes
        self.loss_weight = loss_weight

    def forward(self, logits: Tensor, labels: Tensor) -> Tensor:
        r"""Compute the loss from per-point logits $(N, C)$ and labels $(N,)$."""
        probas = logits.softmax(dim=1)
        return self.loss_weight * _lovasz_softmax(probas, labels, self.classes, self.ignore_index)
