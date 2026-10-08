"""Lovász-Softmax loss."""

from typing import Literal

import torch
from torch import Tensor, nn

__all__ = ["LovaszLoss"]


# Elements sorted per call; bounds the memory of the class-batched sort.
_SORT_CHUNK_ELEMENTS = 2**24


def _lovasz_grad(gt_sorted: Tensor) -> Tensor:
    """Gradient of the Lovász extension w.r.t. the error-sorted ground truth, along the last dimension."""
    gts = gt_sorted.sum(dim=-1, keepdim=True)
    intersection = gts - gt_sorted.cumsum(dim=-1)
    union = gts + (1.0 - gt_sorted).cumsum(dim=-1)
    jaccard = 1.0 - intersection / union
    if gt_sorted.shape[-1] > 1:
        jaccard[..., 1:] = jaccard[..., 1:] - jaccard[..., :-1]
    return jaccard


def lovasz_softmax(
    probas: Tensor,
    labels: Tensor,
    *,
    classes: Literal["present", "all"] = "present",
    ignore_index: int = -1,
) -> Tensor:
    r"""Multi-class Lovász-Softmax loss over per-point class probabilities.

    The smooth surrogate of the mean-IoU objective of
    :arxiv: [The Lovász-Softmax loss: A tractable surrogate for the optimization of the intersection-over-union measure in neural networks](https://arxiv.org/abs/1705.08790) (Berman et al., 2018); the function form of
    [`LovaszLoss`][torch_pointcloud.losses.lovasz.LovaszLoss], which takes logits.

    Args:
        probas: Per-point class probabilities (post-softmax), shape $(N, C)$.
        labels: Per-point ground-truth labels, shape $(N,)$.
        classes: `"present"` averages only classes present in `labels`; `"all"` averages every class.
        ignore_index: Label value excluded from the loss.

    Returns:
        The scalar loss (zero when no point or no class contributes).

    Shape:
        - probas: $(N, C)$
        - labels: $(N,)$
        - output: scalar

    Example:
        ```pycon
        >>> labels = torch.tensor([0, 1, 2])
        >>> probas = torch.nn.functional.one_hot(labels, 3).float()
        >>> lovasz_softmax(probas, labels).item()
        0.0

        ```
    """
    if classes not in ("present", "all"):
        raise ValueError(f"`classes` must be 'present' or 'all', got {classes!r}.")

    valid = labels != ignore_index
    probas, labels = probas[valid], labels[valid]
    num_points, num_classes = probas.shape
    if num_points == 0:
        return probas.sum()

    # The classes that contribute: the ones present in the labels, found with a single synchronization.
    if classes == "present":
        present = torch.unique(labels)
        contributing = present[(present >= 0) & (present < num_classes)]
    else:
        contributing = torch.arange(num_classes, device=probas.device)
    if contributing.numel() == 0:
        # `classes="present"` with no label in $[0, C)$: no class contributes, so the loss is zero.
        return 0.0 * probas.sum()

    # Sort the errors of several classes per call (class-major, contiguous rows) and reduce with the
    # Lovász gradient of each class.
    chunk = max(1, _SORT_CHUNK_ELEMENTS // num_points)
    losses = []
    for start in range(0, contributing.numel(), chunk):
        cls = contributing[start : start + chunk]
        fg = (labels[None, :] == cls[:, None]).to(probas.dtype)  # (chunk, N)
        errors = (fg - probas[:, cls].t()).abs().contiguous()
        errors_sorted, perm = torch.sort(errors, dim=1, descending=True)
        fg_sorted = fg.gather(1, perm)
        losses.append((errors_sorted * _lovasz_grad(fg_sorted)).sum(dim=1))

    return torch.cat(losses).mean()


class LovaszLoss(nn.Module):
    """Lovász-Softmax loss: a smooth surrogate for the mean-IoU objective.

    Optimizes segmentation overlap directly, and is typically summed with
    cross-entropy. See :arxiv: [The Lovász-Softmax loss: A tractable surrogate for the optimization of the intersection-over-union measure in neural networks](https://arxiv.org/abs/1705.08790) (Berman et al., 2018). The function form
    on probabilities is [`lovasz_softmax`][torch_pointcloud.losses.lovasz.lovasz_softmax].

    Args:
        ignore_index: Label value excluded from the loss.
        classes: `"present"` averages only classes present in the targets; `"all"` averages every class.
    """

    def __init__(
        self,
        ignore_index: int = -1,
        classes: Literal["present", "all"] = "present",
    ) -> None:
        super().__init__()
        if classes not in ("present", "all"):
            raise ValueError(f"`classes` must be 'present' or 'all', got {classes!r}.")

        self.ignore_index = ignore_index
        self.classes = classes

    def forward(self, logits: Tensor, labels: Tensor) -> Tensor:
        r"""Compute the loss from per-point logits $(N, C)$ and labels $(N,)$."""
        probas = logits.softmax(dim=1)
        return lovasz_softmax(probas, labels, classes=self.classes, ignore_index=self.ignore_index)
