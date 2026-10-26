r"""Mask losses: the dice loss and the pairwise mask costs of the set-prediction matchers."""

from typing import Literal

import torch.nn.functional as F
from torch import Tensor, nn

__all__ = ["DiceLoss"]


def dice_loss(logits: Tensor, targets: Tensor, *, smooth: float = 1.0) -> Tensor:
    r"""Dice loss between mask logits and binary targets, one value per mask (no reduction).

    $1 - (2 \sum p t + s) / (\sum p + \sum t + s)$ over the last dimension, with $p$ the sigmoid of the logits
    and $s$ the smoothing term, as in :arxiv: [V-Net](https://arxiv.org/abs/1606.04797) (Milletari et al., 2016).

    Args:
        logits: Mask logits, shape $(\ldots, M)$.
        targets: Binary (or soft) targets of the same shape.
        smooth: Smoothing term $s$ added to the numerator and the denominator.

    Returns:
        The loss of every mask, shape $(\ldots)$.

    Shape:
        - logits: $(\ldots, M)$
        - targets: $(\ldots, M)$
        - output: $(\ldots)$

    Example:
        ```pycon
        >>> import torch
        >>> logits = torch.tensor([[10.0, 10.0, -10.0], [-10.0, -10.0, 10.0]])
        >>> targets = torch.tensor([[1.0, 1.0, 0.0], [1.0, 1.0, 0.0]])
        >>> dice_loss(logits, targets).round(decimals=2).tolist()
        [0.0, 0.75]

        ```
    """
    prob = logits.sigmoid()
    numerator = 2 * (prob * targets).sum(dim=-1)
    denominator = prob.sum(dim=-1) + targets.sum(dim=-1)
    return 1 - (numerator + smooth) / (denominator + smooth)


class DiceLoss(nn.Module):
    r"""Module form of [`dice_loss`][torch_pointcloud.losses.mask.dice_loss], reduced over the masks.

    Args:
        smooth: Smoothing term added to the numerator and the denominator.
        reduction: `"none"` returns the per-mask loss, `"sum"` its sum, `"mean"` its mean.

    Shape:
        - input: $(\ldots, M)$ logits
        - target: $(\ldots, M)$
        - output: $(\ldots)$ with `reduction="none"`, else a scalar

    Example:
        ```pycon
        >>> import torch
        >>> logits = torch.tensor([[10.0, 10.0, -10.0], [-10.0, -10.0, 10.0]])
        >>> targets = torch.tensor([[1.0, 1.0, 0.0], [1.0, 1.0, 0.0]])
        >>> DiceLoss(reduction="sum")(logits, targets).round(decimals=2).item()
        0.75

        ```
    """

    def __init__(self, smooth: float = 1.0, reduction: Literal["none", "sum", "mean"] = "mean") -> None:
        super().__init__()
        if reduction not in ("none", "sum", "mean"):
            raise ValueError(f"`reduction` must be 'none', 'sum' or 'mean', got {reduction!r}.")

        self.smooth = smooth
        self.reduction = reduction

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        loss = dice_loss(input, target, smooth=self.smooth)
        if self.reduction == "sum":
            loss = loss.sum()
        elif self.reduction == "mean":
            loss = loss.mean()
        return loss

    def extra_repr(self) -> str:
        return f"smooth={self.smooth}, reduction={self.reduction!r}"


def pairwise_dice_loss(logits: Tensor, targets: Tensor, *, smooth: float = 1.0) -> Tensor:
    r"""Dice loss of every (predicted mask, target mask) pair, the mask cost of a Hungarian matcher.

    Entry $(q, i)$ equals `dice_loss(logits[q], targets[i])`.

    Args:
        logits: Predicted mask logits, shape $(Q, M)$.
        targets: Binary (or soft) target masks, shape $(I, M)$.
        smooth: Smoothing term added to the numerator and the denominator.

    Returns:
        The pairwise losses, shape $(Q, I)$.

    Shape:
        - logits: $(Q, M)$
        - targets: $(I, M)$
        - output: $(Q, I)$

    Example:
        ```pycon
        >>> import torch
        >>> logits = torch.tensor([[10.0, 10.0, -10.0], [-10.0, -10.0, 10.0]])
        >>> targets = torch.tensor([[1.0, 1.0, 0.0]])
        >>> pairwise_dice_loss(logits, targets).round(decimals=2).tolist()
        [[0.0], [0.75]]

        ```
    """
    prob = logits.sigmoid()
    numerator = 2 * prob @ targets.T
    denominator = prob.sum(dim=-1)[:, None] + targets.sum(dim=-1)[None, :]
    return 1 - (numerator + smooth) / (denominator + smooth)


def pairwise_binary_cross_entropy_with_logits(logits: Tensor, targets: Tensor) -> Tensor:
    r"""Mean binary cross-entropy of every (predicted mask, target mask) pair, the mask cost of a Hungarian matcher.

    Entry $(q, i)$ equals `F.binary_cross_entropy_with_logits(logits[q], targets[i])`, computed for all pairs at once
    with two matrix products.

    Args:
        logits: Predicted mask logits, shape $(Q, M)$.
        targets: Binary (or soft) target masks, shape $(I, M)$.

    Returns:
        The pairwise losses averaged over the $M$ elements, shape $(Q, I)$.

    Shape:
        - logits: $(Q, M)$
        - targets: $(I, M)$
        - output: $(Q, I)$

    Example:
        ```pycon
        >>> import torch
        >>> logits = torch.tensor([[0.0, 0.0], [10.0, -10.0]])
        >>> targets = torch.tensor([[1.0, 0.0]])
        >>> pairwise_binary_cross_entropy_with_logits(logits, targets).round(decimals=4)
        tensor([[0.6931],
                [0.0000]])

        ```
    """
    positive = F.binary_cross_entropy_with_logits(logits, logits.new_ones(()).expand_as(logits), reduction="none")
    negative = F.binary_cross_entropy_with_logits(logits, logits.new_zeros(()).expand_as(logits), reduction="none")
    return (positive @ targets.T + negative @ (1 - targets).T) / logits.shape[-1]


def mask_matching_cost(
    class_prob: Tensor,
    mask_logits: Tensor,
    target_masks: Tensor,
    target_labels: Tensor,
    *,
    class_weight: float = 1.0,
    mask_weight: float = 1.0,
    dice_weight: float = 1.0,
) -> Tensor:
    r"""Cost of matching every predicted mask to every target instance, the input of a Hungarian assignment.

    Entry $(q, i)$ is the weighted sum of the negative probability that query $q$ predicts the class of instance $i$,
    the mask binary cross-entropy and the mask dice loss between them (`pairwise_binary_cross_entropy_with_logits`
    and `pairwise_dice_loss`).

    Args:
        class_prob: Per-query class probabilities, shape $(Q, C)$.
        mask_logits: Per-query mask logits, shape $(Q, M)$.
        target_masks: Binary (or soft) target masks, shape $(I, M)$.
        target_labels: Target classes in $[0, C)$, shape $(I,)$.
        class_weight: Weight of the class probability term.
        mask_weight: Weight of the binary cross-entropy term.
        dice_weight: Weight of the dice term.

    Returns:
        The pairwise costs, shape $(Q, I)$.

    Shape:
        - class_prob: $(Q, C)$
        - mask_logits: $(Q, M)$
        - target_masks: $(I, M)$
        - target_labels: $(I,)$
        - output: $(Q, I)$

    Example:
        ```pycon
        >>> import torch
        >>> class_prob = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        >>> mask_logits = torch.tensor([[10.0, -10.0], [-10.0, 10.0]])
        >>> cost = mask_matching_cost(class_prob, mask_logits, torch.tensor([[0.0, 1.0]]), torch.tensor([1]))
        >>> cost.argmin(dim=0).tolist()
        [1]

        ```
    """
    return (
        -class_weight * class_prob[:, target_labels]
        + mask_weight * pairwise_binary_cross_entropy_with_logits(mask_logits, target_masks)
        + dice_weight * pairwise_dice_loss(mask_logits, target_masks)
    )
