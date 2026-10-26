r"""Mask losses: the dice loss and the pairwise mask costs of the set-prediction matchers."""

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
    r"""Dice loss between mask logits and binary targets, averaged over the masks (see `dice_loss`).

    Args:
        smooth: Smoothing term added to the numerator and the denominator.

    Shape:
        - input: $(\ldots, M)$ logits
        - target: $(\ldots, M)$
        - output: scalar
    """

    def __init__(self, smooth: float = 1.0) -> None:
        super().__init__()
        self.smooth = smooth

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return dice_loss(input, target, smooth=self.smooth).mean()


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
        >>> logits = torch.tensor([[0.0, 0.0], [10.0, -10.0]])
        >>> targets = torch.tensor([[1.0, 0.0]])
        >>> pairwise_binary_cross_entropy_with_logits(logits, targets).round(decimals=4).tolist()
        [[0.6931], [0.0]]

        ```
    """
    positive = F.binary_cross_entropy_with_logits(logits, logits.new_ones(()).expand_as(logits), reduction="none")
    negative = F.binary_cross_entropy_with_logits(logits, logits.new_zeros(()).expand_as(logits), reduction="none")
    return (positive @ targets.T + negative @ (1 - targets).T) / logits.shape[-1]
