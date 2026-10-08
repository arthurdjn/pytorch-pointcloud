"""Weighted sum of several `(logits, target)` losses."""

from typing import Optional, Sequence

import torch
from torch import Tensor, nn

__all__ = ["SumLoss"]


class SumLoss(nn.Module):
    """Weighted sum of several loss modules sharing a `(logits, target)` signature.

    Each sub-loss is evaluated on the same inputs and the results are added (e.g. cross-entropy plus
    Lovász), each scaled by its weight when `weights` is given.

    Args:
        losses: The loss modules to sum.
        weights: One scalar per loss; `None` adds them as they are.
    """

    def __init__(self, losses: Sequence[nn.Module], weights: Optional[Sequence[float]] = None) -> None:
        super().__init__()
        if weights is not None and len(weights) != len(losses):
            raise ValueError(f"`weights` lists {len(weights)} values for {len(losses)} losses.")

        self.losses = nn.ModuleList(losses)
        self.weights = None if weights is None else [float(weight) for weight in weights]

    def forward(self, logits: Tensor, target: Tensor) -> Tensor:
        """Compute the weighted sum of the sub-losses."""
        terms = [loss(logits, target) for loss in self.losses]
        if self.weights is not None:
            terms = [weight * term for weight, term in zip(self.weights, terms)]
        return torch.stack(terms).sum()

    def extra_repr(self) -> str:
        return "" if self.weights is None else f"weights={self.weights}"
