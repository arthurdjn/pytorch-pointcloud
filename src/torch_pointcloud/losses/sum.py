"""Composite loss: sum of several `(logits, target)` losses."""

from typing import Sequence

import torch
from torch import Tensor, nn


class SumLoss(nn.Module):
    """Sum of several loss modules sharing a `(logits, target)` signature.

    Each sub-loss is evaluated on the same inputs and the results are added (e.g. cross-entropy plus
    Lovász). Apply per-loss weights through each module's own option (e.g. `LovaszLoss(loss_weight=...)`).

    Args:
        losses: The loss modules to sum.
    """

    def __init__(self, losses: Sequence[nn.Module]) -> None:
        super().__init__()
        self.losses = nn.ModuleList(losses)

    def forward(self, logits: Tensor, target: Tensor) -> Tensor:
        """Compute the summed loss over all sub-losses."""
        return torch.stack([loss(logits, target) for loss in self.losses]).sum()
