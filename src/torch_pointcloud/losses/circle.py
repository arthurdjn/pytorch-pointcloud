r"""Circle loss over a distance matrix, with per-pair weights on the positive terms.

:arxiv: [Circle Loss: A Unified Perspective of Pair Similarity Optimization](https://arxiv.org/abs/2002.10857)
"""

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def weighted_circle_loss(
    distances: Tensor,
    positive_mask: Tensor,
    negative_mask: Tensor,
    positive_weights: Tensor,
    positive_margin: float,
    negative_margin: float,
    positive_optimal: float,
    negative_optimal: float,
    log_scale: float,
) -> Tensor:
    r"""Circle loss over a distance matrix, with per-pair weights on the positive terms.

    The positives of a row are pushed under `positive_margin` and the negatives beyond `negative_margin`, with the
    self-paced weights of the circle loss scaled by `positive_weights` on the positives; the row and column losses
    are averaged over the rows and columns that have both a positive and a negative.

    Args:
        distances: Distances between the elements of two sets, shape $(N, M)$.
        positive_mask: Positive pairs, shape $(N, M)$.
        negative_mask: Negative pairs, shape $(N, M)$.
        positive_weights: Weight of every positive pair, shape $(N, M)$.
        positive_margin: Distance under which a positive pair stops being penalized.
        negative_margin: Distance beyond which a negative pair stops being penalized.
        positive_optimal: Target distance of the positives in the self-paced weights.
        negative_optimal: Target distance of the negatives in the self-paced weights.
        log_scale: Temperature of the log-sum-exp.

    Returns:
        The scalar loss, zero when no row and no column has both a positive and a negative.

    Example:
        ```pycon
        >>> import torch
        >>> from torch_pointcloud.losses import weighted_circle_loss
        >>> distances = torch.tensor([[0.2, 1.0], [1.2, 0.3]])
        >>> positive = torch.eye(2, dtype=torch.bool)
        >>> loss = weighted_circle_loss(distances, positive, ~positive, positive.float(), 0.1, 1.4, 0.1, 1.4, 24.0)
        >>> loss.ndim
        0

        ```
    """
    positive_scales = distances - 1e5 * (~positive_mask).float()  # mask the non-positives
    positive_scales = (positive_scales - positive_optimal).clamp(min=0.0)
    positive_scales = positive_weights * positive_scales.detach()

    negative_scales = distances + 1e5 * (~negative_mask).float()  # mask the non-negatives
    negative_scales = (negative_optimal - negative_scales).clamp(min=0.0).detach()

    positive_logits = log_scale * (distances - positive_margin) * positive_scales
    negative_logits = log_scale * (negative_margin - distances) * negative_scales
    row_loss = F.softplus(torch.logsumexp(positive_logits, dim=-1) + torch.logsumexp(negative_logits, dim=-1))
    col_loss = F.softplus(torch.logsumexp(positive_logits, dim=-2) + torch.logsumexp(negative_logits, dim=-2))
    row_loss, col_loss = row_loss / log_scale, col_loss / log_scale

    row_mask = positive_mask.any(dim=-1) & negative_mask.any(dim=-1)
    col_mask = positive_mask.any(dim=-2) & negative_mask.any(dim=-2)
    zero = distances.sum() * 0.0
    row_term = row_loss[row_mask].mean() if bool(row_mask.any()) else zero
    col_term = col_loss[col_mask].mean() if bool(col_mask.any()) else zero
    return (row_term + col_term) / 2


class WeightedCircleLoss(nn.Module):
    r"""Module form of [`weighted_circle_loss`][torch_pointcloud.losses.circle.weighted_circle_loss].

    Args:
        positive_margin: Distance under which a positive pair stops being penalized.
        negative_margin: Distance beyond which a negative pair stops being penalized.
        positive_optimal: Target distance of the positives in the self-paced weights.
        negative_optimal: Target distance of the negatives in the self-paced weights.
        log_scale: Temperature of the log-sum-exp.
    """

    def __init__(
        self,
        positive_margin: float,
        negative_margin: float,
        positive_optimal: float,
        negative_optimal: float,
        log_scale: float,
    ) -> None:
        super().__init__()
        self.positive_margin = positive_margin
        self.negative_margin = negative_margin
        self.positive_optimal = positive_optimal
        self.negative_optimal = negative_optimal
        self.log_scale = log_scale

    def forward(
        self,
        distances: Tensor,
        positive_mask: Tensor,
        negative_mask: Tensor,
        positive_weights: Tensor,
    ) -> Tensor:
        r"""Compute the loss of the $(N, M)$ `distances` with its positive and negative masks and positive weights."""
        return weighted_circle_loss(
            distances,
            positive_mask,
            negative_mask,
            positive_weights,
            self.positive_margin,
            self.negative_margin,
            self.positive_optimal,
            self.negative_optimal,
            self.log_scale,
        )

    def extra_repr(self) -> str:
        return (
            f"positive_margin={self.positive_margin}, negative_margin={self.negative_margin}, "
            f"positive_optimal={self.positive_optimal}, negative_optimal={self.negative_optimal}, "
            f"log_scale={self.log_scale}"
        )
