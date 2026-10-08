r"""Corner loss of the box-refinement heads (PointRCNN)."""

import math
from typing import Literal

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from torch_pointcloud.ops.box3d import box_corners

__all__ = ["CornerLoss"]


def corner_loss(pred_boxes: Tensor, gt_boxes: Tensor, *, beta: float = 1.0) -> Tensor:
    r"""Per-box mean smooth-$L_1$ distance between the eight corners of two box sets, robust to the heading flip.

    The corner regularization of :arxiv: [PointRCNN: 3D Object Proposal Generation and Detection from Point Cloud](https://arxiv.org/abs/1812.04244) (Shi et al., 2019): each predicted
    corner is compared with the matching corner of the ground-truth box and of that box rotated by
    $\pi$, and the closer of the two distances is penalized, so a heading off by half a turn is not
    punished as a position error.

    Args:
        pred_boxes: Predicted boxes $(c_x, c_y, c_z, d_x, d_y, d_z, \theta)$, shape $(N, 7)$.
        gt_boxes: Matched ground-truth boxes, shape $(N, 7)$.
        beta: Smooth-$L_1$ transition point.

    Returns:
        The per-box loss, shape $(N,)$.

    Shape:
        - pred_boxes: $(N, 7)$
        - gt_boxes: $(N, 7)$
        - output: $(N,)$

    Example:
        ```pycon
        >>> box = torch.tensor([[0.0, 0.0, 0.0, 4.0, 2.0, 1.5, 0.3]])
        >>> corner_loss(box, box).tolist()
        [0.0]
        >>> flipped = box.clone()
        >>> flipped[:, 6] += math.pi
        >>> corner_loss(box, flipped).abs().max().item() < 1e-5
        True

        ```
    """
    pred_corners = box_corners(pred_boxes)
    gt_corners = box_corners(gt_boxes)
    gt_flipped = gt_boxes.clone()
    gt_flipped[:, 6] = gt_flipped[:, 6] + math.pi
    gt_corners_flipped = box_corners(gt_flipped)
    dist = torch.minimum(
        torch.norm(pred_corners - gt_corners, dim=2),
        torch.norm(pred_corners - gt_corners_flipped, dim=2),
    )
    return F.smooth_l1_loss(dist, torch.zeros_like(dist), beta=beta, reduction="none").mean(dim=1)


class CornerLoss(nn.Module):
    r"""Module form of [`corner_loss`][torch_pointcloud.losses.corner.corner_loss].

    Args:
        beta: Smooth-$L_1$ transition point.
        reduction: `"none"` returns the per-box loss, `"sum"` its sum, `"mean"` its mean.

    Example:
        ```python
        criterion = CornerLoss()
        box = torch.tensor([[0.0, 0.0, 0.0, 4.0, 2.0, 1.5, 0.3]])
        criterion(box, box).item()  # 0.0
        ```
    """

    def __init__(
        self,
        beta: float = 1.0,
        reduction: Literal["none", "sum", "mean"] = "mean",
    ) -> None:
        super().__init__()
        if reduction not in ("none", "sum", "mean"):
            raise ValueError(f"`reduction` must be 'none', 'sum' or 'mean', got {reduction!r}.")

        self.beta = beta
        self.reduction = reduction

    def forward(self, pred_boxes: Tensor, gt_boxes: Tensor) -> Tensor:
        r"""Compute the corner loss between matched boxes `pred_boxes` $(N, 7)$ and `gt_boxes` $(N, 7)$."""
        loss = corner_loss(pred_boxes, gt_boxes, beta=self.beta)
        if self.reduction == "sum":
            loss = loss.sum()
        elif self.reduction == "mean":
            loss = loss.mean()
        return loss

    def extra_repr(self) -> str:
        return f"beta={self.beta}, reduction={self.reduction!r}"
