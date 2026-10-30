r"""IoU losses between paired 3D boxes."""

from typing import Literal

import torch
from torch import Tensor, nn

from torch_pointcloud.ops.box3d import box3d_overlap, box_corners

__all__ = ["BoxIoULoss", "box_iou_loss"]


def box_iou_loss(pred_boxes: Tensor, target_boxes: Tensor, *, mode: Literal["iou", "diou"] = "iou") -> Tensor:
    r"""Per-box IoU loss $1 - \text{IoU}$ between paired oriented 3D boxes, or its distance-IoU form.

    The IoU is the exact oriented 3D IoU of [`box3d_overlap`][torch_pointcloud.ops.box3d.box3d_overlap], so the loss
    trains the center, the size and the heading. With `mode="diou"`, the Distance-IoU loss of
    :arxiv: [Distance-IoU Loss: Faster and Better Learning for Bounding Box Regression](https://arxiv.org/abs/1911.08287)
    (Zheng et al., 2020) adds the squared distance between the two centers over the squared diagonal of the smallest
    axis-aligned box enclosing both boxes, which still pulls apart boxes that do not overlap.

    Args:
        pred_boxes: Predicted boxes $(c_x, c_y, c_z, d_x, d_y, d_z, \theta)$, shape $(N, 7)$.
        target_boxes: Matched target boxes, shape $(N, 7)$.
        mode: `"iou"` for $1 - \text{IoU}$, `"diou"` to add the normalized center distance of the Distance-IoU
            loss.

    Returns:
        The per-box loss, shape $(N,)$.

    Shape:
        - pred_boxes: $(N, 7)$
        - target_boxes: $(N, 7)$
        - output: $(N,)$

    Example:
        ```pycon
        >>> import torch
        >>> from torch_pointcloud.losses.iou import box_iou_loss
        >>> target = torch.tensor([[0.0, 0.0, 0.0, 2.0, 2.0, 2.0, 0.0]])
        >>> pred = torch.tensor([[1.0, 0.0, 0.0, 2.0, 2.0, 2.0, 0.0]])
        >>> round(box_iou_loss(pred, target).item(), 4)
        0.6667
        >>> round(box_iou_loss(pred, target, mode="diou").item(), 4)
        0.7255

        ```
    """
    if mode not in ("iou", "diou"):
        raise ValueError(f"`mode` must be 'iou' or 'diou', got {mode!r}.")

    pred_corners, target_corners = box_corners(pred_boxes), box_corners(target_boxes)
    _, iou = box3d_overlap(pred_corners, target_corners, aligned=True)
    loss = 1.0 - iou
    if mode == "diou":
        corners = torch.cat([pred_corners, target_corners], dim=1)
        diagonal = (corners.amax(dim=1) - corners.amin(dim=1)).pow(2).sum(dim=1)
        loss = loss + (pred_boxes[:, :3] - target_boxes[:, :3]).pow(2).sum(dim=1) / diagonal.clamp_min(1e-8)
    return loss


class BoxIoULoss(nn.Module):
    r"""Module form of [`box_iou_loss`][torch_pointcloud.losses.iou.box_iou_loss].

    Args:
        mode: `"iou"` for $1 - \text{IoU}$, `"diou"` to add the normalized center distance of the Distance-IoU
            loss.
        reduction: `"none"` returns the per-box loss, `"sum"` its sum, `"mean"` its mean.

    Example:
        ```python
        import torch
        from torch_pointcloud.losses import BoxIoULoss

        criterion = BoxIoULoss(mode="diou")
        box = torch.tensor([[0.0, 0.0, 0.0, 4.0, 2.0, 1.5, 0.3]])
        criterion(box, box).item()  # 0.0
        ```
    """

    def __init__(
        self,
        mode: Literal["iou", "diou"] = "iou",
        reduction: Literal["none", "sum", "mean"] = "mean",
    ) -> None:
        super().__init__()
        if mode not in ("iou", "diou"):
            raise ValueError(f"`mode` must be 'iou' or 'diou', got {mode!r}.")
        if reduction not in ("none", "sum", "mean"):
            raise ValueError(f"`reduction` must be 'none', 'sum' or 'mean', got {reduction!r}.")

        self.mode = mode
        self.reduction = reduction

    def forward(self, pred_boxes: Tensor, target_boxes: Tensor) -> Tensor:
        r"""Compute the loss between matched boxes `pred_boxes` $(N, 7)$ and `target_boxes` $(N, 7)$."""
        loss = box_iou_loss(pred_boxes, target_boxes, mode=self.mode)
        if self.reduction == "sum":
            loss = loss.sum()
        elif self.reduction == "mean":
            loss = loss.mean()
        return loss

    def extra_repr(self) -> str:
        return f"mode={self.mode!r}, reduction={self.reduction!r}"
