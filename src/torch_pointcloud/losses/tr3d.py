r"""TR3D detection loss: per-class level assignment by center distance, then focal and distance-IoU terms."""

from typing import TYPE_CHECKING, Any, Dict, Sequence, Tuple

import torch
from torch import Tensor, nn

from torch_pointcloud.losses.focal import one_hot_foreground, sigmoid_focal_loss
from torch_pointcloud.losses.iou import box_iou_loss
from torch_pointcloud.ops.box3d import box_corners
from torch_pointcloud.utils.data import DataKeys

if TYPE_CHECKING:
    from torch_pointcloud.models.tr3d import TR3DOutput

__all__ = ["TR3DLoss"]

_NO_BOX = 1e8


class TR3DLoss(nn.Module):
    r"""Loss of [`TR3DDetection`][torch_pointcloud.models.tr3d.TR3DDetection]: assign every location to at most one
    box, then classify and regress the box.

    Every ground-truth box goes to the pyramid level its class is assigned to by `label_levels`, where its
    `num_positives` locations nearest to its center are its positives; a location positive for several boxes
    takes the one with the nearest center.

    The focal loss of the class logits is summed over every location and class of the batch and divided by the
    number of positives, and the distance-IoU loss of the boxes is averaged over the positives. Its center distance
    is taken over the box columns `distance_dims`, the box center $(c_x, c_y, c_z)$ by default.

    Args:
        num_classes: Number of object classes.
        label_levels: Pyramid level of every class, from the finest level.
        num_positives: Positive locations per box.
        distance_dims: Columns of the $(c_x, c_y, c_z, d_x, d_y, d_z, \theta)$ boxes whose squared differences make
            the center distance of the distance-IoU term.

    Example:
        ```python
        import torch
        from torch_pointcloud.losses import TR3DLoss
        from torch_pointcloud.models import create_model

        model = create_model("tr3d.scannet.mmdet3d", task="detection").cuda()
        label_levels = [0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0]
        criterion = TR3DLoss(num_classes=18, label_levels=label_levels, num_positives=6)
        pos = torch.rand(20000, 3, device="cuda") * 4
        out = model(torch.rand(20000, 3, device="cuda"), pos, torch.zeros(20000, dtype=torch.long, device="cuda"))
        data = {
            "box": torch.tensor([[2.0, 2.0, 0.5, 1.0, 1.0, 1.0, 0.0]], device="cuda"),
            "label": torch.tensor([4], device="cuda"),
            "batch_box": torch.tensor([0], device="cuda"),
        }
        criterion(out, data)["loss"].backward()
        ```
    """

    def __init__(
        self,
        num_classes: int,
        label_levels: Sequence[int],
        num_positives: int,
        distance_dims: Tuple[int, int, int] = (0, 1, 2),
    ) -> None:
        if len(label_levels) != num_classes:
            raise ValueError(f"label_levels needs one level per class, got {len(label_levels)} for {num_classes}.")

        super().__init__()
        self.num_classes = num_classes
        self.num_positives = num_positives
        self.label_levels = tuple(label_levels)
        self.distance_dims = tuple(distance_dims)

    @torch.no_grad()
    def assign(self, pos: Tensor, level: Tensor, boxes: Tensor, labels: Tensor) -> Tuple[Tensor, Tensor]:
        r"""Targets of the locations of one scene.

        Args:
            pos: Locations, shape $(P, 3)$.
            level: Pyramid level of every location, shape $(P,)$.
            boxes: Ground-truth boxes, shape $(K, 7)$.
            labels: Ground-truth classes, shape $(K,)$.

        Returns:
            The box target $(P, 7)$ and the class target $(P,)$ of every location, $-1$ for the locations of no box.
        """
        num_points, num_boxes = len(pos), len(boxes)
        if num_boxes == 0:
            return pos.new_zeros(num_points, 7), labels.new_full((num_points,), -1)

        candidate = level[:, None] == labels.new_tensor(self.label_levels)[labels][None, :]
        distances = (pos[:, None, :] - boxes[None, :, :3]).pow(2).sum(dim=-1)
        distances = torch.where(candidate, distances, torch.full_like(distances, _NO_BOX))
        kth = distances.topk(min(self.num_positives + 1, num_points), dim=0, largest=False).values[-1]
        distances = torch.where(distances < kth[None, :], distances, torch.full_like(distances, _NO_BOX))

        nearest, box = distances.min(dim=1)
        cls_targets = torch.where(nearest < _NO_BOX, labels[box], torch.full_like(labels[box], -1))
        return boxes[box], cls_targets

    def forward(self, output: "TR3DOutput", data: Dict[str, Any]) -> Dict[str, Tensor]:
        r"""Compute the TR3D loss and its terms.

        Args:
            output: The model's raw output: the packed locations `pos` $(P, 3)$, `batch` $(P,)$ and `level` $(P,)$,
                their class logits `cls_logits` $(P, \text{num\_classes})$ and decoded `boxes` $(P, 7)$.
            data: Packed ground truth: `DataKeys.BOX` $(K, 7)$ full-extent boxes with counter-clockwise headings,
                `DataKeys.LABEL` $(K,)$ per-box classes and `DataKeys.BATCH_BOX` $(K,)$ per-box scene index.

        Returns:
            A dict with the scalar `loss` and detached `cls_loss` and `box_loss`.
        """
        gt_boxes, gt_labels, gt_batch = data[DataKeys.BOX], data[DataKeys.LABEL], data[DataKeys.BATCH_BOX]
        box_targets = output["boxes"].new_zeros(output["boxes"].shape)
        labels = output["batch"].new_full(output["batch"].shape, -1)
        for scene in range(int(output["batch"].max()) + 1):
            rows = output["batch"] == scene
            gt = gt_batch == scene
            box_targets[rows], labels[rows] = self.assign(
                output["pos"][rows],
                output["level"][rows],
                gt_boxes[gt],
                gt_labels[gt],
            )

        positive = labels >= 0
        cls = output["cls_logits"]
        focal = sigmoid_focal_loss(cls, one_hot_foreground(labels + 1, self.num_classes).to(cls.dtype))
        cls_loss = focal.sum() / positive.sum().clamp(min=1)

        boxes, targets = output["boxes"][positive], box_targets[positive]
        if positive.any():
            corners = torch.cat([box_corners(boxes), box_corners(targets)], dim=1)
            diagonal = (corners.amax(dim=1) - corners.amin(dim=1)).pow(2).sum(dim=1)
            dims = list(self.distance_dims)
            distance = (boxes[:, dims] - targets[:, dims]).pow(2).sum(dim=1) / diagonal.clamp_min(1e-8)
            box_loss = (box_iou_loss(boxes, targets) + distance).mean()
        else:
            box_loss = boxes.sum()
        return {"loss": cls_loss + box_loss, "cls_loss": cls_loss.detach(), "box_loss": box_loss.detach()}

    def extra_repr(self) -> str:
        return (
            f"num_classes={self.num_classes}, label_levels={self.label_levels}, num_positives={self.num_positives}, "
            f"distance_dims={self.distance_dims}"
        )
