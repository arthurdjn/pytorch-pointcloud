r"""FCAF3D detection loss: pyramid-level target assignment, then focal, centerness and box IoU terms."""

from typing import TYPE_CHECKING, Any, Dict, List, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from torch_pointcloud.losses.focal import one_hot_foreground, sigmoid_focal_loss
from torch_pointcloud.losses.iou import box_iou_loss
from torch_pointcloud.utils.data import DataKeys

if TYPE_CHECKING:
    from torch_pointcloud.models.fcaf3d import FCAF3DOutput

__all__ = ["FCAF3DLoss"]

_NO_BOX = 1e8


def _face_distances(pos: Tensor, boxes: Tensor) -> Tensor:
    r"""Distances from every location to the six faces of every box, $(P, 3), (K, 7) \to (P, K, 6)$."""
    shift = pos[:, None, :] - boxes[None, :, :3]
    cos, sin = torch.cos(boxes[:, 6]), torch.sin(boxes[:, 6])
    local_x = shift[..., 0] * cos + shift[..., 1] * sin
    local_y = -shift[..., 0] * sin + shift[..., 1] * cos
    half = boxes[:, 3:6] / 2
    return torch.stack(
        [
            local_x + half[:, 0],
            half[:, 0] - local_x,
            local_y + half[:, 1],
            half[:, 1] - local_y,
            shift[..., 2] + half[:, 2],
            half[:, 2] - shift[..., 2],
        ],
        dim=-1,
    )


class FCAF3DLoss(nn.Module):
    r"""Loss of [`FCAF3DDetection`][torch_pointcloud.models.fcaf3d.FCAF3DDetection]: assign every location to at
    most one box, then classify, score the centerness and regress the box.

    Every ground-truth box goes to one pyramid level: the coarsest one, from the finest down, before the first level
    with fewer than `min_points_per_level` of its locations inside the box (the finest level when even it has fewer,
    the coarsest when none has). On that level, its `num_positives` locations of highest centerness
    $\sqrt{\prod_d \min(f^-_d, f^+_d) / \max(f^-_d, f^+_d)}$, with $f^\pm_d$ the distances to the faces of the box,
    are its positives; a location positive for several boxes takes the smallest one.

    Per scene, the focal loss of the class logits is summed over every location and class, the binary cross-entropy
    of the centerness over the positives, both divided by the number of positives, and the IoU loss of the boxes is
    weighted by the target centerness and divided by its sum. The terms are then averaged over the scenes.

    Args:
        num_classes: Number of object classes.
        min_points_per_level: Locations inside a box that a level needs to keep the box for a coarser level.
        num_positives: Positive locations per box.

    Example:
        ```python
        import torch
        from torch_pointcloud.losses import FCAF3DLoss
        from torch_pointcloud.models import create_model
        from torch_pointcloud.utils.data import DataKeys

        model = create_model("fcaf3d.scannet.mmdet3d", task="detection").cuda()
        criterion = FCAF3DLoss(num_classes=18, min_points_per_level=27, num_positives=18)
        pos = torch.rand(20000, 3, device="cuda") * 4
        out = model(torch.rand(20000, 3, device="cuda"), pos, torch.zeros(20000, dtype=torch.long, device="cuda"))
        data = {
            DataKeys.BOX: torch.tensor([[2.0, 2.0, 0.5, 1.0, 1.0, 1.0, 0.0]], device="cuda"),
            DataKeys.LABEL: torch.tensor([4], device="cuda"),
            DataKeys.BATCH_BOX: torch.tensor([0], device="cuda"),
        }
        criterion(out, data)["loss"].backward()
        ```
    """

    def __init__(self, num_classes: int, min_points_per_level: int, num_positives: int) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.min_points_per_level = min_points_per_level
        self.num_positives = num_positives

    @torch.no_grad()
    def assign(self, pos: Tensor, level: Tensor, boxes: Tensor, labels: Tensor, num_levels: int) -> Tuple[Tensor, ...]:
        r"""Targets of the locations of one scene.

        Args:
            pos: Locations, shape $(P, 3)$.
            level: Pyramid level of every location, shape $(P,)$.
            boxes: Ground-truth boxes, shape $(K, 7)$.
            labels: Ground-truth classes, shape $(K,)$.
            num_levels: Number of pyramid levels.

        Returns:
            The centerness target $(P,)$, the box target $(P, 7)$ and the class target $(P,)$ of every location,
            $-1$ for the locations of no box.
        """
        num_points, num_boxes = len(pos), len(boxes)
        if num_boxes == 0:
            return pos.new_zeros(num_points), pos.new_zeros(num_points, 7), labels.new_full((num_points,), -1)

        faces = _face_distances(pos, boxes)  # (P, K, 6)
        inside = faces.min(dim=-1).values > 0
        counts = torch.zeros(num_levels, num_boxes, device=pos.device).index_add_(0, level, inside.float())
        sparse = counts < self.min_points_per_level
        best = (sparse.int().argmax(dim=0) - 1).clamp(min=0)
        best = torch.where(sparse.any(dim=0), best, torch.full_like(best, num_levels - 1))
        candidate = inside & (level[:, None] == best[None, :])

        pairs = faces.view(num_points, num_boxes, 3, 2)
        centerness = (pairs.min(dim=-1).values / pairs.max(dim=-1).values).prod(dim=-1).sqrt()
        centerness = torch.where(candidate, centerness, torch.full_like(centerness, -1.0))
        kth = centerness.topk(min(self.num_positives + 1, num_points), dim=0).values[-1]
        candidate = candidate & (centerness > kth[None, :])

        volumes = boxes[:, 3:6].prod(dim=1).expand(num_points, num_boxes)
        volumes = torch.where(candidate, volumes, torch.full_like(volumes, _NO_BOX))
        smallest, box = volumes.min(dim=1)
        rows = torch.arange(num_points, device=pos.device)
        cls_targets = torch.where(smallest < _NO_BOX, labels[box], torch.full_like(labels[box], -1))
        return centerness[rows, box], boxes[box], cls_targets

    def forward(self, output: "FCAF3DOutput", data: Dict[str, Any]) -> Dict[str, Tensor]:
        r"""Compute the FCAF3D loss and its terms.

        Args:
            output: The model's raw output: the packed locations `pos` $(P, 3)$, `batch` $(P,)$ and `level` $(P,)$,
                their class logits `cls_logits` $(P, \text{num\_classes})$, centerness logits `centerness`
                $(P, 1)$ and decoded `boxes` $(P, 7)$.
            data: Packed ground truth: `DataKeys.BOX` $(K, 7)$ full-extent boxes with counter-clockwise headings,
                `DataKeys.LABEL` $(K,)$ per-box classes and `DataKeys.BATCH_BOX` $(K,)$ per-box scene index.

        Returns:
            A dict with the scalar `loss` and detached `cls_loss`, `center_loss` and `box_loss`.
        """
        gt_boxes, gt_labels, gt_batch = data[DataKeys.BOX], data[DataKeys.LABEL], data[DataKeys.BATCH_BOX]
        num_levels = int(output["level"].max()) + 1
        terms: Dict[str, List[Tensor]] = {"cls_loss": [], "center_loss": [], "box_loss": []}
        for scene in range(int(output["batch"].max()) + 1):
            rows = output["batch"] == scene
            gt = gt_batch == scene
            center_targets, box_targets, cls_targets = self.assign(
                output["pos"][rows],
                output["level"][rows],
                gt_boxes[gt],
                gt_labels[gt],
                num_levels,
            )
            positive = cls_targets >= 0
            num_positives = positive.sum().clamp(min=1)

            cls = output["cls_logits"][rows]
            focal = sigmoid_focal_loss(cls, one_hot_foreground(cls_targets + 1, self.num_classes).to(cls.dtype))
            terms["cls_loss"].append(focal.sum() / num_positives)

            center = output["centerness"][rows][positive, 0]
            boxes = output["boxes"][rows][positive]
            weights = center_targets[positive]
            if positive.any():
                center_loss = F.binary_cross_entropy_with_logits(center, weights, reduction="sum") / num_positives
                iou_loss = box_iou_loss(boxes, box_targets[positive])
                box_loss = (iou_loss * weights).sum() / weights.sum().clamp(min=1e-6)
            else:
                center_loss, box_loss = center.sum(), boxes.sum()
            terms["center_loss"].append(center_loss)
            terms["box_loss"].append(box_loss)

        losses = {name: torch.stack(values).mean() for name, values in terms.items()}
        total = losses["cls_loss"] + losses["center_loss"] + losses["box_loss"]
        return {"loss": total, **{name: value.detach() for name, value in losses.items()}}

    def extra_repr(self) -> str:
        return (
            f"num_classes={self.num_classes}, min_points_per_level={self.min_points_per_level}, "
            f"num_positives={self.num_positives}"
        )
