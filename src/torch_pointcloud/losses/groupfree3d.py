r"""Group-Free 3D detection loss: candidate sampling, objectness, box and semantic terms over every decoder stage."""

import math
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from torch_pointcloud.losses.focal import sigmoid_focal_loss
from torch_pointcloud.ops.box3d import angle_to_class, points_in_boxes
from torch_pointcloud.utils.data import DataKeys

if TYPE_CHECKING:
    from torch_pointcloud.models.groupfree3d import GroupFree3DOutput

_EPS = 1e-6


class GroupFree3DLoss(nn.Module):
    r"""Loss of `GroupFree3DDetection` (candidate sampling, then objectness, box and semantic terms per stage).

    Reference: :arxiv: [Group-Free 3D Object Detection via Transformers](https://arxiv.org/abs/2104.00678) (Liu et al., 2021).

    Every target is assigned inside the loss from the packed ground-truth boxes. A seed belongs to the first box of its
    scene that contains it, and a candidate inherits the box of the seed it was sampled from.

    - **Sampling:** the `num_positives` seeds of each object closest to its center, in units of its size, are
      positives of the seed objectness logits; a sigmoid focal loss averaged per scene supervises them. The seeds are
      ranked inside their own object, so an object with fewer seeds labels all of them and no seed of another object.
    - **Objectness:** a candidate is positive when its seed lies inside an object; a sigmoid focal loss averaged per
      scene supervises the objectness logits of every stage.
    - **Box and semantic:** positive candidates drive, at every stage, the center (smooth-$L_1$ to the object center),
      heading (bin cross-entropy weighted by `heading_cls_weight` and normalized residual Huber loss), size (direct
      Huber regression, or template cross-entropy weighted by `size_cls_weight` and normalized residual Huber loss)
      and semantic-class terms, each averaged over the positives.

    The total is `loss_weight` times the weighted sampling term plus the weighted objectness, box and semantic terms
    averaged over the stages. The library heading is counter-clockwise about $+z$ while the heading head predicts the
    negated angle, so the heading bins are computed from $-\theta$.

    The recipe defaults are the authors' ScanNet values; their SUN RGB-D values are in the example below.
    `LitDetectionModel` takes this loss built: its criterion factory completes the head geometry of `VoteNetLoss`.

    Args:
        num_classes: Number of semantic classes.
        num_heading_bins: Number of heading-angle bins.
        mean_sizes: Mean box size of each template of a class-aware size head, shape $(C, 3)$ (the size class is the
            semantic class), or `None` when the model regresses the size directly.
        sampling_weight: Weight of the candidate sampling term.
        objectness_weight: Weight of the objectness term.
        center_delta: Smooth-$L_1$ transition point of the center term.
        size_delta: Huber transition point of the size term.
        heading_delta: Huber transition point of the heading residual term.
        num_positives: Seeds per object labeled positive for the sampling term.
        heading_cls_weight: Weight of the heading bin term inside the box term.
        size_cls_weight: Weight of the size template term inside the box term.
        box_weight: Weight of the box term.
        sem_cls_weight: Weight of the semantic-class term.
        loss_weight: Global multiplier applied to the summed loss.

    Example:
        ```python
        import torch
        from torch_pointcloud.losses import GroupFree3DLoss
        from torch_pointcloud.models import create_model
        from torch_pointcloud.utils.data import DataKeys

        model = create_model("groupfree3d-l6-o256.sunrgbd.ze-liu", task="detection").train()
        criterion = GroupFree3DLoss(
            model.num_classes,
            num_heading_bins=model.num_heading_bins,
            mean_sizes=model.mean_sizes,
            sampling_weight=0.2,
            objectness_weight=0.4,
            center_delta=0.111111111111,
            size_delta=0.0625,
            heading_delta=0.04,
        )
        pos = torch.rand(20000, 3) * 4
        batch = torch.zeros(20000, dtype=torch.long)
        data = {
            DataKeys.BOX: torch.tensor([[1.0, 1.0, 0.5, 0.8, 0.6, 1.0, 0.3]]),
            DataKeys.LABEL: torch.tensor([2]),
            DataKeys.BATCH_BOX: torch.tensor([0]),
        }
        losses = criterion(model(None, pos, batch), data)
        losses["loss"].backward()
        ```
    """

    mean_sizes: Optional[Tensor]

    def __init__(
        self,
        num_classes: int,
        *,
        num_heading_bins: int,
        mean_sizes: Union[Tensor, List[List[float]], None],
        sampling_weight: float = 0.8,
        objectness_weight: float = 0.1,
        center_delta: float = 0.04,
        size_delta: float = 0.111111111111,
        heading_delta: float = 1.0,
        num_positives: int = 4,
        heading_cls_weight: float = 0.1,
        size_cls_weight: float = 0.1,
        box_weight: float = 1.0,
        sem_cls_weight: float = 0.1,
        loss_weight: float = 10.0,
    ) -> None:
        super().__init__()
        mean = None if mean_sizes is None else torch.as_tensor(mean_sizes, dtype=torch.float32)
        if mean is not None and mean.shape != (num_classes, 3):
            raise ValueError(f"`mean_sizes` must have shape ({num_classes}, 3), got {tuple(mean.shape)}.")

        self.num_classes = num_classes
        self.num_heading_bins = num_heading_bins
        self.sampling_weight = sampling_weight
        self.objectness_weight = objectness_weight
        self.center_delta = center_delta
        self.size_delta = size_delta
        self.heading_delta = heading_delta
        self.num_positives = num_positives
        self.heading_cls_weight = heading_cls_weight
        self.size_cls_weight = size_cls_weight
        self.box_weight = box_weight
        self.sem_cls_weight = sem_cls_weight
        self.loss_weight = loss_weight
        self.register_buffer("mean_sizes", mean)

    def forward(self, output: "GroupFree3DOutput", data: Dict[str, Any]) -> Dict[str, Tensor]:
        r"""Compute the Group-Free 3D loss and its terms.

        Args:
            output: The model's raw output: the stacked stage predictions $(T, B, Q, \cdot)$, the candidate
                `query_indices` $(B, Q)$ into the packed seeds, and the seeds' `seed_objectness_logits` $(N_s,)$,
                `pos_seed` $(N_s, 3)$ and `batch_seed` $(N_s,)$.
            data: Packed ground truth: `DataKeys.BOX` $(K, 7)$ full-extent boxes with counter-clockwise headings,
                `DataKeys.LABEL` $(K,)$ per-box classes and `DataKeys.BATCH_BOX` $(K,)$ per-box scene index.

        Returns:
            A dict with the scalar `loss` and the detached `sampling_loss`, `objectness_loss`, `box_loss`,
            `center_loss`, `heading_cls_loss`, `heading_res_loss`, `size_cls_loss`, `size_res_loss` and
            `sem_cls_loss`, each stage term summed over the stages; `size_cls_loss` is zero for a class-agnostic
            model, whose size regression is `size_res_loss`.
        """
        pos_seed, batch_seed = output["pos_seed"], output["batch_seed"]
        query_indices = output["query_indices"]
        batch_size, num_queries = query_indices.shape
        num_stages = output["center"].shape[0]

        boxes: Tensor = data[DataKeys.BOX][:, :7].to(pos_seed)
        labels: Tensor = data[DataKeys.LABEL].long().to(pos_seed.device)
        box_batch: Tensor = data[DataKeys.BATCH_BOX].to(pos_seed.device)

        num_boxes = boxes.shape[0]
        inside = points_in_boxes(pos_seed, boxes) & (batch_seed.unsqueeze(1) == box_batch.unsqueeze(0))  # (N_s, K)
        # The first box containing each seed; seeds outside every box take the zero row appended to the targets.
        assignment = torch.cat([inside, inside.new_ones(inside.shape[0], 1)], dim=1).long().argmax(dim=1)
        has_box = assignment < num_boxes
        boxes = torch.cat([boxes, boxes.new_zeros(1, 7)])
        labels = torch.cat([labels, labels.new_zeros(1)])

        sampling = self._sampling_loss(
            output["seed_objectness_logits"],
            pos_seed,
            batch_seed,
            boxes,
            assignment,
            has_box,
            batch_size,
        )

        query_box = assignment[query_indices]  # (B, Q)
        positive = has_box[query_indices].to(pos_seed.dtype)  # (B, Q)
        denom = positive.sum() + _EPS

        scene_weight = positive.new_full((batch_size, num_queries), 1.0 / num_queries)
        objectness = sigmoid_focal_loss(output["objectness_scores"], positive.unsqueeze(-1), scene_weight)
        objectness_loss = objectness.sum() / batch_size

        target = boxes[query_box]  # (B, Q, 7)
        target_label = labels[query_box]  # (B, Q)

        center = F.smooth_l1_loss(
            output["center"],
            target[..., 0:3].expand_as(output["center"]),
            beta=self.center_delta,
            reduction="none",
        )
        center_loss = (center.sum(dim=-1) * positive).sum() / denom

        nh = self.num_heading_bins
        heading_class, heading_residual = angle_to_class(-target[..., 6], nh)
        heading_cls = _cross_entropy(output["heading_scores"], heading_class)
        heading_cls_loss = (heading_cls * positive).sum() / denom
        pred_heading = output["heading_residuals_normalized"].gather(
            -1, heading_class.expand(num_stages, -1, -1).unsqueeze(-1)
        )
        pred_heading = pred_heading.squeeze(-1)
        heading_res = F.huber_loss(
            pred_heading,
            (heading_residual / (math.pi / nh)).expand_as(pred_heading),
            delta=self.heading_delta,
            reduction="none",
        )
        heading_res_loss = (heading_res * positive).sum() / denom

        if self.mean_sizes is None:
            size_res = F.huber_loss(
                output["size"],
                target[..., 3:6].expand_as(output["size"]),
                delta=self.size_delta,
                reduction="none",
            ).sum(dim=-1)
            size_cls_loss = size_res.new_zeros(())
        else:
            mean = self.mean_sizes[target_label]  # (B, Q, 3)
            size_cls = _cross_entropy(output["size_scores"], target_label)
            size_cls_loss = (size_cls * positive).sum() / denom
            index = target_label.view(1, batch_size, num_queries, 1, 1).expand(num_stages, -1, -1, 1, 3)
            pred_size = output["size_residuals_normalized"].gather(3, index).squeeze(3)
            size_res = F.huber_loss(
                pred_size,
                ((target[..., 3:6] - mean) / mean).expand_as(pred_size),
                delta=self.size_delta,
                reduction="none",
            ).sum(dim=-1)
        size_res_loss = (size_res * positive).sum() / denom

        sem_cls = _cross_entropy(output["sem_cls_scores"], target_label)
        sem_cls_loss = (sem_cls * positive).sum() / denom

        box_loss = center_loss + self.heading_cls_weight * heading_cls_loss + heading_res_loss
        box_loss = box_loss + self.size_cls_weight * size_cls_loss + size_res_loss
        stage_loss = self.objectness_weight * objectness_loss + self.box_weight * box_loss
        stage_loss = stage_loss + self.sem_cls_weight * sem_cls_loss
        total = self.loss_weight * (self.sampling_weight * sampling + stage_loss / num_stages)

        return {
            "loss": total,
            "sampling_loss": sampling.detach(),
            "objectness_loss": objectness_loss.detach(),
            "box_loss": box_loss.detach(),
            "center_loss": center_loss.detach(),
            "heading_cls_loss": heading_cls_loss.detach(),
            "heading_res_loss": heading_res_loss.detach(),
            "size_cls_loss": size_cls_loss.detach(),
            "size_res_loss": size_res_loss.detach(),
            "sem_cls_loss": sem_cls_loss.detach(),
        }

    def _sampling_loss(
        self,
        logits: Tensor,
        pos_seed: Tensor,
        batch_seed: Tensor,
        boxes: Tensor,
        assignment: Tensor,
        has_box: Tensor,
        batch_size: int,
    ) -> Tensor:
        r"""Focal loss of the seed objectness, positive on the seeds of each object closest to its center."""
        box = boxes[assignment]
        offset = (pos_seed - box[:, 0:3]) / (box[:, 3:6] + _EPS)
        distance = torch.where(has_box, (offset.pow(2).sum(dim=-1) + _EPS).sqrt(), math.inf)

        # Sort by object then by distance, so the rank of a seed inside its object's run is its distance rank.
        order = distance.argsort()
        order = order[assignment[order].argsort(stable=True)]
        sorted_box = assignment[order]
        rank = torch.arange(order.numel(), device=order.device) - torch.searchsorted(sorted_box, sorted_box)
        label = torch.zeros_like(distance)
        label[order] = ((rank < self.num_positives) & has_box[order]).to(label.dtype)

        counts = torch.zeros(batch_size, device=logits.device).index_add_(0, batch_seed, torch.ones_like(logits))
        weight = 1.0 / counts.clamp(min=1.0)[batch_seed]
        loss = sigmoid_focal_loss(logits.unsqueeze(-1), label.unsqueeze(-1), weight)
        return loss.sum() / batch_size


def _cross_entropy(logits: Tensor, target: Tensor) -> Tensor:
    r"""Cross-entropy of $(T, B, Q, C)$ stage logits against the $(B, Q)$ targets shared by every stage."""
    num_stages = logits.shape[0]
    flat = logits.reshape(-1, logits.shape[-1])
    loss = F.cross_entropy(flat, target.expand(num_stages, -1, -1).reshape(-1), reduction="none")
    return loss.view(logits.shape[:-1])
