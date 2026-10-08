r"""VoteNet detection loss: deep Hough voting target assignment and multi-task objective."""

import math
from typing import TYPE_CHECKING, Any, Dict, List, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from torch_pointcloud.ops.box3d import angle_to_class, points_in_boxes
from torch_pointcloud.utils.data import DataKeys

if TYPE_CHECKING:
    from torch_pointcloud.models.votenet import VoteNetOutput

_EPS = 1e-6


def _nn_distance(src: Tensor, dst: Tensor, *, l1: bool = False) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    r"""Bidirectional nearest-neighbor distances between two batched point sets.

    Args:
        src: Source points, shape $(B, N, 3)$.
        dst: Target points, shape $(B, M, 3)$.
        l1: If `True` use the $L_1$ (sum-of-abs) distance, otherwise the squared $L_2$ distance.

    Returns:
        A tuple `(dist1, idx1, dist2, idx2)`: `dist1`/`idx1` give the distance to and index of each
        source point's nearest target $(B, N)$, and `dist2`/`idx2` the reverse $(B, M)$.
    """
    diff = src.unsqueeze(2) - dst.unsqueeze(1)  # (B, N, M, 3)
    pairwise = diff.abs().sum(dim=-1) if l1 else (diff * diff).sum(dim=-1)

    near = pairwise.min(dim=2)
    far = pairwise.min(dim=1)
    return near.values, near.indices, far.values, far.indices


class _Targets:
    r"""Per-scene ground truth padded to a common object count $M$, in the model's native heading space."""

    def __init__(
        self,
        center: Tensor,
        heading_class: Tensor,
        heading_residual: Tensor,
        size_class: Tensor,
        size_residual: Tensor,
        sem_cls: Tensor,
        mask: Tensor,
    ) -> None:
        self.center = center
        self.heading_class = heading_class
        self.heading_residual = heading_residual
        self.size_class = size_class
        self.size_residual = size_residual
        self.sem_cls = sem_cls
        self.mask = mask


class VoteNetLoss(nn.Module):
    r"""Loss of `VoteNetDetection` (vote, objectness, box, semantic).

    Reference: :arxiv: [Deep Hough Voting for 3D Object Detection in Point Clouds](https://arxiv.org/abs/1904.09664) (Qi et al., 2019).

    Every target is assigned inside the loss from the packed ground-truth boxes:

    - **Vote:** each seed point inside a box votes for its center. A seed collects the centers of the first
      `gt_vote_factor` boxes containing it (repeating the first when inside fewer), and the smooth-$L_1$
      distance of its closest predicted vote to its closest target vote is averaged over the seeds inside
      a box.
    - **Objectness:** proposals are matched to objects by nearest center. A proposal is positive when its
      nearest center is within `near_threshold`, negative beyond `far_threshold`, and ignored in the band
      between; a weighted two-way cross-entropy supervises the objectness logits.
    - **Box and semantic:** positives drive the center (symmetric Chamfer between proposal and object
      centers), heading (bin cross-entropy + normalized residual smooth-$L_1$), size (class cross-entropy +
      normalized residual smooth-$L_1$) and semantic-class terms, each averaged over the positives.

    The library heading is counter-clockwise about $+z$ while the heading head predicts the negated angle,
    so the heading bins are computed from $-\theta$. Scenes are padded to a common object count with zero
    centers that take part in the proposal-to-object matching.

    Args:
        num_classes: Number of semantic classes.
        num_heading_bins: Number of heading-angle bins ($1$ for axis-aligned ScanNet, $12$ for SUN RGB-D).
        num_size_clusters: Number of size templates (the size class is the semantic class).
        mean_sizes: Per-template mean box size, shape $(\text{num\_size\_clusters}, 3)$.
        near_threshold: Distance (meters) below which a proposal is a positive object match.
        far_threshold: Distance (meters) above which a proposal is a negative match.
        objectness_weights: Cross-entropy class weights $[\text{negative}, \text{positive}]$.
        gt_vote_factor: Number of target votes (containing-box centers) collected per seed.
        loss_weight: Global multiplier applied to the summed loss.
    """

    mean_sizes: Tensor

    def __init__(
        self,
        num_classes: int,
        *,
        num_heading_bins: int,
        num_size_clusters: int,
        mean_sizes: Union[Tensor, List[List[float]]],
        near_threshold: float = 0.3,
        far_threshold: float = 0.6,
        objectness_weights: Tuple[float, float] = (0.2, 0.8),
        gt_vote_factor: int = 3,
        loss_weight: float = 10.0,
    ) -> None:
        super().__init__()
        self.num_heading_bins = num_heading_bins
        self.num_size_clusters = num_size_clusters
        self.num_classes = num_classes
        self.near_threshold = near_threshold
        self.far_threshold = far_threshold
        self.objectness_weights = objectness_weights
        self.gt_vote_factor = gt_vote_factor
        self.loss_weight = loss_weight

        mean = torch.as_tensor(mean_sizes, dtype=torch.float32)
        if mean.shape != (num_size_clusters, 3):
            raise ValueError(f"`mean_sizes` must have shape ({num_size_clusters}, 3), got {tuple(mean.shape)}.")

        self.register_buffer("mean_sizes", mean)

    def forward(self, output: "VoteNetOutput", data: Dict[str, Any]) -> Dict[str, Tensor]:
        r"""Compute the VoteNet loss and its terms.

        Args:
            output: The model's raw output: dense head tensors (`objectness_scores`, `center`,
                `heading_scores`, `heading_residuals_normalized`, `size_scores`,
                `size_residuals_normalized`, `sem_cls_scores`, `pos_vote_aggr`) as $(B, K, \cdot)$,
                plus the packed seeds `pos_seed` $(S, 3)$, `batch_seed` $(S,)$ and their votes `pos_vote`
                $(S \cdot \text{vote\_factor}, 3)$.
            data: Packed ground truth: `DataKeys.BOX` $(K, 7)$ full-extent boxes with counter-clockwise
                headings, `DataKeys.LABEL` $(K,)$ per-box classes and `DataKeys.BATCH_BOX` $(K,)$ per-box
                scene index.

        Returns:
            A dict with the scalar `loss` and detached `vote_loss`, `objectness_loss`, `box_loss`,
            `center_loss`, `heading_cls_loss`, `heading_res_loss`, `size_cls_loss`, `size_res_loss`,
            `sem_cls_loss` and the `obj_acc` diagnostic.
        """
        scores = output["objectness_scores"]
        targets = self._densify(data, scores.shape[0])

        vote_loss = self._vote_loss(output, data)
        objectness_loss, objectness_label, objectness_mask, assignment = self._objectness_loss(output, targets)
        center, heading_cls, heading_res, size_cls, size_res, sem_cls = self._box_and_sem_loss(
            output, targets, objectness_label, assignment
        )

        box_loss = center + 0.1 * heading_cls + heading_res + 0.1 * size_cls + size_res
        total = self.loss_weight * (vote_loss + 0.5 * objectness_loss + box_loss + 0.1 * sem_cls)
        obj_acc = self._objectness_accuracy(scores, objectness_label, objectness_mask)

        return {
            "loss": total,
            "vote_loss": vote_loss.detach(),
            "objectness_loss": objectness_loss.detach(),
            "box_loss": box_loss.detach(),
            "center_loss": center.detach(),
            "heading_cls_loss": heading_cls.detach(),
            "heading_res_loss": heading_res.detach(),
            "size_cls_loss": size_cls.detach(),
            "size_res_loss": size_res.detach(),
            "sem_cls_loss": sem_cls.detach(),
            "obj_acc": obj_acc,
        }

    def _densify(self, data: Dict[str, Any], batch_size: int) -> _Targets:
        r"""Pad the packed boxes to a common per-scene count and encode heading, size and semantic targets.

        Headings are negated into the model's native space before binning; the size class is the semantic
        class and the size residual is measured against that class's mean size. Boxes keep their packed
        order within a scene.
        """
        ref = self.mean_sizes
        boxes: Tensor = data[DataKeys.BOX][:, :7].to(ref)
        labels: Tensor = data[DataKeys.LABEL].long().to(ref.device)
        box_batch: Tensor = data[DataKeys.BATCH_BOX].to(ref.device)

        counts = torch.bincount(box_batch, minlength=batch_size)
        max_obj = max(int(counts.max()) if boxes.shape[0] else 0, 1)
        center = ref.new_zeros(batch_size, max_obj, 3)
        heading_class = torch.zeros(batch_size, max_obj, dtype=torch.long, device=ref.device)
        heading_residual = ref.new_zeros(batch_size, max_obj)
        size_class = torch.zeros(batch_size, max_obj, dtype=torch.long, device=ref.device)
        size_residual = ref.new_zeros(batch_size, max_obj, 3)
        sem_cls = torch.zeros(batch_size, max_obj, dtype=torch.long, device=ref.device)
        mask = ref.new_zeros(batch_size, max_obj)
        if boxes.shape[0] == 0:
            return _Targets(center, heading_class, heading_residual, size_class, size_residual, sem_cls, mask)

        # The slot of a box is its rank among the boxes of its scene, in packed order.
        order = torch.argsort(box_batch, stable=True)
        starts = counts.cumsum(0) - counts
        slot = torch.empty_like(box_batch)
        slot[order] = torch.arange(boxes.shape[0], device=ref.device) - starts[box_batch[order]]

        center[box_batch, slot] = boxes[:, 0:3]
        cls, residual = angle_to_class(-boxes[:, 6], self.num_heading_bins)
        heading_class[box_batch, slot] = cls
        heading_residual[box_batch, slot] = residual
        size_class[box_batch, slot] = labels
        size_residual[box_batch, slot] = boxes[:, 3:6] - self.mean_sizes[labels]
        sem_cls[box_batch, slot] = labels
        mask[box_batch, slot] = 1.0

        return _Targets(center, heading_class, heading_residual, size_class, size_residual, sem_cls, mask)

    def _vote_targets(self, pos_seed: Tensor, batch_seed: Tensor, data: Dict[str, Any]) -> Tuple[Tensor, Tensor]:
        r"""Collect, per seed, the centers of the first `gt_vote_factor` boxes of its scene containing it.

        A seed inside fewer boxes repeats the first center in the unfilled slots, so the min-over-votes loss
        can credit either center on overlapping objects. The whole batch is tested at once, each seed against
        the boxes of its own scene.

        Returns:
            `(votes, mask)`: target vote positions $(S, G, 3)$ and the $(S,)$ mask of seeds inside a box.
        """
        num_seed, slots = pos_seed.shape[0], self.gt_vote_factor
        votes = pos_seed.new_zeros(num_seed, slots, 3)
        mask = pos_seed.new_zeros(num_seed)

        boxes: Tensor = data[DataKeys.BOX][:, :7].to(pos_seed)
        box_batch: Tensor = data[DataKeys.BATCH_BOX].to(pos_seed.device)
        if boxes.shape[0] == 0 or num_seed == 0:
            return votes, mask

        inside = points_in_boxes(pos_seed, boxes) & (batch_seed.unsqueeze(1) == box_batch.unsqueeze(0))  # (S, K)
        rank = inside.long().cumsum(dim=1)
        centers = boxes[:, 0:3]

        votes = centers[inside.long().argmax(dim=1)].unsqueeze(1).expand(-1, slots, -1).clone()
        for slot in range(1, slots):
            sel = inside & (rank == slot + 1)
            has = sel.any(dim=1)
            votes[:, slot] = torch.where(has.unsqueeze(1), centers[sel.long().argmax(dim=1)], votes[:, slot])
        mask = inside.any(dim=1).to(mask.dtype)

        return votes, mask

    def _vote_loss(self, output: "VoteNetOutput", data: Dict[str, Any]) -> Tensor:
        r"""$L_1$ vote regression, masked to object seeds (closest predicted vote to the closest target)."""
        pos_seed = output["pos_seed"]
        pos_vote = output["pos_vote"]
        num_seed = pos_seed.shape[0]
        if num_seed == 0:
            raise ValueError("`VoteNetLoss` requires a non-empty batch of seeds.")

        vote_factor = pos_vote.shape[0] // num_seed

        gt_votes, gt_mask = self._vote_targets(pos_seed, output["batch_seed"], data)
        pred = pos_vote.reshape(num_seed, vote_factor, 3)
        dist = (gt_votes.unsqueeze(2) - pred.unsqueeze(1)).abs().sum(dim=-1)  # (S, G, vote_factor)
        votes_dist = dist.min(dim=2).values.min(dim=1).values

        loss: Tensor = (votes_dist * gt_mask).sum() / (gt_mask.sum() + _EPS)
        return loss

    def _objectness_loss(self, output: "VoteNetOutput", targets: _Targets) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        r"""Weighted 2-way cross-entropy with near/far proposal-to-object center assignment."""
        scores = output["objectness_scores"]
        dist1, idx1, _, _ = _nn_distance(output["pos_vote_aggr"], targets.center)
        euclidean = (dist1 + _EPS).sqrt()
        objectness_label: Tensor = (euclidean < self.near_threshold).long()
        objectness_mask: Tensor = ((euclidean < self.near_threshold) | (euclidean > self.far_threshold)).float()

        weights = scores.new_tensor(self.objectness_weights)
        ce = F.cross_entropy(scores.transpose(1, 2), objectness_label, weight=weights, reduction="none")
        loss: Tensor = (ce * objectness_mask).sum() / (objectness_mask.sum() + _EPS)
        return loss, objectness_label, objectness_mask, idx1

    def _box_and_sem_loss(
        self,
        output: "VoteNetOutput",
        targets: _Targets,
        objectness_label: Tensor,
        assignment: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        r"""Center (chamfer), heading (cls + residual), size (cls + residual) and semantic losses."""
        obj = objectness_label.float()
        denom = obj.sum() + _EPS
        nh, ns = self.num_heading_bins, self.num_size_clusters

        dist1, _, dist2, _ = _nn_distance(output["center"], targets.center)
        center_loss = (dist1 * obj).sum() / denom + (dist2 * targets.mask).sum() / (targets.mask.sum() + _EPS)

        heading_class_label = torch.gather(targets.heading_class, 1, assignment)
        heading_residual_label = torch.gather(targets.heading_residual, 1, assignment)
        heading_cls = F.cross_entropy(output["heading_scores"].transpose(1, 2), heading_class_label, reduction="none")
        heading_cls = (heading_cls * obj).sum() / denom

        heading_res_norm_label = heading_residual_label / (math.pi / nh)
        heading_one_hot = F.one_hot(heading_class_label, nh).float()
        pred_heading_res = (output["heading_residuals_normalized"] * heading_one_hot).sum(dim=-1)
        heading_res = F.smooth_l1_loss(pred_heading_res, heading_res_norm_label, reduction="none")
        heading_res = (heading_res * obj).sum() / denom

        size_class_label = torch.gather(targets.size_class, 1, assignment)
        size_cls = F.cross_entropy(output["size_scores"].transpose(1, 2), size_class_label, reduction="none")
        size_cls = (size_cls * obj).sum() / denom

        gather_size = assignment.unsqueeze(-1).expand(-1, -1, 3)
        size_residual_label = torch.gather(targets.size_residual, 1, gather_size)
        size_one_hot = F.one_hot(size_class_label, ns).float().unsqueeze(-1)
        pred_size_res = (output["size_residuals_normalized"] * size_one_hot).sum(dim=2)
        mean_size = (size_one_hot * self.mean_sizes.view(1, 1, ns, 3)).sum(dim=2)
        size_res_norm_label = size_residual_label / mean_size
        size_res = F.smooth_l1_loss(pred_size_res, size_res_norm_label, reduction="none").mean(dim=-1)
        size_res = (size_res * obj).sum() / denom

        sem_cls_label = torch.gather(targets.sem_cls, 1, assignment)
        sem_cls = F.cross_entropy(output["sem_cls_scores"].transpose(1, 2), sem_cls_label, reduction="none")
        sem_cls = (sem_cls * obj).sum() / denom

        return center_loss, heading_cls, heading_res, size_cls, size_res, sem_cls

    @staticmethod
    def _objectness_accuracy(scores: Tensor, label: Tensor, mask: Tensor) -> Tensor:
        r"""Masked accuracy of the objectness classifier (a logged diagnostic, not optimized)."""
        correct = (scores.argmax(dim=2) == label).float() * mask
        return correct.sum() / (mask.sum() + _EPS)
