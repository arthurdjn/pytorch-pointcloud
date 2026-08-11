r"""VoteNet detection loss: deep Hough voting target assignment and multi-task objective."""

import math
from typing import Any, Dict, List, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor, nn

_EPS = 1e-6


def _nn_distance(src: Tensor, dst: Tensor, *, l1: bool = False) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    r"""Bidirectional nearest-neighbour distances between two batched point sets.

    Args:
        src: Source points, shape $(B, N, 3)$.
        dst: Target points, shape $(B, M, 3)$.
        l1: If `True` use the $L_1$ (sum-of-abs) distance, otherwise the squared $L_2$ distance.

    Returns:
        A tuple `(dist1, idx1, dist2, idx2)`: `dist1`/`idx1` give the distance to and index of each
        source point's nearest target $(B, N)$, and `dist2`/`idx2` the reverse $(B, M)$.
    """
    pairwise = torch.cdist(src, dst, p=1.0 if l1 else 2.0)
    if not l1:
        pairwise = pairwise.pow(2)
    near = pairwise.min(dim=2)
    far = pairwise.min(dim=1)
    return near.values, near.indices, far.values, far.indices


class VoteNetLoss(nn.Module):
    r"""Multi-task VoteNet detection loss (vote, objectness, box, semantic).

    Reference: :arxiv: [Qi et al., 2019](https://arxiv.org/abs/1904.09664).

    Proposals are matched to ground-truth objects by nearest center: a proposal is positive when its
    nearest GT center is within `near_threshold`, negative beyond `far_threshold`, and ignored in the
    band between. Positives drive the center, heading, size and semantic terms; the vote term pulls
    each object seed's vote toward its object center (the closest of up to three candidate votes).

    Args:
        num_heading_bin: Number of heading-angle bins ($1$ for axis-aligned ScanNet, $12$ for SUN RGB-D).
        num_size_cluster: Number of size templates.
        num_classes: Number of semantic classes.
        mean_sizes: Per-template mean box size, shape $(\text{num\_size\_cluster}, 3)$.
        near_threshold: Distance (meters) below which a proposal is a positive object match.
        far_threshold: Distance (meters) above which a proposal is a negative match.
        objectness_weights: Cross-entropy class weights $[\text{negative}, \text{positive}]$.
        loss_scale: Global multiplier applied to the summed loss.
    """

    mean_sizes: Tensor

    def __init__(
        self,
        num_heading_bin: int,
        num_size_cluster: int,
        num_classes: int,
        mean_sizes: Union[Tensor, List[List[float]]],
        *,
        near_threshold: float = 0.3,
        far_threshold: float = 0.6,
        objectness_weights: Tuple[float, float] = (0.2, 0.8),
        loss_scale: float = 10.0,
    ) -> None:
        super().__init__()
        self.num_heading_bin = num_heading_bin
        self.num_size_cluster = num_size_cluster
        self.num_classes = num_classes
        self.near_threshold = near_threshold
        self.far_threshold = far_threshold
        self.objectness_weights = objectness_weights
        self.loss_scale = loss_scale

        mean = torch.as_tensor(mean_sizes, dtype=torch.float32)
        if mean.shape != (num_size_cluster, 3):
            raise ValueError(f"`mean_sizes` must have shape ({num_size_cluster}, 3), got {tuple(mean.shape)}.")
        self.register_buffer("mean_sizes", mean)

    def forward(self, output: Dict[str, Tensor], batch: Dict[str, Any]) -> Dict[str, Tensor]:
        r"""Compute the VoteNet loss and its components.

        Args:
            output: The model's raw output: dense head tensors (`objectness_scores`, `center`,
                `heading_scores`, `heading_residuals_normalized`, `size_scores`,
                `size_residuals_normalized`, `sem_cls_scores`, `pos_vote_aggr`) as $(B, K, \cdot)$,
                plus the packed `pos_seed`, `pos_vote` $(S, 3)$ and `seed_indices`, `batch_seed`, `batch_vote` $(S,)$.
            batch: Ground truth (`center_label`, `heading_class_label`, `heading_residual_label`,
                `size_class_label`, `size_residual_label`, `sem_cls_label`, `box_label_mask` as
                $(B, M, \cdot)$, per-point `vote_label` $(B, N, 9)$, `vote_label_mask` $(B, N)$, and the
                per-point `batch` index). The heading labels are binned from counter-clockwise headings
                and re-binned internally into the model's native (negated) heading space.

        Returns:
            A dict with the scalar `loss` (to backprop) and detached `vote_loss`, `objectness_loss`,
            `box_loss`, `center_loss`, `heading_cls_loss`, `heading_res_loss`, `size_cls_loss`,
            `size_res_loss`, `sem_cls_loss` and `obj_acc` diagnostics.
        """
        output = self._densify(output, batch)
        vote_loss = self._vote_loss(output, batch)
        objectness_loss, objectness_label, objectness_mask, assignment = self._objectness_loss(output, batch)
        center, heading_cls, heading_res, size_cls, size_res, sem_cls = self._box_and_sem_loss(
            output, batch, objectness_label, assignment
        )

        box_loss = center + 0.1 * heading_cls + heading_res + 0.1 * size_cls + size_res
        total = self.loss_scale * (vote_loss + 0.5 * objectness_loss + box_loss + 0.1 * sem_cls)
        obj_acc = self._objectness_accuracy(output["objectness_scores"], objectness_label, objectness_mask)

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

    def _densify(self, output: Dict[str, Tensor], batch: Dict[str, Any]) -> Dict[str, Tensor]:
        r"""Reshape the model's packed seed / vote tensors to the dense $(B, S, \cdot)$ the terms expect.

        The proposal tensors are already dense $(B, K, \cdot)$; the seeds are a fixed count per scene, so the
        packed `pos_seed` / `pos_vote` / `seed_indices` (and their batch vectors) reshape by the batch size.
        `seed_indices` are global indices into the packed points, localized to $[0, N)$ by subtracting each
        scene's offset so the per-scene `vote_label` can be gathered onto seeds.
        """
        batch_idx: Tensor = batch["batch"]
        if batch_idx.numel() == 0:
            raise ValueError("`VoteNetLoss` requires a non-empty batch of points.")
        counts = batch_idx.bincount()
        if bool((counts != counts[0]).any()):
            raise ValueError(
                f"`VoteNetLoss` requires the same number of points per scene, got counts {counts.tolist()}."
            )
        batch_size = counts.numel()
        num_points = int(counts[0])
        dense = dict(output)
        for key in ("pos_seed", "pos_vote", "seed_indices", "batch_seed", "batch_vote"):
            tensor = output[key]
            dense[key] = tensor.reshape(batch_size, tensor.shape[0] // batch_size, *tensor.shape[1:])
        dense["seed_indices"] = dense["seed_indices"] - dense["batch_seed"] * num_points
        return dense

    def _vote_loss(self, output: Dict[str, Tensor], batch: Dict[str, Any]) -> Tensor:
        r"""Smooth $L_1$ vote regression, masked to object seeds (closest of the candidate votes)."""
        pos_seed = output["pos_seed"]
        pos_vote = output["pos_vote"]
        seed_indices = output["seed_indices"].long()
        vote_label: Tensor = batch["vote_label"]
        vote_label_mask: Tensor = batch["vote_label_mask"]

        batch_size, num_seed = pos_seed.shape[:2]
        num_candidates = vote_label.size(-1) // 3
        vote_factor = pos_vote.size(1) // num_seed

        seed_gt_votes_mask = torch.gather(vote_label_mask, 1, seed_indices).float()
        gather_idx = seed_indices.unsqueeze(-1).expand(-1, -1, vote_label.size(-1))
        seed_gt_votes = torch.gather(vote_label, 1, gather_idx) + pos_seed.repeat(1, 1, num_candidates)

        pred = pos_vote.reshape(batch_size * num_seed, vote_factor, 3)
        gt = seed_gt_votes.reshape(batch_size * num_seed, num_candidates, 3)
        dist = torch.cdist(gt, pred, p=1.0)
        votes_dist = dist.min(dim=2).values.min(dim=1).values.reshape(batch_size, num_seed)
        loss: Tensor = (votes_dist * seed_gt_votes_mask).sum() / (seed_gt_votes_mask.sum() + _EPS)
        return loss

    def _objectness_loss(
        self, output: Dict[str, Tensor], batch: Dict[str, Any]
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        r"""Weighted 2-way cross-entropy with near/far proposal-to-GT center assignment."""
        scores = output["objectness_scores"]
        gt_center: Tensor = batch["center_label"]

        dist1, idx1, _, _ = _nn_distance(output["pos_vote_aggr"], gt_center)
        euclidean = (dist1 + _EPS).sqrt()
        objectness_label: Tensor = (euclidean < self.near_threshold).long()
        objectness_mask: Tensor = ((euclidean < self.near_threshold) | (euclidean > self.far_threshold)).float()

        weights = scores.new_tensor(self.objectness_weights)
        ce = F.cross_entropy(scores.transpose(1, 2), objectness_label, weight=weights, reduction="none")
        loss: Tensor = (ce * objectness_mask).sum() / (objectness_mask.sum() + _EPS)
        return loss, objectness_label, objectness_mask, idx1

    def _heading_to_native(self, heading_class: Tensor, heading_residual: Tensor) -> Tuple[Tensor, Tensor]:
        r"""Re-bin counter-clockwise heading labels into the model's native (negated) heading space.

        The ground-truth bins encode the library heading convention (counter-clockwise about $+z$), while
        the heading head predicts the negated angle; the continuous angle is reconstructed from the bins,
        negated, and re-binned, which is exact (binning loses no information). With a single bin
        (axis-aligned boxes, zero headings) the conversion is the identity.

        Args:
            heading_class: Heading bin indices (long), shape $(B, K)$.
            heading_residual: In-bin residual angles, shape $(B, K)$.

        Returns:
            The `(heading_class, heading_residual)` pair re-binned in the native heading space.
        """
        two_pi = 2 * math.pi
        angle_per_class = two_pi / self.num_heading_bin
        angle = heading_class.to(heading_residual.dtype) * angle_per_class + heading_residual
        shifted = ((-angle) % two_pi + angle_per_class / 2) % two_pi
        native_class = (shifted / angle_per_class).floor().long().clamp(max=self.num_heading_bin - 1)
        native_residual = shifted - (native_class.to(shifted.dtype) * angle_per_class + angle_per_class / 2)
        return native_class, native_residual

    def _box_and_sem_loss(
        self, output: Dict[str, Tensor], batch: Dict[str, Any], objectness_label: Tensor, assignment: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        r"""Center (chamfer), heading (cls + residual), size (cls + residual) and semantic losses."""
        obj = objectness_label.float()
        denom = obj.sum() + _EPS
        nh, ns = self.num_heading_bin, self.num_size_cluster

        gt_center: Tensor = batch["center_label"]
        box_label_mask: Tensor = batch["box_label_mask"]
        dist1, _, dist2, _ = _nn_distance(output["center"], gt_center)
        center_loss = (dist1 * obj).sum() / denom + (dist2 * box_label_mask).sum() / (box_label_mask.sum() + _EPS)

        heading_class_label = torch.gather(batch["heading_class_label"], 1, assignment)
        heading_residual_label = torch.gather(batch["heading_residual_label"], 1, assignment)
        heading_class_label, heading_residual_label = self._heading_to_native(
            heading_class_label, heading_residual_label
        )
        heading_cls = F.cross_entropy(output["heading_scores"].transpose(1, 2), heading_class_label, reduction="none")
        heading_cls = (heading_cls * obj).sum() / denom

        heading_res_norm_label = heading_residual_label / (math.pi / nh)
        heading_one_hot = F.one_hot(heading_class_label, nh).float()
        pred_heading_res = (output["heading_residuals_normalized"] * heading_one_hot).sum(dim=-1)
        heading_res = F.smooth_l1_loss(pred_heading_res, heading_res_norm_label, reduction="none")
        heading_res = (heading_res * obj).sum() / denom

        size_class_label = torch.gather(batch["size_class_label"], 1, assignment)
        size_cls = F.cross_entropy(output["size_scores"].transpose(1, 2), size_class_label, reduction="none")
        size_cls = (size_cls * obj).sum() / denom

        gather_size = assignment.unsqueeze(-1).expand(-1, -1, 3)
        size_residual_label = torch.gather(batch["size_residual_label"], 1, gather_size)
        size_one_hot = F.one_hot(size_class_label, ns).float().unsqueeze(-1)
        pred_size_res = (output["size_residuals_normalized"] * size_one_hot).sum(dim=2)
        mean_size = (size_one_hot * self.mean_sizes.view(1, 1, ns, 3)).sum(dim=2)
        size_res_norm_label = size_residual_label / mean_size
        size_res = F.smooth_l1_loss(pred_size_res, size_res_norm_label, reduction="none").mean(dim=-1)
        size_res = (size_res * obj).sum() / denom

        sem_cls_label = torch.gather(batch["sem_cls_label"], 1, assignment)
        sem_cls = F.cross_entropy(output["sem_cls_scores"].transpose(1, 2), sem_cls_label, reduction="none")
        sem_cls = (sem_cls * obj).sum() / denom

        return center_loss, heading_cls, heading_res, size_cls, size_res, sem_cls

    @staticmethod
    def _objectness_accuracy(scores: Tensor, label: Tensor, mask: Tensor) -> Tensor:
        r"""Masked accuracy of the objectness classifier (a logged diagnostic, not optimized)."""
        correct = (scores.argmax(dim=2) == label).float() * mask
        return correct.sum() / (mask.sum() + _EPS)
