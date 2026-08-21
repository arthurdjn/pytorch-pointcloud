r"""3DETR set-prediction detection loss: Hungarian query-to-object matching with per-layer aux losses."""

import math
from typing import Any, Dict, List, Tuple

import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torch import Tensor, nn

from torch_pointcloud.transforms.functional import angle_to_class
from torch_pointcloud.utils.box3d import box3d_overlap, box_corners
from torch_pointcloud.utils.data import DataKeys

_EPS = 1e-8


def _huber_loss(error: Tensor, delta: float = 1.0) -> Tensor:
    r"""Element-wise Huber loss: $0.5 x^2$ for $|x| \le \delta$, else $\delta(|x| - 0.5\delta)$.

    Args:
        error: Residual $x$ of any shape.
        delta: Quadratic-to-linear transition point $\delta$.

    Returns:
        Per-element Huber loss, same shape as `error`.

    Shape:
        - error: $(\ldots)$
        - output: $(\ldots)$

    Example:
        >>> _huber_loss(torch.tensor([0.5, 2.0]), delta=1.0).tolist()
        [0.125, 1.5]
    """
    abs_error = error.abs()
    quadratic = abs_error.clamp(max=delta)
    linear = abs_error - quadratic
    return 0.5 * quadratic**2 + delta * linear


class _Targets:
    r"""Densified per-scene ground truth padded to a common object count $M$ (internal loss container)."""

    def __init__(
        self,
        center_unnormalized: Tensor,
        size_unnormalized: Tensor,
        angle: Tensor,
        center_normalized: Tensor,
        size_normalized: Tensor,
        angle_class: Tensor,
        angle_residual_normalized: Tensor,
        label: Tensor,
        present: Tensor,
    ) -> None:
        self.center_unnormalized = center_unnormalized
        self.size_unnormalized = size_unnormalized
        self.angle = angle
        self.center_normalized = center_normalized
        self.size_normalized = size_normalized
        self.angle_class = angle_class
        self.angle_residual_normalized = angle_residual_normalized
        self.label = label
        self.present = present
        self.nactual = present.sum(dim=1).long()


class DETR3DLoss(nn.Module):
    r"""3DETR Hungarian set-prediction detection loss.

    Reference: :arxiv: [Misra et al., 2021](https://arxiv.org/abs/2109.08141).

    Object queries are matched to ground-truth boxes one-to-one per scene by a Hungarian assignment whose
    cost combines the negative predicted class probability, the negative generalized 3D IoU, the $L_1$
    center distance (in the per-scene min-max normalized frame) and the negative objectness. Every decoder
    layer is supervised (the last layer plus the intermediate layers as auxiliary outputs), each with the
    same weighted objective, and the per-layer losses are summed:

    - **Semantic classification:** per-query weighted cross-entropy over the $C + 1$ class logits, with
      unmatched queries assigned the background slot and that slot down-weighted by `loss_no_object_weight`.
    - **Center:** $L_1$ distance between matched query and box centers in the normalized frame.
    - **Size:** $L_1$ distance between matched query and box sizes in the normalized frame.
    - **Angle:** cross-entropy on the heading bin plus a Huber loss on the in-bin residual, over matches.
    - **GIoU:** $1 - \text{gIoU}_{3D}$ between matched query and box, over matches.
    - **Cardinality:** the $L_1$ error between the count of non-background queries and the object count
      (logged only, never optimized).

    Ground truth is read packed from the batch (full-extent $(K, 7)$ boxes with counter-clockwise
    headings, plus per-box classes) and densified per scene; the headings are negated into the model's
    native heading space before binning. The normalization uses the model's `point_cloud_dims`, so the
    loss holds no reference to the model.

    Args:
        num_classes: Number of semantic classes (the class head predicts one extra background slot).
        num_angle_bin: Heading-angle bins ($1$ for axis-aligned ScanNet, $12$ for oriented SUN RGB-D).
        matcher_cls_cost: Matcher weight on the negative class probability.
        matcher_giou_cost: Matcher weight on the negative generalized 3D IoU.
        matcher_center_cost: Matcher weight on the normalized-center $L_1$ distance.
        matcher_objectness_cost: Matcher weight on the negative objectness.
        loss_giou_weight: Weight of the GIoU term in the total. The reference trains with $0$ (the GIoU
            drives only the matcher); note the rotated-box GIoU (scenes with non-zero headings) is
            computed without gradients, so a non-zero weight trains only axis-aligned scenes.
        loss_sem_cls_weight: Weight of the semantic-classification term in the total.
        loss_no_object_weight: Cross-entropy weight of the background class.
        loss_angle_cls_weight: Weight of the heading-bin classification term in the total.
        loss_angle_reg_weight: Weight of the heading-residual regression term in the total.
        loss_center_weight: Weight of the center term in the total.
        loss_size_weight: Weight of the size term in the total.
    """

    semcls_weights: Tensor

    def __init__(
        self,
        num_classes: int,
        num_angle_bin: int,
        *,
        matcher_cls_cost: float = 1.0,
        matcher_giou_cost: float = 2.0,
        matcher_center_cost: float = 0.0,
        matcher_objectness_cost: float = 0.0,
        loss_giou_weight: float = 0.0,
        loss_sem_cls_weight: float = 1.0,
        loss_no_object_weight: float = 0.2,
        loss_angle_cls_weight: float = 0.1,
        loss_angle_reg_weight: float = 0.5,
        loss_center_weight: float = 5.0,
        loss_size_weight: float = 1.0,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.num_angle_bin = num_angle_bin
        self.matcher_cls_cost = matcher_cls_cost
        self.matcher_giou_cost = matcher_giou_cost
        self.matcher_center_cost = matcher_center_cost
        self.matcher_objectness_cost = matcher_objectness_cost
        self.loss_giou_weight = loss_giou_weight
        self.loss_sem_cls_weight = loss_sem_cls_weight
        self.loss_angle_cls_weight = loss_angle_cls_weight
        self.loss_angle_reg_weight = loss_angle_reg_weight
        self.loss_center_weight = loss_center_weight
        self.loss_size_weight = loss_size_weight

        semcls_weights = torch.ones(num_classes + 1)
        semcls_weights[-1] = loss_no_object_weight
        self.register_buffer("semcls_weights", semcls_weights)

    def forward(self, output: Dict[str, Any], batch: Dict[str, Any]) -> Dict[str, Tensor]:
        r"""Compute the 3DETR set-prediction loss and its components.

        Args:
            output: A training-mode `DETR3DTrainOutput`: `aux_outputs` (a per-decoder-layer list of head
                dicts with `sem_cls_logits`, `sem_cls_prob`, `objectness_prob`, `center_normalized`,
                `center_unnormalized`, `size_normalized`, `size_unnormalized`, `angle_logits`,
                `angle_residual_normalized`, `angle_continuous`) and `point_cloud_dims`.
            batch: Packed ground truth: `DataKeys.BOX` $(K, 7)$ full-extent boxes with counter-clockwise
                headings, `DataKeys.LABEL` $(K,)$ per-box classes and `DataKeys.BATCH_BOX` $(K,)$ per-box
                scene index.

        Returns:
            A dict with the scalar `loss` (summed over decoder layers) and detached `loss_sem_cls`,
            `loss_center`, `loss_size`, `loss_angle_cls`, `loss_angle_reg`, `loss_giou`, `loss_cardinality`
            (each summed over decoder layers).
        """
        point_cloud_dims = output["point_cloud_dims"]
        layers: List[Dict[str, Tensor]] = output["aux_outputs"]
        targets = self._densify(batch, point_cloud_dims)

        num_boxes = targets.nactual.sum().clamp(min=1).to(point_cloud_dims[0].dtype)
        has_gt = bool(targets.nactual.sum() > 0)
        rotated = bool(torch.any(targets.angle * targets.present != 0))

        total = point_cloud_dims[0].new_zeros(())
        components = {name: point_cloud_dims[0].new_zeros(()) for name in _COMPONENT_NAMES}
        for layer in layers:
            layer_losses = self._layer_loss(layer, targets, num_boxes, has_gt, rotated)
            total = total + (
                self.loss_giou_weight * layer_losses["loss_giou"]
                + self.loss_sem_cls_weight * layer_losses["loss_sem_cls"]
                + self.loss_angle_cls_weight * layer_losses["loss_angle_cls"]
                + self.loss_angle_reg_weight * layer_losses["loss_angle_reg"]
                + self.loss_center_weight * layer_losses["loss_center"]
                + self.loss_size_weight * layer_losses["loss_size"]
            )
            for name, weight in _COMPONENT_WEIGHTS.items():
                components[name] = components[name] + getattr(self, weight) * layer_losses[name]
            components["loss_cardinality"] = components["loss_cardinality"] + layer_losses["loss_cardinality"]

        result: Dict[str, Tensor] = {"loss": total}
        for name in _COMPONENT_NAMES:
            result[name] = components[name].detach()
        return result

    def _densify(self, batch: Dict[str, Any], point_cloud_dims: Tuple[Tensor, Tensor]) -> _Targets:
        r"""Split the packed GT into per-scene padded tensors in normalized and metric frames.

        The packed boxes are full-extent $(K, 7)$ rows with counter-clockwise headings; the headings are
        negated into the model's native heading space before binning, so the pretrained heading head keeps
        its meaning.
        """
        box: Tensor = batch[DataKeys.BOX]
        cls: Tensor = batch[DataKeys.LABEL].long()
        box_batch: Tensor = batch[DataKeys.BATCH_BOX]
        lo, hi = point_cloud_dims
        batch_size = lo.shape[0]
        device = lo.device

        per_scene = [(box[box_batch == b], cls[box_batch == b]) for b in range(batch_size)]
        max_obj = max((scene.shape[0] for scene, _ in per_scene), default=0)
        max_obj = max(max_obj, 1)

        center = box.new_zeros(batch_size, max_obj, 3)
        size = box.new_zeros(batch_size, max_obj, 3)
        angle = box.new_zeros(batch_size, max_obj)
        label = torch.zeros(batch_size, max_obj, dtype=torch.long, device=device)
        present = box.new_zeros(batch_size, max_obj)
        for b, (scene, scene_cls) in enumerate(per_scene):
            k = scene.shape[0]
            if k == 0:
                continue
            center[b, :k] = scene[:, :3]
            size[b, :k] = scene[:, 3:6]
            angle[b, :k] = -scene[:, 6]
            label[b, :k] = scene_cls
            present[b, :k] = 1.0

        scene_scale = (hi - lo).clamp(min=1e-1)
        center_normalized = (center - lo.unsqueeze(1)) / (hi - lo).unsqueeze(1)
        size_normalized = size / scene_scale.unsqueeze(1)
        angle_class, angle_residual = angle_to_class(angle, self.num_angle_bin)
        angle_residual_normalized = angle_residual / (math.pi / self.num_angle_bin)

        return _Targets(
            center_unnormalized=center,
            size_unnormalized=size,
            angle=angle,
            center_normalized=center_normalized,
            size_normalized=size_normalized,
            angle_class=angle_class,
            angle_residual_normalized=angle_residual_normalized,
            label=label,
            present=present,
        )

    def _layer_loss(
        self,
        layer: Dict[str, Tensor],
        targets: _Targets,
        num_boxes: Tensor,
        has_gt: bool,
        rotated: bool,
    ) -> Dict[str, Tensor]:
        r"""Match one decoder layer's queries to ground truth and compute every raw (unweighted) term."""
        sem_cls_logits = layer["sem_cls_logits"]
        sem_cls_prob = layer["sem_cls_prob"]
        objectness_prob = layer["objectness_prob"]

        gious = self._giou3d(layer, targets, rotated)
        center_dist = torch.cdist(layer["center_normalized"], targets.center_normalized, p=1)
        per_prop_gt_inds, matched_mask = self._match(sem_cls_prob, objectness_prob, center_dist, gious, targets)

        gt_box_label = torch.gather(targets.label, 1, per_prop_gt_inds)
        gt_box_label = gt_box_label.masked_fill(matched_mask == 0, self.num_classes)
        sem_cls_loss = F.cross_entropy(
            sem_cls_logits.transpose(2, 1), gt_box_label, self.semcls_weights, reduction="mean"
        )

        center_loss = torch.gather(center_dist, 2, per_prop_gt_inds.unsqueeze(-1)).squeeze(-1)
        center_loss = (center_loss * matched_mask).sum() / num_boxes

        giou_loss = torch.gather(1 - gious, 2, per_prop_gt_inds.unsqueeze(-1)).squeeze(-1)
        giou_loss = (giou_loss * matched_mask).sum() / num_boxes

        gt_size = torch.gather(targets.size_normalized, 1, per_prop_gt_inds.unsqueeze(-1).expand(-1, -1, 3))
        size_loss = F.l1_loss(layer["size_normalized"], gt_size, reduction="none").sum(dim=-1)
        size_loss = (size_loss * matched_mask).sum() / num_boxes

        if has_gt:
            angle_cls_loss, angle_reg_loss = self._angle_loss(layer, targets, per_prop_gt_inds, matched_mask, num_boxes)
        else:
            angle_cls_loss = sem_cls_logits.new_zeros(())
            angle_reg_loss = sem_cls_logits.new_zeros(())

        pred_objects = (sem_cls_logits.argmax(dim=-1) != self.num_classes).sum(dim=1)
        cardinality = F.l1_loss(pred_objects.to(num_boxes.dtype), targets.nactual.to(num_boxes.dtype))

        return {
            "loss_sem_cls": sem_cls_loss,
            "loss_center": center_loss,
            "loss_size": size_loss,
            "loss_angle_cls": angle_cls_loss,
            "loss_angle_reg": angle_reg_loss,
            "loss_giou": giou_loss,
            "loss_cardinality": cardinality,
        }

    def _angle_loss(
        self,
        layer: Dict[str, Tensor],
        targets: _Targets,
        per_prop_gt_inds: Tensor,
        matched_mask: Tensor,
        num_boxes: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        r"""Heading-bin cross-entropy plus Huber residual regression over matched queries."""
        angle_logits = layer["angle_logits"]
        angle_residual_normalized = layer["angle_residual_normalized"]

        gt_angle_label = torch.gather(targets.angle_class, 1, per_prop_gt_inds)
        angle_cls = F.cross_entropy(angle_logits.transpose(2, 1), gt_angle_label, reduction="none")
        angle_cls_loss = (angle_cls * matched_mask).sum() / num_boxes

        gt_residual = torch.gather(targets.angle_residual_normalized, 1, per_prop_gt_inds)
        one_hot = torch.zeros_like(angle_residual_normalized)
        one_hot.scatter_(2, gt_angle_label.unsqueeze(-1), 1.0)
        residual_for_gt = (angle_residual_normalized * one_hot).sum(dim=-1)
        angle_reg = _huber_loss(residual_for_gt - gt_residual, delta=1.0)
        angle_reg_loss = (angle_reg * matched_mask).sum() / num_boxes
        return angle_cls_loss, angle_reg_loss

    @torch.no_grad()
    def _match(
        self,
        sem_cls_prob: Tensor,
        objectness_prob: Tensor,
        center_dist: Tensor,
        gious: Tensor,
        targets: _Targets,
    ) -> Tuple[Tensor, Tensor]:
        r"""Per-scene Hungarian assignment of queries to ground-truth boxes.

        Returns per-query matched box indices $(B, Q)$ (0 where unmatched) and a matched mask $(B, Q)$.
        """
        batch_size, num_queries = sem_cls_prob.shape[:2]
        num_gt = targets.label.shape[1]
        gt_labels = targets.label.unsqueeze(1).expand(batch_size, num_queries, num_gt)
        class_mat = -torch.gather(sem_cls_prob, 2, gt_labels)
        objectness_mat = -objectness_prob.unsqueeze(-1)
        cost = (
            self.matcher_cls_cost * class_mat
            + self.matcher_objectness_cost * objectness_mat
            + self.matcher_center_cost * center_dist
            + self.matcher_giou_cost * (-gious)
        )
        cost_np = cost.detach().cpu().numpy()

        device = sem_cls_prob.device
        per_prop_gt_inds = torch.zeros(batch_size, num_queries, dtype=torch.long, device=device)
        matched_mask = torch.zeros(batch_size, num_queries, device=device)
        nactual = targets.nactual.tolist()
        for b in range(batch_size):
            n = int(nactual[b])
            if n == 0:
                continue
            row, col = linear_sum_assignment(cost_np[b, :, :n])
            row_t = torch.as_tensor(row, dtype=torch.long, device=device)
            per_prop_gt_inds[b, row_t] = torch.as_tensor(col, dtype=torch.long, device=device)
            matched_mask[b, row_t] = 1.0
        return per_prop_gt_inds, matched_mask

    def _giou3d(self, layer: Dict[str, Tensor], targets: _Targets, rotated: bool) -> Tensor:
        r"""Pairwise generalized 3D IoU between every query and every padded ground-truth box.

        Uses the axis-aligned intersection / enclosing volumes for upright boxes (`rotated=False`), and
        the rotated bird's-eye intersection otherwise. Malformed and padded columns are zeroed, matching
        the reference criterion. Returns $(B, Q, M)$.
        """
        pred_center = layer["center_unnormalized"]
        pred_size = layer["size_unnormalized"]
        pred_angle = layer["angle_continuous"]
        gt_center = targets.center_unnormalized
        gt_size = targets.size_unnormalized
        gt_angle = targets.angle
        present = targets.present

        if not rotated:
            gious = self._giou3d_axis_aligned(pred_center, pred_size, gt_center, gt_size)
        else:
            gious = self._giou3d_rotated(pred_center, pred_size, pred_angle, gt_center, gt_size, gt_angle)
        return gious * present.unsqueeze(1)

    @staticmethod
    def _box_volume(size: Tensor) -> Tensor:
        r"""Per-box volume $\prod \sqrt{\max(d^2, 10^{-6})}$, floored at $10^{-8}$ (reference convention)."""
        return torch.sqrt((size**2).clamp(min=1e-6)).prod(dim=-1).clamp(min=_EPS)

    def _giou3d_axis_aligned(
        self, pred_center: Tensor, pred_size: Tensor, gt_center: Tensor, gt_size: Tensor
    ) -> Tensor:
        r"""Vectorized axis-aligned generalized 3D IoU, $(B, Q, 3) \times (B, M, 3) \to (B, Q, M)$."""
        lo1 = (pred_center - pred_size / 2).unsqueeze(2)
        hi1 = (pred_center + pred_size / 2).unsqueeze(2)
        lo2 = (gt_center - gt_size / 2).unsqueeze(1)
        hi2 = (gt_center + gt_size / 2).unsqueeze(1)
        inter = (torch.minimum(hi1, hi2) - torch.maximum(lo1, lo2)).clamp(min=0).prod(dim=-1)
        enclosing = (torch.maximum(hi1, hi2) - torch.minimum(lo1, lo2)).prod(dim=-1)
        vol1 = self._box_volume(pred_size).unsqueeze(2)
        vol2 = self._box_volume(gt_size).unsqueeze(1)
        return self._giou_from_volumes(inter, enclosing, vol1, vol2)

    @torch.no_grad()
    def _giou3d_rotated(
        self,
        pred_center: Tensor,
        pred_size: Tensor,
        pred_angle: Tensor,
        gt_center: Tensor,
        gt_size: Tensor,
        gt_angle: Tensor,
    ) -> Tensor:
        r"""Per-scene rotated generalized 3D IoU from the BEV-overlap intersection and corner-AABB enclosing.

        `box3d_overlap` computes the rotated intersection without gradients, so the whole rotated GIoU is
        gradient-free (a partially-detached term would push a biased gradient through the union / enclosing
        volumes only); it feeds the matcher and monitoring, not training.
        """
        batch_size, num_queries = pred_center.shape[:2]
        num_gt = gt_center.shape[1]
        gious = pred_center.new_zeros(batch_size, num_queries, num_gt)
        for b in range(batch_size):
            pred_boxes = torch.cat([pred_center[b], pred_size[b], pred_angle[b].unsqueeze(-1)], dim=-1)
            gt_boxes = torch.cat([gt_center[b], gt_size[b], gt_angle[b].unsqueeze(-1)], dim=-1)

            corners1 = box_corners(pred_boxes)
            corners2 = box_corners(gt_boxes)
            inter, _ = box3d_overlap(corners1, corners2)

            lo1, hi1 = corners1.amin(dim=1), corners1.amax(dim=1)
            lo2, hi2 = corners2.amin(dim=1), corners2.amax(dim=1)
            enclosing = (
                torch.maximum(hi1.unsqueeze(1), hi2.unsqueeze(0)) - torch.minimum(lo1.unsqueeze(1), lo2.unsqueeze(0))
            ).prod(dim=-1)
            vol1 = self._box_volume(pred_boxes[:, 3:6]).unsqueeze(1)
            vol2 = self._box_volume(gt_boxes[:, 3:6]).unsqueeze(0)
            gious[b] = self._giou_from_volumes(inter, enclosing, vol1, vol2)
        return gious

    @staticmethod
    def _giou_from_volumes(inter: Tensor, enclosing: Tensor, vol1: Tensor, vol2: Tensor) -> Tensor:
        r"""Assemble the generalized IoU from intersection, enclosing and per-box volumes.

        The enclosing volume is clamped before the division: a degenerate pair (both boxes collapsed on a
        shared plane) has `enclosing == 0`, and the unclamped `inf` would survive the `good` masking as
        `NaN` and poison the Hungarian matcher.
        """
        sum_vols = vol1 + vol2
        good = (enclosing > 2 * _EPS) & (sum_vols > 4 * _EPS)
        union = (sum_vols - inter).clamp(min=_EPS)
        giou = inter / union - (1 - union / enclosing.clamp(min=_EPS))
        return giou * good


_COMPONENT_NAMES = (
    "loss_sem_cls",
    "loss_center",
    "loss_size",
    "loss_angle_cls",
    "loss_angle_reg",
    "loss_giou",
    "loss_cardinality",
)

_COMPONENT_WEIGHTS = {
    "loss_sem_cls": "loss_sem_cls_weight",
    "loss_center": "loss_center_weight",
    "loss_size": "loss_size_weight",
    "loss_angle_cls": "loss_angle_cls_weight",
    "loss_angle_reg": "loss_angle_reg_weight",
    "loss_giou": "loss_giou_weight",
}
