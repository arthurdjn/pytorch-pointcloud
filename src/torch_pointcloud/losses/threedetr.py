r"""3DETR set-prediction detection loss: Hungarian query-to-object matching with per-layer aux losses."""

import math
from typing import TYPE_CHECKING, Any, Dict, List, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from torch_pointcloud.losses.matching import hungarian_match_batched
from torch_pointcloud.ops.box3d import angle_to_class, box3d_overlap, box_corners
from torch_pointcloud.utils.data import DataKeys

if TYPE_CHECKING:
    from torch_pointcloud.models.threedetr import ThreeDETRTrainOutput

_EPS = 1e-8


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


class ThreeDETRLoss(nn.Module):
    r"""3DETR Hungarian set-prediction detection loss.

    Reference: :arxiv: [An End-to-End Transformer Model for 3D Object Detection](https://arxiv.org/abs/2109.08141) (Misra et al., 2021).

    Object queries are matched to ground-truth boxes one-to-one per scene by a Hungarian assignment whose
    cost combines the negative predicted class probability, the negative generalized 3D IoU, the $L_1$
    center distance (in the per-scene min-max normalized frame) and the negative objectness. Every decoder
    layer is supervised (the last layer plus the intermediate layers as auxiliary outputs), each with the
    same weighted objective, and the per-layer losses are summed:

    - **Semantic classification:** per-query weighted cross-entropy over the $C + 1$ class logits, with
      unmatched queries assigned the background slot and that slot down-weighted by `no_object_weight`.
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
        num_heading_bins: Heading-angle bins ($1$ for axis-aligned ScanNet, $12$ for oriented SUN RGB-D).
        matcher_cls_cost: Matcher weight on the negative class probability.
        matcher_giou_cost: Matcher weight on the negative generalized 3D IoU.
        matcher_center_cost: Matcher weight on the normalized-center $L_1$ distance.
        matcher_objectness_cost: Matcher weight on the negative objectness.
        giou_weight: Weight of the GIoU term in the total; $0$ (the published recipe) keeps the GIoU in the
            matcher only. The rotated-box GIoU (scenes with non-zero headings) is computed without gradients,
            so a non-zero weight trains only axis-aligned scenes. Its values differ slightly from
            implementations that skip the polygon intersection of box pairs whose corner-spanned axis-aligned
            rectangles do not overlap: `box3d_overlap` intersects every pair exactly.
        sem_cls_weight: Weight of the semantic-classification term in the total.
        no_object_weight: Cross-entropy weight of the background class.
        angle_cls_weight: Weight of the heading-bin classification term in the total.
        angle_reg_weight: Weight of the heading-residual regression term in the total.
        center_weight: Weight of the center term in the total.
        size_weight: Weight of the size term in the total.
    """

    semcls_weights: Tensor

    def __init__(
        self,
        num_classes: int,
        *,
        num_heading_bins: int,
        matcher_cls_cost: float = 1.0,
        matcher_giou_cost: float = 2.0,
        matcher_center_cost: float = 0.0,
        matcher_objectness_cost: float = 0.0,
        giou_weight: float = 0.0,
        sem_cls_weight: float = 1.0,
        no_object_weight: float = 0.2,
        angle_cls_weight: float = 0.1,
        angle_reg_weight: float = 0.5,
        center_weight: float = 5.0,
        size_weight: float = 1.0,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.num_heading_bins = num_heading_bins
        self.matcher_cls_cost = matcher_cls_cost
        self.matcher_giou_cost = matcher_giou_cost
        self.matcher_center_cost = matcher_center_cost
        self.matcher_objectness_cost = matcher_objectness_cost
        self.giou_weight = giou_weight
        self.sem_cls_weight = sem_cls_weight
        self.angle_cls_weight = angle_cls_weight
        self.angle_reg_weight = angle_reg_weight
        self.center_weight = center_weight
        self.size_weight = size_weight

        semcls_weights = torch.ones(num_classes + 1)
        semcls_weights[-1] = no_object_weight
        self.register_buffer("semcls_weights", semcls_weights)

    def forward(self, output: "ThreeDETRTrainOutput", data: Dict[str, Any]) -> Dict[str, Tensor]:
        r"""Compute the 3DETR set-prediction loss and its terms.

        Args:
            output: A training-mode `ThreeDETRTrainOutput`: `aux_outputs` (a per-decoder-layer list of head
                dicts with `sem_cls_logits`, `sem_cls_prob`, `objectness_prob`, `center_normalized`,
                `center_unnormalized`, `size_normalized`, `size_unnormalized`, `angle_logits`,
                `angle_residual_normalized`, `angle_continuous`) and `point_cloud_dims`.
            data: Packed ground truth: `DataKeys.BOX` $(K, 7)$ full-extent boxes with counter-clockwise
                headings, `DataKeys.LABEL` $(K,)$ per-box classes and `DataKeys.BATCH_BOX` $(K,)$ per-box
                scene index.

        Returns:
            A dict with the scalar `loss` (summed over decoder layers) and detached `sem_cls_loss`,
            `center_loss`, `size_loss`, `angle_cls_loss`, `angle_reg_loss`, `giou_loss` and the
            `cardinality_error` diagnostic (each summed over decoder layers).
        """
        point_cloud_dims = output["point_cloud_dims"]
        layers: List[Dict[str, Tensor]] = output["aux_outputs"]
        targets = self._densify(data, point_cloud_dims)

        nactual = targets.nactual.tolist()
        has_gt = sum(nactual) > 0
        rotated = has_gt and bool(torch.any(targets.angle * targets.present != 0))
        num_boxes = point_cloud_dims[0].new_tensor(float(max(sum(nactual), 1)))

        gious = self._giou3d_layers(layers, targets, rotated)
        with torch.no_grad():
            center_dists = [torch.cdist(layer["center_normalized"], targets.center_normalized, p=1) for layer in layers]
            assignments = self._match(layers, center_dists, gious, targets, nactual)

        total = point_cloud_dims[0].new_zeros(())
        components = {name: point_cloud_dims[0].new_zeros(()) for name in _COMPONENT_NAMES}
        for layer, layer_gious, assignment in zip(layers, gious, assignments):
            layer_losses = self._layer_loss(layer, targets, layer_gious, assignment, num_boxes, has_gt)
            total = total + (
                self.giou_weight * layer_losses["giou_loss"]
                + self.sem_cls_weight * layer_losses["sem_cls_loss"]
                + self.angle_cls_weight * layer_losses["angle_cls_loss"]
                + self.angle_reg_weight * layer_losses["angle_reg_loss"]
                + self.center_weight * layer_losses["center_loss"]
                + self.size_weight * layer_losses["size_loss"]
            )
            for name, weight in _COMPONENT_WEIGHTS.items():
                components[name] = components[name] + getattr(self, weight) * layer_losses[name]
            components["cardinality_error"] = components["cardinality_error"] + layer_losses["cardinality_error"]

        result: Dict[str, Tensor] = {"loss": total}
        for name in _COMPONENT_NAMES:
            result[name] = components[name].detach()
        return result

    def _densify(self, data: Dict[str, Any], point_cloud_dims: Tuple[Tensor, Tensor]) -> _Targets:
        r"""Pad the packed GT to per-scene tensors in normalized and metric frames.

        The packed boxes are full-extent $(K, 7)$ rows with counter-clockwise headings; the headings are
        negated into the model's native heading space before binning, so the pretrained heading head keeps
        its meaning. Boxes keep their packed order within a scene.
        """
        box: Tensor = data[DataKeys.BOX]
        cls: Tensor = data[DataKeys.LABEL].long()
        box_batch: Tensor = data[DataKeys.BATCH_BOX]
        lo, hi = point_cloud_dims
        batch_size = lo.shape[0]
        device = lo.device

        counts = torch.bincount(box_batch, minlength=batch_size)
        max_obj = max(int(counts.max()) if box.shape[0] else 0, 1)
        center = box.new_zeros(batch_size, max_obj, 3)
        size = box.new_zeros(batch_size, max_obj, 3)
        angle = box.new_zeros(batch_size, max_obj)
        label = torch.zeros(batch_size, max_obj, dtype=torch.long, device=device)
        present = box.new_zeros(batch_size, max_obj)

        if box.shape[0]:
            # The slot of a box is its rank among the boxes of its scene, in packed order.
            order = torch.argsort(box_batch, stable=True)
            starts = counts.cumsum(0) - counts
            slot = torch.empty_like(box_batch)
            slot[order] = torch.arange(box.shape[0], device=box_batch.device) - starts[box_batch[order]]

            center[box_batch, slot] = box[:, :3]
            size[box_batch, slot] = box[:, 3:6]
            angle[box_batch, slot] = -box[:, 6]
            label[box_batch, slot] = cls
            present[box_batch, slot] = 1.0

        scene_scale = (hi - lo).clamp(min=1e-1)
        center_normalized = (center - lo.unsqueeze(1)) / (hi - lo).unsqueeze(1)
        size_normalized = size / scene_scale.unsqueeze(1)
        angle_class, angle_residual = angle_to_class(angle, self.num_heading_bins)
        angle_residual_normalized = angle_residual / (math.pi / self.num_heading_bins)

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
        gious: Tensor,
        assignment: Tensor,
        num_boxes: Tensor,
        has_gt: bool,
    ) -> Dict[str, Tensor]:
        r"""Every raw (unweighted) term of one decoder layer, given its query-to-box `assignment` $(B, Q)$."""
        sem_cls_logits = layer["sem_cls_logits"]
        per_prop_gt_inds = assignment.clamp(min=0)
        matched_mask = (assignment >= 0).to(sem_cls_logits.dtype)

        gt_box_label = torch.gather(targets.label, 1, per_prop_gt_inds)
        gt_box_label = gt_box_label.masked_fill(assignment < 0, self.num_classes)
        sem_cls_loss = F.cross_entropy(
            sem_cls_logits.transpose(2, 1), gt_box_label, self.semcls_weights, reduction="mean"
        )

        gt_center = torch.gather(targets.center_normalized, 1, per_prop_gt_inds.unsqueeze(-1).expand(-1, -1, 3))
        center_loss = (layer["center_normalized"] - gt_center).abs().sum(dim=-1)
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
            "sem_cls_loss": sem_cls_loss,
            "center_loss": center_loss,
            "size_loss": size_loss,
            "angle_cls_loss": angle_cls_loss,
            "angle_reg_loss": angle_reg_loss,
            "giou_loss": giou_loss,
            "cardinality_error": cardinality,
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
        residual = residual_for_gt - gt_residual
        angle_reg = F.huber_loss(residual, torch.zeros_like(residual), delta=1.0, reduction="none")
        angle_reg_loss = (angle_reg * matched_mask).sum() / num_boxes
        return angle_cls_loss, angle_reg_loss

    @torch.no_grad()
    def _match(
        self,
        layers: List[Dict[str, Tensor]],
        center_dists: List[Tensor],
        gious: List[Tensor],
        targets: _Targets,
        nactual: List[int],
    ) -> List[Tensor]:
        r"""Hungarian assignment of the queries of every decoder layer and scene to the ground-truth boxes.

        The cost matrices of all layers are stacked and solved after a single device transfer. Returns one
        $(B, Q)$ long tensor per layer holding the matched box index of each query ($-1$ when unmatched).
        """
        batch_size, num_queries = layers[0]["sem_cls_prob"].shape[:2]
        num_gt = targets.label.shape[1]
        gt_labels = targets.label.unsqueeze(1).expand(batch_size, num_queries, num_gt)

        costs = []
        for layer, center_dist, layer_gious in zip(layers, center_dists, gious):
            class_mat = -torch.gather(layer["sem_cls_prob"], 2, gt_labels)
            objectness_mat = -layer["objectness_prob"].unsqueeze(-1)
            costs.append(
                self.matcher_cls_cost * class_mat
                + self.matcher_objectness_cost * objectness_mat
                + self.matcher_center_cost * center_dist
                + self.matcher_giou_cost * (-layer_gious)
            )

        cost = torch.stack(costs).view(len(layers) * batch_size, num_queries, num_gt)
        assignment = hungarian_match_batched(cost, nactual * len(layers))
        return list(assignment.view(len(layers), batch_size, num_queries))

    def _giou3d_layers(self, layers: List[Dict[str, Tensor]], targets: _Targets, rotated: bool) -> List[Tensor]:
        r"""Pairwise generalized 3D IoU between every query of every layer and every padded ground-truth box.

        Uses the axis-aligned intersection / enclosing volumes for upright boxes (`rotated=False`), and
        the rotated bird's-eye intersection otherwise. Malformed and padded columns are zeroed. Returns one
        $(B, Q, M)$ tensor per layer.
        """
        present = targets.present.unsqueeze(1)
        if rotated:
            return [gious * present for gious in self._giou3d_rotated(layers, targets)]

        gious = []
        for layer in layers:
            layer_gious = self._giou3d_axis_aligned(
                layer["center_unnormalized"],
                layer["size_unnormalized"],
                targets.center_unnormalized,
                targets.size_unnormalized,
            )
            gious.append(layer_gious * present)

        return gious

    @staticmethod
    def _prod3(x: Tensor) -> Tensor:
        r"""Product over a trailing dimension of size $3$ as explicit multiplications (a cheap backward)."""
        return x[..., 0] * x[..., 1] * x[..., 2]

    def _box_volume(self, size: Tensor) -> Tensor:
        r"""Per-box volume $\prod \sqrt{\max(d^2, 10^{-6})}$, floored at $10^{-8}$."""
        return self._prod3(torch.sqrt((size**2).clamp(min=1e-6))).clamp(min=_EPS)

    def _giou3d_axis_aligned(
        self,
        pred_center: Tensor,
        pred_size: Tensor,
        gt_center: Tensor,
        gt_size: Tensor,
    ) -> Tensor:
        r"""Vectorized axis-aligned generalized 3D IoU, $(B, Q, 3) \times (B, M, 3) \to (B, Q, M)$."""
        lo1 = (pred_center - pred_size / 2).unsqueeze(2)
        hi1 = (pred_center + pred_size / 2).unsqueeze(2)
        lo2 = (gt_center - gt_size / 2).unsqueeze(1)
        hi2 = (gt_center + gt_size / 2).unsqueeze(1)

        inter = self._prod3((torch.minimum(hi1, hi2) - torch.maximum(lo1, lo2)).clamp(min=0))
        enclosing = self._prod3(torch.maximum(hi1, hi2) - torch.minimum(lo1, lo2))
        vol1 = self._box_volume(pred_size).unsqueeze(2)
        vol2 = self._box_volume(gt_size).unsqueeze(1)
        return self._giou_from_volumes(inter, enclosing, vol1, vol2)

    @torch.no_grad()
    def _giou3d_rotated(self, layers: List[Dict[str, Tensor]], targets: _Targets) -> List[Tensor]:
        r"""Rotated generalized 3D IoU from the BEV-overlap intersection and corner-AABB enclosing.

        `box3d_overlap` computes the rotated intersection without gradients, so the whole rotated GIoU is
        gradient-free (a partially-detached term would push a biased gradient through the union / enclosing
        volumes only); it feeds the matcher and monitoring, not training. The queries of every layer of a
        scene are intersected with its boxes in one call.
        """
        num_layers = len(layers)
        batch_size, num_queries = layers[0]["center_unnormalized"].shape[:2]
        num_gt = targets.center_unnormalized.shape[1]
        gious = targets.center_unnormalized.new_zeros(num_layers, batch_size, num_queries, num_gt)

        for b in range(batch_size):
            # The queries of every layer of the scene, stacked along the rows: (L * Q, 7).
            layer_boxes = []
            for layer in layers:
                center = layer["center_unnormalized"][b]
                size = layer["size_unnormalized"][b]
                angle = layer["angle_continuous"][b].unsqueeze(-1)
                layer_boxes.append(torch.cat([center, size, angle], dim=-1))
            pred_boxes = torch.cat(layer_boxes, dim=0)
            gt_angle = targets.angle[b].unsqueeze(-1)
            gt_boxes = torch.cat([targets.center_unnormalized[b], targets.size_unnormalized[b], gt_angle], dim=-1)

            corners1 = box_corners(pred_boxes)
            corners2 = box_corners(gt_boxes)
            inter, _ = box3d_overlap(corners1, corners2)

            lo1, hi1 = corners1.amin(dim=1), corners1.amax(dim=1)
            lo2, hi2 = corners2.amin(dim=1), corners2.amax(dim=1)
            enclosing = self._prod3(
                torch.maximum(hi1.unsqueeze(1), hi2.unsqueeze(0)) - torch.minimum(lo1.unsqueeze(1), lo2.unsqueeze(0))
            )
            vol1 = self._box_volume(pred_boxes[:, 3:6]).unsqueeze(1)
            vol2 = self._box_volume(gt_boxes[:, 3:6]).unsqueeze(0)
            gious[:, b] = self._giou_from_volumes(inter, enclosing, vol1, vol2).view(num_layers, num_queries, num_gt)

        return list(gious)

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
    "sem_cls_loss",
    "center_loss",
    "size_loss",
    "angle_cls_loss",
    "angle_reg_loss",
    "giou_loss",
    "cardinality_error",
)

_COMPONENT_WEIGHTS = {
    "sem_cls_loss": "sem_cls_weight",
    "center_loss": "center_weight",
    "size_loss": "size_weight",
    "angle_cls_loss": "angle_cls_weight",
    "angle_reg_loss": "angle_reg_weight",
    "giou_loss": "giou_weight",
}
