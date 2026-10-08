r"""Losses of the anchor-based BEV detection heads (SECOND, PointPillars and the multi-group nuScenes variants).

Each loss reads the output dict of its head, rebuilds the anchors from the head geometry, assigns the targets with
[`assign_anchor_targets`][torch_pointcloud.ops.anchors.assign_anchor_targets] and combines the
[`functional`][torch_pointcloud.losses.functional] primitives.
"""

from typing import Any, Dict, List, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from torch_pointcloud.layers.anchors import AnchorHeadMultiOutput, AnchorHeadOutput
from torch_pointcloud.losses.focal import one_hot_foreground, sigmoid_focal_loss
from torch_pointcloud.ops.anchors import assign_anchor_targets, generate_anchors
from torch_pointcloud.ops.box3d import limit_period
from torch_pointcloud.utils.data import DataKeys


class AnchorHeadLoss(nn.Module):
    r"""Loss of the single-group [`AnchorHead`][torch_pointcloud.layers.anchors.AnchorHead] (SECOND, PointPillars).

    Reference: :arxiv: [SECOND: Sparsely Embedded Convolutional Detection](https://www.mdpi.com/1424-8220/18/10/3337) (Yan et al., 2018).

    Per scene each class's axis-aligned anchors are matched to that class's ground-truth boxes
    ([`assign_anchor_targets`][torch_pointcloud.ops.anchors.assign_anchor_targets]) using per-class
    IoU thresholds, giving per-anchor class labels ($-1$ ignore, $0$ background, $\ge 1$ foreground) and
    residual box targets. Three terms are then summed:

    - **Classification:** sigmoid focal loss over one-hot foreground labels, weighted so ignored anchors
      contribute nothing and each scene is normalized by its positive count.
    - **Box regression:** code-weighted smooth-$L_1$ of the residual encodings, with the heading residual
      replaced by $\sin(\theta_p - \theta_g)$ (the sine-difference encoding) so a half-turn error is not
      penalized as a position error.
    - **Direction:** weighted softmax cross-entropy over the discretized heading bin.

    Anchors are rebuilt in the constructor from the same geometry the head uses
    ([`generate_anchors`][torch_pointcloud.ops.anchors.generate_anchors]); the loss holds no reference
    to the model.

    Args:
        num_classes: Number of foreground classes.
        point_cloud_range: Range $(x_\min, y_\min, z_\min, x_\max, y_\max, z_\max)$.
        voxel_size: Voxel size $(v_x, v_y, v_z)$ (used with `point_cloud_range` to size the anchor grid).
        anchor_sizes: Per-class box size $(d_x, d_y, d_z)$, one row per class.
        anchor_bottom_heights: Per-class anchor bottom $z$, one per class.
        feature_map_stride: BEV feature-map stride of the head.
        matched_thresholds: Per-class IoU at or above which an anchor is a positive.
        unmatched_thresholds: Per-class IoU below which an anchor is background.
        anchor_rotations: Yaw angles (radians) shared by all classes.
        code_weights: Per-code regression weights, shape $(7,)$.
        cls_weight: Weight of the classification term in the total.
        loc_weight: Weight of the box-regression term in the total.
        dir_weight: Weight of the direction term in the total.
        num_dir_bins: Number of direction bins.
        dir_offset: Direction-target angle offset.
        dir_limit_offset: Offset used when wrapping the heading before binning.
        focal_alpha: Focal-loss positive/negative balance.
        focal_gamma: Focal-loss focusing exponent.
        smooth_l1_beta: Smooth-$L_1$ transition point $\beta$.
        match_height: Match anchors to boxes by the rotated 3D IoU when `True`, otherwise by the bird's-eye IoU
            of the boxes snapped to their nearest axis-aligned orientation.
    """

    anchors: Tensor
    anchor_class_ids: Tensor
    code_weights: Tensor

    def __init__(
        self,
        num_classes: int,
        *,
        point_cloud_range: Sequence[float],
        voxel_size: Sequence[float],
        anchor_sizes: Sequence[Sequence[float]],
        anchor_bottom_heights: Sequence[float],
        feature_map_stride: int,
        matched_thresholds: Sequence[float],
        unmatched_thresholds: Sequence[float],
        anchor_rotations: Sequence[float] = (0.0, 1.57),
        code_weights: Sequence[float] = (1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
        cls_weight: float = 1.0,
        loc_weight: float = 2.0,
        dir_weight: float = 0.2,
        num_dir_bins: int = 2,
        dir_offset: float = 0.78539,
        dir_limit_offset: float = 0.0,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        smooth_l1_beta: float = 1.0 / 9.0,
        match_height: bool = False,
    ) -> None:
        super().__init__()
        if len(anchor_sizes) != num_classes or len(anchor_bottom_heights) != num_classes:
            raise ValueError("`anchor_sizes` and `anchor_bottom_heights` must have one entry per class.")
        if len(matched_thresholds) != num_classes or len(unmatched_thresholds) != num_classes:
            raise ValueError("`matched_thresholds` and `unmatched_thresholds` must have one entry per class.")

        self.num_classes = num_classes
        self.matched_thresholds = tuple(matched_thresholds)
        self.unmatched_thresholds = tuple(unmatched_thresholds)
        self.cls_weight = cls_weight
        self.loc_weight = loc_weight
        self.dir_weight = dir_weight
        self.num_dir_bins = num_dir_bins
        self.dir_offset = dir_offset
        self.dir_limit_offset = dir_limit_offset
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.smooth_l1_beta = smooth_l1_beta
        self.match_height = match_height
        self.num_rot = len(anchor_rotations)

        grid = [int(round((point_cloud_range[i + 3] - point_cloud_range[i]) / voxel_size[i])) for i in range(3)]
        feature_map_size = (grid[0] // feature_map_stride, grid[1] // feature_map_stride)
        per_class_anchors = [
            generate_anchors(point_cloud_range, feature_map_size, [size], anchor_rotations, [bottom])
            for size, bottom in zip(anchor_sizes, anchor_bottom_heights)
        ]
        anchors = torch.cat(per_class_anchors, dim=-3).view(-1, 7)
        anchor_class_ids = (torch.arange(anchors.shape[0]) // self.num_rot) % num_classes

        self.register_buffer("anchors", anchors, persistent=False)
        self.register_buffer("anchor_class_ids", anchor_class_ids, persistent=False)
        self.register_buffer("code_weights", torch.tensor(list(code_weights), dtype=torch.float32), persistent=False)

    def forward(self, output: AnchorHeadOutput, data: Dict[str, Any]) -> Dict[str, Tensor]:
        r"""Compute the anchor detection loss and its terms.

        Args:
            output: The head's raw output: `cls` $(B, H, W, A_\text{loc} \cdot C)$,
                `box` $(B, H, W, A_\text{loc} \cdot 7)$ and `dir_cls` $(B, H, W, A_\text{loc} \cdot 2)$.
            data: Ground truth: packed `box` $(K, 7)$ full-extent, `label` $(K,)$ ($0$-based classes)
                and `batch_box` $(K,)$ per-box scene index.

        Returns:
            A dict with the scalar `loss` and detached `cls_loss`, `box_loss`, `dir_loss`.
        """
        num_anchors = self.anchors.shape[0]
        batch_size = output["cls"].shape[0]
        cls_preds = output["cls"].view(batch_size, num_anchors, self.num_classes)
        box_preds = output["box"].view(batch_size, num_anchors, 7)
        dir_preds = output["dir_cls"].view(batch_size, num_anchors, self.num_dir_bins)

        box_cls_labels, box_reg_targets = self._assign_targets(data, batch_size)

        positives = box_cls_labels > 0
        cared = box_cls_labels >= 0
        pos_normalizer = positives.sum(dim=1, keepdim=True).to(box_preds.dtype).clamp(min=1.0)
        cls_weights = cared.to(box_preds.dtype) / pos_normalizer
        reg_weights = positives.to(box_preds.dtype) / pos_normalizer

        cls_loss = self.cls_weight * self._cls_loss(cls_preds, box_cls_labels, cls_weights)
        box_loss = self.loc_weight * self._box_loss(box_preds, box_reg_targets, reg_weights)
        dir_loss = self.dir_weight * self._dir_loss(dir_preds, box_reg_targets, reg_weights)

        return {
            "loss": cls_loss + box_loss + dir_loss,
            "cls_loss": cls_loss.detach(),
            "box_loss": box_loss.detach(),
            "dir_loss": dir_loss.detach(),
        }

    def _assign_targets(self, data: Dict[str, Any], batch_size: int) -> Tuple[Tensor, Tensor]:
        r"""Densify the packed ground truth and assign per-class anchor targets for every scene.

        Returns per-anchor class labels $(B, A)$ ($-1$ ignore, $0$ background, $\ge 1$ foreground) and
        residual box targets $(B, A, 7)$, in the head's flat anchor order.
        """
        anchors = self.anchors
        gt_boxes: Tensor = data[DataKeys.BOX]
        gt_labels: Tensor = data[DataKeys.LABEL].long() + 1  # 1-based foreground for `assign_anchor_targets`
        gt_batch: Tensor = data[DataKeys.BATCH_BOX]

        cls_labels = anchors.new_full((batch_size, anchors.shape[0]), -1, dtype=torch.long)
        reg_targets = anchors.new_zeros((batch_size, anchors.shape[0], 7))
        class_anchors = [(self.anchor_class_ids == c).nonzero(as_tuple=True)[0] for c in range(self.num_classes)]

        for c, anchor_idx in enumerate(class_anchors):
            gt_c = gt_labels == (c + 1)
            targets = assign_anchor_targets(
                anchors[anchor_idx],
                gt_boxes[gt_c],
                gt_labels[gt_c],
                matched_threshold=self.matched_thresholds[c],
                unmatched_threshold=self.unmatched_thresholds[c],
                match_height=self.match_height,
                gt_batch=gt_batch[gt_c],
                batch_size=batch_size,
            )
            cls_labels[:, anchor_idx] = targets["cls_labels"]
            reg_targets[:, anchor_idx] = targets["box_reg_targets"]

        return cls_labels, reg_targets

    def _cls_loss(self, cls_preds: Tensor, box_cls_labels: Tensor, cls_weights: Tensor) -> Tensor:
        r"""Sigmoid focal classification loss over one-hot foreground labels."""
        one_hot = one_hot_foreground(box_cls_labels, self.num_classes)
        loss = sigmoid_focal_loss(cls_preds, one_hot, cls_weights, alpha=self.focal_alpha, gamma=self.focal_gamma)
        return loss.sum() / cls_preds.shape[0]

    def _box_loss(self, box_preds: Tensor, box_reg_targets: Tensor, reg_weights: Tensor) -> Tensor:
        r"""Code-weighted smooth-$L_1$ box-regression loss with sine-difference heading encoding."""
        # Sine-difference heading residual: $\sin\theta_p \cos\theta_g - \cos\theta_p \sin\theta_g = \sin(\theta_p - \theta_g)$.
        pred_sin = torch.sin(box_preds[..., 6:7]) * torch.cos(box_reg_targets[..., 6:7])
        target_sin = torch.cos(box_preds[..., 6:7]) * torch.sin(box_reg_targets[..., 6:7])
        box_preds = torch.cat([box_preds[..., :6], pred_sin], dim=-1)
        box_reg_targets = torch.cat([box_reg_targets[..., :6], target_sin], dim=-1)

        diff = (box_preds - box_reg_targets) * self.code_weights.view(1, 1, -1)
        smooth = F.smooth_l1_loss(diff, torch.zeros_like(diff), beta=self.smooth_l1_beta, reduction="none")
        loss = smooth * reg_weights.unsqueeze(-1)
        return loss.sum() / box_preds.shape[0]

    def _dir_loss(self, dir_preds: Tensor, box_reg_targets: Tensor, reg_weights: Tensor) -> Tensor:
        r"""Weighted softmax cross-entropy over the discretized heading bin."""
        anchors = self.anchors.view(1, -1, 7)
        rot_gt = box_reg_targets[..., 6] + anchors[..., 6]
        period = 2 * torch.pi / self.num_dir_bins
        offset_rot = limit_period(rot_gt - self.dir_offset, 0.0, period * self.num_dir_bins)
        dir_targets = torch.floor(offset_rot / period).long().clamp(min=0, max=self.num_dir_bins - 1)

        ce = F.cross_entropy(dir_preds.permute(0, 2, 1), dir_targets, reduction="none") * reg_weights
        return ce.sum() / dir_preds.shape[0]


class MultiGroupAnchorHeadLoss(nn.Module):
    r"""Loss of the [`MultiGroupAnchorHead`][torch_pointcloud.layers.anchors.MultiGroupAnchorHead] (nuScenes SECOND, PointPillars).

    Reference: :arxiv: [Class-balanced Grouping and Sampling for Point Cloud 3D Object Detection](https://arxiv.org/abs/1908.09492) (Zhu et al., 2019).

    The separate-multihead anchor head has several RPN heads, each owning a disjoint class group over a
    shared feature map. Per scene each class's axis-aligned anchors are matched to that class's ground-truth
    boxes ([`assign_anchor_targets`][torch_pointcloud.ops.anchors.assign_anchor_targets]) using per-class
    IoU thresholds, giving per-anchor class labels ($-1$ ignore, $0$ background, $\ge 1$ foreground) and
    residual box targets. Two terms are summed:

    - **Classification:** per head, sigmoid focal loss over the one-hot labels restricted to that head's
      class columns, with positive / negative anchors weighted by `pos_cls_weight` / `neg_cls_weight` and
      each scene normalized by its total positive count.
    - **Box regression:** per head, code-weighted $L_1$ over the $10$-dim box code
      $(x, y, z, d_x, d_y, d_z, \cos\Delta\theta, \sin\Delta\theta, v_x, v_y)$. The heading is encoded as a
      $(\cos, \sin)$ residual, so no separate direction classifier is used.

    !!! note
        The nuScenes ground-truth boxes carry no velocity ($(K, 7)$), so the velocity targets are zero. Set
        the last two `code_weights` entries to $0$ to leave the velocity branch unsupervised; the default
        does so.

    Anchors are rebuilt in the constructor from the same geometry the head uses
    ([`generate_anchors`][torch_pointcloud.ops.anchors.generate_anchors]), in the head's class-group
    order; the loss holds no reference to the model.

    Args:
        num_classes: Number of foreground classes (10 for nuScenes).
        class_groups: Class-index groups, one per RPN head (e.g. `[[0], [1, 2], ...]`), matching the head's
            `head_class_groups`; the classes in each group share one head, and the flattened groups must
            enumerate the classes $0 \ldots C - 1$ in ascending order (the anchor / head layout).
        point_cloud_range: Range $(x_\min, y_\min, z_\min, x_\max, y_\max, z_\max)$.
        voxel_size: Voxel size $(v_x, v_y, v_z)$ (used with `point_cloud_range` to size the anchor grid).
        anchor_sizes: Per-class box size $(d_x, d_y, d_z)$, one row per class.
        anchor_bottom_heights: Per-class anchor bottom $z$, one per class.
        feature_map_stride: BEV feature-map stride of the head.
        matched_thresholds: Per-class IoU at or above which an anchor is a positive.
        unmatched_thresholds: Per-class IoU below which an anchor is background.
        anchor_rotations: Yaw angles (radians) shared by all classes.
        code_weights: Per-code regression weights, shape $(10,)$; the last two (velocity) default to $0$.
        cls_weight: Weight of the classification term in the total.
        loc_weight: Weight of the box-regression term in the total.
        pos_cls_weight: Classification weight of a positive anchor.
        neg_cls_weight: Classification weight of a background anchor.
        focal_alpha: Focal-loss positive/negative balance.
        focal_gamma: Focal-loss focusing exponent.
        match_height: Match anchors to boxes by the rotated 3D IoU when `True`, otherwise by the bird's-eye IoU
            of the boxes snapped to their nearest axis-aligned orientation.
        encode_angle_by_sincos: Encode the heading residual as $(\cos, \sin)$ (always `True` for this head).
    """

    anchors: Tensor
    code_weights: Tensor

    def __init__(
        self,
        num_classes: int,
        *,
        class_groups: Sequence[Sequence[int]],
        point_cloud_range: Sequence[float],
        voxel_size: Sequence[float],
        anchor_sizes: Sequence[Sequence[float]],
        anchor_bottom_heights: Sequence[float],
        feature_map_stride: int,
        matched_thresholds: Sequence[float],
        unmatched_thresholds: Sequence[float],
        anchor_rotations: Sequence[float] = (0.0, 1.57),
        code_weights: Sequence[float] = (1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0),
        cls_weight: float = 1.0,
        loc_weight: float = 0.25,
        pos_cls_weight: float = 1.0,
        neg_cls_weight: float = 2.0,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        match_height: bool = False,
        encode_angle_by_sincos: bool = True,
    ) -> None:
        super().__init__()
        if len(anchor_sizes) != num_classes or len(anchor_bottom_heights) != num_classes:
            raise ValueError("`anchor_sizes` and `anchor_bottom_heights` must have one entry per class.")
        if len(matched_thresholds) != num_classes or len(unmatched_thresholds) != num_classes:
            raise ValueError("`matched_thresholds` and `unmatched_thresholds` must have one entry per class.")
        if not encode_angle_by_sincos:
            raise ValueError("`MultiGroupAnchorHeadLoss` only supports the sincos angle encoding.")

        code_size = len(code_weights)
        if code_size < 8:
            raise ValueError("`code_weights` must cover at least the 6 center/size and 2 sincos-angle codes.")

        # The anchors and the head's per-head outputs are both laid out class 0..C-1; any other grouping
        # would silently misalign the per-head one-hot / target slices.
        if [c for group in class_groups for c in group] != list(range(num_classes)):
            raise ValueError(
                f"`class_groups` must cover classes 0..{num_classes - 1} exactly once, in ascending order, "
                f"got {[list(g) for g in class_groups]}."
            )

        self.num_classes = num_classes
        self.class_groups = [list(g) for g in class_groups]
        self.matched_thresholds = tuple(matched_thresholds)
        self.unmatched_thresholds = tuple(unmatched_thresholds)
        self.cls_weight = cls_weight
        self.loc_weight = loc_weight
        self.pos_cls_weight = pos_cls_weight
        self.neg_cls_weight = neg_cls_weight
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.match_height = match_height
        self.code_size = code_size
        self.num_velocity = code_size - 8

        grid = [int(round((point_cloud_range[i + 3] - point_cloud_range[i]) / voxel_size[i])) for i in range(3)]
        feature_map_size = (grid[0] // feature_map_stride, grid[1] // feature_map_stride)
        per_class_anchors = []
        class_counts = []
        for size, bottom in zip(anchor_sizes, anchor_bottom_heights):
            cls_anchors = generate_anchors(point_cloud_range, feature_map_size, [size], anchor_rotations, [bottom])
            cls_anchors = cls_anchors.permute(3, 4, 0, 1, 2, 5).reshape(-1, 7)
            per_class_anchors.append(cls_anchors)
            class_counts.append(cls_anchors.shape[0])
        self.class_counts = tuple(class_counts)

        self.register_buffer("anchors", torch.cat(per_class_anchors, dim=0), persistent=False)
        self.register_buffer("code_weights", torch.tensor(list(code_weights), dtype=torch.float32), persistent=False)

    def forward(self, output: AnchorHeadMultiOutput, data: Dict[str, Any]) -> Dict[str, Tensor]:
        r"""Compute the multihead anchor detection loss and its terms.

        Args:
            output: The head's raw output: per-head `cls` $(B, A_g, C_g)$ and `box` $(B, A_g, 10)$ lists,
                plus `multihead_label_mapping` (per-head 1-based global class indices).
            data: Ground truth: packed `box` $(K, 7)$ full-extent, `label` $(K,)$ ($0$-based classes) and
                `batch_box` $(K,)$ per-box scene index.

        Returns:
            A dict with the scalar `loss` and detached `cls_loss`, `box_loss`.
        """
        cls_preds = output["cls"]
        box_preds = output["box"]
        label_mapping = output["multihead_label_mapping"]
        batch_size = cls_preds[0].shape[0]

        box_cls_labels, box_reg_targets = self._assign_targets(data, batch_size)

        cls_loss = self.cls_weight * self._cls_loss(cls_preds, box_cls_labels, label_mapping)
        box_loss = self.loc_weight * self._box_loss(box_preds, box_reg_targets, box_cls_labels)

        return {
            "loss": cls_loss + box_loss,
            "cls_loss": cls_loss.detach(),
            "box_loss": box_loss.detach(),
        }

    def _assign_targets(self, data: Dict[str, Any], batch_size: int) -> Tuple[Tensor, Tensor]:
        r"""Densify the packed ground truth and assign per-class anchor targets for every scene.

        Returns per-anchor class labels $(B, A)$ ($-1$ ignore, $0$ background, $\ge 1$ foreground) and
        sincos + velocity residual box targets $(B, A, 10)$, in the head's class-group anchor order.
        """
        anchors = self.anchors
        gt_boxes: Tensor = data[DataKeys.BOX]
        gt_labels: Tensor = data[DataKeys.LABEL].long() + 1  # 1-based foreground for `assign_anchor_targets`
        gt_batch: Tensor = data[DataKeys.BATCH_BOX]

        cls_labels = anchors.new_full((batch_size, anchors.shape[0]), -1, dtype=torch.long)
        reg_targets = anchors.new_zeros((batch_size, anchors.shape[0], self.code_size))

        start = 0
        for c, count in enumerate(self.class_counts):
            idx = slice(start, start + count)
            class_anchors = anchors[idx]
            gt_c = gt_labels == (c + 1)
            targets = assign_anchor_targets(
                class_anchors,
                gt_boxes[gt_c],
                gt_labels[gt_c],
                matched_threshold=self.matched_thresholds[c],
                unmatched_threshold=self.unmatched_thresholds[c],
                match_height=self.match_height,
                gt_batch=gt_batch[gt_c],
                batch_size=batch_size,
            )
            cls_labels[:, idx] = targets["cls_labels"]
            reg_targets[:, idx] = self._encode_sincos(targets["box_reg_targets"], class_anchors)
            start += count

        return cls_labels, reg_targets

    def _encode_sincos(self, box_reg_targets: Tensor, anchors: Tensor) -> Tensor:
        r"""Rewrite the plain-delta heading of a residual target into a $(\cos, \sin)$ residual plus velocity.

        The center / size residuals are shared with the plain encoding; the heading delta $\theta_g - \theta_a$
        (stored in channel 6, zero for non-positive anchors) is expanded to
        $(\cos\theta_g - \cos\theta_a, \sin\theta_g - \sin\theta_a)$ and the velocity codes are appended as
        zeros (the nuScenes ground truth has no velocity).
        """
        anchor_yaw = anchors[:, 6]
        gt_yaw = box_reg_targets[..., 6] + anchor_yaw
        cos_diff = (torch.cos(gt_yaw) - torch.cos(anchor_yaw)).unsqueeze(-1)
        sin_diff = (torch.sin(gt_yaw) - torch.sin(anchor_yaw)).unsqueeze(-1)
        velocity = box_reg_targets.new_zeros((*box_reg_targets.shape[:-1], self.num_velocity))
        return torch.cat([box_reg_targets[..., :6], cos_diff, sin_diff, velocity], dim=-1)

    def _cls_loss(self, cls_preds: List[Tensor], box_cls_labels: Tensor, label_mapping: List[Tensor]) -> Tensor:
        r"""Per-head sigmoid focal classification loss over the head's class columns."""
        positives = box_cls_labels > 0
        negatives = box_cls_labels == 0
        pos_normalizer = positives.sum(dim=1, keepdim=True).to(self.anchors.dtype).clamp(min=1.0)
        cls_weights = (negatives * self.neg_cls_weight + positives * self.pos_cls_weight) / pos_normalizer

        one_hot = one_hot_foreground(box_cls_labels, self.num_classes)
        batch_size = box_cls_labels.shape[0]
        total = box_cls_labels.new_zeros((), dtype=self.anchors.dtype)
        start = 0
        for head_idx, head_cls in enumerate(cls_preds):
            num_head_anchors = head_cls.shape[1]
            columns = label_mapping[head_idx] - 1
            head_one_hot = one_hot[:, start : start + num_head_anchors][:, :, columns]
            head_weights = cls_weights[:, start : start + num_head_anchors]
            loss = sigmoid_focal_loss(
                head_cls, head_one_hot, head_weights, alpha=self.focal_alpha, gamma=self.focal_gamma
            )
            total = total + loss.sum() / batch_size
            start += num_head_anchors
        return total

    def _box_loss(self, box_preds: List[Tensor], box_reg_targets: Tensor, box_cls_labels: Tensor) -> Tensor:
        r"""Per-head code-weighted $L_1$ box-regression loss over the sincos + velocity box code."""
        positives = box_cls_labels > 0
        pos_normalizer = positives.sum(dim=1, keepdim=True).to(self.anchors.dtype).clamp(min=1.0)
        reg_weights = positives.to(self.anchors.dtype) / pos_normalizer

        batch_size = box_cls_labels.shape[0]
        total = box_cls_labels.new_zeros((), dtype=self.anchors.dtype)
        start = 0
        for head_box in box_preds:
            num_head_anchors = head_box.shape[1]
            head_target = box_reg_targets[:, start : start + num_head_anchors]
            head_weights = reg_weights[:, start : start + num_head_anchors]
            diff = (head_box - head_target) * self.code_weights.view(1, 1, -1)
            loss = diff.abs() * head_weights.unsqueeze(-1)
            total = total + loss.sum() / batch_size
            start += num_head_anchors
        return total
