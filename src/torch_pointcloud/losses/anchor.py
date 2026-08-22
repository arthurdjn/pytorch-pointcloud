r"""Anchor-based detection losses for the voxel detectors (SECOND, PointPillars).

- [`AnchorLoss`][torch_pointcloud.losses.anchor.AnchorLoss]: the single-group head loss used by the
  KITTI SECOND / PointPillars detectors (focal cls, sine-difference smooth-$L_1$ box, direction bin).
- [`MultiHeadAnchorLoss`][torch_pointcloud.losses.anchor.MultiHeadAnchorLoss]: the separate-multihead
  loss used by the nuScenes detectors (per-head focal cls, sincos + velocity $L_1$ box).
"""

from typing import Any, Dict, List, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from torch_pointcloud.layers.anchors import AnchorHeadMultiOutput, assign_anchor_targets, generate_anchors
from torch_pointcloud.utils.box3d import limit_period
from torch_pointcloud.utils.data import DataKeys


def sigmoid_focal_loss(preds: Tensor, targets: Tensor, weights: Tensor, *, alpha: float, gamma: float) -> Tensor:
    r"""Anchor-wise weighted sigmoid focal loss (no reduction).

    The classification primitive shared by the anchor heads: sigmoid focal cross-entropy between per-class
    logits and their one-hot targets, scaled by a per-anchor weight.

    Args:
        preds: Per-class logits, shape $(B, A, C)$.
        targets: One-hot foreground targets, shape $(B, A, C)$.
        weights: Per-anchor weights, shape $(B, A)$.
        alpha: Positive/negative balance.
        gamma: Focusing exponent.

    Returns:
        The weighted per-element loss, shape $(B, A, C)$.

    Shape:
        - preds: $(B, A, C)$
        - targets: $(B, A, C)$
        - weights: $(B, A)$
        - output: $(B, A, C)$

    Example:
        ```pycon
        >>> preds = torch.zeros(1, 2, 3)
        >>> targets = torch.tensor([[[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]]])
        >>> weights = torch.ones(1, 2)
        >>> sigmoid_focal_loss(preds, targets, weights, alpha=0.25, gamma=2.0).shape
        torch.Size([1, 2, 3])

        ```
    """
    pred_sigmoid = preds.sigmoid()
    alpha_weight = targets * alpha + (1.0 - targets) * (1.0 - alpha)
    pt = targets * (1.0 - pred_sigmoid) + (1.0 - targets) * pred_sigmoid
    focal_weight = alpha_weight * pt.pow(gamma)
    bce = preds.clamp(min=0) - preds * targets + torch.log1p(torch.exp(-preds.abs()))
    return focal_weight * bce * weights.unsqueeze(-1)


def one_hot_foreground(box_cls_labels: Tensor, num_classes: int) -> Tensor:
    r"""One-hot encode per-anchor class labels, dropping the background column.

    Ignored ($-1$) and background ($0$) anchors map to an all-zero row; a foreground anchor with label
    $\ell \ge 1$ maps to a one-hot row on class $\ell - 1$.

    Args:
        box_cls_labels: Per-anchor class labels ($-1$ ignore, $0$ background, $\ge 1$ foreground), shape $(B, A)$.
        num_classes: Number of foreground classes.

    Returns:
        One-hot foreground targets, shape $(B, A, C)$.

    Shape:
        - box_cls_labels: $(B, A)$
        - output: $(B, A, C)$

    Example:
        ```pycon
        >>> one_hot_foreground(torch.tensor([[2, 0, -1]]), 3).tolist()
        [[[0.0, 1.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]]

        ```
    """
    cared = box_cls_labels >= 0
    cls_targets = box_cls_labels * cared.to(box_cls_labels.dtype)
    one_hot = torch.zeros(*box_cls_labels.shape, num_classes + 1, dtype=torch.float32, device=box_cls_labels.device)
    one_hot.scatter_(-1, cls_targets.unsqueeze(-1), 1.0)
    return one_hot[..., 1:]


class AnchorLoss(nn.Module):
    r"""Single-stage anchor detection loss (classification, box regression, direction).

    Reference: :arxiv: [Yan et al., 2018](https://www.mdpi.com/1424-8220/18/10/3337).

    The loss of the single-group anchor head used by SECOND and PointPillars. Per scene each class's
    axis-aligned anchors are matched to that class's ground-truth boxes
    ([`assign_anchor_targets`][torch_pointcloud.layers.anchors.assign_anchor_targets]) using per-class
    IoU thresholds, giving per-anchor class labels ($-1$ ignore, $0$ background, $\ge 1$ foreground) and
    residual box targets. Three terms are then summed:

    - **Classification:** sigmoid focal loss over one-hot foreground labels, weighted so ignored anchors
      contribute nothing and each scene is normalized by its positive count.
    - **Box regression:** code-weighted smooth-$L_1$ of the residual encodings, with the heading channel
      replaced by the sine-difference encoding $\sin(\theta_p)\cos(\theta_g)$ vs $\cos(\theta_p)\sin(\theta_g)$
      so the smooth-$L_1$ acts on $\sin(\theta_p - \theta_g)$.
    - **Direction:** weighted softmax cross-entropy over the discretized heading bin.

    Anchors are rebuilt in the constructor from the same geometry the head uses
    ([`generate_anchors`][torch_pointcloud.layers.anchors.generate_anchors]); the loss holds no
    reference to the model.

    Args:
        num_classes: Number of foreground classes.
        voxel_size: Voxel size $(v_x, v_y, v_z)$ (used with `point_cloud_range` to size the anchor grid).
        point_cloud_range: Range $(x_\min, y_\min, z_\min, x_\max, y_\max, z_\max)$.
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
        match_height: Match anchors to boxes by 3D IoU when `True`, otherwise bird's-eye IoU.
    """

    anchors: Tensor
    anchor_class_ids: Tensor
    code_weights: Tensor

    def __init__(
        self,
        num_classes: int,
        *,
        voxel_size: Sequence[float],
        point_cloud_range: Sequence[float],
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

    def forward(self, output: Dict[str, Tensor], batch: Dict[str, Any]) -> Dict[str, Tensor]:
        r"""Compute the anchor detection loss and its components.

        Args:
            output: The head's raw output: `cls` $(B, H, W, A_\text{loc} \cdot C)$,
                `box` $(B, H, W, A_\text{loc} \cdot 7)$ and `dir_cls` $(B, H, W, A_\text{loc} \cdot 2)$.
            batch: Ground truth: packed `box` $(K, 7)$ full-extent, `label` $(K,)$ ($0$-based classes)
                and `batch_box` $(K,)$ per-box scene index.

        Returns:
            A dict with the scalar `loss` (to backprop) and detached `cls_loss`, `box_loss`, `dir_loss`.
        """
        num_anchors = self.anchors.shape[0]
        batch_size = output["cls"].shape[0]
        cls_preds = output["cls"].view(batch_size, num_anchors, self.num_classes)
        box_preds = output["box"].view(batch_size, num_anchors, 7)
        dir_preds = output["dir_cls"].view(batch_size, num_anchors, self.num_dir_bins)

        box_cls_labels, box_reg_targets = self._assign_targets(batch, batch_size)

        positives = box_cls_labels > 0
        cared = box_cls_labels >= 0
        pos_normalizer = positives.sum(dim=1, keepdim=True).to(box_preds.dtype).clamp(min=1.0)
        cls_weights = cared.to(box_preds.dtype) / pos_normalizer
        reg_weights = positives.to(box_preds.dtype) / pos_normalizer

        cls_loss = self.cls_weight * self._cls_loss(cls_preds, box_cls_labels, cls_weights)
        box_loss = self.loc_weight * self._box_loss(box_preds, box_reg_targets, reg_weights)
        dir_loss = self.dir_weight * self._dir_loss(dir_preds, box_reg_targets, reg_weights)
        total = cls_loss + box_loss + dir_loss

        return {
            "loss": total,
            "cls_loss": cls_loss.detach(),
            "box_loss": box_loss.detach(),
            "dir_loss": dir_loss.detach(),
        }

    def _assign_targets(self, batch: Dict[str, Any], batch_size: int) -> Tuple[Tensor, Tensor]:
        r"""Densify the packed ground truth and assign per-class anchor targets for every scene.

        Returns per-anchor class labels $(B, A)$ ($-1$ ignore, $0$ background, $\ge 1$ foreground) and
        residual box targets $(B, A, 7)$, in the head's flat anchor order.
        """
        anchors = self.anchors
        gt_boxes: Tensor = batch[DataKeys.BOX]
        gt_labels: Tensor = batch[DataKeys.LABEL].long() + 1  # 1-based foreground for `assign_anchor_targets`
        gt_batch: Tensor = batch[DataKeys.BATCH_BOX]

        cls_labels = anchors.new_full((batch_size, anchors.shape[0]), -1, dtype=torch.long)
        reg_targets = anchors.new_zeros((batch_size, anchors.shape[0], 7))
        class_masks = [self.anchor_class_ids == c for c in range(self.num_classes)]

        for b in range(batch_size):
            scene = gt_batch == b
            scene_boxes = gt_boxes[scene]
            scene_labels = gt_labels[scene]
            for c, class_mask in enumerate(class_masks):
                anchor_idx = class_mask.nonzero(as_tuple=False).squeeze(1)
                gt_c = scene_labels == (c + 1)
                targets = assign_anchor_targets(
                    anchors[anchor_idx],
                    scene_boxes[gt_c],
                    scene_labels[gt_c],
                    matched_threshold=self.matched_thresholds[c],
                    unmatched_threshold=self.unmatched_thresholds[c],
                    match_height=self.match_height,
                )
                cls_labels[b, anchor_idx] = targets["cls_labels"]
                reg_targets[b, anchor_idx] = targets["box_reg_targets"]

        return cls_labels, reg_targets

    def _cls_loss(self, cls_preds: Tensor, box_cls_labels: Tensor, cls_weights: Tensor) -> Tensor:
        r"""Sigmoid focal classification loss over one-hot foreground labels."""
        one_hot = one_hot_foreground(box_cls_labels, self.num_classes)
        loss = sigmoid_focal_loss(cls_preds, one_hot, cls_weights, alpha=self.focal_alpha, gamma=self.focal_gamma)
        return loss.sum() / cls_preds.shape[0]

    def _box_loss(self, box_preds: Tensor, box_reg_targets: Tensor, reg_weights: Tensor) -> Tensor:
        r"""Code-weighted smooth-$L_1$ box-regression loss with sine-difference heading encoding."""
        pred_sin = torch.sin(box_preds[..., 6:7]) * torch.cos(box_reg_targets[..., 6:7])
        target_sin = torch.cos(box_preds[..., 6:7]) * torch.sin(box_reg_targets[..., 6:7])
        box_preds = torch.cat([box_preds[..., :6], pred_sin], dim=-1)
        box_reg_targets = torch.cat([box_reg_targets[..., :6], target_sin], dim=-1)

        diff = (box_preds - box_reg_targets) * self.code_weights.view(1, 1, -1)
        n = diff.abs()
        beta = self.smooth_l1_beta
        smooth_l1 = torch.where(n < beta, 0.5 * n**2 / beta, n - 0.5 * beta)
        loss = smooth_l1 * reg_weights.unsqueeze(-1)
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


class MultiHeadAnchorLoss(nn.Module):
    r"""Separate-multihead anchor detection loss (per-head classification, sincos + velocity box regression).

    Reference: :arxiv: [Zhu et al., 2019](https://arxiv.org/abs/1908.09492).

    The loss of the separate-multihead anchor head used by the nuScenes SECOND and PointPillars detectors,
    where several RPN heads each own a disjoint class group over a shared feature map. Per scene each
    class's axis-aligned anchors are matched to that class's ground-truth boxes
    ([`assign_anchor_targets`][torch_pointcloud.layers.anchors.assign_anchor_targets]) using per-class IoU
    thresholds, giving per-anchor class labels ($-1$ ignore, $0$ background, $\ge 1$ foreground) and residual
    box targets. Two terms are summed:

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
    ([`generate_anchors`][torch_pointcloud.layers.anchors.generate_anchors]), in the head's class-group
    order; the loss holds no reference to the model.

    Args:
        num_classes: Number of foreground classes (10 for nuScenes).
        class_groups: Class-index groups, one per RPN head (e.g. `[[0], [1, 2], ...]`), matching the head's
            `head_class_groups`; the classes in each group share one head, and the flattened groups must
            enumerate the classes $0 \ldots C - 1$ in ascending order (the anchor / head layout).
        voxel_size: Voxel size $(v_x, v_y, v_z)$ (used with `point_cloud_range` to size the anchor grid).
        point_cloud_range: Range $(x_\min, y_\min, z_\min, x_\max, y_\max, z_\max)$.
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
        match_height: Match anchors to boxes by 3D IoU when `True`, otherwise bird's-eye IoU.
        encode_angle_by_sincos: Encode the heading residual as $(\cos, \sin)$ (always `True` for this head).
    """

    anchors: Tensor
    code_weights: Tensor

    def __init__(
        self,
        num_classes: int,
        *,
        class_groups: Sequence[Sequence[int]],
        voxel_size: Sequence[float],
        point_cloud_range: Sequence[float],
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
            raise ValueError("`MultiHeadAnchorLoss` only supports the sincos angle encoding.")
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

    def forward(self, output: AnchorHeadMultiOutput, batch: Dict[str, Any]) -> Dict[str, Tensor]:
        r"""Compute the multihead anchor detection loss and its components.

        Args:
            output: The head's raw output: per-head `cls` $(B, A_g, C_g)$ and `box` $(B, A_g, 10)$ lists,
                plus `multihead_label_mapping` (per-head 1-based global class indices).
            batch: Ground truth: packed `box` $(K, 7)$ full-extent, `label` $(K,)$ ($0$-based classes) and
                `batch_box` $(K,)$ per-box scene index.

        Returns:
            A dict with the scalar `loss` (to backprop) and detached `cls_loss`, `box_loss`, `dir_loss`; the
            separate-multihead head carries no direction classifier, so `dir_loss` is always zero.
        """
        cls_preds = output["cls"]
        box_preds = output["box"]
        label_mapping = output["multihead_label_mapping"]
        batch_size = cls_preds[0].shape[0]

        box_cls_labels, box_reg_targets = self._assign_targets(batch, batch_size)

        cls_loss = self.cls_weight * self._cls_loss(cls_preds, box_cls_labels, label_mapping)
        box_loss = self.loc_weight * self._box_loss(box_preds, box_reg_targets, box_cls_labels)
        dir_loss = torch.zeros_like(cls_loss)
        total = cls_loss + box_loss

        return {
            "loss": total,
            "cls_loss": cls_loss.detach(),
            "box_loss": box_loss.detach(),
            "dir_loss": dir_loss,
        }

    def _assign_targets(self, batch: Dict[str, Any], batch_size: int) -> Tuple[Tensor, Tensor]:
        r"""Densify the packed ground truth and assign per-class anchor targets for every scene.

        Returns per-anchor class labels $(B, A)$ ($-1$ ignore, $0$ background, $\ge 1$ foreground) and
        sincos + velocity residual box targets $(B, A, 10)$, in the head's class-group anchor order.
        """
        anchors = self.anchors
        gt_boxes: Tensor = batch[DataKeys.BOX]
        gt_labels: Tensor = batch[DataKeys.LABEL].long() + 1  # 1-based foreground for `assign_anchor_targets`
        gt_batch: Tensor = batch[DataKeys.BATCH_BOX]

        cls_labels = anchors.new_full((batch_size, anchors.shape[0]), -1, dtype=torch.long)
        reg_targets = anchors.new_zeros((batch_size, anchors.shape[0], self.code_size))

        for b in range(batch_size):
            scene = gt_batch == b
            scene_boxes = gt_boxes[scene]
            scene_labels = gt_labels[scene]
            start = 0
            for c, count in enumerate(self.class_counts):
                idx = slice(start, start + count)
                class_anchors = anchors[idx]
                gt_c = scene_labels == (c + 1)
                targets = assign_anchor_targets(
                    class_anchors,
                    scene_boxes[gt_c],
                    scene_labels[gt_c],
                    matched_threshold=self.matched_thresholds[c],
                    unmatched_threshold=self.unmatched_thresholds[c],
                    match_height=self.match_height,
                )
                cls_labels[b, idx] = targets["cls_labels"]
                reg_targets[b, idx] = self._encode_sincos(targets["box_reg_targets"], class_anchors)
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
        gt_yaw = box_reg_targets[:, 6] + anchor_yaw
        cos_diff = (torch.cos(gt_yaw) - torch.cos(anchor_yaw)).unsqueeze(-1)
        sin_diff = (torch.sin(gt_yaw) - torch.sin(anchor_yaw)).unsqueeze(-1)
        velocity = box_reg_targets.new_zeros((box_reg_targets.shape[0], self.num_velocity))
        return torch.cat([box_reg_targets[:, :6], cos_diff, sin_diff, velocity], dim=-1)

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
