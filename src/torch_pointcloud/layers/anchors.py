r"""Anchor-based dense detection heads for the voxel detectors (PointPillars, SECOND).

A packed-format port of the anchor head from
:github: [open-mmlab/OpenPCDet](https://github.com/open-mmlab/OpenPCDet).

- [`generate_anchors`][torch_pointcloud.layers.anchors.generate_anchors]: axis-aligned anchor
  generation; residuals are decoded with
  [`decode_box_residuals`][torch_pointcloud.utils.box3d.decode_box_residuals].
- [`AnchorHeadSingle`][torch_pointcloud.layers.anchors.AnchorHeadSingle]: the single-stage anchor
  head (per-anchor class logits, box residuals and a direction bin).
- [`AnchorHeadMulti`][torch_pointcloud.layers.anchors.AnchorHeadMulti]: the multi-group
  separate-head variant (sincos + velocity box code) used by the nuScenes detectors.
- [`separate_branch`][torch_pointcloud.layers.anchors.separate_branch]: the per-attribute
  `SeparateHead` branch builder, also used by the Voxel Mamba center head.

"""

import math
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, TypedDict, Union

import torch
import torch.nn as nn
from torch import Tensor

from torch_pointcloud.layers.conv2d_blocks import Conv2dBlock
from torch_pointcloud.utils.box3d import (
    boxes_iou3d,
    boxes_iou_bev,
    decode_box_residuals,
    encode_box_residuals,
    limit_period,
)
from torch_pointcloud.utils.types import Detection3D


class AnchorHeadOutput(TypedDict):
    r"""Raw and decoded predictions of [`AnchorHeadSingle`][torch_pointcloud.layers.anchors.AnchorHeadSingle]."""

    cls: Tensor
    box: Tensor
    dir_cls: Tensor
    batch_cls: Tensor
    batch_box: Tensor


def generate_anchors(
    point_cloud_range: Sequence[float],
    feature_map_size: Tuple[int, int],
    anchor_sizes: Sequence[Sequence[float]],
    anchor_rotations: Sequence[float],
    anchor_bottom_heights: Sequence[float],
    *,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    r"""Generate axis-aligned anchors for a single class over a BEV feature map.

    Mirrors the reference `AnchorGenerator` with `align_center=False`: anchor centers are placed on a
    grid spanning `point_cloud_range` (endpoints inclusive), then `anchor_bottom_heights` are lifted
    by half the box height to box centers.

    Args:
        point_cloud_range: Range $(x_\min, y_\min, z_\min, x_\max, y_\max, z_\max)$.
        feature_map_size: BEV feature map size $(n_x, n_y)$.
        anchor_sizes: Box sizes $(dx, dy, dz)$, one row per size template.
        anchor_rotations: Yaw angles (radians).
        anchor_bottom_heights: Anchor bottom $z$ per height template.
        dtype: Anchor dtype.

    Returns:
        Anchors $(1, n_y, n_x, n_\text{size}, n_\text{rot}, 7)$ as $(x, y, z, dx, dy, dz, \theta)$.
    """
    nx, ny = feature_map_size
    x_min, y_min, _, x_max, y_max, _ = point_cloud_range
    x_stride = (x_max - x_min) / (nx - 1)
    y_stride = (y_max - y_min) / (ny - 1)

    x_shifts = torch.arange(x_min, x_max + 1e-5, step=x_stride, dtype=dtype)
    y_shifts = torch.arange(y_min, y_max + 1e-5, step=y_stride, dtype=dtype)
    z_shifts = torch.tensor(list(anchor_bottom_heights), dtype=dtype)

    sizes = torch.tensor([list(s) for s in anchor_sizes], dtype=dtype)
    rotations = torch.tensor(list(anchor_rotations), dtype=dtype)
    num_size, num_rot = sizes.shape[0], rotations.shape[0]

    xs, ys, zs = torch.meshgrid([x_shifts, y_shifts, z_shifts], indexing="ij")
    anchors = torch.stack((xs, ys, zs), dim=-1)
    anchors = anchors[:, :, :, None, :].repeat(1, 1, 1, num_size, 1)
    sizes = sizes.view(1, 1, 1, -1, 3).repeat(*anchors.shape[:3], 1, 1)
    anchors = torch.cat((anchors, sizes), dim=-1)
    anchors = anchors[:, :, :, :, None, :].repeat(1, 1, 1, 1, num_rot, 1)
    rotations = rotations.view(1, 1, 1, 1, -1, 1).repeat(*anchors.shape[:3], num_size, 1, 1)
    anchors = torch.cat((anchors, rotations), dim=-1)

    anchors = anchors.permute(2, 1, 0, 3, 4, 5).contiguous()
    # lift anchor bottom heights to box-center heights
    anchors[..., 2] += anchors[..., 5] / 2
    return anchors


class AnchorTargets(TypedDict):
    r"""Per-anchor training targets from [`assign_anchor_targets`][torch_pointcloud.layers.anchors.assign_anchor_targets]."""

    cls_labels: Tensor
    box_reg_targets: Tensor


def assign_anchor_targets(
    anchors: Tensor,
    gt_boxes: Tensor,
    gt_labels: Tensor,
    *,
    matched_threshold: float,
    unmatched_threshold: float,
    match_height: bool = False,
) -> AnchorTargets:
    r"""Assign classification and box-regression targets to a single class group of axis-aligned anchors.

    Each anchor is matched to the ground-truth box of highest IoU: IoU $\ge$ `matched_threshold` makes it a
    positive carrying that box's label, IoU $<$ `unmatched_threshold` makes it background, and anything in
    between is ignored. Each ground-truth box additionally force-matches its single highest-IoU anchor, so a
    box with no anchor above threshold still receives one positive. Positive anchors' regression targets are
    the residual encoding of their matched box against the anchor (the inverse of
    [`decode_box_residuals`][torch_pointcloud.utils.box3d.decode_box_residuals]).

    Callers with several class groups (one anchor set per class) invoke this once per group with that class's
    anchors, ground truth, and thresholds, then concatenate the results.

    Args:
        anchors: Anchors $(x, y, z, d_x, d_y, d_z, \theta)$ for one class group, shape $(A, 7)$.
        gt_boxes: Ground-truth boxes $(c_x, c_y, c_z, d_x, d_y, d_z, \theta)$, shape $(G, 7)$.
        gt_labels: Ground-truth class labels ($1$-based foreground indices), shape $(G,)$.
        matched_threshold: IoU at or above which an anchor becomes a positive.
        unmatched_threshold: IoU below which an anchor becomes background.
        match_height: Match by 3D IoU when `True`, otherwise bird's-eye IoU.

    Returns:
        A `TypedDict` with `cls_labels` $(A,)$ ($-1$ ignore, $0$ background, $\ge 1$ foreground class) and
        `box_reg_targets` $(A, 7)$ (residual encodings, zero for non-positive anchors).

    Shape:
        - anchors: $(A, 7)$
        - gt_boxes: $(G, 7)$
        - gt_labels: $(G,)$
        - cls_labels: $(A,)$
        - box_reg_targets: $(A, 7)$

    Example:
        >>> anchors = torch.tensor([[0.0, 0.0, 0.0, 4.0, 2.0, 1.5, 0.0], [20.0, 0.0, 0.0, 4.0, 2.0, 1.5, 0.0]])
        >>> gt_boxes = torch.tensor([[0.0, 0.0, 0.0, 4.0, 2.0, 1.5, 0.0]])
        >>> gt_labels = torch.tensor([1])
        >>> out = assign_anchor_targets(anchors, gt_boxes, gt_labels, matched_threshold=0.6, unmatched_threshold=0.45)
        >>> out["cls_labels"].tolist()
        [1, 0]
    """
    num_anchors = anchors.shape[0]
    num_gt = gt_boxes.shape[0]
    labels = anchors.new_full((num_anchors,), -1, dtype=torch.long)
    box_reg_targets = anchors.new_zeros((num_anchors, 7))

    if num_gt > 0 and num_anchors > 0:
        overlap = boxes_iou3d(anchors, gt_boxes) if match_height else boxes_iou_bev(anchors, gt_boxes)

        anchor_to_gt_argmax = overlap.argmax(dim=1)
        anchor_to_gt_max = overlap[torch.arange(num_anchors, device=anchors.device), anchor_to_gt_argmax]

        gt_to_anchor_argmax = overlap.argmax(dim=0)
        gt_to_anchor_max = overlap[gt_to_anchor_argmax, torch.arange(num_gt, device=anchors.device)]
        gt_to_anchor_max[gt_to_anchor_max == 0] = -1
        anchors_with_max_overlap = (overlap == gt_to_anchor_max).nonzero()[:, 0]
        gt_inds_force = anchor_to_gt_argmax[anchors_with_max_overlap]

        pos_inds = anchor_to_gt_max >= matched_threshold
        labels[pos_inds] = gt_labels[anchor_to_gt_argmax[pos_inds]]
        bg_inds = anchor_to_gt_max < unmatched_threshold
        labels[bg_inds] = 0
        labels[anchors_with_max_overlap] = gt_labels[gt_inds_force]

        fg_inds = (labels > 0).nonzero()[:, 0]
        box_reg_targets[fg_inds] = encode_box_residuals(gt_boxes[anchor_to_gt_argmax[fg_inds]], anchors[fg_inds])
    else:
        labels[:] = 0

    return {
        "cls_labels": labels,
        "box_reg_targets": box_reg_targets,
    }


class AnchorHeadSingle(nn.Module):
    r"""Single-stage anchor head (`AnchorHeadSingle`).

    Three $1\times1$ convs predict, per anchor, class logits, 7-DoF box residuals and a direction
    bin. At inference the residuals are decoded against the precomputed anchors and the predicted
    heading is snapped to the predicted direction bin (`dir_offset` / `num_dir_bins`).

    Args:
        input_channels: Channels of the BEV feature map fed to the head.
        num_classes: Number of foreground classes.
        grid_size: Full voxel grid size $(n_x, n_y)$ (before the head feature-map stride).
        point_cloud_range: Range $(x_\min, y_\min, z_\min, x_\max, y_\max, z_\max)$.
        anchor_sizes: Per-class box size $(dx, dy, dz)$, shape $(\text{num\_classes}, 3)$.
        anchor_bottom_heights: Per-class anchor bottom $z$, shape $(\text{num\_classes},)$.
        anchor_rotations: Yaw angles (radians) shared by all classes.
        feature_map_stride: BEV feature-map stride of the head.
        num_dir_bins: Number of direction bins.
        dir_offset: Direction-classifier angle offset.
        dir_limit_offset: Offset used when wrapping the decoded heading before snapping.
    """

    anchors: Tensor

    def __init__(
        self,
        input_channels: int,
        num_classes: int,
        grid_size: Tuple[int, int],
        point_cloud_range: Sequence[float],
        *,
        anchor_sizes: Sequence[Sequence[float]],
        anchor_bottom_heights: Sequence[float],
        feature_map_stride: int,
        anchor_rotations: Sequence[float] = (0.0, 1.57),
        num_dir_bins: int = 2,
        dir_offset: float = 0.78539,
        dir_limit_offset: float = 0.0,
    ) -> None:
        super().__init__()
        if len(anchor_sizes) != num_classes:
            raise ValueError(f"Expected {num_classes} anchor sizes (one per class), got {len(anchor_sizes)}.")
        self.num_classes = num_classes
        self.num_dir_bins = num_dir_bins
        self.dir_offset = dir_offset
        self.dir_limit_offset = dir_limit_offset
        self.code_size = 7

        feature_map_size = (grid_size[0] // feature_map_stride, grid_size[1] // feature_map_stride)
        anchors_per_class = []
        num_anchors_per_location = 0
        for size, bottom in zip(anchor_sizes, anchor_bottom_heights):
            cls_anchors = generate_anchors(point_cloud_range, feature_map_size, [size], anchor_rotations, [bottom])
            anchors_per_class.append(cls_anchors)
            num_anchors_per_location += cls_anchors.shape[3] * cls_anchors.shape[4]

        anchors = torch.cat(anchors_per_class, dim=-3).view(-1, 7)
        # Anchors are rebuilt from the config, not part of the checkpoint; keep them out of the
        # state dict but move them with the module via a non-persistent buffer.
        self.register_buffer("anchors", anchors, persistent=False)
        self.num_anchors_per_location = num_anchors_per_location

        self.conv_cls = nn.Conv2d(input_channels, num_anchors_per_location * num_classes, 1)
        self.conv_box = nn.Conv2d(input_channels, num_anchors_per_location * self.code_size, 1)
        self.conv_dir_cls = nn.Conv2d(input_channels, num_anchors_per_location * num_dir_bins, 1)

    def forward(self, spatial_features_2d: Tensor) -> AnchorHeadOutput:
        cls_preds = self.conv_cls(spatial_features_2d).permute(0, 2, 3, 1).contiguous()
        box_preds = self.conv_box(spatial_features_2d).permute(0, 2, 3, 1).contiguous()
        dir_cls_preds = self.conv_dir_cls(spatial_features_2d).permute(0, 2, 3, 1).contiguous()

        batch_size = spatial_features_2d.shape[0]
        batch_cls_preds, batch_box_preds = self.generate_predicted_boxes(
            batch_size,
            cls_preds,
            box_preds,
            dir_cls_preds,
        )

        return {
            "cls": cls_preds,
            "box": box_preds,
            "dir_cls": dir_cls_preds,
            "batch_cls": batch_cls_preds,
            "batch_box": batch_box_preds,
        }

    def generate_predicted_boxes(
        self, batch_size: int, cls_preds: Tensor, box_preds: Tensor, dir_cls_preds: Tensor
    ) -> Tuple[Tensor, Tensor]:
        r"""Decode raw head outputs into per-anchor class logits and absolute boxes.

        Returns:
            A tuple `(batch_cls_preds, batch_box_preds)` of shapes $(B, A, \text{num\_classes})$
            (raw logits, not sigmoided) and $(B, A, 7)$ where $A$ is the number of anchors.
        """
        num_anchors = self.anchors.shape[0]
        batch_anchors = self.anchors.view(1, num_anchors, 7).repeat(batch_size, 1, 1)
        batch_cls_preds = cls_preds.view(batch_size, num_anchors, -1).float()
        batch_box_preds = box_preds.view(batch_size, num_anchors, -1)
        batch_box_preds = decode_box_residuals(batch_box_preds, batch_anchors)

        dir_preds = dir_cls_preds.view(batch_size, num_anchors, -1)
        dir_labels = torch.max(dir_preds, dim=-1)[1]
        period = 2 * math.pi / self.num_dir_bins
        dir_rot = limit_period(batch_box_preds[..., 6] - self.dir_offset, self.dir_limit_offset, period)
        batch_box_preds[..., 6] = dir_rot + self.dir_offset + period * dir_labels.to(batch_box_preds.dtype)
        return batch_cls_preds, batch_box_preds

    @torch.no_grad()
    def decode(self, out: AnchorHeadOutput) -> Detection3D:
        r"""Decode a forward output into raw per-anchor detections (no score threshold or NMS).

        Scores each anchor by its top sigmoid class probability and labels it by the argmax class. The
        full per-anchor set is returned; the evaluation pipeline applies score thresholding and per-class
        3D NMS via the `torch_pointcloud.utils.box3d` utilities (see the benchmark examples).

        Returns:
            Packed per-anchor detections `{"boxes": (B * A, 7), "scores": (B * A,), "labels": (B * A,),
            "batch": (B * A,)}` (PyG layout).
        """
        scores, labels = out["batch_cls"].sigmoid().max(dim=-1)
        batch_size, num_anchors = scores.shape
        batch = torch.arange(batch_size, device=scores.device).repeat_interleave(num_anchors)
        return {
            "boxes": out["batch_box"].reshape(-1, 7),
            "scores": scores.reshape(-1),
            "labels": labels.reshape(-1),
            "batch": batch,
        }


class AnchorHeadMultiOutput(TypedDict):
    r"""Predictions of [`AnchorHeadMulti`][torch_pointcloud.layers.anchors.AnchorHeadMulti]."""

    cls: List[Tensor]
    box: List[Tensor]
    batch_box: Tensor
    multihead_label_mapping: List[Tensor]


def separate_branch(
    in_channels: int,
    out_channels: int,
    num_middle_conv: int,
    num_middle_filter: int,
    *,
    act: Union[str, Callable, None] = "relu",
    act_kwargs: Optional[Dict[str, Any]] = None,
    norm: Union[str, Callable, None] = "batch_norm",
    norm_kwargs: Optional[Dict[str, Any]] = None,
    bias: bool = False,
) -> nn.Sequential:
    r"""Build a `SeparateHead`-style prediction branch: middle conv blocks, then a plain output conv.

    The per-attribute branch shared by the separate detection heads (the anchor multi-head and the
    center head): `num_middle_conv` blocks of ($3\times3$ conv, norm, act) followed by a
    $3\times3$ output conv.

    Args:
        in_channels: Input channels.
        out_channels: Output channels of the final conv.
        num_middle_conv: Number of middle conv blocks.
        num_middle_filter: Channel width of the middle convs.
        act: Activation of the middle conv blocks.
        act_kwargs: Extra activation arguments.
        norm: Normalization of the middle conv blocks.
        norm_kwargs: Extra normalization arguments.
        bias: Whether the middle convs carry a bias (the output conv always does).

    Returns:
        The branch as an `nn.Sequential`.

    Shape:
        - Input: $(B, C_\text{in}, H, W)$
        - Output: $(B, C_\text{out}, H, W)$

    Example:
        >>> branch = separate_branch(64, 2, num_middle_conv=1, num_middle_filter=64)
        >>> branch(torch.rand(2, 64, 16, 16)).shape
        torch.Size([2, 2, 16, 16])
    """
    layers: List[nn.Module] = []
    c_in = in_channels
    for _ in range(num_middle_conv):
        block = Conv2dBlock(
            c_in,
            num_middle_filter,
            3,
            padding=1,
            act=act,
            act_kwargs=act_kwargs,
            norm=norm,
            norm_kwargs=norm_kwargs,
            bias=bias,
        )
        layers.append(block)
        c_in = num_middle_filter

    layers.append(nn.Conv2d(c_in, out_channels, 3, stride=1, padding=1, bias=True))
    return nn.Sequential(*layers)


class MultiGroupSingleHead(nn.Module):
    r"""One RPN head of [`AnchorHeadMulti`][torch_pointcloud.layers.anchors.AnchorHeadMulti].

    A `SeparateHead`-style head over the shared feature: a classification branch plus one regression
    branch per box-code group (`reg`, `height`, `size`, `angle`, `velo`), whose outputs are
    concatenated into the per-anchor box code. Predictions are reshaped to the multihead layout
    $(B, A, \cdot)$ with $A$ the anchors of this head's class group.

    Args:
        input_channels: Channels of the shared feature map.
        num_classes: Number of classes handled by this head (separate-multihead).
        num_anchors_per_location: Anchors per BEV cell for this head.
        code_size: Box code size (e.g. 10 for sincos angle + velocity).
        reg_list: Regression-branch spec, e.g. `["reg:2", "height:1", "size:3", "angle:2", "velo:2"]`.
        num_middle_conv: Number of middle convs per branch.
        num_middle_filter: Middle-conv channel width.
        act: Activation type or callable for the middle convs.
        act_kwargs: Extra activation arguments.
        norm: Normalization type or callable for the middle convs.
        norm_kwargs: Extra normalization arguments.
    """

    head_label_indices: Tensor

    def __init__(
        self,
        input_channels: int,
        num_classes: int,
        num_anchors_per_location: int,
        code_size: int,
        reg_list: Sequence[str],
        head_label_indices: Tensor,
        *,
        num_middle_conv: int = 1,
        num_middle_filter: int = 64,
        act: Union[str, Callable, None] = "relu",
        act_kwargs: Optional[Dict[str, Any]] = None,
        norm: Union[str, Callable, None] = "batch_norm",
        norm_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.num_anchors_per_location = num_anchors_per_location
        self.code_size = code_size
        self.register_buffer("head_label_indices", head_label_indices)

        branch_kwargs: Dict[str, Any] = dict(
            num_middle_conv=num_middle_conv,
            num_middle_filter=num_middle_filter,
            act=act,
            act_kwargs=act_kwargs,
            norm=norm,
            norm_kwargs=norm_kwargs,
        )
        # Register conv_box before conv_cls so the parameter order matches the reference
        # checkpoint (its `SeparateHead` lists the regression branches first), keeping conversion a
        # straight positional alignment.
        self.conv_box = nn.ModuleDict()
        self.conv_box_names: List[str] = []
        code_size_cnt = 0
        for reg_config in reg_list:
            name, channels = reg_config.split(":")
            key = f"conv_{name}"
            self.conv_box[key] = separate_branch(
                input_channels,
                num_anchors_per_location * int(channels),
                **branch_kwargs,
            )
            self.conv_box_names.append(key)
            code_size_cnt += int(channels)

        if code_size_cnt != code_size:
            raise ValueError(f"Regression branches sum to {code_size_cnt} channels, expected code_size {code_size}.")

        self.conv_cls = separate_branch(input_channels, num_anchors_per_location * num_classes, **branch_kwargs)

    def forward(self, spatial_features_2d: Tensor) -> Tuple[Tensor, Tensor]:
        cls_preds = self.conv_cls(spatial_features_2d)
        box_preds = torch.cat([self.conv_box[name](spatial_features_2d) for name in self.conv_box_names], dim=1)

        batch_size, _, h, w = box_preds.shape
        box_preds = box_preds.view(-1, self.num_anchors_per_location, self.code_size, h, w)
        box_preds = box_preds.permute(0, 1, 3, 4, 2).contiguous().view(batch_size, -1, self.code_size)
        cls_preds = cls_preds.view(-1, self.num_anchors_per_location, self.num_classes, h, w)
        cls_preds = cls_preds.permute(0, 1, 3, 4, 2).contiguous().view(batch_size, -1, self.num_classes)
        return cls_preds, box_preds


class AnchorHeadMulti(nn.Module):
    r"""Multi-group anchor head (`AnchorHeadMulti`, separate-multihead).

    A shared conv feeds several [`MultiGroupSingleHead`][torch_pointcloud.layers.anchors.MultiGroupSingleHead]s,
    one per class group. Anchors (7-DoF, padded to the box-code size) are decoded with
    [`decode_box_residuals`][torch_pointcloud.utils.box3d.decode_box_residuals] (sincos heading,
    velocity deltas); per-head class scores stay separate (with their global label mapping) for
    class-wise NMS downstream.

    Args:
        input_channels: Channels of the BEV feature map fed to the head.
        num_classes: Number of foreground classes.
        grid_size: Full voxel grid size $(n_x, n_y)$.
        point_cloud_range: Range $(x_\min, y_\min, z_\min, x_\max, y_\max, z_\max)$.
        anchor_sizes: Per-class box size $(dx, dy, dz)$, shape $(\text{num\_classes}, 3)$.
        anchor_bottom_heights: Per-class anchor bottom $z$, shape $(\text{num\_classes},)$.
        head_class_groups: Class-index groups, one per RPN head (e.g. `[[0], [1, 2], ...]`); the
            classes in each group share one `SeparateHead`.
        anchor_rotations: Yaw angles (radians) shared by all classes.
        feature_map_stride: BEV feature-map stride of the head.
        shared_conv_num_filter: Channels of the shared conv.
        reg_list: Regression-branch spec for each head.
        num_middle_conv: Middle convs per branch.
        num_middle_filter: Middle-conv channel width.
        code_size: Base box code size (9 for nuScenes; +1 internally for sincos).
        encode_angle_by_sincos: Encode heading as $(\cos, \sin)$.
        act: Activation type or callable for the shared conv and head middle convs.
        act_kwargs: Extra activation arguments.
        norm: Normalization type or callable for the shared conv and head middle convs.
        norm_kwargs: Extra normalization arguments.
    """

    anchors: Tensor

    def __init__(
        self,
        input_channels: int,
        num_classes: int,
        grid_size: Tuple[int, int],
        point_cloud_range: Sequence[float],
        *,
        anchor_sizes: Sequence[Sequence[float]],
        anchor_bottom_heights: Sequence[float],
        head_class_groups: Sequence[Sequence[int]],
        feature_map_stride: int,
        anchor_rotations: Sequence[float] = (0.0, 1.57),
        shared_conv_num_filter: int = 64,
        reg_list: Sequence[str] = ("reg:2", "height:1", "size:3", "angle:2", "velo:2"),
        num_middle_conv: int = 1,
        num_middle_filter: int = 64,
        code_size: int = 9,
        encode_angle_by_sincos: bool = True,
        act: Union[str, Callable, None] = "relu",
        act_kwargs: Optional[Dict[str, Any]] = None,
        norm: Union[str, Callable, None] = "batch_norm",
        norm_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()
        if len(anchor_sizes) != num_classes:
            raise ValueError(f"Expected {num_classes} anchor sizes (one per class), got {len(anchor_sizes)}.")
        self.encode_angle_by_sincos = encode_angle_by_sincos
        self.code_size = code_size + 1 if encode_angle_by_sincos else code_size

        feature_map_size = (grid_size[0] // feature_map_stride, grid_size[1] // feature_map_stride)
        per_class_anchors = []
        num_anchors_per_class = []
        for size, bottom in zip(anchor_sizes, anchor_bottom_heights):
            anchors = generate_anchors(point_cloud_range, feature_map_size, [size], anchor_rotations, [bottom])
            num_anchors_per_class.append(anchors.shape[3] * anchors.shape[4])
            pad = anchors.new_zeros([*anchors.shape[:-1], self.code_size - anchors.shape[-1]])
            anchors = torch.cat([anchors, pad], dim=-1)
            per_class_anchors.append(anchors.permute(3, 4, 0, 1, 2, 5).reshape(-1, self.code_size))

        self.register_buffer("anchors", torch.cat(per_class_anchors, dim=0), persistent=False)

        self.shared_conv = Conv2dBlock(
            input_channels,
            shared_conv_num_filter,
            3,
            padding=1,
            act=act,
            act_kwargs=act_kwargs,
            norm=norm,
            norm_kwargs=norm_kwargs,
        )

        self.rpn_heads = nn.ModuleList()
        for group in head_class_groups:
            num_anchors = sum(num_anchors_per_class[i] for i in group)
            label_indices = torch.tensor([i + 1 for i in group], dtype=torch.long)
            # The reference per-group SeparateHead uses default BatchNorm hyperparameters (eps 1e-5),
            # unlike the trunk / shared conv (eps 1e-3); `norm_kwargs` is left at the default so a
            # converted checkpoint stays bit-exact.
            head = MultiGroupSingleHead(
                shared_conv_num_filter,
                len(group),
                num_anchors,
                self.code_size,
                reg_list,
                label_indices,
                num_middle_conv=num_middle_conv,
                num_middle_filter=num_middle_filter,
                act=act,
                act_kwargs=act_kwargs,
                norm=norm,
            )
            self.rpn_heads.append(head)

    def forward(self, spatial_features_2d: Tensor) -> AnchorHeadMultiOutput:
        shared = self.shared_conv(spatial_features_2d)
        cls_list: List[Tensor] = []
        box_list: List[Tensor] = []
        label_mapping: List[Tensor] = []
        for head in self.rpn_heads:
            assert isinstance(head, MultiGroupSingleHead)
            cls_preds, box_preds = head(shared)
            cls_list.append(cls_preds)
            box_list.append(box_preds)
            label_mapping.append(head.head_label_indices)

        batch_size = spatial_features_2d.shape[0]
        box_preds_cat = torch.cat(box_list, dim=1)
        batch_anchors = self.anchors.unsqueeze(0).expand(batch_size, -1, -1)
        batch_box_preds = decode_box_residuals(
            box_preds_cat, batch_anchors, angle_by_sincos=self.encode_angle_by_sincos
        )
        return {
            "cls": cls_list,
            "box": box_list,
            "batch_box": batch_box_preds,
            "multihead_label_mapping": label_mapping,
        }

    @torch.no_grad()
    def decode(self, out: AnchorHeadMultiOutput) -> Detection3D:
        r"""Decode a multihead forward output into raw per-anchor detections (no score threshold or NMS).

        Each head scores its anchors by their top sigmoid class probability and maps the argmax to the
        global label; the per-head results are concatenated in head order (matching `batch_box`'s anchor
        order). When the box code carries velocity deltas the decoded $(v_x, v_y)$ columns are returned
        under `velocity`. The full per-anchor set is returned; the evaluation pipeline applies score
        thresholding and per-class 3D NMS via the `torch_pointcloud.utils.box3d` utilities (see the
        benchmark examples).

        Returns:
            Packed per-anchor detections `{"boxes": (B * A, 7), "scores": (B * A,), "labels": (B * A,),
            "batch": (B * A,)}` (PyG layout), plus `"velocity"` $(B \cdot A, 2)$ when the head predicts it.
        """
        boxes_all = out["batch_box"]
        batch_size, num_anchors = boxes_all.shape[:2]
        scores_per_scene, labels_per_scene = [], []
        for b in range(batch_size):
            scene_scores, scene_labels = [], []
            for head_idx, cls_preds in enumerate(out["cls"]):
                head_scores, head_classes = cls_preds[b].sigmoid().max(dim=-1)
                scene_scores.append(head_scores)
                scene_labels.append(out["multihead_label_mapping"][head_idx][head_classes] - 1)
            scores_per_scene.append(torch.cat(scene_scores))
            labels_per_scene.append(torch.cat(scene_labels))

        batch = torch.arange(batch_size, device=boxes_all.device).repeat_interleave(num_anchors)
        det: Detection3D = {
            "boxes": boxes_all[:, :, :7].reshape(-1, 7),
            "scores": torch.stack(scores_per_scene).reshape(-1),
            "labels": torch.stack(labels_per_scene).reshape(-1),
            "batch": batch,
        }
        if boxes_all.shape[-1] >= 9:
            det["velocity"] = boxes_all[:, :, 7:9].reshape(-1, 2)
        return det
