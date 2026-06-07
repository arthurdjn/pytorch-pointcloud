r"""Anchor-based dense detection heads for the voxel detectors (PointPillars, SECOND).

A packed-format port of the OpenPCDet anchor head:
:github: [open-mmlab/OpenPCDet](https://github.com/open-mmlab/OpenPCDet).

- [`generate_anchors`][torch_pointcloud.layers.anchors.generate_anchors] and
  [`ResidualCoder`][torch_pointcloud.layers.anchors.ResidualCoder]: axis-aligned anchor generation
  and the residual box decoder.
- [`AnchorHeadSingle`][torch_pointcloud.layers.anchors.AnchorHeadSingle]: the single-stage anchor
  head (per-anchor class logits, box residuals and a direction bin).
- [`AnchorHeadMulti`][torch_pointcloud.layers.anchors.AnchorHeadMulti]: the multi-group
  separate-head variant (sincos + velocity box code) used by the nuScenes detectors.

Only the inference path (anchor generation + box decoding) is ported; target assignment and
losses are out of scope (these models are shipped for pretrained inference, not training).
"""

import math
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, TypedDict, Union

import torch
import torch.nn as nn
from torch import Tensor

from torch_pointcloud.layers.conv2d_blocks import Conv2dBlock
from torch_pointcloud.utils.box3d import nms3d
from torch_pointcloud.utils.types import Detection3D


class AnchorHeadOutput(TypedDict):
    r"""Raw and decoded predictions of [`AnchorHeadSingle`][torch_pointcloud.layers.anchors.AnchorHeadSingle]."""

    cls_preds: Tensor
    box_preds: Tensor
    dir_cls_preds: Tensor
    batch_cls_preds: Tensor
    batch_box_preds: Tensor


def limit_period(val: Tensor, offset: float = 0.5, period: float = math.pi) -> Tensor:
    r"""Wrap an angle to $[-\text{offset} \cdot \text{period}, (1 - \text{offset}) \cdot \text{period})$."""
    return val - torch.floor(val / period + offset) * period


class ResidualCoder:
    r"""Residual box coder (`ResidualCoder`): decodes box residuals against anchors.

    Encodes $(x, y, z, dx, dy, dz, \theta, \ldots)$ as center offsets normalized by the anchor base
    diagonal, log-size ratios and an angle term (a delta, or a $(\cos, \sin)$ pair when
    `encode_angle_by_sincos`), plus any trailing coordinates (e.g. nuScenes velocity). Only decoding
    is implemented (inference only).

    Args:
        code_size: Base box code size (7 for $(x, y, z, dx, dy, dz, \theta)$, 9 with velocity).
        encode_angle_by_sincos: Encode the heading as $(\cos, \sin)$ (adds one to the code size).
    """

    def __init__(self, code_size: int = 7, encode_angle_by_sincos: bool = False) -> None:
        self.code_size = code_size
        self.encode_angle_by_sincos = encode_angle_by_sincos
        if encode_angle_by_sincos:
            self.code_size += 1

    def decode_torch(self, box_encodings: Tensor, anchors: Tensor) -> Tensor:
        r"""Decode predicted residuals into absolute boxes.

        Args:
            box_encodings: Predicted residuals $(\ldots, \text{code\_size})$.
            anchors: Matching anchors $(\ldots, \geq 7)$.

        Returns:
            Decoded boxes $(\ldots, 7 + C)$ as $(x, y, z, dx, dy, dz, \theta, \ldots)$.
        """
        xa, ya, za, dxa, dya, dza, ra, *cas = torch.split(anchors, 1, dim=-1)
        if not self.encode_angle_by_sincos:
            xt, yt, zt, dxt, dyt, dzt, rt, *cts = torch.split(box_encodings, 1, dim=-1)
        else:
            xt, yt, zt, dxt, dyt, dzt, cost, sint, *cts = torch.split(box_encodings, 1, dim=-1)

        diagonal = torch.sqrt(dxa**2 + dya**2)
        xg = xt * diagonal + xa
        yg = yt * diagonal + ya
        zg = zt * dza + za
        dxg = torch.exp(dxt) * dxa
        dyg = torch.exp(dyt) * dya
        dzg = torch.exp(dzt) * dza
        if self.encode_angle_by_sincos:
            rg = torch.atan2(sint + torch.sin(ra), cost + torch.cos(ra))
        else:
            rg = rt + ra
        cgs = [t + a for t, a in zip(cts, cas)]
        return torch.cat([xg, yg, zg, dxg, dyg, dzg, rg, *cgs], dim=-1)


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

    Mirrors OpenPCDet `AnchorGenerator` with `align_center=False`: anchor centers are placed on a
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
        self.box_coder = ResidualCoder(code_size=7)

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
        self.conv_box = nn.Conv2d(input_channels, num_anchors_per_location * self.box_coder.code_size, 1)
        self.conv_dir_cls = nn.Conv2d(input_channels, num_anchors_per_location * num_dir_bins, 1)

    def forward(self, spatial_features_2d: Tensor) -> AnchorHeadOutput:
        cls_preds = self.conv_cls(spatial_features_2d).permute(0, 2, 3, 1).contiguous()
        box_preds = self.conv_box(spatial_features_2d).permute(0, 2, 3, 1).contiguous()
        dir_cls_preds = self.conv_dir_cls(spatial_features_2d).permute(0, 2, 3, 1).contiguous()

        batch_size = spatial_features_2d.shape[0]
        batch_cls_preds, batch_box_preds = self.generate_predicted_boxes(
            batch_size, cls_preds, box_preds, dir_cls_preds
        )
        return {
            "cls_preds": cls_preds,
            "box_preds": box_preds,
            "dir_cls_preds": dir_cls_preds,
            "batch_cls_preds": batch_cls_preds,
            "batch_box_preds": batch_box_preds,
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
        batch_box_preds = self.box_coder.decode_torch(batch_box_preds, batch_anchors)

        dir_preds = dir_cls_preds.view(batch_size, num_anchors, -1)
        dir_labels = torch.max(dir_preds, dim=-1)[1]
        period = 2 * math.pi / self.num_dir_bins
        dir_rot = limit_period(batch_box_preds[..., 6] - self.dir_offset, self.dir_limit_offset, period)
        batch_box_preds[..., 6] = dir_rot + self.dir_offset + period * dir_labels.to(batch_box_preds.dtype)
        return batch_cls_preds, batch_box_preds

    @torch.no_grad()
    def decode(self, out: AnchorHeadOutput, *, score_threshold: float = 0.1, nms_iou: float = 0.01) -> Detection3D:
        r"""Decode a forward output into packed detections: sigmoid scores, threshold, per-class 3D NMS.

        Returns:
            Packed detections `{"boxes": (K, 7), "scores": (K,), "labels": (K,), "batch": (K,)}` (PyG layout).
        """
        scores_per_class = out["batch_cls_preds"].sigmoid()
        boxes = out["batch_box_preds"]
        out_boxes, out_scores, out_labels, out_batch = [], [], [], []
        for b in range(boxes.shape[0]):
            scores, labels = scores_per_class[b].max(dim=-1)
            keep = scores > score_threshold
            scene_boxes, scene_scores, scene_labels = boxes[b][keep], scores[keep], labels[keep]
            idx = nms3d(scene_boxes, scene_scores, scene_labels, nms_iou)
            out_boxes.append(scene_boxes[idx])
            out_scores.append(scene_scores[idx])
            out_labels.append(scene_labels[idx])
            out_batch.append(torch.full((idx.numel(),), b, dtype=torch.long, device=boxes.device))
        return {
            "boxes": torch.cat(out_boxes),
            "scores": torch.cat(out_scores),
            "labels": torch.cat(out_labels),
            "batch": torch.cat(out_batch),
        }


class AnchorHeadMultiOutput(TypedDict):
    r"""Predictions of [`AnchorHeadMulti`][torch_pointcloud.layers.anchors.AnchorHeadMulti]."""

    cls_preds: List[Tensor]
    box_preds: List[Tensor]
    batch_box_preds: Tensor
    multihead_label_mapping: List[Tensor]


def _separate_branch(
    in_channels: int,
    out_channels: int,
    num_middle_conv: int,
    num_middle_filter: int,
    *,
    act: Union[str, Callable, None] = "relu",
    act_kwargs: Optional[Dict[str, Any]] = None,
    norm: Union[str, Callable, None] = "batch_norm",
    norm_kwargs: Optional[Dict[str, Any]] = None,
) -> nn.Sequential:
    """A `SeparateHead` branch: `num_middle_conv` of (3x3 conv, norm, act) then a 3x3 output conv."""
    layers: List[nn.Module] = []
    c_in = in_channels
    for _ in range(num_middle_conv):
        layers.append(
            Conv2dBlock(
                c_in,
                num_middle_filter,
                3,
                padding=1,
                act=act,
                act_kwargs=act_kwargs,
                norm=norm,
                norm_kwargs=norm_kwargs,
            )
        )
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
        # Register conv_box before conv_cls so the parameter order matches the OpenPCDet reference
        # checkpoint (its `SeparateHead` lists the regression branches first), keeping conversion a
        # straight positional alignment.
        self.conv_box = nn.ModuleDict()
        self.conv_box_names: List[str] = []
        code_size_cnt = 0
        for reg_config in reg_list:
            name, channels = reg_config.split(":")
            key = f"conv_{name}"
            self.conv_box[key] = _separate_branch(
                input_channels, num_anchors_per_location * int(channels), **branch_kwargs
            )
            self.conv_box_names.append(key)
            code_size_cnt += int(channels)
        if code_size_cnt != code_size:
            raise ValueError(f"Regression branches sum to {code_size_cnt} channels, expected code_size {code_size}.")
        self.conv_cls = _separate_branch(input_channels, num_anchors_per_location * num_classes, **branch_kwargs)

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
    one per class group. Anchors (7-DoF, padded to the box-code size) are decoded with a
    velocity-aware sincos [`ResidualCoder`][torch_pointcloud.layers.anchors.ResidualCoder]; per-head
    class scores stay separate (with their global label mapping) for class-wise NMS downstream.

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
        self.box_coder = ResidualCoder(code_size=code_size, encode_angle_by_sincos=encode_angle_by_sincos)

        feature_map_size = (grid_size[0] // feature_map_stride, grid_size[1] // feature_map_stride)
        per_class_anchors = []
        num_anchors_per_class = []
        for size, bottom in zip(anchor_sizes, anchor_bottom_heights):
            anchors = generate_anchors(point_cloud_range, feature_map_size, [size], anchor_rotations, [bottom])
            num_anchors_per_class.append(anchors.shape[3] * anchors.shape[4])
            pad = anchors.new_zeros([*anchors.shape[:-1], self.box_coder.code_size - anchors.shape[-1]])
            anchors = torch.cat([anchors, pad], dim=-1)
            per_class_anchors.append(anchors.permute(3, 4, 0, 1, 2, 5).reshape(-1, self.box_coder.code_size))

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
            # OpenPCDet's per-group SeparateHead uses default BatchNorm hyperparameters (eps 1e-5),
            # unlike the trunk / shared conv (eps 1e-3); `norm_kwargs` is left at the default so a
            # converted checkpoint stays bit-exact.
            self.rpn_heads.append(
                MultiGroupSingleHead(
                    shared_conv_num_filter,
                    len(group),
                    num_anchors,
                    self.box_coder.code_size,
                    reg_list,
                    label_indices,
                    num_middle_conv=num_middle_conv,
                    num_middle_filter=num_middle_filter,
                    act=act,
                    act_kwargs=act_kwargs,
                    norm=norm,
                )
            )

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
        batch_box_preds = self.box_coder.decode_torch(box_preds_cat, batch_anchors)
        return {
            "cls_preds": cls_list,
            "box_preds": box_list,
            "batch_box_preds": batch_box_preds,
            "multihead_label_mapping": label_mapping,
        }

    @torch.no_grad()
    def decode(self, out: AnchorHeadMultiOutput, *, score_threshold: float = 0.1, nms_iou: float = 0.2) -> Detection3D:
        r"""Decode a multihead forward output into packed detections (per-head sigmoid, threshold, 3D NMS).

        Each head's class scores carry their global label mapping; boxes use the 7-DoF pose (velocity
        columns are dropped). Returns `{"boxes": (K, 7), "scores": (K,), "labels": (K,), "batch": (K,)}`.
        """
        boxes_all = out["batch_box_preds"]
        out_boxes, out_scores, out_labels, out_batch = [], [], [], []
        for b in range(boxes_all.shape[0]):
            scene_scores, scene_labels = [], []
            for head_idx, cls_preds in enumerate(out["cls_preds"]):
                head_scores, head_classes = cls_preds[b].sigmoid().max(dim=-1)
                scene_scores.append(head_scores)
                scene_labels.append(out["multihead_label_mapping"][head_idx][head_classes] - 1)
            scores = torch.cat(scene_scores)
            labels = torch.cat(scene_labels)
            boxes = boxes_all[b][:, :7]
            keep = scores > score_threshold
            scene_boxes, scores, labels = boxes[keep], scores[keep], labels[keep]
            idx = nms3d(scene_boxes, scores, labels, nms_iou)
            out_boxes.append(scene_boxes[idx])
            out_scores.append(scores[idx])
            out_labels.append(labels[idx])
            out_batch.append(torch.full((idx.numel(),), b, dtype=torch.long, device=boxes_all.device))
        return {
            "boxes": torch.cat(out_boxes),
            "scores": torch.cat(out_scores),
            "labels": torch.cat(out_labels),
            "batch": torch.cat(out_batch),
        }
