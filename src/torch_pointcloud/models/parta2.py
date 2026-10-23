"""Part-A2 detection model.

{{ paper("1907.03670") }}
"""

import math
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Sequence, Tuple, TypedDict, Union

import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.nn import MLP

import torch_pointcloud.transforms as T
from torch_pointcloud.layers.anchors import AnchorHead, AnchorHeadOutput
from torch_pointcloud.layers.bev_backbone import BEVBackbone
from torch_pointcloud.layers.spconv_blocks import SparseBasicBlock, SparseConvBlock
from torch_pointcloud.ops.box3d import decode_roi_residuals, nms3d, roi_aware_pool3d
from torch_pointcloud.utils.data import DataKeys
from torch_pointcloud.utils.imports import _SPCONV_GITHUB_URL, optional_import
from torch_pointcloud.utils.types import Detection3D

from ._base import DetectionModel
from ._registry import WeightsDict, register_model
from .second import VoxelBackbone8x

if TYPE_CHECKING:
    import spconv.pytorch as spconv

spconv, _ = optional_import("spconv.pytorch", url=_SPCONV_GITHUB_URL)


class PartA2Output(TypedDict):
    r"""Part-A2 output: refined RoI boxes with their second-stage confidence (packed layout).

    Attributes:
        rcnn_cls: Second-stage confidence logit per RoI, shape $(R, 1)$.
        boxes: Refined boxes $(c_x, c_y, c_z, d_x, d_y, d_z, \theta)$, shape $(R, 7)$.
        roi_labels: First-stage class per RoI, shape $(R,)$.
        roi_scores: First-stage score per RoI, shape $(R,)$.
        roi_class_scores: First-stage sigmoid score of every class per RoI, shape $(R, C)$.
        batch: Per-RoI scene index, shape $(R,)$.
    """

    rcnn_cls: Tensor
    boxes: Tensor
    roi_labels: Tensor
    roi_scores: Tensor
    roi_class_scores: Tensor
    batch: Tensor


class SparseUNetDecoder(nn.Module):
    r"""Sparse U-Net decoder that brings the four stages of `VoxelBackbone8x` back to the input voxels.

    From the coarsest stage up, every level runs a lateral residual block on the encoder stage, merges it with the
    upsampled coarser features by a submanifold convolution plus a channel-summing shortcut, and upsamples the
    result with the inverse of the encoder's strided convolution (a submanifold convolution at the input level).

    Args:
        channels: Per-level `(lateral, merge, upsample)` output channels, coarsest level first.
        act: Activation type or callable.
        act_kwargs: Extra activation arguments.
        norm: Normalization type or callable.
        norm_kwargs: Extra normalization arguments.
    """

    def __init__(
        self,
        channels: Sequence[Sequence[int]] = ((64, 64, 64), (64, 64, 32), (32, 32, 16), (16, 16, 16)),
        *,
        act: Union[str, Callable, None] = "relu",
        act_kwargs: Optional[Dict[str, Any]] = None,
        norm: Union[str, Callable, None] = "batch_norm",
        norm_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()
        block_kwargs: Dict[str, Any] = dict(act=act, act_kwargs=act_kwargs, norm=norm, norm_kwargs=norm_kwargs)
        num_levels = len(channels)
        in_channels = 64
        self.laterals = nn.ModuleList()
        self.merges = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        for i, (lateral, merge, upsample) in enumerate(channels):
            level = num_levels - i
            self.laterals.append(SparseBasicBlock(in_channels, lateral, indice_key=f"subm{level}", **block_kwargs))
            self.merges.append(SparseConvBlock(in_channels * 2, merge, 3, indice_key=f"subm{level}", **block_kwargs))
            if level == 1:
                self.upsamples.append(SparseConvBlock(in_channels, upsample, 3, indice_key="subm1", **block_kwargs))
            else:
                self.upsamples.append(
                    SparseConvBlock(
                        in_channels,
                        upsample,
                        3,
                        indice_key=f"spconv{level}",
                        conv_type="inverseconv",
                        **block_kwargs,
                    )
                )
            in_channels = upsample
        self.out_channels = in_channels

    def forward(self, stages: List["spconv.SparseConvTensor"]) -> Tensor:
        r"""Decode the encoder stages (finest first) into features of the input voxels, shape $(V, C)$."""
        x = stages[-1]
        for lateral, merge, upsample, stage in zip(self.laterals, self.merges, self.upsamples, reversed(stages)):
            skip = lateral(stage)
            cat = torch.cat([x.features, skip.features], dim=1)
            merged = merge(skip.replace_feature(cat))
            # The shortcut sums consecutive channel groups of the concatenation down to the merge width.
            shortcut = cat.view(cat.shape[0], merged.features.shape[1], -1).sum(dim=2)
            x = upsample(merged.replace_feature(merged.features + shortcut))
        return x.features


class PartA2BoxHead(nn.Module):
    r"""Part-A2 part-aggregation head: sparse convolutions over the RoI-pooled grids, then two branches.

    The pooled point features and part features of every RoI are sparse grids (a cell is active when it pooled any
    point). Two submanifold branches encode them, a merge stack and a $2\times$ sparse max pooling aggregate the
    concatenation, and the densified grid goes through shared fully connected layers to a confidence and a
    box-residual branch.

    Args:
        seg_channels: Channels of the pooled point features.
        part_channels: Channels of the pooled part features.
        grid_size: Cells per RoI side of the pooled grids.
        seg_conv_channels: Channels of the point-feature branch.
        part_conv_channels: Channels of the part-feature branch.
        merge_channels: Channels of the merge stack.
        down_channels: Channels after the sparse max pooling.
        shared_channels: Channels of the shared fully connected layers.
        cls_channels: Hidden channels of the confidence branch.
        reg_channels: Hidden channels of the box branch.
        dropout: Dropout of the fully connected layers.
        act: Activation type or callable.
        act_kwargs: Extra activation arguments.
        norm: Normalization type or callable.
        norm_kwargs: Extra normalization arguments.
    """

    def __init__(
        self,
        seg_channels: int = 16,
        part_channels: int = 4,
        *,
        grid_size: int = 14,
        seg_conv_channels: Sequence[int] = (64, 64),
        part_conv_channels: Sequence[int] = (64, 64),
        merge_channels: Sequence[int] = (128, 128),
        down_channels: Sequence[int] = (128, 256),
        shared_channels: Sequence[int] = (512, 512, 512),
        cls_channels: Sequence[int] = (256, 256),
        reg_channels: Sequence[int] = (256, 256),
        dropout: float = 0.1,
        act: Union[str, Callable, None] = "relu",
        act_kwargs: Optional[Dict[str, Any]] = None,
        norm: Union[str, Callable, None] = "batch_norm",
        norm_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()
        self.grid_size = grid_size
        block_kwargs: Dict[str, Any] = dict(act=act, act_kwargs=act_kwargs, norm=norm, norm_kwargs=norm_kwargs)
        self.seg_conv = spconv.SparseSequential(
            *[
                SparseConvBlock(c_in, c_out, 3, indice_key=f"rcnn_seg{i}", **block_kwargs)
                for i, (c_in, c_out) in enumerate(zip([seg_channels, *seg_conv_channels], seg_conv_channels))
            ]
        )
        self.part_conv = spconv.SparseSequential(
            *[
                SparseConvBlock(c_in, c_out, 3, indice_key=f"rcnn_part{i}", **block_kwargs)
                for i, (c_in, c_out) in enumerate(zip([part_channels, *part_conv_channels], part_conv_channels))
            ]
        )
        merge_in = seg_conv_channels[-1] + part_conv_channels[-1]
        self.merge_conv = spconv.SparseSequential(
            *[
                SparseConvBlock(c_in, c_out, 3, indice_key="rcnn_down0", **block_kwargs)
                for c_in, c_out in zip([merge_in, *merge_channels], merge_channels)
            ]
        )
        self.pool = spconv.SparseMaxPool3d(2, 2)
        self.down_conv = spconv.SparseSequential(
            *[
                SparseConvBlock(c_in, c_out, 3, indice_key="rcnn_down1", **block_kwargs)
                for c_in, c_out in zip([merge_channels[-1], *down_channels], down_channels)
            ]
        )
        fc_kwargs: Dict[str, Any] = dict(act=act, act_kwargs=act_kwargs, norm=norm, norm_kwargs=norm_kwargs)
        grid_channels = down_channels[-1] * (grid_size // 2) ** 3
        self.shared = MLP(
            [grid_channels, *shared_channels],
            bias=False,
            dropout=[dropout] * (len(shared_channels) - 1) + [0.0],
            plain_last=False,
            **fc_kwargs,
        )
        self.cls = MLP(
            [shared_channels[-1], *cls_channels, 1],
            bias=[False] * len(cls_channels) + [True],
            dropout=[dropout] + [0.0] * len(cls_channels),
            **fc_kwargs,
        )
        self.reg = MLP(
            [shared_channels[-1], *reg_channels, 7],
            bias=[False] * len(reg_channels) + [True],
            dropout=[dropout] + [0.0] * len(reg_channels),
            **fc_kwargs,
        )

    def forward(self, seg_grid: Tensor, part_grid: Tensor) -> Tuple[Tensor, Tensor]:
        r"""Predict every RoI's confidence logit $(R, 1)$ and box residuals $(R, 7)$ from its pooled grids.

        Args:
            seg_grid: Pooled point features, shape $(R, S, S, S, C_s)$.
            part_grid: Pooled part features, shape $(R, S, S, S, C_p)$.
        """
        num_rois = part_grid.shape[0]
        shape = list(part_grid.shape[1:4])
        indices = part_grid.sum(dim=-1).nonzero(as_tuple=False)
        cells = (indices[:, 0], indices[:, 1], indices[:, 2], indices[:, 3])
        # `nonzero` returns column-major indices on CUDA; spconv reads them as row-major, so make them contiguous.
        coords = indices.int().contiguous()

        seg = self.seg_conv(spconv.SparseConvTensor(seg_grid[cells], coords, shape, num_rois))
        part = self.part_conv(spconv.SparseConvTensor(part_grid[cells], coords, shape, num_rois))
        merged = spconv.SparseConvTensor(torch.cat([seg.features, part.features], dim=1), coords, shape, num_rois)
        x = self.down_conv(self.pool(self.merge_conv(merged)))

        shared = self.shared(x.dense().reshape(num_rois, -1))
        return self.cls(shared), self.reg(shared)


class PartA2Detection(DetectionModel):
    r"""Part-A2 two-stage 3D object detector (packed point format).

    Reference: :arxiv: [From Points to Parts: 3D Object Detection from Point Cloud with Part-aware and Part-aggregation Network](https://arxiv.org/abs/1907.03670) (Shi et al., 2020).
    Reference implementation: :github: [open-mmlab/OpenPCDet](https://github.com/open-mmlab/OpenPCDet).

    The first stage is the single-stage detector of
    [`SECONDDetection`][torch_pointcloud.models.second.SECONDDetection] whose best anchors after rotated NMS become
    the RoIs, with a [`SparseUNetDecoder`][torch_pointcloud.models.parta2.SparseUNetDecoder] that decodes the sparse
    stages back to the input voxels. Every voxel predicts a foreground score and its intra-object part location;
    both are pooled inside every RoI by [`roi_aware_pool3d`][torch_pointcloud.ops.box3d.roi_aware_pool3d] and
    aggregated by [`PartA2BoxHead`][torch_pointcloud.models.parta2.PartA2BoxHead].

    Args:
        in_channels: Raw point feature channels including xyz (e.g. $4$ for $x, y, z, \text{intensity}$).
        num_classes: Number of foreground classes.
        voxel_size: Voxel size $(v_x, v_y, v_z)$.
        point_cloud_range: Range $(x_\min, y_\min, z_\min, x_\max, y_\max, z_\max)$.
        anchor_sizes: Per-class box size $(d_x, d_y, d_z)$, one row per class.
        anchor_bottom_heights: Per-class anchor bottom $z$, one per class.
        anchor_rotations: Yaw angles (radians) shared by all classes.
        feature_map_stride: BEV feature-map stride of the anchor head.
        layer_nums: 2D backbone conv counts per level.
        layer_strides: 2D backbone downsample strides per level.
        num_filters: 2D backbone channel widths per level.
        upsample_strides: 2D backbone upsample strides per level.
        num_upsample_filters: 2D backbone upsample channels per level.
        num_dir_bins: Number of direction bins in the anchor head.
        dir_offset: Direction-classifier angle offset.
        dir_limit_offset: Heading wrap offset used during decoding.
        proposal_pre_maxsize: Top-scoring anchors kept per scene before the proposal NMS.
        proposal_post_maxsize: RoIs kept per scene after the proposal NMS.
        proposal_nms_thresh: Rotated BEV IoU threshold of the proposal NMS.
        decoder_channels: Per-level channels of the sparse U-Net decoder.
        part_score_threshold: Foreground score above which a voxel's part location is pooled.
        grid_size: Cells per RoI side of the RoI-aware pooling.
        max_points_per_cell: Voxels pooled per cell.
        act: Activation type or callable.
        act_kwargs: Extra activation arguments.
        norm: Normalization type or callable.
        norm_kwargs: Extra normalization arguments of the backbones and the refinement head.
    """

    def __init__(
        self,
        in_channels: int = 4,
        num_classes: int = 3,
        *,
        voxel_size: Sequence[float] = (0.05, 0.05, 0.1),
        point_cloud_range: Sequence[float] = (0.0, -40.0, -3.0, 70.4, 40.0, 1.0),
        anchor_sizes: Sequence[Sequence[float]],
        anchor_bottom_heights: Sequence[float],
        anchor_rotations: Sequence[float] = (0.0, 1.57),
        feature_map_stride: int = 8,
        layer_nums: Sequence[int] = (5, 5),
        layer_strides: Sequence[int] = (1, 2),
        num_filters: Sequence[int] = (128, 256),
        upsample_strides: Sequence[int] = (1, 2),
        num_upsample_filters: Sequence[int] = (256, 256),
        num_dir_bins: int = 2,
        dir_offset: float = 0.78539,
        dir_limit_offset: float = 0.0,
        proposal_pre_maxsize: int = 1024,
        proposal_post_maxsize: int = 100,
        proposal_nms_thresh: float = 0.7,
        decoder_channels: Sequence[Sequence[int]] = ((64, 64, 64), (64, 64, 32), (32, 32, 16), (16, 16, 16)),
        part_score_threshold: float = 0.3,
        grid_size: int = 14,
        max_points_per_cell: int = 127,
        act: Union[str, Callable, None] = "relu",
        act_kwargs: Optional[Dict[str, Any]] = None,
        norm: Union[str, Callable, None] = "batch_norm",
        norm_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(in_channels=in_channels, num_classes=num_classes)
        self.voxel_size = tuple(voxel_size)
        self.point_cloud_range = tuple(point_cloud_range)
        self.anchor_sizes = anchor_sizes
        self.anchor_bottom_heights = anchor_bottom_heights
        self.anchor_rotations = anchor_rotations
        self.feature_map_stride = feature_map_stride
        self.layer_nums = layer_nums
        self.layer_strides = layer_strides
        self.num_filters = num_filters
        self.upsample_strides = upsample_strides
        self.num_upsample_filters = num_upsample_filters
        self.num_dir_bins = num_dir_bins
        self.dir_offset = dir_offset
        self.dir_limit_offset = dir_limit_offset
        self.proposal_pre_maxsize = proposal_pre_maxsize
        self.proposal_post_maxsize = proposal_post_maxsize
        self.proposal_nms_thresh = proposal_nms_thresh
        self.decoder_channels = decoder_channels
        self.part_score_threshold = part_score_threshold
        self.grid_size = grid_size
        self.max_points_per_cell = max_points_per_cell
        self.act = act
        self.act_kwargs = act_kwargs
        self.norm = norm
        self.norm_kwargs = norm_kwargs

        grid = [int(round((point_cloud_range[i + 3] - point_cloud_range[i]) / voxel_size[i])) for i in range(3)]
        self.grid_size_3d: Tuple[int, int, int] = (grid[0], grid[1], grid[2])
        # spconv spatial shape is (z, y, x) with an extra +1 on z.
        self.sparse_shape: List[int] = [grid[2] + 1, grid[1], grid[0]]

        self.backbone_3d = self.configure_backbone_3d()
        self.decoder = self.configure_decoder()
        self.backbone = self.configure_backbone()
        self.head = self.configure_head()
        self.seg_head = nn.Linear(self.decoder.out_channels, 1)
        self.part_head = nn.Linear(self.decoder.out_channels, 3)
        self.roi_head = self.configure_roi_head()

    def configure_backbone_3d(self) -> VoxelBackbone8x:
        """Build the sparse 3D voxel encoder."""
        return VoxelBackbone8x(
            self.in_channels,
            act=self.act,
            act_kwargs=self.act_kwargs,
            norm=self.norm,
            norm_kwargs=self.norm_kwargs,
        )

    def configure_decoder(self) -> SparseUNetDecoder:
        """Build the sparse U-Net decoder."""
        return SparseUNetDecoder(
            self.decoder_channels,
            act=self.act,
            act_kwargs=self.act_kwargs,
            norm=self.norm,
            norm_kwargs=self.norm_kwargs,
        )

    def configure_backbone(self) -> BEVBackbone:
        """Build the 2D BEV backbone."""
        # Height compression folds the sparse-z output (D=2) into the channel dim.
        bev_input_channels = self.backbone_3d.out_channels * 2
        return BEVBackbone(
            bev_input_channels,
            self.layer_nums,
            self.layer_strides,
            self.num_filters,
            self.upsample_strides,
            self.num_upsample_filters,
            act=self.act,
            act_kwargs=self.act_kwargs,
            norm=self.norm,
            norm_kwargs=self.norm_kwargs,
        )

    @property
    def num_features(self) -> int:
        """Channel count $C$ of the BEV feature map entering the anchor head."""
        return self.backbone.num_bev_features

    def configure_head(self) -> AnchorHead:
        """Build the first-stage anchor head."""
        return AnchorHead(
            self.backbone.num_bev_features,
            self.num_classes,
            (self.grid_size_3d[0], self.grid_size_3d[1]),
            self.point_cloud_range,
            anchor_sizes=self.anchor_sizes,
            anchor_bottom_heights=self.anchor_bottom_heights,
            anchor_rotations=self.anchor_rotations,
            feature_map_stride=self.feature_map_stride,
            num_dir_bins=self.num_dir_bins,
            dir_offset=self.dir_offset,
            dir_limit_offset=self.dir_limit_offset,
        )

    def configure_roi_head(self) -> PartA2BoxHead:
        """Build the part-aggregation refinement head."""
        return PartA2BoxHead(
            self.decoder.out_channels,
            4,
            grid_size=self.grid_size,
            act=self.act,
            act_kwargs=self.act_kwargs,
            norm=self.norm,
            norm_kwargs=self.norm_kwargs,
        )

    def encode(
        self,
        voxels: Tensor,
        pos_voxel: Tensor,
        voxel_num_points: Tensor,
        batch: Tensor,
    ) -> Tuple[Tensor, List["spconv.SparseConvTensor"]]:
        r"""Run the sparse encoder and the BEV backbone.

        Returns:
            `(features, stages)`: the BEV features entering the anchor head and the four sparse stages the
            decoder upsamples.
        """
        voxel_indices = torch.cat([batch.view(-1, 1).to(pos_voxel), pos_voxel], dim=1)
        batch_size = int(batch.max().item()) + 1

        # Average the points (xyz + features) inside each voxel.
        normalizer = torch.clamp_min(voxel_num_points.view(-1, 1), min=1.0).type_as(voxels)
        voxel_features = voxels.sum(dim=1) / normalizer

        x = spconv.SparseConvTensor(
            features=voxel_features,
            indices=voxel_indices.int(),
            spatial_shape=self.sparse_shape,
            batch_size=batch_size,
        )
        x = self.backbone_3d.conv_input(x)
        stages = []
        for conv in (self.backbone_3d.conv1, self.backbone_3d.conv2, self.backbone_3d.conv3, self.backbone_3d.conv4):
            x = conv(x)
            stages.append(x)
        encoded = self.backbone_3d.conv_out(x)

        dense = encoded.dense()
        b, c, d, h, w = dense.shape
        features = self.backbone(dense.view(b, c * d, h, w))
        return features, stages

    def forward_features(self, voxels: Tensor, pos_voxel: Tensor, voxel_num_points: Tensor, batch: Tensor) -> Tensor:
        features, _ = self.encode(voxels, pos_voxel, voxel_num_points, batch)
        return features

    def forward_head(self, features: Tensor) -> AnchorHeadOutput:
        return self.head(features)

    @torch.no_grad()
    def proposals(self, out: AnchorHeadOutput) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        r"""Select the RoIs of every scene from the anchor head output.

        The `proposal_pre_maxsize` best anchors (by their top class score) go through a class-agnostic rotated BEV
        NMS, which keeps at most `proposal_post_maxsize` of them.

        Returns:
            `(rois, roi_scores, roi_labels, roi_class_scores, roi_batch)` of shapes $(R, 7)$, $(R,)$, $(R,)$,
            $(R, C)$ and $(R,)$.
        """
        det = self.head.decode(out)
        class_scores = out["batch_cls"].sigmoid().reshape(-1, self.num_classes)
        keep_all = []
        for b in range(int(det["batch"].max().item()) + 1):
            scene = (det["batch"] == b).nonzero().view(-1)
            top_scores, top = det["scores"][scene].topk(min(self.proposal_pre_maxsize, scene.numel()))
            top = scene[top]
            keep = nms3d(
                det["boxes"][top],
                top_scores,
                self.proposal_nms_thresh,
                rotated=True,
                max_keep=self.proposal_post_maxsize,
            )
            keep_all.append(top[keep])
        keep = torch.cat(keep_all)
        return det["boxes"][keep], det["scores"][keep], det["labels"][keep], class_scores[keep], det["batch"][keep]

    def refine(
        self,
        rois: Tensor,
        roi_batch: Tensor,
        voxel_features: Tensor,
        pos_voxel: Tensor,
        batch: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        r"""Predict the voxel parts, pool them inside every RoI and refine it.

        Returns:
            `(rcnn_cls, rcnn_reg)`: the confidence logit $(R, 1)$ and the box residuals $(R, 7)$ of every RoI.
        """
        seg_scores = self.seg_head(voxel_features).sigmoid()
        part_offsets = self.part_head(voxel_features).sigmoid() * (seg_scores > self.part_score_threshold)
        part_features = torch.cat([part_offsets, seg_scores], dim=1)

        size = pos_voxel.new_tensor(self.voxel_size, dtype=torch.float)
        low = pos_voxel.new_tensor(self.point_cloud_range[:3], dtype=torch.float)
        centers = (pos_voxel[:, [2, 1, 0]].float() + 0.5) * size + low
        pool_kwargs: Dict[str, Any] = dict(
            pos_batch=batch,
            box_batch=roi_batch,
            max_points_per_cell=self.max_points_per_cell,
        )
        seg_grid = roi_aware_pool3d(voxel_features, centers, rois, self.grid_size, reduce="max", **pool_kwargs)
        part_grid = roi_aware_pool3d(part_features, centers, rois, self.grid_size, reduce="mean", **pool_kwargs)
        return self.roi_head(seg_grid, part_grid)

    def forward(
        self,
        voxels: Tensor,
        pos_voxel: Tensor,
        voxel_num_points: Tensor,
        batch: Tensor,
    ) -> PartA2Output:
        features, stages = self.encode(voxels, pos_voxel, voxel_num_points, batch)
        voxel_features = self.decoder(stages)
        out = self.forward_head(features)
        rois, roi_scores, roi_labels, roi_class_scores, roi_batch = self.proposals(out)
        rcnn_cls, rcnn_reg = self.refine(rois, roi_batch, voxel_features, pos_voxel, batch)

        return {
            "rcnn_cls": rcnn_cls,
            "boxes": decode_roi_residuals(rcnn_reg, rois),
            "roi_labels": roi_labels,
            "roi_scores": roi_scores,
            "roi_class_scores": roi_class_scores,
            "batch": roi_batch,
        }

    @torch.no_grad()
    def decode(self, out: PartA2Output) -> Detection3D:
        r"""Decode a forward output into raw per-RoI detections (no score threshold or NMS).

        Every RoI keeps its first-stage label and is scored by the sigmoid of its second-stage confidence.

        Returns:
            Packed per-RoI detections `{"boxes": (R, 7), "scores": (R,), "labels": (R,), "batch": (R,)}`.
        """
        return {
            "boxes": out["boxes"],
            "scores": out["rcnn_cls"].sigmoid().view(-1),
            "labels": out["roi_labels"],
            "batch": out["batch"],
        }


@register_model(
    "part-a2.kitti.mmdet3d",
    task="detection",
    input_keys=("voxel", "pos_voxel", "voxel_num_points", "batch_pos_voxel"),
    weights=WeightsDict(
        url="hf://torch-pointcloud/part-a2.kitti.mmdet3d/resolve/09585010f125f99945f7ac0c0ac31cebda2ae1ad/model.safetensors",
        dataset="kitti",
        metrics={"mAP": 68.34, "AP/Car": 79.03, "AP/Pedestrian": 55.28, "AP/Cyclist": 70.70},
        classes=("Pedestrian", "Cyclist", "Car"),
        author="mmdet3d",
        license="Apache-2.0",
    ),
    # KITTI 3-class inference: lidar reflectance becomes the model's point feature `x`.
    transform=T.Compose(
        [
            T.Cat(keys=[DataKeys.INTENSITY], dst_key=DataKeys.X, dim=1),
            T.HardVoxelize(
                pos_key=DataKeys.POS,
                feature_key=DataKeys.X,
                voxel_size=(0.05, 0.05, 0.1),
                point_cloud_range=(0.0, -40.0, -3.0, 70.4, 40.0, 1.0),
                max_num_points=5,
                max_num_voxels=40000,
            ),
        ]
    ),
    hparams=dict(
        in_channels=4,
        num_classes=3,
        # Pedestrian, Cyclist, Car.
        anchor_sizes=[[0.8, 0.6, 1.73], [1.76, 0.6, 1.73], [3.9, 1.6, 1.56]],
        anchor_bottom_heights=[-0.6, -0.6, -1.78],
        dir_offset=-math.pi / 2,
        norm_kwargs={"eps": 1e-3, "momentum": 0.01},
    ),
)
def part_a2_mmdet3d_kitti(**hparams: Any) -> PartA2Detection:
    return PartA2Detection(**hparams)


@register_model(
    "part-a2.kitti-car.mmdet3d",
    task="detection",
    input_keys=("voxel", "pos_voxel", "voxel_num_points", "batch_pos_voxel"),
    weights=WeightsDict(
        url="hf://torch-pointcloud/part-a2.kitti-car.mmdet3d/resolve/7b43e8570365c5978be01e4c13f583cb76e59869/model.safetensors",
        dataset="kitti",
        metrics={"mAP": 79.08, "AP/Car": 79.08},
        classes=("Car",),
        author="mmdet3d",
        license="Apache-2.0",
    ),
    # KITTI car inference: lidar reflectance becomes the model's point feature `x`.
    transform=T.Compose(
        [
            T.Cat(keys=[DataKeys.INTENSITY], dst_key=DataKeys.X, dim=1),
            T.HardVoxelize(
                pos_key=DataKeys.POS,
                feature_key=DataKeys.X,
                voxel_size=(0.05, 0.05, 0.1),
                point_cloud_range=(0.0, -40.0, -3.0, 70.4, 40.0, 1.0),
                max_num_points=5,
                max_num_voxels=40000,
            ),
        ]
    ),
    hparams=dict(
        in_channels=4,
        num_classes=1,
        anchor_sizes=[[3.9, 1.6, 1.56]],
        anchor_bottom_heights=[-1.78],
        dir_offset=-math.pi / 2,
        norm_kwargs={"eps": 1e-3, "momentum": 0.01},
    ),
)
def part_a2_mmdet3d_kitti_car(**hparams: Any) -> PartA2Detection:
    return PartA2Detection(**hparams)
