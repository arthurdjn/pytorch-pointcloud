"""Voxel R-CNN detection model.

{{ paper("2012.15712") }}
"""

from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Sequence, Tuple, TypedDict, Union

import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.nn import MLP

import torch_pointcloud.transforms as T
from torch_pointcloud.layers.anchors import AnchorHead, AnchorHeadOutput
from torch_pointcloud.layers.bev_backbone import BEVBackbone
from torch_pointcloud.layers.pointnet2_blocks import PointNet2DecomposedQuerySetAbstraction
from torch_pointcloud.ops.box3d import nms3d
from torch_pointcloud.ops.roi import decode_roi_residuals, roi_grid_points, sample_rois
from torch_pointcloud.utils.data import DataKeys
from torch_pointcloud.utils.imports import _SPCONV_GITHUB_URL, optional_import
from torch_pointcloud.utils.types import Detection3D, FeaturesDict, OptTensor

from ._base import DetectionModel
from ._registry import WeightsDict, register_model
from .second import VoxelBackbone8x

if TYPE_CHECKING:
    import spconv.pytorch as spconv

spconv, _ = optional_import("spconv.pytorch", url=_SPCONV_GITHUB_URL)


class VoxelRCNNOutput(TypedDict):
    r"""Voxel R-CNN output: refined RoI boxes with their second-stage confidence (packed layout).

    Attributes:
        rcnn_cls: Second-stage confidence logit per RoI, shape $(R, 1)$.
        boxes: Refined boxes $(c_x, c_y, c_z, d_x, d_y, d_z, \theta)$, shape $(R, 7)$.
        roi_labels: First-stage class per RoI, shape $(R,)$.
        roi_scores: First-stage score per RoI, shape $(R,)$.
        batch: Per-RoI scene index, shape $(R,)$.
    """

    rcnn_cls: Tensor
    boxes: Tensor
    roi_labels: Tensor
    roi_scores: Tensor
    batch: Tensor


class VoxelRCNNTrainOutput(AnchorHeadOutput):
    r"""Voxel R-CNN training output: the anchor head maps and the refinement of the sampled RoIs.

    Attributes:
        rcnn_cls: Second-stage confidence logit per sampled RoI, shape $(R, 1)$.
        rcnn_reg: Second-stage box residuals per sampled RoI, shape $(R, 7)$.
        rois: Sampled RoIs, shape $(R, 7)$.
        roi_batch: Scene index of every sampled RoI, shape $(R,)$.
        roi_iou: 3D IoU of every sampled RoI with its matched box, shape $(R,)$.
        roi_target_boxes: Matched ground-truth box of every sampled RoI, zeros when unmatched, shape $(R, 7)$.
    """

    rcnn_cls: Tensor
    rcnn_reg: Tensor
    rois: Tensor
    roi_batch: Tensor
    roi_iou: Tensor
    roi_target_boxes: Tensor


class VoxelRCNNBoxHead(nn.Module):
    r"""Voxel R-CNN refinement head: voxel RoI pooling of the sparse stages, then a confidence and a box branch.

    Every RoI is filled with a $G \times G \times G$ grid of points, each pooling the voxel features of every sparse
    stage around it with a
    [`PointNet2DecomposedQuerySetAbstraction`][torch_pointcloud.layers.pointnet2_blocks.PointNet2DecomposedQuerySetAbstraction].
    The grid features, concatenated over the stages and flattened grid point by grid point, go through shared fully
    connected layers, then a class-agnostic confidence branch and a box-residual branch.

    Args:
        in_channels: Feature channels of every pooled stage.
        grid_size: Grid points per RoI side $G$.
        pool_channels: Per-stage, per-scale MLP channels of the voxel pooling.
        pool_radii: Per-stage, per-scale ball radius of the voxel pooling.
        pool_num_neighbors: Per-stage, per-scale neighbor count of the voxel pooling.
        shared_channels: Channels of the shared fully connected layers.
        cls_channels: Hidden channels of the confidence branch.
        reg_channels: Hidden channels of the box branch.
        dropout: Dropout after the first layer of the shared layers and of every branch.
        act: Activation type or callable.
        act_kwargs: Extra activation arguments.
        norm: Normalization type or callable.
        norm_kwargs: Extra normalization arguments.
    """

    def __init__(
        self,
        in_channels: Sequence[int],
        *,
        grid_size: int,
        pool_channels: Sequence[Sequence[Sequence[int]]],
        pool_radii: Sequence[Sequence[float]],
        pool_num_neighbors: Sequence[Sequence[int]],
        shared_channels: Sequence[int],
        cls_channels: Sequence[int],
        reg_channels: Sequence[int],
        dropout: float = 0.3,
        act: Union[str, Callable, None] = "relu",
        act_kwargs: Optional[Dict[str, Any]] = None,
        norm: Union[str, Callable, None] = "batch_norm",
        norm_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        if not len(in_channels) == len(pool_channels) == len(pool_radii) == len(pool_num_neighbors):
            raise ValueError(
                "`in_channels`, `pool_channels`, `pool_radii` and `pool_num_neighbors` need one entry per stage."
            )

        super().__init__()
        self.grid_size = grid_size
        layer_kwargs: Dict[str, Any] = dict(act=act, act_kwargs=act_kwargs, norm=norm, norm_kwargs=norm_kwargs)
        grid_pools = [
            PointNet2DecomposedQuerySetAbstraction(channels, mlp_channels, radii=radii, num_neighbors=k, **layer_kwargs)
            for channels, mlp_channels, radii, k in zip(in_channels, pool_channels, pool_radii, pool_num_neighbors)
        ]
        self.grid_pools = nn.ModuleList(grid_pools)
        grid_channels = sum(pool.out_channels for pool in grid_pools) * grid_size**3
        self.shared = MLP(
            [grid_channels, *shared_channels],
            bias=False,
            dropout=[dropout] + [0.0] * (len(shared_channels) - 1),
            plain_last=False,
            **layer_kwargs,
        )
        self.cls = MLP(
            [shared_channels[-1], *cls_channels, 1],
            bias=[False] * len(cls_channels) + [True],
            dropout=[dropout] + [0.0] * len(cls_channels),
            **layer_kwargs,
        )
        self.reg = MLP(
            [shared_channels[-1], *reg_channels, 7],
            bias=[False] * len(reg_channels) + [True],
            dropout=[dropout] + [0.0] * len(reg_channels),
            **layer_kwargs,
        )

    def forward(
        self,
        rois: Tensor,
        roi_batch: Tensor,
        stage_features: List[Tensor],
        stage_pos: List[Tensor],
        stage_batch: List[Tensor],
    ) -> Tuple[Tensor, Tensor]:
        r"""Pool every RoI and predict its confidence logit $(R, 1)$ and box residuals $(R, 7)$.

        Args:
            rois: RoIs, shape $(R, 7)$.
            roi_batch: Scene index of every RoI, sorted, shape $(R,)$.
            stage_features: Features of every pooled stage, shapes $(V_s, C_s)$.
            stage_pos: Voxel centers of every pooled stage, shapes $(V_s, 3)$.
            stage_batch: Sorted scene index of every pooled stage, shapes $(V_s,)$.
        """
        num_points = self.grid_size**3
        grid = roi_grid_points(rois, self.grid_size).view(-1, 3)
        grid_batch = roi_batch.repeat_interleave(num_points)
        pooled = [
            pool(x, pos, batch, grid, grid_batch).view(rois.size(0), num_points, -1)
            for pool, x, pos, batch in zip(self.grid_pools, stage_features, stage_pos, stage_batch)
        ]
        shared = self.shared(torch.cat(pooled, dim=-1).reshape(rois.size(0), -1))
        return self.cls(shared), self.reg(shared)


class VoxelRCNNDetection(DetectionModel):
    r"""Voxel R-CNN two-stage 3D object detector (packed point format).

    Reference: :arxiv: [Voxel R-CNN: Towards High Performance Voxel-based 3D Object Detection](https://arxiv.org/abs/2012.15712) (Deng et al., 2021).
    Reference implementation: :github: [open-mmlab/OpenPCDet](https://github.com/open-mmlab/OpenPCDet).

    The first stage is the single-stage detector of
    [`SECONDDetection`][torch_pointcloud.models.second.SECONDDetection]: a sparse 3D voxel backbone, a dense BEV
    backbone and an anchor head, whose best anchors after rotated NMS become the RoIs. The second stage refines
    every RoI with [`VoxelRCNNBoxHead`][torch_pointcloud.models.voxel_rcnn.VoxelRCNNBoxHead], which pools the
    three strided stages of the sparse backbone around a grid of points inside the RoI.

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
        train_proposal_pre_maxsize: `proposal_pre_maxsize` in training.
        train_proposal_post_maxsize: `proposal_post_maxsize` in training, before the RoI sampling.
        train_proposal_nms_thresh: `proposal_nms_thresh` in training.
        roi_num_samples: RoIs sampled per scene in training.
        roi_foreground_ratio: Largest share of foreground RoIs among the samples of a scene.
        roi_foreground_threshold: IoU at or above which a proposal is a foreground RoI.
        roi_hard_background_threshold: IoU at or above which a background RoI is hard.
        roi_hard_background_ratio: Share of hard RoIs among the background samples.
        grid_size: Grid points per RoI side.
        pool_channels: Per-stage, per-scale MLP channels of the voxel RoI pooling.
        pool_radii: Per-stage, per-scale ball radius of the voxel RoI pooling.
        pool_num_neighbors: Per-stage, per-scale neighbor count of the voxel RoI pooling.
        shared_channels: Channels of the shared fully connected layers of the refinement head.
        cls_channels: Hidden channels of the refinement confidence branch.
        reg_channels: Hidden channels of the refinement box branch.
        dropout: Dropout of the refinement head.
        act: Activation type or callable.
        act_kwargs: Extra activation arguments.
        norm: Normalization type or callable.
        norm_kwargs: Extra normalization arguments of the 3D and 2D backbones.
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        *,
        voxel_size: Sequence[float],
        point_cloud_range: Sequence[float],
        anchor_sizes: Sequence[Sequence[float]],
        anchor_bottom_heights: Sequence[float],
        anchor_rotations: Sequence[float] = (0.0, 1.57),
        feature_map_stride: int = 8,
        layer_nums: Sequence[int] = (5, 5),
        layer_strides: Sequence[int] = (1, 2),
        num_filters: Sequence[int] = (64, 128),
        upsample_strides: Sequence[int] = (1, 2),
        num_upsample_filters: Sequence[int] = (128, 128),
        num_dir_bins: int = 2,
        dir_offset: float = 0.78539,
        dir_limit_offset: float = 0.0,
        proposal_pre_maxsize: int = 2048,
        proposal_post_maxsize: int = 100,
        proposal_nms_thresh: float = 0.7,
        train_proposal_pre_maxsize: int = 9000,
        train_proposal_post_maxsize: int = 512,
        train_proposal_nms_thresh: float = 0.8,
        roi_num_samples: int = 128,
        roi_foreground_ratio: float = 0.5,
        roi_foreground_threshold: float = 0.55,
        roi_hard_background_threshold: float = 0.1,
        roi_hard_background_ratio: float = 0.8,
        grid_size: int = 6,
        pool_channels: Sequence[Sequence[Sequence[int]]] = (((32, 32),), ((32, 32),), ((32, 32),)),
        pool_radii: Sequence[Sequence[float]] = ((0.4,), (0.8,), (1.6,)),
        pool_num_neighbors: Sequence[Sequence[int]] = ((16,), (16,), (16,)),
        shared_channels: Sequence[int] = (256, 256),
        cls_channels: Sequence[int] = (256, 256),
        reg_channels: Sequence[int] = (256, 256),
        dropout: float = 0.3,
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
        self.train_proposal_pre_maxsize = train_proposal_pre_maxsize
        self.train_proposal_post_maxsize = train_proposal_post_maxsize
        self.train_proposal_nms_thresh = train_proposal_nms_thresh
        self.roi_num_samples = roi_num_samples
        self.roi_foreground_ratio = roi_foreground_ratio
        self.roi_foreground_threshold = roi_foreground_threshold
        self.roi_hard_background_threshold = roi_hard_background_threshold
        self.roi_hard_background_ratio = roi_hard_background_ratio
        self.grid_size = grid_size
        self.pool_channels = pool_channels
        self.pool_radii = pool_radii
        self.pool_num_neighbors = pool_num_neighbors
        self.shared_channels = shared_channels
        self.cls_channels = cls_channels
        self.reg_channels = reg_channels
        self.dropout = dropout
        self.act = act
        self.act_kwargs = act_kwargs
        self.norm = norm
        self.norm_kwargs = norm_kwargs

        grid = [int(round((point_cloud_range[i + 3] - point_cloud_range[i]) / voxel_size[i])) for i in range(3)]
        self.grid_size_3d: Tuple[int, int, int] = (grid[0], grid[1], grid[2])
        # spconv spatial shape is (z, y, x) with an extra +1 on z.
        self.sparse_shape: List[int] = [grid[2] + 1, grid[1], grid[0]]

        self.backbone_3d = self.configure_backbone_3d()
        self.backbone = self.configure_backbone()
        self.head = self.configure_head()
        self.roi_head = self.configure_roi_head()

    def configure_backbone_3d(self) -> VoxelBackbone8x:
        """Build the sparse 3D voxel backbone."""
        return VoxelBackbone8x(
            self.in_channels,
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

    def configure_roi_head(self) -> VoxelRCNNBoxHead:
        """Build the voxel RoI pooling refinement head."""
        # The head uses the framework-default norm hyperparameters, so the backbones' `norm_kwargs` are not
        # threaded into it.
        return VoxelRCNNBoxHead(
            self.backbone_3d.stage_channels[1:],
            grid_size=self.grid_size,
            pool_channels=self.pool_channels,
            pool_radii=self.pool_radii,
            pool_num_neighbors=self.pool_num_neighbors,
            shared_channels=self.shared_channels,
            cls_channels=self.cls_channels,
            reg_channels=self.reg_channels,
            dropout=self.dropout,
            act=self.act,
            act_kwargs=self.act_kwargs,
            norm=self.norm,
        )

    def encode(
        self,
        voxels: Tensor,
        pos_voxel: Tensor,
        voxel_num_points: Tensor,
        batch_voxel: Tensor,
    ) -> Tuple[Tensor, List[FeaturesDict]]:
        r"""Run the sparse and BEV backbones.

        Returns:
            `(features, stages)`: the BEV features entering the anchor head and the three strided sparse stages
            (see `VoxelBackbone8x`).
        """
        voxel_indices = torch.cat([batch_voxel.view(-1, 1).to(pos_voxel), pos_voxel], dim=1)
        batch_size = int(batch_voxel.max().item()) + 1

        # Average the points (xyz + features) inside each voxel.
        normalizer = torch.clamp_min(voxel_num_points.view(-1, 1), min=1.0).type_as(voxels)
        voxel_features = voxels.sum(dim=1) / normalizer

        x = spconv.SparseConvTensor(
            features=voxel_features,
            indices=voxel_indices.int(),
            spatial_shape=self.sparse_shape,
            batch_size=batch_size,
        )
        encoded, stages = self.backbone_3d(x, return_intermediates=True)

        dense = encoded.dense()
        b, c, d, h, w = dense.shape
        bev = dense.view(b, c * d, h, w)
        features = self.backbone(bev)
        return features, stages[1:]

    def forward_features(
        self,
        voxels: Tensor,
        pos_voxel: Tensor,
        voxel_num_points: Tensor,
        batch_voxel: Tensor,
    ) -> Tensor:
        features, _ = self.encode(voxels, pos_voxel, voxel_num_points, batch_voxel)
        return features

    def forward_head(self, features: Tensor) -> AnchorHeadOutput:
        return self.head(features)

    @torch.no_grad()
    def proposals(self, out: AnchorHeadOutput) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        r"""Select the RoIs of every scene from the anchor head output.

        The `proposal_pre_maxsize` best anchors of every scene (by their top class score) go through a class-agnostic
        rotated BEV NMS, which keeps at most `proposal_post_maxsize` of them; training uses the `train_proposal_*`
        settings.

        Returns:
            `(rois, roi_scores, roi_labels, roi_batch)` of shapes $(R, 7)$, $(R,)$, $(R,)$ and $(R,)$.
        """
        det = self.head.decode(out)
        keep = nms3d(
            det["boxes"],
            det["scores"],
            self.train_proposal_nms_thresh if self.training else self.proposal_nms_thresh,
            batch=det["batch"],
            rotated=True,
            max_keep=self.train_proposal_post_maxsize if self.training else self.proposal_post_maxsize,
            max_candidates=self.train_proposal_pre_maxsize if self.training else self.proposal_pre_maxsize,
        )
        return det["boxes"][keep], det["scores"][keep], det["labels"][keep], det["batch"][keep]

    def refine(self, rois: Tensor, roi_batch: Tensor, stages: List[FeaturesDict]) -> Tuple[Tensor, Tensor]:
        r"""Pool the strided sparse stages inside every RoI and refine it.

        Args:
            rois: RoIs, shape $(R, 7)$.
            roi_batch: Scene index of every RoI, sorted, shape $(R,)$.
            stages: The strided sparse stages (see `encode`).

        Returns:
            `(rcnn_cls, rcnn_reg)`: the confidence logit $(R, 1)$ and the box residuals $(R, 7)$ of every RoI.
        """
        size = rois.new_tensor(self.voxel_size)
        low = rois.new_tensor(self.point_cloud_range[:3])
        stage_pos = [
            (stage["pos_grid"][:, [2, 1, 0]].float() + 0.5) * (size * stride) + low
            for stage, stride in zip(stages, self.backbone_3d.stage_strides[1:])
        ]
        stage_features = [stage["x"] for stage in stages]
        stage_batch = [stage["batch"] for stage in stages]
        return self.roi_head(rois, roi_batch, stage_features, stage_pos, stage_batch)

    def forward(
        self,
        voxels: Tensor,
        pos_voxel: Tensor,
        voxel_num_points: Tensor,
        batch_voxel: Tensor,
        gt_boxes: OptTensor = None,
        gt_labels: OptTensor = None,
        gt_batch: OptTensor = None,
    ) -> Union[VoxelRCNNOutput, VoxelRCNNTrainOutput]:
        r"""Run both stages; in training the ground truth drives the sampling of the RoIs that are refined.

        Args:
            voxels: Points of every voxel, shape $(V, P, C)$.
            pos_voxel: Voxel coordinates $(z, y, x)$, shape $(V, 3)$.
            voxel_num_points: Points per voxel, shape $(V,)$.
            batch_voxel: Scene index of every voxel, shape $(V,)$.
            gt_boxes: Ground-truth boxes $(K, 7)$, required in training.
            gt_labels: Ground-truth $0$-based classes, shape $(K,)$, required in training.
            gt_batch: Scene index of every box, shape $(K,)$, required in training.

        Returns:
            A `VoxelRCNNTrainOutput` in training, otherwise a `VoxelRCNNOutput`.
        """
        if self.training and (gt_boxes is None or gt_labels is None or gt_batch is None):
            raise ValueError("Training needs `gt_boxes`, `gt_labels` and `gt_batch` to sample the RoIs.")

        features, stages = self.encode(voxels, pos_voxel, voxel_num_points, batch_voxel)
        out = self.forward_head(features)
        rois, roi_scores, roi_labels, roi_batch = self.proposals(out)

        if self.training:
            assert gt_boxes is not None and gt_labels is not None and gt_batch is not None
            samples = sample_rois(
                rois,
                gt_boxes,
                roi_labels=roi_labels,
                box_labels=gt_labels,
                roi_batch=roi_batch,
                box_batch=gt_batch,
                num_samples=self.roi_num_samples,
                foreground_ratio=self.roi_foreground_ratio,
                foreground_threshold=self.roi_foreground_threshold,
                hard_background_threshold=self.roi_hard_background_threshold,
                hard_background_ratio=self.roi_hard_background_ratio,
            )
            rois, roi_batch = rois[samples["index"]], roi_batch[samples["index"]]
            matched = samples["target_index"] >= 0
            target_boxes = torch.zeros_like(rois)
            target_boxes[matched] = gt_boxes[samples["target_index"][matched]]
            rcnn_cls, rcnn_reg = self.refine(rois, roi_batch, stages)
            return {
                **out,
                "rcnn_cls": rcnn_cls,
                "rcnn_reg": rcnn_reg,
                "rois": rois,
                "roi_batch": roi_batch,
                "roi_iou": samples["iou"],
                "roi_target_boxes": target_boxes,
            }

        rcnn_cls, rcnn_reg = self.refine(rois, roi_batch, stages)
        return {
            "rcnn_cls": rcnn_cls,
            "boxes": decode_roi_residuals(rcnn_reg, rois),
            "roi_labels": roi_labels,
            "roi_scores": roi_scores,
            "batch": roi_batch,
        }

    @torch.no_grad()
    def decode(self, out: VoxelRCNNOutput) -> Detection3D:
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
    "voxel-rcnn.kitti-car.openpcdet",
    task="detection",
    input_keys=("voxel", "pos_voxel", "voxel_num_points", "batch_pos_voxel"),
    weights=WeightsDict(
        url="hf://torch-pointcloud/voxel-rcnn.kitti-car.openpcdet/resolve/0000000000000000000000000000000000000000/model.safetensors",
        dataset="kitti",
        metrics={"mAP": 84.53, "AP/Car": 84.53},
        classes=("Car",),
        author="openpcdet",
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
        voxel_size=(0.05, 0.05, 0.1),
        point_cloud_range=(0.0, -40.0, -3.0, 70.4, 40.0, 1.0),
        anchor_sizes=[[3.9, 1.6, 1.56]],
        anchor_bottom_heights=[-1.78],
        feature_map_stride=8,
        norm_kwargs={"eps": 1e-3, "momentum": 0.01},
    ),
)
def voxel_rcnn_openpcdet_kitti_car(**hparams: Any) -> VoxelRCNNDetection:
    return VoxelRCNNDetection(**hparams)
