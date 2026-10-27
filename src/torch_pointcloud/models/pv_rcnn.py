"""PV-RCNN detection model.

{{ paper("1912.13192") }}
"""

from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Sequence, Tuple, TypedDict, Union

import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.nn import MLP

import torch_pointcloud.transforms as T
from torch_pointcloud.layers.anchors import AnchorHead, AnchorHeadOutput
from torch_pointcloud.layers.bev_backbone import BEVBackbone
from torch_pointcloud.layers.pointnet2_blocks import PointNet2QuerySetAbstraction
from torch_pointcloud.ops.box3d import decode_roi_residuals, nms3d
from torch_pointcloud.ops.cluster import fps
from torch_pointcloud.utils.data import DataKeys
from torch_pointcloud.utils.imports import _SPCONV_GITHUB_URL, optional_import
from torch_pointcloud.utils.types import Detection3D

from ._base import DetectionModel
from ._registry import WeightsDict, register_model
from .second import VoxelBackbone8x

if TYPE_CHECKING:
    import spconv.pytorch as spconv

spconv, _ = optional_import("spconv.pytorch", url=_SPCONV_GITHUB_URL)


class PVRCNNOutput(TypedDict):
    r"""PV-RCNN output: refined RoI boxes with their second-stage confidence (packed layout).

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


def bilinear_interpolate(image: Tensor, x: Tensor, y: Tensor) -> Tensor:
    r"""Sample a $(H, W, C)$ map at continuous pixel coordinates by bilinear interpolation.

    The four neighbor indices are clamped to the map and the weights are taken from the clamped indices, so the
    weights of a sample in the last row or column cancel out and it comes out as zero.

    Args:
        image: Feature map, shape $(H, W, C)$.
        x: Column coordinates, shape $(N,)$.
        y: Row coordinates, shape $(N,)$.

    Returns:
        The sampled features, shape $(N, C)$.
    """
    height, width = image.shape[:2]
    x0 = x.floor().long().clamp(0, width - 1)
    x1 = (x.floor().long() + 1).clamp(0, width - 1)
    y0 = y.floor().long().clamp(0, height - 1)
    y1 = (y.floor().long() + 1).clamp(0, height - 1)

    wa = (x1.type_as(x) - x) * (y1.type_as(y) - y)
    wb = (x1.type_as(x) - x) * (y - y0.type_as(y))
    wc = (x - x0.type_as(x)) * (y1.type_as(y) - y)
    wd = (x - x0.type_as(x)) * (y - y0.type_as(y))
    return (
        image[y0, x0] * wa.unsqueeze(1)
        + image[y1, x0] * wb.unsqueeze(1)
        + image[y0, x1] * wc.unsqueeze(1)
        + image[y1, x1] * wd.unsqueeze(1)
    )


def roi_grid_points(rois: Tensor, grid_size: int) -> Tensor:
    r"""Place a regular $G \times G \times G$ grid of points inside every oriented box.

    The points sit at the cell centers of the box, enumerated with $x$ slowest and $z$ fastest.

    Args:
        rois: Boxes $(c_x, c_y, c_z, d_x, d_y, d_z, \theta)$, shape $(R, 7)$.
        grid_size: Points per box side $G$.

    Returns:
        The grid points, shape $(R, G^3, 3)$.
    """
    steps = (torch.arange(grid_size, device=rois.device, dtype=rois.dtype) + 0.5) / grid_size - 0.5
    local = torch.stack(torch.meshgrid(steps, steps, steps, indexing="ij"), dim=-1).view(1, -1, 3)
    local = local * rois[:, None, 3:6]

    cos, sin = torch.cos(rois[:, 6:7]), torch.sin(rois[:, 6:7])
    x = local[..., 0] * cos - local[..., 1] * sin
    y = local[..., 0] * sin + local[..., 1] * cos
    points = torch.stack([x, y, local[..., 2]], dim=-1)
    return points + rois[:, None, 0:3]


class VoxelSetAbstraction(nn.Module):
    r"""Encode keypoints sampled from the raw points with multi-scale point, voxel and BEV features.

    Farthest point sampling picks `num_keypoints` keypoints per scene among the raw points. Every keypoint samples
    the BEV map by bilinear interpolation and pools the raw points and the voxel centers of every sparse stage with
    a [`PointNet2QuerySetAbstraction`][torch_pointcloud.layers.pointnet2_blocks.PointNet2QuerySetAbstraction];
    a linear layer fuses the concatenated features.

    Args:
        num_keypoints: Keypoints sampled per scene.
        voxel_size: Voxel size $(v_x, v_y, v_z)$ of the input grid.
        point_cloud_range: Range $(x_\min, y_\min, z_\min, x_\max, y_\max, z_\max)$.
        point_channels: Feature channels of the raw points (without xyz).
        point_mlp_channels: Per-scale MLP channels of the raw-point abstraction.
        point_radii: Per-scale ball radius of the raw-point abstraction.
        point_num_neighbors: Per-scale neighbor count of the raw-point abstraction.
        voxel_channels: Feature channels of every sparse stage.
        voxel_strides: Downsampling stride of every sparse stage relative to the input grid.
        voxel_mlp_channels: Per-stage, per-scale MLP channels of the voxel abstractions.
        voxel_radii: Per-stage, per-scale ball radius of the voxel abstractions.
        voxel_num_neighbors: Per-stage, per-scale neighbor count of the voxel abstractions.
        bev_channels: Channels of the BEV map.
        bev_stride: Stride of the BEV map relative to the input grid.
        out_channels: Channels of the fused keypoint features.
        act: Activation type or callable.
        act_kwargs: Extra activation arguments.
        norm: Normalization type or callable.
        norm_kwargs: Extra normalization arguments.
    """

    def __init__(
        self,
        num_keypoints: int,
        voxel_size: Sequence[float],
        point_cloud_range: Sequence[float],
        *,
        point_channels: int,
        point_mlp_channels: Sequence[Sequence[int]],
        point_radii: Sequence[float],
        point_num_neighbors: Sequence[int],
        voxel_channels: Sequence[int],
        voxel_strides: Sequence[int],
        voxel_mlp_channels: Sequence[Sequence[Sequence[int]]],
        voxel_radii: Sequence[Sequence[float]],
        voxel_num_neighbors: Sequence[Sequence[int]],
        bev_channels: int,
        bev_stride: int,
        out_channels: int,
        act: Union[str, Callable, None] = "relu",
        act_kwargs: Optional[Dict[str, Any]] = None,
        norm: Union[str, Callable, None] = "batch_norm",
        norm_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        if not len(voxel_channels) == len(voxel_strides) == len(voxel_mlp_channels):
            raise ValueError("`voxel_channels`, `voxel_strides` and `voxel_mlp_channels` need one entry per stage.")

        super().__init__()
        self.num_keypoints = num_keypoints
        self.voxel_size = tuple(voxel_size)
        self.point_cloud_range = tuple(point_cloud_range)
        self.voxel_strides = tuple(voxel_strides)
        self.bev_stride = bev_stride
        layer_kwargs: Dict[str, Any] = dict(act=act, act_kwargs=act_kwargs, norm=norm, norm_kwargs=norm_kwargs)

        self.point_abstraction = PointNet2QuerySetAbstraction(
            point_channels,
            point_mlp_channels,
            radii=point_radii,
            num_neighbors=point_num_neighbors,
            **layer_kwargs,
        )
        voxel_abstractions = [
            PointNet2QuerySetAbstraction(channels, mlp_channels, radii=radii, num_neighbors=k, **layer_kwargs)
            for channels, mlp_channels, radii, k in zip(
                voxel_channels,
                voxel_mlp_channels,
                voxel_radii,
                voxel_num_neighbors,
            )
        ]
        self.voxel_abstractions = nn.ModuleList(voxel_abstractions)
        self.in_channels = (
            bev_channels
            + self.point_abstraction.out_channels
            + sum(abstraction.out_channels for abstraction in voxel_abstractions)
        )
        self.out_channels = out_channels
        self.fusion = MLP([self.in_channels, out_channels], bias=False, plain_last=False, **layer_kwargs)

    def voxel_centers(self, indices: Tensor, stride: int) -> Tensor:
        r"""Metric centers of voxels given as $(b, z, y, x)$ indices on a grid `stride` times coarser."""
        size = indices.new_tensor(self.voxel_size, dtype=torch.float) * stride
        low = indices.new_tensor(self.point_cloud_range[:3], dtype=torch.float)
        return (indices[:, [3, 2, 1]].float() + 0.5) * size + low

    def forward(
        self,
        x: Tensor,
        pos: Tensor,
        batch: Tensor,
        voxel_features: List[Tensor],
        voxel_indices: List[Tensor],
        bev: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        r"""Sample the keypoints and gather their features.

        Args:
            x: Raw point features, shape $(N, C_p)$.
            pos: Raw point coordinates, shape $(N, 3)$.
            batch: Sorted raw point scene index, shape $(N,)$.
            voxel_features: Features of every sparse stage, shapes $(V_s, C_s)$.
            voxel_indices: Indices $(b, z, y, x)$ of every sparse stage, shapes $(V_s, 4)$, sorted by scene.
            bev: BEV feature map, shape $(B, C, H, W)$.

        Returns:
            `(keypoint_pos, keypoint_batch, features, fused)`: keypoints $(B K, 3)$, their scene index
            $(B K,)$, the concatenated features $(B K, C)$ and the fused features $(B K, C_\text{out})$.
        """
        idx = fps(pos, batch, num_nodes=self.num_keypoints, random_start=False)
        keypoint_pos, keypoint_batch = pos[idx], batch[idx]

        x_idx = (keypoint_pos[:, 0] - self.point_cloud_range[0]) / self.voxel_size[0] / self.bev_stride
        y_idx = (keypoint_pos[:, 1] - self.point_cloud_range[1]) / self.voxel_size[1] / self.bev_stride
        bev_maps = bev.permute(0, 2, 3, 1)
        features = [
            torch.cat(
                [
                    bilinear_interpolate(bev_maps[b], x_idx[keypoint_batch == b], y_idx[keypoint_batch == b])
                    for b in range(bev.size(0))
                ]
            )
        ]
        features.append(self.point_abstraction(x, pos, batch, keypoint_pos, keypoint_batch))
        for abstraction, stride, stage_features, indices in zip(
            self.voxel_abstractions,
            self.voxel_strides,
            voxel_features,
            voxel_indices,
        ):
            centers = self.voxel_centers(indices, stride)
            features.append(abstraction(stage_features, centers, indices[:, 0].long(), keypoint_pos, keypoint_batch))

        keypoint_features = torch.cat(features, dim=1)
        fused = self.fusion(keypoint_features)
        return keypoint_pos, keypoint_batch, keypoint_features, fused


class PVRCNNBoxHead(nn.Module):
    r"""PV-RCNN refinement head: RoI-grid pooling of the keypoint features, then a confidence and a box branch.

    Every RoI is filled with a $G \times G \times G$ grid of points, each pooling the weighted keypoint features
    around it with a [`PointNet2QuerySetAbstraction`][torch_pointcloud.layers.pointnet2_blocks.PointNet2QuerySetAbstraction].
    The flattened grid features go through shared fully connected layers, then a class-agnostic confidence branch
    and a box-residual branch.

    Args:
        in_channels: Channels of the keypoint features.
        grid_size: Grid points per RoI side $G$.
        pool_mlp_channels: Per-scale MLP channels of the grid pooling.
        pool_radii: Per-scale ball radius of the grid pooling.
        pool_num_neighbors: Per-scale neighbor count of the grid pooling.
        shared_channels: Channels of the shared fully connected layers.
        cls_channels: Hidden channels of the confidence branch.
        reg_channels: Hidden channels of the box branch.
        dropout: Dropout after the first layer of every branch.
        act: Activation type or callable.
        act_kwargs: Extra activation arguments.
        norm: Normalization type or callable.
        norm_kwargs: Extra normalization arguments.
    """

    def __init__(
        self,
        in_channels: int,
        *,
        grid_size: int,
        pool_mlp_channels: Sequence[Sequence[int]],
        pool_radii: Sequence[float],
        pool_num_neighbors: Sequence[int],
        shared_channels: Sequence[int],
        cls_channels: Sequence[int],
        reg_channels: Sequence[int],
        dropout: float = 0.3,
        act: Union[str, Callable, None] = "relu",
        act_kwargs: Optional[Dict[str, Any]] = None,
        norm: Union[str, Callable, None] = "batch_norm",
        norm_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()
        self.grid_size = grid_size
        layer_kwargs: Dict[str, Any] = dict(act=act, act_kwargs=act_kwargs, norm=norm, norm_kwargs=norm_kwargs)
        self.grid_pool = PointNet2QuerySetAbstraction(
            in_channels,
            pool_mlp_channels,
            radii=pool_radii,
            num_neighbors=pool_num_neighbors,
            **layer_kwargs,
        )
        grid_channels = self.grid_pool.out_channels * grid_size**3
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
        keypoint_features: Tensor,
        keypoint_pos: Tensor,
        keypoint_batch: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        r"""Pool every RoI and predict its confidence logit $(R, 1)$ and box residuals $(R, 7)$."""
        num_points = self.grid_size**3
        grid = roi_grid_points(rois, self.grid_size).view(-1, 3)
        grid_batch = roi_batch.repeat_interleave(num_points)
        pooled = self.grid_pool(keypoint_features, keypoint_pos, keypoint_batch, grid, grid_batch)

        # Flatten channel-major: feature c of grid point g lands at c * G^3 + g.
        pooled = pooled.view(rois.size(0), num_points, -1).transpose(1, 2).reshape(rois.size(0), -1)
        shared = self.shared(pooled)
        return self.cls(shared), self.reg(shared)


class PVRCNNDetection(DetectionModel):
    r"""PV-RCNN two-stage 3D object detector (packed point format).

    Reference: :arxiv: [PV-RCNN: Point-Voxel Feature Set Abstraction for 3D Object Detection](https://arxiv.org/abs/1912.13192) (Shi et al., 2020).
    Reference implementation: :github: [open-mmlab/OpenPCDet](https://github.com/open-mmlab/OpenPCDet).

    The first stage is the single-stage detector of
    [`SECONDDetection`][torch_pointcloud.models.second.SECONDDetection]: a sparse 3D voxel backbone, a dense BEV
    backbone and an anchor head, whose best anchors after rotated NMS become the RoIs. The second stage samples
    keypoints among the raw points, encodes them with
    [`VoxelSetAbstraction`][torch_pointcloud.models.pv_rcnn.VoxelSetAbstraction], weights them by a foreground
    head and refines every RoI with [`PVRCNNBoxHead`][torch_pointcloud.models.pv_rcnn.PVRCNNBoxHead].

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
        num_keypoints: Keypoints sampled per scene.
        point_mlp_channels: Per-scale MLP channels of the raw-point abstraction.
        point_radii: Per-scale ball radius of the raw-point abstraction.
        point_num_neighbors: Per-scale neighbor count of the raw-point abstraction.
        voxel_mlp_channels: Per-stage, per-scale MLP channels of the voxel abstractions.
        voxel_radii: Per-stage, per-scale ball radius of the voxel abstractions.
        voxel_num_neighbors: Per-stage, per-scale neighbor count of the voxel abstractions.
        keypoint_channels: Channels of the fused keypoint features.
        keypoint_head_channels: Hidden channels of the keypoint foreground head.
        grid_size: Grid points per RoI side.
        pool_mlp_channels: Per-scale MLP channels of the RoI-grid pooling.
        pool_radii: Per-scale ball radius of the RoI-grid pooling.
        pool_num_neighbors: Per-scale neighbor count of the RoI-grid pooling.
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
        num_filters: Sequence[int] = (128, 256),
        upsample_strides: Sequence[int] = (1, 2),
        num_upsample_filters: Sequence[int] = (256, 256),
        num_dir_bins: int = 2,
        dir_offset: float = 0.78539,
        dir_limit_offset: float = 0.0,
        proposal_pre_maxsize: int = 1024,
        proposal_post_maxsize: int = 100,
        proposal_nms_thresh: float = 0.7,
        num_keypoints: int = 2048,
        point_mlp_channels: Sequence[Sequence[int]] = ((16, 16), (16, 16)),
        point_radii: Sequence[float] = (0.4, 0.8),
        point_num_neighbors: Sequence[int] = (16, 16),
        voxel_mlp_channels: Sequence[Sequence[Sequence[int]]] = (
            ((16, 16), (16, 16)),
            ((32, 32), (32, 32)),
            ((64, 64), (64, 64)),
            ((64, 64), (64, 64)),
        ),
        voxel_radii: Sequence[Sequence[float]] = ((0.4, 0.8), (0.8, 1.2), (1.2, 2.4), (2.4, 4.8)),
        voxel_num_neighbors: Sequence[Sequence[int]] = ((16, 16), (16, 32), (16, 32), (16, 32)),
        keypoint_channels: int = 128,
        keypoint_head_channels: Sequence[int] = (256, 256),
        grid_size: int = 6,
        pool_mlp_channels: Sequence[Sequence[int]] = ((64, 64), (64, 64)),
        pool_radii: Sequence[float] = (0.8, 1.6),
        pool_num_neighbors: Sequence[int] = (16, 16),
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
        self.num_keypoints = num_keypoints
        self.point_mlp_channels = point_mlp_channels
        self.point_radii = point_radii
        self.point_num_neighbors = point_num_neighbors
        self.voxel_mlp_channels = voxel_mlp_channels
        self.voxel_radii = voxel_radii
        self.voxel_num_neighbors = voxel_num_neighbors
        self.keypoint_channels = keypoint_channels
        self.keypoint_head_channels = keypoint_head_channels
        self.grid_size = grid_size
        self.pool_mlp_channels = pool_mlp_channels
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
        self.keypoint_encoder = self.configure_keypoint_encoder()
        self.keypoint_head = self.configure_keypoint_head()
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

    def configure_keypoint_encoder(self) -> VoxelSetAbstraction:
        """Build the keypoint encoder over the raw points, the sparse stages and the BEV map."""
        # The heads use the framework-default norm hyperparameters, so the backbones' `norm_kwargs` are not
        # threaded into them.
        return VoxelSetAbstraction(
            self.num_keypoints,
            self.voxel_size,
            self.point_cloud_range,
            point_channels=self.in_channels - 3,
            point_mlp_channels=self.point_mlp_channels,
            point_radii=self.point_radii,
            point_num_neighbors=self.point_num_neighbors,
            voxel_channels=(16, 32, 64, 64),
            voxel_strides=(1, 2, 4, 8),
            voxel_mlp_channels=self.voxel_mlp_channels,
            voxel_radii=self.voxel_radii,
            voxel_num_neighbors=self.voxel_num_neighbors,
            bev_channels=self.backbone_3d.out_channels * 2,
            # The keypoints sample the height-compressed sparse output, 8 times coarser than the voxel grid.
            bev_stride=8,
            out_channels=self.keypoint_channels,
            act=self.act,
            act_kwargs=self.act_kwargs,
            norm=self.norm,
        )

    def configure_keypoint_head(self) -> MLP:
        """Build the keypoint foreground head."""
        return MLP(
            [self.keypoint_encoder.in_channels, *self.keypoint_head_channels, 1],
            bias=[False] * len(self.keypoint_head_channels) + [True],
            act=self.act,
            act_kwargs=self.act_kwargs,
            norm=self.norm,
        )

    def configure_roi_head(self) -> PVRCNNBoxHead:
        """Build the RoI-grid refinement head."""
        return PVRCNNBoxHead(
            self.keypoint_channels,
            grid_size=self.grid_size,
            pool_mlp_channels=self.pool_mlp_channels,
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
    ) -> Tuple[Tensor, Tensor, List[Tensor], List[Tensor]]:
        r"""Run the sparse and BEV backbones.

        Returns:
            `(features, bev, stage_features, stage_indices)`: the BEV features entering the anchor head, the
            height-compressed sparse output, and the features and $(b, z, y, x)$ indices of the four sparse stages.
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
        x = self.backbone_3d.conv_input(x)
        x = self.backbone_3d.conv1(x)
        stage_features, stage_indices = [x.features], [x.indices]
        for conv in (self.backbone_3d.conv2, self.backbone_3d.conv3, self.backbone_3d.conv4):
            x = conv(x)
            # Strided sparse convs return their voxels in a library-dependent order; visit them in (b, z, y, x)
            # order so the ball queries, which keep the first neighbors they meet, do not depend on the library.
            indices = x.indices.long()
            depth, height, width = x.spatial_shape
            order = (
                ((indices[:, 0] * depth + indices[:, 1]) * height + indices[:, 2]) * width + indices[:, 3]
            ).argsort()
            stage_features.append(x.features[order])
            stage_indices.append(x.indices[order])
        encoded = self.backbone_3d.conv_out(x)

        dense = encoded.dense()
        b, c, d, h, w = dense.shape
        bev = dense.view(b, c * d, h, w)
        features = self.backbone(bev)
        return features, bev, stage_features, stage_indices

    def forward_features(
        self,
        voxels: Tensor,
        pos_voxel: Tensor,
        voxel_num_points: Tensor,
        batch_voxel: Tensor,
    ) -> Tensor:
        features, _, _, _ = self.encode(voxels, pos_voxel, voxel_num_points, batch_voxel)
        return features

    def forward_head(self, features: Tensor) -> AnchorHeadOutput:
        return self.head(features)

    @torch.no_grad()
    def proposals(self, out: AnchorHeadOutput) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        r"""Select the RoIs of every scene from the anchor head output.

        The `proposal_pre_maxsize` best anchors (by their top class score) go through a class-agnostic rotated BEV
        NMS, which keeps at most `proposal_post_maxsize` of them.

        Returns:
            `(rois, roi_scores, roi_labels, roi_batch)` of shapes $(R, 7)$, $(R,)$, $(R,)$ and $(R,)$.
        """
        det = self.head.decode(out)
        rois, scores, labels, batch = [], [], [], []
        for b in range(int(det["batch"].max().item()) + 1):
            scene = (det["batch"] == b).nonzero().view(-1)
            top_scores, top = det["scores"][scene].topk(min(self.proposal_pre_maxsize, scene.numel()))
            top = scene[top]
            keep = top[
                nms3d(
                    det["boxes"][top],
                    top_scores,
                    self.proposal_nms_thresh,
                    rotated=True,
                    max_keep=self.proposal_post_maxsize,
                )
            ]
            rois.append(det["boxes"][keep])
            scores.append(det["scores"][keep])
            labels.append(det["labels"][keep])
            batch.append(det["batch"][keep])
        return torch.cat(rois), torch.cat(scores), torch.cat(labels), torch.cat(batch)

    def refine(
        self,
        rois: Tensor,
        roi_batch: Tensor,
        x: Tensor,
        pos: Tensor,
        batch: Tensor,
        stage_features: List[Tensor],
        stage_indices: List[Tensor],
        bev: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        r"""Encode the keypoints, weight them by the foreground head and refine every RoI.

        Args:
            rois: RoIs, shape $(R, 7)$.
            roi_batch: Scene index of every RoI, sorted, shape $(R,)$.
            x: Raw point features, shape $(N, C_p)$.
            pos: Raw point coordinates, shape $(N, 3)$.
            batch: Sorted scene index of every raw point, shape $(N,)$.
            stage_features: Features of every sparse stage (see `encode`).
            stage_indices: Indices of every sparse stage (see `encode`).
            bev: Height-compressed sparse output (see `encode`).

        Returns:
            `(rcnn_cls, rcnn_reg)`: the confidence logit $(R, 1)$ and the box residuals $(R, 7)$ of every RoI.
        """
        keypoint_pos, keypoint_batch, keypoint_features, fused = self.keypoint_encoder(
            x,
            pos,
            batch,
            stage_features,
            stage_indices,
            bev,
        )
        weighted = fused * self.keypoint_head(keypoint_features).sigmoid()
        return self.roi_head(rois, roi_batch, weighted, keypoint_pos, keypoint_batch)

    def forward(
        self,
        voxels: Tensor,
        pos_voxel: Tensor,
        voxel_num_points: Tensor,
        batch_voxel: Tensor,
        x: Tensor,
        pos: Tensor,
        batch: Tensor,
    ) -> PVRCNNOutput:
        features, bev, stage_features, stage_indices = self.encode(
            voxels,
            pos_voxel,
            voxel_num_points,
            batch_voxel,
        )
        out = self.forward_head(features)
        rois, roi_scores, roi_labels, roi_batch = self.proposals(out)
        rcnn_cls, rcnn_reg = self.refine(rois, roi_batch, x, pos, batch, stage_features, stage_indices, bev)

        return {
            "rcnn_cls": rcnn_cls,
            "boxes": decode_roi_residuals(rcnn_reg, rois),
            "roi_labels": roi_labels,
            "roi_scores": roi_scores,
            "batch": roi_batch,
        }

    @torch.no_grad()
    def decode(self, out: PVRCNNOutput) -> Detection3D:
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
    "pv-rcnn.kitti.mmdet3d",
    task="detection",
    input_keys=("voxel", "pos_voxel", "voxel_num_points", "batch_pos_voxel", "x", "pos", "batch"),
    weights=WeightsDict(
        url="hf://torch-pointcloud/pv-rcnn.kitti.mmdet3d/resolve/772e84df456a383eb47366efebda099f7cfde296/model.safetensors",
        dataset="kitti",
        metrics={"mAP": 72.16, "AP/Car": 83.54, "AP/Pedestrian": 59.73, "AP/Cyclist": 73.20},
        classes=("Pedestrian", "Cyclist", "Car"),
        author="mmdet3d",
        license="Apache-2.0",
    ),
    # KITTI 3-class inference: the points are cropped to the strict detection range, and lidar reflectance becomes
    # the point feature `x` of both the voxels and the keypoint encoder.
    transform=T.Compose(
        [
            T.Cat(keys=[DataKeys.INTENSITY], dst_key=DataKeys.X, dim=1),
            T.BoxMask(
                keys=DataKeys.POS,
                bbox=(0.0, -40.0, -3.0, 70.4, 40.0, 1.0),
                dst_keys="range_mask",
                strict=True,
            ),
            T.ApplyMask(keys=[DataKeys.POS, DataKeys.X, DataKeys.INTENSITY, "range_mask"], mask_key="range_mask"),
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
        voxel_size=(0.05, 0.05, 0.1),
        point_cloud_range=(0.0, -40.0, -3.0, 70.4, 40.0, 1.0),
        # Pedestrian, Cyclist, Car.
        anchor_sizes=[[0.8, 0.6, 1.73], [1.76, 0.6, 1.73], [3.9, 1.6, 1.56]],
        anchor_bottom_heights=[-0.6, -0.6, -1.78],
        norm_kwargs={"eps": 1e-3, "momentum": 0.01},
    ),
)
def pv_rcnn_mmdet3d_kitti(**hparams: Any) -> PVRCNNDetection:
    return PVRCNNDetection(**hparams)
