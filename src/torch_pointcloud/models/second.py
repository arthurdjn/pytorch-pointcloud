"""SECOND detection models.

{{ paper("second-2018") }}
"""

import math
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Sequence, Tuple, TypedDict, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn import MLP

import torch_pointcloud.transforms as T
from torch_pointcloud.datasets.nuscenes import NUSCENES_DETECTION_CLASSES
from torch_pointcloud.layers import SparseBasicBlock, SparseConvBlock
from torch_pointcloud.layers.act import create_act
from torch_pointcloud.layers.anchors import (
    AnchorHead,
    AnchorHeadMultiOutput,
    AnchorHeadOutput,
    MultiGroupAnchorHead,
)
from torch_pointcloud.layers.bev_backbone import BEVBackbone
from torch_pointcloud.layers.norms import create_norm
from torch_pointcloud.ops.box3d import nms3d
from torch_pointcloud.utils.data import DataKeys
from torch_pointcloud.utils.imports import _SPCONV_GITHUB_URL, optional_import
from torch_pointcloud.utils.types import Detection3D

from ._base import DetectionModel
from ._registry import WeightsDict, register_model

if TYPE_CHECKING:
    import spconv.pytorch as spconv

spconv, _ = optional_import("spconv.pytorch", url=_SPCONV_GITHUB_URL)


class VoxelBackbone8x(nn.Module):
    r"""Sparse 3D convolutional voxel backbone (`VoxelBackBone8x`), $8\times$ downsampling.

    Four sparse conv stages downsample the voxel grid by $2\times$ in $x$/$y$ (stages 2-4) while a
    final $(3, 1, 1)$ sparse conv squeezes the height to 2, yielding a dense BEV tensor after
    height compression.

    Args:
        in_channels: Input voxel feature channels (e.g. $4$ for mean $x, y, z, \text{intensity}$).
        act: Activation type or callable.
        act_kwargs: Extra activation arguments.
        norm: Normalization type or callable.
        norm_kwargs: Extra normalization arguments.
    """

    def __init__(
        self,
        in_channels: int,
        *,
        act: Union[str, Callable, None] = "relu",
        act_kwargs: Optional[Dict[str, Any]] = None,
        norm: Union[str, Callable, None] = "batch_norm",
        norm_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()
        block_kwargs: Dict[str, Any] = dict(act=act, act_kwargs=act_kwargs, norm=norm, norm_kwargs=norm_kwargs)
        self.conv_input = spconv.SparseSequential(
            spconv.SubMConv3d(in_channels, 16, 3, padding=1, bias=False, indice_key="subm1"),
            create_norm(norm, 16, dim=1, **(norm_kwargs or {})),
            create_act(act, **(act_kwargs or {})),
        )
        self.conv1 = spconv.SparseSequential(
            SparseConvBlock(16, 16, 3, indice_key="subm1", **block_kwargs),
        )
        self.conv2 = spconv.SparseSequential(
            SparseConvBlock(16, 32, 3, stride=2, padding=1, indice_key="spconv2", conv_type="spconv", **block_kwargs),
            SparseConvBlock(32, 32, 3, indice_key="subm2", **block_kwargs),
            SparseConvBlock(32, 32, 3, indice_key="subm2", **block_kwargs),
        )
        self.conv3 = spconv.SparseSequential(
            SparseConvBlock(32, 64, 3, stride=2, padding=1, indice_key="spconv3", conv_type="spconv", **block_kwargs),
            SparseConvBlock(64, 64, 3, indice_key="subm3", **block_kwargs),
            SparseConvBlock(64, 64, 3, indice_key="subm3", **block_kwargs),
        )
        self.conv4 = spconv.SparseSequential(
            SparseConvBlock(
                64,
                64,
                3,
                stride=2,
                padding=(0, 1, 1),
                indice_key="spconv4",
                conv_type="spconv",
                **block_kwargs,
            ),
            SparseConvBlock(64, 64, 3, indice_key="subm4", **block_kwargs),
            SparseConvBlock(64, 64, 3, indice_key="subm4", **block_kwargs),
        )
        self.conv_out = spconv.SparseSequential(
            spconv.SparseConv3d(64, 128, (3, 1, 1), stride=(2, 1, 1), padding=0, bias=False, indice_key="spconv_down2"),
            create_norm(norm, 128, dim=1, **(norm_kwargs or {})),
            create_act(act, **(act_kwargs or {})),
        )
        self.out_channels = 128

    def forward(self, x: "spconv.SparseConvTensor") -> "spconv.SparseConvTensor":
        x = self.conv_input(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        return self.conv_out(x)


class SECONDDetection(DetectionModel):
    r"""SECOND 3D object detector (packed point format).

    Reference: :arxiv: [SECOND: Sparsely Embedded Convolutional Detection](https://www.mdpi.com/1424-8220/18/10/3337) (Yan et al., 2018).
    Reference implementation: :github: [open-mmlab/OpenPCDet](https://github.com/open-mmlab/OpenPCDet).

    Args:
        in_channels: Raw point feature channels including xyz (e.g. $4$ for $x, y, z, \text{intensity}$).
        num_classes: Number of foreground classes.
        voxel_size: Voxel size $(v_x, v_y, v_z)$.
        point_cloud_range: Range $(x_\min, y_\min, z_\min, x_\max, y_\max, z_\max)$.
        anchor_sizes: Per-class box size $(d_x, d_y, d_z)$, one row per class.
        anchor_bottom_heights: Per-class anchor bottom $z$, one per class.
        anchor_rotations: Yaw angles (radians) shared by all classes.
        feature_map_stride: BEV feature-map stride of the head.
        layer_nums: 2D backbone conv counts per level.
        layer_strides: 2D backbone downsample strides per level.
        num_filters: 2D backbone channel widths per level.
        upsample_strides: 2D backbone upsample strides per level.
        num_upsample_filters: 2D backbone upsample channels per level.
        num_dir_bins: Number of direction bins in the head.
        dir_offset: Direction-classifier angle offset.
        dir_limit_offset: Heading wrap offset used during decoding.
        act: Activation type or callable for the 3D/2D backbones.
        act_kwargs: Extra activation arguments.
        norm: Normalization type or callable for the 3D/2D backbones.
        norm_kwargs: Extra normalization arguments.
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
        feature_map_stride: int,
        anchor_rotations: Sequence[float] = (0.0, 1.57),
        layer_nums: Sequence[int] = (5, 5),
        layer_strides: Sequence[int] = (1, 2),
        num_filters: Sequence[int] = (128, 256),
        upsample_strides: Sequence[int] = (1, 2),
        num_upsample_filters: Sequence[int] = (256, 256),
        num_dir_bins: int = 2,
        dir_offset: float = 0.78539,
        dir_limit_offset: float = 0.0,
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
        self.feature_map_stride = feature_map_stride
        self.anchor_rotations = anchor_rotations
        self.layer_nums = layer_nums
        self.layer_strides = layer_strides
        self.num_filters = num_filters
        self.upsample_strides = upsample_strides
        self.num_upsample_filters = num_upsample_filters
        self.num_dir_bins = num_dir_bins
        self.dir_offset = dir_offset
        self.dir_limit_offset = dir_limit_offset
        self.act = act
        self.act_kwargs = act_kwargs
        self.norm = norm
        self.norm_kwargs = norm_kwargs

        grid = [int(round((point_cloud_range[i + 3] - point_cloud_range[i]) / voxel_size[i])) for i in range(3)]
        self.grid_size: Tuple[int, int, int] = (grid[0], grid[1], grid[2])
        # spconv spatial shape is (z, y, x) with an extra +1 on z (matches the reference implementation).
        self.sparse_shape: List[int] = [grid[2] + 1, grid[1], grid[0]]

        self.backbone_3d = self.configure_backbone_3d()
        self.backbone = self.configure_backbone()
        self.head = self.configure_head()

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
        # height compression folds the sparse-z output (D=2) into the channel dim
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
        """Channel count $C$ of the BEV feature map entering the head."""
        return self.backbone.num_bev_features

    def configure_head(self) -> AnchorHead:
        """Build the single-group anchor head."""
        return AnchorHead(
            self.backbone.num_bev_features,
            self.num_classes,
            (self.grid_size[0], self.grid_size[1]),
            self.point_cloud_range,
            anchor_sizes=self.anchor_sizes,
            anchor_bottom_heights=self.anchor_bottom_heights,
            anchor_rotations=self.anchor_rotations,
            feature_map_stride=self.feature_map_stride,
            num_dir_bins=self.num_dir_bins,
            dir_offset=self.dir_offset,
            dir_limit_offset=self.dir_limit_offset,
        )

    def forward_features(self, voxels: Tensor, pos_voxel: Tensor, voxel_num_points: Tensor, batch: Tensor) -> Tensor:
        voxel_indices = torch.cat([batch.view(-1, 1).to(pos_voxel), pos_voxel], dim=1)
        batch_size = int(batch.max().item()) + 1

        # MeanVFE: average the points (xyz + features) inside each voxel.
        normalizer = torch.clamp_min(voxel_num_points.view(-1, 1), min=1.0).type_as(voxels)
        voxel_features = voxels.sum(dim=1) / normalizer

        sparse_tensor = spconv.SparseConvTensor(
            features=voxel_features,
            indices=voxel_indices.int(),
            spatial_shape=self.sparse_shape,
            batch_size=batch_size,
        )
        encoded = self.backbone_3d(sparse_tensor)

        dense = encoded.dense()
        b, c, d, h, w = dense.shape
        bev = dense.view(b, c * d, h, w)
        return self.backbone(bev)

    def forward_head(self, features: Tensor) -> AnchorHeadOutput:
        return self.head(features)

    def forward(self, voxels: Tensor, pos_voxel: Tensor, voxel_num_points: Tensor, batch: Tensor) -> AnchorHeadOutput:
        features = self.forward_features(voxels, pos_voxel, voxel_num_points, batch)
        return self.forward_head(features)

    @torch.no_grad()
    def decode(self, out: AnchorHeadOutput) -> Detection3D:
        r"""Decode a forward output into raw per-anchor detections (see `AnchorHead.decode`)."""
        return self.head.decode(out)


class SECONDIoUOutput(TypedDict):
    r"""Predictions of [`SECONDIoUDetection`][torch_pointcloud.models.second.SECONDIoUDetection].

    Attributes:
        rcnn_cls: Second-stage confidence logit of every RoI (its predicted IoU with the object), shape $(R, 1)$.
        boxes: The RoIs, shape $(R, 7)$.
        roi_labels: First-stage class of every RoI, shape $(R,)$.
        roi_scores: First-stage score of every RoI, shape $(R,)$.
        batch: Scene index of every RoI, shape $(R,)$.
    """

    rcnn_cls: Tensor
    boxes: Tensor
    roi_labels: Tensor
    roi_scores: Tensor
    batch: Tensor


class SECONDIoUHead(nn.Module):
    r"""RoI head that scores every proposal by its predicted IoU with the object it covers.

    Every RoI samples a $G \times G$ grid of the bird's-eye-view feature map (see `pool`); the flattened grid goes
    through shared fully connected layers and a confidence branch ending in one logit. The boxes are not refined.

    Args:
        in_channels: Channels of the bird's-eye-view feature map.
        grid_size: Grid points per RoI side $G$.
        origin: Map position $(x_0, y_0)$ of the first feature-map pixel.
        cell_size: Map extent $(s_x, s_y)$ of one feature-map pixel.
        shared_channels: Channels of the shared fully connected layers.
        cls_channels: Hidden channels of the confidence branch.
        dropout: Dropout after the first layer of every branch.
        act: Activation type or callable.
        act_kwargs: Extra activation arguments.
        norm: Normalization type or callable.
        norm_kwargs: Extra normalization arguments.

    Shape:
        - rois: $(R, 7)$ boxes $(c_x, c_y, c_z, d_x, d_y, d_z, \theta)$.
        - roi_batch: $(R,)$ map index of every RoI.
        - features: $(B, C, H, W)$.
        - Output: $(R, 1)$ confidence logits.

    Example:
        ```python
        import torch
        from torch_pointcloud.models.second import SECONDIoUHead

        head = SECONDIoUHead(8, 7, (0.0, -40.0), (0.4, 0.4), shared_channels=(32,), cls_channels=(16,)).eval()
        rois = torch.tensor([[10.0, 0.0, -1.0, 3.9, 1.6, 1.56, 0.3]])
        logits = head(rois, torch.zeros(1, dtype=torch.long), torch.randn(1, 8, 200, 176))
        print(logits.shape)  # (1, 1)
        ```
    """

    def __init__(
        self,
        in_channels: int,
        grid_size: int,
        origin: Sequence[float],
        cell_size: Sequence[float],
        *,
        shared_channels: Sequence[int],
        cls_channels: Sequence[int],
        dropout: float = 0.0,
        act: Union[str, Callable, None] = "relu",
        act_kwargs: Optional[Dict[str, Any]] = None,
        norm: Union[str, Callable, None] = "batch_norm",
        norm_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()
        self.grid_size = grid_size
        self.origin = tuple(origin)
        self.cell_size = tuple(cell_size)
        layer_kwargs: Dict[str, Any] = dict(act=act, act_kwargs=act_kwargs, norm=norm, norm_kwargs=norm_kwargs)
        self.shared = MLP(
            [in_channels * grid_size**2, *shared_channels],
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
        self.reset_parameters()

    def reset_parameters(self) -> None:
        r"""Xavier-normal weights and zero biases on every linear layer."""
        for lin in [*self.shared.lins, *self.cls.lins]:
            nn.init.xavier_normal_(lin.weight)
            if lin.bias is not None:
                nn.init.zeros_(lin.bias)

    def pool(self, rois: Tensor, roi_batch: Tensor, features: Tensor) -> Tensor:
        r"""Bilinearly sample the $G \times G$ grid of every RoI from its feature map, $(R, C, G, G)$ indexed $(v, u)$.

        Map pixel $(i, j)$ sits at $(x_0 + j\,s_x,\; y_0 + i\,s_y)$. The unit grid $(u, v) \in [-1, 1]^2$ is rotated
        by the RoI heading first and scaled by the half extents along the map axes second, so a grid point lands at
        $c_x + \tfrac{d_x}{2}(u \cos\theta - v \sin\theta)$ and $c_y + \tfrac{d_y}{2}(u \sin\theta + v \cos\theta)$.
        This is the box footprint only for a square box or $\theta = 0$: a rotated non-square RoI gets a sheared
        grid, which is what the released weights were trained on. Samples outside the map are zero.
        """
        height, width = features.shape[2:]
        x1 = (rois[:, 0] - rois[:, 3] / 2 - self.origin[0]) / self.cell_size[0]
        x2 = (rois[:, 0] + rois[:, 3] / 2 - self.origin[0]) / self.cell_size[0]
        y1 = (rois[:, 1] - rois[:, 4] / 2 - self.origin[1]) / self.cell_size[1]
        y2 = (rois[:, 1] + rois[:, 4] / 2 - self.origin[1]) / self.cell_size[1]
        cos, sin = torch.cos(rois[:, 6]), torch.sin(rois[:, 6])
        theta = torch.stack(
            [
                (x2 - x1) / (width - 1) * cos,
                (x2 - x1) / (width - 1) * -sin,
                (x1 + x2 - width + 1) / (width - 1),
                (y2 - y1) / (height - 1) * sin,
                (y2 - y1) / (height - 1) * cos,
                (y1 + y2 - height + 1) / (height - 1),
            ],
            dim=1,
        ).view(-1, 2, 3)

        out = features.new_zeros(rois.shape[0], features.shape[1], self.grid_size, self.grid_size)
        for scene in roi_batch.unique().tolist():
            index = (roi_batch == scene).nonzero(as_tuple=False).squeeze(-1)
            size = (index.numel(), features.shape[1], self.grid_size, self.grid_size)
            grid = F.affine_grid(theta[index], list(size), align_corners=True)
            # The map is broadcast to every RoI of the scene, a stride-0 view rather than a copy.
            scene_features = features[scene].unsqueeze(0).expand(index.numel(), -1, -1, -1)
            out[index] = F.grid_sample(scene_features, grid, align_corners=True)
        return out

    def forward(self, rois: Tensor, roi_batch: Tensor, features: Tensor) -> Tensor:
        r"""Predict the confidence logit $(R, 1)$ of every RoI from the $(B, C, H, W)$ feature map."""
        pooled = self.pool(rois, roi_batch, features)
        shared = self.shared(pooled.flatten(1))
        return self.cls(shared)


class SECONDIoUDetection(DetectionModel):
    r"""SECOND with an IoU-scoring second stage (SECOND-IoU, packed point format).

    The [`SECONDDetection`][torch_pointcloud.models.second.SECONDDetection] trunk proposes RoIs from its anchor head
    (top anchors, then a class-agnostic rotated NMS); a [`SECONDIoUHead`][torch_pointcloud.models.second.SECONDIoUHead]
    then scores every RoI by its predicted IoU, which replaces the classification score. The RoIs keep their
    first-stage boxes and classes.

    Reference implementation: :github: [open-mmlab/OpenPCDet](https://github.com/open-mmlab/OpenPCDet).

    Args:
        in_channels: Raw point feature channels including xyz (e.g. $4$ for $x, y, z, \text{intensity}$).
        num_classes: Number of foreground classes.
        voxel_size: Voxel size $(v_x, v_y, v_z)$.
        point_cloud_range: Range $(x_\min, y_\min, z_\min, x_\max, y_\max, z_\max)$.
        anchor_sizes: Per-class box size $(d_x, d_y, d_z)$, one row per class.
        anchor_bottom_heights: Per-class anchor bottom $z$, one per class.
        feature_map_stride: BEV feature-map stride of the head.
        proposal_pre_maxsize: Top-scoring anchors kept per scene before the proposal NMS.
        proposal_post_maxsize: RoIs kept per scene after the proposal NMS.
        proposal_nms_thresh: Rotated BEV IoU threshold of the proposal NMS.
        dir_offset: Direction-classifier angle offset.
        anchor_rotations: Yaw angles (radians) shared by all classes.
        layer_nums: 2D backbone conv counts per level.
        layer_strides: 2D backbone downsample strides per level.
        num_filters: 2D backbone channel widths per level.
        upsample_strides: 2D backbone upsample strides per level.
        num_upsample_filters: 2D backbone upsample channels per level.
        num_dir_bins: Number of direction bins in the head.
        dir_limit_offset: Heading wrap offset used during decoding.
        grid_size: Grid points per RoI side of the IoU head.
        shared_channels: Channels of the shared fully connected layers of the IoU head.
        cls_channels: Hidden channels of the confidence branch of the IoU head.
        dropout: Dropout of the IoU head.
        act: Activation type or callable for the 3D/2D backbones.
        act_kwargs: Extra activation arguments.
        norm: Normalization type or callable for the 3D/2D backbones.
        norm_kwargs: Extra normalization arguments of the 3D/2D backbones.
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
        feature_map_stride: int,
        proposal_pre_maxsize: int,
        proposal_post_maxsize: int,
        proposal_nms_thresh: float,
        dir_offset: float,
        anchor_rotations: Sequence[float] = (0.0, 1.57),
        layer_nums: Sequence[int] = (5, 5),
        layer_strides: Sequence[int] = (1, 2),
        num_filters: Sequence[int] = (128, 256),
        upsample_strides: Sequence[int] = (1, 2),
        num_upsample_filters: Sequence[int] = (256, 256),
        num_dir_bins: int = 2,
        dir_limit_offset: float = 0.0,
        grid_size: int = 7,
        shared_channels: Sequence[int] = (256, 256),
        cls_channels: Sequence[int] = (256, 256),
        dropout: float = 0.0,
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
        self.feature_map_stride = feature_map_stride
        self.proposal_pre_maxsize = proposal_pre_maxsize
        self.proposal_post_maxsize = proposal_post_maxsize
        self.proposal_nms_thresh = proposal_nms_thresh
        self.anchor_rotations = anchor_rotations
        self.layer_nums = layer_nums
        self.layer_strides = layer_strides
        self.num_filters = num_filters
        self.upsample_strides = upsample_strides
        self.num_upsample_filters = num_upsample_filters
        self.num_dir_bins = num_dir_bins
        self.dir_offset = dir_offset
        self.dir_limit_offset = dir_limit_offset
        self.grid_size = grid_size
        self.shared_channels = shared_channels
        self.cls_channels = cls_channels
        self.dropout = dropout
        self.act = act
        self.act_kwargs = act_kwargs
        self.norm = norm
        self.norm_kwargs = norm_kwargs

        grid = [int(round((point_cloud_range[i + 3] - point_cloud_range[i]) / voxel_size[i])) for i in range(3)]
        self.grid_size_3d: Tuple[int, int, int] = (grid[0], grid[1], grid[2])
        # spconv spatial shape is (z, y, x) with an extra +1 on z (matches the reference implementation).
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
        # height compression folds the sparse-z output (D=2) into the channel dim
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
        """Channel count $C$ of the BEV feature map entering the heads."""
        return self.backbone.num_bev_features

    def configure_head(self) -> AnchorHead:
        """Build the single-group anchor head that proposes the RoIs."""
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

    def configure_roi_head(self) -> SECONDIoUHead:
        """Build the IoU head over the BEV feature map."""
        return SECONDIoUHead(
            self.backbone.num_bev_features,
            self.grid_size,
            self.point_cloud_range[:2],
            (self.voxel_size[0] * self.feature_map_stride, self.voxel_size[1] * self.feature_map_stride),
            shared_channels=self.shared_channels,
            cls_channels=self.cls_channels,
            dropout=self.dropout,
            act=self.act,
            act_kwargs=self.act_kwargs,
            # The IoU head keeps the default norm settings of the reference, not the trunk's.
            norm=self.norm,
        )

    def forward_features(self, voxels: Tensor, pos_voxel: Tensor, voxel_num_points: Tensor, batch: Tensor) -> Tensor:
        voxel_indices = torch.cat([batch.view(-1, 1).to(pos_voxel), pos_voxel], dim=1)
        batch_size = int(batch.max().item()) + 1

        # MeanVFE: average the points (xyz + features) inside each voxel.
        normalizer = torch.clamp_min(voxel_num_points.view(-1, 1), min=1.0).type_as(voxels)
        voxel_features = voxels.sum(dim=1) / normalizer

        sparse_tensor = spconv.SparseConvTensor(
            features=voxel_features,
            indices=voxel_indices.int(),
            spatial_shape=self.sparse_shape,
            batch_size=batch_size,
        )
        encoded = self.backbone_3d(sparse_tensor)

        dense = encoded.dense()
        b, c, d, h, w = dense.shape
        bev = dense.view(b, c * d, h, w)
        return self.backbone(bev)

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
        keep = nms3d(
            det["boxes"],
            det["scores"],
            self.proposal_nms_thresh,
            batch=det["batch"],
            rotated=True,
            max_keep=self.proposal_post_maxsize,
            max_candidates=self.proposal_pre_maxsize,
        )
        return det["boxes"][keep], det["scores"][keep], det["labels"][keep], det["batch"][keep]

    def forward(
        self,
        voxels: Tensor,
        pos_voxel: Tensor,
        voxel_num_points: Tensor,
        batch: Tensor,
    ) -> SECONDIoUOutput:
        features = self.forward_features(voxels, pos_voxel, voxel_num_points, batch)
        out = self.forward_head(features)
        rois, roi_scores, roi_labels, roi_batch = self.proposals(out)
        rcnn_cls = self.roi_head(rois, roi_batch, features)
        return {
            "rcnn_cls": rcnn_cls,
            "boxes": rois,
            "roi_labels": roi_labels,
            "roi_scores": roi_scores,
            "batch": roi_batch,
        }

    @torch.no_grad()
    def decode(self, out: SECONDIoUOutput) -> Detection3D:
        r"""Decode a forward output into raw per-RoI detections (no score threshold or NMS).

        Every RoI keeps its first-stage box and label and is scored by the sigmoid of its predicted IoU.

        Returns:
            Packed per-RoI detections `{"boxes": (R, 7), "scores": (R,), "labels": (R,), "batch": (R,)}`.
        """
        return {
            "boxes": out["boxes"],
            "scores": out["rcnn_cls"].sigmoid().view(-1),
            "labels": out["roi_labels"],
            "batch": out["batch"],
        }


class VoxelResBackbone8x(nn.Module):
    r"""Residual sparse 3D voxel backbone (`VoxelResBackBone8x`), $8\times$ downsampling.

    Like [`VoxelBackbone8x`][torch_pointcloud.models.second.VoxelBackbone8x] but with
    [`SparseBasicBlock`][torch_pointcloud.layers.spconv_blocks.SparseBasicBlock] residual stages and a
    128-channel stage 4 (used by the nuScenes SECOND multihead detector).

    Args:
        in_channels: Input voxel feature channels (e.g. $5$ for nuScenes).
        bias: Whether the submanifold convolutions of the residual blocks carry a bias.
        act: Activation type or callable.
        act_kwargs: Extra activation arguments.
        norm: Normalization type or callable.
        norm_kwargs: Extra normalization arguments.
    """

    def __init__(
        self,
        in_channels: int,
        *,
        bias: bool = True,
        act: Union[str, Callable, None] = "relu",
        act_kwargs: Optional[Dict[str, Any]] = None,
        norm: Union[str, Callable, None] = "batch_norm",
        norm_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()
        block_kwargs: Dict[str, Any] = dict(act=act, act_kwargs=act_kwargs, norm=norm, norm_kwargs=norm_kwargs)
        self.conv_input = spconv.SparseSequential(
            spconv.SubMConv3d(in_channels, 16, 3, padding=1, bias=False, indice_key="subm1"),
            create_norm(norm, 16, dim=1, **(norm_kwargs or {})),
            create_act(act, **(act_kwargs or {})),
        )
        self.conv1 = nn.ModuleList(
            [
                SparseBasicBlock(16, 16, bias=bias, indice_key="res1", **block_kwargs),
                SparseBasicBlock(16, 16, bias=bias, indice_key="res1", **block_kwargs),
            ]
        )
        self.conv2 = nn.ModuleList(
            [
                SparseConvBlock(
                    16, 32, 3, stride=2, padding=1, indice_key="spconv2", conv_type="spconv", **block_kwargs
                ),
                SparseBasicBlock(32, 32, bias=bias, indice_key="res2", **block_kwargs),
                SparseBasicBlock(32, 32, bias=bias, indice_key="res2", **block_kwargs),
            ]
        )
        self.conv3 = nn.ModuleList(
            [
                SparseConvBlock(
                    32, 64, 3, stride=2, padding=1, indice_key="spconv3", conv_type="spconv", **block_kwargs
                ),
                SparseBasicBlock(64, 64, bias=bias, indice_key="res3", **block_kwargs),
                SparseBasicBlock(64, 64, bias=bias, indice_key="res3", **block_kwargs),
            ]
        )
        self.conv4 = nn.ModuleList(
            [
                SparseConvBlock(
                    64, 128, 3, stride=2, padding=(0, 1, 1), indice_key="spconv4", conv_type="spconv", **block_kwargs
                ),
                SparseBasicBlock(128, 128, bias=bias, indice_key="res4", **block_kwargs),
                SparseBasicBlock(128, 128, bias=bias, indice_key="res4", **block_kwargs),
            ]
        )
        self.conv_out = spconv.SparseSequential(
            spconv.SparseConv3d(
                128, 128, (3, 1, 1), stride=(2, 1, 1), padding=0, bias=False, indice_key="spconv_down2"
            ),
            create_norm(norm, 128, dim=1, **(norm_kwargs or {})),
            create_act(act, **(act_kwargs or {})),
        )
        self.out_channels = 128

    def forward(self, x: "spconv.SparseConvTensor") -> "spconv.SparseConvTensor":
        x = self.conv_input(x)
        for stage in (self.conv1, self.conv2, self.conv3, self.conv4):
            for module in stage:
                x = module(x)
        return self.conv_out(x)


class SECONDMultiHeadDetection(DetectionModel):
    r"""SECOND with a multi-group anchor head (nuScenes 10-class, packed point format).

    Reference implementation: :github: [open-mmlab/OpenPCDet](https://github.com/open-mmlab/OpenPCDet)
    (`cbgs_second_multihead`). A residual sparse 3D backbone
    ([`VoxelResBackbone8x`][torch_pointcloud.models.second.VoxelResBackbone8x]) feeds the shared 2D
    BEV backbone and an [`MultiGroupAnchorHead`][torch_pointcloud.layers.anchors.MultiGroupAnchorHead] head.
    Input points carry 5 features ($x, y, z, \text{intensity}, \Delta t$).

    Args:
        in_channels: Raw point feature channels including xyz (5 for nuScenes).
        num_classes: Number of foreground classes (10 for nuScenes).
        voxel_size: Voxel size $(v_x, v_y, v_z)$.
        point_cloud_range: Range $(x_\min, y_\min, z_\min, x_\max, y_\max, z_\max)$.
        anchor_sizes: Per-class box size $(d_x, d_y, d_z)$, one row per class.
        anchor_bottom_heights: Per-class anchor bottom $z$, one per class.
        head_class_groups: Class-index groups, one per RPN head (e.g. `[[0], [1, 2], ...]`).
        anchor_rotations: Yaw angles (radians) shared by all classes.
        feature_map_stride: BEV feature-map stride of the head.
        layer_nums: 2D backbone conv counts per level.
        layer_strides: 2D backbone downsample strides per level.
        num_filters: 2D backbone channel widths per level.
        upsample_strides: 2D backbone upsample factors per level.
        num_upsample_filters: 2D backbone upsample channels per level.
        shared_conv_num_filter: Channels of the head's shared conv.
        act: Activation type or callable for the 3D/2D backbones and head.
        act_kwargs: Extra activation arguments.
        norm: Normalization type or callable for the 3D/2D backbones and head.
        norm_kwargs: Extra normalization arguments.
    """

    def __init__(
        self,
        in_channels: int = 5,
        num_classes: int = 10,
        *,
        voxel_size: Sequence[float] = (0.1, 0.1, 0.2),
        point_cloud_range: Sequence[float] = (-51.2, -51.2, -5.0, 51.2, 51.2, 3.0),
        anchor_sizes: Sequence[Sequence[float]],
        anchor_bottom_heights: Sequence[float],
        head_class_groups: Sequence[Sequence[int]],
        feature_map_stride: int,
        anchor_rotations: Sequence[float] = (0.0, 1.57),
        layer_nums: Sequence[int] = (5, 5),
        layer_strides: Sequence[int] = (1, 2),
        num_filters: Sequence[int] = (128, 256),
        upsample_strides: Sequence[float] = (1, 2),
        num_upsample_filters: Sequence[int] = (256, 256),
        shared_conv_num_filter: int = 64,
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
        self.head_class_groups = head_class_groups
        self.feature_map_stride = feature_map_stride
        self.anchor_rotations = anchor_rotations
        self.layer_nums = layer_nums
        self.layer_strides = layer_strides
        self.num_filters = num_filters
        self.upsample_strides = upsample_strides
        self.num_upsample_filters = num_upsample_filters
        self.shared_conv_num_filter = shared_conv_num_filter
        self.act = act
        self.act_kwargs = act_kwargs
        self.norm = norm
        self.norm_kwargs = norm_kwargs

        grid = [int(round((point_cloud_range[i + 3] - point_cloud_range[i]) / voxel_size[i])) for i in range(3)]
        self.grid_size: Tuple[int, int, int] = (grid[0], grid[1], grid[2])
        self.sparse_shape: List[int] = [grid[2] + 1, grid[1], grid[0]]

        self.backbone_3d = self.configure_backbone_3d()
        self.backbone = self.configure_backbone()
        self.head = self.configure_head()

    def configure_backbone_3d(self) -> VoxelResBackbone8x:
        """Build the residual sparse 3D voxel backbone."""
        return VoxelResBackbone8x(
            self.in_channels,
            act=self.act,
            act_kwargs=self.act_kwargs,
            norm=self.norm,
            norm_kwargs=self.norm_kwargs,
        )

    def configure_backbone(self) -> BEVBackbone:
        """Build the 2D BEV backbone."""
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
        """Channel count $C$ of the BEV feature map entering the head."""
        return self.backbone.num_bev_features

    def configure_head(self) -> MultiGroupAnchorHead:
        """Build the multi-group anchor head."""
        return MultiGroupAnchorHead(
            self.backbone.num_bev_features,
            self.num_classes,
            (self.grid_size[0], self.grid_size[1]),
            self.point_cloud_range,
            anchor_sizes=self.anchor_sizes,
            anchor_bottom_heights=self.anchor_bottom_heights,
            head_class_groups=self.head_class_groups,
            anchor_rotations=self.anchor_rotations,
            feature_map_stride=self.feature_map_stride,
            shared_conv_num_filter=self.shared_conv_num_filter,
            act=self.act,
            act_kwargs=self.act_kwargs,
            norm=self.norm,
            norm_kwargs=self.norm_kwargs,
        )

    def forward_features(self, voxels: Tensor, pos_voxel: Tensor, voxel_num_points: Tensor, batch: Tensor) -> Tensor:
        voxel_indices = torch.cat([batch.view(-1, 1).to(pos_voxel), pos_voxel], dim=1)
        batch_size = int(batch.max().item()) + 1

        normalizer = torch.clamp_min(voxel_num_points.view(-1, 1), min=1.0).type_as(voxels)
        voxel_features = voxels.sum(dim=1) / normalizer

        sparse_tensor = spconv.SparseConvTensor(
            features=voxel_features, indices=voxel_indices.int(), spatial_shape=self.sparse_shape, batch_size=batch_size
        )
        encoded = self.backbone_3d(sparse_tensor)
        dense = encoded.dense()
        b, c, d, h, w = dense.shape
        bev = dense.view(b, c * d, h, w)
        return self.backbone(bev)

    def forward_head(self, features: Tensor) -> AnchorHeadMultiOutput:
        return self.head(features)

    def forward(
        self,
        voxels: Tensor,
        pos_voxel: Tensor,
        voxel_num_points: Tensor,
        batch: Tensor,
    ) -> AnchorHeadMultiOutput:
        features = self.forward_features(voxels, pos_voxel, voxel_num_points, batch)
        return self.forward_head(features)

    @torch.no_grad()
    def decode(self, out: AnchorHeadMultiOutput) -> Detection3D:
        r"""Decode a forward output into raw per-anchor detections (see `MultiGroupAnchorHead.decode`)."""
        return self.head.decode(out)


@register_model(
    "second.kitti.openpcdet",
    task="detection",
    input_keys=("voxel", "pos_voxel", "voxel_num_points", "batch_pos_voxel"),
    weights=WeightsDict(
        url="hf://torch-pointcloud/second.kitti.openpcdet/resolve/257ec23dfb951b12756786fcdaba9792cb3916a0/model.safetensors",
        dataset="kitti",
        metrics={"mAP": 66.26, "AP/Car": 78.63, "AP/Pedestrian": 52.98, "AP/Cyclist": 67.16},
        classes=("Car", "Pedestrian", "Cyclist"),
        author="openpcdet",
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
        voxel_size=(0.05, 0.05, 0.1),
        point_cloud_range=(0.0, -40.0, -3.0, 70.4, 40.0, 1.0),
        # Car, Pedestrian, Cyclist.
        anchor_sizes=[[3.9, 1.6, 1.56], [0.8, 0.6, 1.73], [1.76, 0.6, 1.73]],
        anchor_bottom_heights=[-1.78, -0.6, -0.6],
        feature_map_stride=8,
        norm_kwargs={"eps": 1e-3, "momentum": 0.01},
    ),
)
def second_openpcdet_kitti(**hparams: Any) -> SECONDDetection:
    return SECONDDetection(**hparams)


@register_model(
    "second-iou.kitti.openpcdet",
    task="detection",
    input_keys=("voxel", "pos_voxel", "voxel_num_points", "batch_pos_voxel"),
    weights=WeightsDict(
        url="hf://torch-pointcloud/second-iou.kitti.openpcdet/resolve/0000000000000000000000000000000000000000/model.safetensors",
        dataset="kitti",
        metrics={"mAP": 68.71, "AP/Car": 79.09, "AP/Pedestrian": 55.75, "AP/Cyclist": 71.31},
        classes=("Car", "Pedestrian", "Cyclist"),
        author="openpcdet",
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
        voxel_size=(0.05, 0.05, 0.1),
        point_cloud_range=(0.0, -40.0, -3.0, 70.4, 40.0, 1.0),
        # Car, Pedestrian, Cyclist.
        anchor_sizes=[[3.9, 1.6, 1.56], [0.8, 0.6, 1.73], [1.76, 0.6, 1.73]],
        anchor_bottom_heights=[-1.78, -0.6, -0.6],
        feature_map_stride=8,
        proposal_pre_maxsize=1024,
        proposal_post_maxsize=100,
        proposal_nms_thresh=0.7,
        dir_offset=0.78539,
        dropout=0.3,
        norm_kwargs={"eps": 1e-3, "momentum": 0.01},
    ),
)
def second_iou_openpcdet_kitti(**hparams: Any) -> SECONDIoUDetection:
    return SECONDIoUDetection(**hparams)


@register_model(
    "second.kitti.mmdet3d",
    task="detection",
    input_keys=("voxel", "pos_voxel", "voxel_num_points", "batch_pos_voxel"),
    weights=WeightsDict(
        url="hf://torch-pointcloud/second.kitti.mmdet3d/resolve/0000000000000000000000000000000000000000/model.safetensors",
        dataset="kitti",
        metrics={"mAP": 65.55, "AP/Pedestrian": 52.94, "AP/Cyclist": 66.32, "AP/Car": 77.39},
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
        voxel_size=(0.05, 0.05, 0.1),
        point_cloud_range=(0.0, -40.0, -3.0, 70.4, 40.0, 1.0),
        # Pedestrian, Cyclist, Car.
        anchor_sizes=[[0.8, 0.6, 1.73], [1.76, 0.6, 1.73], [3.9, 1.6, 1.56]],
        anchor_bottom_heights=[-0.6, -0.6, -1.78],
        feature_map_stride=8,
        dir_offset=-math.pi / 2,
        norm_kwargs={"eps": 1e-3, "momentum": 0.01},
    ),
)
def second_mmdet3d_kitti(**hparams: Any) -> SECONDDetection:
    return SECONDDetection(**hparams)


@register_model(
    "second.kitti-car.mmdet3d",
    task="detection",
    input_keys=("voxel", "pos_voxel", "voxel_num_points", "batch_pos_voxel"),
    weights=WeightsDict(
        url="hf://torch-pointcloud/second.kitti-car.mmdet3d/resolve/0000000000000000000000000000000000000000/model.safetensors",
        dataset="kitti",
        metrics={"mAP": 78.21, "AP/Car": 78.21},
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
        voxel_size=(0.05, 0.05, 0.1),
        point_cloud_range=(0.0, -40.0, -3.0, 70.4, 40.0, 1.0),
        anchor_sizes=[[3.9, 1.6, 1.56]],
        anchor_bottom_heights=[-1.78],
        feature_map_stride=8,
        dir_offset=-math.pi / 2,
        norm_kwargs={"eps": 1e-3, "momentum": 0.01},
    ),
)
def second_mmdet3d_kitti_car(**hparams: Any) -> SECONDDetection:
    return SECONDDetection(**hparams)


@register_model(
    "second-dynamic-voxel.kitti.mmdet3d",
    task="detection",
    input_keys=("voxel", "pos_voxel", "voxel_num_points", "batch_pos_voxel"),
    weights=WeightsDict(
        url="hf://torch-pointcloud/second-dynamic-voxel.kitti.mmdet3d/resolve/0000000000000000000000000000000000000000/model.safetensors",
        dataset="kitti",
        metrics={"mAP": 65.23, "AP/Pedestrian": 56.78, "AP/Cyclist": 62.26, "AP/Car": 76.66},
        classes=("Pedestrian", "Cyclist", "Car"),
        author="mmdet3d",
        license="Apache-2.0",
    ),
    # KITTI 3-class inference: lidar reflectance becomes the point feature `x`, and every voxel averages all its
    # points.
    transform=T.Compose(
        [
            T.Cat(keys=[DataKeys.INTENSITY], dst_key=DataKeys.X, dim=1),
            T.DynamicVoxelize(
                pos_key=DataKeys.POS,
                feature_key=DataKeys.X,
                voxel_size=(0.05, 0.05, 0.1),
                point_cloud_range=(0.0, -40.0, -3.0, 70.4, 40.0, 1.0),
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
        feature_map_stride=8,
        dir_offset=-math.pi / 2,
        norm_kwargs={"eps": 1e-3, "momentum": 0.01},
    ),
)
def second_dynamic_voxel_mmdet3d_kitti(**hparams: Any) -> SECONDDetection:
    return SECONDDetection(**hparams)


@register_model(
    "second-multihead.nuscenes.openpcdet",
    task="detection",
    input_keys=("voxel", "pos_voxel", "voxel_num_points", "batch_pos_voxel"),
    weights=WeightsDict(
        url="hf://torch-pointcloud/second-multihead.nuscenes.openpcdet/resolve/56430268ff66a20c9ad8f8ecd239421d6dfb3742/model.safetensors",
        dataset="nuscenes",
        metrics={"mAP": 50.76, "NDS": 61.85},
        classes=NUSCENES_DETECTION_CLASSES,
        author="openpcdet",
        license="Apache-2.0",
    ),
    # nuScenes inference: lidar reflectance + sweep timestamp are the model's point features.
    transform=T.Compose(
        [
            T.Cat(keys=[DataKeys.INTENSITY, "timestamp"], dst_key=DataKeys.X, dim=1),
            T.HardVoxelize(
                pos_key=DataKeys.POS,
                feature_key=DataKeys.X,
                voxel_size=(0.1, 0.1, 0.2),
                point_cloud_range=(-51.2, -51.2, -5.0, 51.2, 51.2, 3.0),
                max_num_points=10,
                max_num_voxels=60000,
            ),
        ]
    ),
    hparams=dict(
        in_channels=5,
        num_classes=10,
        voxel_size=(0.1, 0.1, 0.2),
        point_cloud_range=(-51.2, -51.2, -5.0, 51.2, 51.2, 3.0),
        # nuScenes 10-class order: car, truck, construction_vehicle, bus, trailer, barrier,
        # motorcycle, bicycle, pedestrian, traffic_cone.
        anchor_sizes=[
            [4.63, 1.97, 1.74],
            [6.93, 2.51, 2.84],
            [6.37, 2.85, 3.19],
            [10.5, 2.94, 3.47],
            [12.29, 2.90, 3.87],
            [0.50, 2.53, 0.98],
            [2.11, 0.77, 1.47],
            [1.70, 0.60, 1.28],
            [0.73, 0.67, 1.77],
            [0.41, 0.41, 1.07],
        ],
        anchor_bottom_heights=[-0.95, -0.6, -0.225, -0.085, 0.115, -1.33, -1.085, -1.18, -0.935, -1.285],
        head_class_groups=[[0], [1, 2], [3, 4], [5], [6, 7], [8, 9]],
        feature_map_stride=8,
        norm_kwargs={"eps": 1e-3, "momentum": 0.01},
    ),
)
def second_openpcdet_multihead_nuscenes(**hparams: Any) -> SECONDMultiHeadDetection:
    return SECONDMultiHeadDetection(**hparams)
