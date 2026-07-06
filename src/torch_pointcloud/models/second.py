from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
from torch import Tensor

import torch_pointcloud.transforms as T
from torch_pointcloud.layers import SparseConvBlock
from torch_pointcloud.layers.act import create_act
from torch_pointcloud.layers.anchors import (
    AnchorHeadMulti,
    AnchorHeadMultiOutput,
    AnchorHeadOutput,
    AnchorHeadSingle,
)
from torch_pointcloud.layers.bev_backbone import BaseBEVBackbone
from torch_pointcloud.layers.norms import create_norm
from torch_pointcloud.utils.data import DataKeys
from torch_pointcloud.utils.imports import optional_import
from torch_pointcloud.utils.types import Detection3D

from ._base import DetectionModel
from ._registry import register_model

if TYPE_CHECKING:
    import spconv.pytorch as spconv

spconv, _ = optional_import("spconv.pytorch")


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

    Reference: :arxiv: [Yan et al., 2018](https://www.mdpi.com/1424-8220/18/10/3337).
    Reference implementation: :github: [open-mmlab/OpenPCDet](https://github.com/open-mmlab/OpenPCDet).

    Args:
        in_channels: Raw point feature channels including xyz (e.g. $4$ for $x, y, z, \text{intensity}$).
        num_classes: Number of foreground classes.
        voxel_size: Voxel size $(v_x, v_y, v_z)$.
        point_cloud_range: Range $(x_\min, y_\min, z_\min, x_\max, y_\max, z_\max)$.
        anchor_sizes: Per-class box size $(dx, dy, dz)$, one row per class.
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

        grid = [int(round((point_cloud_range[i + 3] - point_cloud_range[i]) / voxel_size[i])) for i in range(3)]
        self.grid_size: Tuple[int, int, int] = (grid[0], grid[1], grid[2])
        # spconv spatial shape is (z, y, x) with an extra +1 on z (matches OpenPCDet).
        self.sparse_shape: List[int] = [grid[2] + 1, grid[1], grid[0]]

        block_kwargs: Dict[str, Any] = dict(act=act, act_kwargs=act_kwargs, norm=norm, norm_kwargs=norm_kwargs)
        self.backbone_3d = VoxelBackbone8x(in_channels, **block_kwargs)
        # height compression folds the sparse-z output (D=2) into the channel dim
        bev_input_channels = self.backbone_3d.out_channels * 2
        self.backbone = BaseBEVBackbone(
            bev_input_channels,
            layer_nums,
            layer_strides,
            num_filters,
            upsample_strides,
            num_upsample_filters,
            **block_kwargs,
        )
        self.head = AnchorHeadSingle(
            self.backbone.num_bev_features,
            num_classes,
            (self.grid_size[0], self.grid_size[1]),
            point_cloud_range,
            anchor_sizes=anchor_sizes,
            anchor_bottom_heights=anchor_bottom_heights,
            anchor_rotations=anchor_rotations,
            feature_map_stride=feature_map_stride,
            num_dir_bins=num_dir_bins,
            dir_offset=dir_offset,
            dir_limit_offset=dir_limit_offset,
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

    def forward(self, voxels: Tensor, pos_voxel: Tensor, voxel_num_points: Tensor, batch: Tensor) -> AnchorHeadOutput:
        spatial_features_2d = self.forward_features(voxels, pos_voxel, voxel_num_points, batch)
        return self.head(spatial_features_2d)

    @torch.no_grad()
    def decode(self, out: AnchorHeadOutput) -> Detection3D:
        r"""Decode a forward output into raw per-anchor detections (see `AnchorHeadSingle.decode`)."""
        return self.head.decode(out)


class SparseBasicBlock(nn.Module):
    r"""Submanifold residual block (`SparseBasicBlock`): two $3\times3\times3$ subm convs + skip.

    A plain `nn.Module` (driven directly rather than via `SparseSequential`) so this file imports
    without `spconv`; the sparse convs are built lazily in `__init__`.

    Args:
        channels: Input and output channels.
        indice_key: Shared submanifold indice key (reuses the rulebook within the block).
        act: Activation type or callable.
        act_kwargs: Extra activation arguments.
        norm: Normalization type or callable.
        norm_kwargs: Extra normalization arguments.
    """

    def __init__(
        self,
        channels: int,
        indice_key: str,
        *,
        act: Union[str, Callable, None] = "relu",
        act_kwargs: Optional[Dict[str, Any]] = None,
        norm: Union[str, Callable, None] = "batch_norm",
        norm_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()
        self.conv1 = spconv.SubMConv3d(channels, channels, 3, padding=1, bias=True, indice_key=indice_key)
        self.bn1 = create_norm(norm, channels, dim=1, **(norm_kwargs or {}))
        self.act = create_act(act, **(act_kwargs or {}))
        self.conv2 = spconv.SubMConv3d(channels, channels, 3, padding=1, bias=True, indice_key=indice_key)
        self.bn2 = create_norm(norm, channels, dim=1, **(norm_kwargs or {}))

    def forward(self, x: "spconv.SparseConvTensor") -> "spconv.SparseConvTensor":
        out = self.conv1(x)
        feat = out.features
        if self.bn1 is not None:
            feat = self.bn1(feat)
        if self.act is not None:
            feat = self.act(feat)
        out = out.replace_feature(feat)
        out = self.conv2(out)
        feat = out.features
        if self.bn2 is not None:
            feat = self.bn2(feat)
        feat = feat + x.features
        if self.act is not None:
            feat = self.act(feat)
        return out.replace_feature(feat)


class VoxelResBackbone8x(nn.Module):
    r"""Residual sparse 3D voxel backbone (`VoxelResBackBone8x`), $8\times$ downsampling.

    Like [`VoxelBackbone8x`][torch_pointcloud.models.second.VoxelBackbone8x] but with
    [`SparseBasicBlock`][torch_pointcloud.models.second.SparseBasicBlock] residual stages and a
    128-channel stage 4 (used by the nuScenes SECOND multihead detector).

    Args:
        in_channels: Input voxel feature channels (e.g. $5$ for nuScenes).
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
        self.conv1 = nn.ModuleList(
            [SparseBasicBlock(16, "res1", **block_kwargs), SparseBasicBlock(16, "res1", **block_kwargs)]
        )
        self.conv2 = nn.ModuleList(
            [
                SparseConvBlock(
                    16, 32, 3, stride=2, padding=1, indice_key="spconv2", conv_type="spconv", **block_kwargs
                ),
                SparseBasicBlock(32, "res2", **block_kwargs),
                SparseBasicBlock(32, "res2", **block_kwargs),
            ]
        )
        self.conv3 = nn.ModuleList(
            [
                SparseConvBlock(
                    32, 64, 3, stride=2, padding=1, indice_key="spconv3", conv_type="spconv", **block_kwargs
                ),
                SparseBasicBlock(64, "res3", **block_kwargs),
                SparseBasicBlock(64, "res3", **block_kwargs),
            ]
        )
        self.conv4 = nn.ModuleList(
            [
                SparseConvBlock(
                    64, 128, 3, stride=2, padding=(0, 1, 1), indice_key="spconv4", conv_type="spconv", **block_kwargs
                ),
                SparseBasicBlock(128, "res4", **block_kwargs),
                SparseBasicBlock(128, "res4", **block_kwargs),
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
    BEV backbone and an [`AnchorHeadMulti`][torch_pointcloud.layers.anchors.AnchorHeadMulti] head.
    Input points carry 5 features ($x, y, z, \text{intensity}, \Delta t$).

    Args:
        in_channels: Raw point feature channels including xyz (5 for nuScenes).
        num_classes: Number of foreground classes (10 for nuScenes).
        voxel_size: Voxel size $(v_x, v_y, v_z)$.
        point_cloud_range: Range $(x_\min, y_\min, z_\min, x_\max, y_\max, z_\max)$.
        anchor_sizes: Per-class box size $(dx, dy, dz)$, one row per class.
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

        grid = [int(round((point_cloud_range[i + 3] - point_cloud_range[i]) / voxel_size[i])) for i in range(3)]
        self.grid_size: Tuple[int, int, int] = (grid[0], grid[1], grid[2])
        self.sparse_shape: List[int] = [grid[2] + 1, grid[1], grid[0]]

        block_kwargs: Dict[str, Any] = dict(act=act, act_kwargs=act_kwargs, norm=norm, norm_kwargs=norm_kwargs)
        self.backbone_3d = VoxelResBackbone8x(in_channels, **block_kwargs)
        bev_input_channels = self.backbone_3d.out_channels * 2
        self.backbone = BaseBEVBackbone(
            bev_input_channels,
            layer_nums,
            layer_strides,
            num_filters,
            upsample_strides,
            num_upsample_filters,
            **block_kwargs,
        )
        self.head = AnchorHeadMulti(
            self.backbone.num_bev_features,
            num_classes,
            (self.grid_size[0], self.grid_size[1]),
            point_cloud_range,
            anchor_sizes=anchor_sizes,
            anchor_bottom_heights=anchor_bottom_heights,
            head_class_groups=head_class_groups,
            anchor_rotations=anchor_rotations,
            feature_map_stride=feature_map_stride,
            shared_conv_num_filter=shared_conv_num_filter,
            **block_kwargs,
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

    def forward(
        self, voxels: Tensor, pos_voxel: Tensor, voxel_num_points: Tensor, batch: Tensor
    ) -> AnchorHeadMultiOutput:
        spatial_features_2d = self.forward_features(voxels, pos_voxel, voxel_num_points, batch)
        return self.head(spatial_features_2d)

    @torch.no_grad()
    def decode(self, out: AnchorHeadMultiOutput) -> Detection3D:
        r"""Decode a forward output into raw per-anchor detections (see `AnchorHeadMulti.decode`)."""
        return self.head.decode(out)


@register_model(
    "second-openpcdet.kitti",
    task="detection",
    weights="hf://torch-pointcloud/second/second-openpcdet.kitti.pt",
    # KITTI 3-class inference: lidar reflectance becomes the model's point feature `x`.
    transforms=T.Compose(
        [
            T.Cat(keys=[DataKeys.INTENSITY], dst_key=DataKeys.X, dim=1),
            T.HardVoxelize(
                pos_key=DataKeys.POS,
                feat_key=DataKeys.X,
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
    "second-openpcdet-multihead.nuscenes",
    task="detection",
    weights="hf://torch-pointcloud/second/second-openpcdet-multihead.nuscenes.pt",
    # nuScenes inference: lidar reflectance + sweep timestamp are the model's point features.
    transforms=T.Compose(
        [
            T.Cat(keys=[DataKeys.INTENSITY, "timestamp"], dst_key=DataKeys.X, dim=1),
            T.HardVoxelize(
                pos_key=DataKeys.POS,
                feat_key=DataKeys.X,
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
