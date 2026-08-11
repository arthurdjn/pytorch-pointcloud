from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.nn import MLP

import torch_pointcloud.transforms as T
from torch_pointcloud.datasets.nuscenes import NUSCENES_DETECTION_CLASSES
from torch_pointcloud.layers.anchors import (
    AnchorHeadMulti,
    AnchorHeadMultiOutput,
    AnchorHeadOutput,
    AnchorHeadSingle,
)
from torch_pointcloud.layers.bev_backbone import BaseBEVBackbone
from torch_pointcloud.utils.data import DataKeys
from torch_pointcloud.utils.types import Detection3D

from ._base import DetectionModel
from ._registry import WeightsDict, register_model


class PFNLayer(nn.Module):
    r"""Single pillar feature-net layer: a per-point [`MLP`][torch_geometric.nn.models.MLP] and pillar max-pool.

    Mirrors the reference `PFNLayer`. For non-final layers the pooled feature is concatenated back
    onto every point (so the output width is doubled before the next layer).

    Args:
        in_channels: Input feature channels per point.
        out_channels: Output feature channels (halved internally for non-final layers).
        last_layer: Whether this is the final layer (return the pooled feature directly).
        act: Activation type or callable.
        act_kwargs: Extra activation arguments.
        norm: Normalization type or callable.
        norm_kwargs: Extra normalization arguments.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        last_layer: bool,
        *,
        act: Union[str, Callable, None] = "relu",
        act_kwargs: Optional[Dict[str, Any]] = None,
        norm: Union[str, Callable, None] = "batch_norm",
        norm_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()
        if not last_layer:
            out_channels = out_channels // 2

        self.last_vfe = last_layer
        self.mlp = MLP(
            [in_channels, out_channels],
            act=act,
            act_kwargs=act_kwargs,
            norm=norm,
            norm_kwargs=norm_kwargs,
            bias=False,
            plain_last=False,
        )

    def forward(self, inputs: Tensor) -> Tensor:
        p, n, _ = inputs.shape
        # The per-point MLP (linear + norm + act) is applied over the flattened pillar-point axis;
        # its BatchNorm normalizes each channel over $P \cdot N$, matching the reference permute.
        x = self.mlp(inputs.reshape(p * n, -1)).reshape(p, n, -1)
        x_max = torch.max(x, dim=1, keepdim=True)[0]
        if self.last_vfe:
            return x_max

        x_repeat = x_max.repeat(1, inputs.shape[1], 1)
        return torch.cat([x, x_repeat], dim=2)


class PillarFeatureNet(nn.Module):
    r"""Pillar feature encoder (`PillarVFE`).

    Augments each point in a pillar with its offset to the pillar's point-cluster mean and to the
    pillar center, then applies a stack of [`PFNLayer`][torch_pointcloud.models.pointpillars.PFNLayer]s.

    Args:
        in_channels: Raw point feature channels (e.g. $4$ for $x, y, z, \text{intensity}$).
        feat_channels: Output channels of each pillar feature-net layer.
        voxel_size: Pillar size $(v_x, v_y, v_z)$.
        point_cloud_range: Range $(x_\min, y_\min, z_\min, x_\max, y_\max, z_\max)$.
        act: Activation type or callable.
        act_kwargs: Extra activation arguments.
        norm: Normalization type or callable.
        norm_kwargs: Extra normalization arguments.
    """

    def __init__(
        self,
        in_channels: int,
        feat_channels: Sequence[int],
        voxel_size: Sequence[float],
        point_cloud_range: Sequence[float],
        *,
        act: Union[str, Callable, None] = "relu",
        act_kwargs: Optional[Dict[str, Any]] = None,
        norm: Union[str, Callable, None] = "batch_norm",
        norm_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()
        # +6: per-point cluster-mean offset (xyz) and pillar-center offset (xyz)
        num_point_features = in_channels + 6
        num_filters = [num_point_features, *feat_channels]

        self.pfn_layers = nn.ModuleList()
        for i in range(len(num_filters) - 1):
            layer = PFNLayer(
                num_filters[i],
                num_filters[i + 1],
                last_layer=(i >= len(num_filters) - 2),
                act=act,
                act_kwargs=act_kwargs,
                norm=norm,
                norm_kwargs=norm_kwargs,
            )
            self.pfn_layers.append(layer)

        self.out_channels = feat_channels[-1]
        self.voxel_x, self.voxel_y, self.voxel_z = voxel_size
        self.x_offset = self.voxel_x / 2 + point_cloud_range[0]
        self.y_offset = self.voxel_y / 2 + point_cloud_range[1]
        self.z_offset = self.voxel_z / 2 + point_cloud_range[2]

    def forward(self, voxels: Tensor, num_points: Tensor, voxel_indices: Tensor) -> Tensor:
        points_mean = voxels[:, :, :3].sum(dim=1, keepdim=True) / num_points.type_as(voxels).view(-1, 1, 1)
        f_cluster = voxels[:, :, :3] - points_mean

        f_center = torch.zeros_like(voxels[:, :, :3])
        f_center[:, :, 0] = voxels[:, :, 0] - (
            voxel_indices[:, 3].to(voxels.dtype).unsqueeze(1) * self.voxel_x + self.x_offset
        )
        f_center[:, :, 1] = voxels[:, :, 1] - (
            voxel_indices[:, 2].to(voxels.dtype).unsqueeze(1) * self.voxel_y + self.y_offset
        )
        f_center[:, :, 2] = voxels[:, :, 2] - (
            voxel_indices[:, 1].to(voxels.dtype).unsqueeze(1) * self.voxel_z + self.z_offset
        )

        features = torch.cat([voxels, f_cluster, f_center], dim=-1)

        voxel_count = features.shape[1]
        mask = torch.arange(voxel_count, device=voxels.device).view(1, -1) < num_points.view(-1, 1)
        features = features * mask.unsqueeze(-1).type_as(voxels)

        for pfn in self.pfn_layers:
            features = pfn(features)
        return features.squeeze(1)


def scatter_to_bev(
    pillar_features: Tensor, voxel_indices: Tensor, batch_size: int, grid_size: Tuple[int, int, int]
) -> Tensor:
    r"""Scatter pillar features back to a dense BEV pseudo-image.

    Stateless scatter of per-pillar features onto a dense bird's-eye-view canvas indexed by each
    pillar's $(y, x)$ voxel grid index. The channel count is taken from `pillar_features`.

    Args:
        pillar_features: Per-pillar features of shape $(P, C)$.
        voxel_indices: Per-pillar voxel indices of shape $(P, 4)$ with columns $(\text{batch}, z, y, x)$ and $z = 0$.
        batch_size: Number of samples $B$ in the batch.
        grid_size: Voxel grid size $(n_x, n_y, n_z)$ with $n_z = 1$.

    Returns:
        Dense BEV pseudo-image of shape $(B, C, n_y, n_x)$.

    Shape:
        - Input: $(P, C)$ and $(P, 4)$.
        - Output: $(B, C, n_y, n_x)$.

    Example:
        ```python
        import torch
        from torch_pointcloud.models.pointpillars import scatter_to_bev

        pillar_features = torch.randn(120, 64)
        voxel_indices = torch.zeros(120, 4, dtype=torch.long)
        voxel_indices[:, 2] = torch.randint(0, 496, (120,))
        voxel_indices[:, 3] = torch.randint(0, 432, (120,))
        bev = scatter_to_bev(pillar_features, voxel_indices, batch_size=1, grid_size=(432, 496, 1))
        print(bev.shape)
        ```
    """
    nx, ny, nz = grid_size
    if nz != 1:
        raise ValueError(f"`grid_size` must have a single height bin (nz == 1), got {nz}.")
    num_bev_features = pillar_features.size(-1)
    # voxel_indices columns: (batch, z, y, x), z == 0.
    flat = voxel_indices[:, 0].long() * (ny * nx) + voxel_indices[:, 2].long() * nx + voxel_indices[:, 3].long()
    canvas = pillar_features.new_zeros(batch_size * ny * nx, num_bev_features)
    canvas[flat] = pillar_features
    canvas = canvas.view(batch_size, ny, nx, num_bev_features)
    return canvas.permute(0, 3, 1, 2).contiguous()


class PointPillarsDetection(DetectionModel):
    r"""PointPillars 3D object detector (packed point format).

    Reference: :arxiv: [Lang et al., 2019](https://arxiv.org/abs/1812.05784).
    Reference implementation: :github: [open-mmlab/OpenPCDet](https://github.com/open-mmlab/OpenPCDet).

    Args:
        in_channels: Raw point feature channels including xyz (e.g. $4$ for $x, y, z, \text{intensity}$).
        num_classes: Number of foreground classes.
        voxel_size: Pillar size $(v_x, v_y, v_z)$.
        point_cloud_range: Range $(x_\min, y_\min, z_\min, x_\max, y_\max, z_\max)$.
        anchor_sizes: Per-class box size $(dx, dy, dz)$, one row per class.
        anchor_bottom_heights: Per-class anchor bottom $z$, one per class.
        anchor_rotations: Yaw angles (radians) shared by all classes.
        feature_map_stride: BEV feature-map stride of the head.
        feat_channels: Output channels of the pillar feature net.
        layer_nums: 2D backbone conv counts per level.
        layer_strides: 2D backbone downsample strides per level.
        num_filters: 2D backbone channel widths per level.
        upsample_strides: 2D backbone upsample strides per level.
        num_upsample_filters: 2D backbone upsample channels per level.
        num_dir_bins: Number of direction bins in the head.
        dir_offset: Direction-classifier angle offset.
        dir_limit_offset: Heading wrap offset used during decoding.
        act: Activation type or callable for the pillar feature net and 2D backbone.
        act_kwargs: Extra activation arguments.
        norm: Normalization type or callable for the pillar feature net and 2D backbone.
        norm_kwargs: Extra normalization arguments.
    """

    def __init__(
        self,
        in_channels: int = 4,
        num_classes: int = 3,
        *,
        voxel_size: Sequence[float] = (0.16, 0.16, 4.0),
        point_cloud_range: Sequence[float] = (0.0, -39.68, -3.0, 69.12, 39.68, 1.0),
        anchor_sizes: Sequence[Sequence[float]],
        anchor_bottom_heights: Sequence[float],
        feature_map_stride: int,
        anchor_rotations: Sequence[float] = (0.0, 1.57),
        feat_channels: Sequence[int] = (64,),
        layer_nums: Sequence[int] = (3, 5, 5),
        layer_strides: Sequence[int] = (2, 2, 2),
        num_filters: Sequence[int] = (64, 128, 256),
        upsample_strides: Sequence[int] = (1, 2, 4),
        num_upsample_filters: Sequence[int] = (128, 128, 128),
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

        block_kwargs: Dict[str, Any] = dict(act=act, act_kwargs=act_kwargs, norm=norm, norm_kwargs=norm_kwargs)
        self.vfe = PillarFeatureNet(in_channels, feat_channels, voxel_size, point_cloud_range, **block_kwargs)
        self.backbone = BaseBEVBackbone(
            feat_channels[-1],
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
        pillar_features = self.vfe(voxels, voxel_num_points, voxel_indices)
        bev = scatter_to_bev(pillar_features, voxel_indices, batch_size, self.grid_size)
        return self.backbone(bev)

    def forward(self, voxels: Tensor, pos_voxel: Tensor, voxel_num_points: Tensor, batch: Tensor) -> AnchorHeadOutput:
        spatial_features_2d = self.forward_features(voxels, pos_voxel, voxel_num_points, batch)
        return self.head(spatial_features_2d)

    @torch.no_grad()
    def decode(self, out: AnchorHeadOutput) -> Detection3D:
        r"""Decode a forward output into raw per-anchor detections (see `AnchorHeadSingle.decode`)."""
        return self.head.decode(out)


class PointPillarsMultiHeadDetection(DetectionModel):
    r"""PointPillars with a multi-group anchor head (nuScenes 10-class, packed point format).

    Reference implementation: :github: [open-mmlab/OpenPCDet](https://github.com/open-mmlab/OpenPCDet)
    (`cbgs_pp_multihead`). Same pillar trunk as
    [`PointPillarsDetection`][torch_pointcloud.models.pointpillars.PointPillarsDetection]
    but with an [`AnchorHeadMulti`][torch_pointcloud.layers.anchors.AnchorHeadMulti] head (per-group
    heads, sincos + velocity box code). Input points carry 5 features ($x, y, z, \text{intensity},
    \Delta t$ from 10-sweep aggregation).

    Args:
        in_channels: Raw point feature channels including xyz (5 for nuScenes).
        num_classes: Number of foreground classes (10 for nuScenes).
        voxel_size: Pillar size $(v_x, v_y, v_z)$.
        point_cloud_range: Range $(x_\min, y_\min, z_\min, x_\max, y_\max, z_\max)$.
        anchor_sizes: Per-class box size $(dx, dy, dz)$, one row per class.
        anchor_bottom_heights: Per-class anchor bottom $z$, one per class.
        head_class_groups: Class-index groups, one per RPN head (e.g. `[[0], [1, 2], ...]`).
        anchor_rotations: Yaw angles (radians) shared by all classes.
        feature_map_stride: BEV feature-map stride of the head.
        feat_channels: Pillar feature-net output channels.
        layer_nums: 2D backbone conv counts per level.
        layer_strides: 2D backbone downsample strides per level.
        num_filters: 2D backbone channel widths per level.
        upsample_strides: 2D backbone upsample factors per level (may be < 1).
        num_upsample_filters: 2D backbone upsample channels per level.
        shared_conv_num_filter: Channels of the head's shared conv.
        act: Activation type or callable for the pillar feature net, 2D backbone and head.
        act_kwargs: Extra activation arguments.
        norm: Normalization type or callable for the pillar feature net, 2D backbone and head.
        norm_kwargs: Extra normalization arguments.
    """

    def __init__(
        self,
        in_channels: int = 5,
        num_classes: int = 10,
        *,
        voxel_size: Sequence[float] = (0.2, 0.2, 8.0),
        point_cloud_range: Sequence[float] = (-51.2, -51.2, -5.0, 51.2, 51.2, 3.0),
        anchor_sizes: Sequence[Sequence[float]],
        anchor_bottom_heights: Sequence[float],
        head_class_groups: Sequence[Sequence[int]],
        feature_map_stride: int,
        anchor_rotations: Sequence[float] = (0.0, 1.57),
        feat_channels: Sequence[int] = (64,),
        layer_nums: Sequence[int] = (3, 5, 5),
        layer_strides: Sequence[int] = (2, 2, 2),
        num_filters: Sequence[int] = (64, 128, 256),
        upsample_strides: Sequence[float] = (0.5, 1, 2),
        num_upsample_filters: Sequence[int] = (128, 128, 128),
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

        block_kwargs: Dict[str, Any] = dict(act=act, act_kwargs=act_kwargs, norm=norm, norm_kwargs=norm_kwargs)
        self.vfe = PillarFeatureNet(in_channels, feat_channels, voxel_size, point_cloud_range, **block_kwargs)
        self.backbone = BaseBEVBackbone(
            feat_channels[-1],
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
        pillar_features = self.vfe(voxels, voxel_num_points, voxel_indices)
        bev = scatter_to_bev(pillar_features, voxel_indices, batch_size, self.grid_size)
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
    "pointpillars.kitti.openpcdet",
    task="detection",
    weights=WeightsDict(
        url="hf://torch-pointcloud/pointpillars/pointpillars.kitti.openpcdet.safetensors",
        dataset="kitti",
        metrics={"mAP": 62.86},
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
                feat_key=DataKeys.X,
                voxel_size=(0.16, 0.16, 4.0),
                point_cloud_range=(0.0, -39.68, -3.0, 69.12, 39.68, 1.0),
                max_num_points=32,
                max_num_voxels=40000,
            ),
        ]
    ),
    hparams=dict(
        in_channels=4,
        num_classes=3,
        voxel_size=(0.16, 0.16, 4.0),
        point_cloud_range=(0.0, -39.68, -3.0, 69.12, 39.68, 1.0),
        # Car, Pedestrian, Cyclist.
        anchor_sizes=[[3.9, 1.6, 1.56], [0.8, 0.6, 1.73], [1.76, 0.6, 1.73]],
        anchor_bottom_heights=[-1.78, -0.6, -0.6],
        feature_map_stride=2,
        norm_kwargs={"eps": 1e-3, "momentum": 0.01},
    ),
)
def pointpillars_openpcdet_kitti(**hparams: Any) -> PointPillarsDetection:
    return PointPillarsDetection(**hparams)


@register_model(
    "pointpillars-multihead.nuscenes.openpcdet",
    task="detection",
    weights=WeightsDict(
        url="hf://torch-pointcloud/pointpillars/pointpillars-multihead.nuscenes.openpcdet.safetensors",
        dataset="nuscenes",
        classes=NUSCENES_DETECTION_CLASSES,
        author="openpcdet",
        license="Apache-2.0",
    ),
    transform=T.Compose(
        [
            T.Cat(keys=[DataKeys.INTENSITY, "timestamp"], dst_key=DataKeys.X, dim=1),
            T.HardVoxelize(
                pos_key=DataKeys.POS,
                feat_key=DataKeys.X,
                voxel_size=(0.2, 0.2, 8.0),
                point_cloud_range=(-51.2, -51.2, -5.0, 51.2, 51.2, 3.0),
                max_num_points=20,
                max_num_voxels=30000,
            ),
        ]
    ),
    hparams=dict(
        in_channels=5,
        num_classes=10,
        voxel_size=(0.2, 0.2, 8.0),
        point_cloud_range=(-51.2, -51.2, -5.0, 51.2, 51.2, 3.0),
        # nuScenes 10-class order:
        # car, truck, construction_vehicle, bus, trailer, barrier, motorcycle, bicycle, pedestrian, traffic_cone
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
        feature_map_stride=4,
        norm_kwargs={"eps": 1e-3, "momentum": 0.01},
    ),
)
def pointpillars_openpcdet_multihead_nuscenes(**hparams: Any) -> PointPillarsMultiHeadDetection:
    return PointPillarsMultiHeadDetection(**hparams)
