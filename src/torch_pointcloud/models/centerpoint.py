"""CenterPoint detection models.

{{ paper("2006.11275") }}
"""

from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import torch
from torch import Tensor

import torch_pointcloud.transforms as T
from torch_pointcloud.datasets.nuscenes import NUSCENES_DETECTION_CLASSES
from torch_pointcloud.layers.bev_backbone import BEVBackbone
from torch_pointcloud.layers.center_head import CenterHead, CenterHeadOutput
from torch_pointcloud.utils.data import DataKeys
from torch_pointcloud.utils.imports import _SPCONV_GITHUB_URL, optional_import
from torch_pointcloud.utils.types import Detection3D

from ._base import DetectionModel
from ._registry import WeightsDict, register_model
from .pointpillars import PillarFeatureNet, scatter_to_bev
from .second import VoxelResBackbone8x

if TYPE_CHECKING:
    import spconv.pytorch as spconv

spconv, _ = optional_import("spconv.pytorch", url=_SPCONV_GITHUB_URL)


class CenterPointDetection(DetectionModel):
    r"""CenterPoint 3D object detector on a sparse voxel backbone (packed point format).

    Reference: :arxiv: [Center-based 3D Object Detection and Tracking](https://arxiv.org/abs/2006.11275) (Yin et al., 2021).
    Reference implementation: :github: [tianweiy/CenterPoint](https://github.com/tianweiy/CenterPoint).

    A residual sparse 3D backbone ([`VoxelResBackbone8x`][torch_pointcloud.models.second.VoxelResBackbone8x])
    encodes the mean-pooled voxels, its output is flattened along height into a dense BEV map refined by the
    SSD-style [`BEVBackbone`][torch_pointcloud.layers.bev_backbone.BEVBackbone], and a
    [`CenterHead`][torch_pointcloud.layers.center_head.CenterHead] predicts one heatmap and box regression per
    class group. Input points carry their raw features ($x, y, z, \text{intensity}, \Delta t$ on nuScenes).

    Args:
        in_channels: Raw point feature channels including xyz (5 for nuScenes).
        num_classes: Number of foreground classes (10 for nuScenes).
        voxel_size: Voxel size $(v_x, v_y, v_z)$.
        point_cloud_range: Range $(x_\min, y_\min, z_\min, x_\max, y_\max, z_\max)$.
        head_class_groups: Class-index groups, one separate head each (e.g. `[[0], [1, 2], ...]`).
        feature_map_stride: BEV feature-map stride of the head.
        with_velocity: Predict the BEV velocity of every box (the nuScenes box code).
        layer_nums: 2D backbone conv counts per level.
        layer_strides: 2D backbone downsample strides per level.
        num_filters: 2D backbone channel widths per level.
        upsample_strides: 2D backbone upsample factors per level.
        num_upsample_filters: 2D backbone upsample channels per level.
        shared_conv_channels: Channels of the head's shared conv.
        num_head_layers: Number of convs per head branch.
        act: Activation type or callable for the backbones and head.
        act_kwargs: Extra activation arguments.
        norm: Normalization type or callable for the backbones and head.
        norm_kwargs: Extra normalization arguments of the 3D and 2D backbones.
    """

    def __init__(
        self,
        in_channels: int = 5,
        num_classes: int = 10,
        *,
        voxel_size: Sequence[float] = (0.1, 0.1, 0.2),
        point_cloud_range: Sequence[float] = (-51.2, -51.2, -5.0, 51.2, 51.2, 3.0),
        head_class_groups: Sequence[Sequence[int]],
        feature_map_stride: int = 8,
        with_velocity: bool = False,
        layer_nums: Sequence[int] = (5, 5),
        layer_strides: Sequence[int] = (1, 2),
        num_filters: Sequence[int] = (128, 256),
        upsample_strides: Sequence[float] = (1, 2),
        num_upsample_filters: Sequence[int] = (256, 256),
        shared_conv_channels: int = 64,
        num_head_layers: int = 2,
        act: Union[str, Callable, None] = "relu",
        act_kwargs: Optional[Dict[str, Any]] = None,
        norm: Union[str, Callable, None] = "batch_norm",
        norm_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(in_channels=in_channels, num_classes=num_classes)
        self.voxel_size = tuple(voxel_size)
        self.point_cloud_range = tuple(point_cloud_range)
        self.head_class_groups = head_class_groups
        self.feature_map_stride = feature_map_stride
        self.with_velocity = with_velocity
        self.layer_nums = layer_nums
        self.layer_strides = layer_strides
        self.num_filters = num_filters
        self.upsample_strides = upsample_strides
        self.num_upsample_filters = num_upsample_filters
        self.shared_conv_channels = shared_conv_channels
        self.num_head_layers = num_head_layers
        self.act = act
        self.act_kwargs = act_kwargs
        self.norm = norm
        self.norm_kwargs = norm_kwargs

        grid = [int(round((point_cloud_range[i + 3] - point_cloud_range[i]) / voxel_size[i])) for i in range(3)]
        self.grid_size: Tuple[int, int, int] = (grid[0], grid[1], grid[2])
        # spconv spatial shape is (z, y, x) with an extra +1 on z.
        self.sparse_shape: List[int] = [grid[2] + 1, grid[1], grid[0]]

        self.backbone_3d = self.configure_backbone_3d()
        self.backbone = self.configure_backbone()
        self.head = self.configure_head()

    def configure_backbone_3d(self) -> VoxelResBackbone8x:
        """Build the residual sparse 3D voxel backbone (its residual convs carry no bias)."""
        return VoxelResBackbone8x(
            self.in_channels,
            bias=False,
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
        """Channel count $C$ of the BEV feature map entering the head."""
        return self.backbone.num_bev_features

    def configure_head(self) -> CenterHead:
        """Build the multi-group center head."""
        # The head uses the framework-default norm hyperparameters, so the backbones' `norm_kwargs` are not
        # threaded into it.
        return CenterHead(
            self.backbone.num_bev_features,
            self.head_class_groups,
            feature_map_stride=self.feature_map_stride,
            voxel_size=self.voxel_size,
            point_cloud_range=self.point_cloud_range,
            shared_conv_channels=self.shared_conv_channels,
            num_head_layers=self.num_head_layers,
            with_velocity=self.with_velocity,
            act=self.act,
            act_kwargs=self.act_kwargs,
            norm=self.norm,
        )

    def forward_features(self, voxels: Tensor, pos_voxel: Tensor, voxel_num_points: Tensor, batch: Tensor) -> Tensor:
        voxel_indices = torch.cat([batch.view(-1, 1).to(pos_voxel), pos_voxel], dim=1)
        batch_size = int(batch.max().item()) + 1

        # Average the points (xyz + features) inside each voxel.
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

    def forward_head(self, features: Tensor) -> CenterHeadOutput:
        return self.head(features)

    def forward(self, voxels: Tensor, pos_voxel: Tensor, voxel_num_points: Tensor, batch: Tensor) -> CenterHeadOutput:
        features = self.forward_features(voxels, pos_voxel, voxel_num_points, batch)
        return self.forward_head(features)

    @torch.no_grad()
    def decode(self, out: CenterHeadOutput, *, top_k: int = 500) -> Detection3D:
        r"""Decode a forward output into raw candidate detections (see `CenterHead.decode`)."""
        return self.head.decode(out, top_k=top_k)


class CenterPointPillarsDetection(DetectionModel):
    r"""CenterPoint 3D object detector on a pillar trunk (packed point format).

    Reference: :arxiv: [Center-based 3D Object Detection and Tracking](https://arxiv.org/abs/2006.11275) (Yin et al., 2021).
    Reference implementation: :github: [tianweiy/CenterPoint](https://github.com/tianweiy/CenterPoint).

    The pillar feature net of [`PointPillarsDetection`][torch_pointcloud.models.pointpillars.PointPillarsDetection]
    scatters its features to a BEV pseudo-image, refined by the SSD-style
    [`BEVBackbone`][torch_pointcloud.layers.bev_backbone.BEVBackbone] and decoded by a
    [`CenterHead`][torch_pointcloud.layers.center_head.CenterHead].

    Args:
        in_channels: Raw point feature channels including xyz (5 for nuScenes).
        num_classes: Number of foreground classes (10 for nuScenes).
        voxel_size: Pillar size $(v_x, v_y, v_z)$.
        point_cloud_range: Range $(x_\min, y_\min, z_\min, x_\max, y_\max, z_\max)$.
        head_class_groups: Class-index groups, one separate head each (e.g. `[[0], [1, 2], ...]`).
        feature_map_stride: BEV feature-map stride of the head.
        with_velocity: Predict the BEV velocity of every box (the nuScenes box code).
        feat_channels: Output channels of the pillar feature net.
        layer_nums: 2D backbone conv counts per level.
        layer_strides: 2D backbone downsample strides per level.
        num_filters: 2D backbone channel widths per level.
        upsample_strides: 2D backbone upsample factors per level.
        num_upsample_filters: 2D backbone upsample channels per level.
        shared_conv_channels: Channels of the head's shared conv.
        num_head_layers: Number of convs per head branch.
        act: Activation type or callable for the pillar feature net, backbone and head.
        act_kwargs: Extra activation arguments.
        norm: Normalization type or callable for the pillar feature net, backbone and head.
        norm_kwargs: Extra normalization arguments of the pillar feature net and backbone.
    """

    def __init__(
        self,
        in_channels: int = 5,
        num_classes: int = 10,
        *,
        voxel_size: Sequence[float] = (0.2, 0.2, 8.0),
        point_cloud_range: Sequence[float] = (-51.2, -51.2, -5.0, 51.2, 51.2, 3.0),
        head_class_groups: Sequence[Sequence[int]],
        feature_map_stride: int = 4,
        with_velocity: bool = False,
        feat_channels: Sequence[int] = (64,),
        layer_nums: Sequence[int] = (3, 5, 5),
        layer_strides: Sequence[int] = (2, 2, 2),
        num_filters: Sequence[int] = (64, 128, 256),
        upsample_strides: Sequence[float] = (0.5, 1, 2),
        num_upsample_filters: Sequence[int] = (128, 128, 128),
        shared_conv_channels: int = 64,
        num_head_layers: int = 2,
        act: Union[str, Callable, None] = "relu",
        act_kwargs: Optional[Dict[str, Any]] = None,
        norm: Union[str, Callable, None] = "batch_norm",
        norm_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(in_channels=in_channels, num_classes=num_classes)
        self.voxel_size = tuple(voxel_size)
        self.point_cloud_range = tuple(point_cloud_range)
        self.head_class_groups = head_class_groups
        self.feature_map_stride = feature_map_stride
        self.with_velocity = with_velocity
        self.feat_channels = feat_channels
        self.layer_nums = layer_nums
        self.layer_strides = layer_strides
        self.num_filters = num_filters
        self.upsample_strides = upsample_strides
        self.num_upsample_filters = num_upsample_filters
        self.shared_conv_channels = shared_conv_channels
        self.num_head_layers = num_head_layers
        self.act = act
        self.act_kwargs = act_kwargs
        self.norm = norm
        self.norm_kwargs = norm_kwargs

        grid = [int(round((point_cloud_range[i + 3] - point_cloud_range[i]) / voxel_size[i])) for i in range(3)]
        self.grid_size: Tuple[int, int, int] = (grid[0], grid[1], grid[2])

        self.vfe = self.configure_vfe()
        self.backbone = self.configure_backbone()
        self.head = self.configure_head()

    def configure_vfe(self) -> PillarFeatureNet:
        """Build the pillar feature net."""
        return PillarFeatureNet(
            self.in_channels,
            self.feat_channels,
            self.voxel_size,
            self.point_cloud_range,
            act=self.act,
            act_kwargs=self.act_kwargs,
            norm=self.norm,
            norm_kwargs=self.norm_kwargs,
        )

    def configure_backbone(self) -> BEVBackbone:
        """Build the 2D BEV backbone."""
        return BEVBackbone(
            self.feat_channels[-1],
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

    def configure_head(self) -> CenterHead:
        """Build the multi-group center head."""
        # The head uses the framework-default norm hyperparameters, so the trunk's `norm_kwargs` are not
        # threaded into it.
        return CenterHead(
            self.backbone.num_bev_features,
            self.head_class_groups,
            feature_map_stride=self.feature_map_stride,
            voxel_size=self.voxel_size,
            point_cloud_range=self.point_cloud_range,
            shared_conv_channels=self.shared_conv_channels,
            num_head_layers=self.num_head_layers,
            with_velocity=self.with_velocity,
            act=self.act,
            act_kwargs=self.act_kwargs,
            norm=self.norm,
        )

    def forward_features(self, voxels: Tensor, pos_voxel: Tensor, voxel_num_points: Tensor, batch: Tensor) -> Tensor:
        voxel_indices = torch.cat([batch.view(-1, 1).to(pos_voxel), pos_voxel], dim=1)
        batch_size = int(batch.max().item()) + 1
        pillar_features = self.vfe(voxels, voxel_num_points, voxel_indices)
        bev = scatter_to_bev(pillar_features, voxel_indices, batch_size, self.grid_size)
        return self.backbone(bev)

    def forward_head(self, features: Tensor) -> CenterHeadOutput:
        return self.head(features)

    def forward(self, voxels: Tensor, pos_voxel: Tensor, voxel_num_points: Tensor, batch: Tensor) -> CenterHeadOutput:
        features = self.forward_features(voxels, pos_voxel, voxel_num_points, batch)
        return self.forward_head(features)

    @torch.no_grad()
    def decode(self, out: CenterHeadOutput, *, top_k: int = 500) -> Detection3D:
        r"""Decode a forward output into raw candidate detections (see `CenterHead.decode`)."""
        return self.head.decode(out, top_k=top_k)


@register_model(
    "centerpoint-voxel01.nuscenes.mmdet3d",
    task="detection",
    input_keys=("voxel", "pos_voxel", "voxel_num_points", "batch_pos_voxel"),
    weights=WeightsDict(
        url="hf://torch-pointcloud/centerpoint-voxel01.nuscenes.mmdet3d/resolve/0000000000000000000000000000000000000000/model.safetensors",
        dataset="nuscenes",
        metrics={"mAP": 56.71, "NDS": 64.60},
        classes=NUSCENES_DETECTION_CLASSES,
        author="mmdet3d",
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
                max_num_voxels=120000,
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
        head_class_groups=[[0], [1, 2], [3, 4], [5], [6, 7], [8, 9]],
        feature_map_stride=8,
        with_velocity=True,
        norm_kwargs={"eps": 1e-3, "momentum": 0.01},
    ),
)
def centerpoint_voxel01_mmdet3d_nuscenes(**hparams: Any) -> CenterPointDetection:
    return CenterPointDetection(**hparams)


@register_model(
    "centerpoint-voxel0075.nuscenes.mmdet3d",
    task="detection",
    input_keys=("voxel", "pos_voxel", "voxel_num_points", "batch_pos_voxel"),
    weights=WeightsDict(
        url="hf://torch-pointcloud/centerpoint-voxel0075.nuscenes.mmdet3d/resolve/0000000000000000000000000000000000000000/model.safetensors",
        dataset="nuscenes",
        metrics={"mAP": 57.06, "NDS": 65.04},
        classes=NUSCENES_DETECTION_CLASSES,
        author="mmdet3d",
        license="Apache-2.0",
    ),
    transform=T.Compose(
        [
            T.Cat(keys=[DataKeys.INTENSITY, "timestamp"], dst_key=DataKeys.X, dim=1),
            T.HardVoxelize(
                pos_key=DataKeys.POS,
                feature_key=DataKeys.X,
                voxel_size=(0.075, 0.075, 0.2),
                point_cloud_range=(-54.0, -54.0, -5.0, 54.0, 54.0, 3.0),
                max_num_points=10,
                max_num_voxels=120000,
            ),
        ]
    ),
    hparams=dict(
        in_channels=5,
        num_classes=10,
        voxel_size=(0.075, 0.075, 0.2),
        point_cloud_range=(-54.0, -54.0, -5.0, 54.0, 54.0, 3.0),
        head_class_groups=[[0], [1, 2], [3, 4], [5], [6, 7], [8, 9]],
        feature_map_stride=8,
        with_velocity=True,
        norm_kwargs={"eps": 1e-3, "momentum": 0.01},
    ),
)
def centerpoint_voxel0075_mmdet3d_nuscenes(**hparams: Any) -> CenterPointDetection:
    return CenterPointDetection(**hparams)


@register_model(
    "centerpoint-pillar02.nuscenes.mmdet3d",
    task="detection",
    input_keys=("voxel", "pos_voxel", "voxel_num_points", "batch_pos_voxel"),
    weights=WeightsDict(
        url="hf://torch-pointcloud/centerpoint-pillar02.nuscenes.mmdet3d/resolve/0000000000000000000000000000000000000000/model.safetensors",
        dataset="nuscenes",
        metrics={"mAP": 49.12, "NDS": 59.04},
        classes=NUSCENES_DETECTION_CLASSES,
        author="mmdet3d",
        license="Apache-2.0",
    ),
    transform=T.Compose(
        [
            T.Cat(keys=[DataKeys.INTENSITY, "timestamp"], dst_key=DataKeys.X, dim=1),
            T.HardVoxelize(
                pos_key=DataKeys.POS,
                feature_key=DataKeys.X,
                voxel_size=(0.2, 0.2, 8.0),
                point_cloud_range=(-51.2, -51.2, -5.0, 51.2, 51.2, 3.0),
                max_num_points=20,
                max_num_voxels=40000,
            ),
        ]
    ),
    hparams=dict(
        in_channels=5,
        num_classes=10,
        voxel_size=(0.2, 0.2, 8.0),
        point_cloud_range=(-51.2, -51.2, -5.0, 51.2, 51.2, 3.0),
        head_class_groups=[[0], [1, 2], [3, 4], [5], [6, 7], [8, 9]],
        feature_map_stride=4,
        with_velocity=True,
        norm_kwargs={"eps": 1e-3, "momentum": 0.01},
    ),
)
def centerpoint_pillar02_mmdet3d_nuscenes(**hparams: Any) -> CenterPointPillarsDetection:
    return CenterPointPillarsDetection(**hparams)
