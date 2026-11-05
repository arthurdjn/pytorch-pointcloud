"""TransFusion-L detection model.

{{ paper("2203.11496") }}
"""

from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import torch
from torch import Tensor

import torch_pointcloud.transforms as T
from torch_pointcloud.datasets.nuscenes import NUSCENES_DETECTION_CLASSES
from torch_pointcloud.layers.bev_backbone import BEVBackbone
from torch_pointcloud.layers.transfusion_head import TransFusionHead, TransFusionHeadOutput
from torch_pointcloud.utils.data import DataKeys
from torch_pointcloud.utils.imports import _SPCONV_GITHUB_URL, optional_import
from torch_pointcloud.utils.types import Detection3D

from ._base import DetectionModel
from ._registry import WeightsDict, register_model
from .second import VoxelResBackbone8x

if TYPE_CHECKING:
    import spconv.pytorch as spconv

spconv, _ = optional_import("spconv.pytorch", url=_SPCONV_GITHUB_URL)


class TransFusionDetection(DetectionModel):
    r"""TransFusion-L: query-based LiDAR-only 3D object detector on a sparse voxel backbone (packed point format).

    Reference: :arxiv: [TransFusion: Robust LiDAR-Camera Fusion for 3D Object Detection with Transformers](https://arxiv.org/abs/2203.11496) (Bai et al., 2022).
    Reference implementation: :github: [open-mmlab/OpenPCDet](https://github.com/open-mmlab/OpenPCDet).

    A residual sparse 3D backbone ([`VoxelResBackbone8x`][torch_pointcloud.models.second.VoxelResBackbone8x])
    encodes the mean-pooled voxels, its output is flattened along height into a dense BEV map refined by the
    SSD-style [`BEVBackbone`][torch_pointcloud.layers.bev_backbone.BEVBackbone], and a
    [`TransFusionHead`][torch_pointcloud.layers.transfusion_head.TransFusionHead] initializes object queries from
    a dense class heatmap and decodes them by attending to the whole BEV map. Input points carry their raw features
    ($x, y, z, \text{intensity}, \Delta t$ on nuScenes).

    Args:
        in_channels: Raw point feature channels including xyz (5 for nuScenes).
        num_classes: Number of foreground classes (10 for nuScenes).
        voxel_size: Voxel size $(v_x, v_y, v_z)$.
        point_cloud_range: Range $(x_\min, y_\min, z_\min, x_\max, y_\max, z_\max)$.
        feature_map_stride: BEV feature-map stride of the head.
        layer_nums: 2D backbone conv counts per level.
        layer_strides: 2D backbone downsample strides per level.
        num_filters: 2D backbone channel widths per level.
        upsample_strides: 2D backbone upsample factors per level.
        num_upsample_filters: 2D backbone upsample channels per level.
        embed_dim: Query / decoder feature width of the head.
        num_proposals: Number of object queries.
        num_heads: Decoder attention heads.
        mlp_dim: Decoder FFN width.
        dropout: Decoder dropout.
        nms_kernel_size: Heatmap local-max pooling kernel of the query initialization.
        local_max_classes: Crowded small-object class indices whose heatmap peaks skip the local-max suppression;
            see [`TransFusionHead`][torch_pointcloud.layers.transfusion_head.TransFusionHead].
        act: Activation type or callable for the backbones.
        act_kwargs: Extra activation arguments.
        norm: Normalization type or callable for the backbones.
        norm_kwargs: Extra normalization arguments of the 3D and 2D backbones.
    """

    def __init__(
        self,
        in_channels: int = 5,
        num_classes: int = 10,
        *,
        voxel_size: Sequence[float] = (0.075, 0.075, 0.2),
        point_cloud_range: Sequence[float] = (-54.0, -54.0, -5.0, 54.0, 54.0, 3.0),
        feature_map_stride: int = 8,
        layer_nums: Sequence[int] = (5, 5),
        layer_strides: Sequence[int] = (1, 2),
        num_filters: Sequence[int] = (128, 256),
        upsample_strides: Sequence[float] = (1, 2),
        num_upsample_filters: Sequence[int] = (256, 256),
        embed_dim: int = 128,
        num_proposals: int = 200,
        num_heads: int = 8,
        mlp_dim: int = 256,
        dropout: float = 0.1,
        nms_kernel_size: int = 3,
        local_max_classes: Sequence[int] = (),
        act: Union[str, Callable, None] = "relu",
        act_kwargs: Optional[Dict[str, Any]] = None,
        norm: Union[str, Callable, None] = "batch_norm",
        norm_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(in_channels=in_channels, num_classes=num_classes)
        self.voxel_size = tuple(voxel_size)
        self.point_cloud_range = tuple(point_cloud_range)
        self.feature_map_stride = feature_map_stride
        self.layer_nums = layer_nums
        self.layer_strides = layer_strides
        self.num_filters = num_filters
        self.upsample_strides = upsample_strides
        self.num_upsample_filters = num_upsample_filters
        self.embed_dim = embed_dim
        self.num_proposals = num_proposals
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim
        self.dropout = dropout
        self.nms_kernel_size = nms_kernel_size
        self.local_max_classes = local_max_classes
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

    def configure_head(self) -> TransFusionHead:
        """Build the query-based TransFusion head, attending to the whole BEV map."""
        return TransFusionHead(
            self.backbone.num_bev_features,
            self.num_classes,
            voxel_size=self.voxel_size,
            point_cloud_range=self.point_cloud_range,
            feature_map_stride=self.feature_map_stride,
            height_code="center",
            embed_dim=self.embed_dim,
            num_proposals=self.num_proposals,
            num_heads=self.num_heads,
            nms_kernel_size=self.nms_kernel_size,
            mlp_dim=self.mlp_dim,
            dropout=self.dropout,
            local_max_classes=self.local_max_classes,
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

    def forward_head(self, features: Tensor) -> TransFusionHeadOutput:
        return self.head(features)

    def forward(
        self, voxels: Tensor, pos_voxel: Tensor, voxel_num_points: Tensor, batch: Tensor
    ) -> TransFusionHeadOutput:
        features = self.forward_features(voxels, pos_voxel, voxel_num_points, batch)
        return self.forward_head(features)

    @torch.no_grad()
    def decode(self, out: TransFusionHeadOutput) -> Detection3D:
        r"""Decode a forward output into raw candidate detections (see `TransFusionHead.decode`)."""
        return self.head.decode(out)


@register_model(
    "transfusion-lidar.nuscenes.openpcdet",
    task="detection",
    input_keys=("voxel", "pos_voxel", "voxel_num_points", "batch_pos_voxel"),
    weights=WeightsDict(
        url="hf://torch-pointcloud/transfusion-lidar.nuscenes.openpcdet/resolve/d7b98f5b95ce86619b35192bacd721fd1def70ea/model.safetensors",
        dataset="nuscenes",
        metrics={"mAP": 65.32, "NDS": 69.71},
        classes=NUSCENES_DETECTION_CLASSES,
        author="openpcdet",
        license="Apache-2.0",
    ),
    transform=T.Compose(
        [
            T.Cat(keys=[DataKeys.INTENSITY, DataKeys.TIMESTAMP], dst_key=DataKeys.X, dim=1),
            T.HardVoxelize(
                pos_key=DataKeys.POS,
                feature_key=DataKeys.X,
                voxel_size=(0.075, 0.075, 0.2),
                point_cloud_range=(-54.0, -54.0, -5.0, 54.0, 54.0, 3.0),
                max_num_points=10,
                max_num_voxels=160000,
            ),
        ]
    ),
    hparams=dict(
        in_channels=5,
        num_classes=10,
        voxel_size=(0.075, 0.075, 0.2),
        point_cloud_range=(-54.0, -54.0, -5.0, 54.0, 54.0, 3.0),
        feature_map_stride=8,
        # nuScenes 10-class order: car, truck, construction_vehicle, bus, trailer, barrier,
        # motorcycle, bicycle, pedestrian, traffic_cone.
        local_max_classes=(8, 9),
        norm_kwargs={"eps": 1e-3, "momentum": 0.01},
    ),
)
def transfusion_lidar_openpcdet_nuscenes(**hparams: Any) -> TransFusionDetection:
    return TransFusionDetection(**hparams)
