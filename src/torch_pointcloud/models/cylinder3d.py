r"""Cylinder3D LiDAR semantic segmentation model.

{{ paper("2011.10033") }}
"""

from typing import TYPE_CHECKING, Any, Callable, Dict, List, Literal, Optional, Sequence, Tuple, Union, overload

import torch
import torch.nn as nn
from torch import Tensor

import torch_pointcloud.transforms as T
from torch_pointcloud.datasets.nuscenes import NUSCENES_LIDARSEG_CLASSES
from torch_pointcloud.datasets.semantickitti import SEMANTIC_KITTI_CLASSES, SEMANTIC_KITTI_LEARNING_MAP
from torch_pointcloud.layers import (
    AsymmetricResidualBlock,
    AsymmetricUpsampleBlock,
    DimensionDecompositionBlock,
    DynamicMaxVFE,
)
from torch_pointcloud.models._base import SemanticSegmentationModel
from torch_pointcloud.models._registry import WeightsDict, register_model
from torch_pointcloud.ops.geometry import cartesian_to_cylindrical
from torch_pointcloud.utils.conversion import convert_from_spconv_tensor
from torch_pointcloud.utils.data import DataKeys
from torch_pointcloud.utils.imports import _SPCONV_GITHUB_URL, optional_import
from torch_pointcloud.utils.types import FeaturesDict

if TYPE_CHECKING:
    import spconv.pytorch as spconv
    from spconv.pytorch import SparseConvTensor


spconv, _ = optional_import("spconv.pytorch", url=_SPCONV_GITHUB_URL)


class Cylinder3DSegmentation(SemanticSegmentationModel):
    r"""Cylinder3D: a sparse U-Net of asymmetric convolutions on a cylindrical voxel partition of a LiDAR scan.

    :arxiv: [Cylindrical and Asymmetrical 3D Convolution Networks for LiDAR Segmentation](https://arxiv.org/abs/2011.10033)

    The points are partitioned on a cylindrical $(\rho, \phi, z)$ grid of `grid_size` cells over `point_cloud_range`,
    the last cell along each axis starting at the upper bound. A `DynamicMaxVFE` encodes each point (offset to its
    voxel center, cylindrical coordinates, Cartesian $x, y$ and the features `x`) and max-pools it per voxel. A sparse
    U-Net of `AsymmetricResidualBlock` and `AsymmetricUpsampleBlock` stages processes the voxels, a
    `DimensionDecompositionBlock` gates the full-resolution features, and a $3 \times 3 \times 3$ submanifold
    convolution classifies each voxel. `forward` returns the logits of every point, those of its voxel; `forward_head`
    returns the per-voxel logits.

    The released checkpoints were trained with an index table shared by all the convolutions of a block, so each
    convolution of a block uses the kernel of the block's first one; the model builds that network. They were also
    trained on the per-voxel logits against the majority label of each voxel, not on the per-point logits of
    `forward`:

    ```{.python notest}
    from torch_pointcloud.ops.voxelization import majority_label

    sparse_x, inverse, skips = model.forward_features(x, pos, batch, return_intermediates=True)
    sparse_x = model.forward_decoder(sparse_x, skips)
    logits = model.forward_head(sparse_x)  # (V, num_classes)
    target = majority_label(segment, inverse, num_classes, logits.shape[0], ignore_index=-1)
    loss = torch.nn.functional.cross_entropy(logits, target, ignore_index=-1)
    ```

    Run mixed precision under `torch.autocast`, not `.half()`: `.half()` also rounds the positions and the grid bounds
    to float16, which moves points to neighboring voxels.

    Args:
        in_channels: Point feature channels, without the coordinates (e.g. $1$ for the LiDAR intensity).
        num_classes: Number of semantic classes; `0` drops the head.
        point_cloud_range: Bounds $(\rho_\min, \phi_\min, z_\min, \rho_\max, \phi_\max, z_\max)$ of the cylindrical
            grid; points outside are clamped to it.
        grid_size: Number of cells $(n_\rho, n_\phi, n_z)$ of the cylindrical grid.
        vfe_channels: Output width of each point MLP layer of the voxel encoder.
        vfe_out_channels: Width of the voxel features entering the U-Net.
        channels: Width of the stem and of each encoder stage; the decoder mirrors them.
        strides: Stride of the downsampling convolution after each encoder stage.
        norm: Normalization passed to `create_norm`.
        norm_kwargs: Extra keyword arguments for the normalization.

    Example:
        ```python
        import torch
        from torch_pointcloud.models import create_model

        model = create_model("cylinder3d.semantickitti.mmdet3d", pretrained=True).cuda().eval()
        pos = torch.randn(20000, 3).cuda() * 20
        intensity = torch.rand(20000, 1).cuda()
        batch = torch.zeros(20000, dtype=torch.long).cuda()
        logits = model(intensity, pos, batch)  # (20000, 19)
        ```
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        *,
        point_cloud_range: Sequence[float],
        grid_size: Sequence[int],
        vfe_channels: Sequence[int] = (64, 128, 256, 256),
        vfe_out_channels: int = 16,
        channels: Sequence[int] = (32, 64, 128, 256, 512),
        strides: Sequence[Tuple[int, int, int]] = ((2, 2, 2), (2, 2, 2), (2, 2, 1), (2, 2, 1)),
        norm: Union[str, Callable, None] = "batch_norm",
        norm_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        if len(point_cloud_range) != 6:
            raise ValueError(f"Expected 6 range bounds (3 lower then 3 upper), got {len(point_cloud_range)}.")
        if len(grid_size) != 3 or min(grid_size) < 2:
            raise ValueError(f"Expected 3 grid sizes of at least 2 cells, got {tuple(grid_size)}.")
        if len(channels) < 2:
            raise ValueError(f"Expected the stem width and at least one encoder stage width, got {tuple(channels)}.")
        if len(strides) != len(channels) - 1:
            raise ValueError(f"Expected one stride per encoder stage ({len(channels) - 1}), got {len(strides)}.")

        super().__init__(in_channels=in_channels, num_classes=num_classes)
        self.grid_size = tuple(grid_size)
        self.channels = tuple(channels)

        range_min = torch.tensor(point_cloud_range[:3], dtype=torch.float32)
        range_max = torch.tensor(point_cloud_range[3:], dtype=torch.float32)
        # The last cell along each axis starts at the upper bound of the range.
        voxel_size = (range_max - range_min) / (torch.tensor(self.grid_size, dtype=torch.float32) - 1)
        self.vfe = DynamicMaxVFE(
            in_channels + 2,
            vfe_channels,
            vfe_out_channels,
            voxel_size=voxel_size.tolist(),
            point_cloud_range=point_cloud_range,
            grid_size=grid_size,
            norm=norm,
            norm_kwargs=norm_kwargs,
        )
        self.stem = AsymmetricResidualBlock(
            vfe_out_channels,
            channels[0],
            ((1, 3, 3), (1, 3, 3)),
            indice_key="stem",
            norm=norm,
            norm_kwargs=norm_kwargs,
        )
        self.encoder = nn.ModuleList(
            AsymmetricResidualBlock(
                channels[i],
                channels[i + 1],
                ((3, 1, 3), (3, 1, 3)),
                indice_key=f"encoder{i}",
                norm=norm,
                norm_kwargs=norm_kwargs,
            )
            for i in range(len(strides))
        )
        self.downsamples = nn.ModuleList(
            spconv.SparseConv3d(
                channels[i + 1],
                channels[i + 1],
                3,
                stride=stride,
                padding=1,
                bias=False,
                indice_key=f"downsample{i}",
            )
            for i, stride in enumerate(strides)
        )
        decoder_channels = [channels[-1], *reversed(channels[1:])]
        self.decoder = nn.ModuleList(
            AsymmetricUpsampleBlock(
                decoder_channels[j],
                decoder_channels[j + 1],
                ((1, 3, 3), (1, 3, 3), (1, 3, 3)),
                indice_key=f"decoder{j}",
                upsample_indice_key=f"downsample{len(strides) - 1 - j}",
                norm=norm,
                norm_kwargs=norm_kwargs,
            )
            for j in range(len(strides))
        )
        self.gate = DimensionDecompositionBlock(
            channels[1],
            ((3, 1, 1), (3, 1, 1), (3, 1, 1)),
            indice_key="gate",
            norm=norm,
            norm_kwargs=norm_kwargs,
        )
        self.head = self.configure_head()

    @property
    def num_features(self) -> int:
        """Channel count $C$ of the gated features concatenated with the decoder features, entering the head."""
        return 2 * self.channels[1]

    def configure_head(self) -> nn.Module:
        if self.num_classes == 0:
            return nn.Identity()
        return spconv.SubMConv3d(self.num_features, self.num_classes, 3, padding=1, bias=True, indice_key="head")

    def reset_classifier(self, num_classes: int) -> None:
        self.num_classes = num_classes
        self.head = self.configure_head()

    @overload
    def forward_features(
        self,
        x: Tensor,
        pos: Tensor,
        batch: Tensor,
        return_intermediates: Literal[True],
    ) -> Tuple["SparseConvTensor", Tensor, List[FeaturesDict]]: ...

    @overload
    def forward_features(
        self,
        x: Tensor,
        pos: Tensor,
        batch: Tensor,
        return_intermediates: Literal[False] = False,
    ) -> Tuple["SparseConvTensor", Tensor]: ...

    def forward_features(
        self,
        x: Tensor,
        pos: Tensor,
        batch: Tensor,
        return_intermediates: bool = False,
    ) -> Any:
        """Encode the voxels into the coarsest sparse features; also return the voxel of each point.

        With `return_intermediates`, the output of every encoder stage is appended, fine to coarse.
        """
        point_x = torch.cat([pos[:, :2], x], dim=1)
        voxel_x, voxel_indices, inverse = self.vfe(point_x, cartesian_to_cylindrical(pos), batch)
        batch_size = int(batch.max().item()) + 1
        # spconv's inference path skips autocast, so the features must already have the weight dtype.
        voxel_x = voxel_x.to(self.stem.conv1.weight.dtype)
        sparse_x = spconv.SparseConvTensor(voxel_x, voxel_indices, list(self.grid_size), batch_size)

        sparse_x = self.stem(sparse_x)
        skips: List[FeaturesDict] = []
        for block, downsample in zip(self.encoder, self.downsamples):
            sparse_x = block(sparse_x)
            x_skip, pos_grid_skip, batch_skip = convert_from_spconv_tensor(sparse_x)
            skips.append({"x": x_skip, "pos_grid": pos_grid_skip, "batch": batch_skip})
            sparse_x = downsample(sparse_x)

        if return_intermediates:
            return sparse_x, inverse, skips
        return sparse_x, inverse

    def forward_decoder(self, x: "SparseConvTensor", skips: List[FeaturesDict]) -> "SparseConvTensor":
        """Upsample through the encoder stages, then concatenate the gated full-resolution features to them."""
        for block, skip in zip(self.decoder, reversed(skips)):
            x = block(x, skip["x"])
        gated = self.gate(x)
        return gated.replace_feature(torch.cat([gated.features, x.features], dim=1))

    def forward_head(self, x: "SparseConvTensor", pre_logits: bool = False) -> Tensor:
        """Per-voxel logits, or the per-voxel features with `pre_logits`; `forward` gathers them per point."""
        return x.features if pre_logits else self.head(x).features

    def forward(self, x: Tensor, pos: Tensor, batch: Tensor) -> Tensor:
        sparse_x, inverse, skips = self.forward_features(x, pos, batch, return_intermediates=True)
        sparse_x = self.forward_decoder(sparse_x, skips)
        logits = self.forward_head(sparse_x)
        return logits[inverse]


@register_model(
    "cylinder3d.semantickitti.mmdet3d",
    task="semantic-segmentation",
    input_keys=("x", "pos", "batch"),
    weights=WeightsDict(
        url="hf://torch-pointcloud/cylinder3d.semantickitti.mmdet3d/resolve/0000000000000000000000000000000000000000/model.safetensors",
        dataset="semantickitti",
        metrics={"mIoU": 63.19, "OA": 91.08},
        classes=SEMANTIC_KITTI_CLASSES,
        author="mmdet3d",
        license="Apache-2.0",
    ),
    transform=T.Compose(
        [
            T.Relabel(keys=DataKeys.SEGMENT, labels=SEMANTIC_KITTI_LEARNING_MAP, default=-1, allow_missing_keys=True),
            T.Cat(keys=[DataKeys.INTENSITY], dst_key=DataKeys.X, dim=1),
        ]
    ),
    hparams=dict(
        in_channels=1,
        num_classes=19,
        point_cloud_range=(0.0, -3.14159265359, -4.0, 50.0, 3.14159265359, 2.0),
        grid_size=(480, 360, 32),
    ),
)
def cylinder3d_semantickitti_mmdet3d(**hparams: Any) -> Cylinder3DSegmentation:
    return Cylinder3DSegmentation(**hparams)


@register_model(
    "cylinder3d-laser-polar-mix.semantickitti.mmdet3d",
    task="semantic-segmentation",
    input_keys=("x", "pos", "batch"),
    weights=WeightsDict(
        url="hf://torch-pointcloud/cylinder3d-laser-polar-mix.semantickitti.mmdet3d/resolve/0000000000000000000000000000000000000000/model.safetensors",
        dataset="semantickitti",
        metrics={"mIoU": 67.54, "OA": 91.87},
        classes=SEMANTIC_KITTI_CLASSES,
        author="mmdet3d",
        license="Apache-2.0",
    ),
    transform=T.Compose(
        [
            T.Relabel(keys=DataKeys.SEGMENT, labels=SEMANTIC_KITTI_LEARNING_MAP, default=-1, allow_missing_keys=True),
            T.Cat(keys=[DataKeys.INTENSITY], dst_key=DataKeys.X, dim=1),
        ]
    ),
    hparams=dict(
        in_channels=1,
        num_classes=19,
        point_cloud_range=(0.0, -3.14159265359, -4.0, 50.0, 3.14159265359, 2.0),
        grid_size=(480, 360, 32),
    ),
)
def cylinder3d_laser_polar_mix_semantickitti_mmdet3d(**hparams: Any) -> Cylinder3DSegmentation:
    return Cylinder3DSegmentation(**hparams)


@register_model(
    "cylinder3d.semantickitti.xinge-zhu",
    task="semantic-segmentation",
    input_keys=("x", "pos", "batch"),
    weights=WeightsDict(
        url="hf://torch-pointcloud/cylinder3d.semantickitti.xinge-zhu/resolve/0000000000000000000000000000000000000000/model.safetensors",
        dataset="semantickitti",
        metrics={"mIoU": 66.91, "OA": 91.49},
        classes=SEMANTIC_KITTI_CLASSES,
        author="xinge-zhu",
        license="Apache-2.0",
    ),
    transform=T.Compose(
        [
            T.Relabel(keys=DataKeys.SEGMENT, labels=SEMANTIC_KITTI_LEARNING_MAP, default=-1, allow_missing_keys=True),
            T.Cat(keys=[DataKeys.INTENSITY], dst_key=DataKeys.X, dim=1),
        ]
    ),
    hparams=dict(
        in_channels=1,
        num_classes=19,
        point_cloud_range=(0.0, -3.1415926, -4.0, 50.0, 3.1415926, 2.0),
        grid_size=(480, 360, 32),
    ),
)
def cylinder3d_semantickitti_xinge_zhu(**hparams: Any) -> Cylinder3DSegmentation:
    return Cylinder3DSegmentation(**hparams)


@register_model(
    "cylinder3d-0p5x.nuscenes.yuenan-hou",
    task="semantic-segmentation",
    input_keys=("x", "pos", "batch"),
    weights=WeightsDict(
        url="hf://torch-pointcloud/cylinder3d-0p5x.nuscenes.yuenan-hou/resolve/0000000000000000000000000000000000000000/model.safetensors",
        dataset="nuscenes",
        metrics={"mIoU": 76.15, "OA": 93.22},
        classes=NUSCENES_LIDARSEG_CLASSES,
        author="yuenan-hou",
        license="MIT",
    ),
    transform=T.Compose(
        [
            # Raw lidarseg category ids onto the 16 benchmark classes.
            T.Relabel(
                keys=DataKeys.SEGMENT,
                labels={
                    9: 0,
                    14: 1,
                    15: 2,
                    16: 2,
                    17: 3,
                    18: 4,
                    21: 5,
                    2: 6,
                    3: 6,
                    4: 6,
                    6: 6,
                    12: 7,
                    22: 8,
                    23: 9,
                    24: 10,
                    25: 11,
                    26: 12,
                    27: 13,
                    28: 14,
                    30: 15,
                },
                default=-1,
                allow_missing_keys=True,
            ),
            T.Cat(keys=[DataKeys.INTENSITY], dst_key=DataKeys.X, dim=1),
        ]
    ),
    hparams=dict(
        in_channels=1,
        num_classes=16,
        point_cloud_range=(0.0, -3.1415926, -5.0, 50.0, 3.1415926, 3.0),
        grid_size=(480, 360, 32),
        vfe_channels=(32, 64, 128, 128),
        vfe_out_channels=16,
        channels=(16, 32, 64, 128, 256),
    ),
)
def cylinder3d_0p5x_nuscenes_yuenan_hou(**hparams: Any) -> Cylinder3DSegmentation:
    return Cylinder3DSegmentation(**hparams)
