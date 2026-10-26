"""MinkUNet segmentation model.

{{ paper("1904.08755") }}
"""

from typing import TYPE_CHECKING, Any, Callable, Dict, List, Literal, Optional, Sequence, Tuple, Union, overload

import torch
import torch.nn as nn
from torch import Tensor

import torch_pointcloud.transforms as T
from torch_pointcloud.datasets.semantickitti import SEMANTIC_KITTI_CLASSES, SEMANTIC_KITTI_LEARNING_MAP
from torch_pointcloud.layers import SparseBasicBlock, SparseConvBlock
from torch_pointcloud.models._base import SemanticSegmentationModel
from torch_pointcloud.models._registry import WeightsDict, register_model
from torch_pointcloud.models.spunet import SparseUNetDecoder
from torch_pointcloud.ops.voxelization import trilinear_sparse_devoxelize
from torch_pointcloud.utils.conversion import convert_from_spconv_tensor, convert_to_spconv_tensor
from torch_pointcloud.utils.data import DataKeys
from torch_pointcloud.utils.imports import _SPCONV_GITHUB_URL, optional_import
from torch_pointcloud.utils.types import FeaturesDict, OptTensor

if TYPE_CHECKING:
    import spconv.pytorch as spconv
    from spconv.pytorch import SparseConvTensor


spconv, _ = optional_import("spconv.pytorch", url=_SPCONV_GITHUB_URL)


class MinkUNetEncoder(nn.Module):
    """Two-convolution submanifold stem, then stages that halve the resolution and run `SparseBasicBlock` blocks.

    The strided convolution of a stage keeps the width of the previous stage; the first residual block of the
    stage widens the features. The stem output and every stage output but the last are returned as decoder skips.
    """

    def __init__(
        self,
        in_channels: int,
        stem_channels: int = 32,
        channels: Sequence[int] = (32, 64, 128, 256),
        depths: Sequence[int] = (2, 2, 2, 2),
        stem_kernel_size: int = 3,
        kernel_size: int = 3,
        spatial_padding: int = 96,
        act: Union[str, Callable, None] = "relu",
        act_kwargs: Optional[Dict[str, Any]] = None,
        norm: Union[str, Callable, None] = "batch_norm",
        norm_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()
        if len(depths) != len(channels):
            raise ValueError(
                f"`depths` and `channels` must have the same length, got {len(depths)} and {len(channels)}."
            )

        self.in_channels = in_channels
        self.stem_channels = stem_channels
        self.channels = tuple(channels)
        self.depths = tuple(depths)
        self.num_stages = len(self.depths)
        self.spatial_padding = spatial_padding

        self.conv_input = spconv.SparseSequential(
            SparseConvBlock(
                in_channels,
                stem_channels,
                kernel_size=stem_kernel_size,
                indice_key="subm0",
                conv_type="subm",
                norm=norm,
                norm_kwargs=norm_kwargs,
                act=act,
                act_kwargs=act_kwargs,
            ),
            SparseConvBlock(
                stem_channels,
                stem_channels,
                kernel_size=stem_kernel_size,
                indice_key="subm0",
                conv_type="subm",
                norm=norm,
                norm_kwargs=norm_kwargs,
                act=act,
                act_kwargs=act_kwargs,
            ),
        )

        self.down = nn.ModuleList()
        self.enc = nn.ModuleList()

        enc_channels = stem_channels
        for s in range(self.num_stages):
            self.down.append(
                SparseConvBlock(
                    enc_channels,
                    enc_channels,
                    kernel_size=2,
                    stride=2,
                    indice_key=f"spconv{s + 1}",
                    conv_type="spconv",
                    norm=norm,
                    norm_kwargs=norm_kwargs,
                    act=act,
                    act_kwargs=act_kwargs,
                )
            )
            blocks = spconv.SparseSequential()
            for i in range(self.depths[s]):
                blocks.add_module(
                    f"block{i}",
                    SparseBasicBlock(
                        enc_channels if i == 0 else self.channels[s],
                        self.channels[s],
                        kernel_size=kernel_size,
                        indice_key=f"subm{s + 1}",
                        act=act,
                        act_kwargs=act_kwargs,
                        norm=norm,
                        norm_kwargs=norm_kwargs,
                    ),
                )
            self.enc.append(blocks)
            enc_channels = self.channels[s]

    @overload
    def forward(
        self,
        x: OptTensor,
        pos_grid: Tensor,
        batch: Tensor,
        *,
        return_intermediates: Literal[True],
    ) -> Tuple["SparseConvTensor", List[FeaturesDict]]: ...

    @overload
    def forward(
        self,
        x: OptTensor,
        pos_grid: Tensor,
        batch: Tensor,
        *,
        return_intermediates: Literal[False] = False,
    ) -> "SparseConvTensor": ...

    def forward(
        self,
        x: OptTensor,
        pos_grid: Tensor,
        batch: Tensor,
        *,
        return_intermediates: bool = False,
    ) -> Any:
        if x is None:
            x = pos_grid.float()

        sparse_x = convert_to_spconv_tensor(x, pos_grid, batch, padding=self.spatial_padding)
        sparse_x = self.conv_input(sparse_x)

        skips: List[FeaturesDict] = []
        if return_intermediates:
            x_skip, pos_grid_skip, batch_skip = convert_from_spconv_tensor(sparse_x)
            skips.append({"x": x_skip, "pos_grid": pos_grid_skip, "batch": batch_skip})
        for s in range(self.num_stages):
            sparse_x = self.down[s](sparse_x)
            sparse_x = self.enc[s](sparse_x)
            if return_intermediates and s < self.num_stages - 1:
                x_skip, pos_grid_skip, batch_skip = convert_from_spconv_tensor(sparse_x)
                skips.append({"x": x_skip, "pos_grid": pos_grid_skip, "batch": batch_skip})

        if return_intermediates:
            return sparse_x, skips
        return sparse_x


class MinkUNetSegmentation(SemanticSegmentationModel):
    r"""MinkUNet segmentation model as described in the paper
    :arxiv: [4D Spatio-Temporal ConvNets: Minkowski Convolutional Neural Networks](https://arxiv.org/abs/1904.08755)
    by Christopher Choy, JunYoung Gwak, Silvio Savarese.

    A sparse residual U-Net over voxelized scans: a two-convolution submanifold stem, stages of strided convolution and
    residual blocks, and a mirrored decoder of inverse convolutions whose stages concatenate the encoder skips. A linear
    head classifies the full-resolution decoder features, whose rows stay aligned with the input voxels. With
    `head_stages`, the head reads the concatenation of several decoding stages instead, the coarser ones interpolated
    trilinearly at the input voxels.

    Args:
        in_channels: Number of input feature channels. Positions are used as features when `x` is `None`.
        num_classes: Number of output classes. $0$ replaces the head with `nn.Identity`.
        stem_channels: Number of channels of the stem, also the width of the shallowest skip.
        encoder_channels: Feature width of every encoder stage.
        decoder_channels: Feature width of every decoder stage, deepest first.
        encoder_depths: Number of residual blocks per encoder stage.
        decoder_depths: Number of residual blocks per decoder stage, aligned with `decoder_channels`.
        head_stages: Feature maps concatenated as the head input, in decoding order: $0$ is the encoder bottleneck and
            $i \geq 1$ the output of the $i$-th decoder stage. `None` uses the last decoder stage alone.
        stem_kernel_size: Kernel size of the two stem convolutions.
        kernel_size: Kernel size of the residual convolutions.
        spatial_padding: Padding added to the sparse spatial shape, so that voxel indices stay in range.
        act: Activation function.
        act_kwargs: Keyword arguments for the activation function.
        norm: Normalization function.
        norm_kwargs: Keyword arguments for the normalization function.

    Shape:
        - `x`: $(N, C_\text{in})$, `pos_grid`: $(N, 3)$ integer voxel coordinates, `batch`: $(N,)$.
        - Output: $(N, \text{num\_classes})$.

    Example:
        ```python
        import torch
        from torch_pointcloud.models import MinkUNetSegmentation

        model = MinkUNetSegmentation(in_channels=4, num_classes=19).cuda()
        pos_grid = torch.randint(0, 256, (1024, 3)).cuda()
        x = torch.randn(1024, 4).cuda()
        batch = torch.zeros(1024, dtype=torch.long).cuda()
        logits = model(x, pos_grid, batch)  # (1024, 19)
        ```
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        *,
        stem_channels: int = 32,
        encoder_channels: Sequence[int] = (32, 64, 128, 256),
        decoder_channels: Sequence[int] = (256, 128, 96, 96),
        encoder_depths: Sequence[int] = (2, 2, 2, 2),
        decoder_depths: Sequence[int] = (2, 2, 2, 2),
        head_stages: Optional[Sequence[int]] = None,
        stem_kernel_size: int = 3,
        kernel_size: int = 3,
        spatial_padding: int = 96,
        act: Union[str, Callable, None] = "relu",
        act_kwargs: Optional[Dict[str, Any]] = None,
        norm: Union[str, Callable, None] = "batch_norm",
        norm_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(in_channels=in_channels, num_classes=num_classes)
        if len(decoder_channels) != len(encoder_channels):
            raise ValueError(
                f"`decoder_channels` and `encoder_channels` must have the same length, "
                f"got {len(decoder_channels)} and {len(encoder_channels)}."
            )

        if len(encoder_depths) != len(encoder_channels):
            raise ValueError(
                f"`encoder_depths` and `encoder_channels` must have the same length, "
                f"got {len(encoder_depths)} and {len(encoder_channels)}."
            )

        if len(decoder_depths) != len(decoder_channels):
            raise ValueError(
                f"`decoder_depths` and `decoder_channels` must have the same length, "
                f"got {len(decoder_depths)} and {len(decoder_channels)}."
            )

        if head_stages is not None and not all(0 <= i <= len(decoder_channels) for i in head_stages):
            raise ValueError(
                f"`head_stages` must index the bottleneck (0) or a decoder stage (1 to {len(decoder_channels)}), "
                f"got {tuple(head_stages)}."
            )

        self.stem_channels = stem_channels
        self.encoder_channels = tuple(encoder_channels)
        self.decoder_channels = tuple(decoder_channels)
        self.encoder_depths = tuple(encoder_depths)
        self.decoder_depths = tuple(decoder_depths)
        self.head_stages = None if head_stages is None else tuple(head_stages)
        self.stem_kernel_size = stem_kernel_size
        self.kernel_size = kernel_size
        self.spatial_padding = spatial_padding
        self.act = act
        self.act_kwargs = act_kwargs
        self.norm = norm
        self.norm_kwargs = norm_kwargs

        self.encoder = self.configure_encoder()
        self.decoder = self.configure_decoder()
        self.head: nn.Module = self.configure_head()

    def configure_encoder(self) -> MinkUNetEncoder:
        """Build the sparse encoder producing the bottleneck features and the per-stage skips."""
        return MinkUNetEncoder(
            in_channels=self.in_channels,
            stem_channels=self.stem_channels,
            channels=self.encoder_channels,
            depths=self.encoder_depths,
            stem_kernel_size=self.stem_kernel_size,
            kernel_size=self.kernel_size,
            spatial_padding=self.spatial_padding,
            act=self.act,
            act_kwargs=self.act_kwargs,
            norm=self.norm,
            norm_kwargs=self.norm_kwargs,
        )

    def configure_decoder(self) -> SparseUNetDecoder:
        """Build the sparse decoder upsampling the bottleneck back to full resolution."""
        skip_channels = [self.stem_channels, *self.encoder_channels[:-1]]
        return SparseUNetDecoder(
            in_channels=self.encoder_channels[-1],
            skip_channels=skip_channels,
            channels=self.decoder_channels,
            depths=self.decoder_depths,
            kernel_size=self.kernel_size,
            act=self.act,
            act_kwargs=self.act_kwargs,
            norm=self.norm,
            norm_kwargs=self.norm_kwargs,
        )

    @property
    def num_features(self) -> int:
        """Channel count $C$ of the features entering the head, one row per input voxel."""
        if self.head_stages is None:
            return self.decoder_channels[-1]

        widths = [self.encoder_channels[-1], *self.decoder_channels]
        return sum(widths[i] for i in self.head_stages)

    def configure_head(self) -> nn.Module:
        if self.num_classes == 0:
            return nn.Identity()
        return nn.Linear(self.num_features, self.num_classes)

    def reset_classifier(self, num_classes: int) -> None:
        self.num_classes = num_classes
        self.head = self.configure_head()

    @overload
    def forward_features(
        self,
        x: OptTensor,
        pos_grid: Tensor,
        batch: Tensor,
        return_intermediates: Literal[True],
    ) -> Tuple["SparseConvTensor", List[FeaturesDict]]: ...

    @overload
    def forward_features(
        self,
        x: OptTensor,
        pos_grid: Tensor,
        batch: Tensor,
        return_intermediates: Literal[False] = False,
    ) -> "SparseConvTensor": ...

    def forward_features(
        self,
        x: OptTensor,
        pos_grid: Tensor,
        batch: Tensor,
        return_intermediates: bool = False,
    ) -> Any:
        if return_intermediates:
            return self.encoder(x, pos_grid, batch, return_intermediates=True)
        return self.encoder(x, pos_grid, batch)

    def forward_decoder(self, x: "SparseConvTensor", skips: List[FeaturesDict]) -> Tensor:
        """Decode the bottleneck `x` into the head input, one row per input voxel."""
        if self.head_stages is None:
            out = self.decoder(x, skips)
            return out.features

        out, stages = self.decoder(x, skips, return_intermediates=True)
        maps = [x, *stages]
        pos_grid, batch = out.indices[:, 1:], out.indices[:, 0]
        features = []
        for i in self.head_stages:
            stride = 2 ** (len(maps) - 1 - i)
            if stride == 1:
                features.append(maps[i].features)
            else:
                stage = maps[i]
                features.append(
                    trilinear_sparse_devoxelize(
                        x_voxel=stage.features,
                        pos_voxel=stage.indices[:, 1:],
                        batch_voxel=stage.indices[:, 0],
                        pos=pos_grid / stride,
                        batch=batch,
                        normalize=True,
                    )
                )
        return torch.cat(features, dim=1)

    def forward_head(self, x: Tensor, pre_logits: bool = False) -> Tensor:
        return x if pre_logits else self.head(x)

    def forward(self, x: OptTensor, pos_grid: Tensor, batch: Tensor) -> Tensor:
        sparse_x, skips = self.forward_features(x, pos_grid, batch, return_intermediates=True)
        features = self.forward_decoder(sparse_x, skips)
        return self.forward_head(features)


_SEMANTICKITTI_VOXEL_SIZE = 0.05


def _semantickitti_transform(learning_map: Dict[int, int]) -> T.Compose:
    return T.Compose(
        [
            T.Relabel(keys=DataKeys.SEGMENT, labels=learning_map, default=255, allow_missing_keys=True),
            T.Cat(keys=[DataKeys.POS, DataKeys.INTENSITY], dst_key=DataKeys.X, dim=1),
            T.CopyItems(
                keys=[DataKeys.POS, DataKeys.SEGMENT],
                dst_keys=[DataKeys.ORIGIN_POS, DataKeys.ORIGIN_SEGMENT],
                allow_missing_keys=True,
            ),
            # The checkpoints round the coordinates to the grid: shifting by half a voxel turns the floor of
            # `Voxelize` into a round.
            T.Translate(keys=DataKeys.POS, offset=_SEMANTICKITTI_VOXEL_SIZE / 2),
            T.Voxelize(
                pos_key=DataKeys.POS,
                pos_reduce="first",
                dst_pos_grid_key=DataKeys.POS_GRID,
                keys=[DataKeys.X, DataKeys.SEGMENT, DataKeys.INTENSITY, DataKeys.INSTANCE],
                reduce="first",
                size=_SEMANTICKITTI_VOXEL_SIZE,
                allow_missing_keys=True,
                dst_inverse_key=DataKeys.INVERSE,
            ),
        ]
    )


# The mit-han-lab checkpoints were trained with bus, on-rails and lane marking ignored instead of merged.
_MIT_HAN_LAB_LEARNING_MAP = {k: v for k, v in SEMANTIC_KITTI_LEARNING_MAP.items() if k not in (13, 16, 60, 256, 257)}


@register_model(
    "minkunet18-w16.semantickitti.mmdet3d",
    task="semantic-segmentation",
    input_keys=("x", "pos_grid", "batch"),
    weights=WeightsDict(
        url="hf://torch-pointcloud/minkunet18-w16.semantickitti.mmdet3d/resolve/0000000000000000000000000000000000000000/model.safetensors",
        dataset="semantickitti",
        metrics={"mIoU": 60.25, "OA": 91.78},
        classes=SEMANTIC_KITTI_CLASSES,
        author="mmdet3d",
        license="Apache-2.0",
    ),
    transform=_semantickitti_transform(SEMANTIC_KITTI_LEARNING_MAP),
    hparams=dict(
        in_channels=4,
        num_classes=19,
        stem_channels=16,
        encoder_channels=(16, 32, 64, 128),
        decoder_channels=(128, 64, 48, 48),
        encoder_depths=(2, 2, 2, 2),
        decoder_depths=(2, 2, 2, 2),
    ),
)
def minkunet18_w16_semantickitti_mmdet3d(**hparams: Any) -> MinkUNetSegmentation:
    return MinkUNetSegmentation(**hparams)


@register_model(
    "minkunet18-w20.semantickitti.mmdet3d",
    task="semantic-segmentation",
    input_keys=("x", "pos_grid", "batch"),
    weights=WeightsDict(
        url="hf://torch-pointcloud/minkunet18-w20.semantickitti.mmdet3d/resolve/0000000000000000000000000000000000000000/model.safetensors",
        dataset="semantickitti",
        metrics={"mIoU": 61.58, "OA": 91.93},
        classes=SEMANTIC_KITTI_CLASSES,
        author="mmdet3d",
        license="Apache-2.0",
    ),
    transform=_semantickitti_transform(SEMANTIC_KITTI_LEARNING_MAP),
    hparams=dict(
        in_channels=4,
        num_classes=19,
        stem_channels=20,
        encoder_channels=(20, 40, 81, 163),
        decoder_channels=(163, 81, 61, 61),
        encoder_depths=(2, 2, 2, 2),
        decoder_depths=(2, 2, 2, 2),
    ),
)
def minkunet18_w20_semantickitti_mmdet3d(**hparams: Any) -> MinkUNetSegmentation:
    return MinkUNetSegmentation(**hparams)


@register_model(
    "minkunet18-w32.semantickitti.mmdet3d",
    task="semantic-segmentation",
    input_keys=("x", "pos_grid", "batch"),
    weights=WeightsDict(
        url="hf://torch-pointcloud/minkunet18-w32.semantickitti.mmdet3d/resolve/0000000000000000000000000000000000000000/model.safetensors",
        dataset="semantickitti",
        metrics={"mIoU": 63.06, "OA": 92.02},
        classes=SEMANTIC_KITTI_CLASSES,
        author="mmdet3d",
        license="Apache-2.0",
    ),
    transform=_semantickitti_transform(SEMANTIC_KITTI_LEARNING_MAP),
    hparams=dict(
        in_channels=4,
        num_classes=19,
        stem_channels=32,
        encoder_channels=(32, 64, 128, 256),
        decoder_channels=(256, 128, 96, 96),
        encoder_depths=(2, 2, 2, 2),
        decoder_depths=(2, 2, 2, 2),
    ),
)
def minkunet18_w32_semantickitti_mmdet3d(**hparams: Any) -> MinkUNetSegmentation:
    return MinkUNetSegmentation(**hparams)


@register_model(
    "minkunet34-w32-torchsparse.semantickitti.mmdet3d",
    task="semantic-segmentation",
    input_keys=("x", "pos_grid", "batch"),
    weights=WeightsDict(
        url="hf://torch-pointcloud/minkunet34-w32-torchsparse.semantickitti.mmdet3d/resolve/0000000000000000000000000000000000000000/model.safetensors",
        dataset="semantickitti",
        metrics={"mIoU": 68.92, "OA": 92.49},
        classes=SEMANTIC_KITTI_CLASSES,
        author="mmdet3d",
        license="Apache-2.0",
    ),
    transform=_semantickitti_transform(SEMANTIC_KITTI_LEARNING_MAP),
    hparams=dict(
        in_channels=4,
        num_classes=19,
        stem_channels=32,
        encoder_channels=(32, 64, 128, 256),
        decoder_channels=(256, 128, 96, 96),
        encoder_depths=(2, 3, 4, 6),
        decoder_depths=(2, 2, 2, 2),
    ),
)
def minkunet34_w32_torchsparse_semantickitti_mmdet3d(**hparams: Any) -> MinkUNetSegmentation:
    return MinkUNetSegmentation(**hparams)


@register_model(
    "minkunet34-w32-torchsparse-amp.semantickitti.mmdet3d",
    task="semantic-segmentation",
    input_keys=("x", "pos_grid", "batch"),
    weights=WeightsDict(
        url="hf://torch-pointcloud/minkunet34-w32-torchsparse-amp.semantickitti.mmdet3d/resolve/0000000000000000000000000000000000000000/model.safetensors",
        dataset="semantickitti",
        metrics={"mIoU": 69.16, "OA": 92.25},
        classes=SEMANTIC_KITTI_CLASSES,
        author="mmdet3d",
        license="Apache-2.0",
    ),
    transform=_semantickitti_transform(SEMANTIC_KITTI_LEARNING_MAP),
    hparams=dict(
        in_channels=4,
        num_classes=19,
        stem_channels=32,
        encoder_channels=(32, 64, 128, 256),
        decoder_channels=(256, 128, 96, 96),
        encoder_depths=(2, 3, 4, 6),
        decoder_depths=(2, 2, 2, 2),
    ),
)
def minkunet34_w32_torchsparse_amp_semantickitti_mmdet3d(**hparams: Any) -> MinkUNetSegmentation:
    return MinkUNetSegmentation(**hparams)


@register_model(
    "minkunet34-w32-spconv.semantickitti.mmdet3d",
    task="semantic-segmentation",
    input_keys=("x", "pos_grid", "batch"),
    weights=WeightsDict(
        url="hf://torch-pointcloud/minkunet34-w32-spconv.semantickitti.mmdet3d/resolve/0000000000000000000000000000000000000000/model.safetensors",
        dataset="semantickitti",
        metrics={"mIoU": 68.64, "OA": 92.06},
        classes=SEMANTIC_KITTI_CLASSES,
        author="mmdet3d",
        license="Apache-2.0",
    ),
    transform=_semantickitti_transform(SEMANTIC_KITTI_LEARNING_MAP),
    hparams=dict(
        in_channels=4,
        num_classes=19,
        stem_channels=32,
        encoder_channels=(32, 64, 128, 256),
        decoder_channels=(256, 128, 96, 96),
        encoder_depths=(2, 3, 4, 6),
        decoder_depths=(2, 2, 2, 2),
    ),
)
def minkunet34_w32_spconv_semantickitti_mmdet3d(**hparams: Any) -> MinkUNetSegmentation:
    return MinkUNetSegmentation(**hparams)


@register_model(
    "minkunet34-w32-spconv-amp.semantickitti.mmdet3d",
    task="semantic-segmentation",
    input_keys=("x", "pos_grid", "batch"),
    weights=WeightsDict(
        url="hf://torch-pointcloud/minkunet34-w32-spconv-amp.semantickitti.mmdet3d/resolve/0000000000000000000000000000000000000000/model.safetensors",
        dataset="semantickitti",
        metrics={"mIoU": 68.57, "OA": 92.79},
        classes=SEMANTIC_KITTI_CLASSES,
        author="mmdet3d",
        license="Apache-2.0",
    ),
    transform=_semantickitti_transform(SEMANTIC_KITTI_LEARNING_MAP),
    hparams=dict(
        in_channels=4,
        num_classes=19,
        stem_channels=32,
        encoder_channels=(32, 64, 128, 256),
        decoder_channels=(256, 128, 96, 96),
        encoder_depths=(2, 3, 4, 6),
        decoder_depths=(2, 2, 2, 2),
    ),
)
def minkunet34_w32_spconv_amp_semantickitti_mmdet3d(**hparams: Any) -> MinkUNetSegmentation:
    return MinkUNetSegmentation(**hparams)


@register_model(
    "minkunet34-w32-minkowski.semantickitti.mmdet3d",
    task="semantic-segmentation",
    input_keys=("x", "pos_grid", "batch"),
    weights=WeightsDict(
        url="hf://torch-pointcloud/minkunet34-w32-minkowski.semantickitti.mmdet3d/resolve/0000000000000000000000000000000000000000/model.safetensors",
        dataset="semantickitti",
        metrics={"mIoU": 69.21, "OA": 92.36},
        classes=SEMANTIC_KITTI_CLASSES,
        author="mmdet3d",
        license="Apache-2.0",
    ),
    transform=_semantickitti_transform(SEMANTIC_KITTI_LEARNING_MAP),
    hparams=dict(
        in_channels=4,
        num_classes=19,
        stem_channels=32,
        encoder_channels=(32, 64, 128, 256),
        decoder_channels=(256, 128, 96, 96),
        encoder_depths=(2, 3, 4, 6),
        decoder_depths=(2, 2, 2, 2),
    ),
)
def minkunet34_w32_minkowski_semantickitti_mmdet3d(**hparams: Any) -> MinkUNetSegmentation:
    return MinkUNetSegmentation(**hparams)


@register_model(
    "minkunet34v2-w32.semantickitti.mmdet3d",
    task="semantic-segmentation",
    input_keys=("x", "pos_grid", "batch"),
    weights=WeightsDict(
        url="hf://torch-pointcloud/minkunet34v2-w32.semantickitti.mmdet3d/resolve/0000000000000000000000000000000000000000/model.safetensors",
        dataset="semantickitti",
        metrics={"mIoU": 70.20, "OA": 92.34},
        classes=SEMANTIC_KITTI_CLASSES,
        author="mmdet3d",
        license="Apache-2.0",
    ),
    transform=_semantickitti_transform(SEMANTIC_KITTI_LEARNING_MAP),
    hparams=dict(
        in_channels=4,
        num_classes=19,
        stem_channels=32,
        encoder_channels=(32, 64, 128, 256),
        decoder_channels=(256, 128, 96, 96),
        encoder_depths=(2, 3, 4, 6),
        decoder_depths=(2, 2, 2, 2),
        head_stages=(0, 2, 4),
    ),
)
def minkunet34v2_w32_semantickitti_mmdet3d(**hparams: Any) -> MinkUNetSegmentation:
    return MinkUNetSegmentation(**hparams)


@register_model(
    "minkunet-29gmacs.semantickitti.mit-han-lab",
    task="semantic-segmentation",
    input_keys=("x", "pos_grid", "batch"),
    weights=WeightsDict(
        url="hf://torch-pointcloud/minkunet-29gmacs.semantickitti.mit-han-lab/resolve/0000000000000000000000000000000000000000/model.safetensors",
        dataset="semantickitti",
        metrics={"mIoU": 58.90, "OA": 91.73},
        classes=SEMANTIC_KITTI_CLASSES,
        author="mit-han-lab",
        license="MIT",
    ),
    transform=_semantickitti_transform(_MIT_HAN_LAB_LEARNING_MAP),
    hparams=dict(
        in_channels=4,
        num_classes=19,
        stem_channels=16,
        encoder_channels=(16, 32, 64, 128),
        decoder_channels=(128, 64, 48, 48),
        encoder_depths=(2, 2, 2, 2),
        decoder_depths=(2, 2, 2, 2),
    ),
)
def minkunet_29gmacs_semantickitti_mit_han_lab(**hparams: Any) -> MinkUNetSegmentation:
    return MinkUNetSegmentation(**hparams)


@register_model(
    "minkunet-46gmacs.semantickitti.mit-han-lab",
    task="semantic-segmentation",
    input_keys=("x", "pos_grid", "batch"),
    weights=WeightsDict(
        url="hf://torch-pointcloud/minkunet-46gmacs.semantickitti.mit-han-lab/resolve/0000000000000000000000000000000000000000/model.safetensors",
        dataset="semantickitti",
        metrics={"mIoU": 60.26, "OA": 91.94},
        classes=SEMANTIC_KITTI_CLASSES,
        author="mit-han-lab",
        license="MIT",
    ),
    transform=_semantickitti_transform(_MIT_HAN_LAB_LEARNING_MAP),
    hparams=dict(
        in_channels=4,
        num_classes=19,
        stem_channels=20,
        encoder_channels=(20, 40, 81, 163),
        decoder_channels=(163, 81, 61, 61),
        encoder_depths=(2, 2, 2, 2),
        decoder_depths=(2, 2, 2, 2),
    ),
)
def minkunet_46gmacs_semantickitti_mit_han_lab(**hparams: Any) -> MinkUNetSegmentation:
    return MinkUNetSegmentation(**hparams)


@register_model(
    "minkunet-114gmacs.semantickitti.mit-han-lab",
    task="semantic-segmentation",
    input_keys=("x", "pos_grid", "batch"),
    weights=WeightsDict(
        url="hf://torch-pointcloud/minkunet-114gmacs.semantickitti.mit-han-lab/resolve/0000000000000000000000000000000000000000/model.safetensors",
        dataset="semantickitti",
        metrics={"mIoU": 61.06, "OA": 91.64},
        classes=SEMANTIC_KITTI_CLASSES,
        author="mit-han-lab",
        license="MIT",
    ),
    transform=_semantickitti_transform(_MIT_HAN_LAB_LEARNING_MAP),
    hparams=dict(
        in_channels=4,
        num_classes=19,
        stem_channels=32,
        encoder_channels=(32, 64, 128, 256),
        decoder_channels=(256, 128, 96, 96),
        encoder_depths=(2, 2, 2, 2),
        decoder_depths=(2, 2, 2, 2),
    ),
)
def minkunet_114gmacs_semantickitti_mit_han_lab(**hparams: Any) -> MinkUNetSegmentation:
    return MinkUNetSegmentation(**hparams)
