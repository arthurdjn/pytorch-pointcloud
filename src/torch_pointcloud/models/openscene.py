"""OpenScene open-vocabulary segmentation model.

{{ paper("2211.15654") }}
"""

from typing import TYPE_CHECKING, Any, Callable, Dict, List, Literal, Optional, Sequence, Tuple, Union, overload

import torch
import torch.nn as nn
from torch import Tensor

import torch_pointcloud.transforms as T
from torch_pointcloud.layers import SparseBasicBlock
from torch_pointcloud.layers.act import create_act
from torch_pointcloud.layers.norms import create_norm
from torch_pointcloud.models._base import OpenVocabularySegmentationModel
from torch_pointcloud.models._registry import WeightsDict, register_model
from torch_pointcloud.models.spunet import SparseUNetDecoder
from torch_pointcloud.utils.conversion import convert_from_spconv_tensor, convert_to_spconv_tensor
from torch_pointcloud.utils.data import DataKeys
from torch_pointcloud.utils.imports import _SPCONV_GITHUB_URL, optional_import
from torch_pointcloud.utils.types import FeaturesDict, OptTensor

if TYPE_CHECKING:
    import spconv.pytorch as spconv
    from spconv.pytorch import SparseConvTensor


spconv, _ = optional_import("spconv.pytorch", url=_SPCONV_GITHUB_URL)
SparseConvTensor, _ = optional_import("spconv.pytorch", "SparseConvTensor", url=_SPCONV_GITHUB_URL)

CLIP_VIT_L_14_336 = "openai/clip-vit-large-patch14-336"
CLIP_VIT_B_32 = "openai/clip-vit-base-patch32"


def _init_openscene_weights(m: nn.Module) -> None:
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, (spconv.SubMConv3d, spconv.SparseConv3d, spconv.SparseInverseConv3d)):
        nn.init.trunc_normal_(m.weight, std=0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm1d):
        nn.init.constant_(m.bias, 0)
        nn.init.constant_(m.weight, 1.0)


class OpenSceneEncoder(nn.Module):
    """Sparse convolutional stem followed by stages that halve the resolution and run `SparseBasicBlock` blocks.

    The stride-2 convolution of a stage keeps the width of the stage before it and the first residual block changes
    it. The stem output and every stage output but the last are returned as decoder skips.
    """

    def __init__(
        self,
        in_channels: int,
        stem_channels: int = 32,
        channels: Sequence[int] = (32, 64, 128, 256),
        depths: Sequence[int] = (2, 2, 2, 2),
        stem_kernel_size: int = 5,
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

        norm_kwargs = norm_kwargs or {}
        act_kwargs = act_kwargs or {}

        self.stem = spconv.SparseSequential(
            spconv.SubMConv3d(
                in_channels,
                stem_channels,
                kernel_size=stem_kernel_size,
                padding=stem_kernel_size // 2,
                bias=False,
                indice_key="stem",
            ),
            create_norm(norm, stem_channels, **norm_kwargs) or nn.Identity(),
            create_act(act, **act_kwargs) or nn.Identity(),
        )

        self.down = nn.ModuleList()
        self.enc = nn.ModuleList()

        width = stem_channels
        for s in range(self.num_stages):
            self.down.append(
                spconv.SparseSequential(
                    spconv.SparseConv3d(width, width, kernel_size=2, stride=2, bias=False, indice_key=f"spconv{s + 1}"),
                    create_norm(norm, width, **norm_kwargs) or nn.Identity(),
                    create_act(act, **act_kwargs) or nn.Identity(),
                )
            )
            blocks = [
                SparseBasicBlock(
                    width if i == 0 else self.channels[s],
                    self.channels[s],
                    kernel_size=kernel_size,
                    indice_key=f"subm{s + 1}",
                    act=act,
                    act_kwargs=act_kwargs,
                    norm=norm,
                    norm_kwargs=norm_kwargs,
                )
                for i in range(self.depths[s])
            ]
            self.enc.append(spconv.SparseSequential(*blocks))
            width = self.channels[s]

        self.apply(_init_openscene_weights)

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
            x = torch.ones(pos_grid.shape[0], self.in_channels, device=pos_grid.device)

        sparse_x = convert_to_spconv_tensor(x, pos_grid, batch, padding=self.spatial_padding)
        sparse_x = self.stem(sparse_x)

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


class OpenSceneSegmentation(OpenVocabularySegmentationModel):
    r"""OpenScene open-vocabulary segmentation model, a sparse residual U-Net distilled into the text space of CLIP, from
    :arxiv: [OpenScene: 3D Scene Understanding with Open Vocabularies](https://arxiv.org/abs/2211.15654)
    by Songyou Peng, Kyle Genova, Chiyu "Max" Jiang, Andrea Tagliasacchi, Marc Pollefeys, Thomas Funkhouser.

    Voxelized coordinates enter a sparse convolutional stem, then a symmetric encoder-decoder of submanifold residual
    blocks with skip connections (MinkUNet18A), and a linear head projects the full-resolution decoder features onto
    the embedding space of a CLIP text encoder. The released weights were distilled from per-pixel OpenSeg or LSeg
    features and read geometry only: `x=None` feeds a constant feature of `in_channels` ones per voxel. A point
    embedding compared by cosine similarity with the embedding of a prompt (`layers.CLIPTextEncoder`) scores any
    word, with no training on it.

    Args:
        in_channels: Number of input feature channels. `x=None` feeds ones.
        embed_dim: Width of the text space: $768$ for CLIP ViT-L/14@336px (OpenSeg), $512$ for ViT-B/32 (LSeg).
        text_model: Hub id of the CLIP checkpoint the embeddings are aligned with, `openai/clip-vit-large-patch14-336`
            or `openai/clip-vit-base-patch32`.
        stem_channels: Number of channels of the stem, also the width of the shallowest skip.
        encoder_channels: Feature width of every encoder stage.
        decoder_channels: Feature width of every decoder stage, deepest first.
        encoder_depths: Number of residual blocks per encoder stage.
        decoder_depths: Number of residual blocks per decoder stage, aligned with `decoder_channels`.
        stem_kernel_size: Kernel size of the stem convolution.
        kernel_size: Kernel size of the residual convolutions.
        spatial_padding: Padding added to the sparse spatial shape, so that voxel indices stay in range.
        act: Activation function.
        act_kwargs: Keyword arguments for the activation function.
        norm: Normalization function.
        norm_kwargs: Keyword arguments for the normalization function.
    """

    def __init__(
        self,
        in_channels: int,
        embed_dim: int,
        text_model: str,
        *,
        stem_channels: int = 32,
        encoder_channels: Sequence[int] = (32, 64, 128, 256),
        decoder_channels: Sequence[int] = (128, 128, 96, 96),
        encoder_depths: Sequence[int] = (2, 2, 2, 2),
        decoder_depths: Sequence[int] = (2, 2, 2, 2),
        stem_kernel_size: int = 5,
        kernel_size: int = 3,
        spatial_padding: int = 96,
        act: Union[str, Callable, None] = "relu",
        act_kwargs: Optional[Dict[str, Any]] = None,
        norm: Union[str, Callable, None] = "batch_norm",
        norm_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(in_channels=in_channels, embed_dim=embed_dim, text_model=text_model)
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

        self.stem_channels = stem_channels
        self.encoder_channels = tuple(encoder_channels)
        self.decoder_channels = tuple(decoder_channels)
        self.encoder_depths = tuple(encoder_depths)
        self.decoder_depths = tuple(decoder_depths)
        self.stem_kernel_size = stem_kernel_size
        self.kernel_size = kernel_size
        self.spatial_padding = spatial_padding
        self.act = act
        self.act_kwargs = act_kwargs
        self.norm = norm
        self.norm_kwargs = norm_kwargs

        self.encoder = self.configure_encoder()
        self.decoder = self.configure_decoder()
        self.head = self.configure_head()
        self.apply(_init_openscene_weights)

    def configure_encoder(self) -> OpenSceneEncoder:
        """Build the sparse encoder producing the bottleneck features and the per-stage skips."""
        return OpenSceneEncoder(
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
        # The decoder reads its stages deepest first; the bottleneck (last encoder stage) is its input, not a skip.
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
        """Channel count $C$ of the full-resolution decoder features entering the projection."""
        return self.decoder_channels[-1]

    def configure_head(self) -> nn.Module:
        return nn.Linear(self.num_features, self.embed_dim, bias=False)

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

    def forward_decoder(self, x: "SparseConvTensor", skips: List[FeaturesDict]) -> "SparseConvTensor":
        return self.decoder(x, skips)

    def forward_head(self, x: "SparseConvTensor", pre_logits: bool = False) -> Tensor:
        if pre_logits:
            return x.features

        embeddings: Tensor = self.head(x.features)
        return embeddings

    def forward(self, x: OptTensor, pos_grid: Tensor, batch: Tensor) -> Tensor:
        sparse_x, skips = self.forward_features(x, pos_grid, batch, return_intermediates=True)
        sparse_x = self.forward_decoder(sparse_x, skips)
        return self.forward_head(sparse_x)


@register_model(
    "openscene-openseg.scannet20.songyou-peng",
    task="open-vocabulary-segmentation",
    input_keys=("x", "pos_grid", "batch"),
    weights=WeightsDict(
        url="hf://torch-pointcloud/openscene-openseg.scannet20.songyou-peng/resolve/0000000000000000000000000000000000000000/model.safetensors",
        dataset="scannet20",
        metrics={"mIoU": 45.33, "mAcc": 66.07, "OA": 69.09},
        author="songyou-peng",
        license="Apache-2.0",
    ),
    transform=T.Compose(
        [
            T.Relabel(keys=DataKeys.SEGMENT, labels=range(1, 21), default=-1),
            T.CopyItems(
                keys=[DataKeys.POS, DataKeys.SEGMENT],
                dst_keys=[DataKeys.ORIGIN_POS, DataKeys.ORIGIN_SEGMENT],
                allow_missing_keys=True,
            ),
            T.Voxelize(
                pos_key=DataKeys.POS,
                pos_reduce="first",
                dst_pos_grid_key=DataKeys.POS_GRID,
                keys=[DataKeys.SEGMENT, DataKeys.COLOR, DataKeys.NORMAL, DataKeys.INSTANCE],
                reduce="first",
                size=0.02,
                allow_missing_keys=True,
                dst_inverse_key=DataKeys.INVERSE,
            ),
        ]
    ),
    hparams=dict(
        in_channels=3,
        embed_dim=768,
        text_model=CLIP_VIT_L_14_336,
        stem_channels=32,
        encoder_channels=(32, 64, 128, 256),
        decoder_channels=(128, 128, 96, 96),
        encoder_depths=(2, 2, 2, 2),
        decoder_depths=(2, 2, 2, 2),
        stem_kernel_size=5,
        kernel_size=3,
    ),
)
def openscene_openseg_scannet20(**hparams: Any) -> OpenSceneSegmentation:
    return OpenSceneSegmentation(**hparams)


@register_model(
    "openscene-lseg.scannet20.songyou-peng",
    task="open-vocabulary-segmentation",
    input_keys=("x", "pos_grid", "batch"),
    weights=WeightsDict(
        url="hf://torch-pointcloud/openscene-lseg.scannet20.songyou-peng/resolve/0000000000000000000000000000000000000000/model.safetensors",
        dataset="scannet20",
        metrics={"mIoU": 52.04, "mAcc": 62.83, "OA": 80.09},
        author="songyou-peng",
        license="Apache-2.0",
    ),
    transform=T.Compose(
        [
            T.Relabel(keys=DataKeys.SEGMENT, labels=range(1, 21), default=-1),
            T.CopyItems(
                keys=[DataKeys.POS, DataKeys.SEGMENT],
                dst_keys=[DataKeys.ORIGIN_POS, DataKeys.ORIGIN_SEGMENT],
                allow_missing_keys=True,
            ),
            T.Voxelize(
                pos_key=DataKeys.POS,
                pos_reduce="first",
                dst_pos_grid_key=DataKeys.POS_GRID,
                keys=[DataKeys.SEGMENT, DataKeys.COLOR, DataKeys.NORMAL, DataKeys.INSTANCE],
                reduce="first",
                size=0.02,
                allow_missing_keys=True,
                dst_inverse_key=DataKeys.INVERSE,
            ),
        ]
    ),
    hparams=dict(
        in_channels=3,
        embed_dim=512,
        text_model=CLIP_VIT_B_32,
        stem_channels=32,
        encoder_channels=(32, 64, 128, 256),
        decoder_channels=(128, 128, 96, 96),
        encoder_depths=(2, 2, 2, 2),
        decoder_depths=(2, 2, 2, 2),
        stem_kernel_size=5,
        kernel_size=3,
    ),
)
def openscene_lseg_scannet20(**hparams: Any) -> OpenSceneSegmentation:
    return OpenSceneSegmentation(**hparams)


@register_model(
    "openscene-openseg.nuscenes.songyou-peng",
    task="open-vocabulary-segmentation",
    input_keys=("x", "pos_grid", "batch"),
    weights=WeightsDict(
        url="hf://torch-pointcloud/openscene-openseg.nuscenes.songyou-peng/resolve/0000000000000000000000000000000000000000/model.safetensors",
        dataset="nuscenes",
        author="songyou-peng",
        license="Apache-2.0",
    ),
    transform=T.Compose(
        [
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
            ),
            T.CopyItems(
                keys=[DataKeys.POS, DataKeys.SEGMENT],
                dst_keys=[DataKeys.ORIGIN_POS, DataKeys.ORIGIN_SEGMENT],
                allow_missing_keys=True,
            ),
            T.Voxelize(
                pos_key=DataKeys.POS,
                pos_reduce="first",
                dst_pos_grid_key=DataKeys.POS_GRID,
                keys=[DataKeys.SEGMENT, DataKeys.INTENSITY, DataKeys.TIMESTAMP],
                reduce="first",
                size=0.05,
                allow_missing_keys=True,
                dst_inverse_key=DataKeys.INVERSE,
            ),
        ]
    ),
    hparams=dict(
        in_channels=3,
        embed_dim=768,
        text_model=CLIP_VIT_L_14_336,
        stem_channels=32,
        encoder_channels=(32, 64, 128, 256),
        decoder_channels=(128, 128, 96, 96),
        encoder_depths=(2, 2, 2, 2),
        decoder_depths=(2, 2, 2, 2),
        stem_kernel_size=5,
        kernel_size=3,
    ),
)
def openscene_openseg_nuscenes(**hparams: Any) -> OpenSceneSegmentation:
    return OpenSceneSegmentation(**hparams)
