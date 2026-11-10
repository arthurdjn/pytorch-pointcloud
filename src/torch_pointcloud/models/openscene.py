"""OpenScene open-vocabulary segmentation model.

{{ paper("2211.15654") }}
"""

from typing import TYPE_CHECKING, Any, Callable, Dict, List, Literal, Optional, Sequence, Tuple, Union, overload

import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

import torch_pointcloud.transforms as T
from torch_pointcloud.datasets.nuscenes import NUSCENES_LIDARSEG_LEARNING_MAP
from torch_pointcloud.models._base import OpenVocabularySegmentationModel
from torch_pointcloud.models._registry import WeightsDict, register_model
from torch_pointcloud.models.spunet import SparseUNetDecoder, SparseUNetEncoder
from torch_pointcloud.utils.data import DataKeys
from torch_pointcloud.utils.types import FeaturesDict, OptTensor

if TYPE_CHECKING:
    from spconv.pytorch import SparseConvTensor


class OpenSceneOpenVocabularySegmentation(OpenVocabularySegmentationModel):
    r"""OpenScene open-vocabulary segmentation model, a sparse residual U-Net distilled into the text space of CLIP,
    from :arxiv: [OpenScene: 3D Scene Understanding with Open Vocabularies](https://arxiv.org/abs/2211.15654)
    by Songyou Peng, Kyle Genova, Chiyu "Max" Jiang, Andrea Tagliasacchi, Marc Pollefeys, Thomas Funkhouser.

    Voxelized coordinates run through a `SparseUNetEncoder` whose stride-2 convolutions keep the width of the stage
    before them (the first residual block of a stage changes it), a `SparseUNetDecoder`, and a linear projection of the
    full-resolution decoder features onto the embedding space of a text encoder. `forward` returns unit vectors, so
    their product with unit text embeddings (`layers.CLIPTextEncoder`) is the cosine similarity that labels a point
    with the nearest prompt, for any vocabulary. The released weights were distilled from per-pixel OpenSeg or LSeg
    features and read geometry only: `x=None` feeds a constant feature of `in_channels` ones per voxel. Each
    registered checkpoint names its text encoder, the revision of that encoder and its prompt template in
    `weights["text_model"]`, `weights["text_revision"]` and `weights["prompt_template"]`.

    Note:
        Run the model in float32 on a GPU: `spconv` has no CPU kernel, and the forward fails under
        `torch.autocast(dtype=torch.float16)`.

    Args:
        in_channels: Number of input feature channels. `x=None` feeds ones.
        embed_dim: Width of the text space the embeddings live in.
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
        super().__init__(in_channels=in_channels, embed_dim=embed_dim)
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

    def configure_encoder(self) -> SparseUNetEncoder:
        """Build the sparse encoder producing the bottleneck features and the per-stage skips."""
        return SparseUNetEncoder(
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
            down_channels=[self.stem_channels, *self.encoder_channels[:-1]],
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
        if x is None:
            weight = self.head.weight
            assert isinstance(weight, Tensor)
            x = weight.new_ones(pos_grid.shape[0], self.in_channels)

        if return_intermediates:
            return self.encoder(x, pos_grid, batch, return_intermediates=True)
        return self.encoder(x, pos_grid, batch)

    def forward_decoder(self, x: "SparseConvTensor", skips: List[FeaturesDict]) -> "SparseConvTensor":
        return self.decoder(x, skips)

    def forward_head(self, x: "SparseConvTensor", pre_logits: bool = False) -> Tensor:
        if pre_logits:
            return x.features

        embeddings = self.head(x.features)
        embeddings = F.normalize(embeddings, dim=1)
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
        url="hf://torch-pointcloud/openscene-openseg.scannet20.songyou-peng/resolve/8cb7996e87c7dccc938367d48da90f3600eb991e/model.safetensors",
        dataset="scannet20",
        metrics={
            "mIoU": 46.11,
            "mAcc": 66.64,
            "OA": 69.76,
            "simple/mIoU": 45.33,
            "simple/mAcc": 66.07,
            "simple/OA": 69.09,
        },
        text_model="openai/clip-vit-large-patch14-336",
        text_revision="ce19dc912ca5cd21c8a653c79e251e808ccabcd1",
        prompt_template="a {} in a scene",
        author="songyou-peng",
        license="Apache-2.0",
    ),
    transform=T.Compose(
        [
            T.Relabel(keys=DataKeys.SEGMENT, labels=range(1, 21), default=-1, allow_missing_keys=True),
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
        stem_channels=32,
        encoder_channels=(32, 64, 128, 256),
        decoder_channels=(128, 128, 96, 96),
        encoder_depths=(2, 2, 2, 2),
        decoder_depths=(2, 2, 2, 2),
        stem_kernel_size=5,
        kernel_size=3,
    ),
)
def openscene_openseg_scannet20(**hparams: Any) -> OpenSceneOpenVocabularySegmentation:
    return OpenSceneOpenVocabularySegmentation(**hparams)


@register_model(
    "openscene-lseg.scannet20.songyou-peng",
    task="open-vocabulary-segmentation",
    input_keys=("x", "pos_grid", "batch"),
    weights=WeightsDict(
        url="hf://torch-pointcloud/openscene-lseg.scannet20.songyou-peng/resolve/a44269075170bd33d1d4f7a5f84abbd0d53b2b9d/model.safetensors",
        dataset="scannet20",
        metrics={
            "mIoU": 52.97,
            "mAcc": 63.43,
            "OA": 80.90,
            "simple/mIoU": 52.04,
            "simple/mAcc": 62.83,
            "simple/OA": 80.10,
        },
        text_model="openai/clip-vit-base-patch32",
        text_revision="3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268",
        prompt_template="a {} in a scene",
        author="songyou-peng",
        license="Apache-2.0",
    ),
    transform=T.Compose(
        [
            T.Relabel(keys=DataKeys.SEGMENT, labels=range(1, 21), default=-1, allow_missing_keys=True),
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
        stem_channels=32,
        encoder_channels=(32, 64, 128, 256),
        decoder_channels=(128, 128, 96, 96),
        encoder_depths=(2, 2, 2, 2),
        decoder_depths=(2, 2, 2, 2),
        stem_kernel_size=5,
        kernel_size=3,
    ),
)
def openscene_lseg_scannet20(**hparams: Any) -> OpenSceneOpenVocabularySegmentation:
    return OpenSceneOpenVocabularySegmentation(**hparams)


@register_model(
    "openscene-openseg.nuscenes.songyou-peng",
    task="open-vocabulary-segmentation",
    input_keys=("x", "pos_grid", "batch"),
    weights=WeightsDict(
        url="hf://torch-pointcloud/openscene-openseg.nuscenes.songyou-peng/resolve/44279170b08beabff330f70e597aeef2ed188cc7/model.safetensors",
        dataset="nuscenes",
        metrics={
            "mIoU": 42.97,
            "mAcc": 56.99,
            "OA": 77.60,
            "simple/mIoU": 41.84,
            "simple/mAcc": 56.05,
            "simple/OA": 77.28,
        },
        text_model="openai/clip-vit-large-patch14-336",
        text_revision="ce19dc912ca5cd21c8a653c79e251e808ccabcd1",
        prompt_template="a {} in a scene",
        author="songyou-peng",
        license="Apache-2.0",
    ),
    transform=T.Compose(
        [
            T.Relabel(
                keys=DataKeys.SEGMENT,
                labels=NUSCENES_LIDARSEG_LEARNING_MAP,
                default=-1,
                allow_missing_keys=True,
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
        stem_channels=32,
        encoder_channels=(32, 64, 128, 256),
        decoder_channels=(128, 128, 96, 96),
        encoder_depths=(2, 2, 2, 2),
        decoder_depths=(2, 2, 2, 2),
        stem_kernel_size=5,
        kernel_size=3,
    ),
)
def openscene_openseg_nuscenes(**hparams: Any) -> OpenSceneOpenVocabularySegmentation:
    return OpenSceneOpenVocabularySegmentation(**hparams)
