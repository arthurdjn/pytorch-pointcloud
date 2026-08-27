"""SPFormer-UNet segmentation model.

{{ paper("2211.15766") }}
"""

from collections import OrderedDict
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Literal,
    Optional,
    Sequence,
    Tuple,
    Union,
    overload,
)

import torch
import torch.nn as nn
from torch import Tensor

import torch_pointcloud.transforms as T
from torch_pointcloud.layers import SparseResidualBlock
from torch_pointcloud.layers.act import create_act
from torch_pointcloud.layers.norms import create_norm
from torch_pointcloud.models._base import SegmentationModel
from torch_pointcloud.models._registry import register_model
from torch_pointcloud.utils.conversion import convert_to_spconv_tensor, ensure_tuple, ensure_tuple_size
from torch_pointcloud.utils.data import DataKeys
from torch_pointcloud.utils.imports import _SPCONV_GITHUB_URL, optional_import
from torch_pointcloud.utils.types import OptTensor

if TYPE_CHECKING:
    import spconv.pytorch as spconv
    from spconv.pytorch import SparseConvTensor


spconv, _ = optional_import("spconv.pytorch", url=_SPCONV_GITHUB_URL)
SparseConvTensor, _ = optional_import("spconv.pytorch", "SparseConvTensor", url=_SPCONV_GITHUB_URL)


def _make_residual_seq(
    in_channels: int,
    out_channels: int,
    depth: int,
    indice_key: str,
    *,
    act: Union[str, Callable, None] = "relu",
    act_kwargs: Optional[Dict[str, Any]] = None,
    norm: Union[str, Callable, None] = "batch_norm",
    norm_kwargs: Optional[Dict[str, Any]] = None,
) -> "spconv.SparseSequential":
    blocks = OrderedDict()
    for i in range(depth):
        blocks[f"block{i}"] = SparseResidualBlock(
            in_channels if i == 0 else out_channels,
            out_channels,
            indice_key=indice_key,
            act=act,
            act_kwargs=act_kwargs,
            norm=norm,
            norm_kwargs=norm_kwargs,
        )
    return spconv.SparseSequential(blocks)


class SPFormerUNetEncoderBlock(nn.Module):
    r"""One encoder stage: an optional stride-2 downsample followed by `depth` residual blocks.

    Args:
        channels: Channel width of this stage (the residual blocks run at this width).
        depth: Number of residual blocks.
        indice_key: SpConv submanifold index key shared by the residual blocks.
        downsample: Stride-2 down-conv applied before the blocks, or `None` for the first
            (full-resolution) stage.
        act: Activation passed to `create_act`.
        act_kwargs: Extra keyword arguments for the activation.
        norm: Normalization passed to `create_norm`.
        norm_kwargs: Extra keyword arguments for the normalization.
    """

    def __init__(
        self,
        channels: int,
        depth: int,
        indice_key: str,
        *,
        downsample: Optional[nn.Module] = None,
        act: Union[str, Callable, None] = "relu",
        act_kwargs: Optional[Dict[str, Any]] = None,
        norm: Union[str, Callable, None] = "batch_norm",
        norm_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()
        self.downsample = downsample
        self.blocks = _make_residual_seq(
            channels,
            channels,
            depth,
            indice_key=indice_key,
            act=act,
            act_kwargs=act_kwargs,
            norm=norm,
            norm_kwargs=norm_kwargs,
        )

    def forward(self, x: "SparseConvTensor") -> "SparseConvTensor":
        if self.downsample is not None:
            x = self.downsample(x)
        return self.blocks(x)


class SPFormerUNetDecoderBlock(nn.Module):
    r"""One decoder stage: upsample, concatenate the encoder skip, then `depth` residual blocks.

    Args:
        channels: Output channel width of this stage.
        depth: Number of residual blocks.
        indice_key: SpConv submanifold index key shared by the residual blocks.
        upsample: Inverse conv mapping the deeper feature to `channels`.
        act: Activation passed to `create_act`.
        act_kwargs: Extra keyword arguments for the activation.
        norm: Normalization passed to `create_norm`.
        norm_kwargs: Extra keyword arguments for the normalization.
    """

    def __init__(
        self,
        channels: int,
        depth: int,
        indice_key: str,
        *,
        upsample: nn.Module,
        act: Union[str, Callable, None] = "relu",
        act_kwargs: Optional[Dict[str, Any]] = None,
        norm: Union[str, Callable, None] = "batch_norm",
        norm_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()
        self.upsample = upsample
        self.blocks = _make_residual_seq(
            channels * 2,
            channels,
            depth,
            indice_key=indice_key,
            act=act,
            act_kwargs=act_kwargs,
            norm=norm,
            norm_kwargs=norm_kwargs,
        )

    def forward(self, x: "SparseConvTensor", skip: "SparseConvTensor") -> "SparseConvTensor":
        x = self.upsample(x)
        x = x.replace_feature(torch.cat([skip.features, x.features], dim=1))
        return self.blocks(x)


class SPFormerUNetEncoder(nn.Module):
    r"""Downsampling path of the SPFormer SpConv U-Net.

    Embeds the input with a submanifold stem, then runs one
    [`SPFormerUNetEncoderBlock`](#) per level (`blocks`); every block but the first
    downsamples by stride 2 before its residual blocks. The output of every level
    but the deepest is returned as a skip connection for the decoder.

    Args:
        in_channels: Number of input feature channels.
        channels: Per-level channel widths, deepest level last.
        layers: Number of residual blocks per level; an `int` is broadcast to every level.
        stem_kernel_size: Kernel size of the submanifold stem convolution.
        spatial_padding: Padding (in voxels) added to the inferred spatial shape.
        act: Activation passed to `create_act`.
        act_kwargs: Extra keyword arguments for the activation.
        norm: Normalization passed to `create_norm`.
        norm_kwargs: Extra keyword arguments for the normalization.

    Shape:
        - Input: packed features $(N, \text{in\_channels})$, grid coordinates, batch.
        - Output: `SparseConvTensor` with `channels[-1]` channels (bottleneck).
    """

    def __init__(
        self,
        in_channels: int,
        channels: Sequence[int],
        layers: Union[int, Sequence[int]],
        *,
        stem_kernel_size: int = 3,
        spatial_padding: int = 96,
        act: Union[str, Callable, None] = "relu",
        act_kwargs: Optional[Dict[str, Any]] = None,
        norm: Union[str, Callable, None] = "batch_norm",
        norm_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.channels = ensure_tuple(channels)
        self.num_levels = len(self.channels)
        self.layers = ensure_tuple_size(
            layers, size=self.num_levels, extra_msg="`layers` must match `channels` length."
        )
        self.spatial_padding = spatial_padding
        norm_kwargs = norm_kwargs or {}
        act_kwargs = act_kwargs or {}

        self.stem = spconv.SubMConv3d(
            in_channels,
            self.channels[0],
            kernel_size=stem_kernel_size,
            padding=stem_kernel_size // 2,
            bias=False,
            indice_key="subm1",
        )

        self.blocks = nn.ModuleList()
        for i in range(self.num_levels):
            downsample: Optional[nn.Module] = None
            if i > 0:
                downsample = spconv.SparseSequential(
                    create_norm(norm, self.channels[i - 1], **norm_kwargs) or nn.Identity(),
                    create_act(act, **act_kwargs) or nn.Identity(),
                    spconv.SparseConv3d(
                        self.channels[i - 1],
                        self.channels[i],
                        kernel_size=2,
                        stride=2,
                        bias=False,
                        indice_key=f"spconv{i}",
                    ),
                )
            self.blocks.append(
                SPFormerUNetEncoderBlock(
                    self.channels[i],
                    self.layers[i],
                    indice_key=f"subm{i + 1}",
                    downsample=downsample,
                    act=act,
                    act_kwargs=act_kwargs,
                    norm=norm,
                    norm_kwargs=norm_kwargs,
                )
            )

    @overload
    def forward(
        self,
        x: OptTensor,
        pos_grid: Tensor,
        batch: Tensor,
        *,
        return_intermediates: Literal[True],
    ) -> Tuple["SparseConvTensor", List["SparseConvTensor"]]: ...

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
        out = self.stem(sparse_x)

        skips: List["SparseConvTensor"] = []
        for i, block in enumerate(self.blocks):
            out = block(out)
            if i < self.num_levels - 1:
                skips.append(out)
        if return_intermediates:
            return out, skips
        return out


class SPFormerUNetDecoder(nn.Module):
    r"""Upsampling path of the SPFormer SpConv U-Net.

    Runs one [`SPFormerUNetDecoderBlock`](#) per upsampling level (`blocks`),
    deepest first. Each block upsamples the deeper feature, concatenates the
    matching encoder skip, and fuses them back to `channels[i]` channels.

    Args:
        channels: Per-level channel widths, deepest level last (same as the encoder).
        layers: Number of residual blocks per level; an `int` is broadcast to every level.
        act: Activation passed to `create_act`.
        act_kwargs: Extra keyword arguments for the activation.
        norm: Normalization passed to `create_norm`.
        norm_kwargs: Extra keyword arguments for the normalization.

    Shape:
        - Input: `SparseConvTensor` with `channels[-1]` channels (bottleneck).
        - Output: `SparseConvTensor` with `channels[0]` channels.
    """

    def __init__(
        self,
        channels: Sequence[int],
        layers: Union[int, Sequence[int]],
        *,
        act: Union[str, Callable, None] = "relu",
        act_kwargs: Optional[Dict[str, Any]] = None,
        norm: Union[str, Callable, None] = "batch_norm",
        norm_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()
        self.channels = ensure_tuple(channels)
        self.num_levels = len(self.channels)
        self.layers = ensure_tuple_size(
            layers, size=self.num_levels, extra_msg="`layers` must match `channels` length."
        )
        norm_kwargs = norm_kwargs or {}
        act_kwargs = act_kwargs or {}

        self.blocks = nn.ModuleList()
        for i in reversed(range(self.num_levels - 1)):
            upsample = spconv.SparseSequential(
                create_norm(norm, self.channels[i + 1], **norm_kwargs) or nn.Identity(),
                create_act(act, **act_kwargs) or nn.Identity(),
                spconv.SparseInverseConv3d(
                    self.channels[i + 1],
                    self.channels[i],
                    kernel_size=2,
                    bias=False,
                    indice_key=f"spconv{i + 1}",
                ),
            )
            block = SPFormerUNetDecoderBlock(
                self.channels[i],
                self.layers[i],
                indice_key=f"subm{i + 1}",
                upsample=upsample,
                act=act,
                act_kwargs=act_kwargs,
                norm=norm,
                norm_kwargs=norm_kwargs,
            )
            self.blocks.append(block)

    def forward(self, x: "SparseConvTensor", skips: List["SparseConvTensor"]) -> "SparseConvTensor":
        out = x
        for block, skip in zip(self.blocks, reversed(skips)):
            out = block(out, skip)
        return out


class SPFormerUNetSegmentation(SegmentationModel):
    r"""SpConv U-Net from SPFormer.

    Reference: :github: [sunjiahao1999/SPFormer](https://github.com/sunjiahao1999/SPFormer).
    A symmetric submanifold sparse-convolution U-Net with pre-norm residual blocks,
    reused as the backbone of OneFormer3D. Distinct from
    [`SparseUNetSegmentation`](#): the residual blocks are pre-norm and the
    stem-level blocks run at full resolution before any downsampling.

    Set `num_classes=0` to drop the classifier: the head keeps only the final normalization
    and activation and `forward` returns per-voxel features, which is how OneFormer3D
    consumes it.

    Args:
        in_channels: Number of input feature channels.
        num_classes: Number of output classes; `0` yields an identity head.
        channels: Per-level channel widths, deepest level last.
        layers: Number of residual blocks per level; an `int` is broadcast to every level.
        stem_kernel_size: Kernel size of the submanifold stem convolution.
        spatial_padding: Padding (in voxels) added to the inferred spatial shape.
        act: Activation passed to `create_act`.
        act_kwargs: Extra keyword arguments for the activation.
        norm: Normalization passed to `create_norm`.
        norm_kwargs: Extra keyword arguments for the normalization (e.g. `eps`, `momentum`).

    Shape:
        - Input: packed features $(N, \text{in\_channels})$, grid coordinates, batch.
        - Output: $(N, \text{num\_classes})$ logits, or $(N, \text{channels}[0])$ features when `num_classes=0`.
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        *,
        channels: Sequence[int] = (32, 64, 96, 128, 160),
        layers: Union[int, Sequence[int]] = 2,
        stem_kernel_size: int = 3,
        spatial_padding: int = 96,
        act: Union[str, Callable, None] = "relu",
        act_kwargs: Optional[Dict[str, Any]] = None,
        norm: Union[str, Callable, None] = "batch_norm",
        norm_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(in_channels=in_channels, num_classes=num_classes)
        self.channels = ensure_tuple(channels)
        self.layers = ensure_tuple_size(
            layers, size=len(self.channels), extra_msg="`layers` must match `channels` length."
        )
        self.stem_kernel_size = stem_kernel_size
        self.spatial_padding = spatial_padding
        self.act = act
        self.act_kwargs = act_kwargs
        self.norm = norm
        self.norm_kwargs = norm_kwargs

        self.encoder = self.configure_encoder()
        self.decoder = self.configure_decoder()
        self.head = self.configure_head()

    @property
    def embedding_dim(self) -> int:
        """Channel count $C$ of the full-resolution decoder features entering the head."""
        return self.channels[0]

    def configure_encoder(self) -> SPFormerUNetEncoder:
        """Builds the sparse encoder producing the bottleneck features and the per-stage skips."""
        return SPFormerUNetEncoder(
            self.in_channels,
            self.channels,
            self.layers,
            stem_kernel_size=self.stem_kernel_size,
            spatial_padding=self.spatial_padding,
            act=self.act,
            act_kwargs=self.act_kwargs,
            norm=self.norm,
            norm_kwargs=self.norm_kwargs,
        )

    def configure_decoder(self) -> SPFormerUNetDecoder:
        """Builds the sparse decoder upsampling the bottleneck back to full resolution."""
        return SPFormerUNetDecoder(
            self.channels,
            self.layers,
            act=self.act,
            act_kwargs=self.act_kwargs,
            norm=self.norm,
            norm_kwargs=self.norm_kwargs,
        )

    def configure_head(self) -> nn.Module:
        norm = create_norm(self.norm, self.channels[0], **(self.norm_kwargs or {}))
        act = create_act(self.act, **(self.act_kwargs or {}))
        modules: List[nn.Module] = [m for m in (norm, act) if m is not None]
        if self.num_classes > 0:
            modules.append(spconv.SubMConv3d(self.channels[0], self.num_classes, kernel_size=1, bias=True))
        return spconv.SparseSequential(*modules)

    def reset_classifier(self, num_classes: int) -> None:
        self.num_classes = num_classes
        self.head = self.configure_head()

    def forward_features(
        self,
        x: OptTensor,
        pos_grid: Tensor,
        batch: Tensor,
    ) -> Tuple["SparseConvTensor", List["SparseConvTensor"]]:
        return self.encoder(x, pos_grid, batch, return_intermediates=True)

    def forward_decoder(
        self,
        x: "SparseConvTensor",
        skips: List["SparseConvTensor"],
    ) -> "SparseConvTensor":
        return self.decoder(x, skips)

    def forward_head(self, x: "SparseConvTensor", pre_logits: bool = False) -> Tensor:
        return x.features if pre_logits else self.head(x).features

    def forward(self, x: OptTensor, pos_grid: Tensor, batch: Tensor) -> Tensor:
        bottleneck, skips = self.forward_features(x, pos_grid, batch)
        sparse_x = self.forward_decoder(bottleneck, skips)
        return self.forward_head(sparse_x)


@register_model(
    "spformer-unet.scannet",
    task="base",
    # No ported pretrained weights for the standalone SPFormer U-Net yet: the released SPFormer checkpoint
    # bundles an instance-segmentation query decoder, so the backbone is registered without weights.
    weights=None,
    transform=T.Compose(
        [
            T.Normalize(keys=DataKeys.COLOR, mean=[127.5, 127.5, 127.5], std=[127.5, 127.5, 127.5]),
            T.CopyItems(keys=DataKeys.POS, names="pos_centered"),
            T.Shift(keys="pos_centered", method="centroid"),
            T.Cat(keys=[DataKeys.COLOR, "pos_centered"], dst_key=DataKeys.X, dim=1),
            T.Shift(keys=DataKeys.POS, method="min"),
            T.CopyItems(
                keys=[DataKeys.POS, DataKeys.SEGMENT],
                names=[DataKeys.ORIGIN_POS, DataKeys.ORIGIN_SEGMENT],
                allow_missing_keys=True,
            ),
            T.Voxelize(
                pos_key=DataKeys.POS,
                pos_reduce="first",
                dst_pos_grid_key=DataKeys.POS_GRID,
                keys=[DataKeys.X, DataKeys.SEGMENT, DataKeys.COLOR, DataKeys.NORMAL, "pos_centered", DataKeys.INSTANCE],
                size=0.02,
                method="fnv",
                allow_missing_keys=True,
                dst_inverse_key=DataKeys.INVERSE,
            ),
        ]
    ),
    hparams=dict(
        in_channels=6,
        num_classes=0,
        channels=[32, 64, 96, 128, 160],
        layers=2,
        stem_kernel_size=3,
        norm_kwargs=dict(eps=1e-4, momentum=0.1),
        spatial_padding=96,
    ),
)
def spformer_unet_scannet(**hparams: Any) -> SPFormerUNetSegmentation:
    return SPFormerUNetSegmentation(**hparams)


@register_model(
    "spformer-unet.scannet20",
    task="segmentation",
    weights=None,
    transform=T.Compose(
        [
            T.Normalize(keys=DataKeys.COLOR, mean=[127.5, 127.5, 127.5], std=[127.5, 127.5, 127.5]),
            T.CopyItems(keys=DataKeys.POS, names="pos_centered"),
            T.Shift(keys="pos_centered", method="centroid"),
            T.Cat(keys=[DataKeys.COLOR, "pos_centered"], dst_key=DataKeys.X, dim=1),
            T.Shift(keys=DataKeys.POS, method="min"),
            T.Relabel(keys=DataKeys.SEGMENT, labels=range(1, 21), default=-1),
            T.CopyItems(
                keys=[DataKeys.POS, DataKeys.SEGMENT],
                names=[DataKeys.ORIGIN_POS, DataKeys.ORIGIN_SEGMENT],
                allow_missing_keys=True,
            ),
            T.Voxelize(
                pos_key=DataKeys.POS,
                pos_reduce="first",
                dst_pos_grid_key=DataKeys.POS_GRID,
                keys=[DataKeys.X, DataKeys.SEGMENT, DataKeys.COLOR, DataKeys.NORMAL, "pos_centered", DataKeys.INSTANCE],
                size=0.02,
                method="fnv",
                allow_missing_keys=True,
                dst_inverse_key=DataKeys.INVERSE,
            ),
        ]
    ),
    hparams=dict(
        in_channels=6,
        num_classes=20,
        channels=[32, 64, 96, 128, 160],
        layers=2,
        stem_kernel_size=3,
        norm_kwargs=dict(eps=1e-4, momentum=0.1),
        spatial_padding=96,
    ),
)
def spformer_unet_scannet20(**hparams: Any) -> SPFormerUNetSegmentation:
    return SPFormerUNetSegmentation(**hparams)
