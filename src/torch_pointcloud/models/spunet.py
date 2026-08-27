"""SpUNet segmentation model.

{{ paper("1904.08755") }}
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
from torch_pointcloud.datasets.scannet import SCANNET20_CLASSES
from torch_pointcloud.layers import SparseModule
from torch_pointcloud.layers.act import create_act
from torch_pointcloud.layers.norms import create_norm
from torch_pointcloud.models._base import SegmentationModel
from torch_pointcloud.models._registry import WeightsDict, register_model
from torch_pointcloud.utils.conversion import convert_to_spconv_tensor
from torch_pointcloud.utils.data import DataKeys
from torch_pointcloud.utils.imports import _SPCONV_GITHUB_URL, optional_import
from torch_pointcloud.utils.types import OptTensor

if TYPE_CHECKING:
    import spconv.pytorch as spconv
    from spconv.pytorch import SparseConvTensor


spconv, _ = optional_import("spconv.pytorch", url=_SPCONV_GITHUB_URL)
SparseConvTensor, _ = optional_import("spconv.pytorch", "SparseConvTensor", url=_SPCONV_GITHUB_URL)


class SparseBasicBlock(SparseModule):
    """Residual block of two submanifold sparse convolutions.

    A pointwise convolution projects the skip connection when the channel count changes.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        bias: bool = False,
        indice_key: Optional[str] = None,
        act: Union[str, Callable, None] = "relu",
        act_kwargs: Optional[Dict[str, Any]] = None,
        norm: Union[str, Callable, None] = "batch_norm",
        norm_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()
        act_kwargs = act_kwargs or {}
        norm_kwargs = norm_kwargs or {}
        padding = kernel_size // 2

        if in_channels == out_channels:
            self.proj: nn.Module = spconv.SparseSequential(nn.Identity())
        else:
            self.proj = spconv.SparseSequential(
                spconv.SubMConv3d(in_channels, out_channels, kernel_size=1, bias=False),
                create_norm(norm, out_channels, **norm_kwargs) or nn.Identity(),
            )

        self.conv1 = spconv.SubMConv3d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=bias,
            indice_key=indice_key,
        )
        self.norm1 = create_norm(norm, out_channels, **norm_kwargs) or nn.Identity()
        self.conv2 = spconv.SubMConv3d(
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=bias,
            indice_key=indice_key,
        )
        self.norm2 = create_norm(norm, out_channels, **norm_kwargs) or nn.Identity()
        self.act = create_act(act, **act_kwargs) or nn.Identity()

    def forward(self, x: "SparseConvTensor") -> "SparseConvTensor":
        residual = x
        out = self.conv1(x)
        out = out.replace_feature(self.act(self.norm1(out.features)))
        out = self.conv2(out)
        out = out.replace_feature(self.norm2(out.features))
        out = out.replace_feature(self.act(out.features + self.proj(residual).features))
        return out


def _init_spunet_weights(m: nn.Module) -> None:
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, spconv.SubMConv3d):
        nn.init.trunc_normal_(m.weight, std=0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm1d):
        nn.init.constant_(m.bias, 0)
        nn.init.constant_(m.weight, 1.0)


def _make_block_seq(
    in_channels: int,
    out_channels: int,
    depth: int,
    indice_key: str,
    *,
    first_in_channels: Optional[int] = None,
    kernel_size: int = 3,
    act: Union[str, Callable, None] = "relu",
    act_kwargs: Optional[Dict[str, Any]] = None,
    norm: Union[str, Callable, None] = "batch_norm",
    norm_kwargs: Optional[Dict[str, Any]] = None,
) -> "spconv.SparseSequential":
    blocks = OrderedDict()
    for i in range(depth):
        if i == 0:
            block_in = first_in_channels if first_in_channels is not None else in_channels
        else:
            block_in = out_channels
        blocks[f"block{i}"] = SparseBasicBlock(
            block_in,
            out_channels,
            kernel_size=kernel_size,
            indice_key=indice_key,
            act=act,
            act_kwargs=act_kwargs,
            norm=norm,
            norm_kwargs=norm_kwargs,
        )
    return spconv.SparseSequential(blocks)


class SparseUNetEncoder(nn.Module):
    """Sparse convolutional stem followed by stages that halve the resolution and run `SparseBasicBlock` blocks.

    The stem output and every stage output but the last are returned as decoder skips.
    """

    def __init__(
        self,
        in_channels: int,
        base_channels: int = 32,
        channels: Sequence[int] = (32, 64, 128, 256),
        layers: Sequence[int] = (2, 3, 4, 6),
        stem_kernel_size: int = 5,
        kernel_size: int = 3,
        spatial_padding: int = 96,
        act: Union[str, Callable, None] = "relu",
        act_kwargs: Optional[Dict[str, Any]] = None,
        norm: Union[str, Callable, None] = "batch_norm",
        norm_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()
        if len(layers) != len(channels):
            raise ValueError(
                f"`layers` and `channels` must have the same length, got {len(layers)} and {len(channels)}."
            )
        self.in_channels = in_channels
        self.base_channels = base_channels
        self.channels = tuple(channels)
        self.layers = tuple(layers)
        self.num_stages = len(self.layers)
        self.spatial_padding = spatial_padding

        norm_kwargs = norm_kwargs or {}
        act_kwargs = act_kwargs or {}

        self.conv_input = spconv.SparseSequential(
            spconv.SubMConv3d(
                in_channels,
                base_channels,
                kernel_size=stem_kernel_size,
                padding=stem_kernel_size // 2,
                bias=False,
                indice_key="stem",
            ),
            create_norm(norm, base_channels, **norm_kwargs) or nn.Identity(),
            create_act(act, **act_kwargs) or nn.Identity(),
        )

        self.down = nn.ModuleList()
        self.enc = nn.ModuleList()

        enc_channels = base_channels
        for s in range(self.num_stages):
            self.down.append(
                spconv.SparseSequential(
                    spconv.SparseConv3d(
                        enc_channels,
                        self.channels[s],
                        kernel_size=2,
                        stride=2,
                        bias=False,
                        indice_key=f"spconv{s + 1}",
                    ),
                    create_norm(norm, self.channels[s], **norm_kwargs) or nn.Identity(),
                    create_act(act, **act_kwargs) or nn.Identity(),
                )
            )
            self.enc.append(
                _make_block_seq(
                    in_channels=self.channels[s],
                    out_channels=self.channels[s],
                    depth=self.layers[s],
                    indice_key=f"subm{s + 1}",
                    kernel_size=kernel_size,
                    act=act,
                    act_kwargs=act_kwargs,
                    norm=norm,
                    norm_kwargs=norm_kwargs,
                )
            )
            enc_channels = self.channels[s]

        self.apply(_init_spunet_weights)

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
        sparse_x = self.conv_input(sparse_x)

        skips: List["SparseConvTensor"] = [sparse_x]
        for s in range(self.num_stages):
            sparse_x = self.down[s](sparse_x)
            sparse_x = self.enc[s](sparse_x)
            if s < self.num_stages - 1:
                skips.append(sparse_x)

        if return_intermediates:
            return sparse_x, skips
        return sparse_x


class SparseUNetDecoder(nn.Module):
    """Mirror of `SparseUNetEncoder`: each stage inverse-convolves, concatenates its skip, and runs residual blocks.

    Stages are built shallowest-first but run deepest-first, so `channels` and `layers` are read back-to-front.
    """

    def __init__(
        self,
        in_channels: int,
        skip_channels: Sequence[int],
        channels: Sequence[int],
        layers: Sequence[int],
        kernel_size: int = 3,
        act: Union[str, Callable, None] = "relu",
        act_kwargs: Optional[Dict[str, Any]] = None,
        norm: Union[str, Callable, None] = "batch_norm",
        norm_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()
        if len(layers) != len(channels):
            raise ValueError(
                f"`layers` and `channels` must have the same length, got {len(layers)} and {len(channels)}."
            )
        if len(skip_channels) != len(channels):
            raise ValueError(
                f"`skip_channels` and `channels` must have the same length, "
                f"got {len(skip_channels)} and {len(channels)}."
            )
        self.in_channels = in_channels
        self.skip_channels = tuple(skip_channels)
        self.channels = tuple(channels)
        self.layers = tuple(layers)
        self.num_stages = len(self.layers)

        norm_kwargs = norm_kwargs or {}
        act_kwargs = act_kwargs or {}

        self.up = nn.ModuleList()
        self.dec = nn.ModuleList()

        # `channels` is given in BUILD order, matching the reference `full_channels[num_stages..]`
        # slice convention. With this convention the build-time `s`-th decoder stage outputs
        # `channels[num_stages - 1 - s]` (i.e., `channels` is read back-to-front during build).
        # At RUNTIME the stages are iterated reversed, so `dec[num_stages-1]` (deepest) runs first.
        for s in range(self.num_stages):
            dec_out = self.channels[self.num_stages - 1 - s]
            # `up[s]` consumes the output of the build-time-PREVIOUS stage:
            # - for s < num_stages - 1, that's the previous stage's `dec_out`,
            #   i.e. `channels[num_stages - s - 2]`;
            # - for s == num_stages - 1 (deepest stage built last, runs first), it's
            #   `in_channels` (the encoder bottleneck).
            up_in = self.channels[self.num_stages - s - 2] if s < self.num_stages - 1 else in_channels

            self.up.append(
                spconv.SparseSequential(
                    spconv.SparseInverseConv3d(
                        up_in,
                        dec_out,
                        kernel_size=2,
                        bias=False,
                        indice_key=f"spconv{s + 1}",
                    ),
                    create_norm(norm, dec_out, **norm_kwargs) or nn.Identity(),
                    create_act(act, **act_kwargs) or nn.Identity(),
                )
            )
            self.dec.append(
                _make_block_seq(
                    in_channels=dec_out,
                    out_channels=dec_out,
                    depth=self.layers[self.num_stages - 1 - s],
                    indice_key=f"subm{s}",
                    first_in_channels=dec_out + self.skip_channels[s],
                    kernel_size=kernel_size,
                    act=act,
                    act_kwargs=act_kwargs,
                    norm=norm,
                    norm_kwargs=norm_kwargs,
                )
            )

        self.apply(_init_spunet_weights)

    def forward(
        self,
        x: "SparseConvTensor",
        skips: List["SparseConvTensor"],
    ) -> "SparseConvTensor":
        skips = list(skips)
        for s in reversed(range(self.num_stages)):
            x = self.up[s](x)
            skip = skips.pop()
            x = x.replace_feature(torch.cat([x.features, skip.features], dim=1))
            x = self.dec[s](x)
        return x


class SparseUNetSegmentation(SegmentationModel):
    r"""SpUNet segmentation model, a sparse residual U-Net in the spirit of
    :arxiv: [4D Spatio-Temporal ConvNets: Minkowski Convolutional Neural Networks](https://arxiv.org/abs/1904.08755)
    by Christopher Choy, JunYoung Gwak, Silvio Savarese.

    Voxelized coordinates enter a sparse convolutional stem, then a symmetric encoder-decoder of
    submanifold residual blocks with skip connections. The head runs on the full-resolution decoder
    output, whose rows stay aligned with the input points.

    Args:
        in_channels: Number of input feature channels. Positions are used as features when `x` is `None`.
        num_classes: Number of output classes. $0$ replaces the head with `nn.Identity`.
        base_channels: Number of channels of the stem, also the width of the shallowest skip.
        channels: Feature width of every stage, encoder stages first then decoder stages. Must have even length.
        layers: Number of residual blocks per stage, aligned with `channels`.
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
        num_classes: int,
        *,
        base_channels: int = 32,
        channels: Sequence[int] = (32, 64, 128, 256, 256, 128, 96, 96),
        layers: Sequence[int] = (2, 3, 4, 6, 2, 2, 2, 2),
        stem_kernel_size: int = 5,
        kernel_size: int = 3,
        spatial_padding: int = 96,
        act: Union[str, Callable, None] = "relu",
        act_kwargs: Optional[Dict[str, Any]] = None,
        norm: Union[str, Callable, None] = "batch_norm",
        norm_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(in_channels=in_channels, num_classes=num_classes)
        if len(layers) != len(channels):
            raise ValueError(
                f"`layers` and `channels` must have the same length, got {len(layers)} and {len(channels)}."
            )
        if len(layers) % 2 != 0:
            raise ValueError(f"`layers` must have an even length, got {len(layers)}.")

        self.base_channels = base_channels
        self.channels = tuple(channels)
        self.layers = tuple(layers)
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
        self.apply(_init_spunet_weights)

    def configure_encoder(self) -> SparseUNetEncoder:
        """Build the sparse encoder producing the bottleneck features and the per-stage skips."""
        num_stages = len(self.layers) // 2
        return SparseUNetEncoder(
            in_channels=self.in_channels,
            base_channels=self.base_channels,
            channels=self.channels[:num_stages],
            layers=self.layers[:num_stages],
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
        num_stages = len(self.layers) // 2
        encoder_channels = self.channels[:num_stages]
        # Skip channels in INSERTION order (stem first, then per-encoder-stage outputs).
        # The bottleneck (last encoder stage) is consumed as `in_channels`, not a skip.
        skip_channels = [self.base_channels, *encoder_channels[:-1]]
        return SparseUNetDecoder(
            in_channels=encoder_channels[-1],
            skip_channels=skip_channels,
            channels=self.channels[num_stages:],
            layers=self.layers[num_stages:],
            kernel_size=self.kernel_size,
            act=self.act,
            act_kwargs=self.act_kwargs,
            norm=self.norm,
            norm_kwargs=self.norm_kwargs,
        )

    def configure_head(self) -> nn.Module:
        if self.num_classes <= 0:
            return nn.Identity()
        return spconv.SubMConv3d(
            self.channels[-1],
            self.num_classes,
            kernel_size=1,
            padding=1,
            bias=True,
        )

    def reset_classifier(self, num_classes: int) -> None:
        self.num_classes = num_classes
        self.head = self.configure_head()
        self.head.apply(_init_spunet_weights)

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
        if pre_logits:
            return x.features
        out = self.head(x)
        return out.features if hasattr(out, "features") else out

    def forward(self, x: OptTensor, pos_grid: Tensor, batch: Tensor) -> Tensor:
        sparse_x, skips = self.forward_features(x, pos_grid, batch)
        sparse_x = self.forward_decoder(sparse_x, skips)
        return self.forward_head(sparse_x)


@register_model(
    "spunet-v1m1.scannet20.pointcept",
    task="segmentation",
    weights=WeightsDict(
        url="hf://torch-pointcloud/spunet/spunet-v1m1.scannet20.pointcept.safetensors",
        dataset="scannet20",
        metrics={"mIoU": 70.02},
        classes=SCANNET20_CLASSES,
        author="pointcept",
        license="MIT",
    ),
    transform=T.Compose(
        [
            T.Shift(keys=DataKeys.POS, method="bbox", axes=[0, 1]),  # XY: bbox midrange
            T.Shift(keys=DataKeys.POS, method="min", axes=[2]),  # Z: min
            T.Divide(keys=DataKeys.COLOR, divisor=255),
            T.Cat(keys=[DataKeys.COLOR, DataKeys.NORMAL], dst_key=DataKeys.X, dim=1),
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
                keys=[DataKeys.X, DataKeys.SEGMENT, DataKeys.COLOR, DataKeys.NORMAL, DataKeys.INSTANCE],
                reduce="first",
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
        base_channels=32,
        channels=(32, 64, 128, 256, 256, 128, 96, 96),
        layers=(2, 3, 4, 6, 2, 2, 2, 2),
        stem_kernel_size=5,
        kernel_size=3,
        norm_kwargs={"eps": 1e-3, "momentum": 0.01},
    ),
)
def spunet_v1m1_scannet20(**hparams: Any) -> SparseUNetSegmentation:
    return SparseUNetSegmentation(**hparams)
