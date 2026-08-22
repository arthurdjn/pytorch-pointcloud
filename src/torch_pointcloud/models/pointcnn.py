"""PointCNN classification and segmentation models.

{{ paper("1801.07791") }}
"""

from typing import (
    Any,
    Callable,
    Dict,
    List,
    Literal,
    NamedTuple,
    Optional,
    Sequence,
    Tuple,
    Union,
    overload,
)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn import MLP

from torch_pointcloud.layers import FPS, PoolLike, XConv, create_pool
from torch_pointcloud.layers.act import create_act
from torch_pointcloud.utils.cluster import knn
from torch_pointcloud.utils.conversion import ensure_list, ensure_list_size, ensure_tuple, ensure_tuple_size
from torch_pointcloud.utils.types import OptTensor

from ._base import ClassificationModel, SegmentationModel
from ._registry import register_model


class PointCNNIntermediate(NamedTuple):
    """Per-stage encoder features and the point cloud they live on, consumed as decoder skips."""

    x: Tensor
    pos: Tensor
    batch: Tensor


class PointCNNEncoderBlock(nn.Module):
    """Optional FPS downsampling followed by an `XConv` over the kNN graph of the sampled points."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        spatial_dim: int,
        kernel_size: int,
        hidden_channels: Optional[int] = None,
        dilation: int = 1,
        bias: bool = True,
        act: Union[str, Callable, None] = "relu",
        act_kwargs: Optional[Dict[str, Any]] = None,
        downsample: Optional[Callable[[Tensor, Tensor], Tensor]] = None,
    ) -> None:
        super().__init__()
        self.downsample = downsample
        self.conv = XConv(
            in_channels,
            out_channels,
            spatial_dim=spatial_dim,
            kernel_size=kernel_size,
            hidden_channels=hidden_channels,
            dilation=dilation,
            bias=bias,
        )
        self.act = create_act(act, **(act_kwargs or {})) or nn.Identity()

    def forward(self, x: Tensor, pos: Tensor, batch: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        if self.downsample is not None:
            idx = self.downsample(pos, batch)
            x, pos, batch = x[idx], pos[idx], batch[idx]

        num_neighbors = self.conv.kernel_size * self.conv.dilation
        edge_index = knn(pos, pos, k=num_neighbors, batch_x=batch, batch_y=batch)
        x = self.conv(x, pos, edge_index)
        x = self.act(x)
        return x, pos, batch


class PointCNNDecoderBlock(nn.Module):
    """Upsamples features to the skip resolution with an `XConv`, then fuses them with the skip features via an MLP."""

    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        spatial_dim: int,
        kernel_size: int,
        hidden_channels: Optional[int] = None,
        dilation: int = 1,
        bias: bool = True,
        act: Union[str, Callable, None] = "relu",
        act_kwargs: Optional[Dict[str, Any]] = None,
        act_first: bool = False,
        norm: Union[str, Callable, None] = "batch_norm",
        norm_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        self.conv = XConv(
            in_channels,
            out_channels,
            spatial_dim=spatial_dim,
            kernel_size=kernel_size,
            hidden_channels=hidden_channels,
            dilation=dilation,
            bias=bias,
        )
        self.fuse = MLP(
            [out_channels + skip_channels, out_channels],
            act=act,
            act_kwargs=act_kwargs,
            act_first=act_first,
            norm=norm,
            norm_kwargs=norm_kwargs,
            plain_last=False,
        )

    def forward(
        self,
        x: Tensor,
        pos: Tensor,
        batch: Tensor,
        x_skip: Tensor,
        pos_skip: Tensor,
        batch_skip: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        num_neighbors = self.conv.kernel_size * self.conv.dilation
        edge_index = knn(pos, pos_skip, k=num_neighbors, batch_x=batch, batch_y=batch_skip)  # flow: source -> target
        x = self.conv((x, None), (pos, pos_skip), edge_index)
        x = torch.cat([x, x_skip], dim=-1)
        x = self.fuse(x)
        return x, pos_skip, batch_skip


class PointCNNEncoder(nn.Module):
    """Stack of `PointCNNEncoderBlock` units that progressively decimate the cloud with FPS.

    A stage with a ratio of `0` keeps every point and only transforms features. When
    `return_intermediates=True` is passed to `forward`, the pre-downsampling features of each
    decimating stage are returned in coarse-to-fine order for `PointCNNDecoder`.
    """

    def __init__(
        self,
        channels: Sequence[int],
        kernel_sizes: Sequence[int],
        spatial_dim: int,
        ratios: Sequence[float],
        hidden_channels: Optional[Union[int, Sequence[int]]] = None,
        dilations: Sequence[int] = (1, 1, 1, 1),
        bias: bool = True,
        act: Union[str, Callable, None] = "relu",
        act_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()
        self.channels = ensure_tuple(channels)

        depth = len(self.channels) - 1
        msg = f"Invalid parameter for {self.__class__.__name__}. Expected `{{param}}` to have length {depth}."
        self.kernel_sizes = ensure_tuple_size(kernel_sizes, size=depth, extra_msg=msg.format(param="kernel_sizes"))
        self.dilations = ensure_tuple_size(dilations, size=depth, extra_msg=msg.format(param="dilations"))
        self.ratios = ensure_tuple_size(ratios, size=depth, extra_msg=msg.format(param="ratios"))
        self.hidden_channels = ensure_tuple_size(
            hidden_channels,
            size=depth,
            extra_msg=msg.format(param="hidden_channels"),
        )

        self.blocks = nn.ModuleList()
        for i in range(depth):
            downsample: Optional[nn.Module] = None
            if self.ratios[i]:
                downsample = FPS(ratio=self.ratios[i])

            block = PointCNNEncoderBlock(
                in_channels=self.channels[i],
                hidden_channels=self.hidden_channels[i],
                out_channels=self.channels[i + 1],
                spatial_dim=spatial_dim,
                kernel_size=self.kernel_sizes[i],
                dilation=self.dilations[i],
                bias=bias,
                act=act,
                act_kwargs=act_kwargs,
                downsample=downsample,
            )
            self.blocks.append(block)

    @overload
    def forward(
        self,
        x: Tensor,
        pos: Tensor,
        batch: Tensor,
        return_intermediates: Literal[True],
    ) -> Tuple[Tensor, Tensor, Tensor, List[PointCNNIntermediate]]: ...

    @overload
    def forward(
        self,
        x: Tensor,
        pos: Tensor,
        batch: Tensor,
        return_intermediates: Literal[False] = False,
    ) -> Tuple[Tensor, Tensor, Tensor]: ...

    def forward(
        self,
        x: Tensor,
        pos: Tensor,
        batch: Tensor,
        return_intermediates: bool = False,
    ) -> Any:
        intermediates = []
        for block in self.blocks:
            if return_intermediates and hasattr(block, "downsample") and block.downsample is not None:
                intermediate = PointCNNIntermediate(x, pos, batch)
                intermediates.append(intermediate)

            x, pos, batch = block(x, pos, batch)

        if return_intermediates:
            return x, pos, batch, intermediates[::-1]
        return x, pos, batch


class PointCNNDecoder(nn.Module):
    """Stack of `PointCNNDecoderBlock` units that walk the encoder intermediates back to full resolution."""

    def __init__(
        self,
        channels: Sequence[int],
        skip_channels: Sequence[int],
        kernel_sizes: Sequence[int],
        spatial_dim: int,
        hidden_channels: Optional[Union[int, Sequence[int]]] = None,
        dilations: Union[int, Sequence[int]] = 1,
        bias: bool = True,
        act: Union[str, Callable, None] = "relu",
        act_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()
        self.channels = ensure_tuple(channels)

        depth = len(self.channels) - 1
        msg = f"Invalid parameter for {self.__class__.__name__}. Expected `{{param}}` to have length {depth}."
        self.kernel_sizes = ensure_list_size(kernel_sizes, size=depth, extra_msg=msg.format(param="kernel_sizes"))
        self.skip_channels = ensure_list_size(skip_channels, size=depth, extra_msg=msg.format(param="skip_channels"))
        self.hidden_channels = ensure_list_size(
            hidden_channels,
            size=depth,
            extra_msg=msg.format(param="hidden_channels"),
        )
        self.dilations = ensure_list_size(dilations, size=depth, extra_msg=msg.format(param="dilations"))

        self.blocks = nn.ModuleList()
        for i in range(depth):
            block = PointCNNDecoderBlock(
                in_channels=self.channels[i],
                skip_channels=self.skip_channels[i],
                hidden_channels=self.hidden_channels[i],
                out_channels=self.channels[i + 1],
                spatial_dim=spatial_dim,
                kernel_size=self.kernel_sizes[i],
                dilation=self.dilations[i],
                bias=bias,
                act=act,
                act_kwargs=act_kwargs,
            )
            self.blocks.append(block)

    def forward(
        self,
        x: Tensor,
        pos: Tensor,
        batch: Tensor,
        intermediates: List[PointCNNIntermediate],
    ) -> Tuple[Tensor, Tensor, Tensor]:
        for block, intermediate in zip(self.blocks, intermediates):
            x, pos, batch = block(x, pos, batch, *intermediate)

        return x, pos, batch


class PointCNNClassification(ClassificationModel):
    r"""
    Classification model as described in the paper
    :arxiv: ["PointCNN: Convolution On X-Transformed Points"](https://arxiv.org/abs/1801.07791)
    by Yangyan Li, Rui Bu, Mingchao Sun, Wei Wu, Xinhan Di, Baoquan Chen.

    This classification model consists of a encoder of XConv layers and FPS downsampling layers,
    and a MLP classification head.

    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        *,
        spatial_dim: int = 3,
        channels: Sequence[int],
        kernel_sizes: Sequence[int],
        ratios: Sequence[float],
        hidden_channels: Optional[Union[int, Sequence[int]]] = None,
        dilations: Sequence[int] = (1, 1, 1, 1),
        bias: bool = True,
        act: Union[str, Callable, None] = "relu",
        act_kwargs: Optional[Dict[str, Any]] = None,
        act_first: bool = False,
        norm: Union[str, Callable, None] = "batch_norm",
        norm_kwargs: Optional[Dict[str, Any]] = None,
        dropout: float = 0.0,
        head_channels: Optional[Union[int, Sequence[int]]] = None,
        global_pool: PoolLike = "max",
    ):
        super().__init__(in_channels=in_channels, num_classes=num_classes)
        self.spatial_dim = spatial_dim
        self.channels = ensure_list(channels)
        self.kernel_sizes = ensure_list(kernel_sizes)
        self.ratios = ensure_list(ratios)
        self.hidden_channels = ensure_list(hidden_channels)
        self.dilations = ensure_list(dilations)
        self.head_channels = ensure_list(head_channels, none_as_empty=True)
        self.bias = bias
        self.act = act
        self.act_kwargs = act_kwargs
        self.act_first = act_first
        self.norm = norm
        self.norm_kwargs = norm_kwargs
        self.dropout = dropout

        self.encoder = self.configure_encoder()
        self.global_pool = create_pool(global_pool)
        self.head = self.configure_head()

    @property
    def embedding_dim(self) -> int:
        """Feature dimension $C$ of the encoder output."""
        return self.channels[-1]

    def configure_encoder(self) -> nn.Module:
        """Build the `PointCNNEncoder` backbone."""
        return PointCNNEncoder(
            channels=[self.in_channels] + self.channels,
            kernel_sizes=self.kernel_sizes,
            spatial_dim=self.spatial_dim,
            ratios=self.ratios,
            hidden_channels=self.hidden_channels,
            dilations=self.dilations,
            bias=self.bias,
            act=self.act,
            act_kwargs=self.act_kwargs,
        )

    def configure_head(self) -> nn.Module:
        if self.num_classes == 0:
            return nn.Identity()
        channels_list = [self.embedding_dim] + self.head_channels + [self.num_classes]
        dropout_list = [0.0] * (len(channels_list) - 1)
        if len(channels_list) > 2:
            dropout_list[-2] = self.dropout

        return MLP(
            channels_list,
            act=self.act,
            act_kwargs=self.act_kwargs,
            act_first=self.act_first,
            norm=self.norm,
            norm_kwargs=self.norm_kwargs,
            bias=self.bias,
            dropout=dropout_list,
            plain_last=True,
        )

    def reset_classifier(self, num_classes: int, global_pool: Optional[PoolLike] = None, **kwargs: Any) -> None:
        self.num_classes = num_classes
        if global_pool is not None:
            self.global_pool = create_pool(global_pool)
        self.head = self.configure_head()

    @overload
    def forward_features(
        self,
        x: OptTensor,
        pos: Tensor,
        batch: Tensor,
        return_intermediates: Literal[True],
    ) -> Tuple[Tensor, Tensor, Tensor, List[PointCNNIntermediate]]: ...

    @overload
    def forward_features(
        self,
        x: OptTensor,
        pos: Tensor,
        batch: Tensor,
        return_intermediates: Literal[False] = False,
    ) -> Tuple[Tensor, Tensor, Tensor]: ...

    def forward_features(
        self,
        x: OptTensor,
        pos: Tensor,
        batch: Tensor,
        return_intermediates: bool = False,
    ) -> Any:
        return self.encoder(x, pos, batch, return_intermediates=return_intermediates)

    def forward_head(self, x: Tensor, batch: Tensor, pre_logits: bool = False) -> Tensor:
        x = self.global_pool(x, batch)
        if len(self.head_channels) == 0:
            x = F.dropout(x, p=self.dropout, training=self.training)
        return x if pre_logits else self.head(x)

    def forward(self, x: OptTensor, pos: Tensor, batch: Tensor) -> Tensor:
        x, _, batch = self.forward_features(x, pos, batch)
        return self.forward_head(x, batch)


class PointCNNSegmentation(SegmentationModel):
    r"""Segmentation model as described in the paper
    :arxiv: ["PointCNN: Convolution On X-Transformed Points"](https://arxiv.org/abs/1801.07791)
    by Yangyan Li, Rui Bu, Mingchao Sun, Wei Wu, Xinhan Di, Baoquan Chen.

    An encoder of XConv layers with FPS downsampling, a decoder of XConv layers upsampling back to
    the skip resolutions, and a per-point MLP head.
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        *,
        spatial_dim: int = 3,
        channels: Sequence[int],
        hidden_channels: Optional[Union[int, Sequence[int]]] = None,
        kernel_sizes: Sequence[int],
        dilations: Sequence[int] = (1, 1, 1, 1),
        ratios: Sequence[float],
        act: Union[str, Callable, None] = "relu",
        act_kwargs: Optional[Dict[str, Any]] = None,
        act_first: bool = False,
        norm: Union[str, Callable, None] = "batch_norm",
        norm_kwargs: Optional[Dict[str, Any]] = None,
        bias: bool = True,
        dropout: float = 0.0,
        head_channels: Optional[Union[int, Sequence[int]]] = None,
    ):
        super().__init__(in_channels=in_channels, num_classes=num_classes)
        self.spatial_dim = spatial_dim
        self.channels = ensure_list(channels)
        self.kernel_sizes = ensure_list(kernel_sizes)
        self.ratios = ensure_list(ratios)
        self.hidden_channels = ensure_list(hidden_channels)
        self.dilations = ensure_list(dilations)
        self.head_channels = ensure_list(head_channels, none_as_empty=True)
        self.bias = bias
        self.act = act
        self.act_kwargs = act_kwargs
        self.act_first = act_first
        self.norm = norm
        self.norm_kwargs = norm_kwargs
        self.dropout = dropout

        self.encoder = self.configure_encoder()
        self.decoder = self.configure_decoder()
        self.head = self.configure_head()

    @property
    def embedding_dim(self) -> int:
        """Feature dimension $C$ of the decoder output."""
        return self.decoder.channels[-1]

    def configure_encoder(self) -> PointCNNEncoder:
        """Build the `PointCNNEncoder` backbone."""
        return PointCNNEncoder(
            channels=[self.in_channels] + self.channels,
            kernel_sizes=self.kernel_sizes,
            spatial_dim=self.spatial_dim,
            ratios=self.ratios,
            hidden_channels=self.hidden_channels,
            dilations=self.dilations,
            bias=self.bias,
            act=self.act,
            act_kwargs=self.act_kwargs,
        )

    def configure_decoder(self) -> PointCNNDecoder:
        """Build the `PointCNNDecoder`, mirroring the decimating encoder stages in reverse."""
        channels = []
        skip_channels = []
        kernel_sizes = []
        hidden_channels = []
        dilations = []

        for i in range(len(self.channels)):
            if not self.ratios[i]:
                continue

            channels.append(self.channels[i])
            skip_channels.append(self.channels[i - 1] if i > 0 else self.in_channels)
            kernel_sizes.append(self.kernel_sizes[i])
            hidden_channels.append(self.hidden_channels[i])
            dilations.append(self.dilations[i])

        return PointCNNDecoder(
            channels=[channels[-1]] + channels[::-1],
            skip_channels=skip_channels[::-1],
            kernel_sizes=kernel_sizes[::-1],
            spatial_dim=self.spatial_dim,
            hidden_channels=hidden_channels[::-1],
            dilations=dilations[::-1],
            bias=self.bias,
            act=self.act,
            act_kwargs=self.act_kwargs,
        )

    def configure_head(self) -> nn.Module:
        if self.num_classes == 0:
            return nn.Identity()
        channels_list = [self.embedding_dim] + self.head_channels + [self.num_classes]
        dropout_list = [0.0] * (len(channels_list) - 1)
        if len(channels_list) > 2:
            dropout_list[-2] = self.dropout

        return MLP(
            channels_list,
            act=self.act,
            act_kwargs=self.act_kwargs,
            act_first=self.act_first,
            norm=self.norm,
            norm_kwargs=self.norm_kwargs,
            bias=self.bias,
            dropout=dropout_list,
            plain_last=True,
        )

    def reset_classifier(self, num_classes: int, **kwargs: Any) -> None:
        self.num_classes = num_classes
        self.head = self.configure_head()

    @overload
    def forward_features(
        self,
        x: OptTensor,
        pos: Tensor,
        batch: Tensor,
        return_intermediates: Literal[True],
    ) -> Tuple[Tensor, Tensor, Tensor, List[PointCNNIntermediate]]: ...

    @overload
    def forward_features(
        self,
        x: OptTensor,
        pos: Tensor,
        batch: Tensor,
        return_intermediates: Literal[False] = False,
    ) -> Tuple[Tensor, Tensor, Tensor]: ...

    def forward_features(
        self,
        x: OptTensor,
        pos: Tensor,
        batch: Tensor,
        return_intermediates: bool = False,
    ) -> Any:
        return self.encoder(x, pos, batch, return_intermediates=return_intermediates)

    def forward_decoder(
        self,
        x: Tensor,
        pos: Tensor,
        batch: Tensor,
        intermediates: List[PointCNNIntermediate],
    ) -> Tuple[Tensor, Tensor, Tensor]:
        return self.decoder(x, pos, batch, intermediates)

    def forward_head(self, x: Tensor, pre_logits: bool = False) -> Tensor:
        if len(self.head_channels) == 0:
            x = F.dropout(x, p=self.dropout, training=self.training)
        return x if pre_logits else self.head(x)

    def forward(self, x: OptTensor, pos: Tensor, batch: Tensor) -> Tensor:
        x, pos, batch, intermediates = self.forward_features(x, pos, batch, return_intermediates=True)
        x, _, _ = self.forward_decoder(x, pos, batch, intermediates)
        return self.forward_head(x)


@register_model(
    "pointcnn-base",
    task="classification",
    hparams=dict(
        spatial_dim=3,
        channels=[48, 96, 192, 384],
        hidden_channels=[32, 64, 128, 256],
        kernel_sizes=[8, 12, 16, 16],
        dilations=[1, 2, 2, 2],
        ratios=[0.0, 0.375, 0.334, 0.0],
        act="relu",
        act_first=False,
        norm="batch_norm",
        bias=True,
        dropout=0.5,
        head_channels=[256, 128],
        global_pool="mean",
    ),
)
def pointcnn_base_cls(**hparams: Any) -> PointCNNClassification:
    return PointCNNClassification(**hparams)


@register_model(
    "pointcnn-base",
    task="segmentation",
    hparams=dict(
        spatial_dim=3,
        channels=[48, 96, 192, 384],
        hidden_channels=[32, 64, 128, 256],
        kernel_sizes=[8, 12, 16, 16],
        dilations=[1, 2, 2, 2],
        ratios=[0.0, 0.375, 0.5, 0.334],
        act="relu",
        act_first=False,
        norm="batch_norm",
        bias=True,
        dropout=0.5,
        head_channels=[256, 128],
    ),
)
def pointcnn_base_seg(**hparams: Any) -> PointCNNSegmentation:
    return PointCNNSegmentation(**hparams)
