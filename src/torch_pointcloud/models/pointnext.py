"""PointNeXt classification and segmentation models.

{{ paper("2206.04670") }}
"""

from typing import Any, Callable, Dict, List, Literal, NamedTuple, Optional, Sequence, Tuple, Union, overload

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn import MLP, global_max_pool, global_mean_pool

import torch_pointcloud.transforms as T
from torch_pointcloud.datasets.modelnet import MODELNET40_CLASSES
from torch_pointcloud.datasets.s3dis import S3DIS_CLASSES
from torch_pointcloud.datasets.scanobjectnn import SCANOBJECTNN_CLASSES
from torch_pointcloud.layers import PoolLike, create_pool
from torch_pointcloud.layers.pointnet2_blocks import PointNet2FeaturePropagation
from torch_pointcloud.layers.pointnext_blocks import PointNeXtResidualBlock, PointNeXtSetAbstraction
from torch_pointcloud.utils.conversion import ensure_list, ensure_tuple, ensure_tuple_size
from torch_pointcloud.utils.data import DataKeys
from torch_pointcloud.utils.types import AggrType, OptTensor

from ._base import ClassificationModel, SegmentationModel
from ._registry import WeightsDict, register_model


class PointNeXtIntermediate(NamedTuple):
    """Input features and point cloud of one encoder block, recorded before it downsamples."""

    x: Tensor
    pos: Tensor
    batch: Tensor


class PointNeXtEncoderBlock(nn.Module):
    """One encoder stage: an optional set-abstraction downsampling followed by `depth` inverted residual blocks."""

    def __init__(
        self,
        spatial_dim: int,
        channels: int,
        depth: int,
        expansion: int,
        radius: float,
        num_neighbors: int,
        act: Union[str, Callable, None] = "relu",
        act_kwargs: Optional[Dict[str, Any]] = None,
        act_first: bool = False,
        norm: Union[str, Callable, None] = "batch_norm",
        norm_kwargs: Optional[Dict[str, Any]] = None,
        bias: Union[bool, List[bool]] = True,
        add_self_loops: bool = False,
        aggr: AggrType = "max",
        downsample: Optional[PointNeXtSetAbstraction] = None,
    ):
        super().__init__()
        act_kwargs = act_kwargs or {}
        norm_kwargs = norm_kwargs or {}

        self.downsample = downsample
        self.layers = nn.ModuleList()
        for _ in range(depth):
            layer = PointNeXtResidualBlock(
                spatial_dim=spatial_dim,
                channels=channels,
                expansion=expansion,
                radius=radius,
                num_neighbors=num_neighbors,
                act=act,
                act_kwargs=act_kwargs,
                act_first=act_first,
                norm=norm,
                norm_kwargs=norm_kwargs,
                bias=bias,
                add_self_loops=add_self_loops,
                aggr=aggr,
            )
            self.layers.append(layer)

    def forward(self, x: Tensor, pos: Tensor, batch: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        if self.downsample is not None:
            x, pos, batch = self.downsample(x, pos, batch)

        for layer in self.layers:
            x = layer(x, pos, batch)

        return x, pos, batch


class PointNeXtEncoder(nn.Module):
    r"""Stack of `PointNeXtEncoderBlock` stages, each preceded by a set-abstraction downsampling.

    When `return_intermediates=True` is passed to `forward`, the pre-downsampling features of every
    stage are returned in coarse-to-fine order, ready to be consumed as decoder skips.

    Note:
        `radiuses` and `num_neighbors` hold one entry per channel: entry $i$ configures the
        set-abstraction of block $i$ and entry $i+1$ its residual blocks.

    Args:
        channels: Channels of the stem output followed by the output of each encoder block.
        spatial_dim: Spatial dimension of the input point cloud.
        depths: Number of residual blocks in each encoder block.
        expansion: Bottleneck expansion factor of the residual blocks.
        ratios: Sampling ratio of the set-abstraction preceding each encoder block.
        radiuses: Ball-query radius for each channel level.
        num_neighbors: Maximum number of neighbors for each channel level.
        sa_layers: Number of MLP layers inside each set-abstraction.
        sa_use_res: Whether the set-abstractions use a residual connection.
        act: Activation function to use for the encoder blocks.
        act_kwargs: Keyword arguments for the activation function.
        act_first: Whether to apply the activation function before the normalization.
        norm: Normalization function to use for the encoder blocks.
        norm_kwargs: Keyword arguments for the normalization function.
        bias: Whether to use a bias for the encoder blocks.
        add_self_loops: Whether the neighborhood graphs include self-loops.
        aggr: Aggregation used to pool neighbor features.
    """

    def __init__(
        self,
        channels: Sequence[int],
        *,
        spatial_dim: int = 3,
        depths: Sequence[int],
        expansion: Union[int, Sequence[int]] = 4,
        ratios: Sequence[float],
        radiuses: Sequence[Union[float, Sequence[float]]],
        num_neighbors: Sequence[Union[int, Sequence[int]]],
        sa_layers: int = 1,
        sa_use_res: bool = True,
        act: Union[str, Callable, None] = "relu",
        act_kwargs: Optional[Dict[str, Any]] = None,
        act_first: bool = False,
        norm: Union[str, Callable, None] = "batch_norm",
        norm_kwargs: Optional[Dict[str, Any]] = None,
        bias: Union[bool, List[bool]] = True,
        add_self_loops: bool = False,
        aggr: AggrType = "max",
    ) -> None:
        super().__init__()
        self.channels = ensure_tuple(channels)

        size = len(self.channels) - 1
        extra_msg = (
            f"Invalid {self.__class__.__name__} parameter: "
            f"expected `{{param}}` to have the same length as the number of blocks ({size})."
        )
        self.depths = ensure_tuple_size(depths, size, extra_msg=extra_msg.format(param="depths"))
        self.expansion = ensure_tuple_size(expansion, size, extra_msg=extra_msg.format(param="expansion"))
        self.ratios = ensure_tuple_size(ratios, size, extra_msg=extra_msg.format(param="ratios"))

        extra_msg = (
            f"Invalid {self.__class__.__name__} parameter: "
            f"expected `{{param}}` to have the same length as the number of channels ({size + 1})."
        )
        self.radiuses = ensure_tuple_size(radiuses, size + 1, extra_msg=extra_msg.format(param="radiuses"))
        self.num_neighbors = ensure_tuple_size(
            num_neighbors,
            size + 1,
            extra_msg=extra_msg.format(param="num_neighbors"),
        )

        self.blocks = nn.ModuleList()
        for i in range(size):
            out_ch = channels[i + 1]
            if sa_layers >= 2:
                mid_ch = out_ch // 2 if out_ch != channels[i] else out_ch
                sa_channels: List[int] = [mid_ch] * (sa_layers - 1) + [out_ch]
            else:
                sa_channels = [out_ch]

            downsample = PointNeXtSetAbstraction(
                spatial_dim=spatial_dim,
                in_channels=channels[i],
                channels=sa_channels,
                ratio=self.ratios[i],
                radius=self.radiuses[i],
                num_neighbors=self.num_neighbors[i],
                use_res=sa_use_res,
                act=act,
                act_kwargs=act_kwargs,
                act_first=act_first,
                norm=norm,
                norm_kwargs=norm_kwargs,
                bias=bias,
                add_self_loops=add_self_loops,
                aggr=aggr,
            )
            block = PointNeXtEncoderBlock(
                spatial_dim=spatial_dim,
                channels=channels[i + 1],
                depth=self.depths[i],
                expansion=self.expansion[i],
                radius=self.radiuses[i + 1],
                num_neighbors=self.num_neighbors[i + 1],
                act=act,
                act_kwargs=act_kwargs,
                act_first=act_first,
                norm=norm,
                norm_kwargs=norm_kwargs,
                bias=bias,
                add_self_loops=add_self_loops,
                aggr=aggr,
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
    ) -> Tuple[Tensor, Tensor, Tensor, List[PointNeXtIntermediate]]: ...

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
            if return_intermediates:
                intermediate = PointNeXtIntermediate(x, pos, batch)
                intermediates.append(intermediate)

            x, pos, batch = block(x, pos, batch)

        if return_intermediates:
            return x, pos, batch, intermediates[::-1]
        return x, pos, batch


class PointNeXtDecoder(nn.Module):
    r"""The PointNeXt decoder, using the Feature Propagation (FP) module from the PointNet++ architecture.

    Note:
        The number of channels should be equal to the number of decoder blocks + 1.

    Args:
        channels: List of channels for each FP block.
            The first element should correspond to the last channel of the encoder.
        skip_channels: List of channels for the skip connections.
            This is usually the same as the first $N-1$ channels of the encoder, in reverse order.
        depths: List of depths for each FP block.
        spatial_dim: Spatial dimension of the input point cloud.
        dropout: Dropout rate before the classification head.
        act: Activation function to use for the FP blocks.
        act_kwargs: Keyword arguments for the activation function.
        act_first: Whether to apply the activation function before the normalization.
        norm: Normalization function to use for the FP blocks.
        norm_kwargs: Keyword arguments for the normalization function.
        bias: Whether to use a bias for the FP blocks.
    """

    def __init__(
        self,
        channels: Sequence[int],
        skip_channels: Sequence[int],
        depths: Sequence[int],
        *,
        spatial_dim: int = 3,
        dropout: float = 0.0,
        act: Union[str, Callable, None] = "relu",
        act_kwargs: Optional[Dict[str, Any]] = None,
        act_first: bool = False,
        norm: Union[str, Callable, None] = "batch_norm",
        norm_kwargs: Optional[Dict[str, Any]] = None,
        bias: Union[bool, List[bool]] = True,
        plain_last: bool = True,
    ):
        super().__init__()
        self.channels = ensure_tuple(channels)
        size = len(channels) - 1

        extra_msg = (
            f"Invalid {self.__class__.__name__} parameter: "
            f"expected `{{param}}` to have the same length as the number of blocks ({size})."
        )
        self.depths = ensure_tuple_size(depths, size, extra_msg=extra_msg.format(param="depths"))
        self.skip_channels = ensure_tuple_size(skip_channels, size, extra_msg=extra_msg.format(param="skip_channels"))

        self.blocks = nn.ModuleList()
        for i in range(size):
            in_channels = self.channels[i] + self.skip_channels[i]
            block = PointNet2FeaturePropagation(
                channels=[in_channels] + [self.channels[i + 1]] * self.depths[i],
                k=spatial_dim,
                act=act,
                act_kwargs=act_kwargs,
                act_first=act_first,
                norm=norm,
                norm_kwargs=norm_kwargs,
                bias=bias,
                dropout=dropout,
                plain_last=plain_last,
            )
            self.blocks.append(block)

    def forward(
        self,
        x: Tensor,
        pos: Tensor,
        batch: Tensor,
        intermediates: List[PointNeXtIntermediate],
    ) -> Tuple[Tensor, Tensor, Tensor]:
        for block, intermediate in zip(self.blocks, intermediates):
            x, pos, batch = block(x, pos, batch, *intermediate)
        return x, pos, batch


class PointNeXtPartDecoder(nn.Module):
    r"""PointNeXt decoder for part segmentation (ShapeNetPart).

    Uses the same FP block layout as `PointNeXtDecoder`, with two
    global feature convolutions and shape-category conditioning. At the
    shallowest decoder stage, the skip features are augmented with max-pooled
    global features from two encoder levels and a shape-category one-hot
    vector before the FP layer.

    This matches the reference `PointNextPartDecoder` with
    `cls_map='curvenet'`.

    Args:
        channels: List of channels for each FP block.
        skip_channels: List of channels for the skip connections.
        depths: List of depths for each FP block.
        global_conv1_in: Input channels for global_conv1 (typically
            `encoder_channels[-2]`).
        global_conv2_in: Input channels for global_conv2 (typically
            `encoder_channels[-1]`).
        num_categories: Number of shape categories (16 for ShapeNetPart).
    """

    def __init__(
        self,
        channels: Sequence[int],
        skip_channels: Sequence[int],
        depths: Sequence[int],
        *,
        global_conv1_in: int,
        global_conv2_in: int,
        num_categories: int = 16,
        spatial_dim: int = 3,
        dropout: float = 0.0,
        act: Union[str, Callable, None] = "relu",
        act_kwargs: Optional[Dict[str, Any]] = None,
        act_first: bool = False,
        norm: Union[str, Callable, None] = "batch_norm",
        norm_kwargs: Optional[Dict[str, Any]] = None,
        bias: Union[bool, List[bool]] = True,
        plain_last: bool = True,
    ):
        super().__init__()
        self.channels = ensure_tuple(channels)
        size = len(channels) - 1

        extra_msg = (
            f"Invalid {self.__class__.__name__} parameter: "
            f"expected `{{param}}` to have the same length as the number of blocks ({size})."
        )
        self.depths = ensure_tuple_size(depths, size, extra_msg=extra_msg.format(param="depths"))
        skip_channels = list(ensure_tuple_size(skip_channels, size, extra_msg=extra_msg.format(param="skip_channels")))
        skip_channels[-1] += 64 + 128 + num_categories
        self.skip_channels = tuple(skip_channels)
        self.num_categories = num_categories

        self.blocks = nn.ModuleList()
        for i in range(size):
            in_channels = self.channels[i] + self.skip_channels[i]
            block = PointNet2FeaturePropagation(
                channels=[in_channels] + [self.channels[i + 1]] * self.depths[i],
                k=spatial_dim,
                act=act,
                act_kwargs=act_kwargs,
                act_first=act_first,
                norm=norm,
                norm_kwargs=norm_kwargs,
                bias=bias,
                dropout=dropout,
                plain_last=plain_last,
            )
            self.blocks.append(block)

        self.global_conv1 = nn.Sequential(nn.Linear(global_conv1_in, 64), nn.ReLU(inplace=True))
        self.global_conv2 = nn.Sequential(nn.Linear(global_conv2_in, 128), nn.ReLU(inplace=True))

    def forward(
        self,
        x: Tensor,
        pos: Tensor,
        batch: Tensor,
        category: Tensor,
        intermediates: List[PointNeXtIntermediate],
    ) -> Tuple[Tensor, Tensor, Tensor]:
        # Global features from the bottleneck and deepest skip
        # (computed BEFORE any decoder blocks modify x, matching the reference convention).
        # intermediates are ordered deep-to-shallow: [0]=deepest, [-1]=shallowest
        x_deep_skip = intermediates[0].x  # encoder_channels[-2] channels
        b_deep_skip = intermediates[0].batch

        emb1 = self.global_conv1(x_deep_skip)  # (N_deep, 64)
        emb1 = global_max_pool(emb1, b_deep_skip)  # (B, 64)

        emb2 = self.global_conv2(x)  # bottleneck, (N_bot, 128)
        emb2 = global_max_pool(emb2, batch)  # (B, 128)

        # Run all decoder blocks except the shallowest (last in the list)
        for block, intermediate in zip(self.blocks[:-1], intermediates[:-1]):
            x, pos, batch = block(x, pos, batch, *intermediate)

        # Expand global features to match the shallowest skip resolution
        skip_x, skip_pos, skip_batch = intermediates[-1]

        # Scatter-expand: (B, C) -> (N, C) using skip_batch
        emb1_exp = emb1[skip_batch]  # (N, 64)
        emb2_exp = emb2[skip_batch]  # (N, 128)
        cls_exp = category[skip_batch]  # (N, num_categories)

        aug_skip_x = torch.cat([skip_x, emb1_exp, emb2_exp, cls_exp], dim=1)
        aug_intermediate = PointNeXtIntermediate(aug_skip_x, skip_pos, skip_batch)

        # Run the shallowest FP block
        x, pos, batch = self.blocks[-1](x, pos, batch, *aug_intermediate)
        return x, pos, batch


class PointNeXtPartSegmentation(SegmentationModel):
    r"""PointNeXt part segmentation model for ShapeNetPart.

    Uses the same encoder as `PointNeXtSegmentation` but replaces the
    decoder with `PointNeXtPartDecoder` (global feature conditioning
    on shape category) and adds a head with global max+avg pooling.

    Args:
        in_channels: Number of input feature channels.
        num_classes: Number of part classes (50 for ShapeNetPart).
        num_categories: Number of shape categories (16 for ShapeNetPart).
        stem_channels: Stem MLP channel sizes.
        encoder_channels: Encoder channel dimensions per stage.
        encoder_depths: Residual block depths per encoder stage.
        encoder_expansion: InvResMLP expansion ratio.
        sa_layers: Number of SA conv layers per block.
        sa_use_res: Whether SA blocks use residual connections.
        decoder_channels: Decoder channel dimensions per stage.
        decoder_depths: FP block depths per decoder stage.
        ratios: FPS downsampling ratios.
        radiuses: Ball-query radii.
        num_neighbors: Max neighbors per ball query.
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        *,
        num_categories: int,
        stem_channels: Optional[Union[int, Sequence[int]]] = None,
        stem_plain_last: bool = False,
        encoder_channels: Sequence[int],
        encoder_depths: Sequence[int],
        encoder_expansion: Union[int, Sequence[int]] = 4,
        sa_layers: int = 1,
        sa_use_res: bool = True,
        decoder_channels: Sequence[int],
        decoder_depths: Sequence[int],
        decoder_plain_last: bool = True,
        ratios: Sequence[float],
        radiuses: Sequence[Union[float, Sequence[float]]],
        num_neighbors: Sequence[Union[int, Sequence[int]]],
        add_self_loops: bool = False,
        spatial_dim: int = 3,
        act: Union[str, Callable, None] = "relu",
        act_kwargs: Optional[Dict[str, Any]] = None,
        act_first: bool = False,
        norm: Union[str, Callable, None] = "batch_norm",
        norm_kwargs: Optional[Dict[str, Any]] = None,
        bias: Union[bool, List[bool]] = True,
        dropout: float = 0.0,
        head_channels: Optional[Sequence[int]] = None,
    ):
        super().__init__(in_channels, num_classes)
        self.num_categories = num_categories
        self.stem_channels = ensure_list(stem_channels, none_as_empty=True)
        self.stem_plain_last = stem_plain_last
        self.encoder_channels = ensure_list(encoder_channels)
        self.encoder_depths = encoder_depths
        self.encoder_expansion = encoder_expansion
        self.sa_layers = sa_layers
        self.sa_use_res = sa_use_res
        self.decoder_channels = ensure_list(decoder_channels)
        self.decoder_depths = decoder_depths
        self.decoder_plain_last = decoder_plain_last
        self.ratios = ensure_tuple(ratios)
        self.radiuses = ensure_tuple(radiuses)
        self.num_neighbors = ensure_tuple(num_neighbors)
        self.add_self_loops = add_self_loops
        self.spatial_dim = spatial_dim
        self.act = act
        self.act_kwargs = act_kwargs
        self.act_first = act_first
        self.norm = norm
        self.norm_kwargs = norm_kwargs
        self.bias = bias
        self.dropout = dropout
        self.head_channels = list(head_channels) if head_channels else []

        self.stem = self.configure_stem()
        self.encoder = self.configure_encoder()
        self.decoder = self.configure_decoder()
        self.head = self.configure_head()

    def configure_stem(self) -> Optional[nn.Module]:
        """Build the stem MLP lifting the input features, or `None` when `stem_channels` is unset."""
        if not self.stem_channels:
            return None
        return MLP(
            [self.in_channels] + self.stem_channels,
            act=self.act,
            act_kwargs=self.act_kwargs,
            act_first=self.act_first,
            norm=self.norm,
            norm_kwargs=self.norm_kwargs,
            bias=self.bias,
            plain_last=self.stem_plain_last,
        )

    def configure_encoder(self) -> PointNeXtEncoder:
        """Build the `PointNeXtEncoder` backbone."""
        in_channels = self.stem_channels[-1] if self.stem_channels else self.in_channels
        return PointNeXtEncoder(
            spatial_dim=self.spatial_dim,
            channels=[in_channels] + self.encoder_channels,
            depths=self.encoder_depths,
            expansion=self.encoder_expansion,
            sa_layers=self.sa_layers,
            sa_use_res=self.sa_use_res,
            ratios=self.ratios,
            radiuses=self.radiuses,
            num_neighbors=self.num_neighbors,
            act=self.act,
            act_kwargs=self.act_kwargs,
            act_first=self.act_first,
            norm=self.norm,
            norm_kwargs=self.norm_kwargs,
            bias=self.bias,
            add_self_loops=self.add_self_loops,
        )

    def configure_decoder(self) -> PointNeXtPartDecoder:
        """Build the `PointNeXtPartDecoder` upsampling the coarsest features back through the encoder skips."""
        in_channels = self.stem_channels[-1] if self.stem_channels else self.in_channels
        return PointNeXtPartDecoder(
            spatial_dim=self.spatial_dim,
            channels=[self.encoder_channels[-1]] + self.decoder_channels,
            skip_channels=self.encoder_channels[:-1][::-1] + [in_channels],
            depths=self.decoder_depths,
            global_conv1_in=self.encoder_channels[-2],
            global_conv2_in=self.encoder_channels[-1],
            num_categories=self.num_categories,
            act=self.act,
            act_kwargs=self.act_kwargs,
            act_first=self.act_first,
            norm=self.norm,
            norm_kwargs=self.norm_kwargs,
            bias=self.bias,
            plain_last=self.decoder_plain_last,
        )

    @property
    def num_features(self) -> int:
        """Channel count $C$ entering the head: decoder features concatenated with their global max and mean pools."""
        return self.decoder.channels[-1] * 3  # point + global_max + global_avg

    def configure_head(self) -> nn.Module:
        if self.num_classes == 0:
            return nn.Identity()
        if not self.head_channels:
            return nn.Linear(self.num_features, self.num_classes)
        return MLP(
            [self.num_features] + list(self.head_channels) + [self.num_classes],
            act=self.act,
            act_kwargs=self.act_kwargs,
            act_first=self.act_first,
            norm=self.norm,
            norm_kwargs=self.norm_kwargs,
            bias=self.bias,
            dropout=self.dropout,
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
    ) -> Tuple[Tensor, Tensor, Tensor, List[PointNeXtIntermediate]]: ...

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
        x = x if x is not None else pos
        if self.stem is not None:
            x = self.stem(x)

        return self.encoder(x, pos, batch, return_intermediates=return_intermediates)

    def forward_decoder(
        self,
        x: Tensor,
        pos: Tensor,
        batch: Tensor,
        category: Tensor,
        intermediates: List[PointNeXtIntermediate],
    ) -> Tuple[Tensor, Tensor, Tensor]:
        return self.decoder(x, pos, batch, category, intermediates)

    def forward_head(self, x: Tensor, batch: Tensor, pre_logits: bool = False) -> Tensor:
        if self.dropout and not isinstance(self.head, MLP):
            x = F.dropout(x, p=float(self.dropout), training=self.training)

        x_max = global_max_pool(x, batch)[batch]  # (N, C)
        x_avg = global_mean_pool(x, batch)[batch]  # (N, C)
        x = torch.cat([x, x_max, x_avg], dim=1)
        return x if pre_logits else self.head(x)

    def forward(self, x: OptTensor, pos: Tensor, batch: Tensor, category: Tensor) -> Tensor:
        x, pos, batch, intermediates = self.forward_features(x, pos, batch, return_intermediates=True)
        x, pos, batch = self.forward_decoder(x, pos, batch, category, intermediates)
        return self.forward_head(x, batch)


class PointNeXtClassification(ClassificationModel):
    r"""
    PointNeXt classification model as described in the paper
    :arxiv: [PointNeXt: Revisiting PointNet++ with Improved Training and Scaling Strategies](https://arxiv.org/abs/2206.04670)
    by Guocheng Qian, Yuchen Li, Houwen Peng, Jinjie Mai, Hasan Abed Al Kader Hammoud, Mohamed Elhoseiny, Bernard Ghanem.

    PointNeXt modernizes PointNet++ through improved training strategies and architectural enhancements,
    achieving state-of-the-art performance while maintaining efficiency. The model introduces Inverted
    Residual MLP (InvResMLP) blocks, separable MLPs, and relative position normalization to enable
    effective model scaling.

    Args:
        in_channels: Number of input channels (typically 3 for XYZ coordinates,
            or 6 for XYZ + RGB, or more with additional features like normal).
        num_classes: Number of output classes for classification.
        stem_channels: Number of channels in the stem MLP layer(s) that map input to higher dimension.
            If None, no stem is used.
        encoder_channels: Channel dimensions for each encoder block.
            The number of channels should be equal to the number of blocks + 1.
        encoder_depths: Number of blocks in each encoder stage after the initial SA block.
        encoder_expansion: Expansion ratio to determine the hidden channels of the encoder blocks.
        ratios: Downsampling sampling ratios for each encoder stage.
            The number of ratios should be equal to the number of blocks.
        radiuses: Query radius for neighborhood grouping in each stage.
            The number of radiuses should be equal to the number of blocks + 1.
        num_neighbors: Maximum number of neighbors for each encoder stage.
            The number of num_neighbors should be equal to the number of blocks + 1.
        add_self_loops: Whether to include the center point as its
            own neighbor in grouping operations.
        spatial_dim: Spatial dimensionality of point clouds (typically 3).
        act: Activation function.
        act_kwargs: Additional arguments for the activation function.
        act_first: Whether to apply activation before normalization.
        norm: Normalization layer type.
        norm_kwargs: Additional arguments for normalization layer.
        bias: Whether to use bias in linear / MLP layers.
        dropout: Dropout probability before the classification head.
        global_pool: Global pooling operation for final feature aggregation.
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        *,
        stem_channels: Optional[Union[int, Sequence[int]]] = None,
        stem_plain_last: bool = False,
        encoder_channels: Sequence[int],
        encoder_depths: Sequence[int],
        encoder_expansion: Union[int, Sequence[int]] = 4,
        sa_layers: int = 1,
        sa_use_res: bool = True,
        ratios: Sequence[float],
        radiuses: Sequence[Union[float, Sequence[float]]],
        num_neighbors: Sequence[Union[int, Sequence[int]]],
        add_self_loops: bool = False,
        spatial_dim: int = 3,
        act: Union[str, Callable, None] = "relu",
        act_kwargs: Optional[Dict[str, Any]] = None,
        act_first: bool = False,
        norm: Union[str, Callable, None] = "batch_norm",
        norm_kwargs: Optional[Dict[str, Any]] = None,
        bias: Union[bool, List[bool]] = True,
        dropout: float = 0.0,
        global_pool: PoolLike = "max",
        head_channels: Optional[Sequence[int]] = None,
        global_sa_channels: Optional[Sequence[int]] = None,
    ):
        super().__init__(in_channels, num_classes)
        self.stem_channels = ensure_list(stem_channels, none_as_empty=True)
        self.stem_plain_last = stem_plain_last
        self.encoder_channels = ensure_list(encoder_channels)
        self.encoder_depths = encoder_depths
        self.encoder_expansion = encoder_expansion
        self.sa_layers = sa_layers
        self.sa_use_res = sa_use_res
        self.ratios = ratios
        self.radiuses = radiuses
        self.num_neighbors = num_neighbors
        self.add_self_loops = add_self_loops
        self.spatial_dim = spatial_dim
        self.act = act
        self.act_kwargs = act_kwargs
        self.act_first = act_first
        self.norm = norm
        self.norm_kwargs = norm_kwargs
        self.bias = bias
        self.dropout = dropout
        self.head_channels = list(head_channels) if head_channels else []
        self.global_sa_channels = list(global_sa_channels) if global_sa_channels is not None else None

        self.stem = self.configure_stem()
        self.encoder = self.configure_encoder()
        self.global_sa = self.configure_global_sa()
        self.global_pool = create_pool(global_pool)
        self.head = self.configure_head()

    def configure_stem(self) -> Optional[nn.Module]:
        """Build the stem MLP lifting the input features, or `None` when `stem_channels` is unset."""
        if not self.stem_channels:
            return None
        return MLP(
            [self.in_channels] + self.stem_channels,
            act=self.act,
            act_kwargs=self.act_kwargs,
            act_first=self.act_first,
            norm=self.norm,
            norm_kwargs=self.norm_kwargs,
            bias=self.bias,
            plain_last=self.stem_plain_last,
        )

    def configure_encoder(self) -> PointNeXtEncoder:
        """Build the `PointNeXtEncoder` backbone."""
        in_channels = self.stem_channels[-1] if self.stem_channels else self.in_channels
        return PointNeXtEncoder(
            spatial_dim=self.spatial_dim,
            channels=[in_channels] + self.encoder_channels,
            depths=self.encoder_depths,
            expansion=self.encoder_expansion,
            sa_layers=self.sa_layers,
            sa_use_res=self.sa_use_res,
            ratios=self.ratios,
            radiuses=self.radiuses,
            num_neighbors=self.num_neighbors,
            act=self.act,
            act_kwargs=self.act_kwargs,
            act_first=self.act_first,
            norm=self.norm,
            norm_kwargs=self.norm_kwargs,
            bias=self.bias,
            add_self_loops=self.add_self_loops,
        )

    def configure_global_sa(self) -> Optional[PointNeXtSetAbstraction]:
        """Build the global set-abstraction applied before pooling, or `None` when `global_sa_channels` is unset."""
        if self.global_sa_channels is None:
            return None
        return PointNeXtSetAbstraction(
            spatial_dim=self.spatial_dim,
            in_channels=self.encoder_channels[-1],
            channels=self.global_sa_channels,
            ratio=1.0,
            radius=1e6,
            num_neighbors=1024,
            use_res=False,
            act=self.act,
            act_kwargs=self.act_kwargs,
            act_first=self.act_first,
            norm=self.norm,
            norm_kwargs=self.norm_kwargs,
            bias=self.bias,
            add_self_loops=self.add_self_loops,
            aggr="max",
        )

    @property
    def num_features(self) -> int:
        """Feature dimension $C$ fed to the classification head, after the optional global set-abstraction."""
        return int(self.global_sa_channels[-1]) if self.global_sa_channels is not None else self.encoder_channels[-1]

    def configure_head(self) -> nn.Module:
        if self.num_classes == 0:
            return nn.Identity()
        if not self.head_channels:
            return nn.Linear(self.num_features, self.num_classes)
        return MLP(
            [self.num_features] + list(self.head_channels) + [self.num_classes],
            act=self.act,
            act_kwargs=self.act_kwargs,
            act_first=self.act_first,
            norm=self.norm,
            norm_kwargs=self.norm_kwargs,
            bias=self.bias,
            dropout=self.dropout,
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
    ) -> Tuple[Tensor, Tensor, Tensor, List[PointNeXtIntermediate]]: ...

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
        x = x if x is not None else pos
        if self.stem is not None:
            x = self.stem(x)
        return self.encoder(x, pos, batch, return_intermediates=return_intermediates)

    def forward_head(self, x: Tensor, pos: Tensor, batch: Tensor, pre_logits: bool = False) -> Tensor:
        if self.global_sa is not None:
            x, pos, batch = self.global_sa(x, pos, batch)
        x = self.global_pool(x, batch)
        if self.dropout and not isinstance(self.head, MLP):
            x = F.dropout(x, p=float(self.dropout), training=self.training)
        return x if pre_logits else self.head(x)

    def forward(self, x: OptTensor, pos: Tensor, batch: Tensor) -> Tensor:
        x, pos, batch = self.forward_features(x, pos, batch)
        return self.forward_head(x, pos, batch)


class PointNeXtSegmentation(SegmentationModel):
    r"""
    PointNeXt segmentation model as described in the paper
    :arxiv: [PointNeXt: Revisiting PointNet++ with Improved Training and Scaling Strategies](https://arxiv.org/abs/2206.04670)
    by Guocheng Qian, Yuchen Li, Houwen Peng, Jinjie Mai, Hasan Abed Al Kader Hammoud, Mohamed Elhoseiny, Bernard Ghanem.

    PointNeXt modernizes PointNet++ through improved training strategies and architectural enhancements,
    achieving state-of-the-art performance while maintaining efficiency. The model introduces Inverted
    Residual MLP (InvResMLP) blocks, separable MLPs, and relative position normalization to enable
    effective model scaling.

    Args:
        in_channels: Number of input channels (typically 3 for XYZ coordinates,
            or 6 for XYZ + RGB, or more with additional features like normal).
        num_classes: Number of output classes for segmentation.
        stem_channels: Number of channels in the stem MLP layer(s) that map input to higher dimension.
            If None, no stem is used.
        encoder_channels: Channel dimensions for each encoder block.
            The number of channels should be equal to the number of blocks + 1.
        encoder_depths: Number of layers in each encoder block after the initial SA block.
        encoder_expansion: Expansion ratio to determine the hidden channels of the encoder blocks.
        decoder_channels: Channel dimensions for each decoder block.
            The number of channels should be equal to the number of decoder blocks + 1.
        decoder_depths: Number of layers in each decoder stage.
        ratios: Downsampling sampling ratios for each encoder stage.
            The number of ratios should be equal to the number of blocks.
        radiuses: Query radius for neighborhood grouping in each stage.
            The number of radiuses should be equal to the number of blocks + 1.
        num_neighbors: Maximum number of neighbors for each encoder stage.
            The number of num_neighbors should be equal to the number of blocks + 1.
        add_self_loops: Whether to include the center point as its
            own neighbor in grouping operations.
        spatial_dim: Spatial dimensionality of point clouds (typically 3).
        act: Activation function.
        act_kwargs: Additional arguments for the activation function.
        act_first: Whether to apply activation before normalization.
        norm: Normalization layer type.
        norm_kwargs: Additional arguments for normalization layer.
        bias: Whether to use bias in linear / MLP layers.
        dropout: Dropout probability before the classification head.
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        *,
        stem_channels: Optional[Union[int, Sequence[int]]] = None,
        stem_plain_last: bool = False,
        encoder_channels: Sequence[int],
        encoder_depths: Sequence[int],
        encoder_expansion: Union[int, Sequence[int]] = 4,
        sa_layers: int = 1,
        sa_use_res: bool = True,
        decoder_channels: Sequence[int],
        decoder_depths: Sequence[int],
        decoder_plain_last: bool = True,
        ratios: Sequence[float],
        radiuses: Sequence[Union[float, Sequence[float]]],
        num_neighbors: Sequence[Union[int, Sequence[int]]],
        add_self_loops: bool = False,
        spatial_dim: int = 3,
        act: Union[str, Callable, None] = "relu",
        act_kwargs: Optional[Dict[str, Any]] = None,
        act_first: bool = False,
        norm: Union[str, Callable, None] = "batch_norm",
        norm_kwargs: Optional[Dict[str, Any]] = None,
        bias: Union[bool, List[bool]] = True,
        dropout: float = 0.0,
        head_channels: Optional[Sequence[int]] = None,
    ):
        super().__init__(in_channels, num_classes)
        self.stem_channels = ensure_list(stem_channels, none_as_empty=True)
        self.stem_plain_last = stem_plain_last
        self.encoder_channels = ensure_list(encoder_channels)
        self.encoder_depths = encoder_depths
        self.encoder_expansion = encoder_expansion
        self.sa_layers = sa_layers
        self.sa_use_res = sa_use_res
        self.decoder_channels = ensure_list(decoder_channels)
        self.decoder_depths = decoder_depths
        self.decoder_plain_last = decoder_plain_last
        self.ratios = ensure_tuple(ratios)
        self.radiuses = ensure_tuple(radiuses)
        self.num_neighbors = ensure_tuple(num_neighbors)
        self.add_self_loops = add_self_loops
        self.spatial_dim = spatial_dim
        self.act = act
        self.act_kwargs = act_kwargs
        self.act_first = act_first
        self.norm = norm
        self.norm_kwargs = norm_kwargs
        self.bias = bias
        self.dropout = dropout
        self.head_channels = list(head_channels) if head_channels else []

        self.stem = self.configure_stem()
        self.encoder = self.configure_encoder()
        self.decoder = self.configure_decoder()
        self.head = self.configure_head()

    def configure_stem(self) -> Optional[nn.Module]:
        """Build the stem MLP lifting the input features, or `None` when `stem_channels` is unset."""
        if not self.stem_channels:
            return None
        return MLP(
            [self.in_channels] + self.stem_channels,
            act=self.act,
            act_kwargs=self.act_kwargs,
            act_first=self.act_first,
            norm=self.norm,
            norm_kwargs=self.norm_kwargs,
            bias=self.bias,
            plain_last=self.stem_plain_last,
        )

    def configure_encoder(self) -> PointNeXtEncoder:
        """Build the `PointNeXtEncoder` backbone."""
        in_channels = self.stem_channels[-1] if self.stem_channels else self.in_channels
        return PointNeXtEncoder(
            spatial_dim=self.spatial_dim,
            channels=[in_channels] + self.encoder_channels,
            depths=self.encoder_depths,
            expansion=self.encoder_expansion,
            sa_layers=self.sa_layers,
            sa_use_res=self.sa_use_res,
            ratios=self.ratios,
            radiuses=self.radiuses,
            num_neighbors=self.num_neighbors,
            act=self.act,
            act_kwargs=self.act_kwargs,
            act_first=self.act_first,
            norm=self.norm,
            norm_kwargs=self.norm_kwargs,
            bias=self.bias,
            add_self_loops=self.add_self_loops,
        )

    def configure_decoder(self) -> PointNeXtDecoder:
        """Build the `PointNeXtDecoder` upsampling the coarsest features back through the encoder skips."""
        in_channels = self.stem_channels[-1] if self.stem_channels else self.in_channels
        return PointNeXtDecoder(
            spatial_dim=self.spatial_dim,
            channels=[self.encoder_channels[-1]] + self.decoder_channels,
            skip_channels=self.encoder_channels[:-1][::-1] + [in_channels],
            depths=self.decoder_depths,
            act=self.act,
            act_kwargs=self.act_kwargs,
            act_first=self.act_first,
            norm=self.norm,
            norm_kwargs=self.norm_kwargs,
            bias=self.bias,
            plain_last=self.decoder_plain_last,
        )

    @property
    def num_features(self) -> int:
        """Feature dimension $C$ of the decoder output."""
        return self.decoder.channels[-1]

    def configure_head(self) -> nn.Module:
        if self.num_classes == 0:
            return nn.Identity()
        if not self.head_channels:
            return nn.Linear(self.num_features, self.num_classes)
        return MLP(
            [self.num_features] + list(self.head_channels) + [self.num_classes],
            act=self.act,
            act_kwargs=self.act_kwargs,
            act_first=self.act_first,
            norm=self.norm,
            norm_kwargs=self.norm_kwargs,
            bias=self.bias,
            dropout=self.dropout,
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
    ) -> Tuple[Tensor, Tensor, Tensor, List[PointNeXtIntermediate]]: ...

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
        x = x if x is not None else pos
        if self.stem is not None:
            x = self.stem(x)

        return self.encoder(x, pos, batch, return_intermediates=return_intermediates)

    def forward_decoder(
        self,
        x: Tensor,
        pos: Tensor,
        batch: Tensor,
        intermediates: List[PointNeXtIntermediate],
    ) -> Tuple[Tensor, Tensor, Tensor]:
        return self.decoder(x, pos, batch, intermediates)

    def forward_head(self, x: Tensor, pre_logits: bool = False) -> Tensor:
        if self.dropout and not isinstance(self.head, MLP):
            x = F.dropout(x, p=float(self.dropout), training=self.training)
        return x if pre_logits else self.head(x)

    def forward(self, x: OptTensor, pos: Tensor, batch: Tensor) -> Tensor:
        x, pos, batch, intermediates = self.forward_features(x, pos, batch, return_intermediates=True)
        x, pos, batch = self.forward_decoder(x, pos, batch, intermediates)
        return self.forward_head(x)


@register_model(
    "pointnext-sm",
    task="classification",
    hparams=dict(
        spatial_dim=3,
        stem_channels=32,
        stem_plain_last=True,
        encoder_channels=[64, 128, 256, 512],
        encoder_depths=[0, 0, 0, 0],
        encoder_expansion=4,
        sa_layers=2,
        sa_use_res=True,
        ratios=[0.25, 0.25, 0.25, 0.25],
        radiuses=[0.1, 0.2, 0.4, 0.8, 1.6],
        num_neighbors=[32, 32, 32, 32, 32],
        act="relu",
        act_first=False,
        norm="batch_norm",
        bias=True,
        add_self_loops=False,
        head_channels=[512, 256],
        global_sa_channels=[512, 512],
    ),
)
def pointnext_sm_clf(**hparams: Any) -> PointNeXtClassification:
    return PointNeXtClassification(**hparams)


@register_model(
    "pointnext-base",
    task="classification",
    hparams=dict(
        spatial_dim=3,
        stem_channels=32,
        stem_plain_last=True,
        encoder_channels=[64, 128, 256, 512],
        encoder_depths=[1, 2, 1, 1],
        encoder_expansion=4,
        sa_layers=1,
        sa_use_res=False,
        ratios=[0.25, 0.25, 0.25, 0.25],
        radiuses=[0.1, 0.2, 0.4, 0.8, 1.6],
        num_neighbors=[32, 32, 32, 32, 32],
        act="relu",
        act_first=False,
        norm="batch_norm",
        bias=True,
        add_self_loops=False,
        head_channels=[512, 256],
        global_sa_channels=[512],
    ),
)
def pointnext_base_clf(**hparams: Any) -> PointNeXtClassification:
    return PointNeXtClassification(**hparams)


@register_model(
    "pointnext-lg",
    task="classification",
    hparams=dict(
        spatial_dim=3,
        stem_channels=32,
        stem_plain_last=True,
        encoder_channels=[64, 128, 256, 512],
        encoder_depths=[2, 4, 2, 2],
        encoder_expansion=4,
        sa_layers=1,
        sa_use_res=False,
        ratios=[0.25, 0.25, 0.25, 0.25],
        radiuses=[0.1, 0.2, 0.4, 0.8, 1.6],
        num_neighbors=[32, 32, 32, 32, 32],
        act="relu",
        act_first=False,
        norm="batch_norm",
        bias=True,
        add_self_loops=False,
        head_channels=[512, 256],
        global_sa_channels=[512],
    ),
)
def pointnext_lg_clf(**hparams: Any) -> PointNeXtClassification:
    return PointNeXtClassification(**hparams)


@register_model(
    "pointnext-xl",
    task="classification",
    hparams=dict(
        spatial_dim=3,
        stem_channels=64,
        stem_plain_last=True,
        encoder_channels=[128, 256, 512, 1024],
        encoder_depths=[3, 6, 3, 3],
        encoder_expansion=4,
        sa_layers=1,
        sa_use_res=False,
        ratios=[0.25, 0.25, 0.25, 0.25],
        radiuses=[0.1, 0.2, 0.4, 0.8, 1.6],
        num_neighbors=[32, 32, 32, 32, 32],
        act="relu",
        act_first=False,
        norm="batch_norm",
        bias=True,
        add_self_loops=False,
        head_channels=[1024, 512],
        global_sa_channels=[1024],
    ),
)
def pointnext_xl_clf(**hparams: Any) -> PointNeXtClassification:
    return PointNeXtClassification(**hparams)


@register_model(
    "pointnext-sm.scanobjectnn.openpoints",
    task="classification",
    weights=WeightsDict(
        url="hf://torch-pointcloud/pointnext-sm.scanobjectnn.openpoints/resolve/main/model.safetensors",
        dataset="scanobjectnn",
        classes=SCANOBJECTNN_CLASSES,
        author="openpoints",
        license="MIT",
    ),
    transform=T.Compose(
        [
            T.CopyItems(keys=DataKeys.POS, names=DataKeys.ORIGIN_POS),
            T.FarthestPointSample(
                pos_key=DataKeys.POS,
                num_samples=1024,
                random_start=False,
                dst_index_key=DataKeys.INDEX,
            ),
            T.AxisMinOffset(keys=DataKeys.POS, axis=1, dst_keys="height"),
            T.Rescale(keys=DataKeys.POS, method="centroid"),
            T.Cat(keys=(DataKeys.POS, "height"), dst_key=DataKeys.X),
        ]
    ),
    hparams=dict(
        in_channels=4,
        num_classes=15,
        spatial_dim=3,
        stem_channels=32,
        stem_plain_last=True,
        encoder_channels=[64, 128, 256, 512],
        encoder_depths=[0, 0, 0, 0],
        encoder_expansion=4,
        sa_layers=2,
        sa_use_res=True,
        ratios=[0.5, 0.5, 0.5, 0.5],
        radiuses=[0.15, 0.225, 0.3375, 0.50625, 0.759375],
        num_neighbors=[32, 32, 32, 32, 32],
        act="relu",
        act_first=False,
        norm="batch_norm",
        bias=True,
        add_self_loops=False,
        dropout=0.5,
        head_channels=[512, 256],
        global_sa_channels=[512, 512],
    ),
)
def pointnext_sm_scanobjectnn_clf(**hparams: Any) -> PointNeXtClassification:
    return PointNeXtClassification(**hparams)


@register_model(
    "pointnext-sm-c64.modelnet40.openpoints",
    task="classification",
    weights=WeightsDict(
        url="hf://torch-pointcloud/pointnext-sm-c64.modelnet40.openpoints/resolve/main/model.safetensors",
        dataset="modelnet40",
        metrics={"OA": 92.1},
        classes=MODELNET40_CLASSES,
        author="openpoints",
        license="MIT",
    ),
    transform=T.Compose(
        [
            T.CopyItems(keys=DataKeys.POS, names=DataKeys.ORIGIN_POS),
            T.FarthestPointSample(
                pos_key=DataKeys.POS,
                keys=[DataKeys.NORMAL],
                num_samples=1024,
                random_start=False,
                dst_index_key=DataKeys.INDEX,
            ),
        ]
    ),
    hparams=dict(
        in_channels=3,
        num_classes=40,
        spatial_dim=3,
        stem_channels=64,
        stem_plain_last=True,
        encoder_channels=[128, 256, 512, 1024],
        encoder_depths=[0, 0, 0, 0],
        encoder_expansion=4,
        sa_layers=2,
        sa_use_res=True,
        ratios=[0.5, 0.5, 0.5, 0.5],
        radiuses=[0.15, 0.225, 0.3375, 0.50625, 0.759375],
        num_neighbors=[32, 32, 32, 32, 32],
        act="relu",
        act_first=False,
        norm="batch_norm",
        bias=True,
        add_self_loops=False,
        dropout=0.5,
        head_channels=[512, 256],
        global_sa_channels=[1024, 1024],
    ),
)
def pointnext_sm_c64_modelnet40_clf(**hparams: Any) -> PointNeXtClassification:
    return PointNeXtClassification(**hparams)


@register_model(
    "pointnext-sm",
    task="segmentation",
    hparams=dict(
        spatial_dim=3,
        stem_channels=32,
        stem_plain_last=True,
        encoder_channels=[64, 128, 256, 512],
        encoder_depths=[0, 0, 0, 0],
        encoder_expansion=4,
        sa_layers=2,
        sa_use_res=True,
        decoder_channels=[512, 256, 128, 64],
        decoder_depths=[2, 2, 2, 2],
        ratios=[0.25, 0.25, 0.25, 0.25],
        radiuses=[0.1, 0.2, 0.4, 0.8, 1.6],
        num_neighbors=[32, 32, 32, 32, 32],
        act="relu",
        act_first=False,
        norm="batch_norm",
        bias=True,
        add_self_loops=False,
    ),
)
def pointnext_sm_seg(**hparams: Any) -> PointNeXtSegmentation:
    return PointNeXtSegmentation(**hparams)


@register_model(
    "pointnext-base",
    task="segmentation",
    hparams=dict(
        spatial_dim=3,
        stem_channels=32,
        stem_plain_last=True,
        encoder_channels=[64, 128, 256, 512],
        encoder_depths=[1, 2, 1, 1],
        encoder_expansion=4,
        sa_layers=1,
        sa_use_res=False,
        decoder_channels=[512, 256, 128, 64],
        decoder_depths=[2, 2, 2, 2],
        ratios=[0.25, 0.25, 0.25, 0.25],
        radiuses=[0.1, 0.2, 0.4, 0.8, 1.6],
        num_neighbors=[32, 32, 32, 32, 32],
        act="relu",
        act_first=False,
        norm="batch_norm",
        bias=True,
        add_self_loops=False,
    ),
)
def pointnext_base_seg(**hparams: Any) -> PointNeXtSegmentation:
    return PointNeXtSegmentation(**hparams)


@register_model(
    "pointnext-lg",
    task="segmentation",
    hparams=dict(
        spatial_dim=3,
        stem_channels=32,
        stem_plain_last=True,
        encoder_channels=[64, 128, 256, 512],
        encoder_depths=[2, 4, 2, 2],
        encoder_expansion=4,
        sa_layers=1,
        sa_use_res=False,
        decoder_channels=[512, 256, 128, 64],
        decoder_depths=[2, 2, 2, 2],
        ratios=[0.25, 0.25, 0.25, 0.25],
        radiuses=[0.1, 0.2, 0.4, 0.8, 1.6],
        num_neighbors=[32, 32, 32, 32, 32],
        act="relu",
        act_first=False,
        norm="batch_norm",
        bias=True,
        add_self_loops=False,
    ),
)
def pointnext_lg_seg(**hparams: Any) -> PointNeXtSegmentation:
    return PointNeXtSegmentation(**hparams)


@register_model(
    "pointnext-xl",
    task="segmentation",
    hparams=dict(
        spatial_dim=3,
        stem_channels=64,
        stem_plain_last=True,
        encoder_channels=[128, 256, 512, 1024],
        encoder_depths=[3, 6, 3, 3],
        encoder_expansion=4,
        sa_layers=1,
        sa_use_res=False,
        decoder_channels=[1024, 512, 256, 128],
        decoder_depths=[2, 2, 2, 2],
        ratios=[0.25, 0.25, 0.25, 0.25],
        radiuses=[0.1, 0.2, 0.4, 0.8, 1.6],
        num_neighbors=[32, 32, 32, 32, 32],
        act="relu",
        act_first=False,
        norm="batch_norm",
        bias=True,
        add_self_loops=False,
    ),
)
def pointnext_xl_seg(**hparams: Any) -> PointNeXtSegmentation:
    return PointNeXtSegmentation(**hparams)


_S3DIS_TRANSFORMS = T.Compose(
    [
        # The PointNeXt benchmark dataset uses a slightly different label mapping than the original S3DIS dataset
        # used by other papers.
        T.Relabel(
            keys=DataKeys.SEGMENT,
            labels=[0, 1, 2, 3, 4, 5, 6, 8, 7, 10, 9, 11, 12],
        ),
        T.Shift(keys=DataKeys.POS, method="min"),
        T.CopyItems(
            keys=[DataKeys.POS, DataKeys.SEGMENT],
            names=[DataKeys.ORIGIN_POS, DataKeys.ORIGIN_SEGMENT],
            allow_missing_keys=True,
        ),
        T.Voxelize(
            pos_key=DataKeys.POS,
            pos_reduce="first",
            keys=[DataKeys.COLOR, DataKeys.SEGMENT, DataKeys.NORM_POS, DataKeys.INSTANCE],
            reduce="first",
            size=0.04,
            method="fnv",  # Use the same method as PointNext, for reproducibility.
            allow_missing_keys=True,
            dst_inverse_key=DataKeys.INVERSE,
        ),
        T.AxisMinOffset(keys=DataKeys.POS, axis=2, dst_keys="height"),
        T.Shift(keys=DataKeys.POS, method="centroid"),
        T.AlignAxis(keys=DataKeys.POS, dim=2),
        T.Divide(keys=DataKeys.COLOR, divisor=255.0),
        T.Normalize(
            keys=DataKeys.COLOR,
            mean=[0.5136457, 0.49523646, 0.44921124],
            std=[0.18308958, 0.18415008, 0.19252081],
        ),
        T.Cat(keys=(DataKeys.COLOR, "height"), dst_key=DataKeys.X),
    ]
)

_S3DIS_COMMON_HPARAMS = dict(
    in_channels=4,
    num_classes=13,
    spatial_dim=3,
    stem_plain_last=True,
    decoder_plain_last=False,
    ratios=[0.25, 0.25, 0.25, 0.25],
    radiuses=[0.1, 0.2, 0.4, 0.8, 1.6],
    num_neighbors=[32, 32, 32, 32, 32],
    act="relu",
    act_first=False,
    norm="batch_norm",
    bias=True,
    add_self_loops=False,
)


_S3DIS_VARIANT_HPARAMS: Dict[str, Dict[str, Any]] = {
    "sm": dict(
        stem_channels=32,
        encoder_channels=[64, 128, 256, 512],
        encoder_depths=[0, 0, 0, 0],
        encoder_expansion=4,
        sa_layers=2,
        sa_use_res=True,
        decoder_channels=[256, 128, 64, 32],
        decoder_depths=[2, 2, 2, 2],
        head_channels=[32],
    ),
    "base": dict(
        stem_channels=32,
        encoder_channels=[64, 128, 256, 512],
        encoder_depths=[1, 2, 1, 1],
        encoder_expansion=4,
        sa_layers=1,
        sa_use_res=False,
        decoder_channels=[256, 128, 64, 32],
        decoder_depths=[2, 2, 2, 2],
        head_channels=[32],
    ),
    "lg": dict(
        stem_channels=32,
        encoder_channels=[64, 128, 256, 512],
        encoder_depths=[2, 4, 2, 2],
        encoder_expansion=4,
        sa_layers=1,
        sa_use_res=False,
        decoder_channels=[256, 128, 64, 32],
        decoder_depths=[2, 2, 2, 2],
        head_channels=[32],
    ),
    "xl": dict(
        stem_channels=64,
        encoder_channels=[128, 256, 512, 1024],
        encoder_depths=[3, 6, 3, 3],
        encoder_expansion=4,
        sa_layers=1,
        sa_use_res=False,
        decoder_channels=[512, 256, 128, 64],
        decoder_depths=[2, 2, 2, 2],
        head_channels=[64],
    ),
}


@register_model(
    "pointnext-sm.s3dis-area1.openpoints",
    task="segmentation",
    weights=WeightsDict(
        url="hf://torch-pointcloud/pointnext-sm.s3dis-area1.openpoints/resolve/main/model.safetensors",
        dataset="s3dis-area1",
        classes=S3DIS_CLASSES,
        author="openpoints",
        license="MIT",
    ),
    transform=_S3DIS_TRANSFORMS,
    hparams={**_S3DIS_COMMON_HPARAMS, **_S3DIS_VARIANT_HPARAMS["sm"]},
)
def pointnext_sm_s3dis_area1_seg(**hparams: Any) -> PointNeXtSegmentation:
    return PointNeXtSegmentation(**hparams)


@register_model(
    "pointnext-sm.s3dis-area2.openpoints",
    task="segmentation",
    weights=WeightsDict(
        url="hf://torch-pointcloud/pointnext-sm.s3dis-area2.openpoints/resolve/main/model.safetensors",
        dataset="s3dis-area2",
        classes=S3DIS_CLASSES,
        author="openpoints",
        license="MIT",
    ),
    transform=_S3DIS_TRANSFORMS,
    hparams={**_S3DIS_COMMON_HPARAMS, **_S3DIS_VARIANT_HPARAMS["sm"]},
)
def pointnext_sm_s3dis_area2_seg(**hparams: Any) -> PointNeXtSegmentation:
    return PointNeXtSegmentation(**hparams)


@register_model(
    "pointnext-sm.s3dis-area3.openpoints",
    task="segmentation",
    weights=WeightsDict(
        url="hf://torch-pointcloud/pointnext-sm.s3dis-area3.openpoints/resolve/main/model.safetensors",
        dataset="s3dis-area3",
        classes=S3DIS_CLASSES,
        author="openpoints",
        license="MIT",
    ),
    transform=_S3DIS_TRANSFORMS,
    hparams={**_S3DIS_COMMON_HPARAMS, **_S3DIS_VARIANT_HPARAMS["sm"]},
)
def pointnext_sm_s3dis_area3_seg(**hparams: Any) -> PointNeXtSegmentation:
    return PointNeXtSegmentation(**hparams)


@register_model(
    "pointnext-sm.s3dis-area4.openpoints",
    task="segmentation",
    weights=WeightsDict(
        url="hf://torch-pointcloud/pointnext-sm.s3dis-area4.openpoints/resolve/main/model.safetensors",
        dataset="s3dis-area4",
        classes=S3DIS_CLASSES,
        author="openpoints",
        license="MIT",
    ),
    transform=_S3DIS_TRANSFORMS,
    hparams={**_S3DIS_COMMON_HPARAMS, **_S3DIS_VARIANT_HPARAMS["sm"]},
)
def pointnext_sm_s3dis_area4_seg(**hparams: Any) -> PointNeXtSegmentation:
    return PointNeXtSegmentation(**hparams)


@register_model(
    "pointnext-sm.s3dis-area5.openpoints",
    task="segmentation",
    weights=WeightsDict(
        url="hf://torch-pointcloud/pointnext-sm.s3dis-area5.openpoints/resolve/main/model.safetensors",
        dataset="s3dis-area5",
        metrics={"mIoU": 63.01},
        classes=S3DIS_CLASSES,
        author="openpoints",
        license="MIT",
    ),
    transform=_S3DIS_TRANSFORMS,
    hparams={**_S3DIS_COMMON_HPARAMS, **_S3DIS_VARIANT_HPARAMS["sm"]},
)
def pointnext_sm_s3dis_area5_seg(**hparams: Any) -> PointNeXtSegmentation:
    return PointNeXtSegmentation(**hparams)


@register_model(
    "pointnext-sm.s3dis-area6.openpoints",
    task="segmentation",
    weights=WeightsDict(
        url="hf://torch-pointcloud/pointnext-sm.s3dis-area6.openpoints/resolve/main/model.safetensors",
        dataset="s3dis-area6",
        classes=S3DIS_CLASSES,
        author="openpoints",
        license="MIT",
    ),
    transform=_S3DIS_TRANSFORMS,
    hparams={**_S3DIS_COMMON_HPARAMS, **_S3DIS_VARIANT_HPARAMS["sm"]},
)
def pointnext_sm_s3dis_area6_seg(**hparams: Any) -> PointNeXtSegmentation:
    return PointNeXtSegmentation(**hparams)


@register_model(
    "pointnext-base.s3dis-area1.openpoints",
    task="segmentation",
    weights=WeightsDict(
        url="hf://torch-pointcloud/pointnext-base.s3dis-area1.openpoints/resolve/main/model.safetensors",
        dataset="s3dis-area1",
        classes=S3DIS_CLASSES,
        author="openpoints",
        license="MIT",
    ),
    transform=_S3DIS_TRANSFORMS,
    hparams={**_S3DIS_COMMON_HPARAMS, **_S3DIS_VARIANT_HPARAMS["base"]},
)
def pointnext_base_s3dis_area1_seg(**hparams: Any) -> PointNeXtSegmentation:
    return PointNeXtSegmentation(**hparams)


@register_model(
    "pointnext-base.s3dis-area2.openpoints",
    task="segmentation",
    weights=WeightsDict(
        url="hf://torch-pointcloud/pointnext-base.s3dis-area2.openpoints/resolve/main/model.safetensors",
        dataset="s3dis-area2",
        classes=S3DIS_CLASSES,
        author="openpoints",
        license="MIT",
    ),
    transform=_S3DIS_TRANSFORMS,
    hparams={**_S3DIS_COMMON_HPARAMS, **_S3DIS_VARIANT_HPARAMS["base"]},
)
def pointnext_base_s3dis_area2_seg(**hparams: Any) -> PointNeXtSegmentation:
    return PointNeXtSegmentation(**hparams)


@register_model(
    "pointnext-base.s3dis-area3.openpoints",
    task="segmentation",
    weights=WeightsDict(
        url="hf://torch-pointcloud/pointnext-base.s3dis-area3.openpoints/resolve/main/model.safetensors",
        dataset="s3dis-area3",
        classes=S3DIS_CLASSES,
        author="openpoints",
        license="MIT",
    ),
    transform=_S3DIS_TRANSFORMS,
    hparams={**_S3DIS_COMMON_HPARAMS, **_S3DIS_VARIANT_HPARAMS["base"]},
)
def pointnext_base_s3dis_area3_seg(**hparams: Any) -> PointNeXtSegmentation:
    return PointNeXtSegmentation(**hparams)


@register_model(
    "pointnext-base.s3dis-area4.openpoints",
    task="segmentation",
    weights=WeightsDict(
        url="hf://torch-pointcloud/pointnext-base.s3dis-area4.openpoints/resolve/main/model.safetensors",
        dataset="s3dis-area4",
        classes=S3DIS_CLASSES,
        author="openpoints",
        license="MIT",
    ),
    transform=_S3DIS_TRANSFORMS,
    hparams={**_S3DIS_COMMON_HPARAMS, **_S3DIS_VARIANT_HPARAMS["base"]},
)
def pointnext_base_s3dis_area4_seg(**hparams: Any) -> PointNeXtSegmentation:
    return PointNeXtSegmentation(**hparams)


@register_model(
    "pointnext-base.s3dis-area5.openpoints",
    task="segmentation",
    weights=WeightsDict(
        url="hf://torch-pointcloud/pointnext-base.s3dis-area5.openpoints/resolve/main/model.safetensors",
        dataset="s3dis-area5",
        classes=S3DIS_CLASSES,
        author="openpoints",
        license="MIT",
    ),
    transform=_S3DIS_TRANSFORMS,
    hparams={**_S3DIS_COMMON_HPARAMS, **_S3DIS_VARIANT_HPARAMS["base"]},
)
def pointnext_base_s3dis_area5_seg(**hparams: Any) -> PointNeXtSegmentation:
    return PointNeXtSegmentation(**hparams)


@register_model(
    "pointnext-base.s3dis-area6.openpoints",
    task="segmentation",
    weights=WeightsDict(
        url="hf://torch-pointcloud/pointnext-base.s3dis-area6.openpoints/resolve/main/model.safetensors",
        dataset="s3dis-area6",
        classes=S3DIS_CLASSES,
        author="openpoints",
        license="MIT",
    ),
    transform=_S3DIS_TRANSFORMS,
    hparams={**_S3DIS_COMMON_HPARAMS, **_S3DIS_VARIANT_HPARAMS["base"]},
)
def pointnext_base_s3dis_area6_seg(**hparams: Any) -> PointNeXtSegmentation:
    return PointNeXtSegmentation(**hparams)


@register_model(
    "pointnext-lg.s3dis-area1.openpoints",
    task="segmentation",
    weights=WeightsDict(
        url="hf://torch-pointcloud/pointnext-lg.s3dis-area1.openpoints/resolve/main/model.safetensors",
        dataset="s3dis-area1",
        classes=S3DIS_CLASSES,
        author="openpoints",
        license="MIT",
    ),
    transform=_S3DIS_TRANSFORMS,
    hparams={**_S3DIS_COMMON_HPARAMS, **_S3DIS_VARIANT_HPARAMS["lg"]},
)
def pointnext_lg_s3dis_area1_seg(**hparams: Any) -> PointNeXtSegmentation:
    return PointNeXtSegmentation(**hparams)


@register_model(
    "pointnext-lg.s3dis-area2.openpoints",
    task="segmentation",
    weights=WeightsDict(
        url="hf://torch-pointcloud/pointnext-lg.s3dis-area2.openpoints/resolve/main/model.safetensors",
        dataset="s3dis-area2",
        classes=S3DIS_CLASSES,
        author="openpoints",
        license="MIT",
    ),
    transform=_S3DIS_TRANSFORMS,
    hparams={**_S3DIS_COMMON_HPARAMS, **_S3DIS_VARIANT_HPARAMS["lg"]},
)
def pointnext_lg_s3dis_area2_seg(**hparams: Any) -> PointNeXtSegmentation:
    return PointNeXtSegmentation(**hparams)


@register_model(
    "pointnext-lg.s3dis-area3.openpoints",
    task="segmentation",
    weights=WeightsDict(
        url="hf://torch-pointcloud/pointnext-lg.s3dis-area3.openpoints/resolve/main/model.safetensors",
        dataset="s3dis-area3",
        classes=S3DIS_CLASSES,
        author="openpoints",
        license="MIT",
    ),
    transform=_S3DIS_TRANSFORMS,
    hparams={**_S3DIS_COMMON_HPARAMS, **_S3DIS_VARIANT_HPARAMS["lg"]},
)
def pointnext_lg_s3dis_area3_seg(**hparams: Any) -> PointNeXtSegmentation:
    return PointNeXtSegmentation(**hparams)


@register_model(
    "pointnext-lg.s3dis-area4.openpoints",
    task="segmentation",
    weights=WeightsDict(
        url="hf://torch-pointcloud/pointnext-lg.s3dis-area4.openpoints/resolve/main/model.safetensors",
        dataset="s3dis-area4",
        classes=S3DIS_CLASSES,
        author="openpoints",
        license="MIT",
    ),
    transform=_S3DIS_TRANSFORMS,
    hparams={**_S3DIS_COMMON_HPARAMS, **_S3DIS_VARIANT_HPARAMS["lg"]},
)
def pointnext_lg_s3dis_area4_seg(**hparams: Any) -> PointNeXtSegmentation:
    return PointNeXtSegmentation(**hparams)


@register_model(
    "pointnext-lg.s3dis-area5.openpoints",
    task="segmentation",
    weights=WeightsDict(
        url="hf://torch-pointcloud/pointnext-lg.s3dis-area5.openpoints/resolve/main/model.safetensors",
        dataset="s3dis-area5",
        classes=S3DIS_CLASSES,
        author="openpoints",
        license="MIT",
    ),
    transform=_S3DIS_TRANSFORMS,
    hparams={**_S3DIS_COMMON_HPARAMS, **_S3DIS_VARIANT_HPARAMS["lg"]},
)
def pointnext_lg_s3dis_area5_seg(**hparams: Any) -> PointNeXtSegmentation:
    return PointNeXtSegmentation(**hparams)


@register_model(
    "pointnext-lg.s3dis-area6.openpoints",
    task="segmentation",
    weights=WeightsDict(
        url="hf://torch-pointcloud/pointnext-lg.s3dis-area6.openpoints/resolve/main/model.safetensors",
        dataset="s3dis-area6",
        classes=S3DIS_CLASSES,
        author="openpoints",
        license="MIT",
    ),
    transform=_S3DIS_TRANSFORMS,
    hparams={**_S3DIS_COMMON_HPARAMS, **_S3DIS_VARIANT_HPARAMS["lg"]},
)
def pointnext_lg_s3dis_area6_seg(**hparams: Any) -> PointNeXtSegmentation:
    return PointNeXtSegmentation(**hparams)


@register_model(
    "pointnext-xl.s3dis-area1.openpoints",
    task="segmentation",
    weights=WeightsDict(
        url="hf://torch-pointcloud/pointnext-xl.s3dis-area1.openpoints/resolve/main/model.safetensors",
        dataset="s3dis-area1",
        classes=S3DIS_CLASSES,
        author="openpoints",
        license="MIT",
    ),
    transform=_S3DIS_TRANSFORMS,
    hparams={**_S3DIS_COMMON_HPARAMS, **_S3DIS_VARIANT_HPARAMS["xl"]},
)
def pointnext_xl_s3dis_area1_seg(**hparams: Any) -> PointNeXtSegmentation:
    return PointNeXtSegmentation(**hparams)


@register_model(
    "pointnext-xl.s3dis-area2.openpoints",
    task="segmentation",
    weights=WeightsDict(
        url="hf://torch-pointcloud/pointnext-xl.s3dis-area2.openpoints/resolve/main/model.safetensors",
        dataset="s3dis-area2",
        classes=S3DIS_CLASSES,
        author="openpoints",
        license="MIT",
    ),
    transform=_S3DIS_TRANSFORMS,
    hparams={**_S3DIS_COMMON_HPARAMS, **_S3DIS_VARIANT_HPARAMS["xl"]},
)
def pointnext_xl_s3dis_area2_seg(**hparams: Any) -> PointNeXtSegmentation:
    return PointNeXtSegmentation(**hparams)


@register_model(
    "pointnext-xl.s3dis-area3.openpoints",
    task="segmentation",
    weights=WeightsDict(
        url="hf://torch-pointcloud/pointnext-xl.s3dis-area3.openpoints/resolve/main/model.safetensors",
        dataset="s3dis-area3",
        classes=S3DIS_CLASSES,
        author="openpoints",
        license="MIT",
    ),
    transform=_S3DIS_TRANSFORMS,
    hparams={**_S3DIS_COMMON_HPARAMS, **_S3DIS_VARIANT_HPARAMS["xl"]},
)
def pointnext_xl_s3dis_area3_seg(**hparams: Any) -> PointNeXtSegmentation:
    return PointNeXtSegmentation(**hparams)


@register_model(
    "pointnext-xl.s3dis-area4.openpoints",
    task="segmentation",
    weights=WeightsDict(
        url="hf://torch-pointcloud/pointnext-xl.s3dis-area4.openpoints/resolve/main/model.safetensors",
        dataset="s3dis-area4",
        classes=S3DIS_CLASSES,
        author="openpoints",
        license="MIT",
    ),
    transform=_S3DIS_TRANSFORMS,
    hparams={**_S3DIS_COMMON_HPARAMS, **_S3DIS_VARIANT_HPARAMS["xl"]},
)
def pointnext_xl_s3dis_area4_seg(**hparams: Any) -> PointNeXtSegmentation:
    return PointNeXtSegmentation(**hparams)


@register_model(
    "pointnext-xl.s3dis-area5.openpoints",
    task="segmentation",
    weights=WeightsDict(
        url="hf://torch-pointcloud/pointnext-xl.s3dis-area5.openpoints/resolve/main/model.safetensors",
        dataset="s3dis-area5",
        classes=S3DIS_CLASSES,
        author="openpoints",
        license="MIT",
    ),
    transform=_S3DIS_TRANSFORMS,
    hparams={**_S3DIS_COMMON_HPARAMS, **_S3DIS_VARIANT_HPARAMS["xl"]},
)
def pointnext_xl_s3dis_area5_seg(**hparams: Any) -> PointNeXtSegmentation:
    return PointNeXtSegmentation(**hparams)


@register_model(
    "pointnext-xl.s3dis-area6.openpoints",
    task="segmentation",
    # No converted checkpoint exists for the xl / Area 6 variant, so it is registered without pretrained weights.
    weights=None,
    transform=_S3DIS_TRANSFORMS,
    hparams={**_S3DIS_COMMON_HPARAMS, **_S3DIS_VARIANT_HPARAMS["xl"]},
)
def pointnext_xl_s3dis_area6_seg(**hparams: Any) -> PointNeXtSegmentation:
    return PointNeXtSegmentation(**hparams)


_SHAPENETPART_TRANSFORMS = T.Compose(
    [
        T.CopyItems(keys=[DataKeys.POS, DataKeys.SEGMENT], names=[DataKeys.ORIGIN_POS, DataKeys.ORIGIN_SEGMENT]),
        T.FarthestPointSample(
            num_samples=2048,
            keys=[DataKeys.POS, DataKeys.NORMAL, DataKeys.SEGMENT],
            pos_key=DataKeys.POS,
            dst_index_key=DataKeys.INDEX,
        ),
        T.AxisMinOffset(keys=DataKeys.POS, axis=1, dst_keys="height"),
        T.Rescale(keys=[DataKeys.POS], method="centroid"),
        T.Cat(keys=[DataKeys.POS, DataKeys.NORMAL, "height"], dst_key=DataKeys.X),
        T.OneHot(keys=DataKeys.CATEGORY, num_classes=16),
    ]
)

_SHAPENETPART_COMMON_HPARAMS = dict(
    in_channels=7,
    num_classes=50,
    num_categories=16,
    spatial_dim=3,
    stem_plain_last=True,
    encoder_depths=[0, 0, 0, 0],
    encoder_expansion=4,
    sa_layers=3,
    sa_use_res=True,
    decoder_depths=[2, 2, 2, 2],
    decoder_plain_last=False,
    ratios=[0.5, 0.5, 0.5, 0.5],
    radiuses=[0.1, 0.25, 0.625, 1.5625, 3.906],
    num_neighbors=[32, 32, 32, 32, 32],
    act="relu",
    act_first=False,
    norm="batch_norm",
    bias=True,
    add_self_loops=False,
)

_SHAPENETPART_VARIANT_HPARAMS = {
    "sm": dict(
        stem_channels=32,
        encoder_channels=[64, 128, 256, 512],
        decoder_channels=[256, 128, 64, 32],
        head_channels=[96],
    ),
    "sm-c64": dict(
        stem_channels=64,
        encoder_channels=[128, 256, 512, 1024],
        decoder_channels=[512, 256, 128, 64],
        head_channels=[192],
    ),
    "sm-c160": dict(
        stem_channels=160,
        encoder_channels=[320, 640, 1280, 2560],
        decoder_channels=[1280, 640, 320, 160],
        head_channels=[480],
    ),
}


@register_model(
    "pointnext-sm.shapenetpart.openpoints",
    task="segmentation",
    weights=WeightsDict(
        url="hf://torch-pointcloud/pointnext-sm.shapenetpart.openpoints/resolve/main/model.safetensors",
        dataset="shapenetpart",
        author="openpoints",
        license="MIT",
    ),
    transform=_SHAPENETPART_TRANSFORMS,
    hparams={**_SHAPENETPART_COMMON_HPARAMS, **_SHAPENETPART_VARIANT_HPARAMS["sm"]},
)
def pointnext_sm_shapenetpart(**hparams: Any) -> PointNeXtPartSegmentation:
    return PointNeXtPartSegmentation(**hparams)


@register_model(
    "pointnext-sm-c64.shapenetpart.openpoints",
    task="segmentation",
    weights=WeightsDict(
        url="hf://torch-pointcloud/pointnext-sm-c64.shapenetpart.openpoints/resolve/main/model.safetensors",
        dataset="shapenetpart",
        author="openpoints",
        license="MIT",
    ),
    transform=_SHAPENETPART_TRANSFORMS,
    hparams={**_SHAPENETPART_COMMON_HPARAMS, **_SHAPENETPART_VARIANT_HPARAMS["sm-c64"]},
)
def pointnext_sm_c64_shapenetpart(**hparams: Any) -> PointNeXtPartSegmentation:
    return PointNeXtPartSegmentation(**hparams)


@register_model(
    "pointnext-sm-c160.shapenetpart.openpoints",
    task="segmentation",
    weights=WeightsDict(
        url="hf://torch-pointcloud/pointnext-sm-c160.shapenetpart.openpoints/resolve/main/model.safetensors",
        dataset="shapenetpart",
        author="openpoints",
        license="MIT",
    ),
    transform=_SHAPENETPART_TRANSFORMS,
    hparams={**_SHAPENETPART_COMMON_HPARAMS, **_SHAPENETPART_VARIANT_HPARAMS["sm-c160"]},
)
def pointnext_sm_c160_shapenetpart(**hparams: Any) -> PointNeXtPartSegmentation:
    return PointNeXtPartSegmentation(**hparams)
