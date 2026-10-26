"""CurveNet classification and part segmentation models.

{{ paper("2105.01288") }}
"""

from typing import Any, Callable, Dict, List, Literal, Optional, Sequence, Tuple, Union, overload

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn import MLP, global_max_pool, global_mean_pool

import torch_pointcloud.transforms as T
from torch_pointcloud.datasets.modelnet import MODELNET40_CLASSES
from torch_pointcloud.layers import (
    CatPool,
    CurveInterventionConv,
    CurveNetFeaturePropagation,
    LocalPointFeatureAggregation,
    PoolLike,
    create_pool,
)
from torch_pointcloud.ops.cluster import knn_table
from torch_pointcloud.utils.conversion import ensure_list, ensure_list_size, is_iterable
from torch_pointcloud.utils.data import DataKeys
from torch_pointcloud.utils.types import FeaturesDict, OptTensor

from ._base import ClassificationModel, PartSegmentationModel
from ._registry import WeightsDict, register_model


class CurveNetEncoder(nn.Module):
    r"""Stack of `CurveInterventionConv` stages.

    The first block of every stage subsamples to `num_points` (when set) with a bottleneck ratio of 2 and maps the
    stage input to its width; the following blocks keep the points with a bottleneck ratio of 4.

    Args:
        in_channels: Number of input channels.
        channels: Output channels of every stage.
        depths: Number of blocks of every stage.
        num_points: Points per sample after every stage's subsampling, `None` to keep the points.
        radii: Ball radius of every stage's subsampling, `None` with `num_points`.
        num_neighbors: Number of neighbors of every stage.
        num_curves: Number of curves per sample of every stage, `None` for stages without curves.
        curve_lengths: Number of points on the curves of every stage, `None` for stages without curves.
        num_layers: Depth of the local aggregation MLP.
        spatial_dim: Dimension $d$ of the positions.
        act: Activation function.
        act_kwargs: Extra arguments for the activation.
        norm: Normalization layer.
        norm_kwargs: Extra arguments for the normalization.
    """

    def __init__(
        self,
        in_channels: int,
        channels: Sequence[int],
        depths: Sequence[int],
        num_points: Sequence[Optional[int]],
        radii: Sequence[Optional[float]],
        num_neighbors: Sequence[int],
        num_curves: Sequence[Optional[int]],
        curve_lengths: Sequence[Optional[int]],
        num_layers: int = 2,
        spatial_dim: int = 3,
        act: Union[str, Callable, None] = "relu",
        act_kwargs: Optional[Dict[str, Any]] = None,
        norm: Union[str, Callable, None] = "batch_norm",
        norm_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()
        self.channels = ensure_list(channels)
        self.stages = nn.ModuleList()
        for i, out_channels in enumerate(self.channels):
            stage = nn.ModuleList()
            for j in range(depths[i]):
                block = CurveInterventionConv(
                    in_channels=in_channels if j == 0 else out_channels,
                    out_channels=out_channels,
                    num_neighbors=num_neighbors[i],
                    bottleneck_ratio=2 if j == 0 else 4,
                    num_layers=num_layers,
                    num_points=num_points[i] if j == 0 else None,
                    radius=radii[i] if j == 0 else None,
                    num_curves=num_curves[i],
                    curve_length=curve_lengths[i],
                    spatial_dim=spatial_dim,
                    act=act,
                    act_kwargs=act_kwargs,
                    norm=norm,
                    norm_kwargs=norm_kwargs,
                )
                stage.append(block)
            self.stages.append(stage)
            in_channels = out_channels

    @overload
    def forward(
        self,
        x: Tensor,
        pos: Tensor,
        batch: Tensor,
        return_intermediates: Literal[True],
    ) -> Tuple[Tensor, Tensor, Tensor, List[FeaturesDict]]: ...

    @overload
    def forward(
        self,
        x: Tensor,
        pos: Tensor,
        batch: Tensor,
        return_intermediates: Literal[False] = False,
    ) -> Tuple[Tensor, Tensor, Tensor]: ...

    def forward(self, x: Tensor, pos: Tensor, batch: Tensor, return_intermediates: bool = False) -> Any:
        intermediates: List[FeaturesDict] = []
        for i, stage in enumerate(self.stages):
            assert isinstance(stage, nn.ModuleList)
            for block in stage:
                x, pos, batch = block(x, pos, batch)
            if i < len(self.stages) - 1:
                intermediates.append(FeaturesDict(x=x, pos=pos, batch=batch))

        if return_intermediates:
            return x, pos, batch, intermediates
        return x, pos, batch


class CurveNetDecoder(nn.Module):
    r"""Feature propagation path of CurveNet: a `CurveNetFeaturePropagation` per encoder skip, each but the last
    followed by a `CurveInterventionConv` block at the restored resolution.

    Args:
        in_channels: Number of channels of the coarsest encoder features.
        skip_channels: Channels of the encoder skips, coarse to fine.
        channels: Output channels of every level, coarse to fine.
        gate_channels: Attention gate width of every level.
        num_neighbors: Number of neighbors of every level's block, coarse to fine.
        num_layers: Depth of the local aggregation MLP.
        spatial_dim: Dimension $d$ of the positions.
        act: Activation function.
        act_kwargs: Extra arguments for the activation.
        norm: Normalization layer.
        norm_kwargs: Extra arguments for the normalization.
    """

    def __init__(
        self,
        in_channels: int,
        skip_channels: Sequence[int],
        channels: Sequence[int],
        gate_channels: Sequence[int],
        num_neighbors: Sequence[int],
        num_layers: int = 2,
        spatial_dim: int = 3,
        act: Union[str, Callable, None] = "relu",
        act_kwargs: Optional[Dict[str, Any]] = None,
        norm: Union[str, Callable, None] = "batch_norm",
        norm_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()
        self.channels = ensure_list(channels)
        self.fps = nn.ModuleList()
        self.blocks = nn.ModuleList()
        for i, out_channels in enumerate(self.channels):
            fp = CurveNetFeaturePropagation(
                in_channels=in_channels,
                skip_channels=skip_channels[i],
                channels=[out_channels, out_channels],
                gate_channels=gate_channels[i],
                act=act,
                act_kwargs=act_kwargs,
                norm=norm,
                norm_kwargs=norm_kwargs,
            )
            self.fps.append(fp)
            if i < len(self.channels) - 1:
                block = CurveInterventionConv(
                    in_channels=out_channels,
                    out_channels=out_channels,
                    num_neighbors=num_neighbors[i],
                    bottleneck_ratio=4,
                    num_layers=num_layers,
                    spatial_dim=spatial_dim,
                    act=act,
                    act_kwargs=act_kwargs,
                    norm=norm,
                    norm_kwargs=norm_kwargs,
                )
                self.blocks.append(block)
            in_channels = out_channels

    def forward(
        self,
        x: Tensor,
        pos: Tensor,
        batch: Tensor,
        intermediates: List[FeaturesDict],
    ) -> Tuple[Tensor, Tensor, Tensor]:
        for i, fp in enumerate(self.fps):
            skip = intermediates[-1 - i]
            x, pos, batch = fp(x, pos, batch, skip["x"], skip["pos"], skip["batch"])
            if i < len(self.blocks):
                x, pos, batch = self.blocks[i](x, pos, batch)
        return x, pos, batch


class CurveNetClassification(ClassificationModel):
    r"""
    Classification model as described in the paper
    :arxiv: [Walk in the Cloud: Learning Curves for Point Clouds Shape Analysis](https://arxiv.org/abs/2105.01288)
    by Tiange Xiang, Chaoyi Zhang, Yang Song, Jianhui Yu, Weidong Cai.

    CurveNet walks curves through the point cloud, starting from the points a learned gate scores highest and
    following a neighbor the agent picks at every step, and aggregates the curve descriptors back into the point
    features between the local aggregations of its residual blocks (`CurveInterventionConv`).

    Args:
        in_channels: Number of input feature channels; `0` embeds the coordinates alone.
        num_classes: Number of output classes.
        spatial_dim: Spatial dimension of the input point cloud.
        embed_channels: Width of the input `LocalPointFeatureAggregation` MLP.
        embed_num_layers: Depth of the input `LocalPointFeatureAggregation` MLP.
        channels: Output channels of every encoder stage.
        depths: Number of blocks of every encoder stage.
        num_points: Points per sample after every stage's subsampling, `None` to keep the points.
        radii: Ball radius of every stage's subsampling, `None` with `num_points`.
        num_neighbors: Number of neighbors of every stage; the input embedding uses the first.
        num_curves: Number of curves per sample of every stage, `None` for stages without curves.
        curve_lengths: Number of points on the curves of every stage, `None` for stages without curves.
        num_layers: Depth of the local aggregation MLP of the encoder blocks.
        proj_channels: Width of the projection applied before pooling.
        head_channels: Hidden channels of the classification head.
        act: Activation of the encoder.
        act_kwargs: Extra arguments for `act`.
        head_act: Activation of the projection and the head.
        head_act_kwargs: Extra arguments for `head_act`.
        norm: Normalization layer type.
        norm_kwargs: Additional arguments for the normalization layer.
        bias: Whether the projection before pooling uses a bias; the head layers always do.
        dropout: Dropout probability after every hidden layer of the head.
        global_pool: Global pooling operation for final feature aggregation.
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        *,
        spatial_dim: int = 3,
        embed_channels: int,
        embed_num_layers: int = 1,
        channels: Sequence[int],
        depths: Union[int, Sequence[int]],
        num_points: Sequence[Optional[int]],
        radii: Sequence[Optional[float]],
        num_neighbors: Union[int, Sequence[int]],
        num_curves: Sequence[Optional[int]],
        curve_lengths: Sequence[Optional[int]],
        num_layers: int = 2,
        proj_channels: int,
        head_channels: Optional[Union[int, Sequence[int]]] = None,
        act: Union[str, Callable, None] = "relu",
        act_kwargs: Optional[Dict[str, Any]] = None,
        head_act: Union[str, Callable, None] = "relu",
        head_act_kwargs: Optional[Dict[str, Any]] = None,
        norm: Union[str, Callable, None] = "batch_norm",
        norm_kwargs: Optional[Dict[str, Any]] = None,
        bias: bool = True,
        dropout: float = 0.0,
        global_pool: PoolLike | Sequence[PoolLike] = "max",
    ):
        super().__init__(in_channels=in_channels, num_classes=num_classes)
        self.spatial_dim = spatial_dim
        self.embed_channels = embed_channels
        self.embed_num_layers = embed_num_layers
        self.channels = ensure_list(channels)
        self.depths = ensure_list_size(depths, size=len(self.channels))
        self.num_points = ensure_list_size(num_points, size=len(self.channels))
        self.radii = ensure_list_size(radii, size=len(self.channels))
        self.num_neighbors = ensure_list_size(num_neighbors, size=len(self.channels))
        self.num_curves = ensure_list_size(num_curves, size=len(self.channels))
        self.curve_lengths = ensure_list_size(curve_lengths, size=len(self.channels))
        self.num_layers = num_layers
        self.proj_channels = proj_channels
        self.head_channels = ensure_list(head_channels, none_as_empty=True)
        self.act = act
        self.act_kwargs = act_kwargs
        self.head_act = head_act
        self.head_act_kwargs = head_act_kwargs
        self.norm = norm
        self.norm_kwargs = norm_kwargs
        self.bias = bias
        self.dropout = dropout

        self.embed = self.configure_embed()
        self.encoder = self.configure_encoder()
        self.proj = self.configure_proj()
        self.global_pool = CatPool(global_pool) if is_iterable(global_pool) else create_pool(global_pool)  # type: ignore[arg-type]
        self.head = self.configure_head()

    def configure_embed(self) -> LocalPointFeatureAggregation:
        """Build the input `LocalPointFeatureAggregation`, max-pooled over the neighbors."""
        return LocalPointFeatureAggregation(
            self.in_channels,
            self.embed_channels,
            num_layers=self.embed_num_layers,
            aggr="max",
            spatial_dim=self.spatial_dim,
            act=self.act,
            act_kwargs=self.act_kwargs,
            norm=self.norm,
            norm_kwargs=self.norm_kwargs,
        )

    def configure_encoder(self) -> CurveNetEncoder:
        """Build the `CurveNetEncoder` backbone."""
        return CurveNetEncoder(
            in_channels=self.embed_channels,
            channels=self.channels,
            depths=self.depths,
            num_points=self.num_points,
            radii=self.radii,
            num_neighbors=self.num_neighbors,
            num_curves=self.num_curves,
            curve_lengths=self.curve_lengths,
            num_layers=self.num_layers,
            spatial_dim=self.spatial_dim,
            act=self.act,
            act_kwargs=self.act_kwargs,
            norm=self.norm,
            norm_kwargs=self.norm_kwargs,
        )

    def configure_proj(self) -> MLP:
        """Build the projection applied to every point before pooling."""
        return MLP(
            [self.channels[-1], self.proj_channels],
            plain_last=False,
            act=self.head_act,
            act_kwargs=self.head_act_kwargs,
            norm=self.norm,
            norm_kwargs=self.norm_kwargs,
            bias=self.bias,
        )

    @property
    def num_features(self) -> int:
        """Channel count $C$ of the pooled features entering the head."""
        return (
            self.proj_channels * self.global_pool.num_pools
            if isinstance(self.global_pool, CatPool)
            else self.proj_channels
        )

    def configure_head(self) -> nn.Module:
        if self.num_classes == 0:
            return nn.Identity()

        channels_list = [self.num_features] + self.head_channels + [self.num_classes]
        dropout_list = [self.dropout] * (len(channels_list) - 1)
        dropout_list[-1] = 0.0
        return MLP(
            channels_list,
            act=self.head_act,
            act_kwargs=self.head_act_kwargs,
            norm=self.norm,
            norm_kwargs=self.norm_kwargs,
            bias=True,
            dropout=dropout_list,
            plain_last=True,
        )

    def reset_classifier(
        self,
        num_classes: int,
        global_pool: PoolLike | Sequence[PoolLike] | None = None,
        **kwargs: Any,
    ) -> None:
        self.num_classes = num_classes
        if global_pool is not None:
            self.global_pool = CatPool(global_pool) if is_iterable(global_pool) else create_pool(global_pool)  # type: ignore[arg-type]
        self.head = self.configure_head()

    def forward_features(self, x: OptTensor, pos: Tensor, batch: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        table = knn_table(pos, pos, self.num_neighbors[0], batch_x=batch, batch_y=batch)
        x = self.embed(x, pos, table)
        x, pos, batch = self.encoder(x, pos, batch)
        x = self.proj(x)
        return x, pos, batch

    def forward_head(self, x: Tensor, batch: Tensor, pre_logits: bool = False) -> Tensor:
        x = self.global_pool(x, batch)
        return x if pre_logits else self.head(x)

    def forward(self, x: OptTensor, pos: Tensor, batch: Tensor) -> Tensor:
        x, _, batch = self.forward_features(x, pos, batch)
        return self.forward_head(x, batch)


class CurveNetPartSegmentation(PartSegmentationModel):
    r"""
    Part segmentation model as described in the paper
    :arxiv: [Walk in the Cloud: Learning Curves for Point Clouds Shape Analysis](https://arxiv.org/abs/2105.01288)
    by Tiange Xiang, Chaoyi Zhang, Yang Song, Jianhui Yu, Weidong Cai.

    The `CurveNetEncoder` is upsampled by a `CurveNetDecoder`; the restored points are concatenated with their
    coordinates, the max-pooled embeddings of the coarsest encoder stages and the one-hot category, refined by two
    `CurveInterventionConv` blocks and scored by a head with a squeeze-and-excitation gate.

    Args:
        in_channels: Number of input feature channels; `0` embeds the coordinates alone.
        num_classes: Number of part classes.
        num_categories: Number of object categories of the one-hot `category` input.
        spatial_dim: Spatial dimension of the input point cloud.
        embed_channels: Width of the input `LocalPointFeatureAggregation` MLP.
        embed_num_layers: Depth of the input `LocalPointFeatureAggregation` MLP.
        channels: Output channels of every encoder stage.
        depths: Number of blocks of every encoder stage.
        num_points: Points per sample after every stage's subsampling, `None` to keep the points.
        radii: Ball radius of every stage's subsampling, `None` with `num_points`.
        num_neighbors: Number of neighbors of every encoder stage. The input embedding and the fuse blocks use the
            first, and every decoder level uses the one of the encoder stage it restores.
        num_curves: Number of curves per sample of every stage, `None` for stages without curves.
        curve_lengths: Number of points on the curves of every stage, `None` for stages without curves.
        num_layers: Depth of the local aggregation MLP of the encoder and decoder blocks.
        decoder_channels: Output channels of every decoder level, coarse to fine.
        gate_channels: Attention gate width of every decoder level.
        global_channels: Width of the global embedding of each of the last encoder stages.
        fuse_channels: Output channels of the blocks refining the conditioned points.
        head_channels: Width of the head projection.
        se_ratio: Width reduction of the squeeze-and-excitation gate.
        act: Activation function.
        act_kwargs: Extra arguments for `act`.
        norm: Normalization layer type.
        norm_kwargs: Additional arguments for the normalization layer.
        bias: Whether the global embedding, the head projection and the squeeze-and-excitation layers use a bias.
        dropout: Dropout probability before the classifier.
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        *,
        num_categories: int,
        spatial_dim: int = 3,
        embed_channels: int,
        embed_num_layers: int = 1,
        channels: Sequence[int],
        depths: Union[int, Sequence[int]],
        num_points: Sequence[Optional[int]],
        radii: Sequence[Optional[float]],
        num_neighbors: Union[int, Sequence[int]],
        num_curves: Sequence[Optional[int]],
        curve_lengths: Sequence[Optional[int]],
        num_layers: int = 2,
        decoder_channels: Sequence[int],
        gate_channels: Sequence[int],
        global_channels: Sequence[int],
        fuse_channels: Sequence[int],
        head_channels: int,
        se_ratio: int,
        act: Union[str, Callable, None] = "relu",
        act_kwargs: Optional[Dict[str, Any]] = None,
        norm: Union[str, Callable, None] = "batch_norm",
        norm_kwargs: Optional[Dict[str, Any]] = None,
        bias: bool = True,
        dropout: float = 0.0,
    ):
        if len(decoder_channels) != len(channels) - 1:
            raise ValueError(
                f"`decoder_channels` must hold one level per encoder skip ({len(channels) - 1}), got {decoder_channels}."
            )

        super().__init__(in_channels=in_channels, num_classes=num_classes, num_categories=num_categories)
        self.spatial_dim = spatial_dim
        self.embed_channels = embed_channels
        self.embed_num_layers = embed_num_layers
        self.channels = ensure_list(channels)
        self.depths = ensure_list_size(depths, size=len(self.channels))
        self.num_points = ensure_list_size(num_points, size=len(self.channels))
        self.radii = ensure_list_size(radii, size=len(self.channels))
        self.num_neighbors = ensure_list_size(num_neighbors, size=len(self.channels))
        self.num_curves = ensure_list_size(num_curves, size=len(self.channels))
        self.curve_lengths = ensure_list_size(curve_lengths, size=len(self.channels))
        self.num_layers = num_layers
        self.decoder_channels = ensure_list(decoder_channels)
        self.gate_channels = ensure_list_size(gate_channels, size=len(self.decoder_channels))
        self.global_channels = ensure_list(global_channels)
        self.fuse_channels = ensure_list(fuse_channels)
        self.head_channels = head_channels
        self.se_ratio = se_ratio
        self.act = act
        self.act_kwargs = act_kwargs
        self.norm = norm
        self.norm_kwargs = norm_kwargs
        self.bias = bias
        self.dropout = dropout

        self.embed = self.configure_embed()
        self.encoder = self.configure_encoder()
        self.decoder = self.configure_decoder()
        self.global_embed = self.configure_global_embed()
        self.fuse = self.configure_fuse()
        self.proj = self.configure_proj()
        self.se = self.configure_se()
        self.head = self.configure_head()

    def configure_embed(self) -> LocalPointFeatureAggregation:
        """Build the input `LocalPointFeatureAggregation`, max-pooled over the neighbors."""
        return LocalPointFeatureAggregation(
            self.in_channels,
            self.embed_channels,
            num_layers=self.embed_num_layers,
            aggr="max",
            spatial_dim=self.spatial_dim,
            act=self.act,
            act_kwargs=self.act_kwargs,
            norm=self.norm,
            norm_kwargs=self.norm_kwargs,
        )

    def configure_encoder(self) -> CurveNetEncoder:
        """Build the `CurveNetEncoder` backbone."""
        return CurveNetEncoder(
            in_channels=self.embed_channels,
            channels=self.channels,
            depths=self.depths,
            num_points=self.num_points,
            radii=self.radii,
            num_neighbors=self.num_neighbors,
            num_curves=self.num_curves,
            curve_lengths=self.curve_lengths,
            num_layers=self.num_layers,
            spatial_dim=self.spatial_dim,
            act=self.act,
            act_kwargs=self.act_kwargs,
            norm=self.norm,
            norm_kwargs=self.norm_kwargs,
        )

    def configure_decoder(self) -> CurveNetDecoder:
        """Build the `CurveNetDecoder` upsampling the coarsest features back through the encoder skips."""
        return CurveNetDecoder(
            in_channels=self.channels[-1],
            skip_channels=self.channels[-2::-1],
            channels=self.decoder_channels,
            gate_channels=self.gate_channels,
            num_neighbors=self.num_neighbors[-2::-1],
            num_layers=self.num_layers,
            spatial_dim=self.spatial_dim,
            act=self.act,
            act_kwargs=self.act_kwargs,
            norm=self.norm,
            norm_kwargs=self.norm_kwargs,
        )

    def configure_global_embed(self) -> nn.ModuleList:
        """Build the MLPs embedding the last encoder stages before their global max pooling."""
        stage_channels = self.channels[len(self.channels) - len(self.global_channels) :]
        return nn.ModuleList(
            MLP(
                [in_channels, out_channels],
                plain_last=False,
                act=self.act,
                act_kwargs=self.act_kwargs,
                norm=self.norm,
                norm_kwargs=self.norm_kwargs,
                bias=self.bias,
            )
            for in_channels, out_channels in zip(stage_channels, self.global_channels)
        )

    def configure_fuse(self) -> nn.ModuleList:
        """Build the `CurveInterventionConv` blocks refining the points conditioned on the global embedding."""
        blocks = nn.ModuleList()
        in_channels = self.spatial_dim + self.decoder_channels[-1] + sum(self.global_channels) + self.num_categories
        for out_channels in self.fuse_channels:
            block = CurveInterventionConv(
                in_channels=in_channels,
                out_channels=out_channels,
                num_neighbors=self.num_neighbors[0],
                bottleneck_ratio=4,
                num_layers=self.num_layers,
                spatial_dim=self.spatial_dim,
                act=self.act,
                act_kwargs=self.act_kwargs,
                norm=self.norm,
                norm_kwargs=self.norm_kwargs,
            )
            blocks.append(block)
            in_channels = out_channels
        return blocks

    def configure_proj(self) -> MLP:
        """Build the projection of the head."""
        return MLP(
            [self.fuse_channels[-1], self.head_channels],
            plain_last=False,
            act=self.act,
            act_kwargs=self.act_kwargs,
            norm=self.norm,
            norm_kwargs=self.norm_kwargs,
            bias=self.bias,
        )

    def configure_se(self) -> MLP:
        """Build the squeeze-and-excitation MLP scoring the mean-pooled head features."""
        return MLP(
            [self.head_channels, self.head_channels // self.se_ratio, self.head_channels],
            plain_last=True,
            act=self.act,
            act_kwargs=self.act_kwargs,
            norm=self.norm,
            norm_kwargs=self.norm_kwargs,
            bias=self.bias,
        )

    @property
    def num_features(self) -> int:
        """Channel count $C$ of the gated head features entering the classifier."""
        return self.head_channels

    def configure_head(self) -> nn.Module:
        if self.num_classes == 0:
            return nn.Identity()
        return nn.Linear(self.num_features, self.num_classes)

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
    ) -> Tuple[Tensor, Tensor, Tensor, List[FeaturesDict]]: ...

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
        table = knn_table(pos, pos, self.num_neighbors[0], batch_x=batch, batch_y=batch)
        x = self.embed(x, pos, table)
        return self.encoder(x, pos, batch, return_intermediates=return_intermediates)

    def forward_decoder(
        self,
        x: Tensor,
        pos: Tensor,
        batch: Tensor,
        category: Tensor,
        intermediates: List[FeaturesDict],
    ) -> Tuple[Tensor, Tensor, Tensor]:
        stages = intermediates[len(intermediates) - len(self.global_embed) + 1 :] + [
            FeaturesDict(x=x, pos=pos, batch=batch)
        ]
        embeddings = [
            global_max_pool(mlp(stage["x"]), stage["batch"], size=category.size(0))
            for mlp, stage in zip(self.global_embed, stages)
        ]
        embedding = torch.cat(embeddings + [category], dim=1)  # (B, sum(global_channels) + num_categories)

        x, pos, batch = self.decoder(x, pos, batch, intermediates)
        x = torch.cat([pos, x, embedding.index_select(0, batch)], dim=1)
        for block in self.fuse:
            x, pos, batch = block(x, pos, batch)
        return x, pos, batch

    def forward_head(self, x: Tensor, batch: Tensor, pre_logits: bool = False) -> Tensor:
        x = self.proj(x)
        gate = torch.sigmoid(self.se(global_mean_pool(x, batch)))  # (B, C)
        x = x * gate.index_select(0, batch)
        x = F.dropout(x, p=self.dropout, training=self.training)
        return x if pre_logits else self.head(x)

    def forward(self, x: OptTensor, pos: Tensor, batch: Tensor, category: Tensor) -> Tensor:
        x, pos, batch, intermediates = self.forward_features(x, pos, batch, return_intermediates=True)
        x, pos, batch = self.forward_decoder(x, pos, batch, category, intermediates)
        return self.forward_head(x, batch)


@register_model(
    "curvenet.modelnet40.tiange-xiang",
    task="classification",
    weights=WeightsDict(
        url="hf://torch-pointcloud/curvenet.modelnet40.tiange-xiang/resolve/0000000000000000000000000000000000000000/model.safetensors",
        dataset="modelnet40",
        metrics={"OA": 93.84, "mAcc": 91.14},
        classes=MODELNET40_CLASSES,
        author="tiange-xiang",
        license="MIT",
    ),
    hparams=dict(
        in_channels=0,
        num_classes=40,
        spatial_dim=3,
        embed_channels=32,
        embed_num_layers=1,
        channels=[64, 128, 256, 512],
        depths=2,
        num_points=[None, None, 256, 64],
        radii=[None, None, 0.1, 0.2],
        num_neighbors=20,
        num_curves=[100, 100, None, None],
        curve_lengths=[5, 5, None, None],
        num_layers=1,
        proj_channels=1024,
        head_channels=512,
        act="leaky_relu",
        act_kwargs={"negative_slope": 0.2},
        head_act="relu",
        norm="batch_norm",
        bias=False,
        dropout=0.5,
        global_pool=["max", "mean"],
    ),
    transform=T.Compose(
        [
            T.CopyItems(keys=DataKeys.POS, dst_keys=DataKeys.ORIGIN_POS),
            T.FarthestPointSample(
                pos_key=DataKeys.POS,
                keys=[DataKeys.NORMAL],
                num_samples=1024,
                dst_index_key=DataKeys.INDEX,
                allow_missing_keys=True,
            ),
        ]
    ),
)
def curvenet_tiange_xiang_modelnet40_cls(**hparams: Any) -> CurveNetClassification:
    # from the repo: https://github.com/tiangexiang/CurveNet
    return CurveNetClassification(**hparams)


@register_model(
    "curvenet.shapenetpart.tiange-xiang",
    task="part-segmentation",
    input_keys=("x", "pos", "batch", "category"),
    weights=WeightsDict(
        url="hf://torch-pointcloud/curvenet.shapenetpart.tiange-xiang/resolve/0000000000000000000000000000000000000000/model.safetensors",
        dataset="shapenetpart",
        metrics={"ins_mIoU": 86.67, "cls_mIoU": 83.79},
        author="tiange-xiang",
        license="MIT",
    ),
    hparams=dict(
        in_channels=0,
        num_classes=50,
        num_categories=16,
        spatial_dim=3,
        embed_channels=32,
        embed_num_layers=1,
        channels=[64, 128, 256, 512, 1024],
        depths=[2, 2, 2, 2, 3],
        num_points=[None, 512, 128, 32, 8],
        radii=[None, 0.4, 0.8, 1.2, 2.0],
        num_neighbors=[32, 32, 32, 31, 7],
        num_curves=[100, 100, None, None, None],
        curve_lengths=[5, 5, None, None, None],
        num_layers=2,
        decoder_channels=[512, 256, 128, 64],
        gate_channels=[256, 128, 64, 32],
        global_channels=[64, 128],
        fuse_channels=[256, 256],
        head_channels=256,
        se_ratio=8,
        act="leaky_relu",
        act_kwargs={"negative_slope": 0.2},
        norm="batch_norm",
        bias=False,
        dropout=0.5,
    ),
    transform=T.Compose(
        [
            T.Rescale(keys=DataKeys.POS, method="centroid"),
            T.CopyItems(
                keys=[DataKeys.POS, DataKeys.SEGMENT],
                dst_keys=[DataKeys.ORIGIN_POS, DataKeys.ORIGIN_SEGMENT],
                allow_missing_keys=True,
            ),
            T.FarthestPointSample(
                keys=[DataKeys.NORMAL, DataKeys.SEGMENT],
                pos_key=DataKeys.POS,
                num_samples=2048,
                dst_index_key=DataKeys.INDEX,
                allow_missing_keys=True,
            ),
            T.OneHot(keys=DataKeys.CATEGORY, num_classes=16),
        ]
    ),
)
def curvenet_tiange_xiang_shapenetpart_partseg(**hparams: Any) -> CurveNetPartSegmentation:
    # from the repo: https://github.com/tiangexiang/CurveNet
    return CurveNetPartSegmentation(**hparams)
