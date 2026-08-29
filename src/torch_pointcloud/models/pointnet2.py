"""PointNet++ classification and segmentation models.

{{ paper("1706.02413") }}
"""

from typing import Any, Callable, Dict, List, Literal, Optional, Sequence, Tuple, Union, overload

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn import MLP

import torch_pointcloud.transforms as T
from torch_pointcloud.datasets.modelnet import MODELNET40_CLASSES
from torch_pointcloud.datasets.s3dis import S3DIS_CLASSES
from torch_pointcloud.datasets.scanobjectnn import SCANOBJECTNN_CLASSES
from torch_pointcloud.layers import PoolLike, create_pool
from torch_pointcloud.layers.pointnet2_blocks import FPModule, SAModule, ensure_msg_list
from torch_pointcloud.utils.conversion import ensure_list, ensure_tuple, ensure_tuple_size
from torch_pointcloud.utils.data import DataKeys
from torch_pointcloud.utils.types import OptTensor

from ._base import ClassificationModel, SegmentationModel
from ._registry import WeightsDict, register_model


class PointNet2Encoder(nn.Module):
    r"""PointNet++ encoder from the paper
    :arxiv: [PointNet++: Deep Hierarchical Feature Learning on Point Sets in a Metric Space](https://arxiv.org/abs/1706.02413)
    by Charles R. Qi, Li Yi, Hao Su, Leonidas J. Guibas.

    Processes raw point clouds through an optional linear stem followed by multiple Set Abstraction (SA)
    blocks that progressively downsample the points while learning local features using radius-based
    grouping. Each SA block can optionally use Multi-Scale Grouping (MSG).

    Args:
        in_channels: Number of input channels (features per point).
        sa_channels: List of channel configurations for Set Abstraction (SA) blocks.
            Each element defines the MLP channels for one SA block.
            For Multi-Scale Grouping (MSG), provide nested lists of channels.
        ratios: Sampling ratios for each SA block (between 0 and 1).
            Mutually exclusive with `num_points`.
        num_points: Absolute number of sampled centroids for each SA block (e.g. PointRCNN's fixed
            $4096, 1024, \ldots$). Exactly one of `ratios` / `num_points` must be given.
        radii: Search radiuses for each SA block's neighborhood.
            For MSG, provide a list of radii per block.
        num_neighbors: Max number of neighbors for each SA block.
            For MSG, provide a list of neighbor counts per block.
        stem_channels: Optional number of channels for initial linear projection.
        spatial_dim: Spatial dimensionality of point coordinates (e.g. 3 for 3D, 2 for 2D).
        act: Activation function type or callable.
        act_kwargs: Additional keyword arguments for the activation function.
        act_first: If `True`, activation is applied before normalization.
        norm: Normalization layer type or callable.
        norm_kwargs: Additional keyword arguments for the normalization layer.
        bias: Whether to use bias in linear layers.
        use_pos: Whether to concatenate per-point relative positions to `x`.
        pos_first: Concatenate the relative positions *before* the grouped features
            (see [`SAModule`][torch_pointcloud.layers.pointnet2_blocks.SAModule]).
        pool: Pooling operation for SA blocks.
    """

    def __init__(
        self,
        in_channels: int,
        sa_channels: Sequence[Sequence[Union[int, Sequence[int]]]],
        *,
        ratios: Optional[Sequence[float]] = None,
        num_points: Optional[Sequence[int]] = None,
        radii: Sequence[Union[float, Sequence[float]]],
        num_neighbors: Sequence[Union[int, Sequence[int]]],
        stem_channels: Optional[int] = None,
        spatial_dim: int = 3,
        act: Union[str, Callable, None] = "relu",
        act_kwargs: Optional[Dict[str, Any]] = None,
        act_first: bool = False,
        norm: Union[str, Callable, None] = "batch_norm",
        norm_kwargs: Optional[Dict[str, Any]] = None,
        bias: bool = False,
        use_pos: bool = True,
        normalize_pos: bool = True,
        pos_first: bool = False,
        pool: PoolLike = "max",
        sort_neighbors: bool = False,
    ) -> None:
        super().__init__()
        if (ratios is None) == (num_points is None):
            raise ValueError("`PointNet2Encoder` needs exactly one of `ratios` or `num_points`.")
        sa_channels = ensure_msg_list(
            sa_channels,
            extra_msg="The parameter `sa_channels` must be a sequence compliant with the Multi-Scale Grouping (MSG) mode.",
        )

        self.in_channels = in_channels
        self.spatial_dim = spatial_dim
        self.stem = nn.Linear(in_channels, stem_channels) if stem_channels else None
        encoder_in = stem_channels if stem_channels else in_channels

        num_blocks = len(sa_channels)
        extra_msg = f"The parameter `{{param}}` must be a sequence matching the number of blocks {num_blocks}."
        if ratios is not None:
            ratios = ensure_tuple_size(ratios, size=num_blocks, extra_msg=extra_msg.format(param="ratios"))
        if num_points is not None:
            num_points = ensure_tuple_size(num_points, size=num_blocks, extra_msg=extra_msg.format(param="num_points"))
        radii = ensure_tuple_size(radii, size=num_blocks, extra_msg=extra_msg.format(param="radii"))
        num_neighbors = ensure_tuple_size(num_neighbors, size=num_blocks, extra_msg=extra_msg.format(param="k"))

        ch = encoder_in
        self.sa_blocks = nn.ModuleList()
        sa_out_channels: List[int] = []
        for i in range(num_blocks):
            block = SAModule(
                in_channels=ch,
                channels=sa_channels[i],
                ratio=ratios[i] if ratios is not None else None,
                num_points=num_points[i] if num_points is not None else None,
                radii=radii[i],
                num_neighbors=num_neighbors[i],
                spatial_dim=spatial_dim,
                act=act,
                act_kwargs=act_kwargs,
                act_first=act_first,
                norm=norm,
                norm_kwargs=norm_kwargs,
                bias=bias,
                use_pos=use_pos,
                normalize_pos=normalize_pos,
                pos_first=pos_first,
                pool=pool,
                sort_neighbors=sort_neighbors,
            )
            self.sa_blocks.append(block)
            ch = sum(c[-1] for c in sa_channels[i])
            sa_out_channels.append(ch)

        self._skip_channels = [encoder_in, *sa_out_channels[:-1]]
        self._out_channels = sa_out_channels[-1]

    @property
    def out_channels(self) -> int:
        """Output channels of the last SA block."""
        return self._out_channels

    @property
    def skip_channels(self) -> List[int]:
        """Skip-connection channel sizes (stem output + each SA output except the last), ordered
        from finest to coarsest resolution."""
        return list(self._skip_channels)

    @overload
    def forward(
        self,
        x: OptTensor,
        pos: Tensor,
        batch: Tensor,
        return_intermediates: Literal[True],
    ) -> Tuple[Tensor, Tensor, Tensor, List[Dict[str, Tensor]]]: ...

    @overload
    def forward(
        self,
        x: OptTensor,
        pos: Tensor,
        batch: Tensor,
        return_intermediates: Literal[False] = False,
    ) -> Tuple[Tensor, Tensor, Tensor]: ...

    def forward(
        self,
        x: OptTensor,
        pos: Tensor,
        batch: Tensor,
        return_intermediates: bool = False,
    ) -> Any:
        if x is None:
            x = pos.new_empty((pos.size(0), 0)) if self.in_channels == 0 else pos
        if self.stem is not None:
            x = self.stem(x)

        intermediates = [{"x": x, "pos": pos, "batch": batch}] if return_intermediates else []
        for i, block in enumerate(self.sa_blocks):
            x, pos, batch = block(x, pos, batch)
            if return_intermediates and i < len(self.sa_blocks) - 1:
                intermediates.append({"x": x, "pos": pos, "batch": batch})

        if return_intermediates:
            return x, pos, batch, intermediates
        return x, pos, batch


class PointNet2Decoder(nn.Module):
    """PointNet++ decoder (feature propagation) from the paper
    :arxiv: [PointNet++: Deep Hierarchical Feature Learning on Point Sets in a Metric Space](https://arxiv.org/abs/1706.02413)
    by Charles R. Qi, Li Yi, Hao Su, Leonidas J. Guibas.

    Upsamples features from the encoder back to the original resolution using kNN interpolation
    and skip connections from encoder intermediates.

    Args:
        in_channels: Number of input channels from the encoder (or aggregation) output.
        skip_channels: Channel sizes for skip connections at each level, ordered from
            coarsest to finest resolution.
        fp_channels: List of channel configurations for Feature Propagation (FP) blocks.
            Each element defines the MLP channels for one FP block.
        spatial_dim: Spatial dimensionality of point coordinates. Also used as the default number
            of neighbors `k` for kNN interpolation in all but the first FP block.
        act: Activation function type or callable.
        act_kwargs: Additional keyword arguments for the activation function.
        act_first: If `True`, activation is applied before normalization.
        norm: Normalization layer type or callable.
        norm_kwargs: Additional keyword arguments for the normalization layer.
        bias: Whether to use bias in linear layers.
        k: Number of neighbors for kNN interpolation, per FP block when a sequence. Defaults to
            `1` for the first (deepest) block and `spatial_dim` for the remaining blocks.
    """

    def __init__(
        self,
        in_channels: int,
        skip_channels: Sequence[int],
        fp_channels: Sequence[Sequence[int]],
        *,
        spatial_dim: int = 3,
        act: Union[str, Callable, None] = "relu",
        act_kwargs: Optional[Dict[str, Any]] = None,
        act_first: bool = False,
        norm: Union[str, Callable, None] = "batch_norm",
        norm_kwargs: Optional[Dict[str, Any]] = None,
        bias: bool = False,
        k: Optional[Union[int, Sequence[int]]] = None,
        weighting: Literal["squared", "inverse"] = "squared",
        eps: float = 1e-16,
    ) -> None:
        super().__init__()
        if len(skip_channels) != len(fp_channels):
            raise ValueError(
                f"The number of skip channels ({len(skip_channels)}) must match "
                f"the number of feature propagation channels ({len(fp_channels)})."
            )

        num_blocks = len(fp_channels)
        if k is None:
            ks: List[int] = [1] + [spatial_dim] * (num_blocks - 1)
        elif isinstance(k, int):
            ks = [k] * num_blocks
        else:
            ks = list(k)
            if len(ks) != num_blocks:
                raise ValueError(f"Length of `k` ({len(ks)}) must match the number of FP blocks ({num_blocks}).")

        self.skip_channels = list(skip_channels)
        self.fp_blocks = nn.ModuleList()
        for i in range(num_blocks):
            ch = in_channels if i == 0 else fp_channels[i - 1][-1]
            block = FPModule(
                in_channels=ch + skip_channels[i],
                channels=fp_channels[i],
                k=ks[i],
                act=act,
                act_kwargs=act_kwargs,
                act_first=act_first,
                norm=norm,
                norm_kwargs=norm_kwargs,
                bias=bias,
                weighting=weighting,
                eps=eps,
            )
            self.fp_blocks.append(block)

    def forward(
        self,
        x: Tensor,
        pos: Tensor,
        batch: Tensor,
        intermediates: List[Dict[str, Tensor]],
    ) -> Tuple[Tensor, Tensor, Tensor]:
        for i, (block, intermediate) in enumerate(zip(self.fp_blocks, reversed(intermediates))):
            x_skip = intermediate["x"] if self.skip_channels[i] > 0 else None
            pos_skip = intermediate["pos"]
            batch_skip = intermediate["batch"]
            x, pos, batch = block(x, pos, batch, x_skip, pos_skip, batch_skip)
        return x, pos, batch


class PointNet2Classification(ClassificationModel):
    """PointNet++ classification model from the paper
    :arxiv: [PointNet++: Deep Hierarchical Feature Learning on Point Sets in a Metric Space](https://arxiv.org/abs/1706.02413)
    by Charles R. Qi, Li Yi, Hao Su, Leonidas J. Guibas.

    This network is a hierarchical point cloud classification model. It processes raw point clouds through
    a `PointNet2Encoder` (optional stem + SA blocks), an optional aggregation MLP, global pooling,
    and a classification head.

    Args:
        in_channels: Number of input channels (features per point).
        num_classes: Number of output classes.
        stem_channels: Optional number of channels for initial linear projection (inside the encoder).
        sa_channels: List of channel configurations for Set Abstraction (SA) blocks.
            Each element defines the MLP channels for one SA block.
            For Multi-Scale Grouping (MSG), provide nested lists of channels.
        aggr_channels: Channel sizes for the post-encoder aggregation MLP.
        ratios: Sampling ratios for each SA block (between 0 and 1).
        radii: Search radiuses for each SA block's neighborhood.
            For MSG, provide a list of radii per block.
        num_neighbors: Max number of neighbors for each SA block.
            For MSG, provide a list of neighbor counts per block.
        spatial_dim: Spatial dimensionality of point coordinates (e.g. 3 for 3D, 2 for 2D).
        act: Activation function type or callable.
        act_kwargs: Additional keyword arguments for the activation function.
        act_first: If `True`, activation is applied before normalization.
        norm: Normalization layer type or callable.
        norm_kwargs: Additional keyword arguments for the normalization layer.
        bias: Whether to use bias in linear layers.
        use_pos: Whether to concatenate per-point relative positions to `x`.
        pool: Pooling operation for SA blocks.
        dropout: Dropout for the classification head: a single rate shared by every hidden layer, or one
            rate per hidden layer.
        global_pool: Global pooling operation.
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        *,
        stem_channels: Optional[int] = None,
        sa_channels: Sequence[Sequence[Union[int, Sequence[int]]]],
        aggr_channels: Optional[Union[int, Sequence[int]]] = None,
        aggr_use_pos: bool = False,
        head_channels: Optional[Union[int, Sequence[int]]] = None,
        ratios: Sequence[float],
        radii: Sequence[Union[float, Sequence[float]]],
        num_neighbors: Sequence[Union[int, Sequence[int]]],
        spatial_dim: int = 3,
        act: Union[str, Callable, None] = "relu",
        act_kwargs: Optional[Dict[str, Any]] = None,
        act_first: bool = False,
        norm: Union[str, Callable, None] = "batch_norm",
        norm_kwargs: Optional[Dict[str, Any]] = None,
        bias: bool = False,
        use_pos: bool = True,
        normalize_pos: bool = True,
        pool: PoolLike = "max",
        dropout: Union[float, Sequence[float]] = 0.0,
        global_pool: PoolLike = "max",
    ) -> None:
        super().__init__(in_channels=in_channels, num_classes=num_classes)
        self.stem_channels = stem_channels
        self.sa_channels = sa_channels
        self.aggr_channels = ensure_tuple(aggr_channels) if aggr_channels else None
        self.aggr_use_pos = aggr_use_pos
        self.head_channels = ensure_list(head_channels, none_as_empty=True)
        self.ratios = ratios
        self.radii = radii
        self.num_neighbors = num_neighbors
        self.spatial_dim = spatial_dim
        self.act = act
        self.act_kwargs = act_kwargs
        self.act_first = act_first
        self.norm = norm
        self.norm_kwargs = norm_kwargs
        self.bias = bias
        self.use_pos = use_pos
        self.normalize_pos = normalize_pos
        self.pool = pool
        self.dropout = dropout

        self.encoder = self.configure_encoder()
        self.aggr = self.configure_aggr()
        self.global_pool = create_pool(global_pool)
        self.head = self.configure_head()

    def configure_encoder(self) -> PointNet2Encoder:
        """Build the `PointNet2Encoder` backbone."""
        return PointNet2Encoder(
            in_channels=self.in_channels,
            sa_channels=self.sa_channels,
            ratios=self.ratios,
            radii=self.radii,
            num_neighbors=self.num_neighbors,
            stem_channels=self.stem_channels,
            spatial_dim=self.spatial_dim,
            act=self.act,
            act_kwargs=self.act_kwargs,
            act_first=self.act_first,
            norm=self.norm,
            norm_kwargs=self.norm_kwargs,
            bias=self.bias,
            use_pos=self.use_pos,
            normalize_pos=self.normalize_pos,
            pool=self.pool,
        )

    def configure_aggr(self) -> Optional[MLP]:
        """Build the aggregation MLP applied to the encoder output, or `None` when `aggr_channels` is unset."""
        if not self.aggr_channels:
            return None
        aggr_in = self.encoder.out_channels + self.spatial_dim if self.aggr_use_pos else self.encoder.out_channels
        return MLP(
            [aggr_in, *self.aggr_channels],
            act=self.act,
            act_kwargs=self.act_kwargs,
            act_first=self.act_first,
            norm=self.norm,
            norm_kwargs=self.norm_kwargs,
            bias=self.bias,
            plain_last=False,
        )

    @property
    def num_features(self) -> int:
        """Feature dimension $C$ of the encoder output, after the optional aggregation MLP."""
        return self.aggr_channels[-1] if self.aggr_channels else self.encoder.out_channels

    def configure_head(self) -> nn.Module:
        if self.num_classes == 0:
            return nn.Identity().train(self.training)
        if not self.head_channels:
            return nn.Linear(self.num_features, self.num_classes).train(self.training)

        channels_list = [self.num_features] + list(self.head_channels) + [self.num_classes]
        if isinstance(self.dropout, (int, float)):
            dropout_list = [float(self.dropout)] * (len(channels_list) - 2) + [0.0]
        else:
            if len(self.dropout) != len(channels_list) - 2:
                raise ValueError(
                    f"`dropout` must provide one rate per head layer ({len(channels_list) - 2}); "
                    f"got {len(self.dropout)}."
                )
            dropout_list = [float(rate) for rate in self.dropout] + [0.0]
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
        ).train(self.training)

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
    ) -> Tuple[Tensor, Tensor, Tensor, List[Dict[str, Tensor]]]: ...

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
        result = self.encoder(x, pos, batch, return_intermediates=return_intermediates)

        if return_intermediates:
            x, pos, batch, intermediates = result
        else:
            x, pos, batch = result

        if self.aggr is not None:
            if self.aggr_use_pos:
                x = torch.cat([x, pos], dim=1)
            x = self.aggr(x)

        if return_intermediates:
            return x, pos, batch, intermediates
        return x, pos, batch

    def forward_head(self, x: Tensor, batch: Tensor, pre_logits: bool = False) -> Tensor:
        x = self.global_pool(x, batch)
        if not self.head_channels:
            rate = self.dropout if isinstance(self.dropout, (int, float)) else next(iter(self.dropout), 0.0)
            if rate:
                x = F.dropout(x, p=float(rate), training=self.training)
        return x if pre_logits else self.head(x)

    def forward(self, x: OptTensor, pos: Tensor, batch: Tensor) -> Tensor:
        x, _, batch = self.forward_features(x, pos, batch)
        return self.forward_head(x, batch)


class PointNet2Segmentation(SegmentationModel):
    """PointNet++ segmentation model from the paper
    :arxiv: [PointNet++: Deep Hierarchical Feature Learning on Point Sets in a Metric Space](https://arxiv.org/abs/1706.02413)
    by Charles R. Qi, Li Yi, Hao Su, Leonidas J. Guibas.

    This network is a hierarchical point cloud segmentation model built from a
    `PointNet2Encoder` (optional stem + SA blocks), an optional aggregation MLP,
    a `PointNet2Decoder` (FP blocks with skip connections), and a per-point
    classification head.

    Args:
        in_channels: Number of input channels (features per point).
        num_classes: Number of output classes.
        stem_channels: Optional number of channels for initial linear projection (inside the encoder).
        sa_channels: List of channel configurations for Set Abstraction (SA) blocks.
            Each element defines the MLP channels for one SA block.
            For Multi-Scale Grouping (MSG), provide nested lists of channels.
        aggr_channels: Channel sizes for the post-encoder aggregation MLP.
        fp_channels: List of channel configurations for Feature Propagation (FP) blocks.
            Each element defines the MLP channels for one FP block.
        ratios: Sampling ratios for each SA block (between 0 and 1).
        radii: Search radiuses for each SA block's neighborhood.
            For MSG, provide a list of radii per block.
        num_neighbors: Max number of neighbors for each SA block.
            For MSG, provide a list of neighbor counts per block.
        spatial_dim: Spatial dimensionality of point coordinates (e.g. 3 for 3D, 2 for 2D).
        act: Activation function type or callable.
        act_kwargs: Additional keyword arguments for the activation function.
        act_first: If `True`, activation is applied before normalization.
        norm: Normalization layer type or callable.
        norm_kwargs: Additional keyword arguments for the normalization layer.
        bias: Whether to use bias in linear layers.
        use_pos: Whether to concatenate per-point relative positions to `x`.
        pool: Pooling operation for SA blocks.
        dropout: Dropout rate for classification head.
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        *,
        stem_channels: Optional[int] = None,
        sa_channels: Sequence[Sequence[Union[int, Sequence[int]]]],
        aggr_channels: Optional[Union[int, Sequence[int]]] = None,
        fp_channels: Sequence[Sequence[int]],
        head_channels: Optional[Union[int, Sequence[int]]] = None,
        ratios: Sequence[float],
        radii: Sequence[Union[float, Sequence[float]]],
        num_neighbors: Sequence[Union[int, Sequence[int]]],
        spatial_dim: int = 3,
        act: Union[str, Callable, None] = "relu",
        act_kwargs: Optional[Dict[str, Any]] = None,
        act_first: bool = False,
        norm: Union[str, Callable, None] = "batch_norm",
        norm_kwargs: Optional[Dict[str, Any]] = None,
        bias: bool = False,
        use_pos: bool = True,
        normalize_pos: bool = True,
        pool: PoolLike = "max",
        dropout: float = 0.0,
        skip_input: bool = True,
        fp_k: Optional[Union[int, Sequence[int]]] = None,
    ) -> None:
        super().__init__(in_channels=in_channels, num_classes=num_classes)
        self.stem_channels = stem_channels
        self.sa_channels = sa_channels
        self.aggr_channels = ensure_tuple(aggr_channels) if aggr_channels else None
        self.fp_channels = fp_channels
        self.head_channels = ensure_list(head_channels, none_as_empty=True)
        self.ratios = ratios
        self.radii = radii
        self.num_neighbors = num_neighbors
        self.spatial_dim = spatial_dim
        self.act = act
        self.act_kwargs = act_kwargs
        self.act_first = act_first
        self.norm = norm
        self.norm_kwargs = norm_kwargs
        self.bias = bias
        self.use_pos = use_pos
        self.normalize_pos = normalize_pos
        self.pool = pool
        self.dropout = dropout
        self.skip_input = skip_input
        self.fp_k = fp_k

        self.encoder = self.configure_encoder()
        self.aggr = self.configure_aggr()
        self.decoder = self.configure_decoder()
        self.head = self.configure_head()

    def configure_encoder(self) -> PointNet2Encoder:
        """Build the `PointNet2Encoder` backbone."""
        return PointNet2Encoder(
            in_channels=self.in_channels,
            sa_channels=self.sa_channels,
            ratios=self.ratios,
            radii=self.radii,
            num_neighbors=self.num_neighbors,
            stem_channels=self.stem_channels,
            spatial_dim=self.spatial_dim,
            act=self.act,
            act_kwargs=self.act_kwargs,
            act_first=self.act_first,
            norm=self.norm,
            norm_kwargs=self.norm_kwargs,
            bias=self.bias,
            use_pos=self.use_pos,
            normalize_pos=self.normalize_pos,
            pool=self.pool,
        )

    def configure_aggr(self) -> Optional[MLP]:
        """Build the aggregation MLP applied to the encoder output, or `None` when `aggr_channels` is unset."""
        if not self.aggr_channels:
            return None
        return MLP(
            [self.encoder.out_channels, *self.aggr_channels],
            act=self.act,
            act_kwargs=self.act_kwargs,
            act_first=self.act_first,
            norm=self.norm,
            norm_kwargs=self.norm_kwargs,
            bias=self.bias,
            plain_last=False,
        )

    def configure_decoder(self) -> PointNet2Decoder:
        """Build the `PointNet2Decoder` upsampling the coarsest features back through the encoder skips."""
        decoder_in = self.aggr_channels[-1] if self.aggr_channels is not None else self.encoder.out_channels
        decoder_skip_channels = list(self.encoder.skip_channels[::-1])
        if not self.skip_input and decoder_skip_channels:
            decoder_skip_channels[-1] = 0
        return PointNet2Decoder(
            in_channels=decoder_in,
            skip_channels=decoder_skip_channels,
            fp_channels=self.fp_channels,
            spatial_dim=self.spatial_dim,
            act=self.act,
            act_kwargs=self.act_kwargs,
            act_first=self.act_first,
            norm=self.norm,
            norm_kwargs=self.norm_kwargs,
            bias=self.bias,
            k=self.fp_k,
        )

    @property
    def num_features(self) -> int:
        """Feature dimension $C$ of the decoder output."""
        return self.fp_channels[-1][-1]

    def configure_head(self) -> nn.Module:
        if self.num_classes == 0:
            return nn.Identity().train(self.training)
        if not self.head_channels:
            return nn.Linear(self.num_features, self.num_classes).train(self.training)

        channels_list = [self.num_features] + list(self.head_channels) + [self.num_classes]
        dropout_list = [self.dropout] * (len(channels_list) - 2) + [0.0]
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
        ).train(self.training)

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
    ) -> Tuple[Tensor, Tensor, Tensor, List[Dict[str, Tensor]]]: ...

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
        result = self.encoder(x, pos, batch, return_intermediates=return_intermediates)

        if return_intermediates:
            x, pos, batch, intermediates = result
        else:
            x, pos, batch = result

        if self.aggr is not None:
            x = self.aggr(x)

        if return_intermediates:
            return x, pos, batch, intermediates
        return x, pos, batch

    def forward_decoder(
        self,
        x: Tensor,
        pos: Tensor,
        batch: Tensor,
        intermediates: List[Dict[str, Tensor]],
    ) -> Tensor:
        x, _, _ = self.decoder(x, pos, batch, intermediates)
        return x

    def forward_head(self, x: Tensor, pre_logits: bool = False) -> Tensor:
        if self.dropout and not self.head_channels:
            x = F.dropout(x, p=float(self.dropout), training=self.training)
        return x if pre_logits else self.head(x)

    def forward(self, x: OptTensor, pos: Tensor, batch: Tensor) -> Tensor:
        x, pos, batch, intermediates = self.forward_features(x, pos, batch, return_intermediates=True)
        x = self.forward_decoder(x, pos, batch, intermediates)
        return self.forward_head(x)


def _apply_yanx27_compat(model: nn.Module) -> None:
    """Match yanx27's reference ops so pretrained weights load deterministically.

    yanx27 / charlesq34 keep the $k$ smallest source indices in each ball
    (PointNet++'s reference `query_ball_point`) and weight FP interpolation by
    `1 / (d^2 + 1e-8)` rather than the `1 / d^2` clamp PyG defaults to.
    """
    for sa in getattr(getattr(model, "encoder", None), "sa_blocks", ()):
        sa.sort_neighbors = True
    for fp in getattr(getattr(model, "decoder", None), "fp_blocks", ()):
        fp.weighting = "squared"
        fp.eps = 1e-8


@register_model(
    "pointnet2-ssg.modelnet40.xu-yan",
    task="classification",
    weights=WeightsDict(
        url="hf://torch-pointcloud/pointnet2-ssg.modelnet40.xu-yan/resolve/main/model.safetensors",
        dataset="modelnet40",
        metrics={"OA": 92.30},
        classes=MODELNET40_CLASSES,
        author="xu-yan",
        license="MIT",
    ),
    hparams=dict(
        in_channels=0,
        num_classes=40,
        sa_channels=[[64, 64, 128], [128, 128, 256]],
        aggr_channels=[256, 512, 1024],
        aggr_use_pos=True,
        head_channels=[512, 256],
        ratios=[0.5, 0.25],
        radii=[0.2, 0.4],
        num_neighbors=[32, 64],
        use_pos=True,
        normalize_pos=False,
        bias=True,
        dropout=0.4,
    ),
    transform=T.Compose(
        [
            T.CopyItems(keys=DataKeys.POS, names=DataKeys.ORIGIN_POS),
            T.FarthestPointSample(
                pos_key=DataKeys.POS,
                keys=[DataKeys.NORMAL],
                num_samples=1024,
                dst_index_key=DataKeys.INDEX,
            ),
            T.Rescale(keys=DataKeys.POS, method="centroid"),
        ]
    ),
)
def pointnet2_yanx27_ssg_modelnet40(**hparams: Any) -> PointNet2Classification:
    # from the repo: https://github.com/yanx27/Pointnet_Pointnet2_pytorch (SSG, no normals)
    model = PointNet2Classification(**hparams)
    _apply_yanx27_compat(model)
    return model


@register_model(
    "pointnet2-msg.modelnet40.xu-yan",
    task="classification",
    weights=WeightsDict(
        url="hf://torch-pointcloud/pointnet2-msg.modelnet40.xu-yan/resolve/main/model.safetensors",
        dataset="modelnet40",
        metrics={"OA": 92.67},
        classes=MODELNET40_CLASSES,
        author="xu-yan",
        license="MIT",
    ),
    hparams=dict(
        in_channels=3,
        num_classes=40,
        sa_channels=[
            [[32, 32, 64], [64, 64, 128], [64, 96, 128]],
            [[64, 64, 128], [128, 128, 256], [128, 128, 256]],
        ],
        aggr_channels=[256, 512, 1024],
        aggr_use_pos=True,
        head_channels=[512, 256],
        ratios=[0.5, 0.25],
        radii=[[0.1, 0.2, 0.4], [0.2, 0.4, 0.8]],
        num_neighbors=[[16, 32, 128], [32, 64, 128]],
        use_pos=True,
        normalize_pos=False,
        bias=True,
        dropout=[0.4, 0.5],
    ),
    transform=T.Compose(
        [
            T.CopyItems(keys=DataKeys.POS, names=DataKeys.ORIGIN_POS),
            T.FarthestPointSample(
                pos_key=DataKeys.POS,
                keys=[DataKeys.NORMAL],
                num_samples=1024,
                dst_index_key=DataKeys.INDEX,
            ),
            T.Rescale(keys=DataKeys.POS, method="centroid"),
        ]
    ),
)
def pointnet2_yanx27_msg_modelnet40(**hparams: Any) -> PointNet2Classification:
    # from the repo: https://github.com/yanx27/Pointnet_Pointnet2_pytorch (MSG, with normals)
    model = PointNet2Classification(**hparams)
    _apply_yanx27_compat(model)
    return model


@register_model(
    "pointnet2.s3dis-area5.xu-yan",
    task="segmentation",
    weights=WeightsDict(
        url="hf://torch-pointcloud/pointnet2.s3dis-area5.xu-yan/resolve/main/model.safetensors",
        dataset="s3dis-area5",
        metrics={"mIoU": 54.28},
        classes=S3DIS_CLASSES,
        author="xu-yan",
        license="MIT",
    ),
    hparams=dict(
        in_channels=9,
        num_classes=13,
        sa_channels=[[32, 32, 64], [64, 64, 128], [128, 128, 256], [256, 256, 512]],
        fp_channels=[[256, 256], [256, 256], [256, 128], [128, 128, 128]],
        head_channels=[128],
        ratios=[0.25, 0.25, 0.25, 0.25],
        radii=[0.1, 0.2, 0.4, 0.8],
        num_neighbors=[32, 32, 32, 32],
        use_pos=True,
        normalize_pos=False,
        bias=True,
        dropout=0.5,
        skip_input=False,
        fp_k=3,
    ),
    transform=T.Compose(
        [
            T.Cat(keys=[DataKeys.POS, DataKeys.COLOR, DataKeys.NORM_POS], dst_key=DataKeys.X),
        ]
    ),
)
def pointnet2_yanx27_s3dis_area5(**hparams: Any) -> PointNet2Segmentation:
    # from the repo: https://github.com/yanx27/Pointnet_Pointnet2_pytorch (sem_seg, S3DIS)
    model = PointNet2Segmentation(**hparams)
    _apply_yanx27_compat(model)
    return model


_OPENPOINTS_CLS_HPARAMS: Dict[str, Any] = dict(
    sa_channels=[[64, 64, 128], [128, 128, 256]],
    aggr_channels=[256, 512, 1024],
    aggr_use_pos=True,
    head_channels=[512, 256],
    ratios=[0.5, 0.25],
    radii=[0.2, 0.4],
    num_neighbors=[32, 64],
    use_pos=True,
    normalize_pos=False,
    bias=True,
    dropout=0.5,
)


@register_model(
    "pointnet2.modelnet40.openpoints",
    task="classification",
    weights=WeightsDict(
        url="hf://torch-pointcloud/pointnet2.modelnet40.openpoints/resolve/main/model.safetensors",
        dataset="modelnet40",
        metrics={"OA": 91.90},
        classes=MODELNET40_CLASSES,
        author="openpoints",
        license="MIT",
    ),
    hparams=dict(
        _OPENPOINTS_CLS_HPARAMS,
        in_channels=3,
        num_classes=40,
    ),
    transform=T.Compose(
        [
            T.CopyItems(keys=DataKeys.POS, names=DataKeys.ORIGIN_POS),
            T.Slice(keys=[DataKeys.POS, DataKeys.NORMAL], stop=1024, dst_index_key=DataKeys.INDEX),
            T.Rescale(keys=DataKeys.POS, method="centroid"),
        ]
    ),
)
def pointnet2_openpoints_modelnet40(**hparams: Any) -> PointNet2Classification:
    return PointNet2Classification(**hparams)


@register_model(
    "pointnet2.scanobjectnn.openpoints",
    task="classification",
    weights=WeightsDict(
        url="hf://torch-pointcloud/pointnet2.scanobjectnn.openpoints/resolve/main/model.safetensors",
        dataset="scanobjectnn",
        metrics={"OA": 86.16},
        classes=SCANOBJECTNN_CLASSES,
        author="openpoints",
        license="MIT",
    ),
    hparams=dict(
        _OPENPOINTS_CLS_HPARAMS,
        in_channels=4,
        num_classes=15,
    ),
    transform=T.Compose(
        [
            T.CopyItems(keys=DataKeys.POS, names=DataKeys.ORIGIN_POS),
            T.FarthestPointSample(pos_key=DataKeys.POS, num_samples=1024, dst_index_key=DataKeys.INDEX),
            T.Slice(keys=DataKeys.POS, start=1, stop=2, dim=1, dst_keys="height"),
            T.Shift(keys="height", method="min"),
            T.Rescale(keys=DataKeys.POS, method="centroid"),
            T.Cat(keys=[DataKeys.POS, "height"], dst_key=DataKeys.X),
        ]
    ),
)
def pointnet2_openpoints_scanobjectnn(**hparams: Any) -> PointNet2Classification:
    return PointNet2Classification(**hparams)


_OPENPOINTS_SEG_HPARAMS: Dict[str, Any] = dict(
    in_channels=4,
    num_classes=13,
    sa_channels=[[32, 32, 64], [64, 64, 128], [128, 128, 256], [256, 256, 512]],
    fp_channels=[[256, 256], [256, 256], [256, 128], [128, 128, 128]],
    head_channels=[128],
    ratios=[0.25, 0.25, 0.25, 0.25],
    radii=[0.1, 0.2, 0.4, 0.8],
    num_neighbors=[32, 32, 32, 32],
    use_pos=True,
    normalize_pos=False,
    bias=True,
    dropout=0.5,
    skip_input=True,
    fp_k=3,
)


_OPENPOINTS_S3DIS_TRANSFORM = T.Compose(
    [
        T.Slice(keys=DataKeys.POS, start=2, stop=3, dim=1, dst_keys="height"),
        T.Shift(keys="height", method="min"),
        T.Divide(keys=DataKeys.COLOR, divisor=255.0),
        T.Normalize(
            keys=DataKeys.COLOR,
            mean=[0.5136457, 0.49523646, 0.44921124],
            std=[0.18308958, 0.18415008, 0.19252081],
        ),
        T.Shift(keys=DataKeys.POS, method="centroid", axes=[0, 1]),
        T.Shift(keys=DataKeys.POS, method="min", axes=[2]),
        T.Cat(keys=[DataKeys.COLOR, "height"], dst_key=DataKeys.X),
    ]
)


def _pointnet2_openpoints_s3dis(**hparams: Any) -> PointNet2Segmentation:
    model = PointNet2Segmentation(**hparams)
    for fp in getattr(getattr(model, "decoder", None), "fp_blocks", ()):
        fp.weighting = "inverse"
        fp.eps = 1e-8
    return model


@register_model(
    "pointnet2.s3dis-area1.openpoints",
    task="segmentation",
    weights=WeightsDict(
        url="hf://torch-pointcloud/pointnet2.s3dis-area1.openpoints/resolve/main/model.safetensors",
        dataset="s3dis-area1",
        metrics={"mIoU": 74.96},
        classes=S3DIS_CLASSES,
        author="openpoints",
        license="MIT",
    ),
    hparams=dict(_OPENPOINTS_SEG_HPARAMS),
    transform=_OPENPOINTS_S3DIS_TRANSFORM,
)
def pointnet2_openpoints_s3dis_area1(**hparams: Any) -> PointNet2Segmentation:
    return _pointnet2_openpoints_s3dis(**hparams)


@register_model(
    "pointnet2.s3dis-area2.openpoints",
    task="segmentation",
    weights=WeightsDict(
        url="hf://torch-pointcloud/pointnet2.s3dis-area2.openpoints/resolve/main/model.safetensors",
        dataset="s3dis-area2",
        metrics={"mIoU": 48.22},
        classes=S3DIS_CLASSES,
        author="openpoints",
        license="MIT",
    ),
    hparams=dict(_OPENPOINTS_SEG_HPARAMS),
    transform=_OPENPOINTS_S3DIS_TRANSFORM,
)
def pointnet2_openpoints_s3dis_area2(**hparams: Any) -> PointNet2Segmentation:
    return _pointnet2_openpoints_s3dis(**hparams)


@register_model(
    "pointnet2.s3dis-area3.openpoints",
    task="segmentation",
    weights=WeightsDict(
        url="hf://torch-pointcloud/pointnet2.s3dis-area3.openpoints/resolve/main/model.safetensors",
        dataset="s3dis-area3",
        metrics={"mIoU": 76.31},
        classes=S3DIS_CLASSES,
        author="openpoints",
        license="MIT",
    ),
    hparams=dict(_OPENPOINTS_SEG_HPARAMS),
    transform=_OPENPOINTS_S3DIS_TRANSFORM,
)
def pointnet2_openpoints_s3dis_area3(**hparams: Any) -> PointNet2Segmentation:
    return _pointnet2_openpoints_s3dis(**hparams)


@register_model(
    "pointnet2.s3dis-area4.openpoints",
    task="segmentation",
    weights=WeightsDict(
        url="hf://torch-pointcloud/pointnet2.s3dis-area4.openpoints/resolve/main/model.safetensors",
        dataset="s3dis-area4",
        metrics={"mIoU": 59.96},
        classes=S3DIS_CLASSES,
        author="openpoints",
        license="MIT",
    ),
    hparams=dict(_OPENPOINTS_SEG_HPARAMS),
    transform=_OPENPOINTS_S3DIS_TRANSFORM,
)
def pointnet2_openpoints_s3dis_area4(**hparams: Any) -> PointNet2Segmentation:
    return _pointnet2_openpoints_s3dis(**hparams)


@register_model(
    "pointnet2.s3dis-area5.openpoints",
    task="segmentation",
    weights=WeightsDict(
        url="hf://torch-pointcloud/pointnet2.s3dis-area5.openpoints/resolve/main/model.safetensors",
        dataset="s3dis-area5",
        metrics={"mIoU": 63.66},
        classes=S3DIS_CLASSES,
        author="openpoints",
        license="MIT",
    ),
    hparams=dict(_OPENPOINTS_SEG_HPARAMS),
    transform=_OPENPOINTS_S3DIS_TRANSFORM,
)
def pointnet2_openpoints_s3dis_area5(**hparams: Any) -> PointNet2Segmentation:
    return _pointnet2_openpoints_s3dis(**hparams)


@register_model(
    "pointnet2.s3dis-area6.openpoints",
    task="segmentation",
    weights=WeightsDict(
        url="hf://torch-pointcloud/pointnet2.s3dis-area6.openpoints/resolve/main/model.safetensors",
        dataset="s3dis-area6",
        metrics={"mIoU": 82.45},
        classes=S3DIS_CLASSES,
        author="openpoints",
        license="MIT",
    ),
    hparams=dict(_OPENPOINTS_SEG_HPARAMS),
    transform=_OPENPOINTS_S3DIS_TRANSFORM,
)
def pointnet2_openpoints_s3dis_area6(**hparams: Any) -> PointNet2Segmentation:
    return _pointnet2_openpoints_s3dis(**hparams)
