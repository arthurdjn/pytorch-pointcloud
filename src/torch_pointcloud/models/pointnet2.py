from typing import Any, Callable, Dict, List, Literal, Optional, Sequence, Tuple, Union, overload

import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn import MLP

from torch_pointcloud.layers import PoolLike, create_cls_head, create_pool
from torch_pointcloud.layers.pointnet2_blocks import FPModule, SAModule, ensure_msg_list
from torch_pointcloud.utils.conversion import ensure_tuple, ensure_tuple_size
from torch_pointcloud.utils.types import OptTensor


class PointNet2Encoder(nn.Module):
    """PointNet++ encoder from the paper
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
        use_coords: Whether to use point coordinates as features.
        pool: Pooling operation for SA blocks.
    """

    def __init__(
        self,
        in_channels: int,
        sa_channels: Sequence[Sequence[Union[int, Sequence[int]]]],
        ratios: Sequence[float],
        radii: Sequence[Union[float, Sequence[float]]],
        num_neighbors: Sequence[Union[int, Sequence[int]]],
        *,
        stem_channels: Optional[int] = None,
        spatial_dim: int = 3,
        act: Union[str, Callable, None] = "relu",
        act_kwargs: Optional[Dict[str, Any]] = None,
        act_first: bool = False,
        norm: Union[str, Callable, None] = "batch_norm",
        norm_kwargs: Optional[Dict[str, Any]] = None,
        bias: bool = False,
        use_coords: bool = True,
        pool: PoolLike = "max",
    ) -> None:
        super().__init__()
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
        ratios = ensure_tuple_size(ratios, size=num_blocks, extra_msg=extra_msg.format(param="ratios"))
        radii = ensure_tuple_size(radii, size=num_blocks, extra_msg=extra_msg.format(param="radii"))
        num_neighbors = ensure_tuple_size(num_neighbors, size=num_blocks, extra_msg=extra_msg.format(param="k"))

        ch = encoder_in
        self.sa_blocks = nn.ModuleList()
        sa_out_channels: List[int] = []
        for i in range(num_blocks):
            block = SAModule(
                in_channels=ch,
                channels=sa_channels[i],
                ratio=ratios[i],
                radii=radii[i],
                num_neighbors=num_neighbors[i],
                spatial_dim=spatial_dim,
                act=act,
                act_kwargs=act_kwargs,
                act_first=act_first,
                norm=norm,
                norm_kwargs=norm_kwargs,
                bias=bias,
                use_coords=use_coords,
                pool=pool,
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
        x = x if x is not None else pos
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
        spatial_dim: Spatial dimensionality of point coordinates. Also used as the default
            number of neighbors `k` for kNN interpolation.
        act: Activation function type or callable.
        act_kwargs: Additional keyword arguments for the activation function.
        act_first: If `True`, activation is applied before normalization.
        norm: Normalization layer type or callable.
        norm_kwargs: Additional keyword arguments for the normalization layer.
        bias: Whether to use bias in linear layers.
        k: Number of neighbors for kNN interpolation. Defaults to `spatial_dim`.
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
        k: Optional[int] = None,
    ) -> None:
        super().__init__()
        if len(skip_channels) != len(fp_channels):
            raise ValueError(
                f"The number of skip channels ({len(skip_channels)}) must match "
                f"the number of feature propagation channels ({len(fp_channels)})."
            )

        if k is None:
            k = spatial_dim

        num_blocks = len(fp_channels)
        self.fp_blocks = nn.ModuleList()
        for i in range(num_blocks):
            ch = in_channels if i == 0 else fp_channels[i - 1][-1]
            block = FPModule(
                in_channels=ch + skip_channels[i],
                channels=fp_channels[i],
                k=1 if i == 0 else k,
                act=act,
                act_kwargs=act_kwargs,
                act_first=act_first,
                norm=norm,
                norm_kwargs=norm_kwargs,
                bias=bias,
            )
            self.fp_blocks.append(block)

    def forward(
        self,
        x: Tensor,
        pos: Tensor,
        batch: Tensor,
        intermediates: List[Dict[str, Tensor]],
    ) -> Tuple[Tensor, Tensor, Tensor]:
        for block, intermediate in zip(self.fp_blocks, reversed(intermediates)):
            x_skip = intermediate["x"]
            pos_skip = intermediate["pos"]
            batch_skip = intermediate["batch"]
            x, pos, batch = block(x, pos, batch, x_skip, pos_skip, batch_skip)
        return x, pos, batch


class PointNet2Classification(nn.Module):
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
        use_coords: Whether to use point coordinates as features.
        pool: Pooling operation for SA blocks.
        dropout: Dropout rate for classification head.
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
        use_coords: bool = True,
        pool: PoolLike = "max",
        dropout: float = 0.0,
        global_pool: PoolLike = "max",
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes

        self.encoder = PointNet2Encoder(
            in_channels=in_channels,
            sa_channels=sa_channels,
            ratios=ratios,
            radii=radii,
            num_neighbors=num_neighbors,
            stem_channels=stem_channels,
            spatial_dim=spatial_dim,
            act=act,
            act_kwargs=act_kwargs,
            act_first=act_first,
            norm=norm,
            norm_kwargs=norm_kwargs,
            bias=bias,
            use_coords=use_coords,
            pool=pool,
        )

        enc_out = self.encoder.out_channels
        aggr_channels = ensure_tuple(aggr_channels) if aggr_channels else None
        self.aggr = (
            MLP(
                [enc_out, *aggr_channels],
                act=act,
                act_kwargs=act_kwargs,
                act_first=act_first,
                norm=norm,
                norm_kwargs=norm_kwargs,
                plain_last=False,
            )
            if aggr_channels
            else None
        )

        self.global_pool = create_pool(global_pool)
        self.dropout = dropout
        self.embedding_dim = aggr_channels[-1] if aggr_channels else enc_out
        self.head = create_cls_head(self.embedding_dim, num_classes)

    def reset_classifier(self, num_classes: int, global_pool: PoolLike = "max", **kwargs: Any) -> None:
        self.num_classes = num_classes
        self.global_pool = create_pool(global_pool)
        self.head = create_cls_head(num_features=self.embedding_dim, num_classes=self.num_classes, **kwargs)

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

    def forward_head(self, x: Tensor, batch: Tensor, pre_logits: bool = False) -> Tensor:
        x = self.global_pool(x, batch)
        if self.dropout:
            x = F.dropout(x, p=float(self.dropout), training=self.training)
        return x if pre_logits else self.head(x)

    def forward(self, x: OptTensor, pos: Tensor, batch: Tensor) -> Tensor:
        x, _, batch = self.forward_features(x, pos, batch)
        return self.forward_head(x, batch)


class PointNet2Segmentation(nn.Module):
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
        use_coords: Whether to use point coordinates as features.
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
        use_coords: bool = True,
        pool: PoolLike = "max",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes

        self.encoder = PointNet2Encoder(
            in_channels=in_channels,
            sa_channels=sa_channels,
            ratios=ratios,
            radii=radii,
            num_neighbors=num_neighbors,
            stem_channels=stem_channels,
            spatial_dim=spatial_dim,
            act=act,
            act_kwargs=act_kwargs,
            act_first=act_first,
            norm=norm,
            norm_kwargs=norm_kwargs,
            bias=bias,
            use_coords=use_coords,
            pool=pool,
        )

        enc_out = self.encoder.out_channels
        aggr_channels = ensure_tuple(aggr_channels) if aggr_channels else None
        self.aggr = (
            MLP(
                [enc_out, *aggr_channels],
                act=act,
                act_kwargs=act_kwargs,
                act_first=act_first,
                norm=norm,
                norm_kwargs=norm_kwargs,
                plain_last=False,
            )
            if aggr_channels
            else None
        )

        decoder_in = aggr_channels[-1] if aggr_channels is not None else enc_out
        self.decoder = PointNet2Decoder(
            in_channels=decoder_in,
            skip_channels=self.encoder.skip_channels[::-1],
            fp_channels=fp_channels,
            spatial_dim=spatial_dim,
            act=act,
            act_kwargs=act_kwargs,
            act_first=act_first,
            norm=norm,
            norm_kwargs=norm_kwargs,
            bias=bias,
        )

        self.dropout = dropout
        self.embedding_dim = fp_channels[-1][-1]
        self.head = create_cls_head(self.embedding_dim, num_classes)

    def reset_classifier(self, num_classes: int, **kwargs: Any) -> None:
        self.num_classes = num_classes
        self.head = create_cls_head(num_features=self.embedding_dim, num_classes=self.num_classes, **kwargs)

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
        if self.dropout:
            x = F.dropout(x, p=float(self.dropout), training=self.training)
        return x if pre_logits else self.head(x)

    def forward(self, x: OptTensor, pos: Tensor, batch: Tensor) -> Tensor:
        x, pos, batch, intermediates = self.forward_features(x, pos, batch, return_intermediates=True)
        x = self.forward_decoder(x, pos, batch, intermediates)
        return self.forward_head(x)
