from typing import Any, Callable, Dict, List, Literal, Optional, Sequence, Tuple, Union, overload

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn import MLP

import torch_pointcloud.transforms as T
from torch_pointcloud.layers import PoolLike, create_cls_head, create_pool
from torch_pointcloud.layers.pointnet2_blocks import FPModule, SAModule, ensure_msg_list
from torch_pointcloud.utils.conversion import ensure_list, ensure_tuple, ensure_tuple_size
from torch_pointcloud.utils.data import DataKeys
from torch_pointcloud.utils.types import KeyCollection, OptTensor

from ._base import ClassificationModel, SegmentationModel
from ._registry import register_model


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
        use_pos: Whether to concatenate per-point relative positions to `x`.
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
        use_pos: bool = True,
        normalize_pos: bool = True,
        pool: PoolLike = "max",
        sort_neighbors: bool = False,
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
                use_pos=use_pos,
                normalize_pos=normalize_pos,
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
        dropout: float = 0.0,
        global_pool: PoolLike = "max",
    ) -> None:
        super().__init__(in_channels=in_channels, num_classes=num_classes)
        self.aggr_use_pos = aggr_use_pos
        self.head_channels = ensure_list(head_channels, none_as_empty=True)
        self.dropout = dropout
        self.act = act
        self.act_kwargs = act_kwargs
        self.act_first = act_first
        self.norm = norm
        self.norm_kwargs = norm_kwargs
        self.bias = bias
        self.spatial_dim = spatial_dim

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
            use_pos=use_pos,
            normalize_pos=normalize_pos,
            pool=pool,
        )

        enc_out = self.encoder.out_channels
        aggr_channels = ensure_tuple(aggr_channels) if aggr_channels else None
        aggr_in = enc_out + spatial_dim if aggr_use_pos else enc_out
        self.aggr = (
            MLP(
                [aggr_in, *aggr_channels],
                act=act,
                act_kwargs=act_kwargs,
                act_first=act_first,
                norm=norm,
                norm_kwargs=norm_kwargs,
                bias=bias,
                plain_last=False,
            )
            if aggr_channels
            else None
        )

        self.global_pool = create_pool(global_pool)
        self.embedding_dim = aggr_channels[-1] if aggr_channels else enc_out
        self.head = self.configure_head()

    def configure_head(self) -> nn.Module:
        if not self.head_channels:
            return create_cls_head(self.embedding_dim, self.num_classes)

        channels_list = [self.embedding_dim] + list(self.head_channels) + [self.num_classes]
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
        )

    def reset_classifier(self, num_classes: int, global_pool: PoolLike = "max", **kwargs: Any) -> None:
        self.num_classes = num_classes
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
        if self.dropout and not self.head_channels:
            x = F.dropout(x, p=float(self.dropout), training=self.training)
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
        self.head_channels = ensure_list(head_channels, none_as_empty=True)
        self.dropout = dropout
        self.act = act
        self.act_kwargs = act_kwargs
        self.act_first = act_first
        self.norm = norm
        self.norm_kwargs = norm_kwargs
        self.bias = bias

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
            use_pos=use_pos,
            normalize_pos=normalize_pos,
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
                bias=bias,
                plain_last=False,
            )
            if aggr_channels
            else None
        )

        decoder_in = aggr_channels[-1] if aggr_channels is not None else enc_out
        decoder_skip_channels = list(self.encoder.skip_channels[::-1])
        if not skip_input and decoder_skip_channels:
            decoder_skip_channels[-1] = 0
        self.decoder = PointNet2Decoder(
            in_channels=decoder_in,
            skip_channels=decoder_skip_channels,
            fp_channels=fp_channels,
            spatial_dim=spatial_dim,
            act=act,
            act_kwargs=act_kwargs,
            act_first=act_first,
            norm=norm,
            norm_kwargs=norm_kwargs,
            bias=bias,
            k=fp_k,
        )

        self.embedding_dim = fp_channels[-1][-1]
        self.head = self.configure_head()

    def configure_head(self) -> nn.Module:
        if not self.head_channels:
            return create_cls_head(self.embedding_dim, self.num_classes)

        channels_list = [self.embedding_dim] + list(self.head_channels) + [self.num_classes]
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
    `1 / (d^2 + 1e-8)` rather than the `1 / d^2` clamp PyG defaults to. These
    inference-time quirks live on the blocks themselves so the public model API
    stays free of them. FPS is already deterministic in `.eval()` mode.
    """
    for sa in getattr(getattr(model, "encoder", None), "sa_blocks", ()):
        sa.sort_neighbors = True
    for fp in getattr(getattr(model, "decoder", None), "fp_blocks", ()):
        fp.weighting = "squared"
        fp.eps = 1e-8


@register_model(
    "pointnet2-yanx27-ssg.modelnet40",
    task="classification",
    weights="hf://torch-pointcloud/pointnet2/pointnet2-yanx27-ssg.modelnet40.pt",
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
    transforms=T.Compose(
        [
            T.FarthestPointSample(pos_key=DataKeys.POS, keys=[], num_samples=1024),
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
    "pointnet2-yanx27-msg.modelnet40",
    task="classification",
    weights="hf://torch-pointcloud/pointnet2/pointnet2-yanx27-msg.modelnet40.pt",
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
        dropout=0.4,
    ),
    transforms=T.Compose(
        [
            T.FarthestPointSample(pos_key=DataKeys.POS, keys=[DataKeys.NORMAL], num_samples=1024),
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
    "pointnet2-yanx27.s3dis-area5",
    task="segmentation",
    weights="hf://torch-pointcloud/pointnet2/pointnet2-yanx27.s3dis-area5.pt",
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
    transforms=T.Compose(
        [
            T.Divide(keys=DataKeys.COLOR, divisor=255.0),
            T.Cat(keys=[DataKeys.POS, DataKeys.COLOR, "norm_pos"], dst_key=DataKeys.X),
        ]
    ),
)
def pointnet2_yanx27_s3dis_area5(**hparams: Any) -> PointNet2Segmentation:
    # from the repo: https://github.com/yanx27/Pointnet_Pointnet2_pytorch (sem_seg, S3DIS)
    model = PointNet2Segmentation(**hparams)
    _apply_yanx27_compat(model)
    return model


# Shared hparams for the openpoints / PointNeXt-trained PointNet++ classifiers
# (`cfgs/{modelnet40ply2048,scanobjectnn}/pointnet++.yaml`).
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


class _TakeFirstN(T.DictTransform):
    """Slice the leading $N$ rows of each listed key.

    Matches openpoints' val protocol on ModelNet40Ply2048 (`current_points = points[:num_points]`):
    HDF5 / resampled ModelNet variants already store points in FPS-sorted order, so the first $N$
    points form a deterministic FPS subset without re-running FPS.
    """

    def __init__(self, keys: KeyCollection, num: int, allow_missing_keys: bool = False) -> None:
        super().__init__(keys, allow_missing_keys)
        self.num = num

    def transform(self, data: Dict[str, Any]) -> Dict[str, Any]:
        d = dict(data)
        for key in self.iter_keys(d):
            d[key] = d[key][: self.num]
        return d


@register_model(
    "pointnet2-openpoints.modelnet40",
    task="classification",
    weights="hf://torch-pointcloud/pointnet2/pointnet2-openpoints.modelnet40.pt",
    hparams=dict(
        _OPENPOINTS_CLS_HPARAMS,
        in_channels=3,
        num_classes=40,
    ),
    transforms=T.Compose(
        [
            _TakeFirstN(keys=DataKeys.POS, num=1024),
            T.Rescale(keys=DataKeys.POS, method="centroid"),
        ]
    ),
)
def pointnet2_openpoints_modelnet40(**hparams: Any) -> PointNet2Classification:
    # from the openpoints model zoo: https://guochengqian.github.io/PointNeXt/modelzoo/
    return PointNet2Classification(**hparams)


class _OpenPointsStashHeight(T.DictTransform):
    """Snapshot openpoints' gravity-axis height in `'height'` before pos rescaling.

    openpoints' `PointCloudCenterAndNormalize` computes `heights = pos[:, gravity_dim] - min`
    on the raw point cloud **before** centering and unit-sphere normalization, so the
    height channel stays in the original cube units even though `pos` is later normalized.
    This transform mirrors that ordering: it must run before `T.Rescale` so the height
    sees raw coordinates.

    openpoints uses `gravity_dim=2` (z-up) on ModelNet40 and `gravity_dim=1` (y-up) on
    ScanObjectNN.
    """

    def __init__(self, gravity_dim: int = 2, dst_key: str = "height") -> None:
        super().__init__(keys=DataKeys.POS, allow_missing_keys=False)
        self.gravity_dim = gravity_dim
        self.dst_key = dst_key

    def transform(self, data: Dict[str, Any]) -> Dict[str, Any]:
        d = dict(data)
        pos = d[DataKeys.POS]
        height = pos[:, self.gravity_dim : self.gravity_dim + 1]
        d[self.dst_key] = height - height.min()
        return d


@register_model(
    "pointnet2-openpoints.scanobjectnn",
    task="classification",
    weights="hf://torch-pointcloud/pointnet2/pointnet2-openpoints.scanobjectnn.pt",
    hparams=dict(
        _OPENPOINTS_CLS_HPARAMS,
        in_channels=4,
        num_classes=15,
    ),
    transforms=T.Compose(
        [
            T.FarthestPointSample(pos_key=DataKeys.POS, keys=[], num_samples=1024),
            _OpenPointsStashHeight(gravity_dim=1),
            T.Rescale(keys=DataKeys.POS, method="centroid"),
            T.Cat(keys=[DataKeys.POS, "height"], dst_key=DataKeys.X),
        ]
    ),
)
def pointnet2_openpoints_scanobjectnn(**hparams: Any) -> PointNet2Classification:
    # from the openpoints model zoo: https://guochengqian.github.io/PointNeXt/modelzoo/
    return PointNet2Classification(**hparams)


def _apply_openpoints_seg_compat(model: nn.Module) -> None:
    """Match openpoints' FP interpolation: $1 / (d + 1\\text{e-}8)$ inverse weighting.

    openpoints' `three_interpolation` uses inverse-distance weighting on Euclidean
    distance (not squared). yanx27 weights by `1 / d^2`; we expose this on each
    `FPModule` so the openpoints factories can swap in the inverse mode.
    """
    for fp in getattr(getattr(model, "decoder", None), "fp_blocks", ()):
        fp.weighting = "inverse"
        fp.eps = 1e-8


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


class _OpenPointsChromaticNormalize(T.DictTransform):
    """openpoints' `ChromaticNormalize`: optionally divide by 255 then standardize.

    Matches the val-time transform from `openpoints/transforms/point_transformer_gpu.py`
    (ImageNet-style RGB stats). The auto-divide branch lets the same transform handle
    HDF5 colors (already in $[0, 1]$) and raw S3DIS rooms (uint8 in $[0, 255]$).
    """

    _MEAN = torch.tensor([0.5136457, 0.49523646, 0.44921124])
    _STD = torch.tensor([0.18308958, 0.18415008, 0.19252081])

    def __init__(self) -> None:
        super().__init__(keys=DataKeys.COLOR, allow_missing_keys=False)

    def transform(self, data: Dict[str, Any]) -> Dict[str, Any]:
        d = dict(data)
        color = d[DataKeys.COLOR].float()
        if color.max() > 1:
            color = color / 255.0
        d[DataKeys.COLOR] = (color - self._MEAN.to(color.device)) / self._STD.to(color.device)
        return d


_OPENPOINTS_S3DIS_TRANSFORM = T.Compose(
    [
        # openpoints' S3DIS model receives `x = cat([color_norm, height])`. Heights come
        # from `pos[:, gravity_dim]` shifted to start at 0; colors go through
        # ChromaticNormalize. We add a final `T.Shift` to mirror `PointCloudXYZAlign`
        # (center pos on its mean, snap z to min) which is what the model saw at training.
        _OpenPointsStashHeight(gravity_dim=2),
        _OpenPointsChromaticNormalize(),
        T.Shift(keys=DataKeys.POS, method="centroid", axes=[0, 1]),
        T.Shift(keys=DataKeys.POS, method="min", axes=[2]),
        T.Cat(keys=[DataKeys.COLOR, "height"], dst_key=DataKeys.X),
    ]
)


def _make_openpoints_s3dis_factory(area: int) -> Callable[..., SegmentationModel]:
    name = f"pointnet2-openpoints.s3dis-area{area}"

    @register_model(
        name,
        task="segmentation",
        weights=f"hf://torch-pointcloud/pointnet2/{name}.pt",
        hparams=dict(_OPENPOINTS_SEG_HPARAMS),
        transforms=_OPENPOINTS_S3DIS_TRANSFORM,
    )
    def factory(**hparams: Any) -> PointNet2Segmentation:
        # from the openpoints model zoo: https://guochengqian.github.io/PointNeXt/modelzoo/
        model = PointNet2Segmentation(**hparams)
        _apply_openpoints_seg_compat(model)
        return model

    factory.__name__ = f"pointnet2_openpoints_s3dis_area{area}"
    return factory


pointnet2_openpoints_s3dis_area1 = _make_openpoints_s3dis_factory(1)
pointnet2_openpoints_s3dis_area2 = _make_openpoints_s3dis_factory(2)
pointnet2_openpoints_s3dis_area3 = _make_openpoints_s3dis_factory(3)
pointnet2_openpoints_s3dis_area4 = _make_openpoints_s3dis_factory(4)
pointnet2_openpoints_s3dis_area5 = _make_openpoints_s3dis_factory(5)
pointnet2_openpoints_s3dis_area6 = _make_openpoints_s3dis_factory(6)
