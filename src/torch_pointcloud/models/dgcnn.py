from typing import Any, Callable, Dict, NamedTuple, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn import MLP, DynamicEdgeConv, global_max_pool

import torch_pointcloud.transforms as T
from torch_pointcloud.layers import CatPool, PoolLike, create_pool
from torch_pointcloud.layers.tnet import DynamicTNet, TNet
from torch_pointcloud.utils.cluster import knn
from torch_pointcloud.utils.conversion import ensure_list, ensure_tuple_size, is_iterable
from torch_pointcloud.utils.data import DataKeys
from torch_pointcloud.utils.types import AggrType, OptTensor

from ._base import ClassificationModel, SegmentationModel
from ._registry import register_model


class DGCNNIntermediate(NamedTuple):
    x: Tensor
    batch: Tensor


class DGCNNEncoderBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: Union[int, Sequence[int]],
        num_neighbors: Union[int, Sequence[int]],
        aggr: AggrType = "max",
        act: Union[str, Callable, None] = "relu",
        act_kwargs: Optional[Dict[str, Any]] = None,
        act_first: bool = False,
        norm: Union[str, Callable, None] = "batch_norm",
        norm_kwargs: Optional[Dict[str, Any]] = None,
        bias: Union[bool, Sequence[bool]] = True,
    ) -> None:
        super().__init__()
        channels_list = [2 * in_channels] + ensure_list(out_channels)
        nn_module = MLP(
            channels_list,
            plain_last=False,
            act=act,
            act_kwargs=act_kwargs,
            norm=norm,
            norm_kwargs=norm_kwargs,
            bias=bias,
        )
        self.k = num_neighbors
        self.conv = DynamicEdgeConv(nn_module, k=num_neighbors, aggr=aggr)

    def forward(self, x: Tensor, batch: Tensor, knn_x: Optional[Tensor] = None) -> Tensor:
        src = knn_x if knn_x is not None else x
        edge_index = knn(src, src, self.k, batch_x=batch, batch_y=batch).flip([0])  # type: ignore[arg-type]
        # Fix, we might integrate our own knn into the DynamicEdgeConv later
        return self.conv.propagate(edge_index, x=(x, x))


class DGCNNEncoder(nn.Module):
    def __init__(
        self,
        channels: Sequence[Union[int, Sequence[int]]],
        num_neighbors: Union[int, Sequence[int]],
        aggr: AggrType = "max",
        act: Union[str, Callable, None] = "relu",
        act_kwargs: Optional[Dict[str, Any]] = None,
        act_first: bool = False,
        norm: Union[str, Callable, None] = "batch_norm",
        norm_kwargs: Optional[Dict[str, Any]] = None,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.channels = ensure_list(channels)
        size = len(self.channels) - 1
        self.num_neighbors = ensure_tuple_size(num_neighbors, size=size)

        self.blocks = nn.ModuleList()
        for i in range(size):
            channels = ensure_list(self.channels[i])
            in_channels = channels[-1]
            block = DGCNNEncoderBlock(
                in_channels=in_channels,
                out_channels=self.channels[i + 1],
                num_neighbors=self.num_neighbors[i],
                act=act,
                act_kwargs=act_kwargs,
                act_first=act_first,
                norm=norm,
                norm_kwargs=norm_kwargs,
                bias=bias,
                aggr=aggr,
            )
            self.blocks.append(block)

    @property
    def out_channels_per_block(self) -> Tuple[int, ...]:
        out = []
        for ch in self.channels[1:]:
            out.append(ch[-1] if isinstance(ch, (list, tuple)) else ch)
        return tuple(out)

    @property
    def out_channels(self) -> int:
        return sum(self.out_channels_per_block)

    def forward(self, x: Tensor, batch: Tensor, knn_x: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        x_list = []
        for i, block in enumerate(self.blocks):
            x = block(x, batch, knn_x=knn_x if i == 0 else None)
            x_list.append(x)

        x = torch.cat(x_list, dim=1)
        return x, batch


class DGCNNClassification(ClassificationModel):
    """
    Classification model as described in the paper
    ["Dynamic Graph CNN for Learning on Point Clouds"](https://arxiv.org/abs/1801.07829)
    by Yue Wang, Yongbin Sun, Ziwei Liu, Sanjay E. Sarma, Michael M. Bronstein, Justin M. Solomon.

    DGCNN introduces the EdgeConv operator, which computes features on dynamically constructed
    k-nearest neighbor (k-NN) graphs at each layer. Graphs are recomputed in the learned feature space,
    allowing the network to capture both local geometric relationships and long-range semantic structures.

    Args:
        in_channels: Number of input channels.
        num_classes: Number of output classes.
        spatial_dim: Spatial dimension of the input point cloud.
        stnet_local_channels: List of channels for the local spatial transformer network.
            If None, the spatial transformer network is not used.
        stnet_global_channels: List of channels for the global spatial transformer network.
            If None, the spatial transformer network is not used.
        channels: List of channels for each encoder block.
        proj_channels: If set, projects the concatenated encoder features through an MLP
            of this width before pooling. Matches the ``conv5`` layer in the original DGCNN paper.
        head_channels: List of channels for each head block.
        num_neighbors: Maximum number of neighbors for each encoder block.
        act: Activation function.
        act_kwargs: Additional arguments for the activation function.
        act_first: Whether to apply activation before normalization.
        norm: Normalization layer type.
        norm_kwargs: Additional arguments for the normalization layer.
        bias: Whether to use bias in linear / MLP layers.
        dropout: Dropout probability before the classification head.
        global_pool: Global pooling operation for final feature aggregation.

    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        *,
        spatial_dim: int = 3,
        stnet_local_channels: Optional[Sequence[int]] = None,
        stnet_global_channels: Optional[Sequence[int]] = None,
        head_channels: Optional[Union[int, Sequence[int]]] = None,
        channels: Sequence[int],
        proj_channels: Optional[int] = None,
        num_neighbors: Union[int, Sequence[int]],
        act: Union[str, Callable, None] = "relu",
        act_kwargs: Optional[Dict[str, Any]] = None,
        act_first: bool = False,
        norm: Union[str, Callable, None] = "batch_norm",
        norm_kwargs: Optional[Dict[str, Any]] = None,
        bias: bool = True,
        dropout: float = 0.0,
        global_pool: PoolLike | Sequence[PoolLike] = "max",
    ):
        super().__init__(in_channels=in_channels + spatial_dim, num_classes=num_classes)
        self.spatial_dim = spatial_dim
        self.stnet_local_channels = ensure_list(stnet_local_channels) if stnet_local_channels is not None else None
        self.stnet_global_channels = ensure_list(stnet_global_channels) if stnet_global_channels is not None else None
        self.head_channels = ensure_list(head_channels, none_as_empty=True)
        self.channels = ensure_list(channels)
        self.proj_channels = proj_channels
        self.num_neighbors = ensure_list(num_neighbors)
        self.act = act
        self.act_kwargs = act_kwargs
        self.act_first = act_first
        self.norm = norm
        self.norm_kwargs = norm_kwargs
        self.bias = bias

        self.stnet: Optional[TNet] = None
        if self.stnet_local_channels is not None and self.stnet_global_channels is not None:
            self.stnet = TNet(
                local_channels=self.stnet_local_channels,
                global_channels=self.stnet_global_channels,
                k=self.spatial_dim,
                act=self.act,
                act_kwargs=self.act_kwargs,
                act_first=self.act_first,
                norm=self.norm,
                norm_kwargs=self.norm_kwargs,
                bias=self.bias,
                aggr="max",
            )

        self.encoder = DGCNNEncoder(
            channels=[self.in_channels] + self.channels,
            num_neighbors=self.num_neighbors,
            act=self.act,
            act_kwargs=self.act_kwargs,
            act_first=self.act_first,
            norm=self.norm,
            norm_kwargs=self.norm_kwargs,
            bias=self.bias,
            aggr="max",
        )

        self.proj: Optional[MLP] = None
        if self.proj_channels is not None:
            self.proj = MLP(
                [self.encoder.out_channels, self.proj_channels],
                plain_last=False,
                act=self.act,
                act_kwargs=self.act_kwargs,
                act_first=self.act_first,
                norm=self.norm,
                norm_kwargs=self.norm_kwargs,
                bias=self.bias,
            )

        self.dropout = dropout
        self.global_pool = CatPool(global_pool) if is_iterable(global_pool) else create_pool(global_pool)  # type: ignore[arg-type]
        self.head = self.configure_head()

    @property
    def embedding_dim(self) -> int:
        # count the output channels of the encoder
        base = self.proj_channels if self.proj_channels is not None else self.encoder.out_channels
        # In case we have multiple global pools, we need to multiply the base by the number of pools
        if isinstance(self.global_pool, CatPool):
            return base * self.global_pool.num_pools

        return base

    def configure_head(self) -> nn.Module:
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

    def reset_classifier(
        self,
        num_classes: int,
        global_pool: PoolLike | Sequence[PoolLike] = "max",
        **kwargs: Any,
    ) -> None:
        self.num_classes = num_classes
        self.global_pool = CatPool(global_pool) if is_iterable(global_pool) else create_pool(global_pool)  # type: ignore[arg-type]
        self.head = self.configure_head()

    def forward_features(self, x: OptTensor, pos: Tensor, batch: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        if self.stnet is not None:
            pos = self.stnet(pos, batch)

        knn_x = pos
        x = torch.cat([x, pos], dim=1) if x is not None else pos
        x, batch = self.encoder(x, batch, knn_x=knn_x)

        if self.proj is not None:
            x = self.proj(x)

        return x, pos, batch

    def forward_head(self, x: Tensor, batch: Tensor, pre_logits: bool = False) -> Tensor:
        x = self.global_pool(x, batch)
        if len(self.head_channels) == 0:
            x = F.dropout(x, p=self.dropout, training=self.training)
        return x if pre_logits else self.head(x)

    def forward(self, x: OptTensor, pos: Tensor, batch: Tensor) -> Tensor:
        x, _, batch = self.forward_features(x, pos, batch)
        return self.forward_head(x, batch)


class DGCNNSegmentation(SegmentationModel):
    """
    Semantic segmentation model as described in the paper
    ["Dynamic Graph CNN for Learning on Point Clouds"](https://arxiv.org/abs/1801.07829)
    by Yue Wang, Yongbin Sun, Ziwei Liu, Sanjay E. Sarma, Michael M. Bronstein, Justin M. Solomon.

    DGCNN introduces the EdgeConv operator, which computes features on dynamically constructed
    k-nearest neighbor (k-NN) graphs at each layer. Graphs are recomputed in the learned feature space,
    allowing the network to capture both local geometric relationships and long-range semantic structures.

    Args:
        in_channels: Number of input channels.
        num_classes: Number of output classes.
        spatial_dim: Spatial dimension of the input point cloud.
        stnet_local_channels: List of channels for the local spatial transformer network.
            If None, the spatial transformer network is not used.
        stnet_global_channels: List of channels for the global spatial transformer network.
            If None, the spatial transformer network is not used.
        proj_channels: Number of channels for the projection layer after the encoder.
        channels: List of channels for each encoder block.
        head_channels: List of channels for each head block.
        num_neighbors: Maximum number of neighbors for each encoder block.
        act: Activation function.
        act_kwargs: Additional arguments for the activation function.
        act_first: Whether to apply activation before normalization.
        norm: Normalization layer type.
        norm_kwargs: Additional arguments for the normalization layer.
        bias: Whether to use bias in linear / MLP layers.
        dropout: Dropout probability before the classification head.
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        *,
        spatial_dim: int = 3,
        stnet_local_channels: Optional[Sequence[int]] = None,
        stnet_global_channels: Optional[Sequence[int]] = None,
        proj_channels: int = 1024,
        channels: Sequence[Union[int, Sequence[int]]],
        head_channels: Optional[Union[int, Sequence[int]]] = None,
        num_neighbors: Union[int, Sequence[int]],
        act: Union[str, Callable, None] = "relu",
        act_kwargs: Optional[Dict[str, Any]] = None,
        act_first: bool = False,
        norm: Union[str, Callable, None] = "batch_norm",
        norm_kwargs: Optional[Dict[str, Any]] = None,
        bias: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__(in_channels=in_channels + spatial_dim, num_classes=num_classes)
        self.spatial_dim = spatial_dim
        self.stnet_local_channels = ensure_list(stnet_local_channels) if stnet_local_channels is not None else None
        self.stnet_global_channels = ensure_list(stnet_global_channels) if stnet_global_channels is not None else None
        self.proj_channels = proj_channels
        self.channels = ensure_list(channels)
        self.head_channels = ensure_list(head_channels, none_as_empty=True)
        self.num_neighbors = ensure_list(num_neighbors)
        self.act = act
        self.act_kwargs = act_kwargs
        self.act_first = act_first
        self.norm = norm
        self.norm_kwargs = norm_kwargs
        self.bias = bias
        self.dropout = dropout

        self.stnet: Optional[TNet] = None
        if self.stnet_local_channels is not None and self.stnet_global_channels is not None:
            self.stnet = TNet(
                local_channels=self.stnet_local_channels,
                global_channels=self.stnet_global_channels,
                k=self.spatial_dim,
                act=self.act,
                act_kwargs=self.act_kwargs,
                act_first=self.act_first,
                norm=self.norm,
                norm_kwargs=self.norm_kwargs,
                bias=self.bias,
                dropout=self.dropout,
                aggr="max",
            )

        self.encoder = DGCNNEncoder(
            channels=[self.in_channels] + self.channels,
            num_neighbors=self.num_neighbors,
            act=self.act,
            act_kwargs=self.act_kwargs,
            act_first=self.act_first,
            norm=self.norm,
            norm_kwargs=self.norm_kwargs,
            bias=self.bias,
            aggr="max",
        )

        self.proj = MLP(
            [self.encoder.out_channels, self.proj_channels],
            plain_last=False,
            act=self.act,
            act_kwargs=self.act_kwargs,
            act_first=self.act_first,
            norm=self.norm,
            norm_kwargs=self.norm_kwargs,
            bias=self.bias,
        )

        self.head = MLP(
            [self.embedding_dim] + self.head_channels + [self.num_classes],
            act=self.act,
            act_kwargs=self.act_kwargs,
            act_first=self.act_first,
            norm=self.norm,
            norm_kwargs=self.norm_kwargs,
            bias=self.bias,
            dropout=self.dropout,
            plain_last=True,
        )

    @property
    def embedding_dim(self) -> int:
        return self.encoder.out_channels + self.proj_channels

    def reset_classifier(self, num_classes: int, **kwargs: Any) -> None:
        self.num_classes = num_classes
        kwargs.setdefault("act", self.act)
        kwargs.setdefault("act_kwargs", self.act_kwargs)
        kwargs.setdefault("act_first", self.act_first)
        kwargs.setdefault("norm", self.norm)
        kwargs.setdefault("norm_kwargs", self.norm_kwargs)
        kwargs.setdefault("bias", self.bias)
        kwargs.setdefault("dropout", self.dropout)
        self.head = MLP(
            [self.embedding_dim] + self.head_channels + [self.num_classes],
            plain_last=True,
            **kwargs,
        )

    def forward_features(self, x: OptTensor, pos: Tensor, batch: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        if self.stnet is not None:
            pos = self.stnet(pos, batch)
        knn_x = pos
        x = torch.cat([x, pos], dim=1) if x is not None else pos
        x, batch = self.encoder(x, batch, knn_x=knn_x)

        x_proj = self.proj(x)
        x_global = global_max_pool(x_proj, batch)  # (B, proj_channels)
        x_global = x_global[batch]  # (N, proj_channels)
        x = torch.cat([x_global, x], dim=1)

        return x, pos, batch

    def forward_head(self, x: Tensor, batch: Tensor, pre_logits: bool = False) -> Tensor:
        return self.head(x)

    def forward(self, x: OptTensor, pos: Tensor, batch: Tensor) -> Tensor:
        x, _, batch = self.forward_features(x, pos, batch)
        return self.forward_head(x, batch)


class DGCNNPartSegmentation(SegmentationModel):
    """
    Part segmentation model as described in the paper
    ["Dynamic Graph CNN for Learning on Point Clouds"](https://arxiv.org/abs/1801.07829)
    by Yue Wang, Yongbin Sun, Ziwei Liu, Sanjay E. Sarma, Michael M. Bronstein, Justin M. Solomon.

    Extends the DGCNN encoder with a category-conditioned global feature branch for
    part-level segmentation (e.g. ShapeNet parts).

    Args:
        in_channels: Number of input channels.
        num_classes: Number of output part classes (across all categories).
        num_categories: Number of object categories for the category embedding.
        cat_embed_channels: Number of channels for the category embedding.
        spatial_dim: Spatial dimension of the input point cloud.
        stnet_edge_channels: Hidden channels for the DynamicTNet EdgeConv MLP.
            If None, the spatial transformer network is not used.
        stnet_local_channels: Channels for the DynamicTNet local (point-wise) MLP.
        stnet_global_channels: Channels for the DynamicTNet global MLP.
        stnet_num_neighbors: Number of kNN neighbors for the DynamicTNet.
        proj_channels: Number of channels for the projection layer after the encoder.
        channels: List of channels for each encoder block.
        head_channels: List of channels for each head block.
        num_neighbors: Maximum number of neighbors for each encoder block.
        act: Activation function.
        act_kwargs: Additional arguments for the activation function.
        act_first: Whether to apply activation before normalization.
        norm: Normalization layer type.
        norm_kwargs: Additional arguments for the normalization layer.
        bias: Whether to use bias in linear / MLP layers.
        dropout: Dropout probability in the decoder head.
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        *,
        num_categories: int,
        cat_embed_channels: int = 64,
        spatial_dim: int = 3,
        stnet_edge_channels: Optional[Sequence[int]] = None,
        stnet_local_channels: Optional[Sequence[int]] = None,
        stnet_global_channels: Optional[Sequence[int]] = None,
        stnet_num_neighbors: int = 20,
        proj_channels: int = 1024,
        channels: Sequence[Union[int, Sequence[int]]],
        head_channels: Optional[Union[int, Sequence[int]]] = None,
        num_neighbors: Union[int, Sequence[int]],
        act: Union[str, Callable, None] = "relu",
        act_kwargs: Optional[Dict[str, Any]] = None,
        act_first: bool = False,
        norm: Union[str, Callable, None] = "batch_norm",
        norm_kwargs: Optional[Dict[str, Any]] = None,
        bias: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__(in_channels=in_channels + spatial_dim, num_classes=num_classes)
        self.spatial_dim = spatial_dim
        self.num_categories = num_categories
        self.cat_embed_channels = cat_embed_channels
        self.proj_channels = proj_channels
        self.channels = ensure_list(channels)
        self.head_channels = ensure_list(head_channels, none_as_empty=True)
        self.num_neighbors = ensure_list(num_neighbors)
        self.act = act
        self.act_kwargs = act_kwargs
        self.act_first = act_first
        self.norm = norm
        self.norm_kwargs = norm_kwargs
        self.bias = bias
        self.dropout = dropout

        self.stnet: Optional[DynamicTNet] = None
        if stnet_edge_channels is not None:
            self.stnet = DynamicTNet(
                edge_channels=stnet_edge_channels,
                local_channels=stnet_local_channels or [],
                global_channels=stnet_global_channels or [],
                k=self.spatial_dim,
                num_neighbors=stnet_num_neighbors,
                act=self.act,
                act_kwargs=self.act_kwargs,
                act_first=self.act_first,
                norm=self.norm,
                norm_kwargs=self.norm_kwargs,
                bias=self.bias,
            )

        self.encoder = DGCNNEncoder(
            channels=[self.in_channels] + self.channels,
            num_neighbors=self.num_neighbors,
            act=self.act,
            act_kwargs=self.act_kwargs,
            act_first=self.act_first,
            norm=self.norm,
            norm_kwargs=self.norm_kwargs,
            bias=self.bias,
            aggr="max",
        )
        self.proj = MLP(
            [self.encoder.out_channels, self.proj_channels],
            plain_last=False,
            act=self.act,
            act_kwargs=self.act_kwargs,
            act_first=self.act_first,
            norm=self.norm,
            norm_kwargs=self.norm_kwargs,
            bias=self.bias,
        )
        self.cat_embed = MLP(
            [self.num_categories, self.cat_embed_channels],
            plain_last=False,
            act=self.act,
            act_kwargs=self.act_kwargs,
            act_first=self.act_first,
            norm=self.norm,
            norm_kwargs=self.norm_kwargs,
            bias=self.bias,
        )

        self.head = MLP(
            [self.embedding_dim] + self.head_channels + [self.num_classes],
            act=self.act,
            act_kwargs=self.act_kwargs,
            act_first=self.act_first,
            norm=self.norm,
            norm_kwargs=self.norm_kwargs,
            bias=self.bias,
            dropout=self.dropout,
            plain_last=True,
        )

    @property
    def embedding_dim(self) -> int:
        return self.encoder.out_channels + self.proj_channels + self.cat_embed_channels

    def reset_classifier(self, num_classes: int, **kwargs: Any) -> None:
        self.num_classes = num_classes
        kwargs.setdefault("act", self.act)
        kwargs.setdefault("act_kwargs", self.act_kwargs)
        kwargs.setdefault("act_first", self.act_first)
        kwargs.setdefault("norm", self.norm)
        kwargs.setdefault("norm_kwargs", self.norm_kwargs)
        kwargs.setdefault("bias", self.bias)
        kwargs.setdefault("dropout", self.dropout)
        self.head = MLP(
            [self.embedding_dim] + self.head_channels + [self.num_classes],
            plain_last=True,
            **kwargs,
        )

    def forward_features(
        self, x: OptTensor, pos: Tensor, batch: Tensor, category: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor]:
        if self.stnet is not None:
            pos = self.stnet(pos, batch)
        knn_x = pos
        x = torch.cat([x, pos], dim=1) if x is not None else pos
        x, batch = self.encoder(x, batch, knn_x=knn_x)

        x_proj = self.proj(x)
        x_global = global_max_pool(x_proj, batch)  # (B, proj_channels)

        x_cat = self.cat_embed(category)  # (B, cat_embed_channels)
        x_global = torch.cat([x_global, x_cat], dim=1)  # (B, proj_channels + cat_embed_channels)
        x_global = x_global[batch]  # (N, proj_channels + cat_embed_channels)

        x = torch.cat([x_global, x], dim=1)
        return x, pos, batch

    def forward_head(self, x: Tensor, batch: Tensor, pre_logits: bool = False) -> Tensor:
        return self.head(x)

    def forward(self, x: OptTensor, pos: Tensor, batch: Tensor, category: Tensor) -> Tensor:
        x, _, batch = self.forward_features(x, pos, batch, category)
        return self.forward_head(x, batch)


def _dgcnn_antao_s3dis_cfg(area: int) -> dict[str, Any]:
    return dict(
        name=f"dgcnn-antao.s3dis.area{area}",
        task="segmentation",
        weights=f"hf://torch-pointcloud/dgcnn/dgcnn-antao.s3dis.area{area}.pt",
        hparams=dict(
            in_channels=6,
            num_classes=13,
            spatial_dim=3,
            proj_channels=1024,
            channels=[[64, 64], [64, 64], [64]],
            head_channels=[512, 256],
            num_neighbors=20,
            act="leaky_relu",
            act_kwargs={"negative_slope": 0.2},
            act_first=False,
            norm="batch_norm",
            bias=True,
            dropout=0.5,
        ),
    )


@register_model(
    "dgcnn-antao.modelnet40.1024",
    task="classification",
    weights="hf://torch-pointcloud/dgcnn/dgcnn-antao.modelnet40.1024.pt",
    hparams=dict(
        in_channels=0,
        num_classes=40,
        spatial_dim=3,
        channels=[64, 64, 128, 256],
        proj_channels=1024,
        head_channels=[512, 256],
        num_neighbors=20,
        act="leaky_relu",
        act_kwargs={"negative_slope": 0.2},
        act_first=False,
        norm="batch_norm",
        bias=True,
        dropout=0.5,
        global_pool=["max", "mean"],
    ),
    transforms=T.SampleFarthestPoints(pos_key="pos", keys=["normal"], num_samples=1024),
)
def dgcnn_antao_modelnet40_1024_cls(**hparams: Any) -> DGCNNClassification:
    # from the repo: https://github.com/antao97/dgcnn.pytorch
    return DGCNNClassification(**hparams)


@register_model(
    "dgcnn-antao.modelnet40.2048",
    task="classification",
    weights="hf://torch-pointcloud/dgcnn/dgcnn-antao.modelnet40.2048.pt",
    hparams=dict(
        in_channels=0,
        num_classes=40,
        spatial_dim=3,
        channels=[64, 64, 128, 256],
        proj_channels=1024,
        head_channels=[512, 256],
        num_neighbors=20,
        act="leaky_relu",
        act_kwargs={"negative_slope": 0.2},
        act_first=False,
        norm="batch_norm",
        bias=True,
        dropout=0.5,
        global_pool=["max", "mean"],
    ),
    transforms=T.SampleFarthestPoints(pos_key="pos", keys=["normal"], num_samples=2048),
)
def dgcnn_antao_modelnet40_2048_cls(**hparams: Any) -> DGCNNClassification:
    # from the repo: https://github.com/antao97/dgcnn.pytorch
    return DGCNNClassification(**hparams)


@register_model(
    "dgcnn-antao.shapenetpart",
    task="segmentation",
    weights="hf://torch-pointcloud/dgcnn/dgcnn-antao.shapenetpart.pt",
    hparams=dict(
        in_channels=0,
        num_classes=50,
        num_categories=16,
        cat_embed_channels=64,
        spatial_dim=3,
        stnet_edge_channels=[64, 128],
        stnet_local_channels=[1024],
        stnet_global_channels=[512, 256],
        stnet_num_neighbors=40,
        proj_channels=1024,
        channels=[[64, 64], [64, 64], [64]],
        head_channels=[256, 256, 128],
        num_neighbors=40,
        act="leaky_relu",
        act_kwargs={"negative_slope": 0.2},
        act_first=False,
        norm="batch_norm",
        bias=True,
        dropout=0.5,
    ),
    transforms=T.SampleFarthestPoints(
        keys=[DataKeys.NORMAL, DataKeys.SEGMENT],
        pos_key=DataKeys.POS,
        num_samples=2048,
    ),
    # transforms=T.RandomSample(keys=[DataKeys.POS, DataKeys.NORMAL, DataKeys.SEGMENT], num_samples=2048),
)
def dgcnn_antao_shapenet_partseg(**hparams: Any) -> DGCNNPartSegmentation:
    return DGCNNPartSegmentation(**hparams)


@register_model(**_dgcnn_antao_s3dis_cfg(1))
def dgcnn_antao_s3dis_area1_seg(**hparams: Any) -> DGCNNSegmentation:
    return DGCNNSegmentation(**hparams)


@register_model(**_dgcnn_antao_s3dis_cfg(2))
def dgcnn_antao_s3dis_area2_seg(**hparams: Any) -> DGCNNSegmentation:
    return DGCNNSegmentation(**hparams)


@register_model(**_dgcnn_antao_s3dis_cfg(3))
def dgcnn_antao_s3dis_area3_seg(**hparams: Any) -> DGCNNSegmentation:
    return DGCNNSegmentation(**hparams)


@register_model(**_dgcnn_antao_s3dis_cfg(4))
def dgcnn_antao_s3dis_area4_seg(**hparams: Any) -> DGCNNSegmentation:
    return DGCNNSegmentation(**hparams)


@register_model(**_dgcnn_antao_s3dis_cfg(5))
def dgcnn_antao_s3dis_area5_seg(**hparams: Any) -> DGCNNSegmentation:
    return DGCNNSegmentation(**hparams)


@register_model(**_dgcnn_antao_s3dis_cfg(6))
def dgcnn_antao_s3dis_area6_seg(**hparams: Any) -> DGCNNSegmentation:
    return DGCNNSegmentation(**hparams)


@register_model(
    "dgcnn-antao.scannet20",
    task="segmentation",
    weights="hf://torch-pointcloud/dgcnn/dgcnn-antao.scannet20.pt",
    hparams=dict(
        in_channels=6,
        num_classes=20,
        spatial_dim=3,
        proj_channels=1024,
        channels=[[64, 64], [64, 64], [64]],
        head_channels=[512, 256],
        num_neighbors=20,
        act="leaky_relu",
        act_kwargs={"negative_slope": 0.2},
        act_first=False,
        norm="batch_norm",
        bias=True,
        dropout=0.5,
    ),
    transforms=T.Compose(
        [
            # The model was trained on 20 classes without a dedicated class for "unknown" objects,
            # so we relabel the segmentation labels from [0, 20] -> [0, 19] and set "unknown" objects to 255.
            T.Relabel(keys=DataKeys.SEGMENT, labels=list(range(1, 21)), default=255),
            T.Divide(keys=DataKeys.COLOR, divisor=255),
            T.CopyItems(keys=DataKeys.POS, names="norm_pos"),
            T.DivideKey(keys="norm_pos", div_keys="scene_max"),
            T.SubtractKey(keys=DataKeys.POS, sub_keys="block_center"),
        ]
    ),
)
def dgcnn_antao_scannet_semseg(**hparams: Any) -> DGCNNSegmentation:
    return DGCNNSegmentation(**hparams)
