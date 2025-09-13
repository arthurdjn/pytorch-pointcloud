from typing import Any, Callable, Dict, NamedTuple, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn import MLP, DynamicEdgeConv, global_max_pool

from torch_pointcloud.layers import PoolLike, create_cls_head, create_pool
from torch_pointcloud.layers.tnet import TNet
from torch_pointcloud.utils.conversion import ensure_list, ensure_tuple, ensure_tuple_size
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
        out_channels: int,
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
        block = MLP(
            [2 * in_channels, out_channels],
            plain_last=False,
            act=act,
            act_kwargs=act_kwargs,
            norm=norm,
            norm_kwargs=norm_kwargs,
            bias=bias,
        )
        self.conv = DynamicEdgeConv(block, k=num_neighbors, aggr=aggr)

    def forward(self, x: Tensor, batch: Tensor) -> Tensor:
        return self.conv(x, batch)


class DGCNNEncoder(nn.Module):
    def __init__(
        self,
        channels: Sequence[int],
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
        self.channels = ensure_tuple(channels)
        size = len(self.channels) - 1
        self.num_neighbors = ensure_tuple_size(num_neighbors, size=size)
        self.blocks = nn.ModuleList()
        for i in range(size):
            block = DGCNNEncoderBlock(
                in_channels=self.channels[i],
                # in_channels=self.channels[i] if i < size - 1 else self.channels[i] * 2,
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

    def forward(self, x: Tensor, batch: Tensor) -> Tuple[Tensor, Tensor]:
        x_list = []
        for block in self.blocks:
            x = block(x, batch)
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
        channels: List of channels for each encoder block.
        num_neighbors: Maximum number of neighbors for each encoder block.
        act: Activation function.
        act_kwargs: Additional arguments for the activation function.
        act_first: Whether to apply activation before normalization.
        norm: Normalization layer type.
        norm_kwargs: Additional arguments for the normalization layer.
        bias: Whether to use bias in linear / MLP layers.
        stnet_local_channels: List of channels for the local spatial transformer network.
        stnet_global_channels: List of channels for the global spatial transformer network.
        dropout: Dropout probability before the classification head.
        global_pool: Global pooling operation for final feature aggregation.

    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        *,
        spatial_dim: int = 3,
        stnet_local_channels: Sequence[int],
        stnet_global_channels: Sequence[int],
        head_channels: Optional[Union[int, Sequence[int]]] = None,
        channels: Sequence[int],
        num_neighbors: Union[int, Sequence[int]],
        act: Union[str, Callable, None] = "relu",
        act_kwargs: Optional[Dict[str, Any]] = None,
        act_first: bool = False,
        norm: Union[str, Callable, None] = "batch_norm",
        norm_kwargs: Optional[Dict[str, Any]] = None,
        bias: bool = True,
        dropout: float = 0.0,
        global_pool: PoolLike = "max",
    ):
        super().__init__(in_channels=in_channels + spatial_dim, num_classes=num_classes)
        self.spatial_dim = spatial_dim
        self.stnet_local_channels = ensure_list(stnet_local_channels)
        self.stnet_global_channels = ensure_list(stnet_global_channels)
        self.head_channels = ensure_list(head_channels, none_as_empty=True)
        self.channels = ensure_list(channels)
        self.num_neighbors = ensure_list(num_neighbors)
        self.act = act
        self.act_kwargs = act_kwargs
        self.act_first = act_first
        self.norm = norm
        self.norm_kwargs = norm_kwargs
        self.bias = bias

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

        self.dropout = dropout
        self.global_pool = create_pool(global_pool)
        self.head = MLP(
            [self.embedding_dim] + self.head_channels + [self.num_classes],
            act=self.act,
            act_kwargs=self.act_kwargs,
            act_first=self.act_first,
            norm=self.norm,
            norm_kwargs=self.norm_kwargs,
            bias=self.bias,
            dropout=[0] * len(self.head_channels) + [self.dropout],
            plain_last=True,
        )

    @property
    def embedding_dim(self) -> int:
        return sum(self.channels)

    def reset_classifier(self, num_classes: int, global_pool: PoolLike = "max", **kwargs: Any) -> None:
        self.num_classes = num_classes
        self.global_pool = create_pool(global_pool)
        self.head = MLP(
            [self.embedding_dim] + self.head_channels + [self.num_classes],
            act=self.act,
            act_kwargs=self.act_kwargs,
            act_first=self.act_first,
            norm=self.norm,
            norm_kwargs=self.norm_kwargs,
            bias=self.bias,
            dropout=[0] * len(self.head_channels) + [self.dropout],
            plain_last=True,
        )

    def forward_features(self, x: OptTensor, pos: Tensor, batch: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        pos = self.stnet(pos, batch)
        x = torch.cat([x, pos], dim=1) if x is not None else pos
        x, batch = self.encoder(x, batch)
        return x, pos, batch

    def forward_head(self, x: Tensor, batch: Tensor, pre_logits: bool = False) -> Tensor:
        x = self.global_pool(x, batch)
        if self.dropout:
            x = F.dropout(x, p=self.dropout, training=self.training)
        return x if pre_logits else self.head(x)

    def forward(self, x: OptTensor, pos: Tensor, batch: Tensor) -> Tensor:
        x, _, batch = self.forward_features(x, pos, batch)
        return self.forward_head(x, batch)


class DGCNNSemanticSegmentation(SegmentationModel):
    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        *,
        spatial_dim: int = 3,
        stnet_local_channels: Union[int, Sequence[int]],
        stnet_global_channels: Union[int, Sequence[int]],
        embedding_channels: int = 1024,
        channels: Sequence[int],
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
        self.stnet_local_channels = ensure_list(stnet_local_channels)
        self.stnet_global_channels = ensure_list(stnet_global_channels)
        self.embedding_channels = embedding_channels
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
        self.embed = nn.Linear(sum(self.channels), self.embedding_channels)

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
        return sum(self.channels) + self.embedding_channels

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
        pos = self.stnet(pos, batch)
        x = torch.cat([x, pos], dim=1) if x is not None else pos
        x, batch = self.encoder(x, batch)

        x_embed = self.embed(x)
        x_global = global_max_pool(x_embed, batch)  # (B, embedding_channels)
        x_global = x_global[batch]  # (N, embedding_channels)
        x = torch.cat([x_global, x], dim=1)

        return x, pos, batch

    def forward_head(self, x: Tensor, batch: Tensor, pre_logits: bool = False) -> Tensor:
        return self.head(x)

    def forward(self, x: OptTensor, pos: Tensor, batch: Tensor) -> Tensor:
        x, _, batch = self.forward_features(x, pos, batch)
        return self.forward_head(x, batch)


@register_model("dgcnn-base", task="classification")
def dgcnn_base_cls(in_channels: int, num_classes: int, **kwargs: Any) -> DGCNNClassification:
    """Base classification model as described in [original paper](https://arxiv.org/abs/1801.07829)
    and [official implementation](https://github.com/WangYueFt/dgcnn).
    """
    hparams = dict(
        in_channels=in_channels,
        num_classes=num_classes,
        spatial_dim=3,
        stnet_local_channels=[64],
        stnet_global_channels=[128, 1024],
        head_channels=[512, 256],
        channels=[64, 64, 128, 256],
        num_neighbors=20,
        act="leaky_relu",
        act_kwargs={"negative_slope": 0.2},
        act_first=False,
        norm="batch_norm",
        bias=True,
        dropout=0.5,
        global_pool="max",
    )
    hparams.update(kwargs)
    return DGCNNClassification(**hparams)  # type: ignore[arg-type]


@register_model("dgcnn-base", task="segmentation")
def dgcnn_base_semseg(in_channels: int, num_classes: int, **kwargs: Any) -> DGCNNSemanticSegmentation:
    """Base semantic segmentation model as described in [original paper](https://arxiv.org/abs/1801.07829)
    and [official implementation](https://github.com/WangYueFt/dgcnn).
    """
    hparams = dict(
        in_channels=in_channels,
        num_classes=num_classes,
        spatial_dim=3,
        stnet_local_channels=[64, 128, 1024],
        stnet_global_channels=[512, 256],
        embedding_channels=1024,
        channels=[64, 64, 128],
        head_channels=[256, 256, 128],
        num_neighbors=20,
        act="leaky_relu",
        act_kwargs={"negative_slope": 0.2},
        act_first=False,
        norm="batch_norm",
        bias=True,
        dropout=0.5,
    )
    hparams.update(kwargs)
    return DGCNNSemanticSegmentation(**hparams)  # type: ignore[arg-type]
