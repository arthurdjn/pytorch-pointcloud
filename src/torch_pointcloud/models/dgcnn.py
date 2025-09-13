from typing import Any, Callable, Dict, List, Literal, NamedTuple, Optional, Sequence, Tuple, Union, overload

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn import MLP, DynamicEdgeConv

from torch_pointcloud.layers import PoolLike, create_cls_head, create_pool
from torch_pointcloud.layers.tnet import TNet
from torch_pointcloud.utils.conversion import ensure_list, ensure_tuple, ensure_tuple_size
from torch_pointcloud.utils.types import AggrType, OptTensor

from ._base import ClassificationModel


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

    @overload
    def forward(
        self,
        x: Tensor,
        batch: Tensor,
        return_intermediates: Literal[True] = True,
    ) -> Union[Tuple[Tensor, Tensor], Tuple[Tensor, Tensor, List[DGCNNIntermediate]]]: ...

    @overload
    def forward(
        self,
        x: Tensor,
        batch: Tensor,
        return_intermediates: Literal[False] = False,
    ) -> Union[Tuple[Tensor, Tensor], Tuple[Tensor, Tensor, List[DGCNNIntermediate]]]: ...

    def forward(
        self,
        x: Tensor,
        batch: Tensor,
        return_intermediates: bool = False,
    ) -> Union[Tuple[Tensor, Tensor], Tuple[Tensor, Tensor, List[DGCNNIntermediate]]]:
        intermediates = []
        for block in self.blocks:
            if return_intermediates:
                intermediates.append(DGCNNIntermediate(x, batch))

            x = block(x, batch)

        if return_intermediates:
            return x, batch, intermediates[::-1]
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
        channels: Sequence[int],
        num_neighbors: Union[int, Sequence[int]],
        act: Union[str, Callable, None] = "relu",
        act_kwargs: Optional[Dict[str, Any]] = None,
        act_first: bool = False,
        norm: Union[str, Callable, None] = "batch_norm",
        norm_kwargs: Optional[Dict[str, Any]] = None,
        bias: bool = True,
        stnet_local_channels: Sequence[int],
        stnet_global_channels: Sequence[int],
        dropout: float = 0.0,
        global_pool: PoolLike = "max",
    ):
        super().__init__(in_channels=in_channels + spatial_dim, num_classes=num_classes)
        self.spatial_dim = spatial_dim

        self.stnet = TNet(
            local_channels=stnet_local_channels,
            global_channels=stnet_global_channels,
            k=spatial_dim,
            act=act,
            act_kwargs=act_kwargs,
            act_first=act_first,
            norm=norm,
            norm_kwargs=norm_kwargs,
            bias=bias,
            dropout=dropout,
            aggr="max",
        )

        channels = [self.in_channels] + ensure_list(channels)
        self.encoder = DGCNNEncoder(
            channels=channels,
            num_neighbors=num_neighbors,
            act=act,
            act_kwargs=act_kwargs,
            act_first=act_first,
            norm=norm,
            norm_kwargs=norm_kwargs,
            bias=bias,
            aggr="max",
        )

        self.dropout = dropout
        self.global_pool = create_pool(global_pool)
        self.head = create_cls_head(num_features=self.embedding_dim, num_classes=self.num_classes)

    @property
    def embedding_dim(self) -> int:
        return sum(self.encoder.channels[1:])

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
    ) -> Tuple[Tensor, Tensor, List[DGCNNIntermediate]]: ...

    @overload
    def forward_features(
        self,
        x: OptTensor,
        pos: Tensor,
        batch: Tensor,
        return_intermediates: Literal[False] = False,
    ) -> Tuple[Tensor, Tensor]: ...

    def forward_features(
        self,
        x: OptTensor,
        pos: Tensor,
        batch: Tensor,
        return_intermediates: bool = False,
    ) -> Any:
        pos = self.stnet(pos, batch)
        x = torch.cat([x, pos], dim=1) if x is not None else pos
        return self.encoder(x, batch, return_intermediates=return_intermediates)

    def forward_head(self, x: Tensor, batch: Tensor, pre_logits: bool = False) -> Tensor:
        x = self.global_pool(x, batch)
        if self.dropout:
            x = F.dropout(x, p=self.dropout, training=self.training)
        return x if pre_logits else self.head(x)

    def forward(self, x: OptTensor, pos: Tensor, batch: Tensor) -> Tensor:
        x, batch, intermediates = self.forward_features(x, pos, batch, return_intermediates=True)
        x = torch.cat([x] + [intermediate.x for intermediate in intermediates[:-1]], dim=1)
        return self.forward_head(x, batch)
