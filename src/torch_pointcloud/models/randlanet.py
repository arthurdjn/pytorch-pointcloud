from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional, Sequence, Tuple, Union, overload

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from torch_pointcloud.layers import MLP, ActLike, NormLike, PoolLike, create_cls_head, create_pool
from torch_pointcloud.utils.conversion import ensure_tuple, ensure_tuple_size
from torch_pointcloud.utils.imports import optional_import
from torch_pointcloud.utils.ops import decimate, softmax
from torch_pointcloud.utils.types import OptTensor

from .pointnet2 import create_fp_blocks

if TYPE_CHECKING:
    from torch_cluster import knn_graph
    from torch_scatter import scatter_add


knn_graph, _ = optional_import("torch_cluster", "knn_graph")
scatter_add, _ = optional_import("torch_scatter", "scatter_add")


class LocalFeatureAggregation(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        bias: bool = False,
        act: ActLike = None,
        norm: NormLike = None,
    ):
        super().__init__()
        self.mlp1 = MLP([in_channels, out_channels // 2], bias=bias, act=act, norm=norm, dropout=None)
        self.mlp_attn = MLP([out_channels, out_channels], bias=None, act=None, norm=None, dropout=None)
        self.mlp2 = MLP([out_channels, out_channels], bias=bias, act=act, norm=norm, dropout=None)

    def forward(self, features: Tensor, coords: Tensor, edge_index: Tensor) -> Tensor:
        src_idx, dst_idx = edge_index

        features_j = features[src_idx]
        coords_i = coords[dst_idx]
        coords_j = coords[src_idx]

        coords_diff = coords_j - coords_i
        distance = torch.sqrt((coords_diff * coords_diff).sum(1, keepdim=True))
        relative_infos = torch.cat([coords_i, coords_j, coords_diff, distance], dim=1)  # E, 10

        local_spatial_encoding = self.mlp1(relative_infos)  # E, channels//2
        local_features = torch.cat([features_j, local_spatial_encoding], dim=1)  # E, channels

        att_features = self.mlp_attn(local_features)  # E, channels
        att_scores = softmax(att_features, dst_idx)  # E, channels

        weighted_features = att_scores * local_features  # E, channels
        out = scatter_add(weighted_features, dst_idx, dim=0, dim_size=features.size(0))  # N, channels

        return self.mlp2(out)  # N, channels


class DilatedResidualBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_neighbors: int,
    ):
        super().__init__()
        self.num_neighbors = num_neighbors
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.mlp1 = MLP([in_channels, out_channels // 8])
        self.shortcut = MLP([in_channels, out_channels], act=None)
        self.mlp2 = MLP([out_channels // 2, out_channels], act=None)

        self.lfa1 = LocalFeatureAggregation(10, out_channels // 4)
        self.lfa2 = LocalFeatureAggregation(10, out_channels // 2)

        # self.lrelu = nn.LeakyReLU(**lrelu02_kwargs)
        self.lrelu = nn.LeakyReLU(0.2)

    def forward(self, x: Tensor, pos: Tensor, batch: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        edge_index = knn_graph(pos, self.num_neighbors, batch=batch, loop=True)

        shortcut_of_x = self.shortcut(x)  # N, d_out
        x = self.mlp1(x)  # N, d_out//8
        x = self.lfa1(x, pos, edge_index)  # N, d_out//2
        x = self.lfa2(x, pos, edge_index)  # N, d_out//2
        x = self.mlp2(x)  # N, d_out
        x = self.lrelu(x + shortcut_of_x)  # N, d_out

        return x, pos, batch


def create_encoder_blocks(
    in_channels: int,
    channels: Sequence[int],
    num_neighbors: Union[int, Sequence[int]],
) -> nn.ModuleList:
    num_blocks = len(channels)
    extra_msg = f"`{{param}}` must be a sequence of the same length as the number of blocks {num_blocks}."
    num_neighbors = ensure_tuple_size(num_neighbors, num_blocks, extra_msg=extra_msg.format(param="num_neighbors"))

    blocks = []
    for out_channels, k in zip(channels, num_neighbors):  # type: ignore[arg-type]
        blocks.append(DilatedResidualBlock(in_channels, out_channels, num_neighbors=k))
        in_channels = out_channels
    return nn.ModuleList(blocks)


class RandLANetClassification(nn.Module):
    """RandLANet classification model as described in the paper
    [RandLA-Net: Efficient Semantic Segmentation of Large-Scale Point Clouds](https://arxiv.org/abs/1911.11236)
    by Qingyong Hu, Bo Yang, Linhai Xie, Stefano Rosa, Yulan Guo, Zhihua Wang, Niki Trigoni, Andrew Markham.

    RandLA-Net is an efficient point cloud processing architecture that uses random sampling instead of more
    computationally expensive alternatives like farthest point sampling. The model consists of encoding blocks
    that progressively downsample the point cloud while learning local features through dilated residual blocks
    and local feature aggregation. For classification, features are aggregated globally after encoding. The
    architecture is memory-efficient and can process large point clouds effectively.

    Args:
        in_channels: Number of input channels.
        num_classes: Number of classes.
        stem_channels: Number of channels in the stem layer.
        encoder_channels: Number of channels in the encoder blocks.
        decimation: Decimation factor.
        num_neighbors: Number of neighbors for each point.
        aggr_channels: Number of channels in the aggregation layer.
        dropout: Dropout rate.
        global_pool: Global pooling operation to use.
        bias: Whether to use bias in the MLPs.
        act: Activation function to use.
        norm: Normalization function to use.
        order: Order of the MLPs.
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        *,
        stem_channels: Optional[int] = 8,
        encoder_channels: Sequence[int],
        decimation: int = 4,
        num_neighbors: Union[int, Sequence[int]] = 16,
        aggr_channels: Optional[Union[int, Sequence[int]]] = None,
        dropout: float = 0.0,
        global_pool: PoolLike = "max",
    ):
        super().__init__()
        self.decimation = decimation

        self.stem = nn.Linear(in_channels, stem_channels) if stem_channels else None
        in_channels = stem_channels if stem_channels else in_channels
        self.encoder_blocks = create_encoder_blocks(in_channels, encoder_channels, num_neighbors)

        in_channels = encoder_channels[-1]
        aggr_channels = ensure_tuple(aggr_channels) if aggr_channels else None
        self.aggr = MLP(in_channels=in_channels, channels=aggr_channels) if aggr_channels else None

        self.embedding_dim = aggr_channels[-1] if aggr_channels else encoder_channels[-1]
        self.global_pool = create_pool(global_pool)
        self.dropout = dropout
        self.head = create_cls_head(self.embedding_dim, num_classes)

    def reset_classifier(self, num_classes: int, global_pool: PoolLike = "max", **kwargs: Any) -> None:
        self.num_classes = num_classes
        self.global_pool = create_pool(global_pool)
        self.head = create_cls_head(num_features=self.embedding_dim, num_classes=self.num_classes, **kwargs)

    @overload
    def forward_features(
        self,
        features: OptTensor,
        coords: Tensor,
        batch: Tensor,
        return_intermediates: Literal[True],
    ) -> Tuple[Tensor, Tensor, Tensor, List[Dict[str, Tensor]]]: ...

    @overload
    def forward_features(
        self,
        features: OptTensor,
        coords: Tensor,
        batch: Tensor,
        return_intermediates: Literal[False] = False,
    ) -> Tuple[Tensor, Tensor, Tensor]: ...

    def forward_features(
        self,
        features: OptTensor,
        coords: Tensor,
        batch: Tensor,
        return_intermediates: bool = False,
    ) -> Any:
        features = features if features is not None else coords
        if self.stem is not None:
            features = self.stem(features)

        # Store the intermediate results if specified with `return_intermediates=True`
        intermediates = [{"features": features, "coords": coords, "batch": batch}] if return_intermediates else []
        for i, block in enumerate(self.encoder_blocks):
            features, coords, batch = block(features, coords, batch)
            (features, coords), batch = decimate((features, coords), batch, self.decimation)
            if return_intermediates and i < len(self.encoder_blocks) - 1:
                # NOTE: Do not store the last result, as it will be the returned output.
                intermediates.append({"features": features, "coords": coords, "batch": batch})

        if self.aggr is not None:
            features = self.aggr(features)

        if return_intermediates:
            return features, coords, batch, intermediates
        return features, coords, batch

    def forward_head(self, x: Tensor, batch: Tensor, pre_logits: bool = False) -> Tensor:
        """Forward pass of the classification head from pre-pooling features.

        Args:
            x: Pre-pooling features of shape $(N, mlp2_dims[-1])$ where $N$ is the batch size.
            batch: Batch indices for each point of shape $(N,)$.
            pre_logits: Whether to return pre-logits. Defaults to False.

        Returns:
            Classification logits of shape $(B, num_classes)$.
        """
        x = self.global_pool(x, batch)
        if self.dropout:
            x = F.dropout(x, p=float(self.dropout), training=self.training)
        return x if pre_logits else self.head(x)

    def forward(self, features: OptTensor, coords: Tensor, batch: Tensor) -> Tensor:
        """Forward pass of the classification model.

        Args:
            features: Additional point features of shape $(N, features_dim)$.
            coords: Point coordinates of shape $(N, coords_dim)$.
            batch_idxs: Batch indices for each point of shape $(N,)$.

        Returns:
            Classification logits of shape $(B, num_classes)$.
        """
        features, _, batch = self.forward_features(features, coords, batch)
        return self.forward_head(features, batch, pre_logits=False)


class RandLANetSegmentation(nn.Module):
    """RandLANet segmentation model as described in the paper
    [RandLA-Net: Efficient Semantic Segmentation of Large-Scale Point Clouds](https://arxiv.org/abs/1911.11236)
    by Qingyong Hu, Bo Yang, Linhai Xie, Stefano Rosa, Yulan Guo, Zhihua Wang, Niki Trigoni, Andrew Markham.

    RandLA-Net is an efficient point cloud processing architecture that uses random sampling instead of more
    computationally expensive alternatives like farthest point sampling. The model consists of encoding blocks
    that progressively downsample the point cloud while learning local features through dilated residual blocks
    and local feature aggregation. For classification, features are aggregated globally after encoding. The
    architecture is memory-efficient and can process large point clouds effectively.

    Args:
        in_channels: Number of input channels.
        num_classes: Number of classes.
        stem_channels: Number of channels in the stem layer.
        encoder_channels: Number of channels in the encoder blocks.
        fp_channels: Number of channels in the feature propagation blocks.
        decimation: Decimation factor.
        num_neighbors: Number of neighbors for each point.
        aggr_channels: Number of channels in the aggregation layer.
        bias: Whether to use bias in the MLPs.
        act: Activation function to use.
        norm: Normalization function to use.
        order: Order of the MLPs.
        dropout: Dropout rate.

    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        *,
        stem_channels: Optional[int] = 8,
        encoder_channels: Sequence[int],
        fp_channels: Sequence[Sequence[int]],
        decimation: int = 4,
        num_neighbors: Union[int, Sequence[int]] = 16,
        aggr_channels: Optional[Union[int, Sequence[int]]] = None,
        bias: bool = False,
        act: ActLike = "relu",
        norm: NormLike = "batch_norm1d",
        order: str = "lan",
        dropout: float = 0.0,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.decimation = decimation

        self.stem = nn.Linear(in_channels, stem_channels) if stem_channels else None
        in_channels = stem_channels if stem_channels else in_channels
        skip_channels = [in_channels]

        self.encoder_blocks = create_encoder_blocks(in_channels, encoder_channels, num_neighbors)
        skip_channels.extend(encoder_channels[:-1])

        aggr_channels = ensure_tuple(aggr_channels) if aggr_channels else None
        self.aggr = MLP(in_channels=encoder_channels[-1], channels=aggr_channels) if aggr_channels else None

        self.fp_blocks = create_fp_blocks(
            in_channels=aggr_channels[-1] if aggr_channels is not None else encoder_channels[-1],
            skip_channels=skip_channels[::-1],
            fp_channels=fp_channels,
            bias=bias,
            act=act,
            norm=norm,
            order=order,
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
        features: OptTensor,
        coords: Tensor,
        batch: Tensor,
        return_intermediates: Literal[True],
    ) -> Tuple[Tensor, Tensor, Tensor, List[Dict[str, Tensor]]]: ...

    @overload
    def forward_features(
        self,
        features: OptTensor,
        coords: Tensor,
        batch: Tensor,
        return_intermediates: Literal[False] = False,
    ) -> Tuple[Tensor, Tensor, Tensor]: ...

    def forward_features(
        self,
        features: OptTensor,
        coords: Tensor,
        batch: Tensor,
        return_intermediates: bool = False,
    ) -> Any:
        features = features if features is not None else coords
        if self.stem is not None:
            features = self.stem(features)

        # NOTE: We only store the intermediate results if specified with `return_intermediates=True`
        intermediates = [{"features": features, "coords": coords, "batch": batch}] if return_intermediates else []
        for i, block in enumerate(self.encoder_blocks):
            features, coords, batch = block(features, coords, batch)
            (features, coords), batch = decimate((features, coords), batch, self.decimation)
            if return_intermediates and i < len(self.encoder_blocks) - 1:
                # NOTE: Do not store the last result, as it will be the returned output.
                intermediates.append({"features": features, "coords": coords, "batch": batch})

        if self.aggr is not None:
            features = self.aggr(features)

        if return_intermediates:
            return features, coords, batch, intermediates
        return features, coords, batch

    def forward_decoder(
        self,
        features: Tensor,
        coords: Tensor,
        batch: Tensor,
        intermediates: List[Dict[str, Tensor]],
    ) -> Tensor:
        for block, intermediate in zip(self.fp_blocks, reversed(intermediates)):
            features_skip = intermediate["features"]
            coords_skip = intermediate["coords"]
            batch_skip = intermediate["batch"]

            features, coords, batch = block(features, coords, batch, features_skip, coords_skip, batch_skip)
        return features

    def forward_head(self, x: Tensor, pre_logits: bool = False) -> Tensor:
        if self.dropout:
            x = F.dropout(x, p=float(self.dropout), training=self.training)
        return x if pre_logits else self.head(x)

    def forward(self, features: OptTensor, coords: Tensor, batch: Tensor) -> Tensor:
        features, coords, batch, intermediates = self.forward_features(
            features, coords, batch, return_intermediates=True
        )
        features = self.forward_decoder(features, coords, batch, intermediates)
        return self.forward_head(features)
