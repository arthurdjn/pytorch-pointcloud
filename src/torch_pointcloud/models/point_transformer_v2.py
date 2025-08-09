from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional, Sequence, Tuple, overload

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from torch_pointcloud.layers import (
    ActLike,
    NormLike,
    PoolLike,
    create_act,
    create_cls_head,
    create_norm,
    create_pool,
    linear_block,
)
from torch_pointcloud.layers.dropouts import DropPath
from torch_pointcloud.utils.conversion import ensure_tuple, ensure_tuple_size
from torch_pointcloud.utils.imports import optional_import
from torch_pointcloud.utils.ops import softmax, voxel_grid
from torch_pointcloud.utils.types import OptTensor, ValueCollection

if TYPE_CHECKING:
    from torch_cluster import knn_graph
    from torch_scatter import scatter_sum, segment_csr

knn_graph, _ = optional_import("torch_cluster", name="knn_graph")
scatter_sum, _ = optional_import("torch_scatter", name="scatter_sum")
segment_csr, _ = optional_import("torch_scatter", name="segment_csr")


class GroupedVectorAttention(nn.Module):
    def __init__(
        self,
        channels: int,
        num_groups: int,
        attn_drop: float = 0.0,
        qkv_bias: bool = True,
        pe_multiplier: bool = False,
        pe_bias: bool = True,
        norm: NormLike = "batch_norm1d",
        act: ActLike = "relu",
    ):
        super().__init__()
        if channels % num_groups != 0:
            raise ValueError(f"channels ({channels}) must be divisible by num_groups ({num_groups})")

        self.channels = channels
        self.num_groups = num_groups

        self.q = nn.Sequential(
            nn.Linear(channels, channels, bias=qkv_bias),
            create_norm(norm, channels),
            create_act(act),
        )
        self.k = nn.Sequential(
            nn.Linear(channels, channels, bias=qkv_bias),
            create_norm(norm, channels),
            create_act(act),
        )
        self.v = nn.Linear(channels, channels, bias=qkv_bias)

        self.pe_multiplier: Optional[nn.Module] = None
        if pe_multiplier:
            self.pe_multiplier = nn.Sequential(
                nn.Linear(3, channels),
                create_norm(norm, channels),
                create_act(act),
                nn.Linear(channels, channels),
            )

        self.pe_bias: Optional[nn.Module] = None
        if pe_bias:
            self.pe_bias = nn.Sequential(
                nn.Linear(3, channels),
                create_norm(norm, channels),
                create_act(act),
                nn.Linear(channels, channels),
            )

        self.weight_encoding = nn.Sequential(
            nn.Linear(channels, num_groups),
            create_norm(norm, num_groups),
            create_act(act),
            nn.Linear(num_groups, num_groups),
        )

        self.attn_drop = nn.Dropout(attn_drop)

    def forward(self, features: Tensor, coords: Tensor, edge_index: Tensor) -> Tensor:
        query, key, value = self.q(features), self.k(features), self.v(features)

        row, col = edge_index
        value = value[row]
        coords = coords[row] - coords[col]
        relation_qk = key[row] - query[col]

        if self.pe_multiplier is not None:
            factor = self.pe_multiplier(coords)
            relation_qk = relation_qk * factor

        if self.pe_bias is not None:
            bias = self.pe_bias(coords)
            relation_qk = relation_qk + bias
            value = value + bias

        weight = self.weight_encoding(relation_qk)
        weight = self.attn_drop(softmax(weight, col))

        value = value.reshape(-1, self.num_groups, self.channels // self.num_groups)
        features = value * weight.unsqueeze(-1)
        features = features.reshape(-1, self.channels)
        features = scatter_sum(features, col, dim=0)
        return features


class Block(nn.Module):
    def __init__(
        self,
        channels: int,
        num_groups: int,
        qkv_bias: bool = True,
        pe_multiplier: bool = False,
        pe_bias: bool = True,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        norm: NormLike = "batch_norm1d",
        act: ActLike = "relu",
    ):
        super().__init__()
        self.attn = GroupedVectorAttention(
            channels=channels,
            num_groups=num_groups,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            pe_multiplier=pe_multiplier,
            pe_bias=pe_bias,
            norm=norm,
            act=act,
        )
        self.fc1 = nn.Linear(channels, channels, bias=False)
        self.fc3 = nn.Linear(channels, channels, bias=False)
        self.norm1 = create_norm(norm, channels)
        self.norm2 = create_norm(norm, channels)
        self.norm3 = create_norm(norm, channels)
        self.act = create_act(act)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, features: Tensor, coords: Tensor, edge_index: Tensor) -> Tensor:
        shortcut = features
        features = self.act(self.norm1(self.fc1(features)))
        features = self.attn(features, coords, edge_index)
        features = self.act(self.norm2(features))
        features = self.norm3(self.fc3(features))
        features = self.drop_path(features) + shortcut
        features = self.act(features)
        return features


class GridPool(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        grid_size: float,
        bias: bool = False,
        reduce: str = "max",
        norm: NormLike = "batch_norm1d",
        act: ActLike = "relu",
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.grid_size = grid_size
        self.reduce = reduce

        self.fc = nn.Linear(in_channels, out_channels, bias=bias)
        self.norm = create_norm(norm, out_channels)
        self.act = create_act(act)

    @overload
    def forward(
        self,
        features: Tensor,
        coords: Tensor,
        batch: Tensor,
        return_inverse: Literal[True] = True,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]: ...

    @overload
    def forward(
        self,
        features: Tensor,
        coords: Tensor,
        batch: Tensor,
        return_inverse: Literal[False] = False,
    ) -> Tuple[Tensor, Tensor, Tensor]: ...

    def forward(
        self,
        features: Tensor,
        coords: Tensor,
        batch: Tensor,
        return_inverse: bool = False,
    ) -> Tuple[Tensor, ...]:
        features = self.act(self.norm(self.fc(features)))

        # NOTE: evaluate difference with this version
        # and the consecutive_cluster version in kpconv.py
        start = segment_csr(
            coords,
            torch.cat([batch.new_zeros(1), torch.cumsum(batch.bincount(), dim=0)]),
            reduce="min",
        )
        cluster = voxel_grid(coords - start[batch], size=self.grid_size, batch=batch, start=0)
        _, cluster, counts = torch.unique(cluster, sorted=True, return_inverse=True, return_counts=True)
        _, sorted_cluster_indices = torch.sort(cluster)
        idx_ptr = torch.cat([counts.new_zeros(1), torch.cumsum(counts, dim=0)])
        coords = segment_csr(coords[sorted_cluster_indices], idx_ptr, reduce="mean")
        features = segment_csr(features[sorted_cluster_indices], idx_ptr, reduce=self.reduce)
        batch = batch[idx_ptr[:-1]]

        if return_inverse:
            return features, coords, batch, cluster
        return features, coords, batch


class InversePool(nn.Module):
    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        bias: bool = True,
        norm: Optional[NormLike] = "batch_norm1d",
        act: Optional[ActLike] = "relu",
    ):
        super().__init__()
        self.in_channels = in_channels
        self.skip_channels = skip_channels
        self.out_channels = out_channels

        self.proj = linear_block(
            in_features=in_channels,
            out_features=out_channels,
            bias=bias,
            norm=norm,
            act=act,
            dropout=None,
            order="lna",
        )
        self.proj_skip = linear_block(
            in_features=in_channels,
            out_features=out_channels,
            bias=bias,
            norm=norm,
            act=act,
            dropout=None,
            order="lna",
        )

    def forward(self, features: Tensor, skip_features: Tensor, inverse: Tensor) -> Tensor:
        features = self.proj(features)
        skip_features = self.proj_skip(skip_features)
        return skip_features + features[inverse]


class EncoderBlock(nn.Module):
    def __init__(
        self,
        depth: int,
        channels: int,
        num_groups: int,
        num_neighbors: int,
        qkv_bias: bool = True,
        pe_multiplier: bool = False,
        pe_bias: bool = True,
        norm: NormLike = "batch_norm1d",
        act: ActLike = "relu",
        attn_drop: ValueCollection[float] = 0.0,
        drop_path: ValueCollection[float] = 0.0,
        downsample: Optional[GridPool] = None,
    ):
        super().__init__()
        attn_drop = ensure_tuple_size(attn_drop, depth)
        drop_path = ensure_tuple_size(drop_path, depth)

        self.num_neighbors = num_neighbors
        self.downsample = downsample

        self.blocks = nn.ModuleList()
        for i in range(depth):
            block = Block(
                channels=channels,
                num_groups=num_groups,
                qkv_bias=qkv_bias,
                pe_multiplier=pe_multiplier,
                pe_bias=pe_bias,
                attn_drop=attn_drop[i],
                drop_path=drop_path[i],
                norm=norm,
                act=act,
            )
            self.blocks.append(block)

    @overload
    def forward(
        self,
        features: Tensor,
        coords: Tensor,
        batch: Tensor,
        return_inverse: Literal[True] = True,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]: ...

    @overload
    def forward(
        self,
        features: Tensor,
        coords: Tensor,
        batch: Tensor,
        return_inverse: Literal[False] = False,
    ) -> Tuple[Tensor, Tensor, Tensor]: ...

    def forward(
        self,
        features: Tensor,
        coords: Tensor,
        batch: Tensor,
        return_inverse: bool = False,
    ) -> Tuple[Tensor, ...]:
        if return_inverse and self.downsample is None:
            raise ValueError("`return_inverse` is only supported if `downsample` is provided")

        if self.downsample is not None:
            features, coords, batch, pooling_inverse = self.downsample(features, coords, batch, return_inverse=True)

        edge_index = knn_graph(coords, self.num_neighbors, batch, loop=True)
        for block in self.blocks:
            features = block(features, coords, edge_index)

        if return_inverse:
            return features, coords, batch, pooling_inverse
        return features, coords, batch


class DecoderBlock(nn.Module):
    def __init__(
        self,
        depth: int,
        channels: int,
        num_groups: int,
        num_neighbors: int,
        qkv_bias: bool = True,
        pe_multiplier: bool = False,
        pe_bias: bool = True,
        norm: NormLike = "batch_norm1d",
        act: ActLike = "relu",
        attn_drop: ValueCollection[float] = 0.0,
        drop_path: ValueCollection[float] = 0.0,
        upsample: Optional[InversePool] = None,
    ):
        super().__init__()
        attn_drop = ensure_tuple_size(attn_drop, depth)
        drop_path = ensure_tuple_size(drop_path, depth)

        self.num_neighbors = num_neighbors
        self.upsample = upsample

        self.blocks = nn.ModuleList()
        for i in range(depth):
            block = Block(
                channels=channels,
                num_groups=num_groups,
                qkv_bias=qkv_bias,
                pe_multiplier=pe_multiplier,
                pe_bias=pe_bias,
                attn_drop=attn_drop[i],
                drop_path=drop_path[i],
                norm=norm,
                act=act,
            )
            self.blocks.append(block)

    def forward(
        self,
        features: Tensor,
        skip_features: Tensor,
        skip_coords: Tensor,
        skip_batch: Tensor,
        pooling_inverse: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        if self.upsample is not None:
            features = self.upsample(features, skip_features, pooling_inverse)

        edge_index = knn_graph(skip_coords, self.num_neighbors, skip_batch, loop=True)
        for block in self.blocks:
            features = block(features, skip_coords, edge_index)
        return features, skip_coords, skip_batch


def create_encoder_blocks(
    depths: Sequence[int],
    channels: Sequence[int],
    num_groups: Sequence[int],
    num_neighbors: Sequence[int],
    grid_sizes: Sequence[float],
    norm: NormLike = "batch_norm1d",
    act: ActLike = "relu",
    qkv_bias: bool = True,
    pe_multiplier: bool = False,
    pe_bias: bool = True,
    attn_drop: float = 0.0,
    drop_path: float = 0.0,
) -> nn.ModuleList:
    depths = ensure_tuple(depths)
    n = len(depths)
    channels = ensure_tuple_size(channels, size=n, extra_msg="Encoder length `channels` != `depths`.")
    num_groups = ensure_tuple_size(num_groups, size=n, extra_msg="Encoder length `num_groups` != `depths`.")
    num_neighbors = ensure_tuple_size(num_neighbors, size=n, extra_msg="Encoder length `num_neighbors` != `depths`.")
    grid_sizes = ensure_tuple_size(grid_sizes, size=n - 1, extra_msg="Encoder length `grid_sizes` != `depths` - 1.")

    # Pre-compute the drop paths for each encoder block.
    # For example, if the drop path is 0.3, and the depths are (2, 3, 4),
    # then the drop paths for each block, at each stage, are:
    # - block 0: [0.0000, 0.0375]
    # - block 1: [0.0750, 0.1125, 0.1500]
    # - block 2: [0.1875, 0.2250, 0.2625, 0.3000]
    drop_paths = torch.split(torch.linspace(0, drop_path, sum(depths)), list(depths))

    blocks = nn.ModuleList()
    for i in range(n):
        downsample: Optional[GridPool] = None
        if i > 0:
            downsample = GridPool(
                in_channels=channels[i - 1],
                out_channels=channels[i],
                grid_size=grid_sizes[i - 1],
                reduce="max",
            )

        block = EncoderBlock(
            depth=depths[i],
            channels=channels[i],
            num_groups=num_groups[i],
            num_neighbors=num_neighbors[i],
            norm=norm,
            act=act,
            qkv_bias=qkv_bias,
            pe_multiplier=pe_multiplier,
            pe_bias=pe_bias,
            attn_drop=attn_drop,
            drop_path=drop_paths[i].tolist(),
            downsample=downsample,
        )
        blocks.append(block)
    return blocks


def create_decoder_blocks(
    depths: Sequence[int],
    channels: Sequence[int],
    skip_channels: Sequence[int],
    num_groups: Sequence[int],
    num_neighbors: Sequence[int],
    norm: NormLike = "batch_norm1d",
    act: ActLike = "relu",
    qkv_bias: bool = True,
    pe_multiplier: bool = False,
    pe_bias: bool = True,
    attn_drop: float = 0.0,
    drop_path: float = 0.0,
) -> nn.ModuleList:
    depths = ensure_tuple(depths)
    n = len(depths)
    channels = ensure_tuple_size(channels, size=n + 1, extra_msg="Decoder length `channels` != `depths` + 1.")
    skip_channels = ensure_tuple_size(skip_channels, size=n, extra_msg="Decoder length `skip_channels` != `depths`.")
    num_groups = ensure_tuple_size(num_groups, size=n, extra_msg="Decoder length `num_groups` != `depths`.")
    num_neighbors = ensure_tuple_size(num_neighbors, size=n, extra_msg="Decoder length `num_neighbors` != `depths`.")

    # Pre-compute the drop paths for each (decoder) block.
    # The drop path is the same as the encoder block, but in reverse order.
    # For example, if the drop path is 0.3, and the depths are (4, 3, 2),
    # then the drop paths for each block at each stage are:
    # - block 0: [0.3000, 0.2625, 0.2250, 0.1875]
    # - block 1: [0.1500, 0.1125, 0.0750]
    # - block 2: [0.0375, 0.0000]
    drop_paths = torch.split(torch.linspace(0, drop_path, sum(depths)), list(depths))[::-1]

    blocks = nn.ModuleList()
    for i in range(n):
        upsample = InversePool(
            in_channels=channels[i],
            skip_channels=skip_channels[i],
            out_channels=channels[i + 1],
            norm=norm,
            act=act,
        )

        # NOTE: For decoder blocks, the drop paths should be in reverse order (i.e. higher -> lower within each block)
        block = DecoderBlock(
            depth=depths[i],
            channels=channels[i + 1],
            num_groups=num_groups[i],
            num_neighbors=num_neighbors[i],
            norm=norm,
            act=act,
            qkv_bias=qkv_bias,
            pe_multiplier=pe_multiplier,
            pe_bias=pe_bias,
            attn_drop=attn_drop,
            drop_path=drop_paths[i].tolist()[::-1],
            upsample=upsample,
        )
        blocks.append(block)
    return blocks


class PointTransformerV2Classification(nn.Module):
    r"""Implementation of the Point Transformer V2 model for classification as described in the paper
    [Point Transformer V2: Grouped Vector Attention and Partition-based Pooling](https://arxiv.org/abs/2210.05666)
    by Xiaoyang Wu, Yixing Lao, Li Jiang, Xihui Liu, Hengshuang Zhao.

    This implementation is based on the original implementation from [Pointcept](https://github.com/Pointcept/Pointcept).

    Note:
        This implementation requires [`torch-cluster`](https://github.com/rusty1s/pytorch_cluster) to be installed.
        and [`torch-scatter`](https://github.com/rusty1s/pytorch_scatter) to be installed.

    Args:
        in_channels: Number of input channels.
        num_classes: Number of output classes.
        encoder_depths: Number of encoder blocks for each stage.
        encoder_channels: Number of channels for each encoder block.
        encoder_num_groups: Number of groups for each encoder block.
        encoder_num_neighbors: Number of edge_index for each encoder block.
        grid_sizes: Size of the grid for each stage.
        norm: Normalization layer to use.
        act: Activation function to use.
        qkv_bias: Whether to use bias in the QKV linear layer.
        pe_multiplier: Whether to use a multiplier for the PE.
        pe_bias: Whether to use bias in the PE linear layer.
        drop_path: Drop path rate.
        dropout: Dropout rate.
        global_pool: Global pooling method to use.

    Inputs:
        features: Float tensor of shape $(N, \text{in_channels})$.
        coords: Float tensor of shape $(N, 3)$.
        batch: Long tensor of shape $(N,)$.

    Outputs:
        Logits tensor of shape $(N, \text{num_classes})$.
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        grid_sizes: Sequence[float] = (0.06, 0.12, 0.24, 0.48),
        encoder_depths: Sequence[int] = (1, 2, 2, 6, 2),
        encoder_channels: Sequence[int] = (48, 96, 192, 384, 512),
        encoder_num_groups: Sequence[int] = (6, 12, 24, 48, 64),
        encoder_num_neighbors: Sequence[int] = (8, 16, 16, 16, 16),
        norm: NormLike = "batch_norm1d",
        act: ActLike = "relu",
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        pe_multiplier: bool = False,
        pe_bias: bool = True,
        drop_path: float = 0.0,
        dropout: float = 0.0,
        global_pool: PoolLike = "max",
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes

        self.embedding = nn.Sequential(
            nn.Linear(in_channels, encoder_channels[0]),
            create_norm(norm, encoder_channels[0]),
            create_act(act),
        )

        self.encoder = self.configure_encoder_blocks(
            depths=encoder_depths,
            channels=encoder_channels,
            num_groups=encoder_num_groups,
            num_neighbors=encoder_num_neighbors,
            grid_sizes=grid_sizes,
            qkv_bias=qkv_bias,
            pe_multiplier=pe_multiplier,
            pe_bias=pe_bias,
            attn_drop=attn_drop,
            drop_path=drop_path,
        )

        self.dropout = dropout
        self.global_pool = create_pool(global_pool)
        self.head = create_cls_head(num_features=self.embedding_dim, num_classes=self.num_classes)

    @property
    def embedding_dim(self) -> int:
        return self.encoder[-1].blocks[-1].fc3.out_features  # type: ignore[index, union-attr]

    def configure_encoder_blocks(self, *args: Any, **kwargs: Any) -> nn.ModuleList:
        return create_encoder_blocks(*args, **kwargs)

    def reset_classifier(self, num_classes: int, global_pool: PoolLike = "max", **kwargs: Any) -> None:
        """Resets the classification head with new parameters.

        Note:
            To set an empty classification head, use `num_classes=0`.

        Args:
            num_classes: Number of output classes.
            global_pool: Pooling method to aggregate point features ("max" or "mean").
            **kwargs: Additional keyword arguments to pass to the classification head.
        """
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
        r"""Forward features through the encoder blocks, before the global pooling.

        Args:
            features: Additional point features of shape $(N, \text{features_dim})$.
            coords: Coordinates of shape $(N, 3)$.
            batch: Batch indices for each point of shape $(N,)$.
            return_intermediates: Whether to return the intermediate features.

        Returns:
            features: Pre-pooling features of shape $(N, \text{embedding_dim})$.
            coords: Coordinates of shape $(N, 3)$.
            batch: Batch indices for each point of shape $(N,)$.
            intermediates: If `return_intermediates` is `True`, a list of dictionaries containing the intermediate features,
                coordinates, batch indices and pooling inverse for each encoder block.
        """
        features = features if features is not None else coords
        features = self.embedding(features)

        intermediates = []
        for i, block in enumerate(self.encoder):
            intermediate = {"features": features, "coords": coords, "batch": batch}

            features, coords, batch, pooling_inverse = block(features, coords, batch, return_inverse=True)
            if i > 0:
                intermediate["pooling_inverse"] = pooling_inverse
                intermediates.append(intermediate)

        if return_intermediates:
            return features, coords, batch, intermediates
        return features, coords, batch

    def forward_head(self, features: Tensor, batch: Tensor, pre_logits: bool = False) -> Tensor:
        r"""Forward pass of the classification head from pre-pooling features.

        Args:
            features: Pre-pooling features of shape $(N, \text{embedding_dim})$.
            batch: Batch indices for each point of shape $(N,)$.
            pre_logits: Whether to return pre-logits.

        Returns:
            Classification logits of shape $(B, \text{num_classes})$.
        """
        features = self.global_pool(features, batch)
        if self.dropout:
            features = F.dropout(features, p=float(self.dropout), training=self.training)
        return features if pre_logits else self.head(features)

    def forward(self, features: OptTensor, coords: Tensor, batch: Tensor) -> Tensor:
        r"""Forward pass of the classification network.

        Args:
            features: Additional point features of shape $(N, \text{features_dim})$.
            coords: Coordinates of shape $(N, 3)$.
            batch: Batch indices for each point of shape $(N,)$.

        Returns:
            Classification logits of shape $(B, \text{num_classes})$.
        """
        features, _, batch = self.forward_features(features, coords, batch)
        return self.forward_head(features, batch)


class PointTransformerV2Segmentation(nn.Module):
    r"""Implementation of the Point Transformer V2 model for semantic segmentation as described in the paper
    [Point Transformer V2: Grouped Vector Attention and Partition-based Pooling](https://arxiv.org/abs/2210.05666)
    by Xiaoyang Wu, Yixing Lao, Li Jiang, Xihui Liu, Hengshuang Zhao.

    This implementation is based on the original implementation from [Pointcept](https://github.com/Pointcept/Pointcept).

    Note:
        This implementation requires [`torch-cluster`](https://github.com/rusty1s/pytorch_cluster) to be installed.
        and [`torch-scatter`](https://github.com/rusty1s/pytorch_scatter) to be installed.

    Args:
        in_channels: Number of input channels.
        num_classes: Number of output classes.
        encoder_depths: Number of encoder blocks for each stage.
        encoder_channels: Number of channels for each encoder block.
        encoder_num_groups: Number of groups for each encoder block.
        encoder_num_neighbors: Number of edge_index for each encoder block.
        grid_sizes: Size of the grid for each stage.
        norm: Normalization layer to use.
        act: Activation function to use.
        qkv_bias: Whether to use bias in the QKV linear layer.
        pe_multiplier: Whether to use a multiplier for the PE.
        pe_bias: Whether to use bias in the PE linear layer.
        drop_path: Drop path rate.
        dropout: Dropout rate.
        global_pool: Global pooling method to use.

    Inputs:
        features: Float tensor of shape $(N, \text{in_channels})$.
        coords: Float tensor of shape $(N, 3)$.
        batch: Long tensor of shape $(N,)$.

    Outputs:
        Segmentation logits of shape $(N, \text{num_classes})$.
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        grid_sizes: Sequence[float] = (0.06, 0.12, 0.24, 0.48),
        encoder_depths: Sequence[int] = (1, 2, 2, 6, 2),
        encoder_channels: Sequence[int] = (48, 96, 192, 384, 512),
        encoder_num_groups: Sequence[int] = (6, 12, 24, 48, 64),
        encoder_num_neighbors: Sequence[int] = (8, 16, 16, 16, 16),
        decoder_depths: Sequence[int] = (1, 1, 1, 1),
        decoder_channels: Sequence[int] = (384, 192, 96, 48),
        decoder_num_groups: Sequence[int] = (48, 24, 12, 6),
        decoder_num_neighbors: Sequence[int] = (16, 16, 16, 16),
        norm: NormLike = "batch_norm1d",
        act: ActLike = "relu",
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        pe_multiplier: bool = False,
        pe_bias: bool = True,
        drop_path: float = 0.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes

        self.embedding = nn.Sequential(
            nn.Linear(in_channels, encoder_channels[0]),
            create_norm(norm, encoder_channels[0]),
            create_act(act),
        )

        self.encoder = self.configure_encoder_blocks(
            depths=encoder_depths,
            channels=encoder_channels,
            grid_sizes=grid_sizes,
            num_groups=encoder_num_groups,
            num_neighbors=encoder_num_neighbors,
            qkv_bias=qkv_bias,
            pe_multiplier=pe_multiplier,
            pe_bias=pe_bias,
            attn_drop=attn_drop,
            drop_path=drop_path,
            norm=norm,
            act=act,
        )

        self.decoder = self.configure_decoder_blocks(
            depths=decoder_depths,
            channels=[encoder_channels[-1]] + list(decoder_channels),
            skip_channels=list(encoder_channels[:-1])[::-1],
            num_groups=decoder_num_groups,
            num_neighbors=decoder_num_neighbors,
            qkv_bias=qkv_bias,
            pe_multiplier=pe_multiplier,
            pe_bias=pe_bias,
            attn_drop=attn_drop,
            norm=norm,
            act=act,
        )

        self.dropout = dropout
        self.head = create_cls_head(num_features=decoder_channels[-1], num_classes=self.num_classes)

    @property
    def embedding_dim(self) -> int:
        return self.encoder[-1].blocks[-1].fc3.out_features  # type: ignore[index, union-attr]

    def configure_encoder_blocks(self, *args: Any, **kwargs: Any) -> nn.ModuleList:
        return create_encoder_blocks(*args, **kwargs)

    def configure_decoder_blocks(self, *args: Any, **kwargs: Any) -> nn.ModuleList:
        return create_decoder_blocks(*args, **kwargs)

    def reset_head(self, num_classes: int, **kwargs: Any) -> None:
        """Resets the head with new class parameters.

        Note:
            To set an empty head, use `num_classes=0`.

        Args:
            num_classes: Number of output classes.
            **kwargs: Additional keyword arguments to pass to the classification head.
        """
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
        r"""Forward features through the encoder blocks, before the global pooling.

        Args:
            features: Additional point features of shape $(N, \text{features_dim})$.
            coords: Coordinates of shape $(N, 3)$.
            batch: Batch indices for each point of shape $(N,)$.
            return_intermediates: Whether to return the intermediate features.

        Returns:
            features: Pre-pooling features of shape $(N, \text{embedding_dim})$.
            coords: Coordinates of shape $(N, 3)$.
            batch: Batch indices for each point of shape $(N,)$.
            intermediates: If `return_intermediates` is `True`, a list of dictionaries containing the intermediate features,
                coordinates, batch indices and pooling inverse for each encoder block.
        """
        features = features if features is not None else coords
        features = self.embedding(features)

        intermediates = []
        for i, block in enumerate(self.encoder):
            intermediate = {"features": features, "coords": coords, "batch": batch}

            features, coords, batch, *pooling_inverse = block(features, coords, batch, return_inverse=i > 0)
            if i > 0:
                intermediate["pooling_inverse"] = pooling_inverse[0]
                intermediates.append(intermediate)

        if return_intermediates:
            return features, coords, batch, intermediates
        return features, coords, batch

    def forward_decoder(
        self,
        features: Tensor,
        intermediates: List[Dict[str, Tensor]],
    ) -> Tuple[Tensor, Tensor, Tensor]:
        for block, intermediate in zip(self.decoder, reversed(intermediates)):
            intermediate = {f"skip_{k}" if k != "pooling_inverse" else k: v for k, v in intermediate.items()}
            features, coords, batch = block(features, **intermediate)
        return features, coords, batch

    def forward_head(self, features: Tensor, pre_logits: bool = False) -> Tensor:
        r"""Forward pass of the classification head from up-sampled features.

        Args:
            features: Pre-pooling features of shape $(N, \text{embedding_dim})$.
            pre_logits: Whether to return pre-logits.

        Returns:
            Segmentation logits of shape $(N, \text{num_classes})$.
        """
        if self.dropout:
            features = F.dropout(features, p=float(self.dropout), training=self.training)
        return features if pre_logits else self.head(features)

    def forward(self, features: OptTensor, coords: Tensor, batch: Tensor) -> Tensor:
        r"""Forward pass of the model.

        Args:
            features: Additional point features of shape $(N, \text{features_dim})$.
            coords: Coordinates of shape $(N, 3)$.
            batch: Batch indices for each point of shape $(N,)$.

        Returns:
            Segmentation logits of shape $(N, \text{num_classes})$.
        """
        features, _, _, intermediates = self.forward_features(features, coords, batch, return_intermediates=True)
        features, _, _ = self.forward_decoder(features, intermediates)
        return self.forward_head(features)
