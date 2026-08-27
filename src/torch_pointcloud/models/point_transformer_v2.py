"""Point Transformer V2 classification and segmentation models.

{{ paper("2210.05666") }}
"""

from typing import TYPE_CHECKING, Any, Callable, Dict, List, Literal, Optional, Sequence, Tuple, Union, overload

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch_geometric.nn import MLP
from torch_geometric.nn.pool import voxel_grid

import torch_pointcloud.transforms as T
from torch_pointcloud.layers import (
    PoolLike,
    create_pool,
)
from torch_pointcloud.layers.act import create_act
from torch_pointcloud.layers.dropouts import DropPath
from torch_pointcloud.layers.norms import create_norm
from torch_pointcloud.models._base import ClassificationModel, SegmentationModel
from torch_pointcloud.models._registry import register_model
from torch_pointcloud.utils.conversion import ensure_tuple, ensure_tuple_size
from torch_pointcloud.utils.data import DataKeys
from torch_pointcloud.utils.imports import _TORCH_CLUSTER_GITHUB_URL, _TORCH_SCATTER_GITHUB_URL, optional_import
from torch_pointcloud.utils.ops import softmax
from torch_pointcloud.utils.types import OptTensor, ValueCollection

if TYPE_CHECKING:
    from torch_cluster import knn_graph
    from torch_scatter import scatter_sum, segment_csr

knn_graph, _ = optional_import("torch_cluster", name="knn_graph", url=_TORCH_CLUSTER_GITHUB_URL)
scatter_sum, _ = optional_import("torch_scatter", name="scatter_sum", url=_TORCH_SCATTER_GITHUB_URL)
segment_csr, _ = optional_import("torch_scatter", name="segment_csr", url=_TORCH_SCATTER_GITHUB_URL)


class GroupedVectorAttention(nn.Module):
    """Vector attention over a neighborhood graph, with one weight vector shared by each group of channels.

    The relation between a query and its neighbor keys is optionally scaled and shifted by a learned
    encoding of their relative position, then mapped to `num_groups` weights and softmax-normalized
    over each destination's neighbors.
    """

    def __init__(
        self,
        channels: int,
        num_groups: int,
        attn_drop: float = 0.0,
        qkv_bias: bool = True,
        pe_multiplier: bool = False,
        pe_bias: bool = True,
        norm: Union[str, Callable, None] = "batch_norm",
        act: Union[str, Callable, None] = "relu",
        act_kwargs: Optional[Dict[str, Any]] = None,
        norm_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        if channels % num_groups != 0:
            raise ValueError(f"channels ({channels}) must be divisible by num_groups ({num_groups})")

        self.channels = channels
        self.num_groups = num_groups

        self.q = MLP(
            [channels, channels],
            act=act,
            norm=norm,
            act_first=False,
            plain_last=False,
            bias=qkv_bias,
            act_kwargs=act_kwargs,
            norm_kwargs=norm_kwargs,
        )
        self.k = MLP(
            [channels, channels],
            act=act,
            norm=norm,
            act_first=False,
            plain_last=False,
            bias=qkv_bias,
            act_kwargs=act_kwargs,
            norm_kwargs=norm_kwargs,
        )
        self.v = nn.Linear(channels, channels, bias=qkv_bias)

        self.pe_multiplier: Optional[nn.Module] = None
        if pe_multiplier:
            self.pe_multiplier = nn.Sequential(
                nn.Linear(3, channels),
                create_norm(norm, channels, **(norm_kwargs or {})) or nn.Identity(),
                create_act(act, **(act_kwargs or {})) or nn.Identity(),
                nn.Linear(channels, channels),
            )

        self.pe_bias: Optional[nn.Module] = None
        if pe_bias:
            self.pe_bias = nn.Sequential(
                nn.Linear(3, channels),
                create_norm(norm, channels, **(norm_kwargs or {})) or nn.Identity(),
                create_act(act, **(act_kwargs or {})) or nn.Identity(),
                nn.Linear(channels, channels),
            )

        self.weight_encoding = nn.Sequential(
            nn.Linear(channels, num_groups),
            create_norm(norm, num_groups, **(norm_kwargs or {})) or nn.Identity(),
            create_act(act, **(act_kwargs or {})) or nn.Identity(),
            nn.Linear(num_groups, num_groups),
        )

        self.attn_drop = nn.Dropout(attn_drop)

    def forward(self, x: Tensor, pos: Tensor, edge_index: Tensor) -> Tensor:
        query, key, value = self.q(x), self.k(x), self.v(x)

        row, col = edge_index
        value = value[row]
        pos = pos[row] - pos[col]
        relation_qk = key[row] - query[col]

        if self.pe_multiplier is not None:
            factor = self.pe_multiplier(pos)
            relation_qk = relation_qk * factor

        if self.pe_bias is not None:
            bias = self.pe_bias(pos)
            relation_qk = relation_qk + bias
            value = value + bias

        weight = self.weight_encoding(relation_qk)
        weight = self.attn_drop(softmax(weight, col))

        value = value.reshape(-1, self.num_groups, self.channels // self.num_groups)
        x = value * weight.unsqueeze(-1)
        x = x.reshape(-1, self.channels)
        x = scatter_sum(x, col, dim=0)
        return x


class PointTransformerV2Block(nn.Module):
    """Residual bottleneck around a `GroupedVectorAttention`, with a linear projection before and after it."""

    def __init__(
        self,
        channels: int,
        num_groups: int,
        qkv_bias: bool = True,
        pe_multiplier: bool = False,
        pe_bias: bool = True,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        norm: Union[str, Callable, None] = "batch_norm",
        act: Union[str, Callable, None] = "relu",
        act_kwargs: Optional[Dict[str, Any]] = None,
        norm_kwargs: Optional[Dict[str, Any]] = None,
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
            act_kwargs=act_kwargs,
            norm_kwargs=norm_kwargs,
        )
        self.fc1 = nn.Linear(channels, channels, bias=False)
        self.fc3 = nn.Linear(channels, channels, bias=False)
        self.norm1 = create_norm(norm, channels, **(norm_kwargs or {})) or nn.Identity()
        self.norm2 = create_norm(norm, channels, **(norm_kwargs or {})) or nn.Identity()
        self.norm3 = create_norm(norm, channels, **(norm_kwargs or {})) or nn.Identity()
        self.act = create_act(act, **(act_kwargs or {})) or nn.Identity()
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x: Tensor, pos: Tensor, edge_index: Tensor) -> Tensor:
        shortcut = x
        x = self.act(self.norm1(self.fc1(x)))
        x = self.attn(x, pos, edge_index)
        x = self.act(self.norm2(x))
        x = self.norm3(self.fc3(x))
        x = self.drop_path(x) + shortcut
        x = self.act(x)
        return x


class PointTransformerV2GridPool(nn.Module):
    """Partition-based pooling: projects the features, then reduces every `grid_size` voxel to a single point.

    Positions are averaged within a voxel while features are reduced with `reduce`. Passing
    `return_inverse=True` to `forward` also returns the point-to-voxel map that `PointTransformerV2InversePool` needs.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        grid_size: float,
        bias: bool = False,
        reduce: str = "max",
        norm: Union[str, Callable, None] = "batch_norm",
        act: Union[str, Callable, None] = "relu",
        act_kwargs: Optional[Dict[str, Any]] = None,
        norm_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.grid_size = grid_size
        self.reduce = reduce

        self.fc = nn.Linear(in_channels, out_channels, bias=bias)
        self.norm = create_norm(norm, out_channels, **(norm_kwargs or {})) or nn.Identity()
        self.act = create_act(act, **(act_kwargs or {})) or nn.Identity()

    @overload
    def forward(
        self,
        x: Tensor,
        pos: Tensor,
        batch: Tensor,
        return_inverse: Literal[True] = True,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]: ...

    @overload
    def forward(
        self,
        x: Tensor,
        pos: Tensor,
        batch: Tensor,
        return_inverse: Literal[False] = False,
    ) -> Tuple[Tensor, Tensor, Tensor]: ...

    def forward(
        self,
        x: Tensor,
        pos: Tensor,
        batch: Tensor,
        return_inverse: bool = False,
    ) -> Tuple[Tensor, ...]:
        x = self.act(self.norm(self.fc(x)))

        # NOTE: evaluate difference with this version
        # and the consecutive_cluster version in kpconv.py
        start = segment_csr(
            pos,
            torch.cat([batch.new_zeros(1), torch.cumsum(batch.bincount(), dim=0)]),
            reduce="min",
        )
        cluster = voxel_grid(pos - start[batch], size=self.grid_size, batch=batch, start=0)
        _, cluster, counts = torch.unique(cluster, sorted=True, return_inverse=True, return_counts=True)
        _, sorted_cluster_indices = torch.sort(cluster)
        idx_ptr = torch.cat([counts.new_zeros(1), torch.cumsum(counts, dim=0)])
        pos = segment_csr(pos[sorted_cluster_indices], idx_ptr, reduce="mean")
        x = segment_csr(x[sorted_cluster_indices], idx_ptr, reduce=self.reduce)
        batch = batch[idx_ptr[:-1]]

        if return_inverse:
            return x, pos, batch, cluster
        return x, pos, batch


class PointTransformerV2InversePool(nn.Module):
    """Undoes a `PointTransformerV2GridPool`: scatters the pooled features back to the finer points and adds the projected skip."""

    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        bias: bool = True,
        norm: Union[str, Callable, None] = "batch_norm",
        act: Union[str, Callable, None] = "relu",
        act_kwargs: Optional[Dict[str, Any]] = None,
        norm_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.skip_channels = skip_channels
        self.out_channels = out_channels

        self.proj = MLP(
            [in_channels, out_channels],
            act=act,
            norm=norm,
            act_first=False,
            plain_last=False,
            bias=bias,
            act_kwargs=act_kwargs,
            norm_kwargs=norm_kwargs,
        )
        self.proj_skip = MLP(
            [skip_channels, out_channels],
            act=act,
            norm=norm,
            act_first=False,
            plain_last=False,
            bias=bias,
            act_kwargs=act_kwargs,
            norm_kwargs=norm_kwargs,
        )

    def forward(self, x: Tensor, x_skip: Tensor, inverse: Tensor) -> Tensor:
        x = self.proj(x)
        x_skip = self.proj_skip(x_skip)
        return x_skip + x[inverse]


class PointTransformerV2EncoderBlock(nn.Module):
    r"""One encoder stage: an optional `PointTransformerV2GridPool` downsampling, then `depth` `PointTransformerV2Block` units sharing a single
    $k$-NN graph built on the stage's own resolution.
    """

    def __init__(
        self,
        depth: int,
        channels: int,
        num_groups: int,
        num_neighbors: int,
        qkv_bias: bool = True,
        pe_multiplier: bool = False,
        pe_bias: bool = True,
        norm: Union[str, Callable, None] = "batch_norm",
        act: Union[str, Callable, None] = "relu",
        act_kwargs: Optional[Dict[str, Any]] = None,
        norm_kwargs: Optional[Dict[str, Any]] = None,
        attn_drop: ValueCollection[float] = 0.0,
        drop_path: ValueCollection[float] = 0.0,
        downsample: Optional[PointTransformerV2GridPool] = None,
    ):
        super().__init__()
        attn_drop = ensure_tuple_size(attn_drop, depth)
        drop_path = ensure_tuple_size(drop_path, depth)

        self.num_neighbors = num_neighbors
        self.downsample = downsample

        self.blocks = nn.ModuleList()
        for i in range(depth):
            block = PointTransformerV2Block(
                channels=channels,
                num_groups=num_groups,
                qkv_bias=qkv_bias,
                pe_multiplier=pe_multiplier,
                pe_bias=pe_bias,
                attn_drop=attn_drop[i],
                drop_path=drop_path[i],
                norm=norm,
                act=act,
                act_kwargs=act_kwargs,
                norm_kwargs=norm_kwargs,
            )
            self.blocks.append(block)

    @overload
    def forward(
        self,
        x: Tensor,
        pos: Tensor,
        batch: Tensor,
        return_inverse: Literal[True] = True,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]: ...

    @overload
    def forward(
        self,
        x: Tensor,
        pos: Tensor,
        batch: Tensor,
        return_inverse: Literal[False] = False,
    ) -> Tuple[Tensor, Tensor, Tensor]: ...

    def forward(
        self,
        x: Tensor,
        pos: Tensor,
        batch: Tensor,
        return_inverse: bool = False,
    ) -> Tuple[Tensor, ...]:
        if return_inverse and self.downsample is None:
            raise ValueError("`return_inverse` is only supported if `downsample` is provided")

        if self.downsample is not None:
            x, pos, batch, pooling_inverse = self.downsample(x, pos, batch, return_inverse=True)

        edge_index = knn_graph(pos, self.num_neighbors, batch, loop=True)
        for block in self.blocks:
            x = block(x, pos, edge_index)

        if return_inverse:
            return x, pos, batch, pooling_inverse
        return x, pos, batch


class PointTransformerV2DecoderBlock(nn.Module):
    r"""One decoder stage: an optional `PointTransformerV2InversePool` upsampling onto the skip resolution, then `depth` `PointTransformerV2Block`
    units sharing a single $k$-NN graph built on that resolution.
    """

    def __init__(
        self,
        depth: int,
        channels: int,
        num_groups: int,
        num_neighbors: int,
        qkv_bias: bool = True,
        pe_multiplier: bool = False,
        pe_bias: bool = True,
        norm: Union[str, Callable, None] = "batch_norm",
        act: Union[str, Callable, None] = "relu",
        act_kwargs: Optional[Dict[str, Any]] = None,
        norm_kwargs: Optional[Dict[str, Any]] = None,
        attn_drop: ValueCollection[float] = 0.0,
        drop_path: ValueCollection[float] = 0.0,
        upsample: Optional[PointTransformerV2InversePool] = None,
    ):
        super().__init__()
        attn_drop = ensure_tuple_size(attn_drop, depth)
        drop_path = ensure_tuple_size(drop_path, depth)

        self.num_neighbors = num_neighbors
        self.upsample = upsample

        self.blocks = nn.ModuleList()
        for i in range(depth):
            block = PointTransformerV2Block(
                channels=channels,
                num_groups=num_groups,
                qkv_bias=qkv_bias,
                pe_multiplier=pe_multiplier,
                pe_bias=pe_bias,
                attn_drop=attn_drop[i],
                drop_path=drop_path[i],
                norm=norm,
                act=act,
                act_kwargs=act_kwargs,
                norm_kwargs=norm_kwargs,
            )
            self.blocks.append(block)

    def forward(
        self,
        x: Tensor,
        x_skip: Tensor,
        pos_skip: Tensor,
        batch_skip: Tensor,
        pooling_inverse: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        if self.upsample is not None:
            x = self.upsample(x, x_skip, pooling_inverse)

        edge_index = knn_graph(pos_skip, self.num_neighbors, batch_skip, loop=True)
        for block in self.blocks:
            x = block(x, pos_skip, edge_index)
        return x, pos_skip, batch_skip


class PointTransformerV2Encoder(nn.Module):
    r"""Point Transformer V2 encoder: `PointTransformerV2EncoderBlock` stages from finest to coarsest, every stage but
    the first preceded by a `PointTransformerV2GridPool` downsampling.

    Args:
        encoder_depths: Number of blocks in each stage.
        encoder_channels: Number of channels in each stage.
        encoder_num_groups: Number of attention groups in each stage.
        encoder_num_neighbors: Number of neighbors of the graph built in each stage.
        grid_sizes: Voxel size of the pooling preceding each stage but the first.
        norm: Normalization layer to use.
        act: Activation function to use.
        act_kwargs: Keyword arguments for the activation function.
        norm_kwargs: Keyword arguments for the normalization layer.
        qkv_bias: Whether to use bias in the QKV linear layers.
        pe_multiplier: Whether to scale the query-key relation by a positional encoding.
        pe_bias: Whether to shift the query-key relation and the values by a positional encoding.
        attn_drop: Dropout rate on the attention weights.
        drop_path: Maximum drop path rate, reached by the last block.

    Inputs:
        x: Embedded point features of shape $(N, \text{encoder\_channels}[0])$.
        pos: Float tensor of shape $(N, 3)$.
        batch: Long tensor of shape $(N,)$.

    Outputs:
        Features, coordinates and batch indices at the coarsest stage. With `return_intermediates=True`, also the
        per-stage skips (features, coordinates, batch indices and pooling inverse) consumed by
        `PointTransformerV2Decoder`.
    """

    def __init__(
        self,
        encoder_depths: Sequence[int] = (1, 2, 2, 6, 2),
        encoder_channels: Sequence[int] = (48, 96, 192, 384, 512),
        encoder_num_groups: Sequence[int] = (6, 12, 24, 48, 64),
        encoder_num_neighbors: Sequence[int] = (8, 16, 16, 16, 16),
        grid_sizes: Sequence[float] = (0.06, 0.12, 0.24, 0.48),
        norm: Union[str, Callable, None] = "batch_norm",
        act: Union[str, Callable, None] = "relu",
        act_kwargs: Optional[Dict[str, Any]] = None,
        norm_kwargs: Optional[Dict[str, Any]] = None,
        qkv_bias: bool = True,
        pe_multiplier: bool = False,
        pe_bias: bool = True,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
    ):
        super().__init__()
        depths = ensure_tuple(encoder_depths)
        n = len(depths)
        channels = ensure_tuple_size(
            encoder_channels, size=n, extra_msg="Encoder length `encoder_channels` != `encoder_depths`."
        )
        num_groups = ensure_tuple_size(
            encoder_num_groups, size=n, extra_msg="Encoder length `encoder_num_groups` != `encoder_depths`."
        )
        num_neighbors = ensure_tuple_size(
            encoder_num_neighbors, size=n, extra_msg="Encoder length `encoder_num_neighbors` != `encoder_depths`."
        )
        grid_sizes = ensure_tuple_size(
            grid_sizes, size=n - 1, extra_msg="Encoder length `grid_sizes` != `encoder_depths` - 1."
        )

        # Stage 0 is the patch embedding and gets no drop path; the schedule ramps linearly across the
        # downsampled stages. For example, drop path 0.3 with depths (1, 2, 3, 4) yields:
        # - stage 0: [0.0000]
        # - stage 1: [0.0000, 0.0375]
        # - stage 2: [0.0750, 0.1125, 0.1500]
        # - stage 3: [0.1875, 0.2250, 0.2625, 0.3000]
        stage_drop_paths = torch.split(torch.linspace(0, drop_path, sum(depths[1:])), list(depths[1:]))
        drop_paths = [torch.zeros(depths[0]), *stage_drop_paths]

        self.blocks = nn.ModuleList()
        for i in range(n):
            downsample: Optional[PointTransformerV2GridPool] = None
            if i > 0:
                downsample = PointTransformerV2GridPool(
                    in_channels=channels[i - 1],
                    out_channels=channels[i],
                    grid_size=grid_sizes[i - 1],
                    reduce="max",
                    norm=norm,
                    act=act,
                    act_kwargs=act_kwargs,
                    norm_kwargs=norm_kwargs,
                )

            block = PointTransformerV2EncoderBlock(
                depth=depths[i],
                channels=channels[i],
                num_groups=num_groups[i],
                num_neighbors=num_neighbors[i],
                norm=norm,
                act=act,
                act_kwargs=act_kwargs,
                norm_kwargs=norm_kwargs,
                qkv_bias=qkv_bias,
                pe_multiplier=pe_multiplier,
                pe_bias=pe_bias,
                attn_drop=attn_drop,
                drop_path=drop_paths[i].tolist(),
                downsample=downsample,
            )
            self.blocks.append(block)

    @property
    def embedding_dim(self) -> int:
        """Feature dimension $C$ of the encoder output."""
        return self.blocks[-1].blocks[-1].fc3.out_features  # type: ignore[index, union-attr]

    @overload
    def forward(
        self,
        x: Tensor,
        pos: Tensor,
        batch: Tensor,
        return_intermediates: Literal[True],
    ) -> Tuple[Tensor, Tensor, Tensor, List[Dict[str, Tensor]]]: ...

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
        for i, block in enumerate(self.blocks):
            intermediate = {"x": x, "pos": pos, "batch": batch}

            x, pos, batch, *rest = block(x, pos, batch, return_inverse=i > 0)
            if i > 0:
                intermediate["pooling_inverse"] = rest[0]
                intermediates.append(intermediate)

        if return_intermediates:
            return x, pos, batch, intermediates
        return x, pos, batch


class PointTransformerV2Decoder(nn.Module):
    r"""Point Transformer V2 decoder: `PointTransformerV2DecoderBlock` stages from coarsest to finest, each preceded
    by a `PointTransformerV2InversePool` upsampling onto its encoder skip.

    Args:
        encoder_channels: Number of channels of each encoder stage, sizing the decoder input and the skips.
        decoder_depths: Number of blocks in each stage.
        decoder_channels: Number of output channels of each stage.
        decoder_num_groups: Number of attention groups in each stage.
        decoder_num_neighbors: Number of neighbors of the graph built in each stage.
        norm: Normalization layer to use.
        act: Activation function to use.
        act_kwargs: Keyword arguments for the activation function.
        norm_kwargs: Keyword arguments for the normalization layer.
        qkv_bias: Whether to use bias in the QKV linear layers.
        pe_multiplier: Whether to scale the query-key relation by a positional encoding.
        pe_bias: Whether to shift the query-key relation and the values by a positional encoding.
        attn_drop: Dropout rate on the attention weights.
        drop_path: Maximum drop path rate, reached by the first block.

    Inputs:
        x: Features at the coarsest encoder stage, of shape $(N', \text{encoder\_channels}[-1])$.
        intermediates: Per-stage skips returned by `PointTransformerV2Encoder`, finest first.

    Outputs:
        Features, coordinates and batch indices at the finest stage.
    """

    def __init__(
        self,
        encoder_channels: Sequence[int] = (48, 96, 192, 384, 512),
        decoder_depths: Sequence[int] = (1, 1, 1, 1),
        decoder_channels: Sequence[int] = (384, 192, 96, 48),
        decoder_num_groups: Sequence[int] = (48, 24, 12, 6),
        decoder_num_neighbors: Sequence[int] = (16, 16, 16, 16),
        norm: Union[str, Callable, None] = "batch_norm",
        act: Union[str, Callable, None] = "relu",
        act_kwargs: Optional[Dict[str, Any]] = None,
        norm_kwargs: Optional[Dict[str, Any]] = None,
        qkv_bias: bool = True,
        pe_multiplier: bool = False,
        pe_bias: bool = True,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
    ):
        super().__init__()
        depths = ensure_tuple(decoder_depths)
        n = len(depths)
        channels = ensure_tuple_size(
            [encoder_channels[-1], *decoder_channels],
            size=n + 1,
            extra_msg="Decoder length `decoder_channels` != `decoder_depths`.",
        )
        skip_channels = ensure_tuple_size(
            list(encoder_channels[:-1])[::-1],
            size=n,
            extra_msg="Decoder length `encoder_channels` - 1 != `decoder_depths`.",
        )
        num_groups = ensure_tuple_size(
            decoder_num_groups, size=n, extra_msg="Decoder length `decoder_num_groups` != `decoder_depths`."
        )
        num_neighbors = ensure_tuple_size(
            decoder_num_neighbors, size=n, extra_msg="Decoder length `decoder_num_neighbors` != `decoder_depths`."
        )

        # The schedule mirrors the encoder's, so it decreases within and across stages. For example, drop path 0.3
        # with depths (4, 3, 2) yields:
        # - stage 0: [0.3000, 0.2625, 0.2250, 0.1875]
        # - stage 1: [0.1500, 0.1125, 0.0750]
        # - stage 2: [0.0375, 0.0000]
        drop_paths = torch.split(torch.linspace(0, drop_path, sum(depths)), list(depths))[::-1]

        self.blocks = nn.ModuleList()
        for i in range(n):
            upsample = PointTransformerV2InversePool(
                in_channels=channels[i],
                skip_channels=skip_channels[i],
                out_channels=channels[i + 1],
                norm=norm,
                act=act,
                act_kwargs=act_kwargs,
                norm_kwargs=norm_kwargs,
            )

            block = PointTransformerV2DecoderBlock(
                depth=depths[i],
                channels=channels[i + 1],
                num_groups=num_groups[i],
                num_neighbors=num_neighbors[i],
                norm=norm,
                act=act,
                act_kwargs=act_kwargs,
                norm_kwargs=norm_kwargs,
                qkv_bias=qkv_bias,
                pe_multiplier=pe_multiplier,
                pe_bias=pe_bias,
                attn_drop=attn_drop,
                drop_path=drop_paths[i].tolist()[::-1],
                upsample=upsample,
            )
            self.blocks.append(block)

    @property
    def out_channels(self) -> int:
        """Feature dimension $C$ of the decoder output."""
        return self.blocks[-1].blocks[-1].fc3.out_features  # type: ignore[index, union-attr]

    def forward(self, x: Tensor, intermediates: List[Dict[str, Tensor]]) -> Tuple[Tensor, Tensor, Tensor]:
        for block, intermediate in zip(self.blocks, reversed(intermediates)):
            skip_kwargs = {f"{k}_skip" if k != "pooling_inverse" else k: v for k, v in intermediate.items()}
            x, pos, batch = block(x, **skip_kwargs)
        return x, pos, batch


class PointTransformerV2Classification(ClassificationModel):
    r"""Implementation of the Point Transformer V2 model for classification as described in the paper
    :arxiv: [Point Transformer V2: Grouped Vector Attention and Partition-based Pooling](https://arxiv.org/abs/2210.05666)
    by Xiaoyang Wu, Yixing Lao, Li Jiang, Xihui Liu, Hengshuang Zhao.

    Note:
        This implementation requires :github: [`torch-cluster`](https://github.com/rusty1s/pytorch_cluster) and
        :github: [`torch-scatter`](https://github.com/rusty1s/pytorch_scatter) to be installed.

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
        x: Float tensor of shape $(N, \text{in\_channels})$.
        pos: Float tensor of shape $(N, 3)$.
        batch: Long tensor of shape $(N,)$.

    Outputs:
        Logits tensor of shape $(B, \text{num\_classes})$.
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
        norm: Union[str, Callable, None] = "batch_norm",
        act: Union[str, Callable, None] = "relu",
        act_kwargs: Optional[Dict[str, Any]] = None,
        norm_kwargs: Optional[Dict[str, Any]] = None,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        pe_multiplier: bool = False,
        pe_bias: bool = True,
        drop_path: float = 0.0,
        dropout: float = 0.0,
        global_pool: PoolLike = "max",
    ):
        super().__init__(in_channels=in_channels, num_classes=num_classes)
        self.grid_sizes = grid_sizes
        self.encoder_depths = encoder_depths
        self.encoder_channels = encoder_channels
        self.encoder_num_groups = encoder_num_groups
        self.encoder_num_neighbors = encoder_num_neighbors
        self.norm = norm
        self.act = act
        self.act_kwargs = act_kwargs
        self.norm_kwargs = norm_kwargs
        self.qkv_bias = qkv_bias
        self.attn_drop = attn_drop
        self.pe_multiplier = pe_multiplier
        self.pe_bias = pe_bias
        self.drop_path = drop_path
        self.dropout = dropout

        self.stem = self.configure_stem()
        self.encoder = self.configure_encoder()
        self.global_pool = create_pool(global_pool)
        self.head = self.configure_head()

    def configure_stem(self) -> nn.Module:
        """Build the linear stem lifting the input features to the first encoder channel."""
        return MLP(
            [self.in_channels, self.encoder_channels[0]],
            act=self.act,
            norm=self.norm,
            act_first=False,
            plain_last=False,
            bias=True,
            act_kwargs=self.act_kwargs,
            norm_kwargs=self.norm_kwargs,
        )

    def configure_encoder(self) -> PointTransformerV2Encoder:
        """Build the `PointTransformerV2Encoder` backbone."""
        return PointTransformerV2Encoder(
            encoder_depths=self.encoder_depths,
            encoder_channels=self.encoder_channels,
            encoder_num_groups=self.encoder_num_groups,
            encoder_num_neighbors=self.encoder_num_neighbors,
            grid_sizes=self.grid_sizes,
            norm=self.norm,
            act=self.act,
            act_kwargs=self.act_kwargs,
            norm_kwargs=self.norm_kwargs,
            qkv_bias=self.qkv_bias,
            pe_multiplier=self.pe_multiplier,
            pe_bias=self.pe_bias,
            attn_drop=self.attn_drop,
            drop_path=self.drop_path,
        )

    @property
    def num_features(self) -> int:
        """Feature dimension $C$ of the encoder output."""
        return self.encoder.embedding_dim

    def configure_head(self) -> nn.Module:
        if self.num_classes == 0:
            return nn.Identity()
        return nn.Linear(self.num_features, self.num_classes)

    def reset_classifier(self, num_classes: int, global_pool: Optional[PoolLike] = None, **kwargs: Any) -> None:
        """Resets the classification head with new parameters.

        Note:
            To set an empty classification head, use `num_classes=0`.

        Args:
            num_classes: Number of output classes.
            global_pool: Pooling method to aggregate point x ("max" or "mean"). If `None`, keeps the current pooling.
            **kwargs: Additional keyword arguments to pass to the classification head.
        """
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
        r"""Forward features through the encoder blocks, before the global pooling.

        Args:
            x: Additional point features of shape $(N, \text{features\_dim})$.
            pos: Coordinates of shape $(N, 3)$.
            batch: Batch indices for each point of shape $(N,)$.
            return_intermediates: Whether to return the intermediate features.

        Returns:
            x: Pre-pooling features of shape $(N, \text{embedding\_dim})$.
            pos: Coordinates of shape $(N, 3)$.
            batch: Batch indices for each point of shape $(N,)$.
            intermediates: If `return_intermediates` is `True`, a list of dictionaries containing the intermediate features,
                coordinates, batch indices and pooling inverse for each encoder block.
        """
        x = x if x is not None else pos
        x = self.stem(x)
        return self.encoder(x, pos, batch, return_intermediates=return_intermediates)

    def forward_head(self, x: Tensor, batch: Tensor, pre_logits: bool = False) -> Tensor:
        r"""Forward pass of the classification head from pre-pooling x.

        Args:
            x: Pre-pooling features of shape $(N, \text{embedding\_dim})$.
            batch: Batch indices for each point of shape $(N,)$.
            pre_logits: Whether to return pre-logits.

        Returns:
            Classification logits of shape $(B, \text{num\_classes})$.
        """
        x = self.global_pool(x, batch)
        if self.dropout:
            x = F.dropout(x, p=float(self.dropout), training=self.training)
        return x if pre_logits else self.head(x)

    def forward(self, x: OptTensor, pos: Tensor, batch: Tensor) -> Tensor:
        r"""Forward pass of the classification network.

        Args:
            x: Additional point features of shape $(N, \text{features\_dim})$.
            pos: Coordinates of shape $(N, 3)$.
            batch: Batch indices for each point of shape $(N,)$.

        Returns:
            Classification logits of shape $(B, \text{num\_classes})$.
        """
        x, _, batch = self.forward_features(x, pos, batch)
        return self.forward_head(x, batch)


class PointTransformerV2Segmentation(SegmentationModel):
    r"""Implementation of the Point Transformer V2 model for semantic segmentation as described in the paper
    :arxiv: [Point Transformer V2: Grouped Vector Attention and Partition-based Pooling](https://arxiv.org/abs/2210.05666)
    by Xiaoyang Wu, Yixing Lao, Li Jiang, Xihui Liu, Hengshuang Zhao.

    Note:
        This implementation requires :github: [`torch-cluster`](https://github.com/rusty1s/pytorch_cluster) and
        :github: [`torch-scatter`](https://github.com/rusty1s/pytorch_scatter) to be installed.

    Args:
        in_channels: Number of input channels.
        num_classes: Number of output classes.
        encoder_depths: Number of encoder blocks for each stage.
        encoder_channels: Number of channels for each encoder block.
        encoder_num_groups: Number of groups for each encoder block.
        encoder_num_neighbors: Number of edge_index for each encoder block.
        decoder_depths: Number of decoder blocks per stage.
        decoder_channels: Number of channels for each decoder block.
        decoder_num_groups: Number of groups for each decoder block.
        decoder_num_neighbors: Neighbor count for each decoder block.
        grid_sizes: Size of the grid for each stage.
        norm: Normalization layer to use.
        act: Activation function to use.
        qkv_bias: Whether to use bias in the QKV linear layer.
        pe_multiplier: Whether to use a multiplier for the PE.
        pe_bias: Whether to use bias in the PE linear layer.
        drop_path: Drop path rate.
        dropout: Dropout rate.
        attn_drop: Attention dropout rate.

    Inputs:
        x: Float tensor of shape $(N, \text{in\_channels})$.
        pos: Float tensor of shape $(N, 3)$.
        batch: Long tensor of shape $(N,)$.

    Outputs:
        Segmentation logits of shape $(N, \text{num\_classes})$.
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
        norm: Union[str, Callable, None] = "batch_norm",
        act: Union[str, Callable, None] = "relu",
        act_kwargs: Optional[Dict[str, Any]] = None,
        norm_kwargs: Optional[Dict[str, Any]] = None,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        pe_multiplier: bool = False,
        pe_bias: bool = True,
        drop_path: float = 0.0,
        dropout: float = 0.0,
    ):
        super().__init__(in_channels=in_channels, num_classes=num_classes)
        self.grid_sizes = grid_sizes
        self.encoder_depths = encoder_depths
        self.encoder_channels = encoder_channels
        self.encoder_num_groups = encoder_num_groups
        self.encoder_num_neighbors = encoder_num_neighbors
        self.decoder_depths = decoder_depths
        self.decoder_channels = decoder_channels
        self.decoder_num_groups = decoder_num_groups
        self.decoder_num_neighbors = decoder_num_neighbors
        self.norm = norm
        self.act = act
        self.act_kwargs = act_kwargs
        self.norm_kwargs = norm_kwargs
        self.qkv_bias = qkv_bias
        self.attn_drop = attn_drop
        self.pe_multiplier = pe_multiplier
        self.pe_bias = pe_bias
        self.drop_path = drop_path
        self.dropout = dropout

        self.stem = self.configure_stem()
        self.encoder = self.configure_encoder()
        self.decoder = self.configure_decoder()
        self.head = self.configure_head()

    def configure_stem(self) -> nn.Module:
        """Build the linear stem lifting the input features to the first encoder channel."""
        return MLP(
            [self.in_channels, self.encoder_channels[0]],
            act=self.act,
            norm=self.norm,
            act_first=False,
            plain_last=False,
            bias=True,
            act_kwargs=self.act_kwargs,
            norm_kwargs=self.norm_kwargs,
        )

    def configure_encoder(self) -> PointTransformerV2Encoder:
        """Build the `PointTransformerV2Encoder` backbone."""
        return PointTransformerV2Encoder(
            encoder_depths=self.encoder_depths,
            encoder_channels=self.encoder_channels,
            encoder_num_groups=self.encoder_num_groups,
            encoder_num_neighbors=self.encoder_num_neighbors,
            grid_sizes=self.grid_sizes,
            norm=self.norm,
            act=self.act,
            act_kwargs=self.act_kwargs,
            norm_kwargs=self.norm_kwargs,
            qkv_bias=self.qkv_bias,
            pe_multiplier=self.pe_multiplier,
            pe_bias=self.pe_bias,
            attn_drop=self.attn_drop,
            drop_path=self.drop_path,
        )

    def configure_decoder(self) -> PointTransformerV2Decoder:
        """Build the `PointTransformerV2Decoder` upsampling the coarsest features back through the encoder skips."""
        return PointTransformerV2Decoder(
            encoder_channels=self.encoder_channels,
            decoder_depths=self.decoder_depths,
            decoder_channels=self.decoder_channels,
            decoder_num_groups=self.decoder_num_groups,
            decoder_num_neighbors=self.decoder_num_neighbors,
            norm=self.norm,
            act=self.act,
            act_kwargs=self.act_kwargs,
            norm_kwargs=self.norm_kwargs,
            qkv_bias=self.qkv_bias,
            pe_multiplier=self.pe_multiplier,
            pe_bias=self.pe_bias,
            attn_drop=self.attn_drop,
            drop_path=self.drop_path,
        )

    @property
    def num_features(self) -> int:
        """Channel count $C$ of the per-point decoder features entering the head."""
        return self.decoder.out_channels

    def configure_head(self) -> nn.Module:
        if self.num_classes == 0:
            return nn.Identity()
        return nn.Linear(self.num_features, self.num_classes)

    def reset_classifier(self, num_classes: int, **kwargs: Any) -> None:
        """Resets the head with new class parameters.

        Note:
            To set an empty head, use `num_classes=0`.

        Args:
            num_classes: Number of output classes.
            **kwargs: Additional keyword arguments to pass to the segmentation head.
        """
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
        r"""Forward features through the encoder blocks, before the global pooling.

        Args:
            x: Additional point features of shape $(N, \text{features\_dim})$.
            pos: Coordinates of shape $(N, 3)$.
            batch: Batch indices for each point of shape $(N,)$.
            return_intermediates: Whether to return the intermediate features.

        Returns:
            x: Pre-pooling features of shape $(N, \text{embedding\_dim})$.
            pos: Coordinates of shape $(N, 3)$.
            batch: Batch indices for each point of shape $(N,)$.
            intermediates: If `return_intermediates` is `True`, a list of dictionaries containing the intermediate features,
                coordinates, batch indices and pooling inverse for each encoder block.
        """
        x = x if x is not None else pos
        x = self.stem(x)
        return self.encoder(x, pos, batch, return_intermediates=return_intermediates)

    def forward_decoder(
        self,
        x: Tensor,
        intermediates: List[Dict[str, Tensor]],
    ) -> Tuple[Tensor, Tensor, Tensor]:
        return self.decoder(x, intermediates)

    def forward_head(self, x: Tensor, pre_logits: bool = False) -> Tensor:
        r"""Forward pass of the classification head from up-sampled x.

        Args:
            x: Pre-pooling features of shape $(N, \text{embedding\_dim})$.
            pre_logits: Whether to return pre-logits.

        Returns:
            Segmentation logits of shape $(N, \text{num\_classes})$.
        """
        if self.dropout:
            x = F.dropout(x, p=float(self.dropout), training=self.training)
        return x if pre_logits else self.head(x)

    def forward(self, x: OptTensor, pos: Tensor, batch: Tensor) -> Tensor:
        r"""Forward pass of the model.

        Args:
            x: Additional point features of shape $(N, \text{features\_dim})$.
            pos: Coordinates of shape $(N, 3)$.
            batch: Batch indices for each point of shape $(N,)$.

        Returns:
            Segmentation logits of shape $(N, \text{num\_classes})$.
        """
        x, _, _, intermediates = self.forward_features(x, pos, batch, return_intermediates=True)
        x, _, _ = self.forward_decoder(x, intermediates)
        return self.forward_head(x)


@register_model(
    "ptv2-base.scannet20",
    task="segmentation",
    weights=None,
    transform=T.Compose(
        [
            T.Shift(keys=DataKeys.POS, method="bbox", axes=[0, 1]),
            T.Shift(keys=DataKeys.POS, method="min", axes=[2]),
            T.Divide(keys=DataKeys.COLOR, divisor=255),
            T.Cat(keys=[DataKeys.POS, DataKeys.COLOR, DataKeys.NORMAL], dst_key=DataKeys.X, dim=1),
            T.Relabel(keys=DataKeys.SEGMENT, labels=range(1, 21), default=-1),
            T.CopyItems(
                keys=[DataKeys.POS, DataKeys.SEGMENT],
                names=[DataKeys.ORIGIN_POS, DataKeys.ORIGIN_SEGMENT],
                allow_missing_keys=True,
            ),
            T.Voxelize(
                pos_key=DataKeys.POS,
                pos_reduce="first",
                keys=[DataKeys.X, DataKeys.SEGMENT, DataKeys.COLOR, DataKeys.NORMAL, DataKeys.INSTANCE],
                reduce="first",
                size=0.02,
                method="fnv",
                allow_missing_keys=True,
                dst_inverse_key=DataKeys.INVERSE,
            ),
        ],
    ),
    hparams=dict(
        in_channels=9,
        num_classes=20,
        grid_sizes=(0.06, 0.15, 0.375, 0.9375),
        encoder_depths=(1, 2, 2, 6, 2),
        encoder_channels=(48, 96, 192, 384, 512),
        encoder_num_groups=(6, 12, 24, 48, 64),
        encoder_num_neighbors=(8, 16, 16, 16, 16),
        decoder_depths=(1, 1, 1, 1),
        decoder_channels=(384, 192, 96, 48),
        decoder_num_groups=(48, 24, 12, 6),
        decoder_num_neighbors=(16, 16, 16, 16),
        qkv_bias=True,
        pe_multiplier=False,
        pe_bias=True,
        attn_drop=0.0,
        drop_path=0.3,
    ),
)
def ptv2_base_scannet20(**hparams: Any) -> PointTransformerV2Segmentation:
    return PointTransformerV2Segmentation(**hparams)


@register_model(
    "ptv2-base.scannet200",
    task="segmentation",
    weights=None,
    transform=T.Compose(
        [
            T.Shift(keys=DataKeys.POS, method="bbox", axes=[0, 1]),
            T.Shift(keys=DataKeys.POS, method="min", axes=[2]),
            T.Divide(keys=DataKeys.COLOR, divisor=255),
            T.Cat(keys=[DataKeys.POS, DataKeys.COLOR, DataKeys.NORMAL], dst_key=DataKeys.X, dim=1),
            T.Relabel(keys=DataKeys.SEGMENT, labels=range(1, 201), default=-1),
            T.CopyItems(
                keys=[DataKeys.POS, DataKeys.SEGMENT],
                names=[DataKeys.ORIGIN_POS, DataKeys.ORIGIN_SEGMENT],
                allow_missing_keys=True,
            ),
            T.Voxelize(
                pos_key=DataKeys.POS,
                pos_reduce="first",
                keys=[DataKeys.X, DataKeys.SEGMENT, DataKeys.COLOR, DataKeys.NORMAL, DataKeys.INSTANCE],
                reduce="first",
                size=0.02,
                method="fnv",
                allow_missing_keys=True,
                dst_inverse_key=DataKeys.INVERSE,
            ),
        ],
    ),
    hparams=dict(
        in_channels=9,
        num_classes=200,
        grid_sizes=(0.06, 0.15, 0.375, 0.9375),
        encoder_depths=(1, 2, 2, 6, 2),
        encoder_channels=(48, 96, 192, 384, 512),
        encoder_num_groups=(6, 12, 24, 48, 64),
        encoder_num_neighbors=(8, 16, 16, 16, 16),
        decoder_depths=(1, 1, 1, 1),
        decoder_channels=(384, 192, 96, 48),
        decoder_num_groups=(48, 24, 12, 6),
        decoder_num_neighbors=(16, 16, 16, 16),
        qkv_bias=True,
        pe_multiplier=False,
        pe_bias=True,
        attn_drop=0.0,
        drop_path=0.3,
    ),
)
def ptv2_base_scannet200(**hparams: Any) -> PointTransformerV2Segmentation:
    return PointTransformerV2Segmentation(**hparams)
