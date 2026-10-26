"""PointNet++ grouping convolution, set abstraction and feature propagation blocks."""

from typing import Any, Callable, Dict, List, Literal, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.nn import MLP, MessagePassing
from torch_geometric.nn.inits import reset
from torch_geometric.typing import Adj, OptTensor, PairOptTensor, PairTensor, SparseTensor, torch_sparse
from torch_geometric.utils import add_self_loops, remove_self_loops, scatter
from typing_extensions import Unpack

from torch_pointcloud.layers.act import create_act
from torch_pointcloud.layers.pools import PoolLike, create_pool
from torch_pointcloud.ops.cluster import fps, knn_interpolate, radius
from torch_pointcloud.utils.conversion import ensure_list, ensure_tuple_size
from torch_pointcloud.utils.types import AggrType, MessagePassingParams


class PointNet2Conv(MessagePassing):
    r"""PointNet++ grouping convolution on top of PyG's `MessagePassing`.

    Each message combines the neighbor features $x_j$ with the relative position $p_j - p_i$ and applies `local_nn`;
    the messages of each centroid are then aggregated (`aggr`, max by default).

    Args:
        local_nn: Network applied to each message of shape $(E, C + D)$, or $(E, C)$ when `use_pos` is false.
        add_self_loops: Whether to add self-loops to the edge index.
        use_pos: Concatenate the relative position $p_j - p_i$ to the neighbor features.
        pos_first: Concatenate the relative position before the features (`cat([p_j - p_i, x_j])`) instead of after.
        pos_scale: Divide the relative position by this value, e.g. the ball-query radius to normalize it.
        **kwargs: Additional `MessagePassing` arguments (`aggr` defaults to `"max"`).
    """

    def __init__(
        self,
        local_nn: nn.Module,
        add_self_loops: bool = True,
        use_pos: bool = True,
        pos_first: bool = False,
        pos_scale: Optional[float] = None,
        **kwargs: Unpack[MessagePassingParams],
    ) -> None:
        kwargs.setdefault("aggr", "max")
        super().__init__(**kwargs)
        self.local_nn = local_nn
        self.add_self_loops = add_self_loops
        self.use_pos = use_pos
        self.pos_first = pos_first
        self.pos_scale = pos_scale

    def reset_parameters(self) -> None:
        super().reset_parameters()
        reset(self.local_nn)

    def forward(
        self,
        x: Union[OptTensor, PairOptTensor],
        pos: Union[Tensor, PairTensor],
        edge_index: Adj,
    ) -> Tensor:
        if not isinstance(x, tuple):
            x = (x, None)

        if isinstance(pos, Tensor):
            pos = (pos, pos)

        if self.add_self_loops:
            if isinstance(edge_index, Tensor):
                edge_index, _ = remove_self_loops(edge_index)
                edge_index, _ = add_self_loops(edge_index, num_nodes=min(pos[0].size(0), pos[1].size(0)))
            elif isinstance(edge_index, SparseTensor):
                edge_index = torch_sparse.set_diag(edge_index)

        return self.propagate(edge_index, x=x, pos=pos)

    def message(self, x_j: Optional[Tensor], pos_i: Tensor, pos_j: Tensor) -> Tensor:
        if not self.use_pos:
            if x_j is None:
                raise ValueError("`PointNet2Conv` needs features `x` when `use_pos` is false.")

            return self.local_nn(x_j)

        rel_pos = pos_j - pos_i
        if self.pos_scale is not None:
            rel_pos = rel_pos / self.pos_scale
        if x_j is None:
            msg = rel_pos
        else:
            msg = torch.cat([rel_pos, x_j], dim=1) if self.pos_first else torch.cat([x_j, rel_pos], dim=1)
        return self.local_nn(msg)

    def extra_repr(self) -> str:
        return f"local_nn={self.local_nn}"

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.extra_repr()})"


class PointNet2SetAbstraction(nn.Module):
    r"""Set-abstraction block (PointNet++ SSG / MSG) built from one `PointNet2Conv` per grouping scale.

    Farthest point sampling selects the centroids, a ball query gathers the neighbors of each centroid per scale, and
    the per-scale outputs are concatenated (Multi-Scale Grouping when `channels` is a nested sequence).

    Args:
        in_channels: Number of input feature channels.
        channels: Per-scale MLP channel sizes; a nested sequence enables Multi-Scale Grouping.
        ratio: Fractional farthest-point-sampling rate. Mutually exclusive with `num_points`.
        num_points: Absolute number of centroids to sample (e.g. VoteNet's fixed $2048, 1024, \ldots$).
            Exactly one of `ratio` / `num_points` must be given. A sample with fewer than `num_points`
            points yields repeated centroids (FPS samples with replacement to keep shapes stable).
        radii: Ball-query radius per scale.
        num_neighbors: Maximum number of neighbors per scale.
        spatial_dim: Dimension of point coordinates.
        dropout: Dropout rate inside the per-scale MLPs.
        bias: Whether the MLP linear layers use a bias.
        use_pos: Concatenate the relative position to the grouped features.
        normalize_pos: Divide the relative position by the ball-query radius.
        pos_first: Concatenate the relative position *before* the grouped features
            (`cat([rel_pos, x])`) instead of after. VoteNet and the reference PointNet++ kernels use
            this order; keeping it a flag lets weights convert as a pure rename without a column swap.
        aggr: Aggregation of the messages of each centroid.
        sort_neighbors: Keep the smallest source indices when a ball holds more than `num_neighbors` points, as the
            reference `query_ball_point` does.
    """

    def __init__(
        self,
        in_channels: int,
        channels: Sequence[Union[int, Sequence[int]]],
        *,
        ratio: Optional[float] = None,
        num_points: Optional[int] = None,
        radii: Union[float, Sequence[float]],
        num_neighbors: Union[int, Sequence[int]],
        spatial_dim: int = 3,
        dropout: float = 0.0,
        act: Union[str, Callable, None] = "relu",
        act_kwargs: Optional[Dict[str, Any]] = None,
        act_first: bool = False,
        norm: Union[str, Callable, None] = "batch_norm",
        norm_kwargs: Optional[Dict[str, Any]] = None,
        bias: bool = False,
        use_pos: bool = True,
        normalize_pos: bool = True,
        pos_first: bool = False,
        aggr: AggrType = "max",
        sort_neighbors: bool = False,
    ) -> None:
        super().__init__()
        if (ratio is None) == (num_points is None):
            raise ValueError("`PointNet2SetAbstraction` needs exactly one of `ratio` or `num_points`.")

        self.in_channels = in_channels
        self.ratio = ratio
        self.num_points = num_points
        self.use_pos = use_pos
        self.normalize_pos = normalize_pos
        self.pos_first = pos_first
        self.sort_neighbors = sort_neighbors

        # Wrap parameters in list of lists to be compatible with Multi-Scale Grouping (MSG) mode
        self.channels = ensure_list(channels, recursive=True)
        self.channels = [self.channels] if not isinstance(self.channels[0], list) else self.channels
        num_scales = len(self.channels)

        extra_msg = f"The parameter `{{param}}` must be a sequence matching the number of scales {num_scales}."
        self.radii = ensure_tuple_size(radii, size=num_scales, extra_msg=extra_msg.format(param="radii"))
        self.num_neighbors = ensure_tuple_size(
            num_neighbors,
            size=num_scales,
            extra_msg=extra_msg.format(param="num_neighbors"),
        )

        mlp_in_channels = in_channels + spatial_dim if use_pos else in_channels
        self.convs = nn.ModuleList()
        for scale_channels, scale_radius in zip(self.channels, self.radii):
            local_nn = MLP(
                [mlp_in_channels, *scale_channels],
                act=act,
                act_kwargs=act_kwargs,
                act_first=act_first,
                norm=norm,
                norm_kwargs=norm_kwargs,
                bias=bias,
                dropout=dropout,
                plain_last=False,
            )
            conv = PointNet2Conv(
                local_nn,
                add_self_loops=False,
                use_pos=use_pos,
                pos_first=pos_first,
                pos_scale=scale_radius if normalize_pos else None,
                aggr=aggr,
            )
            self.convs.append(conv)

    def forward(
        self,
        x: OptTensor,
        pos: Tensor,
        batch: Tensor,
        idx: OptTensor = None,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        # In eval mode pin the FPS start to make predictions reproducible across runs.
        if idx is None:
            idx = fps(pos, batch, ratio=self.ratio, num_nodes=self.num_points, random_start=self.training)

        pos_dst = pos[idx]
        batch_dst = batch[idx]
        out = []
        for r, k, conv in zip(self.radii, self.num_neighbors, self.convs):
            row, col = radius(pos, pos_dst, r, batch, batch_dst, max_num_neighbors=k, sort=self.sort_neighbors)
            edge_index = torch.stack([col, row], dim=0)
            out.append(conv((x, None), (pos, pos_dst), edge_index))

        return torch.cat(out, dim=1), pos_dst, batch_dst


class PointNet2QuerySetAbstraction(nn.Module):
    r"""Multi-scale set abstraction at given query points.

    Unlike [`PointNet2SetAbstraction`][torch_pointcloud.layers.pointnet2_blocks.PointNet2SetAbstraction], the
    centroids are not sampled from the source points: every query point pools, per scale, the first
    `num_neighbors` source points (in index order) inside its ball with a
    [`PointNet2Conv`][torch_pointcloud.layers.pointnet2_blocks.PointNet2Conv] whose messages are the relative
    position $p_j - p_i$ followed by the source features $x_j$, and the per-scale outputs are concatenated. A ball
    holding no source point pools a single all-zero message, so an empty neighborhood maps to the MLP response to
    zero rather than to the reduction identity.

    Args:
        in_channels: Number of source feature channels.
        channels: Per-scale MLP channel sizes, one sequence per scale.
        radii: Ball radius per scale.
        num_neighbors: Maximum number of neighbors per scale.
        spatial_dim: Dimension of point coordinates.
        act: Activation of the MLP layers.
        act_kwargs: Extra activation arguments.
        norm: Normalization of the MLP layers.
        norm_kwargs: Extra normalization arguments.
        bias: Whether the MLP linear layers use a bias.

    Shape:
        - x: $(N, C_\text{in})$
        - pos, pos_query: $(N, D)$, $(M, D)$
        - batch, batch_query: $(N,)$, $(M,)$, both sorted
        - output: $(M, \sum_s C_s)$
    """

    def __init__(
        self,
        in_channels: int,
        channels: Sequence[Sequence[int]],
        *,
        radii: Sequence[float],
        num_neighbors: Sequence[int],
        spatial_dim: int = 3,
        act: Union[str, Callable, None] = "relu",
        act_kwargs: Optional[Dict[str, Any]] = None,
        norm: Union[str, Callable, None] = "batch_norm",
        norm_kwargs: Optional[Dict[str, Any]] = None,
        bias: bool = False,
    ) -> None:
        if not len(channels) == len(radii) == len(num_neighbors):
            raise ValueError(
                f"`channels`, `radii` and `num_neighbors` need one entry per scale, got {len(channels)}, "
                f"{len(radii)} and {len(num_neighbors)}."
            )

        super().__init__()
        self.radii = tuple(radii)
        self.num_neighbors = tuple(num_neighbors)
        self.out_channels = sum(scale_channels[-1] for scale_channels in channels)
        self.convs = nn.ModuleList()
        for scale_channels in channels:
            local_nn = MLP(
                [in_channels + spatial_dim, *scale_channels],
                act=act,
                act_kwargs=act_kwargs,
                norm=norm,
                norm_kwargs=norm_kwargs,
                bias=bias,
                plain_last=False,
            )
            self.convs.append(PointNet2Conv(local_nn, add_self_loops=False, pos_first=True, aggr="max"))

    def forward(self, x: Tensor, pos: Tensor, batch: Tensor, pos_query: Tensor, batch_query: Tensor) -> Tensor:
        num_sources = pos.size(0)
        out = []
        for r, k, conv in zip(self.radii, self.num_neighbors, self.convs):
            row, col = radius(pos, pos_query, r, batch, batch_query, max_num_neighbors=k, sort=True)
            # An empty ball gets one source on the query itself with zero features: its message is all zeros.
            empty = (torch.bincount(row, minlength=pos_query.size(0)) == 0).nonzero().view(-1)
            x_source = torch.cat([x, x.new_zeros(empty.numel(), x.size(1))], dim=0)
            pos_source = torch.cat([pos, pos_query[empty]], dim=0)
            virtual = num_sources + torch.arange(empty.numel(), device=row.device)
            edge_index = torch.stack([torch.cat([col, virtual]), torch.cat([row, empty])], dim=0)
            out.append(conv((x_source, None), (pos_source, pos_query), edge_index))

        return torch.cat(out, dim=1)


class PointNet2DecomposedQuerySetAbstraction(nn.Module):
    r"""Multi-scale set abstraction at query points with the first MLP layer split into a feature and a position term.

    Reference: :arxiv: [Voxel R-CNN: Towards High Performance Voxel-based 3D Object Detection](https://arxiv.org/abs/2012.15712) (Deng et al., 2021).

    Like [`PointNet2QuerySetAbstraction`][torch_pointcloud.layers.pointnet2_blocks.PointNet2QuerySetAbstraction],
    every query point pools, per scale, the first `num_neighbors` source points (in index order) inside its ball. The
    first layer of the per-neighbor MLP is split in two: a linear layer and a norm encode every source point once, a
    second linear layer and norm encode the relative position $p_j - p_i$ of every neighbor, and the activated sum is
    max-pooled over the ball before the remaining MLP layers. A ball holding no source point pools a single message
    with zero encoded features and zero offset.

    Args:
        in_channels: Number of source feature channels.
        channels: Per-scale MLP channel sizes: the width of the split first layer, then the layers after the pooling.
        radii: Ball radius per scale.
        num_neighbors: Maximum number of neighbors per scale.
        spatial_dim: Dimension of point coordinates.
        act: Activation of the summed first layer and of the layers after the pooling.
        act_kwargs: Extra activation arguments.
        norm: Normalization of every layer.
        norm_kwargs: Extra normalization arguments.

    Shape:
        - x: $(N, C_\text{in})$
        - pos, pos_query: $(N, D)$, $(M, D)$
        - batch, batch_query: $(N,)$, $(M,)$, both sorted
        - output: $(M, \sum_s C_s)$

    Example:
        ```pycon
        >>> import torch
        >>> from torch_pointcloud.layers import PointNet2DecomposedQuerySetAbstraction
        >>> sa = PointNet2DecomposedQuerySetAbstraction(4, [[16, 32]], radii=[0.4], num_neighbors=[16]).eval()
        >>> x, pos, batch = torch.randn(100, 4), torch.rand(100, 3), torch.zeros(100, dtype=torch.long)
        >>> pos_query, batch_query = torch.rand(8, 3), torch.zeros(8, dtype=torch.long)
        >>> sa(x, pos, batch, pos_query, batch_query).shape  # doctest: +SKIP
        torch.Size([8, 32])

        ```
    """

    def __init__(
        self,
        in_channels: int,
        channels: Sequence[Sequence[int]],
        *,
        radii: Sequence[float],
        num_neighbors: Sequence[int],
        spatial_dim: int = 3,
        act: Union[str, Callable, None] = "relu",
        act_kwargs: Optional[Dict[str, Any]] = None,
        norm: Union[str, Callable, None] = "batch_norm",
        norm_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        if not len(channels) == len(radii) == len(num_neighbors):
            raise ValueError(
                f"`channels`, `radii` and `num_neighbors` need one entry per scale, got {len(channels)}, "
                f"{len(radii)} and {len(num_neighbors)}."
            )
        if any(len(scale_channels) < 2 for scale_channels in channels):
            raise ValueError(f"`channels` needs at least two sizes per scale, got {channels}.")

        super().__init__()
        self.radii = tuple(radii)
        self.num_neighbors = tuple(num_neighbors)
        self.out_channels = sum(scale_channels[-1] for scale_channels in channels)
        self.act = create_act(act, **(act_kwargs or {})) or nn.Identity()
        self.feature_mlps = nn.ModuleList()
        self.pos_mlps = nn.ModuleList()
        self.out_mlps = nn.ModuleList()
        mlp_kwargs: Dict[str, Any] = dict(norm=norm, norm_kwargs=norm_kwargs, bias=False, plain_last=False)
        for scale_channels in channels:
            self.feature_mlps.append(MLP([in_channels, scale_channels[0]], act=None, **mlp_kwargs))
            self.pos_mlps.append(MLP([spatial_dim, scale_channels[0]], act=None, **mlp_kwargs))
            self.out_mlps.append(MLP(list(scale_channels), act=act, act_kwargs=act_kwargs, **mlp_kwargs))

    def forward(self, x: Tensor, pos: Tensor, batch: Tensor, pos_query: Tensor, batch_query: Tensor) -> Tensor:
        num_sources, num_queries = pos.size(0), pos_query.size(0)
        out = []
        for r, k, feature_mlp, pos_mlp, out_mlp in zip(
            self.radii,
            self.num_neighbors,
            self.feature_mlps,
            self.pos_mlps,
            self.out_mlps,
        ):
            row, col = radius(pos, pos_query, r, batch, batch_query, max_num_neighbors=k, sort=True)
            # An empty ball gets one source on the query itself with zero features: its message is all zeros.
            empty = (torch.bincount(row, minlength=num_queries) == 0).nonzero().view(-1)
            x_source = feature_mlp(x)
            x_source = torch.cat([x_source, x_source.new_zeros(empty.numel(), x_source.size(1))], dim=0)
            pos_source = torch.cat([pos, pos_query[empty]], dim=0)
            virtual = num_sources + torch.arange(empty.numel(), device=row.device)
            source, target = torch.cat([col, virtual]), torch.cat([row, empty])
            offset = pos_source.index_select(0, source) - pos_query.index_select(0, target)
            message = self.act(x_source.index_select(0, source) + pos_mlp(offset))
            pooled = scatter(message, target, dim=0, dim_size=num_queries, reduce="max")
            out.append(out_mlp(pooled))

        return torch.cat(out, dim=1)


class PointNet2GlobalSetAbstraction(nn.Module):
    r"""Global set-abstraction block: a shared MLP followed by a pool over each batch element.

    Args:
        in_channels: Number of input feature channels.
        channels: Per-layer channel sizes of the MLP.
        spatial_dim: Dimension of point coordinates.
        dropout: Dropout rate inside the MLP.
        bias: Whether the MLP linear layers use a bias.
        use_pos: Concatenate the absolute point positions to `x` before the MLP. Unlike
            `PointNet2SetAbstraction` there is no sampled centroid to offset against, so the coordinates enter
            unnormalized (the reference PointNet++ `GroupAll`).
        pos_first: Concatenate the positions *before* the features (`cat([pos, x])`) instead of after.
        aggr: Pooling operation applied per batch element.
    """

    def __init__(
        self,
        in_channels: int,
        channels: Sequence[int],
        spatial_dim: int = 3,
        dropout: float = 0.0,
        act: Union[str, Callable, None] = "relu",
        act_kwargs: Optional[Dict[str, Any]] = None,
        act_first: bool = False,
        norm: Union[str, Callable, None] = "batch_norm",
        norm_kwargs: Optional[Dict[str, Any]] = None,
        bias: bool = False,
        use_pos: bool = False,
        pos_first: bool = False,
        aggr: PoolLike = "max",
    ) -> None:
        super().__init__()
        self.spatial_dim = spatial_dim
        self.use_pos = use_pos
        self.pos_first = pos_first
        self.mlp = MLP(
            [in_channels + spatial_dim if use_pos else in_channels, *ensure_list(channels)],
            act=act,
            act_kwargs=act_kwargs,
            act_first=act_first,
            norm=norm,
            norm_kwargs=norm_kwargs,
            bias=bias,
            dropout=dropout,
            plain_last=False,
        )
        self.pool = create_pool(aggr)

    def forward(self, x: Tensor, pos: Tensor, batch: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        if self.use_pos:
            x = torch.cat([pos, x], dim=1) if self.pos_first else torch.cat([x, pos], dim=1)

        x = self.mlp(x)
        x = self.pool(x, batch)
        pos = pos.new_zeros((x.size(0), self.spatial_dim))
        batch = torch.arange(x.size(0), device=batch.device)
        return x, pos, batch


class PointNet2FeaturePropagation(nn.Module):
    r"""K-NN interpolation + skip concatenation + MLP, as in
    :arxiv: [PointNet++: Deep Hierarchical Feature Learning on Point Sets in a Metric Space](https://arxiv.org/abs/1706.02413).

    The interpolated features are concatenated **before** the skip features
    (`cat([interp, skip])`).

    Args:
        channels: Per-layer channel sizes of the MLP, starting with its input width (interpolated plus skip channels).
        k: Number of neighbors for the K-NN interpolation. PointNet++ uses $k = 3$
            with inverse-distance weighting; RandLA-Net uses $k = 1$ (nearest only).
        dropout: Dropout rate inside the MLP.
        bias: Whether the MLP linear layers use a bias.
        plain_last: Leave the last MLP layer without normalization and activation.
        weighting: Inverse-distance weighting scheme passed to `knn_interpolate`.
            Irrelevant when $k = 1$.
        eps: Numerical stability term added to the interpolation distances.
    """

    def __init__(
        self,
        channels: Sequence[int],
        k: int = 3,
        dropout: float = 0.0,
        act: Union[str, Callable, None] = "relu",
        act_first: bool = False,
        act_kwargs: Optional[Dict[str, Any]] = None,
        norm: Union[str, Callable, None] = "batch_norm",
        norm_kwargs: Optional[Dict[str, Any]] = None,
        bias: Union[bool, List[bool]] = True,
        plain_last: bool = True,
        weighting: Literal["squared", "inverse"] = "inverse",
        eps: float = 1e-16,
    ) -> None:
        super().__init__()
        self.k = k
        self.weighting = weighting
        self.eps = eps
        self.mlp = MLP(
            channels,
            act=act,
            act_first=act_first,
            act_kwargs=act_kwargs,
            norm=norm,
            norm_kwargs=norm_kwargs,
            bias=bias,
            dropout=dropout,
            plain_last=plain_last,
        )

    def forward(
        self,
        x: Tensor,
        pos: Tensor,
        batch: Tensor,
        x_skip: OptTensor,
        pos_skip: Tensor,
        batch_skip: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        x = knn_interpolate(
            x,
            pos,
            pos_skip,
            batch_x=batch,
            batch_y=batch_skip,
            k=self.k,
            weighting=self.weighting,
            eps=self.eps,
        )
        if x_skip is not None:
            x = torch.cat([x, x_skip], dim=1)

        x = self.mlp(x)
        return x, pos_skip, batch_skip
