from typing import Any, Callable, Dict, List, Literal, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.nn import MLP, MessagePassing
from torch_geometric.nn.inits import reset
from torch_geometric.typing import Adj, OptTensor, PairOptTensor, PairTensor, SparseTensor, torch_sparse
from torch_geometric.utils import add_self_loops, remove_self_loops
from typing_extensions import Unpack

from torch_pointcloud.layers.pools import PoolLike, create_pool
from torch_pointcloud.utils.cluster import fps, radius
from torch_pointcloud.utils.conversion import ensure_list, ensure_tuple, ensure_tuple_size, is_iterable
from torch_pointcloud.utils.ops import knn_interpolate
from torch_pointcloud.utils.types import AggrType, MessagePassingParams

from .act import create_act


class SAModule(nn.Module):
    r"""Single-resolution set-abstraction block (PointNet++ SSG/MSG).

    Note:
        `SAModule` / `GlobalSAModule` / `FPModule` form the canonical PointNet++ stack used by the
        registered models. The `PointNet2*` classes in this module are an alternative implementation
        of the same blocks on top of PyG's `MessagePassing`.

    Args:
        ratio: Fractional farthest-point-sampling rate. Mutually exclusive with `num_points`.
        num_points: Absolute number of centroids to sample (e.g. VoteNet's fixed $2048, 1024, \ldots$).
            Exactly one of `ratio` / `num_points` must be given. A sample with fewer than `num_points`
            points yields repeated centroids (FPS samples with replacement to keep shapes stable).
        pos_first: Concatenate the relative position *before* the grouped features
            (`cat([rel_pos, x])`) instead of after. VoteNet and the reference PointNet++ kernels use
            this order; keeping it a flag lets weights convert as a pure rename without a column swap.
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
        if (ratio is None) == (num_points is None):
            raise ValueError("`SAModule` needs exactly one of `ratio` or `num_points`.")

        self.in_channels = in_channels
        self.channels = ensure_list(channels, recursive=True)
        self.ratio = ratio
        self.num_points = num_points
        self.use_pos = use_pos
        self.normalize_pos = normalize_pos
        self.pos_first = pos_first
        self.sort_neighbors = sort_neighbors

        # Wrap parameters in list of lists to be compatible with Multi-Scale Grouping (MSG) mode
        self.channels = [self.channels] if not isinstance(self.channels[0], list) else self.channels
        sizes = [len(channels) for channels in self.channels]

        extra_msg = f"The parameter `{{param}}` must be a sequence matching the number of scales {len(sizes)}."
        self.radii = ensure_tuple_size(radii, size=len(sizes), extra_msg=extra_msg.format(param="radii"))
        self.num_neighbors = ensure_tuple_size(
            num_neighbors,
            size=len(sizes),
            extra_msg=extra_msg.format(param="num_neighbors"),
        )

        in_channels = in_channels + spatial_dim if use_pos else in_channels
        self.mlps = nn.ModuleList()
        for i in range(len(self.channels)):
            mlp = MLP(
                [in_channels, *self.channels[i]],
                act=act,
                act_kwargs=act_kwargs,
                act_first=act_first,
                norm=norm,
                norm_kwargs=norm_kwargs,
                bias=bias,
                plain_last=False,
            )
            self.mlps.append(mlp)

        self.pool = create_pool(pool)

    def forward(
        self,
        x: Tensor,
        pos: Tensor,
        batch: Tensor,
        idx: OptTensor = None,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        # In eval mode pin the FPS start to make predictions reproducible across runs.
        if idx is None:
            idx = fps(pos, batch, ratio=self.ratio, num_nodes=self.num_points, random_start=self.training)

        new_pos = pos[idx]
        new_batch = batch[idx]
        msg_out = []

        for r, k, mlp in zip(self.radii, self.num_neighbors, self.mlps):
            row, col = radius(pos, new_pos, r, batch, new_batch, max_num_neighbors=k, sort=self.sort_neighbors)
            rel_pos = pos[col] - new_pos[row]
            if self.normalize_pos:
                rel_pos = rel_pos / r

            x_j = x[col]
            if self.use_pos:
                x_j = torch.cat([rel_pos, x_j], dim=1) if self.pos_first else torch.cat([x_j, rel_pos], dim=1)

            x_j = mlp(x_j)
            x_j = self.pool(x_j, row)
            msg_out.append(x_j)

        return torch.cat(msg_out, dim=1), new_pos, new_batch


class GlobalSAModule(nn.Module):
    r"""Global set-abstraction block: a shared MLP followed by a pool over each batch element.

    Args:
        use_pos: Concatenate the absolute point positions to `x` before the MLP. Unlike
            [`SAModule`][torch_pointcloud.layers.pointnet2_blocks.SAModule] there is no sampled centroid
            to offset against, so the coordinates enter unnormalized (the reference PointNet++
            `GroupAll`).
        pos_first: Concatenate the positions *before* the features (`cat([pos, x])`) instead of after.
    """

    def __init__(
        self,
        in_channels: int,
        channels: Sequence[int],
        spatial_dim: int = 3,
        act: Union[str, Callable, None] = "relu",
        act_kwargs: Optional[Dict[str, Any]] = None,
        act_first: bool = False,
        norm: Union[str, Callable, None] = "batch_norm",
        norm_kwargs: Optional[Dict[str, Any]] = None,
        bias: bool = False,
        use_pos: bool = False,
        pos_first: bool = False,
        pool: PoolLike = "max",
    ) -> None:
        super().__init__()
        self.spatial_dim = spatial_dim
        self.use_pos = use_pos
        self.pos_first = pos_first
        channels = list(ensure_tuple(channels))
        self.mlp = MLP(
            [in_channels + spatial_dim if use_pos else in_channels, *channels],
            act=act,
            act_kwargs=act_kwargs,
            act_first=act_first,
            norm=norm,
            norm_kwargs=norm_kwargs,
            bias=bias,
            plain_last=False,
        )
        self.pool = create_pool(pool)

    def forward(self, x: Tensor, pos: Tensor, batch: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        if self.use_pos:
            x = torch.cat([pos, x], dim=1) if self.pos_first else torch.cat([x, pos], dim=1)

        x = self.mlp(x)
        x = self.pool(x, batch)
        pos = pos.new_zeros((x.size(0), self.spatial_dim))
        batch = torch.arange(x.size(0), device=batch.device)
        return x, pos, batch


class FPModule(torch.nn.Module):
    r"""Feature-propagation block (PointNet++): $k$-NN interpolation, skip concatenation, and an MLP.

    Args:
        in_channels: Number of input channels after the skip concatenation.
        channels: Per-layer channel sizes of the MLP.
        k: Number of neighbors for the $k$-NN interpolation. PointNet++ uses $k = 3$;
            $k = 1$ copies the nearest coarse feature (RandLA-Net).
        weighting: Inverse-distance weighting scheme passed to `knn_interpolate`. Irrelevant when $k = 1$.
        eps: Numerical stability term added to the interpolation distances.
    """

    def __init__(
        self,
        in_channels: int,
        channels: Sequence[int],
        k: int = 3,
        act: Union[str, Callable, None] = "relu",
        act_kwargs: Optional[Dict[str, Any]] = None,
        act_first: bool = False,
        norm: Union[str, Callable, None] = "batch_norm",
        norm_kwargs: Optional[Dict[str, Any]] = None,
        bias: bool = False,
        weighting: Literal["squared", "inverse"] = "squared",
        eps: float = 1e-16,
    ) -> None:
        super().__init__()
        self.k = k
        self.weighting = weighting
        self.eps = eps
        self.mlp = MLP(
            [in_channels, *channels],
            act=act,
            act_kwargs=act_kwargs,
            act_first=act_first,
            norm=norm,
            norm_kwargs=norm_kwargs,
            bias=bias,
            plain_last=False,
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
        x = knn_interpolate(x, pos, pos_skip, batch, batch_skip, k=self.k, weighting=self.weighting, eps=self.eps)
        if x_skip is not None:
            x = torch.cat([x, x_skip], dim=1)

        x = self.mlp(x)
        return x, pos_skip, batch_skip


def ensure_msg_list(items: Sequence[Any], extra_msg: str = "") -> List[List[List[Any]]]:
    """Utility function to ensure that items are converted in nested lists compatible
    with Multi-Scale Grouping (MSG) mode.
    This function will convert a list of list into a list of list of list.

    Example:
        Let's say we have designed a network where the first two SA blocks are
        not using MSG mode, but the last SA block is using MSG mode.

        Calling `ensure_msg_list` will make sure the provided channels are compliant
        with the MSG mode.

        >>> sa_channels = [[32, 64], [128, 256], [[256, 512, 512], [256, 512, 1024]]]
        >>> ensure_msg_list(sa_channels)
        [[[32, 64]], [[128, 256]], [[256, 512, 512], [256, 512, 1024]]]
    """
    items = ensure_list(items, recursive=True)

    result = []
    if not is_iterable(items):
        raise ValueError(f"Expected a sequence, got {type(items).__name__}. {extra_msg}")

    for i, item in enumerate(items):
        if not is_iterable(item):
            raise ValueError(f"Expected a sequence, got {type(item).__name__} at index {i} from {items}. {extra_msg}")

        # Check if the item is already a list of lists
        if all(is_iterable(subitem) for subitem in item):
            result.append(item)
        elif all(not is_iterable(subitem) for subitem in item):
            result.append([item])
        else:
            raise ValueError(
                "Expected either all items to be iterable or non-iterable, "
                f"got a mix of both at index {i} from {items}. {extra_msg}"
            )

    return result  # type: ignore[return-value]


def ensure_msg_list_size(value: Sequence[Any], size: int, extra_msg: str = "") -> Sequence[Any]:
    """Validate the length of a sequence, then nest it for Multi-Scale Grouping (MSG) compatibility.

    Args:
        value: Sequence of per-block channel specifications.
        size: Expected number of elements in `value`.
        extra_msg: Extra context appended to the error message.

    Returns:
        The value converted to a list of lists of lists (one inner list per grouping scale).

    Raises:
        ValueError: If `value` does not have exactly `size` elements.

    Example:
        >>> ensure_msg_list_size([[32, 64], [64, 128]], size=2)
        [[[32, 64]], [[64, 128]]]
    """
    if len(value) != size:
        raise ValueError(f"Expected a list of size {size}, got {len(value)}. {extra_msg}")
    return ensure_msg_list(value, extra_msg=extra_msg)


class PointNet2Conv(MessagePassing):
    r"""PointNet++ grouping convolution on top of PyG's `MessagePassing`.

    Each message concatenates the neighbor features with the relative position
    (`cat([x_j, pos_j - pos_i])`) and applies `local_nn`; messages are aggregated per centroid.

    Args:
        local_nn: Network applied to each message of shape $(E, C + D)$.
        add_self_loops: Whether to add self-loops to the edge index.
        **kwargs: Additional `MessagePassing` arguments (`aggr` defaults to `"max"`).
    """

    def __init__(self, local_nn: nn.Module, add_self_loops: bool = True, **kwargs: Unpack[MessagePassingParams]):
        kwargs.setdefault("aggr", "max")
        super().__init__(**kwargs)
        self.local_nn = local_nn
        self.add_self_loops = add_self_loops
        self.reset_parameters()

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
        msg = pos_j - pos_i
        if x_j is not None:
            msg = torch.cat([x_j, msg], dim=1)

        return self.local_nn(msg)

    def extra_repr(self) -> str:
        return f"local_nn={self.local_nn}"

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.extra_repr()})"


class PointNet2SetAbstraction(nn.Module):
    r"""Set-abstraction block built from one `PointNet2Conv` per grouping scale.

    Farthest point sampling selects the centroids, a ball query gathers the neighbors of each
    centroid per scale, and the per-scale outputs are concatenated (Multi-Scale Grouping when
    `channels` is a nested sequence).

    Args:
        spatial_dim: Dimension of point coordinates.
        in_channels: Number of input feature channels.
        channels: Per-scale MLP channel sizes; a nested sequence enables Multi-Scale Grouping.
        ratio: Fractional farthest-point-sampling rate.
        radius: Ball-query radius per scale.
        num_neighbors: Maximum number of neighbors per scale.
        dropout: Dropout rate inside the per-scale MLPs.
        add_self_loops: Whether to add self-loops to the grouping edge index.
        aggr: Message aggregation used by the convolutions.
    """

    def __init__(
        self,
        spatial_dim: int,
        in_channels: int,
        channels: Sequence[Union[int, Sequence[int]]],
        ratio: float,
        radius: Union[float, Sequence[float]],
        num_neighbors: Union[int, Sequence[int]],
        dropout: float = 0.0,
        act: Union[str, Callable, None] = "relu",
        act_first: bool = False,
        act_kwargs: Optional[Dict[str, Any]] = None,
        norm: Union[str, Callable, None] = "batch_norm",
        norm_kwargs: Optional[Dict[str, Any]] = None,
        bias: Union[bool, List[bool]] = True,
        add_self_loops: bool = False,
        aggr: AggrType = "max",
    ) -> None:
        super().__init__()
        act_kwargs = act_kwargs or {}
        norm_kwargs = norm_kwargs or {}

        self.spatial_dim = spatial_dim
        self.in_channels = in_channels
        self.dropout = dropout
        self.act = create_act(act, **act_kwargs) or nn.Identity()
        self.act_kwargs = act_kwargs
        self.act_first = act_first
        self.norm = norm
        self.norm_kwargs = norm_kwargs
        self.bias = bias
        self.ratio = ratio
        self.add_self_loops = add_self_loops
        self.aggr = aggr

        channels = ensure_list(channels, recursive=True)
        if len(channels) == 0:
            raise ValueError("The parameter `channels` must be a non-empty sequence.")

        self.channels = [channels] if not isinstance(channels[0], list) else channels

        if not all(isinstance(c, list) for c in self.channels):
            raise ValueError(
                f"All elements in channels must be lists of int to support "
                f"Multi-Scale Grouping (MSG) mode, got: {self.channels}"
            )

        sizes = [len(channels) for channels in self.channels]
        extra_msg = (
            "The parameter `{param}` must be a sequence matching the number of scales "
            f"({self.channels} channels have {len(sizes)} scales)."
        )
        self.radius = ensure_tuple_size(
            radius,
            size=len(sizes),
            extra_msg=extra_msg.format(param="radius"),
        )
        self.num_neighbors = ensure_tuple_size(
            num_neighbors,
            size=len(sizes),
            extra_msg=extra_msg.format(param="num_neighbors"),
        )

        self.convs = nn.ModuleList()
        for scale_channels in self.channels:
            local_nn = MLP(
                [self.in_channels + self.spatial_dim] + ensure_list(scale_channels),
                act=self.act,
                act_first=self.act_first,
                act_kwargs=self.act_kwargs,
                norm=self.norm,
                norm_kwargs=self.norm_kwargs,
                bias=self.bias,
                dropout=self.dropout,
                plain_last=False,
            )
            self.convs.append(PointNet2Conv(local_nn=local_nn, add_self_loops=self.add_self_loops, aggr=self.aggr))

    def forward(self, x: Tensor, pos: Tensor, batch: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        # In eval mode pin the FPS start to make predictions reproducible across runs.
        idx = fps(pos, batch, ratio=self.ratio, random_start=self.training)
        x_dst = None if x is None else x[idx]
        pos_dst = pos[idx]
        batch_dst = batch[idx]

        msg_x = []
        for r, num_neighbors, conv in zip(self.radius, self.num_neighbors, self.convs):
            row, col = radius(pos, pos_dst, r=r, batch_x=batch, batch_y=batch_dst, max_num_neighbors=num_neighbors)
            edge_index = torch.stack([col, row], dim=0)
            out_x = conv((x, x_dst), (pos, pos_dst), edge_index)
            msg_x.append(out_x)

        return torch.cat(msg_x, dim=1), pos_dst, batch_dst


class PointNet2GlobalSetAbstraction(nn.Module):
    r"""Global set-abstraction block: a shared MLP followed by a pool over each batch element.

    Args:
        in_channels: Number of input feature channels.
        channels: Per-layer channel sizes of the MLP.
        dropout: Dropout rate inside the MLP.
        aggr: Pooling operation applied per batch element.
    """

    def __init__(
        self,
        in_channels: int,
        channels: Sequence[int],
        dropout: float = 0.0,
        act: Union[str, Callable, None] = "relu",
        act_first: bool = False,
        act_kwargs: Optional[Dict[str, Any]] = None,
        norm: Union[str, Callable, None] = "batch_norm",
        norm_kwargs: Optional[Dict[str, Any]] = None,
        bias: Union[bool, List[bool]] = True,
        aggr: PoolLike = "max",
    ) -> None:
        super().__init__()
        channels = [in_channels] + ensure_list(channels)
        self.mlp = MLP(
            channels,
            act=act,
            act_first=act_first,
            act_kwargs=act_kwargs,
            norm=norm,
            norm_kwargs=norm_kwargs,
            bias=bias,
            dropout=dropout,
        )
        self.pool = create_pool(aggr)

    def forward(self, x: Tensor, pos: Tensor, batch: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        x = self.mlp(x)
        x = self.pool(x, batch)
        pos = pos.new_zeros((x.size(0), 3))
        batch = torch.arange(x.size(0), device=batch.device)
        return x, pos, batch


class PointNet2FeaturePropagation(nn.Module):
    r"""K-NN interpolation + skip concatenation + MLP, as in
    :arxiv: [PointNet++: Deep Hierarchical Feature Learning on Point Sets in a Metric Space](https://arxiv.org/abs/1706.02413).

    The interpolated features are concatenated **before** the skip features
    (`cat([interp, skip])`). Models with the opposite upstream cat order
    (RandLA-Net's `cat([skip, interp])`) must swap the first linear layer's
    column blocks at conversion time to stay weight-compatible.

    Args:
        channels: Per-layer channel sizes for the post-concat MLP.
        k: Number of neighbors for the K-NN interpolation. PointNet++ uses $k = 3$
            with inverse-distance weighting; RandLA-Net uses $k = 1$ (nearest only).
        weighting: Inverse-distance weighting scheme passed to `knn_interpolate`.
            Irrelevant when $k = 1$.
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
    ) -> None:
        super().__init__()
        self.k = k
        self.weighting = weighting
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
        x = knn_interpolate(x, pos, pos_skip, batch_x=batch, batch_y=batch_skip, k=self.k, weighting=self.weighting)
        if x_skip is not None:
            x = torch.cat([x, x_skip], dim=1)

        x = self.mlp(x)
        return x, pos_skip, batch_skip
