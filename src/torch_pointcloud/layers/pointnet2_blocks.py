"""PointNet++ grouping convolution, set abstraction and feature propagation blocks."""

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
from torch_pointcloud.utils.conversion import ensure_list, ensure_tuple_size, is_iterable
from torch_pointcloud.utils.ops import knn_interpolate
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
    (`cat([interp, skip])`). Models with the opposite upstream cat order
    (RandLA-Net's `cat([skip, interp])`) must swap the first linear layer's
    column blocks at conversion time to stay weight-compatible.

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


def ensure_msg_list(items: Sequence[Any], extra_msg: str = "") -> List[List[List[Any]]]:
    """Utility function to ensure that items are converted in nested lists compatible
    with Multi-Scale Grouping (MSG) mode.
    This function will convert a list of list into a list of list of list.

    Example:
        Let's say we have designed a network where the first two SA blocks are
        not using MSG mode, but the last SA block is using MSG mode.

        Calling `ensure_msg_list` will make sure the provided channels are compliant
        with the MSG mode.

        ```pycon
        >>> sa_channels = [[32, 64], [128, 256], [[256, 512, 512], [256, 512, 1024]]]
        >>> ensure_msg_list(sa_channels)
        [[[32, 64]], [[128, 256]], [[256, 512, 512], [256, 512, 1024]]]

        ```
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
        ```pycon
        >>> ensure_msg_list_size([[32, 64], [64, 128]], size=2)
        [[[32, 64]], [[64, 128]]]

        ```
    """
    if len(value) != size:
        raise ValueError(f"Expected a list of size {size}, got {len(value)}. {extra_msg}")
    return ensure_msg_list(value, extra_msg=extra_msg)
