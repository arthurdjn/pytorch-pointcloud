"""CurveNet blocks: local point-feature aggregation, curve grouping and aggregation, curve intervention convolution."""

from typing import Any, Callable, Dict, List, Literal, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.nn import MLP
from torch_geometric.nn.pool.select.topk import topk

from torch_pointcloud.ops.cluster import knn_interpolate, knn_table
from torch_pointcloud.ops.math import hard_softmax
from torch_pointcloud.utils.conversion import ensure_option
from torch_pointcloud.utils.types import OptTensor

from .act import create_act
from .attention_gate import AttentionGate
from .radius_max_pool import RadiusMaxPool


class LocalPointFeatureAggregation(nn.Module):
    r"""Local point-feature aggregation (LPFA) from
    :arxiv: [Walk in the Cloud: Learning Curves for Point Clouds Shape Analysis](https://arxiv.org/abs/2105.01288).

    Every point gathers the relative geometry of its neighbors, $(\mathbf{p}_i, \mathbf{p}_j, \mathbf{p}_j -
    \mathbf{p}_i)$, lifts it to the feature width and adds it to the feature differences $\mathbf{x}_j - \mathbf{x}_i$
    before an MLP and a pooling over the neighbors. With `in_channels=0` the block embeds the geometry alone, which is
    the input embedding of CurveNet.

    Args:
        in_channels: Number of input feature channels, `0` for a geometry-only embedding.
        out_channels: Number of output channels.
        num_layers: Depth of the MLP applied to every neighbor.
        aggr: Pooling over the neighbors, `"mean"` or `"max"`.
        spatial_dim: Dimension $d$ of the positions.
        act: Activation function.
        act_kwargs: Extra arguments for the activation.
        norm: Normalization layer.
        norm_kwargs: Extra arguments for the normalization.

    Shape:
        - Input: features $(N, C_\text{in})$ (or `None` when `in_channels=0`), positions $(N, d)$ and a
            `knn_table` of neighbor indices $(N, k)$.
        - Output: $(N, C_\text{out})$.

    Example:
        ```python
        import torch
        from torch_pointcloud.layers import LocalPointFeatureAggregation
        from torch_pointcloud.ops.cluster import knn_table

        block = LocalPointFeatureAggregation(16, 32)
        pos = torch.randn(128, 3)
        x = torch.randn(128, 16)
        table = knn_table(pos, pos, k=8)
        y = block(x, pos, table)
        print(y.shape)
        ```
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_layers: int = 2,
        aggr: Literal["mean", "max"] = "mean",
        spatial_dim: int = 3,
        act: Union[str, Callable, None] = "relu",
        act_kwargs: Optional[Dict[str, Any]] = None,
        norm: Union[str, Callable, None] = "batch_norm",
        norm_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.aggr = ensure_option(aggr, ("mean", "max"), name="aggr")
        geometry_channels = 3 * spatial_dim
        if in_channels > 0:
            self.geometry = MLP(
                [geometry_channels, in_channels],
                plain_last=False,
                act=None,
                norm=norm,
                norm_kwargs=norm_kwargs,
                bias=False,
            )
            self.act = create_act(act, **(act_kwargs or {})) or nn.Identity()
        self.mlp = MLP(
            [in_channels or geometry_channels] + [out_channels] * num_layers,
            plain_last=False,
            act=act,
            act_kwargs=act_kwargs,
            norm=norm,
            norm_kwargs=norm_kwargs,
            bias=False,
        )

    def forward(self, x: OptTensor, pos: Tensor, table: Tensor) -> Tensor:
        num_points, k = table.shape
        # `index_select` rather than advanced indexing, whose backward walks the duplicate indices serially.
        index = table.reshape(-1)
        pos_i = pos.unsqueeze(1).expand(-1, k, -1).flatten(0, 1)
        pos_j = pos.index_select(0, index)
        geometry = torch.cat([pos_i, pos_j, pos_j - pos_i], dim=-1)  # (N k, 3 d)
        if self.in_channels == 0:
            edge = geometry
        else:
            assert x is not None
            x_i = x.unsqueeze(1).expand(-1, k, -1).flatten(0, 1)
            edge = self.act(x.index_select(0, index) - x_i + self.geometry(geometry))
        edge = self.mlp(edge).view(num_points, k, -1)
        return edge.amax(dim=1) if self.aggr == "max" else edge.mean(dim=1)


class CurveWalk(nn.Module):
    r"""Walk from starting points along a neighbor table, from
    :arxiv: [Walk in the Cloud: Learning Curves for Point Clouds Shape Analysis](https://arxiv.org/abs/2105.01288).

    At every step an agent scores the neighbors of the current point against the curve descriptor, a learned momentum
    mix of the current point and the previous descriptor, and moves to the best one through a straight-through hard
    softmax. A crossover suppression damps the neighbors that fold the curve back on itself. The walk returns the
    features of the visited points.

    The momentum weights of a sample are read from its channels-first $(2, n)$ score block as $n$ consecutive pairs,
    so curve $i$ mixes with the scores $2i$ and $2i + 1$ of that block rather than with its own two scores.

    Args:
        channels: Number of feature channels $C$.
        curve_length: Number of steps $L$.
        norm: Normalization of the agent and momentum layers.
        norm_kwargs: Extra arguments for the normalization.

    Shape:
        - Input: features $(N, C)$, a neighbor table $(N, k)$ without the point itself, and the starting point
            indices $(B, n)$.
        - Output: $(B, n, L, C)$.

    Example:
        ```python
        import torch
        from torch_pointcloud.layers import CurveWalk
        from torch_pointcloud.ops.cluster import knn_table

        walk = CurveWalk(16, curve_length=5)
        pos = torch.randn(128, 3)
        x = torch.randn(128, 16)
        table = knn_table(pos, pos, k=9)[:, 1:]
        curves = walk(x, table, torch.arange(10).view(1, 10))
        print(curves.shape)
        ```
    """

    def __init__(
        self,
        channels: int,
        curve_length: int,
        norm: Union[str, Callable, None] = "batch_norm",
        norm_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()
        self.curve_length = curve_length
        self.agent = MLP([2 * channels, 1], plain_last=False, act=None, norm=norm, norm_kwargs=norm_kwargs, bias=False)
        self.momentum = MLP(
            [2 * channels, 2],
            plain_last=False,
            act=None,
            norm=norm,
            norm_kwargs=norm_kwargs,
            bias=False,
        )

    def forward(self, x: Tensor, table: Tensor, start: Tensor) -> Tensor:
        batch_size, num_curves = start.shape
        k = table.size(1)
        current = start.reshape(-1)  # (M,) with M = B n
        descriptor = x.index_select(0, current)  # (M, C)
        feature = descriptor
        curves: List[Tensor] = []
        for step in range(self.curve_length):
            if step > 0:
                weights = self.momentum(torch.cat([feature, descriptor], dim=1)).softmax(dim=1)  # (M, 2)
                # The released checkpoints were trained with this pairing of the momentum scores.
                weights = weights.view(batch_size, num_curves, 2).transpose(1, 2).reshape(-1, 2)
                descriptor = feature * weights[:, :1] + descriptor * weights[:, 1:]
            neighbors = table[current]  # (M, k)
            candidates = x.index_select(0, neighbors.reshape(-1)).view(-1, k, x.size(1))  # (M, k, C)
            edge = torch.cat([candidates, descriptor.unsqueeze(1).expand(-1, k, -1)], dim=-1)
            logits = self.agent(edge.flatten(0, 1)).view(-1, k)
            if step > 0:
                logits = logits * crossover_suppression(feature - descriptor, candidates - feature.unsqueeze(1))
            choice = hard_softmax(logits)
            feature = (candidates * choice.unsqueeze(-1)).sum(dim=1)
            current = neighbors.gather(1, choice.argmax(dim=1, keepdim=True)).squeeze(1)
            curves.append(feature)
        return torch.stack(curves, dim=1).view(batch_size, num_curves, self.curve_length, -1)


@torch.no_grad()
def crossover_suppression(direction: Tensor, candidates: Tensor) -> Tensor:
    r"""Damping factor of every candidate step, $\mathrm{clamp}(1 + \cos(\text{direction}, \text{candidate}), 0, 1)$.

    Args:
        direction: Direction of the walk so far, shape $(M, C)$.
        candidates: Candidate steps, shape $(M, k, C)$.

    Returns:
        The factor in $[0, 1]$ of every candidate, shape $(M, k)$.

    Shape:
        Input: $(M, C)$ and $(M, k, C)$
        Output: $(M, k)$

    Example:
        ```pycon
        >>> import torch
        >>> from torch_pointcloud.layers.curvenet_blocks import crossover_suppression
        >>> direction = torch.tensor([[1.0, 0.0]])
        >>> candidates = torch.tensor([[[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0]]])
        >>> crossover_suppression(direction, candidates).tolist()
        [[1.0, 0.0, 1.0]]

        ```
    """
    dot = (candidates @ direction.unsqueeze(-1)).squeeze(-1)  # (M, k)
    norms = direction.norm(dim=1, keepdim=True) * candidates.norm(dim=2)
    cosine = dot / norms.clamp(min=1e-8)
    return (1.0 + cosine).clamp(0.0, 1.0)


class CurveGrouping(nn.Module):
    r"""Curve grouping from
    :arxiv: [Walk in the Cloud: Learning Curves for Point Clouds Shape Analysis](https://arxiv.org/abs/2105.01288).

    Scores every point with a sigmoid gate, starts `num_curves` curves per sample at the highest scores, taken in
    point order, and walks them with `CurveWalk` over the gated features.

    Args:
        channels: Number of feature channels $C$.
        num_curves: Number of curves $n$ per sample.
        curve_length: Number of points $L$ on every curve.
        norm: Normalization of the walk layers.
        norm_kwargs: Extra arguments for the normalization.

    Shape:
        - Input: features $(N, C)$, batch index $(N,)$ and a neighbor table $(N, k)$ without the point itself.
        - Output: curves $(B, n, L, C)$.

    Example:
        ```python
        import torch
        from torch_pointcloud.layers import CurveGrouping
        from torch_pointcloud.ops.cluster import knn_table

        grouping = CurveGrouping(16, num_curves=10, curve_length=5)
        pos = torch.randn(256, 3)
        x = torch.randn(256, 16)
        batch = torch.cat([torch.zeros(128), torch.ones(128)]).long()
        table = knn_table(pos, pos, k=9, batch_x=batch, batch_y=batch)[:, 1:]
        curves = grouping(x, batch, table)
        print(curves.shape)
        ```
    """

    def __init__(
        self,
        channels: int,
        num_curves: int,
        curve_length: int,
        norm: Union[str, Callable, None] = "batch_norm",
        norm_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()
        self.num_curves = num_curves
        self.att = nn.Linear(channels, 1, bias=False)
        self.walk = CurveWalk(channels, curve_length, norm=norm, norm_kwargs=norm_kwargs)

    def forward(self, x: Tensor, batch: Tensor, table: Tensor) -> Tensor:
        score = torch.sigmoid(self.att(x))  # (N, 1)
        start = topk(score.squeeze(1), self.num_curves, batch)
        # The momentum pairing of `CurveWalk` depends on the curve order: the curves start in point order.
        start = start.view(-1, self.num_curves).sort(dim=1).values  # (B, n)
        curves = self.walk(x * score, table, start)
        return curves


class CurveAggregation(nn.Module):
    r"""Curve aggregation from
    :arxiv: [Walk in the Cloud: Learning Curves for Point Clouds Shape Analysis](https://arxiv.org/abs/2105.01288).

    Pools every curve along its length (inter-curve descriptors) and every position across the curves (intra-curve
    descriptors) with a learned attention, then lets every point attend to both descriptor sets of its sample and
    adds the projected result to its features.

    Args:
        channels: Number of feature channels $C$; the descriptors use $C / 2$.
        act: Activation applied to the output.
        act_kwargs: Extra arguments for the activation.
        norm: Normalization of the output projection.
        norm_kwargs: Extra arguments for the normalization.

    Shape:
        - Input: features $(N, C)$, batch index $(N,)$ and curves $(B, n, L, C)$.
        - Output: $(N, C)$.

    Example:
        ```python
        import torch
        from torch_pointcloud.layers import CurveAggregation

        aggregation = CurveAggregation(16)
        x = torch.randn(256, 16)
        batch = torch.cat([torch.zeros(128), torch.ones(128)]).long()
        curves = torch.randn(2, 10, 5, 16)
        y = aggregation(x, batch, curves)
        print(y.shape)
        ```
    """

    def __init__(
        self,
        channels: int,
        act: Union[str, Callable, None] = "relu",
        act_kwargs: Optional[Dict[str, Any]] = None,
        norm: Union[str, Callable, None] = "batch_norm",
        norm_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()
        mid_channels = channels // 2
        self.curve_att = nn.Linear(channels, 1, bias=False)
        self.inter = nn.Linear(channels, mid_channels, bias=False)
        self.intra = nn.Linear(channels, mid_channels, bias=False)
        self.query = nn.Linear(channels, mid_channels, bias=False)
        self.inter_value = nn.Linear(mid_channels, mid_channels, bias=False)
        self.intra_value = nn.Linear(mid_channels, mid_channels, bias=False)
        self.proj = MLP(
            [2 * mid_channels, channels],
            plain_last=False,
            act=None,
            norm=norm,
            norm_kwargs=norm_kwargs,
            bias=False,
        )
        self.act = create_act(act, **(act_kwargs or {})) or nn.Identity()

    def forward(self, x: Tensor, batch: Tensor, curves: Tensor) -> Tensor:
        att = self.curve_att(curves)  # (B, n, L, 1)
        inter = self.inter((curves * att.softmax(dim=2)).sum(dim=2))  # (B, n, C / 2)
        intra = self.intra((curves * att.softmax(dim=1)).sum(dim=1))  # (B, L, C / 2)
        query = self.query(x)  # (N, C / 2)
        inter_weights = torch.einsum("nc,nkc->nk", query, inter.index_select(0, batch)).softmax(dim=-1)  # (N, n)
        intra_weights = torch.einsum("nc,nkc->nk", query, intra.index_select(0, batch)).softmax(dim=-1)  # (N, L)
        inter = torch.einsum("nk,nkc->nc", inter_weights, self.inter_value(inter).index_select(0, batch))
        intra = torch.einsum("nk,nkc->nc", intra_weights, self.intra_value(intra).index_select(0, batch))
        x = x + self.proj(torch.cat([inter, intra], dim=1))
        return self.act(x)


class CurveInterventionConv(nn.Module):
    r"""Curve intervention convolution (CIC) from
    :arxiv: [Walk in the Cloud: Learning Curves for Point Clouds Shape Analysis](https://arxiv.org/abs/2105.01288).

    A bottleneck residual block: an optional `RadiusMaxPool` subsampling, a linear reduction to
    `in_channels / bottleneck_ratio`, curve grouping and aggregation when `num_curves` is set, a
    `LocalPointFeatureAggregation` over the `num_neighbors` nearest points and a linear expansion to `out_channels`
    added to the (projected) input.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        num_neighbors: Number of neighbors $k$ of the local aggregation and of the curve walk.
        bottleneck_ratio: Width reduction of the inner features.
        num_layers: Depth of the local aggregation MLP.
        num_points: Points per sample after subsampling, `None` to keep the input points.
        radius: Ball radius of the subsampling, required with `num_points`.
        num_curves: Number of curves per sample, `None` to skip the curve grouping and aggregation.
        curve_length: Number of points on every curve, required with `num_curves`.
        spatial_dim: Dimension $d$ of the positions.
        act: Activation function.
        act_kwargs: Extra arguments for the activation.
        norm: Normalization layer.
        norm_kwargs: Extra arguments for the normalization.

    Shape:
        - Input: features $(N, C_\text{in})$, positions $(N, d)$ and batch index $(N,)$. Every sample holds more
            than `num_neighbors` points, and at least `num_curves` when curves are grouped.
        - Output: $(M, C_\text{out})$, $(M, d)$ and $(M,)$, with $M = N$ unless `num_points` subsamples.

    Example:
        ```python
        import torch
        from torch_pointcloud.layers import CurveInterventionConv

        block = CurveInterventionConv(32, 64, num_neighbors=8, num_curves=10, curve_length=5)
        pos = torch.randn(256, 3)
        x = torch.randn(256, 32)
        batch = torch.cat([torch.zeros(128), torch.ones(128)]).long()
        y, pos, batch = block(x, pos, batch)
        print(y.shape)
        ```
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_neighbors: int,
        bottleneck_ratio: int = 2,
        num_layers: int = 2,
        num_points: Optional[int] = None,
        radius: Optional[float] = None,
        num_curves: Optional[int] = None,
        curve_length: Optional[int] = None,
        spatial_dim: int = 3,
        act: Union[str, Callable, None] = "relu",
        act_kwargs: Optional[Dict[str, Any]] = None,
        norm: Union[str, Callable, None] = "batch_norm",
        norm_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        if (num_points is None) != (radius is None):
            raise ValueError("`num_points` and `radius` must be set together.")
        if (num_curves is None) != (curve_length is None):
            raise ValueError("`num_curves` and `curve_length` must be set together.")

        super().__init__()
        self.num_neighbors = num_neighbors
        planes = in_channels // bottleneck_ratio
        self.pool = (
            RadiusMaxPool(num_points, radius, max_num_neighbors=num_neighbors)
            if num_points is not None and radius is not None
            else None
        )
        self.conv1 = MLP(
            [in_channels, planes],
            plain_last=False,
            act=act,
            act_kwargs=act_kwargs,
            norm=norm,
            norm_kwargs=norm_kwargs,
            bias=False,
        )
        self.curve_grouping: Optional[CurveGrouping] = None
        self.curve_aggregation: Optional[CurveAggregation] = None
        if num_curves is not None and curve_length is not None:
            self.curve_grouping = CurveGrouping(planes, num_curves, curve_length, norm=norm, norm_kwargs=norm_kwargs)
            self.curve_aggregation = CurveAggregation(
                planes,
                act=act,
                act_kwargs=act_kwargs,
                norm=norm,
                norm_kwargs=norm_kwargs,
            )
        self.lpfa = LocalPointFeatureAggregation(
            planes,
            planes,
            num_layers=num_layers,
            aggr="mean",
            spatial_dim=spatial_dim,
            act=act,
            act_kwargs=act_kwargs,
            norm=norm,
            norm_kwargs=norm_kwargs,
        )
        self.conv2 = MLP(
            [planes, out_channels], plain_last=False, act=None, norm=norm, norm_kwargs=norm_kwargs, bias=False
        )
        self.shortcut = (
            MLP([in_channels, out_channels], plain_last=False, act=None, norm=norm, norm_kwargs=norm_kwargs, bias=False)
            if in_channels != out_channels
            else nn.Identity()
        )
        self.act = create_act(act, **(act_kwargs or {})) or nn.Identity()

    def forward(self, x: Tensor, pos: Tensor, batch: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        if self.pool is not None:
            x, pos, batch = self.pool(x, pos, batch)

        shortcut = x
        x = self.conv1(x)
        # The point itself (or a point at the same position) heads every row: the local aggregation keeps it, the curve
        # walk steps past it.
        table = knn_table(pos, pos, self.num_neighbors + 1, batch_x=batch, batch_y=batch)
        if self.curve_grouping is not None and self.curve_aggregation is not None:
            curves = self.curve_grouping(x, batch, table[:, 1:])
            x = self.curve_aggregation(x, batch, curves)
        x = self.lpfa(x, pos, table[:, :-1])
        x = self.conv2(x)
        return self.act(x + self.shortcut(shortcut)), pos, batch


class CurveNetFeaturePropagation(nn.Module):
    r"""Feature propagation of CurveNet: $k$-NN interpolation of the coarse features, an `AttentionGate` on the skip
    features, concatenation and an MLP.

    The interpolation weights are $1 / (d^2 + \epsilon)$, and the skip features are concatenated **before** the
    interpolated ones.

    Args:
        in_channels: Number of channels of the coarse features.
        skip_channels: Number of channels of the skip features.
        channels: Output channels of every MLP layer.
        gate_channels: Width of the attention gate, `None` for no gate.
        k: Number of neighbors of the interpolation.
        eps: Numerical stability term of the interpolation weights.
        act: Activation function.
        act_kwargs: Extra arguments for the activation.
        norm: Normalization layer.
        norm_kwargs: Extra arguments for the normalization.

    Shape:
        - Input: coarse features $(M, C_\text{in})$, positions $(M, d)$ and batch index $(M,)$, then skip features
            $(N, C_\text{skip})$, positions $(N, d)$ and batch index $(N,)$.
        - Output: $(N, C_\text{out})$, $(N, d)$ and $(N,)$.

    Example:
        ```python
        import torch
        from torch_pointcloud.layers import CurveNetFeaturePropagation

        fp = CurveNetFeaturePropagation(64, 32, channels=[32, 32], gate_channels=16)
        pos, pos_skip = torch.randn(16, 3), torch.randn(128, 3)
        x, x_skip = torch.randn(16, 64), torch.randn(128, 32)
        batch, batch_skip = torch.zeros(16).long(), torch.zeros(128).long()
        y, _, _ = fp(x, pos, batch, x_skip, pos_skip, batch_skip)
        print(y.shape)
        ```
    """

    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        channels: Sequence[int],
        gate_channels: Optional[int] = None,
        k: int = 3,
        eps: float = 1e-8,
        act: Union[str, Callable, None] = "relu",
        act_kwargs: Optional[Dict[str, Any]] = None,
        norm: Union[str, Callable, None] = "batch_norm",
        norm_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()
        self.k = k
        self.eps = eps
        self.gate: Optional[AttentionGate] = None
        if gate_channels is not None:
            self.gate = AttentionGate(
                in_channels,
                skip_channels,
                gate_channels,
                act=act,
                act_kwargs=act_kwargs,
                norm=norm,
                norm_kwargs=norm_kwargs,
            )
        self.mlp = MLP(
            [skip_channels + in_channels] + list(channels),
            plain_last=False,
            act=act,
            act_kwargs=act_kwargs,
            norm=norm,
            norm_kwargs=norm_kwargs,
        )

    def forward(
        self,
        x: Tensor,
        pos: Tensor,
        batch: Tensor,
        x_skip: Tensor,
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
            weighting="squared",
            eps=self.eps,
        )
        if self.gate is not None:
            x_skip = x_skip * self.gate(x, x_skip)
        x = self.mlp(torch.cat([x_skip, x], dim=1))
        return x, pos_skip, batch_skip
