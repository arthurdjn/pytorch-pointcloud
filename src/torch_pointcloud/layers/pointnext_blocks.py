r"""
PointNeXt convolution layer introduced in the
:arxiv: [PointNeXt: Revisiting PointNet++ with Improved Training and Scaling Strategies](https://arxiv.org/abs/2206.04670)
by Guocheng Qian et al.

!!! note
    This layer is also referred to as Local Aggregation in different papers and implementations.

This layer implements the `torch_geometric.nn.conv.MessagePassing` interface from PyTorch Geometric,
which allows for local aggregation of features.

!!! tip
    This layer is similar to the `torch_geometric.nn.conv.PointNetConv` layer from PyTorch Geometric,
    and introduces relative position normalization.

You can use it as follows:

```{.python notest}
import torch
from torch_geometric.nn import MLP, radius_graph
from torch_pointcloud.layers import PointNeXtConv

torch.manual_seed(0)
x = torch.randn(10, 10)
pos = torch.randn(10, 3)
batch = torch.zeros(10, dtype=torch.long)
edge_index = radius_graph(pos, r=1.5, batch=batch, max_num_neighbors=16)

conv = PointNeXtConv(MLP([3 + 10, 10]))

# Normalize the relative position by the query radius
out = conv(x, pos, edge_index, pos_divisor=1.5)

# This will be equivalent to the PointNetConv layer
out = conv(x, pos, edge_index)
```
"""

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn import MLP, MessagePassing, radius, radius_graph
from torch_geometric.nn.inits import reset
from torch_geometric.typing import Adj, NoneType, OptTensor, PairOptTensor, PairTensor, SparseTensor, torch_sparse
from torch_geometric.utils import add_self_loops, remove_self_loops
from typing_extensions import Unpack

from torch_pointcloud.utils.cluster import fps
from torch_pointcloud.utils.conversion import ensure_list, ensure_tuple_size
from torch_pointcloud.utils.types import AggrType, MessagePassingParams

from .act import create_act


class _PlainLastActMLP(MLP):
    """MLP variant that removes activation from the last *in-loop* layer.

    In the base `MLP`, when `plain_last=True` the loop covers all layers
    except the final linear projection.  The activation function is still
    applied at every iteration inside that loop, including the last one.
    This means the penultimate layer's output passes through an activation
    before being fed into the plain final linear layer.

    `_PlainLastActMLP` modifies this behavior by skipping the activation
    at the last loop iteration (`i == len(self.norms) - 1`), so that the
    two outermost layers are separated only by normalization and dropout -
    no non-linearity.

    For a 3-layer network (`channel_list = [d_in, h, h, d_out]`) the
    effective forward passes compare as follows:

    | Variant                          | Forward pass                                                         |
    | -------------------------------- | -------------------------------------------------------------------- |
    | `MLP` (`plain_last=True`)        | lin₀ → act → norm → drop → lin₁ → act  → norm → drop → lin₂ → drop   |
    | `_PlainLastActMLP`               | lin₀ → act → norm → drop → lin₁ → norm → drop → lin₂ → drop          |

    """

    def forward(
        self,
        x: Tensor,
        batch: Optional[Tensor] = None,
        batch_size: Optional[int] = None,
        return_emb: NoneType = None,
    ) -> Tensor:
        # `return_emb` is annotated here as `NoneType` to be compatible with
        # TorchScript, which does not support different return types based on
        # the value of an input argument.
        emb: Optional[Tensor] = None

        # If `plain_last=True`, then `len(norms) = len(lins) - 1`, thus skipping
        # the execution of the last layer inside the for-loop.
        last = len(self.norms) - 1
        for i, (lin, norm) in enumerate(zip(self.lins, self.norms)):
            x = lin(x)
            if self.act is not None and self.act_first and i < last:
                x = self.act(x)
            x = norm(x, batch, batch_size) if self.supports_norm_batch else norm(x)
            if self.act is not None and not self.act_first and i < last:
                x = self.act(x)
            x = F.dropout(x, p=self.dropout[i], training=self.training)
            if isinstance(return_emb, bool) and return_emb is True:
                emb = x

        if self.plain_last:
            x = self.lins[-1](x)
            x = F.dropout(x, p=self.dropout[-1], training=self.training)

        return (x, emb) if isinstance(return_emb, bool) else x  # type: ignore[return-value]


class PointNeXtConv(MessagePassing):
    r"""PointNeXt grouping convolution on top of PyG's `MessagePassing`.

    Each message concatenates the relative position with the neighbor features
    (`cat([pos_j - pos_i, x_j])`) and applies `local_nn`; when a `pos_divisor` is given the relative
    positions are normalized by it (PointNeXt normalizes by the ball-query radius).

    Args:
        local_nn: Network applied to each message of shape $(E, D + C)$.
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
        pos_divisor: Optional[float] = None,
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

        return self.propagate(edge_index, x=x, pos=pos, pos_divisor=pos_divisor)

    def message(
        self,
        x_j: Optional[Tensor],
        pos_i: Tensor,
        pos_j: Tensor,
        pos_divisor: Optional[float] = None,
    ) -> Tensor:
        msg = pos_j - pos_i
        if pos_divisor is not None:
            msg = msg / pos_divisor
        if x_j is not None:
            msg = torch.cat([msg, x_j], dim=1)

        return self.local_nn(msg)

    def extra_repr(self) -> str:
        return f"local_nn={self.local_nn}"

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.extra_repr()})"


class PointNeXtSetAbstraction(nn.Module):
    r"""PointNeXt set-abstraction block: FPS centroids, radius-normalized grouping, and optional
    residual skip connections.

    Farthest point sampling selects the centroids, a ball query gathers the neighbors of each
    centroid per scale (Multi-Scale Grouping when `channels` is a nested sequence), and the relative
    positions are normalized by the query radius before the grouping convolution.

    Args:
        spatial_dim: Dimension of point coordinates.
        in_channels: Number of input feature channels.
        channels: Per-scale MLP channel sizes; a nested sequence enables Multi-Scale Grouping.
        ratio: Fractional farthest-point-sampling rate.
        radius: Ball-query radius per scale, also used to normalize the relative positions.
        num_neighbors: Maximum number of neighbors per scale.
        dropout: Dropout rate inside the per-scale MLPs.
        add_self_loops: Whether to add self-loops to the grouping edge index.
        aggr: Message aggregation used by the convolutions.
        use_res: Whether each scale adds a residual skip connection from the sampled centroid
            features (with a linear projection when the channel counts differ).
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
        use_res: bool = True,
    ) -> None:
        super().__init__()
        act_kwargs = act_kwargs or {}
        norm_kwargs = norm_kwargs or {}

        self.spatial_dim = spatial_dim
        self.in_channels = in_channels
        self.dropout = dropout
        self.act = create_act(act, **act_kwargs) or nn.Identity()
        self.ratio = ratio
        self.add_self_loops = add_self_loops
        self.aggr = aggr
        self.use_res = use_res

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
        self.radius = ensure_tuple_size(radius, size=len(sizes), extra_msg=extra_msg.format(param="radius"))
        self.num_neighbors = ensure_tuple_size(
            num_neighbors,
            size=len(sizes),
            extra_msg=extra_msg.format(param="num_neighbors"),
        )

        mlp_cls = _PlainLastActMLP if use_res else MLP
        self.convs = nn.ModuleList()
        for scale_channels in self.channels:
            local_nn = mlp_cls(
                [in_channels + spatial_dim] + list(scale_channels),
                act=self.act,
                act_first=act_first,
                act_kwargs=act_kwargs,
                norm=norm,
                norm_kwargs=norm_kwargs,
                bias=bias,
                dropout=dropout,
                plain_last=False,
            )
            self.convs.append(PointNeXtConv(local_nn=local_nn, add_self_loops=add_self_loops, aggr=self.aggr))

        self.skip_convs = nn.ModuleList()
        if use_res:
            for scale_channels in self.channels:
                out_channels = scale_channels[-1]
                if out_channels == in_channels:
                    self.skip_convs.append(nn.Identity())
                else:
                    skip_conv = MLP(
                        [in_channels, out_channels],
                        act=self.act,
                        act_first=act_first,
                        act_kwargs=act_kwargs,
                        norm=norm,
                        norm_kwargs=norm_kwargs,
                        bias=bias,
                        dropout=dropout,
                        plain_last=True,
                    )
                    self.skip_convs.append(skip_conv)

    def forward(self, x: Tensor, pos: Tensor, batch: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        # In eval mode pin the FPS start to make predictions reproducible across runs.
        idx = fps(pos, batch, ratio=self.ratio, random_start=self.training)
        x_dst = None if x is None else x[idx]
        pos_dst = pos[idx]
        batch_dst = batch[idx]

        msg_x: List[Tensor] = []
        for r, num_neighbors, conv in zip(self.radius, self.num_neighbors, self.convs):
            row, col = radius(pos, pos_dst, r=r, batch_x=batch, batch_y=batch_dst, max_num_neighbors=num_neighbors)
            edge_index = torch.stack([col, row], dim=0)
            x_out = conv((x, x_dst), (pos, pos_dst), edge_index, pos_divisor=r)

            if self.use_res:
                skip_conv = self.skip_convs[len(msg_x)]
                x_skip = skip_conv(x_dst)
                x_out = self.act(x_out + x_skip)

            msg_x.append(x_out)

        return torch.cat(msg_x, dim=1), pos_dst, batch_dst


class PointNeXtResidualBlock(nn.Module):
    r"""PointNeXt inverted-residual MLP block (InvResMLP).

    A radius-graph grouping convolution followed by an inverted-bottleneck MLP
    (`channels -> channels * expansion -> channels`), wrapped in a single residual connection.
    The resolution is unchanged; downsampling happens in `PointNeXtSetAbstraction`.

    Args:
        spatial_dim: Dimension of point coordinates.
        channels: Number of input and output feature channels.
        expansion: Expansion factor of the bottleneck MLP.
        radius: Radius of the grouping graph, also used to normalize the relative positions.
        num_neighbors: Maximum number of neighbors in the grouping graph.
        add_self_loops: Whether to add self-loops to the grouping edge index.
        aggr: Message aggregation used by the convolution.
    """

    def __init__(
        self,
        spatial_dim: int,
        channels: int,
        expansion: int,
        radius: float,
        num_neighbors: int,
        act: Union[str, Callable, None] = "relu",
        act_kwargs: Optional[Dict[str, Any]] = None,
        act_first: bool = False,
        norm: Union[str, Callable, None] = "batch_norm",
        norm_kwargs: Optional[Dict[str, Any]] = None,
        bias: Union[bool, List[bool]] = True,
        add_self_loops: bool = False,
        aggr: AggrType = "max",
    ):
        super().__init__()
        act_kwargs = act_kwargs or {}
        norm_kwargs = norm_kwargs or {}

        self.spatial_dim = spatial_dim
        self.radius = radius
        self.num_neighbors = num_neighbors
        self.act = create_act(act, **act_kwargs) or nn.Identity()

        # NOTE: use the PointNeXtConv instead of the PointNeXtSetAbstraction layer to create a larger skip connection:
        # x -> conv -> mlp -> x + identity
        # |                          ^
        # +--------------------------+
        local_nn = MLP(
            [channels + self.spatial_dim, channels],
            act=act,
            act_kwargs=act_kwargs,
            act_first=act_first,
            norm=norm,
            norm_kwargs=norm_kwargs,
            bias=bias,
            plain_last=False,
        )
        self.conv = PointNeXtConv(
            local_nn=local_nn,
            add_self_loops=add_self_loops,
            aggr=aggr,
        )

        mid_channels = channels * expansion
        self.mlp = _PlainLastActMLP(
            [channels, mid_channels, channels],
            act=act,
            act_kwargs=act_kwargs,
            act_first=act_first,
            norm=norm,
            norm_kwargs=norm_kwargs,
            bias=bias,
            plain_last=False,
        )

    def forward(self, x: Tensor, pos: Tensor, batch: Tensor) -> Tensor:
        identity = x
        edge_index = radius_graph(pos, r=self.radius, batch=batch, max_num_neighbors=self.num_neighbors)
        x = self.conv(x, pos, edge_index, pos_divisor=self.radius)
        x = self.mlp(x)
        x = self.act(x + identity)
        return x
