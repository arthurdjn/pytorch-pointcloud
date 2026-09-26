"""RandLA-Net local feature aggregation: spatial encoding, attentive pooling, and the dilated residual block."""

from typing import Any, Callable, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.nn import MLP, MessagePassing
from torch_geometric.typing import Adj, OptTensor, PairTensor
from torch_geometric.utils import softmax
from typing_extensions import Unpack

from torch_pointcloud.layers.act import create_act
from torch_pointcloud.ops.cluster import knn_graph
from torch_pointcloud.utils.types import MessagePassingParams


class LocalSpatialEncoding(nn.Module):
    """Per-edge spatial encoding MLP.

    Wraps a single `Linear+norm+act` block that lifts an input feature to `out_channels`.
    Used twice per `LocalFeatureAggregation`: first on the raw 10-channel relative
    positional encoding, then on its output.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        act: Union[str, Callable, None],
        act_kwargs: Optional[Dict[str, Any]] = None,
        act_first: bool = False,
        norm: Union[str, Callable, None],
        norm_kwargs: Optional[Dict[str, Any]] = None,
        bias: bool = False,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.mlp = MLP(
            channel_list=[in_channels, out_channels],
            act=act,
            act_kwargs=act_kwargs,
            act_first=act_first,
            norm=norm,
            norm_kwargs=norm_kwargs,
            bias=bias,
            plain_last=False,
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.mlp(x)


class AttentivePooling(MessagePassing):
    r"""Attention-weighted aggregation of neighbor features.

    Each edge concatenates its source feature with a per-edge spatial encoding, scores the result with a
    no-bias linear layer, softmax-normalizes the scores over the neighbors of each target point, sums the
    score-weighted features per target, then projects them with a `Linear+norm+act` block.

    `edge_index` lists `[source, target]` pairs and `edge_attr` holds the $(E, C_\text{pos})$ spatial encoding
    of each edge.

    Args:
        in_channels: Channels of each edge feature, the source feature and the spatial encoding concatenated.
        out_channels: Output channels after the post-aggregation MLP.
        **kwargs: Extra arguments for `MessagePassing`.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        act: Union[str, Callable, None],
        act_kwargs: Optional[Dict[str, Any]] = None,
        act_first: bool = False,
        norm: Union[str, Callable, None],
        norm_kwargs: Optional[Dict[str, Any]] = None,
        bias: bool = False,
        **kwargs: Unpack[MessagePassingParams],
    ) -> None:
        kwargs.setdefault("aggr", "add")
        super().__init__(**kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.fc = nn.Linear(in_channels, in_channels, bias=False)
        self.mlp = MLP(
            channel_list=[in_channels, out_channels],
            act=act,
            act_kwargs=act_kwargs,
            act_first=act_first,
            norm=norm,
            norm_kwargs=norm_kwargs,
            bias=bias,
            plain_last=False,
        )

    def forward(self, x: Union[Tensor, PairTensor], edge_index: Adj, edge_attr: Tensor) -> Tensor:
        # propagate_type: (x: Tensor, edge_attr: Tensor)
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        return self.mlp(out)

    def message(self, x_j: Tensor, edge_attr: Tensor, index: Tensor, ptr: OptTensor, size_i: Optional[int]) -> Tensor:
        edge_feats = torch.cat([x_j, edge_attr], dim=1)
        att_scores = softmax(self.fc(edge_feats), index, ptr, size_i)
        return att_scores * edge_feats


class LocalFeatureAggregation(nn.Module):
    r"""Local feature aggregation: two rounds of spatial encoding and attentive pooling.

    Each round grows the receptive field, taking a per-point feature of width $d_\text{out} / 2$ to
    $d_\text{out}$. The 10-channel relative positional encoding keeps the channel order of
    :github: [QingyongHu/RandLA-Net](https://github.com/QingyongHu/RandLA-Net)
    (`cat([rel_dist, rel_xyz, xyz_i, xyz_j], dim=-1)`) so pretrained weights load without permuting the first
    kernel, and the second encoding re-projects the output of the first.
    """

    def __init__(
        self,
        d_out: int,
        *,
        act: Union[str, Callable, None],
        act_kwargs: Optional[Dict[str, Any]] = None,
        act_first: bool = False,
        norm: Union[str, Callable, None],
        norm_kwargs: Optional[Dict[str, Any]] = None,
        bias: bool = False,
    ) -> None:
        super().__init__()
        if d_out % 2 != 0:
            raise ValueError(f"`d_out` must be even, got {d_out}.")

        mlp_kwargs: Dict[str, Any] = dict(
            act=act,
            act_kwargs=act_kwargs,
            act_first=act_first,
            norm=norm,
            norm_kwargs=norm_kwargs,
            bias=bias,
        )

        self.d_out = d_out
        self.lse1 = LocalSpatialEncoding(10, d_out // 2, **mlp_kwargs)
        self.att_pooling_1 = AttentivePooling(d_out, d_out // 2, **mlp_kwargs)
        self.lse2 = LocalSpatialEncoding(d_out // 2, d_out // 2, **mlp_kwargs)
        self.att_pooling_2 = AttentivePooling(d_out, d_out, **mlp_kwargs)

    def forward(self, x: Tensor, pos: Tensor, edge_index: Tensor) -> Tensor:
        source, target = edge_index

        # Per-edge 10-channel relative positional encoding.
        pos_i = pos[target]
        pos_j = pos[source]
        rel_xyz = pos_i - pos_j
        rel_dist = torch.linalg.norm(rel_xyz, dim=1, keepdim=True)
        rel = torch.cat([rel_dist, rel_xyz, pos_i, pos_j], dim=1)  # (E, 10)

        f_pos1 = self.lse1(rel)  # (E, d_out//2)
        x = self.att_pooling_1(x, edge_index, f_pos1)  # (N, d_out//2)

        f_pos2 = self.lse2(f_pos1)  # (E, d_out//2)
        x = self.att_pooling_2(x, edge_index, f_pos2)  # (N, d_out)
        return x


class RandLANetResidualBlock(nn.Module):
    r"""RandLA-Net dilated residual block.

    Maps `d_in` channels to $2 \cdot d_\text{out}$ via a residual path of
    `MLP -> LocalFeatureAggregation -> MLP` plus a parallel `Linear+norm` shortcut.
    `mlp2` and `shortcut` carry no activation; the configured activation is applied
    once after the residual sum.

    Args:
        d_in: Number of input channels.
        d_out: "Configuration" channel count; the block actually outputs $2 \cdot d_\text{out}$.
        num_neighbors: Number of neighbors for the local feature aggregation.
    """

    def __init__(
        self,
        d_in: int,
        d_out: int,
        num_neighbors: int,
        *,
        act: Union[str, Callable, None],
        act_kwargs: Optional[Dict[str, Any]] = None,
        act_first: bool = False,
        norm: Union[str, Callable, None],
        norm_kwargs: Optional[Dict[str, Any]] = None,
        bias: bool = False,
    ) -> None:
        super().__init__()
        if d_out % 2 != 0:
            raise ValueError(f"`d_out` must be even, got {d_out}.")

        mlp_kwargs: Dict[str, Any] = dict(
            act=act,
            act_kwargs=act_kwargs,
            act_first=act_first,
            norm=norm,
            norm_kwargs=norm_kwargs,
            bias=bias,
        )

        self.num_neighbors = num_neighbors
        self.d_in = d_in
        self.d_out = d_out
        self.out_channels = 2 * d_out
        self.mlp1 = MLP(channel_list=[d_in, d_out // 2], plain_last=False, **mlp_kwargs)
        self.lfa = LocalFeatureAggregation(d_out, **mlp_kwargs)
        self.mlp2 = MLP(
            channel_list=[d_out, 2 * d_out],
            act=None,
            norm=norm,
            norm_kwargs=norm_kwargs,
            bias=bias,
            plain_last=False,
        )
        self.shortcut = MLP(
            channel_list=[d_in, 2 * d_out],
            act=None,
            norm=norm,
            norm_kwargs=norm_kwargs,
            bias=bias,
            plain_last=False,
        )
        self.act = create_act(act, **(act_kwargs or {})) or nn.Identity()

    def forward(self, x: Tensor, pos: Tensor, batch: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        edge_index = knn_graph(pos, self.num_neighbors, batch=batch, loop=True)
        shortcut = self.shortcut(x)
        x = self.mlp1(x)
        x = self.lfa(x, pos, edge_index)
        x = self.mlp2(x)
        return self.act(x + shortcut), pos, batch
