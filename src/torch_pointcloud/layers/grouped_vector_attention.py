"""Grouped vector attention over a neighborhood graph."""

from typing import Any, Callable, Dict, Optional, Union

from torch import Tensor, nn
from torch_geometric.nn import MLP, MessagePassing
from torch_geometric.typing import Adj, OptTensor, PairTensor
from torch_geometric.utils import softmax
from typing_extensions import Unpack

from torch_pointcloud.layers.act import create_act
from torch_pointcloud.layers.norms import create_norm
from torch_pointcloud.utils.types import MessagePassingParams


class GroupedVectorAttention(MessagePassing):
    """Vector attention over a neighborhood graph, with one weight vector shared by each group of channels.

    The relation between a query and its neighbor keys is optionally scaled and shifted by a learned
    encoding of their relative position, then mapped to `num_groups` weights and softmax-normalized
    over each destination's neighbors.

    `edge_index` lists `[source, target]` pairs: keys and values come from the source points and queries from
    the target points, so a bipartite graph is given as `(x_source, x_target)` and `(pos_source, pos_target)`
    pairs.

    Args:
        channels: Feature width of the queries, keys and values.
        num_groups: Number of channel groups sharing one attention weight.
        attn_drop: Dropout probability on the attention weights.
        qkv_bias: Whether the query, key and value projections carry a bias.
        pe_multiplier: Whether to scale the query-key relation by an encoding of the relative position.
        pe_bias: Whether to shift the query-key relation and the values by an encoding of the relative position.
        norm: Normalization type or callable.
        act: Activation type or callable.
        act_kwargs: Extra activation arguments.
        norm_kwargs: Extra normalization arguments.
        **kwargs: Extra arguments for `MessagePassing`.
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
        **kwargs: Unpack[MessagePassingParams],
    ):
        kwargs.setdefault("aggr", "add")
        super().__init__(**kwargs)
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

    def forward(
        self,
        x: Union[Tensor, PairTensor],
        pos: Union[Tensor, PairTensor],
        edge_index: Adj,
    ) -> Tensor:
        if not isinstance(x, tuple):
            x = (x, x)
        if not isinstance(pos, tuple):
            pos = (pos, pos)

        query, key, value = self.q(x[1]), self.k(x[0]), self.v(x[0])
        size = (x[0].size(0), x[1].size(0))
        # propagate_type: (query: Tensor, key: Tensor, value: Tensor, pos: PairTensor)
        return self.propagate(edge_index, size=size, query=query, key=key, value=value, pos=pos)

    def message(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        pos: PairTensor,
        edge_index_i: Tensor,
        edge_index_j: Tensor,
        index: Tensor,
        ptr: OptTensor,
        size_i: Optional[int],
    ) -> Tensor:
        # Gathered here rather than lifted: `propagate` keeps every lifted argument alive until aggregation, while
        # these per-edge tensors are consumed as soon as they are formed.
        pos_rel = pos[0][edge_index_j] - pos[1][edge_index_i]
        relation_qk = key[edge_index_j] - query[edge_index_i]
        value_j = value[edge_index_j]

        if self.pe_multiplier is not None:
            relation_qk = relation_qk * self.pe_multiplier(pos_rel)

        if self.pe_bias is not None:
            bias = self.pe_bias(pos_rel)
            relation_qk = relation_qk + bias
            value_j = value_j + bias

        weight = self.weight_encoding(relation_qk)
        weight = self.attn_drop(softmax(weight, index, ptr, size_i))

        value_j = value_j.view(-1, self.num_groups, self.channels // self.num_groups) * weight.unsqueeze(-1)
        return value_j.view(-1, self.channels)
