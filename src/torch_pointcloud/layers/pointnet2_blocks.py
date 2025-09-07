"""_summary_

Raises:
    ValueError: _description_
    ValueError: _description_

Returns:
    _description_
"""

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.nn import MLP, MessagePassing, fps, knn_interpolate, radius
from torch_geometric.nn.inits import reset
from torch_geometric.nn.resolver import activation_resolver
from torch_geometric.typing import Adj, OptTensor, PairOptTensor, PairTensor, SparseTensor, torch_sparse
from torch_geometric.utils import add_self_loops, remove_self_loops
from typing_extensions import Unpack

from torch_pointcloud.layers.pools import create_pool
from torch_pointcloud.utils.conversion import ensure_list, ensure_tuple_size
from torch_pointcloud.utils.types import AggrType, MessagePassingParams


class PointNet2Conv(MessagePassing):
    def __init__(self, local_nn: nn.Module, add_self_loops: bool = True, **kwargs: Unpack[MessagePassingParams]):
        kwargs.setdefault("aggr", "max")
        super().__init__(**kwargs)
        self.local_nn = local_nn
        self.add_self_loops = add_self_loops
        self.reset_parameters()

    def reset_parameters(self):
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
        self.act = activation_resolver(act, **act_kwargs)
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
        for i, channels in enumerate(self.channels):
            in_channels = self.in_channels + self.spatial_dim
            conv = self.configure_conv(channels, i)
            self.convs.append(conv)

    def configure_conv(self, channels: Sequence[int], index: int) -> MessagePassing:
        in_channels = self.in_channels + self.spatial_dim
        local_nn = MLP(
            [in_channels] + list(channels),
            act=self.act,
            act_first=self.act_first,
            act_kwargs=self.act_kwargs,
            norm=self.norm,
            norm_kwargs=self.norm_kwargs,
            bias=self.bias,
            dropout=self.dropout,
            plain_last=False,
        )

        return PointNet2Conv(local_nn=local_nn, add_self_loops=self.add_self_loops, aggr=self.aggr)

    def forward(self, x: Tensor, pos: Tensor, batch: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        idx = fps(pos, batch, ratio=self.ratio)
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
        aggr: str = "max",
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
    ) -> None:
        super().__init__()
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

    def forward(
        self,
        x: Tensor,
        pos: Tensor,
        batch: Tensor,
        x_skip: OptTensor,
        pos_skip: Tensor,
        batch_skip: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        x = knn_interpolate(x, pos, pos_skip, batch, batch_skip, k=self.k)
        if x_skip is not None:
            x = torch.cat([x, x_skip], dim=1)

        x = self.mlp(x)
        return x, pos_skip, batch_skip
