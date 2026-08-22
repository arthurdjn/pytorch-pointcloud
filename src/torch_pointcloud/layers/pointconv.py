"""PointConv message-passing convolutions, with and without density reweighting."""

from typing import TYPE_CHECKING, Union

import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.nn import MessagePassing
from torch_geometric.typing import Adj, OptTensor, PairOptTensor, PairTensor, SparseTensor
from torch_geometric.utils import add_self_loops, remove_self_loops
from typing_extensions import Unpack

from torch_pointcloud.utils.imports import _TORCH_SCATTER_GITHUB_URL, _TORCH_SPARSE_GITHUB_URL, optional_import
from torch_pointcloud.utils.types import MessagePassingParams

if TYPE_CHECKING:
    import torch_sparse
    from torch_scatter import scatter_max


torch_sparse, _ = optional_import("torch_sparse", url=_TORCH_SPARSE_GITHUB_URL)
scatter_max, _ = optional_import("torch_scatter", "scatter_max", url=_TORCH_SCATTER_GITHUB_URL)


class PointConv(MessagePassing):
    r"""PointConv message passing: a per-edge feature outer-producted with a learned continuous weight.

    `local_nn` embeds the relative position concatenated with the neighbor feature, `weight_nn` turns the
    relative position alone into the convolution weights, and their outer product is summed over the
    neighbors of each point.

    Args:
        local_nn: Network embedding the relative positions and neighbor features.
        weight_nn: Network mapping a relative position to the convolution weights.
        add_self_loops: Whether to add a self loop to every point before propagating.
        eps: Numerical stability constant.
        **kwargs: Extra arguments for `MessagePassing`.
    """

    def __init__(
        self,
        local_nn: nn.Module,
        weight_nn: nn.Module,
        add_self_loops: bool = False,
        eps: float = 1e-6,
        **kwargs: Unpack[MessagePassingParams],
    ):
        kwargs.setdefault("aggr", "add")
        super().__init__(**kwargs)
        self.local_nn = local_nn
        self.weight_nn = weight_nn
        self.add_self_loops = add_self_loops
        self.eps = eps

    def forward(self, x: Union[OptTensor, PairOptTensor], pos: Union[Tensor, PairTensor], edge_index: Adj) -> Tensor:
        if not isinstance(x, tuple):
            x = (x, None)
        if not isinstance(pos, tuple):
            pos = (pos, pos)

        if self.add_self_loops:
            if isinstance(edge_index, Tensor):
                edge_index, _ = remove_self_loops(edge_index)
                edge_index, _ = add_self_loops(edge_index, num_nodes=min(pos[0].size(0), pos[1].size(0)))
            elif isinstance(edge_index, SparseTensor):
                edge_index = torch_sparse.set_diag(edge_index)

        return self.propagate(edge_index, x=x, pos=pos)

    def message(self, x_j: OptTensor, pos_i: Tensor, pos_j: Tensor, index: Tensor) -> Tensor:
        pos_rel = pos_j - pos_i

        feat = torch.cat([pos_rel, x_j], dim=-1) if x_j is not None else pos_rel
        h = self.local_nn(feat)
        w = self.weight_nn(pos_rel)

        msg = h.unsqueeze(-1) * w.unsqueeze(-2)
        return msg.view(msg.size(0), -1)

    def extra_repr(self) -> str:
        return f"local_nn={self.local_nn}, weight_nn={self.weight_nn}"

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.extra_repr()})"


class PointConvDensity(MessagePassing):
    r"""`PointConv` with inverse density reweighting.

    Each neighbor's embedded feature is scaled by `density_nn` applied to its density relative to the
    densest neighbor, so that densely sampled regions do not dominate the sum.

    Args:
        local_nn: Network embedding the relative positions and neighbor features.
        weight_nn: Network mapping a relative position to the convolution weights.
        density_nn: Network mapping a relative density to a per-edge scale.
        add_self_loops: Whether to add a self loop to every point before propagating.
        eps: Numerical stability constant.
        **kwargs: Extra arguments for `MessagePassing`.
    """

    def __init__(
        self,
        local_nn: nn.Module,
        weight_nn: nn.Module,
        density_nn: nn.Module,
        add_self_loops: bool = False,
        eps: float = 1e-6,
        **kwargs: Unpack[MessagePassingParams],
    ):
        kwargs.setdefault("aggr", "add")
        super().__init__(**kwargs)
        self.local_nn = local_nn
        self.weight_nn = weight_nn
        self.density_nn = density_nn
        self.add_self_loops = add_self_loops
        self.eps = eps

    def forward(
        self,
        x: Union[OptTensor, PairOptTensor],
        pos: Union[Tensor, PairTensor],
        edge_index: Adj,
        density: Union[OptTensor, PairOptTensor],
    ) -> Tensor:
        if not isinstance(x, tuple):
            x = (x, None)
        if not isinstance(pos, tuple):
            pos = (pos, pos)
        if not isinstance(density, tuple):
            density = (density, None)

        if self.add_self_loops:
            if isinstance(edge_index, Tensor):
                edge_index, _ = remove_self_loops(edge_index)
                edge_index, _ = add_self_loops(edge_index, num_nodes=min(pos[0].size(0), pos[1].size(0)))
            elif isinstance(edge_index, SparseTensor):
                edge_index = torch_sparse.set_diag(edge_index)

        return self.propagate(edge_index, x=x, pos=pos, density=density)

    def message(
        self,
        x_j: OptTensor,
        pos_i: Tensor,
        pos_j: Tensor,
        density_j: OptTensor,
        index: Tensor,
    ) -> Tensor:
        pos_rel = pos_j - pos_i

        feat = torch.cat([pos_rel, x_j], dim=-1) if x_j is not None else pos_rel
        h = self.local_nn(feat)
        w = self.weight_nn(pos_rel)

        max_density, _ = scatter_max(density_j, index, dim=0)
        density_rel = density_j / (max_density[index])
        density_scale = self.density_nn(density_rel)
        h = h * density_scale

        msg = h.unsqueeze(-1) * w.unsqueeze(-2)
        return msg.view(msg.size(0), -1)

    def extra_repr(self) -> str:
        return f"local_nn={self.local_nn}, weight_nn={self.weight_nn}, density_nn={self.density_nn}"

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.extra_repr()})"
