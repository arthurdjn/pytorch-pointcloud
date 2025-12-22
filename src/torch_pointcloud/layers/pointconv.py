from typing import TYPE_CHECKING, Any, Callable, Dict, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
from torch import Tensor
from torch_cluster import radius
from torch_geometric.nn import MLP, MessagePassing
from torch_geometric.typing import Adj, OptTensor, PairOptTensor, PairTensor, SparseTensor
from torch_geometric.utils import add_self_loops, remove_self_loops
from typing_extensions import Unpack

from torch_pointcloud.models.pointmlp import LinearBlock
from torch_pointcloud.utils.conversion import ensure_list
from torch_pointcloud.utils.imports import optional_import
from torch_pointcloud.utils.types import MessagePassingParams

if TYPE_CHECKING:
    import torch_sparse
    from torch_cluster import knn, radius
    from torch_scatter import scatter_add, scatter_max


torch_sparse, _ = optional_import("torch_sparse")
knn, _ = optional_import("torch_cluster", "knn")
radius, _ = optional_import("torch_cluster", "radius")
scatter_add, _ = optional_import("torch_scatter", "scatter_add")
scatter_max, _ = optional_import("torch_scatter", "scatter_max")


def gaussian_kernel_density(
    pos: Tensor,
    batch: Tensor,
    bandwidth: float,
    epsilon: float = 1e-4,
) -> Tensor:
    multiplier = torch.sqrt(-2 * torch.tensor(epsilon).log()).item()
    r = bandwidth * multiplier

    row, col = radius(x=pos, y=pos, r=r, batch_x=batch, batch_y=batch)

    diff = pos[col] - pos[row]
    dist2 = (diff * diff).sum(dim=-1)

    kde_edges = torch.exp(-dist2 / (2.0 * bandwidth**2)) / (2.5 * bandwidth)

    density_sum = scatter_add(kde_edges, col, dim=0, dim_size=pos.size(0))

    batch_counts = torch.bincount(batch)
    N_per_point = batch_counts[batch]
    return density_sum / N_per_point.clamp(min=1.0)


class PointConv(MessagePassing):
    def __init__(
        self,
        local_nn: nn.Module,
        weight_nn: nn.Module,
        density_nn: Optional[nn.Module] = None,
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
        density: Union[OptTensor, PairOptTensor] = None,
    ) -> Tensor:
        if not isinstance(x, tuple):
            x = (x, None)
        if not isinstance(pos, tuple):
            pos = (pos, pos)
        if density is not None and not isinstance(density, tuple):
            density = (density, density)

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

        if self.density_nn is not None and density_j is not None:
            max_density, _ = scatter_max(density_j, index, dim=0)

            density_rel = density_j / (max_density[index] + self.eps)
            density_scale = self.density_nn(density_rel)
            h = h * density_scale

        msg = h.unsqueeze(-1) * w.unsqueeze(-2)
        return msg.view(msg.size(0), -1)


class PointConvSetAbstraction(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_neighbors: int,
        bandwidth: float,
        channels: Sequence[int],
        density_channels: Optional[Sequence[int]] = (16, 8),
        weight_channels: Sequence[int] = (8, 8),
        use_pos: bool = True,
        act: Union[str, Callable, None] = "relu",
        act_kwargs: Optional[Dict[str, Any]] = None,
        act_first: bool = False,
        norm: Union[str, Callable, None] = "batch_norm",
        norm_kwargs: Optional[Dict[str, Any]] = None,
        bias: bool = True,
        downsample: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.channels = channels
        self.weight_channels = weight_channels
        self.density_channels = density_channels
        self.use_pos = use_pos
        self.num_neighbors = num_neighbors
        self.bandwidth = bandwidth
        self.act = act
        self.act_kwargs = act_kwargs
        self.act_first = act_first
        self.norm = norm
        self.norm_kwargs = norm_kwargs
        self.bias = bias
        self.downsample = downsample

        self.conv = self.configure_conv()
        self.fc = LinearBlock(
            in_channels=self.out_channels * 16,
            out_channels=self.out_channels,
            act=self.act,
            act_kwargs=self.act_kwargs,
            act_first=self.act_first,
            norm=self.norm,
            norm_kwargs=self.norm_kwargs,
            bias=self.bias,
        )

    @property
    def out_channels(self) -> int:
        return self.channels[-1]

    def configure_conv(self) -> nn.Module:
        kwargs: Dict[str, Any] = dict(
            act=self.act,
            act_kwargs=self.act_kwargs,
            act_first=self.act_first,
            norm=self.norm,
            norm_kwargs=self.norm_kwargs,
            bias=self.bias,
            plain_last=False,
        )

        in_channels = self.in_channels + 3 if self.use_pos else self.in_channels
        local_nn = MLP([in_channels] + ensure_list(self.channels), **kwargs)
        weight_nn = MLP([3] + ensure_list(self.weight_channels) + [16], **kwargs)
        density_nn: Optional[nn.Module] = None
        if self.density_channels is not None:
            density_nn = MLP([1] + ensure_list(self.density_channels) + [1], **kwargs)

        return PointConv(
            local_nn=local_nn,
            weight_nn=weight_nn,
            density_nn=density_nn,
        )

    def forward(self, x: OptTensor, pos: Tensor, batch: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        density = gaussian_kernel_density(pos, batch, self.bandwidth) if self.conv.density_nn is not None else None

        x_dst, pos_dst, batch_dst = x, pos, batch
        if self.downsample is not None:
            idx = self.downsample(pos, batch)
            x_dst, pos_dst, batch_dst = x[idx], pos[idx], batch[idx]

        row, col = knn(pos, pos_dst, k=self.num_neighbors, batch_x=batch, batch_y=batch_dst)
        edge_index = torch.stack([col, row], dim=0)

        msg = self.conv(
            x=(x, x_dst),
            pos=(pos, pos_dst),
            edge_index=edge_index,
            density=(density.view(-1, 1), density[idx].view(-1, 1)) if density is not None else None,
        )

        out = self.fc(msg)
        return out, pos_dst, batch_dst
