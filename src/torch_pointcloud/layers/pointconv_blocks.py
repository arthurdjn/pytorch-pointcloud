"""PointConv message-passing convolutions and the set-abstraction blocks built on them."""

from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.nn import MLP, MessagePassing, global_max_pool
from torch_geometric.typing import Adj, OptTensor, PairOptTensor, PairTensor, SparseTensor, torch_sparse
from torch_geometric.utils import add_self_loops, remove_self_loops, scatter
from typing_extensions import Unpack

from torch_pointcloud.ops.cluster import knn
from torch_pointcloud.ops.density import gaussian_kernel_density
from torch_pointcloud.utils.conversion import ensure_list
from torch_pointcloud.utils.types import MessagePassingParams

from .linear_blocks import LinearBlock
from .pools import PoolLike, create_pool


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

        max_density = scatter(density_j, index, dim=0, reduce="max")
        density_rel = density_j / (max_density[index])
        density_scale = self.density_nn(density_rel)
        h = h * density_scale

        msg = h.unsqueeze(-1) * w.unsqueeze(-2)
        return msg.view(msg.size(0), -1)

    def extra_repr(self) -> str:
        return f"local_nn={self.local_nn}, weight_nn={self.weight_nn}, density_nn={self.density_nn}"

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.extra_repr()})"


# ! IMPORTANT: In original PointConv, the density net has a sigmoid activation function in the last layer,
# ! but is never called due to a bug in the code. For reproducibility, this behavior is replicated here.


class PointConvSetAbstraction(nn.Module):
    r"""PointConv set-abstraction block: optional downsampling, $k$-NN grouping, and a
    weight-net continuous convolution.

    Args:
        in_channels: Number of input feature channels.
        num_neighbors: Number of neighbors gathered per output point.
        channels: Per-layer channel sizes of the feature MLP.
        weight_channels: Hidden channel sizes of the weight net applied to relative positions.
        expansion: Channel expansion factor of the final matrix multiplication.
        spatial_dim: Dimension of point coordinates.
        downsample: Optional module returning the sampled indices (e.g. `FPS`). If `None`, the
            resolution is unchanged.
    """

    def __init__(
        self,
        in_channels: int,
        num_neighbors: int,
        channels: Sequence[int],
        weight_channels: Sequence[int] = (8, 8),
        expansion: int = 16,
        act: Union[str, Callable, None] = "relu",
        act_kwargs: Optional[Dict[str, Any]] = None,
        act_first: bool = False,
        norm: Union[str, Callable, None] = "batch_norm",
        norm_kwargs: Optional[Dict[str, Any]] = None,
        bias: bool = True,
        spatial_dim: int = 3,
        downsample: Optional[nn.Module] = None,
    ):
        super().__init__()
        # Common parameters for MLP blocks
        kwargs: Dict[str, Any] = dict(
            act=act,
            act_kwargs=act_kwargs,
            act_first=act_first,
            norm=norm,
            norm_kwargs=norm_kwargs,
            bias=bias,
        )

        local_nn = MLP([in_channels + spatial_dim] + ensure_list(channels), plain_last=False, **kwargs)
        weight_nn = MLP([spatial_dim] + ensure_list(weight_channels) + [expansion], plain_last=False, **kwargs)

        self.num_neighbors = num_neighbors
        self.downsample = downsample
        self.conv = PointConv(local_nn=local_nn, weight_nn=weight_nn)
        self.fc = LinearBlock(in_channels=channels[-1] * expansion, out_channels=channels[-1], **kwargs)

    def forward(self, x: OptTensor, pos: Tensor, batch: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        x_dst, pos_dst, batch_dst = x, pos, batch
        if self.downsample is not None:
            idx = self.downsample(pos, batch)
            x_dst, pos_dst, batch_dst = x[idx], pos[idx], batch[idx]

        row, col = knn(pos, pos_dst, k=self.num_neighbors, batch_x=batch, batch_y=batch_dst)
        edge_index = torch.stack([col, row], dim=0)
        msg = self.conv(x=(x, x_dst), pos=(pos, pos_dst), edge_index=edge_index)

        x_dst = self.fc(msg)
        return x_dst, pos_dst, batch_dst


class PointConvDensitySetAbstraction(nn.Module):
    r"""PointConv set-abstraction block with inverse-density re-weighting.

    Same layout as `PointConvSetAbstraction`, with a Gaussian kernel density estimated per point;
    the inverse density is transformed by a density net and re-weights the grouped features.

    Args:
        in_channels: Number of input feature channels.
        num_neighbors: Number of neighbors gathered per output point.
        channels: Per-layer channel sizes of the feature MLP.
        bandwidth: Bandwidth of the Gaussian kernel density estimate.
        weight_channels: Hidden channel sizes of the weight net applied to relative positions.
        density_channels: Hidden channel sizes of the density net.
        expansion: Channel expansion factor of the final matrix multiplication.
        spatial_dim: Dimension of point coordinates.
        downsample: Optional module returning the sampled indices (e.g. `FPS`). If `None`, the
            resolution is unchanged.
    """

    def __init__(
        self,
        in_channels: int,
        num_neighbors: int,
        channels: Sequence[int],
        bandwidth: float = 1.0,
        weight_channels: Sequence[int] = (8, 8),
        density_channels: Sequence[int] = (16, 8),
        expansion: int = 16,
        act: Union[str, Callable, None] = "relu",
        act_kwargs: Optional[Dict[str, Any]] = None,
        act_first: bool = False,
        norm: Union[str, Callable, None] = "batch_norm",
        norm_kwargs: Optional[Dict[str, Any]] = None,
        bias: bool = True,
        spatial_dim: int = 3,
        downsample: Optional[nn.Module] = None,
    ):
        super().__init__()
        # Common parameters for MLP blocks
        kwargs: Dict[str, Any] = dict(
            act=act,
            act_kwargs=act_kwargs,
            act_first=act_first,
            norm=norm,
            norm_kwargs=norm_kwargs,
            bias=bias,
        )

        local_nn = MLP([in_channels + spatial_dim] + ensure_list(channels), plain_last=False, **kwargs)
        weight_nn = MLP([spatial_dim] + ensure_list(weight_channels) + [expansion], plain_last=False, **kwargs)
        density_nn = MLP([1] + ensure_list(density_channels) + [1], plain_last=False, **kwargs)

        self.num_neighbors = num_neighbors
        self.bandwidth = bandwidth
        self.downsample = downsample
        self.conv = PointConvDensity(local_nn=local_nn, weight_nn=weight_nn, density_nn=density_nn)
        self.fc = LinearBlock(in_channels=channels[-1] * expansion, out_channels=channels[-1], **kwargs)

    def forward(self, x: OptTensor, pos: Tensor, batch: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        kde = gaussian_kernel_density(pos, batch, self.bandwidth)
        density = 1.0 / kde

        x_dst, pos_dst, batch_dst, density_dst = x, pos, batch, density
        if self.downsample is not None:
            idx = self.downsample(pos, batch)
            x_dst, pos_dst, batch_dst = x[idx], pos[idx], batch[idx]
            density_dst = density[idx]

        row, col = knn(pos, pos_dst, k=self.num_neighbors, batch_x=batch, batch_y=batch_dst)
        edge_index = torch.stack([col, row], dim=0)
        msg = self.conv(
            x=(x, x_dst),
            pos=(pos, pos_dst),
            edge_index=edge_index,
            density=(density.view(-1, 1), density_dst.view(-1, 1)),
        )

        x_dst = self.fc(msg)
        return x_dst, pos_dst, batch_dst


class PointConvGlobalSetAbstraction(nn.Module):
    r"""Global PointConv set-abstraction block: one weight-net convolution over each whole sample.

    Args:
        in_channels: Number of input feature channels.
        channels: Per-layer channel sizes of the feature MLP.
        weight_channels: Hidden channel sizes of the weight net applied to relative positions.
        expansion: Channel expansion factor of the final matrix multiplication.
        aggr: Pooling used to place the single output position of each sample.
        spatial_dim: Dimension of point coordinates.
    """

    def __init__(
        self,
        in_channels: int,
        channels: Sequence[int],
        weight_channels: Sequence[int] = (8, 8),
        expansion: int = 16,
        act: Union[str, Callable, None] = "relu",
        act_kwargs: Optional[Dict[str, Any]] = None,
        act_first: bool = False,
        norm: Union[str, Callable, None] = "batch_norm",
        norm_kwargs: Optional[Dict[str, Any]] = None,
        bias: bool = True,
        aggr: PoolLike = "mean",
        spatial_dim: int = 3,
    ):
        super().__init__()
        # Common parameters for MLP blocks
        kwargs: Dict[str, Any] = dict(
            act=act,
            act_kwargs=act_kwargs,
            act_first=act_first,
            norm=norm,
            norm_kwargs=norm_kwargs,
            bias=bias,
        )

        local_nn = MLP([in_channels + spatial_dim] + ensure_list(channels), plain_last=False, **kwargs)
        weight_nn = MLP([spatial_dim] + ensure_list(weight_channels) + [expansion], plain_last=False, **kwargs)

        self.pool = create_pool(aggr)
        self.conv = PointConv(local_nn=local_nn, weight_nn=weight_nn)
        self.fc = LinearBlock(in_channels=channels[-1] * expansion, out_channels=channels[-1], **kwargs)

    def forward(self, x: OptTensor, pos: Tensor, batch: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        pos_dst = self.pool(pos, batch)
        batch_dst = torch.arange(pos_dst.size(0), device=batch.device)

        row = batch
        col = torch.arange(pos.size(0), device=batch.device)
        edge_index = torch.stack([col, row], dim=0)
        msg = self.conv(x=(x, None), pos=(pos, pos_dst), edge_index=edge_index)

        x_dst = self.fc(msg)
        return x_dst, pos_dst, batch_dst


class PointConvDensityGlobalSetAbstraction(nn.Module):
    r"""Global PointConv set-abstraction block with inverse-density re-weighting.

    Args:
        in_channels: Number of input feature channels.
        channels: Per-layer channel sizes of the feature MLP.
        bandwidth: Bandwidth of the Gaussian kernel density estimate.
        weight_channels: Hidden channel sizes of the weight net applied to relative positions.
        density_channels: Hidden channel sizes of the density net.
        expansion: Channel expansion factor of the final matrix multiplication.
        pool: Pooling used to place the single output position of each sample.
        spatial_dim: Dimension of point coordinates.
    """

    def __init__(
        self,
        in_channels: int,
        channels: Sequence[int],
        bandwidth: float,
        weight_channels: Sequence[int] = (8, 8),
        density_channels: Sequence[int] = (16, 8),
        expansion: int = 16,
        act: Union[str, Callable, None] = "relu",
        act_kwargs: Optional[Dict[str, Any]] = None,
        act_first: bool = False,
        norm: Union[str, Callable, None] = "batch_norm",
        norm_kwargs: Optional[Dict[str, Any]] = None,
        bias: bool = True,
        pool: PoolLike = "mean",
        spatial_dim: int = 3,
    ):
        super().__init__()
        # Common parameters for MLP blocks
        kwargs: Dict[str, Any] = dict(
            act=act,
            act_kwargs=act_kwargs,
            act_first=act_first,
            norm=norm,
            norm_kwargs=norm_kwargs,
            bias=bias,
        )

        local_nn = MLP([in_channels + spatial_dim] + ensure_list(channels), plain_last=False, **kwargs)
        weight_nn = MLP([spatial_dim] + ensure_list(weight_channels) + [expansion], plain_last=False, **kwargs)
        density_nn = MLP([1] + ensure_list(density_channels) + [1], plain_last=False, **kwargs)

        self.bandwidth = bandwidth
        self.pool = create_pool(pool)
        self.conv = PointConvDensity(local_nn=local_nn, weight_nn=weight_nn, density_nn=density_nn)
        self.fc = LinearBlock(in_channels=channels[-1] * expansion, out_channels=channels[-1], **kwargs)

    def forward(self, x: OptTensor, pos: Tensor, batch: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        kde = gaussian_kernel_density(pos, batch, self.bandwidth)
        density = 1.0 / kde
        density_dst = global_max_pool(density.view(-1, 1), batch).squeeze(-1)

        pos_dst = self.pool(pos, batch)
        batch_dst = torch.arange(pos_dst.size(0), device=batch.device)

        row = batch
        col = torch.arange(pos.size(0), device=batch.device)
        edge_index = torch.stack([col, row], dim=0)
        msg = self.conv(
            x=(x, None),
            pos=(pos, pos_dst),
            edge_index=edge_index,
            density=(density.view(-1, 1), density_dst.view(-1, 1)),
        )

        x_dst = self.fc(msg)
        return x_dst, pos_dst, batch_dst
