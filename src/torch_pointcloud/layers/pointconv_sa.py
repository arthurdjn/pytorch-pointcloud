from typing import TYPE_CHECKING, Any, Callable, Dict, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.nn import MLP, global_max_pool
from torch_geometric.typing import OptTensor

from torch_pointcloud.utils.conversion import ensure_list
from torch_pointcloud.utils.imports import optional_import
from torch_pointcloud.utils.neighbors import gaussian_kernel_density

from .linear_blocks import LinearBlock
from .pointconv import PointConv, PointConvDensity
from .pools import PoolLike, create_pool

if TYPE_CHECKING:
    from torch_cluster import knn


knn, _ = optional_import("torch_cluster", "knn")

# ! IMPORTANT: In original PointConv, the density net has a sigmoid activation function in the last layer,
# ! but is never called due to a bug in the code. For reproducibility, this behavior is replicated here.


class PointConvSetAbstraction(nn.Module):
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
