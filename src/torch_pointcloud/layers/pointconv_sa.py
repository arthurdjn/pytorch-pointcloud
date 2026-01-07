from typing import TYPE_CHECKING, Any, Callable, Dict, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.nn import MLP, global_max_pool
from torch_geometric.typing import OptTensor

from torch_pointcloud.models.pointmlp import LinearBlock
from torch_pointcloud.utils.conversion import ensure_list
from torch_pointcloud.utils.imports import optional_import
from torch_pointcloud.utils.neighbors import gaussian_kernel_density

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
        self.in_channels = in_channels
        self.channels = channels
        self.weight_channels = weight_channels
        self.expansion = expansion
        self.num_neighbors = num_neighbors
        self.act = act
        self.act_kwargs = act_kwargs
        self.act_first = act_first
        self.norm = norm
        self.norm_kwargs = norm_kwargs
        self.bias = bias
        self.spatial_dim = spatial_dim
        self.downsample = downsample

        self.conv = self.configure_conv()
        self.fc = LinearBlock(
            in_channels=self.channels[-1] * self.expansion,
            out_channels=self.channels[-1],
            act=self.act,
            act_kwargs=self.act_kwargs,
            act_first=self.act_first,
            norm=self.norm,
            norm_kwargs=self.norm_kwargs,
            bias=self.bias,
        )

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

        in_channels = self.in_channels + self.spatial_dim
        local_nn = MLP([in_channels] + ensure_list(self.channels), **kwargs)
        weight_nn = MLP([self.spatial_dim] + ensure_list(self.weight_channels) + [self.expansion], **kwargs)
        return PointConv(local_nn=local_nn, weight_nn=weight_nn)

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
        self.in_channels = in_channels
        self.channels = channels
        self.weight_channels = weight_channels
        self.density_channels = density_channels
        self.expansion = expansion
        self.num_neighbors = num_neighbors
        self.bandwidth = bandwidth
        self.act = act
        self.act_kwargs = act_kwargs
        self.act_first = act_first
        self.norm = norm
        self.norm_kwargs = norm_kwargs
        self.bias = bias
        self.spatial_dim = spatial_dim
        self.downsample = downsample

        self.conv = self.configure_conv()
        self.fc = LinearBlock(
            in_channels=self.channels[-1] * self.expansion,
            out_channels=self.channels[-1],
            act=self.act,
            act_kwargs=self.act_kwargs,
            act_first=self.act_first,
            norm=self.norm,
            norm_kwargs=self.norm_kwargs,
            bias=self.bias,
        )

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

        in_channels = self.in_channels + self.spatial_dim
        local_nn = MLP([in_channels] + ensure_list(self.channels), **kwargs)
        weight_nn = MLP([self.spatial_dim] + ensure_list(self.weight_channels) + [self.expansion], **kwargs)
        density_nn = MLP([1] + ensure_list(self.density_channels) + [1], **kwargs)
        return PointConvDensity(
            local_nn=local_nn,
            weight_nn=weight_nn,
            density_nn=density_nn,
        )

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
        self.in_channels = in_channels
        self.channels = channels
        self.weight_channels = weight_channels
        self.expansion = expansion
        self.act = act
        self.act_kwargs = act_kwargs
        self.act_first = act_first
        self.norm = norm
        self.norm_kwargs = norm_kwargs
        self.bias = bias
        self.spatial_dim = spatial_dim

        self.conv = self.configure_conv()
        self.pool = create_pool(aggr)
        self.fc = LinearBlock(
            in_channels=self.out_channels * self.expansion,
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

        in_channels = self.in_channels + self.spatial_dim
        local_nn = MLP([in_channels] + ensure_list(self.channels), **kwargs)
        weight_nn = MLP([self.spatial_dim] + ensure_list(self.weight_channels) + [self.expansion], **kwargs)
        return PointConv(local_nn=local_nn, weight_nn=weight_nn)

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
        aggr: PoolLike = "mean",
        spatial_dim: int = 3,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.channels = channels
        self.weight_channels = weight_channels
        self.density_channels = density_channels
        self.expansion = expansion
        self.bandwidth = bandwidth
        self.act = act
        self.act_kwargs = act_kwargs
        self.act_first = act_first
        self.norm = norm
        self.norm_kwargs = norm_kwargs
        self.bias = bias
        self.spatial_dim = spatial_dim

        self.conv = self.configure_conv()
        self.pool = create_pool(aggr)
        self.fc = LinearBlock(
            in_channels=self.channels[-1] * self.expansion,
            out_channels=self.channels[-1],
            act=self.act,
            act_kwargs=self.act_kwargs,
            act_first=self.act_first,
            norm=self.norm,
            norm_kwargs=self.norm_kwargs,
            bias=self.bias,
        )

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

        in_channels = self.in_channels + self.spatial_dim
        local_nn = MLP([in_channels] + ensure_list(self.channels), **kwargs)
        weight_nn = MLP([self.spatial_dim] + ensure_list(self.weight_channels) + [self.expansion], **kwargs)
        density_nn = MLP([1] + ensure_list(self.density_channels) + [1], **kwargs)
        return PointConvDensity(
            local_nn=local_nn,
            weight_nn=weight_nn,
            density_nn=density_nn,
        )

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
