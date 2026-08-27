"""PVCNN building blocks: voxelization, squeeze-and-excitation, and point-voxel convolution."""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.nn import MLP
from torch_geometric.utils import scatter

from torch_pointcloud.layers.act import create_act
from torch_pointcloud.layers.conv3d_blocks import Conv3dBlock
from torch_pointcloud.utils.voxelization import dense_voxelize, trilinear_dense_devoxelize


class Voxelization(nn.Module):
    r"""Averages packed point features into a dense voxel grid.

    Positions are centered per cloud and, when `normalize` is set, rescaled by the cloud's largest radius so
    every cloud fills the grid. The continuous grid coordinates are returned alongside the voxels, for
    trilinear devoxelization.

    Shape:
        Input: $(N, C)$ features, $(N, 3)$ positions, $(N,)$ batch index
        Output: $(B, C, R, R, R)$ voxels, $(N, 3)$ grid coordinates

    Args:
        resolution: Grid resolution $R$ along each axis.
        normalize: Whether to rescale each cloud to the unit grid before voxelizing.
    """

    def __init__(self, resolution: int, normalize: bool = True):
        super().__init__()
        self.resolution = resolution
        self.normalize = normalize

    def forward(self, x: Tensor, pos: Tensor, batch: Tensor) -> Tuple[Tensor, Tensor]:
        batch_size = batch.max().item() + 1

        pos_mean = scatter(pos, batch, dim=0, reduce="mean", dim_size=batch_size)
        pos_centered = pos - pos_mean[batch]

        if self.normalize:
            pos_norm = torch.norm(pos_centered, dim=1, keepdim=True)
            pos_norm_max = scatter(pos_norm.squeeze(-1), batch, dim=0, reduce="max", dim_size=batch_size)
            pos_norm_max = pos_norm_max[batch].unsqueeze(1)
            pos_grid = pos_centered / (pos_norm_max * 2.0 + 1e-6) + 0.5
        else:
            pos_grid = (pos_centered + 1) / 2.0

        pos_grid = torch.clamp(pos_grid * self.resolution, 0, self.resolution - 1)
        pos_voxel = torch.round(pos_grid)

        voxel_features = dense_voxelize(x, pos_voxel, batch, self.resolution, reduce="mean")
        return voxel_features, pos_grid


class SE3d(nn.Module):
    r"""Squeeze-and-excitation gate for dense voxel grids.

    Global-average-pools each channel, passes it through a bottleneck MLP, and rescales the grid by the
    resulting per-channel gate.

    Shape:
        Input: $(B, C, R, R, R)$
        Output: $(B, C, R, R, R)$

    Args:
        channels: Number of channels $C$.
        reduction: Bottleneck ratio of the squeeze layer.
        act: Activation between the squeeze and excitation layers, name resolved by `create_act`.
        act_kwargs: Extra kwargs for the activation.
    """

    def __init__(
        self,
        channels: int,
        reduction: int = 8,
        act: Union[str, Callable, None] = "relu",
        act_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        act_kwargs = act_kwargs or {}

        self.squeeze = nn.Linear(channels, channels // reduction, bias=False)
        self.act = create_act(act, **act_kwargs) or nn.Identity()
        self.excitation = nn.Linear(channels // reduction, channels, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: Tensor) -> Tensor:
        B, C, *_ = x.shape  # (B, C, H, W, D)
        y = x.view(B, C, -1).mean(dim=2)  # (B, C)
        y = self.sigmoid(self.excitation(self.act(self.squeeze(y))))  # (B, C)
        y = y.view(B, C, 1, 1, 1)
        return x * y  # (B, C, H, W, D)


class PVConv(nn.Module):
    r"""Point-voxel convolution: a dense 3D conv branch summed with a per-point MLP branch.

    The voxel branch captures the neighborhood context at a coarse resolution and is devoxelized back with
    trilinear interpolation, while the point branch keeps the fine per-point detail.

    Shape:
        Input: $(N, C_\text{in})$ features, $(N, 3)$ positions, $(N,)$ batch index
        Output: $(N, C_\text{out})$

    Args:
        in_channels: Input channel count.
        out_channels: Output channel count.
        kernel_size: Kernel size of the voxel convolutions.
        resolution: Voxel grid resolution $R$ along each axis.
        use_se: Whether to gate the voxel branch with an `SE3d` block.
        normalize: Whether to rescale each cloud to the unit grid before voxelizing.
        act: Activation, name resolved by `create_act`. `None` disables.
        act_first: If `True`, run activation before normalization.
        act_kwargs: Extra kwargs for the activation.
        norm: Normalization, name resolved by `create_norm`. `None` disables.
        norm_kwargs: Extra kwargs for the normalization.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        resolution: int,
        use_se: bool = False,
        normalize: bool = True,
        act: Union[str, Callable, None] = "relu",
        act_first: bool = False,
        act_kwargs: Optional[Dict[str, Any]] = None,
        norm: Union[str, Callable, None] = "batch_norm",
        norm_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        self.resolution = resolution
        self.voxelization = Voxelization(resolution, normalize=normalize)

        voxel_layers: List[nn.Module] = [
            Conv3dBlock(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                act=act,
                act_first=act_first,
                act_kwargs=act_kwargs,
                norm=norm,
                norm_kwargs=norm_kwargs,
            ),
            Conv3dBlock(
                out_channels,
                out_channels,
                kernel_size=kernel_size,
                act=act,
                act_first=act_first,
                act_kwargs=act_kwargs,
                norm=norm,
                norm_kwargs=norm_kwargs,
            ),
        ]
        if use_se:
            voxel_layers.append(SE3d(out_channels, act=act, act_kwargs=act_kwargs))
        self.voxel_layers = nn.Sequential(*voxel_layers)

        self.mlp = MLP(
            [in_channels, out_channels],
            act=act,
            act_kwargs=act_kwargs,
            act_first=act_first,
            norm=norm,
            norm_kwargs=norm_kwargs,
            plain_last=False,
        )

    def forward(self, x: Tensor, pos: Tensor, batch: Tensor) -> Tensor:
        x_voxels, pos_grid = self.voxelization(x, pos, batch)
        x_voxels = self.voxel_layers(x_voxels)
        x_voxels = trilinear_dense_devoxelize(x_voxels, pos_grid, batch, self.resolution)
        return x_voxels + self.mlp(x)
