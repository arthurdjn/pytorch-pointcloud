from typing import Any, Callable, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.nn import MLP
from torch_geometric.nn.resolver import activation_resolver
from torch_geometric.utils import scatter


def avg_voxelize(x: Tensor, pos: Tensor, batch: Tensor, resolution: int) -> Tensor:
    R = resolution
    if pos.shape[1] != 3:
        raise ValueError(f"Position tensor must be 3D, but got a {pos.shape[1]}-D tensor.")

    _, C = x.shape
    B = batch.max().item() + 1

    # Ensure coordinates are integers and within bounds
    coords = pos.long().clamp(0, R - 1)

    # Create unique voxel indices for clustering
    # Each voxel gets a unique ID based on (batch, x, y, z)
    linear_voxel_idx = coords[:, 2] * (R * R) + coords[:, 1] * R + coords[:, 0]  # z*R² + y*R + x
    batch_offset = batch * (R * R * R)
    voxel_idx = linear_voxel_idx + batch_offset
    # Use scatter to pool features
    max_voxel_idx = B * R * R * R
    x_pooled = scatter(x, voxel_idx, dim=0, reduce="mean", dim_size=max_voxel_idx)
    x_pooled = x_pooled.view(B, R, R, R, C)  # (B, z, y, x, C)
    return x_pooled.permute(0, 4, 3, 2, 1)  # (B, C, x, y, z)


# Adapted from: https://github.com/mit-han-lab/pvcnn/blob/master/modules/functional/src/interpolate/trilinear_devox.cu
def trilinear_devoxelize(x_voxels: Tensor, pos: Tensor, batch: Tensor, resolution: int) -> Tensor:
    device = pos.device
    N, _ = pos.shape
    B, C, R, R1, R2 = x_voxels.shape
    # Operation can fails if the tensors are not all contiguous
    x_voxels = x_voxels.contiguous()
    pos = pos.contiguous()
    batch = batch.contiguous()

    # Sanity checks
    if resolution != R or resolution != R1 or resolution != R2:
        raise ValueError(
            f"Resolution {resolution} must be equal to the voxel grid resolution. "
            f"Got ({R}, {R1}, {R2}) but expected ({resolution}, {resolution}, {resolution})."
        )
    if pos.shape[1] != 3:
        raise ValueError(f"Position tensor must be 3D, but got a {pos.shape[1]}-D tensor.")

    # Ensure coordinates are within bounds [0, R-1]
    pos_clamped = pos.clamp(0, R - 1)

    # Compute floor coordinates and fractional parts
    pos_floor = torch.floor(pos_clamped)
    pos_frac = pos_clamped - pos_floor

    # Convert to integer coordinates
    coords_lo = pos_floor.long()  # (N, 3)
    coords_hi = torch.minimum(coords_lo + 1, torch.tensor(R - 1, device=device))  # (N, 3)

    # Compute interpolation weights for each dimension
    d_0 = pos_frac  # (N, 3) - distance from lower corner
    d_1 = 1.0 - d_0  # (N, 3) - distance from upper corner

    # Create corner offsets representing all binary combinations
    corner_offsets = torch.tensor(
        [
            [0, 0, 0],
            [0, 0, 1],
            [0, 1, 0],
            [0, 1, 1],
            [1, 0, 0],
            [1, 0, 1],
            [1, 1, 0],
            [1, 1, 1],
        ],
        device=device,
    ).unsqueeze(0)  # (1, 8, 3)

    corner_offsets = corner_offsets.expand(N, -1, -1)  # (N, 8, 3)
    coords_lo_expanded = coords_lo.unsqueeze(1).expand(-1, 8, -1)  # (N, 8, 3)
    coords_hi_expanded = coords_hi.unsqueeze(1).expand(-1, 8, -1)  # (N, 8, 3)
    corners = torch.where(corner_offsets == 0, coords_lo_expanded, coords_hi_expanded)  # (N, 8, 3)

    # Compute weights for each corner
    d_0_expanded = d_0.unsqueeze(1).expand(-1, 8, -1)  # (N, 8, 3)
    d_1_expanded = d_1.unsqueeze(1).expand(-1, 8, -1)  # (N, 8, 3)
    weights_3d = torch.where(corner_offsets == 0, d_1_expanded, d_0_expanded)  # (N, 8, 3)

    # Compute trilinear weights (product across 3 dimensions)
    weights = weights_3d.prod(dim=2)  # (N, 8)

    batch_expanded = batch.unsqueeze(1).expand(-1, 8)  # (N, 8)
    linear_indices = corners[:, :, 2] * (R * R) + corners[:, :, 1] * R + corners[:, :, 0]  # (N, 8)
    global_indices = linear_indices + batch_expanded * (R * R * R)  # (N, 8)

    x_voxels_reordered = x_voxels.permute(0, 4, 3, 2, 1)  # (B, C, x, y, z) -> (B, z, y, x, C)
    x_voxels_flat = x_voxels_reordered.reshape(B * R * R * R, C)  # (B*R*R*R, C)

    # Gather features for all corners
    gathered_features = x_voxels_flat[global_indices.view(-1)]  # (N*8, C)
    gathered_features = gathered_features.view(N, 8, C)  # (N, 8, C)

    # Apply trilinear interpolation: weighted sum over 8 neighbors
    weights_expanded = weights.unsqueeze(2)  # (N, 8, 1)
    x_out = (gathered_features * weights_expanded).sum(dim=1)  # (N, C)

    return x_out


class Voxelization(nn.Module):
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
            pos_norm_max = scatter(pos_norm.squeeze(), batch, dim=0, reduce="max", dim_size=batch_size)
            pos_norm_max = pos_norm_max[batch].unsqueeze(1)
            norm_coords = pos_centered / (pos_norm_max * 2.0 + 1e-6) + 0.5
        else:
            norm_coords = (pos_centered + 1) / 2.0

        norm_coords = torch.clamp(norm_coords * self.resolution, 0, self.resolution - 1)
        vox_coords = torch.round(norm_coords)

        voxel_features = avg_voxelize(x, vox_coords, batch, self.resolution)
        return voxel_features, norm_coords


class SE3d(nn.Module):
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
        self.act = activation_resolver(act, **act_kwargs) or nn.Identity()
        self.excitation = nn.Linear(channels // reduction, channels, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: Tensor) -> Tensor:
        B, C, *_ = x.shape  # (B, C, H, W, D)
        y = x.view(B, C, -1).mean(dim=2)  # (B, C)
        y = self.sigmoid(self.excitation(self.act(self.squeeze(y))))  # (B, C)
        y = y.view(B, C, 1, 1, 1)
        return x * y  # (B, C, H, W, D)


class PVConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        resolution: int,
        with_se: bool = False,
        normalize: bool = True,
        act: Union[str, Callable, None] = "relu",
        act_first: bool = False,
        act_kwargs: Optional[Dict[str, Any]] = None,
        norm: Union[str, Callable, None] = "batch_norm",
        norm_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        act_kwargs = act_kwargs or {}
        norm_kwargs = norm_kwargs or {}
        self.resolution = resolution
        self.voxelization = Voxelization(resolution, normalize=normalize)

        if act_first:
            voxel_layers = [
                nn.Conv3d(in_channels, out_channels, kernel_size, stride=1, padding=kernel_size // 2),
                activation_resolver(act, **act_kwargs) or nn.Identity(),
                nn.BatchNorm3d(out_channels),
                nn.Conv3d(out_channels, out_channels, kernel_size, stride=1, padding=kernel_size // 2),
                activation_resolver(act, **act_kwargs) or nn.Identity(),
                nn.BatchNorm3d(out_channels),
            ]
        else:
            voxel_layers = [
                nn.Conv3d(in_channels, out_channels, kernel_size, stride=1, padding=kernel_size // 2),
                nn.BatchNorm3d(out_channels),
                activation_resolver(act, **act_kwargs) or nn.Identity(),
                nn.Conv3d(out_channels, out_channels, kernel_size, stride=1, padding=kernel_size // 2),
                nn.BatchNorm3d(out_channels),
                activation_resolver(act, **act_kwargs) or nn.Identity(),
            ]
        if with_se:
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
        # Voxelize the input point cloud, resulting in a voxelized feature map and voxel coordinates
        # x_voxels: (B, C, R, R, R) - voxel_coords: (N, 3)
        x_voxels, voxel_coords = self.voxelization(x, pos, batch)
        x_voxels = self.voxel_layers(x_voxels)  # (B, C, R, R, R)
        # Devoxelize the features back to the "packed" representation
        x_voxels = trilinear_devoxelize(x_voxels, voxel_coords, batch, self.resolution)  # (N, C)
        return x_voxels + self.mlp(x)  # (N, C)
