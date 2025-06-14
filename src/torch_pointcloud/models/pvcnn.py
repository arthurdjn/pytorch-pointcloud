from typing import Any, Callable, Dict, List, Literal, Optional, Sequence, Tuple, Union, overload

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn import MLP
from torch_geometric.nn.resolver import activation_resolver
from torch_geometric.utils import scatter

from torch_pointcloud.layers import PoolLike, create_cls_head, create_pool
from torch_pointcloud.utils.conversion import ensure_tuple, ensure_tuple_size


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
    def __init__(
        self,
        resolution: int,
        normalize: bool = True,
    ):
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


class PVConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        depth: int,
        kernel_size: int,
        resolution: int,
        with_se: bool = False,
        normalize: bool = True,
        act: Union[str, Callable, None] = "relu",
        act_kwargs: Optional[Dict[str, Any]] = None,
        act_first: bool = False,
        norm: Union[str, Callable, None] = "batch_norm",
        norm_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        kwargs = dict(
            act=act,
            act_kwargs=act_kwargs,
            act_first=act_first,
            norm=norm,
            norm_kwargs=norm_kwargs,
        )

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            if not resolution:
                # In case resolution is 0 or None, use a linear block
                layer = MLP([in_channels, out_channels], plain_last=False, **kwargs)
            else:
                layer = PVConv(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    resolution=resolution,
                    with_se=with_se,
                    normalize=normalize,
                    **kwargs,  # type: ignore[arg-type]
                )

            self.layers.append(layer)
            in_channels = out_channels

    @overload
    def forward(
        self,
        x: Tensor,
        pos: Tensor,
        batch: Tensor,
        return_intermediates: Literal[True],
    ) -> Tuple[Tensor, List[Tensor]]: ...

    @overload
    def forward(
        self,
        x: Tensor,
        pos: Tensor,
        batch: Tensor,
        return_intermediates: Literal[False] = False,
    ) -> Tensor: ...

    def forward(self, x: Tensor, pos: Tensor, batch: Tensor, return_intermediates: bool = False) -> Any:
        intermediates = []
        for layer in self.layers:
            x = layer(x) if isinstance(layer, MLP) else layer(x, pos, batch)
            if return_intermediates:
                intermediates.append(x)

        if return_intermediates:
            return x, intermediates
        return x


class PVCNNClassification(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        *,
        channels: Sequence[int],
        global_channels: Optional[Sequence[int]] = None,
        depths: Sequence[int],
        kernel_sizes: Sequence[int],
        resolutions: Sequence[int],
        with_se: bool = False,
        normalize: bool = True,
        dropout: float = 0.0,
        global_pool: PoolLike = "max",
        act: Union[str, Callable, None] = "relu",
        act_kwargs: Optional[Dict[str, Any]] = None,
        act_first: bool = False,
        norm: Union[str, Callable, None] = "batch_norm",
        norm_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes

        self.depths = ensure_tuple(depths)
        self.channels = ensure_tuple_size([in_channels] + list(channels), size=len(self.depths) + 1)
        self.global_channels = ensure_tuple(global_channels, none_as_empty=True)
        self.kernel_sizes = ensure_tuple_size(kernel_sizes, size=len(self.depths))
        self.resolutions = ensure_tuple_size(resolutions, size=len(self.depths))
        self.with_se = with_se
        self.normalize = normalize
        self.dropout = dropout
        self.act = act
        self.act_kwargs = act_kwargs
        self.act_first = act_first
        self.norm = norm
        self.norm_kwargs = norm_kwargs

        self.blocks = self.configure_blocks()
        self.global_mlp = self.configure_global_mlp()
        self.global_pool = create_pool(global_pool)
        self.head = create_cls_head(self.embedding_dim, self.num_classes)

    @property
    def embedding_dim(self) -> int:
        return self.channels[-1]

    def configure_blocks(self) -> nn.ModuleList:
        blocks = nn.ModuleList()
        for i in range(len(self.depths)):
            in_channels = self.channels[i]
            out_channels = self.channels[i + 1]
            block = PVConvBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                depth=self.depths[i],
                kernel_size=self.kernel_sizes[i],
                resolution=self.resolutions[i],
                with_se=self.with_se,
                normalize=self.normalize,
                act=self.act,
                act_kwargs=self.act_kwargs,
                act_first=self.act_first,
                norm=self.norm,
                norm_kwargs=self.norm_kwargs,
            )
            blocks.append(block)
        return blocks

    def configure_global_mlp(self) -> Optional[MLP]:
        if not self.global_channels:
            return None

        return MLP(
            [self.channels[-1]] + list(self.global_channels),
            act=self.act,
            act_kwargs=self.act_kwargs,
            act_first=self.act_first,
            norm=self.norm,
            norm_kwargs=self.norm_kwargs,
            plain_last=False,
        )

    @overload
    def forward_features(
        self,
        x: Tensor,
        pos: Tensor,
        batch: Tensor,
        return_intermediates: Literal[True],
    ) -> Tuple[Tensor, List[Tensor]]: ...

    @overload
    def forward_features(
        self,
        x: Tensor,
        pos: Tensor,
        batch: Tensor,
        return_intermediates: Literal[False] = False,
    ) -> Tensor: ...

    def forward_features(self, x: Tensor, pos: Tensor, batch: Tensor, return_intermediates: bool = False) -> Any:
        intermediates = []
        for block in self.blocks:
            x = block(x, pos, batch, return_intermediates=return_intermediates)
            if return_intermediates:
                x, x_inters = x
                intermediates.extend(x_inters)

        if return_intermediates:
            return x, intermediates
        return x

    def forward_head(self, x: Tensor, batch: Tensor, pre_logits: bool = False) -> Tensor:
        x = self.global_pool(x, batch)
        if self.dropout:
            x = F.dropout(x, p=self.dropout, training=self.training)
        return x if pre_logits else self.head(x)

    def forward(self, x: Tensor, pos: Tensor, batch: Tensor) -> Tensor:
        x = self.forward_features(x, pos, batch)
        x = self.forward_head(x, batch)
        return x


class PVCNNSegmentation(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        *,
        channels: Sequence[int],
        global_channels: Optional[Sequence[int]] = None,
        depths: Sequence[int],
        kernel_sizes: Sequence[int],
        resolutions: Sequence[int],
        spatial_dim: int = 3,
        with_se: bool = False,
        normalize: bool = True,
        dropout: float = 0.0,
        global_pool: PoolLike = "max",
        act: Union[str, Callable, None] = "relu",
        act_kwargs: Optional[Dict[str, Any]] = None,
        act_first: bool = False,
        norm: Union[str, Callable, None] = "batch_norm",
        norm_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        self.in_channels = max(in_channels, spatial_dim)
        self.num_classes = num_classes

        self.depths = ensure_tuple(depths)
        self.channels = ensure_tuple_size([self.in_channels] + list(channels), size=len(self.depths) + 1)
        self.global_channels = ensure_tuple(global_channels, none_as_empty=True)
        self.kernel_sizes = ensure_tuple_size(kernel_sizes, size=len(self.depths))
        self.resolutions = ensure_tuple_size(resolutions, size=len(self.depths))
        self.with_se = with_se
        self.normalize = normalize
        self.dropout = dropout
        self.act = act
        self.act_kwargs = act_kwargs
        self.act_first = act_first
        self.norm = norm
        self.norm_kwargs = norm_kwargs

        self.blocks = self.configure_blocks()
        self.global_mlp = self.configure_global_mlp()
        self.global_pool = create_pool(global_pool)
        self.head = create_cls_head(self.embedding_dim, self.num_classes)

    @property
    def embedding_dim(self) -> int:
        embedding_dim = sum(channels * depth for channels, depth in zip(self.channels[1:], self.depths))
        if self.global_channels:
            return embedding_dim + self.global_channels[-1]
        return embedding_dim + self.channels[-1]

    def configure_blocks(self) -> nn.ModuleList:
        blocks = nn.ModuleList()
        for i in range(len(self.depths)):
            in_channels = self.channels[i]
            out_channels = self.channels[i + 1]
            block = PVConvBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                depth=self.depths[i],
                kernel_size=self.kernel_sizes[i],
                resolution=self.resolutions[i],
                with_se=self.with_se,
                normalize=self.normalize,
                act=self.act,
                act_kwargs=self.act_kwargs,
                act_first=self.act_first,
                norm=self.norm,
                norm_kwargs=self.norm_kwargs,
            )
            blocks.append(block)
        return blocks

    def configure_global_mlp(self) -> Optional[MLP]:
        if not self.global_channels:
            return None

        return MLP(
            [self.channels[-1]] + list(self.global_channels),
            act=self.act,
            act_kwargs=self.act_kwargs,
            act_first=self.act_first,
            norm=self.norm,
            norm_kwargs=self.norm_kwargs,
            plain_last=False,
        )

    @overload
    def forward_features(
        self,
        x: Tensor,
        pos: Tensor,
        batch: Tensor,
        return_intermediates: Literal[True],
    ) -> Tuple[Tensor, List[Tensor]]: ...

    @overload
    def forward_features(
        self,
        x: Tensor,
        pos: Tensor,
        batch: Tensor,
        return_intermediates: Literal[False] = False,
    ) -> Tensor: ...

    def forward_features(self, x: Tensor, pos: Tensor, batch: Tensor, return_intermediates: bool = False) -> Any:
        intermediates = []
        for block in self.blocks:
            x = block(x, pos, batch, return_intermediates=return_intermediates)
            if return_intermediates:
                x, x_inters = x
                intermediates.extend(x_inters)

        if return_intermediates:
            return x, intermediates
        return x

    # NOTE: Does it make sense to have this forward_decoder?
    # NOTE: Maybe we should raise an error/warning specifying that this method
    # NOTE: is irrelevant for this model. ...And include the global_mlp in the head instead?
    def forward_decoder(self, x: Tensor, batch: Tensor, intermediates: List[Tensor]) -> Tensor:
        x_global = self.global_pool(x, batch)
        if self.global_mlp:
            x_global = self.global_mlp(x_global)

        intermediates.append(x_global[batch])
        return torch.cat(intermediates, dim=1)

    def forward_head(self, x: Tensor, pre_logits: bool = False) -> Tensor:
        if self.dropout:
            x = F.dropout(x, p=self.dropout, training=self.training)
        return x if pre_logits else self.head(x)

    def forward(self, x: Tensor, pos: Tensor, batch: Tensor) -> Tensor:
        x, intermediates = self.forward_features(x, pos, batch, return_intermediates=True)
        x = self.forward_decoder(x, batch, intermediates)
        x = self.forward_head(x)
        return x
