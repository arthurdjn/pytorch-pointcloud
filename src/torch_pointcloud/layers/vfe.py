r"""Voxel feature encoders: the pillar feature-net layer and the dynamic mean and max voxel encoders.

The dynamic mean encoder is a packed-format port of the `DynamicVoxelVFE` from
:github: [gwenzhang/Voxel-Mamba](https://github.com/gwenzhang/Voxel-Mamba), the dynamic max encoder of the
`cylinder_fea` point-voxel encoder of :github: [xinge008/Cylinder3D](https://github.com/xinge008/Cylinder3D).
"""

from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.nn import MLP
from torch_geometric.utils import scatter

from torch_pointcloud.utils.types import OptTensor

from .norms import create_norm


class PillarFeatureLayer(nn.Module):
    r"""Pillar feature-net layer: linear + norm + act on every point, then a max-pool over each pillar.

    Non-final layers halve their output width and concatenate the pooled feature back onto every point of the
    pillar; the final layer returns the pooled per-pillar feature. Padded pillars come as a dense $(P, N, C)$
    tensor pooled over the point axis; dynamically voxelized points come packed as $(N, C)$ with their
    `pillar_index`, and are pooled by index.

    Args:
        in_channels: Input feature channels.
        out_channels: Target output channels (halved internally for non-final layers).
        last: Whether this is the final layer.
        act: Activation type or callable.
        act_kwargs: Extra activation arguments.
        norm: Normalization type or callable.
        norm_kwargs: Extra normalization arguments.

    Shape:
        - Input: $(P, N, C_\text{in})$ padded pillars, or $(N, C_\text{in})$ packed points with an $(N,)$ pillar index.
        - Output: the input layout with $C_\text{out}$ channels for non-final layers; $(P, 1, C_\text{out})$ or
          $(P, C_\text{out})$ pooled features for the final layer.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        last: bool,
        *,
        act: Union[str, Callable, None] = "relu",
        act_kwargs: Optional[Dict[str, Any]] = None,
        norm: Union[str, Callable, None] = "batch_norm",
        norm_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()
        self.last = last
        out_dim = out_channels if last else out_channels // 2
        self.mlp = MLP(
            [in_channels, out_dim],
            act=act,
            act_kwargs=act_kwargs,
            norm=norm,
            norm_kwargs=norm_kwargs,
            bias=False,
            plain_last=False,
        )

    def forward(self, x: Tensor, pillar_index: OptTensor = None) -> Tensor:
        if pillar_index is None:
            num_pillars, points_per_pillar, _ = x.shape
            # The per-point MLP normalizes each channel over every point of every pillar.
            x = self.mlp(x.reshape(num_pillars * points_per_pillar, -1)).reshape(num_pillars, points_per_pillar, -1)
            x_max = x.max(dim=1, keepdim=True).values
            if self.last:
                return x_max
            return torch.cat([x, x_max.expand_as(x)], dim=2)

        x = self.mlp(x)
        x_max = scatter(x, pillar_index, dim=0, reduce="max")
        if self.last:
            return x_max
        return torch.cat([x, x_max[pillar_index]], dim=1)


class DynamicMeanVFE(nn.Module):
    r"""Dynamic mean voxel feature encoder for the Voxel Mamba detector.

    Points are assigned to voxels on the fly (no fixed points-per-voxel), augmented with the
    per-voxel cluster-mean offset and voxel-center offset, then encoded by a stack of
    linear + norm + ReLU PFN layers whose per-voxel max-pool produces one feature vector per voxel.

    Reference implementation: :github:
    [gwenzhang/Voxel-Mamba](https://github.com/gwenzhang/Voxel-Mamba) (`DynamicVoxelVFE`).

    Args:
        in_channels: Raw point feature channels including xyz (e.g. $5$ for Waymo $x, y, z, \text{intensity}, \text{elongation}$).
        num_filters: Output width of each PFN layer; the last entry is the voxel feature dim.
        voxel_size: Voxel size $(v_x, v_y, v_z)$.
        point_cloud_range: Range $(x_\min, y_\min, z_\min, x_\max, y_\max, z_\max)$.
        grid_size: Voxel grid extent $(n_x, n_y, n_z)$.

    Shape:
        - Input: $(N, C_\text{in})$ point features and $(N,)$ batch index.
        - Output: $(M, C_\text{out})$ voxel features and $(M, 4)$ voxel coords.
    """

    voxel_size: Tensor
    point_cloud_range: Tensor
    grid_size: Tensor

    def __init__(
        self,
        in_channels: int,
        num_filters: Sequence[int],
        voxel_size: Sequence[float],
        point_cloud_range: Sequence[float],
        grid_size: Sequence[int],
    ) -> None:
        super().__init__()
        feat_channels = in_channels + 6
        widths = [feat_channels, *num_filters]
        self.pfn_layers = nn.ModuleList(
            PillarFeatureLayer(
                widths[i],
                widths[i + 1],
                last=i >= len(widths) - 2,
                norm_kwargs=dict(eps=1e-3, momentum=0.01),
            )
            for i in range(len(widths) - 1)
        )
        self.out_channels = num_filters[-1]

        self.register_buffer(
            "voxel_size",
            torch.tensor(voxel_size, dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "point_cloud_range",
            torch.tensor(point_cloud_range, dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "grid_size",
            torch.tensor(list(grid_size), dtype=torch.long),
            persistent=False,
        )

        self.scale_xyz = grid_size[0] * grid_size[1] * grid_size[2]
        self.scale_yz = grid_size[1] * grid_size[2]
        self.scale_z = grid_size[2]

    def forward(self, x: OptTensor, pos: Tensor, batch: Tensor) -> Tuple[Tensor, Tensor]:
        point_feats = pos if x is None else torch.cat([pos, x], dim=1)

        pos_grid = torch.floor((pos - self.point_cloud_range[:3]) / self.voxel_size).int()
        mask = ((pos_grid >= 0) & (pos_grid < self.grid_size)).all(dim=1)
        pos_grid = pos_grid[mask]
        pos = pos[mask]
        point_feats = point_feats[mask]
        batch = batch[mask]

        merge = (
            batch.int() * self.scale_xyz
            + pos_grid[:, 0] * self.scale_yz
            + pos_grid[:, 1] * self.scale_z
            + pos_grid[:, 2]
        )
        unq_indices, unq_inv = torch.unique(merge, return_inverse=True)

        points_mean = scatter(pos, unq_inv, dim=0, reduce="mean")
        f_cluster = pos - points_mean[unq_inv]
        center = pos_grid.to(pos.dtype) * self.voxel_size + (self.voxel_size / 2 + self.point_cloud_range[:3])
        f_center = pos - center

        features = torch.cat([point_feats, f_cluster, f_center], dim=1)
        for pfn in self.pfn_layers:
            features = pfn(features, unq_inv)

        unq_indices = unq_indices.int()
        voxel_indices = torch.stack(
            [
                torch.div(unq_indices, self.scale_xyz, rounding_mode="floor"),
                torch.div(unq_indices % self.scale_xyz, self.scale_yz, rounding_mode="floor"),
                torch.div(unq_indices % self.scale_yz, self.scale_z, rounding_mode="floor"),
                unq_indices % self.scale_z,
            ],
            dim=1,
        )
        voxel_indices = voxel_indices[:, [0, 3, 2, 1]]
        return features, voxel_indices


class DynamicMaxVFE(nn.Module):
    r"""Dynamic max-pooled voxel feature encoder: a point MLP max-pooled per voxel, then compressed.

    Points are assigned to voxels on the fly (no fixed points-per-voxel budget): their coordinates are quantized with
    `voxel_size` from the lower bound of `point_cloud_range`, and the points outside the range go to its border
    voxels. Each point gets the offset to its voxel center and its coordinates prepended to its features, a
    normalization layer normalizes the result, and an MLP (linear, norm, ReLU layers, the last one linear only)
    encodes it. The per-voxel maximum of the MLP output goes through a linear + ReLU compression layer.

    The coordinates live in any 3D system: a Cartesian grid, or a cylindrical $(\rho, \phi, z)$ partition when `pos`
    holds cylindrical coordinates and `point_cloud_range` their bounds.

    Args:
        in_channels: Point feature channels, without the coordinates.
        num_filters: Output width of each MLP layer.
        out_channels: Width of the compressed voxel features.
        voxel_size: Voxel size $(v_0, v_1, v_2)$ along each coordinate axis.
        point_cloud_range: Range $(c^0_\min, c^1_\min, c^2_\min, c^0_\max, c^1_\max, c^2_\max)$ of the coordinates.
        grid_size: Voxel grid extent $(n_0, n_1, n_2)$.
        norm: Normalization of the input features and of the MLP layers, passed to `create_norm`.
        norm_kwargs: Extra keyword arguments for the normalization.

    Shape:
        - Input: $(N, C_\text{in})$ point features, $(N, 3)$ coordinates and $(N,)$ batch index.
        - Output: $(M, C_\text{out})$ voxel features, $(M, 4)$ integer voxel indices $(b, i_0, i_1, i_2)$ and the
          $(N,)$ index of the voxel of every point.

    Example:
        ```pycon
        >>> import torch
        >>> from torch_pointcloud.layers import DynamicMaxVFE
        >>> vfe = DynamicMaxVFE(
        ...     1,
        ...     (16, 32),
        ...     8,
        ...     voxel_size=(1.0, 0.2, 0.75),
        ...     point_cloud_range=(0, -3.2, -4, 50, 3.2, 2),
        ...     grid_size=(50, 32, 8),
        ... )
        >>> x = torch.rand(100, 1)
        >>> pos = torch.rand(100, 3) * torch.tensor([50.0, 6.4, 6.0]) - torch.tensor([0.0, 3.2, 4.0])
        >>> batch = torch.zeros(100, dtype=torch.long)
        >>> voxel_x, voxel_indices, inverse = vfe(x, pos, batch)
        >>> voxel_x.shape[1], voxel_indices.shape[1], inverse.shape
        (8, 4, torch.Size([100]))

        ```
    """

    voxel_size: Tensor
    point_cloud_range: Tensor
    grid_size: Tensor

    def __init__(
        self,
        in_channels: int,
        num_filters: Sequence[int],
        out_channels: int,
        *,
        voxel_size: Sequence[float],
        point_cloud_range: Sequence[float],
        grid_size: Sequence[int],
        norm: Union[str, Callable, None] = "batch_norm",
        norm_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        if len(voxel_size) != 3:
            raise ValueError(f"Expected 3 voxel sizes, got {len(voxel_size)}.")
        if len(point_cloud_range) != 6:
            raise ValueError(f"Expected 6 range bounds (3 lower then 3 upper), got {len(point_cloud_range)}.")
        if len(grid_size) != 3:
            raise ValueError(f"Expected 3 grid sizes, got {len(grid_size)}.")

        super().__init__()
        norm_kwargs = norm_kwargs or {}
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.register_buffer(
            "voxel_size",
            torch.tensor(voxel_size, dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "point_cloud_range",
            torch.tensor(point_cloud_range, dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "grid_size",
            torch.tensor(list(grid_size), dtype=torch.long),
            persistent=False,
        )

        self.norm = create_norm(norm, in_channels + 6, **norm_kwargs) or nn.Identity()
        self.mlp = MLP([in_channels + 6, *num_filters], norm=norm, norm_kwargs=norm_kwargs, plain_last=True)
        self.compression = MLP([num_filters[-1], out_channels], norm=None, plain_last=False)

    def forward(self, x: Tensor, pos: Tensor, batch: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        range_min, range_max = self.point_cloud_range[:3], self.point_cloud_range[3:]
        pos_grid = torch.floor((pos.clamp(range_min, range_max) - range_min) / self.voxel_size).long()
        pos_grid = torch.minimum(pos_grid, self.grid_size - 1)
        center = pos_grid.to(pos.dtype) * self.voxel_size + (self.voxel_size / 2 + range_min)
        features = torch.cat([pos - center, pos, x], dim=1)
        features = self.norm(features)
        features = self.mlp(features)

        n_0, n_1, n_2 = self.grid_size.unbind()
        key = ((batch * n_0 + pos_grid[:, 0]) * n_1 + pos_grid[:, 1]) * n_2 + pos_grid[:, 2]
        unique, inverse = torch.unique(key, return_inverse=True)
        voxel_x = scatter(features, inverse, dim=0, dim_size=unique.numel(), reduce="max")
        voxel_x = self.compression(voxel_x)

        voxel_indices = torch.stack(
            [
                unique // (n_0 * n_1 * n_2),
                unique // (n_1 * n_2) % n_0,
                unique // n_2 % n_1,
                unique % n_2,
            ],
            dim=1,
        ).int()
        return voxel_x, voxel_indices, inverse
