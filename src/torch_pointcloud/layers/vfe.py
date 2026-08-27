r"""Dynamic voxel feature encoder shared by the voxel detectors (Voxel Mamba, LION).

A packed-format port of the `DynamicVoxelVFE` from
:github: [gwenzhang/Voxel-Mamba](https://github.com/gwenzhang/Voxel-Mamba).
"""

from typing import TYPE_CHECKING, Sequence, Tuple

import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.nn import MLP

from torch_pointcloud.utils.imports import _TORCH_SCATTER_GITHUB_URL, optional_import
from torch_pointcloud.utils.types import OptTensor

if TYPE_CHECKING:
    from torch_scatter import scatter_max, scatter_mean

scatter_max, _ = optional_import("torch_scatter", "scatter_max", url=_TORCH_SCATTER_GITHUB_URL)
scatter_mean, _ = optional_import("torch_scatter", "scatter_mean", url=_TORCH_SCATTER_GITHUB_URL)


class PFNLayer(nn.Module):
    r"""Pillar feature net layer (the reference's `PFNLayerV2`): linear + norm + ReLU with a per-voxel max-pool.

    Non-final layers halve their output width and concatenate the pooled feature back onto every
    point; the final layer returns the pooled per-voxel feature directly.

    Args:
        in_channels: Input feature channels.
        out_channels: Target output channels (halved internally for non-final layers).
        last: Whether this is the final PFN layer.

    Shape:
        - Input: $(N, C_\text{in})$ point features and $(N,)$ voxel index.
        - Output: $(N, C')$ for non-final layers, $(M, C_\text{out})$ for the final layer.
    """

    def __init__(self, in_channels: int, out_channels: int, last: bool) -> None:
        super().__init__()
        self.last = last
        out_dim = out_channels if last else out_channels // 2
        self.mlp = MLP(
            [in_channels, out_dim],
            act="relu",
            norm="batch_norm",
            norm_kwargs=dict(eps=1e-3, momentum=0.01),
            bias=False,
            plain_last=False,
        )

    def forward(self, inputs: Tensor, unq_inv: Tensor) -> Tensor:
        x = self.mlp(inputs)
        x_max = scatter_max(x, unq_inv, dim=0)[0]
        if self.last:
            return x_max
        return torch.cat([x, x_max[unq_inv]], dim=1)


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
            PFNLayer(widths[i], widths[i + 1], last=i >= len(widths) - 2) for i in range(len(widths) - 1)
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

    def forward(self, pos: Tensor, x: OptTensor, batch: Tensor) -> Tuple[Tensor, Tensor]:
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

        points_mean = scatter_mean(pos, unq_inv, dim=0)
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
