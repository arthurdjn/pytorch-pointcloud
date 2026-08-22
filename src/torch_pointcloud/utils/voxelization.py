"""Dense and sparse voxelization with trilinear devoxelization for packed point clouds."""

import functools
from typing import TYPE_CHECKING, Literal, Sequence, Tuple, overload

import torch
from torch import IntTensor, Tensor
from torch_geometric.nn.pool import voxel_grid
from torch_geometric.nn.pool.consecutive import consecutive_cluster
from torch_geometric.utils import scatter

from torch_pointcloud.utils.imports import _SPCONV_GITHUB_URL, optional_import

if TYPE_CHECKING:
    from spconv.pytorch.utils import PointToVoxel
else:
    PointToVoxel, _ = optional_import("spconv.pytorch.utils", "PointToVoxel", url=_SPCONV_GITHUB_URL)


def dense_voxelize(x: Tensor, pos: Tensor, batch: Tensor, resolution: int, reduce: str = "mean") -> Tensor:
    r"""Pool packed point features into a dense voxel grid, one grid per point cloud.

    Args:
        x: Packed point features $(N, C)$.
        pos: Packed grid coordinates $(N, 3)$, clamped to $[0, \text{resolution} - 1)$.
        batch: Per-point batch index $(N,)$.
        resolution: Number of voxels $R$ per axis.
        reduce: Scatter reduction applied to the points of a voxel (e.g. `mean`, `max`).

    Returns:
        The voxel grid $(B, C, R, R, R)$, zero where a voxel holds no point.
    """
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
    x_pooled = scatter(x, voxel_idx, dim=0, reduce=reduce, dim_size=max_voxel_idx)
    x_pooled = x_pooled.view(B, R, R, R, C)  # (B, z, y, x, C)
    return x_pooled.permute(0, 4, 3, 2, 1)  # (B, C, x, y, z)


# Adapted from: https://github.com/mit-han-lab/pvcnn/blob/master/modules/functional/src/interpolate/trilinear_devox.cu
def trilinear_dense_devoxelize(x_voxel: Tensor, pos: Tensor, batch: Tensor, resolution: int) -> Tensor:
    r"""Interpolate a dense voxel grid back to packed point features, trilinearly.

    Args:
        x_voxel: Voxel grid $(B, C, R, R, R)$.
        pos: Packed grid coordinates $(N, 3)$, clamped to $[0, \text{resolution} - 1)$.
        batch: Per-point batch index $(N,)$.
        resolution: Number of voxels $R$ per axis. Must match the grid.

    Returns:
        The interpolated point features $(N, C)$.
    """
    device = pos.device
    N, _ = pos.shape
    B, C, R, R1, R2 = x_voxel.shape

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
    )  # (8, 3)

    corner_offsets = corner_offsets.unsqueeze(0).expand(N, -1, -1)  # (N, 8, 3)
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

    x_voxel_reordered = x_voxel.permute(0, 4, 3, 2, 1)  # (B, C, x, y, z) -> (B, z, y, x, C)
    x_voxel_flat = x_voxel_reordered.reshape(B * R * R * R, C)  # (B*R*R*R, C)

    # Gather features for all corners
    gathered_features = x_voxel_flat[global_indices.view(-1)]  # (N*8, C)
    gathered_features = gathered_features.view(N, 8, C)

    # Apply trilinear interpolation: weighted sum over 8 neighbors
    weights_expanded = weights.unsqueeze(2)  # (N, 8, 1)
    x_out = (gathered_features * weights_expanded).sum(dim=1)  # (N, C)

    return x_out


@overload
def sparse_voxelize(
    x: Tensor,
    pos: Tensor,
    batch: Tensor,
    voxel_size: float,
    reduce: str,
    return_inverse: Literal[True],
) -> Tuple[Tensor, IntTensor, Tensor, Tensor]: ...


@overload
def sparse_voxelize(
    x: Tensor,
    pos: Tensor,
    batch: Tensor,
    voxel_size: float,
    reduce: str = "mean",
    return_inverse: Literal[False] = False,
) -> Tuple[Tensor, IntTensor, Tensor]: ...


def sparse_voxelize(
    x: Tensor,
    pos: Tensor,
    batch: Tensor,
    voxel_size: float,
    reduce: str = "mean",
    return_inverse: bool = False,
) -> Tuple[Tensor, ...]:
    r"""Pool packed point features into sparse voxels of a regular grid.

    Args:
        x: Packed point features $(N, C)$.
        pos: Packed point coordinates $(N, 3)$.
        batch: Per-point batch index $(N,)$.
        voxel_size: Edge length of a voxel, in the units of `pos`.
        reduce: Scatter reduction applied to the points of a voxel (e.g. `mean`, `max`).
        return_inverse: Also return the per-point voxel index, to broadcast voxel values back to the points.

    Returns:
        The voxel features $(V, C)$, their integer coordinates $(V, 3)$ and batch index $(V,)$, plus the
        per-point voxel index $(N,)$ when `return_inverse` is `True`.
    """
    # Ensure the position is float
    pos = pos.float()
    start = pos.min(0)[0]

    cluster = voxel_grid(pos, batch=batch, size=voxel_size, start=start)
    cluster, perm = consecutive_cluster(cluster)

    x_voxel = scatter(x, cluster, dim=0, reduce=reduce)
    pos_voxel = torch.div(pos[perm] - start, voxel_size, rounding_mode="trunc").int()
    batch_voxel = batch[perm]

    if return_inverse:
        return x_voxel, pos_voxel, batch_voxel, cluster
    return x_voxel, pos_voxel, batch_voxel


@functools.lru_cache(maxsize=8)
def _point_to_voxel_generator(
    voxel_size: Tuple[float, ...],
    point_cloud_range: Tuple[float, ...],
    num_point_features: int,
    max_num_points: int,
    max_num_voxels: int,
    device: str,
) -> "PointToVoxel":
    """Build (and cache) a `spconv` hard-voxel generator for a given config and device."""
    return PointToVoxel(
        vsize_xyz=list(voxel_size),
        coors_range_xyz=list(point_cloud_range),
        num_point_features=num_point_features,
        max_num_voxels=max_num_voxels,
        max_num_points_per_voxel=max_num_points,
        device=torch.device(device),
    )


def hard_voxelize(
    points: Tensor,
    batch: Tensor,
    voxel_size: Sequence[float],
    point_cloud_range: Sequence[float],
    max_num_points: int,
    max_num_voxels: int,
) -> Tuple[Tensor, Tensor, Tensor]:
    r"""Hard voxelization of a packed batch of point clouds (`spconv` voxel generator).

    Reproduces the `transform_points_to_voxels` step of voxel detectors (PointPillars, SECOND):
    each scene is voxelized independently (at most `max_num_points` points per voxel and
    `max_num_voxels` voxels per scene), then the per-scene voxels are concatenated with a leading
    batch index. Points outside `point_cloud_range` are dropped by the generator.

    Args:
        points: Packed point features $(N, C)$ with the first three columns the $xyz$ coordinates.
        batch: Per-point batch index $(N,)$.
        voxel_size: Voxel size $(v_x, v_y, v_z)$.
        point_cloud_range: Range $(x_\min, y_\min, z_\min, x_\max, y_\max, z_\max)$.
        max_num_points: Maximum number of points kept per voxel.
        max_num_voxels: Maximum number of voxels kept per scene.

    Returns:
        A tuple `(voxels, coords, num_points)` where `voxels` is $(V, \text{max\_num\_points}, C)$,
        `coords` is $(V, 4)$ with columns $(\text{batch}, z, y, x)$ and `num_points` is $(V,)$.
    """
    batch_size = int(batch.max().item()) + 1 if batch.numel() else 0
    if batch_size == 0:
        voxels = points.new_zeros((0, max_num_points, points.shape[1]))
        coords = torch.zeros((0, 4), dtype=torch.int32, device=points.device)
        num_points = torch.zeros((0,), dtype=torch.int32, device=points.device)
        return voxels, coords, num_points

    generator = _point_to_voxel_generator(
        tuple(voxel_size),
        tuple(point_cloud_range),
        int(points.shape[1]),
        max_num_points,
        max_num_voxels,
        str(points.device),
    )

    voxels_list, coords_list, num_points_list = [], [], []
    for b in range(batch_size):
        scene = points[batch == b].contiguous()
        voxels, coords, num_points = generator(scene)
        batch_col = torch.full((coords.shape[0], 1), b, dtype=coords.dtype, device=coords.device)
        voxels_list.append(voxels)
        coords_list.append(torch.cat([batch_col, coords], dim=1))
        num_points_list.append(num_points)

    return torch.cat(voxels_list, dim=0), torch.cat(coords_list, dim=0), torch.cat(num_points_list, dim=0)
