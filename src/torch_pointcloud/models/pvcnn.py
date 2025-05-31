import torch
from torch import Tensor
from torch_scatter import scatter


def avg_voxelize(x: Tensor, pos: Tensor, batch: Tensor, r: int) -> Tensor:
    if pos.shape[1] != 3:
        raise ValueError(f"Position tensor must be 3D, but got a {pos.shape[1]}-D tensor.")

    _, C = x.shape
    B = batch.max().item() + 1

    # Ensure coordinates are integers and within bounds
    coords = pos.long().clamp(0, r - 1)

    # Create unique voxel indices for clustering
    # Each voxel gets a unique ID based on (batch, x, y, z)
    linear_voxel_idx = coords[:, 2] * (r * r) + coords[:, 1] * r + coords[:, 0]  # z*r² + y*r + x
    batch_offset = batch * (r * r * r)
    voxel_idx = linear_voxel_idx + batch_offset
    # Use scatter to pool features
    max_voxel_idx = B * r * r * r
    x_pooled = scatter(x, voxel_idx, dim=0, reduce="mean", dim_size=max_voxel_idx)
    x_pooled = x_pooled.view(B, r, r, r, C)  # (B, z, y, x, C)
    return x_pooled.permute(0, 4, 3, 2, 1)  # (B, C, x, y, z) - to match numba


# Adapted from: https://github.com/mit-han-lab/pvcnn/blob/master/modules/functional/src/interpolate/trilinear_devox.cu
def trilinear_devoxelize(x_voxels: Tensor, pos: Tensor, batch: Tensor, resolution: int) -> Tensor:
    device = pos.device
    N, _ = pos.shape
    B, C, R, R1, R2 = x_voxels.shape

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

    # Create corner offsets: (8, 3) representing all binary combinations
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
