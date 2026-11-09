r"""Embedding of non-overlapping cubic voxel patches into tokens, and expansion of the tokens back to the voxels.

Both take the patch of every voxel and its position inside the patch from
[`voxel_patch_partition`](../ops/voxelization.md).
"""

import torch
import torch.nn as nn
from torch import Tensor


class VoxelPatchEmbed(nn.Module):
    r"""Linear embedding of non-overlapping cubic patches of $K^3$ voxels into tokens.

    The features of the voxels of a patch are laid out on its $K^3$ positions, with zeros at the empty ones, flattened
    and projected to `embed_dim`: a sparse 3D convolution of kernel size and stride $K$, evaluated at the occupied
    patches only.

    Args:
        in_channels: Voxel feature channels $C$.
        embed_dim: Token dimension $D$.
        patch_size: Patch side $K$, in voxels.

    Shape:
        - Input: $(N, C)$ voxel features, the $(N,)$ patch of every voxel and its $(N,)$ position inside the patch
            (one voxel per position), and the number of patches $P$.
        - Output: $(P, D)$ tokens.

    Example:
        ```python
        import torch
        from torch_pointcloud.layers import VoxelPatchEmbed
        from torch_pointcloud.ops.voxelization import voxel_patch_partition

        pos_grid = torch.unique(torch.randint(0, 32, (1000, 3)), dim=0)
        batch = torch.zeros(pos_grid.size(0), dtype=torch.long)
        patch_grid, patch_batch, inverse, offset = voxel_patch_partition(pos_grid, batch, patch_size=5)

        embed = VoxelPatchEmbed(in_channels=6, embed_dim=64, patch_size=5)
        tokens = embed(torch.randn(pos_grid.size(0), 6), inverse, offset, patch_grid.size(0))
        ```
    """

    def __init__(self, in_channels: int, embed_dim: int, patch_size: int) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.proj = nn.Linear(patch_size**3 * in_channels, embed_dim)

    def forward(self, x: Tensor, inverse: Tensor, offset: Tensor, num_patches: int) -> Tensor:
        patches = x.new_zeros(num_patches, self.patch_size**3, x.size(1))
        patches[inverse, offset] = x
        return self.proj(patches.flatten(1))


class VoxelPatchExpand(nn.Module):
    r"""Expansion of tokens back to the voxels of their cubic patches of $K^3$ voxels.

    Every token is projected to $K^3$ feature vectors, one per patch position, and every voxel takes the vector of its
    position in its patch, plus a bias shared by all positions: a transposed sparse 3D convolution of kernel size and
    stride $K$, evaluated at the occupied voxels only.

    Args:
        in_channels: Token channels $C$.
        out_channels: Voxel feature channels $D$.
        patch_size: Patch side $K$, in voxels.

    Shape:
        - Input: $(P, C)$ tokens, the $(N,)$ patch of every voxel and its $(N,)$ position inside the patch.
        - Output: $(N, D)$ voxel features.

    Example:
        ```python
        import torch
        from torch_pointcloud.layers import VoxelPatchExpand
        from torch_pointcloud.ops.voxelization import voxel_patch_partition

        pos_grid = torch.unique(torch.randint(0, 32, (1000, 3)), dim=0)
        batch = torch.zeros(pos_grid.size(0), dtype=torch.long)
        patch_grid, patch_batch, inverse, offset = voxel_patch_partition(pos_grid, batch, patch_size=5)

        expand = VoxelPatchExpand(in_channels=64, out_channels=32, patch_size=5)
        x = expand(torch.randn(patch_grid.size(0), 64), inverse, offset)
        ```
    """

    def __init__(self, in_channels: int, out_channels: int, patch_size: int) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.patch_size = patch_size
        self.proj = nn.Linear(in_channels, patch_size**3 * out_channels, bias=False)
        self.bias = nn.Parameter(torch.zeros(out_channels))

    def forward(self, x: Tensor, inverse: Tensor, offset: Tensor) -> Tensor:
        patches = self.proj(x).view(-1, self.patch_size**3, self.out_channels)
        return patches[inverse, offset] + self.bias
