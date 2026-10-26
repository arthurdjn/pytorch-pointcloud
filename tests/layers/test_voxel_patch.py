from typing import Tuple

import torch
from torch import Tensor, nn

from torch_pointcloud.layers.voxel_patch import VoxelPatchEmbed, VoxelPatchExpand
from torch_pointcloud.ops.voxelization import voxel_patch_partition

PATCH_SIZE = 3
GRID = 9


def _occupied_voxels() -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    torch.manual_seed(0)
    pos_grid = torch.unique(torch.randint(0, GRID, (300, 3)), dim=0)
    batch = torch.zeros(pos_grid.size(0), dtype=torch.long)
    patch_grid, _, inverse, offset = voxel_patch_partition(pos_grid, batch, PATCH_SIZE)
    return pos_grid, patch_grid, inverse, offset, batch


def test_voxel_patch_embed_is_a_strided_convolution() -> None:
    pos_grid, patch_grid, inverse, offset, _ = _occupied_voxels()
    x = torch.randn(pos_grid.size(0), 4)
    embed = VoxelPatchEmbed(in_channels=4, embed_dim=8, patch_size=PATCH_SIZE)
    conv = nn.Conv3d(4, 8, kernel_size=PATCH_SIZE, stride=PATCH_SIZE)
    with torch.no_grad():
        conv.weight.copy_(embed.proj.weight.view(8, PATCH_SIZE, PATCH_SIZE, PATCH_SIZE, 4).permute(0, 4, 1, 2, 3))
        conv.bias.copy_(embed.proj.bias)

    tokens = embed(x, inverse, offset, patch_grid.size(0))

    dense = torch.zeros(1, 4, GRID, GRID, GRID)
    dense[0, :, pos_grid[:, 0], pos_grid[:, 1], pos_grid[:, 2]] = x.T
    expected = conv(dense)[0, :, patch_grid[:, 0], patch_grid[:, 1], patch_grid[:, 2]].T
    torch.testing.assert_close(tokens, expected)


def test_voxel_patch_expand_is_a_strided_transposed_convolution() -> None:
    pos_grid, patch_grid, inverse, offset, _ = _occupied_voxels()
    tokens = torch.randn(patch_grid.size(0), 6)
    expand = VoxelPatchExpand(in_channels=6, out_channels=5, patch_size=PATCH_SIZE)
    nn.init.normal_(expand.bias)
    deconv = nn.ConvTranspose3d(6, 5, kernel_size=PATCH_SIZE, stride=PATCH_SIZE)
    with torch.no_grad():
        weight = expand.proj.weight.view(PATCH_SIZE, PATCH_SIZE, PATCH_SIZE, 5, 6)
        deconv.weight.copy_(weight.permute(4, 3, 0, 1, 2))
        deconv.bias.copy_(expand.bias)

    out = expand(tokens, inverse, offset)

    dense = torch.zeros(1, 6, GRID // PATCH_SIZE, GRID // PATCH_SIZE, GRID // PATCH_SIZE)
    dense[0, :, patch_grid[:, 0], patch_grid[:, 1], patch_grid[:, 2]] = tokens.T
    expected = deconv(dense)[0, :, pos_grid[:, 0], pos_grid[:, 1], pos_grid[:, 2]].T
    torch.testing.assert_close(out, expected)
