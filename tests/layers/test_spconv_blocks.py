from typing import TYPE_CHECKING

import pytest
import torch
import torch.nn.functional as F

from torch_pointcloud.layers.spconv_blocks import (
    SparseBasicBlock,
    SparseConvBlock,
    SparseResNetBlock,
    StridedSparseConv3d,
    StridedSparseConvTranspose3d,
    SubMConv3dBlock,
)
from torch_pointcloud.utils.conversion import convert_to_spconv_tensor
from torch_pointcloud.utils.imports import _CUDA_AVAILABLE, _SPCONV_AVAILABLE

if TYPE_CHECKING:
    import spconv.pytorch as spconv

# See: https://docs.pytest.org/en/stable/how-to/skipping.html#summary
pytestmark = [
    pytest.mark.skipif(not _CUDA_AVAILABLE, reason="CUDA is not available"),
    pytest.mark.skipif(not _SPCONV_AVAILABLE, reason="spconv is not installed"),
]


def test_submconv3d_block_forward() -> None:
    block = SubMConv3dBlock(
        in_channels=8,
        out_channels=16,
        kernel_size=3,
        padding=1,
        norm="batch_norm",
        act="relu",
        act_kwargs=None,
        norm_kwargs=None,
        bias=True,
        stem_indice_key=None,
    ).cuda()

    x = torch.randn(20, 8).cuda()
    pos = torch.randint(0, 32, (20, 3)).cuda()
    batch = torch.cat([torch.zeros(8), torch.ones(12)]).long().cuda()

    out = block(x, pos, batch)
    assert out.shape == (20, 16)


def test_sparse_conv_block_state_dict_positions() -> None:
    block = SparseConvBlock(4, 8, 3, indice_key="s1")
    keys = set(block.state_dict().keys())
    assert "0.weight" in keys
    assert "1.module.weight" in keys and "1.module.bias" in keys


def test_sparse_conv_block_none_norm_act_builds_identity() -> None:
    block = SparseConvBlock(4, 8, 3, indice_key="s1", norm=None, act=None)
    modules = dict(block.named_modules())
    assert isinstance(modules["1"], torch.nn.Identity)
    assert isinstance(modules["2"], torch.nn.Identity)


def test_sparse_conv_block_none_norm_act_forward() -> None:
    import spconv.pytorch as spconv

    block = SparseConvBlock(4, 8, 3, indice_key="s1", norm=None, act=None).cuda()
    x = torch.randn(20, 4).cuda()
    indices = (
        torch.cat(
            [torch.cat([torch.zeros(8), torch.ones(12)]).long().unsqueeze(-1), torch.randint(0, 16, (20, 3))],
            dim=1,
        )
        .int()
        .cuda()
    )
    sparse_x = spconv.SparseConvTensor(x, indices, spatial_shape=[16, 16, 16], batch_size=2)
    out = block(sparse_x)
    assert out.features.shape == (20, 8)


def test_sparse_conv_block_invalid_conv_type_raises() -> None:
    with pytest.raises(ValueError, match="Unknown conv_type"):
        SparseConvBlock(4, 8, 3, indice_key="s1", conv_type="invalid")


def test_sparse_basic_block_projects_skip() -> None:
    torch.manual_seed(0)
    x = torch.randn(20, 8).cuda()
    pos = torch.randint(0, 32, (20, 3)).cuda()
    batch = torch.cat([torch.zeros(8), torch.ones(12)]).long().cuda()
    x_sparse = convert_to_spconv_tensor(x, pos, batch)

    block = SparseBasicBlock(8, 16, indice_key="res").cuda()
    assert block(x_sparse).features.shape == (20, 16)

    same = SparseBasicBlock(8, 8, bias=True, indice_key="res").cuda()
    assert not any(p.numel() for p in same.proj.parameters())
    assert same(x_sparse).features.shape == (20, 8)


def _sparse_grid(size: int, channels: int, seed: int = 0) -> "spconv.SparseConvTensor":
    import spconv.pytorch as spconv

    generator = torch.Generator().manual_seed(seed)
    voxels = (torch.rand(size, size, size, generator=generator) < 0.2).nonzero().int()
    # a voxel on the far corner, where a transposed convolution's own grid ends one voxel short at even sizes
    voxels = torch.unique(torch.cat([voxels, torch.full((1, 3), size - 1, dtype=torch.int32)]), dim=0)
    indices = torch.cat([torch.zeros(len(voxels), 1, dtype=torch.int32), voxels], dim=1).contiguous().cuda()
    features = torch.randn(len(voxels), channels, generator=generator).cuda()
    return spconv.SparseConvTensor(features, indices, [size] * 3, 1)


def test_strided_sparse_conv3d_outputs_the_parent_voxels() -> None:
    x = _sparse_grid(12, 3)
    out = StridedSparseConv3d(3, 4, kernel_size=3, stride=2).cuda()(x)

    parents = torch.unique(torch.cat([x.indices[:, :1], x.indices[:, 1:] // 2], dim=1), dim=0)
    assert torch.equal(out.indices, parents.int())


@pytest.mark.parametrize("size", [pytest.param(9, id="odd"), pytest.param(12, id="even")])
def test_strided_sparse_conv_transpose3d_matches_a_dense_transposed_convolution(size: int) -> None:
    fine = _sparse_grid(size, 3)
    coarse = StridedSparseConv3d(3, 5, kernel_size=3, stride=2).cuda()(fine)
    up = StridedSparseConvTranspose3d(5, 4, kernel_size=3, stride=2).cuda()
    out = up(coarse, fine)

    dense = F.conv_transpose3d(coarse.dense(), up.weight.permute(4, 0, 1, 2, 3), stride=2, padding=1, output_padding=1)
    index = fine.indices.long()
    expected = dense[0, :, index[:, 1], index[:, 2], index[:, 3]].T
    assert torch.equal(out.indices, fine.indices)
    torch.testing.assert_close(out.features, expected, atol=1e-5, rtol=1e-5)


def test_sparse_resnet_block_adds_the_projected_input_after_the_activation() -> None:
    x = _sparse_grid(8, 4)
    block = SparseResNetBlock(4, 6, indice_key="subm").cuda().eval()
    out = block(x)

    branch = block.conv1(x)
    branch = branch.replace_feature(block.act(block.norm1(branch.features)))
    branch = block.conv2(branch)
    expected = block.act(block.norm2(branch.features)) + block.proj(x).features
    torch.testing.assert_close(out.features, expected)
    assert set(block.state_dict()) >= {"conv1.weight", "conv2.weight", "proj.0.weight", "proj.1.module.weight"}
    assert not any(p.numel() for p in SparseResNetBlock(4, 4).proj.parameters())
