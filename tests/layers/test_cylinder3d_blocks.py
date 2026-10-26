from typing import TYPE_CHECKING, Tuple

import pytest
import torch

from torch_pointcloud.layers.cylinder3d_blocks import (
    AsymmetricResidualBlock,
    AsymmetricUpsampleBlock,
    DimensionDecompositionBlock,
)
from torch_pointcloud.utils.imports import _CUDA_AVAILABLE, _SPCONV_AVAILABLE, optional_import

if TYPE_CHECKING:
    import spconv.pytorch as spconv
else:
    spconv, _ = optional_import("spconv.pytorch")

# See: https://docs.pytest.org/en/stable/how-to/skipping.html#summary
pytestmark = [
    pytest.mark.skipif(not _CUDA_AVAILABLE, reason="CUDA is not available"),
    pytest.mark.skipif(not _SPCONV_AVAILABLE, reason="spconv is not installed"),
]


def _sparse_tensor(channels: int, num_points: int = 600, size: int = 24) -> "spconv.SparseConvTensor":
    torch.manual_seed(0)
    indices = torch.cat([torch.randint(0, 2, (num_points, 1)), torch.randint(0, size, (num_points, 3))], dim=1)
    indices = torch.unique(indices, dim=0).int().cuda()
    features = torch.randn(len(indices), channels).cuda()
    return spconv.SparseConvTensor(features, indices, [size, size, size], 2)


@pytest.mark.parametrize(
    "kernel_sizes, expected_keys",
    [
        pytest.param(((1, 3, 3), (3, 1, 3)), {"block_1x3x3", "block_3x1x3"}, id="asymmetric"),
        pytest.param(((1, 3, 3), (1, 3, 3)), {"block_1x3x3"}, id="shared-kernel"),
    ],
)
def test_asymmetric_residual_block_shares_one_index_table_per_kernel(
    kernel_sizes: Tuple[Tuple[int, int, int], Tuple[int, int, int]],
    expected_keys: set,
) -> None:
    block = AsymmetricResidualBlock(8, 16, kernel_sizes, indice_key="block").cuda().eval()
    x = _sparse_tensor(8)
    out = block(x)
    assert out.features.shape == (x.features.shape[0], 16)
    assert torch.equal(out.indices, x.indices)
    assert set(out.indice_dict) == expected_keys


def test_asymmetric_residual_block_sums_the_two_branches() -> None:
    block = AsymmetricResidualBlock(8, 16, ((1, 3, 3), (3, 1, 3)), indice_key="block", norm=None).cuda().eval()
    torch.nn.init.zeros_(block.conv3.weight)
    x = _sparse_tensor(8)
    with torch.no_grad():
        hidden = block.conv1(x)
        hidden = block.conv2(hidden.replace_feature(block.act(hidden.features)))
        expected = block.act(hidden.features)
        out = block(x)
    assert torch.allclose(out.features, expected, atol=1e-5)


def test_asymmetric_upsample_block_returns_the_skip_resolution() -> None:
    downsample = spconv.SparseConv3d(16, 16, 3, stride=2, padding=1, bias=False, indice_key="down").cuda()
    block = AsymmetricUpsampleBlock(
        32,
        16,
        ((1, 3, 3), (3, 1, 3), (3, 3, 3)),
        indice_key="up",
        upsample_indice_key="down",
    ).cuda()
    projection = spconv.SubMConv3d(16, 32, 1, bias=False).cuda()
    skip = _sparse_tensor(16)
    coarse = projection(downsample(skip))
    out = block(coarse, skip)
    assert out.features.shape == (skip.features.shape[0], 16)
    assert torch.equal(out.indices, skip.indices)


def test_dimension_decomposition_block_scales_the_input() -> None:
    block = DimensionDecompositionBlock(8, ((3, 1, 1), (1, 3, 1), (1, 1, 3)), indice_key="gate", norm=None).cuda()
    for conv in (block.conv1, block.conv2, block.conv3):
        torch.nn.init.zeros_(conv.weight)
    x = _sparse_tensor(8)
    out = block(x)
    assert torch.allclose(out.features, 1.5 * x.features)
    assert torch.equal(out.indices, x.indices)
