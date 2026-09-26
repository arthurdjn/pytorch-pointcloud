import pytest
import torch

from torch_pointcloud.layers.torchsparse_blocks import TorchSparseConvBlock, TorchSparseResidualBlock
from torch_pointcloud.utils.imports import _CUDA_AVAILABLE, _TORCHSPARSE_AVAILABLE

pytestmark = pytest.mark.skipif(
    not (_TORCHSPARSE_AVAILABLE and _CUDA_AVAILABLE),
    reason="torchsparse and CUDA are required",
)


def _sparse_tensor() -> "torchsparse.SparseTensor":  # type: ignore[name-defined]  # noqa: F821
    from torchsparse import SparseTensor

    torch.manual_seed(0)
    coords = torch.randint(0, 16, (256, 3), dtype=torch.int32)
    coords = torch.cat([torch.zeros(256, 1, dtype=torch.int32), coords], dim=1)
    coords = torch.unique(coords, dim=0)
    feats = torch.randn(coords.shape[0], 8)
    return SparseTensor(feats=feats.cuda(), coords=coords.cuda())


def test_torchsparse_conv_block_forward() -> None:
    block = TorchSparseConvBlock(8, 16, kernel_size=3).cuda()
    out = block(_sparse_tensor())
    assert out.F.shape[1] == 16


def test_torchsparse_residual_block_projects_skip() -> None:
    x = _sparse_tensor()
    block = TorchSparseResidualBlock(8, 16, kernel_size=3).cuda()
    assert block.conv_skip is not None
    out = block(x)
    assert out.F.shape == (x.F.shape[0], 16)

    same = TorchSparseResidualBlock(8, 8, kernel_size=3).cuda()
    assert same.conv_skip is None
    assert same(x).F.shape == x.F.shape
