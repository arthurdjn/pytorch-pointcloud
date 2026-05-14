import pytest
import torch

from torch_pointcloud.layers.spconv_blocks import SubMConv3dBlock
from torch_pointcloud.utils.imports import _CUDA_AVAILABLE, _SPCONV_AVAILABLE

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
