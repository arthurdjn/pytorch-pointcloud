import pytest
import torch

from torch_pointcloud.layers.spconv_blocks import SparseConvBlock, SubMConv3dBlock
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
