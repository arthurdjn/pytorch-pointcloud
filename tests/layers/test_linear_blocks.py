import torch

from torch_pointcloud.layers.linear_blocks import LinearBlock


def test_linear_block_forward() -> None:
    block = LinearBlock(
        in_channels=64,
        out_channels=128,
        act="relu",
        act_kwargs=None,
        act_first=False,
        bias=True,
        norm="batch_norm",
        norm_kwargs=None,
    )
    x = torch.randn(32, 64)
    out = block(x)
    assert out.shape == (32, 128)


def test_linear_block_act_first() -> None:
    block = LinearBlock(
        in_channels=64,
        out_channels=128,
        act="relu",
        act_first=True,
        norm="batch_norm",
    )
    x = torch.randn(32, 64)
    out = block(x)
    assert out.shape == (32, 128)


def test_linear_block_no_act_no_norm() -> None:
    block = LinearBlock(in_channels=64, out_channels=128, act=None, norm=None, bias=False)
    assert block.act is None
    assert block.norm is None
    assert block.stem.bias is None
    x = torch.randn(32, 64)
    assert block(x).shape == (32, 128)


def test_linear_block_extra_args_ignored() -> None:
    block = LinearBlock(in_channels=4, out_channels=8)
    x = torch.randn(10, 4)
    pos = torch.randn(10, 3)
    batch = torch.zeros(10, dtype=torch.long)
    out = block(x, pos=pos, batch=batch)
    assert out.shape == (10, 8)
