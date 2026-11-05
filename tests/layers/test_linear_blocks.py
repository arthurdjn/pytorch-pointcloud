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


def test_linear_block_normalizes_per_batch_element() -> None:
    torch.manual_seed(0)
    block = LinearBlock(in_channels=4, out_channels=8, norm="group_norm", norm_kwargs={"num_groups": 2})
    x = torch.randn(10, 4)
    x[6:] = x[6:] * 10 + 5
    batch = torch.tensor([0] * 6 + [1] * 4)
    out = block(x, batch=batch, batch_size=2)
    assert torch.allclose(out[:6], block(x[:6]), atol=1e-5)
    assert torch.allclose(out[6:], block(x[6:]), atol=1e-5)
    assert not torch.allclose(block(x)[:6], out[:6], atol=1e-3)
