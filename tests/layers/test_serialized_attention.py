import pytest
import torch

from torch_pointcloud.layers.serialized_attention import (
    RelativePositionalEncoding,
    SerializedAttention,
    SerializedAttentionRoPE,
    SerializedAttentionRPE,
)


def test_relative_positional_encoding_forward() -> None:
    rpe = RelativePositionalEncoding(patch_size=8, num_heads=4)
    # (B, K, K, 3) of relative voxel offsets
    coords = torch.randint(-2, 3, (2, 8, 8, 3))
    out = rpe(coords)
    assert out.shape == (2, 4, 8, 8)


def test_serialized_attention_forward_no_flash() -> None:
    attn = SerializedAttention(
        channels=16,
        num_heads=4,
        patch_size=8,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        use_flash_attn=False,
        upcast_attn=True,
        upcast_softmax=True,
    )
    n = 32
    x = torch.randn(n, 16)
    batch = torch.cat([torch.zeros(16), torch.ones(16)]).long()
    out = attn(x, None, batch, serialized_order=None, serialized_inverse=None, pos=None)
    assert out.shape == (n, 16)


def test_serialized_attention_channels_not_divisible_raises() -> None:
    with pytest.raises(ValueError, match="divisible"):
        SerializedAttention(channels=15, num_heads=4, patch_size=8, use_flash_attn=False)


def test_serialized_attention_rpe_forward() -> None:
    attn = SerializedAttentionRPE(
        channels=16,
        num_heads=4,
        patch_size=8,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        upcast_attn=True,
        upcast_softmax=True,
    )
    n = 32
    x = torch.randn(n, 16)
    pos_grid = torch.randint(0, 16, (n, 3))
    batch = torch.cat([torch.zeros(16), torch.ones(16)]).long()
    out = attn(x, pos_grid, batch, serialized_order=None, serialized_inverse=None, pos=None)
    assert out.shape == (n, 16)


def test_serialized_attention_rpe_requires_pos_grid() -> None:
    attn = SerializedAttentionRPE(channels=16, num_heads=4, patch_size=8)
    x = torch.randn(8, 16)
    batch = torch.zeros(8, dtype=torch.long)
    with pytest.raises(ValueError, match="pos_grid"):
        attn(x, None, batch)


def test_serialized_attention_rope_forward_no_flash() -> None:
    attn = SerializedAttentionRoPE(
        channels=12,
        num_heads=2,
        patch_size=8,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        use_flash_attn=False,
        upcast_attn=True,
        upcast_softmax=True,
        rope_base=10.0,
    )
    n = 32
    x = torch.randn(n, 12)
    pos = torch.randn(n, 3)
    batch = torch.cat([torch.zeros(16), torch.ones(16)]).long()
    out = attn(x, None, batch, serialized_order=None, serialized_inverse=None, pos=pos)
    assert out.shape == (n, 12)


def test_serialized_attention_rope_requires_pos() -> None:
    attn = SerializedAttentionRoPE(channels=12, num_heads=2, patch_size=8, use_flash_attn=False)
    x = torch.randn(8, 12)
    batch = torch.zeros(8, dtype=torch.long)
    with pytest.raises(ValueError, match="pos"):
        attn(x, None, batch, pos=None)
