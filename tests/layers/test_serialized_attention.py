from typing import Dict, Tuple

import pytest
import torch
from torch import Tensor

from torch_pointcloud.layers.serialized_attention import (
    RelativePositionalEncoding,
    SerializedAttention,
    SerializedAttentionRoPE,
    SerializedAttentionRPE,
)
from torch_pointcloud.transforms.functional import divisible_pad


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


def test_serialized_attention_rope_pos_follows_feature_permutation(monkeypatch: pytest.MonkeyPatch) -> None:
    """With B=2 and padding, RoPE must see `pos` gathered by the same permutation as the qkv rows."""
    torch.manual_seed(0)
    patch_size = 4
    attn = SerializedAttentionRoPE(channels=12, num_heads=2, patch_size=patch_size, use_flash_attn=False)
    n1, n2 = 6, 4
    x = torch.randn(n1 + n2, 12)
    pos = torch.randn(n1 + n2, 3)
    batch = torch.cat([torch.zeros(n1), torch.ones(n2)]).long()

    captured: Dict[str, Tensor] = {}
    rope_forward = attn.rope.forward

    def capture(q: Tensor, k: Tensor, p: Tensor) -> Tuple[Tensor, Tensor]:
        captured["pos"] = p
        return rope_forward(q, k, p)

    monkeypatch.setattr(attn.rope, "forward", capture)
    attn(x, None, batch, pos=pos)

    padded_indices, _, _ = divisible_pad(batch, patch_size, mode="above", pad_fill="replicate", return_inverse=True)
    assert padded_indices.numel() > batch.numel()  # the config must introduce padding
    torch.testing.assert_close(captured["pos"], pos[padded_indices])


def test_serialized_attention_rope_batched_matches_per_scene() -> None:
    """A padded B=2 forward matches running each scene separately (RoPE positions follow the padding)."""
    torch.manual_seed(0)
    attn = SerializedAttentionRoPE(channels=12, num_heads=2, patch_size=4, use_flash_attn=False)
    n1, n2 = 6, 4
    x = torch.randn(n1 + n2, 12)
    pos = torch.randn(n1 + n2, 3)
    batch = torch.cat([torch.zeros(n1), torch.ones(n2)]).long()

    out = attn(x, None, batch, pos=pos)
    out1 = attn(x[:n1], None, torch.zeros(n1, dtype=torch.long), pos=pos[:n1])
    out2 = attn(x[n1:], None, torch.zeros(n2, dtype=torch.long), pos=pos[n1:])
    torch.testing.assert_close(out[:n1], out1)
    torch.testing.assert_close(out[n1:], out2)


def test_serialized_attention_output_independent_of_co_batched_small_scene() -> None:
    """A scene at or above the patch size gives the same output whether it is batched alone or with a
    scene smaller than the patch: small scenes pad up to a full patch, they never shrink every scene's
    attention window."""
    torch.manual_seed(0)
    variants = [
        SerializedAttention(channels=24, num_heads=2, patch_size=8, use_flash_attn=False),
        SerializedAttentionRPE(channels=24, num_heads=2, patch_size=8),
        SerializedAttentionRoPE(channels=24, num_heads=2, patch_size=8, use_flash_attn=False),
    ]
    big = torch.randn(40, 24)
    small = torch.randn(3, 24)
    x = torch.cat([big, small])
    pos = torch.randn(43, 3)
    pos_grid = torch.randint(0, 16, (43, 3))
    batch = torch.cat([torch.zeros(40), torch.ones(3)]).long()
    solo_batch = torch.zeros(40, dtype=torch.long)

    for attn in variants:
        attn.eval()
        with torch.no_grad():
            out_cobatch = attn(x, pos_grid, batch, pos=pos)
            out_solo = attn(big, pos_grid[:40], solo_batch, pos=pos[:40])
        assert out_cobatch.shape == (43, 24)
        assert torch.equal(out_cobatch[:40], out_solo)


def test_serialized_attention_variants_accept_non_consecutive_batch_ids() -> None:
    """Batch ids {0, 2} behave exactly like {0, 1}; bincount over raw ids used to yield a zero patch size."""
    torch.manual_seed(0)
    n = 12
    x = torch.randn(n, 12)
    pos = torch.randn(n, 3)
    pos_grid = torch.randint(0, 8, (n, 3))
    consecutive = torch.cat([torch.zeros(6), torch.ones(6)]).long()
    gapped = torch.cat([torch.zeros(6), torch.full((6,), 2)]).long()

    attns = [
        SerializedAttention(channels=12, num_heads=2, patch_size=8, use_flash_attn=False),
        SerializedAttentionRPE(channels=12, num_heads=2, patch_size=8),
        SerializedAttentionRoPE(channels=12, num_heads=2, patch_size=8, use_flash_attn=False),
    ]
    for attn in attns:
        out_gapped = attn(x, pos_grid, gapped, pos=pos)
        assert torch.equal(out_gapped, attn(x, pos_grid, consecutive, pos=pos))
