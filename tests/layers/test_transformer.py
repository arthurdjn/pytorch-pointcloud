from typing import Any, Tuple

import pytest
import torch
from torch import Tensor, nn

from torch_pointcloud.layers.rope import AxialRoPE
from torch_pointcloud.layers.transformer import (
    PackedAttention,
    PackedTransformerBlock,
    TransformerDecoderLayer,
    TransformerEncoderLayer,
)
from torch_pointcloud.utils.conversion import batch_to_cu_seqlens
from torch_pointcloud.utils.imports import _CUDA_AVAILABLE, _FLASH_ATTN_AVAILABLE


@pytest.mark.parametrize("norm_first", [True, False])
def test_transformer_encoder_layer_forward(norm_first: bool) -> None:
    torch.manual_seed(0)
    layer = TransformerEncoderLayer(32, 4, 64, dropout=0.0, norm_first=norm_first).eval()
    src = torch.randn(10, 2, 32)
    pos = torch.randn(10, 2, 32)
    mask = torch.zeros(10, 10, dtype=torch.bool)
    out = layer(src, src_mask=mask, pos=pos)
    assert out.shape == (10, 2, 32)
    assert not torch.allclose(out, layer(src))


@pytest.mark.parametrize("norm_first", [True, False])
@pytest.mark.parametrize("pos_in_value", [False, True])
def test_transformer_decoder_layer_forward(norm_first: bool, pos_in_value: bool) -> None:
    torch.manual_seed(0)
    layer = TransformerDecoderLayer(32, 4, 64, dropout=0.0, norm_first=norm_first, pos_in_value=pos_in_value).eval()
    tgt = torch.randn(6, 2, 32)
    memory = torch.randn(10, 2, 32)
    pos = torch.randn(10, 2, 32)
    query_pos = torch.randn(6, 2, 32)
    padding = torch.zeros(2, 10, dtype=torch.bool)
    padding[:, -2:] = True
    out = layer(tgt, memory, pos=pos, query_pos=query_pos, memory_key_padding_mask=padding)
    assert out.shape == (6, 2, 32)


def test_transformer_decoder_layer_pos_in_value_only_matters_with_positions() -> None:
    torch.manual_seed(0)
    plain = TransformerDecoderLayer(32, 4, 64, dropout=0.0).eval()
    in_value = TransformerDecoderLayer(32, 4, 64, dropout=0.0, pos_in_value=True).eval()
    in_value.load_state_dict(plain.state_dict())
    tgt, memory = torch.randn(6, 2, 32), torch.randn(10, 2, 32)
    assert torch.allclose(plain(tgt, memory), in_value(tgt, memory))


def _packed_tokens(lengths: Tuple[int, ...], channels: int) -> Tuple[Tensor, Tensor, Tensor, int]:
    batch = torch.repeat_interleave(torch.arange(len(lengths)), torch.tensor(lengths))
    x = torch.randn(batch.numel(), channels)
    pos = torch.randint(0, 32, (batch.numel(), 3))
    return x, pos, batch_to_cu_seqlens(batch).int(), max(lengths)


def test_packed_attention_matches_each_sample_alone() -> None:
    torch.manual_seed(0)
    attn = PackedAttention(32, 2, qk_norm=True, rope=AxialRoPE((3, 3, 2), base=100.0), use_flash_attn=False)
    x, pos, cu_seqlens, max_seqlen = _packed_tokens((7, 12, 5), 32)

    out = attn(x, cu_seqlens, max_seqlen, pos=pos)

    for start, end in zip(cu_seqlens[:-1].tolist(), cu_seqlens[1:].tolist()):
        alone = attn(x[start:end], torch.tensor([0, end - start], dtype=torch.int32), end - start, pos=pos[start:end])
        torch.testing.assert_close(out[start:end], alone)


def test_packed_attention_without_rope_ignores_positions() -> None:
    torch.manual_seed(0)
    attn = PackedAttention(32, 2, use_flash_attn=False)
    x, pos, cu_seqlens, max_seqlen = _packed_tokens((7, 12), 32)

    out = attn(x, cu_seqlens, max_seqlen)

    assert out.shape == x.shape
    torch.testing.assert_close(attn(x, cu_seqlens, max_seqlen, pos=pos), out)


def test_packed_attention_with_rope_requires_positions() -> None:
    attn = PackedAttention(32, 2, rope=AxialRoPE((3, 3, 2), base=100.0), use_flash_attn=False)
    x, _, cu_seqlens, max_seqlen = _packed_tokens((7, 12), 32)

    with pytest.raises(ValueError, match="pos"):
        attn(x, cu_seqlens, max_seqlen)


@pytest.mark.skipif(not (_CUDA_AVAILABLE and _FLASH_ATTN_AVAILABLE), reason="flash attention needs CUDA and flash-attn")
def test_packed_attention_flash_matches_scaled_dot_product() -> None:
    torch.manual_seed(0)
    attn = PackedAttention(64, 4, qk_norm=True, rope=AxialRoPE((3, 3, 2), base=100.0)).cuda()
    x, pos, cu_seqlens, max_seqlen = _packed_tokens((70, 120, 50), 64)
    x, pos, cu_seqlens = x.cuda(), pos.cuda(), cu_seqlens.cuda()

    flash = attn(x, cu_seqlens, max_seqlen, pos=pos)
    attn.use_flash_attn = False
    reference = attn(x, cu_seqlens, max_seqlen, pos=pos)

    torch.testing.assert_close(flash, reference, atol=2e-3, rtol=1e-2)


def test_packed_transformer_block_backward() -> None:
    torch.manual_seed(0)
    block = PackedTransformerBlock(32, 2, drop_path=0.1, rope=AxialRoPE((3, 3, 2), base=100.0), use_flash_attn=False)
    x, pos, cu_seqlens, max_seqlen = _packed_tokens((9, 4), 32)
    x.requires_grad_(True)

    out = block(x, cu_seqlens, max_seqlen, pos=pos)
    out.sum().backward()

    assert out.shape == x.shape
    assert x.grad is not None


def test_packed_transformer_block_builds_the_given_norm() -> None:
    block = PackedTransformerBlock(32, 2, norm=nn.RMSNorm, norm_kwargs={"eps": 1e-6}, use_flash_attn=False)
    x, _, cu_seqlens, max_seqlen = _packed_tokens((9, 4), 32)

    assert isinstance(block.norm1, nn.RMSNorm) and isinstance(block.norm2, nn.RMSNorm)
    assert block.norm1.eps == 1e-6
    assert block(x, cu_seqlens, max_seqlen).shape == x.shape


def test_packed_transformer_block_requires_a_norm() -> None:
    with pytest.raises(ValueError, match="normalization"):
        PackedTransformerBlock(32, 2, norm=None)


def test_packed_attention_without_flash_attn_installed_names_the_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("torch_pointcloud.layers.transformer._FLASH_ATTN_AVAILABLE", False)

    with pytest.raises(ImportError, match="use_flash_attn=False"):
        PackedAttention(32, 2, use_flash_attn=True)


def test_packed_attention_flash_on_cpu_tensors_names_the_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("torch_pointcloud.layers.transformer._FLASH_ATTN_AVAILABLE", True)
    attn = PackedAttention(32, 2, use_flash_attn=True)
    x, _, cu_seqlens, max_seqlen = _packed_tokens((7, 12), 32)

    with pytest.raises(ValueError, match="use_flash_attn=False"):
        attn(x, cu_seqlens, max_seqlen)


@pytest.mark.parametrize("norm", [pytest.param("layer_norm", id="string"), pytest.param(nn.LayerNorm, id="class")])
def test_packed_transformer_block_keeps_samples_independent(norm: Any) -> None:
    torch.manual_seed(0)
    block = PackedTransformerBlock(32, 2, norm=norm, use_flash_attn=False).eval()
    x, _, cu_seqlens, max_seqlen = _packed_tokens((10, 20), 32)
    changed = x.clone()
    changed[10:] = changed[10:] * 5 + 3

    out = block(x, cu_seqlens, max_seqlen)
    out_changed = block(changed, cu_seqlens, max_seqlen)

    torch.testing.assert_close(out[:10], out_changed[:10])
