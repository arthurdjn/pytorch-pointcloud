from typing import Tuple

import pytest
import torch
from torch import Tensor

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
    attn = PackedAttention(32, 2, rope_frequencies=(3, 3, 2), rope_base=100.0, qk_norm=True, use_flash_attn=False)
    x, pos, cu_seqlens, max_seqlen = _packed_tokens((7, 12, 5), 32)

    out = attn(x, pos, cu_seqlens, max_seqlen)

    for start, end in zip(cu_seqlens[:-1].tolist(), cu_seqlens[1:].tolist()):
        alone = attn(x[start:end], pos[start:end], torch.tensor([0, end - start], dtype=torch.int32), end - start)
        torch.testing.assert_close(out[start:end], alone)


def test_packed_attention_rejects_mismatched_rope_frequencies() -> None:
    with pytest.raises(ValueError, match="rope_frequencies"):
        PackedAttention(32, 2, rope_frequencies=(4, 4, 4), rope_base=100.0)


@pytest.mark.skipif(not (_CUDA_AVAILABLE and _FLASH_ATTN_AVAILABLE), reason="flash attention needs CUDA and flash-attn")
def test_packed_attention_flash_matches_scaled_dot_product() -> None:
    torch.manual_seed(0)
    attn = PackedAttention(64, 4, rope_frequencies=(3, 3, 2), rope_base=100.0, qk_norm=True).cuda()
    x, pos, cu_seqlens, max_seqlen = _packed_tokens((70, 120, 50), 64)
    x, pos, cu_seqlens = x.cuda(), pos.cuda(), cu_seqlens.cuda()

    flash = attn(x, pos, cu_seqlens, max_seqlen)
    attn.use_flash_attn = False
    reference = attn(x, pos, cu_seqlens, max_seqlen)

    torch.testing.assert_close(flash, reference, atol=2e-3, rtol=1e-2)


def test_packed_transformer_block_backward() -> None:
    torch.manual_seed(0)
    block = PackedTransformerBlock(
        32, 2, rope_frequencies=(3, 3, 2), rope_base=100.0, drop_path=0.1, use_flash_attn=False
    )
    x, pos, cu_seqlens, max_seqlen = _packed_tokens((9, 4), 32)
    x.requires_grad_(True)

    out = block(x, pos, cu_seqlens, max_seqlen)
    out.sum().backward()

    assert out.shape == x.shape
    assert x.grad is not None
