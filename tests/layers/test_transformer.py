import pytest
import torch

from torch_pointcloud.layers.transformer import (
    Attention,
    CrossAttention,
    TransformerDecoderLayer,
    TransformerEncoderLayer,
)


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


def test_cross_attention_matches_self_attention_on_its_own_tokens() -> None:
    torch.manual_seed(0)
    attn = Attention(32, num_heads=4, qkv_bias=True).eval()
    cross = CrossAttention(32, num_heads=4, qkv_bias=True).eval()
    with torch.no_grad():
        cross.q.weight.copy_(attn.qkv.weight[:32])
        cross.q.bias.copy_(attn.qkv.bias[:32])
        cross.kv.weight.copy_(attn.qkv.weight[32:])
        cross.kv.bias.copy_(attn.qkv.bias[32:])
        cross.proj.load_state_dict(attn.proj.state_dict())
    x = torch.randn(2, 10, 32)

    assert torch.allclose(cross(x, x), attn(x), atol=1e-6)


def test_cross_attention_output_follows_the_queries() -> None:
    attn = CrossAttention(32, num_heads=4).eval()
    queries = torch.randn(2, 3, 32)
    tokens = torch.randn(2, 10, 32)

    out = attn(queries, tokens)

    assert out.shape == (2, 3, 32)
    assert torch.allclose(attn(queries[:, :1], tokens), out[:, :1], atol=1e-6)
