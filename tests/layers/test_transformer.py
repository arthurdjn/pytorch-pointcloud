import pytest
import torch

from torch_pointcloud.layers.transformer import TransformerDecoderLayer, TransformerEncoderLayer


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
