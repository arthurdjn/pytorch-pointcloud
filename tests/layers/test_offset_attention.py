import torch

from torch_pointcloud.layers.offset_attention import OffsetAttention


def test_offset_attention_forward_shape() -> None:
    attn = OffsetAttention(32)
    x = torch.randn(2, 16, 32)
    assert attn(x).shape == (2, 16, 32)
    assert attn.qk.out_features == 8


def test_offset_attention_matches_dense_formula() -> None:
    torch.manual_seed(0)
    attn = OffsetAttention(16, qk_channels=4).eval()
    x = torch.randn(2, 10, 16)
    q = attn.qk(x)
    energy = (q @ q.transpose(1, 2)).softmax(dim=-1)
    energy = energy / (1e-9 + energy.sum(dim=1, keepdim=True))
    offset = x - torch.einsum("bij,bic->bjc", energy, attn.v(x))
    expected = x + torch.relu(attn.norm(attn.proj(offset).flatten(0, 1)).view_as(x))
    assert torch.allclose(attn(x), expected, atol=1e-6)


def test_offset_attention_without_norm_and_act() -> None:
    attn = OffsetAttention(8, act=None, norm=None)
    assert isinstance(attn.norm, torch.nn.Identity) and isinstance(attn.act, torch.nn.Identity)
    assert attn(torch.randn(1, 4, 8)).shape == (1, 4, 8)
