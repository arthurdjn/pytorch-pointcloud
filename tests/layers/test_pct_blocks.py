import pytest
import torch

from torch_pointcloud.layers.pct_blocks import OffsetAttention, PCTNeighborEmbedding
from torch_pointcloud.utils.imports import _PYG_LIB_AVAILABLE


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


@pytest.mark.skipif(not _PYG_LIB_AVAILABLE, reason="pyg-lib is not installed")
def test_pct_neighbor_embedding_samples_per_sample() -> None:
    torch.manual_seed(0)
    lengths = torch.tensor([128, 96])
    pos = torch.randn(int(lengths.sum()), 3)
    x = torch.randn(int(lengths.sum()), 3)
    batch = torch.repeat_interleave(torch.arange(len(lengths)), lengths)
    block = PCTNeighborEmbedding(3, [16, 16], num_points=20, num_neighbors=8)
    x, pos, batch = block(x, pos, batch)
    assert x.shape == (40, 16) and pos.shape == (40, 3)
    assert torch.equal(batch, torch.arange(2).repeat_interleave(20))
