import pytest
import torch

from torch_pointcloud.layers.grouped_vector_attention import GroupedVectorAttention


def test_grouped_vector_attention_forward() -> None:
    torch.manual_seed(0)
    num_points, num_neighbors = 64, 8
    x = torch.randn(num_points, 32)
    pos = torch.randn(num_points, 3)
    target = torch.arange(num_points).repeat_interleave(num_neighbors)
    source = torch.randint(0, num_points, (num_points * num_neighbors,))
    edge_index = torch.stack([source, target])

    attn = GroupedVectorAttention(channels=32, num_groups=4, pe_multiplier=True, pe_bias=True)
    out = attn(x, pos, edge_index)
    assert out.shape == (num_points, 32)


def test_grouped_vector_attention_channels_not_divisible_raises() -> None:
    with pytest.raises(ValueError, match="divisible"):
        GroupedVectorAttention(channels=30, num_groups=4)


def test_grouped_vector_attention_bipartite() -> None:
    torch.manual_seed(0)
    num_source, num_target, num_neighbors = 64, 16, 8
    x_source = torch.randn(num_source, 32)
    x_target = x_source[:num_target]
    pos_source = torch.randn(num_source, 3)
    pos_target = pos_source[:num_target]
    target = torch.arange(num_target).repeat_interleave(num_neighbors)
    source = torch.randint(0, num_source, (num_target * num_neighbors,))
    edge_index = torch.stack([source, target])

    attn = GroupedVectorAttention(channels=32, num_groups=4)
    out = attn((x_source, x_target), (pos_source, pos_target), edge_index)
    assert out.shape == (num_target, 32)
