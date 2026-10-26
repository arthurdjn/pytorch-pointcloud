import math

import pytest
import torch

from torch_pointcloud.layers.positional_encoding import PositionEmbeddingSine


def test_position_embedding_sine_interleaves_sine_and_cosine_per_axis() -> None:
    embed = PositionEmbeddingSine(8, spatial_dim=2, temperature=100.0)

    out = embed(torch.tensor([[0.5, -1.0]]))

    angles = torch.tensor([0.5, 0.5, 0.05, 0.05, -1.0, -1.0, -0.1, -0.1])
    expected = torch.where(torch.arange(8) % 2 == 0, angles.sin(), angles.cos())
    assert torch.allclose(out[0], expected, atol=1e-6)


@pytest.mark.parametrize(
    "embed_dim,padding",
    [
        pytest.param(384, 0, id="multiple"),
        pytest.param(20, 2, id="padded"),
    ],
)
def test_position_embedding_sine_zero_pads_the_leftover_channels(embed_dim: int, padding: int) -> None:
    embed = PositionEmbeddingSine(embed_dim)
    out = embed(torch.randn(2, 5, 3))

    assert out.shape == (2, 5, embed_dim)
    assert embed.padding == padding
    if padding:
        assert torch.all(out[..., -padding:] == 0)


def test_position_embedding_sine_scale_multiplies_the_coordinates() -> None:
    pos = torch.randn(4, 3)

    scaled = PositionEmbeddingSine(12, scale=2 * math.pi)(pos)

    assert torch.allclose(scaled, PositionEmbeddingSine(12)(pos * 2 * math.pi), atol=1e-6)
