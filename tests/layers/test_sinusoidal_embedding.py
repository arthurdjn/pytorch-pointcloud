import pytest
import torch

from torch_pointcloud.layers import sinusoidal_embedding


def test_sinusoidal_embedding_interleaves_sin_and_cos() -> None:
    indices = torch.tensor([0.0, 1.5])
    embedding = sinusoidal_embedding(indices, dim=4)
    assert embedding.shape == (2, 4)
    assert torch.allclose(embedding[0], torch.tensor([0.0, 1.0, 0.0, 1.0]))
    assert embedding[1, 0].item() == pytest.approx(torch.sin(torch.tensor(1.5)).item())
    with pytest.raises(ValueError, match="even"):
        sinusoidal_embedding(indices, dim=3)
