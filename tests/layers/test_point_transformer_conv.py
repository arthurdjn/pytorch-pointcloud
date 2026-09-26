from typing import Dict

import pytest
import torch
from torch import Tensor

from torch_pointcloud.layers.point_transformer_conv import PointTransformerConv


@pytest.fixture
def data() -> Dict[str, Tensor]:
    torch.manual_seed(42)
    lengths = torch.tensor([256, 512])
    pos = torch.randn(int(lengths.sum()), 3)
    features = torch.randn(int(lengths.sum()), 3)
    batch = torch.repeat_interleave(torch.arange(len(lengths)), lengths)

    # Dummy edge_index connecting each point to 16 nearest indices
    row = torch.arange(len(pos)).repeat_interleave(16)
    cumsum = torch.cat([torch.tensor([0]), torch.cumsum(lengths, dim=0)])
    col = torch.cat([torch.arange(int(lengths[i])).repeat(16) + cumsum[i] for i in range(len(lengths))])
    edge_index = torch.stack([row, col])

    return dict(
        features=features,
        pos=pos,
        batch=batch,
        edge_index=edge_index,
    )


def test_point_transformer_conv(data: Dict[str, Tensor]) -> None:
    conv = PointTransformerConv(
        spatial_dim=3,
        in_channels=3,
        out_channels=32,
    )

    output = conv(data["features"], data["pos"], data["edge_index"])
    assert output.shape == (len(data["pos"]), 32)
