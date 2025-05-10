from typing import Dict

import pytest
import torch
from torch import Tensor

from torch_pointcloud.models.kpconv import EncoderBlock, KPConv, KPConvBlock, KPConvNetClassification, KPResidualBlock
from torch_pointcloud.utils.imports import _TORCH_CLUSTER_AVAILABLE, _TORCH_SCATTER_AVAILABLE


@pytest.fixture
def data() -> Dict[str, Tensor]:
    lengths = torch.tensor([256, 512])
    coords = torch.randn(int(lengths.sum()), 3)
    features = torch.randn(int(lengths.sum()), 3)
    batch = torch.repeat_interleave(torch.arange(len(lengths)), lengths)

    # Create dummy edge_index connecting each point to 16 nearest indices
    num_edges = len(coords) * 16
    row = torch.arange(len(coords)).repeat_interleave(16)
    col = torch.remainder(torch.arange(num_edges), len(coords))
    edge_index = torch.stack([row, col])

    return dict(
        features=features,
        coords=coords,
        batch=batch,
        edge_index=edge_index,
    )


@pytest.mark.skipif(not _TORCH_CLUSTER_AVAILABLE, reason="torch-cluster is not installed")
def test_kpconv_module(data: Dict[str, Tensor]) -> None:
    conv = KPConv(
        spatial_dim=3,
        in_channels=3,
        out_channels=32,
        kernel_size=15,
        kp_radius=0.1,
        kp_sigma=0.1,
    )

    output = conv(data["features"], data["coords"], data["coords"], data["edge_index"])
    assert output.shape == (len(data["coords"]), 32)

    # Test with deformable and modulated options
    conv = KPConv(
        spatial_dim=3,
        in_channels=3,
        out_channels=32,
        kernel_size=15,
        kp_radius=0.1,
        kp_sigma=0.1,
        deformable=True,
        modulated=True,
    )

    output = conv(data["features"], data["coords"], data["coords"], data["edge_index"])
    assert output.shape == (len(data["coords"]), 32)
