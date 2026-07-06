from typing import Dict

import pytest
import torch
from torch import Tensor
from torch_geometric.nn import MLP, radius_graph

from torch_pointcloud.layers.pointnext_blocks import PointNeXtConv, PointNeXtResidualBlock, PointNeXtSetAbstraction
from torch_pointcloud.utils.imports import _TORCH_CLUSTER_AVAILABLE, _TORCH_SCATTER_AVAILABLE

pytestmark = pytest.mark.skipif(
    not _TORCH_CLUSTER_AVAILABLE and not _TORCH_SCATTER_AVAILABLE,
    reason="torch-cluster or torch-scatter is not installed",
)


@pytest.fixture
def data() -> Dict[str, Tensor]:
    torch.manual_seed(42)
    lengths = torch.tensor([256, 512])
    pos = torch.randn(int(lengths.sum()), 3)
    features = torch.randn(int(lengths.sum()), 3)
    batch = torch.repeat_interleave(torch.arange(len(lengths)), lengths)

    edge_index = radius_graph(pos, r=0.1, batch=batch, max_num_neighbors=16)

    return dict(
        features=features,
        pos=pos,
        batch=batch,
        edge_index=edge_index,
    )


def test_pointnext_conv_basic(data: Dict[str, Tensor]) -> None:
    """Test basic PointNeXtConv functionality."""
    local_nn = MLP([3 + 3, 32])  # features + spatial_dim -> out_channels
    conv = PointNeXtConv(local_nn=local_nn, add_self_loops=True)

    output = conv(data["features"], data["pos"], data["edge_index"])
    assert output.shape == (len(data["pos"]), 32)
    assert output.dtype == data["features"].dtype


def test_pointnext_conv_with_pos_divisor(data: Dict[str, Tensor]) -> None:
    """Test PointNeXtConv with position normalization."""
    local_nn = MLP([3 + 3, 32])
    conv = PointNeXtConv(local_nn=local_nn, add_self_loops=True)

    output = conv(data["features"], data["pos"], data["edge_index"], pos_divisor=0.1)
    assert output.shape == (len(data["pos"]), 32)

    output_no_div = conv(data["features"], data["pos"], data["edge_index"])
    assert output_no_div.shape == (len(data["pos"]), 32)


def test_pointnext_conv_with_none_features(data: Dict[str, Tensor]) -> None:
    """Test PointNeXtConv with None features (position-only)."""
    local_nn = MLP([0 + 3, 32])  # no features, only spatial info
    conv = PointNeXtConv(local_nn=local_nn, add_self_loops=True)

    output = conv(None, data["pos"], data["edge_index"])
    assert output.shape == (len(data["pos"]), 32)


def test_pointnext_set_abstraction_basic(data: Dict[str, Tensor]) -> None:
    """Test basic PointNeXtSetAbstraction functionality."""
    sa = PointNeXtSetAbstraction(
        spatial_dim=3,
        in_channels=3,
        channels=[32],
        ratio=0.5,
        radius=0.1,
        num_neighbors=16,
    )

    out_features, out_pos, out_batch = sa(data["features"], data["pos"], data["batch"])

    assert out_features.shape[1] == 32
    assert out_pos.shape[0] == out_features.shape[0] == out_batch.shape[0]
    assert out_pos.shape[1] == 3  # spatial_dim
    assert out_batch.shape[0] <= data["batch"].shape[0]  # downsampled


def test_pointnext_set_abstraction_multiple_radii(data: Dict[str, Tensor]) -> None:
    """Test PointNeXtSetAbstraction with multiple radii."""
    sa = PointNeXtSetAbstraction(
        spatial_dim=3,
        in_channels=3,
        channels=[[32], [64]],
        ratio=0.5,
        radius=[0.1, 0.2],
        num_neighbors=[16, 32],
    )

    out_features, out_pos, out_batch = sa(data["features"], data["pos"], data["batch"])

    # Features from both radii are concatenated.
    assert out_features.shape[1] == 32 + 64
    assert out_pos.shape[0] == out_features.shape[0] == out_batch.shape[0]


def test_pointnext_set_abstraction_skip_connections(data: Dict[str, Tensor]) -> None:
    """Test that skip connections are properly configured."""
    sa = PointNeXtSetAbstraction(
        spatial_dim=3,
        in_channels=3,
        channels=[32],
        ratio=0.5,
        radius=0.1,
        num_neighbors=16,
    )

    assert len(sa.skip_convs) == 1
    assert sa.skip_convs[0] is not None

    out_features, out_pos, out_batch = sa(data["features"], data["pos"], data["batch"])
    assert out_features.shape[1] == 32


def test_pointnext_residual_block_basic(data: Dict[str, Tensor]) -> None:
    """Test basic PointNeXtResidualBlock functionality."""
    block = PointNeXtResidualBlock(
        spatial_dim=3,
        channels=3,
        expansion=4,
        ratio=0.5,
        radius=0.1,
        num_neighbors=16,
    )

    output = block(data["features"], data["pos"], data["batch"])
    assert output.shape == data["features"].shape
    assert output.dtype == data["features"].dtype
