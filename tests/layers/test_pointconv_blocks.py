import pytest
import torch
from torch_geometric.nn import MLP

from torch_pointcloud.layers.fps import FPS
from torch_pointcloud.layers.pointconv_blocks import (
    PointConv,
    PointConvDensity,
    PointConvDensityGlobalSetAbstraction,
    PointConvDensitySetAbstraction,
    PointConvGlobalSetAbstraction,
    PointConvSetAbstraction,
)
from torch_pointcloud.ops.cluster import knn_graph
from torch_pointcloud.utils.imports import _PYG_LIB_AVAILABLE

# See: https://docs.pytest.org/en/stable/how-to/skipping.html#summary
pytestmark = pytest.mark.skipif(
    not _PYG_LIB_AVAILABLE,
    reason="pyg-lib is not installed",
)


def test_point_conv_forward() -> None:
    local_nn = MLP([3 + 3, 16, 32], plain_last=False)
    weight_nn = MLP([3, 16, 8], plain_last=False)
    conv = PointConv(local_nn=local_nn, weight_nn=weight_nn, add_self_loops=True)
    pos = torch.randn(64, 3)
    x = torch.randn(64, 3)
    batch = torch.cat([torch.zeros(32), torch.ones(32)]).long()
    edge_index = knn_graph(pos, k=8, batch=batch)
    out = conv(x, pos, edge_index)
    assert out.shape[0] == 64
    assert out.shape[1] == 32 * 8


def test_point_conv_density_forward() -> None:
    local_nn = MLP([3 + 3, 16, 32], plain_last=False)
    weight_nn = MLP([3, 16, 8], plain_last=False)
    density_nn = MLP([1, 8, 1], plain_last=False)
    conv = PointConvDensity(local_nn=local_nn, weight_nn=weight_nn, density_nn=density_nn, add_self_loops=True)
    pos = torch.randn(64, 3)
    x = torch.randn(64, 3)
    batch = torch.cat([torch.zeros(32), torch.ones(32)]).long()
    edge_index = knn_graph(pos, k=8, batch=batch)
    density = torch.rand(64, 1)
    out = conv(x, pos, edge_index, density)
    assert out.shape[0] == 64
    assert out.shape[1] == 32 * 8


def test_pointconv_set_abstraction_forward() -> None:
    sa = PointConvSetAbstraction(
        in_channels=8,
        num_neighbors=16,
        channels=[16, 32],
        weight_channels=[8, 8],
        expansion=8,
        act="relu",
        norm="batch_norm",
        bias=True,
        spatial_dim=3,
        downsample=FPS(ratio=0.5, random_start=False),
    )
    pos = torch.randn(64, 3)
    x = torch.randn(64, 8)
    batch = torch.cat([torch.zeros(32), torch.ones(32)]).long()
    out_x, out_pos, out_batch = sa(x, pos, batch)
    assert out_x.shape[1] == 32
    assert out_x.shape[0] == out_pos.shape[0] == out_batch.shape[0] == 32


def test_pointconv_density_set_abstraction_forward() -> None:
    sa = PointConvDensitySetAbstraction(
        in_channels=8,
        num_neighbors=16,
        channels=[16, 32],
        bandwidth=0.5,
        weight_channels=[8, 8],
        density_channels=[16, 8],
        expansion=8,
        act="relu",
        norm="batch_norm",
        bias=True,
        spatial_dim=3,
        downsample=FPS(ratio=0.5, random_start=False),
    )
    pos = torch.randn(64, 3)
    x = torch.randn(64, 8)
    batch = torch.cat([torch.zeros(32), torch.ones(32)]).long()
    out_x, out_pos, out_batch = sa(x, pos, batch)
    assert out_x.shape[1] == 32
    assert out_x.shape[0] == out_pos.shape[0] == out_batch.shape[0]


def test_pointconv_global_set_abstraction_forward() -> None:
    sa = PointConvGlobalSetAbstraction(
        in_channels=8,
        channels=[16, 32],
        weight_channels=[8, 8],
        expansion=8,
        act="relu",
        norm="batch_norm",
        bias=True,
        aggr="mean",
        spatial_dim=3,
    )
    pos = torch.randn(64, 3)
    x = torch.randn(64, 8)
    batch = torch.cat([torch.zeros(32), torch.ones(32)]).long()
    out_x, out_pos, out_batch = sa(x, pos, batch)
    assert out_x.shape == (2, 32)
    assert out_pos.shape == (2, 3)


def test_pointconv_density_global_set_abstraction_forward() -> None:
    sa = PointConvDensityGlobalSetAbstraction(
        in_channels=8,
        channels=[16, 32],
        bandwidth=0.5,
        weight_channels=[8, 8],
        density_channels=[16, 8],
        expansion=8,
        act="relu",
        norm="batch_norm",
        bias=True,
        pool="mean",
        spatial_dim=3,
    )
    pos = torch.randn(64, 3)
    x = torch.randn(64, 8)
    batch = torch.cat([torch.zeros(32), torch.ones(32)]).long()
    out_x, out_pos, out_batch = sa(x, pos, batch)
    assert out_x.shape == (2, 32)
