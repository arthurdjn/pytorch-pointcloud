import pytest
import torch
from torch_geometric.nn import MLP, knn_graph

from torch_pointcloud.layers.pointconv import PointConv, PointConvDensity
from torch_pointcloud.utils.imports import _TORCH_CLUSTER_AVAILABLE, _TORCH_SCATTER_AVAILABLE

# See: https://docs.pytest.org/en/stable/how-to/skipping.html#summary
pytestmark = pytest.mark.skipif(
    not (_TORCH_CLUSTER_AVAILABLE and _TORCH_SCATTER_AVAILABLE),
    reason="torch-cluster or torch-scatter is not installed",
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
