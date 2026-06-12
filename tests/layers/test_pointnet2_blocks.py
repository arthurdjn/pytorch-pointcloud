import pytest
import torch
from torch_geometric.nn import MLP, knn_graph

from torch_pointcloud.layers.pointnet2_blocks import (
    GlobalSAModule,
    PointNet2Conv,
    PointNet2FeaturePropagation,
    PointNet2GlobalSetAbstraction,
    PointNet2SetAbstraction,
)
from torch_pointcloud.utils.imports import _TORCH_CLUSTER_AVAILABLE, _TORCH_SCATTER_AVAILABLE

# See: https://docs.pytest.org/en/stable/how-to/skipping.html#summary
pytestmark = pytest.mark.skipif(
    not (_TORCH_CLUSTER_AVAILABLE and _TORCH_SCATTER_AVAILABLE),
    reason="torch-cluster or torch-scatter is not installed",
)


def test_pointnet2_conv_forward() -> None:
    local_nn = MLP([3 + 3, 32], plain_last=False)
    conv = PointNet2Conv(local_nn=local_nn, add_self_loops=True)
    pos = torch.randn(64, 3)
    x = torch.randn(64, 3)
    batch = torch.cat([torch.zeros(32), torch.ones(32)]).long()
    edge_index = knn_graph(pos, k=8, batch=batch)
    out = conv(x, pos, edge_index)
    assert out.shape == (64, 32)


def test_pointnet2_set_abstraction_forward() -> None:
    sa = PointNet2SetAbstraction(
        spatial_dim=3,
        in_channels=3,
        channels=[32],
        ratio=0.5,
        radius=0.2,
        num_neighbors=16,
        dropout=0.0,
        act="relu",
        act_first=False,
        act_kwargs=None,
        norm="batch_norm",
        norm_kwargs=None,
        bias=True,
        add_self_loops=False,
        aggr="max",
    )
    pos = torch.randn(64, 3)
    x = torch.randn(64, 3)
    batch = torch.cat([torch.zeros(32), torch.ones(32)]).long()
    out_x, out_pos, out_batch = sa(x, pos, batch)
    assert out_x.shape[1] == 32
    assert out_x.shape[0] == out_pos.shape[0] == out_batch.shape[0]


def test_pointnet2_global_sa_forward() -> None:
    sa = PointNet2GlobalSetAbstraction(
        in_channels=8,
        channels=[16, 32],
        dropout=0.0,
        act="relu",
        act_first=False,
        act_kwargs=None,
        norm="batch_norm",
        norm_kwargs=None,
        bias=True,
        aggr="max",
    )
    x = torch.randn(64, 8)
    pos = torch.randn(64, 3)
    batch = torch.cat([torch.zeros(32), torch.ones(32)]).long()
    out_x, out_pos, out_batch = sa(x, pos, batch)
    assert out_x.shape == (2, 32)
    assert out_pos.shape == (2, 3)
    assert out_batch.shape == (2,)


def test_global_sa_module_use_pos() -> None:
    sa = GlobalSAModule(in_channels=8, channels=[16, 32], use_pos=True, pos_first=True)
    x = torch.randn(64, 8)
    pos = torch.randn(64, 3)
    batch = torch.cat([torch.zeros(32), torch.ones(32)]).long()
    out_x, out_pos, out_batch = sa(x, pos, batch)
    assert sa.mlp.channel_list[0] == 8 + 3
    assert out_x.shape == (2, 32)
    assert out_pos.shape == (2, 3)
    assert out_batch.tolist() == [0, 1]


def test_pointnet2_feature_propagation_forward() -> None:
    fp = PointNet2FeaturePropagation(
        channels=[16 + 8, 32, 32],
        k=3,
        dropout=0.0,
        act="relu",
        act_first=False,
        act_kwargs=None,
        norm="batch_norm",
        norm_kwargs=None,
        bias=True,
        plain_last=True,
        weighting="inverse",
    )
    pos = torch.randn(20, 3)
    x = torch.randn(20, 16)
    batch = torch.cat([torch.zeros(8), torch.ones(12)]).long()
    pos_skip = torch.randn(40, 3)
    x_skip = torch.randn(40, 8)
    batch_skip = torch.cat([torch.zeros(16), torch.ones(24)]).long()
    out_x, out_pos, out_batch = fp(x, pos, batch, x_skip, pos_skip, batch_skip)
    assert out_x.shape == (40, 32)
    assert out_pos.shape == pos_skip.shape
    assert out_batch.shape == batch_skip.shape
