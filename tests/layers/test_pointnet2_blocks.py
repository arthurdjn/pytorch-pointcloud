import pytest
import torch
from torch_geometric.nn import MLP

from torch_pointcloud.layers.pointnet2_blocks import (
    PointNet2Conv,
    PointNet2FeaturePropagation,
    PointNet2GlobalSetAbstraction,
    PointNet2QuerySetAbstraction,
    PointNet2SetAbstraction,
)
from torch_pointcloud.ops.cluster import knn_graph
from torch_pointcloud.utils.imports import _PYG_LIB_AVAILABLE

# See: https://docs.pytest.org/en/stable/how-to/skipping.html#summary
pytestmark = pytest.mark.skipif(
    not _PYG_LIB_AVAILABLE,
    reason="pyg-lib is not installed",
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
        in_channels=3,
        channels=[32],
        ratio=0.5,
        radii=0.2,
        num_neighbors=16,
        spatial_dim=3,
        dropout=0.0,
        act="relu",
        act_first=False,
        act_kwargs=None,
        norm="batch_norm",
        norm_kwargs=None,
        bias=True,
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


def test_pointnet2_set_abstraction_multi_scale_concatenates_scales() -> None:
    sa = PointNet2SetAbstraction(
        in_channels=3, channels=[[16], [32]], ratio=0.5, radii=[0.1, 0.2], num_neighbors=[8, 16]
    )
    pos = torch.randn(64, 3)
    x = torch.randn(64, 3)
    batch = torch.cat([torch.zeros(32), torch.ones(32)]).long()
    out_x, _, _ = sa(x, pos, batch)
    assert out_x.shape[1] == 16 + 32


def test_pointnet2_set_abstraction_num_points_and_precomputed_idx() -> None:
    sa = PointNet2SetAbstraction(
        in_channels=1,
        channels=[16, 16],
        num_points=64,
        radii=0.4,
        num_neighbors=16,
        pos_first=True,
    ).eval()
    pos = torch.rand(500, 3)
    x = torch.rand(500, 1)
    batch = torch.zeros(500, dtype=torch.long)
    with torch.no_grad():
        new_x, new_pos, new_batch = sa(x, pos, batch)
    assert new_x.shape == (64, 16)
    assert new_pos.shape == (64, 3)
    assert new_batch.shape == (64,)
    # A precomputed sampling index is honored verbatim.
    idx = torch.arange(64)
    with torch.no_grad():
        nx, npos, _ = sa(x, pos, batch, idx)
    assert torch.equal(npos, pos[idx])
    assert nx.shape == (64, 16)


def test_pointnet2_set_abstraction_requires_exactly_one_sampling_spec() -> None:
    with pytest.raises(ValueError, match="ratio"):
        PointNet2SetAbstraction(in_channels=1, channels=[16], radii=0.4, num_neighbors=16)
    with pytest.raises(ValueError, match="ratio"):
        PointNet2SetAbstraction(in_channels=1, channels=[16], ratio=0.5, num_points=64, radii=0.4, num_neighbors=16)


def test_pointnet2_conv_without_pos_requires_features() -> None:
    conv = PointNet2Conv(local_nn=MLP([3, 8]), add_self_loops=False, use_pos=False)
    pos = torch.randn(4, 3)
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]])
    with pytest.raises(ValueError, match="use_pos"):
        conv(None, pos, edge_index)


def test_pointnet2_global_set_abstraction_use_pos() -> None:
    sa = PointNet2GlobalSetAbstraction(in_channels=8, channels=[16, 32], use_pos=True, pos_first=True)
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


def test_pointnet2_query_set_abstraction_concatenates_scales_at_query_points() -> None:
    torch.manual_seed(0)
    sa = PointNet2QuerySetAbstraction(2, [[8], [4, 6]], radii=[0.5, 1.0], num_neighbors=[4, 8]).eval()
    pos = torch.rand(40, 3)
    x = torch.randn(40, 2)
    batch = torch.cat([torch.zeros(20), torch.ones(20)]).long()
    pos_query = torch.rand(6, 3)
    batch_query = torch.tensor([0, 0, 0, 1, 1, 1])
    out = sa(x, pos, batch, pos_query, batch_query)
    assert sa.out_channels == 14
    assert out.shape == (6, 14)


def test_pointnet2_query_set_abstraction_empty_ball_pools_the_mlp_of_zero() -> None:
    torch.manual_seed(0)
    sa = PointNet2QuerySetAbstraction(2, [[4]], radii=[0.1], num_neighbors=[4]).eval()
    pos = torch.zeros(3, 3)
    x = torch.randn(3, 2)
    batch = torch.zeros(3, dtype=torch.long)
    pos_query = torch.tensor([[5.0, 5.0, 5.0]])
    with torch.no_grad():
        out = sa(x, pos, batch, pos_query, torch.zeros(1, dtype=torch.long))
        expected = sa.mlps[0](torch.zeros(1, 5))
    assert torch.allclose(out, expected)


def test_pointnet2_query_set_abstraction_keeps_the_first_neighbors_in_index_order() -> None:
    sa = PointNet2QuerySetAbstraction(1, [[1]], radii=[1.0], num_neighbors=[2], norm=None, act=None).eval()
    mlp = sa.mlps[0]
    assert isinstance(mlp, MLP)
    weight = mlp.lins[0].weight
    assert isinstance(weight, torch.Tensor)
    with torch.no_grad():
        weight.copy_(torch.tensor([[0.0, 0.0, 0.0, 1.0]]))
    pos = torch.zeros(4, 3)
    x = torch.tensor([[1.0], [2.0], [9.0], [8.0]])
    batch = torch.zeros(4, dtype=torch.long)
    with torch.no_grad():
        out = sa(x, pos, batch, torch.zeros(1, 3), torch.zeros(1, dtype=torch.long))
    # The ball holds all four points, but only the first two in index order are pooled.
    assert out.item() == 2.0


def test_pointnet2_query_set_abstraction_rejects_mismatched_scales() -> None:
    with pytest.raises(ValueError, match="one entry per scale"):
        PointNet2QuerySetAbstraction(2, [[4]], radii=[0.1, 0.2], num_neighbors=[4])
