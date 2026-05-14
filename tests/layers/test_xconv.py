import pytest
import torch
from torch_geometric.nn import knn_graph

from torch_pointcloud.layers.xconv import XConv
from torch_pointcloud.utils.imports import _TORCH_CLUSTER_AVAILABLE

# See: https://docs.pytest.org/en/stable/how-to/skipping.html#summary
pytestmark = pytest.mark.skipif(
    not _TORCH_CLUSTER_AVAILABLE,
    reason="torch-cluster is not installed",
)


def test_xconv_forward() -> None:
    k = 8
    conv = XConv(
        in_channels=16,
        out_channels=32,
        spatial_dim=3,
        kernel_size=k,
        hidden_channels=8,
        depth_multiplier=2,
        dilation=1,
        act="elu",
        act_kwargs=None,
        norm="batch_norm",
        norm_kwargs=None,
        bias=True,
        add_self_loops=False,
    )
    torch.manual_seed(0)
    pos = torch.randn(64, 3)
    x = torch.randn(64, 16)
    batch = torch.cat([torch.zeros(32), torch.ones(32)]).long()
    edge_index = knn_graph(pos, k=k, batch=batch)
    out = conv(x, pos, edge_index)
    assert out.shape == (64, 32)
