import pytest
import torch

from torch_pointcloud.layers.tnet import DynamicTNet, TNet
from torch_pointcloud.utils.imports import _TORCH_CLUSTER_AVAILABLE, _TORCH_SCATTER_AVAILABLE

# See: https://docs.pytest.org/en/stable/how-to/skipping.html#summary
pytestmark = pytest.mark.skipif(
    not (_TORCH_CLUSTER_AVAILABLE and _TORCH_SCATTER_AVAILABLE),
    reason="torch-cluster or torch-scatter is not installed",
)


def test_tnet_forward() -> None:
    tnet = TNet(
        local_channels=[16, 32],
        global_channels=[16],
        k=3,
        act="relu",
        act_kwargs=None,
        act_first=False,
        norm="batch_norm",
        norm_kwargs=None,
        bias=True,
        dropout=0.0,
        aggr="max",
    )
    x = torch.randn(20, 3)
    batch = torch.cat([torch.zeros(8), torch.ones(12)]).long()
    out = tnet(x, batch)
    assert out.shape == x.shape


def test_tnet_reset_parameters() -> None:
    tnet = TNet(local_channels=[16], global_channels=[8], k=3)
    tnet.transform.weight.data.fill_(2.0)
    tnet.reset_parameters()
    torch.testing.assert_close(tnet.transform.weight, torch.zeros_like(tnet.transform.weight))


def test_dynamic_tnet_forward() -> None:
    tnet = DynamicTNet(
        edge_channels=[16],
        local_channels=[32],
        global_channels=[16],
        k=3,
        num_neighbors=4,
        act="relu",
        act_kwargs=None,
        act_first=False,
        norm="batch_norm",
        norm_kwargs=None,
        bias=True,
        dropout=0.0,
        aggr="max",
    )
    x = torch.randn(20, 3)
    batch = torch.cat([torch.zeros(8), torch.ones(12)]).long()
    out = tnet(x, batch)
    assert out.shape == x.shape
