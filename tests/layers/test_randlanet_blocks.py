from typing import Any, Dict

import pytest
import torch
from torch import Tensor

from torch_pointcloud.layers.randlanet_blocks import AttentivePooling, LocalFeatureAggregation, RandLANetResidualBlock
from torch_pointcloud.ops.cluster import knn_graph
from torch_pointcloud.utils.imports import _PYG_LIB_AVAILABLE

# See: https://docs.pytest.org/en/stable/how-to/skipping.html#summary
pytestmark = pytest.mark.skipif(
    not _PYG_LIB_AVAILABLE,
    reason="pyg-lib is not installed",
)


@pytest.fixture
def data() -> Dict[str, Tensor]:
    torch.manual_seed(42)
    lengths = torch.tensor([256, 512])
    pos = torch.randn(int(lengths.sum()), 3)
    features = torch.randn(int(lengths.sum()), 6)
    batch = torch.repeat_interleave(torch.arange(len(lengths)), lengths)
    return dict(features=features, pos=pos, batch=batch)


@pytest.fixture
def mlp_kwargs() -> Dict[str, Any]:
    return dict(act="relu", norm="batch_norm", bias=False)


def test_randlanet_attentive_pooling_bipartite(mlp_kwargs: Dict[str, Any]) -> None:
    torch.manual_seed(0)
    num_source, num_target, num_neighbors = 64, 16, 8
    x = torch.randn(num_source, 8)
    target = torch.arange(num_target).repeat_interleave(num_neighbors)
    source = torch.randint(0, num_source, (num_target * num_neighbors,))
    edge_index = torch.stack([source, target])
    edge_attr = torch.randn(edge_index.size(1), 4)

    pooling = AttentivePooling(in_channels=12, out_channels=6, **mlp_kwargs)
    out = pooling((x, x[:num_target]), edge_index, edge_attr)
    assert out.shape == (num_target, 6)


def test_randlanet_local_feature_aggregation(data: Dict[str, Tensor], mlp_kwargs: Dict[str, Any]) -> None:
    lfa = LocalFeatureAggregation(d_out=8, **mlp_kwargs)
    edge_index = knn_graph(data["pos"], 8, batch=data["batch"], loop=True)
    out = lfa(data["features"][:, :4], data["pos"], edge_index)
    assert out.shape == (data["features"].shape[0], 8)


def test_randlanet_residual_block(data: Dict[str, Tensor], mlp_kwargs: Dict[str, Any]) -> None:
    block = RandLANetResidualBlock(d_in=6, d_out=8, num_neighbors=8, **mlp_kwargs)
    x, pos, batch = block(data["features"], data["pos"], data["batch"])
    assert x.shape == (data["features"].shape[0], 16)
    assert pos.shape == data["pos"].shape
    assert batch.shape == data["batch"].shape


def test_randlanet_residual_block_odd_d_out_raises(mlp_kwargs: Dict[str, Any]) -> None:
    with pytest.raises(ValueError, match="must be even"):
        RandLANetResidualBlock(d_in=6, d_out=7, num_neighbors=8, **mlp_kwargs)
