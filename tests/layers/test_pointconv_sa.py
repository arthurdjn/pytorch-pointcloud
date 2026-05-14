import pytest
import torch

from torch_pointcloud.layers.fps import FPS
from torch_pointcloud.layers.pointconv_sa import (
    PointConvDensityGlobalSetAbstraction,
    PointConvDensitySetAbstraction,
    PointConvGlobalSetAbstraction,
    PointConvSetAbstraction,
)
from torch_pointcloud.utils.imports import _TORCH_CLUSTER_AVAILABLE, _TORCH_SCATTER_AVAILABLE

# See: https://docs.pytest.org/en/stable/how-to/skipping.html#summary
pytestmark = pytest.mark.skipif(
    not (_TORCH_CLUSTER_AVAILABLE and _TORCH_SCATTER_AVAILABLE),
    reason="torch-cluster or torch-scatter is not installed",
)


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
