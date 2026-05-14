import pytest
import torch

from torch_pointcloud.layers.fps import FPS
from torch_pointcloud.utils.imports import _TORCH_CLUSTER_AVAILABLE

# See: https://docs.pytest.org/en/stable/how-to/skipping.html#summary
pytestmark = pytest.mark.skipif(
    not _TORCH_CLUSTER_AVAILABLE,
    reason="torch-cluster is not installed",
)


def test_fps_basic() -> None:
    fps = FPS(ratio=0.5, random_start=False)
    pos = torch.randn(100, 3)
    batch = torch.zeros(100, dtype=torch.long)
    idx = fps(pos, batch)
    assert idx.shape == (50,)
    assert idx.dtype == torch.long


def test_fps_multiple_batches() -> None:
    fps = FPS(ratio=0.25, random_start=False)
    pos = torch.randn(200, 3)
    batch = torch.cat([torch.zeros(100), torch.ones(100)]).long()
    idx = fps(pos, batch)
    assert idx.shape == (50,)


def test_fps_repr() -> None:
    fps = FPS(ratio=0.5)
    assert "ratio" in repr(fps)
