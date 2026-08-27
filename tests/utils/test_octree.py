from typing import Literal

import pytest
import torch

from torch_pointcloud.utils.imports import _OCNN_AVAILABLE
from torch_pointcloud.utils.octree import build_octree, octree_interpolate, octree_upsample

pytestmark = pytest.mark.skipif(not _OCNN_AVAILABLE, reason="ocnn is not installed")

_DEPTH = 4


@pytest.fixture
def pos() -> torch.Tensor:
    torch.manual_seed(0)
    return torch.rand(128, 3) * 1.8 - 0.9  # [-0.9, 0.9]


def test_build_octree_returns_octree(pos: torch.Tensor) -> None:
    octree = build_octree(pos, depth=_DEPTH, full_depth=2)
    assert octree.depth == _DEPTH
    assert octree.full_depth == 2
    assert octree.batch_size == 1


def test_build_octree_return_points(pos: torch.Tensor) -> None:
    features = torch.rand(pos.size(0), 5)
    octree, points = build_octree(pos, x=features, depth=_DEPTH, return_points=True)
    assert torch.equal(points.points, pos)
    assert torch.equal(points.features, features)
    assert octree.depth == _DEPTH


def test_build_octree_batched() -> None:
    torch.manual_seed(0)
    pos = torch.rand(64, 3) * 1.8 - 0.9
    batch = torch.repeat_interleave(torch.arange(2), 32)
    octree = build_octree(pos, batch=batch, batch_size=2, depth=_DEPTH)
    assert octree.batch_size == 2


def test_octree_interpolate_shapes_and_no_input_mutation(pos: torch.Tensor) -> None:
    octree = build_octree(pos, depth=_DEPTH, full_depth=2)
    octree.construct_all_neigh()
    x = torch.rand(int(octree.nnum[_DEPTH]), 6)
    pts = torch.cat([pos, torch.zeros(pos.size(0), 1)], dim=1)
    pts_before = pts.clone()

    out = octree_interpolate(x, octree, _DEPTH, pts, method="nearest")
    assert out.shape == (pos.size(0), 6)
    assert torch.equal(pts, pts_before)

    out_linear = octree_interpolate(x, octree, _DEPTH, pts, method="linear")
    assert out_linear.shape == (pos.size(0), 6)
    assert torch.equal(pts, pts_before)


def test_octree_interpolate_invalid_method(pos: torch.Tensor) -> None:
    octree = build_octree(pos, depth=_DEPTH, full_depth=2)
    x = torch.rand(int(octree.nnum[_DEPTH]), 2)
    pts = torch.cat([pos, torch.zeros(pos.size(0), 1)], dim=1)
    with pytest.raises(ValueError, match="Invalid method"):
        octree_interpolate(x, octree, _DEPTH, pts, method="cubic")  # type: ignore[arg-type]


def test_octree_upsample_same_depth_is_identity(pos: torch.Tensor) -> None:
    octree = build_octree(pos, depth=_DEPTH, full_depth=2)
    x = torch.rand(int(octree.nnum[_DEPTH]), 3)
    assert octree_upsample(x, octree, _DEPTH, _DEPTH) is x


def test_octree_upsample_rejects_downsampling(pos: torch.Tensor) -> None:
    octree = build_octree(pos, depth=_DEPTH, full_depth=2)
    x = torch.rand(int(octree.nnum[_DEPTH]), 3)
    with pytest.raises(ValueError, match="Invalid destination depth"):
        octree_upsample(x, octree, _DEPTH, _DEPTH - 1)


@pytest.mark.parametrize("method", [pytest.param("nearest", id="nearest"), pytest.param("linear", id="linear")])
def test_octree_upsample_one_level(pos: torch.Tensor, method: Literal["linear", "nearest"]) -> None:
    octree = build_octree(pos, depth=_DEPTH, full_depth=2)
    octree.construct_all_neigh()
    x = torch.rand(int(octree.nnum[_DEPTH - 1]), 3)
    out = octree_upsample(x, octree, _DEPTH - 1, _DEPTH, method=method)
    assert out.shape == (int(octree.nnum[_DEPTH]), 3)
