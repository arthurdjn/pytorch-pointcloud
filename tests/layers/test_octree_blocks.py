import pytest
import torch

from torch_pointcloud.layers.octree_blocks import OctreeConvBlock, OctreeDeconvBlock
from torch_pointcloud.utils.imports import _OCNN_AVAILABLE
from torch_pointcloud.utils.octree import build_octree

# See: https://docs.pytest.org/en/stable/how-to/skipping.html#summary
pytestmark = pytest.mark.skipif(
    not _OCNN_AVAILABLE,
    reason="ocnn is not installed",
)


def _make_octree(depth: int = 4):  # type: ignore[no-untyped-def]
    torch.manual_seed(0)
    n = 200
    pos = torch.rand(n, 3) * 1.8 - 0.9
    normal = torch.nn.functional.normalize(torch.randn(n, 3), dim=1)
    batch = torch.cat([torch.zeros(80), torch.ones(120)]).long()
    octree = build_octree(pos=pos, normal=normal, batch=batch, batch_size=2, depth=depth, full_depth=2)
    octree.construct_all_neigh()
    return octree


def test_octree_conv_block_forward() -> None:
    octree = _make_octree()
    depth = octree.depth
    x = octree.get_input_feature("ND", nempty=False)
    block = OctreeConvBlock(
        in_channels=x.shape[1],
        out_channels=16,
        kernel_size=3,
        stride=1,
        nempty=False,
        act="relu",
        act_kwargs=None,
        act_first=False,
        norm="batch_norm",
        norm_kwargs=None,
        bias=True,
    )
    out = block(x, octree, depth)
    assert out.shape[1] == 16
    assert out.shape[0] == x.shape[0]


def test_octree_deconv_block_forward() -> None:
    octree = _make_octree()
    depth = octree.depth
    x = octree.get_input_feature("ND", nempty=False)
    block = OctreeDeconvBlock(
        in_channels=x.shape[1],
        out_channels=16,
        kernel_size=3,
        stride=1,
        nempty=False,
        act="relu",
        act_kwargs=None,
        act_first=False,
        norm="batch_norm",
        norm_kwargs=None,
        bias=True,
    )
    out = block(x, octree, depth)
    assert out.shape[1] == 16
    assert out.shape[0] == x.shape[0]
