import pytest
import torch

from torch_pointcloud.layers.octree_attention import RPE, OctreeAttention, OctreeT
from torch_pointcloud.utils.imports import _OCNN_AVAILABLE
from torch_pointcloud.utils.octree import build_octree

# See: https://docs.pytest.org/en/stable/how-to/skipping.html#summary
pytestmark = pytest.mark.skipif(
    not _OCNN_AVAILABLE,
    reason="ocnn is not installed",
)


def _make_octree_t(depth: int = 4, patch_size: int = 4, dilation: int = 1):  # type: ignore[no-untyped-def]
    torch.manual_seed(0)
    n = 200
    pos = torch.rand(n, 3) * 1.8 - 0.9
    normal = torch.nn.functional.normalize(torch.randn(n, 3), dim=1)
    batch = torch.cat([torch.zeros(80), torch.ones(120)]).long()
    octree = build_octree(pos=pos, normal=normal, batch=batch, batch_size=2, depth=depth, full_depth=2)
    octree.construct_all_neigh()
    octree_t = OctreeT.from_octree(octree, patch_size=patch_size, dilation=dilation)
    octree_t.construct_all_attention_context(nempty=False, min_depth=depth, max_depth=depth)
    return octree_t


def test_octree_t_from_octree() -> None:
    octree_t = _make_octree_t()
    assert octree_t.patch_size == 4
    assert octree_t.dilation == 1
    assert octree_t.block_size == 4
    assert octree_t.masks[octree_t.depth] is not None
    assert octree_t.rel_pos[octree_t.depth] is not None


def test_octree_t_repr() -> None:
    octree_t = _make_octree_t()
    assert "OctreeT" in repr(octree_t)


def test_rpe_forward() -> None:
    rpe = RPE(patch_size=4, num_heads=2, dilation=1)
    pos = torch.randint(-2, 3, (2, 4, 4, 3))
    out = rpe(pos)
    assert out.shape == (2, 2, 4, 4)


def test_octree_attention_forward() -> None:
    octree_t = _make_octree_t(depth=4, patch_size=4, dilation=1)
    depth = octree_t.depth
    x = octree_t.get_input_feature("ND", nempty=False)
    # Project to attention channel size
    proj = torch.nn.Linear(x.shape[1], 16)
    x_proj = proj(x)

    attn = OctreeAttention(
        channels=16,
        patch_size=4,
        num_heads=2,
        dilation=1,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        use_rpe=True,
    )
    out = attn(x_proj, octree_t, depth)
    assert out.shape == (x_proj.shape[0], 16)


def test_octree_attention_no_rpe() -> None:
    octree_t = _make_octree_t(depth=4, patch_size=4, dilation=1)
    depth = octree_t.depth
    x = octree_t.get_input_feature("ND", nempty=False)
    proj = torch.nn.Linear(x.shape[1], 16)
    x_proj = proj(x)

    attn = OctreeAttention(channels=16, patch_size=4, num_heads=2, dilation=1, use_rpe=False)
    out = attn(x_proj, octree_t, depth)
    assert out.shape == (x_proj.shape[0], 16)
