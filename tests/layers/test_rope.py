import pytest
import torch

from torch_pointcloud.layers.rope import Point3DRoPE


def test_point3d_rope_forward() -> None:
    rope = Point3DRoPE(head_dim=12, base=10.0)
    q = torch.randn(32, 4, 12)  # (N, heads, head_dim)
    k = torch.randn(32, 4, 12)
    pos = torch.randn(32, 3)
    q_out, k_out = rope(q, k, pos)
    assert q_out.shape == q.shape
    assert k_out.shape == k.shape


def test_point3d_rope_head_dim_not_divisible_raises() -> None:
    with pytest.raises(ValueError, match="divisible by 6"):
        Point3DRoPE(head_dim=10)
    # head_dim=9 passes the divisible-by-3 split but leaves odd per-axis chunks that cannot rotate in pairs.
    with pytest.raises(ValueError, match="divisible by 6"):
        Point3DRoPE(head_dim=9)


def test_point3d_rope_preserves_norm() -> None:
    rope = Point3DRoPE(head_dim=12, base=10.0)
    q = torch.randn(8, 1, 12)
    pos = torch.zeros(8, 3)
    q_out, _ = rope(q, q, pos)
    # At pos=0, cos=1, sin=0, so rotation is identity.
    torch.testing.assert_close(q_out, q)
