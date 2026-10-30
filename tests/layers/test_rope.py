import pytest
import torch

from torch_pointcloud.layers.rope import AxialRoPE, Point3DRoPE


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


def test_axial_rope_matches_complex_rotation() -> None:
    torch.manual_seed(0)
    rope = AxialRoPE(num_frequencies=(3, 3, 2), base=100.0)
    q, k = torch.randn(2, 20, 2, 16).unbind(0)
    pos = torch.randint(0, 50, (20, 3))
    q_out, k_out = rope(q, k, pos)

    freqs = [1.0 / 100.0 ** torch.linspace(0, 1, n) for n in (3, 3, 2)]
    angle = torch.cat([torch.outer(pos[:, axis].float(), freqs[axis]) for axis in range(3)], dim=1)
    cis = torch.polar(torch.ones_like(angle), angle).unsqueeze(1)
    expected = torch.view_as_real(torch.view_as_complex(q.reshape(20, 2, 8, 2)) * cis).flatten(2)
    torch.testing.assert_close(q_out, expected)
    assert k_out.shape == k.shape


def test_axial_rope_logits_depend_on_relative_positions() -> None:
    torch.manual_seed(0)
    rope = AxialRoPE(num_frequencies=(4, 4, 4), base=10.0)
    q, k = torch.randn(2, 1, 1, 24).unbind(0)
    pos = torch.tensor([[3, 1, 4]])
    shift = torch.tensor([[7, 2, 5]])
    q_a, _ = rope(q, q, pos)
    _, k_a = rope(k, k, pos + 2)
    q_b, _ = rope(q, q, pos + shift)
    _, k_b = rope(k, k, pos + 2 + shift)
    torch.testing.assert_close((q_a * k_a).sum(), (q_b * k_b).sum())


def test_axial_rope_requires_three_bands() -> None:
    with pytest.raises(ValueError, match="three positive counts"):
        AxialRoPE(num_frequencies=(4, 4), base=10.0)


def test_axial_rope_keeps_float32_angles_in_a_bfloat16_module() -> None:
    torch.manual_seed(0)
    rope = AxialRoPE(num_frequencies=(4, 4, 4), base=100.0)
    q, k = torch.randn(2, 64, 2, 24).unbind(0)
    pos = torch.randint(0, 400, (64, 3))
    q_ref, k_ref = rope(q, k, pos)

    q_out, k_out = rope.to(torch.bfloat16)(q.bfloat16(), k.bfloat16(), pos)

    torch.testing.assert_close(q_out.float(), q_ref, atol=0.05, rtol=0.02)
    torch.testing.assert_close(k_out.float(), k_ref, atol=0.05, rtol=0.02)


def test_axial_rope_rejects_a_mismatched_head_dimension() -> None:
    rope = AxialRoPE(num_frequencies=(4, 4, 4), base=100.0)
    q = torch.randn(8, 2, 16)

    with pytest.raises(ValueError, match="rotate 24 channels, got 16"):
        rope(q, q, torch.randint(0, 8, (8, 3)))
