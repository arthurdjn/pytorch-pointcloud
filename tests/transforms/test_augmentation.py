import pytest
import torch

import torch_pointcloud.transforms as T
import torch_pointcloud.transforms.functional as F


def test_random_rotate_pos_and_normal_share_rotation() -> None:
    """Same R should be applied to every key listed."""
    pos = torch.tensor([[1.0, 0.0, 0.0]])
    normal = torch.tensor([[1.0, 0.0, 0.0]])
    out = T.RandomRotate(keys=("pos", "normal"), angle_range=(90, 90), axis=2, seed=0)(
        {"pos": pos.clone(), "normal": normal.clone()}
    )
    # 90deg around z: (1, 0, 0) -> (0, 1, 0)
    assert torch.allclose(out["pos"], out["normal"], atol=1e-4)
    assert torch.allclose(out["pos"], torch.tensor([[0.0, 1.0, 0.0]]), atol=1e-4)


def test_random_rotate_p_zero_is_noop() -> None:
    pos = torch.tensor([[1.0, 0.0, 0.0]])
    out = T.RandomRotate(keys="pos", p=0.0)({"pos": pos.clone()})
    assert torch.equal(out["pos"], pos)


def test_random_scale_same_factor_across_keys() -> None:
    """Same factor applies to every point-like key. Direction vectors (e.g. `normal`) must not be listed."""
    pos = torch.tensor([[1.0, 2.0, 3.0]])
    grid_pos = torch.tensor([[4.0, 5.0, 6.0]])
    out = T.RandomScale(keys=("pos", "grid_pos"), scale_range=(2.0, 2.0), seed=0)(
        {"pos": pos.clone(), "grid_pos": grid_pos.clone()}
    )
    assert torch.allclose(out["pos"], pos * 2.0)
    assert torch.allclose(out["grid_pos"], grid_pos * 2.0)


def test_random_scale_anisotropic_per_axis() -> None:
    pos = torch.tensor([[1.0, 1.0, 1.0]])
    out = T.RandomScale(keys="pos", scale_range=(0.5, 2.0), anisotropic=True, seed=0)({"pos": pos.clone()})
    # All axes scaled (possibly differently); shape preserved.
    assert out["pos"].shape == pos.shape


def test_random_flip_p_one_flips_all_listed_axes() -> None:
    pos = torch.tensor([[1.0, 2.0, 3.0]])
    out = T.RandomFlip(keys="pos", axes=(0, 1), p=1.0)({"pos": pos.clone()})
    assert torch.allclose(out["pos"], torch.tensor([[-1.0, -2.0, 3.0]]))


def test_random_jitter_adds_bounded_noise() -> None:
    pos = torch.zeros(100, 3)
    out = T.RandomJitter(keys="pos", sigma=0.1, clip=0.05, seed=0)({"pos": pos})
    assert out["pos"].abs().max().item() <= 0.05 + 1e-6


def test_random_shift_translates_uniformly() -> None:
    pos = torch.zeros(5, 3)
    out = T.RandomShift(keys="pos", shift_range=(1.0, 1.0), seed=0)({"pos": pos})
    assert torch.allclose(out["pos"], torch.ones_like(pos))


def test_random_color_jitter_preserves_dtype_and_range() -> None:
    color = torch.rand(50, 3)
    out = T.RandomColorJitter(keys="color", brightness=0.5, contrast=0.5, saturation=0.3, seed=0)({"color": color})
    assert out["color"].dtype == color.dtype
    assert out["color"].min().item() >= 0.0
    assert out["color"].max().item() <= 1.0


def test_random_color_jitter_applies_same_factors_to_all_keys() -> None:
    """The factors are sampled once per call, so identical inputs under different keys jitter identically."""
    color = torch.rand(50, 3)
    transform = T.RandomColorJitter(keys=["color", "color2"], brightness=0.4, contrast=0.4, saturation=0.2, seed=0)
    out = transform({"color": color.clone(), "color2": color.clone()})
    assert not torch.equal(out["color"], color)
    assert torch.equal(out["color"], out["color2"])


def test_random_color_drop_replaces_with_fill() -> None:
    color = torch.rand(10, 3)
    out = T.RandomColorDrop(keys="color", fill=0.5, p=1.0)({"color": color})
    assert torch.allclose(out["color"], torch.full_like(color, 0.5))


def test_random_color_grayscale_makes_channels_equal() -> None:
    color = torch.rand(10, 3)
    out = T.RandomColorGrayScale(keys="color", p=1.0)({"color": color})
    assert torch.allclose(out["color"][:, 0], out["color"][:, 1])
    assert torch.allclose(out["color"][:, 1], out["color"][:, 2])


def test_random_color_auto_contrast_stretches_range() -> None:
    color = torch.tensor([[0.25, 0.25, 0.25], [0.75, 0.75, 0.75]])
    out = T.RandomColorAutoContrast(keys="color", blend=1.0, p=1.0)({"color": color})
    # Fully stretched: min becomes 0, max becomes 1.
    assert torch.allclose(out["color"].min(dim=0).values, torch.zeros(3), atol=1e-5)
    assert torch.allclose(out["color"].max(dim=0).values, torch.ones(3), atol=1e-5)


def test_random_color_jitter_uint8_keeps_255_scale() -> None:
    color = torch.tensor([[200, 100, 50], [30, 60, 90]], dtype=torch.uint8)
    out = T.RandomColorJitter(keys="color", brightness=0.2, p=1.0, seed=0)({"color": color})
    assert out["color"].dtype == torch.uint8
    assert out["color"].float().max().item() > 100.0


def test_random_color_shift_uint8_clamps_to_255_range() -> None:
    color = torch.full((5, 3), 250, dtype=torch.uint8)
    out = T.RandomColorShift(keys="color", shift_range=(10.0, 10.0), p=1.0, seed=0)({"color": color})
    assert out["color"].dtype == torch.uint8
    assert torch.all(out["color"] == 255)


def test_random_color_shift_float_255_without_flag_raises() -> None:
    color = torch.full((5, 3), 200.0)
    with pytest.raises(ValueError, match="int_color"):
        T.RandomColorShift(keys="color", p=1.0)({"color": color})


def test_random_rotate_choice_same_rotation_across_keys() -> None:
    pos = torch.tensor([[1.0, 0.0, 0.0]])
    normal = torch.tensor([[1.0, 0.0, 0.0]])
    out = T.RandomRotateChoice(
        keys=("pos", "normal"),
        angles=[90.0],
        axis=2,
        seed=7,
    )({"pos": pos.clone(), "normal": normal.clone()})
    # Same R applied to both, so pos and normal are identical.
    assert torch.allclose(out["pos"], out["normal"], atol=1e-5)


def test_random_rotate_choice_empty_raises() -> None:
    with pytest.raises(ValueError, match="at least one angle"):
        T.RandomRotateChoice(keys="pos", angles=[])


def test_random_color_shift_clamps_to_valid_range() -> None:
    color = torch.full((5, 3), 0.95)
    out = T.RandomColorShift(keys="color", shift_range=(0.5, 0.5), seed=0)({"color": color})
    assert torch.all(out["color"] <= 1.0)


def test_random_color_shift_int_dtype_preserved() -> None:
    color = torch.full((5, 3), 128, dtype=torch.uint8)
    out = T.RandomColorShift(
        keys="color",
        shift_range=(5, 5),
        int_color=True,
        seed=0,
    )({"color": color})
    assert out["color"].dtype == torch.uint8


def test_random_color_shift_same_shift_across_keys() -> None:
    """The shift is sampled once per call, so every listed key moves by the same offset."""
    data = {"c1": torch.full((4, 3), 0.5), "c2": torch.full((4, 3), 0.5)}
    out = T.RandomColorShift(keys=("c1", "c2"), shift_range=(-0.2, 0.2), seed=0)(data)
    assert torch.equal(out["c1"], out["c2"])
    assert not torch.equal(out["c1"], data["c1"])


def test_random_elastic_distortion_changes_positions() -> None:
    pos = torch.randn(200, 3)
    out = T.RandomElasticDistortion(
        keys="pos",
        granularity=0.5,
        magnitude=0.1,
        seed=0,
    )({"pos": pos.clone()})
    assert out["pos"].shape == pos.shape
    assert (out["pos"] - pos).abs().max().item() > 0.0


def test_random_elastic_distortion_p_zero_is_noop() -> None:
    pos = torch.randn(20, 3)
    out = T.RandomElasticDistortion(keys="pos", p=0.0)({"pos": pos.clone()})
    assert torch.equal(out["pos"], pos)


def test_random_elastic_distortion_multi_key_shares_field() -> None:
    """Two keys with the same positions receive the same displacement."""
    pos = torch.randn(50, 3)
    out = T.RandomElasticDistortion(
        keys=("pos", "pos_copy"),
        granularity=0.5,
        magnitude=0.5,
        seed=0,
    )({"pos": pos.clone(), "pos_copy": pos.clone()})
    assert torch.equal(out["pos"], out["pos_copy"])
    assert not torch.equal(out["pos"], pos)


def test_random_scale_anisotropic_rejects_mismatched_key_widths() -> None:
    t = T.RandomScale(keys=["pos", "intensity"], scale_range=(0.9, 1.1), anisotropic=True, p=1.0)
    with pytest.raises(ValueError, match="one factor per channel"):
        t({"pos": torch.rand(5, 3), "intensity": torch.rand(5, 1)})


def test_random_shift_rejects_mismatched_key_widths() -> None:
    t = T.RandomShift(keys=["pos", "intensity"], shift_range=(-0.1, 0.1), p=1.0)
    with pytest.raises(ValueError, match="one offset per channel"):
        t({"pos": torch.rand(5, 3), "intensity": torch.rand(5, 1)})


def test_random_jitter_clipped() -> None:
    pos = torch.zeros(100, 3)
    g = torch.Generator().manual_seed(0)
    out = F.random_jitter(pos, sigma=1.0, clip=0.1, generator=g)
    assert out.abs().max().item() <= 0.1 + 1e-6


def test_random_jitter_no_clip() -> None:
    pos = torch.zeros(1000, 3)
    g = torch.Generator().manual_seed(0)
    out = F.random_jitter(pos, sigma=0.1, clip=None, generator=g)
    # Without clip some samples should exceed 0.1.
    assert out.abs().max().item() > 0.1


def test_random_color_jitter_preserves_range() -> None:
    color = torch.rand(50, 3)
    g = torch.Generator().manual_seed(0)
    out = F.random_color_jitter(color, brightness=0.5, contrast=0.5, saturation=0.3, generator=g)
    assert out.min().item() >= 0.0
    assert out.max().item() <= 1.0
    assert out.shape == color.shape


def test_random_color_jitter_int_dtype_preserved() -> None:
    color = (torch.rand(10, 3) * 255).to(torch.uint8)
    g = torch.Generator().manual_seed(0)
    out = F.random_color_jitter(color, brightness=0.2, int_color=True, generator=g)
    assert out.dtype == torch.uint8


def test_random_color_drop_returns_constant() -> None:
    color = torch.rand(10, 3)
    out = F.random_color_drop(color, fill=0.5)
    assert torch.allclose(out, torch.full_like(color, 0.5))


def test_color_grayscale_makes_channels_equal() -> None:
    color = torch.rand(10, 3)
    out = F.color_grayscale(color)
    assert torch.allclose(out[:, 0], out[:, 1])
    assert torch.allclose(out[:, 1], out[:, 2])


def test_color_grayscale_uses_bt601_weights() -> None:
    # Pure red (1, 0, 0) gives luminance 0.299.
    color = torch.tensor([[1.0, 0.0, 0.0]])
    out = F.color_grayscale(color)
    assert torch.allclose(out, torch.full_like(color, 0.299), atol=1e-5)


def test_color_auto_contrast_full_blend_stretches_range() -> None:
    color = torch.tensor([[0.25, 0.25, 0.25], [0.75, 0.75, 0.75]])
    out = F.color_auto_contrast(color, blend=1.0)
    assert torch.allclose(out.min(dim=0).values, torch.zeros(3), atol=1e-5)
    assert torch.allclose(out.max(dim=0).values, torch.ones(3), atol=1e-5)


def test_color_auto_contrast_zero_blend_is_identity() -> None:
    color = torch.tensor([[0.25, 0.25, 0.25], [0.75, 0.75, 0.75]])
    out = F.color_auto_contrast(color, blend=0.0)
    assert torch.allclose(out, color, atol=1e-5)


def test_random_color_jitter_uint8_keeps_255_scale_by_default() -> None:
    color = torch.tensor([[200, 100, 50], [30, 60, 90]], dtype=torch.uint8)
    g = torch.Generator().manual_seed(0)
    out = F.random_color_jitter(color, brightness=0.2, generator=g)
    assert out.dtype == torch.uint8
    # uint8 colors keep their [0, 255] scale instead of collapsing to all-1s.
    assert out.float().max().item() > 100.0


def test_random_color_jitter_float_unit_range_passthrough_at_zero_strength() -> None:
    color = torch.rand(10, 3)
    out = F.random_color_jitter(color)
    assert torch.allclose(out, color, atol=1e-6)


def test_random_color_jitter_float_255_without_flag_raises() -> None:
    color = torch.tensor([[200.0, 100.0, 50.0]])
    with pytest.raises(ValueError, match="int_color"):
        F.random_color_jitter(color, brightness=0.2)


def test_random_color_drop_uint8_fill_rescaled_to_255_range() -> None:
    color = torch.full((4, 3), 200, dtype=torch.uint8)
    out = F.random_color_drop(color)
    assert out.dtype == torch.uint8
    assert torch.all(out == 127)


def test_random_color_drop_float_255_without_flag_raises() -> None:
    color = torch.full((4, 3), 200.0)
    with pytest.raises(ValueError, match="int_color"):
        F.random_color_drop(color)


def test_color_auto_contrast_uint8_stretches_to_255() -> None:
    color = torch.tensor([[10, 10, 10], [110, 110, 110]], dtype=torch.uint8)
    out = F.color_auto_contrast(color, blend=1.0)
    assert out.dtype == torch.uint8
    assert out.min().item() == 0
    assert out.max().item() == 255


def test_color_auto_contrast_float_255_without_flag_raises() -> None:
    color = torch.tensor([[10.0, 10.0, 10.0], [110.0, 110.0, 110.0]])
    with pytest.raises(ValueError, match="int_color"):
        F.color_auto_contrast(color, blend=1.0)


def test_color_auto_contrast_empty_passthrough() -> None:
    color = torch.zeros(0, 3)
    out = F.color_auto_contrast(color, blend=1.0)
    assert out.shape == (0, 3)


def test_color_shift_adds_offset_and_clamps() -> None:
    color = torch.full((4, 3), 0.5)
    out = F.color_shift(color, torch.tensor([0.1, -0.2, 0.6]))
    assert torch.allclose(out, torch.tensor([0.6, 0.3, 1.0]).expand(4, 3))


def test_color_shift_uint8_clamps_to_255_range() -> None:
    color = torch.full((4, 3), 250, dtype=torch.uint8)
    out = F.color_shift(color, torch.full((3,), 10.0))
    assert out.dtype == torch.uint8
    assert torch.all(out == 255)


def test_color_shift_float_255_without_flag_raises() -> None:
    color = torch.full((4, 3), 200.0)
    with pytest.raises(ValueError, match="int_color"):
        F.color_shift(color, torch.zeros(3))


def test_functional_random_elastic_distortion_changes_positions() -> None:
    pos = torch.randn(200, 3)
    g = torch.Generator().manual_seed(0)
    out = F.random_elastic_distortion(pos, granularity=0.5, magnitude=0.1, generator=g)
    assert out.shape == pos.shape
    # Should not be identity at any reasonable magnitude.
    assert (out - pos).abs().max().item() > 0.0


def test_random_elastic_distortion_preserves_local_structure() -> None:
    """Nearby points should still be nearby after distortion (low-frequency field)."""
    pos = torch.tensor([[0.0, 0.0, 0.0], [0.001, 0.0, 0.0]])
    g = torch.Generator().manual_seed(0)
    out = F.random_elastic_distortion(pos, granularity=0.5, magnitude=0.5, generator=g)
    # Displacement at two very close points should also be very close.
    delta_in = (pos[0] - pos[1]).norm().item()
    delta_out = (out[0] - out[1]).norm().item()
    assert abs(delta_out - delta_in) < 0.01


def test_random_elastic_distortion_cell_size_matches_granularity() -> None:
    """Noise-grid nodes are spaced exactly `granularity` apart: points on nodes get the node's noise value."""
    granularity = 0.5
    magnitude = 0.4
    pos = torch.tensor([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]])
    out = F.random_elastic_distortion(
        pos,
        granularity=granularity,
        magnitude=magnitude,
        generator=torch.Generator().manual_seed(0),
    )
    noise = torch.randn(1, 3, 4, 4, 4, generator=torch.Generator().manual_seed(0)) * magnitude
    for _ in range(2):
        noise = torch.nn.functional.avg_pool3d(noise, kernel_size=3, stride=1, padding=1)
    # pos_min sits on grid node (1, 1, 1); one step of `granularity` per axis lands on node (2, 2, 2).
    assert torch.allclose(out[0], pos[0] + noise[0, :, 1, 1, 1], atol=1e-4)
    assert torch.allclose(out[1], pos[1] + noise[0, :, 2, 2, 2], atol=1e-4)


def test_random_elastic_distortion_empty_passthrough() -> None:
    pos = torch.empty(0, 3)
    out = F.random_elastic_distortion(pos, granularity=0.2, magnitude=0.4)
    assert out.shape == (0, 3)


def test_random_elastic_distortion_wrong_shape_raises() -> None:
    pos = torch.randn(10, 2)
    with pytest.raises(ValueError, match=r"\(N, 3\)"):
        F.random_elastic_distortion(pos, granularity=0.2, magnitude=0.4)
