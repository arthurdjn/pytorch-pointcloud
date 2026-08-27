import pytest
import torch

import torch_pointcloud.transforms as T


def test_random_rotate_pos_and_normal_share_rotation() -> None:
    """Same R should be applied to every key listed."""
    pos = torch.tensor([[1.0, 0.0, 0.0]])
    normal = torch.tensor([[1.0, 0.0, 0.0]])
    g = torch.Generator().manual_seed(0)
    out = T.RandomRotate(keys=("pos", "normal"), angle_range=(90, 90), axis=2, generator=g)(
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
    g = torch.Generator().manual_seed(0)
    out = T.RandomScale(keys=("pos", "grid_pos"), scale_range=(2.0, 2.0), generator=g)(
        {"pos": pos.clone(), "grid_pos": grid_pos.clone()}
    )
    assert torch.allclose(out["pos"], pos * 2.0)
    assert torch.allclose(out["grid_pos"], grid_pos * 2.0)


def test_random_scale_anisotropic_per_axis() -> None:
    pos = torch.tensor([[1.0, 1.0, 1.0]])
    g = torch.Generator().manual_seed(0)
    out = T.RandomScale(keys="pos", scale_range=(0.5, 2.0), anisotropic=True, generator=g)({"pos": pos.clone()})
    # All axes scaled (possibly differently); shape preserved.
    assert out["pos"].shape == pos.shape


def test_random_flip_p_one_flips_all_listed_axes() -> None:
    pos = torch.tensor([[1.0, 2.0, 3.0]])
    out = T.RandomFlip(keys="pos", axes=(0, 1), p=1.0)({"pos": pos.clone()})
    assert torch.allclose(out["pos"], torch.tensor([[-1.0, -2.0, 3.0]]))


def test_random_jitter_adds_bounded_noise() -> None:
    pos = torch.zeros(100, 3)
    g = torch.Generator().manual_seed(0)
    out = T.RandomJitter(keys="pos", sigma=0.1, clip=0.05, generator=g)({"pos": pos})
    assert out["pos"].abs().max().item() <= 0.05 + 1e-6


def test_random_shift_translates_uniformly() -> None:
    pos = torch.zeros(5, 3)
    g = torch.Generator().manual_seed(0)
    out = T.RandomShift(keys="pos", shift_range=(1.0, 1.0), generator=g)({"pos": pos})
    assert torch.allclose(out["pos"], torch.ones_like(pos))


def test_random_color_jitter_preserves_dtype_and_range() -> None:
    color = torch.rand(50, 3)
    g = torch.Generator().manual_seed(0)
    out = T.RandomColorJitter(keys="color", brightness=0.5, contrast=0.5, saturation=0.3, generator=g)({"color": color})
    assert out["color"].dtype == color.dtype
    assert out["color"].min().item() >= 0.0
    assert out["color"].max().item() <= 1.0


def test_random_color_jitter_applies_same_factors_to_all_keys() -> None:
    """The factors are sampled once per call, so identical inputs under different keys jitter identically."""
    color = torch.rand(50, 3)
    g = torch.Generator().manual_seed(0)
    transform = T.RandomColorJitter(keys=["color", "color2"], brightness=0.4, contrast=0.4, saturation=0.2, generator=g)
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
    g = torch.Generator().manual_seed(0)
    out = T.RandomColorJitter(keys="color", brightness=0.2, p=1.0, generator=g)({"color": color})
    assert out["color"].dtype == torch.uint8
    assert out["color"].float().max().item() > 100.0


def test_random_color_shift_uint8_clamps_to_255_range() -> None:
    color = torch.full((5, 3), 250, dtype=torch.uint8)
    g = torch.Generator().manual_seed(0)
    out = T.RandomColorShift(keys="color", shift_range=(10.0, 10.0), p=1.0, generator=g)({"color": color})
    assert out["color"].dtype == torch.uint8
    assert torch.all(out["color"] == 255)


def test_random_color_shift_float_255_without_flag_raises() -> None:
    color = torch.full((5, 3), 200.0)
    with pytest.raises(ValueError, match="int_color"):
        T.RandomColorShift(keys="color", p=1.0)({"color": color})


def test_random_rotate_choice_same_rotation_across_keys() -> None:
    pos = torch.tensor([[1.0, 0.0, 0.0]])
    normal = torch.tensor([[1.0, 0.0, 0.0]])
    g = torch.Generator().manual_seed(7)
    out = T.RandomRotateChoice(
        keys=("pos", "normal"),
        angles=[90.0],
        axis=2,
        generator=g,
    )({"pos": pos.clone(), "normal": normal.clone()})
    # Same R applied to both, so pos and normal are identical.
    assert torch.allclose(out["pos"], out["normal"], atol=1e-5)


def test_random_rotate_choice_empty_raises() -> None:
    with pytest.raises(ValueError, match="at least one angle"):
        T.RandomRotateChoice(keys="pos", angles=[])


def test_random_color_shift_clamps_to_valid_range() -> None:
    color = torch.full((5, 3), 0.95)
    g = torch.Generator().manual_seed(0)
    out = T.RandomColorShift(keys="color", shift_range=(0.5, 0.5), generator=g)({"color": color})
    assert torch.all(out["color"] <= 1.0)


def test_random_color_shift_int_dtype_preserved() -> None:
    color = torch.full((5, 3), 128, dtype=torch.uint8)
    g = torch.Generator().manual_seed(0)
    out = T.RandomColorShift(
        keys="color",
        shift_range=(5, 5),
        int_color=True,
        generator=g,
    )({"color": color})
    assert out["color"].dtype == torch.uint8


def test_random_color_shift_same_shift_across_keys() -> None:
    """The shift is sampled once per call, so every listed key moves by the same offset."""
    g = torch.Generator().manual_seed(0)
    data = {"c1": torch.full((4, 3), 0.5), "c2": torch.full((4, 3), 0.5)}
    out = T.RandomColorShift(keys=("c1", "c2"), shift_range=(-0.2, 0.2), generator=g)(data)
    assert torch.equal(out["c1"], out["c2"])
    assert not torch.equal(out["c1"], data["c1"])


def test_random_elastic_distortion_changes_positions() -> None:
    pos = torch.randn(200, 3)
    g = torch.Generator().manual_seed(0)
    out = T.RandomElasticDistortion(
        keys="pos",
        granularity=0.5,
        magnitude=0.1,
        generator=g,
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
    g = torch.Generator().manual_seed(0)
    out = T.RandomElasticDistortion(
        keys=("pos", "pos_copy"),
        granularity=0.5,
        magnitude=0.5,
        generator=g,
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
