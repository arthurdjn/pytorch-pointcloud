import math
from typing import Any, Dict

import pytest
import torch

import torch_pointcloud.transforms as T
import torch_pointcloud.transforms.functional as F


def _mix_pair() -> tuple[Dict[str, Any], Dict[str, Any]]:
    g = torch.Generator().manual_seed(0)
    a = {
        "pos": torch.randn(100, 3, generator=g),
        "segment": torch.randint(0, 10, (100,), generator=g),
        "instance": torch.randint(0, 5, (100,), generator=g),
    }
    b = {
        "pos": torch.randn(120, 3, generator=g),
        "segment": torch.randint(0, 10, (120,), generator=g),
        "instance": torch.randint(0, 7, (120,), generator=g),
    }
    b["instance"][0] = -1  # ignore label
    return a, b


def test_mix3d_concatenates_point_keys() -> None:
    a, b = _mix_pair()
    out = T.Mix3D(keys=("pos", "segment", "instance"))(a, b)
    assert out["pos"].shape[0] == 220
    assert out["segment"].shape[0] == 220
    assert out["instance"].shape[0] == 220


def test_mix3d_instance_offset_respects_ignore_label() -> None:
    """The second scene's instance ids shift past the first scene's max, but `-1` stays `-1`."""
    a, b = _mix_pair()
    out = T.Mix3D(keys=("pos", "segment", "instance"), instance_key="instance", ignore_index=-1)(a, b)
    offset = int(a["instance"].max()) + 1
    assert torch.equal(out["instance"][:100], a["instance"])
    assert out["instance"][100] == -1
    assert torch.equal(out["instance"][101:], b["instance"][1:] + offset)


def test_mix3d_p_zero_is_noop() -> None:
    a, b = _mix_pair()
    out = T.Mix3D(keys=("pos", "segment"), p=0.0)(a, b)
    assert out["pos"].shape[0] == 100
    assert torch.equal(out["pos"], a["pos"])


def test_mix3d_does_not_mutate_inputs() -> None:
    a, b = _mix_pair()
    a_pos = a["pos"].clone()
    b_instance = b["instance"].clone()
    T.Mix3D(keys=("pos", "segment", "instance"))(a, b)
    assert a["pos"].shape[0] == 100 and b["instance"].shape[0] == 120
    assert torch.equal(a["pos"], a_pos)
    assert torch.equal(b["instance"], b_instance)


def test_laser_mix_key_correspondence() -> None:
    """Every masked key keeps the same length as the coordinate key."""
    a, b = _mix_pair()
    out = T.LaserMix(keys=("pos", "segment"), num_areas=(4,), pitch_range=(-25.0, 3.0), seed=1)(a, b)
    assert out["pos"].shape[0] == out["segment"].shape[0]


def test_laser_mix_p_zero_is_noop() -> None:
    a, b = _mix_pair()
    out = T.LaserMix(keys=("pos", "segment"), num_areas=(4,), pitch_range=(-25.0, 3.0), p=0.0)(a, b)
    assert torch.equal(out["pos"], a["pos"])


def test_polar_mix_key_correspondence() -> None:
    a, b = _mix_pair()
    out = T.PolarMix(keys=("pos", "segment"), instance_classes=(1, 2, 3), seed=2)(a, b)
    assert out["pos"].shape[0] == out["segment"].shape[0]


def test_polar_mix_p_zero_is_noop() -> None:
    a, b = _mix_pair()
    out = T.PolarMix(keys=("pos", "segment"), instance_classes=(1, 2, 3), p=0.0)(a, b)
    assert torch.equal(out["pos"], a["pos"])


def test_laser_mix_masks_shapes_and_dtype() -> None:
    g = torch.Generator().manual_seed(0)
    pos = torch.randn(200, 3, generator=g)
    other = torch.randn(150, 3, generator=g)
    mask, other_mask = F.laser_mix_masks(pos, other, num_areas=4, pitch_range=(-25.0, 3.0), generator=g)
    assert mask.shape == (200,) and other_mask.shape == (150,)
    assert mask.dtype == torch.bool and other_mask.dtype == torch.bool


def test_laser_mix_masks_bands_are_complementary() -> None:
    """A point in the same pitch band of both scans is kept from exactly one of them."""
    g = torch.Generator().manual_seed(0)
    pos = torch.randn(300, 3, generator=g)
    mask, other_mask = F.laser_mix_masks(pos, pos, num_areas=6, pitch_range=(-25.0, 3.0), generator=g)
    assert torch.equal(mask, ~other_mask)


def test_laser_mix_masks_invalid_num_areas() -> None:
    pos = torch.randn(10, 3)
    with pytest.raises(ValueError, match="num_areas"):
        F.laser_mix_masks(pos, pos, num_areas=0, pitch_range=(-25.0, 3.0))


def test_polar_mix_masks_shapes_and_dtype() -> None:
    g = torch.Generator().manual_seed(0)
    pos = torch.randn(200, 3, generator=g)
    other = torch.randn(150, 3, generator=g)
    mask, other_mask = F.polar_mix_masks(pos, other, generator=g)
    assert mask.shape == (200,) and other_mask.shape == (150,)
    assert mask.dtype == torch.bool and other_mask.dtype == torch.bool


def test_polar_mix_masks_sector_is_complementary() -> None:
    """The first-scan keep-mask and second-scan paste-mask select opposite sides of the sector."""
    g = torch.Generator().manual_seed(1)
    pos = torch.randn(300, 3, generator=g)
    mask, other_mask = F.polar_mix_masks(pos, pos, generator=g)
    assert torch.equal(mask, ~other_mask)


def test_polar_mix_masks_sector_wraps_at_pi() -> None:
    """The swap sector keeps its half-circle width when the start angle lands near +pi."""
    seed = 155
    start = (torch.rand(1, generator=torch.Generator().manual_seed(seed)).item() * 2.0 - 1.0) * math.pi
    assert start > 3.0, "seed must draw a start angle near +pi so the sector crosses the seam"
    theta = torch.linspace(-math.pi, math.pi, 4001)[:-1]
    pos = torch.stack([torch.cos(theta), torch.sin(theta), torch.zeros_like(theta)], dim=1)
    _, other_mask = F.polar_mix_masks(pos, pos, generator=torch.Generator().manual_seed(seed))
    assert abs(other_mask.float().mean().item() - 0.5) < 0.01
    # A point just past the +pi seam falls inside the wrapped sector.
    seam = torch.tensor([[math.cos(-3.1), math.sin(-3.1), 0.0]])
    _, seam_mask = F.polar_mix_masks(seam, seam, generator=torch.Generator().manual_seed(seed))
    assert seam_mask.tolist() == [True]
