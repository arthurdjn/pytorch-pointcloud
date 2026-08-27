from typing import Any, Dict

import torch

import torch_pointcloud.transforms as T


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
    g = torch.Generator().manual_seed(1)
    out = T.LaserMix(keys=("pos", "segment"), num_areas=(4,), pitch_range=(-25.0, 3.0), generator=g)(a, b)
    assert out["pos"].shape[0] == out["segment"].shape[0]


def test_laser_mix_p_zero_is_noop() -> None:
    a, b = _mix_pair()
    out = T.LaserMix(keys=("pos", "segment"), num_areas=(4,), pitch_range=(-25.0, 3.0), p=0.0)(a, b)
    assert torch.equal(out["pos"], a["pos"])


def test_polar_mix_key_correspondence() -> None:
    a, b = _mix_pair()
    g = torch.Generator().manual_seed(2)
    out = T.PolarMix(keys=("pos", "segment"), instance_classes=(1, 2, 3), generator=g)(a, b)
    assert out["pos"].shape[0] == out["segment"].shape[0]


def test_polar_mix_p_zero_is_noop() -> None:
    a, b = _mix_pair()
    out = T.PolarMix(keys=("pos", "segment"), instance_classes=(1, 2, 3), p=0.0)(a, b)
    assert torch.equal(out["pos"], a["pos"])
