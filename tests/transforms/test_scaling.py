from unittest.mock import sentinel

import pytest
import torch

import torch_pointcloud.transforms as T


def test_rescale_centroid_default() -> None:
    pos = torch.tensor([[0.0, 0.0, 0.0], [4.0, 0.0, 0.0]])
    data = {"pos": pos, "other": sentinel.other}

    result = T.Rescale(keys=["pos"])(data)

    # Centroid: subtract mean (2, 0, 0); divide by max-radius (2). Output spans [-1, 1].
    assert torch.allclose(result["pos"], torch.tensor([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]), atol=1e-5)
    assert result["other"] is sentinel.other


def test_rescale_bbox_method() -> None:
    pos = torch.tensor([[0.0, 0.0, 0.0], [4.0, 2.0, 1.0]])
    result = T.Rescale(keys=["pos"], method="bbox")({"pos": pos})
    # bbox center (2,1,0.5); half-diagonal = max((4,2,1))/2 = 2; output range [-1, 1] on longest axis
    assert result["pos"].abs().max().item() == pytest.approx(1.0, abs=1e-5)


def test_normalize() -> None:
    data = {"x": torch.tensor([[1.0, 2.0, 3.0], [4.0, 6.0, 8.0]])}
    transform = T.Normalize(keys=["x"], mean=[1.0, 2.0, 3.0], std=[1.0, 2.0, 5.0])
    result = transform(data)
    expected = torch.tensor([[0.0, 0.0, 0.0], [3.0, 2.0, 1.0]])
    assert torch.allclose(result["x"], expected)


def test_rescale_empty_passthrough(empty_scene: dict) -> None:
    out = T.Rescale(keys=["pos"])(empty_scene)
    assert out["pos"].shape == (0, 3)


def test_rescale_single_point(single_point_scene: dict) -> None:
    # Single point: radius is 0 → eps prevents NaN. Result should be all zeros.
    out = T.Rescale(keys=["pos"])(single_point_scene)
    assert torch.allclose(out["pos"], torch.zeros_like(out["pos"]))


def test_normalize_zero_std_does_not_divide_by_zero() -> None:
    data = {"x": torch.tensor([[1.0, 2.0]])}
    out = T.Normalize(keys=["x"], mean=[1.0, 2.0], std=[0.0, 0.0], eps=1e-5)(data)
    # (x - mean) is zero, divided by clamped eps; result is zero (not NaN)
    assert torch.all(torch.isfinite(out["x"]))
    assert torch.allclose(out["x"], torch.zeros_like(out["x"]))
