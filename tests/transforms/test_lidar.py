from typing import Dict

import pytest
import torch
from torch import Tensor

import torch_pointcloud.transforms as T
import torch_pointcloud.transforms.functional as F


@pytest.fixture
def lidar_hd_sample() -> Dict[str, Tensor]:
    return {
        "intensity": torch.tensor([[100.0], [2000.0], [50.0]]),
        "return_number": torch.tensor([[1], [1], [2]]),
        "number_of_returns": torch.tensor([[1], [2], [2]]),
        "color": torch.tensor([[65280, 0, 32640], [0, 65280, 0], [65280, 65280, 65280]], dtype=torch.int32),
        "nir": torch.tensor([[0], [65280], [65280]], dtype=torch.int32),
    }


def test_lidar_hd_features_channels(lidar_hd_sample: Dict[str, Tensor]) -> None:
    data = lidar_hd_sample
    x = F.lidar_hd_features(
        data["intensity"], data["return_number"], data["number_of_returns"], data["color"], data["nir"]
    )

    assert x.shape == (3, 9)
    assert x.dtype == torch.float32
    assert x[:, 0].tolist() == [100.0, 2000.0, 50.0]
    assert torch.allclose(x[:, 1], torch.tensor([1.0, 1.0, 2.0]) / 7)
    assert torch.allclose(x[:, 2], torch.tensor([1.0, 2.0, 2.0]) / 7)
    # First point: red 1, green 0, blue 0.5, nir 0.
    assert torch.allclose(x[0, 3:7], torch.tensor([1.0, 0.0, 0.5, 0.0]))
    assert torch.allclose(x[0, 7], torch.tensor(0.5))
    assert torch.allclose(x[0, 8], torch.tensor(-1.0), atol=1e-5)
    # Second point: no red, full near-infrared, NDVI 1.
    assert torch.allclose(x[1, 8], torch.tensor(1.0), atol=1e-5)


def test_lidar_hd_features_zero_colour_beyond_first_return(lidar_hd_sample: Dict[str, Tensor]) -> None:
    data = lidar_hd_sample
    x = F.lidar_hd_features(
        data["intensity"], data["return_number"], data["number_of_returns"], data["color"], data["nir"]
    )
    assert x[2, 3:].tolist() == [0.0] * 6


def test_lidar_hd_features_missing_colour_reads_as_zeros(lidar_hd_sample: Dict[str, Tensor]) -> None:
    data = lidar_hd_sample
    x = F.lidar_hd_features(data["intensity"], data["return_number"], data["number_of_returns"])
    assert x.shape == (3, 9)
    assert x[:, 3:].abs().sum() == 0


def test_lidar_hd_features_transform(lidar_hd_sample: Dict[str, Tensor]) -> None:
    keys = ["intensity", "return_number", "number_of_returns", "color", "nir"]
    data = lidar_hd_sample
    result = T.LidarHDFeatures(keys=keys, dst_key="x")(data)
    expected = F.lidar_hd_features(*(data[key] for key in keys))
    assert torch.equal(result["x"], expected)


def test_lidar_hd_features_transform_missing_colour(lidar_hd_sample: Dict[str, Tensor]) -> None:
    keys = ["intensity", "return_number", "number_of_returns", "color", "nir"]
    data = {key: value for key, value in lidar_hd_sample.items() if key not in ("color", "nir")}

    with pytest.raises(KeyError):
        T.LidarHDFeatures(keys=keys, dst_key="x")(data)

    result = T.LidarHDFeatures(keys=keys, dst_key="x", allow_missing_keys=True)(data)
    assert result["x"][:, 3:].abs().sum() == 0


def test_lidar_hd_features_transform_requires_five_keys() -> None:
    with pytest.raises(ValueError, match="near-infrared"):
        T.LidarHDFeatures(keys=["intensity", "color"], dst_key="x")


def test_standardize_lidar_hd_features_statistics() -> None:
    generator = torch.Generator().manual_seed(0)
    x = torch.rand(1000, 9, generator=generator)
    x[:, 0] = torch.rand(1000, generator=generator) * 3000
    # A wide clamp leaves the standardized channels untouched.
    out = F.standardize_lidar_hd_features(x, clamp_sigma=1e6)

    for channel in (0, 7):
        assert abs(out[:, channel].mean().item()) < 1e-4
        assert abs(out[:, channel].std().item() - 1.0) < 1e-3
    other = [1, 2, 3, 4, 5, 6, 8]
    assert torch.equal(out[:, other], x[:, other])


def test_standardize_lidar_hd_features_clamps_in_raw_standard_deviations() -> None:
    x = torch.zeros(4, 9)
    x[:, 7] = torch.tensor([0.0, 0.1, 0.2, 0.9])
    out = F.standardize_lidar_hd_features(x)

    bound = 3 * (x[:, 7].std() + 1e-6)
    assert out[:, 7].abs().max() <= bound + 1e-6
    assert torch.isclose(out[3, 7], bound)


def test_standardize_lidar_hd_features_single_point() -> None:
    out = F.standardize_lidar_hd_features(torch.ones(1, 9))
    assert torch.isfinite(out).all()


def test_standardize_lidar_hd_features_transform() -> None:
    x = torch.rand(50, 9)
    result = T.StandardizeLidarHDFeatures(keys="x", dst_keys="x_std")({"x": x})
    assert torch.equal(result["x"], x)
    assert torch.equal(result["x_std"], F.standardize_lidar_hd_features(x))
