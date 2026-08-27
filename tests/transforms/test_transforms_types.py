from unittest.mock import sentinel

import pytest
import torch

import torch_pointcloud.transforms as T


def test_to_tensor() -> None:
    data = {"x": [1.0, 2.0, 3.0]}
    transform = T.ToTensor(keys=["x"], dtype=torch.float32)
    result = transform(data)

    assert isinstance(result["x"], torch.Tensor)
    assert result["x"].dtype == torch.float32


def test_ones_like() -> None:
    data = {"pos": torch.randn(5, 3)}
    transform = T.OnesLike(keys=["pos"], dst_keys=["ones"])
    result = transform(data)

    assert torch.equal(result["ones"], torch.ones(5, 3))


def test_to_float() -> None:
    data = {"x": torch.ones(4, dtype=torch.int64), "other": sentinel.other}
    result = T.ToFloat(keys=["x"])(data)
    assert result["x"].dtype == torch.float32
    assert result["other"] is sentinel.other


def test_to_device_cpu() -> None:
    data = {"x": torch.zeros(4), "other": sentinel.other}
    result = T.ToDevice(keys=["x"], device="cpu")(data)
    assert result["x"].device.type == "cpu"
    assert result["other"] is sentinel.other


def test_to_device_non_tensor_raises() -> None:
    with pytest.raises(TypeError, match="tensor"):
        T.ToDevice(keys=["x"], device="cpu")({"x": "not a tensor"})
