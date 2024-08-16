from typing import Any, Dict, List, Literal, Union

import pytest
import torch

import torch_pointcloud.transforms.functional as F


@pytest.mark.parametrize(
    "tensor, num_samples",
    [
        (torch.rand(100, 3), 10),
        (torch.rand(50, 3), 20),
        (torch.rand(10, 3), 5),
    ],
)
def test_random_sample(tensor: torch.Tensor, num_samples: int) -> None:
    sampled_tensor, indices = F.random_sample(tensor, num_samples)
    # Check that the shapes are correct
    assert sampled_tensor.shape[0] == num_samples
    assert indices.shape[0] == num_samples

    # Check that the indices are within valid range
    assert torch.all(indices >= 0) and torch.all(indices < tensor.size(0))
    assert torch.equal(sampled_tensor, tensor[indices]), "Sampled tensor should be a subset of the original tensor"


@pytest.mark.parametrize(
    "data, num_samples, keys",
    [
        ({"xyz": torch.rand(100, 3), "features": torch.rand(100, 6)}, 10, "all"),
        ({"xyz": torch.rand(100, 3), "features": torch.rand(100, 6)}, 10, ["xyz", "features"]),
        ({"xyz": torch.rand(50, 3), "features": torch.rand(50, 6)}, 20, ["xyz"]),
        ({"xyz": torch.rand(10, 3)}, 5, ["xyz"]),
        ({"custom": torch.rand(10)}, 5, ["custom"]),
    ],
)
def test_random_sample_tensors(data: Dict[str, Any], num_samples: int, keys: Union[List[str], Literal["all"]]) -> None:
    sampled_data, indices = F.random_sample_data(data, num_samples=num_samples, keys=keys)
    keys = list(data.keys()) if keys == "all" else keys
    N = data[keys[0]].shape[0]

    # Check that the sampled data has the same keys
    assert set(sampled_data.keys()) == set(data.keys())

    # Check that the shapes are correct
    assert indices.shape[0] == num_samples, "Indices should have num_samples values"
    for key in keys:
        assert sampled_data[key].shape[0] == num_samples, "Sampled tensor should have num_samples rows"

    # Check that the indices are within valid range
    assert torch.all(indices >= 0) and torch.all(indices < N)

    # Verify the sub-sampling is correct
    for key in keys:
        assert torch.equal(sampled_data[key], data[key][indices])


@pytest.mark.parametrize("data, num_points", [({"xyz": torch.rand(100, 3)}, 10), ({"xyz": torch.rand(50, 3)}, 20)])
def test_sample_random_points(data: Dict[str, Any], num_points: int) -> None:
    sampled_data = F.sample_random_points(data, num_points)
    assert sampled_data["xyz"].shape[0] == num_points


@pytest.mark.parametrize("data", [{"xyz": torch.rand(100, 3)}, {"xyz": torch.rand(50, 3)}])
def test_normalize_scale(data: Dict[str, Any]) -> None:
    normalized_data = F.normalize_scale(data)
    assert torch.allclose(normalized_data["xyz"].mean(dim=-2), torch.zeros(3), atol=1e-5)
    assert normalized_data["xyz"].abs().max() <= 1.0


@pytest.mark.parametrize("data, num_points", [({"xyz": torch.rand(100, 3)}, 10), ({"xyz": torch.rand(50, 3)}, 20)])
def test_sample_furthest_points(data: Dict[str, Any], num_points: int) -> None:
    sampled_data = F.sample_furthest_points(data, num_points)
    assert sampled_data["xyz"].shape[0] == num_points


@pytest.mark.parametrize(
    "data, num_points, include_normals",
    [
        ({"xyz": torch.rand(100, 3), "face": torch.randint(0, 100, (50, 3))}, 10, True),
        ({"xyz": torch.rand(100, 3), "face": torch.randint(0, 100, (50, 3))}, 20, False),
    ],
)
def test_sample_mesh_points(data: Dict[str, Any], num_points: int, include_normals: bool) -> None:
    sampled_data = F.sample_mesh_points(data, num_points, include_normals)
    assert sampled_data["xyz"].shape[0] == num_points
    if include_normals:
        assert "normal" in sampled_data
        assert sampled_data["normal"].shape[0] == num_points
    else:
        assert "normal" not in sampled_data
