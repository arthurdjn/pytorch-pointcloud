from typing import Any, Dict, Literal, Sequence, Tuple

import torch

from torch_pointcloud.ops import fps


def random_sample(tensor: torch.Tensor, num_samples: int) -> Tuple[torch.Tensor, torch.Tensor]:
    indices = torch.randint(0, tensor.size(0), (num_samples,))
    return tensor[indices], indices


def random_sample_data(
    data: Dict[str, Any], num_samples: int, keys: Sequence[str] | Literal["all"] = "all"
) -> Tuple[Dict[str, Any], torch.Tensor]:
    keys = list(data.keys()) if keys == "all" else keys
    keys = list(set(keys).intersection(data.keys()))
    assert len(keys) > 0, "keys must be a non-empty list"

    key = keys.pop(0)
    data[key], indices = random_sample(data[key], num_samples)

    for key in keys:
        data[key] = data[key][indices]
    return data, indices


def sample_random_points(data: Dict[str, Any], num_points: int, keys: Sequence[str] = ("xyz",)) -> Dict[str, Any]:
    assert len(keys) > 0, "keys must be a non-empty list"
    key = keys[0]
    N = data[key].size(0)
    indices = torch.randint(0, N, (num_points,))

    for key in keys:
        if key in data:
            data[key] = data[key][indices]
    return data


def normalize_scale(data: Dict[str, Any], keys: Sequence[str] = ("xyz",)) -> Dict[str, Any]:
    for key in keys:
        if key in data.keys():
            data[key] -= data[key].mean(dim=-2, keepdim=True)
            data[key] = data[key] / (data[key].abs().max() + 1e-5)
    return data


def sample_furthest_points(data: Dict[str, Any], num_points: int, keys: Sequence[str] = ("xyz",)) -> Dict[str, Any]:
    assert len(keys) > 0, "keys must be a non-empty list"
    lengths = data.get("lengths", None)
    key = keys[0]
    indices = fps(data[key].unsqueeze(0), num_samples=num_points, lengths=lengths)
    indices = indices.squeeze(0)

    for key in keys:
        if key in data:
            data[key] = data[key][indices]
    return data


def sample_mesh_points(data: Dict[str, Any], num_points: int, include_normals: bool = True) -> Dict[str, Any]:
    assert data["xyz"] is not None
    assert data["face"] is not None

    xyz, face = data["xyz"], data["face"]
    assert xyz.size(1) == 3 and face.size(1) == 3

    pos_max = xyz.abs().max()
    xyz = xyz / pos_max

    vec1 = xyz[face[:, 1]] - xyz[face[:, 0]]
    vec2 = xyz[face[:, 2]] - xyz[face[:, 0]]
    area = vec1.cross(vec2, dim=1)
    area = area.norm(p=2, dim=1).abs() / 2

    prob = area / area.sum()
    samples = torch.multinomial(prob, num_points, replacement=True)
    face = face[samples]

    frac = torch.rand(num_points, 2, device=xyz.device)
    mask = frac.sum(dim=-1) > 1
    frac[mask] = 1 - frac[mask]

    vec1 = xyz[face[:, 1]] - xyz[face[:, 0]]
    vec2 = xyz[face[:, 2]] - xyz[face[:, 0]]

    if include_normals:
        data["normal"] = torch.nn.functional.normalize(vec1.cross(vec2, dim=1), p=2)

    pos_sampled = xyz[face[:, 0]]
    pos_sampled += frac[:, :1] * vec1
    pos_sampled += frac[:, 1:] * vec2

    pos_sampled = pos_sampled * pos_max
    data["xyz"] = pos_sampled

    return data
