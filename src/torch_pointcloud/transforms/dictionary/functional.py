from typing import Optional

import torch_pointcloud.transforms.functional as F
from torch_pointcloud.utils.conversion import ensure_list, ensure_tuple, ensure_tuple_size
from torch_pointcloud.utils.types import DictStr, KeyCollection

from ._utils import key_iterator


def random_sampled(
    data: DictStr, keys: KeyCollection, num_samples: int, seed: Optional[int] = None, allow_missing_keys: bool = False
) -> DictStr:
    """Randomly sample a fixed number of values from a dictionary.
    If multiple keys are provided, the same indices are used for all keys, ensuring
    correspondence between the sampled values.

    Args:
        data: The dictionary data to apply the transform to.
        keys: The keys to sample from.
        num_samples: The number of values to sample.
        seed: The seed for the random number generator.
        allow_missing_keys: If `True`, the transform will not raise an error if the keys are not present in the data.

    Returns:
        The transformed dictionary data.
    """
    d = dict(data)  # avoid modifying the original data
    iterator = key_iterator(d, keys, allow_missing_keys=allow_missing_keys)
    first_key = next(iterator)
    sampled_tensor, indices = F.random_sample(d[first_key], num_samples, return_indices=True, seed=seed)
    d[first_key] = sampled_tensor

    for key in iterator:
        d[key] = d[key][indices]

    return d


def random_sample_face_verticesd(
    data: DictStr,
    keys: KeyCollection,
    face_keys: KeyCollection,
    num_samples: int,
    include_normals: bool = True,
    normals_key: str = "normals",
    seed: Optional[int] = None,
    allow_missing_keys: bool = False,
) -> DictStr:
    """Randomly sample a fixed number of vertices from a dictionary.
    If multiple keys are provided, the same indices are used for all keys, ensuring
    correspondence between the sampled values.

    Args:
        data: The dictionary data to apply the transform to.
        keys: The keys to sample from.
        face_keys: The keys to sample the faces from.
        num_samples: The number of vertices to sample.
        include_normals: If `True`, the normals will be included in the output.
        normals_key: The key to store the normals in.
        seed: The seed for the random number generator.
        allow_missing_keys: If `True`, the transform will not raise an error if the keys are not present in the data.

    Returns:
        The transformed dictionary data.
    """
    d = dict(data)  # avoid modifying the original data
    keys = ensure_tuple(keys)
    face_keys = ensure_tuple_size(face_keys, len(keys))

    iterator = key_iterator(d, keys, face_keys, allow_missing_keys=allow_missing_keys)
    for vertices_key, faces_key in iterator:
        out = F.random_sample_face_vertices(
            d[vertices_key],
            d[faces_key],
            num_samples,
            seed=seed,
            return_normals=include_normals,
        )

        if include_normals:
            vertices, normals = out
            d[normals_key] = normals
        else:
            vertices = out

        d[vertices_key] = vertices

    return d


def sample_farthest_pointsd(
    data: DictStr,
    pos_key: str,
    keys: Optional[KeyCollection] = None,
    num_samples: Optional[int] = None,
    ratio: Optional[float] = None,
    random_start: bool = False,
    allow_missing_keys: bool = False,
) -> DictStr:
    """Sample the farthest points from a dictionary.

    Args:
        data: The dictionary data to apply the transform to.
        keys: The keys to sample the farthest points from.
        num_samples: The number of points to sample.
        ratio: The ratio of points to sample.
        random_start: Whether to start the sampling from a random point.
        allow_missing_keys: If `True`, the transform will not raise an error if the keys are not present in the data.

    Returns:
        The transformed dictionary data.

    Examples:
        >>> import torch
        >>> from torch_pointcloud.transforms.dictionary.functional import sample_farthest_pointsd
        >>> data = {"pos": torch.randn(100, 3)}
        >>> data = sample_farthest_pointsd(data, pos_key="pos", num_samples=10)
        >>> print(data["pos"].shape)
        torch.Size([10, 3])
    """
    d = dict(data)  # avoid modifying the original data
    keys = [pos_key] + ensure_list(keys, none_as_empty=True)
    indices = F.sample_farthest_points(d[pos_key], num_samples=num_samples, ratio=ratio, random_start=random_start)

    for key in key_iterator(d, keys, allow_missing_keys=allow_missing_keys):
        d[key] = d[key][indices]
    return d


def normalize_scaled(data: DictStr, keys: KeyCollection, allow_missing_keys: bool = False) -> DictStr:
    """Normalize the scale of a dictionary.

    Args:
        data: The dictionary data to apply the transform to.
        keys: The keys to normalize the scale of.
        allow_missing_keys: If `True`, the transform will not raise an error if the keys are not present in the data.
    """
    d = dict(data)  # avoid modifying the original data
    iterator = key_iterator(d, keys, allow_missing_keys=allow_missing_keys)
    for key in iterator:
        d[key] = F.normalize_scale(d[key])

    return d


def remove_near_origind(
    data: DictStr,
    pos_key: str,
    keys: Optional[KeyCollection] = None,
    radius: float = 1e-3,
    allow_missing_keys: bool = False,
) -> DictStr:
    r"""Dict-based functional transform version of `torch_pointcloud.transforms.functional.remove_near_origin`.
    This function is used to functionally remove points that are within a given radius of the origin.

    Args:
        data: The dictionary data to apply the transform to.
        pos_key: The key containing the positions / coordinates, used to compute the distance from the origin.
        keys: The keys to remove the near origin points from.
        radius: The radius of the sphere.
        return_mask: Whether to return the mask of the points removed.
        allow_missing_keys: If `True`, the transform will not raise an error if the keys are not present in the data.

    Returns:
        The transformed dictionary data.
    """
    d = dict(data)  # avoid modifying the original data
    keys = [pos_key] + ensure_list(keys, none_as_empty=True)
    _, mask = F.remove_near_origin(d[pos_key], radius=radius, return_mask=True)
    for key in key_iterator(d, keys, allow_missing_keys=allow_missing_keys):
        d[key] = d[key][mask]
    return d
