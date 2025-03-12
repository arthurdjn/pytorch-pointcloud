from typing import Optional

import torch_pointcloud.transforms.functional as F
from torch_pointcloud.utils.conversion import ensure_tuple_size
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


def random_sample_verticesd(
    data: DictStr,
    keys: KeyCollection,
    face_keys: KeyCollection,
    num_samples: int,
    include_normals: bool = True,
    seed: Optional[int] = None,
    allow_missing_keys: bool = False,
    normals_keys: Optional[KeyCollection] = "normals",
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
        seed: The seed for the random number generator.
        allow_missing_keys: If `True`, the transform will not raise an error if the keys are not present in the data.
        normals_keys: The keys to store the normals in.

    Returns:
        The transformed dictionary data.
    """
    face_keys = ensure_tuple_size(face_keys, size=len(keys))
    normals_keys = ensure_tuple_size(normals_keys, size=len(keys))

    d = dict(data)  # avoid modifying the original data
    normals_keys = normals_keys or keys
    iterator = key_iterator(d, keys, face_keys, normals_keys, allow_missing_keys=allow_missing_keys)

    key, face_key, normals_key = next(iterator)

    out = F.random_sample_vertices(
        d[key],
        d[face_key],
        num_samples,
        seed=seed,
        return_normals=include_normals,
        return_indices=True,
    )

    if include_normals:
        sampled_tensor, normals, indices = out
        d[normals_key] = normals
    else:
        sampled_tensor, indices = out

    d[key] = sampled_tensor
    for key, face_key, normals_key in iterator:
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
