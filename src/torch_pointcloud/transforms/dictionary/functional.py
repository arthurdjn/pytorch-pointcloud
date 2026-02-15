from typing import Optional

import torch

import torch_pointcloud.transforms.functional as F
from torch_pointcloud.utils.conversion import ensure_list, ensure_tuple, ensure_tuple_size
from torch_pointcloud.utils.types import DictStr, KeyCollection

from ._utils import assert_keys_in_data, key_iterator


def random_sampled(
    data: DictStr,
    keys: KeyCollection,
    num_samples: int,
    generator: Optional[torch.Generator] = None,
    allow_missing_keys: bool = False,
) -> DictStr:
    """Randomly sample a fixed number of values from a dictionary.
    If multiple keys are provided, the same indices are used for all keys, ensuring
    correspondence between the sampled values.

    Args:
        data: The dictionary data to apply the transform to.
        keys: The keys to sample from.
        num_samples: The number of values to sample.
        generator: The generator for the random number generator.
        allow_missing_keys: If `True`, the transform will not raise an error if the keys are not present in the data.

    Returns:
        The transformed dictionary data.
    """
    d = dict(data)  # avoid modifying the original data
    iterator = key_iterator(d, keys, allow_missing_keys=allow_missing_keys)
    first_key = next(iterator)
    sampled_tensor, indices = F.random_sample(d[first_key], num_samples, return_indices=True, generator=generator)
    d[first_key] = sampled_tensor

    for key in iterator:
        d[key] = d[key][indices]

    return d


def random_sample_face_verticesd(
    data: DictStr,
    *,
    keys: KeyCollection,
    face_key: KeyCollection,
    normal_key: Optional[KeyCollection] = "normals",
    dst_keys: Optional[KeyCollection] = None,
    num_samples: int,
    generator: Optional[torch.Generator] = None,
    allow_missing_keys: bool = False,
) -> DictStr:
    """Randomly sample a fixed number of vertices from a dictionary.
    If multiple keys are provided, the same indices are used for all keys, ensuring
    correspondence between the sampled values.

    Args:
        data: The dictionary data to apply the transform to.
        keys: The keys to sample from.
        face_key: The keys to sample the faces from.
        normal_key: The key to store the normals in.
        dst_keys: The keys to store the sampled vertices in.
        num_samples: The number of vertices to sample.
        include_normals: If `True`, the normals will be included in the output.
        normal_key: The key to store the normals in.
        generator: The generator for the random number generator.
        allow_missing_keys: If `True`, the transform will not raise an error if the keys are not present in the data.

    Returns:
        The transformed dictionary data.
    """
    d = dict(data)  # avoid modifying the original data
    keys = ensure_tuple(keys)
    face_key = ensure_tuple_size(face_key, size=len(keys))
    normal_keys = ensure_tuple_size(normal_key, size=len(keys))
    dst_keys = ensure_tuple_size(dst_keys or keys, size=len(keys))

    iterator = key_iterator(d, keys, dst_keys, face_key, normal_keys, allow_missing_keys=allow_missing_keys)
    for key, dst_key, face_key, normal_key in iterator:
        pos, normal = F.random_sample_face_vertices(
            d[key],
            d[face_key],  # type: ignore[index]
            num_samples,
            generator=generator,
            return_normals=True,
        )

        d[dst_key] = pos
        if normal_key is not None:
            d[normal_key] = normal  # type: ignore[index]

    return d


def sample_farthest_pointsd(
    data: DictStr,
    *,
    keys: Optional[KeyCollection] = None,
    pos_key: str,
    dst_keys: Optional[KeyCollection] = None,
    num_samples: Optional[int] = None,
    ratio: Optional[float] = None,
    random_start: bool = False,
    allow_missing_keys: bool = False,
) -> DictStr:
    """Sample the farthest points from a dictionary.

    See Also:
        `torch_pointcloud.transforms.functional.sample_farthest_points` for more details.

    Args:
        data: The dictionary data to apply the transform to.
        keys: The keys to sample the farthest points from.
        pos_key: The key to store the positions in.
        dst_keys: The keys to store the sampled points in.
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
    dst_keys = ensure_tuple_size(dst_keys or keys, size=len(keys))
    for key, dst_key in key_iterator(d, keys, dst_keys, allow_missing_keys=allow_missing_keys):
        d[dst_key] = d[key][indices]
    return d


def normalize_scaled(
    data: DictStr,
    *,
    keys: KeyCollection,
    dst_keys: Optional[KeyCollection] = None,
    allow_missing_keys: bool = False,
) -> DictStr:
    """Normalize the scale of a dictionary.

    See Also:
        `torch_pointcloud.transforms.functional.normalize_scale` for more details.

    Args:
        data: The dictionary data to apply the transform to.
        keys: The keys to normalize the scale of.
        dst_keys: The keys to store the normalized scale in.
        allow_missing_keys: If `True`, the transform will not raise an error if the keys are not present in the data.

    Returns:
        The transformed dictionary data.

    Example:
        ```python
        import torch
        import torch_pointcloud.transforms.dictionary.functional as F
        data = {"pos": torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])}
        F.normalize_scaled(data, keys=["pos"])
        # {"pos": tensor([[-0.3015, -0.3015, -0.3015], [0.9045, -0.3015, -0.3015], [-0.3015, 0.9045, -0.3015]])}
        ```
    """
    d = dict(data)  # avoid modifying the original data
    keys = ensure_tuple(keys)
    dst_keys = ensure_tuple_size(dst_keys or keys, size=len(keys))
    iterator = key_iterator(d, keys, dst_keys, allow_missing_keys=allow_missing_keys)
    for key, dst_key in iterator:
        d[dst_key] = F.normalize_scale(d[key])

    return d


def remove_near_origind(
    data: DictStr,
    *,
    pos_key: str,
    keys: Optional[KeyCollection] = None,
    dst_keys: Optional[KeyCollection] = None,
    radius: float = 1e-3,
    allow_missing_keys: bool = False,
) -> DictStr:
    r"""Remove points that are within a given radius of the origin.
    This function is used to functionally remove points that are within a given radius of the origin.

    See Also:
        `torch_pointcloud.transforms.functional.remove_near_origin` for more details.

    Args:
        data: The dictionary data to apply the transform to.
        pos_key: The key containing the positions / coordinates, used to compute the distance from the origin.
        keys: The keys to remove the near origin points from.
        dst_keys: The keys to store the near origin points in.
        radius: The radius of the sphere.
        return_mask: Whether to return the mask of the points removed.
        allow_missing_keys: If `True`, the transform will not raise an error if the keys are not present in the data.

    Returns:
        The transformed dictionary data.

    Example:
        ```python
        import torch
        import torch_pointcloud.transforms.dictionary.functional as F
        data = {"pos": torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])}
        F.remove_near_origind(data, keys=["pos"], pos_key="pos", radius=1.0)
        # {"pos": tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])}
        ```
    """
    assert_keys_in_data(data, pos_key)

    d = dict(data)  # avoid modifying the original data
    keys = [pos_key] + ensure_list(keys, none_as_empty=True)
    dst_keys = ensure_tuple_size(dst_keys or keys, size=len(keys))
    _, mask = F.remove_near_origin(d[pos_key], radius=radius, return_mask=True)
    for key, dst_key in key_iterator(d, keys, dst_keys, allow_missing_keys=allow_missing_keys):
        d[dst_key] = d[key][mask]
    return d


def absd(
    data: DictStr,
    *,
    keys: KeyCollection,
    dst_keys: Optional[KeyCollection] = None,
    inplace: bool = False,
    allow_missing_keys: bool = False,
) -> DictStr:
    """Make the input tensor absolute.

    See Also:
        `torch_pointcloud.transforms.functional.abs` for more details.

    Args:
        data: The dictionary data to apply the transform to.
        keys: The keys to make the tensor absolute.
        dst_keys: The keys to store the absolute values in.
        inplace: Whether to perform the operation in place.
        allow_missing_keys: If `True`, the transform will not raise an error if the keys are not present in the data.

    Returns:
        The transformed dictionary data.

    Example:

        ```python
        import torch
        import torch_pointcloud.transforms.dictionary.functional as F
        data = {"pos": torch.tensor([-1.0, 2.0, -3.0])}
        F.absd(data, keys=["pos"])
        # {"pos": tensor([1.0, 2.0, 3.0])}
        ```
    """
    d = dict(data)  # avoid modifying the original data
    keys = ensure_tuple(keys)
    dst_keys = ensure_tuple_size(dst_keys or keys, size=len(keys))
    iterator = key_iterator(d, keys, dst_keys, allow_missing_keys=allow_missing_keys)
    for key, dst_key in iterator:
        d[dst_key] = F.abs(d[key], inplace=inplace)
    return d


def bounding_boxd(
    data: DictStr,
    *,
    keys: KeyCollection,
    dst_keys: Optional[KeyCollection] = None,
    dim: int = -1,
    allow_missing_keys: bool = False,
) -> DictStr:
    """Compute the bounding box of a tensor.

    See Also:
        `torch_pointcloud.transforms.functional.bounding_box` for more details.

    Args:
        data: The dictionary data to apply the transform to.
        keys: The keys to compute the bounding box of.
        dst_keys: The keys to store the bounding box in.
        dim: The dimension to compute the bounding box over.
        allow_missing_keys: If `True`, the transform will not raise an error if the keys are not present in the data.

    Returns:
        The transformed dictionary data.

    Example:
        ```python
        import torch
        import torch_pointcloud.transforms.dictionary.functional as F
        data = {"pos": torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])}
        F.bounding_boxd(data, keys=["pos"])
        # {"pos": (1.0, 2.0, 3.0, 7.0, 8.0, 9.0)}
        ```
    """
    d = dict(data)  # avoid modifying the original data
    keys = ensure_tuple(keys)
    dst_keys = ensure_tuple_size(dst_keys or keys, size=len(keys))

    iterator = key_iterator(d, keys, dst_keys, allow_missing_keys=allow_missing_keys)
    for key, dst_key in iterator:
        d[dst_key] = F.bounding_box(d[key], dim=dim)
    return d


def inbox_maskd(
    data: DictStr,
    *,
    keys: KeyCollection,
    bbox_key: str,
    dst_keys: Optional[KeyCollection] = None,
    dim: int = -1,
    allow_missing_keys: bool = False,
) -> DictStr:
    """Create a mask for the input tensor that is within a given bounding box.

    See Also:
        `torch_pointcloud.transforms.functional.inbox_mask` for more details.

    Args:
        data: The dictionary data to apply the transform to.
        keys: The keys to create the mask for.
        bbox_key: The key to store the bounding box in.
        dst_keys: The keys to store the mask in.
        dim: The dimension to create the mask over.
        allow_missing_keys: If `True`, the transform will not raise an error if the keys are not present in the data.
    """
    d = dict(data)  # avoid modifying the original data
    keys = ensure_tuple(keys)
    dst_keys = ensure_tuple_size(dst_keys or keys, size=len(keys))
    if bbox_key not in d:
        raise KeyError(f"Bounding box key {bbox_key!r} was missing in the data (available keys: {', '.join(d.keys())})")

    bbox = d[bbox_key]
    iterator = key_iterator(d, keys, dst_keys, allow_missing_keys=allow_missing_keys)
    for key, dst_key in iterator:
        d[dst_key] = F.inbox_mask(d[key], bbox, dim=dim)
    return d


def apply_maskd(
    data: DictStr,
    *,
    keys: KeyCollection,
    mask_key: str,
    dst_keys: Optional[KeyCollection] = None,
    allow_missing_keys: bool = False,
) -> DictStr:
    """Functional transform to apply a mask to input tensors stored in a dictionary.

    See Also:
        `torch_pointcloud.transforms.functional.apply_mask` for more details.

    Args:
        data: The dictionary data to apply the transform to.
        keys: The keys to apply the mask to.
        mask_key: The key to store the mask in.
        dst_keys: The keys to store the transformed data in.
        allow_missing_keys: If `True`, the transform will not raise an error if the keys are not present in the data.

    Returns:
        The transformed dictionary data.

    Examples:
        >>> import torch
        >>> from torch_pointcloud.transforms.dictionary.functional import apply_maskd
        >>> data = {"pos": torch.tensor([1.0, 2.0, 3.0]), "mask": torch.tensor([True, False, True])}
        >>> apply_maskd(data, keys=["pos"], mask_key="mask", dst_keys=["pos"])
        {"pos": tensor([1.0, 3.0])}
    """
    d = dict(data)  # avoid modifying the original data
    keys = ensure_tuple(keys)
    dst_keys = ensure_tuple_size(dst_keys or keys, size=len(keys))
    iterator = key_iterator(d, keys, dst_keys, allow_missing_keys=allow_missing_keys)
    assert_keys_in_data(d, mask_key)

    mask = d[mask_key]
    for key, dst_key in iterator:
        d[dst_key] = F.apply_mask(d[key], mask)
    return d
