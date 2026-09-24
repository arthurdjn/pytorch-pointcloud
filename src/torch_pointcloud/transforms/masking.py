"""Transforms that build or apply point masks."""

from typing import Any, Dict, Optional

import torch

from torch_pointcloud.utils.conversion import ensure_tuple, ensure_tuple_size
from torch_pointcloud.utils.types import KeyCollection, ValueCollection

from . import functional as F
from .base import DictTransform

__all__ = [
    "ApplyMask",
    "BoxMask",
    "CubeMask",
    "RemoveNearOrigin",
    "SphereMask",
]


class RemoveNearOrigin(DictTransform):
    """Remove points that are within a given radius of the origin from dictionary entries.

    === "Object"

        ![RemoveNearOrigin on an object](../../assets/transforms/remove_near_origin.png)

    === "Scene"

        ![RemoveNearOrigin on a room](../../assets/transforms/remove_near_origin_scene.png)

    See Also:
        `torch_pointcloud.transforms.functional.remove_near_origin`

    Args:
        pos_key: The key containing the positions / coordinates, used to compute the distance from the origin.
        keys: Extra keys to filter with the same mask.
        radius: The radius of the sphere.
        dst_index_key: Key for the output-to-input row map (see the module docs on sampling keys); `None` (the
            default) disables it.
        allow_missing_keys: If `True`, the transform will not raise an error if the keys are not present in the data.
    """

    def __init__(
        self,
        pos_key: str,
        keys: Optional[KeyCollection] = None,
        radius: float = 1e-3,
        dst_index_key: Optional[str] = None,
        allow_missing_keys: bool = False,
    ) -> None:
        all_keys = ensure_tuple(keys, none_as_empty=True)
        if pos_key not in all_keys:
            all_keys = (pos_key,) + all_keys
        super().__init__(all_keys, allow_missing_keys)
        self.pos_key = pos_key
        self.radius = radius
        self.dst_index_key = dst_index_key

    def transform(self, data: Dict[str, Any]) -> Dict[str, Any]:
        d = dict(data)
        if self.pos_key not in d:
            if self.allow_missing_keys:
                return d
            raise KeyError(f"`RemoveNearOrigin` requires {self.pos_key!r} in data.")
        _, mask = F.remove_near_origin(d[self.pos_key], radius=self.radius, return_mask=True)
        for key in self.iter_keys(d):
            d[key] = d[key][mask]
        if self.dst_index_key is not None:
            index = torch.where(mask)[0]
            prior = d.get(self.dst_index_key)
            d[self.dst_index_key] = index if prior is None else prior[index]
        return d


class BoxMask(DictTransform):
    r"""Create a boolean mask for points inside an axis-aligned bounding box (AABB).

    Membership condition along `dim` (default, boundary points included):

    $$
    \text{bbmin}_j \leq x_j \leq \text{bbmax}_j \quad \forall j
    $$

    where `bbox = (*bbmin, *bbmax)` is the AABB. With `strict=True` the inequalities are strict,
    so boundary points are excluded.

    Sibling masks:

    - `CubeMask` - L∞ ball (center + radius)
    - `SphereMask` - L2 ball (center + radius)

    === "Object"

        ![BoxMask on an object](../../assets/transforms/box_mask.png)

    === "Scene"

        ![BoxMask on a room](../../assets/transforms/box_mask_scene.png)

    See Also:
        `torch_pointcloud.transforms.functional.box_mask`

    Args:
        keys: The keys to create the mask for.
        bbox: The bounding box used to mask input tensors, as `(*bbmin, *bbmax)`.
        dst_keys: The keys to store the mask in.
        dim: The dimension to create the mask over.
        strict: If `True`, use strict inequalities (points exactly on the boundary are excluded).
        allow_missing_keys: If `True`, the transform will not raise an error if
            the keys are not present in the data.
    """

    def __init__(
        self,
        keys: KeyCollection,
        bbox: tuple[float, ...],
        dst_keys: Optional[KeyCollection] = None,
        dim: int = -1,
        strict: bool = False,
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.bbox = bbox
        self.dst_keys = ensure_tuple_size(dst_keys or self.keys, size=len(self.keys))
        self.dim = dim
        self.strict = strict

    def transform(self, data: Dict[str, Any]) -> Dict[str, Any]:
        data = dict(data)
        for key, dst_key in self.iter_keys(data, self.dst_keys):
            data[dst_key] = F.box_mask(data[key], self.bbox, dim=self.dim, strict=self.strict)
        return data


class ApplyMask(DictTransform):
    """Apply a mask stored in a dictionary to other dictionary entries.

    See Also:
        `torch_pointcloud.transforms.functional.apply_mask`

    === "Object"

        ![ApplyMask on an object](../../assets/transforms/apply_mask.png)

    === "Scene"

        ![ApplyMask on a room](../../assets/transforms/apply_mask_scene.png)

    Args:
        keys: The keys to apply the mask to.
        mask_key: The key containing the mask.
        dst_keys: The keys to store the transformed data in.
        dst_index_key: Key for the output-to-input row map (see the module docs on sampling keys); `None` (the
            default) disables it. Leave it unset when `dst_keys` differ from `keys`, since the map would describe
            the `dst_keys` rows.
        allow_missing_keys: If `True`, the transform will not raise an error if the keys are not present in the data.
    """

    def __init__(
        self,
        keys: KeyCollection,
        mask_key: str,
        dst_keys: Optional[KeyCollection] = None,
        dst_index_key: Optional[str] = None,
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.mask_key = mask_key
        self.dst_keys = ensure_tuple_size(dst_keys or self.keys, size=len(self.keys))
        self.dst_index_key = dst_index_key

    def transform(self, data: Dict[str, Any]) -> Dict[str, Any]:
        d = dict(data)
        if self.mask_key not in data:
            if self.allow_missing_keys:
                return d
            raise KeyError(f"Mask key {self.mask_key!r} not found in data.")
        mask = d[self.mask_key]
        for key, dst_key in self.iter_keys(d, self.dst_keys):
            d[dst_key] = F.apply_mask(d[key], mask)
        if self.dst_index_key is not None:
            index = torch.where(mask)[0] if mask.dtype == torch.bool else mask
            prior = d.get(self.dst_index_key)
            d[self.dst_index_key] = index if prior is None else prior[index]
        return d


class CubeMask(DictTransform):
    r"""Create a boolean mask for points inside an axis-aligned cube (L∞ / Chebyshev ball).

    Membership condition along `dim`:

    $$
    \| x - c \|_{\infty} \leq r
    $$

    Geometrically, the L∞ ball of radius $r$ centered at $c$ is a hypercube
    with edge $2r$ aligned to the axes. Pair with `SphereMask` (L2) and
    `BoxMask` (AABB) for the mask family.

    === "Object"

        ![CubeMask on an object](../../assets/transforms/cube_mask.png)

    === "Scene"

        ![CubeMask on a room](../../assets/transforms/cube_mask_scene.png)

    See Also:
        `torch_pointcloud.transforms.functional.cube_mask`

    Args:
        keys: The keys to create the mask for.
        center: The center of the cube.
        radius: The radius (half-edge) of the cube.
        dim: The dimension to create the mask over.
        dst_keys: The keys to store the mask in.
        allow_missing_keys: If `True`, the transform will not raise an error if
            the keys are not present in the data.
    """

    def __init__(
        self,
        keys: KeyCollection,
        center: ValueCollection[float],
        radius: float,
        dim: int = -1,
        dst_keys: Optional[KeyCollection] = None,
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.dst_keys = ensure_tuple_size(dst_keys or self.keys, len(self.keys))
        self.center = center
        self.radius = radius
        self.dim = dim

    def transform(self, data: Dict[str, Any]) -> Dict[str, Any]:
        data = dict(data)
        for key, dst_key in self.iter_keys(data, self.dst_keys):
            x = data[key]
            if not torch.is_tensor(x):
                raise TypeError(f"Expected a tensor, got {type(x).__name__!r}.")
            data[dst_key] = F.cube_mask(x, self.center, self.radius, dim=self.dim)

        return data


class SphereMask(DictTransform):
    r"""Create a boolean mask for points inside an L2 (Euclidean) ball.

    Membership condition along `dim`:

    $$
    \| x - c \|_2 \leq r
    $$

    Pair with `CubeMask` (L∞) and `BoxMask` (AABB) for the mask family.
    `RemoveNearOrigin(radius=r)` is equivalent to
    `Compose([SphereMask(center=(0,0,0), radius=r, invert=True), ApplyMask(...)])`.

    === "Object"

        ![SphereMask on an object](../../assets/transforms/sphere_mask.png)

    === "Scene"

        ![SphereMask on a room](../../assets/transforms/sphere_mask_scene.png)

    See Also:
        `torch_pointcloud.transforms.functional.sphere_mask`

    Args:
        keys: The keys to create the mask for.
        center: The center of the sphere.
        radius: The radius of the sphere.
        dim: The dimension to compute the Euclidean norm over.
        dst_keys: The keys to store the mask in.
        allow_missing_keys: If `True`, the transform will not raise an error if
            the keys are not present in the data.
    """

    def __init__(
        self,
        keys: KeyCollection,
        center: ValueCollection[float],
        radius: float,
        dim: int = -1,
        dst_keys: Optional[KeyCollection] = None,
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.dst_keys = ensure_tuple_size(dst_keys or self.keys, len(self.keys))
        self.center = center
        self.radius = radius
        self.dim = dim

    def transform(self, data: Dict[str, Any]) -> Dict[str, Any]:
        data = dict(data)
        for key, dst_key in self.iter_keys(data, self.dst_keys):
            x = data[key]
            if not torch.is_tensor(x):
                raise TypeError(f"Expected a tensor, got {type(x).__name__!r}.")
            data[dst_key] = F.sphere_mask(x, self.center, self.radius, dim=self.dim)

        return data
