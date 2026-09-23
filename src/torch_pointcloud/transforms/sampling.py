"""Transforms that select a subset of the points."""

from typing import Any, Dict, Optional

import torch
from torch import Tensor

from torch_pointcloud.utils.conversion import ensure_tuple, ensure_tuple_size
from torch_pointcloud.utils.random import Randomizable
from torch_pointcloud.utils.types import KeyCollection

from . import functional as F
from .base import DictTransform

__all__ = [
    "FarthestPointSample",
    "RandomDropout",
    "RandomSample",
    "RandomSampleFaceVertices",
    "ShufflePoint",
    "Slice",
    "SphereCrop",
]


class RandomSample(DictTransform, Randomizable):
    """Randomly sample a fixed number of points from dict entries.

    If multiple keys are provided, the same indices are used for all keys, ensuring
    correspondence between the sampled values.

    === "Object"

        ![RandomSample on an object](../../assets/transforms/random_sample.png)

    === "Scene"

        ![RandomSample on a room](../../assets/transforms/random_sample_scene.png)

    See Also:
        `torch_pointcloud.transforms.functional.random_sample`

    Args:
        keys: The keys to sample from.
        num_samples: The number of values to sample.
        replace: If `True`, sample with replacement (duplicates allowed). If `False`
            (default), sample without replacement when the first sampled key has at least
            `num_samples` points; when `num_samples` exceeds that count the draw falls back
            to replacement so the output always has `num_samples` rows.
        seed: Seed of the transform's own random stream; `None` draws from the global generator (see `Randomizable`).
        dst_index_key: Key for the output-to-input row map (see the module docs on sampling keys); `None` (the
            default) disables it.
        allow_missing_keys: If `True`, the transform will not raise an error if the keys are not present in the data.

    Raises:
        ValueError: If the first sampled tensor is empty and `num_samples > 0`.
    """

    def __init__(
        self,
        keys: KeyCollection,
        num_samples: int,
        replace: bool = False,
        seed: Optional[int] = None,
        dst_index_key: Optional[str] = None,
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.num_samples = num_samples
        self.replace = replace
        self.set_random_state(seed)
        self.dst_index_key = dst_index_key

    def transform(self, data: Dict[str, Any]) -> Dict[str, Any]:
        d = dict(data)
        iterator = self.iter_keys(d)
        try:
            first_key = next(iterator)
        except StopIteration:
            return d
        sampled_tensor, indices = F.random_sample(
            d[first_key],
            self.num_samples,
            return_indices=True,
            replace=self.replace,
            generator=self.R,
        )
        d[first_key] = sampled_tensor
        for key in iterator:
            d[key] = d[key][indices]
        if self.dst_index_key is not None:
            prior = d.get(self.dst_index_key)
            d[self.dst_index_key] = indices if prior is None else prior[indices]
        return d


class RandomSampleFaceVertices(DictTransform, Randomizable):
    """Randomly sample a fixed number of vertices from a 3D mesh stored in a dictionary.

    ![RandomSampleFaceVertices before / after](../../assets/transforms/random_sample_face_vertices.png)

    See Also:
        `torch_pointcloud.transforms.functional.random_sample_face_vertices`

    Args:
        keys: The keys holding vertex positions.
        face_key: The keys holding the face indices.
        normal_key: The key to store the computed normals in.
        num_samples: The number of vertices to sample.
        seed: Seed of the transform's own random stream; `None` draws from the global generator (see `Randomizable`).
        allow_missing_keys: If `True`, the transform will not raise an error if the keys are not present in the data.
    """

    def __init__(
        self,
        *,
        keys: KeyCollection,
        face_key: KeyCollection,
        normal_key: Optional[KeyCollection] = "normal",
        num_samples: int,
        seed: Optional[int] = None,
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.face_key = ensure_tuple_size(face_key, len(self.keys))
        self.num_samples = num_samples
        self.normal_key = ensure_tuple_size(normal_key, len(self.keys))
        self.set_random_state(seed)

    def transform(self, data: Dict[str, Any]) -> Dict[str, Any]:
        data = dict(data)
        for key, face_key, normal_key in self.iter_keys(data, self.face_key, self.normal_key):
            pos, normal = F.random_sample_face_vertices(
                data[key],
                data[face_key],
                self.num_samples,
                generator=self.R,
                return_normals=True,
            )
            data[key] = pos
            if normal_key is not None:
                data[normal_key] = normal
        return data


class FarthestPointSample(DictTransform):
    """Farthest-point sampling (FPS) of a dictionary entry.

    Iteratively picks the point that maximizes the minimum distance to the
    already-selected set, producing a well-distributed subset. Matches the FPS
    convention used by PointNet++, PointNeXt, KPConv, and others.

    === "Object"

        ![FarthestPointSample on an object](../../assets/transforms/farthest_point_sample.png)

    === "Scene"

        ![FarthestPointSample on a room](../../assets/transforms/farthest_point_sample_scene.png)

    See Also:
        `torch_pointcloud.transforms.functional.farthest_point_sample`

    Note:
        The underlying `fps` does not accept a `torch.Generator`. To make
        `random_start=True` reproducible, seed PyTorch globally via
        `torch.manual_seed(...)` before applying this transform.

    Args:
        pos_key: The key holding the positions used for FPS.
        keys: Extra keys to subsample with the same indices.
        num_samples: The number of points to sample.
        ratio: The ratio of points to sample.
        random_start: Whether to start the sampling from a random point.
        dst_index_key: Key for the output-to-input row map (see the module docs on sampling keys); `None` (the
            default) disables it.
        allow_missing_keys: If `True`, the transform will not raise an error if the keys are not present in the data.
    """

    def __init__(
        self,
        pos_key: str,
        keys: Optional[KeyCollection] = None,
        num_samples: Optional[int] = None,
        ratio: Optional[float] = None,
        random_start: bool = False,
        dst_index_key: Optional[str] = None,
        allow_missing_keys: bool = False,
    ) -> None:
        all_keys = ensure_tuple(keys, none_as_empty=True)
        if pos_key not in all_keys:
            all_keys = (pos_key,) + all_keys
        super().__init__(all_keys, allow_missing_keys)
        self.pos_key = pos_key
        self.num_samples = num_samples
        self.ratio = ratio
        self.random_start = random_start
        self.dst_index_key = dst_index_key

    def transform(self, data: Dict[str, Any]) -> Dict[str, Any]:
        d = dict(data)
        if self.pos_key not in d:
            if self.allow_missing_keys:
                return d
            raise KeyError(f"`FarthestPointSample` requires {self.pos_key!r} in data.")
        indices = F.farthest_point_sample(
            d[self.pos_key], num_samples=self.num_samples, ratio=self.ratio, random_start=self.random_start
        )
        for key in self.iter_keys(d):
            d[key] = d[key][indices]
        if self.dst_index_key is not None:
            prior = d.get(self.dst_index_key)
            d[self.dst_index_key] = indices if prior is None else prior[indices]
        return d


class RandomDropout(DictTransform, Randomizable):
    """Randomly drop a fraction of points across all listed keys.

    The same boolean keep-mask is applied to every key so per-point
    correspondence is preserved. Sampling is once per call.

    === "Object"

        ![RandomDropout on an object](../../assets/transforms/random_dropout.png)

    === "Scene"

        ![RandomDropout on a room](../../assets/transforms/random_dropout_scene.png)

    See Also:
        `torch_pointcloud.transforms.functional.random_dropout_mask`

    Args:
        keys: Keys to subset. All must share the same leading dimension $N$.
        p_drop: Fraction of points to drop per call (uniform across points).
            Must lie in $[0, 1)$.
        p: Probability of applying the transform.
        seed: Seed of the transform's own random stream; `None` draws from the global generator (see `Randomizable`).
        dst_index_key: Key for the output-to-input row map (see the module docs on sampling keys); `None` (the
            default) disables it.
        allow_missing_keys: If `True`, silently skip absent keys.
    """

    def __init__(
        self,
        keys: KeyCollection,
        p_drop: float = 0.1,
        p: float = 1.0,
        seed: Optional[int] = None,
        dst_index_key: Optional[str] = None,
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        if not 0.0 <= p_drop < 1.0:
            raise ValueError(f"p_drop must be in [0, 1); got {p_drop}.")
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"p must be in [0, 1]; got {p}.")

        self.p_drop = p_drop
        self.p = p
        self.set_random_state(seed)
        self.dst_index_key = dst_index_key

    def transform(self, data: Dict[str, Any]) -> Dict[str, Any]:
        data = dict(data)
        first_key = next(iter(self.iter_keys(data)), None)
        if first_key is None:
            return data

        n = data[first_key].shape[0]
        device = data[first_key].device
        if torch.rand(1, generator=self.R).item() >= self.p:
            if self.dst_index_key is not None and self.dst_index_key not in data:
                data[self.dst_index_key] = torch.arange(n, device=device)
            return data

        keep = F.random_dropout_mask(n, self.p_drop, device=device, generator=self.R)
        for key in self.iter_keys(data):
            data[key] = data[key][keep]

        if self.dst_index_key is not None:
            index = torch.where(keep)[0]
            prior = data.get(self.dst_index_key)
            data[self.dst_index_key] = index if prior is None else prior[index]
        return data


class SphereCrop(DictTransform, Randomizable):
    """Keep only points inside an L2 sphere of given radius.

    The mask is computed from `pos_key` and applied to every listed `keys`.
    Equivalent to `Compose([SphereMask(...), ApplyMask(...)])`, kept as a
    convenience preset (the dual of `RemoveNearOrigin`).

    When `max_nodes` is set and the sphere holds more than `max_nodes` points,
    only the `max_nodes` nearest the center are kept, bounding memory on large scenes.

    === "Object"

        ![SphereCrop on an object](../../assets/transforms/sphere_crop.png)

    === "Scene"

        ![SphereCrop on a room](../../assets/transforms/sphere_crop_scene.png)

    Args:
        pos_key: Key with positions used to compute the mask.
        keys: Extra keys to filter with the same mask.
        center: Center of the sphere. If `"centroid"`, uses the per-cloud centroid;
            if `"random_point"`, picks a random point as the center; otherwise treat as a 3-vector.
        radius: Radius of the sphere (Euclidean).
        max_nodes: Optional cap on the number of kept points. If `None` (default),
            no cap is applied; otherwise the `max_nodes` points nearest the center are kept.
        p: Probability of applying the transform.
        seed: Seed for the random center (`center="random_point"`) and the probability draw;
            `None` draws from the global generator (see `Randomizable`).
        dst_index_key: Key for the output-to-input row map (see the module docs on sampling keys); `None` (the
            default) disables it.
        allow_missing_keys: If `True`, silently skip absent keys.
    """

    def __init__(
        self,
        pos_key: str,
        radius: float,
        max_nodes: Optional[int] = None,
        keys: Optional[KeyCollection] = None,
        center: Any = "centroid",
        p: float = 1.0,
        seed: Optional[int] = None,
        dst_index_key: Optional[str] = None,
        allow_missing_keys: bool = False,
    ) -> None:
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"p must be in [0, 1]; got {p}.")

        all_keys = ensure_tuple(keys, none_as_empty=True)
        if pos_key not in all_keys:
            all_keys = (pos_key,) + all_keys
        super().__init__(all_keys, allow_missing_keys)
        self.pos_key = pos_key
        self.radius = radius
        self.max_nodes = max_nodes
        self.center = center
        self.p = p
        self.set_random_state(seed)
        self.dst_index_key = dst_index_key

    def _resolve_center(self, pos: torch.Tensor) -> torch.Tensor:
        if isinstance(self.center, str):
            if self.center == "centroid":
                return pos.mean(dim=0)
            if self.center == "random_point":
                if pos.shape[0] == 0:
                    return pos.new_zeros(pos.shape[-1])
                idx = int(torch.randint(0, pos.shape[0], (1,), generator=self.R).item())
                return pos[idx]
            raise ValueError(f"Invalid center: {self.center!r}. Expected 'centroid', 'random_point', or a 3-vector.")
        return torch.as_tensor(self.center, device=pos.device, dtype=pos.dtype)

    def transform(self, data: Dict[str, Any]) -> Dict[str, Any]:
        data = dict(data)
        if self.pos_key not in data:
            if self.allow_missing_keys:
                return data
            raise KeyError(f"`SphereCrop` requires {self.pos_key!r} in data.")
        if torch.rand(1, generator=self.R).item() >= self.p:
            if self.dst_index_key is not None and self.dst_index_key not in data:
                n = data[self.pos_key].shape[0]
                data[self.dst_index_key] = torch.arange(n, device=data[self.pos_key].device)
            return data
        # pos may be integer grid coords (post-Voxelize); norm() needs float.
        pos = data[self.pos_key].float()
        center = self._resolve_center(pos)
        mask = F.sphere_mask(pos, center, self.radius, dim=-1)
        if self.max_nodes is not None and int(mask.sum()) > self.max_nodes:
            dist = (pos - center).norm(dim=-1)
            keep = torch.topk(dist, self.max_nodes, largest=False).indices
            mask = torch.zeros_like(mask)
            mask[keep] = True
        for key in self.iter_keys(data):
            data[key] = data[key][mask]
        if self.dst_index_key is not None:
            index = torch.where(mask)[0]
            prior = data.get(self.dst_index_key)
            data[self.dst_index_key] = index if prior is None else prior[index]
        return data


class Slice(DictTransform):
    """Slice each listed tensor along a chosen dimension via standard Python slicing.

    Useful for taking the first $N$ rows (e.g. on FPS-sorted point clouds), or extracting a
    single column of `pos` into a separate key (set `dim=1` with `start=axis, stop=axis+1`).

    ![Slice diagram](../../assets/transforms/slice.png)

    Args:
        keys: Keys to slice.
        start: Start index (inclusive). `None` is equivalent to `0`.
        stop: Stop index (exclusive). `None` means "to the end".
        step: Stride between selected positions. `None` is equivalent to `1`.
        dim: Dimension along which to slice. Defaults to `0` (the row axis).
        dst_keys: Where to store results. Defaults to `keys`.
        dst_index_key: Key for the output-to-input row map (see the module docs on sampling keys), written only
            when `dim == 0`; `None` (the default) disables it. Leave it unset when `dst_keys` differ from `keys`,
            since the map would describe the `dst_keys` rows.
        allow_missing_keys: If `True`, silently skip absent keys.

    Example:
        ```python
        from torch_pointcloud.transforms import Slice

        # First 1024 rows of `pos` (e.g. an FPS-sorted ModelNet sample).
        Slice(keys="pos", stop=1024)

        # Extract the gravity axis (z=2) into a `(N, 1)` `height` key.
        Slice(keys="pos", start=2, stop=3, dim=1, dst_keys="height")
        ```
    """

    def __init__(
        self,
        keys: KeyCollection,
        start: Optional[int] = None,
        stop: Optional[int] = None,
        step: Optional[int] = None,
        dim: int = 0,
        dst_keys: Optional[KeyCollection] = None,
        dst_index_key: Optional[str] = None,
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.dst_keys = ensure_tuple_size(dst_keys or self.keys, len(self.keys))
        self.start = start
        self.stop = stop
        self.step = step
        self.dim = dim
        self.dst_index_key = dst_index_key

    def transform(self, data: Dict[str, Any]) -> Dict[str, Any]:
        data = dict(data)
        sl = slice(self.start, self.stop, self.step)
        index: Optional[Tensor] = None
        for key, dst_key in self.iter_keys(data, self.dst_keys):
            x = data[key]
            idx: list[Any] = [slice(None)] * x.ndim
            idx[self.dim] = sl
            data[dst_key] = x[tuple(idx)]
            if index is None and self.dim == 0:
                index = torch.arange(x.shape[0], device=getattr(x, "device", None))[sl]
        if index is not None and self.dst_index_key is not None:
            prior = data.get(self.dst_index_key)
            data[self.dst_index_key] = index if prior is None else prior[index]
        return data


class ShufflePoint(DictTransform, Randomizable):
    """Randomly permute the order of points across listed keys.

    The same permutation is applied to every key so per-point correspondence
    is preserved. Useful before `RandomSample` when you want to break any
    structural ordering in the input.

    See Also:
        `torch_pointcloud.transforms.functional.shuffle_indices`

    === "Object"

        ![ShufflePoint on an object](../../assets/transforms/shuffle_point.png)

    === "Scene"

        ![ShufflePoint on a room](../../assets/transforms/shuffle_point_scene.png)

    Args:
        keys: Keys to permute. All must share the same leading dimension $N$.
        p: Probability of applying the transform.
        seed: Seed of the transform's own random stream; `None` draws from the global generator (see `Randomizable`).
        dst_index_key: Key for the output-to-input row map (see the module docs on sampling keys); `None` (the
            default) disables it.
        allow_missing_keys: If `True`, silently skip absent keys.
    """

    def __init__(
        self,
        keys: KeyCollection,
        p: float = 1.0,
        seed: Optional[int] = None,
        dst_index_key: Optional[str] = None,
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"p must be in [0, 1]; got {p}.")
        self.p = p
        self.set_random_state(seed)
        self.dst_index_key = dst_index_key

    def transform(self, data: Dict[str, Any]) -> Dict[str, Any]:
        data = dict(data)
        first_key = next(iter(self.iter_keys(data)), None)
        if first_key is None:
            return data
        n = data[first_key].shape[0]
        device = data[first_key].device
        if torch.rand(1, generator=self.R).item() >= self.p:
            if self.dst_index_key is not None and self.dst_index_key not in data:
                data[self.dst_index_key] = torch.arange(n, device=device)
            return data
        perm = F.shuffle_indices(n, device=device, generator=self.R)
        for key in self.iter_keys(data):
            data[key] = data[key][perm]
        if self.dst_index_key is not None:
            prior = data.get(self.dst_index_key)
            data[self.dst_index_key] = perm if prior is None else prior[perm]
        return data
