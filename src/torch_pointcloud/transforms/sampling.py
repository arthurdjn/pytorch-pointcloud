"""Transforms that select a subset of the points."""

from typing import Any, Dict, Literal, Optional, Tuple, Union, overload

import torch
from torch import Tensor

from torch_pointcloud.ops.cluster import fps
from torch_pointcloud.utils.conversion import ensure_tuple, ensure_tuple_size
from torch_pointcloud.utils.random import Randomizable
from torch_pointcloud.utils.types import KeyCollection

from .base import DictTransform
from .masking import sphere_mask

__all__ = [
    "FarthestPointSample",
    "RandomDropout",
    "RandomSample",
    "RandomSampleFaceVertices",
    "ShufflePoint",
    "Slice",
    "SphereCrop",
]


@overload
def random_sample(
    tensor: Tensor,
    num_samples: int,
    return_indices: Literal[True],
    replace: bool = False,
    generator: Optional[torch.Generator] = None,
) -> Tuple[Tensor, Tensor]: ...


@overload
def random_sample(
    tensor: Tensor,
    num_samples: int,
    return_indices: Literal[False] = False,
    replace: bool = False,
    generator: Optional[torch.Generator] = None,
) -> Tensor: ...


def random_sample(
    tensor: Tensor,
    num_samples: int,
    return_indices: bool = False,
    replace: bool = False,
    generator: Optional[torch.Generator] = None,
) -> Union[Tensor, Tuple[Tensor, Tensor]]:
    r"""Randomly sample a fixed number of values from a tensor.

    Note:
        The data is sampled uniformly along `dim=0`.

    Args:
        tensor: The input tensor of shape $(N, \ldots)$.
        num_samples: The number of values to sample.
        return_indices: Whether to return the indices of the sampled values.
        replace: If `True`, sample with replacement (duplicates allowed). If `False`,
            sample without replacement when $N \geq \text{num\_samples}$; when
            $\text{num\_samples} > N$ the draw falls back to replacement so the output
            always has `num_samples` rows.
        generator: The generator for the random number generator.

    Returns:
        If `return_indices` is `True`, the function returns a tuple of the sampled values and their indices.
        Otherwise, it returns the sampled values.

    Raises:
        ValueError: If `num_samples > 0` and the input is empty.
    """
    n = tensor.size(0)
    if num_samples == 0:
        indices = torch.empty(0, dtype=torch.long, device=tensor.device)
    elif n == 0:
        raise ValueError(f"Cannot sample {num_samples} values from an empty tensor (N=0).")
    elif replace or num_samples > n:
        indices = torch.randint(0, n, (num_samples,), generator=generator, device=tensor.device)
    else:
        indices = torch.randperm(n, generator=generator, device=tensor.device)[:num_samples]

    if return_indices:
        return tensor[indices], indices
    return tensor[indices]


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
        sampled_tensor, indices = random_sample(
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


@overload
def random_sample_face_vertices(
    vertices: Tensor,
    face: Tensor,
    num_samples: int,
    return_normals: Literal[True],
    generator: Optional[torch.Generator] = None,
) -> Tuple[Tensor, Tensor]: ...


@overload
def random_sample_face_vertices(
    vertices: Tensor,
    face: Tensor,
    num_samples: int,
    return_normals: Literal[False] = False,
    generator: Optional[torch.Generator] = None,
) -> Tensor: ...


@overload
def random_sample_face_vertices(
    vertices: Tensor,
    face: Tensor,
    num_samples: int,
    return_normals: bool,
    generator: Optional[torch.Generator] = None,
) -> Union[Tensor, Tuple[Tensor, Tensor]]: ...


def random_sample_face_vertices(
    vertices: Tensor,
    face: Tensor,
    num_samples: int,
    return_normals: bool = False,
    generator: Optional[torch.Generator] = None,
) -> Union[Tensor, Tuple[Tensor, Tensor]]:
    """Randomly sample a fixed number of vertices from a 3D mesh (vertices, face),
    using:

    Note:
        The data is sampled uniformly from the mesh.

    Args:
        vertices: The input tensor.
        face: The input tensor.
        num_samples: The number of vertices to sample.
        return_normals: Whether to return the normal of the sampled vertices.
        generator: The generator for the random number generator.

    Returns:
        If `return_normals` is `True`, the function returns a tuple of the sampled vertices and their normal.
        Otherwise, it returns the sampled vertices.
    """
    pos_max = vertices.abs().max()
    vertices = vertices / pos_max

    v01 = vertices[face[:, 1]] - vertices[face[:, 0]]
    v02 = vertices[face[:, 2]] - vertices[face[:, 0]]
    areas = v01.cross(v02, dim=1)
    areas = areas.norm(p=2, dim=1).abs() / 2

    probs = areas / areas.sum()
    samples = torch.multinomial(probs, num_samples, replacement=True, generator=generator)
    face = face[samples]

    frac = torch.rand(num_samples, 2, device=vertices.device, generator=generator)
    mask = frac.sum(dim=-1) > 1
    frac[mask] = 1 - frac[mask]

    v01 = vertices[face[:, 1]] - vertices[face[:, 0]]
    v02 = vertices[face[:, 2]] - vertices[face[:, 0]]

    if return_normals:
        normal = torch.nn.functional.normalize(v01.cross(v02, dim=1), p=2)

    vertices = vertices[face[:, 0]]
    vertices += frac[:, :1] * v01
    vertices += frac[:, 1:] * v02
    vertices = vertices * pos_max

    if return_normals:
        return vertices, normal
    return vertices


class RandomSampleFaceVertices(DictTransform, Randomizable):
    """Randomly sample a fixed number of vertices from a 3D mesh stored in a dictionary.

    ![RandomSampleFaceVertices before / after](../../assets/transforms/random_sample_face_vertices.png)

    See Also:
        `torch_pointcloud.transforms.functional.random_sample_face_vertices`

    Args:
        keys: The keys holding vertex positions.
        face_key: The keys holding the face indices.
        dst_normal_key: The key to store the computed normals in.
        num_samples: The number of vertices to sample.
        seed: Seed of the transform's own random stream; `None` draws from the global generator (see `Randomizable`).
        allow_missing_keys: If `True`, the transform will not raise an error if the keys are not present in the data.
    """

    def __init__(
        self,
        *,
        keys: KeyCollection,
        face_key: KeyCollection,
        dst_normal_key: Optional[KeyCollection] = "normal",
        num_samples: int,
        seed: Optional[int] = None,
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.face_key = ensure_tuple_size(face_key, len(self.keys))
        self.num_samples = num_samples
        self.dst_normal_key = ensure_tuple_size(dst_normal_key, len(self.keys))
        self.set_random_state(seed)

    def transform(self, data: Dict[str, Any]) -> Dict[str, Any]:
        data = dict(data)
        for key, face_key, dst_normal_key in self.iter_keys(data, self.face_key, self.dst_normal_key):
            pos, normal = random_sample_face_vertices(
                data[key],
                data[face_key],
                self.num_samples,
                generator=self.R,
                return_normals=True,
            )
            data[key] = pos
            if dst_normal_key is not None:
                data[dst_normal_key] = normal
        return data


def farthest_point_sample(
    pos: Tensor,
    num_samples: Optional[int] = None,
    ratio: Optional[float] = None,
    random_start: bool = False,
) -> Tensor:
    """Farthest-point sampling (FPS) from a tensor of positions.

    Thin wrapper around `torch_pointcloud.ops.cluster.fps`, provided for
    convenience and naming symmetry with `random_sample`.

    See Also:
        `torch_pointcloud.ops.cluster.fps` for more details and advanced usage.

    Args:
        pos: The input tensor of shape $(N, D)$.
        num_samples: The number of points to sample.
        ratio: The ratio of points to sample.
        random_start: Whether to start the sampling from a random point.

    Returns:
        The indices of the sampled points.

    Examples:
        ```pycon
        >>> import torch
        >>> from torch_pointcloud.transforms.functional import farthest_point_sample
        >>> pos = torch.randn(100, 3)
        >>> idx = farthest_point_sample(pos, num_samples=10)  # doctest: +SKIP
        >>> print(idx.shape)  # doctest: +SKIP
        torch.Size([10])

        ```
    """
    return fps(pos, num_nodes=num_samples, ratio=ratio, random_start=random_start)


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
        indices = farthest_point_sample(
            d[self.pos_key],
            num_samples=self.num_samples,
            ratio=self.ratio,
            random_start=self.random_start,
        )
        for key in self.iter_keys(d):
            d[key] = d[key][indices]
        if self.dst_index_key is not None:
            prior = d.get(self.dst_index_key)
            d[self.dst_index_key] = indices if prior is None else prior[indices]
        return d


def random_dropout_mask(
    n: int,
    drop_ratio: float,
    device: Optional[torch.device] = None,
    generator: Optional[torch.Generator] = None,
) -> Tensor:
    """Return a boolean keep-mask of length `n` where each entry is kept with probability `1 - drop_ratio`.

    Args:
        n: Number of points.
        drop_ratio: Probability of dropping a point. Must be in $[0, 1)$.
        device: Output device.
        generator: Random generator for reproducibility.

    Returns:
        Boolean tensor of shape $(n,)$.

    Raises:
        ValueError: If `drop_ratio` is not in `[0, 1)`.
    """
    if not 0.0 <= drop_ratio < 1.0:
        raise ValueError(f"drop_ratio must be in [0, 1); got {drop_ratio}.")
    device = device or torch.device("cpu")
    rand = torch.rand(n, device=device, generator=generator)
    return rand >= drop_ratio


class RandomDropout(DictTransform, Randomizable):
    """Randomly drop a fraction of points across all listed keys.

    The same boolean keep-mask is applied to every key so per-point
    correspondence is preserved. Sampling is once per call, with a drop ratio drawn uniformly from
    `drop_ratio_range`.

    === "Object"

        ![RandomDropout on an object](../../assets/transforms/random_dropout.png)

    === "Scene"

        ![RandomDropout on a room](../../assets/transforms/random_dropout_scene.png)

    See Also:
        `torch_pointcloud.transforms.functional.random_dropout_mask`

    Args:
        keys: Keys to subset. All must share the same leading dimension $N$.
        drop_ratio_range: Min and max fraction of points to drop; `(r, r)` drops a fixed fraction. Must lie in
            $[0, 1)$.
        p: Probability of applying the transform.
        seed: Seed of the transform's own random stream; `None` draws from the global generator (see `Randomizable`).
        dst_index_key: Key for the output-to-input row map (see the module docs on sampling keys); `None` (the
            default) disables it.
        allow_missing_keys: If `True`, silently skip absent keys.
    """

    def __init__(
        self,
        keys: KeyCollection,
        drop_ratio_range: Tuple[float, float] = (0.1, 0.1),
        p: float = 1.0,
        seed: Optional[int] = None,
        dst_index_key: Optional[str] = None,
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        if not 0.0 <= drop_ratio_range[0] <= drop_ratio_range[1] < 1.0:
            raise ValueError(f"drop_ratio_range must satisfy 0 <= min <= max < 1; got {drop_ratio_range}.")
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"p must be in [0, 1]; got {p}.")

        self.drop_ratio_range = drop_ratio_range
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

        lo, hi = self.drop_ratio_range
        drop_ratio = torch.empty(1).uniform_(lo, hi, generator=self.R).item()
        keep = random_dropout_mask(n, drop_ratio, device=device, generator=self.R)
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
        mask = sphere_mask(pos, center, self.radius, dim=-1)
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


def shuffle_indices(
    n: int,
    device: Optional[torch.device] = None,
    generator: Optional[torch.Generator] = None,
) -> Tensor:
    """Return a random permutation of `[0, n)`.

    Args:
        n: Sequence length.
        device: Output device.
        generator: Random generator for reproducibility.

    Returns:
        Long tensor of shape $(n,)$.
    """
    device = device or torch.device("cpu")
    return torch.randperm(n, device=device, generator=generator)


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
        perm = shuffle_indices(n, device=device, generator=self.R)
        for key in self.iter_keys(data):
            data[key] = data[key][perm]
        if self.dst_index_key is not None:
            prior = data.get(self.dst_index_key)
            data[self.dst_index_key] = perm if prior is None else prior[perm]
        return data
