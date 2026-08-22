"""Dict transforms.

All transforms in this module operate on a **single scene** (one sample, pre-collate).
The `batch` key is not consumed at this layer; reductions like `mean`, `min`, and `max`
are computed over the whole tensor, not per-batch. Apply transforms before DataLoader
collation if you want per-scene behavior.

Transforms are non-mutating: each transform returns a new shallow-copy dict. Tensors
inside the dict are not cloned unless the transform's documentation says so.
"""

import math
from abc import ABCMeta, abstractmethod
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Generator,
    Iterable,
    Literal,
    Optional,
    Sequence,
    Tuple,
    Union,
    get_args,
)

import numpy as np
import torch
from torch import Tensor
from torch_geometric.nn.pool import voxel_grid
from torch_geometric.nn.pool.consecutive import consecutive_cluster

from torch_pointcloud.utils.conversion import ensure_tuple, ensure_tuple_size
from torch_pointcloud.utils.data import DataKeys
from torch_pointcloud.utils.imports import _TORCH_SCATTER_GITHUB_URL, optional_import
from torch_pointcloud.utils.octree import build_octree
from torch_pointcloud.utils.ops import first_permutation, voxel_grid_fnv
from torch_pointcloud.utils.types import KeyCollection, ValueCollection
from torch_pointcloud.utils.voxelization import hard_voxelize

from . import functional as F
from .functional import RescaleMethod, ShiftMethod

# Type aliases for the literal-valued parameters used by class transforms below.

ReduceOp = Literal["min", "max", "mean", "sum"]
"""Allowed values for `Reduce.op` (per-key reduction operator)."""

VoxelMethod = Literal["fnv", "pyg"]
"""Allowed values for `Voxelize.method` (voxel-id hashing scheme)."""

VoxelReduce = Literal["mean", "min", "max", "sum", "first"]
"""Allowed values for `Voxelize.reduce` (per-key per-voxel reduction)."""

VoxelPosReduce = Literal["mean", "min", "max", "sum", "first", "grid"]
"""Allowed values for `Voxelize.pos_reduce` (per-voxel reduction for positions; `"grid"` keeps integer voxel coords)."""

if TYPE_CHECKING:
    from torch_scatter import scatter

scatter, _ = optional_import("torch_scatter", name="scatter", url=_TORCH_SCATTER_GITHUB_URL)


__all__ = [
    "Abs",
    "AlignAxis",
    "ApplyMask",
    "AxisMinOffset",
    "BBoxCenter",
    "BoxMask",
    "BuildOctree",
    "Cat",
    "Clamp",
    "OctreeFeatures",
    "Compose",
    "CopyItems",
    "CubeMask",
    "DictTransform",
    "Divide",
    "DivideKey",
    "DivisiblePad",
    "EncodeVoteNetTargets",
    "EstimateNormals",
    "FarthestPointSample",
    "GenerateVoteLabels",
    "HardVoxelize",
    "InstanceToBox",
    "KeepItems",
    "LaserMix",
    "Mix3D",
    "Normalize",
    "OneHot",
    "OnesLike",
    "PolarMix",
    "Quantize",
    "RandomColorAutoContrast",
    "RandomColorDrop",
    "RandomColorGrayScale",
    "RandomColorJitter",
    "RandomColorShift",
    "RandomDropout",
    "RandomElasticDistortion",
    "RandomFlip",
    "RandomJitter",
    "RandomRotate",
    "RandomRotateChoice",
    "RandomSample",
    "RandomSampleFaceVertices",
    "RandomScale",
    "RandomShift",
    "Reduce",
    "ReduceOp",
    "Relabel",
    "RelabelBoxes",
    "RemoveNearOrigin",
    "RenameItems",
    "Rescale",
    "RescaleMethod",
    "Scale",
    "SetValue",
    "Shift",
    "ShiftMethod",
    "ShufflePoint",
    "Slice",
    "SphereCrop",
    "SphereMask",
    "SubtractKey",
    "ToDevice",
    "ToFloat",
    "ToTensor",
    "Transform",
    "VoxelMethod",
    "VoxelPosReduce",
    "VoxelReduce",
    "Voxelize",
]


class Transform(metaclass=ABCMeta):
    """Base class for all point cloud transforms.

    A transform is a callable that takes an arbitrary data object
    and returns a transformed version of it.

    While any callable can be used as a transform, this class
    provides a common interface and some convenience features, such as:

    - a `torch_pointcloud.transforms.transforms.Transform.transform` method,
      which implements the actual transformation logic. It will be called by the
      `__call__` method to apply the transform.
    - a `torch_pointcloud.transforms.transforms.Transform.extra_repr` method,
      which returns a string that describes the transform. This will be used by the
      `__repr__` method to represent the transform as a string.

    Note:
        A transform should avoid modifying the input data in place.
        Instead, it should return a new object with the transformed data.

        If the transform is in-place, it should be clearly stated in its documentation.

    Warning:
        Transforms that accept a `generator` keep a reference to it. Under a multi-worker
        `DataLoader`, every worker receives an identical copy of that generator, so all workers
        replay the same "random" augmentations (and with `persistent_workers=False`, so does every
        epoch). Leave `generator=None` for multi-worker training: the global generator is seeded
        per worker by PyTorch (`base_seed + worker_id`), which stays random across workers and
        reproducible under `torch.manual_seed`. Reserve a stored generator for `num_workers=0`, or
        re-seed it per worker in a `worker_init_fn`:

        ```{.python notest}
        def worker_init_fn(worker_id: int) -> None:
            info = torch.utils.data.get_worker_info()
            info.dataset.transform.generator = torch.Generator().manual_seed(info.seed)
        ```

    See Also:
        `torch_pointcloud.transforms.DictTransform` for a version of this class
        that operates on dictionaries.

    Example:
        For example, to create a transform that scales the points in a point cloud,
        we can subclass the `torch_pointcloud.transforms.Transform` class
        and implement the `torch_pointcloud.transforms.Transform.transform` method
        as follows:

        ```python
        from torch import Tensor

        from torch_pointcloud.transforms import Transform

        # 1. Subclass the Transform class
        class MyScale(Transform):
            def __init__(self, factor: float = 1.0):
                self.factor = factor

            def extra_repr(self) -> str:
                return f"factor={self.factor}"

            def transform(self, tensor: Tensor) -> Tensor:
                return tensor * self.factor

        # 2. Initialize the transform
        transform = MyScale()
        # 3. Apply the transform
        tensor = torch.randn(4096, 3)
        tensor = transform(tensor)
        ```
    """

    _repr_indent = 2

    @abstractmethod
    def transform(self, *args: Any, **kwargs: Any) -> Any:
        """Apply the transform to the input data.

        This method should be implemented by all subclasses, and do not
        have any constraints on the input data.
        """

    def extra_repr(self) -> str:
        """Return a string that describes the transform.

        This will be used by the `__repr__` method to represent the transform as a string.
        """
        return ", ".join([f"{k}={v!r}" for k, v in self.__dict__.items() if not k.startswith("_")])

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.transform(*args, **kwargs)

    def __repr__(self) -> str:
        indent = " " * self._repr_indent
        main_str = f"{self.__class__.__name__}("
        extra_lines = self.extra_repr().splitlines()

        if extra_lines:
            if len(extra_lines) == 1:
                main_str += extra_lines[0]
            else:
                main_str += f"\n{indent}" + f"\n{indent}".join(extra_lines) + "\n"

        main_str += ")"
        return main_str


class Compose(Transform):
    """Compose multiple transforms into a single transform.

    This class allows for chaining multiple transforms together.

    Note:
        The order of the transforms is important, as each transform will be applied
        in the order they are added to the `Compose` object.


    Example:
        For example, to chain a random sample and a normalization transform,
        we can do the following:

        ```python
        from torch import Tensor

        from torch_pointcloud.transforms import Compose, RandomSample, Rescale

        # 1. Initialize the transforms
        transform = Compose([
            RandomSample(keys="pos", num_samples=1024),
            Rescale(keys="pos"),
        ])

        # 2. Apply the transform
        data = {"pos": torch.randn(4096, 3)}
        data = transform(data)
        ```
    """

    def __init__(self, transforms: Sequence[Transform]):
        self.transforms = transforms

    def transform(self, data: Any) -> Any:
        """Apply the transforms to the input data.

        This method will apply each transform in the order they were added to the
        `torch_pointcloud.transforms.Compose` object.
        """
        for transform in self.transforms:
            if isinstance(data, (list, tuple)):
                data = [transform(d) for d in data]
            else:
                data = transform(data)
        return data

    def extra_repr(self) -> str:
        return ",\n".join([repr(transform) for transform in self.transforms])


class DictTransform(Transform, metaclass=ABCMeta):
    """Base class for dictionary transforms.

    This class is used to define transforms that operate on a dictionary of data,
    and implements utility methods for key iteration and error handling.

    Note:
        `allow_missing_keys` controls the iteration over `self.keys` performed by
        `iter_keys`. Auxiliary keys read by individual transforms (e.g. `mask_key`
        in `ApplyMask`, `pos_key` in `RemoveNearOrigin` / `FarthestPointSample`,
        `face_key` in `RandomSampleFaceVertices`) document their own missing-key
        behavior in their respective docstrings.

    Args:
        keys: The keys to apply the transform to.
        allow_missing_keys: If `True`, the transform will not raise an error if
            keys listed in `self.keys` are missing from the input dict.

    """

    def __init__(self, keys: Optional[KeyCollection] = None, allow_missing_keys: bool = False) -> None:
        self.keys = ensure_tuple(keys, none_as_empty=True)
        self.allow_missing_keys = allow_missing_keys

    @abstractmethod
    def transform(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply the transform to the dictionary data.

        Args:
            data: The dictionary data to apply the transform to.

        Returns:
            The transformed dictionary data.
        """

    def iter_keys(
        self,
        data: Dict[str, Any],
        *extra_iterables: Iterable[Any],
        extra_msg: str = "",
    ) -> Generator[Any, None, None]:
        """Iterate over `self.keys` present in the data, honoring `allow_missing_keys`.

        Args:
            data: The dictionary data the transform is applied to.
            *extra_iterables: Per-key values (e.g. one output key per input key) zipped with `self.keys`.
            extra_msg: Message appended to the `KeyError` raised on a missing key.

        Returns:
            A generator yielding each present key, or a tuple of the key and its values from
            `extra_iterables` when any is given.
        """
        # inspired by: https://github.com/Project-MONAI/MONAI/blob/main/monai/transforms/transform.py#L456
        # if no extra iterables given, create a dummy list of Nones
        ex_iters: Iterable[Any] = extra_iterables or [[None] * len(self.keys)]
        ex_iters = [ensure_tuple(ex_iter) for ex_iter in ex_iters]

        for key, *_ex_iters in zip(self.keys, *ex_iters):
            if key in data:
                # all normal, yield (what we yield depends on whether extra iterables were given)
                yield (key,) + tuple(_ex_iters) if extra_iterables else key
            elif not self.allow_missing_keys:
                raise KeyError(f"Key {key!r} was missing in the data and `allow_missing_keys==False`. {extra_msg}")

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        return super().__call__(data)


class RandomSample(DictTransform):
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
        generator: The generator for the random number generator.
        allow_missing_keys: If `True`, the transform will not raise an error if the keys are not present in the data.

    Raises:
        ValueError: If the first sampled tensor is empty and `num_samples > 0`.
    """

    def __init__(
        self,
        keys: KeyCollection,
        num_samples: int,
        replace: bool = False,
        generator: Optional[torch.Generator] = None,
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.num_samples = num_samples
        self.replace = replace
        self.generator = generator

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
            generator=self.generator,
        )
        d[first_key] = sampled_tensor
        for key in iterator:
            d[key] = d[key][indices]
        return d


class DivisiblePad(DictTransform):
    r"""Pad per-point tensors so each batch is divisible by `num_samples`.

    Thin dict wrapper around `divisible_pad`; see its docstring for the full
    behavior of each `pad_fill` strategy (`"cycle"`, `"replicate"`, `"random"`).
    The tensor at `ref_key` defines the packed count $n$ and the device. If a
    batch index tensor lives at `batch_key`, padding is done per-batch; otherwise
    a single zero batch is synthesized. Every tensor in the dict whose first dim
    equals $n$ (positions, features, labels, ...) is re-indexed by the same
    gather map, so per-point correspondence is preserved.

    When `dst_inverse_key` is set, the transform also records a source-to-padded
    index map under that dict key: a 1-D long tensor of length $n$ with values
    in $[0, n_\text{padded})$ giving the canonical padded row for each source
    row. If the key already holds a prior inverse map (from an earlier
    invertible transform), the new map composes with it via gather, so the
    stored tensor always maps from the outermost source space to the current
    predictor space. Consumers such as `SlidingWindowInferer` read this key
    once and gather predictions back to the source rows.

    === "Object"

        ![DivisiblePad on an object](../../assets/transforms/divisible_pad.png)

    === "Scene"

        ![DivisiblePad on a room](../../assets/transforms/divisible_pad_scene.png)

    Args:
        num_samples: Target chunk size $k$ for divisibility.
        pad_fill: Fill strategy passed through to `divisible_pad`.
        ref_key: Key whose tensor defines $n$ and the device.
        batch_key: Key for an optional batch index tensor. When present in the
            data, padding runs per-batch; otherwise a single zero batch is
            synthesized for the whole scene.
        generator: Optional `torch.Generator`. Only consumed when
            `pad_fill="random"`. See `Transform` for the multi-worker caveat.
        dst_inverse_key: When set, store the source-to-padded index map under
            this key (auto-composes with any existing value at the same key).
            Leave `None` for training pipelines that never need the inverse.
        allow_missing_keys: If `True`, return the data unchanged when `ref_key`
            is missing instead of raising.

    Example:
        ```python
        from torch_pointcloud.transforms import DivisiblePad

        # Pad a 5000-point block to 8192 (= 2 * 4096) before sliding-window
        # sub-chunking. Random fill duplicates points uniformly at random.
        transform = DivisiblePad(num_samples=4096, pad_fill="random")
        ```
    """

    def __init__(
        self,
        num_samples: int,
        pad_fill: "F.PadFill" = "cycle",
        ref_key: str = DataKeys.POS,
        batch_key: str = DataKeys.BATCH,
        generator: Optional[torch.Generator] = None,
        dst_inverse_key: Optional[str] = None,
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys=ref_key, allow_missing_keys=allow_missing_keys)
        self.num_samples = num_samples
        self.pad_fill = pad_fill
        self.ref_key = ref_key
        self.batch_key = batch_key
        self.generator = generator
        self.dst_inverse_key = dst_inverse_key

    def transform(self, data: Dict[str, Any]) -> Dict[str, Any]:
        d = dict(data)
        if self.ref_key not in d:
            if self.allow_missing_keys:
                return d
            raise KeyError(f"`DivisiblePad` requires {self.ref_key!r} in data.")
        ref = d[self.ref_key]
        if not torch.is_tensor(ref):
            raise TypeError(f"Expected tensor at {self.ref_key!r}, got {type(ref).__name__}.")
        n = int(ref.size(0))
        if n == 0:
            return d
        if self.batch_key in d and torch.is_tensor(d[self.batch_key]):
            batch = d[self.batch_key]
        else:
            batch = torch.zeros(n, dtype=torch.long, device=ref.device)
        indices, inverse_indices, padded_batch = F.divisible_pad(
            batch,
            k=self.num_samples,
            mode="all",
            pad_fill=self.pad_fill,
            return_inverse=True,
            generator=self.generator,
        )
        prior = d.get(self.dst_inverse_key) if self.dst_inverse_key is not None else None
        for key, value in d.items():
            if key == self.dst_inverse_key:
                continue
            if torch.is_tensor(value) and value.ndim > 0 and value.size(0) == n:
                d[key] = value[indices]
        d[self.batch_key] = padded_batch
        if self.dst_inverse_key is not None:
            d[self.dst_inverse_key] = inverse_indices if prior is None else inverse_indices[prior]
        return d


class RandomSampleFaceVertices(DictTransform):
    """Randomly sample a fixed number of vertices from a 3D mesh stored in a dictionary.

    ![RandomSampleFaceVertices before / after](../../assets/transforms/random_sample_face_vertices.png)

    See Also:
        `torch_pointcloud.transforms.functional.random_sample_face_vertices`

    Args:
        keys: The keys holding vertex positions.
        face_key: The keys holding the face indices.
        normal_key: The key to store the computed normals in.
        num_samples: The number of vertices to sample.
        generator: The generator for the random number generator.
        allow_missing_keys: If `True`, the transform will not raise an error if the keys are not present in the data.
    """

    def __init__(
        self,
        *,
        keys: KeyCollection,
        face_key: KeyCollection,
        normal_key: Optional[KeyCollection] = "normal",
        num_samples: int,
        generator: Optional[torch.Generator] = None,
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.face_key = ensure_tuple_size(face_key, len(self.keys))
        self.num_samples = num_samples
        self.normal_key = ensure_tuple_size(normal_key, len(self.keys))
        self.generator = generator

    def transform(self, data: Dict[str, Any]) -> Dict[str, Any]:
        data = dict(data)
        for key, face_key, normal_key in self.iter_keys(data, self.face_key, self.normal_key):
            pos, normal = F.random_sample_face_vertices(
                data[key],
                data[face_key],
                self.num_samples,
                generator=self.generator,
                return_normals=True,
            )
            data[key] = pos
            if normal_key is not None:
                data[normal_key] = normal
        return data


class EstimateNormals(DictTransform):
    r"""Estimate per-point surface normals from coordinates via local PCA.

    Computes unit normals (see `torch_pointcloud.transforms.functional.estimate_normals`) for clouds that
    ship without them (e.g. S3DIS). Each normal is the least-variance direction of a point's $k$ nearest
    neighbours. With `orient_to_centroid`, normals are flipped to face the cloud centroid.

    See Also:
        `torch_pointcloud.transforms.functional.estimate_normals`

    === "Object"

        ![EstimateNormals on an object](../../assets/transforms/estimate_normals.png)

    === "Scene"

        ![EstimateNormals on a room](../../assets/transforms/estimate_normals_scene.png)

    Args:
        keys: Coordinate keys to estimate normals from.
        normal_key: Keys under which to store the normals (one per coordinate key). Defaults to `normal`.
        k: Number of nearest neighbours (the point itself included) per local PCA.
        orient_to_centroid: If `True`, flip each normal to point towards its cloud's centroid (approximates
            the inward-facing normals of meshes scanned from inside a room).
        batch_key: Optional key holding a per-point batch index so neighbours stay within a cloud.
        allow_missing_keys: If `True`, silently skip absent keys.
    """

    def __init__(
        self,
        keys: KeyCollection,
        normal_key: KeyCollection = "normal",
        k: int = 16,
        orient_to_centroid: bool = False,
        batch_key: Optional[str] = None,
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.normal_key = ensure_tuple_size(normal_key, len(self.keys))
        self.k = k
        self.orient_to_centroid = orient_to_centroid
        self.batch_key = batch_key

    def transform(self, data: Dict[str, Any]) -> Dict[str, Any]:
        d = dict(data)
        batch = d.get(self.batch_key) if self.batch_key is not None else None
        for key, normal_key in self.iter_keys(d, self.normal_key):
            d[normal_key] = F.estimate_normals(
                d[key], k=self.k, batch=batch, orient_to_centroid=self.orient_to_centroid
            )
        return d


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
        allow_missing_keys: If `True`, the transform will not raise an error if the keys are not present in the data.
    """

    def __init__(
        self,
        pos_key: str,
        keys: Optional[KeyCollection] = None,
        num_samples: Optional[int] = None,
        ratio: Optional[float] = None,
        random_start: bool = False,
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
        return d


class Rescale(DictTransform):
    r"""Center a point set and rescale it to a unit extent.

    Bundles a centering step and a divide-by-extent step that depend on the
    same statistics. Three methods, each pairing a center and a denominator:

    | `method`      | Center on        | Divide by                        |
    | ------------- | ---------------- | -------------------------------- |
    | `"centroid"`  | centroid (mean)  | max Euclidean distance to center |
    | `"bbox"`      | bbox midpoint    | half of the longest axis extent  |
    | `"linear"`    | centroid (mean)  | longest axis extent              |

    Empty inputs (`N=0`) are returned unchanged.

    === "method=centroid"

        ![Rescale method=centroid on an object](../../assets/transforms/rescale_centroid.png)

    === "method=bbox"

        ![Rescale method=bbox on an object](../../assets/transforms/rescale_bbox.png)

    === "method=linear"

        ![Rescale method=linear on an object](../../assets/transforms/rescale_linear.png)

    See Also:
        `torch_pointcloud.transforms.functional.rescale`

    Args:
        keys: The keys to rescale.
        eps: Small constant added to the denominator for numerical stability.
        method: `"centroid"`, `"bbox"`, or `"linear"`.
        allow_missing_keys: If `True`, the transform will not raise an error if
            the keys are not present in the data.
    """

    def __init__(
        self,
        keys: KeyCollection,
        eps: float = 1e-6,
        method: RescaleMethod = "centroid",
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.eps = eps
        self.method = method

    def transform(self, data: Dict[str, Any]) -> Dict[str, Any]:
        d = dict(data)
        for key in self.iter_keys(d):
            d[key] = F.rescale(d[key], eps=self.eps, method=self.method)
        return d


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
        allow_missing_keys: If `True`, the transform will not raise an error if the keys are not present in the data.
    """

    def __init__(
        self,
        pos_key: str,
        keys: Optional[KeyCollection] = None,
        radius: float = 1e-3,
        allow_missing_keys: bool = False,
    ) -> None:
        all_keys = ensure_tuple(keys, none_as_empty=True)
        if pos_key not in all_keys:
            all_keys = (pos_key,) + all_keys
        super().__init__(all_keys, allow_missing_keys)
        self.pos_key = pos_key
        self.radius = radius

    def transform(self, data: Dict[str, Any]) -> Dict[str, Any]:
        d = dict(data)
        if self.pos_key not in d:
            if self.allow_missing_keys:
                return d
            raise KeyError(f"`RemoveNearOrigin` requires {self.pos_key!r} in data.")
        _, mask = F.remove_near_origin(d[self.pos_key], radius=self.radius, return_mask=True)
        for key in self.iter_keys(d):
            d[key] = d[key][mask]
        return d


class Abs(DictTransform):
    """Make dictionary tensor entries absolute.

    See Also:
        `torch_pointcloud.transforms.functional.abs`

    === "Object"

        ![Abs on an object](../../assets/transforms/abs.png)

    === "Scene"

        ![Abs on a room](../../assets/transforms/abs_scene.png)

    Args:
        keys: The keys to make absolute.
        inplace: Whether to perform the operation in place.
        allow_missing_keys: If `True`, the transform will not raise an error if the keys are not present in the data.
    """

    def __init__(self, keys: KeyCollection, inplace: bool = False, allow_missing_keys: bool = False) -> None:
        super().__init__(keys, allow_missing_keys)
        self.inplace = inplace

    def transform(self, data: Dict[str, Any]) -> Dict[str, Any]:
        d = dict(data)
        for key in self.iter_keys(d):
            d[key] = F.abs(d[key], inplace=self.inplace)
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
        allow_missing_keys: If `True`, the transform will not raise an error if the keys are not present in the data.
    """

    def __init__(
        self,
        keys: KeyCollection,
        mask_key: str,
        dst_keys: Optional[KeyCollection] = None,
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.mask_key = mask_key
        self.dst_keys = ensure_tuple_size(dst_keys or self.keys, size=len(self.keys))

    def transform(self, data: Dict[str, Any]) -> Dict[str, Any]:
        d = dict(data)
        if self.mask_key not in data:
            if self.allow_missing_keys:
                return d
            raise KeyError(f"Mask key {self.mask_key!r} not found in data.")
        mask = d[self.mask_key]
        for key, dst_key in self.iter_keys(d, self.dst_keys):
            d[dst_key] = F.apply_mask(d[key], mask)
        return d


class SetValue(DictTransform):
    """Set values for keys in the dictionary, creating or overwriting them.

    Unlike most `DictTransform` subclasses, `SetValue` does not read existing
    values, so `allow_missing_keys` has no meaning and is not accepted.

    ![SetValue diagram](../../assets/transforms/set_value.png)

    Args:
        keys: The keys to set.
        values: The values to set. Either a single value broadcast to every key,
            or a sequence of values the same length as `keys`.
    """

    def __init__(self, keys: KeyCollection, values: Any) -> None:
        super().__init__(keys, allow_missing_keys=False)
        self.values = ensure_tuple_size(values, len(self.keys))

    def transform(self, data: Dict[str, Any]) -> Dict[str, Any]:
        data = dict(data)
        for key, value in zip(self.keys, self.values):
            data[key] = value
        return data


class Scale(DictTransform):
    """Multiply dictionary tensor entries by a scale factor.

    === "Object"

        ![Scale on an object](../../assets/transforms/scale.png)

    === "Scene"

        ![Scale on a room](../../assets/transforms/scale_scene.png)

    Args:
        keys: The keys to scale.
        scale: The scale factor(s).
        allow_missing_keys: If `True`, the transform will not raise an error if the keys are not present in the data.
    """

    def __init__(
        self,
        keys: KeyCollection,
        scale: float | Sequence[float],
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.scale = ensure_tuple_size(scale, len(self.keys))

    def transform(self, data: Dict[str, Any]) -> Dict[str, Any]:
        data = dict(data)
        for key, scale in self.iter_keys(data, self.scale):
            data[key] = data[key] * scale
        return data


class Divide(DictTransform):
    """Divide dictionary tensor entries by a divisor.

    === "Object"

        ![Divide on an object](../../assets/transforms/divide.png)

    === "Scene"

        ![Divide on a room](../../assets/transforms/divide_scene.png)

    Args:
        keys: The keys to divide.
        divisor: The divisor(s).
        allow_missing_keys: If `True`, the transform will not raise an error if the keys are not present in the data.
    """

    def __init__(
        self,
        keys: KeyCollection,
        divisor: float | Sequence[float],
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.divisor = ensure_tuple_size(divisor, len(self.keys))

    def transform(self, data: Dict[str, Any]) -> Dict[str, Any]:
        data = dict(data)
        for key, divisor in self.iter_keys(data, self.divisor):
            data[key] = data[key] / divisor
        return data


class ToFloat(DictTransform):
    """Cast dictionary tensor entries to float32.

    Useful when tensors are stored in integer formats (e.g. `uint8` for
    colors) and need to be promoted to floating point before arithmetic
    transforms like `Divide` or `Normalize`.

    ![ToFloat diagram](../../assets/transforms/to_float.png)

    Args:
        keys: The keys to cast.
        allow_missing_keys: If `True`, missing keys are silently ignored.
    """

    def __init__(
        self,
        keys: KeyCollection,
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)

    def transform(self, data: Dict[str, Any]) -> Dict[str, Any]:
        data = dict(data)
        for key in self.iter_keys(data):
            data[key] = data[key].float()
        return data


class Normalize(DictTransform):
    r"""Normalize dictionary tensor entries: $x' = (x - \mu) / \max(\sigma, \epsilon)$.

    ![Normalize before / after](../../assets/transforms/normalize.png)

    Args:
        keys: The keys to standardize.
        mean: Per-channel mean(s).  Broadcast against the last dimension of
            each tensor.
        std: Per-channel standard deviation(s).
        eps: Lower bound on $\sigma$ to prevent division by zero. Defaults to $10^{-7}$.
        allow_missing_keys: If `True`, missing keys are silently ignored.
    """

    def __init__(
        self,
        keys: KeyCollection,
        mean: Sequence[float],
        std: Sequence[float],
        eps: float = 1e-7,
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.mean = torch.tensor(mean, dtype=torch.float32)
        self.std = torch.tensor(std, dtype=torch.float32)
        self.eps = eps

    def transform(self, data: Dict[str, Any]) -> Dict[str, Any]:
        data = dict(data)
        for key in self.iter_keys(data):
            data[key] = F.normalize(data[key], self.mean, self.std, eps=self.eps)
        return data


class Shift(DictTransform):
    r"""Shift dictionary tensor entries by subtracting a computed offset.

    Each key is offset independently. The offset is determined by `method`:

    | Method       | Offset                                           |
    | ------------ | ------------------------------------------------ |
    | `"bbox"`     | Midrange: `(min + max) / 2`                      |
    | `"centroid"` | Mean across the reduced dimension                |
    | `"min"`      | Per-axis minimum (shifts to the positive octant) |

    On empty inputs (size $0$ along `dim`) the tensor is returned unchanged.

    === "method=centroid"

        ![Shift method=centroid on an object](../../assets/transforms/shift_centroid.png)

    === "method=bbox"

        ![Shift method=bbox on an object](../../assets/transforms/shift_bbox.png)

    === "method=min"

        ![Shift method=min on an object](../../assets/transforms/shift_min.png)

    === "axes"

        ![Shift restricted to a subset of axes, on an object](../../assets/transforms/shift_axes.png)

    Args:
        keys: The keys to shift.
        method: `"bbox"` (midrange), `"centroid"` (mean), or `"min"` (shift to origin).
        dim: The dimension to reduce over.
        axes: Which axes (last-dim indices) to shift. `None` (default) shifts every
            axis; pass e.g. `axes=[0, 1]` to recenter only XY. Axes outside this list are
            left unchanged - this is the composable knob for mixed-method shifts.
        dst_keys: The keys to store the shifted data in.
        allow_missing_keys: If `True`, skip missing keys silently.

    Example:
        XY shifted by the bbox midpoint, Z shifted by its minimum (equivalent
        to the old `CenterShift(apply_z=True)`):

        ```python
        from torch_pointcloud.transforms import Compose, Shift

        center_shift = Compose([
            Shift(keys="pos", method="bbox", axes=[0, 1]),  # XY: bbox midrange
            Shift(keys="pos", method="min",  axes=[2]),     # Z:  min
        ])
        ```

        Without the Z step (equivalent to the old `CenterShift(apply_z=False)`),
        a single `Shift` suffices:

        ```python
        Shift(keys="pos", method="bbox", axes=[0, 1])
        ```
    """

    def __init__(
        self,
        keys: KeyCollection,
        method: ValueCollection[ShiftMethod],
        dim: int = 0,
        axes: Optional[Sequence[int]] = None,
        dst_keys: Optional[KeyCollection] = None,
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.dim = dim
        self.axes = tuple(axes) if axes is not None else None
        self.dst_keys = ensure_tuple_size(dst_keys or self.keys, len(self.keys))
        self.method = ensure_tuple_size(method, len(self.keys))
        valid = get_args(ShiftMethod)
        if not set(self.method).issubset(valid):
            raise ValueError(f"Invalid method: {method!r}. Expected one of {valid}.")

    def transform(self, data: Dict[str, Any]) -> Dict[str, Any]:
        data = dict(data)
        for key, dst_key, method in self.iter_keys(data, self.dst_keys, self.method):
            x = data[key]
            if not torch.is_tensor(x):
                raise TypeError(f"Expected a tensor, got {type(x).__name__!r}.")
            data[dst_key] = F.shift(x, method=method, dim=self.dim, axes=self.axes)
        return data


class AlignAxis(DictTransform):
    """Shift dictionary tensor entries so that the minimum along a chosen axis is zero.

    Empty inputs (`N=0`) are returned unchanged.

    === "Object"

        ![AlignAxis on an object](../../assets/transforms/align_axis.png)

    === "Scene"

        ![AlignAxis on a room](../../assets/transforms/align_axis_scene.png)

    Args:
        keys: The keys to align.
        dim: The coordinate axis to align.
        inplace: Whether to modify the tensor in place. Non-contiguous inputs are
            materialized to contiguous via `.contiguous()` before the in-place op,
            so the caller's original tensor may not be mutated in that case.
        allow_missing_keys: If `True`, the transform will not raise an error if the keys are not present in the data.
    """

    def __init__(
        self,
        keys: KeyCollection,
        dim: int = -1,
        inplace: bool = False,
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.dim = dim
        self.inplace = inplace

    def transform(self, data: dict) -> dict:
        data = dict(data)

        for key in self.iter_keys(data):
            x = data[key]
            if not torch.is_tensor(x):
                raise TypeError(f"Expected a tensor, got {type(x).__name__!r}.")

            if x.shape[0] == 0:
                data[key] = x
                continue

            if self.inplace:
                x = x.contiguous()
            else:
                x = x.clone()

            x[:, self.dim] -= x[:, self.dim].min()
            data[key] = x

        return data


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


class ToDevice(DictTransform):
    """Convert dictionary tensor entries to the given device.

    ![ToDevice diagram](../../assets/transforms/to_device.png)

    Args:
        keys: The keys to convert the tensors to the given device.
        device: The device to convert the tensors to.
        non_blocking: If `True`, the transfer will be done asynchronously.
        copy: If `True`, the tensor will be copied to the new device.
        memory_format: The memory format to use for the tensor.
        dst_keys: The keys to store the converted tensors in.
        allow_missing_keys: If `True`, the transform will not raise an error if the keys are not present in the data.
    """

    def __init__(
        self,
        keys: KeyCollection,
        device: ValueCollection[str | torch.device],
        non_blocking: ValueCollection[bool] = False,
        copy: ValueCollection[bool] = True,
        memory_format: ValueCollection[torch.memory_format | None] = None,
        dst_keys: Optional[KeyCollection] = None,
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.dst_keys = ensure_tuple_size(dst_keys or self.keys, len(self.keys))
        self.device = ensure_tuple_size(device, len(self.keys))
        self.non_blocking = ensure_tuple_size(non_blocking, len(self.keys))
        self.copy = ensure_tuple_size(copy, len(self.keys))
        self.memory_format = ensure_tuple_size(memory_format, len(self.keys))

    def transform(self, data: Dict[str, Any]) -> Dict[str, Any]:
        data = dict(data)
        for key, dst_key, device, non_blocking, copy, memory_format in self.iter_keys(
            data,
            self.dst_keys,
            self.device,
            self.non_blocking,
            self.copy,
            self.memory_format,
        ):
            x = data[key]
            if not torch.is_tensor(x):
                raise TypeError(f"Expected a tensor, got {type(x).__name__!r}.")

            data[dst_key] = x.to(
                device,
                non_blocking=non_blocking,
                copy=copy,
                memory_format=memory_format,
            )

        return data


class BuildOctree(DictTransform):
    """Build an octree from positions stored in a dictionary.

    === "Object"

        ![BuildOctree on an object](../../assets/transforms/build_octree.png)

    === "Scene"

        ![BuildOctree on a room](../../assets/transforms/build_octree_scene.png)

    Args:
        pos_key: Key holding the point positions.
        octree_key: Key under which the octree is stored.
        depth: Octree depth.
        full_depth: Full depth of the octree.
        batch_size: Batch size.
        normal_key: Key holding surface normals.
        feature_key: Key holding point features.
        label_key: Key holding per-point labels.
        batch_key: Key holding batch indices.
        points_key: Key under which the octree points are stored.
    """

    def __init__(
        self,
        *,
        pos_key: str,
        octree_key: str,
        depth: int,
        full_depth: int = 2,
        batch_size: int = 1,
        normal_key: str | None = None,
        feature_key: str | None = None,
        label_key: str | None = None,
        batch_key: str | None = None,
        points_key: str | None = None,
    ) -> None:
        super().__init__([], False)
        if points_key is not None and points_key == octree_key:
            raise ValueError(f"`points_key` and `octree_key` must be different, got {points_key!r}.")

        self.pos_key = pos_key
        self.depth = depth
        self.octree_key = octree_key
        self.full_depth = full_depth
        self.batch_size = batch_size
        self.normal_key = normal_key
        self.feature_key = feature_key
        self.label_key = label_key
        self.batch_key = batch_key
        self.points_key = points_key

    def transform(self, data: Dict[str, Any]) -> Dict[str, Any]:
        data = dict(data)

        pos = data[self.pos_key]
        normal = data[self.normal_key] if self.normal_key is not None else None
        features = data[self.feature_key] if self.feature_key is not None else None
        batch_id = data[self.batch_key] if self.batch_key is not None else None
        labels = data[self.label_key] if self.label_key is not None else None

        octree, points = build_octree(
            pos=pos,
            normal=normal,
            features=features,
            batch=batch_id,
            labels=labels,
            depth=self.depth,
            full_depth=self.full_depth,
            batch_size=self.batch_size,
            return_points=True,
        )

        data[self.octree_key] = octree
        if self.points_key is not None:
            data[self.points_key] = points

        return data


class HardVoxelize(DictTransform):
    r"""Hard-voxelize a single scene into the per-voxel point stack consumed by voxel detectors.

    Moves the `transform_points_to_voxels` step of voxel detectors (PointPillars, SECOND) out of the
    model and into the data pipeline, mirroring how `BuildOctree` produces an octree for OctFormer.
    The model then receives already-voxelized input and focuses on the network math.

    Reads `pos_key` (and optionally `feat_key`), runs
    `hard_voxelize` on the single sample (the
    batch index is all zeros), and adds three keys while keeping `pos` / `x`:

    - `voxel_key`: the per-voxel point stack.
    - `pos_voxel_key`: integer voxel grid indices $(z, y, x)$ (the single-sample batch column is dropped;
      the per-voxel scene index is synthesized at collation).
    - `num_points_key`: the per-voxel point counts.

    === "Object"

        ![HardVoxelize on an object](../../assets/transforms/hard_voxelize.png)

    === "Scene"

        ![HardVoxelize on a room](../../assets/transforms/hard_voxelize_scene.png)

    Args:
        pos_key: Key holding the point positions $(N, 3)$.
        voxel_size: Voxel size $(v_x, v_y, v_z)$.
        point_cloud_range: Range $(x_\min, y_\min, z_\min, x_\max, y_\max, z_\max)$.
        max_num_points: Maximum number of points kept per voxel.
        max_num_voxels: Maximum number of voxels kept per scene.
        feat_key: Optional key holding extra point features $(N, C)$ concatenated after $xyz$.
        voxel_key: Output key for the per-voxel point stack.
        pos_voxel_key: Output key for the integer voxel grid indices.
        num_points_key: Output key for the per-voxel point counts.
        allow_missing_keys: Unused (`pos_key` is always required); kept for interface parity.

    Shape:
        - `voxel_key`: $(V, \text{max\_num\_points}, 3 + C)$.
        - `pos_voxel_key`: $(V, 3)$ with columns $(z, y, x)$.
        - `num_points_key`: $(V,)$.

    Example:
        ```python
        import torch
        import torch_pointcloud.transforms as T

        data = {"pos": torch.rand(1000, 3) * 50.0, "x": torch.rand(1000, 1)}
        transform = T.HardVoxelize(
            pos_key="pos",
            feat_key="x",
            voxel_size=(0.16, 0.16, 4.0),
            point_cloud_range=(0.0, -39.68, -3.0, 69.12, 39.68, 1.0),
            max_num_points=32,
            max_num_voxels=40000,
        )
        data = transform(data)
        print(data["voxel"].shape, data["pos_voxel"].shape, data["voxel_num_points"].shape)
        ```
    """

    def __init__(
        self,
        pos_key: str,
        voxel_size: Sequence[float],
        point_cloud_range: Sequence[float],
        max_num_points: int,
        max_num_voxels: int,
        feat_key: Optional[str] = None,
        voxel_key: str = DataKeys.VOXEL,
        pos_voxel_key: str = DataKeys.POS_VOXEL,
        num_points_key: str = DataKeys.VOXEL_NUM_POINTS,
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys=pos_key, allow_missing_keys=allow_missing_keys)
        self.pos_key = pos_key
        self.voxel_size = voxel_size
        self.point_cloud_range = point_cloud_range
        self.max_num_points = max_num_points
        self.max_num_voxels = max_num_voxels
        self.feat_key = feat_key
        self.voxel_key = voxel_key
        self.pos_voxel_key = pos_voxel_key
        self.num_points_key = num_points_key

    def transform(self, data: Dict[str, Any]) -> Dict[str, Any]:
        d = dict(data)
        pos = d[self.pos_key]
        feat = d.get(self.feat_key) if self.feat_key is not None else None
        points = pos if feat is None else torch.cat([pos, feat], dim=1)
        batch = pos.new_zeros(points.shape[0], dtype=torch.long)
        voxels, voxel_indices, num_points = hard_voxelize(
            points,
            batch,
            self.voxel_size,
            self.point_cloud_range,
            self.max_num_points,
            self.max_num_voxels,
        )
        d[self.voxel_key] = voxels
        d[self.pos_voxel_key] = voxel_indices[:, 1:]
        d[self.num_points_key] = num_points
        return d


class OctreeFeatures(DictTransform):
    """Extract per-node features from an octree via `octree.get_input_feature`.

    === "Object"

        ![OctreeFeatures on an object](../../assets/transforms/octree_features.png)

    === "Scene"

        ![OctreeFeatures on a room](../../assets/transforms/octree_features_scene.png)

    Args:
        keys: Keys holding `Octree` instances to extract features from.
        features_type: Feature spec passed to `octree.get_input_feature` (e.g.
            `"ND"` for normals + depth, `"NDFP"` for normals + depth + features
            + position).
        nempty: If `True`, return features only for non-empty nodes; otherwise
            include empty-node padding.
        dst_keys: Where to store the extracted feature tensors. Defaults to `keys`.
        allow_missing_keys: If `True`, silently skip absent keys.
    """

    def __init__(
        self,
        keys: KeyCollection,
        features_type: str,
        nempty: bool = False,
        dst_keys: Optional[KeyCollection] = None,
        allow_missing_keys: bool = False,
    ):
        super().__init__(keys, allow_missing_keys)
        self.dst_keys = ensure_tuple_size(dst_keys or self.keys, len(self.keys))
        self.features_type = features_type
        self.nempty = nempty

    def transform(self, data: Dict[str, Any]) -> Dict[str, Any]:
        data = dict(data)
        for key, dst_key in self.iter_keys(data, self.dst_keys):
            octree = data[key]
            data[dst_key] = octree.get_input_feature(self.features_type, self.nempty)
        return data


class Relabel(DictTransform):
    """Remap integer labels in dictionary entries via a lookup table.

    `labels` can be either:

    - a sequence of source values (1:1) - each value at index `i` is mapped to `i`;
    - a `dict[int, int]` (general source → target) - supports N-to-1 merges
      (e.g. SemanticKITTI's `moving-car` and `car` both → 0).

    Source values not listed in `labels` are set to `default`.

    ![Relabel before / after](../../assets/transforms/relabel.png)

    Args:
        keys: Keys holding label tensors to remap.
        labels: Source-value listing (1:1) or explicit `{source: target}` dict (N:1).
        default: Value assigned to source values not listed in `labels`.
        allow_missing_keys: If `True`, skip missing keys instead of raising.

    Example:
        ```python
        # 1:1 - keep raw NYU40 ids 1, 2, 3, 4, 5 and remap them to 0..4
        T.Relabel(keys="segment", labels=[1, 2, 3, 4, 5])

        # N:1 - SemanticKITTI 19-class benchmark (merges moving-* into static)
        T.Relabel(
            keys="segment",
            labels={
                10: 0, 252: 0,    # car        (+ moving-car)
                11: 1,             # bicycle
                15: 2,             # motorcycle
                18: 3, 258: 3,    # truck      (+ moving-truck)
                # ...
            },
            default=255,
        )
        ```
    """

    def __init__(
        self,
        keys: KeyCollection,
        labels: Sequence[int] | Dict[int, int],
        default: int = 0,
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.default = default

        if isinstance(labels, dict):
            self.labels: Dict[int, int] = {int(k): int(v) for k, v in labels.items()}
        else:
            self.labels = {int(value): idx for idx, value in enumerate(labels)}

        if not self.labels:
            raise ValueError("Relabel requires at least one source value in `labels`.")

    def transform(self, data: dict) -> dict:
        data = dict(data)

        for key in self.iter_keys(data):
            tensor = data[key]
            if not isinstance(tensor, torch.Tensor):
                raise TypeError(f"Expected torch.Tensor for key {key!r}, got {type(tensor).__name__}")
            data[key] = F.relabel(tensor, self.labels, default=self.default)

        return data


class RelabelBoxes(DictTransform):
    r"""Map raw box labels to a detection class set and flag don't-care boxes for the AP metric.

    A detection dataset (e.g. `KITTI`) returns the **raw** annotated boxes: every labelled class plus
    per-box attributes such as occlusion / truncation. This transform turns those into the inputs the
    3D AP metric expects, the way `Relabel` turns raw segmentation ids into a benchmark label set:

    - boxes whose raw label is a key of `mapping` are kept as ground truth, relabelled to `mapping[raw]`;
    - boxes whose raw label is a key of `ignore_mapping` (neighbouring classes, e.g. KITTI `Van` for
      `Car`) are kept as **ignore regions** (`ignore_mask = True`), labelled `ignore_mapping[raw]`: the
      evaluated class they excuse. They suppress false positives of that class but are not scored;
    - a kept foreground box that falls outside any range in `ignore_fields` (e.g. KITTI's moderate rule:
      occlusion $\le 1$, truncation $\le 0.3$, 2D height $\ge 25$ px) is downgraded to an ignore region
      attributed to its mapped class;
    - every other box is dropped.

    All keys in `keys` (the box tensor and every per-box attribute, including those named in
    `ignore_fields`) are filtered together by the keep mask so they stay row-aligned. The output adds the
    boolean `ignore_mask_key` consumed by `average_precision3d` / `mean_average_precision3d`, which
    excuse an unmatched prediction only on ignore boxes labelled with the evaluated class.

    ![RelabelBoxes before / after](../../assets/transforms/relabel_boxes.png)

    Args:
        keys: Per-box tensors to filter together (e.g. `DataKeys.BOX`, `DataKeys.LABEL`,
            `DataKeys.TRUNCATION`, `DataKeys.OCCLUSION`). Must include `label_key` and every key
            referenced by `ignore_fields`.
        mapping: Raw-label to detection-label dict; raw labels absent from it (and from `ignore_mapping`)
            are dropped.
        label_key: Key holding the raw integer labels (must be one of `keys`).
        ignore_mapping: Raw labels kept as ignore regions rather than scored ground truth, mapped to the
            detection class they excuse (e.g. KITTI `Van` to the `Car` class index).
        ignore_fields: Per-attribute inclusive ranges `{key: (low, high)}` (use `None` for an open side);
            a foreground box outside any range becomes an ignore region.
        ignore_mask_key: Output key for the written boolean ignore mask.
        allow_missing_keys: If `True`, skip missing keys instead of raising.

    Example:
        ```python
        import torch_pointcloud.transforms as T

        # KITTI: raw 8-class boxes -> 3 detection classes, Van / Person_sitting as ignore regions
        # for Car / Pedestrian, moderate difficulty (occlusion <= 1, truncation <= 0.3,
        # height >= 25 px) as ignore.
        T.RelabelBoxes(
            keys=("box", "label", "truncation", "occlusion", "bbox_height"),
            mapping={0: 0, 3: 1, 5: 2},
            ignore_mapping={1: 0, 4: 1},
            ignore_fields={
                "occlusion": (None, 1),
                "truncation": (None, 0.3),
                "bbox_height": (25, None),
            },
        )
        ```
    """

    def __init__(
        self,
        keys: KeyCollection,
        mapping: Dict[int, int],
        *,
        label_key: str = DataKeys.LABEL,
        ignore_mapping: Optional[Dict[int, int]] = None,
        ignore_fields: Optional[Dict[str, Tuple[Optional[float], Optional[float]]]] = None,
        ignore_mask_key: str = "ignore_mask",
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.mapping = {int(k): int(v) for k, v in mapping.items()}
        self.label_key = label_key
        self.ignore_mapping = {int(k): int(v) for k, v in (ignore_mapping or {}).items()}
        self.ignore_fields = dict(ignore_fields or {})
        self.ignore_mask_key = ignore_mask_key

    def transform(self, data: Dict[str, Any]) -> Dict[str, Any]:
        d = dict(data)
        labels = d[self.label_key].long()
        foreground = torch.isin(labels, torch.tensor(sorted(self.mapping), device=labels.device))
        is_ignore = (
            torch.isin(labels, torch.tensor(sorted(self.ignore_mapping), device=labels.device))
            if self.ignore_mapping
            else torch.zeros_like(labels, dtype=torch.bool)
        )

        hard = torch.zeros_like(labels, dtype=torch.bool)
        for field_key, (low, high) in self.ignore_fields.items():
            value = d[field_key]
            in_range = torch.ones_like(labels, dtype=torch.bool)
            if low is not None:
                in_range &= value >= low
            if high is not None:
                in_range &= value <= high
            hard |= ~in_range

        keep = foreground | is_ignore
        ignore = is_ignore | (foreground & hard)
        new_labels = F.relabel(labels, {**self.ignore_mapping, **self.mapping}, default=-1)

        for key in self.iter_keys(d):
            d[key] = d[key][keep]
        d[self.label_key] = new_labels[keep]
        d[self.ignore_mask_key] = ignore[keep]
        return d


class RenameItems(DictTransform):
    """Rename keys in the dictionary.

    ![RenameItems diagram](../../assets/transforms/rename_items.png)

    Args:
        keys: Source keys to rename.
        names: New key names (same length as `keys`).
        allow_missing_keys: If `True`, silently skip absent source keys.
    """

    def __init__(
        self,
        keys: KeyCollection,
        names: KeyCollection,
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.names = ensure_tuple_size(names, len(self.keys))

    def transform(self, data: Dict[str, Any]) -> Dict[str, Any]:
        data = dict(data)
        for key, dst_key in self.iter_keys(data, self.names):
            data[dst_key] = data.pop(key)
        return data


class CopyItems(DictTransform):
    """Copy values from source keys to new destination keys.

    ![CopyItems diagram](../../assets/transforms/copy_items.png)

    Args:
        keys: Source keys to copy from.
        names: Destination keys to copy to (same length as `keys`).
        allow_missing_keys: If `True`, silently skip absent source keys.
    """

    def __init__(
        self,
        keys: KeyCollection,
        names: KeyCollection,
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.names = ensure_tuple_size(names, len(self.keys))

    def transform(self, data: Dict[str, Any]) -> Dict[str, Any]:
        data = dict(data)
        for key, dst_key in self.iter_keys(data, self.names):
            val = data[key]
            if torch.is_tensor(val):
                val = val.clone()
            elif isinstance(val, np.ndarray):
                val = val.copy()
            data[dst_key] = val
        return data


class SubtractKey(DictTransform):
    """Subtract the value of a reference key from target keys element-wise.

    Computes `data[key] = data[key] - data[sub_key]` for each key. With `axes`
    set, only the listed last-dim indices are subtracted; the other components
    pass through unchanged (useful to shift only XY while keeping Z absolute).

    === "Object"

        ![SubtractKey on an object](../../assets/transforms/subtract_key.png)

    === "Scene"

        ![SubtractKey on a room](../../assets/transforms/subtract_key_scene.png)

    Args:
        keys: Keys whose tensors are modified (subtracted from).
        sub_keys: Keys whose values are subtracted from each target key.
        dst_keys: Where to store results. Defaults to `keys`.
        axes: Optional indices into the last dim restricting which components are
            subtracted. `None` (default) subtracts every component.
        allow_missing_keys: If `True`, silently skip absent target keys.
    """

    def __init__(
        self,
        keys: KeyCollection,
        sub_keys: KeyCollection,
        dst_keys: Optional[KeyCollection] = None,
        axes: Optional[Sequence[int]] = None,
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.sub_keys = ensure_tuple_size(sub_keys, len(self.keys))
        self.dst_keys = ensure_tuple_size(dst_keys or self.keys, len(self.keys))
        self.axes = tuple(axes) if axes is not None else None

    def transform(self, data: Dict[str, Any]) -> Dict[str, Any]:
        data = dict(data)
        for key, sub_key, dst_key in self.iter_keys(data, self.sub_keys, self.dst_keys):
            if self.axes is None:
                data[dst_key] = data[key] - data[sub_key]
            else:
                out = data[key].clone()
                idx = list(self.axes)
                out[..., idx] = out[..., idx] - data[sub_key][..., idx]
                data[dst_key] = out
        return data


class BBoxCenter(DictTransform):
    r"""Derive the center of an axis-aligned bbox stored as a flat tensor.

    Reads a bbox at each source key, laid out as a $(2D,)$ vector
    $[\,\min_0, \ldots, \min_{D-1},\, \max_0, \ldots, \max_{D-1}\,]$, and writes
    the per-axis midpoint $(\min + \max) / 2$ (shape $(D,)$) at the matching
    destination key.

    === "Object"

        ![BBoxCenter on an object](../../assets/transforms/bbox_center.png)

    === "Scene"

        ![BBoxCenter on a room](../../assets/transforms/bbox_center_scene.png)

    Args:
        keys: Source keys holding flat bbox tensors of shape $(2D,)$.
        dst_keys: Destination keys for the centers. Defaults to overwriting
            the source keys.
        allow_missing_keys: If `True`, silently skip absent source keys.

    Example:
        ```python
        from torch_pointcloud.transforms import BBoxCenter

        data = {"block_bbox": torch.tensor([0.0, 0.0, 0.0, 1.5, 1.5, 2.8])}
        BBoxCenter(keys="block_bbox", dst_keys="block_center")(data)
        # data["block_center"] == tensor([0.75, 0.75, 1.40])
        ```
    """

    def __init__(
        self,
        keys: KeyCollection,
        dst_keys: Optional[KeyCollection] = None,
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.dst_keys = ensure_tuple_size(dst_keys or self.keys, len(self.keys))

    def transform(self, data: Dict[str, Any]) -> Dict[str, Any]:
        data = dict(data)
        for key, dst_key in self.iter_keys(data, self.dst_keys):
            bbox = data[key]
            if bbox.numel() % 2 != 0:
                raise ValueError(f"`{key}` must have an even number of elements (got {bbox.numel()}).")
            n_dim = bbox.numel() // 2
            data[dst_key] = (bbox[:n_dim] + bbox[n_dim:]) / 2.0
        return data


class DivideKey(DictTransform):
    """Divide target keys by the value of a reference key element-wise.

    Computes `data[key] = data[key] / data[div_key]` for each key.

    ![DivideKey diagram](../../assets/transforms/divide_key.png)

    Args:
        keys: Keys whose tensors are divided.
        div_keys: Keys whose values are used as the divisors.
        dst_keys: Where to store results. Defaults to `keys`.
        allow_missing_keys: If `True`, silently skip absent target keys.
    """

    def __init__(
        self,
        keys: KeyCollection,
        div_keys: KeyCollection,
        dst_keys: Optional[KeyCollection] = None,
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.div_keys = ensure_tuple_size(div_keys, len(self.keys))
        self.dst_keys = ensure_tuple_size(dst_keys or self.keys, len(self.keys))

    def transform(self, data: Dict[str, Any]) -> Dict[str, Any]:
        data = dict(data)
        for key, div_key, dst_key in self.iter_keys(data, self.div_keys, self.dst_keys):
            data[dst_key] = data[key] / data[div_key]
        return data


class ToTensor(DictTransform):
    """Convert dictionary entries to tensors.

    ![ToTensor diagram](../../assets/transforms/to_tensor.png)

    Args:
        keys: The keys to convert.
        dtype: Target dtype(s).
        device: Target device(s).
        allow_missing_keys: If `True`, the transform will not raise an error if the keys are not present in the data.
    """

    def __init__(
        self,
        keys: KeyCollection,
        dtype: ValueCollection[str | torch.dtype] | None = None,
        device: ValueCollection[str | torch.device] | None = None,
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.dtype = ensure_tuple_size(dtype, len(self.keys))
        self.device = ensure_tuple_size(device, len(self.keys))

    def transform(self, data: Dict[str, Any]) -> Dict[str, Any]:
        data = dict(data)
        for key, dtype, device in self.iter_keys(data, self.dtype, self.device):
            data[key] = torch.as_tensor(data[key], dtype=dtype, device=device)
        return data


class Voxelize(DictTransform):
    r"""Voxelize a point cloud by grid-binning and per-voxel reduction.

    Sub-samples a point cloud to one representative point per occupied voxel,
    and optionally records a source-to-voxel index map for full-resolution
    back-projection.

    Operates on a single sample (pre-collate). With `dst_inverse_key` set, the stored
    tensor has shape $(N_\text{full},)$ with values in $[0, N_\text{voxel})$:
    for each original point $i$, the voxel it belongs to. Downstream code can
    recover full-resolution predictions with `preds_full = preds_voxel[inverse]`.

    If the key already holds a prior inverse map (e.g. from an earlier
    invertible transform), the new map composes with it via gather, so the
    stored tensor always maps from the outermost source space to the current
    predictor space.

    === "Object"

        ![Voxelize on an object](../../assets/transforms/voxelize.png)

    === "Scene"

        ![Voxelize on a room](../../assets/transforms/voxelize_scene.png)

    Args:
        pos_key: Key holding the positions to sub-sample.
        pos_reduce: How to reduce positions per voxel (`mean`/`min`/`max`/`sum`/`first`/`grid`).
        size: Voxel edge length in the same units as the positions. Must be positive.
        method: Voxel-id hashing scheme (`fnv` matches FNV-1a-based reference pipelines; `pyg` is the default).
        reduce: Per-key reduction for `keys`. `None` (the default) resolves per key to `mean` for
            floating-point tensors and `first` for integer tensors (e.g. `segment`). Integer keys keep
            their dtype: non-`first` reductions compute in float and cast back. The `first`
            representative is the first point of each voxel in input order, deterministic across
            devices (unless `random_sample=True`).
        keys: Additional per-point keys to sub-sample (e.g. `color`, `segment`).
        dst_inverse_key: When set, store the source-to-voxel index map under
            this key (auto-composes with any existing value at the same key).
            Leave `None` for training pipelines that never need the inverse.
        grid_pos_key: When set together with a non-`grid` `pos_reduce`, also store
            the integer voxel-grid coordinates under this key. Useful when a model
            needs both real-valued positions (e.g. for rotary position embedding)
            and integer grid coordinates (for serialization / sparse-conv stems).
        random_sample: If `True`, the per-voxel representative used by `reduce="first"`
            (and the `pos`/`grid_pos` derivations) is chosen *randomly* within each
            voxel on every call. Per-voxel random sampling is a meaningful
            training augmentation; leave
            `False` (default) for deterministic validation.
        generator: Optional `torch.Generator` for `random_sample` reproducibility. See `Transform`
            for the multi-worker caveat.
        allow_missing_keys: If `True`, missing keys are skipped silently.

    Raises:
        ValueError: If `size` is not positive, or `pos_reduce` / `method` / any `reduce` entry is
            not one of its allowed values.
    """

    def __init__(
        self,
        pos_key: str,
        pos_reduce: VoxelPosReduce,
        size: float,
        method: VoxelMethod = "pyg",
        reduce: Optional[ValueCollection[VoxelReduce]] = None,
        keys: Optional[KeyCollection] = None,
        dst_inverse_key: Optional[str] = None,
        grid_pos_key: Optional[str] = None,
        random_sample: bool = False,
        generator: Optional[torch.Generator] = None,
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        if size <= 0:
            raise ValueError(f"size must be positive; got {size}.")
        if pos_reduce not in get_args(VoxelPosReduce):
            raise ValueError(f"Invalid pos_reduce: {pos_reduce!r}. Expected one of {get_args(VoxelPosReduce)}.")
        if method not in get_args(VoxelMethod):
            raise ValueError(f"Invalid method: {method!r}. Expected one of {get_args(VoxelMethod)}.")
        self.pos_key = pos_key
        self.pos_reduce = pos_reduce
        self.size = size
        self.reduce = ensure_tuple_size(reduce, len(self.keys))
        invalid = set(self.reduce) - set(get_args(VoxelReduce)) - {None}
        if invalid:
            raise ValueError(f"Invalid reduce(s): {invalid}. Expected one of {get_args(VoxelReduce)}.")
        self.method = method
        self.dst_inverse_key = dst_inverse_key
        self.grid_pos_key = grid_pos_key
        self.random_sample = random_sample
        self.generator = generator

    def _random_perm(self, cluster: torch.Tensor, num_clusters: int) -> torch.Tensor:
        """Pick one random representative-index per cluster (replaces the deterministic perm)."""
        sort_idx = torch.argsort(cluster, stable=True)
        counts = torch.bincount(cluster, minlength=num_clusters)
        idx_ptr = torch.cat([counts.new_zeros(1), counts.cumsum(0)[:-1]])
        rand = torch.rand(num_clusters, device=cluster.device, generator=self.generator)
        offsets = torch.minimum((rand * counts.float()).long(), counts - 1)
        return sort_idx[idx_ptr + offsets]

    def _reduce(
        self,
        tensor: torch.Tensor,
        reduce: str,
        cluster: torch.Tensor,
        perm: torch.Tensor,
    ) -> torch.Tensor:
        if reduce == "first":
            return tensor[perm]

        # Integer min/max/sum scatter natively; the float32 round trip below corrupts values above 2^24.
        if not tensor.is_floating_point() and tensor.dtype != torch.bool and reduce != "mean":
            return scatter(tensor, cluster, dim=0, reduce=reduce)

        # Mean (and any bool reduction) needs float input; cast back so the key keeps its dtype.
        out = scatter(tensor.float(), cluster, dim=0, reduce=reduce)
        return out if tensor.is_floating_point() else out.to(tensor.dtype)

    def transform(self, data: Dict[str, Any]) -> Dict[str, Any]:
        data = dict(data)
        pos = data[self.pos_key]

        if pos.shape[0] == 0:
            if self.dst_inverse_key is not None:
                data[self.dst_inverse_key] = torch.empty(0, dtype=torch.long, device=pos.device)
            if self.grid_pos_key is not None:
                data[self.grid_pos_key] = torch.empty(0, pos.shape[-1], dtype=torch.long, device=pos.device)
            return data

        start = torch.floor(pos.min(dim=0).values / self.size) * self.size

        if self.method == "fnv":
            # This method is supported only for debugging and reproducibility against FNV-hash-based
            # grid subsampling. This method might be removed in the future (?)
            cluster = voxel_grid_fnv(pos, size=self.size, start=start)
        else:
            cluster = voxel_grid(pos, size=self.size, start=start)

        cluster, _ = consecutive_cluster(cluster)
        num_clusters = int(cluster.max().item()) + 1

        if self.random_sample:
            perm = self._random_perm(cluster, num_clusters=num_clusters)
        else:
            perm = first_permutation(cluster, num_clusters=num_clusters)

        if self.pos_reduce == "grid":
            pos_grid = torch.floor((pos[perm] - start) / self.size).long()
            data[self.pos_key] = pos_grid - pos_grid.min(dim=0).values
        else:
            data[self.pos_key] = self._reduce(pos, self.pos_reduce, cluster, perm)
            if self.grid_pos_key is not None:
                pos_grid = torch.floor((pos[perm] - start) / self.size).long()
                data[self.grid_pos_key] = pos_grid - pos_grid.min(dim=0).values

        for key, reduce in self.iter_keys(data, self.reduce):
            tensor = data[key]
            if reduce is None:
                reduce = "mean" if tensor.is_floating_point() else "first"
            data[key] = self._reduce(tensor, reduce, cluster, perm)

        if self.dst_inverse_key is not None:
            prior = data.get(self.dst_inverse_key)
            data[self.dst_inverse_key] = cluster if prior is None else cluster[prior]

        return data


class Quantize(DictTransform):
    r"""Integer voxel-grid coordinates of every point, keeping the cloud at full resolution.

    Stores $\lfloor p / s \rfloor$ (shifted so the per-axis minimum is $0$) for each point of `keys` under
    `dst_keys`. Unlike `Voxelize`, no reduction happens: points sharing a voxel keep their own rows and get equal
    coordinates. This is how a voxel-partition evaluation feeds sparse models with every raw point (each
    sub-cloud holds one point per voxel, so its rows are exactly the voxels), and how test-time views recompute
    grid coordinates after rotating or scaling the positions.

    === "Object"

        ![Quantize on an object](../../assets/transforms/quantize.png)

    === "Scene"

        ![Quantize on a room](../../assets/transforms/quantize_scene.png)

    Args:
        keys: Keys holding point positions of shape $(N, D)$.
        size: Voxel side length in the units of the positions.
        dst_keys: Keys under which the grid coordinates are stored. Defaults to `keys` (in-place overwrite).
        allow_missing_keys: If `True`, missing keys are skipped.

    Example:
        ```python
        import torch
        from torch_pointcloud.transforms import Quantize

        transform = Quantize(keys="pos", size=0.02, dst_keys="pos_grid")
        data = transform({"pos": torch.tensor([[0.0, 0.0, 0.0], [0.03, 0.0, 0.0], [0.05, 0.0, 0.0]])})
        data["pos_grid"]  # tensor([[0, 0, 0], [1, 0, 0], [2, 0, 0]])
        ```
    """

    def __init__(
        self,
        keys: KeyCollection,
        size: float,
        dst_keys: Optional[KeyCollection] = None,
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        if size <= 0.0:
            raise ValueError(f"`size` must be > 0, got {size}.")

        self.size = size
        self.dst_keys = ensure_tuple_size(dst_keys, len(self.keys)) if dst_keys is not None else self.keys

    def transform(self, data: Dict[str, Any]) -> Dict[str, Any]:
        data = dict(data)
        for key, dst_key in self.iter_keys(data, self.dst_keys):
            data[dst_key] = F.quantize(data[key], self.size)
        return data


class OnesLike(DictTransform):
    """Adds a tensor of ones shaped like existing dictionary entries.

    ![OnesLike diagram](../../assets/transforms/ones_like.png)

    Args:
        keys: Reference keys used to determine tensor shape.
        dst_keys: Keys under which the ones tensors are stored.
    """

    def __init__(
        self,
        keys: KeyCollection,
        memory_format: ValueCollection[torch.memory_format] | None = None,
        dtype: ValueCollection[torch.dtype] | None = None,
        layout: ValueCollection[torch.layout] | None = None,
        device: ValueCollection[torch.device] | None = None,
        pin_memory: ValueCollection[bool] | None = False,
        requires_grad: ValueCollection[bool] | None = False,
        dst_keys: KeyCollection | None = None,
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.dst_keys = ensure_tuple_size(dst_keys or self.keys, len(self.keys))
        self.memory_format = ensure_tuple_size(memory_format, len(self.keys))
        self.dtype = ensure_tuple_size(dtype, len(self.keys))
        self.layout = ensure_tuple_size(layout, len(self.keys))
        self.device = ensure_tuple_size(device, len(self.keys))
        self.pin_memory = ensure_tuple_size(pin_memory, len(self.keys))
        self.requires_grad = ensure_tuple_size(requires_grad, len(self.keys))

    def transform(self, data: Dict[str, Any]) -> Dict[str, Any]:
        data = dict(data)
        for key, dst_key, memory_format, dtype, layout, device, pin_memory, requires_grad in self.iter_keys(
            data,
            self.dst_keys,
            self.memory_format,
            self.dtype,
            self.layout,
            self.device,
            self.pin_memory,
            self.requires_grad,
        ):
            data[dst_key] = torch.ones_like(
                data[key],
                memory_format=memory_format,
                dtype=dtype,
                layout=layout,
                device=device,
                pin_memory=pin_memory,
                requires_grad=requires_grad,
            )
        return data


class AxisMinOffset(DictTransform):
    r"""Per-point offset from a floor reference along a chosen coordinate axis.

    For each point and a given axis $a$ along tensor dimension $d$, computes:

    $$
    o_i = p_{i,a} - r
    $$

    where the floor reference $r$ is either the strict minimum $\min_j p_{j,a}$
    (default) or, when `quantile` is set, the empirical quantile
    $Q_q(p_{\cdot,a})$. A small positive quantile gives an outlier-robust floor
    estimate: `quantile=0.0099` reproduces VoteNet's `np.percentile(z, 0.99)`
    height feature.

    The result has the same shape as the input with the coordinate dimension
    reduced to size 1 (e.g. $(N, 3) \to (N, 1)$ or $(B, N, 3) \to (B, N, 1)$).
    For batched inputs, the minimum is computed per-sample.

    === "Object"

        ![AxisMinOffset on an object](../../assets/transforms/axis_min_offset.png)

    === "Scene"

        ![AxisMinOffset on a room](../../assets/transforms/axis_min_offset_scene.png)

    Args:
        keys: Keys holding point positions of shape $(N, D)$.
        axis: Coordinate axis $a$ along which to compute the offset.
        quantile: Optional quantile $q \in [0, 1]$ for the floor reference. When
            `None`, the strict per-axis minimum is used.
        dst_keys: Keys under which the offset tensors are stored. Defaults to `keys`
            (in-place overwrite).
        allow_missing_keys: If True, skip missing keys instead of raising.

    Example:
        Let's say you have a point cloud with positions $(N, 3)$ in XYZ order
        and you want to compute the offset from the minimum along the z-axis,
        i.e. computing the height above the local floor.

        ```python
        from torch_pointcloud.transforms import AxisMinOffset

        data = {
            "pos": torch.randn(10, 3),
        }
        transform = AxisMinOffset(keys="pos", dst_keys="pos_offset", axis=2)
        data = transform(data)
        ```

        Now, the data dictionary will contain the key `pos_offset` with the shape $(N, 1)$.
    """

    def __init__(
        self,
        keys: KeyCollection,
        axis: ValueCollection[int],
        quantile: Optional[float] = None,
        dst_keys: KeyCollection | None = None,
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.dst_keys = ensure_tuple_size(dst_keys or self.keys, len(self.keys))
        self.axis = ensure_tuple_size(axis, len(self.keys))
        self.quantile = quantile

    def transform(self, data: Dict[str, Any]) -> Dict[str, Any]:
        data = dict(data)
        for key, dst_key, axis in self.iter_keys(data, self.dst_keys, self.axis):
            data[dst_key] = F.axis_min_offset(data[key], axis=axis, quantile=self.quantile)
        return data


class Cat(DictTransform):
    """Concatenates tensors from multiple keys into a single feature tensor.

    Note:
        This transform is mostly used to concatenate multiple features into a single tensor to feed into your model.

    Integer inputs are cast to `float32`; floating inputs keep their dtype. When the inputs mix
    floating dtypes, the result uses the widest one (so `float64` is preserved, never downcast).

    ![Cat diagram](../../assets/transforms/cat.png)

    Args:
        keys: Keys whose tensors are concatenated (in order).
        dst_key: Key under which the result is stored.
        dim: Dimension along which to concatenate.
        allow_missing_keys: If `True`, silently skip absent keys.

    Example:
        If you have a point cloud data containing position, color and normal and want to concatenate them
        into a single feature tensor (to feed into your model), you can do the following:

        ```python
        from torch_pointcloud.transforms import Cat

        data = {
            "pos": torch.randn(10, 3),
            "color": torch.randn(10, 3),
            "normal": torch.randn(10, 3),
        }
        transform = Cat(keys=["pos", "color", "normal"], dst_key="x", dim=1)
        data = transform(data)
        ```

        Now, the data dictionary will contain the key `x` with the shape $(10, 9)$.
    """

    def __init__(
        self,
        keys: KeyCollection,
        dst_key: str,
        dim: int = -1,
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.dst_key = dst_key
        self.dim = dim

    def transform(self, data: Dict[str, Any]) -> Dict[str, Any]:
        data = dict(data)
        tensors = [data[key] if data[key].is_floating_point() else data[key].float() for key in self.iter_keys(data)]
        if not tensors:
            return data
        dtype = tensors[0].dtype
        for tensor in tensors[1:]:
            dtype = torch.promote_types(dtype, tensor.dtype)
        data[self.dst_key] = torch.cat([tensor.to(dtype) for tensor in tensors], dim=self.dim)
        return data


class OneHot(DictTransform):
    r"""One-hot encode integer-class tensors.

    Wraps `torch.nn.functional.one_hot` and casts the result to float so the
    output is ready to feed into a model.

    ![OneHot diagram](../../assets/transforms/one_hot.png)

    Args:
        keys: Keys holding integer (long) class indices.
        num_classes: Number of classes $C$ in the one-hot encoding.
        dst_keys: Where to store the one-hot tensors. Defaults to `keys`.
        allow_missing_keys: If `True`, silently skip absent keys.

    Shape:
        Input class tensor of shape $(N,)$ becomes $(N, C)$. A scalar input
        becomes shape $(C,)$, which after batched collate stacks to $(B, C)$.
    """

    def __init__(
        self,
        keys: KeyCollection,
        num_classes: ValueCollection[int],
        dst_keys: Optional[KeyCollection] = None,
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.dst_keys = ensure_tuple_size(dst_keys or self.keys, len(self.keys))
        self.num_classes = ensure_tuple_size(num_classes, len(self.keys))

    def transform(self, data: Dict[str, Any]) -> Dict[str, Any]:
        data = dict(data)
        for key, dst_key, num_classes in self.iter_keys(data, self.dst_keys, self.num_classes):
            x = data[key].long()
            out = torch.nn.functional.one_hot(x, num_classes=num_classes).float()
            # A 0-d input (per-sample scalar label) one-hots to `(num_classes,)`. Unsqueeze
            # so packed-batch collate yields `(B, num_classes)` after `torch.cat(dim=0)`.
            if x.ndim == 0:
                out = out.unsqueeze(0)
            data[dst_key] = out
        return data


class Reduce(DictTransform):
    r"""Reduce a tensor along a dimension and store the scalar/vector result.

    Useful for capturing per-sample statistics (e.g. axis-wise scene maxima or
    centroids) as standalone keys that downstream transforms can reference.

    ![Reduce diagram](../../assets/transforms/reduce.png)

    Args:
        keys: Keys to reduce.
        op: Reduction operator: `"min"`, `"max"`, `"mean"`, or `"sum"` (matches the
            vocabulary used by `Voxelize`). `"mean"` keeps the input's floating dtype
            (`float64` included); integer inputs are cast to `float32`.
        dim: Dimension to reduce. Defaults to `0`.
        keepdim: Pass `keepdim=True` to keep the reduced axis as size $1$. This
            is helpful when the result is meant to broadcast against a $(N, D)$
            tensor (e.g. per-sample bbox stats) and to survive the packed-batch
            collate - a $(1, D)$ tensor collates to $(B, D)$ via `torch.cat`,
            whereas a $(D,)$ tensor would concatenate to $(B \cdot D,)$.
        dst_keys: Output keys. Defaults to `keys` (in-place overwrite).
        allow_missing_keys: If `True`, silently skip absent keys.
    """

    _OP_FUNCS: Dict[str, Any] = {
        "min": torch.amin,
        "max": torch.amax,
        "sum": torch.sum,
    }

    def __init__(
        self,
        keys: KeyCollection,
        op: ValueCollection[ReduceOp],
        dim: ValueCollection[int] = 0,
        keepdim: bool = False,
        dst_keys: Optional[KeyCollection] = None,
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.dst_keys = ensure_tuple_size(dst_keys or self.keys, len(self.keys))
        self.op = ensure_tuple_size(op, len(self.keys))
        self.dim = ensure_tuple_size(dim, len(self.keys))
        self.keepdim = keepdim

        valid = get_args(ReduceOp)
        invalid = set(self.op) - set(valid)
        if invalid:
            raise ValueError(f"Invalid op(s): {invalid}. Expected one of {valid}.")

    def transform(self, data: Dict[str, Any]) -> Dict[str, Any]:
        data = dict(data)
        for key, dst_key, op, dim in self.iter_keys(data, self.dst_keys, self.op, self.dim):
            x = data[key]
            if op == "mean":
                x_float = x if x.is_floating_point() else x.float()
                data[dst_key] = x_float.mean(dim=dim, keepdim=self.keepdim)
            else:
                data[dst_key] = self._OP_FUNCS[op](x, dim=dim, keepdim=self.keepdim)
        return data


class KeepItems(DictTransform):
    r"""Keep only items in the data dictionary that are in the keys list.

    Note:
        This transform is useful if during augmentation process you constructed multiple tensors and want
        to drop intermediate tensors for memory efficiency.

    ![KeepItems diagram](../../assets/transforms/keep_items.png)

    Args:
        keys: The keys to keep in the data dictionary.
        allow_missing_keys: If `True`, the transform will not raise an error if the keys are not present in the data.

    Example:
        If you have a data dictionary containing position, color and normal and want to keep only the position and color,
        you can do the following:

        ```python
        from torch_pointcloud.transforms import KeepItems

        data = {
            "pos": torch.randn(10, 3),
            "color": torch.randn(10, 3),
            "normal": torch.randn(10, 3),
        }
        transform = KeepItems(keys=["pos", "color"])
        data = transform(data)
        ```

        Now, the data dictionary will contain only the keys `pos` and `color`.
        The key `normal` will be removed.
    """

    def __init__(
        self,
        keys: KeyCollection,
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)

    def transform(self, data: Dict[str, Any]) -> Dict[str, Any]:
        data = dict(data)
        return {key: data[key] for key in self.iter_keys(data)}


class RandomRotate(DictTransform):
    r"""Rotate one or more keys (and optionally oriented boxes) by a uniformly random angle around an axis.

    Sampling is done once per call: every listed key and the optional box get the same rotation. Each key is a
    $(\ldots, 3)$ field or a packed $(N, 3G)$ field of tiled 3D offsets (e.g. VoteNet votes). Pair
    `keys=("pos", "normal")` to keep positions and normals consistent, or pass `box_key` to also rotate a
    $(K, 7)$ oriented-box tensor (centers rotated, heading incremented). Box headings are counterclockwise
    yaw about the up axis, so `box_key` requires `axis=2`.

    === "Object"

        ![RandomRotate on an object](../../assets/transforms/random_rotate.png)

    === "Scene"

        ![RandomRotate on a room](../../assets/transforms/random_rotate_scene.png)

    === "Per axis"

        ![RandomRotate around each axis on an object](../../assets/transforms/rotate_axes.png)

    See Also:
        `torch_pointcloud.transforms.functional.rotate_vectors`,
        `torch_pointcloud.transforms.functional.rotate_boxes`,
        `torch_pointcloud.transforms.functional.rotation_matrix`

    Args:
        keys: Keys to rotate. Each must be a $(\ldots, 3)$ or $(N, 3G)$ vector field.
        angle_range: Min and max rotation angle, in **degrees**.
        axis: Axis index to rotate around (0=X, 1=Y, 2=Z).
        p: Probability of applying the transform.
        box_key: Optional key of a $(K, 7)$ oriented-box tensor to rotate jointly (requires `axis=2`).
        dst_keys: Where to store the rotated tensors. Defaults to `keys` (in-place).
        dst_box_key: Where to store the rotated boxes. Defaults to `box_key` (in-place).
        generator: Optional `torch.Generator` for reproducibility. See `Transform` for the
            multi-worker caveat.
        allow_missing_keys: If `True`, silently skip absent keys.
    """

    def __init__(
        self,
        keys: KeyCollection,
        angle_range: Tuple[float, float] = (-180.0, 180.0),
        axis: int = 2,
        p: float = 1.0,
        box_key: Optional[str] = None,
        dst_keys: Optional[KeyCollection] = None,
        dst_box_key: Optional[str] = None,
        generator: Optional[torch.Generator] = None,
        allow_missing_keys: bool = False,
    ) -> None:
        if box_key is not None and axis != 2:
            raise ValueError(f"box_key rotation is only defined about the up axis (axis=2), got axis={axis}.")
        super().__init__(keys, allow_missing_keys)
        self.angle_range = angle_range
        self.axis = axis
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"p must be in [0, 1]; got {p}.")
        self.p = p
        self.box_key = box_key
        self.dst_keys = ensure_tuple_size(dst_keys or self.keys, len(self.keys))
        self.dst_box_key = dst_box_key or box_key
        self.generator = generator

    def transform(self, data: Dict[str, Any]) -> Dict[str, Any]:
        data = dict(data)
        if torch.rand(1, generator=self.generator).item() >= self.p:
            return data
        lo, hi = self.angle_range
        angle = math.radians(torch.empty(1).uniform_(lo, hi, generator=self.generator).item())
        rotation = F.rotation_matrix(angle, self.axis)
        box_key = self.box_key
        if box_key is not None and box_key in data:
            assert self.dst_box_key is not None
            data[self.dst_box_key] = F.rotate_boxes(data[box_key], rotation, angle)
        for key, dst_key in self.iter_keys(data, self.dst_keys):
            data[dst_key] = F.rotate_vectors(data[key], rotation)
        return data


class RandomScale(DictTransform):
    """Scale one or more keys (and optionally oriented boxes) by a uniformly random factor.

    Sampling is done once per call: every listed key and the optional box are scaled by the same factor (or
    per-axis factor vector when `anisotropic=True`). Pass `box_key` to also scale a $(K, 7)$ oriented-box
    tensor (centers and sizes). An oriented box has no per-axis scale, so `box_key` is incompatible with
    `anisotropic=True`.

    List only point-like keys. Do not list direction vectors such as `normal`: a scaled normal is no
    longer unit length, while a true surface normal is unchanged by an isotropic scale (and an
    anisotropic scale would require the inverse-transpose rule). Simply omit normal keys.

    === "Object"

        ![RandomScale on an object](../../assets/transforms/random_scale.png)

    === "Scene"

        ![RandomScale on a room](../../assets/transforms/random_scale_scene.png)

    See Also:
        `torch_pointcloud.transforms.functional.scale_boxes`

    Args:
        keys: Keys to scale. Point-like keys only; do not list direction vectors such as `normal`.
        scale_range: Min and max scaling factor.
        anisotropic: If `True`, sample a separate scale per axis of the last dim (incompatible with `box_key`).
        p: Probability of applying the transform.
        box_key: Optional key of a $(K, 7)$ oriented-box tensor to scale jointly.
        dst_keys: Where to store the scaled tensors.
        dst_box_key: Where to store the scaled boxes. Defaults to `box_key` (in-place).
        generator: Optional `torch.Generator` for reproducibility. See `Transform` for the
            multi-worker caveat.
        allow_missing_keys: If `True`, silently skip absent keys.
    """

    def __init__(
        self,
        keys: KeyCollection,
        scale_range: Tuple[float, float] = (0.8, 1.25),
        anisotropic: bool = False,
        p: float = 1.0,
        box_key: Optional[str] = None,
        dst_keys: Optional[KeyCollection] = None,
        dst_box_key: Optional[str] = None,
        generator: Optional[torch.Generator] = None,
        allow_missing_keys: bool = False,
    ) -> None:
        if anisotropic and box_key is not None:
            raise ValueError("box_key cannot be scaled anisotropically (an oriented box has no per-axis scale).")
        super().__init__(keys, allow_missing_keys)
        self.scale_range = scale_range
        self.anisotropic = anisotropic
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"p must be in [0, 1]; got {p}.")
        self.p = p
        self.box_key = box_key
        self.dst_keys = ensure_tuple_size(dst_keys or self.keys, len(self.keys))
        self.dst_box_key = dst_box_key or box_key
        self.generator = generator

    def transform(self, data: Dict[str, Any]) -> Dict[str, Any]:
        data = dict(data)
        if torch.rand(1, generator=self.generator).item() >= self.p:
            return data
        lo, hi = self.scale_range
        box_key = self.box_key
        has_box = box_key is not None and box_key in data
        first_key = next(iter(self.iter_keys(data)), None)
        if first_key is None and not has_box:
            return data
        if self.anisotropic and first_key is not None:
            scale = torch.empty(data[first_key].shape[-1]).uniform_(lo, hi, generator=self.generator)
        else:
            scale = torch.empty(1).uniform_(lo, hi, generator=self.generator)
        if box_key is not None and box_key in data:
            assert self.dst_box_key is not None
            data[self.dst_box_key] = F.scale_boxes(data[box_key], scale.to(data[box_key]))
        for key, dst_key in self.iter_keys(data, self.dst_keys):
            x = data[key]
            if self.anisotropic and x.shape[-1] != scale.numel():
                raise ValueError(
                    f"RandomScale(anisotropic=True) draws one factor per channel of the first key "
                    f"({scale.numel()}); key '{key}' has {x.shape[-1]} channels."
                )
            data[dst_key] = x * scale.to(x.dtype).to(x.device)
        return data


class RandomFlip(DictTransform):
    r"""Flip listed axes (and optionally oriented boxes) with probability `p` each.

    Sampling is done once per call: every listed key and the optional box are flipped on the same axes. Each
    key is a $(\ldots, 3)$ field or a packed $(N, 3G)$ field of tiled 3D offsets (e.g. VoteNet votes). Pass
    `box_key` to also flip a $(K, 7)$ oriented-box tensor (centers negated, heading remapped).

    === "Object"

        ![RandomFlip on an object](../../assets/transforms/random_flip.png)

    === "Scene"

        ![RandomFlip on a room](../../assets/transforms/random_flip_scene.png)

    === "Per axis"

        ![RandomFlip across each axis on an object](../../assets/transforms/flip_axes.png)

    See Also:
        `torch_pointcloud.transforms.functional.flip_vectors`,
        `torch_pointcloud.transforms.functional.flip_boxes`

    Args:
        keys: Keys to flip. Each must be a $(\ldots, 3)$ or $(N, 3G)$ vector field.
        axes: Axis indices (into each 3D triple) to consider for flipping.
        p: Per-axis flip probability.
        box_key: Optional key of a $(K, 7)$ oriented-box tensor to flip jointly.
        dst_keys: Where to store the flipped tensors.
        dst_box_key: Where to store the flipped boxes. Defaults to `box_key` (in-place).
        generator: Optional `torch.Generator` for reproducibility. See `Transform` for the
            multi-worker caveat.
        allow_missing_keys: If `True`, silently skip absent keys.
    """

    def __init__(
        self,
        keys: KeyCollection,
        axes: Sequence[int] = (0, 1),
        p: float = 0.5,
        box_key: Optional[str] = None,
        dst_keys: Optional[KeyCollection] = None,
        dst_box_key: Optional[str] = None,
        generator: Optional[torch.Generator] = None,
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.axes = tuple(axes)
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"p must be in [0, 1]; got {p}.")
        self.p = p
        self.box_key = box_key
        self.dst_keys = ensure_tuple_size(dst_keys or self.keys, len(self.keys))
        self.dst_box_key = dst_box_key or box_key
        self.generator = generator

    def transform(self, data: Dict[str, Any]) -> Dict[str, Any]:
        data = dict(data)
        box_key = self.box_key
        has_box = box_key is not None and box_key in data
        if next(iter(self.iter_keys(data)), None) is None and not has_box:
            return data
        flipped = [axis for axis in self.axes if torch.rand(1, generator=self.generator).item() < self.p]
        if not flipped:
            return data
        if box_key is not None and box_key in data:
            assert self.dst_box_key is not None
            box = data[box_key]
            for axis in flipped:
                box = F.flip_boxes(box, axis)
            data[self.dst_box_key] = box
        for key, dst_key in self.iter_keys(data, self.dst_keys):
            x = data[key]
            for axis in flipped:
                x = F.flip_vectors(x, axis)
            data[dst_key] = x
        return data


class RandomJitter(DictTransform):
    """Add Gaussian noise to listed keys, optionally clipped.

    Each key gets its own independent noise tensor (because the noise shape
    matches the key shape). Pair-rotation-style consistency does not apply here.

    === "Object"

        ![RandomJitter on an object](../../assets/transforms/random_jitter.png)

    === "Scene"

        ![RandomJitter on a room](../../assets/transforms/random_jitter_scene.png)

    See Also:
        `torch_pointcloud.transforms.functional.random_jitter`

    Args:
        keys: Keys to jitter.
        sigma: Standard deviation of the Gaussian noise.
        clip: If not `None`, clip the noise to `[-clip, clip]`.
        p: Probability of applying the transform.
        dst_keys: Where to store the jittered tensors.
        generator: Optional `torch.Generator` for reproducibility. See `Transform` for the
            multi-worker caveat.
        allow_missing_keys: If `True`, silently skip absent keys.
    """

    def __init__(
        self,
        keys: KeyCollection,
        sigma: float = 0.01,
        clip: Optional[float] = 0.05,
        p: float = 1.0,
        dst_keys: Optional[KeyCollection] = None,
        generator: Optional[torch.Generator] = None,
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.sigma = sigma
        self.clip = clip
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"p must be in [0, 1]; got {p}.")
        self.p = p
        self.dst_keys = ensure_tuple_size(dst_keys or self.keys, len(self.keys))
        self.generator = generator

    def transform(self, data: Dict[str, Any]) -> Dict[str, Any]:
        data = dict(data)
        if torch.rand(1, generator=self.generator).item() >= self.p:
            return data
        for key, dst_key in self.iter_keys(data, self.dst_keys):
            data[dst_key] = F.random_jitter(data[key], self.sigma, self.clip, generator=self.generator)
        return data


class RandomShift(DictTransform):
    """Translate listed keys (and optionally oriented boxes) by a uniformly random vector.

    Sampling is done once per call: all listed keys and the optional box are shifted by the same
    translation vector. Pass `box_key` to also shift a $(K, 7)$ oriented-box tensor (centers only;
    sizes and heading unchanged).

    List only point-like keys. Do not list direction vectors such as `normal`: directions are
    translation-invariant, so a shifted normal is wrong. Simply omit normal keys.

    === "Object"

        ![RandomShift on an object](../../assets/transforms/random_shift.png)

    === "Scene"

        ![RandomShift on a room](../../assets/transforms/random_shift_scene.png)

    See Also:
        `torch_pointcloud.transforms.functional.shift_boxes`

    Args:
        keys: Keys to shift. Point-like keys only; do not list direction vectors such as `normal`.
        shift_range: Min and max per-axis translation.
        p: Probability of applying the transform.
        box_key: Optional key of a $(K, 7)$ oriented-box tensor to shift jointly.
        dst_keys: Where to store the shifted tensors.
        dst_box_key: Where to store the shifted boxes. Defaults to `box_key` (in-place).
        generator: Optional `torch.Generator` for reproducibility. See `Transform` for the
            multi-worker caveat.
        allow_missing_keys: If `True`, silently skip absent keys.
    """

    def __init__(
        self,
        keys: KeyCollection,
        shift_range: Tuple[float, float] = (-0.2, 0.2),
        p: float = 1.0,
        box_key: Optional[str] = None,
        dst_keys: Optional[KeyCollection] = None,
        dst_box_key: Optional[str] = None,
        generator: Optional[torch.Generator] = None,
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.shift_range = shift_range
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"p must be in [0, 1]; got {p}.")
        self.p = p
        self.box_key = box_key
        self.dst_keys = ensure_tuple_size(dst_keys or self.keys, len(self.keys))
        self.dst_box_key = dst_box_key or box_key
        self.generator = generator

    def transform(self, data: Dict[str, Any]) -> Dict[str, Any]:
        data = dict(data)
        if torch.rand(1, generator=self.generator).item() >= self.p:
            return data
        lo, hi = self.shift_range
        box_key = self.box_key
        has_box = box_key is not None and box_key in data
        first_key = next(iter(self.iter_keys(data)), None)
        if first_key is None and not has_box:
            return data
        d = data[first_key].shape[-1] if first_key is not None else 3
        shift = torch.empty(d).uniform_(lo, hi, generator=self.generator)
        if box_key is not None and box_key in data:
            assert self.dst_box_key is not None
            data[self.dst_box_key] = F.shift_boxes(data[box_key], shift[:3])
        for key, dst_key in self.iter_keys(data, self.dst_keys):
            x = data[key]
            if x.shape[-1] != shift.numel():
                raise ValueError(
                    f"RandomShift draws one offset per channel of the first key ({shift.numel()}); "
                    f"key '{key}' has {x.shape[-1]} channels."
                )
            data[dst_key] = x + shift.to(x.dtype).to(x.device)
        return data


class RandomDropout(DictTransform):
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
        generator: Optional `torch.Generator` for reproducibility. See `Transform` for the
            multi-worker caveat.
        allow_missing_keys: If `True`, silently skip absent keys.
    """

    def __init__(
        self,
        keys: KeyCollection,
        p_drop: float = 0.1,
        p: float = 1.0,
        generator: Optional[torch.Generator] = None,
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        if not 0.0 <= p_drop < 1.0:
            raise ValueError(f"p_drop must be in [0, 1); got {p_drop}.")
        self.p_drop = p_drop
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"p must be in [0, 1]; got {p}.")
        self.p = p
        self.generator = generator

    def transform(self, data: Dict[str, Any]) -> Dict[str, Any]:
        data = dict(data)
        if torch.rand(1, generator=self.generator).item() >= self.p:
            return data
        first_key = next(iter(self.iter_keys(data)), None)
        if first_key is None:
            return data
        n = data[first_key].shape[0]
        keep = F.random_dropout_mask(n, self.p_drop, device=data[first_key].device, generator=self.generator)
        for key in self.iter_keys(data):
            data[key] = data[key][keep]
        return data


class RandomColorJitter(DictTransform):
    """Jitter colors by brightness, contrast, and saturation strengths.

    Each strength is a relative delta uniformly sampled from `[-x, x]`. Sampling
    is once per call, so the same factors are applied to every listed key.

    ![RandomColorJitter before / after](../../assets/transforms/color_jitter.png)

    See Also:
        `torch_pointcloud.transforms.functional.color_jitter`

    Args:
        keys: Color keys to jitter, shape $(N, 3)$.
        brightness: Max relative brightness change in $[0, 1]$.
        contrast: Max relative contrast change in $[0, 1]$.
        saturation: Max relative saturation change in $[0, 1]$.
        int_color: If `True`, treat float colors as `[0, 255]` values; otherwise `[0, 1]`.
            `uint8` colors are always treated as `[0, 255]` regardless of the flag; float colors
            above 1 with `int_color=False` raise a ValueError.
        p: Probability of applying the transform.
        dst_keys: Where to store the jittered tensors.
        generator: Optional `torch.Generator` for reproducibility. See `Transform` for the
            multi-worker caveat.
        allow_missing_keys: If `True`, silently skip absent keys.
    """

    def __init__(
        self,
        keys: KeyCollection,
        brightness: float = 0.4,
        contrast: float = 0.4,
        saturation: float = 0.2,
        int_color: bool = False,
        p: float = 1.0,
        dst_keys: Optional[KeyCollection] = None,
        generator: Optional[torch.Generator] = None,
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.int_color = int_color
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"p must be in [0, 1]; got {p}.")
        self.p = p
        self.dst_keys = ensure_tuple_size(dst_keys or self.keys, len(self.keys))
        self.generator = generator

    def transform(self, data: Dict[str, Any]) -> Dict[str, Any]:
        data = dict(data)
        if torch.rand(1, generator=self.generator).item() >= self.p:
            return data
        b, c, s = self.brightness, self.contrast, self.saturation
        brightness = torch.empty(1).uniform_(1 - b, 1 + b, generator=self.generator).item() if b > 0 else None
        contrast = torch.empty(1).uniform_(1 - c, 1 + c, generator=self.generator).item() if c > 0 else None
        saturation = torch.empty(1).uniform_(1 - s, 1 + s, generator=self.generator).item() if s > 0 else None
        for key, dst_key in self.iter_keys(data, self.dst_keys):
            data[dst_key] = F.color_jitter(
                data[key],
                brightness=brightness,
                contrast=contrast,
                saturation=saturation,
                int_color=self.int_color,
            )
        return data


class RandomColorDrop(DictTransform):
    """Replace colors with a constant gray value with probability `p`.

    ![RandomColorDrop before / after](../../assets/transforms/color_drop.png)

    See Also:
        `torch_pointcloud.transforms.functional.random_color_drop`

    Args:
        keys: Color keys to drop.
        fill: Replacement value in the range implied by `int_color` (`[0, 1]` when `False`,
            `[0, 255]` when `True`); rescaled to the input's actual range when that differs, so
            the default `0.5` fills `127` on `uint8` colors.
        int_color: If `True`, treat float colors as `[0, 255]` values; otherwise `[0, 1]`.
            `uint8` colors are always treated as `[0, 255]` regardless of the flag; float colors
            above 1 with `int_color=False` raise a ValueError.
        p: Probability of dropping colors.
        dst_keys: Where to store the result.
        generator: Optional `torch.Generator` for reproducibility. See `Transform` for the
            multi-worker caveat.
        allow_missing_keys: If `True`, silently skip absent keys.
    """

    def __init__(
        self,
        keys: KeyCollection,
        fill: float = 0.5,
        int_color: bool = False,
        p: float = 0.2,
        dst_keys: Optional[KeyCollection] = None,
        generator: Optional[torch.Generator] = None,
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.fill = fill
        self.int_color = int_color
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"p must be in [0, 1]; got {p}.")
        self.p = p
        self.dst_keys = ensure_tuple_size(dst_keys or self.keys, len(self.keys))
        self.generator = generator

    def transform(self, data: Dict[str, Any]) -> Dict[str, Any]:
        data = dict(data)
        if torch.rand(1, generator=self.generator).item() >= self.p:
            return data
        for key, dst_key in self.iter_keys(data, self.dst_keys):
            data[dst_key] = F.random_color_drop(data[key], fill=self.fill, int_color=self.int_color)
        return data


class RandomColorGrayScale(DictTransform):
    """Convert listed color keys to grayscale (BT.601 luminance) with probability `p`.

    ![RandomColorGrayScale before / after](../../assets/transforms/color_grayscale.png)

    See Also:
        `torch_pointcloud.transforms.functional.color_grayscale`

    Args:
        keys: Color keys, shape $(N, 3)$.
        int_color: If `True`, treat colors as `[0, 255]` ints; otherwise `[0, 1]` floats.
        p: Probability of converting to grayscale.
        dst_keys: Where to store the result.
        generator: Optional `torch.Generator` for reproducibility. See `Transform` for the
            multi-worker caveat.
        allow_missing_keys: If `True`, silently skip absent keys.
    """

    def __init__(
        self,
        keys: KeyCollection,
        int_color: bool = False,
        p: float = 0.2,
        dst_keys: Optional[KeyCollection] = None,
        generator: Optional[torch.Generator] = None,
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.int_color = int_color
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"p must be in [0, 1]; got {p}.")
        self.p = p
        self.dst_keys = ensure_tuple_size(dst_keys or self.keys, len(self.keys))
        self.generator = generator

    def transform(self, data: Dict[str, Any]) -> Dict[str, Any]:
        data = dict(data)
        if torch.rand(1, generator=self.generator).item() >= self.p:
            return data
        for key, dst_key in self.iter_keys(data, self.dst_keys):
            data[dst_key] = F.color_grayscale(data[key], int_color=self.int_color)
        return data


class RandomColorAutoContrast(DictTransform):
    """Stretch per-cloud color range to the full extent, then blend back, with probability `p`.

    ![RandomColorAutoContrast before / after](../../assets/transforms/color_auto_contrast.png)

    See Also:
        `torch_pointcloud.transforms.functional.color_auto_contrast`

    Args:
        keys: Color keys, shape $(N, 3)$.
        blend: Blend weight in `[0, 1]`. `1.0` is fully auto-contrasted; `0.0` is the input.
        int_color: If `True`, treat float colors as `[0, 255]` values; otherwise `[0, 1]`.
            `uint8` colors are always treated as `[0, 255]` regardless of the flag; float colors
            above 1 with `int_color=False` raise a ValueError.
        p: Probability of applying the transform.
        dst_keys: Where to store the result.
        generator: Optional `torch.Generator` for reproducibility. See `Transform` for the
            multi-worker caveat.
        allow_missing_keys: If `True`, silently skip absent keys.
    """

    def __init__(
        self,
        keys: KeyCollection,
        blend: float = 0.5,
        int_color: bool = False,
        p: float = 0.2,
        dst_keys: Optional[KeyCollection] = None,
        generator: Optional[torch.Generator] = None,
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.blend = blend
        self.int_color = int_color
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"p must be in [0, 1]; got {p}.")
        self.p = p
        self.dst_keys = ensure_tuple_size(dst_keys or self.keys, len(self.keys))
        self.generator = generator

    def transform(self, data: Dict[str, Any]) -> Dict[str, Any]:
        data = dict(data)
        if torch.rand(1, generator=self.generator).item() >= self.p:
            return data
        for key, dst_key in self.iter_keys(data, self.dst_keys):
            data[dst_key] = F.color_auto_contrast(data[key], blend=self.blend, int_color=self.int_color)
        return data


class SphereCrop(DictTransform):
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
        generator: Optional `torch.Generator` for reproducibility (used when
            `center="random_point"`). See `Transform` for the multi-worker caveat.
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
        generator: Optional[torch.Generator] = None,
        allow_missing_keys: bool = False,
    ) -> None:
        all_keys = ensure_tuple(keys, none_as_empty=True)
        if pos_key not in all_keys:
            all_keys = (pos_key,) + all_keys
        super().__init__(all_keys, allow_missing_keys)
        self.pos_key = pos_key
        self.radius = radius
        self.max_nodes = max_nodes
        self.center = center
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"p must be in [0, 1]; got {p}.")
        self.p = p
        self.generator = generator

    def _resolve_center(self, pos: torch.Tensor) -> torch.Tensor:
        if isinstance(self.center, str):
            if self.center == "centroid":
                return pos.mean(dim=0)
            if self.center == "random_point":
                if pos.shape[0] == 0:
                    return pos.new_zeros(pos.shape[-1])
                idx = int(torch.randint(0, pos.shape[0], (1,), generator=self.generator).item())
                return pos[idx]
            raise ValueError(f"Invalid center: {self.center!r}. Expected 'centroid', 'random_point', or a 3-vector.")
        return torch.as_tensor(self.center, device=pos.device, dtype=pos.dtype)

    def transform(self, data: Dict[str, Any]) -> Dict[str, Any]:
        data = dict(data)
        if self.pos_key not in data:
            if self.allow_missing_keys:
                return data
            raise KeyError(f"`SphereCrop` requires {self.pos_key!r} in data.")
        if torch.rand(1, generator=self.generator).item() >= self.p:
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
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.dst_keys = ensure_tuple_size(dst_keys or self.keys, len(self.keys))
        self.start = start
        self.stop = stop
        self.step = step
        self.dim = dim

    def transform(self, data: Dict[str, Any]) -> Dict[str, Any]:
        data = dict(data)
        sl = slice(self.start, self.stop, self.step)
        for key, dst_key in self.iter_keys(data, self.dst_keys):
            x = data[key]
            index: list[Any] = [slice(None)] * x.ndim
            index[self.dim] = sl
            data[dst_key] = x[tuple(index)]
        return data


class ShufflePoint(DictTransform):
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
        generator: Optional `torch.Generator` for reproducibility. See `Transform` for the
            multi-worker caveat.
        allow_missing_keys: If `True`, silently skip absent keys.
    """

    def __init__(
        self,
        keys: KeyCollection,
        p: float = 1.0,
        generator: Optional[torch.Generator] = None,
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"p must be in [0, 1]; got {p}.")
        self.p = p
        self.generator = generator

    def transform(self, data: Dict[str, Any]) -> Dict[str, Any]:
        data = dict(data)
        if torch.rand(1, generator=self.generator).item() >= self.p:
            return data
        first_key = next(iter(self.iter_keys(data)), None)
        if first_key is None:
            return data
        n = data[first_key].shape[0]
        perm = F.shuffle_indices(n, device=data[first_key].device, generator=self.generator)
        for key in self.iter_keys(data):
            data[key] = data[key][perm]
        return data


class Clamp(DictTransform):
    """Clamp tensor entries to a range (a thin wrapper over `torch.clamp`).

    === "Object"

        ![Clamp on an object](../../assets/transforms/clamp.png)

    === "Scene"

        ![Clamp on a room](../../assets/transforms/clamp_scene.png)

    Args:
        keys: Keys to clamp.
        min: Lower bound. `None` disables the lower clamp.
        max: Upper bound. `None` disables the upper clamp.
        dst_keys: Where to store the result. Defaults to `keys` (in-place overwrite).
        allow_missing_keys: If `True`, silently skip absent keys.
    """

    def __init__(
        self,
        keys: KeyCollection,
        min: Optional[float] = None,
        max: Optional[float] = None,
        dst_keys: Optional[KeyCollection] = None,
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        if min is None and max is None:
            raise ValueError("Clamp requires at least one of `min` or `max`.")
        self.min = min
        self.max = max
        self.dst_keys = ensure_tuple_size(dst_keys or self.keys, len(self.keys))

    def transform(self, data: Dict[str, Any]) -> Dict[str, Any]:
        data = dict(data)
        for key, dst_key in self.iter_keys(data, self.dst_keys):
            data[dst_key] = data[key].clamp(min=self.min, max=self.max)
        return data


class RandomRotateChoice(DictTransform):
    """Rotate one or more keys by an angle chosen uniformly from a discrete list.

    Common use: ModelNet / ScanObjectNN augmentation with `angles=[0, 90, 180, 270]`
    around the z-axis. Sampling is done once per call: every listed key gets
    the same rotation matrix.

    See Also:
        `torch_pointcloud.transforms.functional.rotation_matrix`,
        `torch_pointcloud.transforms.functional.rotate_vectors`

    === "Object"

        ![RandomRotateChoice on an object](../../assets/transforms/random_rotate_choice.png)

    === "Scene"

        ![RandomRotateChoice on a room](../../assets/transforms/random_rotate_choice_scene.png)

    Args:
        keys: Keys to rotate. Each must have shape `(..., 3)`.
        angles: Candidate rotation angles, in **degrees**. Must be non-empty.
        axis: Axis index to rotate around (0=X, 1=Y, 2=Z).
        p: Probability of applying the transform.
        dst_keys: Where to store the rotated tensors. Defaults to `keys` (in-place).
        generator: Optional `torch.Generator` for reproducibility. See `Transform` for the
            multi-worker caveat.
        allow_missing_keys: If `True`, silently skip absent keys.
    """

    def __init__(
        self,
        keys: KeyCollection,
        angles: Sequence[float],
        axis: int = 2,
        p: float = 1.0,
        dst_keys: Optional[KeyCollection] = None,
        generator: Optional[torch.Generator] = None,
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        if len(angles) == 0:
            raise ValueError("RandomRotateChoice requires at least one angle.")
        self.angles = tuple(float(a) for a in angles)
        self.axis = axis
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"p must be in [0, 1]; got {p}.")
        self.p = p
        self.dst_keys = ensure_tuple_size(dst_keys or self.keys, len(self.keys))
        self.generator = generator

    def transform(self, data: Dict[str, Any]) -> Dict[str, Any]:
        data = dict(data)
        if torch.rand(1, generator=self.generator).item() >= self.p:
            return data
        idx = int(torch.randint(0, len(self.angles), (1,), generator=self.generator).item())
        angle_deg = self.angles[idx]
        for key, dst_key in self.iter_keys(data, self.dst_keys):
            x = data[key]
            R = F.rotation_matrix(math.radians(angle_deg), self.axis, device=x.device)
            data[dst_key] = F.rotate_vectors(x, R)
        return data


class RandomColorShift(DictTransform):
    """Additive per-channel color shift sampled uniformly per channel.

    For each of the 3 channels, sample one offset uniformly from `shift_range`
    and add it to every point's value. Sampling is once per call (same shift
    across all listed keys). Result is clamped to the valid color range.

    ![RandomColorShift before / after](../../assets/transforms/color_shift.png)

    See Also:
        `torch_pointcloud.transforms.functional.color_shift`

    Args:
        keys: Color keys to shift, shape $(N, 3)$.
        shift_range: Min and max per-channel offset (in the same range as the colors).
        int_color: If `True`, treat float colors as `[0, 255]` values; otherwise `[0, 1]`.
            `uint8` colors are always treated as `[0, 255]` regardless of the flag; float colors
            above 1 with `int_color=False` raise a ValueError.
        p: Probability of applying the transform.
        dst_keys: Where to store the shifted tensors.
        generator: Optional `torch.Generator` for reproducibility. See `Transform` for the
            multi-worker caveat.
        allow_missing_keys: If `True`, silently skip absent keys.
    """

    def __init__(
        self,
        keys: KeyCollection,
        shift_range: Tuple[float, float] = (-0.05, 0.05),
        int_color: bool = False,
        p: float = 1.0,
        dst_keys: Optional[KeyCollection] = None,
        generator: Optional[torch.Generator] = None,
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.shift_range = shift_range
        self.int_color = int_color
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"p must be in [0, 1]; got {p}.")
        self.p = p
        self.dst_keys = ensure_tuple_size(dst_keys or self.keys, len(self.keys))
        self.generator = generator

    def transform(self, data: Dict[str, Any]) -> Dict[str, Any]:
        data = dict(data)
        if torch.rand(1, generator=self.generator).item() >= self.p:
            return data
        lo, hi = self.shift_range
        shift = torch.empty(3).uniform_(lo, hi, generator=self.generator)
        for key, dst_key in self.iter_keys(data, self.dst_keys):
            data[dst_key] = F.color_shift(data[key], shift, int_color=self.int_color)
        return data


class RandomElasticDistortion(DictTransform):
    """Apply a smooth random displacement field (elastic distortion).

    Used in sparse-voxel indoor segmentation recipes. Sampling is done once
    per call so multi-key consistency is preserved (the same displacement
    field is applied to every listed key).

    For multi-scale distortion (the common default), compose two
    `RandomElasticDistortion` calls with different `granularity` / `magnitude`.

    === "Object"

        ![RandomElasticDistortion on an object](../../assets/transforms/random_elastic_distortion.png)

    === "Scene"

        ![RandomElasticDistortion on a room](../../assets/transforms/random_elastic_distortion_scene.png)

    See Also:
        `torch_pointcloud.transforms.functional.random_elastic_distortion`

    Args:
        keys: Position keys to distort, shape $(N, 3)$. All listed keys must
            share the same leading dimension $N$: the per-point displacement is
            computed once from the first present key and added to every key.
        granularity: Size of the displacement-field grid cells. Smaller values
            give higher-frequency distortion.
        magnitude: Standard deviation of the per-cell Gaussian noise. Larger
            values give stronger deformation.
        p: Probability of applying the transform.
        dst_keys: Where to store the distorted tensors.
        generator: Optional `torch.Generator` for reproducibility. See `Transform` for the
            multi-worker caveat.
        allow_missing_keys: If `True`, silently skip absent keys.
    """

    def __init__(
        self,
        keys: KeyCollection,
        granularity: float = 0.2,
        magnitude: float = 0.4,
        p: float = 1.0,
        dst_keys: Optional[KeyCollection] = None,
        generator: Optional[torch.Generator] = None,
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.granularity = granularity
        self.magnitude = magnitude
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"p must be in [0, 1]; got {p}.")
        self.p = p
        self.dst_keys = ensure_tuple_size(dst_keys or self.keys, len(self.keys))
        self.generator = generator

    def transform(self, data: Dict[str, Any]) -> Dict[str, Any]:
        data = dict(data)
        if torch.rand(1, generator=self.generator).item() >= self.p:
            return data
        first_key = next(iter(self.iter_keys(data)), None)
        if first_key is None:
            return data
        reference = data[first_key]
        displacement = (
            F.random_elastic_distortion(reference, self.granularity, self.magnitude, generator=self.generator)
            - reference
        )
        for key, dst_key in self.iter_keys(data, self.dst_keys):
            x = data[key]
            data[dst_key] = x + displacement.to(x.dtype).to(x.device)
        return data


class InstanceToBox(DictTransform):
    r"""Axis-aligned bounding boxes from per-point instance ids (e.g. ScanNet detection targets).

    Each distinct non-negative instance id in `instance_key` becomes one axis-aligned box covering its
    `pos_key` points: the center and full extents with heading $0$, written to `dst_box_key` as $(K, 7)$
    rows $[c_x, c_y, c_z, d_x, d_y, d_z, 0]$. The box class (the instance's most common `semantic_key`
    value) is written separately to `dst_class_key` as a $(K,)$ long tensor. Negative instance ids mark
    unlabeled points and never form a box. Instances whose class equals `ignore_index` are dropped, so
    mapping the stuff / non-target semantics to `ignore_index` with a `Relabel` upstream filters the boxes
    down to the detection classes.

    ![InstanceToBox before / after](../../assets/transforms/instance_to_box.png)

    Args:
        instance_key: Key of the $(N,)$ per-point instance ids.
        semantic_key: Key of the $(N,)$ per-point class labels the box class is read from.
        pos_key: Key of the $(N, 3)$ coordinates.
        dst_box_key: Key to write the $(K, 7)$ boxes to.
        dst_class_key: Key to write the $(K,)$ per-box classes to.
        ignore_index: Class value whose instances are dropped (e.g. unlabeled / stuff).
        allow_missing_keys: If `True`, return the data unchanged when an input key is missing instead of raising.
    """

    def __init__(
        self,
        instance_key: str = "instance",
        semantic_key: str = "segment",
        pos_key: str = "pos",
        dst_box_key: str = "box",
        dst_class_key: str = "label",
        ignore_index: int = -1,
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__([instance_key, semantic_key, pos_key], allow_missing_keys)
        self.instance_key = instance_key
        self.semantic_key = semantic_key
        self.pos_key = pos_key
        self.dst_box_key = dst_box_key
        self.dst_class_key = dst_class_key
        self.ignore_index = ignore_index

    def transform(self, data: Dict[str, Any]) -> Dict[str, Any]:
        d = dict(data)
        for key in (self.instance_key, self.semantic_key, self.pos_key):
            if key not in d:
                if self.allow_missing_keys:
                    return d
                raise KeyError(f"Key {key!r} was missing in the data and `allow_missing_keys==False`.")

        boxes, classes = self._instance_boxes(d[self.pos_key], d[self.instance_key], d[self.semantic_key])
        d[self.dst_box_key] = boxes
        d[self.dst_class_key] = classes
        return d

    def _instance_boxes(self, pos: Tensor, instance: Tensor, segment: Tensor) -> Tuple[Tensor, Tensor]:
        boxes: list[Tensor] = []
        classes: list[Tensor] = []
        for inst in torch.unique(instance):
            if int(inst) < 0:
                continue
            mask = instance == inst
            cls = segment[mask].mode().values
            if int(cls) == self.ignore_index:
                continue
            lo, hi = pos[mask].amin(dim=0), pos[mask].amax(dim=0)
            boxes.append(torch.cat([(lo + hi) / 2, hi - lo, pos.new_zeros(1)]))
            classes.append(cls.long())

        if not boxes:
            return pos.new_zeros((0, 7)), torch.zeros(0, dtype=torch.long, device=pos.device)
        return torch.stack(boxes), torch.stack(classes)


class GenerateVoteLabels(DictTransform):
    r"""Generate per-point vote offsets and a vote mask from oriented GT boxes.

    Each point collects the offsets to the centers of the first `gt_vote_factor` boxes containing it, in box
    order, matching the VoteNet ScanNet and SUN RGB-D vote layout: a point inside fewer boxes repeats its
    first offset in the unfilled slots, so the min-over-votes loss can credit either center on overlapping
    objects. Points inside no box receive zero offsets and a zero mask. Boxes are $(K, 7)$ rows
    $[c_x, c_y, c_z, d_x, d_y, d_z, \theta]$ with full extents and heading in radians counterclockwise about
    $+z$. When `oriented` is `True` containment is yaw-aware, otherwise an axis-aligned test is used.

    See Also:
        `torch_pointcloud.transforms.functional.points_in_oriented_box`

    ![GenerateVoteLabels before / after](../../assets/transforms/generate_vote_labels.png)

    Args:
        pos_key: Key of the $(N, 3)$ coordinate tensor.
        box_key: Key of the $(K, 7)$ box tensor (full extents, counterclockwise heading).
        vote_key: Key to write the $(N, 3 G)$ vote offsets to.
        mask_key: Key to write the $(N,)$ vote mask to.
        oriented: If `True`, use yaw-aware containment, otherwise an axis-aligned test.
        gt_vote_factor: Number $G$ of vote slots per point.
        allow_missing_keys: If `True`, return the data unchanged when `pos_key` or `box_key` is missing
            instead of raising.
    """

    def __init__(
        self,
        pos_key: str = "pos",
        box_key: str = "box",
        vote_key: str = "vote_label",
        mask_key: str = "vote_label_mask",
        oriented: bool = True,
        gt_vote_factor: int = 3,
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__([pos_key, box_key], allow_missing_keys)
        self.pos_key = pos_key
        self.box_key = box_key
        self.vote_key = vote_key
        self.mask_key = mask_key
        self.oriented = oriented
        self.gt_vote_factor = gt_vote_factor

    def transform(self, data: Dict[str, Any]) -> Dict[str, Any]:
        d = dict(data)
        if self.pos_key not in d or self.box_key not in d:
            if self.allow_missing_keys:
                return d

            missing = self.pos_key if self.pos_key not in d else self.box_key
            raise KeyError(f"Key {missing!r} was missing in the data and `allow_missing_keys==False`.")

        pos = d[self.pos_key]
        boxes = d[self.box_key]
        n = pos.shape[0]
        num_slots = self.gt_vote_factor
        votes = torch.zeros(n, 3 * num_slots, device=pos.device, dtype=pos.dtype)
        mask = torch.zeros(n, device=pos.device, dtype=torch.long)
        counts = torch.zeros(n, device=pos.device, dtype=torch.long)

        for k in range(boxes.shape[0]):
            box = boxes[k]
            if self.oriented:
                half_box = torch.cat([box[0:3], box[3:6] / 2, box[6:7]])
                inside = F.points_in_oriented_box(pos, half_box)
            else:
                inside = ((pos - box[0:3]).abs() <= box[3:6] / 2).all(dim=1)
            mask[inside] = 1
            idx = (inside & (counts < num_slots)).nonzero(as_tuple=True)[0]
            offsets = box[0:3] - pos[idx]
            first = counts[idx] == 0
            votes[idx[first]] = offsets[first].repeat(1, num_slots)
            rest = idx[~first]
            cols = counts[rest, None] * 3 + torch.arange(3, device=pos.device)
            votes[rest[:, None], cols] = offsets[~first]
            counts[inside] += 1

        d[self.vote_key] = votes
        d[self.mask_key] = mask
        return d


class EncodeVoteNetTargets(DictTransform):
    r"""Encode oriented GT boxes into the padded label tensors the VoteNet loss consumes.

    Each $(K, 7)$ box row $[c_x, c_y, c_z, d_x, d_y, d_z, \theta]$ (full extents) and its class from
    `class_key` are converted to fixed-size $(M, \ldots)$ targets where $M$ is `max_num_obj`. Headings are
    binned with `angle_to_class`. The size class is the semantic class and the size residual is computed
    against `mean_sizes` (full edge lengths).

    See Also:
        `torch_pointcloud.transforms.functional.angle_to_class`,
        `torch_pointcloud.transforms.functional.class_to_size`

    ![EncodeVoteNetTargets before / after](../../assets/transforms/encode_votenet_targets.png)

    Args:
        box_key: Key of the $(K, 7)$ box tensor (full extents).
        class_key: Key of the $(K,)$ per-box class tensor.
        center_key: Key to write the $(M, 3)$ center labels to.
        heading_class_key: Key to write the $(M,)$ heading class labels to.
        heading_residual_key: Key to write the $(M,)$ heading residual labels to.
        size_class_key: Key to write the $(M,)$ size class labels to.
        size_residual_key: Key to write the $(M, 3)$ size residual labels to.
        sem_cls_key: Key to write the $(M,)$ semantic class labels to.
        box_mask_key: Key to write the $(M,)$ box mask to.
        num_heading_bin: Number of heading bins.
        mean_sizes: Template sizes of shape $(C, 3)$ holding full edge lengths per class.
        max_num_obj: Padded number of objects $M$.
        allow_missing_keys: If `True`, return the data unchanged when `box_key` or `class_key` is missing
            instead of raising.

    Raises:
        ValueError: If `mean_sizes` is not provided.
    """

    def __init__(
        self,
        box_key: str = "box",
        class_key: str = "label",
        center_key: str = "center_label",
        heading_class_key: str = "heading_class_label",
        heading_residual_key: str = "heading_residual_label",
        size_class_key: str = "size_class_label",
        size_residual_key: str = "size_residual_label",
        sem_cls_key: str = "sem_cls_label",
        box_mask_key: str = "box_label_mask",
        num_heading_bin: int = 12,
        mean_sizes: Optional[Union[Tensor, Sequence[Sequence[float]]]] = None,
        max_num_obj: int = 64,
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__([box_key, class_key], allow_missing_keys)
        if mean_sizes is None:
            raise ValueError("mean_sizes must be provided with full edge lengths of shape (C, 3).")

        self.box_key = box_key
        self.class_key = class_key
        self.center_key = center_key
        self.heading_class_key = heading_class_key
        self.heading_residual_key = heading_residual_key
        self.size_class_key = size_class_key
        self.size_residual_key = size_residual_key
        self.sem_cls_key = sem_cls_key
        self.box_mask_key = box_mask_key
        self.num_heading_bin = num_heading_bin
        self.mean_sizes = torch.as_tensor(mean_sizes, dtype=torch.float32)
        self.max_num_obj = max_num_obj

    def transform(self, data: Dict[str, Any]) -> Dict[str, Any]:
        d = dict(data)
        for key in (self.box_key, self.class_key):
            if key not in d:
                if self.allow_missing_keys:
                    return d
                raise KeyError(f"Key {key!r} was missing in the data and `allow_missing_keys==False`.")

        boxes = d[self.box_key]
        classes = d[self.class_key]
        device = boxes.device
        dtype = boxes.dtype
        m = self.max_num_obj
        k = min(boxes.shape[0], m)
        mean_sizes = self.mean_sizes.to(device=device, dtype=dtype)

        center = torch.zeros(m, 3, device=device, dtype=dtype)
        heading_class = torch.zeros(m, device=device, dtype=torch.long)
        heading_residual = torch.zeros(m, device=device, dtype=dtype)
        size_class = torch.zeros(m, device=device, dtype=torch.long)
        size_residual = torch.zeros(m, 3, device=device, dtype=dtype)
        sem_cls = torch.zeros(m, device=device, dtype=torch.long)
        box_mask = torch.zeros(m, device=device, dtype=dtype)

        if k > 0:
            valid = boxes[:k]
            sem = classes[:k].long()
            center[:k] = valid[:, 0:3]
            cls, residual = F.angle_to_class(valid[:, 6], self.num_heading_bin)
            heading_class[:k] = cls
            heading_residual[:k] = residual
            size_class[:k] = sem
            size_residual[:k] = valid[:, 3:6] - mean_sizes[sem]
            sem_cls[:k] = sem
            box_mask[:k] = 1

        d[self.center_key] = center
        d[self.heading_class_key] = heading_class
        d[self.heading_residual_key] = heading_residual
        d[self.size_class_key] = size_class
        d[self.size_residual_key] = size_residual
        d[self.sem_cls_key] = sem_cls
        d[self.box_mask_key] = box_mask
        return d


class Mix3D(Transform):
    r"""Concatenate two scenes into one, offsetting the second scene's instance ids.

    :arxiv: [Mix3D: Out-of-Context Data Augmentation for 3D Scenes](https://arxiv.org/abs/2110.02210)

    Every point-aligned key in `keys` is concatenated along the point dimension, so the mixed scene
    holds all points of both inputs. When `instance_key` is present in both scenes, the second
    scene's instance ids are shifted past the first scene's maximum id so the merged instances stay
    disjoint; points labelled `ignore_index` keep that label and are excluded from the offset.

    Unlike the other pairwise mixes, `Mix3D` keeps all points of both scenes, so the mixed scene has
    roughly twice as many points as either input.

    ![Mix3D before / after](../../assets/transforms/mix3d.png)

    Args:
        keys: Point-aligned keys concatenated jointly (e.g. `pos`, `color`, `normal`, `segment`).
        instance_key: Key of per-point instance ids to offset, or `None` to skip instance handling.
        ignore_index: Instance id treated as "no instance" (kept as-is, ignored by the offset).
        p: Probability of applying the mix; below it the first scene is returned unchanged.
        generator: Optional `torch.Generator` for the probability draw. See `Transform` for the
            multi-worker caveat.

    Shape:
        - each key in `keys`: $(N, \ldots)$ and $(M, \ldots)$ inputs, $(N + M, \ldots)$ output.

    Example:
        ```python
        import torch
        import torch_pointcloud.transforms as T

        a = {"pos": torch.randn(100, 3), "segment": torch.randint(0, 10, (100,))}
        b = {"pos": torch.randn(120, 3), "segment": torch.randint(0, 10, (120,))}
        mix = T.Mix3D(keys=("pos", "segment"), instance_key=None)
        out = mix(a, b)
        print(out["pos"].shape)
        ```
    """

    def __init__(
        self,
        keys: KeyCollection,
        instance_key: Optional[str] = "instance",
        ignore_index: int = -1,
        p: float = 1.0,
        generator: Optional[torch.Generator] = None,
    ) -> None:
        self.keys = ensure_tuple(keys)
        self.instance_key = instance_key
        self.ignore_index = ignore_index
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"p must be in [0, 1]; got {p}.")
        self.p = p
        self.generator = generator

    def _merge_instances(self, instance: Tensor, other_instance: Tensor) -> Tensor:
        valid = instance != self.ignore_index
        offset = int(instance[valid].max()) + 1 if valid.any() else 0
        other = other_instance.clone()
        other_valid = other != self.ignore_index
        other[other_valid] = other[other_valid] + offset
        return torch.cat([instance, other], dim=0)

    def transform(self, data: Dict[str, Any], other: Dict[str, Any]) -> Dict[str, Any]:
        d = dict(data)
        if torch.rand(1, generator=self.generator).item() >= self.p:
            return d
        for key in self.keys:
            d[key] = torch.cat([data[key], other[key]], dim=0)
        ik = self.instance_key
        if ik is not None and ik in data and ik in other:
            d[ik] = self._merge_instances(data[ik], other[ik])
        return d


class LaserMix(Transform):
    r"""Mix two LiDAR scans by swapping alternating inclination (pitch) bands.

    :arxiv: [LaserMix for Semi-Supervised LiDAR Semantic Segmentation](https://arxiv.org/abs/2207.00026)

    Both scans are partitioned into `num_areas` inclination bands (one count is drawn per call), and
    alternating bands are taken from each scan so the mixed scene tiles the full field of view. Every
    key in `keys` is masked with the same per-scan selection, keeping per-point correspondence.

    See Also:
        `torch_pointcloud.transforms.functional.laser_mix_masks`

    ![LaserMix before / after](../../assets/transforms/laser_mix.png)

    Args:
        keys: Point-aligned keys masked jointly (must include `pos_key`).
        num_areas: Candidate band counts; one is sampled uniformly per call.
        pitch_range: Inclination range `(min, max)` in degrees.
        pos_key: Key of the coordinates used to compute inclination bands.
        p: Probability of applying the mix; below it the first scene is returned unchanged.
        generator: Optional `torch.Generator` for the band count, parity, and probability draws.
            See `Transform` for the multi-worker caveat.

    Shape:
        - each key in `keys`: $(N, \ldots)$ and $(M, \ldots)$ inputs, $(N' + M', \ldots)$ output.

    Example:
        ```python
        import torch
        import torch_pointcloud.transforms as T

        a = {"pos": torch.randn(100, 3), "segment": torch.randint(0, 10, (100,))}
        b = {"pos": torch.randn(120, 3), "segment": torch.randint(0, 10, (120,))}
        mix = T.LaserMix(keys=("pos", "segment"), num_areas=(3, 4, 5, 6), pitch_range=(-25.0, 3.0))
        out = mix(a, b)
        ```
    """

    def __init__(
        self,
        keys: KeyCollection,
        num_areas: Sequence[int],
        pitch_range: Tuple[float, float],
        pos_key: str = "pos",
        p: float = 1.0,
        generator: Optional[torch.Generator] = None,
    ) -> None:
        self.keys = ensure_tuple(keys)
        self.num_areas = tuple(num_areas)
        self.pitch_range = pitch_range
        self.pos_key = pos_key
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"p must be in [0, 1]; got {p}.")
        self.p = p
        self.generator = generator

    def transform(self, data: Dict[str, Any], other: Dict[str, Any]) -> Dict[str, Any]:
        d = dict(data)
        if torch.rand(1, generator=self.generator).item() >= self.p:
            return d
        index = int(torch.randint(len(self.num_areas), (1,), generator=self.generator).item())
        num_areas = self.num_areas[index]
        mask, other_mask = F.laser_mix_masks(
            data[self.pos_key], other[self.pos_key], num_areas, self.pitch_range, generator=self.generator
        )
        for key in self.keys:
            d[key] = torch.cat([data[key][mask], other[key][other_mask]], dim=0)
        return d


class PolarMix(Transform):
    r"""Mix two LiDAR scans by swapping an azimuth sector and rotate-pasting instance-class points.

    :arxiv: [PolarMix: A General Data Augmentation Technique for LiDAR Point Clouds](https://arxiv.org/abs/2208.00223)

    Two independent sub-augmentations run per call. With probability `swap_ratio`, a random azimuth
    half-sector of the first scan is replaced by the same sector of the second scan. With probability
    `rotate_paste_ratio`, points of the second scan whose `segment_key` label is in `instance_classes`
    are rotated by a random angle about the up axis and appended. Only `pos_key` is rotated for the
    pasted points; the other keys are copied unchanged.

    See Also:
        `torch_pointcloud.transforms.functional.polar_mix_masks`

    ![PolarMix before / after](../../assets/transforms/polar_mix.png)

    Args:
        keys: Point-aligned keys masked and concatenated jointly (must include `pos_key`).
        instance_classes: Semantic labels whose points are rotate-pasted from the second scan.
        swap_ratio: Probability of swapping the azimuth sector.
        rotate_paste_ratio: Probability of rotate-pasting the instance-class points.
        pos_key: Key of the coordinates used to compute azimuth sectors and to rotate pasted points.
        segment_key: Key of per-point semantic labels used to select the instance classes.
        p: Probability of applying the mix; below it the first scene is returned unchanged.
        generator: Optional `torch.Generator` for the sector, rotation, and probability draws.
            See `Transform` for the multi-worker caveat.

    Shape:
        - each key in `keys`: $(N, \ldots)$ and $(M, \ldots)$ inputs, $(K, \ldots)$ output.

    Example:
        ```python
        import torch
        import torch_pointcloud.transforms as T

        a = {"pos": torch.randn(100, 3), "segment": torch.randint(0, 10, (100,))}
        b = {"pos": torch.randn(120, 3), "segment": torch.randint(0, 10, (120,))}
        mix = T.PolarMix(keys=("pos", "segment"), instance_classes=(1, 2, 3))
        out = mix(a, b)
        ```
    """

    def __init__(
        self,
        keys: KeyCollection,
        instance_classes: Sequence[int],
        swap_ratio: float = 0.5,
        rotate_paste_ratio: float = 1.0,
        pos_key: str = "pos",
        segment_key: str = "segment",
        p: float = 1.0,
        generator: Optional[torch.Generator] = None,
    ) -> None:
        self.keys = ensure_tuple(keys)
        self.instance_classes = tuple(instance_classes)
        self.swap_ratio = swap_ratio
        self.rotate_paste_ratio = rotate_paste_ratio
        self.pos_key = pos_key
        self.segment_key = segment_key
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"p must be in [0, 1]; got {p}.")
        self.p = p
        self.generator = generator

    def transform(self, data: Dict[str, Any], other: Dict[str, Any]) -> Dict[str, Any]:
        d = dict(data)
        if torch.rand(1, generator=self.generator).item() >= self.p:
            return d
        if torch.rand(1, generator=self.generator).item() < self.swap_ratio:
            mask, other_mask = F.polar_mix_masks(data[self.pos_key], other[self.pos_key], generator=self.generator)
            for key in self.keys:
                d[key] = torch.cat([data[key][mask], other[key][other_mask]], dim=0)
        if torch.rand(1, generator=self.generator).item() < self.rotate_paste_ratio:
            segment = other[self.segment_key]
            paste = torch.isin(segment, segment.new_tensor(self.instance_classes))
            angle = torch.empty(1).uniform_(-math.pi, math.pi, generator=self.generator).item()
            rotation = F.rotation_matrix(angle, axis=2)
            for key in self.keys:
                pasted = other[key][paste]
                if key == self.pos_key:
                    pasted = F.rotate_vectors(pasted, rotation)
                d[key] = torch.cat([d[key], pasted], dim=0)
        return d
