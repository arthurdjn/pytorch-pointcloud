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
from typing import TYPE_CHECKING, Any, Dict, Generator, Iterable, Literal, Optional, Sequence, Tuple, get_args

import numpy as np
import torch

from torch_pointcloud.utils.conversion import ensure_tuple, ensure_tuple_size
from torch_pointcloud.utils.imports import optional_import
from torch_pointcloud.utils.octree import build_octree
from torch_pointcloud.utils.ops import consecutive_cluster, voxel_grid, voxel_grid_fnv
from torch_pointcloud.utils.types import KeyCollection, ValueCollection

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

scatter, _ = optional_import("torch_scatter", name="scatter")


__all__ = [
    "Abs",
    "AlignAxis",
    "ApplyMask",
    "AxisMinOffset",
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
    "FarthestPointSample",
    "KeepItems",
    "Normalize",
    "OneHot",
    "OnesLike",
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
    "RemoveNearOrigin",
    "RenameItems",
    "Rescale",
    "RescaleMethod",
    "Scale",
    "SetValue",
    "Shift",
    "ShiftMethod",
    "ShufflePoint",
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

    See Also:
        `torch_pointcloud.transforms.functional.random_sample`

    Args:
        keys: The keys to sample from.
        num_samples: The number of values to sample.
        replace: If `True`, sample with replacement (duplicates allowed). If `False`
            (default), raise `ValueError` when `num_samples > N`.
        generator: The generator for the random number generator.
        allow_missing_keys: If `True`, the transform will not raise an error if the keys are not present in the data.

    Raises:
        ValueError: If `replace=False` and `num_samples > N` for the first sampled key,
            or if the first sampled tensor is empty and `num_samples > 0`.
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


class RandomSampleFaceVertices(DictTransform):
    """Randomly sample a fixed number of vertices from a 3D mesh stored in a dictionary.

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


class FarthestPointSample(DictTransform):
    """Farthest-point sampling (FPS) of a dictionary entry.

    Iteratively picks the point that maximizes the minimum distance to the
    already-selected set, producing a well-distributed subset. Matches the FPS
    convention used by PointNet++, PointNeXt, KPConv, and others.

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
        _, mask = F.remove_near_origin(d[self.pos_key], radius=self.radius, return_mask=True)
        for key in self.iter_keys(d):
            d[key] = d[key][mask]
        return d


class Abs(DictTransform):
    """Make dictionary tensor entries absolute.

    See Also:
        `torch_pointcloud.transforms.functional.abs`

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

    Membership condition along `dim`:

    $$
    \text{bbmin}_j < x_j < \text{bbmax}_j \quad \forall j
    $$

    where `bbox = (*bbmin, *bbmax)` is the AABB.

    Sibling masks:

    - `CubeMask` - L∞ ball (center + radius)
    - `SphereMask` - L2 ball (center + radius)

    See Also:
        `torch_pointcloud.transforms.functional.box_mask`

    Args:
        keys: The keys to create the mask for.
        bbox: The bounding box used to mask input tensors, as `(*bbmin, *bbmax)`.
        dst_keys: The keys to store the mask in.
        dim: The dimension to create the mask over.
        strict: Whether to use strict inequality.
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
            data[dst_key] = F.box_mask(data[key], self.bbox, dim=self.dim)
        return data


class ApplyMask(DictTransform):
    """Apply a mask stored in a dictionary to other dictionary entries.

    See Also:
        `torch_pointcloud.transforms.functional.apply_mask`

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

    Args:
        keys: The keys to shift.
        method: `"bbox"` (midrange), `"centroid"` (mean), or `"min"` (shift to origin).
        dim: The dimension to reduce over.
        axes: Which axes (last-dim indices) to shift. `None` (default) shifts every
            axis; pass e.g. `axes=[0, 1]` to recenter only XY (matching Open3D-ML's
            `recenter: dim: [0, 1]` augmentation). Axes outside this list are
            left unchanged - this is the composable knob for mixed-method shifts.
        dst_keys: The keys to store the shifted data in.
        allow_missing_keys: If `True`, skip missing keys silently.

    Example:
        Pointcept-style centering - XY shifted by the bbox midpoint, Z shifted
        by its minimum (equivalent to the old `CenterShift(apply_z=True)`):

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


class OctreeFeatures(DictTransform):
    """Extract per-node features from an octree via `octree.get_input_feature`.

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


class RenameItems(DictTransform):
    """Rename keys in the dictionary.

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

    Computes `data[key] = data[key] - data[sub_key]` for each key.

    Args:
        keys: Keys whose tensors are modified (subtracted from).
        sub_keys: Keys whose values are subtracted from each target key.
        dst_keys: Where to store results. Defaults to `keys`.
        allow_missing_keys: If `True`, silently skip absent target keys.
    """

    def __init__(
        self,
        keys: KeyCollection,
        sub_keys: KeyCollection,
        dst_keys: Optional[KeyCollection] = None,
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.sub_keys = ensure_tuple_size(sub_keys, len(self.keys))
        self.dst_keys = ensure_tuple_size(dst_keys or self.keys, len(self.keys))

    def transform(self, data: Dict[str, Any]) -> Dict[str, Any]:
        data = dict(data)
        for key, sub_key, dst_key in self.iter_keys(data, self.sub_keys, self.dst_keys):
            data[dst_key] = data[key] - data[sub_key]
        return data


class DivideKey(DictTransform):
    """Divide target keys by the value of a reference key element-wise.

    Computes `data[key] = data[key] / data[div_key]` for each key.

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
    """Voxelize a point cloud by grid-binning and per-voxel reduction.

    Sub-samples a point cloud to one representative point per occupied voxel,
    and optionally preserves the inverse cluster mapping for full-resolution
    back-projection.

    Operates on a single sample (pre-collate); equivalent to Pointcept's
    `GridSample` for the convention. With `cluster_key` set,
    `data[pos_key][cluster[i]]` is the voxel-mean position of original point
    `i` - handy when the model evaluates at sub-resolution but mIoU is reported
    at full resolution.

    Args:
        pos_key: Key holding the positions to sub-sample.
        pos_reduce: How to reduce positions per voxel (`mean`/`min`/`max`/`sum`/`first`/`grid`).
        size: Voxel edge length in the same units as the positions.
        method: Voxel-id hashing scheme (`fnv` matches Pointcept; `pyg` is the default).
        reduce: Per-key reduction for `keys` (defaults to `mean` if `None`).
        keys: Additional per-point keys to sub-sample (e.g. `color`, `segment`).
        cluster_key: When set, store the inverse cluster mapping shaped $(N_\text{full},)$
            under this key.
        grid_pos_key: When set together with a non-`grid` `pos_reduce`, also store
            the integer voxel-grid coordinates under this key. Useful when a model
            needs both real-valued positions (e.g. for rotary position embedding)
            and integer grid coordinates (for serialization / sparse-conv stems).
        allow_missing_keys: If `True`, missing keys are skipped silently.
    """

    def __init__(
        self,
        pos_key: str,
        pos_reduce: VoxelPosReduce,
        size: float,
        method: VoxelMethod = "pyg",
        reduce: Optional[ValueCollection[VoxelReduce]] = None,
        keys: Optional[KeyCollection] = None,
        cluster_key: Optional[str] = None,
        grid_pos_key: Optional[str] = None,
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.pos_key = pos_key
        self.pos_reduce = pos_reduce
        self.size = size
        self.reduce = ensure_tuple_size(reduce, len(self.keys))
        self.method = method
        self.cluster_key = cluster_key
        self.grid_pos_key = grid_pos_key

    def _reduce(
        self,
        tensor: torch.Tensor,
        reduce: str,
        cluster: torch.Tensor,
        perm: torch.Tensor,
    ) -> torch.Tensor:
        if reduce == "first":
            return tensor[perm]

        # NOTE: Tensor is automatically converted to float before reduction.
        return scatter(tensor.float(), cluster, dim=0, reduce=reduce)

    def transform(self, data: Dict[str, Any]) -> Dict[str, Any]:
        data = dict(data)
        pos = data[self.pos_key]

        if pos.shape[0] == 0:
            if self.cluster_key is not None:
                data[self.cluster_key] = torch.empty(0, dtype=torch.long, device=pos.device)
            if self.grid_pos_key is not None:
                data[self.grid_pos_key] = torch.empty(0, pos.shape[-1], dtype=torch.long, device=pos.device)
            return data

        start = torch.floor(pos.min(dim=0).values / self.size) * self.size

        if self.method == "fnv":
            # This method is supported only for debugging and reproducibility, such that it behaves and produces
            # the same output as Pointcept grid subsampling.
            # This method might be removed in the future (?)
            cluster = voxel_grid_fnv(pos, size=self.size, start=start)
        else:
            cluster = voxel_grid(pos, size=self.size, start=start)

        cluster, perm = consecutive_cluster(cluster, return_permutation=True)

        if self.pos_reduce == "grid":
            pos_grid = torch.floor((pos[perm] - start) / self.size).long()
            data[self.pos_key] = pos_grid - pos_grid.min(dim=0).values
        else:
            data[self.pos_key] = self._reduce(pos, self.pos_reduce, cluster, perm)
            if self.grid_pos_key is not None:
                pos_grid = torch.floor((pos[perm] - start) / self.size).long()
                data[self.grid_pos_key] = pos_grid - pos_grid.min(dim=0).values

        for key, reduce in self.iter_keys(data, self.reduce):
            data[key] = self._reduce(data[key], reduce, cluster, perm)

        if self.cluster_key is not None:
            data[self.cluster_key] = cluster

        return data


class OnesLike(DictTransform):
    """Adds a tensor of ones shaped like existing dictionary entries.

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
    r"""Per-point offset from the minimum along a chosen coordinate axis.

    For each point and a given axis $a$ along tensor dimension $d$, computes:

    $$
    o_i = p_{i,a} - \min_j p_{j,a}
    $$

    The result has the same shape as the input with the coordinate dimension
    reduced to size 1 (e.g. $(N, 3) \to (N, 1)$ or $(B, N, 3) \to (B, N, 1)$).
    For batched inputs, the minimum is computed per-sample.

    Args:
        keys: Keys holding point positions of shape $(N, D)$.
        axis: Coordinate axis $a$ along which to compute the offset.
        dst_keys: Keys under which the offset tensors are stored. Defaults to `keys`
            (in-place overwrite).
        allow_missing_keys: If True, skip missing keys instead of raising.

    Example:
        Let's say you have a point cloud with positions `(N, 3)` in XYZ order
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

        Now, the data dictionary will contain the key `pos_offset` with the shape `(N, 1)`.
    """

    def __init__(
        self,
        keys: KeyCollection,
        axis: ValueCollection[int],
        dst_keys: KeyCollection | None = None,
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.dst_keys = ensure_tuple_size(dst_keys or self.keys, len(self.keys))
        self.axis = ensure_tuple_size(axis, len(self.keys))

    def transform(self, data: Dict[str, Any]) -> Dict[str, Any]:
        data = dict(data)
        for key, dst_key, axis in self.iter_keys(data, self.dst_keys, self.axis):
            data[dst_key] = F.axis_min_offset(data[key], axis=axis)
        return data


class Cat(DictTransform):
    """Concatenates tensors from multiple keys into a single feature tensor.

    Note:
        This transform is mostly used to concatenate multiple features into a single tensor to feed into your model.

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

        Now, the data dictionary will contain the key `x` with the shape `(10, 9)`.
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
        tensors = [data[key].float() for key in self.iter_keys(data)]
        data[self.dst_key] = torch.cat(tensors, dim=self.dim)
        return data


class OneHot(DictTransform):
    r"""One-hot encode integer-class tensors.

    Wraps `torch.nn.functional.one_hot` and casts the result to float so the
    output is ready to feed into a model.

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

    Args:
        keys: Keys to reduce.
        op: Reduction operator: `"min"`, `"max"`, `"mean"`, or `"sum"` (matches the
            vocabulary used by `Voxelize`).
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
                data[dst_key] = x.float().mean(dim=dim, keepdim=self.keepdim)
            else:
                data[dst_key] = self._OP_FUNCS[op](x, dim=dim, keepdim=self.keepdim)
        return data


class KeepItems(DictTransform):
    r"""Keep only items in the data dictionary that are in the keys list.

    Note:
        This transform is useful if during augmentation process you constructed multiple tensors and want
        to drop intermediate tensors for memory efficiency.

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
    """Rotate one or more keys by a uniformly random angle around an axis.

    Sampling is done once per call: all listed keys get the same rotation
    matrix. Pair `keys=("pos", "normal")` to keep positions and normals
    consistent.

    See Also:
        `torch_pointcloud.transforms.functional.random_rotate`,
        `torch_pointcloud.transforms.functional.rotation_matrix`

    Args:
        keys: Keys to rotate. Each must have shape `(..., 3)`.
        angle_range: Min and max rotation angle, in **degrees**.
        axis: Axis index to rotate around (0=X, 1=Y, 2=Z).
        p: Probability of applying the transform.
        dst_keys: Where to store the rotated tensors. Defaults to `keys` (in-place).
        generator: Optional `torch.Generator` for reproducibility.
        allow_missing_keys: If `True`, silently skip absent keys.
    """

    def __init__(
        self,
        keys: KeyCollection,
        angle_range: Tuple[float, float] = (-180.0, 180.0),
        axis: int = 2,
        p: float = 1.0,
        dst_keys: Optional[KeyCollection] = None,
        generator: Optional[torch.Generator] = None,
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.angle_range = angle_range
        self.axis = axis
        self.p = p
        self.dst_keys = ensure_tuple_size(dst_keys or self.keys, len(self.keys))
        self.generator = generator

    def transform(self, data: Dict[str, Any]) -> Dict[str, Any]:
        data = dict(data)
        if torch.rand(1, generator=self.generator).item() >= self.p:
            return data
        lo, hi = self.angle_range
        angle_deg = torch.empty(1).uniform_(lo, hi, generator=self.generator).item()
        for key, dst_key in self.iter_keys(data, self.dst_keys):
            x = data[key]
            R = F.rotation_matrix(math.radians(angle_deg), self.axis, device=x.device)
            data[dst_key] = F.rotate(x, R)
        return data


class RandomScale(DictTransform):
    """Scale one or more keys by a uniformly random factor.

    Sampling is done once per call: all listed keys are scaled by the same
    factor (or per-axis factor vector when `anisotropic=True`).

    See Also:
        `torch_pointcloud.transforms.functional.random_scale`

    Args:
        keys: Keys to scale.
        scale_range: Min and max scaling factor.
        anisotropic: If `True`, sample a separate scale per axis of the last dim.
        p: Probability of applying the transform.
        dst_keys: Where to store the scaled tensors.
        generator: Optional `torch.Generator` for reproducibility.
        allow_missing_keys: If `True`, silently skip absent keys.
    """

    def __init__(
        self,
        keys: KeyCollection,
        scale_range: Tuple[float, float] = (0.8, 1.25),
        anisotropic: bool = False,
        p: float = 1.0,
        dst_keys: Optional[KeyCollection] = None,
        generator: Optional[torch.Generator] = None,
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.scale_range = scale_range
        self.anisotropic = anisotropic
        self.p = p
        self.dst_keys = ensure_tuple_size(dst_keys or self.keys, len(self.keys))
        self.generator = generator

    def transform(self, data: Dict[str, Any]) -> Dict[str, Any]:
        data = dict(data)
        if torch.rand(1, generator=self.generator).item() >= self.p:
            return data
        lo, hi = self.scale_range
        first_key = next(iter(self.iter_keys(data)), None)
        if first_key is None:
            return data
        d = data[first_key].shape[-1]
        if self.anisotropic:
            scale = torch.empty(d).uniform_(lo, hi, generator=self.generator)
        else:
            scale = torch.empty(1).uniform_(lo, hi, generator=self.generator)
        for key, dst_key in self.iter_keys(data, self.dst_keys):
            x = data[key]
            data[dst_key] = x * scale.to(x.dtype).to(x.device)
        return data


class RandomFlip(DictTransform):
    """Flip listed axes independently with probability `p` each.

    Sampling is done once per call: all listed keys are flipped on the same
    axes.

    See Also:
        `torch_pointcloud.transforms.functional.random_flip`

    Args:
        keys: Keys to flip.
        axes: Axis indices (into the last dim) to consider for flipping.
        p: Per-axis flip probability.
        dst_keys: Where to store the flipped tensors.
        generator: Optional `torch.Generator` for reproducibility.
        allow_missing_keys: If `True`, silently skip absent keys.
    """

    def __init__(
        self,
        keys: KeyCollection,
        axes: Sequence[int] = (0, 1),
        p: float = 0.5,
        dst_keys: Optional[KeyCollection] = None,
        generator: Optional[torch.Generator] = None,
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.axes = tuple(axes)
        self.p = p
        self.dst_keys = ensure_tuple_size(dst_keys or self.keys, len(self.keys))
        self.generator = generator

    def transform(self, data: Dict[str, Any]) -> Dict[str, Any]:
        data = dict(data)
        first_key = next(iter(self.iter_keys(data)), None)
        if first_key is None:
            return data
        d = data[first_key].shape[-1]
        flips = torch.zeros(d)
        for ax in self.axes:
            flips[ax] = 1.0 if torch.rand(1, generator=self.generator).item() < self.p else 0.0
        sign = torch.where(flips > 0, -torch.ones(d), torch.ones(d))
        for key, dst_key in self.iter_keys(data, self.dst_keys):
            x = data[key]
            data[dst_key] = x * sign.to(x.dtype).to(x.device)
        return data


class RandomJitter(DictTransform):
    """Add Gaussian noise to listed keys, optionally clipped.

    Each key gets its own independent noise tensor (because the noise shape
    matches the key shape). Pair-rotation-style consistency does not apply here.

    See Also:
        `torch_pointcloud.transforms.functional.random_jitter`

    Args:
        keys: Keys to jitter.
        sigma: Standard deviation of the Gaussian noise.
        clip: If not `None`, clip the noise to `[-clip, clip]`.
        p: Probability of applying the transform.
        dst_keys: Where to store the jittered tensors.
        generator: Optional `torch.Generator` for reproducibility.
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
    """Translate listed keys by a uniformly random vector.

    Sampling is done once per call: all listed keys are shifted by the same
    translation vector.

    See Also:
        `torch_pointcloud.transforms.functional.random_shift`

    Args:
        keys: Keys to shift.
        shift_range: Min and max per-axis translation.
        p: Probability of applying the transform.
        dst_keys: Where to store the shifted tensors.
        generator: Optional `torch.Generator` for reproducibility.
        allow_missing_keys: If `True`, silently skip absent keys.
    """

    def __init__(
        self,
        keys: KeyCollection,
        shift_range: Tuple[float, float] = (-0.2, 0.2),
        p: float = 1.0,
        dst_keys: Optional[KeyCollection] = None,
        generator: Optional[torch.Generator] = None,
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.shift_range = shift_range
        self.p = p
        self.dst_keys = ensure_tuple_size(dst_keys or self.keys, len(self.keys))
        self.generator = generator

    def transform(self, data: Dict[str, Any]) -> Dict[str, Any]:
        data = dict(data)
        if torch.rand(1, generator=self.generator).item() >= self.p:
            return data
        lo, hi = self.shift_range
        first_key = next(iter(self.iter_keys(data)), None)
        if first_key is None:
            return data
        d = data[first_key].shape[-1]
        shift = torch.empty(d).uniform_(lo, hi, generator=self.generator)
        for key, dst_key in self.iter_keys(data, self.dst_keys):
            x = data[key]
            data[dst_key] = x + shift.to(x.dtype).to(x.device)
        return data


class RandomDropout(DictTransform):
    """Randomly drop a fraction of points across all listed keys.

    The same boolean keep-mask is applied to every key so per-point
    correspondence is preserved. Sampling is once per call.

    See Also:
        `torch_pointcloud.transforms.functional.random_dropout_mask`

    Args:
        keys: Keys to subset. All must share the same leading dimension `N`.
        p_drop: Fraction of points to drop per call (uniform across points).
            Must lie in $[0, 1)$.
        p: Probability of applying the transform.
        generator: Optional `torch.Generator` for reproducibility.
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

    Each strength is a relative delta uniformly sampled from `[-x, x]`. Same
    factors are applied to every listed key in one call.

    See Also:
        `torch_pointcloud.transforms.functional.random_color_jitter`

    Args:
        keys: Color keys to jitter, shape `(N, 3)`.
        brightness: Max relative brightness change in $[0, 1]$.
        contrast: Max relative contrast change in $[0, 1]$.
        saturation: Max relative saturation change in $[0, 1]$.
        int_color: If `True`, treat colors as `[0, 255]` ints; otherwise `[0, 1]` floats.
        p: Probability of applying the transform.
        dst_keys: Where to store the jittered tensors.
        generator: Optional `torch.Generator` for reproducibility.
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
        self.p = p
        self.dst_keys = ensure_tuple_size(dst_keys or self.keys, len(self.keys))
        self.generator = generator

    def transform(self, data: Dict[str, Any]) -> Dict[str, Any]:
        data = dict(data)
        if torch.rand(1, generator=self.generator).item() >= self.p:
            return data
        for key, dst_key in self.iter_keys(data, self.dst_keys):
            data[dst_key] = F.random_color_jitter(
                data[key],
                brightness=self.brightness,
                contrast=self.contrast,
                saturation=self.saturation,
                int_color=self.int_color,
                generator=self.generator,
            )
        return data


class RandomColorDrop(DictTransform):
    """Replace colors with a constant gray value with probability `p`.

    See Also:
        `torch_pointcloud.transforms.functional.random_color_drop`

    Args:
        keys: Color keys to drop.
        fill: Replacement value in the same range as the colors. For
            `int_color=False`, sensible default is `0.5`; for `int_color=True`, `128`.
        int_color: If `True`, treat colors as `[0, 255]` ints; otherwise `[0, 1]` floats.
        p: Probability of dropping colors.
        dst_keys: Where to store the result.
        generator: Optional `torch.Generator` for reproducibility.
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

    See Also:
        `torch_pointcloud.transforms.functional.color_grayscale`

    Args:
        keys: Color keys, shape `(N, 3)`.
        int_color: If `True`, treat colors as `[0, 255]` ints; otherwise `[0, 1]` floats.
        p: Probability of converting to grayscale.
        dst_keys: Where to store the result.
        generator: Optional `torch.Generator` for reproducibility.
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

    See Also:
        `torch_pointcloud.transforms.functional.color_auto_contrast`

    Args:
        keys: Color keys, shape `(N, 3)`.
        blend: Blend weight in `[0, 1]`. `1.0` is fully auto-contrasted; `0.0` is the input.
        int_color: If `True`, treat colors as `[0, 255]` ints; otherwise `[0, 1]` floats.
        p: Probability of applying the transform.
        dst_keys: Where to store the result.
        generator: Optional `torch.Generator` for reproducibility.
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

    Args:
        pos_key: Key with positions used to compute the mask.
        keys: Extra keys to filter with the same mask.
        center: Center of the sphere. If `"centroid"`, uses the per-cloud centroid;
            if `"random_point"`, picks a random point as the center; otherwise treat as a 3-vector.
        radius: Radius of the sphere (Euclidean).
        p: Probability of applying the transform.
        generator: Optional `torch.Generator` for reproducibility (used when
            `center="random_point"`).
        allow_missing_keys: If `True`, silently skip absent keys.
    """

    def __init__(
        self,
        pos_key: str,
        radius: float,
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
        self.center = center
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
        if torch.rand(1, generator=self.generator).item() >= self.p:
            return data
        pos = data[self.pos_key]
        center = self._resolve_center(pos)
        mask = F.sphere_mask(pos, center, self.radius, dim=-1)
        for key in self.iter_keys(data):
            data[key] = data[key][mask]
        return data


class ShufflePoint(DictTransform):
    """Randomly permute the order of points across listed keys.

    The same permutation is applied to every key so per-point correspondence
    is preserved. Useful before `RandomSample` when you want to break any
    structural ordering in the input.

    See Also:
        `torch_pointcloud.transforms.functional.shuffle_indices`

    Args:
        keys: Keys to permute. All must share the same leading dimension `N`.
        p: Probability of applying the transform.
        generator: Optional `torch.Generator` for reproducibility.
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
        `torch_pointcloud.transforms.functional.random_rotate_choice`

    Args:
        keys: Keys to rotate. Each must have shape `(..., 3)`.
        angles: Candidate rotation angles, in **degrees**. Must be non-empty.
        axis: Axis index to rotate around (0=X, 1=Y, 2=Z).
        p: Probability of applying the transform.
        dst_keys: Where to store the rotated tensors. Defaults to `keys` (in-place).
        generator: Optional `torch.Generator` for reproducibility.
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
            data[dst_key] = F.rotate(x, R)
        return data


class RandomColorShift(DictTransform):
    """Additive per-channel color shift sampled uniformly per channel.

    For each of the 3 channels, sample one offset uniformly from `shift_range`
    and add it to every point's value. Sampling is once per call (same shift
    across all listed keys). Result is clamped to the valid color range.

    See Also:
        `torch_pointcloud.transforms.functional.random_color_shift`

    Args:
        keys: Color keys to shift, shape `(N, 3)`.
        shift_range: Min and max per-channel offset (in the same range as the colors).
        int_color: If `True`, treat colors as `[0, 255]` ints; otherwise `[0, 1]` floats.
        p: Probability of applying the transform.
        dst_keys: Where to store the shifted tensors.
        generator: Optional `torch.Generator` for reproducibility.
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
        self.p = p
        self.dst_keys = ensure_tuple_size(dst_keys or self.keys, len(self.keys))
        self.generator = generator

    def transform(self, data: Dict[str, Any]) -> Dict[str, Any]:
        data = dict(data)
        if torch.rand(1, generator=self.generator).item() >= self.p:
            return data
        lo, hi = self.shift_range
        max_val = 255.0 if self.int_color else 1.0
        for key, dst_key in self.iter_keys(data, self.dst_keys):
            x = data[key]
            shift = torch.empty(3, device=x.device).uniform_(lo, hi, generator=self.generator)
            out = x.float() + shift
            data[dst_key] = out.clamp(0.0, max_val).to(x.dtype)
        return data


class RandomElasticDistortion(DictTransform):
    """Apply a smooth random displacement field (elastic distortion).

    Used in SparseConvNet / MinkowskiEngine / Pointcept indoor segmentation
    recipes. Sampling is done once per call so multi-key consistency is
    preserved (the same displacement field is applied to every listed key).

    For multi-scale distortion (the standard Pointcept default), compose two
    `RandomElasticDistortion` calls with different `granularity` / `magnitude`.

    See Also:
        `torch_pointcloud.transforms.functional.random_elastic_distortion`

    Args:
        keys: Position keys to distort, shape `(N, 3)`.
        granularity: Size of the displacement-field grid cells. Smaller values
            give higher-frequency distortion.
        magnitude: Standard deviation of the per-cell Gaussian noise. Larger
            values give stronger deformation.
        p: Probability of applying the transform.
        dst_keys: Where to store the distorted tensors.
        generator: Optional `torch.Generator` for reproducibility.
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
        self.p = p
        self.dst_keys = ensure_tuple_size(dst_keys or self.keys, len(self.keys))
        self.generator = generator

    def transform(self, data: Dict[str, Any]) -> Dict[str, Any]:
        data = dict(data)
        if torch.rand(1, generator=self.generator).item() >= self.p:
            return data
        for key, dst_key in self.iter_keys(data, self.dst_keys):
            data[dst_key] = F.random_elastic_distortion(
                data[key], self.granularity, self.magnitude, generator=self.generator
            )
        return data
