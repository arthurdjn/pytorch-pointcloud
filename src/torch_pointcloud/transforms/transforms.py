from abc import ABCMeta, abstractmethod
from typing import TYPE_CHECKING, Any, Dict, Generator, Iterable, Literal, Optional, Sequence

import numpy as np
import torch
from torch import Tensor

from torch_pointcloud.utils.conversion import ensure_tuple, ensure_tuple_size
from torch_pointcloud.utils.imports import optional_import
from torch_pointcloud.utils.octree import build_octree
from torch_pointcloud.utils.ops import consecutive_cluster, voxel_grid, voxel_grid_fnv
from torch_pointcloud.utils.types import KeyCollection, ValueCollection

from . import functional as F

if TYPE_CHECKING:
    from torch_scatter import scatter

scatter, _ = optional_import("torch_scatter", name="scatter")


__all__ = [
    "Abs",
    "AlignAxis",
    "ApplyMask",
    "AxisMinOffset",
    "BallMask",
    "BuildOctree",
    "Cat",
    "CenterShift",
    "Compose",
    "CopyItems",
    "DictTransform",
    "Divide",
    "DivideKey",
    "VoxelGrid",
    "InboxMask",
    "KeepItems",
    "NormalizeScale",
    "OnesLike",
    "RandomSample",
    "RandomSampleFaceVertices",
    "Relabel",
    "RemoveNearOrigin",
    "RenameItems",
    "SampleFarthestPoints",
    "Scale",
    "SetValue",
    "Shift",
    "Normalize",
    "SubtractKey",
    "ToDevice",
    "ToFloat",
    "ToTensor",
    "Transform",
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

        from torch_pointcloud.transforms import Compose, RandomSample, NormalizeScale

        # 1. Initialize the transforms
        transform = Compose([
            RandomSample(keys="pos", num_samples=1024),
            NormalizeScale(keys="pos"),
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
    and implement utility methods for key iteration and error handling.

    Args:
        keys: The keys to apply the transform to.
        allow_missing_keys: If `True`, the transform will not raise an error if the keys are not present in the data.

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
        generator: The generator for the random number generator.
        allow_missing_keys: If `True`, the transform will not raise an error if the keys are not present in the data.
    """

    def __init__(
        self,
        keys: KeyCollection,
        num_samples: int,
        generator: Optional[torch.Generator] = None,
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.num_samples = num_samples
        self.generator = generator

    def transform(self, data: Dict[str, Any]) -> Dict[str, Any]:
        d = dict(data)
        iterator = self.iter_keys(d)
        first_key = next(iterator)
        sampled_tensor, indices = F.random_sample(
            d[first_key], self.num_samples, return_indices=True, generator=self.generator
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


class SampleFarthestPoints(DictTransform):
    """Sample the farthest points from a dictionary entry.

    See Also:
        `torch_pointcloud.transforms.functional.sample_farthest_points`

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
        indices = F.sample_farthest_points(
            d[self.pos_key], num_samples=self.num_samples, ratio=self.ratio, random_start=self.random_start
        )
        for key in self.iter_keys(d):
            d[key] = d[key][indices]
        return d


class NormalizeScale(DictTransform):
    r"""Normalize point coordinates for dictionary entries.

    See Also:
        `torch_pointcloud.transforms.functional.normalize_scale`

    Args:
        keys: The keys to normalize the scale of.
        eps: Small constant passed to `normalize_scale`.
        method: `"centroid"` or `"bbox"`; see `functional.normalize_scale`.
        allow_missing_keys: If `True`, the transform will not raise an error if the keys are not present in the data.
    """

    def __init__(
        self,
        keys: KeyCollection,
        eps: float = 1e-6,
        method: Literal["centroid", "bbox"] = "centroid",
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.eps = eps
        self.method = method

    def transform(self, data: Dict[str, Any]) -> Dict[str, Any]:
        d = dict(data)
        for key in self.iter_keys(d):
            d[key] = F.normalize_scale(d[key], eps=self.eps, method=self.method)
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


class InboxMask(DictTransform):
    """Create a mask for dictionary tensor entries that are within a given bounding box.

    See Also:
        `torch_pointcloud.transforms.functional.inbox_mask`

    Args:
        keys: The keys to create the mask for.
        bbox: The bounding box used to mask input tensors.
        dst_keys: The keys to store the mask in.
        dim: The dimension to create the mask over.
        strict: Whether to use strict inequality.
        allow_missing_keys: If `True`, the transform will not raise an error if the keys are not present in the data.
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
            data[dst_key] = F.inbox_mask(data[key], self.bbox, dim=self.dim)
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
        if self.mask_key not in data:
            raise KeyError(f"Mask key {self.mask_key!r} not found in data.")
        d = dict(data)
        mask = d[self.mask_key]
        for key, dst_key in self.iter_keys(d, self.dst_keys):
            d[dst_key] = F.apply_mask(d[key], mask)
        return d


class SetValue(DictTransform):
    """Set a value to a key in the dictionary.

    Args:
        keys: The keys to set the values to.
        values: The values to set.
    """

    def __init__(self, keys: KeyCollection, values: Any) -> None:
        super().__init__(keys, False)
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

    Useful when tensors are stored in integer formats (e.g. ``uint8`` for
    colors) and need to be promoted to floating point before arithmetic
    transforms like `Divide` or `Normalize`.

    Args:
        keys: The keys to cast.
        allow_missing_keys: If ``True``, missing keys are silently ignored.
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
    r"""Normalize dictionary tensor entries: :math:`x' = (x - \mu) / \sigma`.

    Args:
        keys: The keys to standardize.
        mean: Per-channel mean(s).  Broadcast against the last dimension of
            each tensor.
        std: Per-channel standard deviation(s).
        allow_missing_keys: If ``True``, missing keys are silently ignored.
    """

    def __init__(
        self,
        keys: KeyCollection,
        mean: Sequence[float],
        std: Sequence[float],
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.mean = torch.tensor(mean, dtype=torch.float32)
        self.std = torch.tensor(std, dtype=torch.float32)

    def transform(self, data: Dict[str, Any]) -> Dict[str, Any]:
        data = dict(data)
        mean = self.mean.to(data[next(iter(self.keys))].device)
        std = self.std.to(data[next(iter(self.keys))].device)
        for key in self.iter_keys(data):
            data[key] = (data[key] - mean) / std
        return data


class Shift(DictTransform):
    """Shift dictionary tensor entries by subtracting a computed offset.

    Each key is offset independently. The offset is determined by `method`:

    | Method   | Offset                                           |
    | -------- | ------------------------------------------------ |
    | `"bbox"` | Midrange: `(min + max) / 2`                      |
    | `"mean"` | Centroid: `mean`                                 |
    | `"min"`  | Per-axis minimum (shifts to the positive octant) |

    Args:
        keys: The keys to shift.
        method: `"bbox"` (midrange), `"mean"` (centroid), or `"min"` (shift to origin).
        dim: The dimension to reduce over.
        dst_keys: The keys to store the shifted data in.
        allow_missing_keys: If `True`, skip missing keys silently.
    """

    def __init__(
        self,
        keys: KeyCollection,
        method: ValueCollection[Literal["bbox", "mean", "min"]],
        dim: int = 0,
        dst_keys: Optional[KeyCollection] = None,
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.dim = dim
        self.dst_keys = ensure_tuple_size(dst_keys or self.keys, len(self.keys))
        self.method = ensure_tuple_size(method, len(self.keys))
        if not set(self.method).issubset({"bbox", "mean", "min"}):
            raise ValueError(f"Invalid method: {method!r}. Expected 'bbox', 'mean', or 'min'.")

    def offset(self, x: Tensor, method: Literal["bbox", "mean", "min"]) -> Tensor:
        if method == "bbox":
            return (x.min(dim=self.dim).values + x.max(dim=self.dim).values) / 2
        if method == "mean":
            return x.mean(dim=self.dim)
        return x.min(dim=self.dim).values

    def transform(self, data: Dict[str, Any]) -> Dict[str, Any]:
        data = dict(data)
        for key, dst_key, method in self.iter_keys(data, self.dst_keys, self.method):
            x = data[key]
            if not torch.is_tensor(x):
                raise TypeError(f"Expected a tensor, got {type(x).__name__!r}.")
            data[dst_key] = x - self.offset(x, method)
        return data


class CenterShift(DictTransform):
    """Shift positions by bbox center in XY and optionally by minimum in Z.

    Matches the Pointcept `CenterShift` convention: the X and Y axes are
    shifted by their respective bbox midpoints, and the Z axis is shifted
    by its minimum when `apply_z=True` or left unchanged when `apply_z=False`.

    Args:
        keys: The position keys to shift (each must be `(N, 3)`).
        apply_z: If `True`, shift Z by its minimum; otherwise leave Z unchanged.
        allow_missing_keys: If `True`, silently skip absent keys.
    """

    def __init__(
        self,
        keys: KeyCollection,
        apply_z: bool = True,
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.apply_z = apply_z

    def transform(self, data: Dict[str, Any]) -> Dict[str, Any]:
        data = dict(data)
        for key in self.iter_keys(data):
            pos = data[key]
            mins = pos.min(dim=0).values
            maxs = pos.max(dim=0).values
            shift = torch.stack(
                [
                    (mins[0] + maxs[0]) / 2,
                    (mins[1] + maxs[1]) / 2,
                    mins[2] if self.apply_z else torch.zeros_like(mins[2]),
                ]
            )
            data[key] = pos - shift
        return data


class AlignAxis(DictTransform):
    """Shift dictionary tensor entries so that the minimum along a chosen axis is zero.

    Args:
        keys: The keys to align.
        dim: The coordinate axis to align.
        inplace: Whether to modify the tensor in place.
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

            if not self.inplace:
                x = x.clone()

            x[:, self.dim] -= x[:, self.dim].min()
            data[key] = x

        return data


class BallMask(DictTransform):
    """Create a ball mask (Chebyshev ball) for dictionary tensor entries.

    The ball is defined as the set of points that are within a given radius
    of a center point, i.e. the set of points that satisfy the inequality:

    $$
    \\| x - c \\|_{\\infty} \\leq r
    $$

    where $x$ is the point, $c$ is the center of the ball, and $r$ is the radius of the ball.

    Args:
        keys: The keys to create the mask for.
        center: The center of the ball.
        radius: The radius of the ball.
        dim: The dimension to create the mask over.
        dst_keys: The keys to store the mask in.
        allow_missing_keys: If `True`, the transform will not raise an error if the keys are not present in the data.
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

            center = torch.as_tensor(self.center, device=x.device)
            data[dst_key] = (x - center).abs().amax(dim=self.dim) <= self.radius

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


class Relabel(DictTransform):
    """Remap integer labels in dictionary entries via a lookup table.

    Args:
        keys: Keys holding label tensors to remap.
        labels: Valid source label values; mapped to 0..len-1.
        default: Value assigned to out-of-range labels.
        allow_missing_keys: If `True`, skip missing keys instead of raising.
    """

    def __init__(
        self,
        keys: KeyCollection,
        labels: Sequence[int],
        default: int = 0,
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.labels = ensure_tuple(labels)
        self.default = default

        num_labels = max(self.labels) + 1
        self._lookup = torch.full((num_labels,), self.default)
        self._lookup[torch.tensor(self.labels)] = torch.arange(len(self.labels))

    def transform(self, data: dict) -> dict:
        data = dict(data)

        for key in self.iter_keys(data):
            labels = data[key].long()

            if not isinstance(labels, torch.Tensor):
                raise TypeError(f"Expected torch.Tensor for key {key!r}, got {type(labels).__name__}")

            lookup = self._lookup.to(labels.device)

            mask = (labels >= 0) & (labels < lookup.numel())
            dst_labels = torch.full_like(labels, self.default)
            dst_labels[mask] = lookup[labels[mask]]
            data[key] = dst_labels.to(data[key].dtype)

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


class VoxelGrid(DictTransform):
    def __init__(
        self,
        pos_key: str,
        pos_reduce: Literal["mean", "min", "max", "sum", "first", "grid"],
        size: float,
        method: Literal["fnv", "pyg"] = "pyg",
        reduce: Optional[ValueCollection[Literal["mean", "min", "max", "sum", "first"]]] = None,
        keys: Optional[KeyCollection] = None,
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.pos_key = pos_key
        self.pos_reduce = pos_reduce
        self.size = size
        self.reduce = ensure_tuple_size(reduce, len(self.keys))
        self.method = method

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

        for key, reduce in self.iter_keys(data, self.reduce):
            data[key] = self._reduce(data[key], reduce, cluster, perm)

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
            x = data[key]
            col = x[:, axis]
            data[dst_key] = (col - col.min()).unsqueeze(-1).to(x.dtype)
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
