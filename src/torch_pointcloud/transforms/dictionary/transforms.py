from abc import ABCMeta, abstractmethod
from typing import TYPE_CHECKING, Any, Dict, Generator, Iterable, Literal, Optional, Sequence

import numpy as np
import torch

from torch_pointcloud.transforms.dictionary._utils import key_iterator
from torch_pointcloud.transforms.transforms import Transform
from torch_pointcloud.utils.conversion import ensure_tuple, ensure_tuple_size
from torch_pointcloud.utils.imports import optional_import
from torch_pointcloud.utils.octree import build_octree
from torch_pointcloud.utils.ops import consecutive_cluster, voxel_grid
from torch_pointcloud.utils.types import KeyCollection, ValueCollection

from . import functional as F

if TYPE_CHECKING:
    from torch_scatter import scatter

scatter, _ = optional_import("torch_scatter", name="scatter")


__all__ = [
    "Absd",
    "AlignAxisd",
    "ApplyMaskd",
    "AxisMinOffsetd",
    "BallMaskd",
    "BoundingBoxd",
    "BuildOctreed",
    "Catd",
    "Centerd",
    "CopyItemsd",
    "Divided",
    "DivideKeyd",
    "GridSubsamplingd",
    "InboxMaskd",
    "KeepItemsd",
    "NormalizeScaled",
    "OnesLiked",
    "RandomSampled",
    "RandomSampleFaceVerticesd",
    "Relabeld",
    "RemoveNearOrigind",
    "RenameItemsd",
    "SampleFarthestPointsd",
    "Scaled",
    "SetValued",
    "SubtractKeyd",
    "ToDeviced",
    "ToTensord",
    "Transformd",
]


class Transformd(Transform, metaclass=ABCMeta):
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
        return key_iterator(
            data,
            self.keys,
            *extra_iterables,
            allow_missing_keys=self.allow_missing_keys,
            extra_msg=extra_msg,
        )

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        return super().__call__(data)


class RandomSampled(Transformd):
    """Dictionary transform version of :class:`torch_pointcloud.transforms.RandomSample`.

    Args:
        keys: The keys to sample from.
        num_samples: The number of values to sample.
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
        """Apply the transform to the dictionary data.

        Args:
            data: The dictionary data to apply the transform to.

        Returns:
            The transformed dictionary data.
        """
        return F.random_sampled(
            data,
            keys=self.keys,
            num_samples=self.num_samples,
            generator=self.generator,
            allow_missing_keys=self.allow_missing_keys,
        )


class RandomSampleFaceVerticesd(Transformd):
    """Dictionary transform version of :class:`torch_pointcloud.transforms.RandomSampleFaceVertices`.

    Args:
        keys: The keys to sample from.
        face_key: The keys to sample the face from.
        num_samples: The number of vertices to sample.
        include_normals: If `True`, the normal will be included in the output.
        normal_key: The key to store the normal in.
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
        self.normal_key = normal_key
        self.generator = generator

    def transform(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply the transform to the dictionary data.

        Args:
            data: The dictionary data to apply the transform to.

        Returns:
            The transformed dictionary data.
        """
        return F.random_sample_face_verticesd(
            data,
            keys=self.keys,
            face_key=self.face_key,
            normal_key=self.normal_key,
            num_samples=self.num_samples,
            generator=self.generator,
            allow_missing_keys=self.allow_missing_keys,
        )


class SampleFarthestPointsd(Transformd):
    """Dictionary transform version of :class:`torch_pointcloud.transforms.SampleFarthestPoints`.

    Args:
        pos_key: The key to store the positions in.
        keys: The keys to sample the farthest points from.
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
        super().__init__(keys, allow_missing_keys)
        self.pos_key = pos_key
        self.num_samples = num_samples
        self.ratio = ratio
        self.random_start = random_start

    def transform(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply the transform to the dictionary data.

        Args:
            data: The dictionary data to apply the transform to.

        Returns:
            The transformed dictionary data.
        """
        return F.sample_farthest_pointsd(
            data,
            pos_key=self.pos_key,
            keys=self.keys,
            num_samples=self.num_samples,
            ratio=self.ratio,
            random_start=self.random_start,
            allow_missing_keys=self.allow_missing_keys,
        )


class NormalizeScaled(Transformd):
    """Dictionary transform version of :class:`torch_pointcloud.transforms.NormalizeScale`.

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
        """Apply the transform to the dictionary data.

        Args:
            data: The dictionary data to apply the transform to.

        Returns:
            The transformed dictionary data.
        """
        return F.normalize_scaled(
            data,
            keys=self.keys,
            eps=self.eps,
            method=self.method,
            allow_missing_keys=self.allow_missing_keys,
        )


class RemoveNearOrigind(Transformd):
    """Dict-based class transform of `torch_pointcloud.transforms.dictionary.functional.remove_near_origind`.
    This transform is designed to remove points that are within a given radius of the origin.

    Args:
        pos_key: The key containing the positions / coordinates, used to compute the distance from the origin.
        keys: The keys to remove the near origin points from.
        radius: The radius of the sphere.
        return_mask: Whether to return the mask of the points removed.
        allow_missing_keys: If `True`, the transform will not raise an error if the keys are not present in the data.
    """

    def __init__(
        self,
        pos_key: str,
        keys: Optional[KeyCollection] = None,
        radius: float = 1e-3,
        return_mask: bool = False,
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.pos_key = pos_key
        self.radius = radius
        self.return_mask = return_mask

    def transform(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply the transform to the dictionary data.

        Args:
            data: The dictionary data to apply the transform to.

        Returns:
            The transformed dictionary data.
        """
        return F.remove_near_origind(
            data,
            pos_key=self.pos_key,
            keys=self.keys,
            radius=self.radius,
            allow_missing_keys=self.allow_missing_keys,
        )


class Absd(Transformd):
    """Dictionary transform version of :class:`torch_pointcloud.transforms.Absolute`.

    Args:
        keys: The keys to make the tensor absolute.
        allow_missing_keys: If `True`, the transform will not raise an error if the keys are not present in the data.

    Returns:
        The transformed dictionary data.

    Examples:
        >>> import torch
        >>> from torch_pointcloud.transforms.dictionary.transforms import Absd
        >>> data = {"pos": torch.tensor([-1.0, 2.0, -3.0])}
        >>> transform = Absd(keys=["pos"])
        >>> transform(data)
        {"pos": tensor([1.0, 2.0, 3.0])}
    """

    def __init__(self, keys: KeyCollection, inplace: bool = False, allow_missing_keys: bool = False) -> None:
        super().__init__(keys, allow_missing_keys)
        self.inplace = inplace

    def transform(self, data: Dict[str, Any]) -> Dict[str, Any]:
        return F.absd(
            data,
            keys=self.keys,
            inplace=self.inplace,
            allow_missing_keys=self.allow_missing_keys,
        )


class BoundingBoxd(Transformd):
    """Dictionary transform version of :class:`torch_pointcloud.transforms.BoundingBox`.

    Args:
        keys: The keys to compute the bounding box of.
        dst_keys: The keys to store the bounding box in.
        dim: The dimension to compute the bounding box over.
        allow_missing_keys: If `True`, the transform will not raise an error if the keys are not present in the data.
    """

    def __init__(
        self,
        keys: KeyCollection,
        dst_keys: Optional[KeyCollection] = None,
        dim: int = 0,
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.dst_keys = dst_keys
        self.dim = dim

    def transform(self, data: Dict[str, Any]) -> Dict[str, Any]:
        return F.bounding_boxd(
            data,
            keys=self.keys,
            dst_keys=self.dst_keys,
            dim=self.dim,
            allow_missing_keys=self.allow_missing_keys,
        )


class InboxMaskd(Transformd):
    """Dictionary transform version of :class:`torch_pointcloud.transforms.InboxMask`.

    Args:
        keys: The keys to create the mask for.
        bbox: The bounding box used to mask input tensors.
        dst_keys: The keys to store the mask in.
        dim: The dimension to create the mask over.
        allow_missing_keys: If `True`, the transform will not raise an error if the keys are not present in the data.

    Examples:
        >>> import torch
        >>> from torch_pointcloud.transforms.dictionary.transforms import InboxMaskd
        >>> data = {
        ...     "pos": torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]),
        ... }
        >>> transform = InboxMaskd(keys=["pos"], bbox=(0.0, 10.0, 0.0, 10.0, 0.0, 10.0), dst_keys=["mask"])
        >>> transform(data)
        {"pos": tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]),
         "mask": tensor([[True, True, True], [True, True, True], [True, True, True]])}
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
        self.dst_keys = dst_keys
        self.dim = dim
        self.strict = strict

    def transform(self, data: Dict[str, Any]) -> Dict[str, Any]:
        return F.inbox_maskd(
            data,
            keys=self.keys,
            bbox=self.bbox,
            dst_keys=self.dst_keys,
            dim=self.dim,
            allow_missing_keys=self.allow_missing_keys,
        )


class ApplyMaskd(Transformd):
    """Class based variant of the dictionary transform `torch_pointcloud.transforms.dictionary.functional.apply_maskd`.
    This transform is designed to apply a mask to input tensors stored in a dictionary.

    Args:
        keys: The keys to apply the mask to.
        mask_key: The key to store the mask in.
        dst_keys: The keys to store the transformed data in.
        allow_missing_keys: If `True`, the transform will not raise an error if the keys are not present in the data.

    Examples:
        >>> import torch
        >>> from torch_pointcloud.transforms.dictionary.transforms import ApplyMaskd
        >>> data = {"pos": torch.tensor([1.0, 2.0, 3.0]), "mask": torch.tensor([True, False, True])}
        >>> transform = ApplyMaskd(keys=["pos"], mask_key="mask")
        >>> transform(data)
        {"pos": tensor([1.0, 3.0])}
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
        self.dst_keys = dst_keys

    def transform(self, data: Dict[str, Any]) -> Dict[str, Any]:
        return F.apply_maskd(
            data,
            keys=self.keys,
            mask_key=self.mask_key,
            dst_keys=self.dst_keys,
            allow_missing_keys=self.allow_missing_keys,
        )


class SetValued(Transformd):
    """Set a value to a key in the dictionary.

    Args:
        keys: The keys to set the values to.
        values: The values to set.
        allow_missing_keys: If `True`, the transform will not raise an error if the keys are not present in the data.
    """

    def __init__(self, keys: KeyCollection, values: Any) -> None:
        super().__init__(keys, False)
        self.values = ensure_tuple_size(values, len(self.keys))

    def transform(self, data: Dict[str, Any]) -> Dict[str, Any]:
        data = dict(data)
        for key, value in zip(self.keys, self.values):
            data[key] = value
        return data


class Scaled(Transformd):
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


class Divided(Transformd):
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


class Centerd(Transformd):
    def __init__(
        self,
        keys: KeyCollection,
        method: Literal["bbox", "mean"] = "bbox",
        dim: int = 0,
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        if method not in ["bbox", "mean"]:
            raise ValueError(f"Invalid method: {method!r}. Expected 'midrange' or 'mean'.")

        self.method = method
        self.dim = dim

    def transform(self, data: Dict[str, Any]) -> Dict[str, Any]:
        data = dict(data)

        iterator = self.iter_keys(data)
        first_key = next(iterator)

        x = data[first_key]
        if not torch.is_tensor(x):
            raise TypeError(f"Expected a tensor, got {type(x).__name__!r}.")

        if self.method == "bbox":
            center = (x.min(dim=self.dim).values + x.max(dim=self.dim).values) / 2
        else:
            center = x.mean(dim=self.dim)

        for key in self.iter_keys(data):
            x = data[key]
            if not torch.is_tensor(x):
                raise TypeError(f"Expected a tensor, got {type(x).__name__!r}.")

            data[key] = x - center
        return data


class AlignAxisd(Transformd):
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


class BallMaskd(Transformd):
    """Create a ball mask (Chebyshev ball) for the input tensors.
    The ball is defined as the set of points that are within a given radius
    of a center point, i.e. the set of points that satisfy the inequality:

    $$
    \| x - c \|_{\infty} \leq r
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
        self.dst_keys = ensure_tuple_size(dst_keys or keys, len(self.keys))
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


class ToDeviced(Transformd):
    """Convert the input tensors to the given device.

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
        self.dst_keys = ensure_tuple_size(dst_keys or keys, len(self.keys))
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


class BuildOctreed(Transformd):
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


class Relabeld(Transformd):
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

            # out-of-range source labels (255, etc.) route to default
            mask = (labels >= 0) & (labels < lookup.numel())
            dst_labels = torch.full_like(labels, self.default)
            dst_labels[mask] = lookup[labels[mask]]
            data[key] = dst_labels.to(data[key].dtype)

        return data


class RenameItemsd(Transformd):
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


class CopyItemsd(Transformd):
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


class SubtractKeyd(Transformd):
    """Subtract the value of a reference key from target keys element-wise.

    Computes `data[key] = data[key] - data[ref_key]` for each key. Broadcasting
    is handled by PyTorch.

    Args:
        keys: Keys whose tensors are modified in-place (subtracted from).
        ref_key: Key whose value is subtracted from each target key.
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
        self.dst_keys = ensure_tuple_size(dst_keys or keys, len(self.keys))

    def transform(self, data: Dict[str, Any]) -> Dict[str, Any]:
        data = dict(data)
        for key, sub_key, dst_key in self.iter_keys(data, self.sub_keys, self.dst_keys):
            data[dst_key] = data[key] - data[sub_key]
        return data


class DivideKeyd(Transformd):
    """Divide target keys by the value of a reference key element-wise.

    Computes `data[key] = data[key] / data[ref_key]` for each key. Broadcasting
    is handled by PyTorch.

    Args:
        keys: Keys whose tensors are divided.
        div_keys: Keys whose values are used as the divisors.
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
        self.dst_keys = ensure_tuple_size(dst_keys or keys, len(self.keys))

    def transform(self, data: Dict[str, Any]) -> Dict[str, Any]:
        data = dict(data)
        for key, div_key, dst_key in self.iter_keys(data, self.div_keys, self.dst_keys):
            data[dst_key] = data[key] / data[div_key]
        return data


class ToTensord(Transformd):
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


class GridSubsamplingd(Transformd):
    """Voxel-grid subsampling for point cloud dictionaries.

    Assigns each point to a voxel and aggregates within each voxel:
    positions and float feature keys are averaged; label keys retain a single
    representative point per voxel (the first point returned by the cluster).

    Args:
        pos_key: Key holding xyz positions `(N, 3)`.
        feature_keys: Keys for float features to average per voxel.
        label_keys: Keys for integer labels — one representative value per voxel.
        dl: Voxel edge length.
        allow_missing_keys: If `True`, silently skip absent feature/label keys.
    """

    def __init__(
        self,
        pos_key: str,
        feature_keys: Optional[KeyCollection] = None,
        label_keys: Optional[KeyCollection] = None,
        dl: float = 0.04,
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(None, allow_missing_keys)
        self.pos_key = pos_key
        self.feature_keys: list = list(ensure_tuple(feature_keys)) if feature_keys is not None else []
        self.label_keys: list = list(ensure_tuple(label_keys)) if label_keys is not None else []
        self.dl = dl

    def transform(self, data: Dict[str, Any]) -> Dict[str, Any]:
        data = dict(data)
        pos = data[self.pos_key].float()

        start = torch.floor(pos.min(dim=0).values / self.dl) * self.dl
        cluster = voxel_grid(pos, size=self.dl, start=start)
        cluster, perm = consecutive_cluster(cluster, return_permutation=True)

        data[self.pos_key] = scatter(pos, cluster, dim=0, reduce="mean")

        for key in self.feature_keys:
            if key in data:
                data[key] = scatter(data[key].float(), cluster, dim=0, reduce="mean")
            elif not self.allow_missing_keys:
                raise KeyError(f"Key {key!r} not found in data.")

        for key in self.label_keys:
            if key in data:
                data[key] = data[key][perm]
            elif not self.allow_missing_keys:
                raise KeyError(f"Key {key!r} not found in data.")

        return data


class OnesLiked(Transformd):
    """Adds a column of ones `(N, 1)` to the data dictionary.

    Args:
        keys: Reference keys used to determine the number of points `N`.
        dst_key: Key under which the ones tensor is stored.
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
        self.dst_keys = ensure_tuple_size(dst_keys or keys, len(self.keys))
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


class AxisMinOffsetd(Transformd):
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


        ``python
        from torch_pointcloud.transforms import AxisMinOffsetd

        data = {
            "pos": torch.randn(10, 3),
        }
        transform = AxisMinOffsetd(keys="pos", dst_keys="pos_offset", axis=2)
        data = transform(data)
        ``

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
        self.dst_keys = ensure_tuple_size(dst_keys or keys, len(self.keys))
        self.axis = ensure_tuple_size(axis, len(self.keys))

    def transform(self, data: Dict[str, Any]) -> Dict[str, Any]:
        data = dict(data)
        for key, dst_key, axis in self.iter_keys(data, self.dst_keys, self.axis):
            x = data[key]
            col = x[:, axis]
            data[dst_key] = (col - col.min()).unsqueeze(-1).to(x.dtype)
        return data


class Catd(Transformd):
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

        ``python
        from torch_pointcloud.transforms import Catd

        data = {
            "pos": torch.randn(10, 3),
            "color": torch.randn(10, 3),
            "normal": torch.randn(10, 3),
        }
        transform = Catd(keys=["pos", "color", "normal"], dst_key="x", dim=1)
        data = transform(data)
        ``

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


class KeepItemsd(Transformd):
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

        ``python
        from torch_pointcloud.transforms import KeepItemsd

        data = {
            "pos": torch.randn(10, 3),
            "color": torch.randn(10, 3),
            "normal": torch.randn(10, 3),
        }
        transform = KeepItemsd(keys=["pos", "color"])
        data = transform(data)
        ``

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
