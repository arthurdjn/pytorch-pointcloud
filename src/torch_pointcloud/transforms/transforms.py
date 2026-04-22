from abc import ABCMeta, abstractmethod
from typing import Any, Literal, Optional, Sequence, Tuple, Union, overload

import torch
from torch import Tensor

from . import functional as F

__all__ = [
    "Abs",
    "ApplyMask",
    "BoundingBox",
    "Compose",
    "InboxMask",
    "NormalizeScale",
    "RandomSample",
    "RandomSampleFaceVertices",
    "RemoveNearOrigin",
    "SampleFarthestPoints",
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
        `torch_pointcloud.transforms.dictionary.Transformd` for a version of this class
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
        class Scale(Transform):
            def __init__(self, factor: float = 1.0):
                self.factor = factor

            def extra_repr(self) -> str:
                return f"factor={self.factor}"

            def transform(self, tensor: Tensor) -> Tensor:
                return tensor * self.factor

        # 2. Initialize the transform
        transform = Scale()
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
                # simple one-liner info, which most builtin transforms will use
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
            RandomSample(1024),
            NormalizeScale(),
        ])

        # 2. Apply the transform
        tensor = torch.randn(4096, 3)
        tensor = transform(tensor)
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


class RandomSample(Transform):
    """Randomly sample a fixed number of points from a tensor.

    See Also:
        `torch_pointcloud.transforms.functional.random_sample` for more details.

    Args:
        num_samples: The number of values to sample.
        return_indices: Whether to return the indices of the sampled values.
        generator: The generator for the random number generator.
    """

    def __init__(self, num_samples: int, return_indices: bool = False, generator: Optional[torch.Generator] = None):
        self.num_samples = num_samples
        self.return_indices = return_indices
        self.generator = generator

    def transform(self, tensor: Tensor) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        """Apply the transform to the input tensor.

        Args:
            tensor: The input tensor.

        Returns:
            The transformed tensor.
        """
        return F.random_sample(
            tensor,
            num_samples=self.num_samples,
            return_indices=self.return_indices,
            generator=self.generator,
        )


class RandomSampleFaceVertices(Transform):
    """Randomly sample a fixed number of vertices from a 3D mesh (vertices, face).

    See Also:
        `torch_pointcloud.transforms.functional.random_sample_face_vertices` for more details.

    Args:
        num_samples: The number of vertices to sample.
        return_normals: Whether to return the normal of the sampled vertices.
        generator: The generator for the random number generator.
    """

    def __init__(
        self,
        num_samples: int,
        return_normals: bool = True,
        generator: Optional[torch.Generator] = None,
    ):
        self.num_samples = num_samples
        self.return_normals = return_normals
        self.generator = generator

    def transform(self, points: Tensor, face: Tensor) -> Any:
        """Apply the transform to the input tensor.

        Args:
            points: The input tensor.
            face: The input tensor.

        Returns:
            The transformed tensor.
        """
        return F.random_sample_face_vertices(
            points,
            face,
            num_samples=self.num_samples,
            return_normals=self.return_normals,
            generator=self.generator,
        )


class SampleFarthestPoints(Transform):
    """Sample the farthest points from a tensor.

    See Also:
        `torch_pointcloud.transforms.functional.sample_farthest_points` for more details.

    Args:
        num_samples: The number of points to sample.
        ratio: The ratio of points to sample.
        random_start: Whether to start the sampling from a random point.

    Returns:
        The indices of the sampled points.
    """

    def __init__(self, num_samples: Optional[int] = None, ratio: Optional[float] = None, random_start: bool = False):
        self.num_samples = num_samples
        self.ratio = ratio
        self.random_start = random_start

    def transform(self, pos: Tensor) -> Tensor:
        return F.sample_farthest_points(
            pos,
            num_samples=self.num_samples,
            ratio=self.ratio,
            random_start=self.random_start,
        )


class NormalizeScale(Transform):
    r"""Normalize point coordinates along the point dimension (see `functional.normalize_scale`).

    See Also:
        `torch_pointcloud.transforms.functional.normalize_scale`.

    Args:
        eps: Small constant added to the scale denominator.
        method: ``"centroid"`` (mean centering and max $\ell_2$ radius) or ``"bbox"`` (midrange
            centering and half the longest axis-aligned box edge).
    """

    def __init__(self, eps: float = 1e-8, method: Literal["centroid", "bbox"] = "centroid") -> None:
        self.eps = eps
        self.method = method

    def transform(self, tensor: Tensor) -> Tensor:
        """Apply the transform to the input tensor.

        Args:
            tensor: The input tensor.

        Returns:
            The transformed tensor.
        """
        return F.normalize_scale(tensor, eps=self.eps, method=self.method)


class RemoveNearOrigin(Transform):
    """Remove points that are within a given radius of the origin.

    See Also:
        `torch_pointcloud.transforms.functional.remove_near_origin` for more details.

    Args:
        radius: The radius of the sphere.

    Returns:
        The tensor with the points removed, and optionally the mask of the points removed.
    """

    def __init__(self, radius: float = 1e-3):
        self.radius = radius

    @overload
    def transform(self, pos: Tensor, return_mask: Literal[True]) -> Tuple[Tensor, Tensor]: ...

    @overload
    def transform(self, pos: Tensor, return_mask: Literal[False] = False) -> Tensor: ...

    def transform(self, pos: Tensor, return_mask: bool = False) -> Any:
        return F.remove_near_origin(pos, radius=self.radius, return_mask=return_mask)


class Abs(Transform):
    """Make the input tensor absolute.

    See Also:
        `torch_pointcloud.transforms.functional.abs` for more details.

    Args:
        inplace: Whether to perform the operation in place.

    Returns:
        The absolute tensor.

    Examples:
        >>> import torch
        >>> from torch_pointcloud.transforms.transforms import Abs
        >>> x = torch.tensor([-1.0, 2.0, -3.0])
        >>> transform = Abs()
        >>> transform(x)
        tensor([1.0, 2.0, 3.0])
    """

    def __init__(self, inplace: bool = False):
        self.inplace = inplace

    def transform(self, tensor: Tensor) -> Tensor:
        return F.abs(tensor, inplace=self.inplace)


class BoundingBox(Transform):
    """Compute the bounding box of a tensor.

    See Also:
        `torch_pointcloud.transforms.functional.bounding_box` for more details.

    Args:
        dim: The dimension to compute the bounding box over.
    """

    def __init__(self, dim: int = 0):
        self.dim = dim

    def transform(self, tensor: Tensor) -> tuple[float, ...]:
        return F.bounding_box(tensor, dim=self.dim)


class InboxMask(Transform):
    """Create a mask for the input tensor that is within a given bounding box.

    See Also:
        `torch_pointcloud.transforms.functional.inbox_mask` for more details.

    Args:
        bbox: The bounding box.
    """

    def __init__(self, dim: int = -1):
        self.dim = dim

    def transform(self, tensor: Tensor, bbox: tuple[float, ...]) -> Tensor:
        return F.inbox_mask(tensor, bbox, dim=self.dim)


class ApplyMask(Transform):
    """Apply a mask to a tensor.

    See Also:
        `torch_pointcloud.transforms.functional.apply_mask` for more details.

    Args:
        mask: The mask.
    """

    def __init__(self, mask: Tensor):
        self.mask = mask

    def transform(self, tensor: Tensor) -> Tensor:
        return F.apply_mask(tensor, self.mask)
