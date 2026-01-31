from abc import ABCMeta, abstractmethod
from typing import Any, Optional, Sequence, Tuple, Union

from torch import Tensor

from . import functional as F


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
    def transform(self, *_: Any, **__: Any) -> Any:
        """Apply the transform to the input data.

        This method should be implemented by all subclasses, and do not
        have any constraints on the input data.
        """

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

    def extra_repr(self) -> str:
        """Return a string that describes the transform.

        This will be used by the `__repr__` method to represent the transform as a string.
        """
        return ""


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
        seed: The seed for the random number generator.
    """

    def __init__(self, num_samples: int, return_indices: bool = False, seed: Optional[int] = None):
        self.num_samples = num_samples
        self.return_indices = return_indices
        self.seed = seed

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
            seed=self.seed,
        )


class RandomSampleFaceVertices(Transform):
    """Randomly sample a fixed number of vertices from a 3D mesh (vertices, faces).

    See Also:
        `torch_pointcloud.transforms.functional.random_sample_face_vertices` for more details.

    Args:
        num_samples: The number of vertices to sample.
        return_normals: Whether to return the normals of the sampled vertices.
        seed: The seed for the random number generator.
    """

    def __init__(
        self,
        num_samples: int,
        return_normals: bool = True,
        seed: Optional[int] = None,
    ):
        self.num_samples = num_samples
        self.return_normals = return_normals
        self.seed = seed

    def transform(self, points: Tensor, faces: Tensor) -> Any:
        """Apply the transform to the input tensor.

        Args:
            points: The input tensor.
            faces: The input tensor.

        Returns:
            The transformed tensor.
        """
        return F.random_sample_face_vertices(
            points,
            faces,
            num_samples=self.num_samples,
            return_normals=self.return_normals,
            seed=self.seed,
        )


class NormalizeScale(Transform):
    r"""Normalize the scale of a 3D tensor as follows:

    $$
    \mathbf{x} = \frac{\mathbf{x} - \mathbf{\mu}}{\max(\sqrt{\sum_{i=1}^3 x_i^2}, \epsilon)}
    $$

    See Also:
        `torch_pointcloud.transforms.functional.normalize_scale`.

    Args:
        eps: The epsilon value to use to avoid division by zero.
    """

    def __init__(self, eps: float = 1e-8):
        self.eps = eps

    def transform(self, tensor: Tensor) -> Tensor:
        """Apply the transform to the input tensor.

        Args:
            tensor: The input tensor.

        Returns:
            The transformed tensor.
        """
        return F.normalize_scale(tensor, eps=self.eps)
