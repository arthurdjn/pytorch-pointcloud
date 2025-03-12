from abc import ABCMeta, abstractmethod
from typing import Any, Dict, Generator, Iterable, Optional, Tuple, Union

from torch_pointcloud.transforms.transforms import Transform
from torch_pointcloud.utils.conversion import ensure_tuple
from torch_pointcloud.utils.types import KeyCollection

from . import functional as F
from ._utils import key_iterator


class Transformd(Transform, metaclass=ABCMeta):
    """Base class for dictionary transforms.

    This class is used to define transforms that operate on a dictionary of data,
    and implement utility methods for key iteration and error handling.

    Args:
        keys: The keys to apply the transform to.
        allow_missing_keys: If ``True``, the transform will not raise an error if the keys are not present in the data.

    Example:
        To create a transform that scales the points in a point cloud (in dict format),
        we can subclass the :class:`Transformd` class and implement the :func:`Transformd.transform` method as follows:

        ```python
        from torch_pointcloud.transforms import Transformd

        # 1. Subclass the Transformd class
        class ScalePoints(Transformd):
            def __init__(self, keys: KeyCollection, scale: float = 1.0, allow_missing_keys: bool = False):
                super().__init__(keys, allow_missing_keys)
                self.scale = scale

            def transform(self, data: Dict[str, Any]) -> Dict[str, Any]:
                d = dict(data)  # Avoid modifying the input data in place
                for key in self.key_iterator(data):
                    d[key] = d[key] * self.scale
                return d

        # 2. Initialize the transform
        transform = ScalePoints(keys=["points", "normals])

        # 3. Apply the transform
        data = {"points": torch.randn(10, 3), "normals": torch.randn(10, 3)}
        data = transform(data)
        ```
    """

    def __init__(self, keys: KeyCollection, allow_missing_keys: bool = False) -> None:
        self.keys = ensure_tuple(keys)
        self.allow_missing_keys = allow_missing_keys

    def key_iterator(
        self, data: Dict[str, Any], *extra_iterables: Iterable[Any]
    ) -> Generator[Union[str, Tuple], None, None]:
        """Utility method to iterate over the keys of the data.
        If extra iterables are provided, they will be iterated over in parallel.

        Args:
            data: The data to iterate over.
            *extra_iterables: Additional iterables to iterate over.

        Returns:
            A generator of the keys.
        """
        expected_keys = ", ".join(self.keys)
        return key_iterator(
            data,
            self.keys,
            *extra_iterables,
            allow_missing_keys=self.allow_missing_keys,
            extra_msg=f"Hint: The transform {self.__class__.__name__!r} expects the following keys: {expected_keys!r}.",
        )

    @abstractmethod
    def transform(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply the transform to the dictionary data.

        Args:
            data: The dictionary data to apply the transform to.

        Returns:
            The transformed dictionary data.
        """

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        return super().__call__(data)


class RandomSampled(Transformd):
    """Dictionary transform version of :class:`torch_pointcloud.transforms.RandomSample`.

    Args:
        keys: The keys to sample from.
        num_samples: The number of values to sample.
        allow_missing_keys: If ``True``, the transform will not raise an error if the keys are not present in the data.
    """

    def __init__(
        self, keys: KeyCollection, num_samples: int, seed: Optional[int] = None, allow_missing_keys: bool = False
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.num_samples = num_samples
        self.seed = seed

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
            seed=self.seed,
            allow_missing_keys=self.allow_missing_keys,
        )


class RandomSampleVerticesd(Transformd):
    """Dictionary transform version of :class:`torch_pointcloud.transforms.RandomSampleVertices`.

    Args:
        keys: The keys to sample from.
        face_keys: The keys to sample the faces from.
        num_samples: The number of vertices to sample.
        include_normals: If ``True``, the normals will be included in the output.
        seed: The seed for the random number generator.
        allow_missing_keys: If ``True``, the transform will not raise an error if the keys are not present in the data.
    """

    def __init__(
        self,
        keys: KeyCollection,
        face_keys: KeyCollection,
        num_samples: int,
        include_normals: bool = True,
        seed: Optional[int] = None,
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.face_keys = ensure_tuple(face_keys)
        self.num_samples = num_samples
        self.include_normals = include_normals
        self.seed = seed

    def transform(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply the transform to the dictionary data.

        Args:
            data: The dictionary data to apply the transform to.

        Returns:
            The transformed dictionary data.
        """
        return F.random_sample_verticesd(
            data,
            keys=self.keys,
            face_keys=self.face_keys,
            num_samples=self.num_samples,
            include_normals=self.include_normals,
            seed=self.seed,
            allow_missing_keys=self.allow_missing_keys,
        )


class NormalizeScaled(Transformd):
    """Dictionary transform version of :class:`torch_pointcloud.transforms.NormalizeScale`.

    Args:
        keys: The keys to normalize the scale of.
        allow_missing_keys: If ``True``, the transform will not raise an error if the keys are not present in the data.
    """

    def __init__(self, keys: KeyCollection, allow_missing_keys: bool = False) -> None:
        super().__init__(keys, allow_missing_keys)

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
            allow_missing_keys=self.allow_missing_keys,
        )
