"""Base classes of the dict transforms and their composition."""

from abc import ABCMeta, abstractmethod
from typing import Any, Dict, Generator, Iterable, Optional, Sequence

import torch
from typing_extensions import Self

from torch_pointcloud.utils.conversion import ensure_tuple
from torch_pointcloud.utils.random import Randomizable
from torch_pointcloud.utils.types import KeyCollection

__all__ = [
    "Compose",
    "DictTransform",
    "Transform",
]


class Transform(metaclass=ABCMeta):
    """Base class for all point cloud transforms.

    A transform is a callable that takes an arbitrary data object
    and returns a transformed version of it.

    While any callable can be used as a transform, this class
    provides a common interface and some convenience features, such as:

    - a `torch_pointcloud.transforms.base.Transform.transform` method,
      which implements the actual transformation logic. It will be called by the
      `__call__` method to apply the transform.
    - a `torch_pointcloud.transforms.base.Transform.extra_repr` method,
      which returns a string that describes the transform. This will be used by the
      `__repr__` method to represent the transform as a string.

    Note:
        A transform should avoid modifying the input data in place.
        Instead, it should return a new object with the transformed data.

        If the transform is in-place, it should be clearly stated in its documentation.

    Note:
        Random transforms take a `seed`. `None` draws from the global generator, which PyTorch seeds per `DataLoader`
        worker and per epoch. An int gives the transform its own stream, re-derived inside each worker so that workers
        and epochs keep drawing different numbers (see `Randomizable`).

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


class Compose(Transform, Randomizable):
    """Compose multiple transforms into a single transform.

    This class allows for chaining multiple transforms together.

    Note:
        The order of the transforms is important, as each transform will be applied
        in the order they are added to the `Compose` object.

    Args:
        transforms: The transforms to apply, in order.
        allow_missing_keys: When set, assigned to every `DictTransform` child, recursively through nested
            `Compose` objects, overriding what each child was built with (assigning the attribute later does the
            same). Registered pipelines are shared objects, so setting it on one mutates its children for every
            user. `None` leaves the children as built.

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

    def __init__(self, transforms: Sequence[Transform], allow_missing_keys: Optional[bool] = None) -> None:
        self.transforms = transforms
        # Recursively set all children's allow_missing_keys
        self.allow_missing_keys = allow_missing_keys

    @property
    def allow_missing_keys(self) -> Optional[bool]:
        return self._allow_missing_keys

    @allow_missing_keys.setter
    def allow_missing_keys(self, value: Optional[bool]) -> None:
        self._allow_missing_keys = value
        if value is None:
            return

        for transform in self.transforms:
            if isinstance(transform, (Compose, DictTransform)):
                transform.allow_missing_keys = value

    def set_random_state(self, seed: Optional[int] = None, state: Optional[torch.Generator] = None) -> Self:
        """Seed every random transform of the pipeline, each from its own draw of the pipeline's stream.

        Args:
            seed: Seed of the pipeline's stream. `None` (with no `state`) returns every transform to the global
                generator.
            state: Generator the pipeline draws the per-transform seeds from.

        Returns:
            The pipeline itself, for chaining.
        """
        super().set_random_state(seed=seed, state=state)
        for transform in self.transforms:
            if isinstance(transform, Randomizable):
                child_seed = None if self.R is None else int(torch.randint(2**62, (1,), generator=self.R).item())
                transform.set_random_state(seed=child_seed)
        return self

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
