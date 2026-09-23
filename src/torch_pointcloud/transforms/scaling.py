"""Transforms that rescale, normalize or combine values."""

from typing import Any, Dict, Optional, Sequence

import torch

from torch_pointcloud.utils.conversion import ensure_tuple_size
from torch_pointcloud.utils.types import KeyCollection

from . import functional as F
from .base import DictTransform
from .functional import RescaleMethod

__all__ = [
    "Abs",
    "Clamp",
    "Divide",
    "DivideKey",
    "Normalize",
    "Rescale",
    "RescaleMethod",
    "Scale",
    "SubtractKey",
]


class Rescale(DictTransform):
    r"""Center a point set and rescale it to a unit extent.

    Bundles a centering step and a divide-by-extent step that depend on the
    same statistics. Four methods, each pairing a center and a denominator:

    | `method`              | Center on               | Divide by                        |
    | --------------------- | ----------------------- | -------------------------------- |
    | `"centroid"`          | centroid (mean)         | max Euclidean distance to center |
    | `"bbox"`              | bbox midpoint           | half of the longest axis extent  |
    | `"centroid_extent"`   | centroid (mean)         | longest axis extent              |
    | `"min_sphere"`        | min-sphere center       | min-sphere radius                |

    Empty inputs (`N=0`) are returned unchanged.

    === "method=centroid"

        ![Rescale method=centroid on an object](../../assets/transforms/rescale_centroid.png)

    === "method=bbox"

        ![Rescale method=bbox on an object](../../assets/transforms/rescale_bbox.png)

    === "method=centroid_extent"

        ![Rescale method=centroid_extent on an object](../../assets/transforms/rescale_centroid_extent.png)

    See Also:
        `torch_pointcloud.transforms.functional.rescale`

    Args:
        keys: The keys to rescale.
        eps: Small constant added to the denominator for numerical stability.
        method: `"centroid"`, `"bbox"`, `"centroid_extent"`, or `"min_sphere"`.
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
