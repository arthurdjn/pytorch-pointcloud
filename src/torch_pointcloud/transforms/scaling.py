"""Transforms that rescale or normalize values."""

from typing import Any, Dict, Sequence

import torch

from torch_pointcloud.utils.types import KeyCollection

from . import functional as F
from .base import DictTransform
from .functional import RescaleMethod

__all__ = [
    "Normalize",
    "Rescale",
    "RescaleMethod",
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
