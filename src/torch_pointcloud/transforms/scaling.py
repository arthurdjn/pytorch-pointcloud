"""Transforms that rescale or normalize values."""

from typing import Any, Dict, List, Literal, Sequence, Tuple, Union, get_args

import torch
from torch import Tensor

from torch_pointcloud.utils.types import KeyCollection

from .base import DictTransform

RescaleMethod = Literal["centroid", "bbox", "centroid_extent", "min_sphere"]


__all__ = [
    "Normalize",
    "Rescale",
    "RescaleMethod",
]


def _circumsphere(support: Sequence[Tensor]) -> Tuple[Tensor, Tensor]:
    a = support[0]
    if len(support) == 1:
        return a, torch.zeros((), dtype=a.dtype)
    d = torch.stack([r - a for r in support[1:]])
    gram = 2 * d @ d.T
    rhs = (d * d).sum(dim=1)
    coeffs = torch.linalg.lstsq(gram, rhs.unsqueeze(1)).solution.squeeze(1)
    center = a + coeffs @ d
    return center, torch.norm(center - a)


def _welzl(points: List[Tensor], support: List[Tensor], dim: int) -> Tuple[Tensor, Tensor]:
    if support:
        center, radius = _circumsphere(support)
    else:
        center, radius = torch.zeros(dim, dtype=torch.float64), torch.tensor(-1.0, dtype=torch.float64)
    if len(support) == dim + 1:
        return center, radius
    for i in range(len(points)):
        if torch.norm(points[i] - center) > radius + 1e-9:
            center, radius = _welzl(points[:i], support + [points[i]], dim)
            points.insert(0, points.pop(i))
    return center, radius


def minimal_enclosing_ball(points: Tensor) -> Tuple[Tensor, Tensor]:
    r"""Compute the smallest ball enclosing a point set (Welzl's move-to-front algorithm).

    The recursion only descends on the support set (at most $C + 1$ points), so the depth is bounded by the
    dimension while the point loop stays iterative; points already known to be inside are moved to the front so
    that later checks succeed early.

    Args:
        points: Tensor of shape $(N, C)$ with $N \geq 1$.

    Returns:
        The ball center $(C,)$ and its radius (a scalar), in the dtype and on the device of `points`.

    Shape:
        - Input: $(N, C)$
        - Output: $(C,)$ and $()$

    Example:
        ```python
        import torch
        from torch_pointcloud.transforms import functional as F

        points = torch.tensor([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.5, 0.0]])
        center, radius = F.minimal_enclosing_ball(points)  # tensor([0., 0., 0.]), tensor(1.)
        ```
    """
    if points.dim() != 2 or points.shape[0] == 0:
        raise ValueError(f"Expected a non-empty (N, C) tensor, got shape {tuple(points.shape)}.")
    rows = list(points.detach().to("cpu", torch.float64))
    center, radius = _welzl(rows, [], points.shape[1])
    return center.to(points.device, points.dtype), radius.to(points.device, points.dtype)


def rescale(
    points: Tensor,
    eps: float = 1e-6,
    method: RescaleMethod = "centroid",
) -> Tensor:
    r"""Center a point set and rescale it to a unit extent.

    Operates along the point dimension `dim=-2`. Pairs a centering step with a
    scale-by-extent step that share the same statistics. The scale denominator is a
    single statistic over **all** leading dimensions, so the input is treated as one
    point cloud: rescale packed batches per sample (pre-collate), never on
    concatenated clouds.

    Args:
        points: Tensor of shape $(\ldots, N, C)$ with $C \geq 1$; min/max and means are over $N$.
        eps: Small constant added to the scale denominator for numerical stability.
        method:

            * `"centroid"`: subtract the mean over points, then divide by the max Euclidean
              distance from the centroid plus $\epsilon$:

              $$
              \mathbf{x} \leftarrow \frac{\mathbf{x} - \boldsymbol{\mu}}{\max_i \|\mathbf{x}_i - \boldsymbol{\mu}\|_2 + \epsilon}
              $$

            * `"bbox"`: subtract the axis-aligned bounding-box midpoint (midrange center),
              then divide by half the longest edge of that box plus $\epsilon$ (matches common
              ModelNet-style normalization):

              $$
              \mathbf{c} = \frac{\mathbf{x}_{\min} + \mathbf{x}_{\max}}{2}, \quad
              r = \frac{1}{2}\max_j (x_{\max,j} - x_{\min,j}) + \epsilon, \quad
              \mathbf{x} \leftarrow \frac{\mathbf{x} - \mathbf{c}}{r}
              $$

            * `"centroid_extent"`: subtract the centroid then divide by the longest axis-aligned
              span (the convention used by the published RandLA-Net Toronto-3D /
              Semantic3D checkpoints):

              $$
              \mathbf{x} \leftarrow \frac{\mathbf{x} - \boldsymbol{\mu}}{\max_j (x_{\max,j} - x_{\min,j}) + \epsilon}
              $$

            * `"min_sphere"`: subtract the center of the minimal enclosing sphere of all points and divide by
              its radius plus $\epsilon$ (see `minimal_enclosing_ball`; the OctFormer ModelNet40 normalization).

    Returns:
        Normalized tensor, same shape as `points`.

    Raises:
        ValueError: If `method` is not `"centroid"`, `"bbox"`, `"centroid_extent"`, or `"min_sphere"`.
    """
    if method not in get_args(RescaleMethod):
        raise ValueError(f"Invalid method: {method!r}. Expected one of {get_args(RescaleMethod)}.")

    if points.shape[-2] == 0:
        return points

    if method == "bbox":
        bbmin = points.min(dim=-2).values
        bbmax = points.max(dim=-2).values
        center = (bbmin + bbmax) / 2
        radius = (bbmax - bbmin).max() / 2
        return (points - center) / (radius + eps)

    if method == "centroid_extent":
        bbmin = points.min(dim=-2).values
        bbmax = points.max(dim=-2).values
        scale = (bbmax - bbmin).max()
        centroid = points.mean(dim=-2, keepdim=True)
        return (points - centroid) / (scale + eps)

    if method == "min_sphere":
        center, radius = minimal_enclosing_ball(points.reshape(-1, points.shape[-1]))
        return (points - center) / (radius + eps)

    centroid = points.mean(dim=-2, keepdim=True)
    points = points - centroid
    scale = torch.norm(points, dim=-1, keepdim=True).max()
    return points / (scale + eps)


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
            d[key] = rescale(d[key], eps=self.eps, method=self.method)
        return d


def normalize(
    x: Tensor,
    mean: Union[Tensor, Sequence[float], float],
    std: Union[Tensor, Sequence[float], float],
    eps: float = 1e-7,
) -> Tensor:
    r"""Per-channel standardization: $x' = (x - \mu) / \max(\sigma, \epsilon)$.

    Args:
        x: Input tensor. The last dimension is treated as the channel dim.
        mean: Per-channel mean(s). Broadcast against the last dimension.
        std: Per-channel standard deviation(s).
        eps: Lower bound on $\sigma$ to prevent division by zero.

    Returns:
        Standardized tensor, same shape as `x`.
    """
    if not torch.is_floating_point(x):
        x = x.float()
    mean_t = torch.as_tensor(mean, dtype=x.dtype, device=x.device)
    std_t = torch.as_tensor(std, dtype=x.dtype, device=x.device).clamp(min=eps)
    return (x - mean_t) / std_t


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
            data[key] = normalize(data[key], self.mean, self.std, eps=self.eps)
        return data
