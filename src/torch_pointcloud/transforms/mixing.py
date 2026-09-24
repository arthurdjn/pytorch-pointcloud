"""Transforms that mix two samples into one."""

import math
from typing import Any, Dict, Optional, Sequence, Tuple

import torch
from torch import Tensor

from torch_pointcloud.utils.conversion import ensure_tuple
from torch_pointcloud.utils.random import Randomizable
from torch_pointcloud.utils.types import KeyCollection

from .base import Transform
from .geometry import rotate_vectors, rotation_matrix

__all__ = [
    "LaserMix",
    "Mix3D",
    "PolarMix",
]


class Mix3D(Transform, Randomizable):
    r"""Concatenate two scenes into one, offsetting the second scene's instance ids.

    :arxiv: [Mix3D: Out-of-Context Data Augmentation for 3D Scenes](https://arxiv.org/abs/2110.02210)

    Every point-aligned key in `keys` is concatenated along the point dimension, so the mixed scene
    holds all points of both inputs. When `instance_key` is present in both scenes, the second
    scene's instance ids are shifted past the first scene's maximum id so the merged instances stay
    disjoint; points labeled `ignore_index` keep that label and are excluded from the offset.

    Unlike the other pairwise mixes, `Mix3D` keeps all points of both scenes, so the mixed scene has
    roughly twice as many points as either input.

    ![Mix3D before / after](../../assets/transforms/mix3d.png)

    Args:
        keys: Point-aligned keys concatenated jointly (e.g. `pos`, `color`, `normal`, `segment`).
        instance_key: Key of per-point instance ids to offset, or `None` to skip instance handling.
        ignore_index: Instance id treated as "no instance" (kept as-is, ignored by the offset).
        p: Probability of applying the mix; below it the first scene is returned unchanged.
        seed: Seed for the probability draw; `None` draws from the global generator (see `Randomizable`).

    Shape:
        - each key in `keys`: $(N, \ldots)$ and $(M, \ldots)$ inputs, $(N + M, \ldots)$ output.

    Example:
        ```python
        import torch
        import torch_pointcloud.transforms as T

        a = {"pos": torch.randn(100, 3), "segment": torch.randint(0, 10, (100,))}
        b = {"pos": torch.randn(120, 3), "segment": torch.randint(0, 10, (120,))}
        mix = T.Mix3D(keys=("pos", "segment"), instance_key=None)
        out = mix(a, b)
        print(out["pos"].shape)
        ```
    """

    def __init__(
        self,
        keys: KeyCollection,
        instance_key: Optional[str] = "instance",
        ignore_index: int = -1,
        p: float = 1.0,
        seed: Optional[int] = None,
    ) -> None:
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"p must be in [0, 1]; got {p}.")

        self.keys = ensure_tuple(keys)
        self.instance_key = instance_key
        self.ignore_index = ignore_index
        self.p = p
        self.set_random_state(seed)

    def _merge_instances(self, instance: Tensor, other_instance: Tensor) -> Tensor:
        valid = instance != self.ignore_index
        offset = int(instance[valid].max()) + 1 if valid.any() else 0
        other = other_instance.clone()
        other_valid = other != self.ignore_index
        other[other_valid] = other[other_valid] + offset
        return torch.cat([instance, other], dim=0)

    def transform(self, data: Dict[str, Any], other: Dict[str, Any]) -> Dict[str, Any]:
        d = dict(data)
        if torch.rand(1, generator=self.R).item() >= self.p:
            return d
        for key in self.keys:
            d[key] = torch.cat([data[key], other[key]], dim=0)
        ik = self.instance_key
        if ik is not None and ik in data and ik in other:
            d[ik] = self._merge_instances(data[ik], other[ik])
        return d


def laser_mix_masks(
    pos: Tensor,
    other_pos: Tensor,
    num_areas: int,
    pitch_range: Tuple[float, float],
    generator: Optional[torch.Generator] = None,
) -> Tuple[Tensor, Tensor]:
    r"""Return keep-masks that swap alternating inclination (pitch) bands between two LiDAR scans.

    Each point's inclination is $\phi = \arctan2(z, \sqrt{x^2 + y^2})$ in degrees. The range
    `pitch_range` is split into `num_areas` equal bands; a random parity picks whether the even or
    odd bands are kept from the first scan, with the complementary bands kept from the second. The
    mixed scene is `torch.cat([pos[mask], other_pos[other_mask]])`, so the two masks tile the sky.

    Args:
        pos: Coordinates of the first scan of shape $(N, 3)$.
        other_pos: Coordinates of the second scan of shape $(M, 3)$.
        num_areas: Number of inclination bands to split `pitch_range` into.
        pitch_range: Inclination range `(min, max)` in degrees.
        generator: Random generator for reproducibility.

    Returns:
        A tuple `(mask, other_mask)` of boolean tensors of shape $(N,)$ and $(M,)$ that select the
        points kept from `pos` and from `other_pos` respectively.

    Shape:
        - `pos`: $(N, 3)$
        - `other_pos`: $(M, 3)$
        - output: $(N,)$ and $(M,)$

    Raises:
        ValueError: If `num_areas` is not positive.

    Example:
        ```python
        import torch
        from torch_pointcloud.transforms.functional import laser_mix_masks

        pos = torch.randn(100, 3)
        other = torch.randn(120, 3)
        g = torch.Generator().manual_seed(0)
        mask, other_mask = laser_mix_masks(pos, other, num_areas=4, pitch_range=(-25.0, 3.0), generator=g)
        mixed = torch.cat([pos[mask], other[other_mask]], dim=0)
        ```
    """
    if num_areas <= 0:
        raise ValueError(f"num_areas must be positive; got {num_areas}.")
    lo, hi = pitch_range
    edges = torch.linspace(lo, hi, num_areas + 1, device=pos.device)[1:-1]

    def bands(p: Tensor) -> Tensor:
        rho = torch.sqrt(p[:, 0] ** 2 + p[:, 1] ** 2)
        pitch = torch.rad2deg(torch.atan2(p[:, 2], rho))
        return torch.bucketize(pitch, edges.to(pitch))

    start = int(torch.randint(2, (1,), generator=generator).item())
    mask = (bands(pos) % 2) == start
    other_mask = (bands(other_pos) % 2) != start
    return mask, other_mask


class LaserMix(Transform, Randomizable):
    r"""Mix two LiDAR scans by swapping alternating inclination (pitch) bands.

    :arxiv: [LaserMix for Semi-Supervised LiDAR Semantic Segmentation](https://arxiv.org/abs/2207.00026)

    Both scans are partitioned into `num_areas` inclination bands (one count is drawn per call), and
    alternating bands are taken from each scan so the mixed scene tiles the full field of view. Every
    key in `keys` is masked with the same per-scan selection, keeping per-point correspondence.

    See Also:
        `torch_pointcloud.transforms.functional.laser_mix_masks`

    ![LaserMix before / after](../../assets/transforms/laser_mix.png)

    Args:
        keys: Point-aligned keys masked jointly (must include `pos_key`).
        num_areas: Candidate band counts; one is sampled uniformly per call.
        pitch_range: Inclination range `(min, max)` in degrees.
        pos_key: Key of the coordinates used to compute inclination bands.
        p: Probability of applying the mix; below it the first scene is returned unchanged.
        seed: Seed for the band count, parity and probability draws;
            `None` draws from the global generator (see `Randomizable`).

    Shape:
        - each key in `keys`: $(N, \ldots)$ and $(M, \ldots)$ inputs, $(N' + M', \ldots)$ output.

    Example:
        ```python
        import torch
        import torch_pointcloud.transforms as T

        a = {"pos": torch.randn(100, 3), "segment": torch.randint(0, 10, (100,))}
        b = {"pos": torch.randn(120, 3), "segment": torch.randint(0, 10, (120,))}
        mix = T.LaserMix(keys=("pos", "segment"), num_areas=(3, 4, 5, 6), pitch_range=(-25.0, 3.0))
        out = mix(a, b)
        ```
    """

    def __init__(
        self,
        keys: KeyCollection,
        num_areas: Sequence[int],
        pitch_range: Tuple[float, float],
        pos_key: str = "pos",
        p: float = 1.0,
        seed: Optional[int] = None,
    ) -> None:
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"p must be in [0, 1]; got {p}.")

        self.keys = ensure_tuple(keys)
        self.num_areas = tuple(num_areas)
        self.pitch_range = pitch_range
        self.pos_key = pos_key
        self.p = p
        self.set_random_state(seed)

    def transform(self, data: Dict[str, Any], other: Dict[str, Any]) -> Dict[str, Any]:
        d = dict(data)
        if torch.rand(1, generator=self.R).item() >= self.p:
            return d
        index = int(torch.randint(len(self.num_areas), (1,), generator=self.R).item())
        num_areas = self.num_areas[index]
        mask, other_mask = laser_mix_masks(
            data[self.pos_key],
            other[self.pos_key],
            num_areas,
            self.pitch_range,
            generator=self.R,
        )
        for key in self.keys:
            d[key] = torch.cat([data[key][mask], other[key][other_mask]], dim=0)
        return d


def polar_mix_masks(
    pos: Tensor,
    other_pos: Tensor,
    generator: Optional[torch.Generator] = None,
) -> Tuple[Tensor, Tensor]:
    r"""Return keep-masks that swap a random azimuth half-sector between two LiDAR scans.

    Each point's azimuth is $\theta = \arctan2(y, x)$. A random start angle in $[-\pi, \pi)$ defines a
    half-circle sector $[\theta_0, \theta_0 + \pi)$ that wraps around the $\pm\pi$ seam, so half of the
    azimuth range is swapped regardless of the start angle. Points of the first scan outside the sector
    are kept, and points of the second scan inside the sector are added, so the mixed scene is
    `torch.cat([pos[mask], other_pos[other_mask]])`.

    Args:
        pos: Coordinates of the first scan of shape $(N, 3)$.
        other_pos: Coordinates of the second scan of shape $(M, 3)$.
        generator: Random generator for reproducibility.

    Returns:
        A tuple `(mask, other_mask)` of boolean tensors of shape $(N,)$ and $(M,)$ that select the
        points kept from `pos` and pasted from `other_pos` respectively.

    Shape:
        - `pos`: $(N, 3)$
        - `other_pos`: $(M, 3)$
        - output: $(N,)$ and $(M,)$

    Example:
        ```python
        import torch
        from torch_pointcloud.transforms.functional import polar_mix_masks

        pos = torch.randn(100, 3)
        other = torch.randn(120, 3)
        g = torch.Generator().manual_seed(0)
        mask, other_mask = polar_mix_masks(pos, other, generator=g)
        mixed = torch.cat([pos[mask], other[other_mask]], dim=0)
        ```
    """
    start = (torch.rand(1, generator=generator).item() * 2.0 - 1.0) * math.pi
    yaw = torch.atan2(pos[:, 1], pos[:, 0])
    other_yaw = torch.atan2(other_pos[:, 1], other_pos[:, 0])
    inside = (yaw - start) % (2 * math.pi) < math.pi
    other_inside = (other_yaw - start) % (2 * math.pi) < math.pi
    return ~inside, other_inside


class PolarMix(Transform, Randomizable):
    r"""Mix two LiDAR scans by swapping an azimuth sector and rotate-pasting instance-class points.

    :arxiv: [PolarMix: A General Data Augmentation Technique for LiDAR Point Clouds](https://arxiv.org/abs/2208.00223)

    Two independent sub-augmentations run per call. With probability `swap_ratio`, a random azimuth
    half-sector of the first scan is replaced by the same sector of the second scan. With probability
    `rotate_paste_ratio`, points of the second scan whose `segment_key` label is in `instance_classes`
    are rotated by a random angle about the up axis and appended. Only `pos_key` is rotated for the
    pasted points; the other keys are copied unchanged.

    See Also:
        `torch_pointcloud.transforms.functional.polar_mix_masks`

    ![PolarMix before / after](../../assets/transforms/polar_mix.png)

    Args:
        keys: Point-aligned keys masked and concatenated jointly (must include `pos_key`).
        instance_classes: Semantic labels whose points are rotate-pasted from the second scan.
        swap_ratio: Probability of swapping the azimuth sector.
        rotate_paste_ratio: Probability of rotate-pasting the instance-class points.
        pos_key: Key of the coordinates used to compute azimuth sectors and to rotate pasted points.
        segment_key: Key of per-point semantic labels used to select the instance classes.
        p: Probability of applying the mix; below it the first scene is returned unchanged.
        seed: Seed for the sector, rotation and probability draws;
            `None` draws from the global generator (see `Randomizable`).

    Shape:
        - each key in `keys`: $(N, \ldots)$ and $(M, \ldots)$ inputs, $(K, \ldots)$ output.

    Example:
        ```python
        import torch
        import torch_pointcloud.transforms as T

        a = {"pos": torch.randn(100, 3), "segment": torch.randint(0, 10, (100,))}
        b = {"pos": torch.randn(120, 3), "segment": torch.randint(0, 10, (120,))}
        mix = T.PolarMix(keys=("pos", "segment"), instance_classes=(1, 2, 3))
        out = mix(a, b)
        ```
    """

    def __init__(
        self,
        keys: KeyCollection,
        instance_classes: Sequence[int],
        swap_ratio: float = 0.5,
        rotate_paste_ratio: float = 1.0,
        pos_key: str = "pos",
        segment_key: str = "segment",
        p: float = 1.0,
        seed: Optional[int] = None,
    ) -> None:
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"p must be in [0, 1]; got {p}.")

        self.keys = ensure_tuple(keys)
        self.instance_classes = tuple(instance_classes)
        self.swap_ratio = swap_ratio
        self.rotate_paste_ratio = rotate_paste_ratio
        self.pos_key = pos_key
        self.segment_key = segment_key
        self.p = p
        self.set_random_state(seed)

    def transform(self, data: Dict[str, Any], other: Dict[str, Any]) -> Dict[str, Any]:
        d = dict(data)
        if torch.rand(1, generator=self.R).item() >= self.p:
            return d
        if torch.rand(1, generator=self.R).item() < self.swap_ratio:
            mask, other_mask = polar_mix_masks(data[self.pos_key], other[self.pos_key], generator=self.R)
            for key in self.keys:
                d[key] = torch.cat([data[key][mask], other[key][other_mask]], dim=0)
        if torch.rand(1, generator=self.R).item() < self.rotate_paste_ratio:
            segment = other[self.segment_key]
            paste = torch.isin(segment, segment.new_tensor(self.instance_classes))
            angle = torch.empty(1).uniform_(-math.pi, math.pi, generator=self.R).item()
            rotation = rotation_matrix(angle, axis=2)
            for key in self.keys:
                pasted = other[key][paste]
                if key == self.pos_key:
                    pasted = rotate_vectors(pasted, rotation)
                d[key] = torch.cat([d[key], pasted], dim=0)
        return d
