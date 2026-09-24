"""Random geometric and color augmentations."""

import math
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import torch
from torch import Tensor

from torch_pointcloud.utils.conversion import ensure_tuple_size
from torch_pointcloud.utils.random import Randomizable
from torch_pointcloud.utils.types import KeyCollection

from .base import DictTransform
from .geometry import rotate_vectors, rotation_matrix

__all__ = [
    "RandomColorAutoContrast",
    "RandomColorDrop",
    "RandomColorGrayScale",
    "RandomColorJitter",
    "RandomColorShift",
    "RandomElasticDistortion",
    "RandomFlip",
    "RandomJitter",
    "RandomRotate",
    "RandomRotateChoice",
    "RandomScale",
    "RandomShift",
]


def rotate_boxes(boxes: Tensor, rotation: Tensor, angle: float) -> Tensor:
    r"""Rotate oriented 3D boxes about the up axis.

    Box centers are rotated by `rotation` (`centers @ rotation.transpose(-1, -2)`) and the heading is
    incremented by `angle`, so a counterclockwise rotation about $+z$ keeps the counterclockwise heading
    aligned with the jointly rotated points. Sizes are unchanged.

    Args:
        boxes: Box tensor of shape $(K, 7)$ as $[c_x, c_y, c_z, d_x, d_y, d_z, \theta]$.
        rotation: A $3 \times 3$ rotation matrix rotating by `angle` counterclockwise about the $z$ axis.
        angle: Rotation angle in **radians**, added to the heading.

    Returns:
        The rotated box tensor of shape $(K, 7)$.
    """
    boxes = boxes.clone()
    boxes[:, 0:3] = boxes[:, 0:3] @ rotation.to(boxes).transpose(-1, -2)
    boxes[:, 6] = boxes[:, 6] + angle
    return boxes


class RandomRotate(DictTransform, Randomizable):
    r"""Rotate one or more keys (and optionally oriented boxes) by a uniformly random angle around an axis.

    Sampling is done once per call: every listed key and the optional box get the same rotation. Each key is a
    $(\ldots, 3)$ field or a packed $(N, 3G)$ field of tiled 3D offsets (e.g. VoteNet votes). Pair
    `keys=("pos", "normal")` to keep positions and normals consistent, or pass `box_key` to also rotate a
    $(K, 7)$ oriented-box tensor (centers rotated, heading incremented). Box headings are counterclockwise
    yaw about the up axis, so `box_key` requires `axis=2`.

    === "Object"

        ![RandomRotate on an object](../../assets/transforms/random_rotate.png)

    === "Scene"

        ![RandomRotate on a room](../../assets/transforms/random_rotate_scene.png)

    === "Per axis"

        ![RandomRotate around each axis on an object](../../assets/transforms/rotate_axes.png)

    See Also:
        `torch_pointcloud.transforms.functional.rotate_vectors`,
        `torch_pointcloud.transforms.functional.rotate_boxes`,
        `torch_pointcloud.transforms.functional.rotation_matrix`

    Args:
        keys: Keys to rotate. Each must be a $(\ldots, 3)$ or $(N, 3G)$ vector field.
        angle_range: Min and max rotation angle, in **degrees**.
        axis: Axis index to rotate around (0=X, 1=Y, 2=Z).
        p: Probability of applying the transform.
        box_key: Optional key of a $(K, 7)$ oriented-box tensor to rotate jointly (requires `axis=2`).
        dst_keys: Where to store the rotated tensors. Defaults to `keys` (in-place).
        dst_box_key: Where to store the rotated boxes. Defaults to `box_key` (in-place).
        seed: Seed of the transform's own random stream; `None` draws from the global generator (see `Randomizable`).
        allow_missing_keys: If `True`, silently skip absent keys.
    """

    def __init__(
        self,
        keys: KeyCollection,
        angle_range: Tuple[float, float] = (-180.0, 180.0),
        axis: int = 2,
        p: float = 1.0,
        box_key: Optional[str] = None,
        dst_keys: Optional[KeyCollection] = None,
        dst_box_key: Optional[str] = None,
        seed: Optional[int] = None,
        allow_missing_keys: bool = False,
    ) -> None:
        if box_key is not None and axis != 2:
            raise ValueError(f"box_key rotation is only defined about the up axis (axis=2), got axis={axis}.")
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"p must be in [0, 1]; got {p}.")

        super().__init__(keys, allow_missing_keys)
        self.angle_range = angle_range
        self.axis = axis
        self.p = p
        self.box_key = box_key
        self.dst_keys = ensure_tuple_size(dst_keys or self.keys, len(self.keys))
        self.dst_box_key = dst_box_key or box_key
        self.set_random_state(seed)

    def transform(self, data: Dict[str, Any]) -> Dict[str, Any]:
        data = dict(data)
        if torch.rand(1, generator=self.R).item() >= self.p:
            return data

        lo, hi = self.angle_range
        angle = math.radians(torch.empty(1).uniform_(lo, hi, generator=self.R).item())
        rotation = rotation_matrix(angle, self.axis)
        box_key = self.box_key
        if box_key is not None and box_key in data:
            assert self.dst_box_key is not None
            data[self.dst_box_key] = rotate_boxes(data[box_key], rotation, angle)
        for key, dst_key in self.iter_keys(data, self.dst_keys):
            data[dst_key] = rotate_vectors(data[key], rotation)
        return data


def scale_boxes(boxes: Tensor, scale: Union[float, Tensor]) -> Tensor:
    r"""Scale oriented 3D boxes by an isotropic factor.

    Both centers and extents (columns $0$ to $6$) are multiplied by `scale`. Heading is unchanged.

    Args:
        boxes: Box tensor of shape $(K, 7)$.
        scale: Isotropic scalar factor applied to centers and sizes.

    Returns:
        The scaled box tensor of shape $(K, 7)$.
    """
    boxes = boxes.clone()
    factor = scale.to(boxes) if isinstance(scale, Tensor) else scale
    boxes[:, 0:6] = boxes[:, 0:6] * factor
    return boxes


class RandomScale(DictTransform, Randomizable):
    """Scale one or more keys (and optionally oriented boxes) by a uniformly random factor.

    Sampling is done once per call: every listed key and the optional box are scaled by the same factor (or
    per-axis factor vector when `anisotropic=True`). Pass `box_key` to also scale a $(K, 7)$ oriented-box
    tensor (centers and sizes). An oriented box has no per-axis scale, so `box_key` is incompatible with
    `anisotropic=True`.

    List only point-like keys. Do not list direction vectors such as `normal`: a scaled normal is no
    longer unit length, while a true surface normal is unchanged by an isotropic scale (and an
    anisotropic scale would require the inverse-transpose rule). Simply omit normal keys.

    === "Object"

        ![RandomScale on an object](../../assets/transforms/random_scale.png)

    === "Scene"

        ![RandomScale on a room](../../assets/transforms/random_scale_scene.png)

    See Also:
        `torch_pointcloud.transforms.functional.scale_boxes`

    Args:
        keys: Keys to scale. Point-like keys only; do not list direction vectors such as `normal`.
        scale_range: Min and max scaling factor.
        anisotropic: If `True`, sample a separate scale per axis of the last dim (incompatible with `box_key`).
        p: Probability of applying the transform.
        box_key: Optional key of a $(K, 7)$ oriented-box tensor to scale jointly.
        dst_keys: Where to store the scaled tensors.
        dst_box_key: Where to store the scaled boxes. Defaults to `box_key` (in-place).
        seed: Seed of the transform's own random stream; `None` draws from the global generator (see `Randomizable`).
        allow_missing_keys: If `True`, silently skip absent keys.
    """

    def __init__(
        self,
        keys: KeyCollection,
        scale_range: Tuple[float, float] = (0.8, 1.25),
        anisotropic: bool = False,
        p: float = 1.0,
        box_key: Optional[str] = None,
        dst_keys: Optional[KeyCollection] = None,
        dst_box_key: Optional[str] = None,
        seed: Optional[int] = None,
        allow_missing_keys: bool = False,
    ) -> None:
        if anisotropic and box_key is not None:
            raise ValueError("box_key cannot be scaled anisotropically (an oriented box has no per-axis scale).")
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"p must be in [0, 1]; got {p}.")

        super().__init__(keys, allow_missing_keys)
        self.scale_range = scale_range
        self.anisotropic = anisotropic
        self.p = p
        self.box_key = box_key
        self.dst_keys = ensure_tuple_size(dst_keys or self.keys, len(self.keys))
        self.dst_box_key = dst_box_key or box_key
        self.set_random_state(seed)

    def transform(self, data: Dict[str, Any]) -> Dict[str, Any]:
        data = dict(data)
        if torch.rand(1, generator=self.R).item() >= self.p:
            return data

        lo, hi = self.scale_range
        box_key = self.box_key
        has_box = box_key is not None and box_key in data
        first_key = next(iter(self.iter_keys(data)), None)
        if first_key is None and not has_box:
            return data
        if self.anisotropic and first_key is not None:
            scale = torch.empty(data[first_key].shape[-1]).uniform_(lo, hi, generator=self.R)
        else:
            scale = torch.empty(1).uniform_(lo, hi, generator=self.R)
        if box_key is not None and box_key in data:
            assert self.dst_box_key is not None
            data[self.dst_box_key] = scale_boxes(data[box_key], scale.to(data[box_key]))
        for key, dst_key in self.iter_keys(data, self.dst_keys):
            x = data[key]
            if self.anisotropic and x.shape[-1] != scale.numel():
                raise ValueError(
                    f"RandomScale(anisotropic=True) draws one factor per channel of the first key "
                    f"({scale.numel()}); key '{key}' has {x.shape[-1]} channels."
                )
            data[dst_key] = x * scale.to(x.dtype).to(x.device)
        return data


def flip_boxes(boxes: Tensor, axis: int) -> Tensor:
    r"""Flip oriented 3D boxes along a spatial axis.

    Boxes are stored as $(K, 7)$ rows $[c_x, c_y, c_z, d_x, d_y, d_z, \theta]$ with full extents and heading
    in radians counterclockwise about $+z$ from $+x$. A flip negates the center component along `axis`. A
    flip along `axis` $0$ (the $yz$ plane) maps the heading to $\pi - \theta$; a flip along `axis` $1$ (the
    $xz$ plane) maps the heading to $-\theta$. Sizes are unchanged.

    Args:
        boxes: Box tensor of shape $(K, 7)$.
        axis: Center axis index to negate (0=X, 1=Y).

    Returns:
        The flipped box tensor of shape $(K, 7)$.
    """
    boxes = boxes.clone()
    boxes[:, axis] = -boxes[:, axis]
    if axis == 0:
        boxes[:, 6] = math.pi - boxes[:, 6]
    elif axis == 1:
        boxes[:, 6] = -boxes[:, 6]
    return boxes


def flip_vectors(x: Tensor, axis: int) -> Tensor:
    r"""Flip a packed field of 3D vectors along a spatial axis.

    Negates component `axis` of every contiguous triple of the last dimension, so it handles both a plain
    $(N, 3)$ field (e.g. coordinates or normals) and a $(N, 3 G)$ field of $G$ tiled offsets (e.g. VoteNet
    vote offsets $(\text{center} - \text{point})$) alike.

    Args:
        x: Vector field of shape $(N, 3)$ or $(N, 3 G)$.
        axis: Axis index within each triple to negate.

    Returns:
        The flipped tensor with the same shape as `x`.
    """
    x = x.clone()
    x[..., axis::3] = -x[..., axis::3]
    return x


class RandomFlip(DictTransform, Randomizable):
    r"""Flip listed axes (and optionally oriented boxes) with probability `p` each.

    Sampling is done once per call: every listed key and the optional box are flipped on the same axes. Each
    key is a $(\ldots, 3)$ field or a packed $(N, 3G)$ field of tiled 3D offsets (e.g. VoteNet votes). Pass
    `box_key` to also flip a $(K, 7)$ oriented-box tensor (centers negated, heading remapped).

    === "Object"

        ![RandomFlip on an object](../../assets/transforms/random_flip.png)

    === "Scene"

        ![RandomFlip on a room](../../assets/transforms/random_flip_scene.png)

    === "Per axis"

        ![RandomFlip across each axis on an object](../../assets/transforms/flip_axes.png)

    See Also:
        `torch_pointcloud.transforms.functional.flip_vectors`,
        `torch_pointcloud.transforms.functional.flip_boxes`

    Args:
        keys: Keys to flip. Each must be a $(\ldots, 3)$ or $(N, 3G)$ vector field.
        axes: Axis indices (into each 3D triple) to consider for flipping.
        p: Per-axis flip probability.
        box_key: Optional key of a $(K, 7)$ oriented-box tensor to flip jointly.
        dst_keys: Where to store the flipped tensors.
        dst_box_key: Where to store the flipped boxes. Defaults to `box_key` (in-place).
        seed: Seed of the transform's own random stream; `None` draws from the global generator (see `Randomizable`).
        allow_missing_keys: If `True`, silently skip absent keys.
    """

    def __init__(
        self,
        keys: KeyCollection,
        axes: Sequence[int] = (0, 1),
        p: float = 0.5,
        box_key: Optional[str] = None,
        dst_keys: Optional[KeyCollection] = None,
        dst_box_key: Optional[str] = None,
        seed: Optional[int] = None,
        allow_missing_keys: bool = False,
    ) -> None:
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"p must be in [0, 1]; got {p}.")

        super().__init__(keys, allow_missing_keys)
        self.axes = tuple(axes)
        self.p = p
        self.box_key = box_key
        self.dst_keys = ensure_tuple_size(dst_keys or self.keys, len(self.keys))
        self.dst_box_key = dst_box_key or box_key
        self.set_random_state(seed)

    def transform(self, data: Dict[str, Any]) -> Dict[str, Any]:
        data = dict(data)
        box_key = self.box_key
        has_box = box_key is not None and box_key in data
        if next(iter(self.iter_keys(data)), None) is None and not has_box:
            return data

        flipped = [axis for axis in self.axes if torch.rand(1, generator=self.R).item() < self.p]
        if not flipped:
            return data

        if box_key is not None and box_key in data:
            assert self.dst_box_key is not None
            box = data[box_key]
            for axis in flipped:
                box = flip_boxes(box, axis)
            data[self.dst_box_key] = box
        for key, dst_key in self.iter_keys(data, self.dst_keys):
            x = data[key]
            for axis in flipped:
                x = flip_vectors(x, axis)
            data[dst_key] = x
        return data


def random_jitter(
    x: Tensor,
    sigma: float = 0.01,
    clip: Optional[float] = 0.05,
    generator: Optional[torch.Generator] = None,
) -> Tensor:
    """Add Gaussian noise to `x`, optionally clipped.

    Args:
        x: Input tensor.
        sigma: Standard deviation of the Gaussian noise.
        clip: If not `None`, clip the noise to `[-clip, clip]`.
        generator: Random generator for reproducibility.

    Returns:
        Jittered tensor with the same shape as `x`.
    """
    noise = torch.empty_like(x).normal_(mean=0.0, std=sigma, generator=generator)
    if clip is not None:
        noise = noise.clamp(min=-clip, max=clip)
    return x + noise


class RandomJitter(DictTransform, Randomizable):
    """Add Gaussian noise to listed keys, optionally clipped.

    Each key gets its own independent noise tensor (because the noise shape
    matches the key shape). Pair-rotation-style consistency does not apply here.

    === "Object"

        ![RandomJitter on an object](../../assets/transforms/random_jitter.png)

    === "Scene"

        ![RandomJitter on a room](../../assets/transforms/random_jitter_scene.png)

    See Also:
        `torch_pointcloud.transforms.functional.random_jitter`

    Args:
        keys: Keys to jitter.
        sigma: Standard deviation of the Gaussian noise.
        clip: If not `None`, clip the noise to `[-clip, clip]`.
        p: Probability of applying the transform.
        dst_keys: Where to store the jittered tensors.
        seed: Seed of the transform's own random stream; `None` draws from the global generator (see `Randomizable`).
        allow_missing_keys: If `True`, silently skip absent keys.
    """

    def __init__(
        self,
        keys: KeyCollection,
        sigma: float = 0.01,
        clip: Optional[float] = 0.05,
        p: float = 1.0,
        dst_keys: Optional[KeyCollection] = None,
        seed: Optional[int] = None,
        allow_missing_keys: bool = False,
    ) -> None:
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"p must be in [0, 1]; got {p}.")

        super().__init__(keys, allow_missing_keys)
        self.sigma = sigma
        self.clip = clip
        self.p = p
        self.dst_keys = ensure_tuple_size(dst_keys or self.keys, len(self.keys))
        self.set_random_state(seed)

    def transform(self, data: Dict[str, Any]) -> Dict[str, Any]:
        data = dict(data)
        if torch.rand(1, generator=self.R).item() >= self.p:
            return data

        for key, dst_key in self.iter_keys(data, self.dst_keys):
            data[dst_key] = random_jitter(data[key], self.sigma, self.clip, generator=self.R)
        return data


def shift_boxes(boxes: Tensor, shift: Tensor) -> Tensor:
    r"""Translate oriented 3D boxes by a fixed offset.

    Centers (columns $0$ to $3$) are offset by `shift`. Sizes and heading are unchanged.

    Args:
        boxes: Box tensor of shape $(K, 7)$ as $[c_x, c_y, c_z, d_x, d_y, d_z, \theta]$.
        shift: Translation vector of shape $(3,)$.

    Returns:
        The shifted box tensor of shape $(K, 7)$.
    """
    boxes = boxes.clone()
    boxes[:, 0:3] = boxes[:, 0:3] + shift.to(boxes)
    return boxes


class RandomShift(DictTransform, Randomizable):
    """Translate listed keys (and optionally oriented boxes) by a uniformly random vector.

    Sampling is done once per call: all listed keys and the optional box are shifted by the same
    translation vector. Pass `box_key` to also shift a $(K, 7)$ oriented-box tensor (centers only;
    sizes and heading unchanged).

    List only point-like keys. Do not list direction vectors such as `normal`: directions are
    translation-invariant, so a shifted normal is wrong. Simply omit normal keys.

    === "Object"

        ![RandomShift on an object](../../assets/transforms/random_shift.png)

    === "Scene"

        ![RandomShift on a room](../../assets/transforms/random_shift_scene.png)

    See Also:
        `torch_pointcloud.transforms.functional.shift_boxes`

    Args:
        keys: Keys to shift. Point-like keys only; do not list direction vectors such as `normal`.
        shift_range: Min and max per-axis translation.
        p: Probability of applying the transform.
        box_key: Optional key of a $(K, 7)$ oriented-box tensor to shift jointly.
        dst_keys: Where to store the shifted tensors.
        dst_box_key: Where to store the shifted boxes. Defaults to `box_key` (in-place).
        seed: Seed of the transform's own random stream; `None` draws from the global generator (see `Randomizable`).
        allow_missing_keys: If `True`, silently skip absent keys.
    """

    def __init__(
        self,
        keys: KeyCollection,
        shift_range: Tuple[float, float] = (-0.2, 0.2),
        p: float = 1.0,
        box_key: Optional[str] = None,
        dst_keys: Optional[KeyCollection] = None,
        dst_box_key: Optional[str] = None,
        seed: Optional[int] = None,
        allow_missing_keys: bool = False,
    ) -> None:
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"p must be in [0, 1]; got {p}.")

        super().__init__(keys, allow_missing_keys)
        self.shift_range = shift_range
        self.p = p
        self.box_key = box_key
        self.dst_keys = ensure_tuple_size(dst_keys or self.keys, len(self.keys))
        self.dst_box_key = dst_box_key or box_key
        self.set_random_state(seed)

    def transform(self, data: Dict[str, Any]) -> Dict[str, Any]:
        data = dict(data)
        if torch.rand(1, generator=self.R).item() >= self.p:
            return data

        lo, hi = self.shift_range
        box_key = self.box_key
        has_box = box_key is not None and box_key in data
        first_key = next(iter(self.iter_keys(data)), None)
        if first_key is None and not has_box:
            return data
        d = data[first_key].shape[-1] if first_key is not None else 3
        shift = torch.empty(d).uniform_(lo, hi, generator=self.R)
        if box_key is not None and box_key in data:
            assert self.dst_box_key is not None
            data[self.dst_box_key] = shift_boxes(data[box_key], shift[:3])
        for key, dst_key in self.iter_keys(data, self.dst_keys):
            x = data[key]
            if x.shape[-1] != shift.numel():
                raise ValueError(
                    f"RandomShift draws one offset per channel of the first key ({shift.numel()}); "
                    f"key '{key}' has {x.shape[-1]} channels."
                )
            data[dst_key] = x + shift.to(x.dtype).to(x.device)
        return data


def _color_max(color: Tensor, int_color: bool) -> float:
    """Resolve the color range maximum from the tensor dtype, validating the `int_color` flag."""
    if color.dtype == torch.uint8 or int_color:
        return 255.0
    if color.numel() > 0 and float(color.max()) > 1.0:
        raise ValueError(
            f"Float colors with `int_color=False` must lie in [0, 1], but got a maximum of {float(color.max()):.4g}. "
            "Pass `int_color=True` for [0, 255] float colors, or divide by 255 first."
        )
    return 1.0


def color_jitter(
    color: Tensor,
    brightness: Optional[float] = None,
    contrast: Optional[float] = None,
    saturation: Optional[float] = None,
    int_color: bool = False,
) -> Tensor:
    """Apply brightness, contrast, and saturation factors to colors, in that order.

    Each factor multiplies its component directly (`1.0` is identity); `None`
    skips the component entirely.

    Args:
        color: Color tensor of shape $(N, 3)$.
        brightness: Multiplicative brightness factor (e.g. `1.2` brightens by 20%).
        contrast: Contrast factor, scaling the deviation from the per-channel mean.
        saturation: Saturation factor, scaling the deviation from the per-point
            grayscale luminance.
        int_color: If `True`, treat float colors as `[0, 255]` values; otherwise `[0, 1]`.
            `uint8` colors are always treated as `[0, 255]` regardless of the flag.

    Returns:
        Jittered colors with the same shape and dtype as `color`.

    Raises:
        ValueError: If `color` is a float tensor with values above 1 while `int_color=False`.
    """
    max_val = _color_max(color, int_color)
    out = color.float() / max_val

    if brightness is not None:
        out = out * brightness
    if contrast is not None:
        mean = out.mean(dim=0, keepdim=True)
        out = (out - mean) * contrast + mean
    if saturation is not None:
        # Luminance per point, broadcast across channels.
        gray = (out * torch.tensor([0.299, 0.587, 0.114], device=out.device)).sum(dim=-1, keepdim=True)
        out = (out - gray) * saturation + gray

    out = out.clamp(0.0, 1.0) * max_val
    return out.to(color.dtype)


def random_color_jitter(
    color: Tensor,
    brightness: float = 0.0,
    contrast: float = 0.0,
    saturation: float = 0.0,
    int_color: bool = False,
    generator: Optional[torch.Generator] = None,
) -> Tensor:
    """Jitter colors by brightness, contrast, and saturation, in that order.

    Each strength is a relative delta sampled uniformly from `[-x, x]` and
    applied multiplicatively (`out = x * factor`) via `color_jitter`.

    Args:
        color: Color tensor of shape $(N, 3)$.
        brightness: Max relative brightness change. `0.2` means ±20%.
        contrast: Max relative contrast change.
        saturation: Max relative saturation change. Saturation moves toward
            (or away from) the per-channel grayscale luminance.
        int_color: If `True`, treat float colors as `[0, 255]` values; otherwise `[0, 1]`.
            `uint8` colors are always treated as `[0, 255]` regardless of the flag.
        generator: Random generator for reproducibility.

    Returns:
        Jittered colors with the same shape and dtype as `color`.

    Raises:
        ValueError: If `color` is a float tensor with values above 1 while `int_color=False`.
    """
    b = torch.empty(1).uniform_(1 - brightness, 1 + brightness, generator=generator).item() if brightness > 0 else None
    c = torch.empty(1).uniform_(1 - contrast, 1 + contrast, generator=generator).item() if contrast > 0 else None
    s = torch.empty(1).uniform_(1 - saturation, 1 + saturation, generator=generator).item() if saturation > 0 else None
    return color_jitter(color, brightness=b, contrast=c, saturation=s, int_color=int_color)


class RandomColorJitter(DictTransform, Randomizable):
    """Jitter colors by brightness, contrast, and saturation strengths.

    Each strength is a relative delta uniformly sampled from `[-x, x]`. Sampling
    is once per call, so the same factors are applied to every listed key.

    ![RandomColorJitter before / after](../../assets/transforms/color_jitter.png)

    See Also:
        `torch_pointcloud.transforms.functional.color_jitter`

    Args:
        keys: Color keys to jitter, shape $(N, 3)$.
        brightness: Max relative brightness change in $[0, 1]$.
        contrast: Max relative contrast change in $[0, 1]$.
        saturation: Max relative saturation change in $[0, 1]$.
        int_color: If `True`, treat float colors as `[0, 255]` values; otherwise `[0, 1]`.
            `uint8` colors are always treated as `[0, 255]` regardless of the flag; float colors
            above 1 with `int_color=False` raise a ValueError.
        p: Probability of applying the transform.
        dst_keys: Where to store the jittered tensors.
        seed: Seed of the transform's own random stream; `None` draws from the global generator (see `Randomizable`).
        allow_missing_keys: If `True`, silently skip absent keys.
    """

    def __init__(
        self,
        keys: KeyCollection,
        brightness: float = 0.4,
        contrast: float = 0.4,
        saturation: float = 0.2,
        int_color: bool = False,
        p: float = 1.0,
        dst_keys: Optional[KeyCollection] = None,
        seed: Optional[int] = None,
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"p must be in [0, 1]; got {p}.")

        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.int_color = int_color
        self.p = p
        self.dst_keys = ensure_tuple_size(dst_keys or self.keys, len(self.keys))
        self.set_random_state(seed)

    def transform(self, data: Dict[str, Any]) -> Dict[str, Any]:
        data = dict(data)
        if torch.rand(1, generator=self.R).item() >= self.p:
            return data

        b, c, s = self.brightness, self.contrast, self.saturation
        brightness = torch.empty(1).uniform_(1 - b, 1 + b, generator=self.R).item() if b > 0 else None
        contrast = torch.empty(1).uniform_(1 - c, 1 + c, generator=self.R).item() if c > 0 else None
        saturation = torch.empty(1).uniform_(1 - s, 1 + s, generator=self.R).item() if s > 0 else None
        for key, dst_key in self.iter_keys(data, self.dst_keys):
            data[dst_key] = color_jitter(
                data[key],
                brightness=brightness,
                contrast=contrast,
                saturation=saturation,
                int_color=self.int_color,
            )
        return data


def random_color_drop(
    color: Tensor,
    fill: float = 0.5,
    int_color: bool = False,
) -> Tensor:
    """Replace colors with a constant gray value (drops chromatic information).

    Args:
        color: Color tensor of shape $(N, 3)$.
        fill: Replacement value, expressed in the range implied by `int_color` (`[0, 1]` when
            `False`, `[0, 255]` when `True`). It is rescaled to the input's actual range when
            that differs, so the default `0.5` fills `127` on `uint8` colors.
        int_color: If `True`, treat float colors as `[0, 255]` values; otherwise `[0, 1]`.
            `uint8` colors are always treated as `[0, 255]` regardless of the flag.

    Returns:
        Tensor of the same shape and dtype as `color`, filled with the rescaled `fill`.

    Raises:
        ValueError: If `color` is a float tensor with values above 1 while `int_color=False`.
    """
    flag_max = 255.0 if int_color else 1.0
    return torch.full_like(color, fill * _color_max(color, int_color) / flag_max)


class RandomColorDrop(DictTransform, Randomizable):
    """Replace colors with a constant gray value with probability `p`.

    ![RandomColorDrop before / after](../../assets/transforms/color_drop.png)

    See Also:
        `torch_pointcloud.transforms.functional.random_color_drop`

    Args:
        keys: Color keys to drop.
        fill: Replacement value in the range implied by `int_color` (`[0, 1]` when `False`,
            `[0, 255]` when `True`); rescaled to the input's actual range when that differs, so
            the default `0.5` fills `127` on `uint8` colors.
        int_color: If `True`, treat float colors as `[0, 255]` values; otherwise `[0, 1]`.
            `uint8` colors are always treated as `[0, 255]` regardless of the flag; float colors
            above 1 with `int_color=False` raise a ValueError.
        p: Probability of dropping colors.
        dst_keys: Where to store the result.
        seed: Seed of the transform's own random stream; `None` draws from the global generator (see `Randomizable`).
        allow_missing_keys: If `True`, silently skip absent keys.
    """

    def __init__(
        self,
        keys: KeyCollection,
        fill: float = 0.5,
        int_color: bool = False,
        p: float = 0.2,
        dst_keys: Optional[KeyCollection] = None,
        seed: Optional[int] = None,
        allow_missing_keys: bool = False,
    ) -> None:
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"p must be in [0, 1]; got {p}.")

        super().__init__(keys, allow_missing_keys)
        self.fill = fill
        self.int_color = int_color

        self.p = p
        self.dst_keys = ensure_tuple_size(dst_keys or self.keys, len(self.keys))
        self.set_random_state(seed)

    def transform(self, data: Dict[str, Any]) -> Dict[str, Any]:
        data = dict(data)
        if torch.rand(1, generator=self.R).item() >= self.p:
            return data

        for key, dst_key in self.iter_keys(data, self.dst_keys):
            data[dst_key] = random_color_drop(data[key], fill=self.fill, int_color=self.int_color)
        return data


def color_grayscale(color: Tensor, int_color: bool = False) -> Tensor:
    """Convert RGB colors to grayscale using the BT.601 luminance weights.

    Args:
        color: Color tensor of shape $(N, 3)$.
        int_color: If `True`, treat colors as `[0, 255]` ints; otherwise `[0, 1]` floats.

    Returns:
        Tensor with the same shape and dtype as `color`, with R=G=B = luminance.
    """
    weights = torch.tensor([0.299, 0.587, 0.114], device=color.device)
    if int_color:
        lum = (color.float() * weights).sum(dim=-1, keepdim=True)
        return lum.expand_as(color).to(color.dtype)
    lum = (color * weights).sum(dim=-1, keepdim=True)
    return lum.expand_as(color).to(color.dtype)


class RandomColorGrayScale(DictTransform, Randomizable):
    """Convert listed color keys to grayscale (BT.601 luminance) with probability `p`.

    ![RandomColorGrayScale before / after](../../assets/transforms/color_grayscale.png)

    See Also:
        `torch_pointcloud.transforms.functional.color_grayscale`

    Args:
        keys: Color keys, shape $(N, 3)$.
        int_color: If `True`, treat colors as `[0, 255]` ints; otherwise `[0, 1]` floats.
        p: Probability of converting to grayscale.
        dst_keys: Where to store the result.
        seed: Seed of the transform's own random stream; `None` draws from the global generator (see `Randomizable`).
        allow_missing_keys: If `True`, silently skip absent keys.
    """

    def __init__(
        self,
        keys: KeyCollection,
        int_color: bool = False,
        p: float = 0.2,
        dst_keys: Optional[KeyCollection] = None,
        seed: Optional[int] = None,
        allow_missing_keys: bool = False,
    ) -> None:
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"p must be in [0, 1]; got {p}.")

        super().__init__(keys, allow_missing_keys)
        self.int_color = int_color
        self.p = p
        self.dst_keys = ensure_tuple_size(dst_keys or self.keys, len(self.keys))
        self.set_random_state(seed)

    def transform(self, data: Dict[str, Any]) -> Dict[str, Any]:
        data = dict(data)
        if torch.rand(1, generator=self.R).item() >= self.p:
            return data
        for key, dst_key in self.iter_keys(data, self.dst_keys):
            data[dst_key] = color_grayscale(data[key], int_color=self.int_color)
        return data


def color_auto_contrast(color: Tensor, blend: float = 0.5, int_color: bool = False) -> Tensor:
    """Stretch per-cloud color range to the full `[0, max]` interval, then blend.

    For each channel, the min becomes 0 and the max becomes `max_val`. The
    output is then linearly blended with the original by `blend`
    (`blend=1.0` is the fully stretched version, `blend=0.0` is the input).

    Args:
        color: Color tensor of shape $(N, 3)$.
        blend: Blend weight in `[0, 1]`.
        int_color: If `True`, treat float colors as `[0, 255]` values; otherwise `[0, 1]`.
            `uint8` colors are always treated as `[0, 255]` regardless of the flag.

    Returns:
        Auto-contrast tensor with the same shape and dtype as `color`.

    Raises:
        ValueError: If `color` is a float tensor with values above 1 while `int_color=False`.
    """
    if color.shape[0] == 0:
        return color
    max_val = _color_max(color, int_color)
    out = color.float()
    lo = out.min(dim=0).values
    hi = out.max(dim=0).values
    scale = max_val / (hi - lo).clamp(min=1e-6)
    stretched = (out - lo) * scale
    blended = blend * stretched + (1.0 - blend) * out
    return blended.clamp(0.0, max_val).to(color.dtype)


class RandomColorAutoContrast(DictTransform, Randomizable):
    """Stretch per-cloud color range to the full extent, then blend back, with probability `p`.

    ![RandomColorAutoContrast before / after](../../assets/transforms/color_auto_contrast.png)

    See Also:
        `torch_pointcloud.transforms.functional.color_auto_contrast`

    Args:
        keys: Color keys, shape $(N, 3)$.
        blend: Blend weight in `[0, 1]`. `1.0` is fully auto-contrasted; `0.0` is the input.
        int_color: If `True`, treat float colors as `[0, 255]` values; otherwise `[0, 1]`.
            `uint8` colors are always treated as `[0, 255]` regardless of the flag; float colors
            above 1 with `int_color=False` raise a ValueError.
        p: Probability of applying the transform.
        dst_keys: Where to store the result.
        seed: Seed of the transform's own random stream; `None` draws from the global generator (see `Randomizable`).
        allow_missing_keys: If `True`, silently skip absent keys.
    """

    def __init__(
        self,
        keys: KeyCollection,
        blend: float = 0.5,
        int_color: bool = False,
        p: float = 0.2,
        dst_keys: Optional[KeyCollection] = None,
        seed: Optional[int] = None,
        allow_missing_keys: bool = False,
    ) -> None:
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"p must be in [0, 1]; got {p}.")

        super().__init__(keys, allow_missing_keys)
        self.blend = blend
        self.int_color = int_color
        self.p = p
        self.dst_keys = ensure_tuple_size(dst_keys or self.keys, len(self.keys))
        self.set_random_state(seed)

    def transform(self, data: Dict[str, Any]) -> Dict[str, Any]:
        data = dict(data)
        if torch.rand(1, generator=self.R).item() >= self.p:
            return data
        for key, dst_key in self.iter_keys(data, self.dst_keys):
            data[dst_key] = color_auto_contrast(data[key], blend=self.blend, int_color=self.int_color)
        return data


class RandomRotateChoice(DictTransform, Randomizable):
    """Rotate one or more keys by an angle chosen uniformly from a discrete list.

    Common use: ModelNet / ScanObjectNN augmentation with `angles=[0, 90, 180, 270]`
    around the z-axis. Sampling is done once per call: every listed key gets
    the same rotation matrix.

    See Also:
        `torch_pointcloud.transforms.functional.rotation_matrix`,
        `torch_pointcloud.transforms.functional.rotate_vectors`

    === "Object"

        ![RandomRotateChoice on an object](../../assets/transforms/random_rotate_choice.png)

    === "Scene"

        ![RandomRotateChoice on a room](../../assets/transforms/random_rotate_choice_scene.png)

    Args:
        keys: Keys to rotate. Each must have shape `(..., 3)`.
        angles: Candidate rotation angles, in **degrees**. Must be non-empty.
        axis: Axis index to rotate around (0=X, 1=Y, 2=Z).
        p: Probability of applying the transform.
        dst_keys: Where to store the rotated tensors. Defaults to `keys` (in-place).
        seed: Seed of the transform's own random stream; `None` draws from the global generator (see `Randomizable`).
        allow_missing_keys: If `True`, silently skip absent keys.
    """

    def __init__(
        self,
        keys: KeyCollection,
        angles: Sequence[float],
        axis: int = 2,
        p: float = 1.0,
        dst_keys: Optional[KeyCollection] = None,
        seed: Optional[int] = None,
        allow_missing_keys: bool = False,
    ) -> None:
        if len(angles) == 0:
            raise ValueError("RandomRotateChoice requires at least one angle.")
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"p must be in [0, 1]; got {p}.")

        super().__init__(keys, allow_missing_keys)
        self.angles = tuple(float(a) for a in angles)
        self.axis = axis
        self.p = p
        self.dst_keys = ensure_tuple_size(dst_keys or self.keys, len(self.keys))
        self.set_random_state(seed)

    def transform(self, data: Dict[str, Any]) -> Dict[str, Any]:
        data = dict(data)
        if torch.rand(1, generator=self.R).item() >= self.p:
            return data
        idx = int(torch.randint(0, len(self.angles), (1,), generator=self.R).item())
        angle_deg = self.angles[idx]
        for key, dst_key in self.iter_keys(data, self.dst_keys):
            x = data[key]
            R = rotation_matrix(math.radians(angle_deg), self.axis, device=x.device)
            data[dst_key] = rotate_vectors(x, R)
        return data


def color_shift(color: Tensor, shift: Tensor, int_color: bool = False) -> Tensor:
    """Add a per-channel offset to colors, clamped to the valid color range.

    Args:
        color: Color tensor of shape $(N, 3)$.
        shift: Per-channel offset of shape $(3,)$, in the same range as the colors.
        int_color: If `True`, treat float colors as `[0, 255]` values; otherwise `[0, 1]`.
            `uint8` colors are always treated as `[0, 255]` regardless of the flag.

    Returns:
        Shifted colors with the same shape and dtype as `color`.

    Raises:
        ValueError: If `color` is a float tensor with values above 1 while `int_color=False`.
    """
    max_val = _color_max(color, int_color)
    out = color.float() + shift.to(color.device)
    return out.clamp(0.0, max_val).to(color.dtype)


class RandomColorShift(DictTransform, Randomizable):
    """Additive per-channel color shift sampled uniformly per channel.

    For each of the 3 channels, sample one offset uniformly from `shift_range`
    and add it to every point's value. Sampling is once per call (same shift
    across all listed keys). Result is clamped to the valid color range.

    ![RandomColorShift before / after](../../assets/transforms/color_shift.png)

    See Also:
        `torch_pointcloud.transforms.functional.color_shift`

    Args:
        keys: Color keys to shift, shape $(N, 3)$.
        shift_range: Min and max per-channel offset (in the same range as the colors).
        int_color: If `True`, treat float colors as `[0, 255]` values; otherwise `[0, 1]`.
            `uint8` colors are always treated as `[0, 255]` regardless of the flag; float colors
            above 1 with `int_color=False` raise a ValueError.
        p: Probability of applying the transform.
        dst_keys: Where to store the shifted tensors.
        seed: Seed of the transform's own random stream; `None` draws from the global generator (see `Randomizable`).
        allow_missing_keys: If `True`, silently skip absent keys.
    """

    def __init__(
        self,
        keys: KeyCollection,
        shift_range: Tuple[float, float] = (-0.05, 0.05),
        int_color: bool = False,
        p: float = 1.0,
        dst_keys: Optional[KeyCollection] = None,
        seed: Optional[int] = None,
        allow_missing_keys: bool = False,
    ) -> None:
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"p must be in [0, 1]; got {p}.")

        super().__init__(keys, allow_missing_keys)
        self.shift_range = shift_range
        self.int_color = int_color
        self.p = p
        self.dst_keys = ensure_tuple_size(dst_keys or self.keys, len(self.keys))
        self.set_random_state(seed)

    def transform(self, data: Dict[str, Any]) -> Dict[str, Any]:
        data = dict(data)
        if torch.rand(1, generator=self.R).item() >= self.p:
            return data
        lo, hi = self.shift_range
        shift = torch.empty(3).uniform_(lo, hi, generator=self.R)
        for key, dst_key in self.iter_keys(data, self.dst_keys):
            data[dst_key] = color_shift(data[key], shift, int_color=self.int_color)
        return data


def random_elastic_distortion(
    pos: Tensor,
    granularity: float = 0.2,
    magnitude: float = 0.4,
    generator: Optional[torch.Generator] = None,
) -> Tensor:
    r"""Apply a smooth random displacement field to `pos`.

    Implements the elastic distortion recipe common in sparse-voxel indoor
    segmentation pipelines: sample Gaussian noise on a coarse 3D grid (cells of
    side `granularity`), smooth it with two passes of a $3 \times 3 \times 3$ mean filter,
    trilinear-interpolate the smoothed displacement at each point, and add it
    to the position. Net effect is a locally-coherent, low-frequency
    deformation that preserves nearby-point relationships.

    Args:
        pos: Input positions of shape $(N, 3)$.
        granularity: Size of the noise grid cells (in the same units as `pos`).
            Smaller values give higher-frequency distortion.
        magnitude: Standard deviation of the per-cell Gaussian noise (in the
            same units as `pos`). Larger values give stronger deformation.
        generator: Random generator for reproducibility.

    Returns:
        Distorted positions of shape $(N, 3)$.
    """
    if pos.shape[0] == 0:
        return pos
    if pos.shape[-1] != 3:
        raise ValueError(f"random_elastic_distortion expects shape (N, 3); got {tuple(pos.shape)}.")

    pos_min = pos.min(dim=0).values
    pos_max = pos.max(dim=0).values
    extent = (pos_max - pos_min).clamp(min=granularity)

    # Noise grid with node spacing `granularity` and one pad node on each side for safe interpolation
    grid_int = (extent / granularity).ceil().to(torch.long) + 3
    grid_x, grid_y, grid_z = (int(grid_int[i].item()) for i in range(3))

    # Sample noise on the coarse grid: (N, C, D, H, W) for grid_sample input
    noise = (
        torch.randn(
            1,
            3,
            grid_z,
            grid_y,
            grid_x,
            generator=generator,
            device=pos.device,
            dtype=torch.float32,
        )
        * magnitude
    )

    # Smooth via two passes of 3x3x3 mean filter
    for _ in range(2):
        noise = torch.nn.functional.avg_pool3d(noise, kernel_size=3, stride=1, padding=1)

    # Node j sits at pos_min + granularity * (j - 1), so one grid cell spans exactly `granularity`.
    # grid_sample's grid last dim is (x, y, z) which indexes (W, H, D) of the input.
    index = (pos - pos_min) / granularity + 1.0
    normalized = 2.0 * index / (grid_int.to(pos.dtype) - 1.0) - 1.0
    sample_grid = normalized.to(noise.dtype).view(1, 1, 1, -1, 3)

    displacement = torch.nn.functional.grid_sample(
        noise,
        sample_grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )
    # displacement: (1, 3, 1, 1, N) -> (N, 3)
    displacement = displacement.squeeze(2).squeeze(2).squeeze(0).T
    return pos + displacement.to(pos.dtype)


class RandomElasticDistortion(DictTransform, Randomizable):
    """Apply a smooth random displacement field (elastic distortion).

    Used in sparse-voxel indoor segmentation recipes. Sampling is done once
    per call so multi-key consistency is preserved (the same displacement
    field is applied to every listed key).

    For multi-scale distortion (the common default), compose two
    `RandomElasticDistortion` calls with different `granularity` / `magnitude`.

    === "Object"

        ![RandomElasticDistortion on an object](../../assets/transforms/random_elastic_distortion.png)

    === "Scene"

        ![RandomElasticDistortion on a room](../../assets/transforms/random_elastic_distortion_scene.png)

    See Also:
        `torch_pointcloud.transforms.functional.random_elastic_distortion`

    Args:
        keys: Position keys to distort, shape $(N, 3)$. All listed keys must
            share the same leading dimension $N$: the per-point displacement is
            computed once from the first present key and added to every key.
        granularity: Size of the displacement-field grid cells. Smaller values
            give higher-frequency distortion.
        magnitude: Standard deviation of the per-cell Gaussian noise. Larger
            values give stronger deformation.
        p: Probability of applying the transform.
        dst_keys: Where to store the distorted tensors.
        seed: Seed of the transform's own random stream; `None` draws from the global generator (see `Randomizable`).
        allow_missing_keys: If `True`, silently skip absent keys.
    """

    def __init__(
        self,
        keys: KeyCollection,
        granularity: float = 0.2,
        magnitude: float = 0.4,
        p: float = 1.0,
        dst_keys: Optional[KeyCollection] = None,
        seed: Optional[int] = None,
        allow_missing_keys: bool = False,
    ) -> None:
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"p must be in [0, 1]; got {p}.")

        super().__init__(keys, allow_missing_keys)
        self.granularity = granularity
        self.magnitude = magnitude
        self.p = p
        self.dst_keys = ensure_tuple_size(dst_keys or self.keys, len(self.keys))
        self.set_random_state(seed)

    def transform(self, data: Dict[str, Any]) -> Dict[str, Any]:
        data = dict(data)
        if torch.rand(1, generator=self.R).item() >= self.p:
            return data
        first_key = next(iter(self.iter_keys(data)), None)
        if first_key is None:
            return data
        reference = data[first_key]
        displacement = (
            random_elastic_distortion(reference, self.granularity, self.magnitude, generator=self.R) - reference
        )
        for key, dst_key in self.iter_keys(data, self.dst_keys):
            x = data[key]
            data[dst_key] = x + displacement.to(x.dtype).to(x.device)
        return data
