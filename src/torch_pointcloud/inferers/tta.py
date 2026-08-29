r"""Test-time augmentation (TTA) inferer.

Wraps any `Inferer` and runs it N times, each time under a different spatial
augmentation of the input, then aggregates the per-point predictions. Because
point cloud segmentation outputs are indexed by point ID rather than by spatial
position, predictions from rotated or flipped views are already aligned and can
be averaged directly without inverting the transform.
"""

import itertools
import warnings
from typing import Any, Callable, Dict, List, Literal, Optional, Sequence, Union

import torch
from torch import Tensor

from torch_pointcloud.transforms import Compose, RandomFlip, RandomRotate, RandomScale, Transform
from torch_pointcloud.utils.data import DataKeys
from torch_pointcloud.utils.types import KeyCollection

from .inferer import Inferer

AggregateMode = Literal["mean", "ema"]
TransformFn = Callable[[Dict[str, Any]], Dict[str, Any]]


class TTAInferer(Inferer):
    r"""Test-time augmentation inferer.

    Runs the wrapped `base` inferer once per augmentation pass and aggregates
    the per-point predictions across passes.

    Two augmentation modes are supported:

    - **Single callable**: re-sampled independently each pass. Use for random
      augmentations such as uniformly random rotation. Requires `num_passes`.
    - **Sequence of callables**: each element is applied to exactly one pass in
      order. Use for a fixed view set (e.g. 8 evenly-spaced rotations).
      `num_passes` is inferred from the sequence length.

    Args:
        base: Underlying `Inferer` invoked once per pass. Any concrete inferer
            works: `SimpleInferer()`, `SlidingWindowInferer(...)`,
            `KNNWindowInferer(...)`.
        transforms: Single callable (re-sampled each pass) or a sequence of
            callables for fixed views. Any `Dict[str, Any] -> Dict[str, Any]`
            callable works, including `Compose`. When a sequence is given,
            `num_passes` is ignored.
        num_passes: Number of TTA passes when `transforms` is a single callable.
            Must be $\geq 1$.
        include_identity: If `True`, run one extra pass on the un-augmented input
            before the augmented passes (the "clean + N random views" voting
            protocol), so the total pass count is `num_passes + 1`.
        aggregate: How per-point predictions are combined across passes.
            `"mean"` averages per-pass outputs (works for logits or probabilities).
            `"ema"` maintains an exponential moving average of softmax
            probabilities.
        ema_smoothing: EMA factor $\alpha \in [0, 1)$ used when `aggregate="ema"`.
        ema_softmax: When `aggregate="ema"`, softmax each pass's output before
            accumulating. Set `False` if the base inferer already returns
            probabilities (e.g. `SlidingWindowInferer(softmax=True)`).
        pos_key: Dict key for the position tensor (used for the empty-output fallback).

    Example:
        A 4-pass TTA over random Z rotations and X/Y flips:

        ```python
        from torch_pointcloud.inferers import TTAInferer, SlidingWindowInferer
        from torch_pointcloud.transforms import Compose, RandomRotate, RandomFlip

        base = SlidingWindowInferer(block_size=6.0)
        aug = Compose([
            RandomRotate(keys="pos", angle_range=(-180.0, 180.0), axis=2, p=1.0),
            RandomFlip(keys="pos", axes=[0, 1], p=0.5),
        ])
        inferer = TTAInferer(base=base, transforms=aug, num_passes=4,
                             aggregate="mean")
        probs = inferer(data, predictor=lambda d: model(d["pos"], d["pos"], d["batch"]))
        ```

        Enumerated 8-view TTA (ScanNet ablation):

        ```python
        views = [Compose([RandomRotate(keys="pos", angle_range=(a, a), axis=2, p=1.0)])
                 for a in (0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0)]
        inferer = TTAInferer(base=base, transforms=views, aggregate="mean")
        ```
    """

    def __init__(
        self,
        base: Inferer,
        transforms: Union[TransformFn, Sequence[TransformFn]],
        num_passes: Optional[int] = None,
        include_identity: bool = False,
        aggregate: AggregateMode = "mean",
        ema_smoothing: float = 0.95,
        ema_softmax: bool = True,
        pos_key: str = DataKeys.POS,
    ) -> None:
        if callable(transforms):
            self._sequence: Optional[Sequence[TransformFn]] = None
            self._sample: Optional[TransformFn] = transforms
            if num_passes is None or num_passes < 1:
                raise ValueError(
                    f"`num_passes` must be an int >= 1 when `transforms` is a single callable, got {num_passes!r}."
                )
            self.num_passes = int(num_passes)
        else:
            seq = list(transforms)
            if len(seq) == 0:
                raise ValueError("`transforms` sequence must contain at least one callable.")
            if num_passes is not None and num_passes != len(seq):
                warnings.warn(
                    f"`num_passes={num_passes}` is ignored when `transforms` is a sequence "
                    f"(using len(transforms)={len(seq)} instead).",
                    stacklevel=2,
                )
            self._sequence = seq
            self._sample = None
            self.num_passes = len(seq)

        if aggregate not in ("mean", "ema"):
            raise ValueError(f"`aggregate` must be 'mean' or 'ema', got {aggregate!r}.")
        if not 0.0 <= ema_smoothing < 1.0:
            raise ValueError(f"`ema_smoothing` must be in [0, 1), got {ema_smoothing}.")

        self.base = base
        self.include_identity = include_identity
        self.aggregate = aggregate
        self.ema_smoothing = ema_smoothing
        self.ema_softmax = ema_softmax
        self.pos_key = pos_key

    @torch.no_grad()
    def forward(
        self,
        data: Dict[str, Any],
        predictor: Callable[[Dict[str, Any]], Tensor],
    ) -> Tensor:
        if self.pos_key not in data:
            raise KeyError(f"`data` is missing the required key {self.pos_key!r}.")

        output: Optional[Tensor] = None

        passes: List[Optional[TransformFn]] = [None] if self.include_identity else []
        if self._sequence is not None:
            passes.extend(self._sequence)
        else:
            passes.extend([self._sample] * self.num_passes)

        for aug in passes:
            data_aug = dict(data) if aug is None else aug(dict(data))
            pass_output = self.base(data_aug, predictor)

            if self.aggregate == "ema":
                preds = torch.softmax(pass_output, dim=-1) if self.ema_softmax else pass_output
                if output is None:
                    output = preds.clone()
                else:
                    output = self.ema_smoothing * output + (1.0 - self.ema_smoothing) * preds
            else:
                if output is None:
                    output = pass_output.clone()
                else:
                    output = output + pass_output

        if output is None:
            return data[self.pos_key].new_zeros((0, 0))
        if self.aggregate == "mean":
            output = output / float(len(passes))
        return output


def simple_tta_transforms(
    rotations: Sequence[float] = (0.0, 90.0, 180.0, 270.0),
    scales: Sequence[float] = (1.0, 0.95, 1.05),
    flip: bool = True,
    axis: int = 2,
    keys: KeyCollection = (DataKeys.POS, DataKeys.NORMAL),
    scale_keys: KeyCollection = DataKeys.POS,
) -> List[Compose]:
    r"""Default test-time augmentation transforms for indoor semantic segmentation.

    One view per (scale, rotation) pair, scales outermost, plus one x/y flip view when `flip` is set: the defaults
    give the 13-view precise-evaluation protocol of the ScanNet / ScanNet200 benchmarks. Rotations and the flip act on
    every key in `keys` (positions and normals); scales act on `scale_keys` only. Missing keys are skipped, so the
    same views serve scenes with and without normals. A rotation of $0$ or a scale of $1$ adds no transform.

    Args:
        rotations: Rotation angles in degrees about `axis`, applied at every scale.
        scales: Isotropic scale factors.
        flip: Append one view mirrored along the x and y axes.
        axis: Rotation axis (`2` is the up axis).
        keys: Keys rotated and flipped.
        scale_keys: Keys scaled.

    Returns:
        A list of `len(rotations) * len(scales) + flip` composed transforms for `TTAInferer`.

    Example:
        ```python
        from torch_pointcloud.inferers import TTAInferer, VoxelPartitionInferer, simple_tta_transforms

        inferer = TTAInferer(
            base=VoxelPartitionInferer(voxel_size=0.02, softmax=True, reduce="sum"),
            transforms=simple_tta_transforms(),
        )
        ```
    """
    transforms: List[Compose] = []
    for scale, angle in itertools.product(scales, rotations):
        steps: List[Transform] = []
        if angle != 0.0:
            steps.append(RandomRotate(keys=keys, angle_range=(angle, angle), axis=axis, p=1.0, allow_missing_keys=True))
        if scale != 1.0:
            steps.append(RandomScale(keys=scale_keys, scale_range=(scale, scale), p=1.0, allow_missing_keys=True))
        transforms.append(Compose(steps))
    if flip:
        transforms.append(Compose([RandomFlip(keys=keys, axes=(0, 1), p=1.0, allow_missing_keys=True)]))
    return transforms
