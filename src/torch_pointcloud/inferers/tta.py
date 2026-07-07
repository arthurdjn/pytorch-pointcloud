r"""Test-time augmentation (TTA) inferer.

Wraps any `Inferer` and runs it N times, each time under a different spatial
augmentation of the input, then aggregates the per-point predictions. Because
point cloud segmentation outputs are indexed by point ID rather than by spatial
position, predictions from rotated or flipped views are already aligned and can
be averaged directly without inverting the transform.
"""

import warnings
from typing import Any, Callable, Dict, Literal, Optional, Sequence, Union, cast

import torch
from torch import Tensor

from torch_pointcloud.utils.data import DataKeys

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
        Pointcept-style 4-pass TTA over random Z rotations and X/Y flips:

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

        for pass_i in range(self.num_passes):
            aug = cast(TransformFn, self._sequence[pass_i] if self._sequence is not None else self._sample)
            data_aug = aug(dict(data))
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
            output = output / float(self.num_passes)
        return output
