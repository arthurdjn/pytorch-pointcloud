r"""Test-time augmentation (TTA) inferers.

Run a model under several spatial augmentations of the input and merge the
outputs. Segmentation outputs are indexed by point ID rather than by spatial
position, so predictions from rotated or flipped views are already aligned and
are averaged directly. Detection outputs are boxes in the coordinates of the
view, so each view's boxes are mapped back through the inverse augmentation
before they are concatenated.
"""

import itertools
import math
import warnings
from typing import Any, Callable, Dict, List, Literal, Optional, Sequence, Tuple, Union

import torch
from torch import Tensor
from tqdm import tqdm

import torch_pointcloud.transforms.functional as F
from torch_pointcloud.transforms import Compose, RandomFlip, RandomRotate, RandomScale, Transform
from torch_pointcloud.utils.conversion import ensure_tuple
from torch_pointcloud.utils.data import DataKeys, collate
from torch_pointcloud.utils.types import Detection3D, KeyCollection

from ._utils import check_batch_alignment, index_select_dict
from .inferer import DetectionInferer, Inferer, SegmentationInferer

AggregateMode = Literal["mean", "ema"]
TransformFn = Callable[[Dict[str, Any]], Dict[str, Any]]


class TTASegmentationInferer(SegmentationInferer):
    r"""Test-time augmentation inferer for segmentation.

    Runs the wrapped `base` inferer once per augmentation pass and aggregates
    the per-point predictions across passes.

    Three modes are supported:

    - **Single callable**: re-sampled independently each pass. Use for random
      augmentations such as uniformly random rotation. Requires `num_passes`.
    - **Sequence of callables**: each element is applied to exactly one pass in
      order. Use for a fixed view set (e.g. 8 evenly-spaced rotations).
      `num_passes` is inferred from the sequence length.
    - **`None`**: the input is left as-is and the base inferer is simply run
      `num_passes` times, voting over its own randomness (random sub-batches,
      random padding).

    Args:
        base: Underlying `Inferer` invoked once per pass. Any concrete inferer
            works: `SimpleInferer()`, `SlidingWindowSegmentationInferer(...)`,
            `KNNWindowSegmentationInferer(...)`.
        transforms: Single callable (re-sampled each pass), a sequence of
            callables for fixed views, or `None` for un-augmented passes. Any
            `Dict[str, Any] -> Dict[str, Any]` callable works, including
            `Compose`. When a sequence is given, `num_passes` is ignored.
        num_passes: Number of TTA passes when `transforms` is a single callable
            or `None`. Must be $\geq 1$.
        include_identity: If `True`, run one extra pass on the un-augmented input
            before the augmented passes (the "clean + N random views" voting
            protocol), so the total pass count is `num_passes + 1`.
        aggregate: How per-point predictions are combined across passes.
            `"mean"` averages per-pass outputs. `"ema"` maintains an exponential
            moving average of them.
        ema_smoothing: EMA factor $\alpha \in [0, 1)$ used when `aggregate="ema"`.
        softmax: If `True`, softmax each pass's output before aggregating. Leave
            `False` when the base inferer already returns probabilities (e.g.
            `SlidingWindowSegmentationInferer(softmax=True)`).
        pos_key: Dict key for the position tensor (used for the empty-output fallback).
        progress: If `True`, show a `tqdm` progress bar over the passes.

    Example:
        A 4-pass TTA over random Z rotations and X/Y flips:

        ```python
        import torch

        import torch_pointcloud as tp
        from torch_pointcloud.inferers import SlidingWindowSegmentationInferer, TTASegmentationInferer
        from torch_pointcloud.transforms import Compose, RandomFlip, RandomRotate

        model = tp.create_model("pointnet.s3dis-area5").eval()
        data = {
            "pos": torch.rand(4096, 3) * 10,
            "x": torch.rand(4096, 9),
            "batch": torch.zeros(4096, dtype=torch.long),
        }

        base = SlidingWindowSegmentationInferer(block_size=6.0)
        aug = Compose([
            RandomRotate(keys="pos", angle_range=(-180.0, 180.0), axis=2, p=1.0),
            RandomFlip(keys="pos", axes=[0, 1], p=0.5),
        ])
        inferer = TTASegmentationInferer(base=base, transforms=aug, num_passes=4, aggregate="mean")
        probs = inferer(data, predictor=lambda d: model(d["x"], d["pos"], d["batch"]))
        ```

        Enumerated 8-view TTA (ScanNet ablation):

        ```python
        from torch_pointcloud.inferers import SlidingWindowSegmentationInferer, TTASegmentationInferer
        from torch_pointcloud.transforms import Compose, RandomRotate

        base = SlidingWindowSegmentationInferer(block_size=6.0)
        views = [Compose([RandomRotate(keys="pos", angle_range=(a, a), axis=2, p=1.0)])
                 for a in (0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0)]
        inferer = TTASegmentationInferer(base=base, transforms=views, aggregate="mean")
        ```
    """

    def __init__(
        self,
        base: Inferer[Tensor],
        transforms: Union[TransformFn, Sequence[TransformFn], None] = None,
        num_passes: Optional[int] = None,
        include_identity: bool = False,
        aggregate: AggregateMode = "mean",
        ema_smoothing: float = 0.95,
        softmax: bool = False,
        pos_key: str = DataKeys.POS,
        progress: bool = False,
    ) -> None:
        if transforms is None or callable(transforms):
            self._sequence: Optional[Sequence[TransformFn]] = None
            self._sample: Optional[TransformFn] = transforms
            if num_passes is None or num_passes < 1:
                raise ValueError(
                    "`num_passes` must be an int >= 1 when `transforms` is a single callable or `None`, "
                    f"got {num_passes!r}."
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
        self.softmax = softmax
        self.pos_key = pos_key
        self.progress = progress

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

        for aug in tqdm(passes, desc="TTA", leave=False, disable=not self.progress):
            data_aug = dict(data) if aug is None else aug(dict(data))
            pass_output = self.base(data_aug, predictor)
            if self.softmax:
                pass_output = torch.softmax(pass_output, dim=-1)

            if self.aggregate == "ema":
                if output is None:
                    output = pass_output.clone()
                else:
                    output = self.ema_smoothing * output + (1.0 - self.ema_smoothing) * pass_output
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
        A list of `len(rotations) * len(scales) + flip` composed transforms for `TTASegmentationInferer`.

    Example:
        ```python
        from torch_pointcloud.inferers import (
            TTASegmentationInferer,
            VoxelPartitionSegmentationInferer,
            simple_tta_transforms,
        )

        inferer = TTASegmentationInferer(
            base=VoxelPartitionSegmentationInferer(voxel_size=0.02, softmax=True, aggregate="sum"),
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


def _check_detection_keys(det: Detection3D) -> None:
    """Raise when the predictor returns a per-box entry the view inversion does not know how to map back."""
    unknown = sorted(set(det) - {"boxes", "scores", "labels", "batch", "ignore_mask", "class_probs", "velocity"})
    if unknown:
        raise ValueError(
            f"The predictor returned {unknown}, which `TTADetectionInferer` cannot map back to the input scene: it "
            f"maps `boxes` and `velocity` back and keeps `scores`, `labels`, `batch`, `ignore_mask` and "
            f"`class_probs` as they are."
        )


def _cat_detections(parts: List[Detection3D], like: Tensor) -> Detection3D:
    """Concatenate detections box-wise; no parts give an empty set on the device and dtype of `like`."""
    if not parts:
        return {
            "boxes": like.new_zeros((0, 7)),
            "scores": like.new_zeros(0),
            "labels": like.new_zeros(0, dtype=torch.long),
            "batch": like.new_zeros(0, dtype=torch.long),
        }

    keys = set(parts[0])
    for part in parts[1:]:
        if set(part) != keys:
            raise ValueError(f"The predictor returned different keys across calls: {sorted(keys)} and {sorted(part)}.")

    merged: Detection3D = {
        "boxes": torch.cat([part["boxes"] for part in parts]),
        "scores": torch.cat([part["scores"] for part in parts]),
        "labels": torch.cat([part["labels"] for part in parts]),
        "batch": torch.cat([part["batch"] for part in parts]),
    }
    if "ignore_mask" in keys:
        merged["ignore_mask"] = torch.cat([part["ignore_mask"] for part in parts])
    if "class_probs" in keys:
        merged["class_probs"] = torch.cat([part["class_probs"] for part in parts])
    if "velocity" in keys:
        merged["velocity"] = torch.cat([part["velocity"] for part in parts])
    return merged


def _split_scenes(data: Dict[str, Any], pos_key: str, batch_key: str) -> Tuple[Tensor, List[Dict[str, Any]]]:
    """Split a packed batch into one sample per scene.

    Per-point entries are split by `batch_key`, and every entry `<key>` with a `batch_<key>` scene index (the
    `cat_keys` of `collate`, e.g. ground-truth boxes) by that index; the scene indices are dropped.
    """
    pos, batch = data[pos_key], data[batch_key]
    check_batch_alignment(pos, batch, pos_key, batch_key)
    packed_keys = [key for key in data if f"batch_{key}" in data]
    scenes = torch.unique(batch)
    samples: List[Dict[str, Any]] = []
    for scene in scenes.tolist():
        sample = index_select_dict(data, torch.where(batch == scene)[0], pos.size(0))
        for key in packed_keys:
            sample[key] = data[key][data[f"batch_{key}"] == scene]
            del sample[f"batch_{key}"]
        del sample[batch_key]
        samples.append(sample)
    return scenes, samples


def _map_view_back(det: Detection3D, angle: float, scale: float, flips: Tuple[int, ...]) -> Detection3D:
    """Map boxes predicted on a view (rotated by `angle` degrees, scaled, flipped) back to the input scene."""
    velocity = det.get("velocity")
    boxes = det["boxes"] if velocity is None else torch.cat([det["boxes"], velocity], dim=1)
    for axis in flips:
        boxes = F.flip_boxes(boxes, axis)
    if scale != 1.0:
        boxes = F.scale_boxes(boxes, 1.0 / scale)
    if angle != 0.0:
        radians = -math.radians(angle)
        boxes = F.rotate_boxes(boxes, F.rotation_matrix(radians, axis=2), radians)

    mapped = det.copy()
    mapped["boxes"] = boxes[:, :7]
    if velocity is not None:
        mapped["velocity"] = boxes[:, 7:9]
    return mapped


class TTADetectionInferer(DetectionInferer):
    r"""Test-time augmentation inferer for 3D detection.

    Runs the predictor once per view of the scene and maps each view's boxes back to the coordinates of the
    input. A view rotates the scene about the up axis through the origin, scales it isotropically and mirrors
    it along $x$ and/or $y$, in that order; the views are every combination of `scales`, `rotations` and
    `flips`, scales outermost. Each view's boxes go back through the inverse steps in reverse order
    (`flip_boxes`, `scale_boxes`, `rotate_boxes`), a `velocity` with them, and the views' detections are
    concatenated. The duplicates of one object across views are left to the postprocess `nms3d`, which keeps
    the most confident one. A view must keep the scene inside the model's point-cloud range: on the KITTI
    front-view range only the $y$ flip does.

    A view moves the points, so every step that depends on their positions must run after it: the inferer
    splits the batch into its scenes and runs `transform`, the model's per-scene pipeline (for a voxel
    detector, its registered `transform`: feature stacking and voxelization within the point-cloud range), on
    each scene of each view, then collates the results, `sw_batch_size` scenes per predictor call. Entries
    the pipeline derives from the points (voxels, features) must therefore come from `transform`, not from
    the batch.

    Args:
        transform: Callable applied to each scene of each view before the predictor; `lambda sample: sample`
            for a predictor that reads the points directly.
        rotations: Rotation angles in degrees about the up axis ($z$). The boxes come back rotated, so a
            rotation needs a detector that predicts headings; an axis-aligned detector takes flips and scales.
        scales: Isotropic scale factors, each $> 0$.
        flips: Axis sets to mirror along, each a sequence of distinct axes from `(0, 1)`; `()` leaves the view
            unflipped. `[(), (0,), (1,), (0, 1)]` gives the four bird's-eye-view flips.
        keys: Keys rotated and flipped (positions, normals). Missing keys are skipped.
        scale_keys: Keys scaled. Missing keys are skipped.
        sw_batch_size: Number of transformed scenes collated into one predictor call.
        stack_keys: Keys `collate` stacks into a leading scene dimension.
        cat_keys: Keys `collate` keeps packed with a `batch_<key>` scene index, e.g. `DataKeys.POS_VOXEL` for
            the `batch_pos_voxel` of the voxel detectors.
        pos_key: Dict key for the position tensor.
        batch_key: Dict key for the per-point batch index.
        progress: If `True`, show a `tqdm` progress bar over the views.

    Example:
        PointPillars on a KITTI frame and its mirror image along $y$:

        ```python
        import torch

        import torch_pointcloud as tp
        from torch_pointcloud.inferers import TTADetectionInferer
        from torch_pointcloud.utils.data import DataKeys, select_inputs

        model, info = tp.create_model("pointpillars.kitti.openpcdet", task="detection", return_info=True)
        model.eval()
        scene = {
            "pos": torch.rand(20_000, 3) * torch.tensor([60.0, 60.0, 3.0]) - torch.tensor([0.0, 30.0, 2.0]),
            "intensity": torch.rand(20_000, 1),
            "batch": torch.zeros(20_000, dtype=torch.long),
        }

        inferer = TTADetectionInferer(
            transform=info["transform"],
            flips=[(), (1,)],
            cat_keys=[DataKeys.POS_VOXEL],
        )
        det = inferer(scene, predictor=lambda d: model.decode(model(*select_inputs(d, info["input_keys"]))))
        ```
    """

    def __init__(
        self,
        transform: TransformFn,
        rotations: Sequence[float] = (0.0,),
        scales: Sequence[float] = (1.0,),
        flips: Sequence[Sequence[int]] = ((),),
        keys: KeyCollection = (DataKeys.POS, DataKeys.NORMAL),
        scale_keys: KeyCollection = DataKeys.POS,
        sw_batch_size: int = 1,
        stack_keys: Optional[KeyCollection] = None,
        cat_keys: Optional[KeyCollection] = None,
        pos_key: str = DataKeys.POS,
        batch_key: str = DataKeys.BATCH,
        progress: bool = False,
    ) -> None:
        if len(rotations) == 0 or len(scales) == 0 or len(flips) == 0:
            raise ValueError("`rotations`, `scales` and `flips` must each hold at least one entry.")

        if any(scale <= 0.0 for scale in scales):
            raise ValueError(f"`scales` must all be > 0, got {list(scales)}.")

        for axes in flips:
            if any(axis not in (0, 1) for axis in axes) or len(set(axes)) != len(axes):
                raise ValueError(f"Each entry of `flips` must list distinct axes from (0, 1), got {tuple(axes)}.")

        if sw_batch_size < 1:
            raise ValueError(f"`sw_batch_size` must be >= 1, got {sw_batch_size}.")

        self.transform = transform
        self.rotations = [float(angle) for angle in rotations]
        self.scales = [float(scale) for scale in scales]
        self.flips = [tuple(int(axis) for axis in axes) for axes in flips]
        self.keys = ensure_tuple(keys)
        self.scale_keys = ensure_tuple(scale_keys)
        self.sw_batch_size = sw_batch_size
        self.stack_keys = stack_keys
        self.cat_keys = cat_keys
        self.pos_key = pos_key
        self.batch_key = batch_key
        self.progress = progress

    def forward(
        self,
        data: Dict[str, Any],
        predictor: Callable[[Dict[str, Any]], Detection3D],
    ) -> Detection3D:
        for key in (self.pos_key, self.batch_key):
            if key not in data:
                raise KeyError(f"`data` is missing the required key {key!r}.")

        scenes, samples = _split_scenes(data, self.pos_key, self.batch_key)
        views = list(itertools.product(self.scales, self.rotations, self.flips))
        parts: List[Detection3D] = []
        for scale, angle, flips in tqdm(views, desc="TTA", leave=False, disable=not self.progress):
            for start in range(0, len(samples), self.sw_batch_size):
                viewed = [
                    self._apply_view(sample, angle, scale, flips)
                    for sample in samples[start : start + self.sw_batch_size]
                ]
                transformed = [self.transform(sample) for sample in viewed]
                packed = collate(
                    transformed,
                    batch_from=self.pos_key,
                    batch_key=self.batch_key,
                    stack_keys=self.stack_keys,
                    cat_keys=self.cat_keys,
                )
                det = predictor(packed)
                _check_detection_keys(det)
                det = _map_view_back(det, angle, scale, flips)
                det["batch"] = scenes[start : start + self.sw_batch_size][det["batch"]]
                parts.append(det)
        return _cat_detections(parts, like=data[self.pos_key])

    def _apply_view(self, data: Dict[str, Any], angle: float, scale: float, flips: Tuple[int, ...]) -> Dict[str, Any]:
        """Rotate `keys` by `angle` degrees about $z$ and flip them along `flips`; scale `scale_keys` (it commutes)."""
        view = dict(data)
        rotation = F.rotation_matrix(math.radians(angle), axis=2)
        for key in self.keys:
            if key not in view:
                continue

            value = view[key]
            if angle != 0.0:
                value = F.rotate_vectors(value, rotation)
            for axis in flips:
                value = F.flip_vectors(value, axis)
            view[key] = value

        if scale != 1.0:
            for key in self.scale_keys:
                if key in view:
                    view[key] = view[key] * scale
        return view


class TTAInferer(TTASegmentationInferer):
    """Deprecated alias of `TTASegmentationInferer`."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        warnings.warn(
            "`TTAInferer` is deprecated since 0.0.19 and will be removed in 0.0.20; use "
            "`TTASegmentationInferer` instead.",
            FutureWarning,
            stacklevel=2,
        )
        super().__init__(*args, **kwargs)
