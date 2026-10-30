import math
from typing import Any, Callable, Dict, List, Sequence
from unittest.mock import Mock

import pytest
import torch
from torch import Tensor

from torch_pointcloud.inferers import SimpleInferer, TTADetectionInferer, TTASegmentationInferer
from torch_pointcloud.transforms import Compose, RandomFlip, RandomRotate
from torch_pointcloud.utils.data import DataKeys
from torch_pointcloud.utils.types import Detection3D


def _toy_data(n: int = 64, *, seed: int = 0) -> Dict[str, Any]:
    g = torch.Generator().manual_seed(seed)
    return {
        DataKeys.POS: torch.randn(n, 3, generator=g),
        DataKeys.BATCH: torch.zeros(n, dtype=torch.long),
    }


def _pos_logits(num_classes: int) -> Callable[[Dict[str, Any]], Tensor]:
    def predictor(window: Dict[str, Any]) -> Tensor:
        pos = window[DataKeys.POS]
        first = pos[:, :num_classes] if pos.size(1) >= num_classes else pos[:, :1].expand(-1, num_classes)
        return first.contiguous()

    return predictor


def test_tta_single_pass_with_identity_compose_equals_base() -> None:
    """A single TTA pass with a no-op augmentation is equivalent to the base inferer called directly."""
    data = _toy_data()
    base = SimpleInferer()
    predictor = _pos_logits(3)

    inferer = TTASegmentationInferer(base=base, transforms=Compose([]), num_passes=1, aggregate="mean")
    out_tta = inferer(data, predictor=predictor)
    out_base = base(data, predictor=predictor)
    assert torch.allclose(out_tta, out_base)


def test_tta_progress_does_not_change_the_output() -> None:
    data = _toy_data()
    predictor = _pos_logits(num_classes=4)
    quiet = TTASegmentationInferer(base=SimpleInferer(), num_passes=2)(data, predictor=predictor)
    shown = TTASegmentationInferer(base=SimpleInferer(), num_passes=2, progress=True)(data, predictor=predictor)
    assert torch.equal(quiet, shown)


def test_tta_mean_of_identical_passes_equals_one_pass() -> None:
    """With a no-op rotation (angle range collapsed to 0), four passes produce
    four identical logits whose mean equals one pass."""
    data = _toy_data(seed=1)
    base = SimpleInferer()
    predictor = _pos_logits(4)

    noop_rot = Compose([RandomRotate(keys=DataKeys.POS, angle_range=(0.0, 0.0), axis=2, p=1.0)])
    inferer = TTASegmentationInferer(base=base, transforms=noop_rot, num_passes=4, aggregate="mean")
    out_tta = inferer(data, predictor=predictor)
    out_base = base(data, predictor=predictor)
    assert torch.allclose(out_tta, out_base, atol=1e-5)


def test_tta_include_identity_adds_a_clean_pass() -> None:
    """`include_identity=True` prepends one un-augmented pass: with a deterministic flip view the mean is
    the average of the clean and the flipped predictions, i.e. two passes."""
    data = _toy_data(seed=3)
    base = SimpleInferer()
    predictor = _pos_logits(3)
    flip = Compose([RandomFlip(keys=DataKeys.POS, axes=[0], p=1.0)])

    out = TTASegmentationInferer(base=base, transforms=flip, num_passes=1, include_identity=True)(
        data, predictor=predictor
    )
    clean = base(data, predictor=predictor)
    flipped = base(flip(dict(data)), predictor=predictor)
    assert torch.allclose(out, (clean + flipped) / 2.0)

    out_seq = TTASegmentationInferer(base=base, transforms=[flip], include_identity=True)(data, predictor=predictor)
    assert torch.allclose(out_seq, out)


def test_tta_enumerated_sequence_uses_each_view_once() -> None:
    """With a sequence of Composes, `num_passes` is overridden by the sequence length and the aggregate
    equals the mean of the per-view predictions (each view is deterministic, so they can be replayed)."""
    data = _toy_data(seed=2)
    base = SimpleInferer()
    predictor = _pos_logits(3)

    views = [
        Compose([RandomRotate(keys=DataKeys.POS, angle_range=(0.0, 0.0), axis=2, p=1.0)]),
        Compose([RandomRotate(keys=DataKeys.POS, angle_range=(90.0, 90.0), axis=2, p=1.0)]),
        Compose([RandomRotate(keys=DataKeys.POS, angle_range=(180.0, 180.0), axis=2, p=1.0)]),
    ]
    inferer = TTASegmentationInferer(base=base, transforms=views, aggregate="mean")
    assert inferer.num_passes == 3
    out = inferer(data, predictor=predictor)
    assert out.shape == (data[DataKeys.POS].size(0), 3)
    assert torch.isfinite(out).all()

    per_view = torch.stack([base(view(dict(data)), predictor=predictor) for view in views])
    assert torch.allclose(out, per_view.mean(dim=0), atol=1e-6)


def test_tta_flip_changes_predictions_but_preserves_shape() -> None:
    """An always-flip pass alters the per-point predictions but keeps shape."""
    data = _toy_data(seed=3)
    base = SimpleInferer()
    predictor = _pos_logits(3)

    aug = Compose([RandomFlip(keys=DataKeys.POS, axes=[0], p=1.0)])
    inferer = TTASegmentationInferer(base=base, transforms=aug, num_passes=1, aggregate="mean")
    out_tta = inferer(data, predictor=predictor)
    out_base = base(data, predictor=predictor)
    assert out_tta.shape == out_base.shape
    assert not torch.allclose(out_tta, out_base)


def test_tta_ema_aggregation_returns_probabilities() -> None:
    """EMA aggregation over softmaxed per-pass outputs always yields a valid probability distribution."""
    data = _toy_data(seed=4)
    base = SimpleInferer()

    def random_logits(window: Dict[str, Any]) -> Tensor:
        return torch.randn(window[DataKeys.POS].size(0), 5)

    aug = Compose([RandomRotate(keys=DataKeys.POS, angle_range=(0.0, 0.0), axis=2, p=1.0)])
    inferer = TTASegmentationInferer(
        base=base, transforms=aug, num_passes=3, aggregate="ema", ema_smoothing=0.5, softmax=True
    )
    out = inferer(data, predictor=random_logits)
    assert (out >= 0).all()
    sums = out.sum(dim=1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)


def test_tta_without_transforms_votes_over_the_base_inferer() -> None:
    """`transforms=None` runs the base `num_passes` times on the untouched input and averages the passes."""
    data = _toy_data()
    calls: List[int] = []

    def predictor(window: Dict[str, Any]) -> Tensor:
        assert torch.equal(window[DataKeys.POS], data[DataKeys.POS])
        calls.append(len(calls))
        return torch.full((window[DataKeys.POS].size(0), 2), float(len(calls)))

    out = TTASegmentationInferer(base=SimpleInferer(), num_passes=3)(data, predictor=predictor)
    assert len(calls) == 3
    assert torch.allclose(out, torch.full((64, 2), 2.0))
    with pytest.raises(ValueError, match="num_passes"):
        TTASegmentationInferer(base=SimpleInferer())


def test_tta_invalid_num_passes_raises() -> None:
    """`num_passes` must be a positive integer when `transforms` is a single callable."""
    base = SimpleInferer()
    with pytest.raises(ValueError, match="num_passes"):
        TTASegmentationInferer(base=base, transforms=Compose([]), num_passes=None)
    with pytest.raises(ValueError, match="num_passes"):
        TTASegmentationInferer(base=base, transforms=Compose([]), num_passes=0)


def test_tta_invalid_aggregate_raises() -> None:
    """`aggregate` only accepts "mean" or "ema"; unknown values raise ValueError."""
    base = SimpleInferer()
    with pytest.raises(ValueError, match="aggregate"):
        TTASegmentationInferer(base=base, transforms=Compose([]), num_passes=1, aggregate="vote")  # type: ignore[arg-type]


def test_tta_base_of_another_task_raises() -> None:
    """A detection base returns boxes, which the per-point aggregation cannot merge."""
    with pytest.raises(TypeError, match="TTADetectionInferer"):
        TTASegmentationInferer(base=TTADetectionInferer(_identity), num_passes=2)  # type: ignore[arg-type]


def test_tta_empty_sequence_raises() -> None:
    """An empty augmentation sequence has no passes to run and raises ValueError."""
    base = SimpleInferer()
    with pytest.raises(ValueError, match="at least one"):
        TTASegmentationInferer(base=base, transforms=[])


def test_tta_empty_scene_passes_through_base_output() -> None:
    """With $N = 0$ the aggregate of the base inferer's empty outputs is returned unchanged in shape."""
    data: Dict[str, Any] = {
        DataKeys.POS: torch.zeros(0, 3),
        DataKeys.BATCH: torch.zeros(0, dtype=torch.long),
    }

    def predictor(window: Dict[str, Any]) -> Tensor:
        return torch.zeros(0, 5)

    def identity(sample: Dict[str, Any]) -> Dict[str, Any]:
        return dict(sample)

    out = TTASegmentationInferer(base=SimpleInferer(), transforms=identity, num_passes=2)(data, predictor=predictor)
    assert out.shape == (0, 5)


def test_simple_tta_transforms_is_the_13_view_indoor_protocol() -> None:
    """Every rotation at every scale (scales outermost) plus one flip view; identity factors add no transform."""
    from torch_pointcloud.inferers import simple_tta_transforms

    transforms = simple_tta_transforms()
    assert len(transforms) == 13
    assert len(transforms[0].transforms) == 0
    assert [type(t).__name__ for t in transforms[1].transforms] == ["RandomRotate"]
    assert [type(t).__name__ for t in transforms[5].transforms] == ["RandomRotate", "RandomScale"]
    assert [type(t).__name__ for t in transforms[-1].transforms] == ["RandomFlip"]
    assert len(simple_tta_transforms(rotations=(0.0,), scales=(0.9, 1.1), flip=False)) == 2


def test_simple_tta_transforms_flip_acts_on_positions_and_normals_and_skips_missing_keys() -> None:
    """The flip view mirrors both `pos` and `normal`, and still applies to a scene without normals."""
    from torch_pointcloud.inferers import simple_tta_transforms

    data = _toy_data(seed=5)
    data[DataKeys.NORMAL] = torch.randn(64, 3)
    view = simple_tta_transforms(rotations=(0.0,), scales=(1.0,))[-1]
    out = view(dict(data))
    assert torch.allclose(out[DataKeys.POS][:, :2], -data[DataKeys.POS][:, :2])
    assert torch.allclose(out[DataKeys.NORMAL][:, :2], -data[DataKeys.NORMAL][:, :2])
    assert torch.allclose(out[DataKeys.POS][:, 2], data[DataKeys.POS][:, 2])
    out_no_normal = view(_toy_data(seed=5))
    assert DataKeys.NORMAL not in out_no_normal


def _scenes(num_scenes: int, *, seed: int = 0) -> Dict[str, Any]:
    g = torch.Generator().manual_seed(seed)
    return {
        DataKeys.POS: torch.randn(10 * num_scenes, 3, generator=g) + 5.0,
        DataKeys.BATCH: torch.arange(num_scenes).repeat_interleave(10),
    }


def _identity(sample: Dict[str, Any]) -> Dict[str, Any]:
    return sample


def _equivariant_detections(window: Dict[str, Any]) -> Detection3D:
    """One box per scene built from its points, so it moves with them: centered at their mean, sized and headed by
    the offset of the first point, with the planar offset of the second point as velocity."""
    pos, batch = window[DataKeys.POS], window[DataKeys.BATCH]
    boxes: List[Tensor] = []
    velocity: List[Tensor] = []
    for scene in range(int(batch.max()) + 1):
        points = pos[batch == scene]
        center = points.mean(dim=0)
        offset = points[0] - center
        size = offset.norm() * torch.tensor([1.0, 0.5, 0.25], device=pos.device)
        heading = torch.atan2(offset[1], offset[0]).reshape(1)
        boxes.append(torch.cat([center, size, heading]))
        velocity.append((points[1] - center)[:2])

    num_scenes = len(boxes)
    index = torch.arange(num_scenes, device=pos.device)
    return {
        "boxes": torch.stack(boxes),
        "scores": (index + 1) / num_scenes,
        "labels": index,
        "batch": index,
        "ignore_mask": index % 2 == 0,
        "velocity": torch.stack(velocity),
        "class_probs": torch.softmax(torch.arange(3.0 * num_scenes, device=pos.device).reshape(num_scenes, 3), dim=1),
    }


def _no_detections() -> Detection3D:
    return {
        "boxes": torch.zeros(0, 7),
        "scores": torch.zeros(0),
        "labels": torch.zeros(0, dtype=torch.long),
        "batch": torch.zeros(0, dtype=torch.long),
    }


@pytest.mark.parametrize(
    ("rotations", "scales", "flips"),
    [
        pytest.param((30.0,), (1.0,), ((),), id="rotation"),
        pytest.param((0.0,), (1.5,), ((),), id="scale"),
        pytest.param((0.0,), (1.0,), ((0,),), id="flip_x"),
        pytest.param((0.0,), (1.0,), ((1,),), id="flip_y"),
        pytest.param((0.0,), (1.0,), ((0, 1),), id="flip_xy"),
        pytest.param((0.0, 30.0, 90.0), (1.0, 1.2), ((), (0,), (1,), (0, 1)), id="all_combinations"),
    ],
)
def test_tta_detection_maps_every_view_back_to_the_input_scene(
    rotations: Sequence[float],
    scales: Sequence[float],
    flips: Sequence[Sequence[int]],
) -> None:
    """A predictor whose boxes move with the points gives the same boxes, headings and velocities from every view
    once they are mapped back; scores, labels, batch indices, ignore masks and class probabilities are kept."""
    data = _scenes(2)
    expected = _equivariant_detections(data)
    inferer = TTADetectionInferer(_identity, rotations=rotations, scales=scales, flips=flips, sw_batch_size=2)
    out = inferer(data, predictor=_equivariant_detections)

    num_views = len(rotations) * len(scales) * len(flips)
    rows = torch.arange(2).repeat(num_views)
    assert out["boxes"].shape == (2 * num_views, 7)
    torch.testing.assert_close(out["boxes"][:, :6], expected["boxes"][rows, :6], atol=1e-5, rtol=0.0)
    heading_error = out["boxes"][:, 6] - expected["boxes"][rows, 6]
    wrapped_error = torch.atan2(torch.sin(heading_error), torch.cos(heading_error))
    torch.testing.assert_close(wrapped_error, torch.zeros_like(wrapped_error), atol=1e-5, rtol=0.0)
    torch.testing.assert_close(out["velocity"], expected["velocity"][rows], atol=1e-5, rtol=0.0)
    torch.testing.assert_close(out["scores"], expected["scores"][rows])
    torch.testing.assert_close(out["labels"], expected["labels"][rows])
    torch.testing.assert_close(out["batch"], expected["batch"][rows])
    torch.testing.assert_close(out["ignore_mask"], expected["ignore_mask"][rows])
    torch.testing.assert_close(out["class_probs"], expected["class_probs"][rows])


def test_tta_detection_default_is_one_unaugmented_pass() -> None:
    """The default views are the identity alone: one predictor call on the input, whose detections come back as is."""
    data = _scenes(2)
    calls: List[Dict[str, Any]] = []

    def predictor(window: Dict[str, Any]) -> Detection3D:
        calls.append(window)
        return _equivariant_detections(window)

    out = TTADetectionInferer(_identity, sw_batch_size=2)(data, predictor=predictor)
    expected = _equivariant_detections(data)
    assert len(calls) == 1
    assert torch.equal(calls[0][DataKeys.POS], data[DataKeys.POS])
    assert torch.equal(out["boxes"], expected["boxes"])
    assert torch.equal(out["velocity"], expected["velocity"])


def test_tta_detection_views_rotate_and_flip_keys_and_scale_positions_only() -> None:
    """A view rotates positions and normals about $z$, scales the positions only and mirrors both; the views run
    scales outermost, then rotations, then flips."""
    data: Dict[str, Any] = {
        DataKeys.POS: torch.tensor([[1.0, 0.0, 0.5]]),
        DataKeys.NORMAL: torch.tensor([[1.0, 0.0, 0.0]]),
        DataKeys.BATCH: torch.zeros(1, dtype=torch.long),
    }
    seen_pos: List[Tensor] = []
    seen_normal: List[Tensor] = []

    def predictor(window: Dict[str, Any]) -> Detection3D:
        seen_pos.append(window[DataKeys.POS][0])
        seen_normal.append(window[DataKeys.NORMAL][0])
        return _no_detections()

    inferer = TTADetectionInferer(_identity, rotations=(0.0, 90.0), scales=(1.0, 2.0), flips=((), (0,)))
    out = inferer(data, predictor=predictor)

    expected_pos = torch.tensor(
        [
            [1.0, 0.0, 0.5],
            [-1.0, 0.0, 0.5],
            [0.0, 1.0, 0.5],
            [0.0, 1.0, 0.5],
            [2.0, 0.0, 1.0],
            [-2.0, 0.0, 1.0],
            [0.0, 2.0, 1.0],
            [0.0, 2.0, 1.0],
        ]
    )
    expected_normal = torch.tensor([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0]]).repeat(2, 1)
    torch.testing.assert_close(torch.stack(seen_pos), expected_pos, atol=1e-6, rtol=0.0)
    torch.testing.assert_close(torch.stack(seen_normal), expected_normal, atol=1e-6, rtol=0.0)
    assert out["boxes"].shape == (0, 7)


def test_tta_detection_transform_runs_per_scene_after_the_view() -> None:
    """With a `transform`, each scene of each view is transformed on its own after the view, `sw_batch_size` scenes
    are collated per predictor call (with the `cat_keys` scene indices) and the boxes get the scenes' batch indices."""
    data = _scenes(3)
    transformed: List[Dict[str, Any]] = []
    calls: List[Dict[str, Any]] = []

    def transform(sample: Dict[str, Any]) -> Dict[str, Any]:
        transformed.append(sample)
        return {**sample, DataKeys.POS_VOXEL: sample[DataKeys.POS][:3]}

    def predictor(packed: Dict[str, Any]) -> Detection3D:
        calls.append(packed)
        return _equivariant_detections(packed)

    inferer = TTADetectionInferer(
        transform,
        flips=((), (1,)),
        sw_batch_size=2,
        cat_keys=[DataKeys.POS_VOXEL],
    )
    out = inferer(data, predictor=predictor)

    assert len(transformed) == 6
    assert all(DataKeys.BATCH not in sample and sample[DataKeys.POS].shape == (10, 3) for sample in transformed)
    mirrored = data[DataKeys.POS][10:20] * torch.tensor([1.0, -1.0, 1.0])
    torch.testing.assert_close(transformed[4][DataKeys.POS], mirrored)
    assert len(calls) == 4
    assert calls[0][f"batch_{DataKeys.POS_VOXEL}"].tolist() == [0, 0, 0, 1, 1, 1]
    assert out["batch"].tolist() == [0, 1, 2, 0, 1, 2]
    expected = _equivariant_detections(data)
    torch.testing.assert_close(out["boxes"][:, :6], expected["boxes"][out["batch"], :6], atol=1e-5, rtol=0.0)


def test_tta_detection_empty_input_with_a_transform_returns_no_boxes() -> None:
    """Without points there is no scene to transform: the predictor is never called and the set is empty."""
    data: Dict[str, Any] = {
        DataKeys.POS: torch.zeros(0, 3),
        DataKeys.BATCH: torch.zeros(0, dtype=torch.long),
    }
    predictor = Mock()
    out = TTADetectionInferer(_identity, flips=((), (1,)))(data, predictor=predictor)
    predictor.assert_not_called()
    assert out["boxes"].shape == (0, 7)
    assert out["batch"].dtype == torch.long


def test_tta_detection_splits_packed_keys_by_their_scene_index() -> None:
    """A collated batch splits into its scenes: ground-truth boxes by `batch_box`, whose scene index is dropped."""
    data = {
        **_scenes(2),
        DataKeys.BOX: torch.arange(3.0).unsqueeze(1).expand(3, 7),
        f"batch_{DataKeys.BOX}": torch.tensor([0, 0, 1]),
    }
    transformed: List[Dict[str, Any]] = []

    def transform(sample: Dict[str, Any]) -> Dict[str, Any]:
        transformed.append(sample)
        return sample

    TTADetectionInferer(transform, cat_keys=[DataKeys.BOX])(data, predictor=_equivariant_detections)

    assert [sample[DataKeys.BOX][:, 0].tolist() for sample in transformed] == [[0.0, 1.0], [2.0]]
    assert all(f"batch_{DataKeys.BOX}" not in sample for sample in transformed)


def test_tta_detection_unknown_predictor_key_raises() -> None:
    """A per-box entry the inversion does not know could be geometric, so it raises instead of passing through."""

    def predictor(window: Dict[str, Any]) -> Any:
        return {**_equivariant_detections(window), "direction_scores": torch.zeros(1, 2)}

    with pytest.raises(ValueError, match="direction_scores"):
        TTADetectionInferer(_identity)(_scenes(2), predictor=predictor)


def test_tta_detection_predictor_keys_changing_across_calls_raise() -> None:
    """Merging calls that disagree on the optional keys would drop or misalign them, so it raises."""
    calls: List[int] = []

    def predictor(window: Dict[str, Any]) -> Any:
        calls.append(1)
        det = _equivariant_detections(window)
        return det if len(calls) == 1 else {key: value for key, value in det.items() if key != "velocity"}

    with pytest.raises(ValueError, match="different keys"):
        TTADetectionInferer(_identity, flips=((), (1,)))(_scenes(1), predictor=predictor)


def test_tta_detection_splits_stacked_keys_by_scene_row() -> None:
    """Each scene gets its own row of a stacked key, and every predictor call the rows of its scenes."""
    calib = torch.arange(3.0).view(3, 1, 1).expand(3, 3, 4)
    data = {**_scenes(3), DataKeys.CALIB: calib}
    transformed: List[Tensor] = []
    seen: List[Tensor] = []

    def transform(sample: Dict[str, Any]) -> Dict[str, Any]:
        transformed.append(sample[DataKeys.CALIB])
        return sample

    def predictor(packed: Dict[str, Any]) -> Detection3D:
        seen.append(packed[DataKeys.CALIB])
        return _equivariant_detections(packed)

    inferer = TTADetectionInferer(transform, sw_batch_size=2, stack_keys=[DataKeys.CALIB])
    inferer(data, predictor=predictor)

    assert [tuple(value.shape) for value in transformed] == [(3, 4), (3, 4), (3, 4)]
    assert [value[0, 0].item() for value in transformed] == [0.0, 1.0, 2.0]
    assert [tuple(value.shape) for value in seen] == [(2, 3, 4), (1, 3, 4)]
    assert torch.equal(seen[1][0], calib[2])


def test_tta_detection_drops_packed_keys_outside_cat_keys() -> None:
    """Ground truth packed without being listed in `cat_keys` belongs to no scene sample and never reaches the
    predictor."""
    data = {
        **_scenes(2),
        DataKeys.BOX: torch.zeros(3, 7),
        f"batch_{DataKeys.BOX}": torch.tensor([0, 0, 1]),
    }
    transformed: List[Dict[str, Any]] = []

    def transform(sample: Dict[str, Any]) -> Dict[str, Any]:
        transformed.append(sample)
        return sample

    TTADetectionInferer(transform)(data, predictor=_equivariant_detections)
    assert all(set(sample) == {DataKeys.POS} for sample in transformed)


def test_tta_detection_keeps_the_scene_ids_of_a_sparse_unsorted_batch() -> None:
    """Scenes are the ids present in `batch`, interleaved or not; an id without points gets no predictor call."""
    pos = torch.randn(20, 3)
    batch = torch.tensor([0, 5] * 10)
    data: Dict[str, Any] = {DataKeys.POS: pos, DataKeys.BATCH: batch}
    out = TTADetectionInferer(_identity, flips=((), (1,)), sw_batch_size=2)(data, predictor=_equivariant_detections)

    assert out["batch"].tolist() == [0, 5, 0, 5]
    torch.testing.assert_close(out["boxes"][[0, 2], :3], pos[batch == 0].mean(dim=0).expand(2, 3))


def test_tta_detection_wraps_the_headings_of_rotated_and_flipped_views() -> None:
    """An x flip maps a heading t to pi - t and a rotation shifts it, so those views come back wrapped to
    [-pi, pi); the identity view returns the predictor's heading as is."""

    def predictor(window: Dict[str, Any]) -> Detection3D:
        det = _equivariant_detections(window)
        det["boxes"][:, 6] = -3.0
        return det

    inferer = TTADetectionInferer(_identity, rotations=(0.0, 170.0), flips=((), (0,)))
    out = inferer(_scenes(1), predictor=predictor)

    headings = out["boxes"][:, 6]
    assert headings[0].item() == -3.0
    assert bool(((headings[1:] >= -math.pi) & (headings[1:] < math.pi)).all())
    torch.testing.assert_close(headings[1], torch.tensor(math.pi + 3.0 - 2 * math.pi))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_tta_detection_predictor_on_another_device_returns_on_the_input_device() -> None:
    """A scene kept on the CPU with the model on the GPU: the boxes come back on the CPU, equal to a CPU run."""
    data = _scenes(2)

    def predictor(packed: Dict[str, Any]) -> Detection3D:
        cuda = {key: value.cuda() if isinstance(value, Tensor) else value for key, value in packed.items()}
        return _equivariant_detections(cuda)

    inferer = TTADetectionInferer(_identity, flips=((), (1,)), sw_batch_size=2)
    out = inferer(data, predictor=predictor)
    expected = inferer(data, predictor=_equivariant_detections)

    torch.testing.assert_close(dict(out), dict(expected))


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        pytest.param({"rotations": ()}, "at least one", id="no_rotation"),
        pytest.param({"scales": (1.0, 0.0)}, "> 0", id="zero_scale"),
        pytest.param({"flips": ((2,),)}, "distinct axes", id="flip_up_axis"),
        pytest.param({"flips": ((0, 0),)}, "distinct axes", id="repeated_flip_axis"),
        pytest.param({"sw_batch_size": 0}, "sw_batch_size", id="zero_sw_batch_size"),
        pytest.param({"keys": (DataKeys.NORMAL,)}, "pos_key", id="positions_not_rotated"),
        pytest.param({"scale_keys": ()}, "pos_key", id="positions_not_scaled"),
        pytest.param({"pos_key": "coord"}, "pos_key", id="other_pos_key"),
    ],
)
def test_tta_detection_invalid_arguments_raise(kwargs: Dict[str, Any], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        TTADetectionInferer(_identity, **kwargs)
