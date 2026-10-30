# mypy: disable-error-code="arg-type,call-overload"
import json
import shutil
from pathlib import Path
from typing import Any, Callable, Dict, Optional
from unittest.mock import Mock, patch

import numpy as np
import pytest
import torch

from torch_pointcloud.datasets import NuScenes, NuScenesMini
from torch_pointcloud.datasets.nuscenes import (
    NUSCENES_ATTRIBUTES,
    NUSCENES_DETECTION_CLASSES,
    NUSCENES_LIDARSEG_CLASSES,
    NUSCENES_LIDARSEG_LABEL_NAMES,
    NUSCENES_LIDARSEG_LEARNING_MAP,
    NUSCENES_MINI_TRAIN_SCENES,
    NUSCENES_MINI_VAL_SCENES,
    NUSCENES_VAL_SCENES,
    _annotation_velocity,
    _pose_matrix,
    _remove_ego_points,
    load_nuscenes_sweeps,
)
from torch_pointcloud.utils.data import DataKeys


def test_pose_matrix_combines_rotation_and_translation() -> None:
    """_pose_matrix places the rotation and translation into a homogeneous transform."""
    matrix = _pose_matrix([1.0, 2.0, 3.0], [1.0, 0.0, 0.0, 0.0])
    assert np.allclose(matrix[:3, :3], np.eye(3))
    assert np.allclose(matrix[:3, 3], [1.0, 2.0, 3.0])
    assert np.allclose(matrix[3], [0.0, 0.0, 0.0, 1.0])


def test_remove_ego_points_drops_near_origin() -> None:
    """_remove_ego_points discards points within the ego radius on the x/y plane."""
    points = np.array([[0.0, 0.0, 1.0, 0.0], [5.0, 5.0, 1.0, 0.0], [0.5, -0.5, 1.0, 0.0]], dtype=np.float32)
    kept = _remove_ego_points(points)
    assert kept.shape == (1, 4)
    assert np.allclose(kept[0, :2], [5.0, 5.0])


def test_annotation_velocity_finite_difference() -> None:
    """_annotation_velocity divides the neighbor translation delta by the sample-timestamp delta."""
    ann_by_token = {
        "a": {"token": "a", "prev": "", "next": "b", "sample_token": "sa", "translation": [0.0, 0.0, 0.0]},
        "b": {"token": "b", "prev": "a", "next": "c", "sample_token": "sb", "translation": [1.0, 2.0, 0.0]},
        "c": {"token": "c", "prev": "b", "next": "", "sample_token": "sc", "translation": [2.0, 4.0, 0.0]},
    }
    timestamps = {"sa": 10.0, "sb": 10.5, "sc": 11.0}
    assert np.allclose(_annotation_velocity(ann_by_token["b"], ann_by_token, timestamps), [2.0, 4.0, 0.0])
    assert np.allclose(_annotation_velocity(ann_by_token["a"], ann_by_token, timestamps), [2.0, 4.0, 0.0])
    assert np.allclose(_annotation_velocity(ann_by_token["c"], ann_by_token, timestamps), [2.0, 4.0, 0.0])


def test_annotation_velocity_without_neighbors_is_zero() -> None:
    """No prev/next neighbor (or a dangling token) yields a zero velocity."""
    isolated = {"token": "a", "prev": "", "next": "", "sample_token": "sa", "translation": [3.0, 1.0, 0.0]}
    assert np.allclose(_annotation_velocity(isolated, {"a": isolated}, {"sa": 0.0}), 0.0)
    dangling = {"token": "b", "prev": "missing", "next": "", "sample_token": "sb", "translation": [3.0, 1.0, 0.0]}
    assert np.allclose(_annotation_velocity(dangling, {"b": dangling}, {"sb": 0.0}), 0.0)


def test_nuscenes_dataset_not_found() -> None:
    """Raises an error if the dataset is not found"""
    with pytest.raises(RuntimeError, match="not found"):
        _ = NuScenesMini(root="not-found")


def test_nuscenes_dataset_shapes(datasets_dir_factory: Callable[..., Path]) -> None:
    """Test that samples load with the expected keys and consistent shapes"""
    datasets_dir = datasets_dir_factory("NuScenesMini/**/*")
    dataset = NuScenesMini(root=datasets_dir)
    assert len(dataset) == 2

    sample = dataset[0]
    num_points = sample[DataKeys.POS].shape[0]
    assert num_points > 0
    assert sample[DataKeys.POS].shape == (num_points, 3)
    assert sample[DataKeys.INTENSITY].shape == (num_points, 1)
    assert sample[DataKeys.TIMESTAMP].shape == (num_points, 1)


def test_nuscenes_dataset_dtypes(datasets_dir_factory: Callable[..., Path]) -> None:
    """Test that returned tensors have the expected dtypes"""
    datasets_dir = datasets_dir_factory("NuScenesMini/**/*")
    sample = NuScenesMini(root=datasets_dir)[0]
    assert sample[DataKeys.POS].dtype == torch.float32
    assert sample[DataKeys.INTENSITY].dtype == torch.float32
    assert sample[DataKeys.TIMESTAMP].dtype == torch.float32
    assert sample[DataKeys.BOX].dtype == torch.float32
    assert sample[DataKeys.LABEL].dtype == torch.int64
    assert sample[DataKeys.VELOCITY].dtype == torch.float32
    assert sample[DataKeys.NUM_POINTS].dtype == torch.int64
    assert sample[DataKeys.ATTRIBUTE].dtype == torch.int64


def test_nuscenes_dataset_boxes(datasets_dir_factory: Callable[..., Path]) -> None:
    """Test that ground-truth boxes are 7-DoF and labels index into the detection classes"""
    datasets_dir = datasets_dir_factory("NuScenesMini/**/*")
    sample = NuScenesMini(root=datasets_dir)[0]
    boxes, labels = sample[DataKeys.BOX], sample[DataKeys.LABEL]
    assert boxes.shape[1] == 7
    assert boxes.shape[0] == labels.shape[0] >= 1
    assert labels.min() >= 0
    assert labels.max() < len(NUSCENES_DETECTION_CLASSES)


def test_nuscenes_dataset_box_extras(datasets_dir_factory: Callable[..., Path]) -> None:
    """Per-box velocity, LiDAR point count and attribute id align with the boxes and stay in range."""
    datasets_dir = datasets_dir_factory("NuScenesMini/**/*")
    sample = NuScenesMini(root=datasets_dir)[0]
    num_boxes = sample[DataKeys.BOX].shape[0]
    velocity = sample[DataKeys.VELOCITY]
    assert velocity.shape == (num_boxes, 2)
    assert torch.isfinite(velocity).all()
    # The fixture annotations all have prev/next neighbors, so velocities resolve to non-zero values.
    assert velocity.abs().sum() > 0
    assert sample[DataKeys.NUM_POINTS].shape == (num_boxes,)
    assert (sample[DataKeys.NUM_POINTS] >= 0).all()
    attribute = sample[DataKeys.ATTRIBUTE]
    assert attribute.shape == (num_boxes,)
    assert (attribute >= -1).all()
    assert (attribute < len(NUSCENES_ATTRIBUTES)).all()


def test_nuscenes_dataset_sweep_aggregation(datasets_dir_factory: Callable[..., Path]) -> None:
    """Test that the keyframe carries dt == 0 and prior sweeps add points with a positive time lag"""
    datasets_dir = datasets_dir_factory("NuScenesMini/**/*")
    aggregated = NuScenesMini(root=datasets_dir)[0]
    keyframe_only = NuScenesMini(root=datasets_dir, max_sweeps=1)[0]
    assert aggregated[DataKeys.POS].shape[0] > keyframe_only[DataKeys.POS].shape[0]
    assert torch.equal(keyframe_only[DataKeys.TIMESTAMP], torch.zeros_like(keyframe_only[DataKeys.TIMESTAMP]))
    assert (aggregated[DataKeys.TIMESTAMP] == 0).any()
    assert aggregated[DataKeys.TIMESTAMP].max() > 0


def test_nuscenes_dataset_transform(datasets_dir_factory: Callable[..., Path]) -> None:
    """Test that the transform is called once per sample"""
    datasets_dir = datasets_dir_factory("NuScenesMini/**/*")
    transform = Mock(side_effect=lambda data: data)
    dataset = NuScenesMini(root=datasets_dir, transform=transform)
    _ = list(dataset)
    assert transform.call_count == len(dataset)


def test_nuscenes_dataset_cache_meta_mismatch_raises(datasets_dir_factory: Callable[..., Path]) -> None:
    """A processed cache written with different classes raises instead of serving mislabeled boxes."""
    datasets_dir = datasets_dir_factory("NuScenesMini/**/*")
    _ = NuScenesMini(root=datasets_dir, classes=("pedestrian", "car"), show_progress=False)

    with pytest.raises(RuntimeError, match="force_process=True"):
        _ = NuScenesMini(root=datasets_dir, show_progress=False)

    dataset = NuScenesMini(root=datasets_dir, show_progress=False, force_process=True)
    assert dataset.classes == NUSCENES_DETECTION_CLASSES


def test_nuscenes_dataset_stale_format_version_raises(datasets_dir_factory: Callable[..., Path]) -> None:
    """A cache from an older format version raises, so the box extras are never silently missing."""
    datasets_dir = datasets_dir_factory("NuScenesMini/**/*")
    dataset = NuScenesMini(root=datasets_dir, show_progress=False)
    meta_path = Path(dataset.processed_dir, "meta.json")
    meta = json.loads(meta_path.read_text())
    meta["format_version"] = 1
    meta_path.write_text(json.dumps(meta))

    with pytest.raises(RuntimeError, match="force_process=True"):
        _ = NuScenesMini(root=datasets_dir, show_progress=False)

    reprocessed = NuScenesMini(root=datasets_dir, show_progress=False, force_process=True)
    assert reprocessed[0][DataKeys.VELOCITY].shape[1] == 2


def test_nuscenes_dataset_legacy_cache_without_meta_loads(datasets_dir_factory: Callable[..., Path]) -> None:
    """A processed cache from before cache metadata existed is accepted as-is."""
    datasets_dir = datasets_dir_factory("NuScenesMini/**/*")
    dataset = NuScenesMini(root=datasets_dir, show_progress=False)
    Path(dataset.processed_dir, "meta.json").unlink()

    reloaded = NuScenesMini(root=datasets_dir, show_progress=False)
    assert len(reloaded) == len(dataset)


def test_nuscenes_dataset_interrupted_process_not_marked_complete(datasets_dir_factory: Callable[..., Path]) -> None:
    """A crash during processing leaves no keyframes sentinel, so the next construction reprocesses."""
    datasets_dir = datasets_dir_factory("NuScenesMini/**/*")
    with patch("torch_pointcloud.datasets.nuscenes.json.dumps", side_effect=RuntimeError("interrupted")):
        with pytest.raises(RuntimeError, match="interrupted"):
            _ = NuScenesMini(root=datasets_dir, show_progress=False)

    dataset = NuScenesMini(root=datasets_dir, show_progress=False)
    assert len(dataset) > 0


def test_velocity_attributes_follow_class_and_speed() -> None:
    """Moving vehicles, cycles and pedestrians get their moving attribute, slow ones the stationary one, and the
    two attribute-free classes get -1."""
    from torch_pointcloud.datasets.nuscenes import (
        NUSCENES_ATTRIBUTES,
        NUSCENES_DETECTION_CLASSES,
        velocity_attributes,
    )

    labels = torch.tensor(
        [NUSCENES_DETECTION_CLASSES.index(name) for name in ("car", "car", "bus", "bicycle", "pedestrian", "barrier")]
    )
    velocity = torch.tensor([[3.0, 0.0], [0.2, 0.0], [0.0, 0.0], [0.0, 2.0], [0.5, 0.5], [9.0, 0.0]])
    attributes = velocity_attributes(labels, velocity)
    names = [NUSCENES_ATTRIBUTES[i] if i >= 0 else None for i in attributes.tolist()]
    assert names == [
        "vehicle.moving",
        "vehicle.parked",
        "vehicle.stopped",
        "cycle.with_rider",
        "pedestrian.standing",
        None,
    ]


def _copy_fixture_as_nuscenes(datasets_dir: Path) -> Path:
    shutil.copytree(datasets_dir / "NuScenesMini" / "raw", datasets_dir / "NuScenes" / "raw")
    return datasets_dir


def test_nuscenes_official_split_lists() -> None:
    """The vendored scene lists match the devkit split sizes and mini_val is part of val."""
    assert len(NUSCENES_VAL_SCENES) == len(set(NUSCENES_VAL_SCENES)) == 150
    assert len(NUSCENES_MINI_TRAIN_SCENES) == 8
    assert len(NUSCENES_MINI_VAL_SCENES) == 2
    assert set(NUSCENES_MINI_VAL_SCENES) <= set(NUSCENES_VAL_SCENES)


def test_nuscenes_unknown_split_raises() -> None:
    """An unknown split name is rejected before any disk access."""
    with pytest.raises(ValueError, match="Unknown nuScenes split"):
        _ = NuScenes(root="not-found", split="validation")
    with pytest.raises(ValueError, match="version"):
        _ = NuScenes(root="not-found", split=None)


def test_nuscenes_split_resolves_version_and_cache_dir(datasets_dir_factory: Callable[..., Path]) -> None:
    """`mini_val` reads `v1.0-mini`, keeps only its scenes and caches under a split-specific directory."""
    datasets_dir = _copy_fixture_as_nuscenes(datasets_dir_factory("NuScenesMini/**/*"))
    dataset = NuScenes(root=datasets_dir, split="mini_val", show_progress=False)
    assert dataset.version == "v1.0-mini"
    assert Path(dataset.processed_dir).name == "v1.0-mini_mini_val_sweeps10"
    assert len(dataset) == 2
    assert dataset[0][DataKeys.BOX].shape[1] == 7


def test_nuscenes_split_without_keyframes_raises(datasets_dir_factory: Callable[..., Path]) -> None:
    """A split whose scenes are absent from the metadata raises instead of producing an empty dataset."""
    datasets_dir = _copy_fixture_as_nuscenes(datasets_dir_factory("NuScenesMini/**/*"))
    with pytest.raises(RuntimeError, match="No LiDAR keyframes"):
        _ = NuScenes(root=datasets_dir, split="mini_train", show_progress=False)


def test_nuscenes_missing_lidar_file_raises(datasets_dir_factory: Callable[..., Path]) -> None:
    """A keyframe whose LiDAR file was not extracted raises before any cache is written."""
    datasets_dir = _copy_fixture_as_nuscenes(datasets_dir_factory("NuScenesMini/**/*"))
    keyframes = sorted((datasets_dir / "NuScenes" / "raw" / "samples" / "LIDAR_TOP").glob("*.bin"))
    keyframes[0].unlink()
    with pytest.raises(RuntimeError, match="1 of 2 LiDAR keyframes are missing"):
        _ = NuScenes(root=datasets_dir, split="mini_val", show_progress=False)


def test_nuscenes_parallel_processing_matches_sequential(datasets_dir_factory: Callable[..., Path]) -> None:
    """Worker processes write the same cache as the sequential path."""
    datasets_dir = _copy_fixture_as_nuscenes(datasets_dir_factory("NuScenesMini/**/*"))
    sequential = NuScenes(root=datasets_dir, split="mini_val", show_progress=False)
    expected = [sequential[i] for i in range(len(sequential))]
    parallel = NuScenes(root=datasets_dir, split="mini_val", show_progress=False, force_process=True, num_workers=2)
    assert parallel.tokens == sequential.tokens
    for index, sample in enumerate(expected):
        for key in (DataKeys.POS, DataKeys.BOX, DataKeys.LABEL, DataKeys.VELOCITY):
            assert torch.equal(parallel[index][key], sample[key])


def test_nuscenes_lidarseg_label_tables() -> None:
    """The raw lidarseg id table covers ids 0..31 and the learning map sends them onto the 16 benchmark classes."""
    assert sorted(NUSCENES_LIDARSEG_LABEL_NAMES) == list(range(32))
    assert len(NUSCENES_LIDARSEG_CLASSES) == len(set(NUSCENES_LIDARSEG_CLASSES)) == 16
    assert set(NUSCENES_LIDARSEG_LEARNING_MAP) <= set(NUSCENES_LIDARSEG_LABEL_NAMES)
    assert sorted(set(NUSCENES_LIDARSEG_LEARNING_MAP.values())) == list(range(len(NUSCENES_LIDARSEG_CLASSES)))


def test_load_nuscenes_sweeps_keeps_keyframe_ego_points(tmp_path: Path) -> None:
    """The keyframe comes first and keeps its ego points; only the prior sweeps drop them."""
    identity = {"translation": [0.0, 0.0, 0.0], "rotation": [1.0, 0.0, 0.0, 0.0]}
    clouds = {
        "keyframe": np.array([[0.5, 0.5, 0.0, 1.0, 0.0], [5.0, 5.0, 0.0, 2.0, 0.0]], dtype=np.float32),
        "sweep": np.array([[0.5, -0.5, 0.0, 3.0, 0.0], [6.0, 6.0, 0.0, 4.0, 0.0]], dtype=np.float32),
    }
    sample_data: Dict[str, Dict[str, Any]] = {}
    for token, prev, timestamp in (("keyframe", "sweep", 1_000_000), ("sweep", "", 950_000)):
        clouds[token].tofile(tmp_path / f"{token}.bin")
        sample_data[token] = {
            "token": token,
            "filename": f"{token}.bin",
            "timestamp": timestamp,
            "prev": prev,
            "ego_pose_token": "pose",
            "calibrated_sensor_token": "calib",
        }

    points = load_nuscenes_sweeps(
        tmp_path,
        sample_data["keyframe"],
        {"pose": identity},
        {"calib": identity},
        sample_data,
        max_sweeps=2,
    )
    assert np.allclose(points[:, :4], [[0.5, 0.5, 0.0, 1.0], [5.0, 5.0, 0.0, 2.0], [6.0, 6.0, 0.0, 4.0]])
    assert np.allclose(points[:, 4], [0.0, 0.0, 0.05])


def test_nuscenes_segment_labels_every_keyframe_point(datasets_dir_factory: Callable[..., Path]) -> None:
    """A keyframe-only sample is the raw scan with one raw lidarseg id per point, and no box keys."""
    datasets_dir = _copy_fixture_as_nuscenes(datasets_dir_factory("NuScenesMini/**/*"))
    dataset = NuScenes(
        root=datasets_dir,
        split="mini_val",
        max_sweeps=1,
        return_boxes=False,
        return_segment=True,
        show_progress=False,
    )
    assert len(dataset) == 2

    sample = dataset[0]
    raw_dir = datasets_dir / "NuScenes" / "raw"
    records = json.loads((raw_dir / "v1.0-mini" / "sample_data.json").read_text())
    record = next(r for r in records if r["is_key_frame"] and r["sample_token"] == sample[DataKeys.TOKEN])
    scan = np.fromfile(raw_dir / record["filename"], dtype=np.float32).reshape(-1, 5)
    assert torch.equal(sample[DataKeys.POS], torch.from_numpy(scan[:, :3]))
    assert torch.equal(sample[DataKeys.INTENSITY], torch.from_numpy(scan[:, 3:4]))

    segment = sample[DataKeys.SEGMENT]
    assert segment.shape == (scan.shape[0],)
    assert segment.dtype == torch.int64
    assert set(segment.unique().tolist()) <= set(NUSCENES_LIDARSEG_LABEL_NAMES)
    # The synthetic scan hits the road and the facade, so both surfaces are labelled.
    assert {24, 28} <= set(segment.unique().tolist())
    assert DataKeys.BOX not in sample
    assert DataKeys.LABEL not in sample


def test_nuscenes_segment_with_sweeps_and_boxes(datasets_dir_factory: Callable[..., Path]) -> None:
    """With sweeps, the keyframe points keep their labels, prior-sweep points get -1 and the boxes stay."""
    datasets_dir = datasets_dir_factory("NuScenesMini/**/*")
    keyframe = NuScenesMini(root=datasets_dir, max_sweeps=1, return_segment=True, show_progress=False)[0]
    sample = NuScenesMini(root=datasets_dir, return_segment=True, show_progress=False)[0]
    num_keyframe_points = keyframe[DataKeys.POS].shape[0]
    assert sample[DataKeys.POS].shape[0] > num_keyframe_points
    assert torch.equal(sample[DataKeys.POS][:num_keyframe_points], keyframe[DataKeys.POS])

    segment = sample[DataKeys.SEGMENT]
    assert segment.shape == (sample[DataKeys.POS].shape[0],)
    assert torch.equal(segment[:num_keyframe_points], keyframe[DataKeys.SEGMENT])
    assert (segment[num_keyframe_points:] == -1).all()
    assert torch.equal(segment == -1, sample[DataKeys.TIMESTAMP][:, 0] > 0)
    assert sample[DataKeys.BOX].shape[1] == 7


def test_nuscenes_segment_without_lidarseg_raises(datasets_dir_factory: Callable[..., Path]) -> None:
    """`return_segment` without `lidarseg.json` raises before processing."""
    datasets_dir = _copy_fixture_as_nuscenes(datasets_dir_factory("NuScenesMini/**/*"))
    (datasets_dir / "NuScenes" / "raw" / "v1.0-mini" / "lidarseg.json").unlink()
    with pytest.raises(RuntimeError, match="nuScenes-lidarseg labels not found"):
        _ = NuScenes(root=datasets_dir, split="mini_val", return_segment=True, show_progress=False)
    assert not (datasets_dir / "NuScenes" / "processed" / "v1.0-mini_mini_val_sweeps10" / "keyframes.json").exists()


def test_nuscenes_missing_label_file_raises(datasets_dir_factory: Callable[..., Path]) -> None:
    """A keyframe whose label file was not extracted raises before any cache is written."""
    datasets_dir = _copy_fixture_as_nuscenes(datasets_dir_factory("NuScenesMini/**/*"))
    label_files = sorted((datasets_dir / "NuScenes" / "raw" / "lidarseg" / "v1.0-mini").glob("*.bin"))
    label_files[0].unlink()
    with pytest.raises(RuntimeError, match="1 of 2 lidarseg label files are missing"):
        _ = NuScenes(root=datasets_dir, split="mini_val", show_progress=False)
    assert not (datasets_dir / "NuScenes" / "processed" / "v1.0-mini_mini_val_sweeps10" / "keyframes.json").exists()


def test_nuscenes_segment_on_cache_without_labels_raises(datasets_dir_factory: Callable[..., Path]) -> None:
    """A cache processed before the lidarseg labels were extracted asks for `force_process=True`."""
    datasets_dir = datasets_dir_factory("NuScenesMini/**/*")
    lidarseg_path = datasets_dir / "NuScenesMini" / "raw" / "v1.0-mini" / "lidarseg.json"
    lidarseg = lidarseg_path.read_text()
    lidarseg_path.unlink()
    _ = NuScenesMini(root=datasets_dir, show_progress=False)
    lidarseg_path.write_text(lidarseg)

    with pytest.raises(RuntimeError, match="No lidarseg labels"):
        _ = NuScenesMini(root=datasets_dir, return_segment=True, show_progress=False)

    dataset = NuScenesMini(root=datasets_dir, return_segment=True, show_progress=False, force_process=True)
    assert dataset[0][DataKeys.SEGMENT].shape == (dataset[0][DataKeys.POS].shape[0],)


def test_nuscenes_mini_process_stamps_cache_meta(datasets_dir_factory: Callable[..., Path]) -> None:
    """Processing writes `cache_meta` next to the cache, with the class's `format_version`."""
    datasets_dir = datasets_dir_factory("NuScenesMini/**/*")
    dataset = NuScenesMini(root=datasets_dir, show_progress=False, force_process=True)
    meta = json.loads(Path(dataset.processed_dir, "meta.json").read_text())
    assert meta == dataset.cache_meta
    assert meta["format_version"] == NuScenesMini.format_version


def test_nuscenes_mini_stale_format_version_raises(datasets_dir_factory: Callable[..., Path]) -> None:
    """A cache stamped with another `format_version` raises instead of being served, until it is regenerated."""
    datasets_dir = datasets_dir_factory("NuScenesMini/**/*")
    dataset = NuScenesMini(root=datasets_dir, show_progress=False, force_process=True)
    meta_path = Path(dataset.processed_dir, "meta.json")
    meta = json.loads(meta_path.read_text())
    meta["format_version"] += 1
    meta_path.write_text(json.dumps(meta))

    with pytest.raises(RuntimeError, match="force_process=True"):
        NuScenesMini(root=datasets_dir, show_progress=False, force_process=False)

    rebuilt = NuScenesMini(root=datasets_dir, show_progress=False, force_process=True)
    assert json.loads(meta_path.read_text()) == rebuilt.cache_meta


def test_nuscenes_mini_cache_without_meta_is_served(datasets_dir_factory: Callable[..., Path]) -> None:
    """A cache from before the stamp existed is served as is and left unstamped."""
    datasets_dir = datasets_dir_factory("NuScenesMini/**/*")
    dataset = NuScenesMini(root=datasets_dir, show_progress=False, force_process=True)
    meta_path = Path(dataset.processed_dir, "meta.json")
    meta_path.unlink()

    legacy = NuScenesMini(root=datasets_dir, show_progress=False, force_process=False)
    assert len(legacy) == len(dataset)
    assert not meta_path.exists()


def test_nuscenes_mini_stamp_without_format_version_is_version_one(datasets_dir_factory: Callable[..., Path]) -> None:
    """A stamp written before `format_version` existed counts as version 1, below NuScenesMini's version, and raises."""
    datasets_dir = datasets_dir_factory("NuScenesMini/**/*")
    dataset = NuScenesMini(root=datasets_dir, show_progress=False, force_process=True)
    meta_path = Path(dataset.processed_dir, "meta.json")
    meta = json.loads(meta_path.read_text())
    del meta["format_version"]
    meta_path.write_text(json.dumps(meta))

    with pytest.raises(RuntimeError, match="force_process=True"):
        NuScenesMini(root=datasets_dir, show_progress=False, force_process=False)


def test_nuscenes_mini_interrupted_rebuild_leaves_no_stamp(
    datasets_dir_factory: Callable[..., Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The stamp is removed before the cache is rewritten, so a failed forced rebuild is never served as complete."""
    datasets_dir = datasets_dir_factory("NuScenesMini/**/*")
    dataset = NuScenesMini(root=datasets_dir, show_progress=False, force_process=True)
    meta_path = Path(dataset.processed_dir, "meta.json")

    def interrupted(self: NuScenesMini, num_workers: Optional[int] = None, show_progress: bool = True) -> bool:
        raise KeyboardInterrupt

    monkeypatch.setattr(NuScenesMini, "write_processed_files", interrupted)
    with pytest.raises(KeyboardInterrupt):
        NuScenesMini(root=datasets_dir, show_progress=False, force_process=True)
    assert not meta_path.exists()


def test_nuscenes_mini_incomplete_cache_is_not_stamped(
    datasets_dir_factory: Callable[..., Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """`write_processed_files` returning False leaves the cache unstamped."""
    datasets_dir = datasets_dir_factory("NuScenesMini/**/*")
    dataset = NuScenesMini(root=datasets_dir, show_progress=False, force_process=True)
    meta_path = Path(dataset.processed_dir, "meta.json")

    def incomplete(self: NuScenesMini, num_workers: Optional[int] = None, show_progress: bool = True) -> bool:
        return False

    monkeypatch.setattr(NuScenesMini, "write_processed_files", incomplete)
    NuScenesMini(root=datasets_dir, show_progress=False, force_process=True)
    assert not meta_path.exists()
