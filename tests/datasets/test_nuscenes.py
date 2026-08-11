# mypy: disable-error-code="arg-type,call-overload"
import json
from pathlib import Path
from typing import Callable
from unittest.mock import Mock, patch

import numpy as np
import pytest
import torch

from torch_pointcloud.datasets import NuScenesMini
from torch_pointcloud.datasets.nuscenes import (
    NUSCENES_ATTRIBUTES,
    NUSCENES_DETECTION_CLASSES,
    _annotation_velocity,
    _pose_matrix,
    _remove_ego_points,
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
