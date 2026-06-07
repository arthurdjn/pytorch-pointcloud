# mypy: disable-error-code="arg-type,call-overload"
from pathlib import Path
from typing import Callable
from unittest.mock import Mock

import numpy as np
import pytest
import torch

from torch_pointcloud.datasets import NuScenesMini
from torch_pointcloud.datasets.nuscenes import (
    NUSCENES_DETECTION_CLASSES,
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
    num_points = sample["pos"].shape[0]
    assert num_points > 0
    assert sample["pos"].shape == (num_points, 3)
    assert sample["intensity"].shape == (num_points, 1)
    assert sample["timestamp"].shape == (num_points, 1)


def test_nuscenes_dataset_dtypes(datasets_dir_factory: Callable[..., Path]) -> None:
    """Test that returned tensors have the expected dtypes"""
    datasets_dir = datasets_dir_factory("NuScenesMini/**/*")
    sample = NuScenesMini(root=datasets_dir)[0]
    assert sample["pos"].dtype == torch.float32
    assert sample["intensity"].dtype == torch.float32
    assert sample["timestamp"].dtype == torch.float32
    assert sample[DataKeys.BOX].dtype == torch.float32
    assert sample[DataKeys.LABEL].dtype == torch.int64


def test_nuscenes_dataset_boxes(datasets_dir_factory: Callable[..., Path]) -> None:
    """Test that ground-truth boxes are 7-DoF and labels index into the detection classes"""
    datasets_dir = datasets_dir_factory("NuScenesMini/**/*")
    sample = NuScenesMini(root=datasets_dir)[0]
    boxes, labels = sample[DataKeys.BOX], sample[DataKeys.LABEL]
    assert boxes.shape[1] == 7
    assert boxes.shape[0] == labels.shape[0] >= 1
    assert labels.min() >= 0
    assert labels.max() < len(NUSCENES_DETECTION_CLASSES)


def test_nuscenes_dataset_sweep_aggregation(datasets_dir_factory: Callable[..., Path]) -> None:
    """Test that the keyframe carries dt == 0 and prior sweeps add points with a positive time lag"""
    datasets_dir = datasets_dir_factory("NuScenesMini/**/*")
    aggregated = NuScenesMini(root=datasets_dir)[0]
    keyframe_only = NuScenesMini(root=datasets_dir, max_sweeps=1)[0]
    assert aggregated["pos"].shape[0] > keyframe_only["pos"].shape[0]
    assert torch.equal(keyframe_only["timestamp"], torch.zeros_like(keyframe_only["timestamp"]))
    assert (aggregated["timestamp"] == 0).any()
    assert aggregated["timestamp"].max() > 0


def test_nuscenes_dataset_transform(datasets_dir_factory: Callable[..., Path]) -> None:
    """Test that the transform is called once per sample"""
    datasets_dir = datasets_dir_factory("NuScenesMini/**/*")
    transform = Mock(side_effect=lambda data: data)
    dataset = NuScenesMini(root=datasets_dir, transform=transform)
    _ = list(dataset)
    assert transform.call_count == len(dataset)
