# mypy: disable-error-code="arg-type,call-overload"
import json
import shutil
from pathlib import Path
from typing import Callable
from unittest.mock import Mock, patch

import numpy as np
import pytest
import torch

from torch_pointcloud.datasets import KITTI
from torch_pointcloud.datasets.kitti import KITTI_CLASSES, fov_flag, lidar_to_rect, load_kitti_calib, rect_to_lidar
from torch_pointcloud.utils.data import DataKeys

_IDENTITY_CALIB = "P2: 1 0 0 0 0 1 0 0 0 0 1 0\nR0_rect: 1 0 0 0 1 0 0 0 1\nTr_velo_to_cam: 1 0 0 0 0 1 0 0 0 0 1 0\n"


def test_kitti_calibration_rect_lidar_roundtrip(datasets_dir_factory: Callable[..., Path]) -> None:
    """rect_to_lidar inverts lidar_to_rect for the real fixture calibration."""
    datasets_dir = datasets_dir_factory("KITTI/**/*")
    calib = load_kitti_calib(datasets_dir / "KITTI" / "raw" / "training" / "calib" / "000000.txt")
    points = np.array([[1.0, 2.0, 30.0], [-4.0, 1.5, 12.0]], dtype=np.float32)
    assert np.allclose(rect_to_lidar(lidar_to_rect(points, calib), calib), points, atol=1e-5)


def test_fov_flag_keeps_front_drops_behind(tmp_path: Path) -> None:
    """fov_flag keeps points that project into the image and drops those behind or out of bounds."""
    calib_file = tmp_path / "calib.txt"
    calib_file.write_text(_IDENTITY_CALIB)
    calib = load_kitti_calib(calib_file)
    # Identity calibration: a point projects to (x/z, y/z) at depth z. Row 0 is in front and in
    # bounds; row 1 is behind the camera; row 2 projects past the image width.
    points = np.array([[1.0, 1.0, 5.0, 0.0], [1.0, 1.0, -5.0, 0.0], [10000.0, 0.0, 5.0, 0.0]], dtype=np.float32)
    mask = fov_flag(points, (375, 1242), calib)
    assert mask.tolist() == [True, False, False]


def test_kitti_dataset_not_found() -> None:
    """Raises an error if the dataset is not found"""
    with pytest.raises(RuntimeError, match="not found"):
        _ = KITTI(root="not-found", train=True, fov=False)


def test_kitti_dataset_download_unsupported(datasets_dir_factory: Callable[..., Path]) -> None:
    """download=True raises because KITTI must be downloaded manually."""
    datasets_dir = datasets_dir_factory("KITTI/**/*")
    with pytest.raises(RuntimeError, match="does not support automatic download"):
        _ = KITTI(root=datasets_dir, train=True, fov=False, download=True)


def test_kitti_dataset_shapes(datasets_dir_factory: Callable[..., Path]) -> None:
    """Test that samples load with the expected keys and shapes"""
    datasets_dir = datasets_dir_factory("KITTI/**/*")
    dataset = KITTI(root=datasets_dir, train=True, fov=False)
    assert len(dataset) == 3

    sample = dataset[0]
    assert sample[DataKeys.POS].shape == (1024, 3)
    assert sample[DataKeys.INTENSITY].shape == (1024, 1)
    assert sample["frame"] == "000000"


def test_kitti_dataset_dtypes(datasets_dir_factory: Callable[..., Path]) -> None:
    """Test that returned tensors have the expected dtypes"""
    datasets_dir = datasets_dir_factory("KITTI/**/*")
    sample = KITTI(root=datasets_dir, train=True, fov=False)[0]
    assert sample[DataKeys.POS].dtype == torch.float32
    assert sample[DataKeys.INTENSITY].dtype == torch.float32
    assert sample[DataKeys.BOX].dtype == torch.float32
    assert sample[DataKeys.LABEL].dtype == torch.int64


def test_kitti_dataset_boxes(datasets_dir_factory: Callable[..., Path]) -> None:
    """Test that ground-truth boxes are 7-DoF and labels are raw indices into the KITTI classes"""
    datasets_dir = datasets_dir_factory("KITTI/**/*")
    sample = KITTI(root=datasets_dir, train=True, fov=False)[0]
    boxes, labels = sample[DataKeys.BOX], sample[DataKeys.LABEL]
    assert boxes.shape[1] == 7
    assert boxes.shape[0] == labels.shape[0] >= 1
    assert labels.min() >= 0
    assert labels.max() < len(KITTI_CLASSES)


def test_kitti_dataset_raw_boxes_and_fields(datasets_dir_factory: Callable[..., Path]) -> None:
    """The dataset returns raw classes (not the detection subset) plus per-box truncation/occlusion/2D height."""
    datasets_dir = datasets_dir_factory("KITTI/**/*")
    sample = KITTI(root=datasets_dir, train=True, fov=False)[0]
    # Frame 000000 holds a single Pedestrian (raw index 3) with truncation 0, occlusion 0, height 307.92 - 143.00.
    assert sample[DataKeys.LABEL].tolist() == [KITTI_CLASSES.index("Pedestrian")]
    assert sample[DataKeys.TRUNCATION].tolist() == [0.0]
    assert sample[DataKeys.OCCLUSION].tolist() == [0]
    assert sample[DataKeys.BBOX_HEIGHT].item() == pytest.approx(164.92, abs=1e-2)
    assert sample[DataKeys.OCCLUSION].dtype == torch.int64
    assert sample[DataKeys.TRUNCATION].dtype == torch.float32


def test_kitti_dataset_caches_npy(datasets_dir_factory: Callable[..., Path]) -> None:
    """Processing writes a per-frame .npy cache under processed/<split>/<frame>/ plus a completion marker."""
    datasets_dir = datasets_dir_factory("KITTI/**/*")
    _ = KITTI(root=datasets_dir, train=True, fov=False)
    processed = datasets_dir / "KITTI" / "processed" / "training"
    assert sorted(p.name for p in processed.glob("*")) == ["000000", "000001", "000002", "meta.json"]
    assert json.loads((processed / "meta.json").read_text())["format_version"] == 1
    assert {p.name for p in (processed / "000000").glob("*.npy")} == {
        "pos.npy",
        "intensity.npy",
        "boxes.npy",
        "labels.npy",
        "truncation.npy",
        "occlusion.npy",
        "bbox_height.npy",
    }


def test_kitti_dataset_legacy_cache_without_marker_loads(datasets_dir_factory: Callable[..., Path]) -> None:
    """A complete cache without a completion marker (legacy layout) still loads without reprocessing."""
    datasets_dir = datasets_dir_factory("KITTI/**/*")
    _ = KITTI(root=datasets_dir, train=True, fov=False)
    (datasets_dir / "KITTI" / "processed" / "training" / "meta.json").unlink()
    with patch("torch_pointcloud.datasets.kitti.parallel_map") as mock_map:
        dataset = KITTI(root=datasets_dir, train=True, fov=False)
    mock_map.assert_not_called()
    assert len(dataset) == 3


def test_kitti_dataset_interrupted_cache_detected(datasets_dir_factory: Callable[..., Path]) -> None:
    """An unmarked cache with a torn frame raises instead of silently loading."""
    datasets_dir = datasets_dir_factory("KITTI/**/*")
    dataset = KITTI(root=datasets_dir, train=True, fov=False)
    (dataset.processed_split_dir / "meta.json").unlink()
    (dataset.processed_split_dir / "000001" / "labels.npy").unlink()

    with pytest.raises(RuntimeError, match="force_process"):
        _ = KITTI(root=datasets_dir, train=True, fov=False)

    reprocessed = KITTI(root=datasets_dir, train=True, fov=False, force_process=True)
    assert len(reprocessed) == 3


def test_kitti_dataset_missing_frame_detected(datasets_dir_factory: Callable[..., Path]) -> None:
    """An unmarked cache missing a raw frame raises instead of silently loading a partial split."""
    datasets_dir = datasets_dir_factory("KITTI/**/*")
    dataset = KITTI(root=datasets_dir, train=True, fov=False)
    (dataset.processed_split_dir / "meta.json").unlink()
    shutil.rmtree(dataset.processed_split_dir / "000002")

    with pytest.raises(RuntimeError, match="000002"):
        _ = KITTI(root=datasets_dir, train=True, fov=False)


def test_kitti_dataset_reuses_cache(datasets_dir_factory: Callable[..., Path]) -> None:
    """A second construction reuses the existing cache instead of reprocessing."""
    datasets_dir = datasets_dir_factory("KITTI/**/*")
    _ = KITTI(root=datasets_dir, train=True, fov=False)
    with patch("torch_pointcloud.datasets.kitti.parallel_map") as mock_map:
        _ = KITTI(root=datasets_dir, train=True, fov=False)
    mock_map.assert_not_called()


def test_kitti_dataset_force_process(datasets_dir_factory: Callable[..., Path]) -> None:
    """force_process reprocesses the split even when a cache already exists."""
    datasets_dir = datasets_dir_factory("KITTI/**/*", symlinks=False)
    _ = KITTI(root=datasets_dir, train=True, fov=False)
    with patch("torch_pointcloud.datasets.kitti.parallel_map") as mock_map:
        _ = KITTI(root=datasets_dir, train=True, fov=False, force_process=True)
    mock_map.assert_called_once()


def test_kitti_dataset_fov_filter(datasets_dir_factory: Callable[..., Path]) -> None:
    """Test that fov=True bakes the fov_flag mask into the cache when the camera image is present"""
    datasets_dir = datasets_dir_factory("KITTI/**/*")
    image = datasets_dir / "KITTI" / "raw" / "training" / "image_2" / "000000.png"
    image.parent.mkdir(parents=True, exist_ok=True)
    image.touch()
    mock_fov = Mock(side_effect=lambda points, image_shape, calib: np.arange(len(points)) < 5)
    with patch("torch_pointcloud.datasets.kitti.fov_flag", mock_fov):
        sample = KITTI(root=datasets_dir, train=True, fov=True)[0]
    assert mock_fov.called
    assert sample[DataKeys.POS].shape == (5, 3)


def test_kitti_dataset_split_file(datasets_dir_factory: Callable[..., Path]) -> None:
    """Test that a split file restricts the loaded frames to the listed ids"""
    datasets_dir = datasets_dir_factory("KITTI/**/*")
    split_file = datasets_dir / "val.txt"
    split_file.write_text("000000\n000002\n")
    dataset = KITTI(root=datasets_dir, train=True, split_file=split_file, fov=False)
    assert [dataset[i]["frame"] for i in range(len(dataset))] == ["000000", "000002"]


def test_kitti_dataset_split_file_missing_frames_raise(datasets_dir_factory: Callable[..., Path]) -> None:
    """A split file referencing frames absent from the cache raises listing the missing ids."""
    datasets_dir = datasets_dir_factory("KITTI/**/*")
    split_file = datasets_dir / "val.txt"
    split_file.write_text("000000\n000099\n")
    with pytest.raises(RuntimeError, match="000099"):
        _ = KITTI(root=datasets_dir, train=True, split_file=split_file, fov=False)


def test_kitti_dataset_transform(datasets_dir_factory: Callable[..., Path]) -> None:
    """Test that the transform is called once per sample"""
    datasets_dir = datasets_dir_factory("KITTI/**/*")
    transform = Mock(side_effect=lambda data: data)
    dataset = KITTI(root=datasets_dir, train=True, transform=transform, fov=False)
    _ = list(dataset)
    assert transform.call_count == len(dataset)
