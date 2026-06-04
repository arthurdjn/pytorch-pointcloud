# mypy: disable-error-code="arg-type,call-overload"
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Callable
from unittest.mock import Mock, patch

import numpy as np
import pytest
import torch

from torch_pointcloud.datasets import SunRGBD
from torch_pointcloud.datasets.sunrgbd import (
    SUNRGBD_CLASS_TO_IDX,
    SUNRGBD_CLASSES,
    decode_depth,
    parse_boxes,
    rebase_sequence,
    unproject,
)
from torch_pointcloud.utils.data import DataKeys


@pytest.mark.parametrize(
    "path,expected",
    [
        ("/n/fs/sun3d/data/SUNRGBD/kv1/NYUdata/NYU0001", "kv1/NYUdata/NYU0001"),
        ("//n/fs/sun3d/data/SUNRGBD/kv1/b3do/img_0063", "kv1/b3do/img_0063"),
        ("SUNRGBD/kv1/NYUdata/NYU0001", "kv1/NYUdata/NYU0001"),
        (
            "/n/fs/sun3d/data/SUNRGBD/kv2/kinect2data/000385-resize//depth/0000087.png",
            "kv2/kinect2data/000385-resize/depth/0000087.png",
        ),
    ],
)
def test_rebase_sequence(path: str, expected: str) -> None:
    """Absolute scene paths rebase to a sequence id; internal double slashes collapse to match zip members."""
    assert rebase_sequence(path) == expected


def test_decode_depth_shape_and_dtype() -> None:
    """The bit-shifted 16-bit depth decodes to metric float32 meters."""
    raw = np.array([[8 << 3, 0], [16 << 3, 32 << 3]], dtype=np.uint16)
    depth = decode_depth(raw)
    assert depth.shape == (2, 2)
    assert depth.dtype == np.float32
    assert depth[0, 0] == pytest.approx(8 / 1000.0)
    assert depth[0, 1] == 0.0
    assert depth[1, 0] == pytest.approx(16 / 1000.0)


def test_decode_depth_truncates_at_8m() -> None:
    """Depths beyond 8 m are clamped to the truncation distance."""
    raw = np.array([[1]], dtype=np.uint16)
    depth = decode_depth(raw)
    assert depth[0, 0] == 8.0


def test_unproject_identity_intrinsics() -> None:
    """Pixels unproject into the reordered depth frame [x, z, -y] under identity rtilt."""
    depth = np.array([[1.0, 2.0]], dtype=np.float32)
    k = np.array([[1.0, 0.0, 1.0], [0.0, 1.0, 1.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    rtilt = np.eye(3, dtype=np.float32)
    pts = unproject(depth, k, rtilt)
    assert pts.shape == (2, 3)
    assert pts.dtype == np.float32
    np.testing.assert_allclose(pts[0], [0.0, 1.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(pts[1], [2.0, 2.0, 0.0], atol=1e-6)


def _box(classname: str, orientation: tuple[float, float]) -> SimpleNamespace:
    return SimpleNamespace(
        classname=classname,
        centroid=np.array([1.0, 2.0, 3.0], dtype=np.float32),
        coeffs=np.array([0.5, 0.6, 0.7], dtype=np.float32),
        basis=np.eye(3, dtype=np.float32),
        orientation=np.array(orientation, dtype=np.float32),
        _fieldnames=["classname", "centroid", "coeffs", "basis", "orientation"],
    )


def test_parse_boxes_heading_sign_and_reordered_halfextents() -> None:
    box = _box("chair", (0.9302, -0.3650))
    out = parse_boxes(box, SUNRGBD_CLASS_TO_IDX)
    assert out.shape == (1, 8)
    np.testing.assert_allclose(out[0, :3], [1.0, 2.0, 3.0], atol=1e-6)
    # coeffs [0.5, 0.6, 0.7] -> [c1, c0, c2] = [0.6, 0.5, 0.7], matching votenet's [l, w, h].
    np.testing.assert_allclose(out[0, 3:6], [0.6, 0.5, 0.7], atol=1e-6)
    assert out[0, 6] == pytest.approx(-math.atan2(-0.3650, 0.9302), abs=1e-5)
    assert int(out[0, 7]) == SUNRGBD_CLASS_TO_IDX["chair"]


def test_parse_boxes_filters_unknown_classes() -> None:
    boxes = np.array([_box("chair", (1.0, 0.0)), _box("swivelchair", (1.0, 0.0)), _box("wall", (1.0, 0.0))])
    out = parse_boxes(boxes, SUNRGBD_CLASS_TO_IDX)
    assert out.shape == (1, 8)
    assert int(out[0, 7]) == SUNRGBD_CLASS_TO_IDX["chair"]


def test_parse_boxes_empty_returns_zero_rows() -> None:
    assert parse_boxes(None, SUNRGBD_CLASS_TO_IDX).shape == (0, 8)
    assert parse_boxes(np.array([_box("wall", (1.0, 0.0))]), SUNRGBD_CLASS_TO_IDX).shape == (0, 8)


def test_sunrgbd_class_to_idx_covers_all_classes() -> None:
    """The class-to-index map covers all 10 detection classes in declaration order."""
    expected = {
        "bed": 0,
        "table": 1,
        "sofa": 2,
        "chair": 3,
        "toilet": 4,
        "desk": 5,
        "dresser": 6,
        "night_stand": 7,
        "bookshelf": 8,
        "bathtub": 9,
    }
    assert len(SUNRGBD_CLASSES) == 10
    assert SUNRGBD_CLASS_TO_IDX == expected


def test_sunrgbd_dataset_not_found() -> None:
    """Raises an error if the dataset is not found"""
    with pytest.raises(RuntimeError, match="Dataset not found"):
        _ = SunRGBD(root="not-found", show_progress=False)


def test_sunrgbd_dataset_invalid_split() -> None:
    """Raises an error if the split is invalid or not supported"""
    with pytest.raises(ValueError, match="Invalid split"):
        _ = SunRGBD(root="not-found", split="bogus", show_progress=False)


@pytest.mark.parametrize("split", ["train", "val"])
def test_sunrgbd_dataset_raw_files_exist(datasets_dir_factory: Callable[..., Path], split: str) -> None:
    """Test that the raw files exist"""
    datasets_dir = datasets_dir_factory("SunRGBD/raw/**/*")
    dataset = SunRGBD(root=datasets_dir, split=split, show_progress=False)
    assert dataset.raw_files_exist()


@pytest.mark.parametrize("split", ["train", "val"])
def test_sunrgbd_dataset_processed_files_exist(datasets_dir_factory: Callable[..., Path], split: str) -> None:
    """Test that the processed files exist"""
    datasets_dir = datasets_dir_factory("SunRGBD/processed/**/*")
    dataset = SunRGBD(root=datasets_dir, split=split, show_progress=False)
    assert dataset.processed_files_exist()


@pytest.mark.parametrize("split", ["train", "val"])
@patch.object(SunRGBD, "process_scene", autospec=True, side_effect=SunRGBD.process_scene)
def test_sunrgbd_dataset_already_processed(
    mock_process: Mock, datasets_dir_factory: Callable[..., Path], split: str
) -> None:
    """Test that no scene is re-processed when the processed cache already exists"""
    datasets_dir = datasets_dir_factory("SunRGBD/processed/**/*")

    dataset = SunRGBD(root=datasets_dir, split=split, show_progress=False)
    assert len(dataset) > 0
    _ = list(dataset)

    assert mock_process.call_count == 0


@pytest.mark.parametrize("split", ["train", "val"])
@patch.object(SunRGBD, "process_scene", autospec=True, side_effect=SunRGBD.process_scene)
def test_sunrgbd_dataset_process_split(
    mock_process: Mock, datasets_dir_factory: Callable[..., Path], split: str
) -> None:
    """Test that every scene is processed exactly once when no cache exists"""
    datasets_dir = datasets_dir_factory("SunRGBD/raw/**/*")

    dataset = SunRGBD(root=datasets_dir, split=split, show_progress=False)
    assert len(dataset) > 0

    assert mock_process.call_count == len(dataset)


@pytest.mark.parametrize("split", ["train", "val"])
@patch.object(SunRGBD, "process_scene", autospec=True, side_effect=SunRGBD.process_scene)
def test_sunrgbd_dataset_process_split_forced(
    mock_process: Mock, datasets_dir_factory: Callable[..., Path], split: str
) -> None:
    """Test that `force_process` re-processes every scene even when the cache exists"""
    datasets_dir = datasets_dir_factory("SunRGBD/**/*", symlinks=False)

    dataset = SunRGBD(root=datasets_dir, split=split, show_progress=False, force_process=True)
    assert len(dataset) > 0

    assert mock_process.call_count == len(dataset)


def test_sunrgbd_dataset_progress(
    datasets_dir_factory: Callable[..., Path], capsys: pytest.CaptureFixture[str]
) -> None:
    """Test that the dataset displays a progress bar during processing"""
    datasets_dir = datasets_dir_factory("SunRGBD/raw/**/*")

    dataset = SunRGBD(root=datasets_dir, split="val", show_progress=True)
    assert len(dataset) > 0
    captured = capsys.readouterr()
    assert "Processing" in captured.err
    assert captured.out == ""


def test_sunrgbd_dataset_without_progress(
    datasets_dir_factory: Callable[..., Path], capsys: pytest.CaptureFixture[str]
) -> None:
    """Test that the dataset does not display a progress bar during processing"""
    datasets_dir = datasets_dir_factory("SunRGBD/raw/**/*")

    dataset = SunRGBD(root=datasets_dir, split="val", show_progress=False)
    assert len(dataset) > 0
    captured = capsys.readouterr()
    assert captured.err == ""
    assert captured.out == ""


def test_sunrgbd_dataset_progress_with_cached_processed(
    datasets_dir_factory: Callable[..., Path], capsys: pytest.CaptureFixture[str]
) -> None:
    """Test that no processing progress bar is shown if the processed dataset already exists"""
    datasets_dir = datasets_dir_factory("SunRGBD/processed/**/*")

    dataset = SunRGBD(root=datasets_dir, split="val", show_progress=True)
    assert len(dataset) > 0
    captured = capsys.readouterr()
    assert "Processing" not in captured.err
    assert "Loading" in captured.err
    assert captured.out == ""


def test_sunrgbd_dataset_transform(datasets_dir_factory: Callable[..., Path]) -> None:
    """Test that the dataset is transformed correctly after being processed"""
    datasets_dir = datasets_dir_factory("SunRGBD/processed/**/*")

    transform = Mock(side_effect=lambda data: data)
    dataset = SunRGBD(root=datasets_dir, split="val", transform=transform, show_progress=False)
    _ = list(dataset)
    assert transform.call_count == len(dataset)


def test_sunrgbd_dataset_classes(datasets_dir_factory: Callable[..., Path]) -> None:
    """Test that the dataset exposes the canonical class names and class-to-index map"""
    datasets_dir = datasets_dir_factory("SunRGBD/processed/**/*")

    dataset = SunRGBD(root=datasets_dir, split="val", show_progress=False)
    assert tuple(dataset.classes) == SUNRGBD_CLASSES
    assert dataset.class_to_idx == SUNRGBD_CLASS_TO_IDX


@pytest.mark.parametrize("split", ["train", "val"])
def test_sunrgbd_dataset_loads_processed(datasets_dir_factory: Callable[..., Path], split: str) -> None:
    """Test that processed scenes load with the expected keys, shapes and dtypes"""
    datasets_dir = datasets_dir_factory("SunRGBD/processed/**/*")

    dataset = SunRGBD(root=datasets_dir, split=split, show_progress=False)
    assert len(dataset) == 3

    sample = dataset[0]
    assert sample[DataKeys.POS].shape[1] == 3
    assert sample[DataKeys.POS].dtype == torch.float32
    assert sample[DataKeys.BOX].shape[1] == 8
    assert sample[DataKeys.CLASS].shape[0] == sample[DataKeys.BOX].shape[0]
    assert sample[DataKeys.CLASS].dtype == torch.long
    assert sample[DataKeys.COLOR].shape == sample[DataKeys.POS].shape


def test_sunrgbd_dataset_processes_from_raw(datasets_dir_factory: Callable[..., Path]) -> None:
    """Test that scenes are reconstructed from the raw archives when no cache exists"""
    datasets_dir = datasets_dir_factory("SunRGBD/raw/**/*")

    dataset = SunRGBD(root=datasets_dir, split="val", show_progress=False)
    assert len(dataset) == 3

    sample = dataset[0]
    assert sample[DataKeys.POS].shape[1] == 3
    assert sample[DataKeys.POS].shape[0] > 2048
    assert sample[DataKeys.BOX].shape[1] == 8
    assert sample[DataKeys.BOX].shape[0] == sample[DataKeys.CLASS].shape[0]
    assert sample[DataKeys.BOX].shape[0] >= 1
