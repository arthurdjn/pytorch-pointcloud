# mypy: disable-error-code="arg-type,call-overload,attr-defined"
from pathlib import Path
from typing import Callable
from unittest.mock import Mock, patch

import pytest
import torch

from torch_pointcloud.datasets import S3DIS
from torch_pointcloud.datasets.s3dis import S3DIS_UNK_IDX, load_s3dis_room_data


def test_load_s3dis_room_data(data_dir: Path) -> None:
    """Test that the S3DIS room data is loaded correctly"""
    room_dir = data_dir / "S3DIS" / "raw" / "Area_1" / "conferenceRoom_1"
    data = load_s3dis_room_data(room_dir)

    assert isinstance(data["pos"], torch.Tensor)
    assert isinstance(data["color"], torch.Tensor)
    assert isinstance(data["instance"], torch.Tensor)
    assert isinstance(data["segment"], torch.Tensor)
    assert data["pos"].shape[1] == 3
    assert data["color"].shape[1] == 3
    assert data["instance"].ndim == 1
    assert data["segment"].ndim == 1


def test_s3dis_dataset_not_found() -> None:
    """Raises an error if the dataset is not found"""
    with pytest.raises(RuntimeError, match="Dataset not found"):
        _ = S3DIS(root="not-found", show_progress=False)


@pytest.mark.parametrize("areas", [["Area_1"], ["Area_1", "Area_2"], "all"])
def test_s3dis_dataset_raw_files_exist(data_dir_factory: Callable[..., Path], areas: str | list[str]) -> None:
    """Test that the raw files exist"""
    data_dir = data_dir_factory("S3DIS/raw/**/*")
    dataset = S3DIS(root=data_dir, areas=areas, show_progress=False)
    assert dataset.raw_files_exist()


@pytest.mark.parametrize("areas", [["Area_1"], ["Area_1", "Area_2"], "all"])
def test_s3dis_dataset_raw_files_not_exist(areas: str | list[str]) -> None:
    """Test that an error is raised if the raw files do not exist"""
    with pytest.raises(RuntimeError, match="Dataset not found"):
        _ = S3DIS(root="not-found", areas=areas, show_progress=False)


@pytest.mark.parametrize("areas", [["Area_1"], ["Area_1", "Area_2"], "all"])
def test_s3dis_dataset_processed_files_exist(data_dir_factory: Callable[..., Path], areas: str | list[str]) -> None:
    """Test that the processed files exist"""
    data_dir = data_dir_factory("S3DIS/processed/**/*")
    dataset = S3DIS(root=data_dir, areas=areas, show_progress=False)
    assert dataset.processed_files_exist()


@pytest.mark.parametrize("areas", [["Area_1"], ["Area_1", "Area_2"], "all"])
@patch("torch_pointcloud.datasets.s3dis.load_s3dis_room_data", wraps=load_s3dis_room_data)
def test_s3dis_dataset_split(
    mock_load: Mock,
    data_dir_factory: Callable[..., Path],
    areas: str | list[str],
) -> None:
    """Test that the dataset does not load raw data if the processed data exists"""
    data_dir = data_dir_factory("S3DIS/processed/**/*")

    dataset = S3DIS(root=data_dir, areas=areas, show_progress=False)
    assert len(dataset) > 0
    _ = list(dataset)

    assert mock_load.call_count == 0


@pytest.mark.parametrize("areas", [["Area_1"], ["Area_1", "Area_2"], "all"])
@patch("torch_pointcloud.datasets.s3dis.load_s3dis_room_data", wraps=load_s3dis_room_data)
def test_s3dis_dataset_process_split(
    mock_load: Mock,
    data_dir_factory: Callable[..., Path],
    areas: str | list[str],
) -> None:
    """Test that the dataset loads raw data if the processed data does not exist"""
    data_dir = data_dir_factory("S3DIS/raw/**/*")

    dataset = S3DIS(root=data_dir, areas=areas, show_progress=False)
    assert len(dataset) > 0
    _ = list(dataset)

    assert mock_load.call_count > 0


def test_s3dis_dataset_progress(data_dir_factory: Callable[..., Path], capsys: pytest.CaptureFixture[str]) -> None:
    """Test that the dataset displays a progress bar during processing"""
    data_dir = data_dir_factory("S3DIS/raw/**/*")

    dataset = S3DIS(root=data_dir, show_progress=True)
    assert len(dataset) > 0
    captured = capsys.readouterr()
    assert "Processing" in captured.err
    assert captured.out == ""


def test_s3dis_dataset_without_progress(
    data_dir_factory: Callable[..., Path], capsys: pytest.CaptureFixture[str]
) -> None:
    """Test that the dataset does not display a progress bar during processing"""
    data_dir = data_dir_factory("S3DIS/raw/**/*")

    dataset = S3DIS(root=data_dir, show_progress=False)
    assert len(dataset) > 0
    captured = capsys.readouterr()
    assert captured.err == ""
    assert captured.out == ""


def test_s3dis_dataset_progress_with_cached_processed(
    data_dir_factory: Callable[..., Path],
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Test that no progress bar is shown if the processed dataset already exists"""
    data_dir = data_dir_factory("S3DIS/processed/**/*")

    dataset = S3DIS(root=data_dir, show_progress=True)
    assert len(dataset) > 0
    captured = capsys.readouterr()
    assert "Processing" not in captured.err
    assert captured.out == ""


@pytest.mark.parametrize("classes", [["wall"], ["wall", "floor", "ceiling"]])
def test_s3dis_dataset_classes(
    data_dir_factory: Callable[..., Path],
    classes: list[str],
) -> None:
    """Test that the dataset loads specific classes"""
    data_dir = data_dir_factory("S3DIS/raw/**/*")

    dataset = S3DIS(root=data_dir, classes=classes, show_progress=False)
    assert len(dataset) > 0
    assert all(cls in dataset.classes for cls in classes)

    class_ids = set([*dataset.class_to_idx.values(), S3DIS_UNK_IDX])
    for data in dataset:
        labels = data["segment"].unique().tolist()
        assert set(labels).issubset(class_ids)


def test_s3dis_dataset_all_classes(data_dir_factory: Callable[..., Path]) -> None:
    """Test that the dataset loads all classes"""
    data_dir = data_dir_factory("S3DIS/raw/**/*")

    dataset = S3DIS(root=data_dir, classes="all", show_progress=False)
    assert len(dataset) > 0


def test_s3dis_dataset_transform(data_dir_factory: Callable[..., Path]) -> None:
    """Test that the dataset is transformed correctly after being processed"""
    data_dir = data_dir_factory("S3DIS/processed/**/*")

    transform = Mock(side_effect=lambda data: data)
    dataset = S3DIS(root=data_dir, transform=transform, show_progress=False)
    _ = list(dataset)
    assert transform.call_count == len(dataset)
