# mypy: disable-error-code="arg-type,call-overload"
from pathlib import Path
from typing import Callable, List
from unittest.mock import Mock, patch

import numpy as np
import pytest

from torch_pointcloud.datasets.shapenetpart import ShapeNetPart, load_shapenet_part_data


def test_load_shapenet_part(datasets_dir_factory: Callable[..., Path]) -> None:
    """Test that the shapenet part is loaded correctly"""
    datasets_dir = datasets_dir_factory("ShapeNetPart/raw/**/*")
    file_path = datasets_dir / "ShapeNetPart" / "raw" / "02691156" / "103c9e43cdf6501c62b600da24e0965.txt"
    data = load_shapenet_part_data(file_path)

    assert data is not None
    assert isinstance(data["pos"], np.ndarray)
    assert isinstance(data["normal"], np.ndarray)
    assert isinstance(data["segment"], np.ndarray)

    assert data["pos"].shape == (10, 3)
    assert data["normal"].shape == (10, 3)
    assert data["segment"].shape == (10,)


def test_shapenet_dataset_not_found() -> None:
    """Raises an error if the dataset is not found"""
    with pytest.raises(RuntimeError, match="Dataset not found"):
        _ = ShapeNetPart(root="not-found", split="train", show_progress=False)


@pytest.mark.parametrize("split", ["train", "val", "test"])
def test_shapenet_dataset_raw_files_exist(datasets_dir_factory: Callable[..., Path], split: str) -> None:
    """Test that the raw files exist"""
    datasets_dir = datasets_dir_factory("ShapeNetPart/raw/**/*")

    dataset = ShapeNetPart(root=datasets_dir, split=split, show_progress=False)
    assert dataset.raw_files_exist()


@pytest.mark.parametrize("split", ["train", "val", "test"])
def test_shapenet_dataset_raw_files_not_exist(split: str) -> None:
    """Test that the raw files do not exist"""

    with pytest.raises(RuntimeError, match="Dataset not found"):
        _ = ShapeNetPart(root="not-found", split=split, show_progress=False)


@pytest.mark.parametrize("split", ["train", "val", "test"])
def test_shapenet_dataset_processed_files_exist(datasets_dir_factory: Callable[..., Path], split: str) -> None:
    """Test that the processed files exist"""
    datasets_dir = datasets_dir_factory("ShapeNetPart/processed/**/*")

    dataset = ShapeNetPart(root=datasets_dir, split=split, show_progress=False)
    assert dataset.processed_files_exist()


@pytest.mark.parametrize("split", ["train", "val", "test"])
def test_shapenet_dataset_processed_files_not_exist(split: str) -> None:
    """Test that the processed files do not exist"""
    with pytest.raises(RuntimeError, match="Dataset not found"):
        _ = ShapeNetPart(root="not-found", split=split, show_progress=False)


@pytest.mark.parametrize("split", ["train", "val", "test"])
@patch("torch_pointcloud.datasets.shapenetpart.load_shapenet_part_data")
def test_shapenet_dataset_split(
    mock_load: Mock,
    datasets_dir_factory: Callable[..., Path],
    split: str,
) -> None:
    """Test that the dataset is loaded correctly for different splits"""
    mock_load.side_effect = load_shapenet_part_data
    datasets_dir = datasets_dir_factory("ShapeNetPart/processed/**/*")

    dataset = ShapeNetPart(root=datasets_dir, split=split, show_progress=False)
    assert len(dataset) > 0
    _ = list(dataset)

    assert mock_load.call_count == 0


@pytest.mark.parametrize("split", ["train", "val", "test"])
@patch("torch_pointcloud.datasets.shapenetpart.load_shapenet_part_data")
def test_shapenet_dataset_process_split(
    mock_load: Mock,
    datasets_dir_factory: Callable[..., Path],
    split: str,
) -> None:
    """Test that the dataset is processed correctly for different splits"""
    mock_load.side_effect = load_shapenet_part_data
    datasets_dir = datasets_dir_factory("ShapeNetPart/raw/**/*")

    dataset = ShapeNetPart(root=datasets_dir, split=split, show_progress=False)
    assert len(dataset) > 0
    _ = list(dataset)

    assert mock_load.call_count == len(dataset)


@pytest.mark.parametrize("split", ["train", "val", "test"])
@patch("torch_pointcloud.datasets.shapenetpart.load_shapenet_part_data")
def test_shapenet_dataset_process_split_forced(
    mock_load: Mock,
    datasets_dir_factory: Callable[..., Path],
    split: str,
) -> None:
    """Test that the dataset is processed correctly for different splits regardless of whether the processed data already exists"""
    mock_load.side_effect = load_shapenet_part_data
    datasets_dir = datasets_dir_factory("ShapeNetPart/**/*", symlinks=False)

    dataset = ShapeNetPart(root=datasets_dir, split=split, show_progress=False, force_process=True)
    assert len(dataset) > 0
    _ = list(dataset)

    assert mock_load.call_count == len(dataset)


def test_shapenet_dataset_invalid_split(datasets_dir_factory: Callable[..., Path]) -> None:
    """Raises an error if the split is invalid or not supported"""
    datasets_dir = datasets_dir_factory("ShapeNetPart/**/*")

    with pytest.raises(ValueError):
        _ = ShapeNetPart(root=datasets_dir, split="invalid", show_progress=False)


def test_shapenet_dataset_progress(
    datasets_dir_factory: Callable[..., Path], capsys: pytest.CaptureFixture[str]
) -> None:
    """Test that the dataset is processed correctly with a progress bar"""
    datasets_dir = datasets_dir_factory("ShapeNetPart/raw/**/*")

    dataset = ShapeNetPart(root=datasets_dir, split="train", show_progress=True)
    assert len(dataset) > 0
    captured = capsys.readouterr()
    assert "Reading" in captured.err
    assert captured.out == ""


def test_shapenet_dataset_without_progress(
    datasets_dir_factory: Callable[..., Path],
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Test that the dataset is processed correctly without a progress bar"""
    datasets_dir = datasets_dir_factory("ShapeNetPart/raw/**/*")

    dataset = ShapeNetPart(root=datasets_dir, split="train", show_progress=False)
    assert len(dataset) > 0
    captured = capsys.readouterr()
    assert captured.err == ""
    assert captured.out == ""


def test_shapenet_dataset_progress_with_cached_processed(
    datasets_dir_factory: Callable[..., Path],
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Test that no processing progress bar is shown if the processed dataset already exists"""
    datasets_dir = datasets_dir_factory("ShapeNetPart/processed/**/*")

    dataset = ShapeNetPart(root=datasets_dir, split="train", show_progress=True)
    assert len(dataset) > 0
    captured = capsys.readouterr()
    assert "Reading" not in captured.err
    assert captured.out == ""


@pytest.mark.parametrize(
    "category",
    [
        "Airplane",
        "Bag",
        "Cap",
        "Car",
        "Chair",
        "Earphone",
        "Guitar",
        "Knife",
        "Lamp",
        "Laptop",
        "Motorbike",
        "Mug",
        "Pistol",
        "Rocket",
        "Skateboard",
        "Table",
    ],
)
def test_shapenet_dataset_category(datasets_dir_factory: Callable[..., Path], category: str) -> None:
    """Test that the dataset is loaded correctly for a specific category"""
    datasets_dir = datasets_dir_factory("ShapeNetPart/processed/**/*")

    dataset = ShapeNetPart(root=datasets_dir, split="train", categories=category, show_progress=False)
    assert len(dataset) > 0
    assert len(dataset.categories) == 1
    assert dataset.categories[0] == category


@pytest.mark.parametrize("categories", [["Airplane", "Table"]])
def test_shapenet_dataset_categories(datasets_dir_factory: Callable[..., Path], categories: List[str]) -> None:
    """Test that the dataset is loaded correctly for multiple categories"""
    datasets_dir = datasets_dir_factory("ShapeNetPart/processed/**/*")

    dataset = ShapeNetPart(root=datasets_dir, split="train", categories=categories, show_progress=False)
    assert len(dataset) > 0
    assert len(dataset.categories) == len(categories)
    assert all(category in dataset.categories for category in categories)


def test_shapenet_dataset_invalid_category(datasets_dir_factory: Callable[..., Path]) -> None:
    """Raises an error if the category is invalid or not supported"""
    datasets_dir = datasets_dir_factory("ShapeNetPart/raw/**/*")

    with pytest.raises(KeyError):
        _ = ShapeNetPart(root=datasets_dir, split="train", categories="Not a category", show_progress=False)


def test_shapenet_dataset_transform_called(datasets_dir_factory: Callable[..., Path]) -> None:
    """Test that the dataset is transformed correctly after being processed"""
    datasets_dir = datasets_dir_factory("ShapeNetPart/processed/**/*")

    transform = Mock(side_effect=lambda data: data)
    dataset = ShapeNetPart(root=datasets_dir, split="train", transform=transform, show_progress=False)
    _ = list(dataset)
    assert transform.call_count == len(dataset)
