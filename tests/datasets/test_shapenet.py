import shutil
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import Mock

import pytest
import torch

from torch_pointcloud.datasets import ShapeNetPart
from torch_pointcloud.datasets.shapenet import load_shapenet_part_data


class DummyTransform:
    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        data = dict(data)
        data["dummy"] = "dummy"
        return data


def test_load_shapenet_part(data_dir: Path) -> None:
    """Test that the shapenet part is loaded correctly"""
    file_path = data_dir / "ShapeNetPart" / "raw" / "02691156" / "103c9e43cdf6501c62b600da24e0965.txt"
    data = load_shapenet_part_data(file_path, 0)

    assert isinstance(data["category"], torch.Tensor)
    assert isinstance(data["segmentation"], torch.Tensor)
    assert isinstance(data["coords"], torch.Tensor)
    assert isinstance(data["normals"], torch.Tensor)

    assert data["category"].item() == 0
    assert data["segmentation"].shape == (10,)
    assert data["coords"].shape == (10, 3)
    assert data["normals"].shape == (10, 3)


def test_shapenet_dataset_not_found(tmp_path: Path) -> None:
    """Raises an error if the dataset is not found"""
    with pytest.raises(RuntimeError, match="Dataset not found"):
        _ = ShapeNetPart(root=tmp_path, split="train", show_progress=False)


@pytest.mark.parametrize("split", ["train", "val", "test"])
def test_shapenet_dataset_split(data_dir: Path, monkeypatch: pytest.MonkeyPatch, split: str) -> None:
    """Test that the dataset is loaded correctly for different splits"""
    mock_load = Mock(wraps=load_shapenet_part_data)
    monkeypatch.setattr("torch_pointcloud.datasets.shapenet.load_shapenet_part_data", mock_load)

    dataset = ShapeNetPart(root=data_dir, split=split, show_progress=False)
    assert len(dataset) > 0
    _ = list(dataset)

    assert mock_load.call_count == 0


@pytest.mark.parametrize("split", ["train", "val", "test"])
def test_shapenet_dataset_process_split(data_dir: Path, monkeypatch: pytest.MonkeyPatch, split: str) -> None:
    """Test that the dataset is processed correctly for different splits"""
    shutil.rmtree(data_dir / "ShapeNetPart" / "processed")
    mock_load = Mock(wraps=load_shapenet_part_data)
    monkeypatch.setattr("torch_pointcloud.datasets.shapenet.load_shapenet_part_data", mock_load)

    dataset = ShapeNetPart(root=data_dir, split=split, show_progress=False)
    assert len(dataset) > 0
    _ = list(dataset)

    assert mock_load.call_count == len(dataset)


@pytest.mark.parametrize("split", ["train", "val", "test"])
def test_shapenet_dataset_process_split_forced(data_dir: Path, monkeypatch: pytest.MonkeyPatch, split: str) -> None:
    """Test that the dataset is processed correctly for different splits regardless of whether the processed data already exists"""
    mock_load = Mock(wraps=load_shapenet_part_data)
    monkeypatch.setattr("torch_pointcloud.datasets.shapenet.load_shapenet_part_data", mock_load)

    dataset = ShapeNetPart(root=data_dir, split=split, show_progress=False, force_process=True)
    assert len(dataset) > 0
    _ = list(dataset)

    assert mock_load.call_count == len(dataset)


def test_shapenet_dataset_invalid_split(data_dir: Path) -> None:
    """Raises an error if the split is invalid or not supported"""
    with pytest.raises(ValueError):
        _ = ShapeNetPart(root=data_dir, split="invalid", show_progress=False)


def test_shapenet_dataset_progress(data_dir: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Test that the dataset is processed correctly with a show_progress bar"""
    shutil.rmtree(data_dir / "ShapeNetPart" / "processed")

    dataset = ShapeNetPart(root=data_dir, split="train", show_progress=True)
    assert len(dataset) > 0
    captured = capsys.readouterr()
    assert "Processing" in captured.err
    assert captured.out == ""


def test_shapenet_dataset_without_progress(data_dir: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Test that the dataset is processed correctly without a show_progress bar"""
    shutil.rmtree(data_dir / "ShapeNetPart" / "processed")

    dataset = ShapeNetPart(root=data_dir, split="train", show_progress=False)
    assert len(dataset) > 0
    captured = capsys.readouterr()
    assert captured.err == ""
    assert captured.out == ""


def test_shapenet_dataset_progress_with_cached_processed(data_dir: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Test that no show_progress bar is shown if the processed dataset already exists"""
    dataset = ShapeNetPart(root=data_dir, split="train", show_progress=True)
    assert len(dataset) > 0
    captured = capsys.readouterr()
    assert captured.err == ""
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
def test_shapenet_dataset_category(data_dir: Path, category: str) -> None:
    """Test that the dataset is loaded correctly for a specific category"""
    shutil.rmtree(data_dir / "ShapeNetPart" / "processed")

    dataset = ShapeNetPart(root=data_dir, split="train", categories=category, show_progress=False)
    assert len(dataset) > 0
    assert len(dataset.categories) == 1
    assert dataset.categories[0] == category


@pytest.mark.parametrize("categories", [["Airplane", "Table"]])
def test_shapenet_dataset_categories(data_dir: Path, categories: List[str]) -> None:
    """Test that the dataset is loaded correctly for multiple categories"""
    shutil.rmtree(data_dir / "ShapeNetPart" / "processed")

    dataset = ShapeNetPart(root=data_dir, split="train", categories=categories, show_progress=False)
    assert len(dataset) > 0
    assert len(dataset.categories) == len(categories)
    assert all(category in dataset.categories for category in categories)


def test_shapenet_dataset_invalid_category(data_dir: Path) -> None:
    """Raises an error if the category is invalid or not supported"""
    shutil.rmtree(data_dir / "ShapeNetPart" / "processed")

    with pytest.raises(KeyError):
        _ = ShapeNetPart(root=data_dir, split="train", categories="Not a category", show_progress=False)


def test_shapenet_dataset_pre_transform(data_dir: Path) -> None:
    """Test that the dataset is transformed correctly before being processed"""
    shutil.rmtree(data_dir / "ShapeNetPart" / "processed")

    dataset = ShapeNetPart(root=data_dir, split="train", pre_transform=DummyTransform(), show_progress=False)
    assert len(dataset) > 0
    assert all("dummy" in data for data in dataset)


def test_shapenet_dataset_pre_filter(data_dir: Path) -> None:
    """Test that the dataset is filtered correctly before being processed"""
    shutil.rmtree(data_dir / "ShapeNetPart" / "processed")

    dataset = ShapeNetPart(root=data_dir, split="train", pre_filter=lambda x: False, show_progress=False)
    assert len(dataset) == 0


def test_shapenet_dataset_transform(data_dir: Path) -> None:
    """Test that the dataset is transformed correctly after being processed"""
    dataset = ShapeNetPart(root=data_dir, split="train", transform=DummyTransform(), show_progress=False)
    assert len(dataset) > 0
    assert all("dummy" in data for data in dataset)
