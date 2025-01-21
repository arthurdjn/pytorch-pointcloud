import shutil
from pathlib import Path
from typing import Any, Dict, List

import pytest
import torch

from torch_pointcloud.datasets import ShapeNetPart


class DummyTransform:
    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        data = dict(data)
        data["dummy"] = "dummy"
        return data


def test_shapenet_dataset_not_found(tmp_path: Path) -> None:
    """Raises an error if the dataset is not found"""
    with pytest.raises(RuntimeError):
        _ = ShapeNetPart(root=tmp_path, split="train", progress=False)


@pytest.mark.parametrize("split", ["train", "val", "test"])
def test_shapenet_dataset_split(data_dir: Path, split: str) -> None:
    """Test that the dataset is loaded correctly for different splits"""
    dataset = ShapeNetPart(root=data_dir, split=split, progress=False)
    assert len(dataset) > 0

    for data in dataset:
        category = data["category"]
        segmentation = data["segmentation"]
        coords = data["coords"]
        normals = data["normals"]

        assert torch.is_tensor(category)
        assert torch.is_tensor(segmentation)
        assert torch.is_tensor(coords)
        assert torch.is_tensor(normals)

        # Category is a scalar
        assert category.numel() == 1
        assert 0 <= category.item() < len(dataset.category_ids)
        assert segmentation.shape == (10,)
        assert coords.shape == (10, 3)
        assert normals.shape == (10, 3)


@pytest.mark.parametrize("split", ["train", "val", "test"])
def test_shapenet_dataset_process_split(data_dir: Path, split: str) -> None:
    """Test that the dataset is processed correctly for different splits"""
    shutil.rmtree(data_dir / "ShapeNetPart" / "processed")

    dataset = ShapeNetPart(root=data_dir, split=split, progress=False)
    assert len(dataset) > 0

    for data in dataset:
        category = data["category"]
        segmentation = data["segmentation"]
        coords = data["coords"]
        normals = data["normals"]

        assert torch.is_tensor(category)
        assert torch.is_tensor(segmentation)
        assert torch.is_tensor(coords)
        assert torch.is_tensor(normals)

        # Category is a scalar
        assert category.numel() == 1
        assert 0 <= category.item() < len(dataset.category_ids)
        assert segmentation.shape == (10,)
        assert coords.shape == (10, 3)
        assert normals.shape == (10, 3)


def test_shapenet_dataset_invalid_split(data_dir: Path) -> None:
    """Raises an error if the split is invalid or not supported"""
    with pytest.raises(ValueError):
        _ = ShapeNetPart(root=data_dir, split="invalid", progress=False)


def test_shapenet_dataset_progress(data_dir: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Test that the dataset is processed correctly with a progress bar"""
    shutil.rmtree(data_dir / "ShapeNetPart" / "processed")

    dataset = ShapeNetPart(root=data_dir, split="train", progress=True)
    assert len(dataset) > 0
    captured = capsys.readouterr()
    assert "Processing" in captured.err
    assert captured.out == ""


def test_shapenet_dataset_without_progress(data_dir: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Test that the dataset is processed correctly without a progress bar"""
    shutil.rmtree(data_dir / "ShapeNetPart" / "processed")

    dataset = ShapeNetPart(root=data_dir, split="train", progress=False)
    assert len(dataset) > 0
    captured = capsys.readouterr()
    assert captured.err == ""
    assert captured.out == ""


def test_shapenet_dataset_progress_with_cached_processed(data_dir: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Test that no progress bar is shown if the processed dataset already exists"""
    dataset = ShapeNetPart(root=data_dir, split="train", progress=True)
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

    dataset = ShapeNetPart(root=data_dir, split="train", categories=category, progress=False)
    assert len(dataset) > 0
    assert len(dataset.categories) == 1
    assert dataset.categories[0] == category


@pytest.mark.parametrize("categories", [["Airplane", "Table"]])
def test_shapenet_dataset_categories(data_dir: Path, categories: List[str]) -> None:
    """Test that the dataset is loaded correctly for multiple categories"""
    shutil.rmtree(data_dir / "ShapeNetPart" / "processed")

    dataset = ShapeNetPart(root=data_dir, split="train", categories=categories, progress=False)
    assert len(dataset) > 0
    assert len(dataset.categories) == len(categories)
    assert all(category in dataset.categories for category in categories)


def test_shapenet_dataset_invalid_category(data_dir: Path) -> None:
    """Raises an error if the category is invalid or not supported"""
    shutil.rmtree(data_dir / "ShapeNetPart" / "processed")

    with pytest.raises(KeyError):
        _ = ShapeNetPart(root=data_dir, split="train", categories="Not a category", progress=False)


def test_shapenet_dataset_pre_transform(data_dir: Path) -> None:
    """Test that the dataset is transformed correctly before being processed"""
    shutil.rmtree(data_dir / "ShapeNetPart" / "processed")

    dataset = ShapeNetPart(root=data_dir, split="train", pre_transform=DummyTransform(), progress=False)
    assert len(dataset) > 0
    assert all("dummy" in data for data in dataset)


def test_shapenet_dataset_pre_filter(data_dir: Path) -> None:
    """Test that the dataset is filtered correctly before being processed"""
    shutil.rmtree(data_dir / "ShapeNetPart" / "processed")

    dataset = ShapeNetPart(root=data_dir, split="train", pre_filter=lambda x: False, progress=False)
    assert len(dataset) == 0


def test_shapenet_dataset_transform(data_dir: Path) -> None:
    """Test that the dataset is transformed correctly after being processed"""
    dataset = ShapeNetPart(root=data_dir, split="train", transform=DummyTransform(), progress=False)
    assert len(dataset) > 0
    assert all("dummy" in data for data in dataset)
