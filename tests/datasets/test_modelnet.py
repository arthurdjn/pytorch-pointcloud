import shutil
from pathlib import Path
from typing import Type
from unittest.mock import Mock

import pytest
import torch

from torch_pointcloud.datasets import ModelNet10, ModelNet40
from torch_pointcloud.datasets.modelnet import _ModelNet, load_modelnet_data


@pytest.mark.parametrize("dataset_cls", [ModelNet10, ModelNet40])
def test_load_modelnet_data(data_dir: Path, dataset_cls: Type[_ModelNet]) -> None:
    """Test that the modelnet data is loaded correctly"""
    file_path = data_dir / dataset_cls.__name__ / "raw" / "chair" / "train" / "chair_0001.off"
    data = load_modelnet_data(file_path, 0)

    assert isinstance(data["coords"], torch.Tensor)
    assert isinstance(data["faces"], torch.Tensor)
    assert isinstance(data["target"], torch.Tensor)

    assert data["target"].item() == 0
    assert data["coords"].shape == torch.Size([10, 3])
    assert data["faces"].shape == torch.Size([10, 3])


@pytest.mark.parametrize("dataset_cls", [ModelNet10, ModelNet40])
def test_modelnet_dataset_not_found(tmp_path: Path, dataset_cls: Type[_ModelNet]) -> None:
    """Raises an error if the dataset is not found"""
    with pytest.raises(RuntimeError, match="Dataset not found"):
        _ = dataset_cls(root=tmp_path, show_progress=False)


@pytest.mark.parametrize("dataset_cls", [ModelNet10, ModelNet40])
@pytest.mark.parametrize("train", [True, False])
def test_modelnet_dataset_split(
    data_dir: Path, monkeypatch: pytest.MonkeyPatch, dataset_cls: Type[_ModelNet], train: bool
) -> None:
    """Test that the dataset is loaded correctly for different splits"""
    mock_load = Mock(wraps=load_modelnet_data)
    monkeypatch.setattr("torch_pointcloud.datasets.modelnet.load_modelnet_data", mock_load)

    dataset = dataset_cls(root=data_dir, train=train, show_progress=False)
    assert len(dataset) > 0
    _ = list(dataset)

    assert mock_load.call_count == 0


@pytest.mark.parametrize("dataset_cls", [ModelNet10, ModelNet40])
@pytest.mark.parametrize("train", [True, False])
def test_modelnet_dataset_process_split(
    data_dir: Path, monkeypatch: pytest.MonkeyPatch, dataset_cls: Type[_ModelNet], train: bool
) -> None:
    """Test that the dataset is processed correctly for different splits"""
    shutil.rmtree(data_dir / dataset_cls.__name__ / "processed", ignore_errors=True)
    mock_load = Mock(wraps=load_modelnet_data)
    monkeypatch.setattr("torch_pointcloud.datasets.modelnet.load_modelnet_data", mock_load)

    dataset = dataset_cls(root=data_dir, train=train, show_progress=False)
    assert len(dataset) > 0
    _ = list(dataset)

    assert mock_load.call_count == len(dataset)


@pytest.mark.parametrize("dataset_cls", [ModelNet10, ModelNet40])
@pytest.mark.parametrize("train", [True, False])
def test_modelnet_dataset_process_split_forced(
    data_dir: Path, monkeypatch: pytest.MonkeyPatch, dataset_cls: Type[_ModelNet], train: bool
) -> None:
    """Test that the dataset is processed correctly for different splits regardless of whether the processed data already exists"""
    mock_load = Mock(wraps=load_modelnet_data)
    monkeypatch.setattr("torch_pointcloud.datasets.modelnet.load_modelnet_data", mock_load)

    dataset = dataset_cls(root=data_dir, train=train, show_progress=False, force_process=True)
    assert len(dataset) > 0
    _ = list(dataset)

    assert mock_load.call_count == len(dataset)


@pytest.mark.parametrize("dataset_cls", [ModelNet10, ModelNet40])
def test_modelnet_dataset_progress(
    data_dir: Path, dataset_cls: Type[_ModelNet], capsys: pytest.CaptureFixture[str]
) -> None:
    """Test that the dataset is processed correctly with a progress bar"""
    shutil.rmtree(data_dir / dataset_cls.__name__ / "processed", ignore_errors=True)

    dataset = dataset_cls(root=data_dir, show_progress=True)
    assert len(dataset) > 0
    captured = capsys.readouterr()
    assert "Processing" in captured.err
    assert captured.out == ""


@pytest.mark.parametrize("dataset_cls", [ModelNet10, ModelNet40])
def test_modelnet_dataset_without_progress(
    data_dir: Path, dataset_cls: Type[_ModelNet], capsys: pytest.CaptureFixture[str]
) -> None:
    """Test that the dataset is processed correctly without a progress bar"""
    shutil.rmtree(data_dir / dataset_cls.__name__ / "processed", ignore_errors=True)

    dataset = dataset_cls(root=data_dir, show_progress=False)
    assert len(dataset) > 0
    captured = capsys.readouterr()
    assert captured.err == ""
    assert captured.out == ""


@pytest.mark.parametrize("dataset_cls", [ModelNet10, ModelNet40])
def test_modelnet_dataset_progress_with_cached_processed(
    data_dir: Path, dataset_cls: Type[_ModelNet], capsys: pytest.CaptureFixture[str]
) -> None:
    """Test that no progress bar is shown if the processed dataset already exists"""
    dataset = dataset_cls(root=data_dir, show_progress=True)
    assert len(dataset) > 0
    captured = capsys.readouterr()
    assert captured.err == ""
    assert captured.out == ""


@pytest.mark.parametrize(
    "dataset_cls,category",
    [
        (ModelNet10, "chair"),
        (ModelNet40, "airplane"),
    ],
)
def test_modelnet_dataset_category(data_dir: Path, dataset_cls: Type[_ModelNet], category: str) -> None:
    """Test that the dataset is loaded correctly for a specific category"""
    shutil.rmtree(data_dir / dataset_cls.__name__ / "processed", ignore_errors=True)

    dataset = dataset_cls(root=data_dir, classes=category, show_progress=False)
    assert len(dataset) > 0
    assert dataset.classes == (category,)


@pytest.mark.parametrize(
    "dataset_cls,categories",
    [
        (ModelNet10, ["chair", "table"]),
        (ModelNet40, ["airplane", "car"]),
    ],
)
def test_modelnet_dataset_categories(data_dir: Path, dataset_cls: Type[_ModelNet], categories: list[str]) -> None:
    """Test that the dataset is loaded correctly for multiple categories"""
    shutil.rmtree(data_dir / dataset_cls.__name__ / "processed", ignore_errors=True)

    dataset = dataset_cls(root=data_dir, classes=categories, show_progress=False)
    assert len(dataset) > 0
    assert len(dataset.classes) == len(categories)
    assert all(category in dataset.classes for category in categories)


@pytest.mark.parametrize("dataset_cls", [ModelNet10, ModelNet40])
def test_modelnet_dataset_pre_transform(data_dir: Path, dataset_cls: Type[_ModelNet]) -> None:
    """Test that the dataset is transformed correctly before being processed"""
    shutil.rmtree(data_dir / dataset_cls.__name__ / "processed", ignore_errors=True)

    pre_transform = Mock(side_effect=lambda x: x)
    dataset = dataset_cls(root=data_dir, pre_transform=pre_transform, show_progress=False)
    assert pre_transform.call_count == len(dataset)


@pytest.mark.parametrize("dataset_cls", [ModelNet10, ModelNet40])
def test_modelnet_dataset_pre_filter(data_dir: Path, dataset_cls: Type[_ModelNet]) -> None:
    """Test that the dataset is filtered correctly before being processed"""
    shutil.rmtree(data_dir / dataset_cls.__name__ / "processed", ignore_errors=True)

    pre_filter = Mock(side_effect=lambda x: True)
    dataset = dataset_cls(root=data_dir, pre_filter=pre_filter, show_progress=False)
    assert pre_filter.call_count == len(dataset)


@pytest.mark.parametrize("dataset_cls", [ModelNet10, ModelNet40])
def test_modelnet_dataset_transform(data_dir: Path, dataset_cls: Type[_ModelNet]) -> None:
    """Test that the dataset is transformed correctly after being processed"""
    transform = Mock(side_effect=lambda data: data)
    dataset = dataset_cls(root=data_dir, transform=transform, show_progress=False)
    _ = list(dataset)
    assert transform.call_count == len(dataset)
