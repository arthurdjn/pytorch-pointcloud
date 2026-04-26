# mypy: disable-error-code="arg-type,call-overload"
import functools
from pathlib import Path
from typing import Callable, Type, Union
from unittest.mock import Mock, patch

import pytest
import torch

from torch_pointcloud.datasets import ModelNet10, ModelNet40, ModelNetNormalResampled
from torch_pointcloud.datasets.modelnet import (
    load_modelnet_data,
    load_modelnet_normal_resampled_data,
)

ModelNet10NormalResampled = functools.partial(ModelNetNormalResampled, variant="10")
ModelNet10NormalResampled.__name__ = "ModelNetNormalResampled"  # type: ignore[attr-defined]
ModelNet40NormalResampled = functools.partial(ModelNetNormalResampled, variant="40")
ModelNet40NormalResampled.__name__ = "ModelNetNormalResampled"  # type: ignore[attr-defined]
ModelNetDataset = Union[ModelNet10, ModelNet40]  # TODO: Type hint the ModelNetXXNormalResampled datasets?


@pytest.mark.parametrize("dataset_name", ["ModelNet10", "ModelNet40"])
def test_load_modelnet_data(datasets_dir: Path, dataset_name: str) -> None:
    """Test that the modelnet data is loaded correctly"""
    file_path = datasets_dir / dataset_name / "raw" / "chair" / "train" / "chair_0001.off"
    data = load_modelnet_data(file_path, 0)

    assert isinstance(data["pos"], torch.Tensor)
    assert isinstance(data["face"], torch.Tensor)
    assert isinstance(data["label"], torch.Tensor)

    assert data["label"].item() == 0
    assert data["pos"].shape == torch.Size([10, 3])
    assert data["face"].shape == torch.Size([10, 3])


@pytest.mark.parametrize("dataset_name", ["ModelNetNormalResampled"])
def test_load_modelnet_normal_resampled_data(datasets_dir: Path, dataset_name: str) -> None:
    """Test that the modelnet normal resampled data is loaded correctly"""
    file_path = datasets_dir / dataset_name / "raw" / "chair" / "chair_0001.txt"
    data = load_modelnet_normal_resampled_data(file_path, 0)

    assert isinstance(data["pos"], torch.Tensor)
    assert isinstance(data["normal"], torch.Tensor)
    assert isinstance(data["label"], torch.Tensor)

    assert data["pos"].shape == torch.Size([128, 3])
    assert data["normal"].shape == torch.Size([128, 3])
    assert data["label"].item() == 0


@pytest.mark.parametrize("dataset_cls", [ModelNet10, ModelNet40, ModelNet10NormalResampled, ModelNet40NormalResampled])
def test_modelnet_dataset_not_found(tmp_path: Path, dataset_cls: Type[ModelNetDataset]) -> None:
    """Raises an error if the dataset is not found"""
    with pytest.raises(RuntimeError, match="Dataset not found"):
        _ = dataset_cls(root=tmp_path, show_progress=False)


@pytest.mark.parametrize("dataset_cls", [ModelNet10, ModelNet40, ModelNet10NormalResampled, ModelNet40NormalResampled])
@pytest.mark.parametrize("train", [True, False])
def test_modelnet_dataset_raw_files_exist(
    datasets_dir_factory: Callable[..., Path],
    dataset_cls: Type[ModelNetDataset],
    train: bool,
) -> None:
    """Test that the raw files exist"""
    datasets_dir = datasets_dir_factory(f"{dataset_cls.__name__}/raw/**/*")
    dataset = dataset_cls(root=datasets_dir, train=train, show_progress=False)
    assert dataset.raw_files_exist()


@pytest.mark.parametrize("dataset_cls", [ModelNet10, ModelNet40, ModelNet10NormalResampled, ModelNet40NormalResampled])
@pytest.mark.parametrize("train", [True, False])
def test_modelnet_dataset_raw_files_not_exist(dataset_cls: Type[ModelNetDataset], train: bool) -> None:
    """Test that the raw files do not exist"""
    with pytest.raises(RuntimeError, match="Dataset not found"):
        _ = dataset_cls(root="not-found", train=train, show_progress=False)


@pytest.mark.parametrize("dataset_cls", [ModelNet10, ModelNet40, ModelNet10NormalResampled, ModelNet40NormalResampled])
@pytest.mark.parametrize("train", [True, False])
def test_modelnet_dataset_processed_files_exist(
    datasets_dir_factory: Callable[..., Path], dataset_cls: Type[ModelNetDataset], train: bool
) -> None:
    """Test that the processed files exist"""
    datasets_dir = datasets_dir_factory(f"{dataset_cls.__name__}/processed/**/*")
    dataset = dataset_cls(root=datasets_dir, train=train, show_progress=False)
    assert dataset.processed_files_exist()


@pytest.mark.parametrize("dataset_cls", [ModelNet10, ModelNet40, ModelNet10NormalResampled, ModelNet40NormalResampled])
@pytest.mark.parametrize("train", [True, False])
def test_modelnet_dataset_processed_files_not_exist(dataset_cls: Type[ModelNetDataset], train: bool) -> None:
    """Test that the processed files do not exist"""
    with pytest.raises(RuntimeError, match="Dataset not found"):
        _ = dataset_cls(root="not-found", train=train, show_progress=False)


@pytest.mark.parametrize("dataset_cls", [ModelNet10, ModelNet40])
@pytest.mark.parametrize("train", [True, False])
@patch("torch_pointcloud.datasets.modelnet.load_modelnet_data")
def test_modelnet_dataset_already_processed(
    mock_load: Mock,
    datasets_dir_factory: Callable[..., Path],
    dataset_cls: Type[ModelNetDataset],
    train: bool,
) -> None:
    """Test that the dataset is loaded correctly for different splits"""
    mock_load.side_effect = load_modelnet_data
    datasets_dir = datasets_dir_factory(f"{dataset_cls.__name__}/processed/**/*")

    dataset = dataset_cls(root=datasets_dir, train=train, show_progress=False)
    assert len(dataset) > 0
    _ = list(dataset)

    assert mock_load.call_count == 0


@pytest.mark.parametrize("dataset_cls", [ModelNet10NormalResampled, ModelNet40NormalResampled])
@pytest.mark.parametrize("train", [True, False])
@patch("torch_pointcloud.datasets.modelnet.load_modelnet_normal_resampled_data")
def test_modelnet_normal_resampled_dataset_already_processed(
    mock_load: Mock,
    datasets_dir_factory: Callable[..., Path],
    dataset_cls: Type[ModelNetDataset],
    train: bool,
) -> None:
    """Test that the dataset is loaded correctly for different splits"""
    mock_load.side_effect = load_modelnet_normal_resampled_data
    datasets_dir = datasets_dir_factory(f"{dataset_cls.__name__}/processed/**/*")

    dataset = dataset_cls(root=datasets_dir, train=train, show_progress=False)
    assert len(dataset) > 0
    _ = list(dataset)

    assert mock_load.call_count == 0


@pytest.mark.parametrize("dataset_cls", [ModelNet10, ModelNet40])
@pytest.mark.parametrize("train", [True, False])
@patch("torch_pointcloud.datasets.modelnet.load_modelnet_data")
def test_modelnet_dataset_process_split(
    mock_load: Mock,
    datasets_dir_factory: Callable[..., Path],
    dataset_cls: Type[ModelNetDataset],
    train: bool,
) -> None:
    """Test that the dataset is processed correctly for different splits
    when the processed data does not already exist"""
    mock_load.side_effect = load_modelnet_data
    datasets_dir = datasets_dir_factory(f"{dataset_cls.__name__}/raw/**/*")

    dataset = dataset_cls(root=datasets_dir, train=train, show_progress=False)
    assert len(dataset) > 0
    _ = list(dataset)

    assert mock_load.call_count == len(dataset)


@pytest.mark.parametrize("dataset_cls", [ModelNet10NormalResampled, ModelNet40NormalResampled])
@pytest.mark.parametrize("train", [True, False])
@patch("torch_pointcloud.datasets.modelnet.load_modelnet_normal_resampled_data")
def test_modelnet_normal_resampled_dataset_process_split(
    mock_load: Mock,
    datasets_dir_factory: Callable[..., Path],
    dataset_cls: Type[ModelNetDataset],
    train: bool,
) -> None:
    """Test that the dataset is processed correctly for different splits
    when the processed data does not already exist"""
    mock_load.side_effect = load_modelnet_normal_resampled_data
    datasets_dir = datasets_dir_factory(f"{dataset_cls.__name__}/raw/**/*")

    dataset = dataset_cls(root=datasets_dir, train=train, show_progress=False)
    assert len(dataset) > 0
    _ = list(dataset)

    assert mock_load.call_count == len(dataset)


@pytest.mark.parametrize("dataset_cls", [ModelNet10, ModelNet40])
@pytest.mark.parametrize("train", [True, False])
@patch("torch_pointcloud.datasets.modelnet.load_modelnet_data")
def test_modelnet_dataset_process_split_forced(
    mock_load: Mock,
    datasets_dir_factory: Callable[..., Path],
    dataset_cls: Type[ModelNetDataset],
    train: bool,
) -> None:
    """Test that the dataset is processed correctly for different splits
    regardless of whether the processed data already exists"""
    mock_load.side_effect = load_modelnet_data
    datasets_dir = datasets_dir_factory(f"{dataset_cls.__name__}/**/*", symlinks=False)

    dataset = dataset_cls(root=datasets_dir, train=train, show_progress=False, force_process=True)
    assert len(dataset) > 0
    _ = list(dataset)

    assert mock_load.call_count == len(dataset)


@pytest.mark.parametrize("dataset_cls", [ModelNet10NormalResampled, ModelNet40NormalResampled])
@pytest.mark.parametrize("train", [True, False])
@patch("torch_pointcloud.datasets.modelnet.load_modelnet_normal_resampled_data")
def test_modelnet_normal_resampled_dataset_process_split_forced(
    mock_load: Mock,
    datasets_dir_factory: Callable[..., Path],
    dataset_cls: Type[ModelNetDataset],
    train: bool,
) -> None:
    """Test that the dataset is processed correctly for different splits
    regardless of whether the processed data already exists"""
    mock_load.side_effect = load_modelnet_normal_resampled_data
    datasets_dir = datasets_dir_factory(f"{dataset_cls.__name__}/**/*")

    dataset = dataset_cls(root=datasets_dir, train=train, show_progress=False, force_process=True)
    assert len(dataset) > 0
    _ = list(dataset)

    assert mock_load.call_count == len(dataset)


@pytest.mark.parametrize("dataset_cls", [ModelNet10, ModelNet40, ModelNet10NormalResampled, ModelNet40NormalResampled])
def test_modelnet_dataset_progress(
    datasets_dir_factory: Callable[..., Path],
    dataset_cls: Type[ModelNetDataset],
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Test that the dataset is processed correctly with a progress bar"""
    datasets_dir = datasets_dir_factory(f"{dataset_cls.__name__}/raw/**/*")

    dataset = dataset_cls(root=datasets_dir, show_progress=True)
    assert len(dataset) > 0
    captured = capsys.readouterr()
    assert "Processing" in captured.err
    assert captured.out == ""


@pytest.mark.parametrize("dataset_cls", [ModelNet10, ModelNet40, ModelNet10NormalResampled, ModelNet40NormalResampled])
def test_modelnet_dataset_without_progress(
    datasets_dir_factory: Callable[..., Path],
    dataset_cls: Type[ModelNetDataset],
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Test that the dataset is processed correctly without a progress bar"""
    datasets_dir = datasets_dir_factory(f"{dataset_cls.__name__}/raw/**/*")

    dataset = dataset_cls(root=datasets_dir, show_progress=False)
    assert len(dataset) > 0
    captured = capsys.readouterr()
    assert captured.err == ""
    assert captured.out == ""


@pytest.mark.parametrize("dataset_cls", [ModelNet10, ModelNet40, ModelNet10NormalResampled, ModelNet40NormalResampled])
def test_modelnet_dataset_progress_with_cached_processed(
    datasets_dir_factory: Callable[..., Path],
    dataset_cls: Type[ModelNetDataset],
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Test that no progress bar is shown if the processed dataset already exists"""
    datasets_dir = datasets_dir_factory(f"{dataset_cls.__name__}/processed/**/*")

    dataset = dataset_cls(root=datasets_dir, show_progress=True)
    assert len(dataset) > 0
    captured = capsys.readouterr()
    assert captured.err == ""
    assert captured.out == ""


@pytest.mark.parametrize(
    "dataset_cls,category",
    [
        (ModelNet10, "chair"),
        (ModelNet40, "airplane"),
        (ModelNet10NormalResampled, "chair"),
        (ModelNet40NormalResampled, "airplane"),
    ],
)
def test_modelnet_dataset_category(
    datasets_dir_factory: Callable[..., Path],
    dataset_cls: Type[ModelNetDataset],
    category: str,
) -> None:
    """Test that the dataset is loaded correctly for a specific category"""
    datasets_dir = datasets_dir_factory(f"{dataset_cls.__name__}/raw/**/*")

    dataset = dataset_cls(root=datasets_dir, classes=category, show_progress=False)
    assert len(dataset) > 0
    assert dataset.classes == (category,)


@pytest.mark.parametrize(
    "dataset_cls,categories",
    [
        (ModelNet10, ["chair", "table"]),
        (ModelNet40, ["airplane", "car"]),
        (ModelNet10NormalResampled, ["chair", "table"]),
        (ModelNet40NormalResampled, ["airplane", "car"]),
    ],
)
def test_modelnet_dataset_categories(
    datasets_dir_factory: Callable[..., Path],
    dataset_cls: Type[ModelNetDataset],
    categories: list[str],
) -> None:
    """Test that the dataset is loaded correctly for multiple categories"""
    datasets_dir = datasets_dir_factory(f"{dataset_cls.__name__}/raw/**/*")

    dataset = dataset_cls(root=datasets_dir, classes=categories, show_progress=False)
    assert len(dataset) > 0
    assert len(dataset.classes) == len(categories)
    assert all(category in dataset.classes for category in categories)


@pytest.mark.parametrize("dataset_cls", [ModelNet10, ModelNet40, ModelNet10NormalResampled, ModelNet40NormalResampled])
def test_modelnet_dataset_pre_transform(
    datasets_dir_factory: Callable[..., Path], dataset_cls: Type[ModelNetDataset]
) -> None:
    """Test that the dataset is transformed correctly before being processed"""
    datasets_dir = datasets_dir_factory(f"{dataset_cls.__name__}/raw/**/*")

    pre_transform = Mock(side_effect=lambda x: x)
    dataset = dataset_cls(root=datasets_dir, pre_transform=pre_transform, show_progress=False)
    assert pre_transform.call_count == len(dataset)


@pytest.mark.parametrize("dataset_cls", [ModelNet10, ModelNet40, ModelNet10NormalResampled, ModelNet40NormalResampled])
def test_modelnet_dataset_pre_filter(
    datasets_dir_factory: Callable[..., Path], dataset_cls: Type[ModelNetDataset]
) -> None:
    """Test that the dataset is filtered correctly before being processed"""
    datasets_dir = datasets_dir_factory(f"{dataset_cls.__name__}/raw/**/*")

    pre_filter = Mock(side_effect=lambda x: True)
    dataset = dataset_cls(root=datasets_dir, pre_filter=pre_filter, show_progress=False)
    assert pre_filter.call_count == len(dataset)


@pytest.mark.parametrize("dataset_cls", [ModelNet10, ModelNet40, ModelNet10NormalResampled, ModelNet40NormalResampled])
def test_modelnet_dataset_transform(
    datasets_dir_factory: Callable[..., Path], dataset_cls: Type[ModelNetDataset]
) -> None:
    """Test that the dataset is transformed correctly after being processed"""
    datasets_dir = datasets_dir_factory(f"{dataset_cls.__name__}/processed/**/*")

    transform = Mock(side_effect=lambda data: data)
    dataset = dataset_cls(root=datasets_dir, transform=transform, show_progress=False)
    _ = list(dataset)
    assert transform.call_count == len(dataset)
