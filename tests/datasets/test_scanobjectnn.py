# mypy: disable-error-code="arg-type,call-overload"
from pathlib import Path
from typing import Callable
from unittest.mock import Mock

import pytest

from torch_pointcloud.datasets import ScanObjectNN


@pytest.mark.parametrize("split", ["main", "split1", "split2", "split3", "split4"])
@pytest.mark.parametrize(
    "variant", [None, "augmented25_norot", "augmented25rot", "augmentedrot", "augmentedrot_scale75"]
)
@pytest.mark.parametrize("background", [True, False])
@pytest.mark.parametrize("train", [True, False])
def test_scanobjectnn_dataset_not_found(split: str, variant: str, background: bool, train: bool) -> None:
    """Raises an error if the dataset is not found"""
    with pytest.raises(RuntimeError, match="Dataset not found"):
        _ = ScanObjectNN(
            root="not-found",
            split=split,
            variant=variant,
            background=background,
            train=train,
            show_progress=False,
        )


@pytest.mark.parametrize("split", ["main", "split1", "split2", "split3", "split4"])
@pytest.mark.parametrize(
    "variant", [None, "augmented25_norot", "augmented25rot", "augmentedrot", "augmentedrot_scale75"]
)
@pytest.mark.parametrize("background", [True, False])
@pytest.mark.parametrize("train", [True, False])
def test_scanobjectnn_dataset_raw_files_exist(
    datasets_dir_factory: Callable[..., Path], split: str, variant: str, background: bool, train: bool
) -> None:
    """Test that the raw files exist"""
    datasets_dir = datasets_dir_factory("ScanObjectNN/raw/**/*")

    dataset = ScanObjectNN(
        root=datasets_dir,
        split=split,
        variant=variant,
        background=background,
        train=train,
        show_progress=False,
    )
    assert dataset.raw_files_exist()
    assert len(dataset) > 0


@pytest.mark.parametrize("split", ["main", "split1", "split2", "split3", "split4"])
@pytest.mark.parametrize(
    "variant", [None, "augmented25_norot", "augmented25rot", "augmentedrot", "augmentedrot_scale75"]
)
@pytest.mark.parametrize("background", [True, False])
@pytest.mark.parametrize("train", [True, False])
def test_scanobjectnn_dataset_raw_files_not_exist(split: str, variant: str, background: bool, train: bool) -> None:
    """Test that the raw files do not exist"""

    with pytest.raises(RuntimeError, match="Dataset not found"):
        _ = ScanObjectNN(
            root="not-found",
            split=split,
            variant=variant,
            background=background,
            train=train,
            show_progress=False,
        )


@pytest.mark.parametrize("split", ["main", "split1", "split2", "split3", "split4"])
@pytest.mark.parametrize(
    "variant", [None, "augmented25_norot", "augmented25rot", "augmentedrot", "augmentedrot_scale75"]
)
@pytest.mark.parametrize("background", [True, False])
@pytest.mark.parametrize("train", [True, False])
def test_scanobjectnn_dataset_processed_files_exist(
    datasets_dir_factory: Callable[..., Path], split: str, variant: str, background: bool, train: bool
) -> None:
    """Test that the processed files exist"""
    datasets_dir = datasets_dir_factory("ScanObjectNN/processed/**/*")

    dataset = ScanObjectNN(
        root=datasets_dir,
        split=split,
        variant=variant,
        background=background,
        train=train,
        show_progress=False,
    )
    assert dataset.processed_files_exist()
    assert len(dataset) > 0


@pytest.mark.parametrize("split", ["main", "split1", "split2", "split3", "split4"])
@pytest.mark.parametrize(
    "variant", [None, "augmented25_norot", "augmented25rot", "augmentedrot", "augmentedrot_scale75"]
)
@pytest.mark.parametrize("background", [True, False])
@pytest.mark.parametrize("train", [True, False])
def test_scanobjectnn_dataset_processed_files_not_exist(
    split: str, variant: str, background: bool, train: bool
) -> None:
    """Test that the processed files do not exist"""
    with pytest.raises(RuntimeError, match="Dataset not found"):
        _ = ScanObjectNN(
            root="not-found",
            split=split,
            variant=variant,
            background=background,
            train=train,
            show_progress=False,
        )


@pytest.mark.parametrize("split", ["main", "split1", "split2", "split3", "split4"])
@pytest.mark.parametrize(
    "variant", [None, "augmented25_norot", "augmented25rot", "augmentedrot", "augmentedrot_scale75"]
)
@pytest.mark.parametrize("background", [True, False])
@pytest.mark.parametrize("train", [True, False])
def test_scanobjectnn_dataset_force_process(
    datasets_dir_factory: Callable[..., Path], split: str, variant: str, background: bool, train: bool
) -> None:
    """Test that the dataset is processed correctly for different splits regardless of whether the processed data already exists"""
    datasets_dir = datasets_dir_factory("ScanObjectNN/**/*", symlinks=False)

    dataset = ScanObjectNN(
        root=datasets_dir,
        split=split,
        variant=variant,
        background=background,
        train=train,
        show_progress=False,
        force_process=True,
    )
    assert len(dataset) > 0


@pytest.mark.parametrize("split", ["invalid"])
@pytest.mark.parametrize("variant", ["invalid"])
def test_scanobjectnn_dataset_invalid_split(
    datasets_dir_factory: Callable[..., Path], split: str, variant: str
) -> None:
    """Raises an error if the split is invalid or not supported"""
    datasets_dir = datasets_dir_factory("ScanObjectNN/**/*")

    with pytest.raises(ValueError):
        _ = ScanObjectNN(root=datasets_dir, split=split, variant=variant, show_progress=False)


@pytest.mark.parametrize("label", list(ScanObjectNN.original_classes))
def test_scanobjectnn_dataset_labels(datasets_dir_factory: Callable[..., Path], label: str) -> None:
    """Test that the dataset is loaded correctly for a specific category"""
    datasets_dir = datasets_dir_factory("ScanObjectNN/processed/**/*")

    dataset = ScanObjectNN(root=datasets_dir, split="main", train=False, classes=[label], show_progress=False)  # type: ignore[list-item]
    assert len(dataset) > 0
    assert len(dataset.classes) == 1
    assert dataset.classes[0] == label
    assert dataset.class_to_idx[label] == 0


def test_scanobjectnn_dataset_transform_called(datasets_dir_factory: Callable[..., Path]) -> None:
    """Test that the dataset is transformed correctly after being processed"""
    datasets_dir = datasets_dir_factory("ScanObjectNN/processed/**/*")

    transform = Mock(side_effect=lambda data: data)
    dataset = ScanObjectNN(root=datasets_dir, split="main", train=False, transform=transform, show_progress=False)
    _ = list(dataset)
    assert transform.call_count == len(dataset)
