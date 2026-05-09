# mypy: disable-error-code="arg-type,call-overload,attr-defined"
from pathlib import Path
from typing import Callable
from unittest.mock import Mock

import pytest
import torch

from torch_pointcloud.datasets import ParisLille3D
from torch_pointcloud.datasets.parislille3d import (
    PARISLILLE3D_CLASSES,
    PARISLILLE3D_IGNORE_IDX,
    load_parislille3d_data,
)

# Must match generate.py's `--num-points` default.
NUM_POINTS_PER_SCAN = 1024
ALL_SPLITS = ["train", "val", "trainval", "all"]
SPLIT_TO_FILES = {
    "train": {"Lille1_1", "Lille1_2", "Paris"},
    "val": {"Lille2"},
    "trainval": {"Lille1_1", "Lille1_2", "Lille2", "Paris"},
    "all": {"Lille1_1", "Lille1_2", "Lille2", "Paris"},
}


def test_load_parislille3d_data_shapes(datasets_dir: Path) -> None:
    """`.ply` loader returns positions, reflectance, and segment with expected shapes/dtypes."""
    ply_path = datasets_dir / "ParisLille3D" / "raw" / "Lille1_1.ply"
    data = load_parislille3d_data(ply_path)

    assert isinstance(data["pos"], torch.Tensor)
    assert data["pos"].shape == (NUM_POINTS_PER_SCAN, 3)
    assert data["pos"].dtype == torch.float32

    assert isinstance(data["reflectance"], torch.Tensor)
    assert data["reflectance"].shape == (NUM_POINTS_PER_SCAN, 1)
    assert data["reflectance"].dtype == torch.uint8

    assert isinstance(data["segment"], torch.Tensor)
    assert data["segment"].shape == (NUM_POINTS_PER_SCAN,)
    assert data["segment"].dtype == torch.int64

    # Sanity: every class id must be in the documented 10-class set.
    valid_ids = set(range(len(PARISLILLE3D_CLASSES)))
    assert set(int(v) for v in data["segment"].unique().tolist()).issubset(valid_ids)


def test_parislille3d_dataset_not_found() -> None:
    """Raises a clear error when the dataset directory is missing."""
    with pytest.raises(FileNotFoundError, match="Dataset not found"):
        _ = ParisLille3D(root="not-found")


@pytest.mark.parametrize("split", ALL_SPLITS)
def test_parislille3d_dataset_raw_files_exist(datasets_dir_factory: Callable[..., Path], split: str) -> None:
    """Happy path: the dataset can locate the raw files for each split."""
    datasets_dir = datasets_dir_factory("ParisLille3D/raw/**/*")
    dataset = ParisLille3D(root=datasets_dir, split=split)
    assert dataset.raw_files_exist()
    assert len(dataset) == len(SPLIT_TO_FILES[split])


def test_parislille3d_dataset_invalid_split(datasets_dir_factory: Callable[..., Path]) -> None:
    """Unknown split values are rejected up-front."""
    datasets_dir = datasets_dir_factory("ParisLille3D/raw/**/*")
    with pytest.raises(ValueError, match="Unknown split"):
        _ = ParisLille3D(root=datasets_dir, split="bogus")


@pytest.mark.parametrize("split", ALL_SPLITS)
def test_parislille3d_dataset_files_match_split(datasets_dir_factory: Callable[..., Path], split: str) -> None:
    """Each split exposes the exact set of files documented for the 10-class benchmark."""
    datasets_dir = datasets_dir_factory("ParisLille3D/raw/**/*")
    dataset = ParisLille3D(root=datasets_dir, split=split)
    file_stems = {Path(f).stem for f in dataset.files}
    assert file_stems == SPLIT_TO_FILES[split]


def test_parislille3d_dataset_custom_files_override_split(datasets_dir_factory: Callable[..., Path]) -> None:
    """Passing `files=` lets users mix and match scans for ad-hoc tasks."""
    datasets_dir = datasets_dir_factory("ParisLille3D/raw/**/*")
    custom = ("Lille1_1.ply", "Paris.ply")
    dataset = ParisLille3D(root=datasets_dir, files=custom)
    # `files=` ignores `split=` (which kept its "val" default).
    assert dataset.files == custom
    assert len(dataset) == 2


def test_parislille3d_dataset_returns_expected_shapes(datasets_dir_factory: Callable[..., Path]) -> None:
    """Each sample exposes the expected keys / dtypes / shapes plus the source name."""
    datasets_dir = datasets_dir_factory("ParisLille3D/raw/**/*")
    dataset = ParisLille3D(root=datasets_dir, split="all")

    for index in range(len(dataset)):
        sample = dataset[index]
        assert sample["pos"].shape == (NUM_POINTS_PER_SCAN, 3)
        assert sample["pos"].dtype == torch.float32
        assert sample["reflectance"].shape == (NUM_POINTS_PER_SCAN, 1)
        assert sample["reflectance"].dtype == torch.uint8
        assert sample["segment"].shape == (NUM_POINTS_PER_SCAN,)
        assert sample["segment"].dtype == torch.int64
        assert isinstance(sample["name"], str) and sample["name"]


def test_parislille3d_dataset_segment_ids_valid(datasets_dir_factory: Callable[..., Path]) -> None:
    """Returned segment ids are within the 10-class benchmark range."""
    datasets_dir = datasets_dir_factory("ParisLille3D/raw/**/*")
    dataset = ParisLille3D(root=datasets_dir, split="all")

    valid_ids = set(range(len(PARISLILLE3D_CLASSES)))
    for sample in dataset:
        unique = set(int(v) for v in sample["segment"].unique().tolist())
        assert unique.issubset(valid_ids), f"unexpected ids: {unique - valid_ids}"


def test_parislille3d_dataset_ignore_index_in_class_set() -> None:
    """`PARISLILLE3D_IGNORE_IDX` matches the documented `unclassified` class."""
    assert PARISLILLE3D_CLASSES[PARISLILLE3D_IGNORE_IDX] == "unclassified"


def test_parislille3d_dataset_transform_called(datasets_dir_factory: Callable[..., Path]) -> None:
    """Transform is invoked exactly once per `__getitem__`."""
    datasets_dir = datasets_dir_factory("ParisLille3D/raw/**/*")

    transform = Mock(side_effect=lambda data: data)
    dataset = ParisLille3D(root=datasets_dir, split="all", transform=transform)
    _ = list(dataset)
    assert transform.call_count == len(dataset)


def test_parislille3d_dataset_load_is_lazy(datasets_dir_factory: Callable[..., Path]) -> None:
    """Construction enumerates files but doesn't read PLYs; reads happen in `__getitem__`."""
    datasets_dir = datasets_dir_factory("ParisLille3D/raw/**/*")

    transform = Mock(side_effect=lambda data: data)
    dataset = ParisLille3D(root=datasets_dir, split="all", transform=transform)
    assert transform.call_count == 0
    _ = dataset[0]
    assert transform.call_count == 1
