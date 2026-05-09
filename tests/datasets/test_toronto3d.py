# mypy: disable-error-code="arg-type,call-overload,attr-defined"
from pathlib import Path
from typing import Callable
from unittest.mock import Mock

import pytest
import torch

from torch_pointcloud.datasets import Toronto3D
from torch_pointcloud.datasets.toronto3d import (
    TORONTO3D_CLASSES,
    TORONTO3D_IGNORE_IDX,
    TORONTO3D_UTM_OFFSET,
    load_toronto3d_data,
)

# Must match generate.py's `--num-points` default.
NUM_POINTS_PER_SCAN = 1024
ALL_SPLITS = ["train", "val", "test", "trainval", "all"]
SPLIT_TO_FILES = {
    "train": {"L001", "L003", "L004"},
    "val": {"L002"},
    "test": {"L002"},
    "trainval": {"L001", "L002", "L003", "L004"},
    "all": {"L001", "L002", "L003", "L004"},
}


def test_load_toronto3d_data_shapes(datasets_dir: Path) -> None:
    """Loader returns the documented keys/dtypes/shapes."""
    ply_path = datasets_dir / "Toronto3D" / "raw" / "L001.ply"
    data = load_toronto3d_data(ply_path)

    assert isinstance(data["pos"], torch.Tensor)
    assert data["pos"].shape == (NUM_POINTS_PER_SCAN, 3)
    assert data["pos"].dtype == torch.float32

    assert isinstance(data["color"], torch.Tensor)
    assert data["color"].shape == (NUM_POINTS_PER_SCAN, 3)
    assert data["color"].dtype == torch.uint8

    assert isinstance(data["intensity"], torch.Tensor)
    assert data["intensity"].shape == (NUM_POINTS_PER_SCAN, 1)
    assert data["intensity"].dtype == torch.float32

    assert isinstance(data["gps_time"], torch.Tensor)
    assert data["gps_time"].shape == (NUM_POINTS_PER_SCAN, 1)
    assert data["gps_time"].dtype == torch.float32

    assert isinstance(data["segment"], torch.Tensor)
    assert data["segment"].shape == (NUM_POINTS_PER_SCAN,)
    assert data["segment"].dtype == torch.int64


def test_load_toronto3d_data_default_offset_keeps_local_range(datasets_dir: Path) -> None:
    """Default UTM offset brings raw 6.27e5/4.84e6 UTM into a small numerical range."""
    ply_path = datasets_dir / "Toronto3D" / "raw" / "L001.ply"
    data = load_toronto3d_data(ply_path)
    # After subtracting the offset, |x| and |y| should be well below the raw UTM scale.
    assert data["pos"][:, 0].abs().max().item() < 1e5
    assert data["pos"][:, 1].abs().max().item() < 1e5


def test_load_toronto3d_data_zero_offset_keeps_raw_utm(datasets_dir: Path) -> None:
    """Passing `utm_offset=(0, 0, 0)` yields raw UTM coordinates."""
    ply_path = datasets_dir / "Toronto3D" / "raw" / "L001.ply"
    data = load_toronto3d_data(ply_path, utm_offset=(0.0, 0.0, 0.0))
    # Raw Toronto UTM eastings are ~6.27e5; northings ~4.84e6.
    assert data["pos"][:, 0].mean().item() > 1e5
    assert data["pos"][:, 1].mean().item() > 1e6


def test_toronto3d_dataset_not_found() -> None:
    """Raises a clear error when the dataset directory is missing."""
    with pytest.raises(FileNotFoundError, match="Dataset not found"):
        _ = Toronto3D(root="not-found")


def test_toronto3d_dataset_invalid_split(datasets_dir_factory: Callable[..., Path]) -> None:
    """Unknown split values are rejected up-front."""
    datasets_dir = datasets_dir_factory("Toronto3D/raw/**/*")
    with pytest.raises(ValueError, match="Unknown split"):
        _ = Toronto3D(root=datasets_dir, split="bogus")


@pytest.mark.parametrize("split", ALL_SPLITS)
def test_toronto3d_dataset_raw_files_exist(datasets_dir_factory: Callable[..., Path], split: str) -> None:
    """Happy path: every documented split resolves to files that exist on disk."""
    datasets_dir = datasets_dir_factory("Toronto3D/raw/**/*")
    dataset = Toronto3D(root=datasets_dir, split=split)
    assert dataset.raw_files_exist()
    assert len(dataset) == len(SPLIT_TO_FILES[split])


@pytest.mark.parametrize("split", ALL_SPLITS)
def test_toronto3d_dataset_files_match_split(datasets_dir_factory: Callable[..., Path], split: str) -> None:
    """Each split exposes the exact set of files documented for the benchmark."""
    datasets_dir = datasets_dir_factory("Toronto3D/raw/**/*")
    dataset = Toronto3D(root=datasets_dir, split=split)
    file_stems = {Path(f).stem for f in dataset.files}
    assert file_stems == SPLIT_TO_FILES[split]


def test_toronto3d_dataset_custom_files_override_split(datasets_dir_factory: Callable[..., Path]) -> None:
    """Passing `files=` lets users mix and match scans for ad-hoc tasks."""
    datasets_dir = datasets_dir_factory("Toronto3D/raw/**/*")
    custom = ("L001.ply", "L002.ply")
    dataset = Toronto3D(root=datasets_dir, files=custom)
    assert dataset.files == custom
    assert len(dataset) == 2


def test_toronto3d_dataset_returns_expected_shapes(datasets_dir_factory: Callable[..., Path]) -> None:
    """Each sample exposes the expected keys / dtypes / shapes plus the source name."""
    datasets_dir = datasets_dir_factory("Toronto3D/raw/**/*")
    dataset = Toronto3D(root=datasets_dir, split="all")

    for index in range(len(dataset)):
        sample = dataset[index]
        assert sample["pos"].shape == (NUM_POINTS_PER_SCAN, 3)
        assert sample["pos"].dtype == torch.float32
        assert sample["color"].shape == (NUM_POINTS_PER_SCAN, 3)
        assert sample["color"].dtype == torch.uint8
        assert sample["intensity"].shape == (NUM_POINTS_PER_SCAN, 1)
        assert sample["gps_time"].shape == (NUM_POINTS_PER_SCAN, 1)
        assert sample["segment"].shape == (NUM_POINTS_PER_SCAN,)
        assert sample["segment"].dtype == torch.int64
        assert isinstance(sample["name"], str) and sample["name"]


def test_toronto3d_dataset_segment_ids_valid(datasets_dir_factory: Callable[..., Path]) -> None:
    """Segment ids returned by the dataset stay within the 9-class range."""
    datasets_dir = datasets_dir_factory("Toronto3D/raw/**/*")
    dataset = Toronto3D(root=datasets_dir, split="all")

    valid_ids = set(range(len(TORONTO3D_CLASSES)))
    for sample in dataset:
        unique = set(int(v) for v in sample["segment"].unique().tolist())
        assert unique.issubset(valid_ids), f"unexpected ids: {unique - valid_ids}"


def test_toronto3d_dataset_utm_offset_passthrough(datasets_dir_factory: Callable[..., Path]) -> None:
    """The dataset forwards `utm_offset` to the loader."""
    datasets_dir = datasets_dir_factory("Toronto3D/raw/**/*")
    dataset = Toronto3D(root=datasets_dir, files=["L001.ply"], utm_offset=(0.0, 0.0, 0.0))
    sample = dataset[0]
    assert sample["pos"][:, 0].mean().item() > 1e5


def test_toronto3d_ignore_index_in_class_set() -> None:
    """`TORONTO3D_IGNORE_IDX` matches the documented `Unclassified` class."""
    assert TORONTO3D_CLASSES[TORONTO3D_IGNORE_IDX] == "Unclassified"


def test_toronto3d_default_utm_offset_is_three_vector() -> None:
    """`TORONTO3D_UTM_OFFSET` is a 3-vector with non-zero easting/northing."""
    assert len(TORONTO3D_UTM_OFFSET) == 3
    assert TORONTO3D_UTM_OFFSET[0] > 0 and TORONTO3D_UTM_OFFSET[1] > 0


def test_toronto3d_dataset_transform_called(datasets_dir_factory: Callable[..., Path]) -> None:
    """Transform is invoked exactly once per `__getitem__`."""
    datasets_dir = datasets_dir_factory("Toronto3D/raw/**/*")

    transform = Mock(side_effect=lambda data: data)
    dataset = Toronto3D(root=datasets_dir, split="all", transform=transform)
    _ = list(dataset)
    assert transform.call_count == len(dataset)


def test_toronto3d_dataset_load_is_lazy(datasets_dir_factory: Callable[..., Path]) -> None:
    """Construction enumerates files but doesn't read PLYs; reads happen in `__getitem__`."""
    datasets_dir = datasets_dir_factory("Toronto3D/raw/**/*")

    transform = Mock(side_effect=lambda data: data)
    dataset = Toronto3D(root=datasets_dir, split="all", transform=transform)
    assert transform.call_count == 0
    _ = dataset[0]
    assert transform.call_count == 1
