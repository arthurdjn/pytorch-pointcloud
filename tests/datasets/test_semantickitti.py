# mypy: disable-error-code="call-overload,attr-defined"
from pathlib import Path
from typing import Callable
from unittest.mock import Mock

import pytest
import torch

from torch_pointcloud.datasets import SemanticKITTI
from torch_pointcloud.datasets.semantickitti import (
    SEMANTIC_KITTI_LABEL_NAMES,
    SEMANTIC_KITTI_SEQUENCES_PER_SPLIT,
    load_semantickitti_labels,
    load_semantickitti_scan,
)

# Splits backed by the test fixture (see `tests/data/datasets/SemanticKITTI/scripts/generate.py`).
# `train` and `val` have labels; `test` doesn't (matches the real release).
ALL_SPLITS = ["train", "val", "trainval", "test"]
SPLITS_WITH_LABELS = ["train", "val", "trainval"]
NUM_POINTS_PER_SCAN = 1024  # must match generate.py's `--num-points` default


def test_load_semantickitti_scan(datasets_dir: Path) -> None:
    """`.bin` loader returns positions $(N, 3)$ float32 and intensity $(N, 1)$ float32."""
    bin_path = datasets_dir / "SemanticKITTI" / "raw" / "sequences" / "00" / "velodyne" / "000000.bin"
    pos, intensity = load_semantickitti_scan(bin_path)

    assert isinstance(pos, torch.Tensor) and isinstance(intensity, torch.Tensor)
    assert pos.shape == (NUM_POINTS_PER_SCAN, 3)
    assert pos.dtype == torch.float32
    assert intensity.shape == (NUM_POINTS_PER_SCAN, 1)
    assert intensity.dtype == torch.float32


def test_load_semantickitti_labels(datasets_dir: Path) -> None:
    """`.label` loader returns segment / instance int64 tensors of shape $(N,)$."""
    label_path = datasets_dir / "SemanticKITTI" / "raw" / "sequences" / "00" / "labels" / "000000.label"
    segment, instance = load_semantickitti_labels(label_path)

    assert isinstance(segment, torch.Tensor) and isinstance(instance, torch.Tensor)
    assert segment.shape == (NUM_POINTS_PER_SCAN,) and segment.dtype == torch.int64
    assert instance.shape == (NUM_POINTS_PER_SCAN,) and instance.dtype == torch.int64
    # Sanity: every semantic id is one of the documented raw labels.
    valid_ids = set(SEMANTIC_KITTI_LABEL_NAMES.keys())
    assert set(int(v) for v in segment.unique().tolist()).issubset(valid_ids)


@pytest.mark.parametrize("split", ALL_SPLITS)
def test_semantickitti_dataset_not_found(split: str) -> None:
    """Raises a clear error when the dataset directory is missing."""
    with pytest.raises(RuntimeError, match="Dataset not found"):
        _ = SemanticKITTI(root="not-found", split=split)


@pytest.mark.parametrize("split", ALL_SPLITS)
def test_semantickitti_dataset_raw_files_exist(datasets_dir_factory: Callable[..., Path], split: str) -> None:
    """Happy path: the dataset can locate the raw files for each split present in the fixture."""
    datasets_dir = datasets_dir_factory("SemanticKITTI/raw/**/*")

    # The fixture only ships seq 00 / 08 / 11. Trim each split to the sequences we have.
    available = {"00", "08", "11"}
    sequences = tuple(s for s in SEMANTIC_KITTI_SEQUENCES_PER_SPLIT[split] if s in available)
    assert sequences, f"fixture has no scans for split={split!r}"

    dataset = SemanticKITTI(root=datasets_dir, split=split, sequences=sequences)
    assert dataset.raw_files_exist()
    assert len(dataset) > 0


def test_semantickitti_dataset_missing_sequences_raise(datasets_dir_factory: Callable[..., Path]) -> None:
    """A partial download is rejected: `split='train'` expects seqs 00-10 but the fixture ships only
    seq 00, so the dataset raises listing the missing sequences instead of silently loading a subset."""
    datasets_dir = datasets_dir_factory("SemanticKITTI/raw/sequences/00/**/*")

    with pytest.raises(RuntimeError, match="Missing sequence"):
        _ = SemanticKITTI(root=datasets_dir, split="train")

    with pytest.raises(RuntimeError, match="01, 02, 03, 04, 05, 06, 07, 09, 10"):
        _ = SemanticKITTI(root=datasets_dir, split="train")


def test_semantickitti_dataset_explicit_sequence_subset_loads(datasets_dir_factory: Callable[..., Path]) -> None:
    """Explicitly restricting `sequences` to what is on disk loads the subset."""
    datasets_dir = datasets_dir_factory("SemanticKITTI/raw/sequences/00/**/*")

    dataset = SemanticKITTI(root=datasets_dir, split="train", sequences=("00",))
    assert dataset.raw_files_exist()
    assert len(dataset) > 0
    assert {seq for seq, *_ in dataset.scans} == {"00"}


def test_semantickitti_dataset_processed_dir_aliases_raw(datasets_dir_factory: Callable[..., Path]) -> None:
    """`SemanticKITTI` consumes raw `.bin` directly; `processed_dir` should alias `raw_dir`."""
    datasets_dir = datasets_dir_factory("SemanticKITTI/raw/**/*")

    dataset = SemanticKITTI(root=datasets_dir, split="train", sequences=("00",))
    assert dataset.processed_dir == dataset.raw_dir
    assert dataset.processed_files_exist()


def test_semantickitti_dataset_default_split_is_train() -> None:
    """When split is omitted we should hit the default — `train`."""
    with pytest.raises(RuntimeError, match="Dataset not found"):
        _ = SemanticKITTI(root="not-found")

    # And bogus splits raise ValueError, not RuntimeError.
    with pytest.raises(ValueError, match="Unknown split"):
        _ = SemanticKITTI(root="not-found", split="bogus")


def test_semantickitti_dataset_invalid_sequence(datasets_dir_factory: Callable[..., Path]) -> None:
    """Unknown sequence ids are rejected up-front."""
    datasets_dir = datasets_dir_factory("SemanticKITTI/raw/**/*")
    with pytest.raises(ValueError, match="Unknown sequence"):
        _ = SemanticKITTI(root=datasets_dir, split="val", sequences=("99",))


@pytest.mark.parametrize("split", SPLITS_WITH_LABELS)
def test_semantickitti_dataset_returns_labels(datasets_dir_factory: Callable[..., Path], split: str) -> None:
    """For splits with `.label` files, samples include `segment` + `instance`."""
    datasets_dir = datasets_dir_factory("SemanticKITTI/raw/**/*")
    available = {"00", "08", "11"}
    sequences = tuple(s for s in SEMANTIC_KITTI_SEQUENCES_PER_SPLIT[split] if s in available)
    dataset = SemanticKITTI(root=datasets_dir, split=split, sequences=sequences)
    sample = dataset[0]

    assert sample["pos"].shape == (NUM_POINTS_PER_SCAN, 3)
    assert sample["pos"].dtype == torch.float32
    assert sample["intensity"].shape == (NUM_POINTS_PER_SCAN, 1)
    assert sample["intensity"].dtype == torch.float32
    assert sample["segment"].shape == (NUM_POINTS_PER_SCAN,)
    assert sample["segment"].dtype == torch.int64
    assert sample["instance"].shape == (NUM_POINTS_PER_SCAN,)
    assert sample["instance"].dtype == torch.int64
    assert isinstance(sample["sequence"], str) and isinstance(sample["frame"], str)


def test_semantickitti_dataset_test_split_has_no_labels(datasets_dir_factory: Callable[..., Path]) -> None:
    """Test split lacks `.label` files; samples expose `pos` / `intensity` only."""
    datasets_dir = datasets_dir_factory("SemanticKITTI/raw/**/*")
    dataset = SemanticKITTI(root=datasets_dir, split="test", sequences=("11",))
    sample = dataset[0]

    assert "segment" not in sample
    assert "instance" not in sample
    assert sample["pos"].shape == (NUM_POINTS_PER_SCAN, 3)
    assert sample["intensity"].shape == (NUM_POINTS_PER_SCAN, 1)


def test_semantickitti_dataset_raw_segment_ids(datasets_dir_factory: Callable[..., Path]) -> None:
    """Dataset returns RAW segment ids (no remap baked in)."""
    datasets_dir = datasets_dir_factory("SemanticKITTI/raw/**/*")
    dataset = SemanticKITTI(root=datasets_dir, split="val", sequences=("08",))

    valid_ids = set(SEMANTIC_KITTI_LABEL_NAMES.keys())
    for sample in dataset:
        unique = set(int(v) for v in sample["segment"].unique().tolist())
        assert unique.issubset(valid_ids), f"segment contained non-raw ids: {unique - valid_ids}"


def test_semantickitti_dataset_transform_called(datasets_dir_factory: Callable[..., Path]) -> None:
    """Transform is invoked exactly once per `__getitem__`."""
    datasets_dir = datasets_dir_factory("SemanticKITTI/raw/**/*")

    transform = Mock(side_effect=lambda data: data)
    dataset = SemanticKITTI(
        root=datasets_dir,
        split="val",
        sequences=("08",),
        transform=transform,
    )
    _ = list(dataset)
    assert transform.call_count == len(dataset)


def test_semantickitti_dataset_load_is_lazy(datasets_dir_factory: Callable[..., Path]) -> None:
    """Construction enumerates files but doesn't read scans; reads happen in `__getitem__`."""
    datasets_dir = datasets_dir_factory("SemanticKITTI/raw/**/*")

    transform = Mock(side_effect=lambda data: data)
    dataset = SemanticKITTI(
        root=datasets_dir,
        split="val",
        sequences=("08",),
        transform=transform,
    )
    assert transform.call_count == 0  # construction shouldn't trigger reads
    _ = dataset[0]
    assert transform.call_count == 1


def test_semantickitti_dataset_download_unsupported(datasets_dir_factory: Callable[..., Path]) -> None:
    """`download()` raises because SemanticKITTI must be downloaded manually."""
    datasets_dir = datasets_dir_factory("SemanticKITTI/raw/**/*")
    dataset = SemanticKITTI(root=datasets_dir, split="train", sequences=("00",))
    with pytest.raises(RuntimeError, match="does not support automatic download"):
        dataset.download()
