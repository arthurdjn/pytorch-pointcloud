# mypy: disable-error-code="arg-type,call-overload,attr-defined"
from pathlib import Path
from typing import Callable
from unittest.mock import Mock

import pytest
import torch

from torch_pointcloud.datasets import Semantic3D
from torch_pointcloud.datasets.semantic3d import (
    SEMANTIC3D_CLASSES,
    SEMANTIC3D_IGNORE_IDX,
    load_semantic3d_data,
)

# Must match generate.py's `--num-points` default.
NUM_POINTS_PER_SCENE = 1024
TRAIN_SCENE = "bildstein_station1_xyz_intensity_rgb"
TEST_SCENE = "MarketplaceFeldkirch_Station4_rgb_intensity-reduced"


def test_load_semantic3d_data_with_labels(datasets_dir: Path) -> None:
    """Loader returns positions, intensity, color, and segment for train scenes."""
    raw = datasets_dir / "Semantic3D" / "raw"
    data = load_semantic3d_data(raw / f"{TRAIN_SCENE}.txt", raw / f"{TRAIN_SCENE}.labels")

    assert isinstance(data["pos"], torch.Tensor)
    assert data["pos"].shape == (NUM_POINTS_PER_SCENE, 3)
    assert data["pos"].dtype == torch.float32

    assert isinstance(data["intensity"], torch.Tensor)
    assert data["intensity"].shape == (NUM_POINTS_PER_SCENE, 1)
    assert data["intensity"].dtype == torch.float32

    assert isinstance(data["color"], torch.Tensor)
    assert data["color"].shape == (NUM_POINTS_PER_SCENE, 3)
    assert data["color"].dtype == torch.uint8

    assert isinstance(data["segment"], torch.Tensor)
    assert data["segment"].shape == (NUM_POINTS_PER_SCENE,)
    assert data["segment"].dtype == torch.int64


def test_load_semantic3d_data_without_labels(datasets_dir: Path) -> None:
    """Loader skips `segment` when no `.labels` file is provided (held-out test scenes)."""
    raw = datasets_dir / "Semantic3D" / "raw"
    data = load_semantic3d_data(raw / f"{TEST_SCENE}.txt")

    assert "segment" not in data
    assert data["pos"].shape == (NUM_POINTS_PER_SCENE, 3)
    assert data["intensity"].shape == (NUM_POINTS_PER_SCENE, 1)
    assert data["color"].shape == (NUM_POINTS_PER_SCENE, 3)


def test_load_semantic3d_data_skips_missing_labels(
    datasets_dir_factory: Callable[..., Path],
) -> None:
    """If a `.labels` path is passed but doesn't exist, the loader still works."""
    datasets_dir = datasets_dir_factory("Semantic3D/raw/**/*")
    raw = datasets_dir / "Semantic3D" / "raw"
    data = load_semantic3d_data(raw / f"{TEST_SCENE}.txt", raw / f"{TEST_SCENE}.labels")
    assert "segment" not in data


def test_semantic3d_dataset_not_found() -> None:
    """Raises a clear error when the dataset directory is missing."""
    with pytest.raises(FileNotFoundError, match="Dataset not found"):
        _ = Semantic3D(root="not-found")


def test_semantic3d_dataset_invalid_split(datasets_dir_factory: Callable[..., Path]) -> None:
    """Unknown split values are rejected up-front."""
    datasets_dir = datasets_dir_factory("Semantic3D/raw/**/*")
    with pytest.raises(ValueError, match="Unknown split"):
        _ = Semantic3D(root=datasets_dir, split="bogus")


def test_semantic3d_dataset_train_scene(datasets_dir_factory: Callable[..., Path]) -> None:
    """Default `train` split exposes the fixture's train scene with labels."""
    datasets_dir = datasets_dir_factory("Semantic3D/raw/**/*")
    dataset = Semantic3D(root=datasets_dir, scenes=[TRAIN_SCENE])
    assert dataset.raw_files_exist()
    assert len(dataset) == 1

    sample = dataset[0]
    assert sample["pos"].shape == (NUM_POINTS_PER_SCENE, 3)
    assert sample["pos"].dtype == torch.float32
    assert sample["intensity"].shape == (NUM_POINTS_PER_SCENE, 1)
    assert sample["color"].shape == (NUM_POINTS_PER_SCENE, 3)
    assert sample["segment"].shape == (NUM_POINTS_PER_SCENE,)
    assert sample["name"] == TRAIN_SCENE


def test_semantic3d_dataset_test_scene_has_no_labels(
    datasets_dir_factory: Callable[..., Path],
) -> None:
    """Held-out test scenes lack `.labels`; loader should not fabricate a segment key."""
    datasets_dir = datasets_dir_factory("Semantic3D/raw/**/*")
    dataset = Semantic3D(root=datasets_dir, split="test", scenes=[TEST_SCENE])
    sample = dataset[0]
    assert "segment" not in sample
    assert sample["pos"].shape == (NUM_POINTS_PER_SCENE, 3)
    assert sample["name"] == TEST_SCENE


def test_semantic3d_dataset_custom_scenes_override_split(
    datasets_dir_factory: Callable[..., Path],
) -> None:
    """Passing `scenes=` lets users mix and match scenes for ad-hoc tasks."""
    datasets_dir = datasets_dir_factory("Semantic3D/raw/**/*")
    dataset = Semantic3D(root=datasets_dir, scenes=[TRAIN_SCENE, TEST_SCENE])
    assert dataset.scenes == (TRAIN_SCENE, TEST_SCENE)
    assert len(dataset) == 2


def test_semantic3d_dataset_segment_ids_in_class_set(
    datasets_dir_factory: Callable[..., Path],
) -> None:
    """Segment ids returned for train scenes stay within the documented 9-class range."""
    datasets_dir = datasets_dir_factory("Semantic3D/raw/**/*")
    dataset = Semantic3D(root=datasets_dir, scenes=[TRAIN_SCENE])
    sample = dataset[0]

    valid_ids = set(range(len(SEMANTIC3D_CLASSES)))
    unique = set(int(v) for v in sample["segment"].unique().tolist())
    assert unique.issubset(valid_ids)


def test_semantic3d_dataset_ignore_index_in_class_set() -> None:
    """`SEMANTIC3D_IGNORE_IDX` matches the documented `unlabelled` class."""
    assert SEMANTIC3D_CLASSES[SEMANTIC3D_IGNORE_IDX] == "unlabelled"


def test_semantic3d_dataset_transform_called(datasets_dir_factory: Callable[..., Path]) -> None:
    """Transform is invoked exactly once per `__getitem__`."""
    datasets_dir = datasets_dir_factory("Semantic3D/raw/**/*")

    transform = Mock(side_effect=lambda data: data)
    dataset = Semantic3D(root=datasets_dir, scenes=[TRAIN_SCENE], transform=transform)
    _ = list(dataset)
    assert transform.call_count == len(dataset)


def test_semantic3d_dataset_load_is_lazy(datasets_dir_factory: Callable[..., Path]) -> None:
    """Construction enumerates files but doesn't read scenes; reads happen in `__getitem__`."""
    datasets_dir = datasets_dir_factory("Semantic3D/raw/**/*")

    transform = Mock(side_effect=lambda data: data)
    dataset = Semantic3D(root=datasets_dir, scenes=[TRAIN_SCENE], transform=transform)
    assert transform.call_count == 0
    _ = dataset[0]
    assert transform.call_count == 1


def test_semantic3d_dataset_download_unsupported(datasets_dir_factory: Callable[..., Path]) -> None:
    """`download()` raises because Semantic3D must be downloaded manually."""
    datasets_dir = datasets_dir_factory("Semantic3D/raw/**/*")
    dataset = Semantic3D(root=datasets_dir, scenes=[TRAIN_SCENE])
    with pytest.raises(RuntimeError, match="does not support automatic download"):
        dataset.download()
