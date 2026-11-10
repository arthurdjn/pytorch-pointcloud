from pathlib import Path
from typing import Callable
from unittest.mock import Mock

import pytest
import torch

from torch_pointcloud.datasets import DALES
from torch_pointcloud.datasets.dales import DALES_CLASSES, DALES_IGNORE_IDX, load_dales_data
from torch_pointcloud.utils.data import DataKeys
from torch_pointcloud.utils.imports import _LASPY_AVAILABLE

pytestmark = pytest.mark.skipif(not _LASPY_AVAILABLE, reason="laspy is not installed")

# Must match generate.py's `TILES` and `--num-points` default.
TILES = {"train": ["5080_54435", "5190_54400"], "test": ["5080_54470", "5100_54440"]}
NUM_POINTS_PER_TILE = 1024


def test_load_dales_data_shapes(datasets_dir: Path) -> None:
    """Loader returns the documented keys, dtypes and shapes."""
    data = load_dales_data(datasets_dir / "DALES" / "raw" / "dales_las" / "test" / "5080_54470.las")

    expected = {
        DataKeys.POS: ((NUM_POINTS_PER_TILE, 3), torch.float32),
        DataKeys.RETURN_NUMBER: ((NUM_POINTS_PER_TILE, 1), torch.int64),
        DataKeys.NUMBER_OF_RETURNS: ((NUM_POINTS_PER_TILE, 1), torch.int64),
        DataKeys.SEGMENT: ((NUM_POINTS_PER_TILE,), torch.int64),
    }
    assert set(data) == set(expected)
    for key, (shape, dtype) in expected.items():
        assert data[key].shape == shape, key
        assert data[key].dtype == dtype, key


def test_load_dales_data_positions_relative_to_corner(datasets_dir: Path) -> None:
    """Positions start at the minimum corner of the tile, so UTM coordinates keep their precision."""
    pos = load_dales_data(datasets_dir / "DALES" / "raw" / "dales_las" / "test" / "5080_54470.las")[DataKeys.POS]

    assert torch.allclose(pos.min(dim=0).values, torch.zeros(3))
    assert (pos[:, :2].max(dim=0).values <= 500.0).all()


@pytest.mark.parametrize("train", [True, False])
def test_dales_dataset_reads_every_tile(datasets_dir: Path, train: bool) -> None:
    """Each split lists its tiles in name order, each with its file name."""
    dataset = DALES(root=datasets_dir, train=train)

    assert dataset.raw_files_exist()
    assert len(dataset) == 2
    assert [dataset[i][DataKeys.NAME] for i in range(len(dataset))] == TILES["train" if train else "test"]


def test_dales_dataset_segment_ids_valid(datasets_dir: Path) -> None:
    """Segment ids stay within the 9-class range of the release."""
    dataset = DALES(root=datasets_dir, train=False)
    for index in range(len(dataset)):
        segment = dataset[index][DataKeys.SEGMENT]
        assert int(segment.min()) >= 0
        assert int(segment.max()) < len(DALES_CLASSES)


def test_dales_ignore_index_in_class_set() -> None:
    assert DALES_CLASSES[DALES_IGNORE_IDX] == "unknown"
    assert len(DALES_CLASSES) == 9


def test_dales_dataset_transform_called(datasets_dir: Path) -> None:
    """The transform runs once per `__getitem__`, not at construction."""
    transform = Mock(side_effect=lambda data: data)
    dataset = DALES(root=datasets_dir, train=False, transform=transform)
    assert transform.call_count == 0

    _ = dataset[0]
    assert transform.call_count == 1


def test_dales_dataset_not_found(datasets_dir_factory: Callable[..., Path]) -> None:
    """A split without tiles raises a clear error."""
    datasets_dir = datasets_dir_factory("DALES/raw/dales_las/train/*")
    with pytest.raises(FileNotFoundError, match="Dataset not found"):
        _ = DALES(root=datasets_dir, train=False)


def test_dales_dataset_download_unsupported(datasets_dir: Path) -> None:
    """Automatic download is not supported: the archive is requested from the authors."""
    dataset = DALES(root=datasets_dir, train=False)
    with pytest.raises(RuntimeError, match="does not support automatic download"):
        dataset.download()
