import json
from pathlib import Path
from typing import Callable

import pytest
import torch

import torch_pointcloud.transforms as T
from torch_pointcloud.datasets import KITTIOdometry
from torch_pointcloud.ops.geometry import transform_points
from torch_pointcloud.utils.data import DataKeys, PointCloudDataLoader
from torch_pointcloud.utils.imports import _PYG_LIB_AVAILABLE


def test_kittiodometry_dataset_not_found() -> None:
    with pytest.raises(RuntimeError, match="not found"):
        KITTIOdometry(root="not-found", split="test")


def test_kittiodometry_unknown_split(datasets_dir_factory: Callable[..., Path]) -> None:
    datasets_dir = datasets_dir_factory("KITTIOdometry/**/*")
    with pytest.raises(ValueError, match="split"):
        KITTIOdometry(root=datasets_dir, split="trainval", show_progress=False)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "split, sequence, expected",
    [
        pytest.param("train", "00", 3, id="train"),
        pytest.param("val", "06", 1, id="val"),
        pytest.param("test", "08", 3, id="test"),
    ],
)
def test_kittiodometry_splits(
    datasets_dir_factory: Callable[..., Path], split: str, sequence: str, expected: int
) -> None:
    datasets_dir = datasets_dir_factory("KITTIOdometry/**/*")
    dataset = KITTIOdometry(root=datasets_dir, split=split, show_progress=False)  # type: ignore[arg-type]
    assert len(dataset) == expected

    sample = dataset[0]
    assert sample[DataKeys.POS].ndim == 2 and sample[DataKeys.POS].shape[1] == 3
    assert sample[DataKeys.REF_POS].ndim == 2 and sample[DataKeys.REF_POS].shape[1] == 3
    assert sample[DataKeys.POS].dtype == torch.float32
    assert sample[DataKeys.TRANSFORM].shape == (4, 4)
    assert (int(sample[DataKeys.FRAME]), int(sample[DataKeys.REF_FRAME])) == (1, 0)
    assert sample[DataKeys.SEQUENCE] == sequence


def test_kittiodometry_transform_aligns_the_pair(datasets_dir_factory: Callable[..., Path]) -> None:
    datasets_dir = datasets_dir_factory("KITTIOdometry/**/*")
    dataset = KITTIOdometry(root=datasets_dir, split="test", show_progress=False)
    sample = dataset[2]
    aligned = transform_points(sample[DataKeys.POS], sample[DataKeys.TRANSFORM])
    nearest = torch.cdist(aligned, sample[DataKeys.REF_POS]).min(dim=1).values
    # The scans are crops of one sampling of the road, so the overlap coincides once aligned.
    assert nearest.median().item() < 1e-3


@pytest.mark.skipif(not _PYG_LIB_AVAILABLE, reason="pyg-lib is not installed")
def test_kittiodometry_voxelize_transform(datasets_dir_factory: Callable[..., Path]) -> None:
    datasets_dir = datasets_dir_factory("KITTIOdometry/**/*")
    transform = T.Compose(
        [
            T.Voxelize(pos_key=DataKeys.POS, pos_reduce="mean", size=0.3, origin="min"),
            T.Voxelize(pos_key=DataKeys.REF_POS, pos_reduce="mean", size=0.3, origin="min"),
        ]
    )
    raw = KITTIOdometry(root=datasets_dir, split="test", show_progress=False)
    voxelized = KITTIOdometry(root=datasets_dir, split="test", transform=transform, show_progress=False)
    assert voxelized[0][DataKeys.POS].size(0) < raw[0][DataKeys.POS].size(0)
    assert voxelized[0][DataKeys.REF_POS].size(0) < raw[0][DataKeys.REF_POS].size(0)
    assert torch.equal(voxelized[0][DataKeys.TRANSFORM], raw[0][DataKeys.TRANSFORM])


def test_kittiodometry_processed_cache_is_reused(datasets_dir_factory: Callable[..., Path]) -> None:
    datasets_dir = datasets_dir_factory("KITTIOdometry/**/*")
    first = KITTIOdometry(root=datasets_dir, split="test", show_progress=False)
    pairs_file = Path(first.processed_dir, "pairs", "test.json")
    meta_file = Path(first.processed_dir, "pairs", "test.meta.json")
    assert pairs_file.exists()
    assert json.loads(meta_file.read_text()) == first.cache_meta
    stamp = pairs_file.stat().st_mtime_ns
    second = KITTIOdometry(root=datasets_dir, split="test", show_progress=False)
    assert pairs_file.stat().st_mtime_ns == stamp
    assert len(second) == len(first)


def test_kittiodometry_collate(datasets_dir_factory: Callable[..., Path]) -> None:
    datasets_dir = datasets_dir_factory("KITTIOdometry/**/*")
    dataset = KITTIOdometry(root=datasets_dir, split="test", show_progress=False)
    loader = PointCloudDataLoader(
        dataset,
        batch_size=2,
        cat_keys=[DataKeys.REF_POS],
        stack_keys=[DataKeys.TRANSFORM],
    )
    batch = next(iter(loader))
    assert batch[DataKeys.TRANSFORM].shape == (2, 4, 4)
    assert batch[DataKeys.FRAME].shape == (2,) and batch[DataKeys.REF_FRAME].shape == (2,)
    assert batch[DataKeys.BATCH].max().item() == 1
    assert batch[DataKeys.BATCH_REF_POS].size(0) == batch[DataKeys.REF_POS].size(0)
    assert batch[DataKeys.SEQUENCE] == ["08", "08"]
