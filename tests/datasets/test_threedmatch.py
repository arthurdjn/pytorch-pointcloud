import json
from pathlib import Path
from typing import Callable

import pytest
import torch

from torch_pointcloud.datasets import ThreeDMatch
from torch_pointcloud.datasets.threedmatch import load_threedmatch_fragment
from torch_pointcloud.ops.geometry import transform_points
from torch_pointcloud.utils.data import DataKeys, PointCloudDataLoader


def test_load_threedmatch_fragment(datasets_dir_factory: Callable[..., Path]) -> None:
    datasets_dir = datasets_dir_factory("ThreeDMatch/**/*")
    points = load_threedmatch_fragment(
        datasets_dir / "ThreeDMatch" / "raw" / "indoor" / "test" / "7-scenes-redkitchen" / "cloud_bin_0.pth"
    )
    assert points.ndim == 2 and points.shape[1] == 3
    assert points.dtype.name == "float32"


def test_load_threedmatch_fragment_refuses_other_globals(tmp_path: Path) -> None:
    torch.save({"a": torch.zeros(2)}, tmp_path / "bad.pth", _use_new_zipfile_serialization=False)
    with pytest.raises(Exception, match="Refusing"):
        load_threedmatch_fragment(tmp_path / "bad.pth")


def test_threedmatch_dataset_not_found() -> None:
    with pytest.raises(RuntimeError, match="not found"):
        ThreeDMatch(root="not-found", split="test")


@pytest.mark.parametrize(
    "split, benchmark, expected",
    [
        pytest.param("test", "3DMatch", 6, id="3dmatch"),
        pytest.param("test", "3DLoMatch", 2, id="3dlomatch"),
        pytest.param("train", "3DMatch", 3, id="train"),
        pytest.param("val", "3DMatch", 1, id="val"),
    ],
)
def test_threedmatch_splits(
    datasets_dir_factory: Callable[..., Path], split: str, benchmark: str, expected: int
) -> None:
    datasets_dir = datasets_dir_factory("ThreeDMatch/**/*")
    dataset = ThreeDMatch(root=datasets_dir, split=split, benchmark=benchmark, show_progress=False)  # type: ignore[arg-type]
    assert len(dataset) == expected

    sample = dataset[0]
    assert sample[DataKeys.POS].ndim == 2 and sample[DataKeys.POS].shape[1] == 3
    assert sample[DataKeys.REF_POS].ndim == 2 and sample[DataKeys.REF_POS].shape[1] == 3
    assert sample[DataKeys.POS].dtype == torch.float32
    assert sample[DataKeys.TRANSFORM].shape == (4, 4)
    assert sample[DataKeys.FRAME].tolist() == ([0, 1] if benchmark == "3DMatch" else [0, 2])
    assert isinstance(sample[DataKeys.SCENE], str)
    assert (DataKeys.INFORMATION in sample) == (split == "test")

    # The ground truth aligns the source onto the reference: most moved source points land on a reference point.
    aligned = transform_points(sample[DataKeys.POS], sample[DataKeys.TRANSFORM])
    nearest = torch.cdist(aligned, sample[DataKeys.REF_POS]).min(dim=1).values
    assert nearest.median().item() < 1e-4


def test_threedmatch_processed_cache_is_reused(datasets_dir_factory: Callable[..., Path]) -> None:
    datasets_dir = datasets_dir_factory("ThreeDMatch/**/*")
    first = ThreeDMatch(root=datasets_dir, split="test", show_progress=False)
    pairs_file = Path(first.processed_dir, "pairs", "3DMatch.json")
    assert pairs_file.exists()
    stamp = pairs_file.stat().st_mtime_ns
    second = ThreeDMatch(root=datasets_dir, split="test", show_progress=False)
    assert pairs_file.stat().st_mtime_ns == stamp
    assert len(second) == len(first)


def test_threedmatch_process_stamps_each_pair_list(datasets_dir_factory: Callable[..., Path]) -> None:
    """Each pair list gets its own `cache_meta` stamp, so the 3DMatch cache does not stand in for the 3DLoMatch one."""
    datasets_dir = datasets_dir_factory("ThreeDMatch/**/*")
    dataset = ThreeDMatch(root=datasets_dir, split="test", show_progress=False, force_process=True)
    meta_path = Path(dataset.processed_dir, "pairs", "3DMatch.meta.json")
    assert json.loads(meta_path.read_text()) == dataset.cache_meta
    assert not Path(dataset.processed_dir, "pairs", "3DLoMatch.meta.json").exists()

    lomatch = ThreeDMatch(root=datasets_dir, split="test", benchmark="3DLoMatch", show_progress=False)
    assert Path(lomatch.processed_dir, "pairs", "3DLoMatch.json").exists()
    assert Path(lomatch.processed_dir, "pairs", "3DLoMatch.meta.json").exists()


def test_threedmatch_collate(datasets_dir_factory: Callable[..., Path]) -> None:
    datasets_dir = datasets_dir_factory("ThreeDMatch/**/*")
    dataset = ThreeDMatch(root=datasets_dir, split="test", show_progress=False)
    loader = PointCloudDataLoader(
        dataset,
        batch_size=2,
        cat_keys=[DataKeys.REF_POS],
        stack_keys=[DataKeys.TRANSFORM, DataKeys.INFORMATION, DataKeys.FRAME],
    )
    data = next(iter(loader))
    num_src = [len(dataset[i][DataKeys.POS]) for i in range(2)]
    num_ref = [len(dataset[i][DataKeys.REF_POS]) for i in range(2)]
    assert data[DataKeys.POS].shape == (sum(num_src), 3)
    assert data[DataKeys.BATCH].tolist() == [0] * num_src[0] + [1] * num_src[1]
    assert data[DataKeys.BATCH_REF_POS].tolist() == [0] * num_ref[0] + [1] * num_ref[1]
    assert data[DataKeys.TRANSFORM].shape == (2, 4, 4)
    assert data[DataKeys.INFORMATION].shape == (2, 6, 6)
    assert data[DataKeys.FRAME].shape == (2, 2)
    assert len(data[DataKeys.SCENE]) == 2
