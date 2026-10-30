from pathlib import Path
from typing import Callable
from unittest.mock import Mock

import pytest
import torch

from torch_pointcloud.datasets import GridNetHD
from torch_pointcloud.datasets.gridnethd import (
    GRIDNETHD_CLASSES,
    GRIDNETHD_IGNORE_IDX,
    GRIDNETHD_LABEL_GROUPS,
    load_gridnethd_data,
)
from torch_pointcloud.utils.data import DataKeys
from torch_pointcloud.utils.imports import _LASPY_AVAILABLE

pytestmark = pytest.mark.skipif(not _LASPY_AVAILABLE, reason="laspy is not installed")

# Must match generate.py's `SPLIT` and `--num-points` default.
SPLIT = {"train": ["t1z6a"], "val": ["t1z8"], "test": ["t1z4"]}
NUM_POINTS_PER_ZONE = 1024
# The grouping table of the release card: annotated `ground_truth` id to benchmark class. 12, 13 and 255 are
# ignored, as is 22, which the card does not group.
RELEASE_GROUPS = {
    0: "pylon",
    1: "pylon",
    2: "pylon",
    3: "pylon",
    4: "pylon",
    5: "conductor_cable",
    6: "structural_cable",
    7: "structural_cable",
    8: "insulator",
    9: "insulator",
    10: "insulator",
    11: "insulator",
    14: "high_vegetation",
    15: "low_vegetation",
    16: "herbaceous_vegetation",
    17: "rock_gravel_soil",
    18: "rock_gravel_soil",
    19: "impervious_soil",
    20: "water",
    21: "building",
}


def test_load_gridnethd_data_shapes(datasets_dir: Path) -> None:
    """Loader returns the documented keys, dtypes and shapes."""
    data = load_gridnethd_data(datasets_dir / "GridNetHD" / "raw" / "t1z8" / "lidar" / "t1z8.las")

    expected = {
        DataKeys.POS: ((NUM_POINTS_PER_ZONE, 3), torch.float32),
        DataKeys.COLOR: ((NUM_POINTS_PER_ZONE, 3), torch.uint8),
        DataKeys.INTENSITY: ((NUM_POINTS_PER_ZONE, 1), torch.float32),
        DataKeys.RETURN_NUMBER: ((NUM_POINTS_PER_ZONE, 1), torch.int64),
        DataKeys.NUMBER_OF_RETURNS: ((NUM_POINTS_PER_ZONE, 1), torch.int64),
        DataKeys.SEGMENT: ((NUM_POINTS_PER_ZONE,), torch.int64),
    }
    assert set(data) == set(expected)
    for key, (shape, dtype) in expected.items():
        assert data[key].shape == shape, key
        assert data[key].dtype == dtype, key


def test_load_gridnethd_data_positions_relative_to_corner(datasets_dir: Path) -> None:
    """Positions start at the minimum corner of the zone, so UTM coordinates keep their precision."""
    pos = load_gridnethd_data(datasets_dir / "GridNetHD" / "raw" / "t1z8" / "lidar" / "t1z8.las")[DataKeys.POS]

    assert torch.allclose(pos.min(dim=0).values, torch.zeros(3))
    assert (pos[:, :2].max(dim=0).values <= 200.0).all()


def test_load_gridnethd_data_test_zone_has_no_labels(datasets_dir: Path) -> None:
    """A zone without the `ground_truth` field returns no `segment`."""
    data = load_gridnethd_data(datasets_dir / "GridNetHD" / "raw" / "t1z4" / "lidar" / "t1z4.las")
    assert DataKeys.SEGMENT not in data


@pytest.mark.parametrize("split", ["train", "val", "test"])
def test_gridnethd_dataset_reads_the_zones_of_the_split(datasets_dir: Path, split: str) -> None:
    """Each split lists its zones as `split.json` does, each sample named after its zone."""
    dataset = GridNetHD(root=datasets_dir, split=split)  # type: ignore[arg-type]

    assert dataset.raw_files_exist()
    assert len(dataset) == len(SPLIT[split])
    assert [dataset[i][DataKeys.NAME] for i in range(len(dataset))] == SPLIT[split]


def test_gridnethd_dataset_segment_is_the_release_group_of_each_point(datasets_dir: Path) -> None:
    """Each point's `segment` is the class the release card groups its annotated `ground_truth` id into."""
    laspy = pytest.importorskip("laspy")
    las = laspy.read((datasets_dir / "GridNetHD" / "raw" / "t1z8" / "lidar" / "t1z8.las").as_posix())
    ground_truth = las["ground_truth"].tolist()
    expected = [
        GRIDNETHD_CLASSES.index(RELEASE_GROUPS[label]) if label in RELEASE_GROUPS else GRIDNETHD_IGNORE_IDX
        for label in ground_truth
    ]

    segment = GridNetHD(root=datasets_dir, split="val")[0][DataKeys.SEGMENT]

    assert set(RELEASE_GROUPS) <= set(ground_truth)
    assert segment.tolist() == expected


def test_gridnethd_label_groups_cover_the_benchmark_classes() -> None:
    assert len(GRIDNETHD_CLASSES) == 11
    assert len(GRIDNETHD_LABEL_GROUPS) == len(GRIDNETHD_CLASSES)
    annotated = sorted(label for group in GRIDNETHD_LABEL_GROUPS for label in group)
    assert annotated == [label for label in range(22) if label not in (12, 13)]


def test_gridnethd_dataset_invalid_split(datasets_dir: Path) -> None:
    with pytest.raises(ValueError, match="Invalid split"):
        _ = GridNetHD(root=datasets_dir, split="validation")  # type: ignore[arg-type]


def test_gridnethd_dataset_transform_called(datasets_dir: Path) -> None:
    """The transform runs once per `__getitem__`, not at construction."""
    transform = Mock(side_effect=lambda data: data)
    dataset = GridNetHD(root=datasets_dir, split="val", transform=transform)
    assert transform.call_count == 0

    _ = dataset[0]
    assert transform.call_count == 1


def test_gridnethd_dataset_not_found(datasets_dir_factory: Callable[..., Path]) -> None:
    """A split whose zones are missing raises a clear error."""
    datasets_dir = datasets_dir_factory("GridNetHD/raw/t1z8/*")
    with pytest.raises(FileNotFoundError, match="Dataset not found"):
        _ = GridNetHD(root=datasets_dir, split="val")


def test_gridnethd_dataset_download_fetches_the_split_and_its_zones(
    datasets_dir_factory: Callable[..., Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """`download=True` fetches `split.json` first, then the `.las` file of every zone of the split."""
    datasets_dir = datasets_dir_factory("GridNetHD/README.md")
    split_path = datasets_dir / "GridNetHD" / "raw" / "split.json"
    source_root = Path(__file__).resolve().parents[1] / "data" / "datasets" / "GridNetHD" / "raw"

    def fake_download(url: str, file_path: str, **kwargs: object) -> str:
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        Path(file_path).write_bytes((source_root / url.split(f"/{GridNetHD.revision}/", 1)[1]).read_bytes())
        return file_path

    download = Mock(side_effect=fake_download)
    monkeypatch.setattr("torch_pointcloud.datasets.gridnethd.download_url", download)

    dataset = GridNetHD(root=datasets_dir, split="val", download=True, show_progress=False)

    base_url = f"{GridNetHD.data_url}/resolve/{GridNetHD.revision}"
    assert [call.args[0] for call in download.call_args_list] == [
        f"{base_url}/split.json",
        f"{base_url}/t1z8/lidar/t1z8.las",
    ]
    assert download.call_args_list[0].args[1] == split_path.as_posix()
    assert [call.kwargs["description"] for call in download.call_args_list] == [
        "Downloading split.json",
        "Downloading t1z8.las",
    ]
    assert all(call.kwargs["show_progress"] is False for call in download.call_args_list)
    assert len(dataset) == 1
