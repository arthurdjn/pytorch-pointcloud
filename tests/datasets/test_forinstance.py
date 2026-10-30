from pathlib import Path
from unittest.mock import Mock

import pytest
import torch

from torch_pointcloud.datasets import FORinstance
from torch_pointcloud.datasets.forinstance import FORINSTANCE_CLASSES, FORINSTANCE_OUT_POINTS, load_forinstance_data
from torch_pointcloud.utils.data import DataKeys
from torch_pointcloud.utils.imports import _LASPY_AVAILABLE

pytestmark = pytest.mark.skipif(not _LASPY_AVAILABLE, reason="laspy is not installed")

# Must match generate.py's `PLOTS`, `NUM_TREES`, `--points-per-tree` default, header offsets and `PLOT_SIZE`.
PLOTS = {True: ["CULS_plot_1_annotated"], False: ["CULS_plot_2_annotated", "TUWIEN_test"]}
NUM_POINTS = 4 * 600 + 600 + 4 * 600
UTM_ORIGIN = torch.tensor([600000.0, 6600000.0, 100.0], dtype=torch.float64)


@pytest.mark.parametrize(
    "name", [pytest.param("CULS/plot_2_annotated.las", id="int32"), pytest.param("TUWIEN/test.las", id="float64")]
)
def test_load_forinstance_data(datasets_dir: Path, name: str) -> None:
    """Every plot reads to the documented keys: UTM positions at the millimetre of the LAS file, tree ids as int64
    whatever their LAS type."""
    data = load_forinstance_data(datasets_dir / "FORinstance" / "raw" / name)
    local = data[DataKeys.POS] - UTM_ORIGIN

    assert data[DataKeys.POS].shape == (NUM_POINTS, 3)
    assert data[DataKeys.POS].dtype == torch.float64
    assert local.min() > -1.0 and local.max() < 30.0
    torch.testing.assert_close(local, local.mul(1000).round().div(1000), rtol=0.0, atol=1e-6)
    assert data[DataKeys.SEGMENT].dtype == data[DataKeys.INSTANCE].dtype == torch.int64
    assert int(data[DataKeys.SEGMENT].max()) < len(FORINSTANCE_CLASSES)


def test_load_forinstance_data_marks_points_of_no_tree(datasets_dir: Path) -> None:
    data = load_forinstance_data(datasets_dir / "FORinstance" / "raw" / "CULS" / "plot_2_annotated.las")
    segment, instance = data[DataKeys.SEGMENT], data[DataKeys.INSTANCE]

    assert (instance[segment <= 2] == -1).all()
    assert (instance[segment >= 4] > 0).all()
    assert instance[segment == FORINSTANCE_OUT_POINTS].unique().numel() == 1


@pytest.mark.parametrize("train", [pytest.param(True, id="dev"), pytest.param(False, id="test")])
def test_forinstance_dataset_lists_the_plots_of_the_split(datasets_dir: Path, train: bool) -> None:
    """Plots of the split file that the archive does not hold are skipped."""
    dataset = FORinstance(root=datasets_dir, train=train)

    assert [dataset[i][DataKeys.NAME] for i in range(len(dataset))] == PLOTS[train]


def test_forinstance_dataset_applies_the_transform(datasets_dir: Path) -> None:
    transform = Mock(side_effect=lambda data: data)
    dataset = FORinstance(root=datasets_dir, train=False, transform=transform)
    sample = dataset[0]

    transform.assert_called_once()
    assert transform.call_args.args[0] is sample


def test_forinstance_dataset_missing_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="download=True"):
        FORinstance(root=tmp_path)


def test_forinstance_download_checks_the_archive(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A corrupted archive is refused before extraction."""
    download = Mock()
    monkeypatch.setattr("torch_pointcloud.datasets.forinstance.download_url", download)
    monkeypatch.setattr("torch_pointcloud.datasets.forinstance.is_hash_valid", Mock(return_value=False))
    with pytest.raises(RuntimeError, match="MD5"):
        FORinstance(root=tmp_path, download=True, show_progress=False)
    assert download.call_args.args[0] == FORinstance.data_url
