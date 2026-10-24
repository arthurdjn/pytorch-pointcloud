import zipfile
from pathlib import Path
from typing import Any, Callable, List
from unittest.mock import Mock, patch

import pytest
import torch

from torch_pointcloud.datasets import FRACTAL
from torch_pointcloud.datasets.fractal import FRACTAL_CLASSES, FRACTAL_LEARNING_MAP, load_fractal_data
from torch_pointcloud.utils.data import DataKeys
from torch_pointcloud.utils.imports import _LASPY_AVAILABLE

pytestmark = pytest.mark.skipif(not _LASPY_AVAILABLE, reason="laspy is not installed")

# Must match generate.py's `NUM_ARCHIVES` and `--num-points` default.
NUM_PATCHES = 10
NUM_POINTS_PER_PATCH = 1024


def test_load_fractal_data_shapes(datasets_dir: Path) -> None:
    """Loader returns the documented keys, dtypes and shapes."""
    with zipfile.ZipFile(datasets_dir / "FRACTAL" / "raw" / "test-00.zip") as archive:
        content = archive.read(archive.namelist()[0])
    data = load_fractal_data(content)

    expected = {
        DataKeys.POS: ((NUM_POINTS_PER_PATCH, 3), torch.float64),
        DataKeys.INTENSITY: ((NUM_POINTS_PER_PATCH, 1), torch.float32),
        DataKeys.RETURN_NUMBER: ((NUM_POINTS_PER_PATCH, 1), torch.int64),
        DataKeys.NUMBER_OF_RETURNS: ((NUM_POINTS_PER_PATCH, 1), torch.int64),
        DataKeys.COLOR: ((NUM_POINTS_PER_PATCH, 3), torch.int32),
        DataKeys.NIR: ((NUM_POINTS_PER_PATCH, 1), torch.int32),
        DataKeys.SEGMENT: ((NUM_POINTS_PER_PATCH,), torch.int64),
    }
    assert set(data) == set(expected)
    for key, (shape, dtype) in expected.items():
        assert data[key].shape == shape, key
        assert data[key].dtype == dtype, key


def test_load_fractal_data_positions_are_lambert93(datasets_dir: Path) -> None:
    """Positions are the Lambert-93 coordinates of the patch, finer than the 0.5 m that float32 resolves there."""
    with zipfile.ZipFile(datasets_dir / "FRACTAL" / "raw" / "test-00.zip") as archive:
        content = archive.read(archive.namelist()[0])
    pos = load_fractal_data(content)[DataKeys.POS]

    corner = torch.tensor([743050.0, 6246650.0], dtype=torch.float64)
    assert (pos[:, :2] >= corner).all()
    assert (pos[:, :2] <= corner + 50.0).all()
    assert (pos[:, 1] % 0.5 != 0).any()


def test_load_fractal_data_reads_a_path(datasets_dir: Path, tmp_path: Path) -> None:
    """A file path reads the same as its bytes."""
    with zipfile.ZipFile(datasets_dir / "FRACTAL" / "raw" / "test-00.zip") as archive:
        content = archive.read(archive.namelist()[0])
    path = tmp_path / "patch.las"
    path.write_bytes(content)

    from_path, from_bytes = load_fractal_data(path), load_fractal_data(content)
    for key, value in from_bytes.items():
        assert torch.equal(from_path[key], value), key


def test_fractal_dataset_reads_every_patch(datasets_dir: Path) -> None:
    """The test split lists the patches of its 10 archives, each with its file name."""
    dataset = FRACTAL(root=datasets_dir, split="test")

    assert len(dataset) == NUM_PATCHES
    names = [dataset[i][DataKeys.NAME] for i in range(len(dataset))]
    assert names == sorted(names)
    assert all(name.startswith("TEST-") for name in names)


def test_fractal_dataset_codes_cover_the_classes(datasets_dir: Path) -> None:
    """Every class of the learning map is reachable from the codes of the fixture."""
    dataset = FRACTAL(root=datasets_dir, split="test")
    codes = set(dataset[0][DataKeys.SEGMENT].unique().tolist())

    assert {FRACTAL_LEARNING_MAP[code] for code in codes if code in FRACTAL_LEARNING_MAP} == set(
        range(len(FRACTAL_CLASSES))
    )


def test_fractal_dataset_transform_called(datasets_dir: Path) -> None:
    """The transform runs once per `__getitem__`, not at construction."""
    transform = Mock(side_effect=lambda data: data)
    dataset = FRACTAL(root=datasets_dir, split="test", transform=transform)
    assert transform.call_count == 0

    _ = dataset[0]
    assert transform.call_count == 1


def test_fractal_dataset_invalid_split(datasets_dir: Path) -> None:
    """Unknown split values are rejected up-front."""
    with pytest.raises(ValueError, match="Unknown split"):
        _ = FRACTAL(root=datasets_dir, split="bogus")  # type: ignore[arg-type]


def test_fractal_dataset_not_found(datasets_dir: Path) -> None:
    """A split whose archives are missing raises a clear error."""
    with pytest.raises(FileNotFoundError, match="Dataset not found"):
        _ = FRACTAL(root=datasets_dir, split="val")


def test_fractal_dataset_download_fetches_missing_archives(datasets_dir: Path) -> None:
    """`download=True` asks for every archive of the split, keeping the complete ones."""
    calls: List[Any] = []
    with patch("torch_pointcloud.datasets.fractal.download_url", side_effect=lambda *a, **k: calls.append((a, k))):
        dataset = FRACTAL(root=datasets_dir, split="test", download=True, show_progress=False)

    assert len(dataset) == NUM_PATCHES
    assert [Path(args[1]).name for args, _ in calls] == [f"test-{i:02d}.zip" for i in range(NUM_PATCHES)]
    assert all(args[0] == f"{FRACTAL.data_url}/data/test/{Path(args[1]).name}" for args, _ in calls)
    assert all(kwargs["overwrite"] == "incomplete" for _, kwargs in calls)


def test_fractal_dataset_download_force(datasets_dir_factory: Callable[..., Path]) -> None:
    """`download(force=True)` downloads every archive again."""
    datasets_dir = datasets_dir_factory()
    dataset = FRACTAL(root=datasets_dir, split="test", show_progress=False)
    with patch("torch_pointcloud.datasets.fractal.download_url") as download_url:
        dataset.download(force=True)

    assert download_url.call_count == NUM_PATCHES
    assert all(call.kwargs["overwrite"] is True for call in download_url.call_args_list)
