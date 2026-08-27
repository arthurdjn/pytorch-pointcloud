import shutil
from pathlib import Path
from typing import Callable, Dict, Optional

import pytest
import torch
from torch.utils.data import Dataset

from torch_pointcloud.datasets import (
    S3DIS,
    ModelNetNormalResampled,
    S3DISHdf5,
    ScanNet20,
    ScanObjectNN,
    SemanticKITTI,
    ShapeNetPart,
)

DATA_DIR = Path(__file__).parent / "data"
DATASETS_DIR = DATA_DIR / "datasets"
MODELS_DIR = DATA_DIR / "models"


def modelnet_dataset(transform: Optional[Callable] = None) -> ModelNetNormalResampled:
    return ModelNetNormalResampled(
        root=DATASETS_DIR,
        variant="40",
        train=False,
        show_progress=False,
        transform=transform,
    )


def scanobjectnn_dataset(transform: Optional[Callable] = None) -> ScanObjectNN:
    return ScanObjectNN(
        root=DATASETS_DIR,
        train=False,
        partition="split1",
        background=False,
        show_progress=False,
        transform=transform,
    )


def shapenetpart_dataset(transform: Optional[Callable] = None) -> ShapeNetPart:
    return ShapeNetPart(
        root=DATASETS_DIR,
        split="test",
        show_progress=False,
        transform=transform,
    )


def s3dis_dataset(transform: Optional[Callable] = None) -> S3DIS:
    return S3DIS(
        root=DATASETS_DIR,
        areas=("Area_5",),
        aligned=True,
        download=False,
        show_progress=False,
        transform=transform,
    )


def s3dis_hdf5_dataset(transform: Optional[Callable] = None) -> S3DISHdf5:
    return S3DISHdf5(
        root=DATASETS_DIR,
        areas=("Area_5",),
        show_progress=False,
        transform=transform,
    )


def scannet20_dataset(transform: Optional[Callable] = None) -> ScanNet20:
    return ScanNet20(
        root=DATASETS_DIR,
        split="val",
        show_progress=False,
        transform=transform,
    )


def scannet20_blocks_dataset(transform: Optional[Callable] = None) -> ScanNet20:
    # tile_scannet_scene samples points per block with torch RNG; seed so the snapshot stays reproducible.
    torch.manual_seed(0)
    return ScanNet20(
        root=DATASETS_DIR,
        split="val",
        use_axis_alignment=False,
        block_size=1.5,
        block_stride=0.75,
        num_nodes=8192,
        show_progress=False,
        transform=transform,
    )


def semantickitti_dataset(transform: Optional[Callable] = None) -> SemanticKITTI:
    return SemanticKITTI(
        root=DATASETS_DIR,
        sequences=("00",),
        transform=transform,
    )


@pytest.fixture
def dataset_factory() -> Callable[..., Dataset]:
    """Build one of the datasets shipped under `tests/data/datasets` by name, e.g. `dataset_factory("s3dis")`."""
    constructors: Dict[str, Callable[..., Dataset]] = {
        "modelnet": modelnet_dataset,
        "scanobjectnn": scanobjectnn_dataset,
        "shapenetpart": shapenetpart_dataset,
        "s3dis": s3dis_dataset,
        "s3dis_hdf5": s3dis_hdf5_dataset,
        "scannet20": scannet20_dataset,
        "scannet20_blocks": scannet20_blocks_dataset,
        "semantickitti": semantickitti_dataset,
    }

    def factory(name: str, transform: Optional[Callable] = None) -> Dataset:
        return constructors[name](transform=transform)

    return factory


def _create_dir_factory(source: Path, dest: Path, symlinks: bool = False) -> Callable[..., Path]:
    default_symlinks = symlinks

    def factory(pattern: str = "*", symlinks: bool = default_symlinks) -> Path:
        for file_path in source.rglob(pattern):
            if not file_path.is_file():
                continue
            out_path = dest / file_path.relative_to(source)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            if out_path.exists():
                out_path.unlink()
            if symlinks:
                out_path.symlink_to(file_path.absolute())
            else:
                shutil.copy(file_path, out_path)
        return dest

    return factory


@pytest.fixture
def data_dir_factory(tmp_path: Path) -> Callable[..., Path]:
    """Utility fixture to mock the test `data` directory.
    This fixture is used to add data files within the test temporary directory `tmp_path`.
    Files are real copies by default, so in-place writes cannot leak into the committed data directory;
    pass `symlinks=True` to symlink instead when a test only reads large files.

    The purpose of this function is to allow data modification in an isolated folder,
    without affecting the actual data directory.
    For example, if a test modifies the data files created by this fixture,
    the changes will not be reflected in the actual data directory.

    Args:
        pattern: The pattern to match the data files to be copied.
        symlinks: Whether to symlink the data files instead of copying them.

    Returns:
        The path to the test `data` directory.

    Example:
        ```python
        def test_data_dir(data_dir_factory: Callable[..., Path]) -> None:
            data_dir = data_dir_factory("ShapeNetPart/raw/**/*")
            # data_dir is the path to the tmp directory with the copied data,
            # e.g. /tmp/pytest-of-user/test_data_dir/test_data_dir0/data
            # containing the files matching the pattern "ShapeNetPart/raw/**/*"
            # from the actual data directory.

            # The data directory can be used in the test as if it were the actual data directory.
            # For example, the test can use the data directory to load data.

            dataset = ShapeNetPart(root=data_dir)

            # Finally, if the test corrupts the data, the changes will not be reflected in the actual data directory.
            # This is useful to avoid modifying the actual data directory.

            # The below will not remove the actual data directory
            shutil.rmtree(data_dir)
        ```
    """
    return _create_dir_factory(DATA_DIR, tmp_path / "data")


@pytest.fixture
def data_dir(data_dir_factory: Callable[..., Path]) -> Path:
    """Utility fixture to get a copy of the full data directory"""
    return data_dir_factory()


@pytest.fixture
def datasets_dir_factory(tmp_path: Path) -> Callable[..., Path]:
    """Utility fixture to create a directory factory for the datasets directory"""
    return _create_dir_factory(DATASETS_DIR, tmp_path / "datasets")


@pytest.fixture
def datasets_dir(datasets_dir_factory: Callable[..., Path]) -> Path:
    """Utility fixture to get a copy of the datasets directory"""
    return datasets_dir_factory()


@pytest.fixture
def models_dir_factory(tmp_path: Path) -> Callable[..., Path]:
    """Utility fixture to create a directory factory for the models directory.

    Defaults to symlinks: the model snapshots weigh hundreds of MB and every consumer only reads them
    (snapshot regeneration writes to the source directory directly, never through the returned paths).
    """
    return _create_dir_factory(MODELS_DIR, tmp_path / "models", symlinks=True)


@pytest.fixture
def models_dir(models_dir_factory: Callable[..., Path]) -> Path:
    """Utility fixture to get a copy of the models directory"""
    return models_dir_factory()


def pytest_addoption(parser: pytest.Parser) -> None:
    """Register a custom flag to force the regeneration of the expected files used for anti-regression tests.
    While one could use the [pytest-regressions](https://pytest-regressions.readthedocs.io) plugin to do this,
    it will not support SimpleITK images out of the box.

    This fixture follow the same pattern, but really simpler: just a boolean flag and you will have to
    manually update the expected files yourself.

    References:
        - https://pytest-regressions.readthedocs.io/en/latest/overview.html#using-data-regression
        - https://docs.pytest.org/en/stable/reference/reference.html#pytest.hookspec.pytest_addoption
    """
    parser.addoption(
        "--force-regen",
        action="store_true",
        default=False,
        help="Regenerate the expected files used for anti-regression tests",
    )
    parser.addoption(
        "--run-pretrained",
        action="store_true",
        default=False,
        help="Run pretrained model regression tests (require local weights, skipped by default in CI)",
    )
    parser.addoption(
        "--run-experiment",
        action="store_true",
        default=False,
        help="Run experiment-fit tests that train each experiment config on dummy data (slow, skipped by default)",
    )


def pytest_configure(config: pytest.Config) -> None:
    """Register custom markers."""
    config.addinivalue_line(
        "markers",
        "pretrained: tests that load pretrained weights and compare numerical outputs. "
        "Skipped unless --run-pretrained is passed.",
    )
    config.addinivalue_line(
        "markers",
        "experiment: tests that fit an experiment config end-to-end on dummy data. "
        "Slow; skipped unless --run-experiment is passed.",
    )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Skip tests marked `pretrained` or `experiment` unless their opt-in flag is passed.

    `pretrained` tests need local weights CI does not cache; `experiment` tests fit every experiment
    config end-to-end on dummy data (slow). Both stay opt-in via `--run-pretrained` / `--run-experiment`.
    """
    if not config.getoption("--run-pretrained"):
        skip_pretrained = pytest.mark.skip(reason="needs --run-pretrained to run")
        for item in items:
            if "pretrained" in item.keywords:
                item.add_marker(skip_pretrained)

    if not config.getoption("--run-experiment"):
        skip_experiment = pytest.mark.skip(reason="needs --run-experiment to run")
        for item in items:
            if "experiment" in item.keywords:
                item.add_marker(skip_experiment)


@pytest.fixture
def force_regen(request: pytest.FixtureRequest) -> bool:
    """Store the CLI flag value as a fixture to be injected into test functions."""
    return request.config.getoption("--force-regen")
