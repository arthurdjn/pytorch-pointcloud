import shutil
from pathlib import Path
from typing import Any, Callable, Dict, Optional

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


def modelnet_resampled_dataset(transform: Optional[Callable] = None) -> ModelNetNormalResampled:
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
        "modelnet_resampled": modelnet_resampled_dataset,
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
        "--run-examples",
        action="store_true",
        default=False,
        help="Run every example script on the dummy datasets (slow, needs local weights, skipped by default)",
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
        "example: tests that run an example script end-to-end on the dummy datasets. "
        "Slow; skipped unless --run-examples is passed.",
    )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Skip tests marked `pretrained` or `example` unless their opt-in flag is passed.

    `pretrained` tests need local weights CI does not cache; `example` tests run every example script on
    the dummy datasets (slow). Both stay opt-in via `--run-pretrained` / `--run-examples`.
    """
    if not config.getoption("--run-pretrained"):
        skip_pretrained = pytest.mark.skip(reason="needs --run-pretrained to run")
        for item in items:
            if "pretrained" in item.keywords:
                item.add_marker(skip_pretrained)

    if not config.getoption("--run-examples"):
        skip_example = pytest.mark.skip(reason="needs --run-examples to run")
        for item in items:
            if "example" in item.keywords:
                item.add_marker(skip_example)


@pytest.fixture
def force_regen(request: pytest.FixtureRequest) -> bool:
    """Store the CLI flag value as a fixture to be injected into test functions."""
    return request.config.getoption("--force-regen")


@pytest.fixture
def sample_scene() -> Dict[str, Any]:
    """100-point single-scene dict with all standard Pointcept-style keys."""
    g = torch.Generator().manual_seed(0)
    return {
        "pos": torch.randn(100, 3, generator=g),
        "color": (torch.rand(100, 3, generator=g) * 255).to(torch.uint8),
        "normal": torch.nn.functional.normalize(torch.randn(100, 3, generator=g), dim=-1),
        "segment": torch.randint(0, 10, (100,), generator=g),
    }


@pytest.fixture
def empty_scene() -> Dict[str, Any]:
    """Empty single-scene dict (N=0) with all standard keys."""
    return {
        "pos": torch.empty(0, 3),
        "color": torch.empty(0, 3, dtype=torch.uint8),
        "normal": torch.empty(0, 3),
        "segment": torch.empty(0, dtype=torch.long),
    }


@pytest.fixture
def single_point_scene() -> Dict[str, Any]:
    """Single-point (N=1) dict with all standard keys."""
    return {
        "pos": torch.tensor([[1.0, 2.0, 3.0]]),
        "color": torch.tensor([[128, 64, 32]], dtype=torch.uint8),
        "normal": torch.tensor([[0.0, 0.0, 1.0]]),
        "segment": torch.tensor([5], dtype=torch.long),
    }
