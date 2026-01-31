import shutil
from pathlib import Path
from typing import Callable

import pytest

DATA_DIR = Path("tests/data")


@pytest.fixture
def data_dir_factory(tmp_path: Path) -> Callable[..., Path]:
    """Utility fixture to mock the test `data` directory.
    This fixture is used to add data files within the test temporary directory `tmp_path`.
    For convenience, the copied data is symlinked to the `data` directory within the test temporary directory.

    The purpose of this function is to allow data modification in an isolated folder,
    without affecting the actual data directory.
    For example, if a test modifies the data files created by this fixture,
    the changes will not be reflected in the actual data directory.

    Args:
        pattern: The pattern to match the data files to be copied.
        symlinks: Whether to create symlinks to the data files.

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

    def data_dir(pattern: str = "*", symlinks: bool = True) -> Path:
        out_dir = tmp_path / "data"
        for file_path in DATA_DIR.rglob(pattern):
            if not file_path.is_file():
                continue

            out_path = out_dir / file_path.relative_to(DATA_DIR)
            out_path.parent.mkdir(parents=True, exist_ok=True)

            if out_path.exists():
                out_path.unlink()

            if symlinks:
                out_path.symlink_to(file_path.absolute())
            else:
                shutil.copy(file_path, out_path)

        return out_dir

    return data_dir


@pytest.fixture
def data_dir(data_dir_factory: Callable[..., Path]) -> Path:
    """Utility fixture to get a copy of the full data directory"""
    return data_dir_factory()
