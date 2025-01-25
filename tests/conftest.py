from pathlib import Path
from typing import Callable

import pytest

DATA_DIR = Path("tests/data")


@pytest.fixture
def data_dir_factory(tmp_path: Path) -> Callable[..., Path]:
    def data_dir(pattern: str = "*") -> Path:
        out_dir = tmp_path / "data"
        for file_path in DATA_DIR.rglob(pattern):
            if not file_path.is_file():
                continue

            out_path = out_dir / file_path.relative_to(DATA_DIR)
            out_path.parent.mkdir(parents=True, exist_ok=True)

            if out_path.exists():
                out_path.unlink()

            out_path.symlink_to(file_path.absolute())

        return out_dir

    return data_dir


@pytest.fixture
def data_dir(data_dir_factory: Callable[..., Path]) -> Path:
    return data_dir_factory()
