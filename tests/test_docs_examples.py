"""Run the ```python fences of the `torch_pointcloud` docstrings and the docs pages as pasted.

A plain `python` fence runs in the default suite. A `{.python data}` fence reads the fixture datasets under `data/`
or downloads pretrained weights, so it runs only with `--run-docs`, from a copy of `tests/data/datasets`. A
`{.python notest}` fence never runs.
"""

import re
import shutil
import textwrap
from pathlib import Path
from typing import Iterator, Tuple

import pytest

from torch_pointcloud.utils.imports import _PYG_LIB_AVAILABLE

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "torch_pointcloud"
DOCS = ROOT / "docs"
DATASETS_DIR = ROOT / "tests" / "data" / "datasets"
FENCE = re.compile(
    r"^(?P<indent>[ \t]*)```(?P<info>python|\{\.python data\})\n(?P<code>.*?)^(?P=indent)```", re.MULTILINE | re.DOTALL
)


def docs_pages() -> Iterator[Path]:
    """Yield the hand-written docs pages, leaving out the generated API stubs, dataset and model pages and notebooks."""
    for path in sorted(DOCS.rglob("*.md")):
        parts = path.relative_to(DOCS).parts
        if parts[0] in ("api", "examples", "scripts"):
            continue
        if parts[0] in ("datasets", "models") and path.name != "overview.md":
            continue
        yield path


def python_fences(path: Path) -> Iterator[Tuple[int, str, str]]:
    """Yield `(line, info, code)` for every `python` or `{.python data}` fence of `path`, dedented."""
    text = path.read_text()
    for match in FENCE.finditer(text):
        yield text.count("\n", 0, match.start()) + 1, match.group("info"), textwrap.dedent(match.group("code"))


FENCES = [
    (info, pytest.param(code, id=f"{path.relative_to(ROOT)}:{line}"))
    for path in [*sorted(SRC.rglob("*.py")), *docs_pages()]
    for line, info, code in python_fences(path)
]


@pytest.fixture(scope="session")
def docs_data_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Directory holding a copy of the fixture datasets as `data/`, the root the docs examples read."""
    root = tmp_path_factory.mktemp("docs")
    shutil.copytree(DATASETS_DIR, root / "data")
    return root


@pytest.mark.skipif(not _PYG_LIB_AVAILABLE, reason="pyg-lib is not installed")
@pytest.mark.parametrize("code", [param for info, param in FENCES if info == "python"])
def test_docs_example_runs(code: str) -> None:
    exec(compile(code, "<docs>", "exec"), {"__name__": "__docs__"})


@pytest.mark.docs
@pytest.mark.skipif(not _PYG_LIB_AVAILABLE, reason="pyg-lib is not installed")
@pytest.mark.parametrize("code", [param for info, param in FENCES if info != "python"])
def test_docs_data_example_runs(code: str, docs_data_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(docs_data_dir)
    exec(compile(code, "<docs>", "exec"), {"__name__": "__docs__"})
