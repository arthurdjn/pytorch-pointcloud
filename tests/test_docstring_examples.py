"""Run every ```python fence of the `torch_pointcloud` docstrings as pasted (`pycon` and `{.python notest}` are skipped)."""

import re
import textwrap
from pathlib import Path
from typing import Iterator, Tuple

import pytest

from torch_pointcloud.utils.imports import _PYG_LIB_AVAILABLE

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "torch_pointcloud"
FENCE = re.compile(r"^(?P<indent>[ \t]*)```python\n(?P<code>.*?)^(?P=indent)```", re.MULTILINE | re.DOTALL)


def python_fences(path: Path) -> Iterator[Tuple[int, str]]:
    """Yield `(line, code)` for every fence of `path` whose info string is exactly `python`, dedented."""
    text = path.read_text()
    for match in FENCE.finditer(text):
        yield text.count("\n", 0, match.start()) + 1, textwrap.dedent(match.group("code"))


FENCES = [
    pytest.param(code, id=f"{path.relative_to(ROOT)}:{line}")
    for path in sorted(SRC.rglob("*.py"))
    for line, code in python_fences(path)
]


@pytest.mark.skipif(not _PYG_LIB_AVAILABLE, reason="pyg-lib is not installed")
@pytest.mark.parametrize("code", FENCES)
def test_docstring_example_runs(code: str) -> None:
    exec(compile(code, "<docstring>", "exec"), {"__name__": "__docstring__"})
