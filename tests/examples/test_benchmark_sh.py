"""Every command line of `examples/benchmark.sh` matches the argparse interface of the script it calls."""

import importlib.util
import shlex
import sys
from functools import cache
from pathlib import Path
from types import ModuleType
from typing import List

import pytest

REPO_DIR = Path(__file__).resolve().parents[2]
BENCHMARK_SH = REPO_DIR / "examples" / "benchmark.sh"

COMMANDS = [
    pytest.param(shlex.split(line)[1:], id=f"benchmark.sh:{lineno}")
    for lineno, line in enumerate(BENCHMARK_SH.read_text().splitlines(), 1)
    if line.startswith("$RUN ")
]


@cache
def _import(script: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(script, REPO_DIR / script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("command", COMMANDS)
def test_command_matches_script(command: List[str], monkeypatch: pytest.MonkeyPatch) -> None:
    script, *args = command
    try:
        module = _import(script)
    except ModuleNotFoundError as exc:
        pytest.skip(f"{script} imports {exc.name}")
    monkeypatch.setattr(sys, "argv", [script, *args])
    module.parse_args()
