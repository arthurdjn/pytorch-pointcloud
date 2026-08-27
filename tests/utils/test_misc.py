from types import SimpleNamespace
from typing import Any

import pytest

from torch_pointcloud.utils.misc import deep_getattr, parallel_map


@pytest.mark.parametrize(
    ("obj", "path", "expected"),
    [
        pytest.param(SimpleNamespace(a=SimpleNamespace(b=3)), "a.b", 3, id="attr-then-attr"),
        pytest.param({"a": {"b": 3}}, "a.b", 3, id="dict-then-dict"),
        pytest.param({"octree": SimpleNamespace(depth=5)}, "octree.depth", 5, id="dict-then-attr"),
        pytest.param(SimpleNamespace(cfg={"lr": 0.1}), "cfg.lr", 0.1, id="attr-then-dict"),
        pytest.param(
            SimpleNamespace(criterion=SimpleNamespace(ignore_index=-1)),
            "criterion.ignore_index",
            -1,
            id="criterion-ignore-index",
        ),
        pytest.param({"x": 7}, "x", 7, id="single-segment"),
    ],
)
def test_deep_getattr_resolves(obj: Any, path: str, expected: Any) -> None:
    assert deep_getattr(obj, path) == expected


@pytest.mark.parametrize(
    ("obj", "path"),
    [
        pytest.param({"a": {"b": 1}}, "a.c", id="missing-dict-key"),
        pytest.param(SimpleNamespace(), "missing", id="missing-attr"),
        pytest.param(SimpleNamespace(criterion=SimpleNamespace()), "criterion.ignore_index", id="missing-nested-attr"),
        pytest.param(SimpleNamespace(criterion=None), "criterion.ignore_index", id="none-mid-path"),
        pytest.param({}, "x", id="empty-mapping"),
    ],
)
def test_deep_getattr_missing_returns_default(obj: Any, path: str) -> None:
    sentinel = object()
    assert deep_getattr(obj, path, sentinel) is sentinel
    assert deep_getattr(obj, path) is None


@pytest.mark.parametrize("num_workers", [None, 0, 2])
def test_parallel_map_zero_workers_runs_sequentially(num_workers: Any) -> None:
    assert parallel_map(lambda x: x * 2, [1, 2, 3], num_workers=num_workers) == [2, 4, 6]
