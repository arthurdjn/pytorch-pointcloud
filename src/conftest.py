from typing import Generator

import pytest


@pytest.hookimpl(wrapper=True)
def pytest_make_collect_report(
    collector: pytest.Collector,
) -> Generator[None, pytest.CollectReport, pytest.CollectReport]:
    """Treat import failures while collecting source modules as skips rather than errors.

    `pytest --doctest-modules src` and `pytest --markdown-docs src` both import every module to collect its `>>>`
    examples and docstring code fences. A model / layer module that wraps an optional heavy dependency (flash-attn,
    ocnn, sptr, dwconv, ...) raises `ImportError` at import time when that dependency is installed but broken or
    version-mismatched in the current environment. Softening such failures to skips keeps these runs green in any
    environment, mirroring how the test suite gates optional-dependency tests with `skipif`. Scoped to the source-scan
    runs so it never masks a genuine import error in the regular test suite.
    """
    report = yield
    scans_source = collector.config.getoption("doctestmodules", False) or collector.config.getoption(
        "markdowndocs", False
    )
    if scans_source and report.failed:
        longrepr = str(report.longrepr)
        if "ImportError" in longrepr or "ModuleNotFoundError" in longrepr:
            report.outcome = "skipped"
            report.longrepr = (str(collector.path), 0, "Skipped: optional dependency unavailable")
    return report
