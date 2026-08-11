import importlib
import os
import subprocess
import sys
from pathlib import Path
from typing import Callable, Generator, List
from unittest.mock import MagicMock

import pytest

from torch_pointcloud.utils.imports import check_requirement, module_available, optional_import, package_available


@pytest.fixture
def mock_package() -> Generator[str, None, None]:
    package_name = "mock_package"

    class MockModule:
        __spec__ = MagicMock()
        __version__ = "1.0.0"

    sys.modules[package_name] = MockModule()  # type: ignore[assignment]
    yield package_name

    del sys.modules[package_name]


def test_package_available() -> None:
    assert package_available("os") is True


def test_package_not_available() -> None:
    assert package_available("fake_package") is False


def test_module_available() -> None:
    assert module_available("os.path") is True


def test_module_not_available() -> None:
    assert module_available("fake.module.path") is False


def test_check_requirement() -> None:
    assert check_requirement("os") is True


def test_check_missing_requirement() -> None:
    assert check_requirement("nonexistent_package") is False


def test_check_version_requirement(mock_package: str) -> None:
    assert check_requirement(f"{mock_package}>=0.9.0") is True
    assert check_requirement(f"{mock_package}<=1.1.0") is True
    assert check_requirement(f"{mock_package}>=2.0.0") is False


def test_optional_import() -> None:
    os_module, is_available = optional_import("os")
    assert is_available is True
    assert os_module.name == "posix" or os_module.name == "nt"  # Unix/Linux or Windows


def test_optional_import_fails() -> None:
    fake_module, is_available = optional_import("nonexistent_module")
    assert is_available is False

    with pytest.raises(ImportError):
        fake_module()

    with pytest.raises(ImportError):
        fake_module.some_function()


def test_optional_import_proxy_is_subclassable() -> None:
    """A missing optional dependency resolves to a real class, so `class X(Dep): ...` imports cleanly
    while the proxy and any subclass still raise `ImportError` when instantiated."""
    proxy, is_available = optional_import("nonexistent_module")
    assert is_available is False
    assert isinstance(proxy, type)

    class Subclass(proxy):  # type: ignore[valid-type, misc]
        pass

    with pytest.raises(ImportError):
        proxy()
    with pytest.raises(ImportError):
        Subclass()

    assert "nonexistent_module" in repr(proxy)


def test_optional_import_missing_submodule_raises_on_first_use() -> None:
    """A missing submodule of an installed package cannot be detected without importing the package, so
    the lazy proxy reports available and raises `ImportError` on first use instead of at import time."""
    proxy, is_available = optional_import("os.nonexistent_submodule")
    assert is_available is True
    with pytest.raises(ImportError):
        _ = proxy.anything


def test_optional_import_name() -> None:
    path_module, is_available = optional_import("os", name="path")
    assert is_available is True
    assert hasattr(path_module, "join")


def test_optional_import_requirement() -> None:
    fake_module, is_available = optional_import("fake_package", requirement=">=1.0.0", url="https://fake-package.org")
    assert is_available is False

    with pytest.raises(ImportError) as exc_info:
        fake_module()

    with pytest.raises(ImportError) as exc_info:
        fake_module.some_function()

    assert "https://fake-package.org" in str(exc_info.value)


def test_optional_import_version_requirement(mock_package: str) -> None:
    module, is_available = optional_import(mock_package, requirement=">=0.9.0")
    assert is_available is True
    assert module is not None


def test_availability_flag_resolves_lazily_and_caches() -> None:
    """The `_*_AVAILABLE` flags are computed on first attribute access (PEP 562) and cached in the
    module globals, so `import torch_pointcloud` does not probe every optional heavy dependency."""
    from torch_pointcloud.utils import imports

    value = imports._TORCH_SCATTER_AVAILABLE
    assert isinstance(value, bool)
    assert vars(imports)["_TORCH_SCATTER_AVAILABLE"] is value


def test_unknown_module_attribute_raises() -> None:
    from torch_pointcloud.utils import imports

    with pytest.raises(AttributeError, match="_NOT_A_FLAG"):
        _ = imports._NOT_A_FLAG


def test_optional_import_version_requirement_invalid(mock_package: str) -> None:
    module, is_available = optional_import(mock_package, requirement=">=2.0.0")
    assert is_available is False

    with pytest.raises(ImportError):
        module()

    with pytest.raises(ImportError):
        module.some_function()


@pytest.fixture
def lazy_module_factory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Generator[Callable[[str, str], str], None, None]:
    created: List[str] = []

    def factory(module_name: str, source: str) -> str:
        (tmp_path / f"{module_name}.py").write_text(source)
        monkeypatch.syspath_prepend(str(tmp_path))
        importlib.invalidate_caches()
        created.append(module_name)
        return module_name

    yield factory

    for module_name in created:
        sys.modules.pop(module_name, None)


def test_optional_import_module_resolves_on_first_attribute_access(
    lazy_module_factory: Callable[[str, str], str],
) -> None:
    module_name = lazy_module_factory("lazy_probe_module", "VALUE = 3\n")
    module, is_available = optional_import(module_name)
    assert is_available is True
    assert module_name not in sys.modules
    assert module.VALUE == 3
    assert module_name in sys.modules


def test_optional_import_attribute_resolves_on_call(lazy_module_factory: Callable[[str, str], str]) -> None:
    module_name = lazy_module_factory("lazy_probe_function", "def double(x):\n    return 2 * x\n")
    double, is_available = optional_import(module_name, "double")
    assert is_available is True
    assert module_name not in sys.modules
    assert double(3) == 6
    assert module_name in sys.modules


def test_optional_import_attribute_supports_subclassing_and_isinstance(
    lazy_module_factory: Callable[[str, str], str],
) -> None:
    """Subclassing a lazy proxy resolves the real class (PEP 560 `__mro_entries__`), and the proxy
    answers `isinstance` / `issubclass` checks by delegating to it."""
    module_name = lazy_module_factory("lazy_probe_base", "class Base:\n    pass\n")
    base_proxy, is_available = optional_import(module_name, "Base")
    assert is_available is True
    assert module_name not in sys.modules

    class Subclass(base_proxy):  # type: ignore[valid-type, misc]
        pass

    assert module_name in sys.modules
    assert Subclass.__mro__[1] is sys.modules[module_name].Base
    assert isinstance(Subclass(), base_proxy)
    assert issubclass(Subclass, base_proxy)


def test_optional_import_dunder_probe_does_not_import(lazy_module_factory: Callable[[str, str], str]) -> None:
    """Dunder lookups come from introspection machinery (e.g. doctest probing `__wrapped__` through
    `hasattr`) and must not force the import."""
    module_name = lazy_module_factory("lazy_probe_dunder", "VALUE = 1\n")
    module, _ = optional_import(module_name)
    assert not hasattr(module, "__wrapped__")
    assert module_name not in sys.modules


def test_import_does_not_load_lazy_optional_dependencies() -> None:
    """`import torch_pointcloud` must not import dependencies that are only reached through
    `optional_import` proxies (dependencies resolved at module scope, e.g. subclassed ones, are exempt)."""
    code = (
        "import sys\n"
        "import torch_pointcloud\n"
        "loaded = [m for m in ('dwconv', 'flash_attn', 'fvdb', 'mamba_ssm', 'sptr') if m in sys.modules]\n"
        "print(','.join(loaded))\n"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert result.stdout.strip() == ""


def test_config_import_has_no_dotenv_side_effect(tmp_path: Path) -> None:
    """Importing the library must not read `.env` from the working directory: `TORCH_POINTCLOUD_*`
    settings come from the process environment only, and `DATA_DIR` defaults to the relative `data`."""
    (tmp_path / ".env").write_text("TPC_DOTENV_PROBE=injected\nTORCH_POINTCLOUD_DATA_DIR=/from-dotenv\n")
    env = {key: value for key, value in os.environ.items() if key != "TORCH_POINTCLOUD_DATA_DIR"}
    code = (
        "import os\n"
        "import torch_pointcloud.config as config\n"
        "print(os.getenv('TPC_DOTENV_PROBE', '<unset>'))\n"
        "print(config.DATA_DIR)\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True, cwd=tmp_path, env=env
    )
    assert result.stdout.strip().splitlines()[-2:] == ["<unset>", "data"]


def test_package_available_ignores_namespace_shadow(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A same-named plain directory on `sys.path` (e.g. `tests/lightning` when pytest prepends `tests/`)
    resolves to a namespace-package spec and must not count as an installed dependency."""
    (tmp_path / "fake_namespace_shadow").mkdir()
    monkeypatch.syspath_prepend(str(tmp_path))
    package_available.cache_clear()
    try:
        assert package_available("fake_namespace_shadow") is False
    finally:
        package_available.cache_clear()


def test_lazy_proxy_subclassing_unresolvable_target_defers_error() -> None:
    """Subclassing a proxy whose target cannot resolve must not fail at class-definition (import) time;
    the informative `ImportError` surfaces when the subclass is instantiated."""
    proxy, available = optional_import("importlib", name="does_not_exist_attribute")
    assert available is True

    class Consumer(proxy):  # type: ignore[valid-type, misc]
        pass

    with pytest.raises(ImportError, match="could not provide 'does_not_exist_attribute'"):
        Consumer()
