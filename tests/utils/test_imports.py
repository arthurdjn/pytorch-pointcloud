import sys
from typing import Generator
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


def test_optional_import_version_requirement_invalid(mock_package: str) -> None:
    module, is_available = optional_import(mock_package, requirement=">=2.0.0")
    assert is_available is False

    with pytest.raises(ImportError):
        module()

    with pytest.raises(ImportError):
        module.some_function()
