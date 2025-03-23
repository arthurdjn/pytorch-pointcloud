import importlib
import warnings
from functools import lru_cache
from importlib import import_module
from importlib.util import find_spec
from typing import Any, Optional, Tuple

from packaging.requirements import Requirement
from packaging.version import Version


class OptionalImportError(ImportError):
    """Error raised when an optional import fails."""


@lru_cache
def package_available(package_name: str) -> bool:
    """Check if a package is available in your environment.

    >>> package_available('os')
    True
    >>> package_available('bla')
    False

    """
    try:
        return find_spec(package_name) is not None
    except ModuleNotFoundError:
        return False


@lru_cache
def module_available(module_path: str) -> bool:
    """Check if a module path is available in your environment.

    >>> module_available('os')
    True
    >>> module_available('os.bla')
    False
    >>> module_available('bla.bla')
    False

    """
    module_names = module_path.split(".")
    if not package_available(module_names[0]):
        return False
    try:
        importlib.import_module(module_path)
    except ImportError:
        return False
    return True


def check_requirement(requirement: str) -> bool:
    """Check if a package with specified version is available.

    Args:
        requirement: Package with optional version requirement (e.g., "torch>=1.10.0")

    Returns:
        Boolean indicating if the requirement is satisfied
    """
    req = Requirement(requirement)

    try:
        base_name, *_ = req.name.split(".")
        base_module = import_module(base_name)

        if req.specifier and hasattr(base_module, "__version__"):
            if not req.specifier.contains(Version(base_module.__version__)):
                warnings.warn(
                    f"Module '{req.name}' found but version requirement not met: "
                    f"installed={base_module.__version__}, required={req.specifier}"
                )
                return False
        return True
    except ImportError:
        return False


def optional_import(requirement: str, url: Optional[str] = None) -> Tuple[Any, bool]:
    """Import a module with a version check and return a boolean indicating availability.

    Args:
        requirement: Module to import with optional version requirement (e.g., "torch>=1.10.0")
        url: Documentation URL for the module. Will be used in case of an import error.

    Returns:
        Imported module (or proxy if not available) and boolean indicating availability

    Examples:
        >>> torch, _IS_TORCH_AVAILABLE = optional_import("torch>=2.5.0")
        >>> _IS_TORCH_AVAILABLE
        True
        >>> pkg, _IS_PKG_AVAILABLE = optional_import("missing_package")
        >>> _IS_PKG_AVAILABLE
        False
        >>> pkg.some_function()
        OptionalImportError: ...

    """
    req = Requirement(requirement)
    is_available = check_requirement(requirement)

    if is_available:
        return import_module(req.name), True

    # Create a proxy that will raise import error when used
    base_name, *_ = req.name.split(".")
    base_requirement = requirement.replace(req.name, base_name)
    msg = f"Optional module '{req.name}' is not available, but expected {base_requirement}."
    if url:
        msg += f" Check official documentation to install it: {url}."

    class ModuleNotFoundProxy:
        def __getattr__(self, name: str) -> Any:
            raise OptionalImportError(msg)

        def __call__(self, *args: Any, **kwargs: Any) -> Any:
            raise OptionalImportError(msg)

        def __repr__(self) -> str:
            return msg

    return ModuleNotFoundProxy(), False
