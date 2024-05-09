import importlib
from importlib.util import find_spec


def _package_available(package_name: str) -> bool:
    """Check if a package is available in your environment.

    Args:
        package_name: Name of the package to check.

    Returns:
        True if the package exists.

    Examples:
        >>> _package_available('os')
        True
        >>> _package_available('bla')
        False
    """
    try:
        return find_spec(package_name) is not None
    except ModuleNotFoundError:
        return False


def _module_available(module_path: str) -> bool:
    """Check if a module path is available in your environment.

    Args:
        module_path: Path to the module to check.

    Returns:
        True if the module exists.

    Examples:
        >>> _module_available('os')
        True
        >>> _module_available('os.bla')
        False
        >>> _module_available('bla.bla')
        False
    """
    module_names = module_path.split(".")
    if not _package_available(module_names[0]):
        return False
    try:
        module = importlib.import_module(module_names[0])
    except ImportError:
        return False
    for name in module_names[1:]:
        if not hasattr(module, name):
            return False
        module = getattr(module, name)
    return True


_TORCHSPARSE_AVAILABLE = _module_available("torchsparse")
_TORCH_GEOMETRIC_AVAILABLE = _module_available("torch_geometric")
