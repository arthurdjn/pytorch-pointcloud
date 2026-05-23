import importlib
from functools import lru_cache
from importlib import import_module
from importlib.util import find_spec
from typing import Any, Optional, Tuple

import torch
from packaging.requirements import Requirement
from packaging.version import Version


@lru_cache
def package_available(package_name: str) -> bool:
    """Check if a package is available in your environment.

    Args:
        package_name: Name of the package to check (e.g. `os`)

    Examples:
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

    Args:
        module_path: Path to the module to check (e.g. `os.bla`)

    Examples:
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


@lru_cache
def check_requirement(requirement: str) -> bool:
    """Check if a package with specified version is available.

    Args:
        requirement: Package with optional version requirement (e.g., "torch>=1.10.0")

    Returns:
        Boolean indicating if the requirement is satisfied

    Examples:
        >>> check_requirement("torch>=1.10.0")
        True
        >>> check_requirement("torch<1.10.0")
        False
    """
    req = Requirement(requirement)

    try:
        base_name, *_ = req.name.split(".")
        base_module = import_module(base_name)

        if req.specifier and hasattr(base_module, "__version__"):
            if not req.specifier.contains(Version(base_module.__version__)):
                return False
        return True
    except ImportError:
        return False


@lru_cache
def optional_import(
    module_path: str,
    name: str = "",
    requirement: str = "",
    url: Optional[str] = None,
) -> Tuple[Any, bool]:
    """Import a module with a version check and return a boolean indicating availability.
    Args:
        module_path: Path to the module to import (e.g. `torch`)
        name: Name of the class, function or attribute to import from the module
            (e.g. `from torch import nn`, here `module="torch", name="nn"`)
        requirement: Requirement for the module (e.g., ">=1.10.0")
        url: URL of the module documentation, if any. Will be used in case of an import error.

    Returns:
        Imported module (or proxy if not available) and boolean indicating availability

    Examples:
        >>> torch, IS_TORCH_AVAILABLE = optional_import("torch>=2.5.0")
        >>> IS_TORCH_AVAILABLE
        True
        >>> pkg, IS_PKG_AVAILABLE = optional_import("missing_package")
        >>> IS_PKG_AVAILABLE
        False
        >>> pkg.some_function()
        OptionalImportError: ...
    """
    package_name = module_path.split(".")[0]
    # In case the requirement is in the format "package>=1.0.0"
    requirement = requirement.replace(package_name, "")

    # Extra message in case the module is not installed
    msg = ""

    if not module_available(module_path):
        msg = f"Optional module '{module_path}' is not installed, but expected {package_name}{requirement}."

    if check_requirement(f"{package_name}{requirement}"):
        module = import_module(module_path)
        if not name:
            return module, True

        try:
            return getattr(module, name), True
        except AttributeError:
            msg = f"Optional module '{module_path}' is available but could not import '{name}' from '{module_path}'."

    msg = msg or f"Optional module '{module_path}' does not meet the requirement {package_name}{requirement}."
    if url:
        msg += f" Check official documentation to install it: {url}."

    # Create a proxy that will raise an import error when used
    class ModuleNotFoundProxy:
        def __getattr__(self, name: str) -> Any:
            raise ImportError(msg)

        def __call__(self, *args: Any, **kwargs: Any) -> Any:
            raise ImportError(msg)

        def __repr__(self) -> str:
            return msg

    return ModuleNotFoundProxy(), False


_FLASH_ATTN_GITHUB_URL = "https://github.com/Dao-AILab/flash-attention"
_OCNN_GITHUB_URL = "https://github.com/octree-nn/ocnn-pytorch"
_SPCONV_GITHUB_URL = "https://github.com/traveller59/spconv"
_TORCH_CLUSTER_GITHUB_URL = "https://github.com/rusty1s/pytorch_cluster"
_TORCH_SCATTER_GITHUB_URL = "https://github.com/rusty1s/pytorch_scatter"
_TORCHSPARSE_GITHUB_URL = "https://github.com/mit-han-lab/torchsparse"

_CUDA_AVAILABLE = torch.cuda.is_available()
_FLASH_ATTN_AVAILABLE = module_available("flash_attn")
_OCNN_AVAILABLE = module_available("ocnn")
_SPCONV_AVAILABLE = module_available("spconv.pytorch")
_TORCH_CLUSTER_AVAILABLE = module_available("torch_cluster")
_TORCH_SCATTER_AVAILABLE = module_available("torch_scatter")
_TORCHSPARSE_AVAILABLE = module_available("torchsparse")
_MAMBA_SSM_AVAILABLE = module_available("mamba_ssm")
_DWCONV_AVAILABLE = module_available("dwconv")
_LIGHTNING_AVAILABLE = module_available("lightning.pytorch")
_HYDRA_AVAILABLE = module_available("hydra")
