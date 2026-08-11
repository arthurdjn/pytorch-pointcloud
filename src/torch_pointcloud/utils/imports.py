import importlib
from functools import lru_cache, partial
from importlib import import_module
from importlib.util import find_spec
from typing import Any, Callable, Dict, Optional, Tuple

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
    except (ModuleNotFoundError, ValueError):
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
    except Exception:
        # An installed module can still fail to import for reasons other than ImportError (e.g. a CUDA /
        # triton driver probe on a GPU-less machine); treat any import failure as "not available".
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
    # A broken optional dependency (e.g. a CUDA / ABI mismatch in a source-built wheel) can raise more than
    # ImportError at import time; treat any failure to load as "requirement not met" so callers get a proxy.
    except Exception:
        return False


_UNRESOLVED = object()


class _LazyImportProxy:
    """Stand-in for an installed optional dependency that defers the import to first use.

    Resolution happens on the first attribute access, call, `isinstance` / `issubclass` check, or
    subclassing (PEP 560 `__mro_entries__`), so module-scope `optional_import` calls do not load heavy
    dependencies at import time. Dunder lookups on an unresolved proxy raise `AttributeError`: they come
    from introspection machinery (e.g. doctest probing `__wrapped__` through `hasattr`) and must not
    force the import.

    The defer-to-first-use pattern follows TensorFlow's `LazyLoader`
    (https://github.com/tensorflow/tensorflow/blob/v2.17.0/tensorflow/python/util/lazy_loader.py#L28), written
    as a plain wrapper instead of a `types.ModuleType` subclass. Unlike `importlib.util.LazyLoader`, no import
    machinery is hooked: the first use runs an ordinary `import_module`, so a failed import cannot leave a
    half-initialized module in `sys.modules` and there is no loader-compatibility constraint.
    """

    def __init__(self, module_path: str, name: str, url: Optional[str]) -> None:
        self._module_path = module_path
        self._name = name
        self._url = url
        self._target: Any = _UNRESOLVED

    def _resolve(self) -> Any:
        if self._target is _UNRESOLVED:
            try:
                target: Any = import_module(self._module_path)
                if self._name:
                    target = getattr(target, self._name)
            except Exception as error:
                detail = f"could not provide '{self._name}'" if self._name else "failed to load"
                msg = f"Optional module '{self._module_path}' is installed but {detail}."
                if self._url:
                    msg += f" Check official documentation to install it: {self._url}."
                raise ImportError(msg) from error
            self._target = target
        return self._target

    def __getattr__(self, name: str) -> Any:
        if self._target is _UNRESOLVED and name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        return getattr(self._resolve(), name)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        target = self._resolve()
        return target(*args, **kwargs)

    def __mro_entries__(self, bases: Tuple[Any, ...]) -> Tuple[type, ...]:
        target = self._resolve()
        assert isinstance(target, type), f"'{self._module_path}.{self._name}' is not a class"
        return (target,)

    def __instancecheck__(self, instance: Any) -> bool:
        target = self._resolve()
        assert isinstance(target, type), f"'{self._module_path}.{self._name}' is not a class"
        return isinstance(instance, target)

    def __subclasscheck__(self, subclass: type) -> bool:
        target = self._resolve()
        assert isinstance(target, type), f"'{self._module_path}.{self._name}' is not a class"
        return issubclass(subclass, target)

    def __repr__(self) -> str:
        qualifier = f"{self._module_path}.{self._name}" if self._name else self._module_path
        return f"<lazy optional import '{qualifier}'>"


@lru_cache
def optional_import(
    module_path: str,
    name: str = "",
    requirement: str = "",
    url: Optional[str] = None,
) -> Tuple[Any, bool]:
    """Import a module lazily and return a boolean indicating availability.

    When the top-level package is installed and no `requirement` is given, the returned object is a
    lightweight proxy that defers the actual import to first use (attribute access, call, `isinstance`
    check, or subclassing), so module-scope calls do not load heavy dependencies at import time. With a
    `requirement`, the module is imported eagerly to check its version. When the dependency is missing,
    the returned proxy raises an informative `ImportError` on any use.

    The `(module, available)` return contract and the raising placeholder for missing dependencies are
    modeled on MONAI's `optional_import`
    (https://github.com/Project-MONAI/MONAI/blob/1.4.0/monai/utils/module.py#L283).

    Args:
        module_path: Path to the module to import (e.g. `torch`)
        name: Name of the class, function or attribute to import from the module
            (e.g. `from torch import nn`, here `module="torch", name="nn"`)
        requirement: Requirement for the module (e.g., ">=1.10.0")
        url: URL of the module documentation, if any. Will be used in case of an import error.

    Returns:
        Imported module (or proxy) and boolean indicating availability

    Examples:
        >>> torch, IS_TORCH_AVAILABLE = optional_import("torch", requirement=">=2.5.0")
        >>> IS_TORCH_AVAILABLE
        True
        >>> pkg, IS_PKG_AVAILABLE = optional_import("missing_package")
        >>> IS_PKG_AVAILABLE
        False
        >>> pkg.some_function()  # doctest: +SKIP
        ImportError: Optional module 'missing_package' does not meet the requirement missing_package.
    """
    package_name = module_path.split(".")[0]
    # In case the requirement is in the format "package>=1.0.0"
    requirement = requirement.replace(package_name, "")

    # Extra message in case the module is not installed
    msg = ""

    if requirement:
        if check_requirement(f"{package_name}{requirement}"):
            try:
                module = import_module(module_path)
                return (getattr(module, name) if name else module), True
            except ImportError:
                msg = f"Optional module '{module_path}' is not installed, but expected {package_name}{requirement}."
            except AttributeError:
                msg = (
                    f"Optional module '{module_path}' is available but could not import '{name}' from '{module_path}'."
                )
            except Exception:
                msg = f"Optional module '{module_path}' is installed but failed to load."
    elif package_available(package_name):
        return _LazyImportProxy(module_path, name, url), True

    msg = msg or f"Optional module '{module_path}' does not meet the requirement {package_name}{requirement}."
    if url:
        msg += f" Check official documentation to install it: {url}."

    # Create a proxy that raises an informative ImportError whenever the missing dependency is used. Dunder lookups
    # are answered with AttributeError instead: they come from introspection machinery (e.g. doctest / inspect probing
    # `__wrapped__` through `hasattr`, which only swallows AttributeError), so raising ImportError there would crash
    # module collection on Python < 3.12 even though no real use of the dependency occurred.
    def _getattr(name: str) -> Any:
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        raise ImportError(msg)

    class _Meta(type):
        def __getattr__(cls, name: str) -> Any:
            return _getattr(name)

        def __call__(cls, *args: Any, **kwargs: Any) -> Any:
            raise ImportError(msg)

        def __repr__(cls) -> str:
            return msg

    class ModuleNotFoundProxy(metaclass=_Meta):
        def __getattr__(self, name: str) -> Any:
            return _getattr(name)

        def __call__(self, *args: Any, **kwargs: Any) -> Any:
            raise ImportError(msg)

        def __repr__(self) -> str:
            return msg

    return ModuleNotFoundProxy, False


_DWCONV_GITHUB_URL = "https://github.com/octree-nn/dwconv"
_FLASH_ATTN_GITHUB_URL = "https://github.com/Dao-AILab/flash-attention"
_FVDB_GITHUB_URL = "https://github.com/voxel-foundation/fvdb"
_LIGHTNING_GITHUB_URL = "https://github.com/Lightning-AI/pytorch-lightning"
_MAMBA_SSM_GITHUB_URL = "https://github.com/state-spaces/mamba"
_OCNN_GITHUB_URL = "https://github.com/octree-nn/ocnn-pytorch"
_SPCONV_GITHUB_URL = "https://github.com/traveller59/spconv"
_SPTR_GITHUB_URL = "https://github.com/JIA-Lab-research/SparseTransformer"
_TORCH_CLUSTER_GITHUB_URL = "https://github.com/rusty1s/pytorch_cluster"
_TORCH_SCATTER_GITHUB_URL = "https://github.com/rusty1s/pytorch_scatter"
_TORCH_SPARSE_GITHUB_URL = "https://github.com/rusty1s/pytorch_sparse"
_TORCHMETRICS_GITHUB_URL = "https://github.com/Lightning-AI/torchmetrics"
_TORCHSPARSE_GITHUB_URL = "https://github.com/mit-han-lab/torchsparse"

# Availability probes fully import their dependency (spconv alone costs seconds), so the flags are
# resolved lazily on first attribute access (PEP 562) and cached in the module globals.
_AVAILABILITY_FLAGS: Dict[str, Callable[[], bool]] = {
    "_CUDA_AVAILABLE": torch.cuda.is_available,
    "_DWCONV_AVAILABLE": partial(module_available, "dwconv"),
    "_FLASH_ATTN_AVAILABLE": partial(module_available, "flash_attn"),
    "_FVDB_AVAILABLE": partial(module_available, "fvdb"),
    "_HYDRA_AVAILABLE": partial(module_available, "hydra"),
    "_LIGHTNING_AVAILABLE": partial(module_available, "lightning.pytorch"),
    "_MAMBA_SSM_AVAILABLE": partial(module_available, "mamba_ssm"),
    "_OCNN_AVAILABLE": partial(module_available, "ocnn"),
    "_SPCONV_AVAILABLE": partial(module_available, "spconv.pytorch"),
    "_SPTR_AVAILABLE": partial(module_available, "sptr"),
    "_TORCH_CLUSTER_AVAILABLE": partial(module_available, "torch_cluster"),
    "_TORCH_SCATTER_AVAILABLE": partial(module_available, "torch_scatter"),
    "_TORCHSPARSE_AVAILABLE": partial(module_available, "torchsparse"),
}


def __getattr__(name: str) -> bool:
    flag = _AVAILABILITY_FLAGS.get(name)
    if flag is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = flag()
    globals()[name] = value
    return value
