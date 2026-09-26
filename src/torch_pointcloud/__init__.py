"""PyTorch library for 3D point cloud deep learning: models, datasets, transforms, and pretrained weights."""

from importlib import import_module
from importlib.metadata import version as _version
from typing import TYPE_CHECKING, Any, List

if TYPE_CHECKING:
    from . import config, datasets, inferers, layers, losses, metrics, models, ops, transforms, utils
    from .models import create_model, list_models, register_model

__version__ = _version("torch_pointcloud")

__all__ = [
    "__version__",
    "config",
    "create_model",
    "datasets",
    "inferers",
    "layers",
    "list_models",
    "losses",
    "metrics",
    "models",
    "ops",
    "register_model",
    "transforms",
    "utils",
]

_SUBMODULES = {"config", "datasets", "inferers", "layers", "losses", "metrics", "models", "ops", "transforms", "utils"}
_MODEL_FUNCTIONS = {"create_model", "list_models", "register_model"}


# Subpackages load on first access, so `import torch_pointcloud.transforms` does not import the models and their
# sparse-convolution dependencies.
def __getattr__(name: str) -> Any:
    if name in _SUBMODULES:
        return import_module(f".{name}", __name__)
    if name in _MODEL_FUNCTIONS:
        return getattr(import_module(".models", __name__), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> List[str]:
    return sorted(__all__)
