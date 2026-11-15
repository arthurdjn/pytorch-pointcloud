"""PyTorch library for 3D point cloud deep learning: models, datasets, transforms, and pretrained weights."""

from importlib import import_module
from importlib.metadata import version as _version
from typing import TYPE_CHECKING, Any, List

if TYPE_CHECKING:
    from . import config, datasets, inferers, layers, lightning, losses, metrics, models, ops, optim, transforms, utils
    from .models import create_model, deprecated_model, list_models, model_info, register_model

__version__ = _version("torch_pointcloud")

__all__ = [
    "__version__",
    "config",
    "create_model",
    "datasets",
    "deprecated_model",
    "inferers",
    "layers",
    "list_models",
    "losses",
    "metrics",
    "model_info",
    "models",
    "ops",
    "optim",
    "register_model",
    "transforms",
    "utils",
]

_SUBMODULES = {
    "config",
    "datasets",
    "inferers",
    "layers",
    "losses",
    "metrics",
    "models",
    "ops",
    "optim",
    "transforms",
    "utils",
}
# `lightning` needs the optional dependency, so it resolves on access without being part of `__all__`.
_OPTIONAL_SUBMODULES = {"lightning"}
_MODEL_FUNCTIONS = {"create_model", "deprecated_model", "list_models", "model_info", "register_model"}


# Subpackages load on first access, so `import torch_pointcloud.transforms` does not import the models and their
# sparse-convolution dependencies.
def __getattr__(name: str) -> Any:
    if name in _SUBMODULES or name in _OPTIONAL_SUBMODULES:
        return import_module(f".{name}", __name__)

    if name in _MODEL_FUNCTIONS:
        return getattr(import_module(".models", __name__), name)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> List[str]:
    return sorted(__all__)
