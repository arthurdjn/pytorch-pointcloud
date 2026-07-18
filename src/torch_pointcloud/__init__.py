import importlib.metadata

from . import datasets, inferers, losses, models, transforms, utils
from .models import create_model, list_models, register_model

__version__ = importlib.metadata.version("torch_pointcloud")

__all__ = [
    "__version__",
    "create_model",
    "datasets",
    "inferers",
    "list_models",
    "losses",
    "models",
    "register_model",
    "transforms",
    "utils",
]
