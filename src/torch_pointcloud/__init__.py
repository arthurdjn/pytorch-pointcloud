import importlib.metadata

from .models import create_model, list_models, register_model

__version__ = importlib.metadata.version("torch_pointcloud")
