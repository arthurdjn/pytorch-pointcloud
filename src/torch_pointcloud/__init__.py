import importlib.metadata

from .models import create_model, register_model

__version__ = importlib.metadata.version("torch_pointcloud")
