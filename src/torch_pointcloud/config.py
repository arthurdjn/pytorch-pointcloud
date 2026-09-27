"""Environment-variable configuration of cache, model, and data directories and randomness defaults.

This module contains the following global variables for configuration:

| Variable | Description | Default |
|----------|-------------|---------|
| `HOME_DIR` | The home directory of the user. | `Path.home().as_posix()` |
| `CACHE_DIR` | The cache directory for the package. | `Path(HOME_DIR, ".cache", "torch-pointcloud").as_posix()` |
| `MODELS_DIR` | The directory for the models. | `Path(CACHE_DIR, "models").as_posix()` |
| `DATA_DIR` | The directory for the data. | `"data"` |
"""

import os
from pathlib import Path

HOME_DIR = Path.home().as_posix()
CACHE_DIR = os.getenv("TORCH_POINTCLOUD_CACHE_DIR", Path(HOME_DIR, ".cache", "torch-pointcloud").as_posix())
MODELS_DIR = os.getenv("TORCH_POINTCLOUD_MODELS_DIR", Path(CACHE_DIR, "models").as_posix())
DATA_DIR = os.getenv("TORCH_POINTCLOUD_DATA_DIR", "data")
