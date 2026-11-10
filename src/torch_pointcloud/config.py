"""Cache and data directories of the package.

| Variable | Description | Default | Environment variable |
|----------|-------------|---------|----------------------|
| `HOME_DIR` | The home directory of the user. | `Path.home().as_posix()` | |
| `CACHE_DIR` | The cache directory of the package. | `Path(HOME_DIR, ".cache", "torch-pointcloud").as_posix()` | `TORCH_POINTCLOUD_CACHE_DIR` |
| `MODELS_DIR` | The pretrained weights, inside the cache. | `Path(CACHE_DIR, "models").as_posix()` | |
| `DATA_DIR` | The datasets. | `"data"` | `TORCH_POINTCLOUD_DATA_DIR` |
"""

import os
from pathlib import Path

HOME_DIR = Path.home().as_posix()
CACHE_DIR = os.getenv("TORCH_POINTCLOUD_CACHE_DIR", Path(HOME_DIR, ".cache", "torch-pointcloud").as_posix())
MODELS_DIR = Path(CACHE_DIR, "models").as_posix()
DATA_DIR = os.getenv("TORCH_POINTCLOUD_DATA_DIR", "data")
