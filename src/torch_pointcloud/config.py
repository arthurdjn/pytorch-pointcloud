"""Environment-variable configuration of cache, model, and data directories and randomness defaults.

This module contains the following global variables for configuration:

| Variable | Description | Default |
|----------|-------------|---------|
| `HOME_DIR` | The home directory of the user. | `Path.home().as_posix()` |
| `CACHE_DIR` | The cache directory for the package. | `Path(HOME_DIR, ".cache", "torch-pointcloud").as_posix()` |
| `MODELS_DIR` | The directory for the models. | `Path(CACHE_DIR, "models").as_posix()` |
| `DATA_DIR` | The directory for the data. | `"data"` |
| `FPS_RANDOM_START` | Overrides the `random_start` of every farthest point sampling call; `None` keeps each call's own. | `None` |
| `KNN_DENSE_BUDGET` | Largest $B N_x N_y$ (clouds times points squared) for which `knn` computes dense pairwise distances. | `16_000_000` |
"""

import os
from pathlib import Path
from typing import Optional


def asbool(value: Optional[str]) -> Optional[bool]:
    """Parse an environment variable as a boolean.

    Args:
        value: The raw variable value, or None when it is unset.

    Returns:
        True when the value reads as `true`, `1`, `yes` or `y` (case-insensitive), False for anything
        else, and None when the variable is unset.
    """
    if value is None:
        return None

    return value.strip().lower() in ["true", "1", "yes", "y"]


def asint(value: Optional[str]) -> Optional[int]:
    """Parse an environment variable as an integer.

    Args:
        value: The raw variable value, or None when it is unset.

    Returns:
        The parsed integer, or None when the variable is unset or does not parse.
    """
    if value is None:
        return None

    try:
        return int(value)
    except ValueError:
        return None


HOME_DIR = Path.home().as_posix()
CACHE_DIR = os.getenv("TORCH_POINTCLOUD_CACHE_DIR", Path(HOME_DIR, ".cache", "torch-pointcloud").as_posix())
MODELS_DIR = os.getenv("TORCH_POINTCLOUD_MODELS_DIR", Path(CACHE_DIR, "models").as_posix())
DATA_DIR = os.getenv("TORCH_POINTCLOUD_DATA_DIR", "data")

FPS_RANDOM_START = asbool(os.getenv("TORCH_POINTCLOUD_FPS_RANDOM_START", None))
KNN_DENSE_BUDGET = asint(os.getenv("TORCH_POINTCLOUD_KNN_DENSE_BUDGET", None)) or 16_000_000
