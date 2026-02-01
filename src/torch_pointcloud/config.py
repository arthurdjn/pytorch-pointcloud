import os
from pathlib import Path
from typing import Optional


def asbool(value: Optional[str]) -> Optional[bool]:
    if value is None:
        return None

    return value.strip().lower() in ["true", "1", "yes", "y"]


def asint(value: Optional[str]) -> Optional[int]:
    if value is None:
        return None

    try:
        return int(value)
    except ValueError:
        return None


# NOTE: We could use pydantic to define this config. Maybe later.
# For simplicity, we use os.getenv for now (as we have few settings).
HOME_DIR = Path.home().as_posix()
CWD_DIR = Path.cwd().as_posix()
CACHE_DIR = os.getenv("TORCH_POINTCLOUD_CACHE_DIR", Path(HOME_DIR, ".cache", "torch-pointcloud").as_posix())
MODELS_DIR = os.getenv("TORCH_POINTCLOUD_MODELS_DIR", Path(CACHE_DIR, "models").as_posix())
DATA_DIR = os.getenv("TORCH_POINTCLOUD_DATA_DIR", Path(CWD_DIR, "data").as_posix())

# Some variables to affect how random operations are performed.
RANDOM_SEED = asint(os.getenv("TORCH_POINTCLOUD_RANDOM_SEED", None))
FPS_RANDOM_START = asbool(os.getenv("TORCH_POINTCLOUD_FPS_RANDOM_START", None))
