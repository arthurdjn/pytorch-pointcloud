import os
from pathlib import Path

HOME_DIR = Path.home().as_posix()
CACHE_DIR = os.getenv("TORCH_POINTCLOUD_CACHE_DIR", Path(HOME_DIR, ".cache", "torch_pointcloud").as_posix())
