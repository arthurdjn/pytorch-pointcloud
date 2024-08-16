import logging
import os
from pathlib import Path

CACHE_DIR = os.getenv("TORCH_POINTCLOUD_CACHE_DIR", Path.home() / ".cache" / "torch_pointcloud")
LOG_LEVEL = os.getenv("TORCH_POINTCLOUD_LOG_LEVEL", "INFO")


def get_logger() -> logging.Logger:
    logger = logging.getLogger("torch_pointcloud")
    logger.setLevel(LOG_LEVEL)
    return logger
