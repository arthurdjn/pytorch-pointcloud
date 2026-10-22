"""Base class for point cloud datasets cached on disk under a `raw/` + `processed/` layout."""

from pathlib import Path
from typing import Any, Dict

from torch.utils.data import Dataset

from torch_pointcloud.utils.types import PathLike


class PointCloudDataset(Dataset):
    """Base class for point-cloud datasets stored under a `raw/` + `processed/` disk layout.

    Data for a concrete dataset lives under `<root>/<name>/`, with the original download in
    `raw/` and the preprocessed cache in `processed/`. Subclasses implement `__getitem__` /
    `__len__` and return per-sample `Dict[str, Tensor]` objects keyed by `DataKeys` members;
    batching is left to the caller (see `torch_pointcloud.utils.data.collate`).

    Note:
        Datasets that keep their samples in an in-memory cache return a shallow `dict` copy from
        `__getitem__`, so adding or replacing keys on a returned sample never mutates the cache.
        The tensor values still share storage with the cache: treat them as read-only and let
        transforms return new tensors instead of writing in place.

    `process` stamps the cache it writes with a `meta.json` holding `cache_meta()`, and `processed_files_exist`
    raises when a cache carries a different stamp. Bump `format_version` whenever `process` changes the files it
    writes or their contents, so caches from older releases are regenerated instead of served.

    Args:
        root: The root directory under which the dataset directory is created.
    """

    _repr_indent = 4
    format_version = 1

    def __init__(self, root: PathLike) -> None:
        super().__init__()
        self.root = Path(root).as_posix()

    @property
    def name(self) -> str:
        """Name of the dataset directory."""
        return self.__class__.__name__

    @property
    def data_dir(self) -> str:
        """Path to the dataset directory `<root>/<name>`."""
        return Path(self.root, f"{self.name}").absolute().as_posix()

    @property
    def raw_dir(self) -> str:
        """Path to the raw download directory."""
        return Path(self.data_dir, "raw").absolute().as_posix()

    @property
    def processed_dir(self) -> str:
        """Path to the processed cache directory."""
        return Path(self.data_dir, "processed").absolute().as_posix()

    def raw_files_exist(self) -> bool:
        return Path(self.raw_dir).exists()

    def processed_files_exist(self) -> bool:
        return Path(self.processed_dir).exists()

    def cache_meta(self) -> Dict[str, Any]:
        """Parameters the processed cache depends on, stamped in its `meta.json` and checked before it is served.

        Subclasses extend it with the constructor arguments that change the cache content, such as a class subset.
        """
        return {"format_version": self.format_version}

    def __getitem__(self, index: int) -> Any:
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError

    def __repr__(self) -> str:
        head = "Dataset " + self.__class__.__name__
        body = [f"Number of samples: {self.__len__():,}"]
        if self.data_dir is not None:
            body.append(f"Data location: {self.data_dir}")
        body += self.extra_repr().splitlines()
        lines = [head] + [" " * self._repr_indent + line for line in body]
        return "\n".join(lines)

    def extra_repr(self) -> str:
        """Extra lines appended to the dataset `__repr__` (none by default)."""
        return ""
