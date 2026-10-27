"""Base class for point cloud datasets cached on disk under a `raw/` + `processed/` layout."""

from pathlib import Path
from typing import Any, Dict, List, Optional

from torch.utils.data import Dataset

from torch_pointcloud.utils.types import PathLike

from .utils import check_cache_meta, write_cache_meta


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

    Subclasses that keep a processed cache implement `write_processed_files` and `processed_files_exist`.
    `process` calls them: it stamps every path in `cache_meta_paths` with a `meta.json` holding `cache_meta`
    and raises when an existing stamp differs. Bump `format_version` whenever `write_processed_files` changes
    the files it writes or their contents, so caches from older releases are refused instead of served.

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

    @property
    def cache_meta_paths(self) -> List[str]:
        """Paths of the `meta.json` stamps, one per cache that `write_processed_files` builds as a whole."""
        return [Path(self.processed_dir, "meta.json").as_posix()]

    @property
    def cache_meta(self) -> Dict[str, Any]:
        """Parameters the processed cache depends on, stamped in its `meta.json` and checked before it is served.

        Subclasses extend it with the constructor arguments that change the cache content, such as a class subset.
        """
        return {"format_version": self.format_version}

    def process(self, force: bool = False, num_workers: Optional[int] = None, show_progress: bool = True) -> None:
        """Build the processed cache unless a complete one stamped with `cache_meta` already exists.

        A cache without a stamp is served as is when `processed_files_exist` accepts it. The stamps are removed
        before `write_processed_files` runs and written back once it reports the cache complete, so an
        interrupted or partial build is never served as complete.

        Args:
            force: Rebuild the cache even when a matching one exists.
            num_workers: Worker processes for `write_processed_files`, or `None` for sequential processing.
            show_progress: Show a progress bar while processing.

        Raises:
            RuntimeError: If a stamp differs from `cache_meta` and `force` is False.
        """
        meta_paths = [Path(meta_path) for meta_path in self.cache_meta_paths]
        if not force:
            for meta_path in meta_paths:
                check_cache_meta(meta_path, self.cache_meta)
            if all(meta_path.exists() for meta_path in meta_paths) or self.processed_files_exist():
                return

        for meta_path in meta_paths:
            meta_path.unlink(missing_ok=True)

        complete = self.write_processed_files(num_workers=num_workers, show_progress=show_progress)
        if complete:
            for meta_path in meta_paths:
                write_cache_meta(meta_path, self.cache_meta)

    def write_processed_files(self, num_workers: Optional[int] = None, show_progress: bool = True) -> bool:
        """Write the processed cache from the raw files and return whether it is complete.

        Returning False leaves the cache unstamped, so the next construction audits it instead of serving it.

        Args:
            num_workers: Worker processes, or `None` for sequential processing.
            show_progress: Show a progress bar while processing.
        """
        raise NotImplementedError

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
