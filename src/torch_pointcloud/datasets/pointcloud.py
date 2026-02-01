from pathlib import Path
from typing import Any

from torch.utils.data import Dataset

from torch_pointcloud.utils.types import PathLike


class PointCloudDataset(Dataset):
    _repr_indent = 4

    def __init__(self, root: PathLike) -> None:
        super().__init__()
        self.root = Path(root).as_posix()

    @property
    def name(self) -> str:
        return self.__class__.__name__

    @property
    def data_dir(self) -> str:
        return Path(self.root, f"{self.name}").absolute().as_posix()

    @property
    def raw_dir(self) -> str:
        return Path(self.data_dir, "raw").absolute().as_posix()

    @property
    def processed_dir(self) -> str:
        return Path(self.data_dir, "processed").absolute().as_posix()

    def raw_files_exist(self) -> bool:
        return Path(self.raw_dir).exists()

    def processed_files_exist(self) -> bool:
        return Path(self.processed_dir).exists()

    def __getitem__(self, index: int) -> Any:
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError

    def __repr__(self) -> str:
        head = "Dataset " + self.__class__.__name__
        body = [f"Number of points: {self.__len__():,}"]
        if self.data_dir is not None:
            body.append(f"Data location: {self.data_dir}")
        body += self.extra_repr().splitlines()
        lines = [head] + [" " * self._repr_indent + line for line in body]
        return "\n".join(lines)

    def extra_repr(self) -> str:
        return ""
