"""The Toronto-3D dataset.

{{ paper("2003.08284") }}

:arxiv: [Toronto-3D: A Large-scale Mobile LiDAR Dataset for Semantic Segmentation of Urban Roadways](https://arxiv.org/abs/2003.08284)
by Tan, Ma, Liu, Bobkov, Pukhalskaya, Eichenberger, Tatarchenko, Kosinka, et al.

The dataset contains four large-scale outdoor mobile LiDAR scans (`L001.ply`,
`L002.ply`, `L003.ply`, `L004.ply`) covering ~1 km of Toronto roadways. The
official benchmark uses `L002.ply` as the held-out test split.
"""

from pathlib import Path
from typing import Any, Callable, Dict, Literal, Optional, Sequence, Tuple, get_args

import numpy as np
import plyfile
import torch
from torch import Tensor
from typing_extensions import override

from torch_pointcloud.utils.conversion import ensure_tuple
from torch_pointcloud.utils.data import DataKeys
from torch_pointcloud.utils.types import PathLike

from .pointcloud import PointCloudDataset

# 9 raw classes, in the order the original dataset exports labels.
# Class 0 ('Unclassified') is ignored at evaluation, leaving the 8-class benchmark
# (Ground / Road_markings / Natural / Building / Utility_line / Pole / Car / Fence).
Toronto3DClass = Literal[
    "Unclassified",
    "Ground",
    "Road_markings",
    "Natural",
    "Building",
    "Utility_line",
    "Pole",
    "Car",
    "Fence",
]
TORONTO3D_CLASSES: Tuple[Toronto3DClass, ...] = get_args(Toronto3DClass)
TORONTO3D_IGNORE_IDX = 0  # 'Unclassified'

# Local UTM offset applied to every point so coordinates are not in raw UTM range
# (which is ~6.27e5, 4.84e6 for southern Ontario), matching the published checkpoints' convention.
TORONTO3D_UTM_OFFSET = (627285.0, 4841948.0, 0.0)

Toronto3DSplit = Literal["train", "val", "test", "trainval", "all"]


def load_toronto3d_data(path: PathLike, /, utm_offset: Sequence[float] = TORONTO3D_UTM_OFFSET) -> Dict[str, Tensor]:
    """Parse a Toronto-3D PLY scan and return per-key tensors.

    The PLY files are CloudCompare exports with these vertex properties (in order):

    - `x`, `y`, `z` (`double`),
    - `red`, `green`, `blue` (`uchar`),
    - `scalar_Intensity`, `scalar_GPSTime`, `scalar_ScanAngleRank` (`float`),
    - `scalar_Label` (`float`).

    Coordinates are returned in `float32` after the UTM offset is subtracted; colors
    are `uint8` (0-255); `intensity` and `gps_time` stay `float32`; `segment`
    is `int64` (raw class id, 0..8).

    | Key         | Shape    | Dtype   |
    | ----------- | -------- | ------- |
    | `pos`       | $(N, 3)$ | float32 |
    | `color`     | $(N, 3)$ | uint8   |
    | `intensity` | $(N, 1)$ | float32 |
    | `gps_time`  | $(N, 1)$ | float32 |
    | `segment`   | $(N,)$   | int64   |

    Args:
        path: Path to the `.ply` file.
        utm_offset: 3-vector subtracted from `(x, y, z)` to keep coordinates in a
            small numerical range. Defaults to `TORONTO3D_UTM_OFFSET`; pass
            `(0, 0, 0)` to keep raw UTM coordinates.
    """
    plydata = plyfile.PlyData.read(Path(path).as_posix())
    v = plydata["vertex"].data

    # Per-property arrays returned by `plyfile` are non-contiguous views into the
    # interleaved buffer; copy via `np.ascontiguousarray` before the dtype cast so
    # `torch.from_numpy` can take ownership without re-strided slices.
    # Subtract the UTM offset in float64 before the float32 cast: raw UTM coordinates are ~4.8e6,
    # where float32 resolution is 0.5 m, so casting first quantizes the positions.
    pos = np.stack([v["x"], v["y"], v["z"]], axis=1) - np.asarray(utm_offset, dtype=np.float64)
    pos = np.ascontiguousarray(pos.astype(np.float32))
    color = np.ascontiguousarray(np.stack([v["red"], v["green"], v["blue"]], axis=1)).astype(np.uint8)
    intensity = np.ascontiguousarray(v["scalar_Intensity"]).astype(np.float32).reshape(-1, 1)
    gps_time = np.ascontiguousarray(v["scalar_GPSTime"]).astype(np.float32).reshape(-1, 1)
    segment = np.ascontiguousarray(v["scalar_Label"]).astype(np.int64)

    return {
        DataKeys.POS: torch.from_numpy(pos),
        DataKeys.COLOR: torch.from_numpy(color),
        DataKeys.INTENSITY: torch.from_numpy(intensity),
        DataKeys.GPS_TIME: torch.from_numpy(gps_time),
        DataKeys.SEGMENT: torch.from_numpy(segment),
    }


class Toronto3D(PointCloudDataset):
    """The Toronto-3D dataset.

    Each sample is a single CloudCompare-exported PLY scan from the four-tile mobile
    LiDAR sweep, returned as a dictionary:

    | Key         | Shape    | Dtype   | Meaning                                           |
    | ----------- | -------- | ------- | ------------------------------------------------- |
    | `pos`       | $(N, 3)$ | float32 | XYZ (UTM offset by `utm_offset`)                  |
    | `color`     | $(N, 3)$ | uint8   | RGB (0-255)                                       |
    | `intensity` | $(N, 1)$ | float32 | LiDAR intensity                                   |
    | `gps_time`  | $(N, 1)$ | float32 | GPS timestamp                                     |
    | `segment`   | $(N,)$   | int64   | Raw class id, 0-8 (0 = `Unclassified`, ignored)   |
    | `name`      | str      |         | Source file name without extension                |

    Args:
        root: Dataset root. Files are read from `<root>/Toronto3D/raw/<file>.ply`.
        split: One of `"train"` / `"val"` / `"test"` / `"trainval"` / `"all"`.
            Test labels are publicly available so val and test are the same file
            (`L002.ply`).
        files: Optional explicit list of file names. Overrides `split`.
        utm_offset: 3-vector subtracted from raw UTM `(x, y, z)` to keep
            coordinates in a small numerical range. Defaults to
            `TORONTO3D_UTM_OFFSET`; pass `(0, 0, 0)` to keep raw UTM coordinates.
        transform: Callable applied to each loaded sample dict at `__getitem__` time.

    !!! note
        The raw dataset must be downloaded manually from
        :github: [WeikaiTan/Toronto-3D](https://github.com/WeikaiTan/Toronto-3D) (a license
        must be accepted). The expected layout is `<root>/Toronto3D/raw/L00{1,2,3,4}.ply`.
    """

    data_url: str = "https://github.com/WeikaiTan/Toronto-3D"

    #: Default file groupings: `L002.ply` is the canonical held-out test/val tile.
    files_per_split: Dict[str, Tuple[str, ...]] = {
        "train": ("L001.ply", "L003.ply", "L004.ply"),
        "val": ("L002.ply",),
        "test": ("L002.ply",),
        "trainval": ("L001.ply", "L002.ply", "L003.ply", "L004.ply"),
        "all": ("L001.ply", "L002.ply", "L003.ply", "L004.ply"),
    }

    def __init__(
        self,
        root: PathLike,
        *,
        split: Toronto3DSplit = "test",
        files: Optional[Sequence[str]] = None,
        utm_offset: Sequence[float] = TORONTO3D_UTM_OFFSET,
        transform: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
    ) -> None:
        super().__init__(root)
        self.split = split
        self.utm_offset = tuple(utm_offset)
        self.transform = transform

        if files is not None:
            self.files: Tuple[str, ...] = ensure_tuple(files)
        elif split in self.files_per_split:
            self.files = self.files_per_split[split]
        else:
            valid = ", ".join(self.files_per_split.keys())
            raise ValueError(f"Unknown split {split!r}. Expected one of: {valid}.")

        self.paths: Tuple[Path, ...] = tuple(Path(self.raw_dir, f) for f in self.files)
        for p in self.paths:
            if not p.exists():
                raise FileNotFoundError(
                    f"Dataset not found: expected Toronto-3D scan at {p.as_posix()}; "
                    f"download from {self.data_url} and place under {self.raw_dir!r}."
                )

    @override
    def raw_files_exist(self) -> bool:
        return all(p.exists() for p in self.paths)

    def download(self, force: bool = False) -> None:
        """Toronto-3D must be downloaded manually (a license must be accepted).

        Args:
            force: Unused; present to mirror the other datasets' `download` signature.
        Raises:
            RuntimeError: Always; automatic download is not supported.
        """
        raise RuntimeError(
            f"{self.__class__.__name__} does not support automatic download. Download the Toronto-3D archive "
            f"from {self.data_url!r} and place the `.ply` files under {self.raw_dir!r}."
        )

    def __len__(self) -> int:
        return len(self.paths)

    @override
    def __getitem__(self, index: int) -> Dict[str, Any]:
        path = self.paths[index]
        data: Dict[str, Any] = load_toronto3d_data(path, utm_offset=self.utm_offset)
        data[DataKeys.NAME] = path.stem
        if self.transform is not None:
            data = self.transform(data)
        return data

    def extra_repr(self) -> str:
        return f"split={self.split!r}, files={list(self.files)}"
