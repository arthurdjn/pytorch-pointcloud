"""The Paris-Lille-3D (NPM3D) dataset.

[Paris-Lille-3D: A Point Cloud Dataset for Urban Scene Segmentation and Classification](https://arxiv.org/abs/1712.00032)
by Roynard, Deschaud and Goulette (2018).

The 10-class benchmark used by the Open3D-ML model zoo splits the data into:

- train: `Lille1_1.ply`, `Lille1_2.ply`, `Paris.ply`
- val / held-out: `Lille2.ply`

There is also a 50-class research split with three large `.ply` files plus
per-class XML annotations; this loader targets the standard 10-class benchmark
used by published RandLA-Net checkpoints.
"""

from pathlib import Path
from typing import Any, Callable, Dict, Literal, Optional, Sequence, Tuple, get_args

import numpy as np
import plyfile
import torch
from torch import Tensor
from typing_extensions import override

from torch_pointcloud.utils.conversion import ensure_tuple
from torch_pointcloud.utils.types import PathLike

from .pointcloud import PointCloudDataset

# 10 raw classes; class 0 ("unclassified") is the standard ignored label.
ParisLille3DClass = Literal[
    "unclassified",
    "ground",
    "building",
    "pole-road_sign-traffic_light",
    "bollard-small_pole",
    "trash_can",
    "barrier",
    "pedestrian",
    "car",
    "natural-vegetation",
]
PARISLILLE3D_CLASSES: Tuple[ParisLille3DClass, ...] = get_args(ParisLille3DClass)
PARISLILLE3D_IGNORE_IDX = 0  # 'unclassified'

ParisLille3DSplit = Literal["train", "val", "trainval", "all"]

# Default file groupings for the 10-class benchmark, matching Open3D-ML.
_FILES_PER_SPLIT: Dict[str, Tuple[str, ...]] = {
    "train": ("Lille1_1.ply", "Lille1_2.ply", "Paris.ply"),
    "val": ("Lille2.ply",),
    "trainval": ("Lille1_1.ply", "Lille1_2.ply", "Lille2.ply", "Paris.ply"),
    "all": ("Lille1_1.ply", "Lille1_2.ply", "Lille2.ply", "Paris.ply"),
}


def load_parislille3d_data(ply_path: PathLike, /) -> Dict[str, Tensor]:
    """Parse a Paris-Lille-3D 10-class PLY file.

    The training files store per-vertex `x`, `y`, `z` (`float32`), `reflectance`
    (`uchar`) and `class` (`int32`). The `class` column is omitted in test files;
    callers receive only `pos` and `reflectance` in that case.

    | Key           | Shape    | Dtype   |
    | ------------- | -------- | ------- |
    | `pos`         | $(N, 3)$ | float32 |
    | `reflectance` | $(N, 1)$ | uint8   |
    | `segment`     | $(N,)$   | int64   | (optional)
    """
    plydata = plyfile.PlyData.read(Path(ply_path).as_posix())
    v = plydata["vertex"].data
    fields = set(v.dtype.names or ())

    pos = np.ascontiguousarray(np.stack([v["x"], v["y"], v["z"]], axis=1)).astype(np.float32)
    reflectance = np.ascontiguousarray(v["reflectance"]).astype(np.uint8).reshape(-1, 1)
    data = {
        "pos": torch.from_numpy(pos),
        "reflectance": torch.from_numpy(reflectance),
    }

    if "class" in fields:
        segment = np.ascontiguousarray(v["class"]).astype(np.int64)
        data["segment"] = torch.from_numpy(segment)

    return data


class ParisLille3D(PointCloudDataset):
    """The Paris-Lille-3D 10-class benchmark.

    Each sample is a single PLY scan, returned as a dictionary:

    | Key           | Shape    | Dtype   | Meaning                                            |
    | ------------- | -------- | ------- | -------------------------------------------------- |
    | `pos`         | $(N, 3)$ | float32 | XYZ                                                |
    | `reflectance` | $(N, 1)$ | uint8   | LiDAR reflectance                                  |
    | `segment`     | $(N,)$   | int64   | Raw class id, 0-9 (0 = `unclassified`, ignored)    |
    | `name`        | str      |         | Source file name without extension                 |

    Args:
        root: Dataset root. Files are read from `<root>/ParisLille3D/raw/<file>.ply`.
        split: One of `"train"` / `"val"` / `"trainval"` / `"all"`. The 10-class
            benchmark holds out `Lille2.ply` as the val split; the public test files
            (`test_10_classes/`) have no labels and are not loaded here.
        files: Optional explicit list of file names. Overrides `split`.
        transform: Callable applied to each loaded sample dict at `__getitem__` time.

    !!! note
        The raw dataset must be downloaded manually from
        [npm3d.fr/paris-lille-3d](https://npm3d.fr/paris-lille-3d). The expected
        layout is `<root>/ParisLille3D/raw/<Lille1_1, Lille1_2, Lille2, Paris>.ply`.
    """

    def __init__(
        self,
        root: PathLike,
        *,
        split: ParisLille3DSplit = "val",
        files: Optional[Sequence[str]] = None,
        transform: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
    ) -> None:
        super().__init__(root)
        self.split = split
        self.transform = transform

        if files is not None:
            self.files: Tuple[str, ...] = ensure_tuple(files)
        elif split in _FILES_PER_SPLIT:
            self.files = _FILES_PER_SPLIT[split]
        else:
            valid = ", ".join(_FILES_PER_SPLIT.keys())
            raise ValueError(f"Unknown split {split!r}. Expected one of: {valid}.")

        self.paths: Tuple[Path, ...] = tuple(Path(self.raw_dir, f) for f in self.files)
        for p in self.paths:
            if not p.exists():
                raise FileNotFoundError(
                    f"Dataset not found: expected Paris-Lille-3D scan at {p.as_posix()}; "
                    f"download from https://npm3d.fr/paris-lille-3d and place under {self.raw_dir!r}."
                )

    @override
    def raw_files_exist(self) -> bool:
        return all(p.exists() for p in self.paths)

    def __len__(self) -> int:
        return len(self.paths)

    @override
    def __getitem__(self, index: int) -> Dict[str, Any]:
        path = self.paths[index]
        data: Dict[str, Any] = load_parislille3d_data(path)
        data["name"] = path.stem
        if self.transform is not None:
            data = self.transform(data)
        return data

    def extra_repr(self) -> str:
        return f"split={self.split!r}, files={list(self.files)}"
