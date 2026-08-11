"""The Semantic3D dataset.

:arxiv: [Semantic3D.net: A new Large-scale Point Cloud Classification Benchmark](https://arxiv.org/abs/1704.03847)
by Hackel, Savinov, Ladicky, Wegner, Schindler and Pollefeys (2017).

Each scene is distributed as a pair of plain-text files:

- `<scene>.txt` - one row per point with `x y z intensity r g b` (white-space
  separated, ASCII, no header). XYZ are float64 metres; intensity is float; RGB
  are `uint8` 0-255.
- `<scene>.labels` - one row per point with the integer class id ($0$ = unlabelled,
  $1$-$8$ = the eight benchmark classes). `.labels` is only present for the
  `reduced-8` and `training` splits.
"""

from pathlib import Path
from typing import Any, Callable, Dict, Literal, Optional, Sequence, Tuple, get_args

import numpy as np
import torch
from torch import Tensor
from typing_extensions import override

from torch_pointcloud.utils.conversion import ensure_tuple
from torch_pointcloud.utils.data import DataKeys
from torch_pointcloud.utils.types import PathLike

from .pointcloud import PointCloudDataset

# 9 raw classes; class 0 ("unlabelled") is the standard ignored label.
Semantic3DClass = Literal[
    "unlabelled",
    "man-made_terrain",
    "natural_terrain",
    "high_vegetation",
    "low_vegetation",
    "buildings",
    "hard_scape",
    "scanning_artefacts",
    "cars",
]
SEMANTIC3D_CLASSES: Tuple[Semantic3DClass, ...] = get_args(Semantic3DClass)
SEMANTIC3D_IGNORE_IDX = 0  # 'unlabelled'

Semantic3DSplit = Literal["train", "test", "all"]


def load_semantic3d_data(path_txt: PathLike, /, path_labels: Optional[PathLike] = None) -> Dict[str, Tensor]:
    """Parse a Semantic3D `<scene>.txt` (and optional `<scene>.labels`) file pair.

    The `.txt` files are large ASCII (often 100M+ rows) - we use `np.loadtxt`
    which is slow but free of compiled dependencies. The `segment` key is only
    populated when `path_labels` is provided and exists (held-out test scenes
    receive no `segment` key).

    | Key         | Shape    | Dtype   |
    | ----------- | -------- | ------- |
    | `pos`       | $(N, 3)$ | float32 |
    | `intensity` | $(N, 1)$ | float32 |
    | `color`     | $(N, 3)$ | uint8   |
    | `segment`   | $(N,)$   | int64   | (optional)
    """
    arr = np.loadtxt(Path(path_txt).as_posix(), dtype=np.float64)
    pos = arr[:, 0:3].astype(np.float32, copy=True)
    intensity = arr[:, 3:4].astype(np.float32, copy=True)
    color = arr[:, 4:7].astype(np.uint8, copy=True)
    data: Dict[str, Tensor] = {
        DataKeys.POS: torch.from_numpy(pos),
        DataKeys.INTENSITY: torch.from_numpy(intensity),
        DataKeys.COLOR: torch.from_numpy(color),
    }

    if path_labels is not None and Path(path_labels).exists():
        segment = np.loadtxt(Path(path_labels).as_posix(), dtype=np.int64)
        data[DataKeys.SEGMENT] = torch.from_numpy(segment)

    return data


class Semantic3D(PointCloudDataset):
    """The Semantic3D dataset.

    Each sample is a single ASCII-text scene, returned as a dictionary:

    | Key         | Shape    | Dtype   | Meaning                                                                   |
    | ----------- | -------- | ------- | ------------------------------------------------------------------------- |
    | `pos`       | $(N, 3)$ | float32 | XYZ                                                                       |
    | `intensity` | $(N, 1)$ | float32 | LiDAR intensity                                                           |
    | `color`     | $(N, 3)$ | uint8   | RGB (0-255)                                                               |
    | `segment`   | $(N,)$   | int64   | Raw class id 0-8 (0 = `unlabelled`, ignored). Absent for held-out scenes. |
    | `name`      | str      |         | Source scene name                                                         |

    Args:
        root: Dataset root. Files are read from `<root>/Semantic3D/raw/<scene>.txt`.
        split: One of `"train"` / `"test"` / `"all"`. Defaults to `"train"`.
            `"test"` returns the four `reduced-8` benchmark scenes (without labels).
        scenes: Optional explicit list of scene names (without the `.txt` / `.labels`
            suffix). Overrides `split`.
        transform: Callable applied to each loaded sample dict at `__getitem__` time.

    !!! note
        The dataset must be downloaded manually from
        [semantic3d.net](http://semantic3d.net) (a license must be accepted). The
        expected layout under `<root>/Semantic3D/raw/` is one `<scene>.txt` (and
        matching `<scene>.labels` for training scenes) per scene.

    !!! warning
        Loading is done with `np.loadtxt`, which reads the (often 100M+-row) ASCII
        files into RAM in a single shot. Each scene needs ~5 GB of RAM to parse;
        consider extracting only the scenes you need for evaluation.
    """

    data_url: str = "http://semantic3d.net"
    train_scenes: Tuple[str, ...] = (
        "bildstein_station1_xyz_intensity_rgb",
        "bildstein_station3_xyz_intensity_rgb",
        "bildstein_station5_xyz_intensity_rgb",
        "domfountain_station1_xyz_intensity_rgb",
        "domfountain_station2_xyz_intensity_rgb",
        "domfountain_station3_xyz_intensity_rgb",
        "neugasse_station1_xyz_intensity_rgb",
        "sg27_station1_intensity_rgb",
        "sg27_station2_intensity_rgb",
        "sg27_station4_intensity_rgb",
        "sg27_station5_intensity_rgb",
        "sg27_station9_intensity_rgb",
        "sg28_station4_intensity_rgb",
        "untermaederbrunnen_station1_xyz_intensity_rgb",
        "untermaederbrunnen_station3_xyz_intensity_rgb",
    )
    test_scenes: Tuple[str, ...] = (
        "MarketplaceFeldkirch_Station4_rgb_intensity-reduced",
        "sg27_station10_rgb_intensity-reduced",
        "sg28_Station2_rgb_intensity-reduced",
        "StGallenCathedral_station6_rgb_intensity-reduced",
    )

    def __init__(
        self,
        root: PathLike,
        *,
        split: Semantic3DSplit = "train",
        scenes: Optional[Sequence[str]] = None,
        transform: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
    ) -> None:
        super().__init__(root)
        self.split = split
        self.transform = transform

        if scenes is not None:
            self.scenes: Tuple[str, ...] = ensure_tuple(scenes)
        elif split == "train":
            self.scenes = self.train_scenes
        elif split == "test":
            self.scenes = self.test_scenes
        elif split == "all":
            self.scenes = self.train_scenes + self.test_scenes
        else:
            raise ValueError(f"Unknown split {split!r}. Expected one of: train, test, all.")

        self.txt_paths: Tuple[Path, ...] = tuple(Path(self.raw_dir, f"{s}.txt") for s in self.scenes)
        self.label_paths: Tuple[Optional[Path], ...] = tuple(
            (Path(self.raw_dir, f"{s}.labels") if Path(self.raw_dir, f"{s}.labels").exists() else None)
            for s in self.scenes
        )
        for txt in self.txt_paths:
            if not txt.exists():
                raise FileNotFoundError(
                    f"Dataset not found: expected Semantic3D scene at {txt.as_posix()}; "
                    f"download from {self.data_url} and place under {self.raw_dir!r}."
                )

    @override
    def raw_files_exist(self) -> bool:
        return all(p.exists() for p in self.txt_paths)

    def download(self, force: bool = False) -> None:
        """Semantic3D must be downloaded manually (a license must be accepted).

        Args:
            force: Unused; present to mirror the other datasets' `download` signature.
        Raises:
            RuntimeError: Always; automatic download is not supported.
        """
        raise RuntimeError(
            f"{self.__class__.__name__} does not support automatic download. Download the Semantic3D scenes "
            f"from {self.data_url!r} and place the `.txt` / `.labels` files under {self.raw_dir!r}."
        )

    def __len__(self) -> int:
        return len(self.scenes)

    @override
    def __getitem__(self, index: int) -> Dict[str, Any]:
        txt = self.txt_paths[index]
        lbl = self.label_paths[index]
        data: Dict[str, Any] = load_semantic3d_data(txt, lbl)
        data[DataKeys.NAME] = self.scenes[index]
        if self.transform is not None:
            data = self.transform(data)
        return data

    def extra_repr(self) -> str:
        return f"split={self.split!r}, scenes={list(self.scenes)}"
