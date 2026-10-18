"""The DALES dataset.

{{ paper("2004.11985") }}

:arxiv: [DALES: A Large-scale Aerial LiDAR Data Set for Semantic Segmentation](https://arxiv.org/abs/2004.11985)
by Nina Varney, Vijayan K. Asari, Quinn Graehling.

40 aerial LiDAR tiles of 500 x 500 m over the City of Surrey, British Columbia (UTM zone 10N), labelled with eight
classes: 29 tiles for training and 11 for testing. The LAS release ships one `.las` file per tile.
"""

from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, Literal, Optional, Tuple, get_args

import numpy as np
import torch
from torch import Tensor
from typing_extensions import override

from torch_pointcloud.utils.data import DataKeys
from torch_pointcloud.utils.imports import _LASPY_GITHUB_URL, optional_import
from torch_pointcloud.utils.types import PathLike

from .pointcloud import PointCloudDataset

if TYPE_CHECKING:
    import laspy
else:
    laspy, _ = optional_import("laspy", url=_LASPY_GITHUB_URL)

# 9 raw classes, in the order the release stores them in the LAS classification field.
# Class 0 ("unknown") is ignored at evaluation, leaving the 8-class benchmark.
DalesClass = Literal[
    "unknown",
    "ground",
    "vegetation",
    "cars",
    "trucks",
    "power_lines",
    "fences",
    "poles",
    "buildings",
]
DALES_CLASSES: Tuple[DalesClass, ...] = get_args(DalesClass)
DALES_IGNORE_IDX = 0  # "unknown"


def load_dales_data(path: PathLike, /) -> Dict[str, Tensor]:
    r"""Read a DALES LAS tile.

    Positions are returned relative to the minimum corner of the tile, subtracted in float64, so that the UTM
    coordinates (northings around $5.4 \cdot 10^6$ m, where float32 resolves 0.5 m) keep their centimetre
    precision.

    | Key                 | Shape    | Dtype   |
    | ------------------- | -------- | ------- |
    | `pos`               | $(N, 3)$ | float32 |
    | `return_number`     | $(N, 1)$ | int64   |
    | `number_of_returns` | $(N, 1)$ | int64   |
    | `segment`           | $(N,)$   | int64   |

    Args:
        path: Path to a `.las` file.

    Returns:
        The point attributes, keyed by `DataKeys`. `segment` holds the raw class ids 0-8.
    """
    las = laspy.read(Path(path).as_posix())
    xyz = np.stack([las.x, las.y, las.z], axis=1)
    pos = (xyz - las.header.mins).astype(np.float32)
    return {
        DataKeys.POS: torch.from_numpy(pos),
        DataKeys.RETURN_NUMBER: torch.from_numpy(np.asarray(las.return_number, dtype=np.int64)).unsqueeze(1),
        DataKeys.NUMBER_OF_RETURNS: torch.from_numpy(np.asarray(las.number_of_returns, dtype=np.int64)).unsqueeze(1),
        DataKeys.SEGMENT: torch.from_numpy(np.asarray(las.classification, dtype=np.int64)),
    }


class DALES(PointCloudDataset):
    """The DALES dataset.

    Each sample is one 500 x 500 m tile, returned as a dictionary:

    | Key                 | Shape    | Dtype   | Meaning                                              |
    | ------------------- | -------- | ------- | ---------------------------------------------------- |
    | `pos`               | $(N, 3)$ | float32 | UTM zone 10N metres, minus the tile's minimum corner |
    | `return_number`     | $(N, 1)$ | int64   | Return number, 1-based                               |
    | `number_of_returns` | $(N, 1)$ | int64   | Number of returns of the pulse                       |
    | `segment`           | $(N,)$   | int64   | Raw class id, 0-8 (0 = `unknown`, ignored)           |
    | `name`              | str      |         | Tile file name without extension                     |

    Reading the LAS files needs the optional `laspy` package.

    Args:
        root: Dataset root. Tiles are read from `<root>/DALES/raw/dales_las/<split>/<tile>.las`, the layout of
            the `dales_semantic_segmentation_las.tar.gz` archive extracted under `raw/`.
        train: Whether to load the 29 training tiles (the default) or the 11 test tiles.
        transform: Callable applied to each loaded sample dict at `__getitem__` time.

    !!! note
        The raw dataset must be requested from the authors (the download form linked from `data_url`; the data
        are released under CC BY 3.0 for non-commercial use) and extracted by hand. The expected layout is
        `<root>/DALES/raw/dales_las/{train,test}/<tile>.las`.

    Example:
        ```{.python notest}
        from torch_pointcloud.datasets import DALES

        dataset = DALES(root="data", train=False)
        sample = dataset[0]
        ```
    """

    data_url: str = "https://sites.google.com/a/udayton.edu/vasari1/research/earth-vision/dales"

    def __init__(
        self,
        root: PathLike,
        *,
        train: bool = True,
        transform: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
    ) -> None:
        super().__init__(root)
        self.train = train
        self.transform = transform

        self.paths: Tuple[Path, ...] = tuple(sorted(Path(self.split_dir).glob("*.las")))
        if not self.paths:
            raise FileNotFoundError(
                f"Dataset not found: expected DALES tiles at {self.split_dir}/*.las; request the archive from "
                f"{self.data_url} and extract it under {self.raw_dir!r}."
            )

    @property
    def split(self) -> str:
        """Name of the split directory, `"train"` or `"test"`."""
        return "train" if self.train else "test"

    @property
    def split_dir(self) -> str:
        """Directory holding the `.las` tiles of the split."""
        return Path(self.raw_dir, "dales_las", self.split).as_posix()

    @override
    def raw_files_exist(self) -> bool:
        return any(Path(self.split_dir).glob("*.las"))

    def download(self, force: bool = False) -> None:
        """DALES must be requested from the authors and downloaded by hand.

        Args:
            force: Unused; present to mirror the other datasets' `download` signature.

        Raises:
            RuntimeError: Always; automatic download is not supported.
        """
        raise RuntimeError(
            f"{self.__class__.__name__} does not support automatic download. Request the DALES archive from "
            f"{self.data_url!r} and extract it under {self.raw_dir!r}."
        )

    def __len__(self) -> int:
        return len(self.paths)

    @override
    def __getitem__(self, index: int) -> Dict[str, Any]:
        path = self.paths[index]
        data: Dict[str, Any] = load_dales_data(path)
        data[DataKeys.NAME] = path.stem
        if self.transform is not None:
            data = self.transform(data)
        return data

    def extra_repr(self) -> str:
        return f"train={self.train}"
