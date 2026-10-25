"""The GridNet-HD dataset.

{{ paper("2601.13052") }}

:arxiv: [GridNet-HD: A High-Resolution Multi-Modal Dataset for LiDAR-Image Fusion on Power Line Infrastructure](https://arxiv.org/abs/2601.13052)
by Antoine Carreaud, Shanci Li, Malo De Lacour, Digre Frinde, Jan Skaloud, Adrien Gressin.

36 UAV LiDAR zones of a high-voltage line in the Swiss Alps (EPSG:32632), 2.4 billion coloured points labelled
with eleven classes: 21 zones for training, 6 for validation and 9 for testing, whose labels are kept for the
leaderboard. The Hub release ships one `.las` file per zone.
"""

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Literal, Optional, Tuple, get_args

import numpy as np
import torch
from torch import Tensor
from typing_extensions import override

from torch_pointcloud.transforms.utils import relabel
from torch_pointcloud.utils.data import DataKeys
from torch_pointcloud.utils.imports import _LASPY_GITHUB_URL, optional_import
from torch_pointcloud.utils.types import PathLike

from .pointcloud import PointCloudDataset
from .utils import download_url

if TYPE_CHECKING:
    import laspy
else:
    laspy, _ = optional_import("laspy", url=_LASPY_GITHUB_URL)

# The 11 benchmark classes, in the order the release groups its annotated classes (`GRIDNETHD_LABEL_GROUPS`).
GridNetHDClass = Literal[
    "pylon",
    "conductor_cable",
    "structural_cable",
    "insulator",
    "high_vegetation",
    "low_vegetation",
    "herbaceous_vegetation",
    "rock_gravel_soil",
    "impervious_soil",
    "water",
    "building",
]
GRIDNETHD_CLASSES: Tuple[GridNetHDClass, ...] = get_args(GridNetHDClass)
GRIDNETHD_IGNORE_IDX = 255

# Annotated class ids of the `ground_truth` field grouped into each benchmark class; 12, 13 and 255 are ignored.
GRIDNETHD_LABEL_GROUPS: Tuple[Tuple[int, ...], ...] = (
    (0, 1, 2, 3, 4),
    (5,),
    (6, 7),
    (8, 9, 10, 11),
    (14,),
    (15,),
    (16,),
    (17, 18),
    (19,),
    (20,),
    (21,),
)
GRIDNETHD_LABEL_MAPPING: Dict[int, int] = {
    source: label for label, sources in enumerate(GRIDNETHD_LABEL_GROUPS) for source in sources
}
GridNetHDSplit = Literal["train", "val", "test"]


def load_gridnethd_data(path: PathLike, /) -> Dict[str, Tensor]:
    r"""Read a GridNet-HD LAS zone.

    Positions are returned relative to the minimum corner of the zone, subtracted in float64, so that the UTM
    coordinates (northings around $5.1 \cdot 10^6$ m, where float32 resolves 0.5 m) keep their millimetre
    precision. Colours are the 16-bit LAS values reduced to 8 bits.

    | Key                 | Shape    | Dtype   |
    | ------------------- | -------- | ------- |
    | `pos`               | $(N, 3)$ | float32 |
    | `color`             | $(N, 3)$ | uint8   |
    | `intensity`         | $(N, 1)$ | float32 |
    | `return_number`     | $(N, 1)$ | int64   |
    | `number_of_returns` | $(N, 1)$ | int64   |
    | `segment`           | $(N,)$   | int64   |

    Args:
        path: Path to a `.las` file.

    Returns:
        The point attributes, keyed by `DataKeys`. `segment` holds the benchmark class ids 0-10, with
        `GRIDNETHD_IGNORE_IDX` for the unassigned points, and is absent from the test zones (no `ground_truth` field).
    """
    las = laspy.read(Path(path).as_posix())
    # A zone holds up to 110 million points: the labels come first, while nothing else is resident, and the
    # positions one float64 axis at a time.
    segment = _read_segment(las) if "ground_truth" in las.point_format.dimension_names else None
    pos = np.empty((las.header.point_count, 3), dtype=np.float32)
    for axis, coordinates in enumerate((las.x, las.y, las.z)):
        coordinates = np.asarray(coordinates)
        pos[:, axis] = coordinates - coordinates.min()
    rgb = np.stack([las.red, las.green, las.blue], axis=1)
    data: Dict[str, Tensor] = {
        DataKeys.POS: torch.from_numpy(pos),
        DataKeys.COLOR: torch.from_numpy((rgb >> 8).astype(np.uint8)),
        DataKeys.INTENSITY: torch.from_numpy(np.asarray(las.intensity, dtype=np.float32)).unsqueeze(1),
        DataKeys.RETURN_NUMBER: torch.from_numpy(np.asarray(las.return_number, dtype=np.int64)).unsqueeze(1),
        DataKeys.NUMBER_OF_RETURNS: torch.from_numpy(np.asarray(las.number_of_returns, dtype=np.int64)).unsqueeze(1),
    }
    if segment is not None:
        data[DataKeys.SEGMENT] = segment
    return data


def _read_segment(las: "laspy.LasData") -> Tensor:
    """The benchmark class of every point, grouped from the `ground_truth` field."""
    ground_truth = torch.from_numpy(np.asarray(las["ground_truth"], dtype=np.int64))
    return relabel(ground_truth, GRIDNETHD_LABEL_MAPPING, default=GRIDNETHD_IGNORE_IDX)


class GridNetHD(PointCloudDataset):
    """The GridNet-HD dataset.

    Each sample is one zone of the line, returned as a dictionary:

    | Key                 | Shape    | Dtype   | Meaning                                                      |
    | ------------------- | -------- | ------- | ------------------------------------------------------------ |
    | `pos`               | $(N, 3)$ | float32 | EPSG:32632 metres, minus the zone's minimum corner           |
    | `color`             | $(N, 3)$ | uint8   | RGB                                                          |
    | `intensity`         | $(N, 1)$ | float32 | LiDAR intensity                                              |
    | `return_number`     | $(N, 1)$ | int64   | Return number, 1-based                                       |
    | `number_of_returns` | $(N, 1)$ | int64   | Number of returns of the pulse                               |
    | `segment`           | $(N,)$   | int64   | Benchmark class id, 0-10 (255 = unassigned); `train` and `val` only |
    | `name`              | str      |         | Zone name                                                    |

    The `segment` labels are the eleven benchmark classes the release scores (`GRIDNETHD_CLASSES`), grouped from the
    annotated classes of the `ground_truth` field as `GRIDNETHD_LABEL_GROUPS` lists. The test zones carry no labels:
    the leaderboard keeps them.

    Reading the LAS files needs the optional `laspy` package.

    Args:
        root: Dataset root. Zones are read from `<root>/GridNetHD/raw/<zone>/lidar/<zone>.las` and the zone lists
            from `<root>/GridNetHD/raw/split.json`, the layout of the Hub repository.
        split: Which zones to load, `"train"` (21 zones, the default), `"val"` (6) or `"test"` (9).
        transform: Callable applied to each loaded sample dict at `__getitem__` time.
        download: Whether to download the split file and the zones of `split` from the Hub (CC BY 4.0).
        force_download: Whether to download the files again when they exist.

    !!! note
        A zone weighs 1.2 to 4.3 GB (40 to 110 million points): the training split is 52 GB, the validation
        split 16 GB and the test split 24 GB.

    Example:
        ```{.python notest}
        from torch_pointcloud.datasets import GridNetHD

        dataset = GridNetHD(root="data", split="val", download=True)
        sample = dataset[0]
        ```
    """

    data_url: str = "https://huggingface.co/datasets/heig-vd-geo/GridNet-HD"
    revision: str = "944b561090d1fed9803ccc96bfafb6ab3049726c"

    def __init__(
        self,
        root: PathLike,
        *,
        split: GridNetHDSplit = "train",
        transform: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
        download: bool = False,
        force_download: bool = False,
    ) -> None:
        if split not in get_args(GridNetHDSplit):
            raise ValueError(f"Invalid split {split!r}; expected one of {get_args(GridNetHDSplit)}.")

        super().__init__(root)
        self.split = split
        self.transform = transform

        if download or force_download:
            self.download(force=force_download)

        if not self.raw_files_exist():
            raise FileNotFoundError(
                f"Dataset not found: expected {self.split_path!r} and the {split!r} zones at "
                f"{self.raw_dir}/<zone>/lidar/<zone>.las. Pass `download=True` or download them from {self.data_url!r}."
            )

        self.zones: Tuple[str, ...] = tuple(self.read_split()[split])
        self.paths: Tuple[Path, ...] = tuple(self.zone_path(zone) for zone in self.zones)

    @property
    def split_path(self) -> str:
        """Path of the `split.json` file listing the zones of each split."""
        return Path(self.raw_dir, "split.json").as_posix()

    def zone_path(self, zone: str) -> Path:
        """Path of the `.las` file of a zone."""
        return Path(self.raw_dir, zone, "lidar", f"{zone}.las")

    def read_split(self) -> Dict[str, List[str]]:
        """The zone names of each split, as `split.json` lists them."""
        with open(self.split_path) as f:
            split: Dict[str, List[str]] = json.load(f)
        return split

    @override
    def raw_files_exist(self) -> bool:
        if not Path(self.split_path).is_file():
            return False
        return all(self.zone_path(zone).is_file() for zone in self.read_split()[self.split])

    def download(self, force: bool = False) -> None:
        """Download `split.json` and the `.las` file of every zone of the split from the Hub.

        Args:
            force: Whether to download the files again when they exist.
        """
        base_url = f"{self.data_url}/resolve/{self.revision}"
        download_url(f"{base_url}/split.json", self.split_path, overwrite=force)
        for zone in self.read_split()[self.split]:
            download_url(f"{base_url}/{zone}/lidar/{zone}.las", self.zone_path(zone), overwrite=force)

    def __len__(self) -> int:
        return len(self.paths)

    @override
    def __getitem__(self, index: int) -> Dict[str, Any]:
        data: Dict[str, Any] = load_gridnethd_data(self.paths[index])
        data[DataKeys.NAME] = self.zones[index]
        if self.transform is not None:
            data = self.transform(data)
        return data

    def extra_repr(self) -> str:
        return f"split={self.split!r}"
