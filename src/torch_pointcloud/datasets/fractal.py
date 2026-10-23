"""The FRACTAL dataset.

{{ paper("2405.04634") }}

:arxiv: [FRACTAL: An Ultra-Large-Scale Aerial Lidar Dataset for 3D Semantic Segmentation of Diverse Landscapes](https://arxiv.org/abs/2405.04634)
by Charles Gaydon, Michel Daab, Floryne Roche.

100,000 aerial LiDAR patches of 50 x 50 m from the IGN LiDAR HD program, colourised with RGB and near-infrared
orthoimages, sampled from five regions of southern France: 80,000 for training, 10,000 each for validation and test.
Each split ships as zip archives of 1,000 LAZ files, which the dataset reads in place.
"""

import io
import zipfile
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Literal, Optional, Tuple, Union, get_args

import numpy as np
import torch
from torch import Tensor
from typing_extensions import override

from torch_pointcloud.utils.data import DataKeys
from torch_pointcloud.utils.imports import _LASPY_GITHUB_URL, optional_import
from torch_pointcloud.utils.types import PathLike

from .pointcloud import PointCloudDataset
from .utils import download_url

if TYPE_CHECKING:
    import laspy
else:
    laspy, _ = optional_import("laspy", url=_LASPY_GITHUB_URL)


FractalClass = Literal[
    "other",
    "ground",
    "vegetation",
    "building",
    "water",
    "bridge",
    "permanent_structure",
]
FRACTAL_CLASSES: Tuple[FractalClass, ...] = get_args(FractalClass)
# LiDAR HD classification codes of each FRACTAL class: the three vegetation heights (3, 4, 5) merge into one class.
# The artifacts (65) and virtual points (66) belong to no class.
FRACTAL_LEARNING_MAP: Dict[int, int] = {1: 0, 2: 1, 3: 2, 4: 2, 5: 2, 6: 3, 9: 4, 17: 5, 64: 6}

FractalSplit = Literal["train", "val", "test"]


def load_fractal_data(source: Union[PathLike, bytes]) -> Dict[str, Tensor]:
    r"""Read a FRACTAL (or any colourised LiDAR HD) LAS / LAZ file.

    Positions are the Lambert-93 coordinates in float64: around $6.5 \cdot 10^6$ m, float32 only resolves 0.5 m.

    | Key                 | Shape    | Dtype   |
    | ------------------- | -------- | ------- |
    | `pos`               | $(N, 3)$ | float64 |
    | `intensity`         | $(N, 1)$ | float32 |
    | `return_number`     | $(N, 1)$ | int64   |
    | `number_of_returns` | $(N, 1)$ | int64   |
    | `color`             | $(N, 3)$ | int32   |
    | `nir`               | $(N, 1)$ | int32   |
    | `segment`           | $(N,)$   | int64   |

    Args:
        source: Path to a `.las` / `.laz` file, or its bytes. LAZ needs a laspy backend (`pip install laspy[lazrs]`).

    Returns:
        The point attributes, keyed by `DataKeys`. `segment` holds the raw LiDAR HD classification codes.
    """
    las = laspy.read(io.BytesIO(source) if isinstance(source, bytes) else source)
    pos = np.stack([las.x, las.y, las.z], axis=1)
    color = np.stack([las.red, las.green, las.blue], axis=1).astype(np.int32)
    return {
        DataKeys.POS: torch.from_numpy(pos),
        DataKeys.INTENSITY: torch.from_numpy(np.asarray(las.intensity, dtype=np.float32)).unsqueeze(1),
        DataKeys.RETURN_NUMBER: torch.from_numpy(np.asarray(las.return_number, dtype=np.int64)).unsqueeze(1),
        DataKeys.NUMBER_OF_RETURNS: torch.from_numpy(np.asarray(las.number_of_returns, dtype=np.int64)).unsqueeze(1),
        DataKeys.COLOR: torch.from_numpy(color),
        DataKeys.NIR: torch.from_numpy(np.asarray(las.nir, dtype=np.int32)).unsqueeze(1),
        DataKeys.SEGMENT: torch.from_numpy(np.asarray(las.classification, dtype=np.int64)),
    }


class FRACTAL(PointCloudDataset):
    """The FRACTAL dataset.

    Each sample is one 50 x 50 m patch, returned as a dictionary:

    | Key                 | Shape    | Dtype   | Meaning                                                  |
    | ------------------- | -------- | ------- | -------------------------------------------------------- |
    | `pos`               | $(N, 3)$ | float64 | Lambert-93 (EPSG:2154) metres                            |
    | `intensity`         | $(N, 1)$ | float32 | LiDAR intensity                                          |
    | `return_number`     | $(N, 1)$ | int64   | Return number, 1-based                                   |
    | `number_of_returns` | $(N, 1)$ | int64   | Number of returns of the pulse                           |
    | `color`             | $(N, 3)$ | int32   | RGB, 16-bit                                              |
    | `nir`               | $(N, 1)$ | int32   | Near-infrared, 16-bit                                    |
    | `segment`           | $(N,)$   | int64   | LiDAR HD classification code, see `FRACTAL_LEARNING_MAP` |
    | `name`              | str      |         | Patch file name without extension                        |

    Reading the LAZ files needs `laspy` with a LAZ backend: `pip install laspy[lazrs]`.

    Args:
        root: Dataset root. Archives are read from `<root>/FRACTAL/raw/<split>-<NN>.zip`.
        split: One of `"train"` (the default), `"val"` or `"test"`.
        transform: Callable applied to each loaded sample dict at `__getitem__` time.
        download: Download the archives of `split` missing from `raw/` (9 GB for `val` or `test`, 64 GB for `train`).
        show_progress: Whether to display download progress bars.

    Example:
        ```{.python notest}
        from torch_pointcloud.datasets import FRACTAL

        dataset = FRACTAL(root="data", split="test", download=True)
        sample = dataset[0]
        ```
    """

    data_url: str = "https://huggingface.co/datasets/IGNF/FRACTAL/resolve/36e00245ef92fd180f655f6ee410f2e935693cc2"
    num_archives: Dict[str, int] = {"train": 80, "val": 10, "test": 10}

    def __init__(
        self,
        root: PathLike,
        *,
        split: FractalSplit = "train",
        transform: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
        download: bool = False,
        show_progress: bool = True,
    ) -> None:
        if split not in self.num_archives:
            raise ValueError(f"Unknown split {split!r}. Expected one of: {', '.join(self.num_archives)}.")

        super().__init__(root)
        self.split = split
        self.transform = transform
        self.show_progress = show_progress

        if download:
            self.download()

        if not self.raw_files_exist():
            raise FileNotFoundError(
                f"Dataset not found: expected {self.num_archives[split]} archives {split}-NN.zip under "
                f"{self.raw_dir!r}; pass `download=True` or fetch `data/{split}` from "
                "https://huggingface.co/datasets/IGNF/FRACTAL."
            )

        self.members: List[Tuple[Path, str]] = []
        for archive in self.archives:
            with zipfile.ZipFile(archive) as zip_file:
                names = sorted(name for name in zip_file.namelist() if name.endswith((".laz", ".las")))
            self.members.extend((archive, name) for name in names)

    @property
    def archives(self) -> Tuple[Path, ...]:
        """Paths of the zip archives of the split."""
        return tuple(Path(self.raw_dir, f"{self.split}-{i:02d}.zip") for i in range(self.num_archives[self.split]))

    @override
    def raw_files_exist(self) -> bool:
        return all(path.exists() for path in self.archives)

    def download(self, force: bool = False) -> None:
        """Download the archives of the split from the Hugging Face Hub.

        Args:
            force: Download every archive again, even the ones already in `raw/`.
        """
        for path in self.archives:
            download_url(
                f"{self.data_url}/data/{self.split}/{path.name}",
                path,
                description=f"Downloading {path.name}",
                show_progress=self.show_progress,
                overwrite=True if force else "incomplete",
            )

    def __len__(self) -> int:
        return len(self.members)

    @override
    def __getitem__(self, index: int) -> Dict[str, Any]:
        archive, name = self.members[index]
        with zipfile.ZipFile(archive) as zip_file:
            content = zip_file.read(name)

        data: Dict[str, Any] = load_fractal_data(content)
        data[DataKeys.NAME] = Path(name).stem

        if self.transform is not None:
            data = self.transform(data)
        return data

    @override
    def extra_repr(self) -> str:
        return f"split={self.split!r}"
