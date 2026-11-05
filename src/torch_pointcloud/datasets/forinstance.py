"""The FOR-instance dataset.

{{ paper("2309.01279") }}

:arxiv: [FOR-instance: a UAV laser scanning benchmark dataset for semantic and instance segmentation of individual trees](https://arxiv.org/abs/2309.01279)
by Stefano Puliti, Grant Pearse, Peter Surový, Luke Wallace, Markus Hollaus, Maciej Wielgosz, Rasmus Astrup.

32 forest plots scanned from UAVs over Norway (NIBIO), the Czech Republic (CULS), Austria (TUWIEN), New Zealand
(SCION) and Australia (RMIT), each point labelled with a semantic class and the id of its tree. The release splits
them into 21 development and 11 test plots and ships one `.las` file per plot.
"""

import csv
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
from .utils import download_url, extract_zip, is_hash_valid

if TYPE_CHECKING:
    import laspy
else:
    laspy, _ = optional_import("laspy", url=_LASPY_GITHUB_URL)

# The semantic classes, in the order of the LAS classification codes.
FORinstanceClass = Literal[
    "unclassified",
    "low_vegetation",
    "terrain",
    "out_points",
    "stem",
    "live_branches",
    "woody_branches",
]
FORINSTANCE_CLASSES: Tuple[FORinstanceClass, ...] = get_args(FORinstanceClass)
FORINSTANCE_OUT_POINTS = 3  # trees outside the annotated plot
# Class codes to the five labelled classes, `low_vegetation` to `woody_branches`.
FORINSTANCE_LEARNING_MAP: Dict[int, int] = {1: 0, 2: 1, 4: 2, 5: 3, 6: 4}
# Class codes to not a tree (low vegetation, terrain) and tree (stem, live and woody branches).
FORINSTANCE_TREE_LEARNING_MAP: Dict[int, int] = {1: 0, 2: 0, 4: 1, 5: 1, 6: 1}


def load_forinstance_data(path: PathLike, /) -> Dict[str, Tensor]:
    r"""Read a FOR-instance LAS plot.

    Positions are the UTM coordinates and elevations in float64: northings around $6 \cdot 10^6$ m, where float32
    only resolves 0.5 m.

    | Key        | Shape    | Dtype   |
    | ---------- | -------- | ------- |
    | `pos`      | $(N, 3)$ | float64 |
    | `segment`  | $(N,)$   | int64   |
    | `instance` | $(N,)$   | int64   |

    Args:
        path: Path to a `.las` file.

    Returns:
        The point attributes, keyed by `DataKeys`. `segment` holds the classification codes 0-6, `instance` the
        tree id, $-1$ for points of no tree.
    """
    las = laspy.read(Path(path).as_posix())
    pos = np.stack([las.x, las.y, las.z], axis=1)
    tree = np.asarray(las["treeID"]).astype(np.int64)
    return {
        DataKeys.POS: torch.from_numpy(pos),
        DataKeys.SEGMENT: torch.from_numpy(np.asarray(las.classification, dtype=np.int64)),
        DataKeys.INSTANCE: torch.from_numpy(np.where(tree > 0, tree, -1)),
    }


class FORinstance(PointCloudDataset):
    """The FOR-instance dataset.

    Each sample is one plot, returned as a dictionary:

    | Key        | Shape    | Dtype   | Meaning                                                |
    | ---------- | -------- | ------- | ------------------------------------------------------ |
    | `pos`      | $(N, 3)$ | float64 | UTM metres and elevation                               |
    | `segment`  | $(N,)$   | int64   | Class code 0-6, see `FORINSTANCE_CLASSES`              |
    | `instance` | $(N,)$   | int64   | Tree id, unique within the plot; $-1$ for no tree      |
    | `name`     | str      |         | Collection and plot, e.g. `CULS_plot_2_annotated`      |

    The `out_points` class holds trees outside the annotated plot, whose points carry no reliable label; benchmarks
    remove them before inference. Reading the LAS files needs the optional `laspy` package.

    Args:
        root: Dataset root. Plots are read from `<root>/FORinstance/raw/<collection>/<plot>.las`, the layout of
            the extracted `FORinstance_dataset.zip`.
        train: Whether to load the 21 development plots (the default) or the 11 test plots, as listed by the
            release's `data_split_metadata.csv`.
        transform: Callable applied to each loaded sample dict at `__getitem__` time.
        download: Download and extract the archive (1.6 GB, 5.8 GB extracted) when it is missing.
        show_progress: Show progress bars while downloading and extracting.

    !!! note
        `data_split_metadata.csv` also lists 50 `NIBIO2` plots that the archive does not hold; they are skipped.

    Example:
        ```{.python notest}
        from torch_pointcloud.datasets import FORinstance

        dataset = FORinstance(root="data", train=False, download=True)
        sample = dataset[0]
        ```
    """

    data_url: str = "https://zenodo.org/records/8287792/files/FORinstance_dataset.zip?download=1"
    data_md5: str = "01fd03ba4068648e21fb4591fbff34c4"

    def __init__(
        self,
        root: PathLike,
        *,
        train: bool = True,
        transform: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
        download: bool = False,
        show_progress: bool = True,
    ) -> None:
        super().__init__(root)
        self.train = train
        self.transform = transform
        self.show_progress = show_progress

        if download and not self.raw_files_exist():
            self.download()

        if not self.raw_files_exist():
            raise FileNotFoundError(
                f"Dataset not found: expected the extracted FOR-instance archive under {self.raw_dir!r}; pass "
                f"`download=True` or extract {self.data_url!r} there."
            )

        with open(self.split_file, newline="") as f:
            rows = list(csv.DictReader(f))
        split = "dev" if train else "test"
        paths = (Path(self.raw_dir, row["path"]) for row in rows if row["split"] == split)
        self.paths: Tuple[Path, ...] = tuple(sorted(path for path in paths if path.exists()))

    @property
    def split_file(self) -> str:
        """Path of the release's `data_split_metadata.csv`."""
        return Path(self.raw_dir, "data_split_metadata.csv").as_posix()

    @override
    def raw_files_exist(self) -> bool:
        return Path(self.split_file).exists()

    def download(self, force: bool = False) -> None:
        """Download the archive from Zenodo and extract it into `raw/`.

        Args:
            force: Download and extract the archive again, even when it is already in `raw/`.

        Raises:
            RuntimeError: If the downloaded archive does not match its MD5 checksum.
        """
        archive = Path(self.raw_dir, "FORinstance_dataset.zip")
        download_url(
            self.data_url,
            archive,
            description=f"Downloading {archive.name}",
            show_progress=self.show_progress,
            overwrite=True if force else "incomplete",
        )
        if not is_hash_valid(archive, self.data_md5):
            raise RuntimeError(f"{archive.as_posix()} does not match its MD5 checksum {self.data_md5}.")

        extract_zip(archive, self.raw_dir, show_progress=self.show_progress)

    def __len__(self) -> int:
        return len(self.paths)

    @override
    def __getitem__(self, index: int) -> Dict[str, Any]:
        path = self.paths[index]
        data: Dict[str, Any] = load_forinstance_data(path)
        data[DataKeys.NAME] = f"{path.parent.name}_{path.stem}"
        if self.transform is not None:
            data = self.transform(data)
        return data

    @override
    def extra_repr(self) -> str:
        return f"train={self.train}"
