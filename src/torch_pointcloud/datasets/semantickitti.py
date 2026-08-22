"""
The SemanticKITTI dataset, as described in the paper
:arxiv: [SemanticKITTI: A Dataset for Semantic Scene Understanding of LiDAR Sequences](https://arxiv.org/abs/1904.01416).
"""

from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, Sequence, Union, get_args

import numpy as np
import torch
from torch import Tensor
from typing_extensions import override

from torch_pointcloud.utils.conversion import ensure_tuple
from torch_pointcloud.utils.data import DataKeys
from torch_pointcloud.utils.types import PathLike, ValueCollection

from .pointcloud import PointCloudDataset

# All 22 SemanticKITTI sequences. The split into train/val/test follows the official benchmark.
SemanticKittiSequence = Literal[
    "00",
    "01",
    "02",
    "03",
    "04",
    "05",
    "06",
    "07",
    "08",
    "09",
    "10",
    "11",
    "12",
    "13",
    "14",
    "15",
    "16",
    "17",
    "18",
    "19",
    "20",
    "21",
]
SEMANTIC_KITTI_SEQUENCES = get_args(SemanticKittiSequence)

SemanticKittiSplit = Literal["train", "val", "trainval", "test"]

# Sequence groups per split, following the official SemanticKITTI benchmark.
SEMANTIC_KITTI_SEQUENCES_PER_SPLIT: Dict[str, tuple[str, ...]] = {
    "train": ("00", "01", "02", "03", "04", "05", "06", "07", "09", "10"),
    "val": ("08",),
    "trainval": ("00", "01", "02", "03", "04", "05", "06", "07", "08", "09", "10"),
    "test": ("11", "12", "13", "14", "15", "16", "17", "18", "19", "20", "21"),
}

# Raw label name for every label id present in the SemanticKITTI `.label` files.
# Source: https://github.com/PRBonn/semantic-kitti-api/blob/master/config/semantic-kitti.yaml
SEMANTIC_KITTI_LABEL_NAMES: Dict[int, str] = {
    0: "unlabeled",
    1: "outlier",
    10: "car",
    11: "bicycle",
    13: "bus",
    15: "motorcycle",
    16: "on-rails",
    18: "truck",
    20: "other-vehicle",
    30: "person",
    31: "bicyclist",
    32: "motorcyclist",
    40: "road",
    44: "parking",
    48: "sidewalk",
    49: "other-ground",
    50: "building",
    51: "fence",
    52: "other-structure",
    60: "lane-marking",
    70: "vegetation",
    71: "trunk",
    72: "terrain",
    80: "pole",
    81: "traffic-sign",
    99: "other-object",
    252: "moving-car",
    253: "moving-bicyclist",
    254: "moving-person",
    255: "moving-motorcyclist",
    256: "moving-on-rails",
    257: "moving-bus",
    258: "moving-truck",
    259: "moving-other-vehicle",
}

# 19-class single-scan benchmark taxonomy: `SEMANTIC_KITTI_CLASSES[i]` names learning-map index i
# (the raw-id to contiguous-index convention used by the SemanticKITTI segmentation checkpoints).
SEMANTIC_KITTI_CLASSES = (
    "car",
    "bicycle",
    "motorcycle",
    "truck",
    "other-vehicle",
    "person",
    "bicyclist",
    "motorcyclist",
    "road",
    "parking",
    "sidewalk",
    "other-ground",
    "building",
    "fence",
    "vegetation",
    "trunk",
    "terrain",
    "pole",
    "traffic-sign",
)


def _check_sequences(sequences: Sequence[str]) -> None:
    for seq in sequences:
        if seq not in SEMANTIC_KITTI_SEQUENCES:
            available = ", ".join(SEMANTIC_KITTI_SEQUENCES)
            raise ValueError(f"Unknown sequence: {seq!r}. Must be one of {available}.")


def load_semantickitti_scan(file_path: PathLike) -> tuple[Tensor, Tensor]:
    r"""Load a single SemanticKITTI velodyne scan from a `.bin` file.

    Each `.bin` file contains a raw float32 array of shape $(N \cdot 4,)$
    interpreted as $(N, 4)$ points with columns `(x, y, z, intensity)`.

    Args:
        file_path: Path to the `.bin` file.

    Returns:
        A pair `(pos, intensity)` of tensors with shapes $(N, 3)$ and $(N, 1)$.

    Example:
        ```python
        from torch_pointcloud.datasets.semantickitti import load_semantickitti_scan

        pos, intensity = load_semantickitti_scan(
            "data/SemanticKITTI/raw/sequences/00/velodyne/000000.bin"
        )
        ```
    """
    raw = np.fromfile(file_path, dtype=np.float32).reshape(-1, 4)
    pos = torch.from_numpy(raw[:, :3].copy())
    intensity = torch.from_numpy(raw[:, 3:4].copy())
    return pos, intensity


def load_semantickitti_labels(file_path: PathLike) -> tuple[Tensor, Tensor]:
    """Load a single SemanticKITTI `.label` file.

    Each `.label` file contains a raw uint32 array of shape $(N,)$ where the
    lower 16 bits encode the semantic label and the upper 16 bits encode the
    instance id.

    Args:
        file_path: Path to the `.label` file.

    Returns:
        A pair `(segment, instance)` of int64 tensors with shape $(N,)$, where
        `segment` contains the raw semantic label ids (see
        `SEMANTIC_KITTI_LABEL_NAMES`) and `instance` contains the per-class
        instance ids.

    Example:
        ```python
        from torch_pointcloud.datasets.semantickitti import load_semantickitti_labels

        segment, instance = load_semantickitti_labels(
            "data/SemanticKITTI/raw/sequences/00/labels/000000.label"
        )
        ```
    """
    raw = np.fromfile(file_path, dtype=np.uint32)
    segment = torch.from_numpy((raw & 0xFFFF).astype(np.int64))
    instance = torch.from_numpy((raw >> 16).astype(np.int64))
    return segment, instance


class SemanticKITTI(PointCloudDataset):
    """The SemanticKITTI dataset, as described in the paper
    :arxiv: [SemanticKITTI: A Dataset for Semantic Scene Understanding of LiDAR Sequences](https://arxiv.org/abs/1904.01416).

    The dataset contains a sequence of LiDAR scans collected from a vehicle driving in
    several urban areas, with point-wise semantic and instance annotations. The 22
    sequences are split into train (`00-07, 09, 10`), val (`08`), and test (`11-21`).
    Test labels are not publicly available; only the velodyne scans are released.

    Note:
        The raw dataset must be downloaded manually from
        https://www.semantic-kitti.org/dataset.html (a license must be accepted).
        The expected layout under `<root>/SemanticKITTI/raw` is:

        ```text
        sequences/
            00/
                velodyne/{frame:06d}.bin
                labels/{frame:06d}.label
            01/
                ...
        ```

    Each sample is returned as a dict with the following keys:

    | Key         | Shape    | Dtype   | Description                                |
    | ----------- | -------- | ------- | ------------------------------------------ |
    | `pos`       | $(N, 3)$ | float32 | XYZ coordinates                            |
    | `intensity` | $(N, 1)$ | float32 | Reflected LiDAR intensity                  |
    | `segment`   | $(N,)$   | int64   | Raw per-point semantic id (when available) |
    | `instance`  | $(N,)$   | int64   | Per-point instance id (when available)     |
    | `sequence`  | -        | str     | Source sequence id (e.g. `"00"`)           |
    | `frame`     | -        | str     | Source frame id (e.g. `"000000"`)          |

    Note:
        Scans are loaded from disk on demand (lazy loading), so the dataset can be used
        with very large splits without exhausting host memory. As a consequence, the
        dataset has *no* `process` step: it just enumerates `.bin` files at construction
        time. Augmentation, voxelisation, feature normalisation, and label remapping
        are intentionally left out so they can be composed with
        `torch_pointcloud.transforms` and shared across models.

    Note:
        `segment` contains the **raw** SemanticKITTI label ids (see
        `SEMANTIC_KITTI_LABEL_NAMES`); no class merging or contiguous remapping is
        applied. Compose a downstream `torch_pointcloud.transforms.Relabel` to project
        them onto the (model-specific) class set you train against.

    Args:
        root: Root directory of the dataset. Raw data is expected under
            `<root>/SemanticKITTI/raw/sequences/<seq>/...`.
        split: One of `"train"`, `"val"`, `"trainval"`, or `"test"`. Selects the
            sequences used by the official benchmark. Ignored when `sequences` is set.
        sequences: Optional explicit list of sequences to use. Overrides `split`.
        transform: Callable applied to each sample dict at `__getitem__` time. Used
            for augmentation, voxelisation, label remapping, feature construction, etc.

    Example:
        Assuming you have downloaded the raw dataset and extracted it under
        `data/SemanticKITTI/raw/sequences/...`, you can load the validation split:

        ```python
        from torch_pointcloud.datasets import SemanticKITTI

        dataset = SemanticKITTI(root="data", split="val")
        sample = dataset[0]
        sample["pos"].shape         # torch.Size([N, 3])
        sample["intensity"].shape   # torch.Size([N, 1])
        sample["segment"].shape     # torch.Size([N]) - raw label ids in [0, 259]
        ```

        To map raw labels onto a 19-class training set, compose a `Relabel` transform yourself.
        The mapping below follows the evaluation protocol this library's SemanticKITTI
        pretrained weights were trained against. It is *not* the official
        `semantic-kitti-api` learning map: the official 19-class map merges bus (13) and
        on-rails (16) into other-vehicle and lane-marking (60) into road, whereas this one
        sends them to the ignore index.

        ```python
        import torch_pointcloud.transforms as T
        from torch_pointcloud.datasets import SemanticKITTI

        # `{raw_id: contiguous_index}`
        # (moving-* are merged with their static counterpart; bus/on-rails/
        # lane-marking/other-* fall through to `default=255`).
        labels = {
            10: 0, 252: 0,                  # car (+ moving-car)
            11: 1,                          # bicycle
            15: 2,                          # motorcycle
            18: 3, 258: 3,                  # truck (+ moving-truck)
            20: 4, 259: 4,                  # other-vehicle (+ moving-other-vehicle)
            30: 5, 254: 5,                  # person (+ moving-person)
            31: 6, 253: 6,                  # bicyclist (+ moving-bicyclist)
            32: 7, 255: 7,                  # motorcyclist (+ moving-motorcyclist)
            40: 8, 44: 9, 48: 10, 49: 11,   # road, parking, sidewalk, other-ground
            50: 12, 51: 13,                 # building, fence
            70: 14, 71: 15, 72: 16,         # vegetation, trunk, terrain
            80: 17, 81: 18,                 # pole, traffic-sign
        }
        dataset = SemanticKITTI(
            root="data",
            split="val",
            transform=T.Relabel(keys="segment", labels=labels, default=255),
        )
        ```
    """

    data_url: str = "https://www.semantic-kitti.org/dataset.html"

    # NOTE: We use a union of Literal types and a str subclass (not bare `str`)
    # so that IDEs provide autocompletion for known values while still accepting
    # arbitrary strings without type-checker errors.
    def __init__(
        self,
        root: PathLike,
        *,
        split: Union[SemanticKittiSplit, str] = "train",
        sequences: Optional[ValueCollection[Union[SemanticKittiSequence, str]]] = None,
        transform: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
    ) -> None:
        super().__init__(root)

        if split not in SEMANTIC_KITTI_SEQUENCES_PER_SPLIT:
            available = ", ".join(SEMANTIC_KITTI_SEQUENCES_PER_SPLIT)
            raise ValueError(f"Unknown split: {split!r}. Must be one of {available}.")

        self.split = split
        self.sequences = ensure_tuple(sequences if sequences is not None else SEMANTIC_KITTI_SEQUENCES_PER_SPLIT[split])
        self.transform = transform

        _check_sequences(self.sequences)

        self.load()

    @property
    def sequences_dir(self) -> str:
        """Path to the raw `sequences` directory."""
        return Path(self.raw_dir, "sequences").absolute().as_posix()

    @property
    @override
    def processed_dir(self) -> str:
        """Path to the processed cache directory, which aliases `raw_dir` since the scans are read as-is."""
        # SemanticKITTI is consumed directly from its raw form (no processing step),
        # so `processed_dir` aliases `raw_dir` rather than pointing to a non-existent
        # `<root>/SemanticKITTI/processed` location.
        return self.raw_dir

    @override
    def raw_files_exist(self) -> bool:
        sequences_dir = Path(self.sequences_dir)
        if not sequences_dir.is_dir():
            return False

        return all(any((sequences_dir / seq / "velodyne").glob("*.bin")) for seq in self.sequences)

    @override
    def processed_files_exist(self) -> bool:
        return self.raw_files_exist()

    def download(self, force: bool = False) -> None:
        """SemanticKITTI must be downloaded manually (a license must be accepted).

        Args:
            force: Unused; present to mirror the other datasets' `download` signature.
        Raises:
            RuntimeError: Always; automatic download is not supported.
        """
        raise RuntimeError(
            f"{self.__class__.__name__} does not support automatic download. Download the velodyne scans and "
            f"labels from {self.data_url!r} and extract the `sequences/` tree under {self.raw_dir!r}."
        )

    def load(self) -> None:
        """Enumerate the velodyne scans for the configured sequences.

        This populates `self.scans` with `(sequence, frame, bin_path, label_path)` tuples,
        where `label_path` is `None` when no `.label` file is present (e.g. test split).
        Raises a `RuntimeError` listing the missing sequences when only part of the requested
        sequences has velodyne scans on disk.
        """
        sequences_dir = Path(self.sequences_dir)
        if not sequences_dir.is_dir():
            raise RuntimeError(
                f"Dataset not found at {self.sequences_dir!r}. "
                "You can download the raw dataset from https://www.semantic-kitti.org/dataset.html "
                f"and extract it under {self.raw_dir!r}."
            )

        missing = [seq for seq in self.sequences if not any((sequences_dir / seq / "velodyne").glob("*.bin"))]
        if missing:
            raise RuntimeError(
                f"Missing sequence(s) {', '.join(missing)} under {self.sequences_dir!r}. "
                "Download the full split from https://www.semantic-kitti.org/dataset.html, "
                "or pass `sequences=(...)` restricted to the sequences on disk."
            )
        scans: List[tuple[str, str, Path, Optional[Path]]] = []
        for seq in self.sequences:
            velodyne_dir = sequences_dir / seq / "velodyne"
            labels_dir = sequences_dir / seq / "labels"
            for bin_path in sorted(velodyne_dir.glob("*.bin")):
                frame = bin_path.stem
                label_path = labels_dir / f"{frame}.label"
                scans.append((seq, frame, bin_path, label_path if label_path.exists() else None))

        self.scans = scans

    @override
    def __len__(self) -> int:
        return len(self.scans)

    @override
    def __getitem__(self, index: int) -> Dict[str, Any]:
        seq, frame, bin_path, label_path = self.scans[index]

        pos, intensity = load_semantickitti_scan(bin_path)
        data: Dict[str, Any] = {
            DataKeys.POS: pos,
            DataKeys.INTENSITY: intensity,
            DataKeys.SEQUENCE: seq,
            DataKeys.FRAME: frame,
        }

        if label_path is not None:
            segment, instance = load_semantickitti_labels(label_path)
            data[DataKeys.SEGMENT] = segment
            data[DataKeys.INSTANCE] = instance

        if self.transform is not None:
            data = self.transform(data)
        return data
