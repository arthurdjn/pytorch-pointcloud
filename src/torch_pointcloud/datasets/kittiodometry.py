"""The KITTI odometry pairwise registration benchmark.

{{ paper("kitti-2012") }}
"""

import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple

import torch
from typing_extensions import override

from torch_pointcloud.utils.data import DataKeys
from torch_pointcloud.utils.io import load_pickle
from torch_pointcloud.utils.types import PathLike

from .pointcloud import PointCloudDataset
from .semantickitti import load_semantickitti_scan
from .utils import download_url, extract_zip

KITTIOdometrySplit = Literal["train", "val", "test"]

# Sequences of each split, the protocol of FCGF and PREDATOR.
KITTI_ODOMETRY_SEQUENCES: Dict[str, Tuple[str, ...]] = {
    "train": ("00", "01", "02", "03", "04", "05"),
    "val": ("06", "07"),
    "test": ("08", "09", "10"),
}


class KITTIOdometry(PointCloudDataset):
    r"""The KITTI odometry pairwise registration benchmark, as described in the paper
    :arxiv: [PREDATOR: Registration of 3D Point Clouds with Low Overlap](https://arxiv.org/abs/2011.13005)
    by Huang, Gojcic, Usvyatsov, Wieser, Schindler, on the velodyne scans of
    [KITTI odometry](https://www.cvlibs.net/datasets/kitti/eval_odometry.php) by Geiger, Lenz, Urtasun.

    Every sample pairs two scans of one sequence at least 10 m apart, with the rigid transform moving the source into
    the reference frame: the GPS/IMU pose refined by ICP, as published in the GeoTransformer pair lists. The splits
    are sequences `00-05` (1 358 pairs), `06-07` (180 pairs) and `08-10` (555 test pairs).

    | Key         | Shape    | Dtype   | Description                                |
    | ----------- | -------- | ------- | ------------------------------------------ |
    | `pos`       | $(N, 3)$ | float32 | Source scan points                         |
    | `ref_pos`   | $(M, 3)$ | float32 | Reference scan points                      |
    | `transform` | $(4, 4)$ | float32 | Transform of the source onto the reference |
    | `frame`     | str      |         | Source scan id, its file stem (`000123`)   |
    | `ref_frame` | str      |         | Reference scan id                          |
    | `sequence`  | str      |         | Sequence name                              |

    The scans are read as released, about 120k points each; the benchmark protocol averages them in 30 cm voxels,
    which `Voxelize(size=0.3, pos_reduce="mean", origin="min")` on `pos` and `ref_pos` does. The velodyne scans
    (80 GB) are downloaded by hand from the KITTI website into `raw/sequences/<sequence>/velodyne/`, the layout the
    SemanticKITTI release shares; `download=True` fetches the pair lists.

    Args:
        root: Root directory where the dataset is stored.
        split: `"train"`, `"val"` or `"test"`.
        transform: A callable that transforms the data when retrieved from the dataset.
        download: Whether to download the missing pair lists.
        force_download: Whether to force the download of the pair lists.
        force_process: Whether to force the processing of the pair lists.
        show_progress: Whether to show a progress bar during the download.

    Example:
        ```{.python notest}
        from torch_pointcloud.datasets import KITTIOdometry

        dataset = KITTIOdometry(root="data", split="test", download=True)
        sample = dataset[0]
        sample["pos"].shape, sample["ref_pos"].shape, sample["transform"].shape
        ```
    """

    data_url = "https://www.cvlibs.net/datasets/kitti/eval_odometry.php"
    metadata_url = "https://github.com/qinzheng93/GeoTransformer/archive/e7a135af4c318ff3b8d7f6c963df094d7e4ea540.zip"
    metadata_prefix = "GeoTransformer-e7a135af4c318ff3b8d7f6c963df094d7e4ea540/data/Kitti/metadata"

    def __init__(
        self,
        root: PathLike,
        *,
        split: KITTIOdometrySplit = "train",
        transform: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
        download: bool = False,
        force_download: bool = False,
        force_process: bool = False,
        show_progress: bool = True,
    ) -> None:
        super().__init__(root)
        if split not in KITTI_ODOMETRY_SEQUENCES:
            raise ValueError(f"Unknown split {split!r}. Expected 'train', 'val' or 'test'.")

        self.split = split
        self.transform = transform
        self.show_progress = show_progress

        if download or force_download:
            self.download(force=force_download)

        self.process(force=force_process, show_progress=show_progress)
        self.pairs = self.load_pairs()

    @property
    def sequences(self) -> Tuple[str, ...]:
        """Sequences of the split."""
        return KITTI_ODOMETRY_SEQUENCES[self.split]

    @property
    def sequences_dir(self) -> str:
        return Path(self.raw_dir, "sequences").as_posix()

    @property
    def metadata_dir(self) -> str:
        return Path(self.raw_dir, "metadata").as_posix()

    def scans_exist(self) -> bool:
        """Whether a sequence of the split has velodyne scans on disk; the pairs name the scans they need."""
        sequences_dir = Path(self.sequences_dir)
        return any(any((sequences_dir / seq / "velodyne").glob("*.bin")) for seq in self.sequences)

    @override
    def raw_files_exist(self) -> bool:
        return self.scans_exist() and Path(self.metadata_dir, f"{self.split}.pkl").exists()

    @property
    @override
    def cache_meta_paths(self) -> List[str]:
        return [Path(self.processed_dir, "pairs", f"{self.split}.meta.json").as_posix()]

    @override
    def processed_files_exist(self) -> bool:
        return Path(self.processed_dir, "pairs", f"{self.split}.json").exists()

    def download(self, force: bool = False) -> None:
        if self.raw_files_exist() and not force:
            return

        raw_dir = Path(self.raw_dir)
        raw_dir.mkdir(parents=True, exist_ok=True)
        if force or not Path(self.metadata_dir).is_dir():
            archive = Path(raw_dir, "metadata.zip")
            download_url(self.metadata_url, archive, show_progress=self.show_progress, overwrite=True)
            extract_zip(
                archive,
                self.metadata_dir,
                relative_to=self.metadata_prefix,
                show_progress=self.show_progress,
                prefix=self.metadata_prefix,
            )
            archive.unlink()

        if not self.scans_exist():
            raise RuntimeError(
                f"Velodyne scans of sequences {', '.join(self.sequences)} not found under {self.sequences_dir!r}. "
                f"Download the odometry velodyne laser data from {self.data_url!r} and extract its `sequences/` "
                f"tree under {self.raw_dir!r}."
            )

    def write_processed_files(self, num_workers: Optional[int] = None, show_progress: bool = True) -> bool:
        if not self.raw_files_exist():
            raise RuntimeError(
                f"Dataset not found at {self.raw_dir!r}. Pass `download=True` for the pair lists, and download the "
                f"odometry velodyne laser data from {self.data_url!r} and extract its `sequences/` tree under "
                f"{self.raw_dir!r}."
            )

        pairs = self.read_pairs_pickle()
        missing = sorted(
            {
                pair["sequence"]
                for pair in pairs
                if not self.scan_path(pair["sequence"], pair["ref_frame"]).exists()
                or not self.scan_path(pair["sequence"], pair["src_frame"]).exists()
            }
        )
        if missing:
            raise RuntimeError(
                f"Missing scans of sequence(s) {', '.join(missing)} under {self.sequences_dir!r}. Download the "
                f"odometry velodyne laser data from {self.data_url!r} and extract its `sequences/` tree under "
                f"{self.raw_dir!r}."
            )

        pairs_dir = Path(self.processed_dir, "pairs")
        pairs_dir.mkdir(parents=True, exist_ok=True)
        tmp_path = pairs_dir / f"{self.split}.json.tmp"
        tmp_path.write_text(json.dumps(pairs))
        tmp_path.replace(pairs_dir / f"{self.split}.json")
        return True

    def read_pairs_pickle(self) -> List[Dict[str, Any]]:
        """The pairs of the split, read from the GeoTransformer pair list."""
        records = load_pickle(Path(self.metadata_dir, f"{self.split}.pkl"))

        pairs = []
        for record in records:
            pairs.append(
                {
                    "sequence": f"{int(record['seq_id']):02d}",
                    "ref_frame": int(record["frame0"]),
                    "src_frame": int(record["frame1"]),
                    "transform": record["transform"].tolist(),
                }
            )
        return pairs

    def load_pairs(self) -> List[Dict[str, Any]]:
        return json.loads(Path(self.processed_dir, "pairs", f"{self.split}.json").read_text())

    def scan_path(self, sequence: str, frame: int) -> Path:
        """Path of the velodyne scan `frame` of `sequence`."""
        return Path(self.sequences_dir, sequence, "velodyne", f"{frame:06d}.bin")

    def __len__(self) -> int:
        return len(self.pairs)

    @override
    def __getitem__(self, index: int) -> Dict[str, Any]:
        pair = self.pairs[index]
        pos, _ = load_semantickitti_scan(self.scan_path(pair["sequence"], pair["src_frame"]))
        ref_pos, _ = load_semantickitti_scan(self.scan_path(pair["sequence"], pair["ref_frame"]))
        data: Dict[str, Any] = {
            DataKeys.POS: pos,
            DataKeys.REF_POS: ref_pos,
            DataKeys.TRANSFORM: torch.tensor(pair["transform"], dtype=torch.float32),
            DataKeys.FRAME: f"{pair['src_frame']:06d}",
            DataKeys.REF_FRAME: f"{pair['ref_frame']:06d}",
            DataKeys.SEQUENCE: pair["sequence"],
        }
        if self.transform is not None:
            data = self.transform(data)
        return data

    def extra_repr(self) -> str:
        return f"Split: {self.split}"
