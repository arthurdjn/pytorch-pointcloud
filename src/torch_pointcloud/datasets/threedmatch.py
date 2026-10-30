"""The 3DMatch and 3DLoMatch pairwise registration benchmarks.

{{ paper("1603.08182") }}
"""

import json
import shutil
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple

import numpy as np
import torch
from typing_extensions import override

from torch_pointcloud.utils.data import DataKeys
from torch_pointcloud.utils.io import ArrayUnpickler, load_pickle
from torch_pointcloud.utils.misc import parallel_map
from torch_pointcloud.utils.types import PathLike

from .pointcloud import PointCloudDataset
from .utils import download_url, extract_zip

ThreeDMatchSplit = Literal["train", "val", "test"]
ThreeDMatchBenchmark = Literal["3DMatch", "3DLoMatch"]


def load_threedmatch_fragment(path: PathLike) -> np.ndarray:
    r"""Read the points of a fragment file of the PREDATOR release (`cloud_bin_*.pth`).

    The files hold one NumPy array saved with the legacy `torch.save` format (a magic number, a protocol version,
    a system info dict, then the array). Only the array constructors are allowed when unpickling.

    Args:
        path: Path to a `cloud_bin_*.pth` file.

    Returns:
        The fragment points, shape $(N, 3)$, as `float32`.

    Example:
        ```{.python notest}
        from torch_pointcloud.datasets.threedmatch import load_threedmatch_fragment

        points = load_threedmatch_fragment("data/ThreeDMatch/raw/indoor/test/7-scenes-redkitchen/cloud_bin_0.pth")
        ```
    """
    with open(path, "rb") as f:
        for _ in range(3):
            ArrayUnpickler(f).load()
        points = ArrayUnpickler(f).load()
    if not isinstance(points, np.ndarray) or points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"Expected an (N, 3) array in {Path(path).as_posix()!r}, got {type(points).__name__}.")
    return np.ascontiguousarray(points, dtype=np.float32)


def _read_log_matrices(path: Path, rows: int) -> Dict[Tuple[int, int], np.ndarray]:
    r"""Parse a 3DMatch `gt.log` (`rows=4`) or `gt.info` (`rows=6`) file into `(ref, src) -> matrix`."""
    lines = [line for line in Path(path).read_text().splitlines() if line.strip()]
    matrices: Dict[Tuple[int, int], np.ndarray] = {}
    for start in range(0, len(lines), rows + 1):
        ref_frame, src_frame, _ = (int(token) for token in lines[start].split())
        block = [line.split() for line in lines[start + 1 : start + 1 + rows]]
        matrices[(ref_frame, src_frame)] = np.array(block, dtype=np.float64)
    return matrices


def _read_overlap_log(path: Path) -> Dict[Tuple[int, int], float]:
    overlaps: Dict[Tuple[int, int], float] = {}
    for line in Path(path).read_text().splitlines():
        if line.strip():
            ref_frame, src_frame, overlap = line.strip().split(",")
            overlaps[(int(ref_frame), int(src_frame))] = float(overlap)
    return overlaps


class ThreeDMatch(PointCloudDataset):
    r"""The 3DMatch and 3DLoMatch pairwise registration benchmarks, as described in the papers
    :arxiv: [3DMatch: Learning Local Geometric Descriptors from RGB-D Reconstructions](https://arxiv.org/abs/1603.08182)
    by Zeng, Song, Nießner, Fisher, Xiao, Funkhouser, and
    :arxiv: [PREDATOR: Registration of 3D Point Clouds with Low Overlap](https://arxiv.org/abs/2011.13005)
    by Huang, Gojcic, Usvyatsov, Wieser, Schindler.

    Every sample is a pair of indoor RGB-D fragments, each fused from 50 depth frames and voxel-downsampled at 2.5 cm
    by the PREDATOR release, with the rigid transform moving the source into the reference frame. The train and val
    pairs are the ones PREDATOR trains on (overlap above 30%); the test split holds the 8 benchmark scenes, paired
    as the 3DMatch benchmark (1 623 pairs, overlap above 30%) or the 3DLoMatch benchmark (1 781 pairs, overlap
    between 10% and 30%), with the ground truth and the information matrices of the official log files.

    | Key           | Shape      | Dtype   | Description                                                           |
    | ------------- | ---------- | ------- | --------------------------------------------------------------------- |
    | `pos`         | $(N, 3)$   | float32 | Source fragment points                                                |
    | `ref_pos`     | $(M, 3)$   | float32 | Reference fragment points                                             |
    | `transform`   | $(4, 4)$   | float32 | Transform of the source onto the reference                            |
    | `overlap`     | scalar     | float32 | Overlap ratio of the pair                                             |
    | `information` | $(6, 6)$   | float32 | Information matrix of the ground truth, test split only               |
    | `frame`       | str        |         | Source fragment id, the index of its `cloud_bin_<id>` file            |
    | `ref_frame`   | str        |         | Reference fragment id                                                 |
    | `scene`       | str        |         | Scene name                                                            |

    The fragments are downloaded from the PREDATOR release and the pair lists from the GeoTransformer repository;
    the code of both is MIT and the fragments keep the terms of the 3DMatch scenes they come from. The first use
    converts the fragments of the split, with every pair list reading them, to a `float32` cache under `processed/`.

    Args:
        root: Root directory where the dataset is stored or will be downloaded.
        split: `"train"`, `"val"` or `"test"`.
        benchmark: Pairing of the test split, `"3DMatch"` or `"3DLoMatch"`; the other splits take `"3DMatch"`.
        transform: A callable that transforms the data when retrieved from the dataset.
        download: Whether to download the missing raw files.
        force_download: Whether to force the download of the raw data.
        force_process: Whether to force the processing of the raw data.
        show_progress: Whether to show a progress bar during the download and the processing.
        num_workers: Worker processes for preprocessing, or `None` for sequential processing.

    Example:
        ```{.python notest}
        from torch_pointcloud.datasets import ThreeDMatch

        dataset = ThreeDMatch(root="data", split="test", benchmark="3DLoMatch", download=True)
        sample = dataset[0]
        sample["pos"].shape, sample["ref_pos"].shape, sample["transform"].shape
        ```
    """

    data_url = "https://share.phys.ethz.ch/~gseg/Predator/data.zip"
    metadata_url = "https://github.com/qinzheng93/GeoTransformer/archive/e7a135af4c318ff3b8d7f6c963df094d7e4ea540.zip"
    metadata_prefix = "GeoTransformer-e7a135af4c318ff3b8d7f6c963df094d7e4ea540/data/3DMatch/metadata"

    def __init__(
        self,
        root: PathLike,
        *,
        split: ThreeDMatchSplit = "train",
        benchmark: ThreeDMatchBenchmark = "3DMatch",
        transform: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
        download: bool = False,
        force_download: bool = False,
        force_process: bool = False,
        show_progress: bool = True,
        num_workers: Optional[int] = None,
    ) -> None:
        super().__init__(root)
        if split not in ("train", "val", "test"):
            raise ValueError(f"Unknown split {split!r}. Expected 'train', 'val' or 'test'.")
        if benchmark not in ("3DMatch", "3DLoMatch"):
            raise ValueError(f"Unknown benchmark {benchmark!r}. Expected '3DMatch' or '3DLoMatch'.")
        if split != "test" and benchmark != "3DMatch":
            raise ValueError(f"The {split!r} split has a single pairing; `benchmark` selects the pairs of 'test'.")

        self.split = split
        self.benchmark = benchmark
        self.transform = transform
        self.show_progress = show_progress

        if download or force_download:
            self.download(force=force_download)

        self.process(force=force_process, num_workers=num_workers, show_progress=show_progress)
        self.pairs = self.load_pairs()

    @property
    def pairs_name(self) -> str:
        """Name of the pair list: the split, or the benchmark for the test split."""
        return self.benchmark if self.split == "test" else self.split

    @property
    def pair_lists(self) -> Tuple[str, ...]:
        """Names of the pair lists reading the fragments of the split: both benchmarks, or the train and val pairs."""
        return ("3DMatch", "3DLoMatch") if self.split == "test" else ("train", "val")

    @property
    def fragments_split(self) -> str:
        """Fragment directory the pairs read from, `train` for the train and val splits and `test` otherwise."""
        return "test" if self.split == "test" else "train"

    @property
    def metadata_dir(self) -> str:
        return Path(self.raw_dir, "metadata").as_posix()

    @override
    def raw_files_exist(self) -> bool:
        fragments = Path(self.raw_dir, "indoor", self.fragments_split)
        if not fragments.is_dir() or not any(fragments.glob("*/cloud_bin_*.pth")):
            return False

        if self.split == "test":
            return all(Path(self.metadata_dir, "benchmarks", name).is_dir() for name in self.pair_lists)
        return all(Path(self.metadata_dir, f"{name}.pkl").exists() for name in self.pair_lists)

    @property
    @override
    def cache_meta_paths(self) -> List[str]:
        return [Path(self.processed_dir, f"{self.fragments_split}.meta.json").as_posix()]

    @override
    def processed_files_exist(self) -> bool:
        pairs_dir = Path(self.processed_dir, "pairs")
        pairs_exist = all(Path(pairs_dir, f"{name}.json").exists() for name in self.pair_lists)
        return pairs_exist and Path(self.processed_dir, self.fragments_split).is_dir()

    def download(self, force: bool = False) -> None:
        if self.raw_files_exist() and not force:
            return

        raw_dir = Path(self.raw_dir)
        raw_dir.mkdir(parents=True, exist_ok=True)

        if force or not Path(raw_dir, "indoor").is_dir():
            archive = Path(raw_dir, "data.zip")
            download_url(self.data_url, archive, show_progress=self.show_progress, overwrite="incomplete")
            extract_zip(archive, raw_dir, relative_to="data", show_progress=self.show_progress, prefix="data/indoor")
            archive.unlink()

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

    def write_processed_files(self, num_workers: Optional[int] = None, show_progress: bool = True) -> bool:
        if not self.raw_files_exist():
            raise RuntimeError(
                f"Dataset not found at {self.raw_dir!r}. Pass `download=True`, or download the fragments from "
                f"{self.data_url!r} and the pair lists from {self.metadata_url!r} and place them under "
                f"{self.raw_dir!r} as `indoor/` and `metadata/`."
            )

        self.process_fragments(num_workers=num_workers, show_progress=show_progress)
        pairs_dir = Path(self.processed_dir, "pairs")
        pairs_dir.mkdir(parents=True, exist_ok=True)
        for name in self.pair_lists:
            pairs = self.read_test_pairs(name) if self.split == "test" else self.read_pairs_pickle(name)
            tmp_path = pairs_dir / f"{name}.json.tmp"
            tmp_path.write_text(json.dumps(pairs))
            tmp_path.replace(pairs_dir / f"{name}.json")
        return True

    def process_fragments(self, num_workers: Optional[int] = None, show_progress: bool = True) -> None:
        """Convert every fragment of the split to a `float32` `.npy` file, replacing the previous cache."""
        src_dir = Path(self.raw_dir, "indoor", self.fragments_split)
        dst_dir = Path(self.processed_dir, self.fragments_split)
        paths = sorted(src_dir.glob("*/cloud_bin_*.pth"))
        records = [(p, dst_dir / p.parent.name / f"{p.stem}.npy") for p in paths]
        parallel_map(
            _convert_fragment,
            records,
            num_workers=num_workers,
            total=len(records),
            desc=f"Processing {self.fragments_split} fragments",
            show_progress=show_progress,
        )

    def read_pairs_pickle(self, name: str) -> List[Dict[str, Any]]:
        """The `"train"` or `"val"` pairs of the PREDATOR split, read from the GeoTransformer pair list."""
        records = load_pickle(Path(self.metadata_dir, f"{name}.pkl"))

        pairs = []
        for record in records:
            transform = np.eye(4)
            transform[:3, :3] = record["rotation"]
            transform[:3, 3] = record["translation"]
            pairs.append(
                {
                    "scene": record["scene_name"],
                    "ref_frame": int(record["frag_id0"]),
                    "src_frame": int(record["frag_id1"]),
                    "transform": transform.tolist(),
                    "overlap": float(record["overlap"]),
                }
            )
        return pairs

    def read_test_pairs(self, benchmark: str) -> List[Dict[str, Any]]:
        """The pairs of a test benchmark, with the ground truth of `gt.log`, `gt.info` and `gt_overlap.log`."""
        pairs = []
        scene_dirs = sorted(p for p in Path(self.metadata_dir, "benchmarks", benchmark).iterdir() if p.is_dir())
        for scene_dir in scene_dirs:
            scene = scene_dir.name
            transforms = _read_log_matrices(scene_dir / "gt.log", rows=4)
            informations = _read_log_matrices(scene_dir / "gt.info", rows=6)
            overlaps = _read_overlap_log(scene_dir / "gt_overlap.log")
            for (ref_frame, src_frame), transform in transforms.items():
                pairs.append(
                    {
                        "scene": scene,
                        "ref_frame": ref_frame,
                        "src_frame": src_frame,
                        "transform": transform.tolist(),
                        "overlap": overlaps[(ref_frame, src_frame)],
                        "information": informations[(ref_frame, src_frame)].tolist(),
                    }
                )
        return pairs

    def load_pairs(self) -> List[Dict[str, Any]]:
        return json.loads(Path(self.processed_dir, "pairs", f"{self.pairs_name}.json").read_text())

    def fragment_path(self, scene: str, frame: int) -> Path:
        """Path of the processed fragment `frame` of `scene`."""
        return Path(self.processed_dir, self.fragments_split, scene, f"cloud_bin_{frame}.npy")

    def __len__(self) -> int:
        return len(self.pairs)

    @override
    def __getitem__(self, index: int) -> Dict[str, Any]:
        pair = self.pairs[index]
        data: Dict[str, Any] = {
            DataKeys.POS: torch.from_numpy(np.load(self.fragment_path(pair["scene"], pair["src_frame"]))),
            DataKeys.REF_POS: torch.from_numpy(np.load(self.fragment_path(pair["scene"], pair["ref_frame"]))),
            DataKeys.TRANSFORM: torch.tensor(pair["transform"], dtype=torch.float32),
            DataKeys.OVERLAP: torch.tensor(pair["overlap"], dtype=torch.float32),
            DataKeys.FRAME: str(pair["src_frame"]),
            DataKeys.REF_FRAME: str(pair["ref_frame"]),
            DataKeys.SCENE: pair["scene"],
        }
        if "information" in pair:
            data[DataKeys.INFORMATION] = torch.tensor(pair["information"], dtype=torch.float32)
        if self.transform is not None:
            data = self.transform(data)
        return data

    def extra_repr(self) -> str:
        return f"Split: {self.split}" + (f"\nBenchmark: {self.benchmark}" if self.split == "test" else "")


def _convert_fragment(record: Tuple[Path, Path]) -> None:
    src, dst = record
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".tmp")
    with open(tmp, "wb") as f:
        np.save(f, load_threedmatch_fragment(src))
    shutil.move(tmp, dst)
