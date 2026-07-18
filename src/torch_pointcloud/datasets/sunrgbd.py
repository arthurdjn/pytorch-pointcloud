"""
The SUN RGB-D dataset for 3D object detection, as described in the paper
:arxiv: [SUN RGB-D: A RGB-D Scene Understanding Benchmark Suite](https://arxiv.org/abs/1505.05554).

"""

import io
import json
import math
import shutil
import zipfile
from functools import cached_property
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple

import numpy as np
import scipy.io as sio
import torch
from PIL import Image
from tqdm import tqdm
from typing_extensions import override

from torch_pointcloud.utils.data import DataKeys
from torch_pointcloud.utils.misc import parallel_map
from torch_pointcloud.utils.types import PathLike

from .pointcloud import PointCloudDataset
from .utils import download_url, is_hash_valid

SUNRGBD_BASE_URL = "https://rgbd.cs.princeton.edu/data"

SUNRGBD_RELEASE_ZIP = "SUNRGBD.zip"
SUNRGBD_TOOLBOX_ZIP = "SUNRGBDtoolbox.zip"
SUNRGBD_META3D_V2 = "SUNRGBDMeta3DBB_v2.mat"
SUNRGBD_META2D_V2 = "SUNRGBDMeta2DBB_v2.mat"
TOOLBOX_META_MEMBER = "SUNRGBDtoolbox/Metadata/SUNRGBDMeta.mat"
TOOLBOX_SPLIT_MEMBER = "SUNRGBDtoolbox/traintestSUNRGBD/allsplit.mat"

DEPTH_TRUNC = 8.0
DEPTH_SCALE = 1000.0

SUNRGBD_CLASSES = (
    "bed",
    "table",
    "sofa",
    "chair",
    "toilet",
    "desk",
    "dresser",
    "night_stand",
    "bookshelf",
    "bathtub",
)
SUNRGBD_CLASS_TO_IDX = {name: i for i, name in enumerate(SUNRGBD_CLASSES)}


def rebase_sequence(path: str) -> str:
    r"""Rebase an absolute SUN RGB-D scene path to a sequence id relative to `SUNRGBD/`.

    The split lists and metadata store absolute paths such as
    `/n/fs/sun3d/data/SUNRGBD/kv1/NYUdata/NYU0001`, sometimes with a leading double slash.
    Members inside `SUNRGBD.zip` are keyed as `SUNRGBD/kv1/NYUdata/NYU0001/...`, so the
    sequence id keeps everything after the last `/SUNRGBD/` segment.

    Args:
        path: Absolute scene path or a `SUNRGBD/...` sequence path.

    Returns:
        The sequence id, e.g. `kv1/NYUdata/NYU0001`.

    Examples:
        >>> rebase_sequence("/n/fs/sun3d/data/SUNRGBD/kv1/NYUdata/NYU0001")
        'kv1/NYUdata/NYU0001'
    """
    marker = "/SUNRGBD/"
    idx = path.rfind(marker)
    tail = path[idx + len(marker) :] if idx != -1 else path.split("SUNRGBD/", 1)[-1]
    while "//" in tail:
        tail = tail.replace("//", "/")
    return tail.rstrip("/")


def decode_depth(
    depth_png: np.ndarray,
    depth_trunc: float = DEPTH_TRUNC,
    depth_scale: float = DEPTH_SCALE,
) -> np.ndarray:
    r"""Decode a 16-bit SUN RGB-D depth image into metric depth.

    The raw PNG stores depth bit-shifted by 3. The value is recovered as
    `(d >> 3) | (d << 13)` in `uint16`, scaled to meters and truncated at $8$ m.

    Args:
        depth_png: Raw depth image of shape $(H, W)$ and dtype `uint16`.

    Returns:
        Metric depth of shape $(H, W)$ and dtype `float32`.

    Shape:
        - Input: $(H, W)$.
        - Output: $(H, W)$.
    """
    d = depth_png.astype(np.uint16)
    d16 = (d >> 3) | (d << 13)
    depth_m = d16.astype(np.float32) / depth_scale
    depth_m[depth_m > depth_trunc] = depth_trunc
    return depth_m


def unproject(depth_m: np.ndarray, k: np.ndarray, rtilt: np.ndarray) -> np.ndarray:
    r"""Unproject a metric depth map into the upright depth coordinate frame.

    Pixels are unprojected on a 1-indexed grid using the intrinsics `k`, reordered into the
    depth frame as $[x, z, -y]$, then rotated upright by `rtilt`. Invalid (zero-depth) pixels
    are kept in place here and dropped by the caller via the depth mask.

    Args:
        depth_m: Metric depth of shape $(H, W)$.
        k: Camera intrinsics of shape $(3, 3)$.
        rtilt: Upright rotation of shape $(3, 3)$.

    Returns:
        Unprojected points of shape $(H \cdot W, 3)$ in row-major order, dtype `float32`.

    Shape:
        - Input `depth_m`: $(H, W)$.
        - Output: $(H \cdot W, 3)$.
    """
    cx, cy = k[0, 2], k[1, 2]
    fx, fy = k[0, 0], k[1, 1]
    h, w = depth_m.shape
    xx, yy = np.meshgrid(np.arange(1, w + 1, dtype=np.float32), np.arange(1, h + 1, dtype=np.float32))
    x3 = (xx - cx) * depth_m / fx
    y3 = (yy - cy) * depth_m / fy
    z3 = depth_m
    pts = np.stack([x3.reshape(-1), z3.reshape(-1), -y3.reshape(-1)], axis=1)
    pts_upright = pts @ rtilt.T
    return pts_upright.astype(np.float32)


def _box_list(gt: Any) -> List[Any]:
    r"""Normalize the `groundtruth3DBB` field into a list of box structs.

    The field is a single struct when a scene has one box, a struct array when it has many,
    and empty or `None` when it has none.

    Args:
        gt: The raw `groundtruth3DBB` value from the metadata struct.

    Returns:
        A list of per-box structs (possibly empty).
    """
    if gt is None:
        return []
    if hasattr(gt, "_fieldnames"):
        return [gt]

    arr = np.atleast_1d(gt)
    return [b for b in arr if b is not None and hasattr(b, "_fieldnames")]


def parse_boxes(gt: Any, class_to_idx: Dict[str, int]) -> np.ndarray:
    r"""Parse SUN RGB-D 3D boxes into the packed detection encoding.

    Each kept box is $[cx, cy, cz, dx, dy, dz, \text{heading}, \text{sem\_cls}]$ where the
    centroid is the box center and the half-extents are the abs `coeffs` reordered to
    $[c_1, c_0, c_2]$. This matches votenet's `_bbox.npy` `[l, w, h]` mapping ($l = \text{coeffs}[1]$
    along the heading axis, $w = \text{coeffs}[0]$, $h = \text{coeffs}[2]$); storing the raw
    `coeffs` order swaps $dx \leftrightarrow dy$ and collapses oriented-box AP@0.5. The heading is
    $-\text{atan2}(o_1, o_0)$ from the orientation vector, and `sem_cls` is the index of an exact
    class-name match. Objects whose class name is not one of the 10 SUN RGB-D detection classes
    are dropped.

    Args:
        gt: The raw `groundtruth3DBB` value from the metadata struct.
        class_to_idx: Mapping from class name to integer index.

    Returns:
        Boxes of shape $(K, 8)$ and dtype `float32` (empty $(0, 8)$ when no box is kept).

    Shape:
        - Output: $(K, 8)$.
    """
    rows = []
    for b in _box_list(gt):
        name = str(b.classname)
        if name not in class_to_idx:
            continue
        cen = np.asarray(b.centroid, dtype=np.float32).reshape(-1)
        co = np.abs(np.asarray(b.coeffs, dtype=np.float32).reshape(-1))
        ori = np.asarray(b.orientation, dtype=np.float32).reshape(-1)
        heading = -math.atan2(float(ori[1]), float(ori[0]))
        rows.append(
            np.array(
                [cen[0], cen[1], cen[2], co[1], co[0], co[2], heading, float(class_to_idx[name])],
                dtype=np.float32,
            )
        )

    if not rows:
        return np.zeros((0, 8), dtype=np.float32)
    return np.stack(rows, axis=0)


class SunRGBD(PointCloudDataset):
    r"""The SUN RGB-D dataset for 3D object detection, as described in the paper
    :arxiv: [SUN RGB-D: A RGB-D Scene Understanding Benchmark Suite](https://arxiv.org/abs/1505.05554).

    SUN RGB-D provides 10335 RGB-D frames with amodal oriented 3D bounding boxes. This dataset
    reconstructs the upright point cloud from the raw depth frames following the votenet recipe
    (:github: [facebookresearch/votenet](https://github.com/facebookresearch/votenet)) and
    exposes per-scene clouds with their ground-truth boxes over the 10 detection classes.

    The dataset reads the depth and RGB frames directly from the 6.8 GB `SUNRGBD.zip` release
    and the metadata from `SUNRGBDtoolbox.zip`, without extracting either archive wholesale.
    Each scene is processed into a `<processed_dir>/<split>/<sequence_id>/` directory holding one `.npy` per
    attribute; `pos` is stored as float16 and `color` as uint8 to keep the cache compact.
    Cached boxes keep the on-disk $(K, 8)$ encoding (half extents, clockwise heading, trailing class
    column); `load` converts them to the emitted $(K, 7)$ format below, so existing caches stay valid.

    Args:
        root: Root directory where the dataset is stored or will be downloaded.
        split: The split to load, one of `train` or `val`.
        transform: A callable that transforms the data when retrieved from the dataset.
        download: Whether to download missing raw inputs if not present.
        force_download: Whether to force the download of the raw data.
        force_process: Whether to force the processing of the raw data.
        show_progress: Whether to show a progress bar during processing.
        num_workers: Worker processes for preprocessing, or `None` for sequential processing.

    Shape:
        - `pos`: $(N, 3)$ point coordinates in the upright depth frame.
        - `color`: $(N, 3)$ RGB values in $[0, 255]$ (uint8).
        - `box`: $(K, 7)$ boxes as $[cx, cy, cz, dx, dy, dz, \text{heading}]$ with full extents and a
          counter-clockwise heading about $+z$ from $+x$.
        - `class`: $(K,)$ class indices.

    Example:
        Assuming you have downloaded `SUNRGBD.zip` and `SUNRGBDtoolbox.zip` under
        `data/SunRGBD/raw`, you can load the dataset as follows:

        ```python
        from torch_pointcloud.datasets import SunRGBD

        dataset = SunRGBD(root="data", split="val")
        sample = dataset[0]
        sample["pos"].shape  # (N, 3)
        sample["box"].shape  # (K, 7)
        ```
    """

    data_url = SUNRGBD_BASE_URL
    md5: Dict[str, str] = {
        SUNRGBD_RELEASE_ZIP: "59a6919f60ecd6acb1c9a850fcb543b8",
        SUNRGBD_TOOLBOX_ZIP: "18d22e1761d36352f37232cba102f91f",
        SUNRGBD_META3D_V2: "6aa268dabdd0293ff18aeda78ea558a7",
        SUNRGBD_META2D_V2: "27397cfaf79277b70493940317817a51",
    }
    classes = SUNRGBD_CLASSES

    def __init__(
        self,
        root: PathLike,
        *,
        split: Literal["train", "val"] = "train",
        transform: Optional[Callable] = None,
        download: bool = False,
        force_download: bool = False,
        force_process: bool = False,
        show_progress: bool = True,
        num_workers: Optional[int] = None,
    ) -> None:
        super().__init__(root)
        if split not in ["train", "val"]:
            raise ValueError(f"Invalid split {split!r}, expected one of 'train' or 'val'.")

        self.split = split
        self.transform = transform
        self.show_progress = show_progress

        if download or force_download:
            self.download(force=force_download)

        self.process(force=force_process, num_workers=num_workers, show_progress=show_progress)
        self.load(show_progress=show_progress)

    @property
    def release_zip_path(self) -> str:
        return Path(self.raw_dir, SUNRGBD_RELEASE_ZIP).as_posix()

    @property
    def toolbox_zip_path(self) -> str:
        return Path(self.raw_dir, SUNRGBD_TOOLBOX_ZIP).as_posix()

    @cached_property
    def class_to_idx(self) -> Dict[str, int]:
        return dict(SUNRGBD_CLASS_TO_IDX)

    @override
    def raw_files_exist(self) -> bool:
        return Path(self.release_zip_path).is_file() and Path(self.toolbox_zip_path).is_file()

    @property
    def processed_files(self) -> List[Path]:
        scene_paths = Path(self.processed_dir, self.split).glob(f"*/{DataKeys.POS}.npy")
        return sorted(p.parent for p in scene_paths if not p.parent.name.endswith(".tmp"))

    @override
    def processed_files_exist(self) -> bool:
        split_dir = Path(self.processed_dir, self.split)
        if (split_dir / "meta.json").exists():
            return True

        scene_dirs = self.processed_files
        if not scene_dirs:
            return False

        file_names = tuple(f"{key}.npy" for key in (DataKeys.POS, DataKeys.COLOR, DataKeys.BOX, DataKeys.CLASS))
        incomplete = [d.name for d in scene_dirs if not all((d / name).exists() for name in file_names)]
        missing: List[str] = []
        if self.raw_files_exist():
            present = {d.name for d in scene_dirs}
            missing = [sid for sid in self.read_split() if sid.replace("/", "_") not in present]
            if missing:
                # Only sequences present in the toolbox metadata are processed; exonerate the rest.
                meta = self.read_meta()
                missing = [sid for sid in missing if sid in meta]

        if incomplete or missing:
            raise RuntimeError(
                f"Incomplete processed cache at {split_dir.as_posix()!r}: {len(missing)} missing scene(s) "
                f"{missing[:5]}, {len(incomplete)} incomplete scene(s) {incomplete[:5]}. "
                "Pass `force_process=True` to reprocess the raw data."
            )
        return True

    def download(self, force: bool = False) -> None:
        if self.raw_files_exist() and not force:
            return

        Path(self.raw_dir).mkdir(parents=True, exist_ok=True)

        for file_name in (SUNRGBD_TOOLBOX_ZIP, SUNRGBD_META3D_V2, SUNRGBD_META2D_V2, SUNRGBD_RELEASE_ZIP):
            out_path = Path(self.raw_dir, file_name)

            if (
                not out_path.exists()
                or not is_hash_valid(out_path, expected_hash=self.md5[file_name], hash_type="md5")
                or force
            ):
                download_url(
                    f"{self.data_url}/{file_name}",
                    out_path,
                    description=f"Downloading {file_name!r}",
                    show_progress=self.show_progress,
                    overwrite=True if force else "incomplete",
                )

            if not is_hash_valid(out_path, expected_hash=self.md5[file_name], hash_type="md5"):
                raise RuntimeError(
                    f"File corrupted: MD5 hash mismatch for {out_path!r}. "
                    "HINT: Make sure the file was downloaded correctly."
                )

    def read_split(self) -> List[str]:
        with zipfile.ZipFile(self.toolbox_zip_path) as z:
            split = sio.loadmat(io.BytesIO(z.read(TOOLBOX_SPLIT_MEMBER)), struct_as_record=False, squeeze_me=True)

        if self.split == "train":
            return [rebase_sequence(str(p)) for p in split["alltrain"]]
        return [rebase_sequence(str(p)) for p in split["alltest"]]

    def read_meta(self) -> Dict[str, Any]:
        with zipfile.ZipFile(self.toolbox_zip_path) as z:
            meta = sio.loadmat(io.BytesIO(z.read(TOOLBOX_META_MEMBER)), struct_as_record=False, squeeze_me=True)
        return {rebase_sequence(str(e.sequenceName)): e for e in meta["SUNRGBDMeta"]}

    def process(self, force: bool = False, num_workers: Optional[int] = None, show_progress: bool = True) -> None:
        if not force and self.processed_files_exist():
            return
        if not self.raw_files_exist():
            raise RuntimeError(
                f"Dataset not found at {self.raw_dir!r}. "
                f"You can download the raw dataset from {self.data_url!r}, "
                f"and place {SUNRGBD_RELEASE_ZIP!r} and {SUNRGBD_TOOLBOX_ZIP!r} under {self.raw_dir!r}."
            )

        split_dir = Path(self.processed_dir, self.split)
        split_dir.mkdir(parents=True, exist_ok=True)
        for stale in split_dir.glob("*.tmp"):
            if stale.is_dir():
                shutil.rmtree(stale)
            else:
                stale.unlink()

        sequences = self.read_split()
        meta = self.read_meta()
        records = [(sid, meta[sid], split_dir.as_posix()) for sid in sequences if sid in meta]

        parallel_map(
            self.process_scene,
            records,
            num_workers=num_workers,
            total=len(records),
            desc=f"Processing {self.split}",
            show_progress=show_progress,
        )

        meta_path = split_dir / "meta.json"
        tmp_path = split_dir / "meta.json.tmp"
        tmp_path.write_text(json.dumps({"format_version": 1}))
        tmp_path.replace(meta_path)

    def process_scene(self, args: Tuple[str, Any, str]) -> None:
        sequence_id, entry, split_dir = args
        coords, colors = self.read_scene_cloud(entry)
        boxes = parse_boxes(entry.groundtruth3DBB, self.class_to_idx)
        classes = np.asarray(boxes[:, 7], dtype=np.int64)

        scene_dir = Path(split_dir, sequence_id.replace("/", "_"))
        tmp_dir = scene_dir.with_name(f"{scene_dir.name}.tmp")
        tmp_dir.mkdir(parents=True, exist_ok=True)
        np.save(Path(tmp_dir, f"{DataKeys.POS}.npy"), coords.astype(np.float16))
        np.save(Path(tmp_dir, f"{DataKeys.COLOR}.npy"), colors)
        np.save(Path(tmp_dir, f"{DataKeys.BOX}.npy"), boxes)
        np.save(Path(tmp_dir, f"{DataKeys.CLASS}.npy"), classes)
        if scene_dir.exists():
            shutil.rmtree(scene_dir)
        tmp_dir.replace(scene_dir)

    def read_scene_cloud(self, entry: Any) -> Tuple[np.ndarray, np.ndarray]:
        depth_member = f"SUNRGBD/{rebase_sequence(str(entry.depthpath))}"
        rgb_member = f"SUNRGBD/{rebase_sequence(str(entry.rgbpath))}"
        with zipfile.ZipFile(self.release_zip_path) as z:
            depth_png = np.array(Image.open(io.BytesIO(z.read(depth_member))))
            rgb = np.array(Image.open(io.BytesIO(z.read(rgb_member))).convert("RGB"))

        k = np.asarray(entry.K, dtype=np.float32)
        rtilt = np.asarray(entry.Rtilt, dtype=np.float32)
        depth_m = decode_depth(depth_png)
        valid = depth_m.reshape(-1) != 0
        points: np.ndarray = unproject(depth_m, k, rtilt)[valid]
        colors: np.ndarray = rgb.reshape(-1, 3)[valid]
        return points, colors

    def load(self, show_progress: bool = True) -> None:
        self.data: List[Dict[str, Any]] = []
        for scene_dir in tqdm(
            self.processed_files,
            total=len(self.processed_files),
            desc="Loading",
            disable=not show_progress,
        ):
            # The cache stores half extents, a clockwise heading, and a trailing class column;
            # convert to the emitted format at load time so existing caches stay valid.
            boxes = torch.from_numpy(np.load(Path(scene_dir, f"{DataKeys.BOX}.npy")))
            scene: Dict[str, Any] = {
                DataKeys.POS: torch.from_numpy(np.load(Path(scene_dir, f"{DataKeys.POS}.npy"))).float(),
                DataKeys.COLOR: torch.from_numpy(np.load(Path(scene_dir, f"{DataKeys.COLOR}.npy"))),
                DataKeys.BOX: torch.cat([boxes[:, :3], boxes[:, 3:6] * 2, -boxes[:, 6:7]], dim=1),
                DataKeys.CLASS: torch.from_numpy(np.load(Path(scene_dir, f"{DataKeys.CLASS}.npy"))),
            }
            self.data.append(scene)

    @override
    def __len__(self) -> int:
        return len(self.data)

    @override
    def __getitem__(self, index: int) -> Dict[str, Any]:
        data = self.data[index]
        if self.transform is not None:
            data = self.transform(data)
        return data
