import json
import shutil
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, TypedDict

import numpy as np
import torch
from typing_extensions import override

from torch_pointcloud.utils.data import DataKeys
from torch_pointcloud.utils.misc import parallel_map
from torch_pointcloud.utils.types import PathLike

from .pointcloud import PointCloudDataset

KITTI_CLASSES = ("Car", "Van", "Truck", "Pedestrian", "Person_sitting", "Cyclist", "Tram", "Misc")
KITTI_CLASS_TO_INDEX = {name: i for i, name in enumerate(KITTI_CLASSES)}


class KittiCalib(TypedDict):
    """KITTI camera/LiDAR calibration"""

    P2: np.ndarray
    R0_rect: np.ndarray
    Tr_velo_to_cam: np.ndarray


def load_kitti_calib(calib_file: PathLike) -> KittiCalib:
    r"""Parse a KITTI `calib/{id}.txt` into its projection / rectification / LiDAR-to-camera matrices.

    Only the three matrices used for box conversion and the FOV filter are kept: the left-color camera
    projection `P2` $(3, 4)$, the rectifying rotation `R0_rect` $(3, 3)$, and the LiDAR-to-camera transform
    `Tr_velo_to_cam` $(3, 4)$.

    Args:
        calib_file: Path to a KITTI calibration text file.

    Returns:
        A `KittiCalib` dict holding the `P2`, `R0_rect`, and `Tr_velo_to_cam` arrays.

    Example:
        ```python
        from torch_pointcloud.datasets.kitti import load_kitti_calib

        calib = load_kitti_calib("data/KITTI/raw/training/calib/000000.txt")
        calib["P2"].shape  # (3, 4)
        ```
    """
    values: Dict[str, np.ndarray] = {}
    for line in Path(calib_file).read_text().splitlines():
        if ":" not in line:
            continue

        key, raw = line.split(":", 1)
        if raw.strip():
            values[key.strip()] = np.array(raw.strip().split(" "), dtype=np.float32)

    return KittiCalib(
        P2=values["P2"].reshape(3, 4),
        R0_rect=values["R0_rect"].reshape(3, 3),
        Tr_velo_to_cam=values["Tr_velo_to_cam"].reshape(3, 4),
    )


def lidar_to_rect(points: np.ndarray, calib: KittiCalib) -> np.ndarray:
    r"""Transform LiDAR points into the rectified camera frame.

    Args:
        points: LiDAR XYZ coordinates.
        calib: Calibration matrices from `load_kitti_calib`.

    Returns:
        The points expressed in the rectified camera frame.

    Shape:
        - Input `points`: $(N, 3)$.
        - Output: $(N, 3)$.

    Example:
        ```python
        from torch_pointcloud.datasets.kitti import lidar_to_rect, load_kitti_calib

        calib = load_kitti_calib("data/KITTI/raw/training/calib/000000.txt")
        rect = lidar_to_rect(points, calib)
        ```
    """
    hom = np.hstack((points, np.ones((points.shape[0], 1), dtype=np.float32)))
    return hom @ (calib["Tr_velo_to_cam"].T @ calib["R0_rect"].T)


def rect_to_img(points: np.ndarray, calib: KittiCalib) -> Tuple[np.ndarray, np.ndarray]:
    r"""Project rectified-camera points onto the `P2` image plane.

    Args:
        points: Points in the rectified camera frame.
        calib: Calibration matrices from `load_kitti_calib`.

    Returns:
        A pair `(pixels, depth)` of the projected image coordinates and the per-point camera depth.

    Shape:
        - Input `points`: $(N, 3)$.
        - Output `pixels`: $(N, 2)$, `depth`: $(N,)$.

    Example:
        ```python
        from torch_pointcloud.datasets.kitti import lidar_to_rect, load_kitti_calib, rect_to_img

        calib = load_kitti_calib("data/KITTI/raw/training/calib/000000.txt")
        pixels, depth = rect_to_img(lidar_to_rect(points, calib), calib)
        ```
    """
    hom = np.hstack((points, np.ones((points.shape[0], 1), dtype=np.float32)))
    pts_2d_hom = hom @ calib["P2"].T
    pts_img = (pts_2d_hom[:, 0:2].T / points[:, 2]).T
    depth = pts_2d_hom[:, 2] - calib["P2"].T[3, 2]
    return pts_img, depth


def rect_to_lidar(points: np.ndarray, calib: KittiCalib) -> np.ndarray:
    r"""Transform rectified-camera points back into the LiDAR frame (inverse of `lidar_to_rect`).

    Args:
        points: Points in the rectified camera frame (e.g. label box centers).
        calib: Calibration matrices from `load_kitti_calib`.

    Returns:
        The points expressed in the LiDAR frame.

    Shape:
        - Input `points`: $(N, 3)$.
        - Output: $(N, 3)$.

    Example:
        ```python
        from torch_pointcloud.datasets.kitti import load_kitti_calib, rect_to_lidar

        calib = load_kitti_calib("data/KITTI/raw/training/calib/000000.txt")
        lidar = rect_to_lidar(centers, calib)
        ```
    """
    r0_ext = np.eye(4, dtype=np.float32)
    r0_ext[:3, :3] = calib["R0_rect"]
    v2c_ext = np.eye(4, dtype=np.float32)
    v2c_ext[:3, :4] = calib["Tr_velo_to_cam"]
    hom = np.hstack((points, np.ones((points.shape[0], 1), dtype=np.float32)))
    return (hom @ np.linalg.inv(r0_ext @ v2c_ext).T)[:, :3]


def fov_flag(points: np.ndarray, image_shape: Tuple[int, int], calib: KittiCalib) -> np.ndarray:
    r"""Boolean mask of LiDAR points that project into the front-camera image.

    A point is kept when its `P2` projection falls inside the image bounds and it lies in front of the
    camera (positive depth).

    Args:
        points: LiDAR points; only the first three columns (XYZ) are used.
        image_shape: Front-camera image `(height, width)` in pixels.
        calib: Calibration matrices from `load_kitti_calib`.

    Returns:
        A boolean mask selecting the in-FOV points.

    Shape:
        - Input `points`: $(N, C)$ with $C \ge 3$.
        - Output: $(N,)$.

    Example:
        ```python
        from torch_pointcloud.datasets.kitti import fov_flag, load_kitti_calib

        calib = load_kitti_calib("data/KITTI/raw/training/calib/000000.txt")
        points = points[fov_flag(points, (375, 1242), calib)]
        ```
    """
    pts_rect = lidar_to_rect(points[:, 0:3], calib)
    pts_img, depth = rect_to_img(pts_rect, calib)
    in_w = (pts_img[:, 0] >= 0) & (pts_img[:, 0] < image_shape[1])
    in_h = (pts_img[:, 1] >= 0) & (pts_img[:, 1] < image_shape[0])
    return in_w & in_h & (depth >= 0)


def load_kitti_boxes(label_file: PathLike, calib: KittiCalib) -> Dict[str, np.ndarray]:
    r"""Parse `label_2/{id}.txt` into raw LiDAR 7-DoF boxes and their per-box attributes.

    Each non-`DontCare` row is converted from its camera-frame `(h, w, l, x, y, z, ry)` annotation to a
    LiDAR box $(cx, cy, cz, dx, dy, dz, \text{heading})$ via `rect_to_lidar`; classes without a 3D box are
    skipped. The 2D box height is `bbox_bottom - bbox_top` in pixels.

    Args:
        label_file: Path to a KITTI `label_2` file. A missing file yields empty arrays (e.g. test split).
        calib: Calibration matrices from `load_kitti_calib`, used to convert box centers to the LiDAR frame.

    Returns:
        A dict of arrays keyed by cache file name: `boxes` $(K, 7)$, `labels` $(K,)$, `truncation` $(K,)$,
        `occlusion` $(K,)$, and `bbox_height` $(K,)$.

    Example:
        ```python
        from torch_pointcloud.datasets.kitti import load_kitti_boxes, load_kitti_calib

        calib = load_kitti_calib("data/KITTI/raw/training/calib/000000.txt")
        annotations = load_kitti_boxes("data/KITTI/raw/training/label_2/000000.txt", calib)
        annotations["boxes"].shape  # (K, 7)
        ```
    """
    rows: List[List[float]] = []
    labels: List[int] = []
    truncation: List[float] = []
    occlusion: List[int] = []
    bbox_height: List[float] = []
    label_path = Path(label_file)
    if label_path.exists():
        for line in label_path.read_text().splitlines():
            fields = line.split(" ")
            name = fields[0]
            if name not in KITTI_CLASS_TO_INDEX:
                continue

            height, width, length = (float(v) for v in fields[8:11])
            loc = np.array([[float(v) for v in fields[11:14]]], dtype=np.float32)
            rotation_y = float(fields[14])
            center = rect_to_lidar(loc, calib)[0]
            center[2] += height / 2
            rows.append([center[0], center[1], center[2], length, width, height, -(rotation_y + np.pi / 2)])
            labels.append(KITTI_CLASS_TO_INDEX[name])
            truncation.append(float(fields[1]))
            occlusion.append(int(fields[2]))
            bbox_height.append(float(fields[7]) - float(fields[5]))

    return {
        DataKeys.BOX: np.array(rows, dtype=np.float32).reshape(-1, 7),
        DataKeys.LABEL: np.array(labels, dtype=np.int64),
        DataKeys.TRUNCATION: np.array(truncation, dtype=np.float32),
        DataKeys.OCCLUSION: np.array(occlusion, dtype=np.int64),
        DataKeys.BBOX_HEIGHT: np.array(bbox_height, dtype=np.float32),
    }


def _read_image_shape(image_path: PathLike) -> Tuple[int, int]:
    r"""Read a PNG's `(height, width)` from its IHDR header (no image library needed).

    Args:
        image_path: Path to a PNG file.

    Returns:
        The image `(height, width)` in pixels.

    Example:
        ```python
        from torch_pointcloud.datasets.kitti import _read_image_shape

        height, width = _read_image_shape("data/KITTI/raw/training/image_2/000000.png")
        ```
    """
    with open(image_path, "rb") as f:
        f.seek(16)
        width, height = int.from_bytes(f.read(4), "big"), int.from_bytes(f.read(4), "big")
    return height, width


class KITTI(PointCloudDataset):
    r"""KITTI 3D object-detection dataset (LiDAR points + raw LiDAR-frame ground-truth boxes).

    The raw object split is read from `<root>/KITTI/raw/<split>/` and processed once into a per-frame
    `.npy` cache (`<root>/KITTI/processed_fov/<split>/` when `fov=True`, else `<root>/KITTI/processed/`),
    so each `__getitem__` is a flat array read. The whole split is processed; `split_file` only selects
    which cached frames to load. Frames are loaded lazily from the cache, so a full split does not need to
    fit in host memory.

    KITTI requires manual download (a license must be accepted), so `download=True` raises with the
    download URL rather than fetching anything.

    Args:
        root: Dataset root; raw data is read from `<root>/KITTI/raw/<split>/`.
        split: KITTI object split directory (`"training"` or `"testing"`).
        split_file: Optional text file of frame ids (one per line) selecting which cached frames to load;
            defaults to every processed frame.
        fov: Restrict points to the front-camera field of view (requires `image_2/`). Baked into the cache.
        transform: Callable applied to each sample dict (e.g. `RelabelBoxes` + the model's transform).
        download: Unsupported; raises a `RuntimeError` pointing at the manual download page when set.
        force_download: Unsupported; raises like `download`.
        force_process: Reprocess the raw split even if a cache already exists.
        show_progress: Show a progress bar while processing.
        num_workers: Worker processes for processing, or `None` for sequential processing.

    Example:
        Assuming the raw split is extracted under `data/KITTI/raw/training/`:

        ```python
        from torch_pointcloud.datasets import KITTI

        dataset = KITTI(root="data", split="training")
        sample = dataset[0]
        sample["pos"].shape  # (N, 3)
        sample["box"].shape  # (K, 7)
        ```
    """

    data_url = "https://www.cvlibs.net/datasets/kitti/eval_object.php"

    def __init__(
        self,
        root: PathLike,
        *,
        split: str = "training",
        split_file: Optional[PathLike] = None,
        fov: bool = True,
        transform: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
        download: bool = False,
        force_download: bool = False,
        force_process: bool = False,
        show_progress: bool = True,
        num_workers: Optional[int] = None,
    ) -> None:
        super().__init__(root)
        self.split = split
        self.split_file = split_file
        self.fov = fov
        self.transform = transform
        self.show_progress = show_progress
        self.num_workers = num_workers

        if download or force_download:
            self.download(force=force_download)

        self.process(force=force_process, num_workers=num_workers, show_progress=show_progress)
        self.load()

    @property
    @override
    def processed_dir(self) -> str:
        suffix = "processed_fov" if self.fov else "processed"
        return Path(self.data_dir, suffix).absolute().as_posix()

    @property
    def raw_split_dir(self) -> Path:
        return Path(self.raw_dir, self.split)

    @property
    def processed_split_dir(self) -> Path:
        return Path(self.processed_dir, self.split)

    @property
    def frame_ids(self) -> List[str]:
        return [frame for _, frame in self.frames]

    @override
    def raw_files_exist(self) -> bool:
        velodyne_dir = self.raw_split_dir / "velodyne"
        return velodyne_dir.is_dir() and any(velodyne_dir.glob("*.bin"))

    @override
    def processed_files_exist(self) -> bool:
        if (self.processed_split_dir / "meta.json").exists():
            return True

        frame_paths = self.processed_split_dir.glob(f"*/{DataKeys.POS}.npy")
        frame_dirs = sorted(p.parent for p in frame_paths if not p.parent.name.endswith(".tmp"))
        if not frame_dirs:
            return False

        file_names = (
            "pos.npy",
            "intensity.npy",
            "boxes.npy",
            "labels.npy",
            "truncation.npy",
            "occlusion.npy",
            "bbox_height.npy",
        )
        incomplete = [d.name for d in frame_dirs if not all((d / name).exists() for name in file_names)]
        missing: List[str] = []
        if self.raw_files_exist():
            expected = {p.stem for p in (self.raw_split_dir / "velodyne").glob("*.bin")}
            missing = sorted(expected - {d.name for d in frame_dirs})

        if incomplete or missing:
            raise RuntimeError(
                f"Incomplete processed cache at {self.processed_split_dir.as_posix()!r}: {len(missing)} missing "
                f"frame(s) {missing[:5]}, {len(incomplete)} incomplete frame(s) {incomplete[:5]}. "
                "Pass `force_process=True` to reprocess the raw data."
            )
        return True

    def download(self, force: bool = False) -> None:
        r"""Raise: KITTI must be downloaded manually after accepting its license.

        Args:
            force: Unused; present to mirror the other datasets' `download` signature.
        """
        raise RuntimeError(
            f"{self.__class__.__name__} does not support automatic download. Register and download the KITTI 3D object "
            f"detection benchmark from {self.data_url!r} and extract the velodyne / calib / label_2 (and "
            f"optionally image_2) folders under {self.raw_dir!r}."
        )

    def process(self, force: bool = False, num_workers: Optional[int] = None, show_progress: bool = True) -> None:
        r"""Convert every raw frame in the split into its `.npy` cache directory.

        Args:
            force: Reprocess even if a cache already exists.
            num_workers: Worker processes, or `None` for sequential processing.
            show_progress: Show a progress bar while processing.
        """
        if not force and self.processed_files_exist():
            return
        if not self.raw_files_exist():
            raise RuntimeError(
                f"Dataset not found at {self.raw_split_dir.as_posix()!r}. KITTI must be downloaded manually "
                f"from {self.data_url!r} and extracted under {self.raw_dir!r}."
            )

        frames = sorted(p.stem for p in (self.raw_split_dir / "velodyne").glob("*.bin"))
        self.processed_split_dir.mkdir(parents=True, exist_ok=True)
        for stale in self.processed_split_dir.glob("*.tmp"):
            if stale.is_dir():
                shutil.rmtree(stale)
            else:
                stale.unlink()

        parallel_map(
            self.process_frame,
            frames,
            num_workers=num_workers,
            total=len(frames),
            desc=f"Processing {self.split}",
            show_progress=show_progress,
        )

        meta_path = self.processed_split_dir / "meta.json"
        tmp_path = self.processed_split_dir / "meta.json.tmp"
        tmp_path.write_text(json.dumps({"format_version": 1, "fov": self.fov}))
        tmp_path.replace(meta_path)

    def process_frame(self, frame: str) -> None:
        r"""Read one raw frame (optionally FOV-filtered) and write its `.npy` cache.

        Args:
            frame: Frame id, e.g. `"000000"`.
        """
        points = np.fromfile(self.raw_split_dir / "velodyne" / f"{frame}.bin", dtype=np.float32).reshape(-1, 4)
        calib = load_kitti_calib(self.raw_split_dir / "calib" / f"{frame}.txt")

        if self.fov:
            image_path = self.raw_split_dir / "image_2" / f"{frame}.png"
            if image_path.exists():
                points = points[fov_flag(points, _read_image_shape(image_path), calib)]

        annotations = load_kitti_boxes(self.raw_split_dir / "label_2" / f"{frame}.txt", calib)
        frame_dir = self.processed_split_dir / frame
        tmp_dir = self.processed_split_dir / f"{frame}.tmp"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        np.save(tmp_dir / "pos.npy", points[:, :3])
        np.save(tmp_dir / "intensity.npy", points[:, 3:4])
        np.save(tmp_dir / "boxes.npy", annotations[DataKeys.BOX])
        np.save(tmp_dir / "labels.npy", annotations[DataKeys.LABEL])
        np.save(tmp_dir / "truncation.npy", annotations[DataKeys.TRUNCATION])
        np.save(tmp_dir / "occlusion.npy", annotations[DataKeys.OCCLUSION])
        np.save(tmp_dir / "bbox_height.npy", annotations[DataKeys.BBOX_HEIGHT])
        if frame_dir.exists():
            shutil.rmtree(frame_dir)
        tmp_dir.replace(frame_dir)

    def load(self) -> None:
        r"""Enumerate the cached frames to load, honoring `split_file` when given.

        Raises a `RuntimeError` listing the missing frame ids when `split_file` references frames
        absent from the processed cache.
        """
        if self.split_file is not None:
            frame_ids = [line.strip() for line in Path(self.split_file).read_text().splitlines() if line.strip()]
            missing = [
                frame for frame in frame_ids if not (self.processed_split_dir / frame / f"{DataKeys.POS}.npy").exists()
            ]
            if missing:
                raise RuntimeError(
                    f"{len(missing)} frame(s) listed in {Path(self.split_file).as_posix()!r} are missing from the "
                    f"processed cache at {self.processed_split_dir.as_posix()!r}: {missing[:10]}. "
                    "Pass `force_process=True` to reprocess the raw data."
                )
        else:
            frame_paths = self.processed_split_dir.glob(f"*/{DataKeys.POS}.npy")
            frame_ids = sorted(p.parent.name for p in frame_paths if not p.parent.name.endswith(".tmp"))

        self.frames: List[Tuple[Path, str]] = [(self.processed_split_dir / frame, frame) for frame in frame_ids]
        if not self.frames:
            raise RuntimeError(f"No processed frames found under {self.processed_split_dir.as_posix()!r}.")

    @override
    def __len__(self) -> int:
        return len(self.frames)

    @override
    def __getitem__(self, index: int) -> Dict[str, Any]:
        frame_dir, frame = self.frames[index]
        data: Dict[str, Any] = {
            DataKeys.POS: torch.from_numpy(np.load(frame_dir / "pos.npy")),
            DataKeys.INTENSITY: torch.from_numpy(np.load(frame_dir / "intensity.npy")),
            DataKeys.BOX: torch.from_numpy(np.load(frame_dir / "boxes.npy")),
            DataKeys.LABEL: torch.from_numpy(np.load(frame_dir / "labels.npy")),
            DataKeys.TRUNCATION: torch.from_numpy(np.load(frame_dir / "truncation.npy")),
            DataKeys.OCCLUSION: torch.from_numpy(np.load(frame_dir / "occlusion.npy")),
            DataKeys.BBOX_HEIGHT: torch.from_numpy(np.load(frame_dir / "bbox_height.npy")),
            DataKeys.FRAME: frame,
        }

        if self.transform is not None:
            data = self.transform(data)
        return data
