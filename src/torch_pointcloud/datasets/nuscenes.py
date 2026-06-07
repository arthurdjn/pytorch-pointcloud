import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from tqdm import tqdm
from typing_extensions import override

from torch_pointcloud.utils.data import DataKeys
from torch_pointcloud.utils.types import PathLike

from .pointcloud import PointCloudDataset

NUSCENES_DETECTION_CLASSES = (
    "car",
    "truck",
    "construction_vehicle",
    "bus",
    "trailer",
    "barrier",
    "motorcycle",
    "bicycle",
    "pedestrian",
    "traffic_cone",
)

# nuScenes category name -> detection class (the official 10-class mapping; unlisted -> ignored).
_CATEGORY_TO_DETECTION: Dict[str, str] = {
    "vehicle.car": "car",
    "vehicle.truck": "truck",
    "vehicle.construction": "construction_vehicle",
    "vehicle.bus.bendy": "bus",
    "vehicle.bus.rigid": "bus",
    "vehicle.trailer": "trailer",
    "movable_object.barrier": "barrier",
    "vehicle.motorcycle": "motorcycle",
    "vehicle.bicycle": "bicycle",
    "human.pedestrian.adult": "pedestrian",
    "human.pedestrian.child": "pedestrian",
    "human.pedestrian.construction_worker": "pedestrian",
    "human.pedestrian.police_officer": "pedestrian",
    "movable_object.trafficcone": "traffic_cone",
}


def _quaternion_to_rotation(quaternion: Sequence[float]) -> np.ndarray:
    """Rotation matrix $(3, 3)$ from a Hamilton `[w, x, y, z]` quaternion."""
    w, x, y, z = quaternion
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _pose_matrix(translation: Sequence[float], rotation: Sequence[float]) -> np.ndarray:
    """Homogeneous $(4, 4)$ transform from a `(translation, [w,x,y,z])` pose."""
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = _quaternion_to_rotation(rotation)
    matrix[:3, 3] = np.asarray(translation, dtype=np.float64)
    return matrix


def _remove_ego_points(points: np.ndarray, radius: float = 1.0) -> np.ndarray:
    return points[~((np.abs(points[:, 0]) < radius) & (np.abs(points[:, 1]) < radius))]


def _lidar_to_global(record: Dict[str, Any], ego_pose: Dict[str, Any], calib: Dict[str, Any]) -> np.ndarray:
    """LiDAR-to-global transform for `record`, composing its ego pose and sensor calibration."""
    ego = ego_pose[record["ego_pose_token"]]
    sensor = calib[record["calibrated_sensor_token"]]
    return _pose_matrix(ego["translation"], ego["rotation"]) @ _pose_matrix(sensor["translation"], sensor["rotation"])


def read_nuscenes_table(version_dir: PathLike, name: str) -> List[Dict[str, Any]]:
    """Read one nuScenes metadata table from `<version_dir>/<name>.json`."""
    path = Path(version_dir, f"{name}.json")
    if not path.exists():
        raise RuntimeError(f"nuScenes metadata table not found at {path!r}.")
    with open(path) as f:
        records: List[Dict[str, Any]] = json.load(f)
    return records


def load_nuscenes_sweeps(
    raw_dir: PathLike,
    record: Dict[str, Any],
    ego_pose: Dict[str, Any],
    calib: Dict[str, Any],
    sample_data: Dict[str, Any],
    max_sweeps: int,
) -> np.ndarray:
    r"""Aggregate up to `max_sweeps` LiDAR sweeps for a keyframe into its sensor frame.

    Each prior sweep is transformed into the keyframe frame and tagged with its time lag to the keyframe.

    Args:
        raw_dir: Dataset raw directory (sweep `filename`s are resolved against it).
        record: The keyframe `sample_data` record (a `LIDAR_TOP` key frame).
        ego_pose: `ego_pose` table indexed by token.
        calib: `calibrated_sensor` table indexed by token.
        sample_data: `sample_data` table indexed by token (used to walk the `prev` chain).
        max_sweeps: Total sweeps to aggregate (keyframe + prior sweeps).

    Returns:
        Packed points $(N, 5)$ of $(x, y, z, \text{intensity}, \Delta t)$.
    """
    ref_from_global = np.linalg.inv(_lidar_to_global(record, ego_pose, calib))
    keyframe_time = record["timestamp"] * 1e-6
    clouds: List[np.ndarray] = []
    current: Optional[Dict[str, Any]] = record
    while current is not None and len(clouds) < max_sweeps:
        raw = np.fromfile(Path(raw_dir, current["filename"]), dtype=np.float32).reshape(-1, 5)[:, :4]
        xyz = _remove_ego_points(raw)
        if current["token"] != record["token"]:
            transform = ref_from_global @ _lidar_to_global(current, ego_pose, calib)
            hom = np.hstack((xyz[:, :3], np.ones((xyz.shape[0], 1), dtype=np.float32)))
            xyz = np.hstack(((hom @ transform.T)[:, :3].astype(np.float32), xyz[:, 3:4]))

        dt = np.full((xyz.shape[0], 1), keyframe_time - current["timestamp"] * 1e-6, dtype=np.float32)
        clouds.append(np.hstack((xyz, dt)))
        prev_token = current["prev"]
        current = sample_data.get(prev_token) if prev_token else None

    return np.concatenate(clouds, axis=0)


def load_nuscenes_boxes(
    record: Dict[str, Any],
    ego_pose: Dict[str, Any],
    calib: Dict[str, Any],
    annotations: Dict[str, List[Dict[str, Any]]],
    class_to_idx: Dict[str, int],
) -> Tuple[np.ndarray, np.ndarray]:
    r"""Convert a keyframe's global-frame annotations to LiDAR 7-DoF boxes and labels.

    Args:
        record: The keyframe `sample_data` record.
        ego_pose: `ego_pose` table indexed by token.
        calib: `calibrated_sensor` table indexed by token.
        annotations: `sample_annotation` records grouped by `sample_token`, each carrying a
            `detection_name` (resolved detection class, or `None`).
        class_to_idx: Detection-class-name to label-index map; annotations whose class is absent are dropped.

    Returns:
        Boxes $(K, 7)$ of $(c_x, c_y, c_z, dx, dy, dz, \theta)$ and integer labels $(K,)$.
    """
    ref_from_global = np.linalg.inv(_lidar_to_global(record, ego_pose, calib))
    rows: List[List[float]] = []
    labels: List[int] = []
    for ann in annotations.get(record["sample_token"], []):
        name = ann["detection_name"]
        if name not in class_to_idx:
            continue

        center = (ref_from_global @ np.array([*ann["translation"], 1.0]))[:3]
        rotation = ref_from_global[:3, :3] @ _quaternion_to_rotation(ann["rotation"])
        yaw = float(np.arctan2(rotation[1, 0], rotation[0, 0]))
        width, length, height = ann["size"]
        rows.append([center[0], center[1], center[2], length, width, height, yaw])
        labels.append(class_to_idx[name])

    boxes = np.asarray(rows, dtype=np.float32).reshape(-1, 7)
    return boxes, np.asarray(labels, dtype=np.int64)


class NuScenesMini(PointCloudDataset):
    r"""nuScenes mini 3D object-detection dataset (LiDAR keyframes + LiDAR-frame ground-truth boxes).

    Reference: :arxiv: [nuScenes: A multimodal dataset for autonomous driving](https://arxiv.org/abs/1903.11027).

    Each LiDAR keyframe aggregates `max_sweeps` sweeps into the keyframe frame (ego-point removal + per-sweep
    ego/sensor transform, with a per-point time lag), and the global-frame annotations are converted to LiDAR
    7-DoF boxes $(c_x, c_y, c_z, dx, dy, dz, \theta)$ mapped onto the official 10-class detection set
    (`NUSCENES_DETECTION_CLASSES`). The raw split is processed once into per-keyframe numpy files under
    `<root>/NuScenesMini/processed/<version>_sweeps<max_sweeps>/`, then loaded from there (the same
    raw -> processed -> load flow as `S3DIS` / `ModelNet40`).

    Note:
        nuScenes cannot be downloaded automatically (it requires registration and accepting the dataset
        EULA). Download the mini split manually from https://www.nuscenes.org/nuscenes#download and extract
        it under `<root>/NuScenesMini/raw/` so that `raw/<version>/*.json`, `raw/samples/LIDAR_TOP/` and
        `raw/sweeps/LIDAR_TOP/` exist.

    Tip:
        The processed cache is keyed by `version` and `max_sweeps`, so different sweep counts coexist. After
        changing any other processing argument (e.g. `classes`), delete the processed directory or pass
        `force_process=True` to reprocess.

    Each sample is a dict:

    | Key         | Shape    | Dtype   | Description                                       |
    | ----------- | -------- | ------- | ------------------------------------------------- |
    | `pos`       | $(N, 3)$ | float32 | LiDAR XYZ (sweeps aggregated into keyframe frame) |
    | `intensity` | $(N, 1)$ | float32 | LiDAR reflectance                                 |
    | `timestamp` | $(N, 1)$ | float32 | Per-point time lag to the keyframe (seconds)      |
    | `box`       | $(K, 7)$ | float32 | GT boxes $(c_x, c_y, c_z, dx, dy, dz, \theta)$    |
    | `label`     | $(K,)$   | int64   | Detection class index into `classes`              |
    | `token`     | -        | str     | Source keyframe `sample_token`                    |

    Args:
        root: Dataset root; raw data is read from `<root>/NuScenesMini/raw/`.
        version: Metadata version directory (the mini split is `"v1.0-mini"`).
        max_sweeps: Total LiDAR sweeps aggregated per keyframe (keyframe + prior sweeps).
        classes: Foreground class names kept (order defines the label index).
        transform: Callable applied to each sample dict (e.g. the model's registered transform).
        download: If `True`, call `download` (which raises, since nuScenes needs a manual download).
        force_download: Forwarded to `download` as `force`.
        force_process: If `True`, reprocess the raw data even if a processed cache exists.
        show_progress: If `True`, show a progress bar while processing.

    Example:
        Assuming the raw mini split is extracted under `data/NuScenesMini/raw/`:

        ```python
        from torch_pointcloud.datasets import NuScenesMini

        dataset = NuScenesMini(root="data")
        sample = dataset[0]
        sample["pos"].shape   # torch.Size([N, 3])
        sample["box"].shape   # torch.Size([K, 7])
        ```
    """

    data_url = "https://www.nuscenes.org/nuscenes#download"

    def __init__(
        self,
        root: PathLike,
        *,
        version: str = "v1.0-mini",
        max_sweeps: int = 10,
        classes: Sequence[str] = NUSCENES_DETECTION_CLASSES,
        transform: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
        download: bool = False,
        force_download: bool = False,
        force_process: bool = False,
        show_progress: bool = True,
    ) -> None:
        super().__init__(root)
        self.version = version
        self.max_sweeps = max_sweeps
        self.classes = tuple(classes)
        self._class_to_idx = {name: i for i, name in enumerate(self.classes)}
        self.transform = transform
        self.show_progress = show_progress

        if download or force_download:
            self.download(force=force_download)

        self.process(force=force_process)
        self.load()

    @property
    @override
    def processed_dir(self) -> str:
        return Path(self.data_dir, "processed", f"{self.version}_sweeps{self.max_sweeps}").absolute().as_posix()

    def download(self, force: bool = False) -> None:
        raise RuntimeError(
            f"{self.__class__.__name__} cannot be downloaded automatically (registration and EULA acceptance are required). "
            f"Download the {self.version!r} split from {self.data_url!r} and extract it under {self.raw_dir!r} "
            "so that the raw layout (<version>/*.json, samples/LIDAR_TOP/, sweeps/LIDAR_TOP/) exists."
        )

    @override
    def raw_files_exist(self) -> bool:
        version_dir = Path(self.raw_dir, self.version)
        return (version_dir / "sample_data.json").exists() and Path(self.raw_dir, "samples", "LIDAR_TOP").is_dir()

    @override
    def processed_files_exist(self) -> bool:
        return Path(self.processed_dir, "keyframes.json").exists()

    def process(self, force: bool = False) -> None:
        if self.processed_files_exist() and not force:
            return
        if not self.raw_files_exist():
            raise RuntimeError(
                f"Dataset not found at {self.raw_dir!r}. "
                f"You can download the {self.version!r} split from {self.data_url!r} "
                f"and extract it under {self.raw_dir!r}."
            )

        version_dir = Path(self.raw_dir, self.version)
        ego_pose = {r["token"]: r for r in read_nuscenes_table(version_dir, "ego_pose")}
        calib = {r["token"]: r for r in read_nuscenes_table(version_dir, "calibrated_sensor")}
        category = {r["token"]: r["name"] for r in read_nuscenes_table(version_dir, "category")}
        instance = {r["token"]: category[r["category_token"]] for r in read_nuscenes_table(version_dir, "instance")}
        annotations: Dict[str, List[Dict[str, Any]]] = {}
        for ann in read_nuscenes_table(version_dir, "sample_annotation"):
            ann["detection_name"] = _CATEGORY_TO_DETECTION.get(instance[ann["instance_token"]])
            annotations.setdefault(ann["sample_token"], []).append(ann)

        sample_data = {r["token"]: r for r in read_nuscenes_table(version_dir, "sample_data")}
        keyframes = [r for r in sample_data.values() if r["is_key_frame"] and "LIDAR_TOP" in r["filename"]]

        tokens: List[str] = []
        for record in tqdm(keyframes, desc="Processing", disable=not self.show_progress):
            points = load_nuscenes_sweeps(self.raw_dir, record, ego_pose, calib, sample_data, self.max_sweeps)
            boxes, labels = load_nuscenes_boxes(record, ego_pose, calib, annotations, self._class_to_idx)

            token = record["sample_token"]
            keyframe_dir = Path(self.processed_dir, token)
            keyframe_dir.mkdir(parents=True, exist_ok=True)
            np.save(keyframe_dir / "pos.npy", points[:, :3])
            np.save(keyframe_dir / "intensity.npy", points[:, 3:4])
            np.save(keyframe_dir / "timestamp.npy", points[:, 4:5])
            np.save(keyframe_dir / "box.npy", boxes)
            np.save(keyframe_dir / "label.npy", labels)
            tokens.append(token)

        Path(self.processed_dir, "keyframes.json").write_text(json.dumps(tokens))

    def load(self) -> None:
        if not self.processed_files_exist():
            raise RuntimeError(f"Processed data not found at {self.processed_dir!r}. Run `process` first.")
        self.tokens: List[str] = json.loads(Path(self.processed_dir, "keyframes.json").read_text())

    @override
    def __len__(self) -> int:
        return len(self.tokens)

    @override
    def __getitem__(self, index: int) -> Dict[str, Any]:
        token = self.tokens[index]
        keyframe_dir = Path(self.processed_dir, token)
        data: Dict[str, Any] = {
            DataKeys.POS: torch.from_numpy(np.load(keyframe_dir / "pos.npy")),
            DataKeys.INTENSITY: torch.from_numpy(np.load(keyframe_dir / "intensity.npy")),
            DataKeys.TIMESTAMP: torch.from_numpy(np.load(keyframe_dir / "timestamp.npy")),
            DataKeys.BOX: torch.from_numpy(np.load(keyframe_dir / "box.npy")),
            DataKeys.LABEL: torch.from_numpy(np.load(keyframe_dir / "label.npy")),
            DataKeys.TOKEN: token,
        }

        if self.transform is not None:
            data = self.transform(data)
        return data
