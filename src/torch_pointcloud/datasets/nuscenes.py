"""nuScenes 3D object detection datasets with sweep aggregation and annotation loading helpers.

{{ paper("1903.11027") }}
"""

import json
from functools import partial
from pathlib import Path
from typing import Any, Callable, Dict, FrozenSet, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import Tensor
from typing_extensions import override

from torch_pointcloud.utils.data import DataKeys
from torch_pointcloud.utils.misc import parallel_map
from torch_pointcloud.utils.types import PathLike

from .pointcloud import PointCloudDataset
from .utils import check_cache_meta

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

NUSCENES_SPLITS = ("train", "val", "test", "mini_train", "mini_val")

# The official `val` scenes of `v1.0-trainval` (nuscenes-devkit `nuscenes.utils.splits.val`).
# `train` is the rest.
NUSCENES_VAL_SCENES = (
    "scene-0003",
    "scene-0012",
    "scene-0013",
    "scene-0014",
    "scene-0015",
    "scene-0016",
    "scene-0017",
    "scene-0018",
    "scene-0035",
    "scene-0036",
    "scene-0038",
    "scene-0039",
    "scene-0092",
    "scene-0093",
    "scene-0094",
    "scene-0095",
    "scene-0096",
    "scene-0097",
    "scene-0098",
    "scene-0099",
    "scene-0100",
    "scene-0101",
    "scene-0102",
    "scene-0103",
    "scene-0104",
    "scene-0105",
    "scene-0106",
    "scene-0107",
    "scene-0108",
    "scene-0109",
    "scene-0110",
    "scene-0221",
    "scene-0268",
    "scene-0269",
    "scene-0270",
    "scene-0271",
    "scene-0272",
    "scene-0273",
    "scene-0274",
    "scene-0275",
    "scene-0276",
    "scene-0277",
    "scene-0278",
    "scene-0329",
    "scene-0330",
    "scene-0331",
    "scene-0332",
    "scene-0344",
    "scene-0345",
    "scene-0346",
    "scene-0519",
    "scene-0520",
    "scene-0521",
    "scene-0522",
    "scene-0523",
    "scene-0524",
    "scene-0552",
    "scene-0553",
    "scene-0554",
    "scene-0555",
    "scene-0556",
    "scene-0557",
    "scene-0558",
    "scene-0559",
    "scene-0560",
    "scene-0561",
    "scene-0562",
    "scene-0563",
    "scene-0564",
    "scene-0565",
    "scene-0625",
    "scene-0626",
    "scene-0627",
    "scene-0629",
    "scene-0630",
    "scene-0632",
    "scene-0633",
    "scene-0634",
    "scene-0635",
    "scene-0636",
    "scene-0637",
    "scene-0638",
    "scene-0770",
    "scene-0771",
    "scene-0775",
    "scene-0777",
    "scene-0778",
    "scene-0780",
    "scene-0781",
    "scene-0782",
    "scene-0783",
    "scene-0784",
    "scene-0794",
    "scene-0795",
    "scene-0796",
    "scene-0797",
    "scene-0798",
    "scene-0799",
    "scene-0800",
    "scene-0802",
    "scene-0904",
    "scene-0905",
    "scene-0906",
    "scene-0907",
    "scene-0908",
    "scene-0909",
    "scene-0910",
    "scene-0911",
    "scene-0912",
    "scene-0913",
    "scene-0914",
    "scene-0915",
    "scene-0916",
    "scene-0917",
    "scene-0919",
    "scene-0920",
    "scene-0921",
    "scene-0922",
    "scene-0923",
    "scene-0924",
    "scene-0925",
    "scene-0926",
    "scene-0927",
    "scene-0928",
    "scene-0929",
    "scene-0930",
    "scene-0931",
    "scene-0962",
    "scene-0963",
    "scene-0966",
    "scene-0967",
    "scene-0968",
    "scene-0969",
    "scene-0971",
    "scene-0972",
    "scene-1059",
    "scene-1060",
    "scene-1061",
    "scene-1062",
    "scene-1063",
    "scene-1064",
    "scene-1065",
    "scene-1066",
    "scene-1067",
    "scene-1068",
    "scene-1069",
    "scene-1070",
    "scene-1071",
    "scene-1072",
    "scene-1073",
)

# The official `mini_train` / `mini_val` scenes of `v1.0-mini`.
NUSCENES_MINI_TRAIN_SCENES = (
    "scene-0061",
    "scene-0553",
    "scene-0655",
    "scene-0757",
    "scene-0796",
    "scene-1077",
    "scene-1094",
    "scene-1100",
)
NUSCENES_MINI_VAL_SCENES = (
    "scene-0103",
    "scene-0916",
)

_SPLIT_VERSIONS = {
    "train": "v1.0-trainval",
    "val": "v1.0-trainval",
    "test": "v1.0-test",
    "mini_train": "v1.0-mini",
    "mini_val": "v1.0-mini",
}

# The official nuScenes attribute set, in `attribute.json` table order; annotation attribute ids index into it.
NUSCENES_ATTRIBUTES = (
    "vehicle.moving",
    "vehicle.stopped",
    "vehicle.parked",
    "cycle.with_rider",
    "cycle.without_rider",
    "pedestrian.sitting_lying_down",
    "pedestrian.standing",
    "pedestrian.moving",
)

# Attribute of a moving / stationary box per detection class; `barrier` and `traffic_cone` carry no attribute.
NUSCENES_MOVING_ATTRIBUTE = {
    "car": "vehicle.moving",
    "truck": "vehicle.moving",
    "construction_vehicle": "vehicle.moving",
    "bus": "vehicle.moving",
    "trailer": "vehicle.moving",
    "motorcycle": "cycle.with_rider",
    "bicycle": "cycle.with_rider",
    "pedestrian": "pedestrian.moving",
}
NUSCENES_STATIONARY_ATTRIBUTE = {
    "car": "vehicle.parked",
    "truck": "vehicle.parked",
    "construction_vehicle": "vehicle.parked",
    "bus": "vehicle.stopped",
    "trailer": "vehicle.parked",
    "motorcycle": "cycle.without_rider",
    "bicycle": "cycle.without_rider",
    "pedestrian": "pedestrian.standing",
}

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


def _annotation_velocity(
    ann: Dict[str, Any], ann_by_token: Dict[str, Dict[str, Any]], sample_timestamp: Dict[str, float]
) -> np.ndarray:
    r"""Global-frame velocity $(3,)$ of an annotation: the finite difference of its neighbors' translations.

    Uses the `prev` / `next` annotations of the same instance (falling back to the annotation itself when a
    neighbor is missing) and divides by the sample-timestamp delta in seconds; zeros when neither neighbor
    resolves.
    """
    first = ann_by_token.get(ann["prev"]) if ann["prev"] else None
    last = ann_by_token.get(ann["next"]) if ann["next"] else None
    if first is None and last is None:
        return np.zeros(3, dtype=np.float64)

    first = first if first is not None else ann
    last = last if last is not None else ann
    delta = np.asarray(last["translation"], dtype=np.float64) - np.asarray(first["translation"], dtype=np.float64)
    dt = sample_timestamp[last["sample_token"]] - sample_timestamp[first["sample_token"]]
    return delta / dt


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
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    r"""Convert a keyframe's global-frame annotations to LiDAR 7-DoF boxes with labels and box extras.

    Args:
        record: The keyframe `sample_data` record.
        ego_pose: `ego_pose` table indexed by token.
        calib: `calibrated_sensor` table indexed by token.
        annotations: `sample_annotation` records grouped by `sample_token`, each carrying a
            `detection_name` (resolved detection class, or `None`), a global-frame `velocity` $(3,)$ and
            an `attribute_id` (index into `NUSCENES_ATTRIBUTES`, $-1$ when unset).
        class_to_idx: Detection-class-name to label-index map; annotations whose class is absent are dropped.

    Returns:
        Boxes $(K, 7)$ of $(c_x, c_y, c_z, dx, dy, dz, \theta)$, integer labels $(K,)$, LiDAR-frame BEV
        velocities $(K, 2)$, LiDAR point counts $(K,)$ and attribute ids $(K,)$.
    """
    ref_from_global = np.linalg.inv(_lidar_to_global(record, ego_pose, calib))
    rows: List[List[float]] = []
    labels: List[int] = []
    velocities: List[List[float]] = []
    num_points: List[int] = []
    attributes: List[int] = []
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
        velocity = ref_from_global[:3, :3] @ ann["velocity"]
        velocities.append([velocity[0], velocity[1]])
        num_points.append(int(ann["num_lidar_pts"]))
        attributes.append(int(ann["attribute_id"]))

    boxes = np.asarray(rows, dtype=np.float32).reshape(-1, 7)
    return (
        boxes,
        np.asarray(labels, dtype=np.int64),
        np.asarray(velocities, dtype=np.float32).reshape(-1, 2),
        np.asarray(num_points, dtype=np.int64),
        np.asarray(attributes, dtype=np.int64),
    )


def _keyframe_item(
    record: Dict[str, Any],
    sample_data: Dict[str, Any],
    ego_pose: Dict[str, Any],
    calib: Dict[str, Any],
    annotations: List[Dict[str, Any]],
    max_sweeps: int,
) -> Dict[str, Any]:
    """Self-contained work item for one keyframe: its sweep chain, the poses the chain uses and its annotations."""
    chain: Dict[str, Any] = {}
    current: Optional[Dict[str, Any]] = record
    while current is not None and len(chain) < max_sweeps:
        chain[current["token"]] = current
        current = sample_data.get(current["prev"]) if current["prev"] else None

    return {
        "record": record,
        "sample_data": chain,
        "ego_pose": {r["ego_pose_token"]: ego_pose[r["ego_pose_token"]] for r in chain.values()},
        "calib": {r["calibrated_sensor_token"]: calib[r["calibrated_sensor_token"]] for r in chain.values()},
        "annotations": annotations,
    }


def _process_keyframe(
    item: Dict[str, Any], raw_dir: str, processed_dir: str, max_sweeps: int, class_to_idx: Dict[str, int]
) -> str:
    """Aggregate one keyframe's sweeps and boxes and write its cache directory; returns its sample token."""
    record = item["record"]
    points = load_nuscenes_sweeps(raw_dir, record, item["ego_pose"], item["calib"], item["sample_data"], max_sweeps)
    boxes, labels, velocity, num_points, attribute_ids = load_nuscenes_boxes(
        record,
        item["ego_pose"],
        item["calib"],
        {record["sample_token"]: item["annotations"]},
        class_to_idx,
    )

    token: str = record["sample_token"]
    keyframe_dir = Path(processed_dir, token)
    keyframe_dir.mkdir(parents=True, exist_ok=True)
    np.save(keyframe_dir / "pos.npy", points[:, :3])
    np.save(keyframe_dir / "intensity.npy", points[:, 3:4])
    np.save(keyframe_dir / "timestamp.npy", points[:, 4:5])
    np.save(keyframe_dir / "box.npy", boxes)
    np.save(keyframe_dir / "label.npy", labels)
    np.save(keyframe_dir / "velocity.npy", velocity)
    np.save(keyframe_dir / "num_points.npy", num_points)
    np.save(keyframe_dir / "attribute.npy", attribute_ids)
    return token


class NuScenes(PointCloudDataset):
    r"""nuScenes 3D object-detection dataset (LiDAR keyframes + LiDAR-frame ground-truth boxes).

    Reference: :arxiv: [nuScenes: A multimodal dataset for autonomous driving](https://arxiv.org/abs/1903.11027).

    Each LiDAR keyframe aggregates `max_sweeps` sweeps into the keyframe frame (ego-point removal + per-sweep
    ego/sensor transform, with a per-point time lag), and the global-frame annotations are converted to LiDAR
    7-DoF boxes $(c_x, c_y, c_z, dx, dy, dz, \theta)$ mapped onto the official 10-class detection set
    (`NUSCENES_DETECTION_CLASSES`). `split` selects the keyframes through the official scene lists and resolves
    the metadata version it reads (`train` / `val` read `v1.0-trainval`, `test` reads `v1.0-test`, `mini_train` /
    `mini_val` read `v1.0-mini`); `split=None` takes every keyframe of an explicit `version`. The split is processed
    once into per-keyframe numpy files under `<root>/NuScenes/processed/<version>_<split>_sweeps<max_sweeps>/`,
    then loaded from there (the same raw -> processed -> load flow as `S3DIS` / `ModelNet40`).

    Note:
        nuScenes cannot be downloaded automatically (it requires registration and accepting the dataset
        EULA). Download the metadata and every `*_blobs.tgz` archive of the version manually from
        https://www.nuscenes.org/nuscenes#download and extract them under `<root>/NuScenes/raw/` so that
        `raw/<version>/*.json`, `raw/samples/LIDAR_TOP/` and `raw/sweeps/LIDAR_TOP/` exist; each blob archive
        holds a different set of scenes, so a split is complete only once all of them are extracted.

    Tip:
        The processed cache is keyed by `version`, `split` and `max_sweeps`, so different sweep counts coexist.
        The cache also records the `classes` it was built with and refuses to load under a different class set;
        pass `force_process=True` to regenerate it.

    Each sample is a dict:

    | Key          | Shape    | Dtype   | Description                                              |
    | ------------ | -------- | ------- | -------------------------------------------------------- |
    | `pos`        | $(N, 3)$ | float32 | LiDAR XYZ (sweeps aggregated into keyframe frame)        |
    | `intensity`  | $(N, 1)$ | float32 | LiDAR reflectance                                        |
    | `timestamp`  | $(N, 1)$ | float32 | Per-point time lag to the keyframe (seconds)             |
    | `box`        | $(K, 7)$ | float32 | GT boxes $(c_x, c_y, c_z, dx, dy, dz, \theta)$           |
    | `label`      | $(K,)$   | int64   | Detection class index into `classes`                     |
    | `velocity`   | $(K, 2)$ | float32 | Per-box LiDAR-frame BEV velocity $(v_x, v_y)$            |
    | `num_points` | $(K,)$   | int64   | Per-box LiDAR point count (`num_lidar_pts`)              |
    | `attribute`  | $(K,)$   | int64   | Index into `NUSCENES_ATTRIBUTES` ($-1$ when unset)       |
    | `token`      | -        | str     | Source keyframe `sample_token`                           |

    Per-box velocities are the finite difference of the annotation translation across the `prev` / `next`
    annotations divided by their sample-timestamp delta, computed in the global frame, rotated into the
    keyframe LiDAR frame, and zero when neither neighbor exists. The `test` split ships no annotations, so its
    box tensors are empty.

    Args:
        root: Dataset root; raw data is read from `<root>/NuScenes/raw/`.
        split: One of `NUSCENES_SPLITS` (`"train"`, `"val"`, `"test"`, `"mini_train"`, `"mini_val"`), or `None`
            for every keyframe of `version`.
        version: Metadata version directory; resolved from `split` when omitted, required when `split` is `None`.
        max_sweeps: Total LiDAR sweeps aggregated per keyframe (keyframe + prior sweeps).
        classes: Foreground class names kept (order defines the label index).
        transform: Callable applied to each sample dict (e.g. the model's registered transform).
        download: If `True`, call `download` (which raises, since nuScenes needs a manual download).
        force_download: Forwarded to `download` as `force`.
        force_process: If `True`, reprocess the raw data even if a processed cache exists.
        show_progress: If `True`, show a progress bar while processing.
        num_workers: Worker processes for processing, or `None` for sequential processing.

    Example:
        Assuming the raw `v1.0-trainval` release is extracted under `data/NuScenes/raw/`:

        ```python
        from torch_pointcloud.datasets import NuScenes

        dataset = NuScenes(root="data", split="val", num_workers=8)
        len(dataset)          # 6019
        sample = dataset[0]
        sample["pos"].shape   # torch.Size([N, 3])
        sample["box"].shape   # torch.Size([K, 7])
        ```
    """

    data_url = "https://www.nuscenes.org/nuscenes#download"

    def __init__(
        self,
        root: PathLike,
        split: Optional[str] = "train",
        *,
        version: Optional[str] = None,
        max_sweeps: int = 10,
        classes: Sequence[str] = NUSCENES_DETECTION_CLASSES,
        transform: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
        download: bool = False,
        force_download: bool = False,
        force_process: bool = False,
        show_progress: bool = True,
        num_workers: Optional[int] = None,
    ) -> None:
        super().__init__(root)
        if split is not None and split not in NUSCENES_SPLITS:
            raise ValueError(f"Unknown nuScenes split {split!r}; expected one of {NUSCENES_SPLITS} or None.")
        if version is None:
            if split is None:
                raise ValueError("`version` is required when `split` is None.")
            version = _SPLIT_VERSIONS[split]

        self.split = split
        self.version = version
        self.max_sweeps = max_sweeps
        self.classes = tuple(classes)
        self._class_to_idx = {name: i for i, name in enumerate(self.classes)}
        self.transform = transform
        self.show_progress = show_progress
        self.num_workers = num_workers

        if download or force_download:
            self.download(force=force_download)

        self.process(force=force_process)
        self.load()

    @property
    @override
    def processed_dir(self) -> str:
        """Path to the processed cache directory, one per version, split and sweep count."""
        split = "" if self.split is None else f"_{self.split}"
        return Path(self.data_dir, "processed", f"{self.version}{split}_sweeps{self.max_sweeps}").absolute().as_posix()

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

    def _cache_meta(self) -> Dict[str, Any]:
        """Snapshot of the constructor parameters the processed cache content depends on."""
        return {"format_version": 2, "classes": list(self.classes)}

    @override
    def processed_files_exist(self) -> bool:
        if not Path(self.processed_dir, "keyframes.json").exists():
            return False
        check_cache_meta(Path(self.processed_dir, "meta.json"), self._cache_meta())
        return True

    def split_scenes(self) -> Optional[FrozenSet[str]]:
        """Scene names of `split`, or `None` when every scene of `version` belongs to it."""
        if self.split == "train":
            names = {record["name"] for record in read_nuscenes_table(Path(self.raw_dir, self.version), "scene")}
            return frozenset(names.difference(NUSCENES_VAL_SCENES))
        if self.split == "val":
            return frozenset(NUSCENES_VAL_SCENES)
        if self.split == "mini_train":
            return frozenset(NUSCENES_MINI_TRAIN_SCENES)
        if self.split == "mini_val":
            return frozenset(NUSCENES_MINI_VAL_SCENES)
        return None

    def _load_annotations(self, version_dir: Path) -> Dict[str, List[Dict[str, Any]]]:
        """`sample_annotation` records grouped by sample token, each with its detection class, velocity and attribute."""
        if not (version_dir / "sample_annotation.json").exists():
            return {}
        ann_records = read_nuscenes_table(version_dir, "sample_annotation")
        if not ann_records:
            return {}

        category = {r["token"]: r["name"] for r in read_nuscenes_table(version_dir, "category")}
        instance = {r["token"]: category[r["category_token"]] for r in read_nuscenes_table(version_dir, "instance")}
        sample_timestamp = {r["token"]: r["timestamp"] * 1e-6 for r in read_nuscenes_table(version_dir, "sample")}
        attribute = {r["token"]: r["name"] for r in read_nuscenes_table(version_dir, "attribute")}
        attribute_to_idx = {name: i for i, name in enumerate(NUSCENES_ATTRIBUTES)}
        ann_by_token = {ann["token"]: ann for ann in ann_records}

        annotations: Dict[str, List[Dict[str, Any]]] = {}
        for ann in ann_records:
            ann["detection_name"] = _CATEGORY_TO_DETECTION.get(instance[ann["instance_token"]])
            ann["velocity"] = _annotation_velocity(ann, ann_by_token, sample_timestamp)
            attr_tokens = ann["attribute_tokens"]
            ann["attribute_id"] = attribute_to_idx[attribute[attr_tokens[0]]] if attr_tokens else -1
            annotations.setdefault(ann["sample_token"], []).append(ann)
        return annotations

    def process(self, force: bool = False) -> None:
        if not force and self.processed_files_exist():
            return
        if not self.raw_files_exist():
            raise RuntimeError(
                f"Dataset not found at {self.raw_dir!r}. "
                f"You can download the {self.version!r} split from {self.data_url!r} "
                f"and extract it under {self.raw_dir!r}."
            )

        Path(self.processed_dir, "keyframes.json").unlink(missing_ok=True)

        version_dir = Path(self.raw_dir, self.version)
        ego_pose = {r["token"]: r for r in read_nuscenes_table(version_dir, "ego_pose")}
        calib = {r["token"]: r for r in read_nuscenes_table(version_dir, "calibrated_sensor")}
        annotations = self._load_annotations(version_dir)
        scene_of_sample = {r["token"]: r["scene_token"] for r in read_nuscenes_table(version_dir, "sample")}
        sample_data = {r["token"]: r for r in read_nuscenes_table(version_dir, "sample_data")}
        keyframes = [r for r in sample_data.values() if r["is_key_frame"] and "LIDAR_TOP" in r["filename"]]

        scenes = self.split_scenes()
        if scenes is not None:
            scene_tokens = {r["token"] for r in read_nuscenes_table(version_dir, "scene") if r["name"] in scenes}
            keyframes = [r for r in keyframes if scene_of_sample[r["sample_token"]] in scene_tokens]

        keyframes.sort(key=lambda r: (scene_of_sample[r["sample_token"]], r["timestamp"]))
        if not keyframes:
            raise RuntimeError(f"No LiDAR keyframes of split {self.split!r} found under {version_dir.as_posix()!r}.")

        missing = [r["filename"] for r in keyframes if not Path(self.raw_dir, r["filename"]).exists()]
        if missing:
            raise RuntimeError(
                f"{len(missing)} of {len(keyframes)} LiDAR keyframes are missing under {self.raw_dir!r} "
                f"(e.g. {missing[0]!r}); extract every `*_blobs.tgz` archive of {self.version!r}."
            )

        items = [
            _keyframe_item(r, sample_data, ego_pose, calib, annotations.get(r["sample_token"], []), self.max_sweeps)
            for r in keyframes
        ]

        Path(self.processed_dir).mkdir(parents=True, exist_ok=True)
        tokens = parallel_map(
            partial(
                _process_keyframe,
                raw_dir=self.raw_dir,
                processed_dir=self.processed_dir,
                max_sweeps=self.max_sweeps,
                class_to_idx=self._class_to_idx,
            ),
            items,
            num_workers=self.num_workers,
            desc=f"Processing {self.split or self.version}",
            show_progress=self.show_progress,
        )

        processed_dir = Path(self.processed_dir)
        meta_tmp_path = processed_dir / "meta.json.tmp"
        meta_tmp_path.write_text(json.dumps(self._cache_meta()))
        meta_tmp_path.replace(processed_dir / "meta.json")
        tokens_tmp_path = processed_dir / "keyframes.json.tmp"
        tokens_tmp_path.write_text(json.dumps(tokens))
        tokens_tmp_path.replace(processed_dir / "keyframes.json")

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
            DataKeys.VELOCITY: torch.from_numpy(np.load(keyframe_dir / "velocity.npy")),
            DataKeys.NUM_POINTS: torch.from_numpy(np.load(keyframe_dir / "num_points.npy")),
            DataKeys.ATTRIBUTE: torch.from_numpy(np.load(keyframe_dir / "attribute.npy")),
            DataKeys.TOKEN: token,
        }

        if self.transform is not None:
            data = self.transform(data)
        return data


class NuScenesMini(NuScenes):
    r"""nuScenes mini 3D object-detection dataset: every keyframe of the `v1.0-mini` release.

    A `NuScenes` preset for the ten-scene mini release (404 keyframes, no train / val split) that reads its own
    `<root>/NuScenesMini/raw/` directory, so the mini download can live next to the full dataset. Samples and
    the processed cache follow `NuScenes`; the cache lives under
    `<root>/NuScenesMini/processed/<version>_sweeps<max_sweeps>/`.

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
        num_workers: Worker processes for processing, or `None` for sequential processing.

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
        num_workers: Optional[int] = None,
    ) -> None:
        super().__init__(
            root,
            None,
            version=version,
            max_sweeps=max_sweeps,
            classes=classes,
            transform=transform,
            download=download,
            force_download=force_download,
            force_process=force_process,
            show_progress=show_progress,
            num_workers=num_workers,
        )


def velocity_attributes(labels: Tensor, velocity: Tensor, speed_threshold: float = 1.0) -> Tensor:
    r"""Attribute id of each detected box from its class and BEV speed (the nuScenes submission convention).

    A box faster than `speed_threshold` gets its class's moving attribute (`vehicle.moving`, `cycle.with_rider`,
    `pedestrian.moving`), a slower one its stationary attribute (`vehicle.parked` / `vehicle.stopped`,
    `cycle.without_rider`, `pedestrian.standing`); `barrier` and `traffic_cone` carry no attribute ($-1$).

    Args:
        labels: Detection class indices into `NUSCENES_DETECTION_CLASSES`, shape $(N,)$.
        velocity: BEV velocities in m/s, shape $(N, 2)$.
        speed_threshold: Speed above which a box counts as moving.

    Returns:
        Attribute ids into `NUSCENES_ATTRIBUTES`, shape $(N,)$, $-1$ where the class has none.

    Example:
        ```python
        from torch_pointcloud.datasets.nuscenes import velocity_attributes

        attributes = velocity_attributes(det["labels"], det["velocity"])
        ```
    """
    moving = torch.linalg.norm(velocity, dim=1) > speed_threshold
    attributes = torch.full_like(labels, -1)
    for index, name in enumerate(NUSCENES_DETECTION_CLASSES):
        if name not in NUSCENES_MOVING_ATTRIBUTE:
            continue
        mask = labels == index
        attributes[mask & moving] = NUSCENES_ATTRIBUTES.index(NUSCENES_MOVING_ATTRIBUTE[name])
        attributes[mask & ~moving] = NUSCENES_ATTRIBUTES.index(NUSCENES_STATIONARY_ATTRIBUTE[name])
    return attributes
