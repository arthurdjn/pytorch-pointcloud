import math
from collections import defaultdict
from functools import cached_property
from pathlib import Path
from typing import Any, Callable, Literal, Optional, Sequence, TypedDict, Union, get_args
from urllib.parse import urljoin

import h5py
import numpy as np
import torch
from torch import Tensor
from tqdm import tqdm
from typing_extensions import override

from torch_pointcloud.utils.conversion import ensure_tuple
from torch_pointcloud.utils.data import DataKeys
from torch_pointcloud.utils.geometry import rodrigues_rotation_matrix
from torch_pointcloud.utils.misc import parallel_map
from torch_pointcloud.utils.types import PathLike, ValueCollection

from .pointcloud import PointCloudDataset
from .utils import download_url, extract_zip, is_hash_valid

# Areas available in the S3DIS dataset.
S3DISArea = Literal["Area_1", "Area_2", "Area_3", "Area_4", "Area_5", "Area_6"]
S3DIS_AREAS = get_args(S3DISArea)

# Classes used by the S3DIS dataset. Order matters: the same order used in SOTA reference implementations.
# Note that the clutter class (unknown objects) is the last one.
S3DISClass = Literal[
    "ceiling",
    "floor",
    "wall",
    "beam",
    "column",
    "window",
    "door",
    "chair",
    "table",
    "bookcase",
    "sofa",
    "board",
    "clutter",
]
S3DIS_CLASSES = get_args(S3DISClass)

# Create a mapping between classes and their indices.
S3DIS_CLASS_TO_IDX = {cls: idx for idx, cls in enumerate(S3DIS_CLASSES)}
# In S3DIS original convention, unknown classes are grouped into the 'clutter' class.
S3DIS_UNK_CLS = "clutter"
S3DIS_UNK_IDX = S3DIS_CLASS_TO_IDX[S3DIS_UNK_CLS]


class S3DISRoomData(TypedDict, total=False):
    pos: Tensor
    color: Tensor
    segment: Tensor
    instance: Tensor


def _check_areas(areas: Sequence[str]) -> None:
    r"""Verify that the provided areas are valid, and raises a ValueError if not."""
    for area in areas:
        if area not in S3DIS_AREAS:
            available_areas = ", ".join(S3DIS_AREAS)
            raise ValueError(f"Unknown area: {area!r}. Must be one of {available_areas}.")


def _check_classes(classes: Sequence[str]) -> None:
    r"""Verify that the provided classes are valid, and raises a ValueError if not."""
    for cls_name in classes:
        if cls_name not in S3DIS_CLASSES:
            available_classes = ", ".join(S3DIS_CLASSES)
            raise ValueError(f"Unknown class: {cls_name!r}. Must be one of {available_classes}.")


def load_s3dis_room(room_dir: PathLike, alignment_angle: float | None = None) -> S3DISRoomData:
    """Load the full S3DIS room from a given directory and (optionally) apply an alignment angle.

    Args:
        room_dir: Path to the room directory.
        alignment_angle: Alignment angle to apply to the room. Defaults to None.

    Returns:
        A dictionary containing the room data.

    Example:
        Assuming you have downloaded the raw dataset and extracted it under `data/S3DIS/raw`,
        you can load the alignment angles for the Area_1 as follows:

        ```python
        from torch_pointcloud.datasets.s3dis import load_s3dis_room

        room_dir = "data/S3DIS/raw/Area_1/conferenceRoom_1"
        data = load_s3dis_room(room_dir)
        ```
    """
    room = defaultdict(list)

    for obj_idx, obj_path in enumerate(Path(room_dir).rglob("./Annotations/*.txt")):
        class_name, *_ = obj_path.stem.split("_")
        class_idx = S3DIS_CLASS_TO_IDX.get(class_name, S3DIS_UNK_IDX)

        data = np.loadtxt(obj_path, dtype=np.float32, delimiter=" ")
        points = torch.from_numpy(data)
        N, _ = points.shape

        if alignment_angle is not None:
            pos = points[:, 0:3]
            theta = alignment_angle * (torch.pi / 180.0)
            R = rodrigues_rotation_matrix(torch.tensor([0.0, 0.0, 1.0]), theta)
            points[:, 0:3] = pos @ R

        room["pos"].append(points[:, 0:3])
        room["color"].append(points[:, 3:6].to(torch.uint8))
        room["segment"].append(torch.full((N,), class_idx, dtype=torch.int64))
        room["instance"].append(torch.full((N,), obj_idx, dtype=torch.int64))

    for key, values in room.items():
        values = torch.cat(values, dim=0)  # type: ignore[assignment]
        if key == "instance":
            _, values = torch.unique(values, return_inverse=True)
        room[key] = values

    return S3DISRoomData(**room)  # type: ignore[typeddict-item]


def load_s3dis_alignment_angles(file_path: PathLike) -> dict[str, float]:
    """Load the alignment angles for a given area from a text file.
    In S3DIS dataset, one file is provided for each area, containing the alignment angles for each room in the area.
    The file is a text file with the following format:

    ```txt
    ## Global alignment angle per disjoint space in Area_1 ##
    ## Disjoint Space Name Global Alignment Angle ##
    conferenceRoom_1 0
    conferenceRoom_2 180
    ```

    Example:
        Assuming you have downloaded the raw dataset and extracted it under `data/S3DIS/raw`,
        you can load the alignment angles for the Area_1 as follows:

        ```python
        from torch_pointcloud.datasets.s3dis import load_s3dis_alignment_angles

        file_path = "data/S3DIS/raw/Area_1/Area_1_alignmentAngle.txt"
        angles = load_s3dis_alignment_angles(file_path)
        ```
    """
    with open(file_path, "r") as f:
        lines = f.readlines()

    lines = [line for line in lines if not line.startswith("#")]
    return {room_name: float(angle) for room_name, angle in [line.split() for line in lines]}


def tile_s3dis_room(
    room: dict[str, Any],
    block_size: float = 1.0,
    block_stride: float = 1.0,
    num_nodes: int = 4096,
    min_num_nodes: int = 100,
) -> list[dict[str, Any]]:
    r"""Split a single room dict into fixed-size spatial blocks.

    The algorithm shifts the room so that the minimum point is at the origin, then
    sweeps a $\text{block_size} \times \text{block_size}$ window (full Z extent) over the room with
    the given stride. This matches the procedure in multiple SOTA reference
    implementations (KPFCNN, ...).

    Args:
        room: Dict with at least `DataKeys.POS` (float32, $(N, 3)$).
            All other tensors with a leading dimension of $N$ are sliced in parallel.
        block_size: Side length of each square block in meters.
        block_stride: Step size for the sliding window in meters. Must be $\\leq$ `block_size`.
        num_nodes: Fixed number of nodes per block. Nodes are randomly subsampled
            (or duplicated if the block has fewer nodes).
        min_num_nodes: Minimum number of raw nodes for a block to be kept.

    Returns:
        List of dicts, one per retained block. Each block has exactly
        `num_nodes` nodes (randomly subsampled or oversampled).
        Positions are origin-shifted (room minimum at origin).

    Example:
        Assuming you have downloaded the raw dataset and extracted it under `data/S3DIS/raw`,
        you can tile the room into 1m x 1m blocks as follows:

        ```python
        from torch_pointcloud.datasets.s3dis import load_s3dis_room, tile_s3dis_room

        room_dir = "data/S3DIS/raw/Area_1/conferenceRoom_1"
        data = load_s3dis_room(room_dir)
        blocks = tile_s3dis_room(data)
        ```
    """
    pos = room[DataKeys.POS]
    N = pos.shape[0]

    pos_min = pos.min(dim=0).values
    pos_shifted = pos - pos_min
    room_max = pos_shifted.max(dim=0).values

    num_block_x = math.ceil((room_max[0].item() - block_size) / block_stride) + 1
    num_block_y = math.ceil((room_max[1].item() - block_size) / block_stride) + 1

    x = pos_shifted[:, 0]
    y = pos_shifted[:, 1]

    blocks: list[dict[str, Any]] = []
    for i in range(num_block_x):
        for j in range(num_block_y):
            x_min = i * block_stride
            y_min = j * block_stride
            x_max = x_min + block_size
            y_max = y_min + block_size

            mask = (x >= x_min) & (x <= x_max) & (y >= y_min) & (y <= y_max)
            indices = mask.nonzero(as_tuple=True)[0]
            n = indices.numel()
            if n < min_num_nodes:
                continue

            if n >= num_nodes:
                chosen = indices[torch.randperm(n)[:num_nodes]]
            else:
                chosen = indices[torch.randint(0, n, (num_nodes,))]

            block: dict[str, Any] = {}
            for key, val in room.items():
                if isinstance(val, Tensor) and val.shape[0] == N:
                    block[key] = (pos_shifted[chosen] if key == DataKeys.POS else val[chosen]).clone()
                else:
                    block[key] = val

            # Store the room maximum coordinates for normalization (might be used in transforms)
            block["room_max"] = room_max
            blocks.append(block)

    return blocks


class S3DIS(PointCloudDataset):
    """
    The Stanford 3D Indoor Spaces Dataset (S3DIS) dataset, as described in the original paper
    [3D Indoor Spaces Dataset: Collection, Annotations, and Methods](https://openaccess.thecvf.com/content_cvpr_2016/papers/Armeni_3D_Semantic_Parsing_CVPR_2016_paper.pdf).

    You can download the raw dataset from https://cvg-data.inf.ethz.ch/s3dis/ official website.

    The S3DIS dataset contains 6 diverse areas (one used for testing) covering a total of 6020 square meters.
    Each area contains multiple rooms (e.g. office, conference room, etc.), and each room contains multiple segment regions
    (e.g. wall, floor, ceiling, etc.) with instance-level annotations.

    The dataset will be processed automatically and saved in the `S3DIS/processed` directory.
    Each room is stored as its own folder, mirroring
    [Pointcept's preprocessing layout](https://github.com/Pointcept/Pointcept/blob/main/pointcept/datasets/preprocessing/s3dis/preprocess_s3dis.py):
    `<processed_dir>/<Area_i>/<room_name>/{coord,color,segment,instance}.npy`. If the processed
    data already exists, it will be loaded from the `S3DIS/processed` directory and processing
    will be skipped.

    Tip:
        If you change the preprocessing parameters, you can delete the processed data to reprocess the dataset
        or use the `force_process` argument to force the processing of the raw data.

    Args:
        root: The root directory of the dataset, where the raw and processed data will be stored.
        areas: The areas to load, either a list of area names or "all".
        classes: The classes to load, either a list of class names or "all".
        align: Whether to return aligned (axis-aligned) coordinates.  The raw
            download is `Stanford3dDataset_v1.2_Aligned_Version` whose coordinates
            already contain per-room alignment rotations.  When `True` (default)
            those coordinates are used as-is.  When `False` the *inverse* of each
            room's alignment angle is applied to recover the original V1.2
            (non-aligned) coordinate frame — this is required when benchmarking
            pretrained weights that were trained on non-aligned data (e.g. the DGCNN
            reference weights whose HDF5 blocks use non-aligned coordinates).
            Aligned and unaligned data are stored in separate processed directories
            so they can coexist.
        tile_blocks: Optional block tiling configuration. When provided, each room is split
            into fixed-size spatial blocks at load time (not at save/process time), so changing
            this parameter does not require reprocessing. See `TileBlocksConfig` for details.
        transform: A callable that transforms the data when retrieved from the dataset.
        download: Whether to download the raw data.
        force_download: Whether to force the download of the raw data.
        force_process: Whether to force the processing of the raw data.
        show_progress: Whether to show a progress bar during processing.
        num_workers: Number of worker processes for parallel room processing. If `None`,
            rooms are processed sequentially.

    Example:
        Assuming you have downloaded the raw dataset from https://cvg-data.inf.ethz.ch/s3dis/,
        and extracted it under `data/S3DIS/raw`, you can load the dataset as follows:

        ``python
        from torch_pointcloud.datasets import S3DIS

        dataset = S3DIS(
            root="data",
            areas=["Area_1", "Area_2", "Area_3", "Area_4", "Area_6"],
        )
        ``

        To split rooms into 1m x 1m blocks (matching the DGCNN evaluation protocol):

        ``python
        from torch_pointcloud.datasets.s3dis import TileBlocksConfig

        dataset = S3DIS(
            root="data",
            areas=["Area_5"],
            tile_blocks=TileBlocksConfig(block_size=1.0, stride=1.0, num_points=4096),
        )
        ``
    """

    data_url = "https://cvg-data.inf.ethz.ch/s3dis/"

    resources = [
        "ReadMe.txt",
        # NOTE: We always use the updated aligned version of the dataset.
        # To get the original unaligned version, we used the provided alignment angles to revert (if desired).
        "Stanford3dDataset_v1.2_Aligned_Version.zip",
    ]
    md5 = "ca095ff6721a379f2fbd97b82d3a9960"

    def __init__(
        self,
        root: PathLike,
        *,
        areas: Union[ValueCollection[S3DISArea], Literal["all"]] = "all",
        classes: Union[ValueCollection[S3DISClass], Literal["all"]] = "all",
        align: bool = True,
        block_size: Optional[float] = None,
        block_stride: float = 1.0,
        num_nodes: int = 4096,
        min_num_nodes: int = 100,
        transform: Optional[Callable] = None,
        download: bool = False,
        force_download: bool = False,
        force_process: bool = False,
        show_progress: bool = True,
        num_workers: Optional[int] = None,
    ) -> None:
        super().__init__(root)

        self.areas = ensure_tuple(areas if areas != "all" else S3DIS_AREAS)
        self.classes = ensure_tuple(classes if classes != "all" else S3DIS_CLASSES)
        self.align = align
        self.transform = transform
        self.show_progress = show_progress
        self.num_workers = num_workers

        _check_areas(self.areas)
        _check_classes(self.classes)

        if download or force_download:
            self.download(force=force_download)

        self.process(force=force_process, num_workers=num_workers, show_progress=show_progress)
        self.load(
            show_progress=show_progress,
            block_size=block_size,
            block_stride=block_stride,
            num_nodes=num_nodes,
            min_num_nodes=min_num_nodes,
        )

    @property
    def processed_dir(self) -> str:
        base = Path(self.data_dir, "processed")
        if not self.align:
            base = base.with_name("processed_unaligned")
        return base.absolute().as_posix()

    @cached_property
    def class_to_idx(self) -> dict[str, int]:
        return {cls: idx for idx, cls in enumerate(self.classes)}

    @override
    def raw_files_exist(self) -> bool:
        for area in self.areas:
            if not Path(self.raw_dir, area).is_dir():
                return False

            room_dirs = [path for path in Path(self.raw_dir, area).iterdir() if path.is_dir()]
            for room_dir in room_dirs:
                if not any(room_dir.rglob("Annotations/*.txt")):
                    return False

        return True

    @override
    def processed_files_exist(self) -> bool:
        return all(self.is_area_processed(area) for area in self.areas)

    def is_area_processed(self, area: str) -> bool:
        file_names = ["pos.npy", "color.npy", "segment.npy", "instance.npy", "offset.npy"]
        area_dir = Path(self.processed_dir, area)
        return all((area_dir / name).exists() for name in file_names)

    def download(self, force: bool = False) -> None:
        if self.raw_files_exist() and not force:
            return

        # Download the README file
        readme_url = urljoin(self.data_url, self.resources[0])
        readme_path = Path(self.raw_dir, self.resources[0])

        if not readme_path.exists() or force:
            download_url(
                readme_url,
                readme_path,
                description=f"Downloading {readme_path.name}",
                show_progress=self.show_progress,
            )

        # Download the dataset
        resource_url = urljoin(self.data_url, self.resources[1])
        resource_path = Path(self.raw_dir, self.resources[1])

        if (
            not resource_path.exists()
            or not is_hash_valid(resource_path, expected_hash=self.md5, hash_type="md5")
            or force
        ):
            download_url(
                resource_url,
                resource_path,
                description=f"Downloading {resource_path.name}",
                show_progress=self.show_progress,
            )

        # Extract the dataset
        extract_zip(
            resource_path,
            self.raw_dir,
            relative_to=resource_path.stem,
            show_progress=self.show_progress,
        )

        # clean up files stored in the archive
        for ds_store in Path(self.raw_dir).rglob(".DS_Store"):
            ds_store.unlink()

        for icon in Path(self.raw_dir).rglob("Icon"):
            icon.unlink()

        # Remove duplicated file
        file_path = Path(self.raw_dir, "Area_5/office_36/Annotations/wall_3 (1).txt")
        if file_path.exists():
            file_path.unlink()

        # fix corrupted file(s)
        file_path = Path(self.raw_dir, "Area_5/hallway_6/Annotations/ceiling_1.txt")
        with open(file_path, "r") as f:
            lines = f.readlines()
            lines = [line.replace("\00", " ") for line in lines]

        with open(file_path, "w") as f:
            f.writelines(lines)

    def process(
        self,
        force: bool = False,
        num_workers: Optional[int] = None,
        show_progress: bool = True,
    ) -> None:
        """Process the raw dataset for easier loading. This will NOT apply tiling or align rooms."""
        if self.processed_files_exist() and not force:
            return
        if not self.raw_files_exist():
            raise RuntimeError(
                f"Dataset not found at {self.raw_dir!r}. "
                f"You can download it from {self.data_url!r} "
                f"and extract it under {self.raw_dir!r}."
            )

        for area in self.areas:
            if not force and self.is_area_processed(area):
                continue

            area_dir = Path(self.raw_dir, area)
            angles: dict[str, float] = {}
            if not self.align:
                angle_path = area_dir / f"{area}_alignmentAngle.txt"
                raw_angles = load_s3dis_alignment_angles(angle_path) if angle_path.exists() else {}
                angles = {room: -angle for room, angle in raw_angles.items()}

            jobs = [
                (room_dir, angles.get(room_dir.name))
                for room_dir in sorted(p for p in area_dir.iterdir() if p.is_dir())
            ]

            rooms = parallel_map(
                lambda job: load_s3dis_room(*job),
                jobs,
                num_workers=num_workers,
                total=len(jobs),
                desc=f"Processing {area}",
                show_progress=show_progress,
            )
            if not rooms:
                raise RuntimeError(f"Found no valid rooms in {area!r}.")

            area_dir = Path(self.processed_dir, area)
            area_dir.mkdir(parents=True, exist_ok=True)

            sizes = np.array([r["pos"].shape[0] for r in rooms], dtype=np.int64)
            offsets = np.concatenate(([0], np.cumsum(sizes))).astype(np.int64)

            np.save(area_dir / "pos.npy", np.concatenate([r["pos"] for r in rooms], dtype=np.float32))
            np.save(area_dir / "color.npy", np.concatenate([r["color"] for r in rooms], dtype=np.uint8))
            np.save(area_dir / "segment.npy", np.concatenate([r["segment"] for r in rooms], dtype=np.int16))
            np.save(area_dir / "instance.npy", np.concatenate([r["instance"] for r in rooms], dtype=np.int16))
            np.save(area_dir / "offset.npy", offsets)

    def load(
        self,
        block_size: Optional[float] = None,
        block_stride: float = 1.0,
        num_nodes: int = 4096,
        min_num_nodes: int = 100,
        show_progress: bool = True,
    ) -> None:
        """Load the processed dataset into memory.
        If the provided block_size is not `None` and greater than 0, the rooms will be split into fixed-size
        spatial blocks.

        Args:
            block_size: Side length of each square block in meters.
            block_stride: Step size for the sliding window in meters. Must be $\\leq$ `block_size`.
            num_nodes: Fixed number of nodes per block. Nodes are randomly subsampled
                (or duplicated if the block has fewer nodes).
            min_num_nodes: Minimum number of raw nodes for a block to be kept.
            show_progress: Whether to show a progress bar during loading.
        """
        # Build the mapping between original classes and selected classes,
        # i.e. an array of shape $(C,)$ where $C$ is the number of classes and ignored classes are mapped to `unk_idx`.
        remap: np.ndarray | None = None
        if tuple(self.classes) != tuple(S3DIS_CLASSES):
            remap = np.full(len(S3DIS_CLASSES), S3DIS_UNK_IDX, dtype=np.int64)
            for new_id, cls_name in enumerate(self.classes):
                remap[S3DIS_CLASS_TO_IDX[cls_name]] = new_id

        # Load each room from all areas
        self.data: list[dict[str, Any]] = []
        for area in tqdm(self.areas, total=len(self.areas), desc="Loading", disable=not show_progress):
            area_dir = Path(self.processed_dir, area)
            offsets = np.load(area_dir / "offset.npy")
            pos = np.load(area_dir / "pos.npy")
            color = np.load(area_dir / "color.npy")
            segment = np.load(area_dir / "segment.npy")
            instance = np.load(area_dir / "instance.npy")

            n_rooms = len(offsets) - 1
            for i in range(n_rooms):
                s, e = int(offsets[i]), int(offsets[i + 1])
                seg = segment[s:e] if remap is None else remap[segment[s:e]]

                room: dict[str, Any] = {
                    DataKeys.POS: torch.from_numpy(pos[s:e].copy()),
                    DataKeys.COLOR: torch.from_numpy(color[s:e].copy()),
                    DataKeys.SEGMENT: torch.from_numpy(seg.astype(np.int64)),
                    DataKeys.INSTANCE: torch.from_numpy(instance[s:e].astype(np.int64)),
                }

                if block_size is not None and block_size > 0:
                    blocks = tile_s3dis_room(
                        room,
                        block_size=block_size,
                        block_stride=block_stride,
                        num_nodes=num_nodes,
                        min_num_nodes=min_num_nodes,
                    )
                    self.data.extend(blocks)
                else:
                    self.data.append(room)

    @override
    def __len__(self) -> int:
        return len(self.data)

    @override
    def __getitem__(self, index: int) -> dict[str, Any]:
        data = self.data[index]
        if self.transform is not None:
            data = self.transform(data)
        return data


class S3DISHdf5(PointCloudDataset):
    """Pre-processed HDF5 version of the S3DIS dataset used by multiple SOTA reference implementations.

    Unlike `S3DIS`, which loads from the raw annotated rooms, this class loads
    the pre-tiled 4096-point blocks distributed as HDF5 files.
    The blocks are already spatially tiled and fixed-size, so no additional tiling step is needed.

    Each sample is a dict with the following keys:

    | Key        | Shape          | Dtype   | Description                                     |
    | ---------- | -------------- | ------- | ----------------------------------------------- |
    | `pos`      | $(4096, 3)$    | float32 | XYZ coordinates                                 |
    | `color`    | $(4096, 3)$    | float32 | RGB values                                      |
    | `norm_pos` | $(4096, 3)$    | float32 | Room-normalised XYZ coordinates (range $[0,1]$) |
    | `segment`  | $(4096,)$      | int64   | Per-point semantic label (13 classes)           |

    Important:
        This dataset is already processed and the HDF5 files are used directly.
        They are stored in the `S3DIS/indoor3d_sem_seg_hdf5_data` directory,
        meaning that they are co-located with the `S3DIS` dataset.

    Args:
        root: Root directory where `S3DIS/indoor3d_sem_seg_hdf5_data/` is stored.
        areas: Areas to load, either a sequence of area names or `"all"`.
        classes: Classes to load, either a sequence of class names or `"all"`.
        transform: Optional callable applied to each sample dict at `__getitem__` time.
        download: Whether to download the HDF5 archive if not already present.
        force_download: Whether to re-download even if the archive already exists.
        force_process: Whether to force re-processing (no-op for this variant since
            the HDF5 files are used directly).
        show_progress: Whether to show progress bars during download and loading.

    Example:
        Assuming you have downloaded the HDF5 files from https://shapenet.cs.stanford.edu/media/,
        and extracted it under `data/S3DIS/indoor3d_sem_seg_hdf5_data`, you can load the dataset as follows:

        ```python
        from torch_pointcloud.datasets import S3DISHdf5

        dataset = S3DISHdf5(root="data", areas=["Area_1"])
        sample = dataset[0]
        sample["pos"].shape   # torch.Size([4096, 3])
        sample["segment"].shape  # torch.Size([4096])
        ```
    """

    data_url = "https://shapenet.cs.stanford.edu/media/"
    resource = "indoor3d_sem_seg_hdf5_data.zip"
    md5 = "f07d79acdea1f497b3fb3d32f34f1428"
    classes = S3DIS_CLASSES

    def __init__(
        self,
        root: PathLike,
        *,
        areas: Union[Sequence[S3DISArea], Literal["all"]] = "all",
        classes: Union[ValueCollection[S3DISClass], Literal["all"]] = "all",
        transform: Optional[Callable] = None,
        download: bool = True,
        force_download: bool = False,
        force_process: bool = False,
        show_progress: bool = True,
    ) -> None:
        super().__init__(root)
        self.areas = ensure_tuple(areas if areas != "all" else S3DIS_AREAS)
        self.classes = ensure_tuple(classes if classes != "all" else S3DIS_CLASSES)
        self.transform = transform
        self.show_progress = show_progress

        _check_areas(self.areas)

        if download:
            self.download(force=force_download, show_progress=show_progress)

        self.load(show_progress=show_progress)

    @property
    def name(self) -> str:
        return "S3DIS"

    @property
    def raw_dir(self) -> str:
        return Path(self.root, self.name, "indoor3d_sem_seg_hdf5_data").as_posix()

    @property
    def processed_dir(self) -> str:
        return Path(self.root, self.name, "indoor3d_sem_seg_hdf5_data").as_posix()

    def raw_files_exist(self) -> bool:
        file_names = ["all_files.txt", "room_filelist.txt"]
        for block_idx in range(24):
            file_names.append(f"ply_data_all_{block_idx}.h5")

        return all(Path(self.raw_dir, file_name).exists() for file_name in file_names)

    def processed_files_exist(self) -> bool:
        return self.raw_files_exist()

    def download(self, force: bool = False, show_progress: bool = True) -> None:
        if self.raw_files_exist() and not force:
            return

        resource_path = Path(self.data_dir, self.resource)
        resource_url = urljoin(self.data_url, self.resource)

        if (
            not resource_path.exists()
            or not is_hash_valid(resource_path, expected_hash=self.md5, hash_type="md5")
            or force
        ):
            download_url(
                resource_url,
                resource_path,
                description=f"Downloading {self.resource}",
                show_progress=show_progress,
            )

        extract_zip(resource_path, self.raw_dir, show_progress=show_progress)

        # Cleanup the downloaded zipped file
        resource_path.unlink()

    def load(self, show_progress: bool = True) -> None:
        if not self.processed_files_exist():
            raise RuntimeError(
                f"Dataset not found at {self.processed_dir!r}. "
                f"You can download it from {self.data_url!r} "
                f"and extract it under {self.processed_dir!r}."
            )

        with open(Path(self.raw_dir, "all_files.txt")) as f:
            file_names = [line.strip().lstrip("indoor3d_sem_seg_hdf5_data/") for line in f]
        with open(Path(self.raw_dir, "room_filelist.txt")) as f:
            room_names = [line.strip() for line in f]

        data_chunks: list[np.ndarray] = []
        label_chunks: list[np.ndarray] = []
        offset = 0
        for file_name in tqdm(file_names, total=len(file_names), desc="Loading", disable=not show_progress):
            file_path = Path(self.raw_dir, file_name)
            with h5py.File(file_path, "r") as hf:
                chunk_data = hf["data"][:]
                chunk_labels = hf["label"][:]

            n = chunk_data.shape[0]
            keep = np.array(
                ["_".join(room_names[offset + i].split("_")[:2]) in self.areas for i in range(n)],
                dtype=bool,
            )
            offset += n

            if keep.any():
                data_chunks.append(chunk_data[keep])
                label_chunks.append(chunk_labels[keep])

        data = np.concatenate(data_chunks, axis=0)
        labels = np.concatenate(label_chunks, axis=0).astype(np.int64)
        if labels.ndim == 2 and labels.shape[1] == 1:
            labels = labels.squeeze(1)

        self.data = data
        self.labels = labels

    @override
    def __len__(self) -> int:
        return len(self.data)

    @override
    def __getitem__(self, index: int) -> dict[str, Any]:
        block = self.data[index]
        label = self.labels[index]

        data: dict[str, Any] = {
            DataKeys.POS: torch.from_numpy(block[:, 0:3]),
            DataKeys.COLOR: torch.from_numpy(block[:, 3:6]),
            "norm_pos": torch.from_numpy(block[:, 6:9]),
            DataKeys.SEGMENT: torch.from_numpy(label),
        }

        if self.transform is not None:
            data = self.transform(data)
        return data
