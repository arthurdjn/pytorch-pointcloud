from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Literal, Optional, Sequence, Tuple, TypedDict, Union
from urllib.parse import urljoin

import numpy as np
import torch
from torch import Tensor
from tqdm import tqdm
from typing_extensions import override

from torch_pointcloud.utils.conversion import ensure_tuple
from torch_pointcloud.utils.data import DataKeys
from torch_pointcloud.utils.geometry import rodrigues_rotation_matrix
from torch_pointcloud.utils.misc import parallel_map
from torch_pointcloud.utils.types import PathLike

from .pointcloud import PointCloudDataset
from .utils import download_url, extract_zip, is_hash_valid

S3DIS_AREAS = ["Area_1", "Area_2", "Area_3", "Area_4", "Area_5", "Area_6"]
S3DIS_CLASSES = [
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
S3DIS_CLASS_TO_IDX = {cls: idx for idx, cls in enumerate(S3DIS_CLASSES)}
S3DIS_UNK_IDX: int | None = None
S3DIS_UNK_CLASS = "<unk>"


class S3DISRoomData(TypedDict, total=False):
    pos: Tensor
    color: Tensor
    segment: Tensor
    instance: Tensor


def _check_areas(areas: Sequence[str]) -> None:
    for area in areas:
        if area not in S3DIS_AREAS:
            available_areas = ", ".join(S3DIS_AREAS)
            raise ValueError(f"Unknown area: {area!r}. Must be one of {available_areas}.")


def _check_classes(classes: Sequence[str]) -> None:
    for cls_name in classes:
        if cls_name not in S3DIS_CLASSES:
            available_classes = ", ".join(S3DIS_CLASSES)
            raise ValueError(f"Unknown class: {cls_name!r}. Must be one of {available_classes}.")


def load_s3dis_room_data(
    room_dir: PathLike,
    alignment_angle: float | None = None,
    class_to_idx: dict[str, int] | None = None,
    unk_idx: int | None = S3DIS_UNK_IDX,
) -> S3DISRoomData:
    class_to_idx = class_to_idx or S3DIS_CLASS_TO_IDX
    room = defaultdict(list)

    for obj_idx, obj_path in enumerate(Path(room_dir).rglob("./Annotations/*.txt")):
        # Get the associated class (e.g. 'chair_24' -> 'chair')
        # NOTE: some rooms have an unknown class 'stairs', that should be treated as 'clutter'
        class_name, *_ = obj_path.stem.split("_")
        class_idx = class_to_idx.get(class_name, unk_idx)

        if class_idx is None:
            # Skip loading the data of the room if it has an unknown class
            continue

        data = np.loadtxt(obj_path, dtype=np.float32, delimiter=" ")
        points = torch.from_numpy(data)
        N, _ = points.shape

        if alignment_angle is not None:
            coords = points[:, 0:3]
            R = rodrigues_rotation_matrix(torch.tensor([0, 0, 1]), alignment_angle)
            points[:, 0:3] = coords @ R

        room["pos"].append(points[:, 0:3])
        room["color"].append(points[:, 3:6].to(torch.uint8))
        room["segment"].append(torch.full((N,), class_idx, dtype=torch.int64))
        room["instance"].append(torch.full((N,), obj_idx, dtype=torch.int64))

    # Stack the data
    for key, values in room.items():
        values = torch.cat(values, dim=0)  # type: ignore[assignment]
        if key == "instance":
            _, values = torch.unique(values, return_inverse=True)
        room[key] = values

    return S3DISRoomData(**room)  # type: ignore[typeddict-item]


def load_s3dis_alignment_angles(file_path: PathLike) -> dict[str, float]:
    with open(file_path, "r") as f:
        lines = f.readlines()

    lines = [line for line in lines if not line.startswith("#")]
    return {room_name: float(angle) for room_name, angle in [line.split() for line in lines]}


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

    > [!TIP]
    > If you change the preprocessing parameters, you can delete the processed data to reprocess the dataset
    > or use the `force_process` argument to force the processing of the raw data.

    Args:
        root: The root directory of the dataset, where the raw and processed data will be stored.
        areas: The areas to load, either a list of area names or "all".
        classes: The classes to load, either a list of class names or "all".
        unk_idx: The id to use for unknown classes.
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

        ```python
        from torch_pointcloud.datasets import S3DIS

        dataset = S3DIS(
            root="data",
            areas=["Area_1", "Area_2", "Area_3", "Area_4", "Area_6"],
        )
        ```

        You can select specific classes to load by passing a list of class names to the `classes` argument.
        For example, to load only the "wall" and "floor" classes, you can do:

        ```python
        dataset = S3DIS(
            root="data",
            classes=["wall", "floor"],
            unk_idx=-1,  # Use -1 to treat unknown classes as a single class
        )
        ```
    """

    data_url = "https://cvg-data.inf.ethz.ch/s3dis/"
    resources = [
        "ReadMe.txt",
        "Stanford3dDataset_v1.2_Aligned_Version.zip",
    ]
    md5 = "ca095ff6721a379f2fbd97b82d3a9960"

    def __init__(
        self,
        root: PathLike,
        *,
        areas: Union[list[str], Literal["all"]] = "all",
        classes: Optional[Union[str, Sequence[str]]] = "all",
        unk_idx: Optional[int] = None,
        unk_class: str = S3DIS_UNK_CLASS,
        transform: Optional[Callable] = None,
        download: bool = False,
        force_download: bool = False,
        force_process: bool = False,
        show_progress: bool = True,
        num_workers: Optional[int] = None,
    ) -> None:
        super().__init__(root)

        self.areas = areas if areas != "all" else S3DIS_AREAS
        self.classes = ensure_tuple(S3DIS_CLASS_TO_IDX.keys() if classes == "all" else classes)
        self.unk_idx = unk_idx
        self.unk_class = unk_class
        self.transform = transform
        self.show_progress = show_progress
        self.num_workers = num_workers

        _check_areas(self.areas)
        _check_classes(self.classes)

        if download or force_download:
            self.download(force=force_download)

        self.process(force=force_process, num_workers=num_workers, show_progress=show_progress)
        self.load(show_progress=show_progress)

    @property
    def class_to_idx(self) -> dict[str, int]:
        mapping = {cls: idx for idx, cls in enumerate(self.classes)}
        if self.unk_idx is not None:
            mapping[self.unk_class] = self.unk_idx
        return mapping

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
            angle_path = area_dir / f"{area}_alignmentAngle.txt"
            angles = load_s3dis_alignment_angles(angle_path) if angle_path.exists() else {}

            jobs = [
                (room_dir, angles.get(room_dir.name))
                for room_dir in sorted(p for p in area_dir.iterdir() if p.is_dir())
            ]

            def load_room(job: Tuple[Path, Optional[float]]) -> S3DISRoomData:
                return load_s3dis_room_data(*job, self.class_to_idx, self.unk_idx)

            rooms = parallel_map(
                load_room,
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

    def load(self, show_progress: bool = True) -> None:
        self.samples: list[dict[str, Any]] = []

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
                self.samples.append(
                    {
                        # NOTE: We use a copy to avoid modifying the original array
                        # in case a transform is modifying data in-place.
                        DataKeys.POS: torch.from_numpy(pos[s:e].copy()),
                        DataKeys.COLOR: torch.from_numpy(color[s:e].copy()),
                        DataKeys.SEGMENT: torch.from_numpy(segment[s:e].astype(np.int64)),
                        DataKeys.INSTANCE: torch.from_numpy(instance[s:e].astype(np.int64)),
                    }
                )

    @override
    def __len__(self) -> int:
        return len(self.samples)

    @override
    def __getitem__(self, index: int) -> dict[str, Any]:
        data = self.samples[index]
        if self.transform is not None:
            data = self.transform(data)
        return data
