from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Literal, Optional, Sequence, Tuple, TypedDict, Union
from urllib.parse import urljoin

import numpy as np
import torch
from torch import Tensor
from typing_extensions import override

from torch_pointcloud.utils.conversion import ensure_tuple
from torch_pointcloud.utils.data import DataKeys
from torch_pointcloud.utils.geometry import rodrigues_rotation_matrix
from torch_pointcloud.utils.misc import parallel_map
from torch_pointcloud.utils.types import PathLike

from .pointcloud import PointCloudDataset
from .utils import download_url, extract_zip, is_hash_valid

S3DIS_CLASS_TO_IDX = {
    "ceiling": 0,
    "floor": 1,
    "wall": 2,
    "beam": 3,
    "column": 4,
    "window": 5,
    "door": 6,
    "chair": 7,
    "table": 8,
    "bookcase": 9,
    "sofa": 10,
    "board": 11,
    "clutter": 12,
}


class S3DISRoomData(TypedDict, total=False):
    pos: Tensor
    color: Tensor
    semantic: Tensor
    instance: Tensor


def load_s3dis_room_data(
    room_dir: PathLike,
    alignment_angle: float | None = None,
    class_to_idx: Optional[dict[str, int]] = None,
    unk_id: Optional[int] = None,
) -> S3DISRoomData:
    class_to_idx = class_to_idx or S3DIS_CLASS_TO_IDX
    room = defaultdict(list)

    for obj_idx, obj_path in enumerate(Path(room_dir).rglob("./Annotations/*.txt")):
        # Get the associated class (e.g. 'chair_24' -> 'chair')
        # NOTE: some rooms have an unknown class 'stairs', that should be treated as 'clutter'
        category, *_ = obj_path.stem.split("_")
        category_idx = class_to_idx.get(category, unk_id)

        if category_idx is None:
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
        room["semantic"].append(torch.full((N,), category_idx, dtype=torch.int64))
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
    Each area contains multiple rooms (e.g. office, conference room, etc.), and each room contains multiple semantic regions
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
        unk_id: The id to use for unknown classes.
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
            unk_id=-1,  # Use -1 to treat unknown classes as a single class
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
        areas: Union[list[str], Literal["all"]] = "all",
        classes: Optional[Union[str, Sequence[str]]] = "all",
        unk_id: Optional[int] = None,
        transform: Optional[Callable] = None,
        download: bool = False,
        force_download: bool = False,
        force_process: bool = False,
        show_progress: bool = True,
        num_workers: Optional[int] = None,
    ) -> None:
        super().__init__(root)

        self.areas = areas if areas != "all" else ["Area_1", "Area_2", "Area_3", "Area_4", "Area_5", "Area_6"]
        self.classes = ensure_tuple(S3DIS_CLASS_TO_IDX.keys() if classes == "all" else classes)
        self.unk_id = unk_id
        self.transform = transform
        self.show_progress = show_progress
        self.num_workers = num_workers

        if download or force_download:
            self.download(force=force_download)

        self.process(force=force_process)
        self.data = self._load_processed_data()

    @property
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
        def is_room_processed(room_dir: Path) -> bool:
            return all((room_dir / f"{name}.npy").exists() for name in ("coord", "color", "segment", "instance"))

        for area in self.areas:
            area_dir = Path(self.processed_dir, area)
            if not area_dir.is_dir():
                return False

            room_dirs = [p for p in area_dir.iterdir() if p.is_dir()]
            if not room_dirs or not all(is_room_processed(p) for p in room_dirs):
                return False

        return True

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

    def process(self, force: bool = False) -> None:
        if self.processed_files_exist() and not force:
            return
        if not self.raw_files_exist():
            raise RuntimeError(
                f"Dataset not found at {self.root!r}. "
                f"You can download the raw dataset from {self.data_url!r}, "
                f"and extract it under {self.raw_dir!r}."
            )

        jobs: list[Tuple[Path, Path, Optional[float]]] = []
        for area in self.areas:
            area_dir = Path(self.raw_dir, area)
            out_area_dir = Path(self.processed_dir, area)
            angle_path = area_dir / f"{area}_alignmentAngle.txt"
            angles = load_s3dis_alignment_angles(angle_path) if angle_path.exists() else {}
            for room_dir in sorted(p for p in area_dir.iterdir() if p.is_dir()):
                jobs.append((room_dir, out_area_dir / room_dir.name, angles.get(room_dir.name)))

        parallel_map(
            lambda job: self._process_room(*job),
            jobs,
            num_workers=self.num_workers,
            desc="Processing",
            show_progress=self.show_progress,
        )

    def _process_room(
        self,
        room_dir: PathLike,
        out_dir: Path,
        alignment_angle: Optional[float] = None,
    ) -> None:
        room = load_s3dis_room_data(
            room_dir,
            alignment_angle=alignment_angle,
            class_to_idx=self.class_to_idx,
            unk_id=self.unk_id,
        )
        if not room:
            return

        out_dir.mkdir(parents=True, exist_ok=True)
        np.save(out_dir / "coord.npy", room["pos"].numpy().astype(np.float32))
        np.save(out_dir / "color.npy", room["color"].numpy().astype(np.uint8))
        np.save(out_dir / "segment.npy", room["semantic"].numpy().astype(np.int16))
        np.save(out_dir / "instance.npy", room["instance"].numpy().astype(np.int16))

    def _load_processed_data(self) -> list[dict[str, Any]]:
        def _load_processed_room_data(room_dir: Path) -> dict[str, Any]:
            pos = np.load(room_dir / "coord.npy")
            color = np.load(room_dir / "color.npy")
            semantic = np.load(room_dir / "segment.npy")
            instance = np.load(room_dir / "instance.npy")
            return {
                DataKeys.POS: torch.from_numpy(pos).float(),
                DataKeys.COLOR: torch.from_numpy(color).to(torch.uint8),
                DataKeys.SEMANTIC: torch.from_numpy(semantic).long(),
                DataKeys.INSTANCE: torch.from_numpy(instance).long(),
            }

        room_dirs = []
        for area in self.areas:
            area_dir = Path(self.processed_dir, area)
            room_dirs.extend(sorted(p for p in area_dir.iterdir() if p.is_dir()))

        return parallel_map(
            _load_processed_room_data,
            room_dirs,
            num_workers=self.num_workers,
        )

    @override
    def __getitem__(self, index: int) -> dict[str, Any]:
        data = self.data[index]
        if self.transform is not None:
            data = self.transform(data)
        return data

    @override
    def __len__(self) -> int:
        return len(self.data)
