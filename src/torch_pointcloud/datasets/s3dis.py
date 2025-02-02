import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, Generator, List, Literal, Optional, Sequence, Tuple, TypedDict, Union
from urllib.parse import urljoin

import numpy as np
import torch
from torch import Tensor
from tqdm import tqdm
from typing_extensions import override

from torch_pointcloud.utils.conversion import ensure_tuple
from torch_pointcloud.utils.geometry import axis_aligned_bounding_box, rodrigues_rotation_matrix
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
    coords: torch.Tensor
    colors: torch.Tensor
    semantic: torch.Tensor
    instances: torch.Tensor
    bboxes: torch.Tensor


def load_s3dis_room_data(
    room_dir: PathLike,
    with_coords: bool = True,
    with_colors: bool = True,
    with_semantic: bool = False,
    with_instances: bool = False,
    with_bboxes: bool = False,
    alignment_angle: float | None = None,
    class_to_idx: Optional[Dict[str, int]] = None,
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

        if with_coords:
            room["coords"].append(points[:, 0:3])
        if with_colors:
            room["colors"].append(points[:, 3:6].to(torch.uint8))
        if with_semantic:
            room["semantic"].append(torch.full((N,), category_idx, dtype=torch.int64))
        if with_instances:
            room["instances"].append(torch.full((N,), obj_idx, dtype=torch.int64))
        if with_bboxes:
            bboxes = axis_aligned_bounding_box(points[:, 0:3])
            bboxes = torch.cat([bboxes, torch.tensor([category_idx])])
            room["boxes"].append(bboxes.reshape(1, -1))

    # Stack the data
    for key, values in room.items():
        values = torch.cat(values, dim=0)  # type: ignore[assignment]
        if key == "instances":
            _, values = torch.unique(values, return_inverse=True)
        room[key] = values

    return S3DISRoomData(**room)  # type: ignore[typeddict-item]


def load_s3dis_alignment_angles(file_path: PathLike) -> Dict[str, float]:
    with open(file_path, "r") as f:
        lines = f.readlines()

    lines = [line for line in lines if not line.startswith("#")]
    return {room_name: float(angle) for room_name, angle in [line.split() for line in lines]}


def iter_blocks(
    coords: torch.Tensor,
    block_size: float,
    stride: float,
) -> Generator[Tuple[Tensor, Tensor], None, None]:
    x_max, y_max, _ = torch.max(coords, dim=0).values
    x_min, y_min, _ = torch.min(coords, dim=0).values
    num_block_x = abs(math.ceil((x_max - block_size) / stride)) + 1
    num_block_y = abs(math.ceil((y_max - block_size) / stride)) + 1

    x_starts, y_starts = np.meshgrid(np.arange(num_block_x), np.arange(num_block_y))
    x_starts = torch.from_numpy(x_starts.flatten() * stride) + x_min
    y_starts = torch.from_numpy(y_starts.flatten() * stride) + y_min
    indices = torch.arange(coords.size(0))

    # Collect blocks
    for x_start, y_start in zip(x_starts, y_starts):
        x_cond = (coords[:, 0] >= x_start) & (coords[:, 0] <= x_start + block_size)
        y_cond = (coords[:, 1] >= y_start) & (coords[:, 1] <= y_start + block_size)
        cond = x_cond & y_cond

        if cond.sum() == 0:
            continue

        yield coords[cond], indices[cond]


class S3DIS(PointCloudDataset):
    """
    The Stanford 3D Indoor Spaces Dataset (S3DIS) dataset, as described in the original paper
    [3D Indoor Spaces Dataset: Collection, Annotations, and Methods](https://openaccess.thecvf.com/content_cvpr_2016/papers/Armeni_3D_Semantic_Parsing_CVPR_2016_paper.pdf).

    You can download the raw dataset from https://cvg-data.inf.ethz.ch/s3dis/ official website.

    The S3DIS dataset contains 6 diverse areas (one used for testing) covering a total of 6020 square meters.
    Each area contains multiple rooms (e.g. office, conference room, etc.), and each room contains multiple semantic regions
    (e.g. wall, floor, ceiling, etc.) with instance-level annotations.

    The dataset will be processed automatically and saved in the `S3DIS/processed` directory.
    If the processed data already exists, it will be loaded from the `S3DIS/processed` directory
    and processing steps will be skipped.
    The raw dataset is processed by blocks of size `block_size` with a stride of `block_stride`.

    > [!TIP]
    > If you change the preprocessing parameters, you can delete the processed data to reprocess the dataset
    > or use the `force_process` argument to force the processing of the raw data.

    Args:
        root: The root directory of the dataset, where the raw and processed data will be stored.
        areas: The areas to load, either a list of area names or "all".
        classes: The classes to load, either a list of class names or "all".
        unk_id: The id to use for unknown classes.
        block_size: The size of the blocks to process.
        block_stride: The stride of the blocks to process.
        transform: A callable that transforms the data when retrieved from the dataset.
        normalize_coords: Whether to normalize and center the coordinates of the block.
        pre_transform: Used to transform the data before saving it in the processed directory.
        pre_filter: Used to filter the data before saving it in the processed directory.
        download: Whether to download the raw data.
        force_download: Whether to force the download of the raw data.
        force_process: Whether to force the processing of the raw data.
        show_progress: Whether to show a progress bar during processing.

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

        You can select the block size to process the dataset by passing the `block_size` argument.
        For example, to process blocks of size 1 meter, you can do:

        ```python
        dataset = S3DIS(
            root="data",
            block_size=1,  # The size of the blocks to process.
            block_stride=0.5,  # The stride of the blocks to process.
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
        areas: Union[List[str], Literal["all"]] = "all",
        classes: Optional[Union[str, Sequence[str]]] = "all",
        unk_id: Optional[int] = None,
        block_size: float = 1,
        block_stride: float = 0.5,
        transform: Optional[Callable] = None,
        normalize_coords: bool = True,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None,
        download: bool = False,
        force_download: bool = False,
        force_process: bool = False,
        show_progress: bool = True,
    ) -> None:
        super().__init__(root)

        self.areas = areas if areas != "all" else ["Area_1", "Area_2", "Area_3", "Area_4", "Area_5", "Area_6"]
        self.classes = ensure_tuple(S3DIS_CLASS_TO_IDX.keys() if classes == "all" else classes)
        self.unk_id = unk_id
        self.block_size = block_size
        self.block_stride = block_stride
        self.transform = transform
        self.normalize_coords = normalize_coords
        self.pre_transform = pre_transform
        self.pre_filter = pre_filter
        self.show_progress = show_progress

        if download:
            self.download(force=force_download)

        self.process(force=force_process)
        self.data = self._load_processed_data()

    @property
    def class_to_idx(self) -> Dict[str, int]:
        return {cls: idx for idx, cls in enumerate(self.classes)}

    @override
    def raw_files_exist(self) -> bool:
        for area in self.areas:
            if not Path(self.raw_dir, area).is_dir():
                return False

            for room_dir in Path(self.raw_dir, area).iterdir():
                if not any(room_dir.rglob("Annotations/*.txt")):
                    return False
                return False
        return True

    @override
    def processed_files_exist(self) -> bool:
        for area in self.areas:
            if not Path(self.processed_dir, f"{area}.pt").exists():
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

        for area in self.areas:
            area_dir = Path(self.raw_dir, area)

            alignment_angles = {}
            alignment_angle_path = area_dir / f"{area}_alignmentAngle.txt"
            if alignment_angle_path.exists():
                alignment_angles = load_s3dis_alignment_angles(alignment_angle_path)

            blocks = []

            room_dirs = [path for path in area_dir.iterdir() if path.is_dir()]
            pbar = tqdm(room_dirs, total=len(room_dirs), desc=f"Processing {area}")
            for room_dir in pbar:
                alignment_angle = alignment_angles.get(room_dir.name, None)
                room_blocks = self._process_room(room_dir, alignment_angle, self.class_to_idx)
                room_blocks = [block for block in room_blocks if block is not None]
                blocks.extend(room_blocks)

            out_path = Path(self.processed_dir, f"{area}.pt")
            out_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(blocks, out_path)

    def _process_room(
        self,
        room_dir: PathLike,
        alignment_angle: float | None = None,
        class_to_idx: Optional[Dict[str, int]] = None,
    ) -> List[Optional[Dict[str, Any]]]:
        room = load_s3dis_room_data(
            room_dir,
            with_coords=True,
            with_colors=True,
            with_semantic=True,
            with_instances=True,
            alignment_angle=alignment_angle,
            class_to_idx=class_to_idx,
            unk_id=self.unk_id,
        )

        blocks: List[Optional[Dict[str, Any]]] = []
        for block_coords, block_idxs in iter_blocks(
            room["coords"],
            block_size=self.block_size,
            stride=self.block_stride,
        ):
            block_data = {"coords": block_coords}
            for key in set(room.keys()) - set(block_data.keys()):
                block_data[key] = room[key][block_idxs]  # type: ignore[literal-required]

            if self.pre_filter is not None and not self.pre_filter(block_data):
                continue

            # TODO: move to a transform
            if self.normalize_coords:
                x_min, y_min, _ = torch.min(block_data["coords"], dim=0).values
                delta = self.block_size / 2
                block_data["coords"] = block_data["coords"] - torch.tensor([x_min + delta, y_min + delta, 0])

            if self.pre_transform is not None:
                block_data = self.pre_transform(block_data)

            blocks.append(block_data)

        return blocks

    def _load_processed_data(self) -> List[Dict[str, Tensor]]:
        data = []
        for area in self.areas:
            file_path = Path(self.processed_dir, f"{area}.pt")
            data.extend(torch.load(file_path, weights_only=True))
        return data

    @override
    def __getitem__(self, index: int) -> Dict[str, Any]:
        data = self.data[index]
        if self.transform is not None:
            data = self.transform(data)
        return data

    @override
    def __len__(self) -> int:
        return len(self.data)
