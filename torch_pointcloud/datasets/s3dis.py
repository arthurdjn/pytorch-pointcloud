import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, Generator, List, Literal, Optional, Tuple, TypedDict, Union
from urllib.parse import urljoin

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset
from tqdm import tqdm

import torch_pointcloud.transforms.functional as F
from torch_pointcloud.utils import rodrigues_rotation_matrix
from torch_pointcloud.utils.types import PATH_LIKE

from .utils import download_file, extract_zip

S3DIS_FORM = "https://goo.gl/forms/4SoGp4KtH1jfRqEj2"
S3DIS_URL = "https://cvg-data.inf.ethz.ch/s3dis/"
CLASS_TO_IDX = {
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


class S3DISRoom(TypedDict, total=False):
    xyz: torch.Tensor
    rgb: torch.Tensor
    semantic: torch.Tensor
    instance: torch.Tensor


def xyz_to_bbox(xyz: Tensor) -> Tensor:
    # Compute axis aligned box
    # An axis aligned bounding box is parameterized by (cx,cy,cz) and (dx,dy,dz) and label id
    # where (cx,cy,cz) is the center point of the box, dx is the x-axis length of the box
    xmin, ymin, zmin, *_ = torch.min(xyz, dim=0).values
    xmax, ymax, zmax, *_ = torch.max(xyz, dim=0).values
    cx, cy, cz = (xmin + xmax) / 2.0, (ymin + ymax) / 2.0, (zmin + zmax) / 2.0
    dx, dy, dz = xmax - xmin, ymax - ymin, zmax - zmin
    return torch.tensor([cx, cy, cz, dx, dy, dz])


def load_s3dis_room(
    room_dir: PATH_LIKE,
    with_xyz: bool = True,
    with_rgb: bool = True,
    with_semantic: bool = False,
    with_instance: bool = False,
    with_bbox: bool = False,
    with_oriented_bbox: bool = False,
    alignment_angle: float | None = None,
) -> S3DISRoom:
    room = defaultdict(list)

    for obj_idx, obj_path in enumerate(Path(room_dir).rglob("./Annotations/*.txt")):
        if obj_path.name == "wall_3 (1).txt":
            # NOTE: This is a duplicate file in the original dataset.
            # In case this file still exists, we skip it.
            continue

        # get the associated class (e.g. 'chair_24' -> 'chair')
        # NOTE: some rooms have an unknown class 'stairs', that should be treated as 'clutter'
        category = obj_path.stem.split("_")[0]
        category_idx = CLASS_TO_IDX.get(category, CLASS_TO_IDX["clutter"])
        data = np.loadtxt(obj_path, dtype=np.float32, delimiter=" ")
        points = torch.from_numpy(data)
        N, _ = points.shape

        if alignment_angle is not None:
            xyz = points[:, 0:3]
            R = rodrigues_rotation_matrix(torch.tensor([0, 0, 1]), alignment_angle)
            points[:, 0:3] = xyz @ R

        if with_xyz:
            room["xyz"].append(points[:, 0:3])
        if with_rgb:
            room["rgb"].append(points[:, 3:6].to(torch.uint8))
        if with_semantic:
            room["semantic"].append(torch.full((N,), category_idx, dtype=torch.int64))
        if with_instance:
            room["instance"].append(torch.full((N,), obj_idx, dtype=torch.int64))
        if with_bbox:
            bbox = xyz_to_bbox(points[:, 0:3])
            bbox = torch.cat([bbox, torch.tensor([category_idx])])
            room["bbox"].append(bbox.reshape(1, -1))
        if with_oriented_bbox:
            raise NotImplementedError("Oriented bounding box not implemented yet.")

    for key, values in room.items():
        room[key] = torch.cat(values, dim=0)  # type: ignore[assignment]
    return dict(room)  # type: ignore[return-value]


def iter_xy_tiles(
    xyz: torch.Tensor,
    tile_size: float,
    stride: float,
) -> Generator[Tuple[Tensor, Tensor], None, None]:
    x_max, y_max, _ = torch.max(xyz, dim=0).values
    x_min, y_min, _ = torch.min(xyz, dim=0).values
    num_block_x = abs(math.ceil((x_max - tile_size) / stride)) + 1
    num_block_y = abs(math.ceil((y_max - tile_size) / stride)) + 1

    x_starts, y_starts = np.meshgrid(np.arange(num_block_x), np.arange(num_block_y))
    x_starts = torch.from_numpy(x_starts.flatten() * stride) + x_min
    y_starts = torch.from_numpy(y_starts.flatten() * stride) + y_min
    indices = torch.arange(xyz.size(0))

    # Collect blocks
    for x_start, y_start in zip(x_starts, y_starts):
        x_cond = (xyz[:, 0] >= x_start) & (xyz[:, 0] <= x_start + tile_size)
        y_cond = (xyz[:, 1] >= y_start) & (xyz[:, 1] <= y_start + tile_size)
        cond = x_cond & y_cond

        yield xyz[cond], indices[cond]


class S3DIS(Dataset):
    mirror = "https://cvg-data.inf.ethz.ch/s3dis/"
    resources = [
        "ReadMe.txt",
        "Stanford3dDataset_v1.2_Aligned_Version.zip",
    ]
    classes = [
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

    def __init__(
        self,
        root: PATH_LIKE,
        areas: Union[List[str], Literal["all"]] = "all",
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None,
        download: bool = False,
    ) -> None:
        super().__init__()
        self.root = Path(root).as_posix()
        self.areas = areas if areas != "all" else ["Area_1", "Area_2", "Area_3", "Area_4", "Area_5", "Area_6"]
        self.transform = transform
        self.pre_transform = pre_transform
        self.pre_filter = pre_filter

        if download:
            self.download()

        self.process()
        self.data = self._load_processed_data()

    @property
    def data_dir(self) -> str:
        return Path(self.root, self.__class__.__name__).as_posix()

    @property
    def raw_dir(self) -> str:
        return Path(self.data_dir, "raw").as_posix()

    @property
    def processed_dir(self) -> str:
        return Path(self.data_dir, "processed").as_posix()

    def _check_raw_files_exists(self) -> bool:
        for area in self.areas:
            if not Path(self.raw_dir, area).is_dir():
                for room_dir in Path(self.raw_dir, area).iterdir():
                    if not any(room_dir.rglob("Annotations/*.txt")):
                        return False
                return False
        return True

    def download(self) -> None:
        if self._check_raw_files_exists():
            return

        for resource in self.resources:
            url = urljoin(self.mirror, resource)
            out_path = download_file(url, self.raw_dir)
            if resource.endswith(".zip"):
                extract_zip(out_path, self.raw_dir, relative_to=Path(resource).stem)

        # clean up files stored in the archive
        for ds_store in Path(self.raw_dir).rglob(".DS_Store"):
            ds_store.unlink()

        for icon in Path(self.raw_dir).rglob("Icon"):
            icon.unlink()

        file_path = Path(self.raw_dir, "/Area_5/office_36/Annotations/wall_3 (1).txt")
        if file_path.exists():
            file_path.unlink()

        # fix corrupted file(s)
        file_path = Path(self.raw_dir, "/Area_5/hallway_6/Annotations/ceiling_1.txt")
        with open(file_path, "r") as f:
            lines = f.readlines()
            lines = [line.replace("\00", " ") for line in lines]
        with open(file_path, "w") as f:
            f.writelines(lines)

    def _check_processed_files_exists(self) -> bool:
        for area in self.areas:
            if not Path(self.processed_dir, f"{area}.pt").exists():
                return False
        return True

    def process(self) -> None:
        if self._check_processed_files_exists():
            return

        tile_size = 1
        stride = 0.5

        num_rooms = sum(len(list(Path(self.raw_dir, area).iterdir())) for area in self.areas)
        pbar = tqdm(self.areas, total=num_rooms, desc="Processing")
        for area in pbar:
            tiles = []
            for room_dir in Path(self.raw_dir, area).iterdir():
                # TODO: load alignment angle from txt
                room = load_s3dis_room(room_dir, with_semantic=True, with_xyz=True)
                x_max, y_max, z_max = torch.max(room["xyz"], dim=0).values

                for tile_xyz, tile_idxs in iter_xy_tiles(room["xyz"], tile_size=tile_size, stride=stride):
                    if len(tile_xyz) < 100:
                        continue

                    tile_data = {"xyz": tile_xyz}
                    for key in set(room.keys()) - set(tile_data.keys()):
                        tile_data[key] = room[key][tile_idxs]  # type: ignore[literal-required]

                    tile_data, _ = F.random_sample_data(tile_data, num_samples=4096)
                    tile_data["xyz_norm"] = tile_data["xyz"] / torch.tensor([x_max, y_max, z_max])
                    x_min, y_min, _ = torch.min(tile_data["xyz"], dim=0).values
                    delta = tile_size / 2
                    tile_data["xyz_shifted"] = tile_data["xyz"] - torch.tensor([x_min + delta, y_min + delta, 0])
                    tiles.append(tile_data)

                pbar.update(1)

            out_path = Path(self.processed_dir, f"{area}.pt")
            out_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(tiles, out_path)

    def _load_processed_data(self) -> List[Dict[str, Tensor]]:
        data = []
        for area in self.areas:
            file_path = Path(self.processed_dir, f"{area}.pt")
            data.extend(torch.load(file_path))
        return data

    def __getitem__(self, index: int) -> Any:
        data = self.data[index]
        if self.transform is not None:
            data = self.transform(data)
        return data

    def __len__(self) -> int:
        return len(self.data)


# class S3DIS(Dataset):
#     def __init__(
#         self,
#         split="train",
#         data_root="trainval_fullarea",
#         num_point=4096,
#         test_area=5,
#         block_size=1.0,
#         sample_rate=1.0,
#         transform=None,
#     ):
#         super().__init__()
#         self.num_point = num_point
#         self.block_size = block_size
#         self.transform = transform
#         rooms = sorted(os.listdir(data_root))
#         rooms = [room for room in rooms if "Area_" in room]
#         if split == "train":
#             rooms_split = [room for room in rooms if not "Area_{}".format(test_area) in room]
#         else:
#             rooms_split = [room for room in rooms if "Area_{}".format(test_area) in room]

#         self.room_points, self.room_labels = [], []
#         self.room_coord_min, self.room_coord_max = [], []
#         num_point_all = []
#         labelweights = np.zeros(13)

#         for room_name in tqdm(rooms_split, total=len(rooms_split)):
#             room_path = os.path.join(data_root, room_name)
#             room_data = np.load(room_path)  # xyzrgbl, N*7
#             points, labels = room_data[:, 0:6], room_data[:, 6]  # xyzrgb, N*6; l, N
#             tmp, _ = np.histogram(labels, range(14))
#             labelweights += tmp
#             coord_min, coord_max = np.amin(points, axis=0)[:3], np.amax(points, axis=0)[:3]
#             self.room_points.append(points), self.room_labels.append(labels)
#             self.room_coord_min.append(coord_min), self.room_coord_max.append(coord_max)
#             num_point_all.append(labels.size)
#         labelweights = labelweights.astype(np.float32)
#         labelweights = labelweights / np.sum(labelweights)
#         self.labelweights = np.power(np.amax(labelweights) / labelweights, 1 / 3.0)
#         print(self.labelweights)
#         sample_prob = num_point_all / np.sum(num_point_all)
#         num_iter = int(np.sum(num_point_all) * sample_rate / num_point)
#         room_idxs = []
#         for index in range(len(rooms_split)):
#             room_idxs.extend([index] * int(round(sample_prob[index] * num_iter)))
#         self.room_idxs = np.array(room_idxs)
#         print("Totally {} samples in {} set.".format(len(self.room_idxs), split))

#     def __getitem__(self, idx):
#         room_idx = self.room_idxs[idx]
#         points = self.room_points[room_idx]  # N * 6
#         labels = self.room_labels[room_idx]  # N
#         N_points = points.shape[0]

#         while True:
#             center = points[np.random.choice(N_points)][:3]
#             block_min = center - [self.block_size / 2.0, self.block_size / 2.0, 0]
#             block_max = center + [self.block_size / 2.0, self.block_size / 2.0, 0]
#             point_idxs = np.where(
#                 (points[:, 0] >= block_min[0])
#                 & (points[:, 0] <= block_max[0])
#                 & (points[:, 1] >= block_min[1])
#                 & (points[:, 1] <= block_max[1])
#             )[0]
#             if point_idxs.size > 1024:
#                 break

#         if point_idxs.size >= self.num_point:
#             selected_point_idxs = np.random.choice(point_idxs, self.num_point, replace=False)
#         else:
#             selected_point_idxs = np.random.choice(point_idxs, self.num_point, replace=True)

#         # normalize
#         selected_points = points[selected_point_idxs, :]  # num_point * 6
#         current_points = np.zeros((self.num_point, 9))  # num_point * 9
#         current_points[:, 6] = selected_points[:, 0] / self.room_coord_max[room_idx][0]
#         current_points[:, 7] = selected_points[:, 1] / self.room_coord_max[room_idx][1]
#         current_points[:, 8] = selected_points[:, 2] / self.room_coord_max[room_idx][2]
#         selected_points[:, 0] = selected_points[:, 0] - center[0]
#         selected_points[:, 1] = selected_points[:, 1] - center[1]
#         selected_points[:, 3:6] /= 255.0
#         current_points[:, 0:6] = selected_points
#         current_labels = labels[selected_point_idxs]
#         if self.transform is not None:
#             current_points, current_labels = self.transform(current_points, current_labels)
#         return current_points, current_labels

#     def __len__(self):
#         return len(self.room_idxs)
