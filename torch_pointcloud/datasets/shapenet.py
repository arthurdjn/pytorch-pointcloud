import json
import shutil
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, TypedDict
from urllib.parse import urljoin

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

from torch_pointcloud.utils.types import PATH_LIKE

from .utils import download_file, extract_zip

TRANSFORM_TYPE = Callable[[Dict[str, Any]], Dict[str, Any]]
SHAPENET_CATEGORY_TYPE = Literal[
    "Airplane",
    "Bag",
    "Cap",
    "Car",
    "Chair",
    "Earphone",
    "Guitar",
    "Knife",
    "Lamp",
    "Laptop",
    "Motorbike",
    "Mug",
    "Pistol",
    "Rocket",
    "Skateboard",
    "Table",
]


class ShapeNetData(TypedDict, total=False):
    xyz: Tensor
    face: Tensor
    segmentation_target: Tensor
    category_target: Tensor


class ShapeNet(Dataset):
    data_url = "https://shapenet.cs.stanford.edu/media/"
    resources = ["shapenetcore_partanno_segmentation_benchmark_v0_normal.zip"]

    category_ids: Dict[SHAPENET_CATEGORY_TYPE, str] = {
        "Airplane": "02691156",
        "Bag": "02773838",
        "Cap": "02954340",
        "Car": "02958343",
        "Chair": "03001627",
        "Earphone": "03261776",
        "Guitar": "03467517",
        "Knife": "03624134",
        "Lamp": "03636649",
        "Laptop": "03642806",
        "Motorbike": "03790512",
        "Mug": "03797390",
        "Pistol": "03948459",
        "Rocket": "04099429",
        "Skateboard": "04225987",
        "Table": "04379243",
    }

    seg_classes: Dict[SHAPENET_CATEGORY_TYPE, List[int]] = {
        "Airplane": [0, 1, 2, 3],
        "Bag": [4, 5],
        "Cap": [6, 7],
        "Car": [8, 9, 10, 11],
        "Chair": [12, 13, 14, 15],
        "Earphone": [16, 17, 18],
        "Guitar": [19, 20, 21],
        "Knife": [22, 23],
        "Lamp": [24, 25, 26, 27],
        "Laptop": [28, 29],
        "Motorbike": [30, 31, 32, 33, 34, 35],
        "Mug": [36, 37],
        "Pistol": [38, 39, 40],
        "Rocket": [41, 42, 43],
        "Skateboard": [44, 45, 46],
        "Table": [47, 48, 49],
    }

    def __init__(
        self,
        root: PATH_LIKE,
        *,
        split: Literal["train", "val", "test"],
        categories: Optional[List[SHAPENET_CATEGORY_TYPE]] = None,
        include_normals: bool = True,
        transform: Optional[TRANSFORM_TYPE] = None,
        pre_transform: Optional[TRANSFORM_TYPE] = None,
        pre_filter: Optional[Callable[[Dict[str, Any]], bool]] = None,
        download: bool = False,
    ) -> None:
        super().__init__()

        self.root = root
        self.split = split
        self.categories = categories or list(self.category_ids.keys())
        self.include_normals = include_normals
        self.transform = transform
        self.pre_filter = pre_filter
        self.pre_transform = pre_transform

        if download:
            self.download()

        if not self._check_raw_exists():
            raise RuntimeError("Dataset not found. You can use `download=True` to download it")

        if not self._check_processed_exists():
            self.process()

        self.data = self._load_processed_data()

    @property
    def data_dir(self) -> str:
        return Path(self.root, f"{self.__class__.__name__}").as_posix()

    @property
    def raw_dir(self) -> str:
        return Path(self.data_dir, "raw").as_posix()

    @property
    def processed_dir(self) -> str:
        return Path(self.data_dir, "processed").as_posix()

    @property
    def category_to_id(self) -> Dict[str, int]:
        return {cat: i for i, cat in enumerate(self.categories)}

    @property
    def seg_to_id(self) -> Dict[str, int]:
        return {seg: i for i, seg in enumerate(self.seg_classes)}

    def download(self) -> None:
        if self._check_raw_exists():
            return

        # Download resource
        file_name = self.resources[0]
        url = urljoin(self.data_url, file_name)
        zip_path = Path(self.data_dir, file_name)
        download_file(url=url, out_path=zip_path)

        # Remove previous raw data
        shutil.rmtree(self.raw_dir, ignore_errors=True)

        # Extract content as raw directory
        extract_zip(zip_path, self.data_dir)
        extract_dir = Path(self.data_dir) / Path(file_name).stem
        extract_dir.rename(self.raw_dir)
        zip_path.unlink()

    def process(self) -> None:
        split_path = Path(self.raw_dir, "train_test_split", f"shuffled_{self.split}_file_list.json")

        with open(split_path, "r") as f:
            split_data = json.load(f)

        category_id_to_idx = {self.category_ids[cat]: i for i, cat in enumerate(self.categories)}

        data_list = []
        for file_name in split_data:
            file_path = Path(self.raw_dir, file_name.replace("shape_data/", "")).with_suffix(".txt")
            category_id = file_path.parent.name
            if category_id not in category_id_to_idx:
                continue

            points = np.loadtxt(file_path, delimiter=" ")
            xyz = points[:, :3]
            face = points[:, 3:6]
            seg_target = points[:, -1]

            data = {
                "xyz": torch.from_numpy(xyz).float(),
                "face": torch.from_numpy(face).float(),
                "segmentation_target": torch.from_numpy(seg_target).long(),
                "category_target": torch.tensor(category_id_to_idx[category_id]),
            }

            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            if self.pre_transform is not None:
                data = self.pre_transform(data)

            data_list.append(data)

        Path(self.processed_dir).mkdir(parents=True, exist_ok=True)
        torch.save(data_list, Path(self.processed_dir, f"{self.split}.pt"))

    def _check_raw_exists(self) -> bool:
        if not Path(self.raw_dir).exists():
            return False

        for category_id in self.category_ids.values():
            if not Path(self.raw_dir, category_id).exists():
                return False

            if not any(Path(self.raw_dir, category_id).rglob("*.txt")):
                return False

        return True

    def _check_processed_exists(self) -> bool:
        return Path(self.processed_dir, f"{self.split}.pt").exists()

    def _load_processed_data(self) -> Any:
        return torch.load(Path(self.processed_dir, f"{self.split}.pt"))

    def __getitem__(self, index: int) -> Dict[str, Any]:
        data = self.data[index]

        if self.transform is not None:
            data = self.transform(data)

        return data

    def __len__(self) -> int:
        return len(self.data)
