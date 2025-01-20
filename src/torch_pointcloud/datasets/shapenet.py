import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, Union

import numpy as np
import torch
from joblib import Parallel, delayed
from torch.utils.data import Dataset
from tqdm import tqdm

from torch_pointcloud.utils.conversion import ensure_tuple
from torch_pointcloud.utils.types import PathLike

TransformLike = Callable[[Dict[str, Any]], Dict[str, Any]]
ShapeNetCategory = Literal[
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


class ShapeNet(Dataset):
    category_ids: Dict[ShapeNetCategory, str] = {
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

    seg_ids: Dict[ShapeNetCategory, List[int]] = {
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
        root: PathLike,
        *,
        split: Literal["train", "val", "test"],
        categories: Optional[Union[List[ShapeNetCategory], ShapeNetCategory]] = None,
        transform: Optional[TransformLike] = None,
        pre_transform: Optional[TransformLike] = None,
        pre_filter: Optional[Callable[[Dict[str, Any]], bool]] = None,
        progress: bool = True,
        num_workers: int = -1,
    ) -> None:
        super().__init__()

        if split not in ["train", "val", "test"]:
            raise ValueError(f"Invalid split: {split}. Must be one of 'train', 'val' or 'test'.")

        self.root = root
        self.split = split
        self.categories = ensure_tuple(categories or self.category_ids.keys())
        self.transform = transform
        self.pre_filter = pre_filter
        self.pre_transform = pre_transform
        self.progress = progress
        self.num_workers = num_workers

        if not self.raw_files_exists:
            raise RuntimeError(
                "Dataset not found. You can download the raw dataset from https://shapenet.org/, "
                f"and extract it under {self.raw_dir!r}."
            )

        if not self.processed_file_exists:
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
    def raw_files_exists(self) -> bool:
        if not Path(self.raw_dir).exists():
            return False

        if not Path(self.raw_dir, "train_test_split", f"shuffled_{self.split}_file_list.json").exists():
            return False

        for category_id in self.category_ids.values():
            if not Path(self.raw_dir, category_id).exists():
                return False

            if not any(Path(self.raw_dir, category_id).rglob("*.txt")):
                return False

        return True

    @property
    def processed_file_exists(self) -> bool:
        return Path(self.processed_dir, f"{self.split}.pt").exists()

    @property
    def category_to_id(self) -> Dict[str, int]:
        return {cat: i for i, cat in enumerate(self.categories)}

    @property
    def seg_to_id(self) -> Dict[str, int]:
        return {seg: i for i, seg in enumerate(self.seg_ids)}

    def process(self) -> None:
        split_path = Path(self.raw_dir, "train_test_split", f"shuffled_{self.split}_file_list.json")

        with open(split_path, "r") as f:
            split_data = json.load(f)

        category_id_to_idx = {self.category_ids[cat]: i for i, cat in enumerate(self.categories)}

        with Parallel(n_jobs=self.num_workers) as parallel:
            data_list = parallel(
                delayed(self._process_data)(file_name, category_id_to_idx)
                for file_name in tqdm(split_data, total=len(split_data), desc="Processing", disable=not self.progress)
            )
            data_list = [data for data in data_list if data is not None]

        Path(self.processed_dir).mkdir(parents=True, exist_ok=True)
        torch.save(data_list, Path(self.processed_dir, f"{self.split}.pt"))

    def _process_data(self, file_name: str, category_id_to_idx: Dict[str, int]) -> Optional[Dict[str, Any]]:
        file_path = Path(self.raw_dir, file_name.replace("shape_data/", "")).with_suffix(".txt")
        category_id = file_path.parent.name

        if category_id not in category_id_to_idx:
            return None

        points = np.loadtxt(file_path, delimiter=" ")
        coords = points[:, :3]
        normals = points[:, 3:6]
        seg_target = points[:, -1]

        data = {
            "coords": torch.from_numpy(coords).float(),
            "normals": torch.from_numpy(normals).float(),
            "segmentation_target": torch.from_numpy(seg_target).long(),
            "category_target": torch.tensor(category_id_to_idx[category_id]),
        }

        if self.pre_filter is not None and not self.pre_filter(data):
            return None

        if self.pre_transform is not None:
            data = self.pre_transform(data)

        return data

    def _load_processed_data(self) -> Any:
        return torch.load(Path(self.processed_dir, f"{self.split}.pt"), weights_only=True)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        data = self.data[index]

        if self.transform is not None:
            data = self.transform(data)

        return data

    def __len__(self) -> int:
        return len(self.data)
