import json
import os
import os.path as osp
from pathlib import Path
from typing import Callable, Dict, List, Literal, Optional, Tuple, TypedDict, Union

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset
from torch_geometric.data import Data, InMemoryDataset, download_url, extract_zip
from torch_geometric.io import fs, read_txt_array

from torch_pointcloud.utils.io import load_json
from torch_pointcloud.utils.types import PATH_LIKE

TRANSFORM_TYPE = Callable[[Dict[str, Tensor]], Dict[str, Tensor]]
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


def aslist(value: Union[None, List, Tuple]) -> List:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


class ShapeNetData(TypedDict, total=False):
    xyz: Tensor
    normal: Tensor
    segmentation_target: Tensor
    category_target: Tensor


class ShapeNet(Dataset):
    data_url = "https://shapenet.cs.stanford.edu/media"
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
        categories: Optional[Union[SHAPENET_CATEGORY_TYPE, List[SHAPENET_CATEGORY_TYPE]]] = None,
        include_normals: bool = True,
        transform: Optional[TRANSFORM_TYPE] = None,
        target_transform: Optional[TRANSFORM_TYPE] = None,
        transforms: Optional[TRANSFORM_TYPE] = None,
        download: bool = False,
    ) -> None:
        super().__init__()

        self.root = root
        self.split = split
        self.categories = categories
        self.include_normals = include_normals
        self.transform = transform
        self.target_transform = target_transform
        self.transforms = transforms

        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError("Dataset not found. You can use download=True to download it")

        if not Path(self.processed_dir, f"{self.split}.pt").exists():
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

    def download(self) -> None:
        pass

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

            # Process file
            points = np.loadtxt(file_path, delimiter=" ")
            xyz = points[:, :3]
            normal = points[:, 3:6]
            seg_target = points[:, -1]

            data = {
                "xyz": torch.from_numpy(xyz).float(),
                "normal": torch.from_numpy(normal).float(),
                "seg_target": torch.from_numpy(seg_target).long(),
                "cat_target": category_id_to_idx[category_id],
            }
            # End process file

            # if self.pre_filter is not None and not self.pre_filter(data):
            #     continue
            # if self.pre_transform is not None:
            #     data = self.pre_transform(data)

            data_list.append(data)

    def _check_exists(self) -> bool:
        return Path(self.processed_dir).exists()

    def _load_processed_data(self) -> Tensor:
        return torch.load(Path(self.processed_dir, f"{self.split}.pt"))


class ShapeNet_original(InMemoryDataset):
    url = "https://shapenet.cs.stanford.edu/media/" "shapenetcore_partanno_segmentation_benchmark_v0_normal.zip"

    # In case `shapenet.cs.stanford.edu` is offline, try to download the data
    # from Kaggle instead (requires login):
    # https://www.kaggle.com/datasets/mitkir/shapenet/download?datasetVersionNumber=1

    category_ids = {
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

    seg_classes = {
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
        root: str,
        categories: Optional[Union[str, List[str]]] = None,
        include_normals: bool = True,
        split: str = "trainval",
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None,
        force_reload: bool = False,
    ) -> None:
        if categories is None:
            categories = list(self.category_ids.keys())
        if isinstance(categories, str):
            categories = [categories]
        assert all(category in self.category_ids for category in categories)
        self.categories = categories
        super().__init__(root, transform, pre_transform, pre_filter, force_reload=force_reload)

        if split == "train":
            path = self.processed_paths[0]
        elif split == "val":
            path = self.processed_paths[1]
        elif split == "test":
            path = self.processed_paths[2]
        elif split == "trainval":
            path = self.processed_paths[3]
        else:
            raise ValueError((f"Split {split} found, but expected either " "train, val, trainval or test"))

        self.load(path)

        assert isinstance(self._data, Data)
        self._data.x = self._data.x if include_normals else None

        self.y_mask = torch.zeros((len(self.seg_classes.keys()), 50), dtype=torch.bool)
        for i, labels in enumerate(self.seg_classes.values()):
            self.y_mask[i, labels] = 1

    @property
    def num_classes(self) -> int:
        return self.y_mask.size(-1)

    @property
    def raw_file_names(self) -> List[str]:
        return list(self.category_ids.values()) + ["train_test_split"]

    @property
    def processed_file_names(self) -> List[str]:
        cats = "_".join([cat[:3].lower() for cat in self.categories])
        return [osp.join(f"{cats}_{split}.pt") for split in ["train", "val", "test", "trainval"]]

    def download(self) -> None:
        path = download_url(self.url, self.root)
        extract_zip(path, self.root)
        os.unlink(path)
        fs.rm(self.raw_dir)
        name = self.url.split("/")[-1].split(".")[0]
        os.rename(osp.join(self.root, name), self.raw_dir)

    def process_filenames(self, filenames: List[str]) -> List[Data]:
        data_list = []
        categories_ids = [self.category_ids[cat] for cat in self.categories]
        cat_idx = {categories_ids[i]: i for i in range(len(categories_ids))}

        for name in filenames:
            cat = name.split(osp.sep)[0]
            if cat not in categories_ids:
                continue

            tensor = read_txt_array(osp.join(self.raw_dir, name))
            pos = tensor[:, :3]
            x = tensor[:, 3:6]
            y = tensor[:, -1].type(torch.long)
            data = Data(pos=pos, x=x, y=y, category=cat_idx[cat])
            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            if self.pre_transform is not None:
                data = self.pre_transform(data)
            data_list.append(data)

        return data_list

    def process(self) -> None:
        trainval = []
        for i, split in enumerate(["train", "val", "test"]):
            path = osp.join(self.raw_dir, "train_test_split", f"shuffled_{split}_file_list.json")
            with open(path, "r") as f:
                filenames = [
                    osp.sep.join(name.split("/")[1:]) + ".txt" for name in json.load(f)
                ]  # Removing first directory.
            data_list = self.process_filenames(filenames)
            if split == "train" or split == "val":
                trainval += data_list
            self.save(data_list, self.processed_paths[i])
        self.save(trainval, self.processed_paths[3])

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({len(self)}, " f"categories={self.categories})"
