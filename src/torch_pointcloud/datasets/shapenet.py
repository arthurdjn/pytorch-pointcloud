import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, TypedDict, Union

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


class ShapeNetPartData(TypedDict):
    coords: torch.Tensor
    normals: torch.Tensor
    segmentation: torch.Tensor
    category: torch.Tensor


def load_shapenet_part(file_path: PathLike, category: int) -> ShapeNetPartData:
    points = np.loadtxt(file_path, delimiter=" ")
    coords = points[:, :3]
    normals = points[:, 3:6]
    segmentation = points[:, -1]

    return ShapeNetPartData(
        coords=torch.from_numpy(coords).float(),
        normals=torch.from_numpy(normals).float(),
        segmentation=torch.from_numpy(segmentation).long(),
        category=torch.tensor(category),
    )


class ShapeNetPart(Dataset):
    """The ShapeNetPart dataset as described in the original paper
    [A Scalable Active Framework for Region Annotation in 3D Shape Collections](http://web.stanford.edu/~ericyi/papers/part_annotation_16_small.pdf).

    You can download the raw dataset from https://shapenet.org/ official website.

    The ShapeNetPart dataset is a subset of the ShapeNetCore dataset.
    It contains approximately 17,000 3D shapes from 16 categories.
    Each category is annotated with 2 to 6 semantic parts.

    The dataset will be processed automatically and saved in the `ShapeNetPart/processed` directory.
    If the processed data already exists, it will be loaded from the `ShapeNetPart/processed` directory
    and processing steps will be skipped.

    Args:
        root: The root directory of the dataset, where the raw and processed data will be stored.
        split: The split to load, one of "train", "val", or "test".
        categories: The categories to load, either a list of ShapeNetPart categories or a single category.
        transform: A callable that transforms the data when retrieved from the dataset.
        pre_transform: Used to transform the data before saving it in the processed directory.
        pre_filter: Used to filter the data before saving it in the processed directory.
        process: Whether to process the raw data and save it in the processed directory.
            If `False`, the processed data will be loaded from the processed directory.
            If `True`, the raw data will be processed and saved in the processed directory,
            regardless of whether the processed data already exists.
        progress: Whether to show a progress bar during processing.
        num_workers: If specified, the number of workers to use for processing the data.
            If unspecified or `None`, the data will be processed sequentially.

    Example:
        Assuming you have downloaded the raw dataset from https://shapenet.org/,
        and extracted it under `data/ShapeNetPart/raw`, you can load the dataset as follows:

        ```python
        from torch_pointcloud.datasets import ShapeNetPart

        dataset = ShapeNetPart(
            root="data",
            split="train",
            categories=["Airplane", "Chair"],
            progress=False,
        )
        ```

        This will process the raw data and save it in the `data/ShapeNetPart/processed` directory,
        and re-running the above code will load the processed data from the `data/ShapeNetPart/processed` directory.
    """

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
        process: bool = False,
        progress: bool = True,
        num_workers: Optional[int] = None,
    ) -> None:
        super().__init__()

        if split not in ["train", "val", "test"]:
            raise ValueError(f"Invalid split: {split}. Must be one of 'train', 'val' or 'test'.")

        self.root = Path(root).as_posix()
        self.split = split
        self.categories = ensure_tuple(categories or self.category_ids.keys())
        self.transform = transform
        self.pre_filter = pre_filter
        self.pre_transform = pre_transform
        self.progress = progress
        self.num_workers = num_workers

        if not self.raw_files_exists():
            root = Path(self.root).resolve().as_posix()

            raise RuntimeError(
                f"Dataset not found at {root!r}. "
                f"You can download the raw dataset from https://shapenet.org/, "
                f"and extract it under {self.raw_dir!r}."
            )

        if process or not self.processed_file_exists():
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
        return {seg: i for i, seg in enumerate(self.seg_ids)}

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

    def processed_file_exists(self) -> bool:
        return Path(self.processed_dir, f"{self.split}.pt").exists()

    def process(self) -> None:
        split_path = Path(self.raw_dir, "train_test_split", f"shuffled_{self.split}_file_list.json")

        with open(split_path, "r") as f:
            split_files = json.load(f)

        category_id_to_idx = {self.category_ids[cat]: i for i, cat in enumerate(self.categories)}

        data_list = self._process_data_list(split_files, category_id_to_idx)
        data_list = [data for data in data_list if data is not None]

        Path(self.processed_dir).mkdir(parents=True, exist_ok=True)
        torch.save(data_list, Path(self.processed_dir, f"{self.split}.pt"))

    def _process_data_list(self, file_names: List[str], category_id_to_idx: Dict[str, int]) -> List[Dict[str, Any]]:
        pbar = tqdm(file_names, total=len(file_names), desc="Processing", disable=not self.progress)
        if self.num_workers is None:
            data_list = [self._process_data(file_name, category_id_to_idx) for file_name in pbar]
        else:
            with Parallel(n_jobs=self.num_workers) as parallel:
                data_list = parallel(delayed(self._process_data)(file_name, category_id_to_idx) for file_name in pbar)

        return [data for data in data_list if data is not None]

    def _process_data(self, file_name: str, category_id_to_idx: Dict[str, int]) -> Optional[Dict[str, Any]]:
        file_path = Path(self.raw_dir, file_name.replace("shape_data/", "")).with_suffix(".txt")

        category_id = file_path.parent.name
        category = category_id_to_idx.get(category_id)

        if category is None:
            return None

        data: Dict[str, Any] = load_shapenet_part(file_path, category)  # type: ignore[assignment]

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
