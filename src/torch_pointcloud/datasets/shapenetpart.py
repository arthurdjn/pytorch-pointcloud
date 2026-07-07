import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, TypedDict, Union

import numpy as np
import torch
from tqdm import tqdm
from typing_extensions import override

from torch_pointcloud.utils.conversion import ensure_tuple
from torch_pointcloud.utils.data import DataKeys
from torch_pointcloud.utils.misc import parallel_map
from torch_pointcloud.utils.types import PathLike

from .pointcloud import PointCloudDataset

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
    pos: np.ndarray
    normal: np.ndarray
    segment: np.ndarray


def load_shapenet_part_data(file_path: PathLike) -> Optional[ShapeNetPartData]:
    data = np.loadtxt(file_path, delimiter=" ")
    if data.shape[0] == 0:
        return None

    return ShapeNetPartData(
        pos=data[:, :3].astype(np.float32),
        normal=data[:, 3:6].astype(np.float32),
        segment=data[:, -1].astype(np.int16),
    )


class ShapeNetPart(PointCloudDataset):
    """ShapeNetPart dataset packed into per-split `.npy` files.

    Each split is packed once into:

        <processed_dir>/<split>/
            pos.npy           # float32, (total_points, 3)
            normal.npy        # float32, (total_points, 3)
            segment.npy       # int16,   (total_points,)
            offset.npy        # int64,   (n_samples + 1,)
            category.npy      # int16,   (n_samples,)

    The packed files contain every category; `categories` filters the
    sample index at load time, so changing the subset is free.

    Args:
        root: Dataset root directory.
        split: One of "train", "val", "test".
        categories: Which categories to expose. Defaults to all 16.
        transform: Callable applied to each sample in `__getitem__`.
        force_process: Re-pack raw data even if processed files exist.
        show_progress: Show a progress bar during processing.
        num_workers: Parallelism for raw-file reading during processing.
    """

    data_url = "https://shapenet.org/"

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
        force_process: bool = False,
        show_progress: bool = True,
        num_workers: Optional[int] = None,
    ) -> None:
        super().__init__(root)
        if split not in ("train", "val", "test"):
            raise ValueError(f"Invalid split: {split!r}. Must be one of 'train', 'val', 'test'.")

        self.root = Path(root).as_posix()
        self.split = split
        self.categories = ensure_tuple(categories or self.category_ids.keys())
        self.transform = transform
        self.show_progress = show_progress
        self.num_workers = num_workers

        for category in self.categories:
            if category not in self.category_ids:
                raise KeyError(f"Unknown {self.__class__.__name__} category: {category!r}")

        self.process(force=force_process, num_workers=num_workers, show_progress=show_progress)
        self.load(show_progress=show_progress)

    @override
    @property
    def data_dir(self) -> str:
        return Path(self.root, self.__class__.__name__).as_posix()

    @override
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

    @override
    def raw_files_exist(self) -> bool:
        if not Path(self.raw_dir).exists():
            return False
        if not Path(self.raw_dir, "train_test_split", f"shuffled_{self.split}_file_list.json").exists():
            return False
        for category_id in self.category_ids.values():
            cat_dir = Path(self.raw_dir, category_id)
            if not cat_dir.exists() or not any(cat_dir.rglob("*.txt")):
                return False
        return True

    @override
    def processed_files_exist(self) -> bool:
        split_dir = Path(self.processed_dir, self.split)
        file_names = ["pos.npy", "normal.npy", "segment.npy", "offset.npy", "category.npy"]
        return all((split_dir / name).exists() for name in file_names)

    def process(self, force: bool = False, num_workers: Optional[int] = None, show_progress: bool = True) -> None:
        if self.processed_files_exist() and not force:
            return
        if not self.raw_files_exist():
            raise RuntimeError(
                f"Dataset not found at {self.raw_dir!r}. "
                f"You can download it from {self.data_url!r} "
                f"and extract it under {self.raw_dir!r}."
            )

        split_path = Path(self.raw_dir, "train_test_split", f"shuffled_{self.split}_file_list.json")
        with open(split_path, "r") as f:
            split_files = json.load(f)

        samples = parallel_map(
            self._load_raw_file,
            split_files,
            num_workers=num_workers,
            total=len(split_files),
            desc="Reading",
            show_progress=show_progress,
        )
        samples = [s for s in samples if s is not None]
        if not samples:
            raise RuntimeError(f"Found no samples in split {self.split!r}.")

        split_dir = Path(self.processed_dir, self.split)
        split_dir.mkdir(parents=True, exist_ok=True)

        sizes = np.array([s["pos"].shape[0] for s in samples], dtype=np.int64)  # type: ignore[index]
        offsets = np.concatenate(([0], np.cumsum(sizes))).astype(np.int64)
        pos = np.concatenate([s["pos"] for s in samples], dtype=np.float32)  # type: ignore[index]
        normal = np.concatenate([s["normal"] for s in samples], dtype=np.float32)  # type: ignore[index]
        segment = np.concatenate([s["segment"] for s in samples], dtype=np.int16)  # type: ignore[index]
        category = np.asarray([s["category"] for s in samples], dtype=np.int16)  # type: ignore[index]

        np.save(split_dir / "pos.npy", pos)
        np.save(split_dir / "normal.npy", normal)
        np.save(split_dir / "segment.npy", segment)
        np.save(split_dir / "offset.npy", offsets)
        np.save(split_dir / "category.npy", category)

    def _load_raw_file(self, file_name: str) -> Optional[Dict[str, Any]]:
        file_path = Path(self.raw_dir, file_name.removeprefix("shape_data/")).with_suffix(".txt")
        data = load_shapenet_part_data(file_path)
        if not data:
            return None

        category_id = file_path.parent.name
        category_idx = list(self.category_ids.values()).index(category_id)
        return {
            **data,
            "category": category_idx,
        }

    def load(self, show_progress: bool = True) -> None:
        split_dir = Path(self.processed_dir, self.split)
        offsets = np.load(split_dir / "offset.npy")
        category_idxs = np.load(split_dir / "category.npy")

        category_name_to_idx = {name: i for i, name in enumerate(self.category_ids)}
        selected_category_idxs = np.array(
            [category_name_to_idx[c] for c in self.categories],
            dtype=category_idxs.dtype,
        )

        indices = np.nonzero(np.isin(category_idxs, selected_category_idxs))[0]
        pos = np.load(split_dir / "pos.npy")
        normal = np.load(split_dir / "normal.npy")
        segment = np.load(split_dir / "segment.npy")

        self.samples = [
            {
                # NOTE: We use a copy to avoid modifying the original array
                # in case a transform is modifying data in-place.
                DataKeys.POS: torch.from_numpy(pos[offsets[i] : offsets[i + 1]].copy()),
                DataKeys.NORMAL: torch.from_numpy(normal[offsets[i] : offsets[i + 1]].copy()),
                DataKeys.SEGMENT: torch.from_numpy(segment[offsets[i] : offsets[i + 1]].astype(np.int64)),
                # The category index is NOT remapped to the new selected categories subset (if specified)
                # meaning that one MUST relabel the categories if the subset changed for classification tasks.
                DataKeys.CATEGORY: torch.tensor(category_idxs[i], dtype=torch.long),
            }
            for i in tqdm(indices, total=len(indices), desc="Loading", disable=not show_progress)
        ]

    @override
    def __len__(self) -> int:
        return len(self.samples)

    @override
    def __getitem__(self, index: int) -> Dict[str, Any]:
        data: dict = self.samples[index]
        if self.transform is not None:
            data = self.transform(data)
        return data
