import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset
from tqdm import tqdm

from torch_pointcloud.utils.io import load_off
from torch_pointcloud.utils.types import PATH_LIKE

from .utils import download_file, extract_zip


class ModelNet40(Dataset):

    mirror = "https://shapenet.cs.stanford.edu/media/"
    resources = ["modelnet40_normal_resampled.zip"]

    def __init__(
        self,
        root: PATH_LIKE,
        name: Literal["10", "40"],
        split: Literal["train", "test"],
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        transforms: Optional[Callable] = None,
        download: bool = False,
    ) -> None:
        super().__init__()
        assert name in ["10", "40"], "ModelNet name must be either '10' or '40'"
        assert split in ["train", "test"], "ModelNet split must be either 'train' or 'test'"

        self.root = Path(root).as_posix()
        self.name = name
        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        self.transforms = transforms

        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError("Dataset not found. You can use download=True to download it")

        if not Path(self.processed_dir, f"{self.split}.pt").exists():
            self.process()

        self.classes = self._load_classes()
        self.data = self._load_processed_data()

    @property
    def data_dir(self) -> str:
        return Path(self.root, f"{self.__class__.__name__}{self.name}").as_posix()

    @property
    def raw_dir(self) -> str:
        return Path(self.data_dir, "raw").as_posix()

    @property
    def processed_dir(self) -> str:
        return Path(self.data_dir, "processed").as_posix()

    @property
    def class_to_idx(self) -> dict[str, int]:
        return {label: target for target, label in enumerate(self.classes)}

    def _load_classes(self) -> list[str]:
        # Text file containing the list of the 10/40 classes
        file_path = Path(self.raw_dir, f"modelnet{self.name}_shape_names.txt")
        with open(file_path, "r") as f:
            return sorted(f.read().splitlines())

    def _load_object_parts(self) -> list[str]:
        # Text file containing the list of the objects from the 10/40 classes
        file_path = Path(self.raw_dir, f"modelnet{self.name}_{self.split}.txt")
        with open(file_path, "r") as f:
            return f.read().splitlines()

    def process(self) -> None:
        classes = self._load_classes()
        object_parts = self._load_object_parts()

        # Group the objects by class
        class_name_to_object_paths = defaultdict(list)
        for object_part in object_parts:
            class_name, _ = object_part.rsplit("_", 1)
            object_path = Path(self.raw_dir, class_name, f"{object_part}.txt")
            class_name_to_object_paths[class_name].append(object_path)

        # Make sure that objects are available for all classes
        assert len(classes) == len(class_name_to_object_paths), "Number of classes and objects do not match"

        # Load and stack all the 3D point clouds of each object
        data_list = []
        for target, (class_name, object_paths) in enumerate(class_name_to_object_paths.items()):
            for object_path in object_paths:
                raw = np.loadtxt(object_path, dtype=np.float32, delimiter=",")
                data = dict(
                    pos=torch.tensor(raw[:, :3]),
                    features=torch.tensor(raw[:, 3:]),
                    target=torch.tensor([target]),
                )
                data_list.append(data)

        out_path = Path(self.processed_dir, f"{self.split}.pt")
        torch.save(data_list, out_path)

    def _load_processed_data(self) -> List[Dict[str, Tensor]]:
        file_path = Path(self.processed_dir, f"{self.split}.pt")
        return torch.load(file_path)

    def _check_exists(self) -> bool:
        for resource in self.resources:
            if not Path(self.raw_dir, resource).exists():
                return False
        return True

    def download(self) -> None:
        if self._check_exists():
            return

        for resource in self.resources:
            url = f"{self.mirror}/{resource}"
            out_path = download_file(url, self.raw_dir)
            extract_zip(out_path, self.raw_dir, relative_to=Path(resource).stem)

    def __getitem__(self, index: int) -> Any:
        data = self.data[index]
        if self.transforms is not None:
            data = self.transforms(data)
        return data

    def __len__(self) -> int:
        return len(self.data)


class ModelNet10(Dataset):

    mirror = "http://vision.princeton.edu/projects/2014/3DShapeNets"
    resource = "ModelNet10.zip"

    classes = ("bathtub", "bed", "chair", "desk", "dresser", "monitor", "night_stand", "sofa", "table", "toilet")

    def __init__(
        self,
        root: PATH_LIKE,
        train: bool = False,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None,
        download: bool = False,
    ) -> None:
        super().__init__()
        self.root = Path(root).as_posix()
        self.train = train
        self.transform = transform
        self.pre_transform = pre_transform
        self.pre_filter = pre_filter

        if download:
            self.download()
        self.process()

        self.data = self._load_processed_data()

    @property
    def split(self) -> str:
        return "train" if self.train else "test"

    @property
    def data_dir(self) -> str:
        return Path(self.root, self.__class__.__name__).as_posix()

    @property
    def raw_dir(self) -> str:
        return Path(self.data_dir, "raw").as_posix()

    @property
    def processed_dir(self) -> str:
        return Path(self.data_dir, "processed").as_posix()

    @property
    def class_to_idx(self) -> dict[str, int]:
        return {label: target for target, label in enumerate(self.classes)}

    def process(self) -> None:
        if self._check_processed_files_exists():
            return

        data_list = []
        files = list(Path(self.raw_dir).rglob(f"**/{self.split}/*.off"))
        for off_path in tqdm(files, desc="Processing"):
            label = off_path.parent.parent.name
            target = self.class_to_idx[label]
            xyz, faces = load_off(off_path)
            data = {
                "xyz": xyz.float(),
                "face": faces.long(),
                "target": torch.tensor([target]),
            }

            if self.pre_filter is not None and not self.pre_filter(data):
                continue

            if self.pre_transform is not None:
                data = self.pre_transform(data)

            data_list.append(data)

        out_path = Path(self.processed_dir, f"{self.split}.pt")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(data_list, out_path)

    def _load_processed_data(self) -> List[Dict[str, Tensor]]:
        file_path = Path(self.processed_dir, f"{self.split}.pt")
        return torch.load(file_path)

    def _check_raw_files_exists(self) -> bool:
        raw_files = list(Path(self.raw_dir).rglob("*.off"))
        return len(raw_files) == 4899

    def _check_processed_files_exists(self) -> bool:
        if not Path(self.processed_dir, f"{self.split}.pt").exists():
            return False
        return True

    def download(self) -> None:
        if self._check_raw_files_exists():
            return

        url = f"{self.mirror}/{self.resource}"
        print(f"Downloading {url!r} to {self.raw_dir!r}...")
        out_path = download_file(url, self.raw_dir)
        extract_zip(out_path, self.raw_dir, relative_to=Path(self.resource).stem)

        # clean up files stored in the archive
        macosx_dir = Path(self.raw_dir, "__MACOSX")
        if macosx_dir.exists():
            shutil.rmtree(macosx_dir)
        # remove .DS_Store files
        for ds_store in Path(self.raw_dir).rglob(".DS_Store"):
            ds_store.unlink()

    def __getitem__(self, index: int) -> Any:
        data = self.data[index]
        if self.transform is not None:
            data = self.transform(data)
        return data

    def __len__(self) -> int:
        return len(self.data)
