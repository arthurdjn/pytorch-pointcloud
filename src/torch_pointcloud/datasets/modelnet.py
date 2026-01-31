import pickle
import shutil
from functools import partial
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, Sequence, Tuple, TypedDict, Union

import numpy as np
import torch
from torch import Tensor
from tqdm import tqdm
from typing_extensions import override

from torch_pointcloud.utils.conversion import convert_to_numpy, convert_to_tensor, ensure_tuple
from torch_pointcloud.utils.io import load_off
from torch_pointcloud.utils.misc import parallel_map
from torch_pointcloud.utils.types import PathLike

from .pointcloud import PointCloudDataset
from .utils import download_url, extract_tar, extract_zip, is_hash_valid


class ModelNetData(TypedDict):
    coords: Tensor
    faces: Tensor
    target: Tensor


def load_modelnet_data(file_path: PathLike, target: int) -> ModelNetData:
    coords, faces = load_off(file_path)
    return {
        "coords": coords,
        "faces": faces,
        "target": torch.tensor(target, dtype=torch.long),
    }


def load_modelnet_normal_resampled_data(file_path: PathLike, target: int) -> Dict[str, Tensor]:
    data = np.loadtxt(file_path, delimiter=",").astype(np.float32)

    # NOTE: we could directly return the data in numpy-format but for consistency
    # we convert it to tensors so that it is ready to be used in transforms.
    return {
        "pos": torch.from_numpy(data[:, :3]),
        "normal": torch.from_numpy(data[:, 3:]),
        "target": torch.tensor(target, dtype=torch.long),
    }


class _ModelNet(PointCloudDataset):
    data_url: str
    resource: str
    md5: str
    original_classes: Tuple[str, ...]

    def __init__(
        self,
        root: PathLike,
        train: bool = False,
        classes: Union[str, Sequence[str]] = "all",
        transform: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
        pre_transform: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
        pre_filter: Optional[Callable[[Dict[str, Any]], bool]] = None,
        download: bool = False,
        force_download: bool = False,
        force_process: bool = False,
        show_progress: bool = True,
        num_workers: Optional[int] = None,
    ) -> None:
        super().__init__(root)
        classes = self.original_classes if classes == "all" else ensure_tuple(classes)
        self.train = train
        self.classes = tuple(sorted(classes))
        self.transform = transform
        self.pre_transform = pre_transform
        self.pre_filter = pre_filter
        self.show_progress = show_progress
        self.num_workers = num_workers

        if download:
            self.download(force=force_download)

        self.process(force=force_process)

        self.data = self._load_processed_data()

    @property
    def split(self) -> str:
        return "train" if self.train else "test"

    @property
    def class_to_idx(self) -> dict[str, int]:
        return {label: target for target, label in enumerate(self.classes)}

    @override
    def raw_files_exist(self) -> bool:
        raw_files = list(Path(self.raw_dir).rglob("*.off"))
        return len(raw_files) > 0

    @override
    def processed_files_exist(self) -> bool:
        if not Path(self.processed_dir, f"{self.split}.pt").exists():
            return False
        return True

    def download(self, force: bool = False) -> None:
        if self.raw_files_exist() and not force:
            return

        url = f"{self.data_url}/{self.resource}"
        resource_path = Path(self.raw_dir, self.resource)

        if (
            not resource_path.exists()
            or not is_hash_valid(resource_path, expected_hash=self.md5, hash_type="md5")
            or force
        ):
            download_url(url, resource_path, show_progress=self.show_progress)

        extract_zip(resource_path, self.raw_dir, relative_to=resource_path.stem, show_progress=self.show_progress)

        # clean up files stored in the archive
        macosx_dir = Path(self.raw_dir, "__MACOSX")
        if macosx_dir.exists():
            shutil.rmtree(macosx_dir)

        # remove .DS_Store files
        for ds_store in Path(self.raw_dir).rglob(".DS_Store"):
            ds_store.unlink()

        # Remove the downloaded resource
        resource_path.unlink()

    def process(self, force: bool = False) -> None:
        if self.processed_files_exist() and not force:
            return
        elif not self.raw_files_exist():
            raise RuntimeError(
                f"Dataset not found at {self.raw_dir!r}. "
                f"You can download the raw dataset from {self.data_url!r}, "
                f"and extract it under {self.raw_dir!r}."
            )

        file_paths = sorted(list(Path(self.raw_dir).rglob(f"**/{self.split}/*.off")))

        pbar = tqdm(file_paths, desc="Processing", disable=not self.show_progress)
        func = partial(self._process_data, class_to_idx=self.class_to_idx)
        data_list = parallel_map(func, pbar, num_workers=self.num_workers)
        data_list = [data for data in data_list if data is not None]

        out_path = Path(self.processed_dir, f"{self.split}.pt")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(data_list, out_path)

    def _process_data(self, file_path: PathLike, class_to_idx: Dict[str, int]) -> Optional[Dict[str, Any]]:
        label = Path(file_path).parent.parent.name
        target = class_to_idx.get(label)
        if target is None:
            return None

        data: Dict[str, Any] = load_modelnet_data(file_path, target)  # type: ignore[assignment]

        if self.pre_filter is not None and not self.pre_filter(data):
            return None

        if self.pre_transform is not None:
            data = self.pre_transform(data)

        return data

    def _load_processed_data(self) -> List[Dict[str, Tensor]]:
        file_path = Path(self.processed_dir, f"{self.split}.pt")
        return torch.load(file_path, weights_only=True)

    def __getitem__(self, index: int) -> Any:
        data = self.data[index]
        if self.transform is not None:
            data = self.transform(data)
        return data

    def __len__(self) -> int:
        return len(self.data)


class ModelNet10(_ModelNet):
    """
    The ModelNet10 dataset as described in the paper
    [3D ShapeNets: A Deep Representation for Volumetric Shapes](https://people.csail.mit.edu/khosla/papers/cvpr2015_wu.pdf).

    You can download the official dataset from the [Princeton dedicated website](https://modelnet.cs.princeton.edu/).

    The ModelNet10 dataset consists of 10 classes, containing 3,991 training and 908 test examples.

    Args:
        root: Root directory where the dataset should be stored.
        train: If `True`, creates the dataset from the training set, otherwise from the test set.
        classes: The class names to include in the dataset. If `"all"`, all classes are included.
        transform: A function/transform that takes in a dictionary containing the data and returns a transformed version.
        pre_transform: A function/transform that takes in a dictionary containing the data and returns a transformed version.
        pre_filter: A function that takes in a dictionary containing the data and returns a boolean value indicating whether the data should be included in the dataset.
        download: If `True`, downloads the dataset from the internet and puts it in `root`.
        force_download: If `True`, forces to download the dataset from the internet, even if it is already downloaded.
        force_process: If `True`, forces to process the dataset, even if it is already processed.
        show_progress: If `True`, displays a progress bar of the download and processing.
        num_workers: The number of workers to use for parallel processing.

    Example:
        ```python
        from torch_pointcloud.datasets import ModelNet10

        train_dataset = ModelNet10(root="data", train=True, download=True)
        test_dataset = ModelNet10(root="data", train=False, download=True)
        ```
    """

    data_url = "http://3dvision.princeton.edu/projects/2014/3DShapeNets"
    resource = "ModelNet10.zip"
    md5 = "18f4c73879802c35aa6178f8e773a99e"

    original_classes = (
        "bathtub",
        "bed",
        "chair",
        "desk",
        "dresser",
        "monitor",
        "night_stand",
        "sofa",
        "table",
        "toilet",
    )


class ModelNet40(_ModelNet):
    """
    The ModelNet40 dataset as described in the paper
    [3D ShapeNets: A Deep Representation for Volumetric Shapes](https://people.csail.mit.edu/khosla/papers/cvpr2015_wu.pdf).

    You can download the official dataset from the [Princeton dedicated website](https://modelnet.cs.princeton.edu/).

    The ModelNet40 dataset consists of 40 classes, containing 19,686 training and 4,936 test examples.

    Args:
        root: Root directory where the dataset should be stored.
        train: If `True`, creates the dataset from the training set, otherwise from the test set.
        classes: The class names to include in the dataset. If `"all"`, all classes are included.
        transform: A function/transform that takes in a dictionary containing the data and returns a transformed version.
        pre_transform: A function/transform that takes in a dictionary containing the data and returns a transformed version.
        pre_filter: A function that takes in a dictionary containing the data and returns a boolean value indicating whether the data should be included in the dataset.
        download: If `True`, downloads the dataset from the internet and puts it in `root`.
        force_download: If `True`, forces to download the dataset from the internet, even if it is already downloaded.
        force_process: If `True`, forces to process the dataset, even if it is already processed.
        show_progress: If `True`, displays a progress bar of the download and processing.
        num_workers: The number of workers to use for parallel processing.

    Example:
        ```python
        from torch_pointcloud.datasets import ModelNet40

        train_dataset = ModelNet40(root="data", train=True, download=True)
        test_dataset = ModelNet40(root="data", train=False, download=True)
        ```
    """

    data_url = "http://modelnet.cs.princeton.edu"
    resource = "ModelNet40.zip"
    md5 = "79bcee68fdf02f581938ba15f4cdca51"

    original_classes = (
        "airplane",
        "bathtub",
        "bed",
        "bench",
        "bookshelf",
        "bottle",
        "bowl",
        "car",
        "chair",
        "cone",
        "cup",
        "curtain",
        "desk",
        "door",
        "dresser",
        "flower_pot",
        "glass_box",
        "guitar",
        "keyboard",
        "lamp",
        "laptop",
        "mantel",
        "monitor",
        "night_stand",
        "person",
        "piano",
        "plant",
        "radio",
        "range_hood",
        "sink",
        "sofa",
        "stairs",
        "stool",
        "table",
        "tent",
        "toilet",
        "tv_stand",
        "vase",
        "wardrobe",
        "xbox",
    )


class ModelNetNormalResampled(PointCloudDataset):
    data_url: str = "https://huggingface.co/datasets/Pointcept/modelnet40_normal_resampled-compressed/resolve/main"
    resource: str = "modelnet40_normal_resampled.tar.gz"
    md5: str = "46ed374f8ef5f7504ec4870298772ac2"

    def __init__(
        self,
        root: PathLike,
        variant: Literal["10", "40"],
        train: bool = False,
        classes: Union[str, Sequence[str]] = "all",
        transform: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
        pre_transform: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
        pre_filter: Optional[Callable[[Dict[str, Any]], bool]] = None,
        download: bool = False,
        force_download: bool = False,
        force_process: bool = False,
        show_progress: bool = True,
        num_workers: Optional[int] = None,
    ) -> None:
        if variant not in ("10", "40"):
            raise ValueError(f"Invalid variant {variant!r} for {self.__class__.__name__}. Must be '10' or '40'.")

        super().__init__(root)
        self.variant = variant
        self.train = train
        self.transform = transform
        self.pre_transform = pre_transform
        self.pre_filter = pre_filter
        self.show_progress = show_progress
        self.num_workers = num_workers

        if download:
            self.download(force=force_download)

        classes = self.original_classes if classes == "all" else ensure_tuple(classes)
        self.classes = tuple(sorted(classes))

        self.process(force=force_process)

        self.data = self._load_processed_data()

    @property
    def original_classes(self) -> Tuple[str, ...]:
        shape_name_path = Path(self.raw_dir, f"modelnet{self.variant}_shape_names.txt")
        original_classes = shape_name_path.read_text().splitlines()
        return tuple(sorted(original_classes))

    @property
    def split(self) -> str:
        return "train" if self.train else "test"

    @property
    def class_to_idx(self) -> dict[str, int]:
        return {label: target for target, label in enumerate(self.classes)}

    @override
    def raw_files_exist(self) -> bool:
        raw_files = list(Path(self.raw_dir).rglob("*.txt"))
        return len(raw_files) > 0

    @override
    def processed_files_exist(self) -> bool:
        if not Path(self.processed_dir, f"modelnet{self.variant}_{self.split}.dat").exists():
            return False
        return True

    def download(self, force: bool = False) -> None:
        if self.raw_files_exist() and not force:
            return

        url = f"{self.data_url}/{self.resource}"
        resource_path = Path(self.raw_dir, self.resource)

        if (
            not resource_path.exists()
            or not is_hash_valid(resource_path, expected_hash=self.md5, hash_type="md5")
            or force
        ):
            download_url(url, resource_path, show_progress=self.show_progress)

        if not is_hash_valid(resource_path, expected_hash=self.md5, hash_type="md5"):
            raise RuntimeError(
                f"File corrupted: MD5 hash mismatch for {resource_path!r}. "
                "HINT: Make sure the file was downloaded correctly."
            )

        extract_tar(resource_path, self.raw_dir, relative_to=resource_path.stem, show_progress=self.show_progress)

        # Remove the downloaded resource
        resource_path.unlink()

    def process(self, force: bool = False) -> None:
        if self.processed_files_exist() and not force:
            return
        elif not self.raw_files_exist():
            raise RuntimeError(
                f"Dataset not found at {self.raw_dir!r}. "
                f"You can download the raw dataset from {self.data_url!r}, "
                f"and extract it under {self.raw_dir!r}."
            )

        split_path = Path(self.raw_dir, f"modelnet{self.variant}_{self.split}.txt")
        file_ids = split_path.read_text().splitlines()
        file_paths = [
            Path(self.raw_dir, "_".join(file_id.split("_")[:-1]), f"{file_id}.txt")  # fmt: skip
            for file_id in file_ids
        ]

        pbar = tqdm(file_paths, desc="Processing", disable=not self.show_progress)
        func = partial(self._process_data, class_to_idx=self.class_to_idx)
        data_list = parallel_map(func, pbar, num_workers=self.num_workers)
        # Ignore None data (could have been discarded, filtered, etc. with `pre_transform` or `pre_filter`)
        data_list = [data for data in data_list if data is not None]

        out_path = Path(self.processed_dir, f"modelnet{self.variant}_{self.split}.dat")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "wb") as f:
            pickle.dump(data_list, f)

    def _process_data(self, file_path: PathLike, class_to_idx: Dict[str, int]) -> Optional[Dict[str, Any]]:
        label = Path(file_path).parent.name
        target = class_to_idx.get(label)
        if target is None:
            return None

        data = load_modelnet_normal_resampled_data(file_path, target)

        if self.pre_filter is not None and not self.pre_filter(data):
            return None

        if self.pre_transform is not None:
            data = self.pre_transform(data)

        # Ensure to convert the processed data to numpy-compatible format
        # so that it is optimized for pickle serialization.
        return convert_to_numpy(data, strict=False)

    def _load_processed_data(self) -> List[Dict[str, Any]]:
        file_path = Path(self.processed_dir, f"modelnet{self.variant}_{self.split}.dat")
        with open(file_path, "rb") as f:
            return pickle.load(f)

    def __getitem__(self, index: int) -> Any:
        data = self.data[index]

        # Convert back the processed data to tensor-compatible format
        data = convert_to_tensor(data, strict=False)

        if self.transform is not None:
            data = self.transform(data)

        return data

    def __len__(self) -> int:
        return len(self.data)
