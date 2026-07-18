import json
import shutil
import zipfile
from functools import partial
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch import Tensor
from tqdm import tqdm
from typing_extensions import override

from torch_pointcloud.utils.conversion import ensure_tuple
from torch_pointcloud.utils.data import DataKeys
from torch_pointcloud.utils.io import load_off
from torch_pointcloud.utils.misc import parallel_map
from torch_pointcloud.utils.types import PathLike

from .pointcloud import PointCloudDataset
from .utils import compute_hash, download_url, extract_tar, extract_zip, is_hash_valid

MODELNET10_CLASSES = (
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

MODELNET40_CLASSES = (
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


def load_modelnet_data(file_path: PathLike, target: int) -> Dict[str, Tensor]:
    pos, face = load_off(file_path)
    return {
        DataKeys.POS: torch.from_numpy(pos).float(),
        DataKeys.FACE: torch.from_numpy(face).long(),
        DataKeys.LABEL: torch.tensor(target, dtype=torch.long),
    }


def load_modelnet_normal_resampled_data(file_path: PathLike, target: int) -> Dict[str, Tensor]:
    data = np.loadtxt(file_path, delimiter=",").astype(np.float32)

    # NOTE: we could directly return the data in numpy-format but for consistency
    # we convert it to tensors so that it is ready to be used in transforms.
    return {
        DataKeys.POS: torch.from_numpy(data[:, :3]).float(),
        DataKeys.NORMAL: torch.from_numpy(data[:, 3:]).float(),
        DataKeys.LABEL: torch.tensor(target, dtype=torch.long),
    }


def _transform_name(transform: Optional[Callable[..., Any]]) -> Optional[str]:
    if transform is None:
        return None
    return f"{type(transform).__module__}.{type(transform).__qualname__}"


def _cache_meta(
    classes: Sequence[str],
    pre_transform: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]],
    pre_filter: Optional[Callable[[Dict[str, Any]], bool]],
) -> Dict[str, Any]:
    """Snapshot of the constructor parameters the processed cache content depends on."""
    return {
        "classes": list(classes),
        "pre_transform": _transform_name(pre_transform),
        "pre_filter": _transform_name(pre_filter),
    }


def _check_cache_meta(meta_path: Path, meta: Dict[str, Any]) -> None:
    """Raise if the cache metadata records different parameters; a missing meta (legacy cache) is accepted."""
    if not meta_path.exists():
        return
    cached_meta = json.loads(meta_path.read_text())
    if cached_meta != meta:
        raise RuntimeError(
            f"Processed cache at {meta_path.parent.as_posix()!r} was created with different parameters "
            f"(cached: {cached_meta}, requested: {meta}). Pass force_process=True to regenerate it."
        )


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
            force
            or not resource_path.exists()
            or not is_hash_valid(resource_path, expected_hash=self.md5, hash_type="md5")
        ):
            download_url(url, resource_path, show_progress=self.show_progress, overwrite=True)

        if not is_hash_valid(resource_path, expected_hash=self.md5, hash_type="md5"):
            raise RuntimeError(
                f"File corrupted: MD5 hash mismatch for {resource_path.as_posix()!r} "
                f"(expected {self.md5}, got {compute_hash(resource_path)})."
            )

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

        dst_path = Path(self.processed_dir, f"{self.split}.pt")
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(data_list, dst_path)
        meta = _cache_meta(self.classes, self.pre_transform, self.pre_filter)
        dst_path.with_suffix(".meta.json").write_text(json.dumps(meta))

    def _process_data(self, file_path: PathLike, class_to_idx: Dict[str, int]) -> Optional[Dict[str, Any]]:
        label = Path(file_path).parent.parent.name
        target = class_to_idx.get(label)
        if target is None:
            return None

        data = load_modelnet_data(file_path, target)

        if self.pre_filter is not None and not self.pre_filter(data):
            return None

        if self.pre_transform is not None:
            data = self.pre_transform(data)

        return data

    def _load_processed_data(self) -> List[Dict[str, Tensor]]:
        file_path = Path(self.processed_dir, f"{self.split}.pt")
        meta = _cache_meta(self.classes, self.pre_transform, self.pre_filter)
        _check_cache_meta(file_path.with_suffix(".meta.json"), meta)
        # Sample dicts are keyed by the DataKeys enum, which `weights_only=True` only unpickles when allowlisted.
        with torch.serialization.safe_globals([DataKeys]):
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

    data_url = "https://3dvision.princeton.edu/projects/2014/3DShapeNets"
    resource = "ModelNet10.zip"
    md5 = "18f4c73879802c35aa6178f8e773a99e"

    original_classes = MODELNET10_CLASSES


class ModelNet40(_ModelNet):
    """
    The ModelNet40 dataset as described in the paper
    [3D ShapeNets: A Deep Representation for Volumetric Shapes](https://people.csail.mit.edu/khosla/papers/cvpr2015_wu.pdf).

    You can download the official dataset from the [Princeton dedicated website](https://modelnet.cs.princeton.edu/).

    The ModelNet40 dataset consists of 40 classes, containing 9,843 training and 2,468 test examples.

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

    data_url = "https://modelnet.cs.princeton.edu"
    resource = "ModelNet40.zip"
    md5 = "79bcee68fdf02f581938ba15f4cdca51"

    original_classes = MODELNET40_CLASSES


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
        classes = self.original_classes if classes == "all" else ensure_tuple(classes)
        self.classes = tuple(sorted(classes))
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
        return MODELNET10_CLASSES if self.variant == "10" else MODELNET40_CLASSES

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
            force
            or not resource_path.exists()
            or not is_hash_valid(resource_path, expected_hash=self.md5, hash_type="md5")
        ):
            download_url(url, resource_path, show_progress=self.show_progress, overwrite=True)

        if not is_hash_valid(resource_path, expected_hash=self.md5, hash_type="md5"):
            raise RuntimeError(
                f"File corrupted: MD5 hash mismatch for {resource_path.as_posix()!r} "
                f"(expected {self.md5}, got {compute_hash(resource_path)})."
            )

        archive_stem = resource_path.name.removesuffix(".tar.gz").removesuffix(".tgz").removesuffix(".tar")
        extract_tar(resource_path, self.raw_dir, relative_to=archive_stem, show_progress=self.show_progress)

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

        dst_path = Path(self.processed_dir, f"modelnet{self.variant}_{self.split}.dat")
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(data_list, dst_path)
        meta = _cache_meta(self.classes, self.pre_transform, self.pre_filter)
        dst_path.with_suffix(".meta.json").write_text(json.dumps(meta))

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

        return data

    def _load_processed_data(self) -> List[Dict[str, Any]]:
        file_path = Path(self.processed_dir, f"modelnet{self.variant}_{self.split}.dat")
        if not zipfile.is_zipfile(file_path):
            raise RuntimeError(
                f"Stale processed cache at {file_path.as_posix()!r}: it was written with pickle by an older "
                "version of this dataset. Pass force_process=True to regenerate it."
            )
        meta = _cache_meta(self.classes, self.pre_transform, self.pre_filter)
        _check_cache_meta(file_path.with_suffix(".meta.json"), meta)
        # Sample dicts are keyed by the DataKeys enum, which `weights_only=True` only unpickles when allowlisted.
        with torch.serialization.safe_globals([DataKeys]):
            return torch.load(file_path, weights_only=True)

    def __getitem__(self, index: int) -> Any:
        data = self.data[index]

        if self.transform is not None:
            data = self.transform(data)

        return data

    def __len__(self) -> int:
        return len(self.data)
