import pickle
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, Sequence, Union

import h5py
import numpy as np
import torch
from torch import Tensor
from typing_extensions import get_args, override

from torch_pointcloud.utils.conversion import convert_to_numpy, convert_to_tensor, ensure_tuple
from torch_pointcloud.utils.types import PathLike

from .pointcloud import PointCloudDataset
from .utils import download_url, extract_zip, is_hash_valid

PAPER_TITLE = (
    "Revisiting Point Cloud Classification: A New Benchmark Dataset and Classification Model on Real-World Data"
)
PAPER_URL = "https://arxiv.org/abs/1908.04616"
PAPER_AUTHORS = "Mikaela Angelina Uy, Quang-Hieu Pham, Binh-Son Hua, Duc Thanh Nguyen, Sai-Kit Yeung"
PAPER_YEAR = "2019"
PAPER_CITATION = f":arxiv: [{PAPER_TITLE}]({PAPER_URL}) by {PAPER_AUTHORS} (submitted on {PAPER_YEAR})"


ScanObjectNNSplit = Literal["main", "split1", "split2", "split3", "split4"]
ScanObjectNNVariant = Literal["augmented25_norot", "augmented25rot", "augmentedrot", "augmentedrot_scale75"]


class ScanObjectNN(PointCloudDataset):
    rf"""
    The ScanObjectNN dataset is a dataset of 3D object point clouds as described in the paper
    {PAPER_CITATION}.

    """

    data_url = "http://hkust-vgd.ust.hk/scanobjectnn/"
    resource = "h5_files.zip"
    md5 = "36876af479f9ad39abad5ebcd89038dd"

    original_classes = (
        "bag",
        "bin",
        "box",
        "cabinet",
        "chair",
        "desk",
        "display",
        "door",
        "shelf",
        "table",
        "bed",
        "pillow",
        "sink",
        "sofa",
        "toilet",
    )

    def __init__(
        self,
        root: PathLike,
        split: ScanObjectNNSplit = "main",
        background: bool = False,
        train: bool = False,
        variant: Optional[ScanObjectNNVariant] = None,
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
        valid_splits = get_args(ScanObjectNNSplit)
        if split not in valid_splits:
            raise ValueError(f"Invalid split {split!r}, expected one of {', '.join(valid_splits)}.")

        valid_variants = [None] + list(get_args(ScanObjectNNVariant))
        if variant not in valid_variants:
            raise ValueError(f"Invalid variant {variant!r}, expected one of {', '.join(valid_variants)}.")

        super().__init__(root)
        classes = self.original_classes if classes == "all" else ensure_tuple(classes)
        self.split = split
        self.background = background
        self.train = train
        self.variant = variant
        self.classes = tuple(classes)
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
    def class_to_idx(self) -> dict[str, int]:
        return {label: target for target, label in enumerate(self.classes)}

    @property
    def raw_file(self) -> str:
        dir_name = f"{self.split}"
        if self.split == "main":
            dir_name += "_split"
        if not self.background:
            dir_name += "_nobg"

        file_name_parts = ["training" if self.train else "test", "objectdataset"]
        if self.variant:
            file_name_parts.append(self.variant)

        file_name = "_".join(file_name_parts) + ".h5"
        return Path(self.raw_dir, dir_name, file_name).as_posix()

    @property
    def processed_file(self) -> str:
        raw_dir = Path(self.raw_dir).absolute()
        raw_file = Path(self.raw_file).absolute()
        processed_file = Path(self.processed_dir, raw_file.relative_to(raw_dir))
        return processed_file.with_suffix(".dat").as_posix()

    @override
    def raw_files_exist(self) -> bool:
        return Path(self.raw_file).exists()

    @override
    def processed_files_exist(self) -> bool:
        return Path(self.processed_file).exists()

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

        extract_zip(resource_path, self.raw_dir, relative_to=resource_path.stem, show_progress=self.show_progress)

        # Cleanup the downloaded zipped file
        resource_path.unlink()

    def process(self, force: bool = False) -> None:
        if self.processed_files_exist() and not force:
            return
        elif not self.raw_files_exist():
            raise RuntimeError(
                f"Dataset not found at {self.raw_dir!r}. "
                f"You can download the raw dataset from {self.data_url!r}, "
                f"and extract it under {self.raw_dir!r}.\n"
                "Please agree to the terms of use at the following link: https://forms.gle/ZZRnnmaUdwfRucoy7."
            )

        with h5py.File(self.raw_file, "r") as f:
            data = f["data"][:]
            labels = f["label"][:].tolist()

        if len(data) != len(labels):
            raise RuntimeError(
                f"Data and labels have different lengths: {len(data)} != {len(labels)}for {self.raw_file!r}, "
                f"but expected them to have the same length "
                "(expected shape (N, 2048, 3) and (N,) respectively)."
            )

        # If a subset of classes is specified, then the class_to_idx only contains
        # class -> idx mapping for indices between [0, num_classes]
        # However in the h5 files, the original class indices are used (containing ALL classes),
        # so we need to map the original class indices back to the desired indices (containing fewer classes)
        idx_to_class = {target: label for target, label in enumerate(self.original_classes)}
        class_to_idx = self.class_to_idx.copy()
        labels = map(lambda x: idx_to_class.get(x, None), labels)

        data_list = []
        for pos, label in zip(data, labels):
            data = self._process_data(pos, label, class_to_idx)
            if data is None:
                continue

            data = convert_to_numpy(data)
            data_list.append(data)

        Path(self.processed_file).parent.mkdir(parents=True, exist_ok=True)
        with open(self.processed_file, "wb") as f:
            pickle.dump(data_list, f)

    def _process_data(
        self,
        pos: np.ndarray,
        label: Optional[str],
        class_to_idx: dict[str, int],
    ) -> Optional[Dict[str, Any]]:
        if label is None:
            return None

        target = class_to_idx.get(label)
        if target is None:
            return None

        data = {"pos": torch.from_numpy(pos), "label": torch.tensor(target, dtype=torch.long)}
        if self.pre_filter is not None and not self.pre_filter(data):
            return None

        if self.pre_transform is not None:
            data = self.pre_transform(data)

        return data

    def _load_processed_data(self) -> List[Dict[str, Tensor]]:
        with open(self.processed_file, "rb") as f:
            data_list = pickle.load(f)
            return [convert_to_tensor(data, strict=False) for data in data_list]

    def __getitem__(self, index: int) -> Any:
        data = self.data[index]
        if self.transform is not None:
            data = self.transform(data)
        return data

    def __len__(self) -> int:
        return len(self.data)
