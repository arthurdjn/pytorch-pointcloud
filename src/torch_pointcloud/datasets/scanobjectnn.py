from pathlib import Path
from typing import Any, Callable, Dict, Literal, Optional, Sequence, Union

import h5py
import numpy as np
import torch
from typing_extensions import get_args, override

from torch_pointcloud.utils.conversion import ensure_tuple
from torch_pointcloud.utils.types import PathLike

from .pointcloud import PointCloudDataset
from .utils import download_url, extract_zip, is_hash_valid

ScanObjectNNSplit = Literal["main", "split1", "split2", "split3", "split4"]
SCANOBJECTNN_SPLITS = get_args(ScanObjectNNSplit)

ScanObjectNNVariant = Literal["augmented25_norot", "augmented25rot", "augmentedrot", "augmentedrot_scale75"]
SCANOBJECTNN_VARIANTS = get_args(ScanObjectNNVariant)

ScanObjectNNClasses = Literal[
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
]
SCANOBJECTNN_CLASSES = get_args(ScanObjectNNClasses)


def _check_split(split: str) -> None:
    if split not in SCANOBJECTNN_SPLITS:
        raise ValueError(f"Invalid split {split!r}, expected one of {', '.join(SCANOBJECTNN_SPLITS)}.")


def _check_variant(variant: str | None) -> None:
    if variant is not None and variant not in SCANOBJECTNN_VARIANTS:
        raise ValueError(f"Invalid variant {variant!r}, expected one of {', '.join(SCANOBJECTNN_VARIANTS)}.")


def _check_classes(classes: Sequence[str]) -> None:
    for cls_name in classes:
        if cls_name not in SCANOBJECTNN_CLASSES:
            raise ValueError(f"Invalid class {cls_name!r}, expected one of {', '.join(SCANOBJECTNN_CLASSES)}.")


class ScanObjectNN(PointCloudDataset):
    r"""The ScanObjectNN dataset of 3D object point clouds, as described in the paper
    :arxiv: [Revisiting Point Cloud Classification: A New Benchmark Dataset and Classification Model on Real-World Data](https://arxiv.org/abs/1908.04616)
    by Mikaela Angelina Uy, Quang-Hieu Pham, Binh-Son Hua, Duc Thanh Nguyen, Sai-Kit Yeung (submitted on 2019).
    """

    data_url = "http://hkust-vgd.ust.hk/scanobjectnn/"
    resource = "h5_files.zip"
    md5 = "36876af479f9ad39abad5ebcd89038dd"

    original_classes = tuple(SCANOBJECTNN_CLASSES)

    def __init__(
        self,
        root: PathLike,
        split: ScanObjectNNSplit = "main",
        background: bool = False,
        train: bool = False,
        variant: Optional[ScanObjectNNVariant] = None,
        classes: Union[ScanObjectNNClasses, Sequence[ScanObjectNNClasses], Literal["all"]] = "all",
        transform: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
        download: bool = False,
        force_download: bool = False,
        force_process: bool = False,
        show_progress: bool = True,
    ) -> None:
        super().__init__(root)
        self.split = split
        self.background = background
        self.train = train
        self.variant = variant
        self.classes = tuple(self.original_classes if classes == "all" else ensure_tuple(classes))
        self.transform = transform

        _check_split(self.split)
        _check_variant(self.variant)
        _check_classes(self.classes)

        if download:
            self.download(force=force_download, show_progress=show_progress)

        self.process(force=force_process, show_progress=show_progress)
        self.load(show_progress=show_progress)

    @property
    def class_to_idx(self) -> Dict[str, int]:
        return {cls_name: cls_idx for cls_idx, cls_name in enumerate(self.classes)}

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
        return processed_file.with_suffix(".npz").as_posix()

    @override
    def raw_files_exist(self) -> bool:
        return Path(self.raw_file).exists()

    @override
    def processed_files_exist(self) -> bool:
        return Path(self.processed_file).exists()

    def download(self, force: bool = False, show_progress: bool = True) -> None:
        if self.raw_files_exist() and not force:
            return

        url = f"{self.data_url}/{self.resource}"
        resource_path = Path(self.raw_dir, self.resource)

        if (
            not resource_path.exists()
            or not is_hash_valid(resource_path, expected_hash=self.md5, hash_type="md5")
            or force
        ):
            download_url(url, resource_path, show_progress=show_progress)

        if not is_hash_valid(resource_path, expected_hash=self.md5, hash_type="md5"):
            raise RuntimeError(
                f"File corrupted: MD5 hash mismatch for {resource_path!r}. "
                "HINT: Make sure the file was downloaded correctly."
            )

        extract_zip(resource_path, self.raw_dir, relative_to=resource_path.stem, show_progress=show_progress)

        # Cleanup the downloaded zipped file
        resource_path.unlink()

    def process(self, force: bool = False, show_progress: bool = True) -> None:
        if self.processed_files_exist() and not force:
            return
        if not self.raw_files_exist():
            raise RuntimeError(
                f"Dataset not found at {self.raw_dir!r}. "
                f"You can download the raw dataset from {self.data_url!r}, "
                f"and extract it under {self.raw_dir!r}.\n"
                "Please agree to the terms of use at the following link: https://forms.gle/ZZRnnmaUdwfRucoy7."
            )

        if show_progress:
            file_name = Path(self.raw_file).absolute().relative_to(Path(self.root).absolute()).as_posix()
            print(f"Processing {file_name}...", end=" ")

        with h5py.File(self.raw_file, "r") as f:
            pos = f["data"][:]
            labels = f["label"][:]

        if len(pos) != len(labels):
            raise RuntimeError(
                f"Data and labels have different lengths: {len(pos)} != {len(labels)} for {self.raw_file!r}. "
                "Expected shapes (N, 2048, 3) and (N,) respectively."
            )

        Path(self.processed_file).parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            self.processed_file,
            pos=pos.astype(np.float32),
            label=labels.astype(np.int16),
        )

        if show_progress:
            print("Done!")

    def load(self, show_progress: bool = True) -> None:
        if show_progress:
            print(f"Loading {self.processed_file}...", end=" ")

        with np.load(self.processed_file) as f:
            pos = f["pos"]
            labels = f["label"]

        # Remap the labels to the selected classes if they differ from the original classes.
        # This step will drop samples for classes that are not in the selected subset (order matters).
        remap: np.ndarray | None = None
        if self.classes != self.original_classes:
            # we use -1 as a special class index to indicate that the sample should be dropped
            # and construct a mapping (numpy array) from the original classes to the new classes.
            remap = np.full(len(self.original_classes), -1, dtype=np.int64)
            original_to_idx = {cls_name: cls_idx for cls_idx, cls_name in enumerate(self.original_classes)}
            for cls_idx, cls_name in enumerate(self.classes):
                original_idx = original_to_idx[cls_name]
                remap[original_idx] = cls_idx

            # first, remap the labels to the new subset of classes.
            labels = remap[labels]
            # then, drop samples that are not in the selected subset (i.e. have a -1 class index)
            mask = labels >= 0
            pos = pos[mask]
            labels = labels[mask]

        self.data = [
            {
                "pos": torch.from_numpy(pos[i].copy()).float(),
                "label": torch.tensor(int(labels[i]), dtype=torch.long),
            }
            for i in range(len(pos))
        ]

        if show_progress:
            print("Done!")

    @override
    def __getitem__(self, index: int) -> Any:
        data = self.data[index]
        if self.transform is not None:
            data = self.transform(data)
        return data

    @override
    def __len__(self) -> int:
        return len(self.data)
