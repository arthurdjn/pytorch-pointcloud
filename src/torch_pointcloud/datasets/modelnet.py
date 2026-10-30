"""ModelNet10/40 shape classification datasets, including normal-resampled and HDF5 variants.

{{ paper("1406.5670") }}
"""

import shutil
import zipfile
from functools import partial
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, Sequence, Tuple, Union

import h5py
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
from .utils import (
    compute_hash,
    download_url,
    extract_tar,
    extract_zip,
    is_hash_valid,
)

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
    r"""Load one ModelNet mesh from a raw OFF file.

    Args:
        file_path: Path to a raw `<class>/<split>/<model_id>.off` mesh.
        target: Label index of the shape.

    Returns:
        The mesh's `pos` $(N, 3)$ vertices, `face` $(F, 3)$ triangles and scalar `label`.
    """
    pos, face = load_off(file_path)
    return {
        DataKeys.POS: torch.from_numpy(pos).float(),
        DataKeys.FACE: torch.from_numpy(face).long(),
        DataKeys.LABEL: torch.tensor(target, dtype=torch.long),
    }


def load_modelnet_normal_resampled_data(file_path: PathLike, target: int) -> Dict[str, Tensor]:
    r"""Load one ModelNet shape from a raw normal-resampled `.txt` file.

    Args:
        file_path: Path to a raw `<class>/<model_id>.txt` shape.
        target: Label index of the shape.

    Returns:
        The shape's `pos` $(N, 3)$, `normal` $(N, 3)$ and scalar `label`.
    """
    data = np.loadtxt(file_path, delimiter=",").astype(np.float32)

    # NOTE: we could directly return the data in numpy-format but for consistency
    # we convert it to tensors so that it is ready to be used in transforms.
    return {
        DataKeys.POS: torch.from_numpy(data[:, :3]).float(),
        DataKeys.NORMAL: torch.from_numpy(data[:, 3:]).float(),
        DataKeys.LABEL: torch.tensor(target, dtype=torch.long),
    }


class _ModelNet(PointCloudDataset):
    data_url: str
    resource: str
    md5: str
    original_classes: Tuple[str, ...]

    def __init__(
        self,
        root: PathLike,
        *,
        train: bool = True,
        classes: Union[str, Sequence[str]] = "all",
        transform: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
        download: bool = False,
        force_download: bool = False,
        force_process: bool = False,
        show_progress: bool = True,
        num_workers: Optional[int] = None,
    ) -> None:
        super().__init__(root)
        classes = self.original_classes if classes == "all" else ensure_tuple(classes)
        self.train = train
        self._split = "train" if train else "test"
        self.classes = tuple(sorted(classes))
        self.transform = transform
        self.show_progress = show_progress

        if download or force_download:
            self.download(force=force_download)

        self.process(force=force_process, num_workers=num_workers, show_progress=show_progress)

        self.data = self._load_processed_data()

    @property
    def class_to_idx(self) -> dict[str, int]:
        return {label: target for target, label in enumerate(self.classes)}

    @override
    def raw_files_exist(self) -> bool:
        # The archive is removed only after a completed extraction, so its presence marks a partial raw tree.
        if Path(self.raw_dir, self.resource).exists():
            return False

        raw_files = list(Path(self.raw_dir).rglob("*.off"))
        return len(raw_files) > 0

    @property
    @override
    def cache_meta(self) -> Dict[str, Any]:
        return {**super().cache_meta, "classes": list(self.classes)}

    @property
    @override
    def cache_meta_paths(self) -> List[str]:
        return [Path(self.processed_dir, f"{self._split}.meta.json").as_posix()]

    @override
    def processed_files_exist(self) -> bool:
        return Path(self.processed_dir, f"{self._split}.pt").exists()

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

    def write_processed_files(self, num_workers: Optional[int] = None, show_progress: bool = True) -> bool:
        if not self.raw_files_exist():
            raise RuntimeError(
                f"Dataset not found at {self.raw_dir!r}. "
                f"You can download the raw dataset from {self.data_url!r}, "
                f"and extract it under {self.raw_dir!r}."
            )

        file_paths = sorted(list(Path(self.raw_dir).rglob(f"**/{self._split}/*.off")))

        pbar = tqdm(file_paths, desc="Processing", disable=not show_progress)
        func = partial(self._process_data, class_to_idx=self.class_to_idx)
        data_list = parallel_map(func, pbar, num_workers=num_workers)
        data_list = [data for data in data_list if data is not None]

        dst_path = Path(self.processed_dir, f"{self._split}.pt")
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = dst_path.with_name(dst_path.name + ".tmp")
        torch.save(data_list, tmp_path)
        tmp_path.replace(dst_path)
        return True

    def _process_data(self, file_path: PathLike, class_to_idx: Dict[str, int]) -> Optional[Dict[str, Any]]:
        label = Path(file_path).parent.parent.name
        target = class_to_idx.get(label)
        if target is None:
            return None
        return load_modelnet_data(file_path, target)

    def _load_processed_data(self) -> List[Dict[str, Tensor]]:
        file_path = Path(self.processed_dir, f"{self._split}.pt")
        # Sample dicts are keyed by the DataKeys enum, which `weights_only=True` only unpickles when allowlisted.
        with torch.serialization.safe_globals([DataKeys]):
            return torch.load(file_path, weights_only=True)

    def __getitem__(self, index: int) -> Any:
        data = dict(self.data[index])
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
        train: If `True`, loads the training set, otherwise the test set.
        classes: The class names to include in the dataset. If `"all"`, all classes are included.
        transform: A function/transform that takes in a dictionary containing the data and returns a transformed version.
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
        train: If `True`, loads the training set, otherwise the test set.
        classes: The class names to include in the dataset. If `"all"`, all classes are included.
        transform: A function/transform that takes in a dictionary containing the data and returns a transformed version.
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
    r"""The ModelNet shapes resampled to 10,000 surface points with normals, as described in the paper
    :arxiv: [PointNet++: Deep Hierarchical Feature Learning on Point Sets in a Metric Space](https://arxiv.org/abs/1706.02413).

    Each shape from ModelNet10 / ModelNet40 (see `ModelNet10` and `ModelNet40` for the mesh datasets) is
    stored as a `.txt` file of 10,000 rows with six comma-separated columns: the $xyz$ coordinates
    (normalized to the unit sphere) followed by the surface normal. This is the standard input for the
    point-based classification benchmarks (PointNet++, PointMLP, Point-MAE, ...), which typically sample
    the first $1024$ points. The `variant` selects the 10-class or 40-class label set from the same
    resampled release.

    Args:
        root: Root directory where the dataset should be stored.
        variant: The label set to load, `"10"` (ModelNet10 classes) or `"40"` (ModelNet40 classes).
        train: If `True`, loads the training set, otherwise the test set.
        classes: The class names to include in the dataset. If `"all"`, all classes are included.
        transform: A function/transform that takes in a dictionary containing the data and returns a transformed version.
        download: If `True`, downloads the dataset from the internet and puts it in `root`.
        force_download: If `True`, forces to download the dataset from the internet, even if it is already downloaded.
        force_process: If `True`, forces to process the dataset, even if it is already processed.
        show_progress: If `True`, displays a progress bar of the download and processing.
        num_workers: The number of workers to use for parallel processing.

    Shape:
        - `pos`: $(10000, 3)$ unit-sphere coordinates.
        - `normal`: $(10000, 3)$ surface normals.
        - `label`: scalar class index.

    Example:
        ```python
        from torch_pointcloud.datasets import ModelNetNormalResampled

        train_dataset = ModelNetNormalResampled(root="data", variant="40", train=True, download=True)
        test_dataset = ModelNetNormalResampled(root="data", variant="40", train=False, download=True)
        ```
    """

    data_url: str = "https://huggingface.co/datasets/Pointcept/modelnet40_normal_resampled-compressed/resolve/main"
    resource: str = "modelnet40_normal_resampled.tar.gz"
    md5: str = "46ed374f8ef5f7504ec4870298772ac2"

    def __init__(
        self,
        root: PathLike,
        *,
        variant: Literal["10", "40"],
        train: bool = True,
        classes: Union[str, Sequence[str]] = "all",
        transform: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
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
        self._split = "train" if train else "test"
        classes = self.original_classes if classes == "all" else ensure_tuple(classes)
        self.classes = tuple(sorted(classes))
        self.transform = transform
        self.show_progress = show_progress

        if download or force_download:
            self.download(force=force_download)

        self.process(force=force_process, num_workers=num_workers, show_progress=show_progress)

        self.data = self._load_processed_data()

    @property
    def original_classes(self) -> Tuple[str, ...]:
        """Class names of the variant, in the original release order."""
        return MODELNET10_CLASSES if self.variant == "10" else MODELNET40_CLASSES

    @property
    def class_to_idx(self) -> dict[str, int]:
        """Mapping from class name to label index."""
        return {label: target for target, label in enumerate(self.classes)}

    @override
    def raw_files_exist(self) -> bool:
        # The archive is removed only after a completed extraction, so its presence marks a partial raw tree.
        if Path(self.raw_dir, self.resource).exists():
            return False

        raw_files = list(Path(self.raw_dir).rglob("*.txt"))
        return len(raw_files) > 0

    @property
    @override
    def cache_meta(self) -> Dict[str, Any]:
        return {**super().cache_meta, "classes": list(self.classes)}

    @property
    @override
    def cache_meta_paths(self) -> List[str]:
        return [Path(self.processed_dir, f"modelnet{self.variant}_{self._split}.meta.json").as_posix()]

    @override
    def processed_files_exist(self) -> bool:
        return Path(self.processed_dir, f"modelnet{self.variant}_{self._split}.dat").exists()

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

    def write_processed_files(self, num_workers: Optional[int] = None, show_progress: bool = True) -> bool:
        if not self.raw_files_exist():
            raise RuntimeError(
                f"Dataset not found at {self.raw_dir!r}. "
                f"You can download the raw dataset from {self.data_url!r}, "
                f"and extract it under {self.raw_dir!r}."
            )

        split_path = Path(self.raw_dir, f"modelnet{self.variant}_{self._split}.txt")
        file_ids = split_path.read_text().splitlines()
        file_paths = [
            Path(self.raw_dir, "_".join(file_id.split("_")[:-1]), f"{file_id}.txt")  # fmt: skip
            for file_id in file_ids
        ]

        pbar = tqdm(file_paths, desc="Processing", disable=not show_progress)
        func = partial(self._process_data, class_to_idx=self.class_to_idx)
        data_list = parallel_map(func, pbar, num_workers=num_workers)
        # Ignore None data (samples whose class is outside the selected `classes` subset)
        data_list = [data for data in data_list if data is not None]

        dst_path = Path(self.processed_dir, f"modelnet{self.variant}_{self._split}.dat")
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = dst_path.with_name(dst_path.name + ".tmp")
        torch.save(data_list, tmp_path)
        tmp_path.replace(dst_path)
        return True

    def _process_data(self, file_path: PathLike, class_to_idx: Dict[str, int]) -> Optional[Dict[str, Any]]:
        label = Path(file_path).parent.name
        target = class_to_idx.get(label)
        if target is None:
            return None
        return load_modelnet_normal_resampled_data(file_path, target)

    def _load_processed_data(self) -> List[Dict[str, Any]]:
        file_path = Path(self.processed_dir, f"modelnet{self.variant}_{self._split}.dat")
        if not zipfile.is_zipfile(file_path):
            raise RuntimeError(
                f"Stale processed cache at {file_path.as_posix()!r}: it was written with pickle by an older "
                "version of this dataset. Pass force_process=True to regenerate it."
            )

        # Sample dicts are keyed by the DataKeys enum, which `weights_only=True` only unpickles when allowlisted.
        with torch.serialization.safe_globals([DataKeys]):
            return torch.load(file_path, weights_only=True)

    def __getitem__(self, index: int) -> Any:
        data = dict(self.data[index])

        if self.transform is not None:
            data = self.transform(data)

        return data

    def __len__(self) -> int:
        return len(self.data)


class ModelNet40Hdf5(PointCloudDataset):
    r"""The pre-sampled HDF5 version of ModelNet40 (`modelnet40_ply_hdf5_2048`), introduced with
    :arxiv: [PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation](https://arxiv.org/abs/1612.00593).

    Each ModelNet40 mesh is sampled to 2,048 surface points with normals, normalized to the unit
    sphere, and the pre-shuffled clouds are sharded into HDF5 files listed by `train_files.txt` /
    `test_files.txt` (9,840 train and 2,468 test clouds). This is the standard evaluation input for
    point-based classification benchmarks (DGCNN, PointMLP, PointNeXt, ...), which take the first
    $N$ points of each cloud rather than resampling.

    Each sample is a dict with the following keys:

    | Key      | Shape       | Dtype   | Description                       |
    | -------- | ----------- | ------- | --------------------------------- |
    | `pos`    | $(2048, 3)$ | float32 | Unit-sphere XYZ coordinates       |
    | `normal` | $(2048, 3)$ | float32 | Surface normals                   |
    | `label`  | scalar      | int64   | Class index (40 classes)          |

    The HDF5 shards are used directly (no separate processed cache): `force_process` is accepted for
    contract parity and is a no-op.

    Args:
        root: Root directory where the dataset should be stored.
        train: If `True`, loads the training set, otherwise the test set.
        transform: A function/transform that takes in a dictionary containing the data and returns a transformed version.
        download: If `True`, downloads the dataset from the internet and puts it in `root`.
        force_download: If `True`, forces to download the dataset from the internet, even if it is already downloaded.
        force_process: Accepted for contract parity; the HDF5 shards are used directly, so this is a no-op.
        show_progress: If `True`, displays a progress bar of the download and loading.

    Example:
        ```python
        from torch_pointcloud.datasets import ModelNet40Hdf5

        train_dataset = ModelNet40Hdf5(root="data", train=True, download=True)
        test_dataset = ModelNet40Hdf5(root="data", train=False, download=True)
        ```
    """

    data_url = "https://huggingface.co/datasets/Msun/modelnet40/resolve/main"
    resource = "modelnet40_ply_hdf5_2048.zip"
    md5 = "5ae9e37d1527ff1500e41a7e9e2d50e4"

    classes = MODELNET40_CLASSES

    def __init__(
        self,
        root: PathLike,
        *,
        train: bool = True,
        transform: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
        download: bool = False,
        force_download: bool = False,
        force_process: bool = False,
        show_progress: bool = True,
    ) -> None:
        super().__init__(root)
        self.train = train
        self._split = "train" if train else "test"
        self.transform = transform
        self.show_progress = show_progress

        if download or force_download:
            self.download(force=force_download)

        self.load()

    @property
    def split_file(self) -> str:
        """Path to the raw file list of the split."""
        return Path(self.raw_dir, f"{self._split}_files.txt").as_posix()

    def _shard_paths(self) -> List[Path]:
        # The file lists carry the release's own directory prefix (`data/modelnet40_ply_hdf5_2048/...`);
        # only the shard basenames are meaningful once extracted under `raw/`.
        lines = Path(self.split_file).read_text().splitlines()
        return [Path(self.raw_dir, Path(line.strip()).name) for line in lines if line.strip()]

    @override
    def raw_files_exist(self) -> bool:
        # The archive is removed only after a completed extraction, so its presence marks a partial raw tree.
        if Path(self.raw_dir, self.resource).exists():
            return False

        if not Path(self.split_file).exists():
            return False
        return all(path.exists() for path in self._shard_paths())

    @override
    def processed_files_exist(self) -> bool:
        return self.raw_files_exist()

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

        # Remove the downloaded resource
        resource_path.unlink()

    def load(self) -> None:
        if not self.raw_files_exist():
            raise RuntimeError(
                f"Dataset not found at {self.raw_dir!r}. "
                f"You can download the raw dataset from {self.data_url!r}, "
                f"and extract it under {self.raw_dir!r}."
            )

        pos_chunks: List[np.ndarray] = []
        normal_chunks: List[np.ndarray] = []
        label_chunks: List[np.ndarray] = []
        shard_paths = self._shard_paths()
        for shard_path in tqdm(shard_paths, total=len(shard_paths), desc="Loading", disable=not self.show_progress):
            with h5py.File(shard_path, "r") as f:
                pos_chunks.append(f["data"][:])
                normal_chunks.append(f["normal"][:])
                label_chunks.append(f["label"][:])

        self.pos = np.concatenate(pos_chunks, axis=0).astype(np.float32)
        self.normal = np.concatenate(normal_chunks, axis=0).astype(np.float32)
        labels = np.concatenate(label_chunks, axis=0).astype(np.int64)
        if labels.ndim == 2 and labels.shape[1] == 1:
            labels = labels.squeeze(1)
        self.labels = labels

    @override
    def __getitem__(self, index: int) -> Dict[str, Any]:
        # Copies detach the returned tensors from the cached numpy arrays, so in-place user edits
        # cannot corrupt the dataset across epochs.
        data: Dict[str, Any] = {
            DataKeys.POS: torch.from_numpy(self.pos[index].copy()),
            DataKeys.NORMAL: torch.from_numpy(self.normal[index].copy()),
            DataKeys.LABEL: torch.tensor(int(self.labels[index]), dtype=torch.long),
        }
        if self.transform is not None:
            data = self.transform(data)
        return data

    @override
    def __len__(self) -> int:
        return len(self.labels)


# The 32 ModelNet40 classes without a rotational symmetry, the ones the RPM-Net registration benchmark keeps.
MODELNET40_ASYMMETRIC_LABELS = (
    0, 1, 2, 3, 4, 7, 8, 11, 12, 13, 14, 16, 17, 18, 20, 21,
    22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 35, 36, 38, 39,
)  # fmt: skip

# The classes of the RPM-Net seen / unseen category split: the first 20 and the last 20 labels.
MODELNET40_SEEN_CLASSES = MODELNET40_CLASSES[:20]
MODELNET40_UNSEEN_CLASSES = MODELNET40_CLASSES[20:]


def _euler_zyx_rotation(angles: np.ndarray) -> np.ndarray:
    r"""Rotation matrix of extrinsic $z$-$y$-$x$ Euler angles in radians: a rotation about $z$, then $y$, then $x$ of
    the fixed frame."""
    (cz, cy, cx), (sz, sy, sx) = np.cos(angles), np.sin(angles)
    rz = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]])
    ry = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]])
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]])
    return rx @ ry @ rz


class ModelNet40Registration(ModelNet40Hdf5):
    r"""Synthetic registration pairs of ModelNet40 shapes, the benchmark of
    :arxiv: [RPM-Net: Robust Point Matching using Learned Features](https://arxiv.org/abs/2003.13479)
    by Yew and Lee, as GeoTransformer samples it.

    Every sample pairs two partial views of one `ModelNet40Hdf5` shape: the shape is centered in the unit sphere,
    a random rigid transform (Euler angles up to `rotation_magnitude` degrees about each axis, a translation up to
    `translation_magnitude`) separates the two copies, each copy is cropped by a random plane keeping `keep_ratio`
    of its points, randomly subsampled to `num_points`, jittered by Gaussian noise of standard deviation `noise_std`
    clipped at `noise_magnitude`, and shuffled; the normals follow their points and the rotation, without noise.
    The 8 rotationally symmetric classes are left out when `asymmetric` is set. `MODELNET40_SEEN_CLASSES` and
    `MODELNET40_UNSEEN_CLASSES` are the two halves of the benchmark's category split.

    The draws come from NumPy: with `deterministic` (the default on the test set) the generator of sample $i$ is
    seeded with $i$, so a pair is the same every time it is read and matches the published evaluation protocol;
    otherwise the seed comes from the torch generator, so training pairs follow `seed_everything` and the DataLoader
    worker seeds.

    | Key          | Shape    | Dtype   | Description                                |
    | ------------ | -------- | ------- | ------------------------------------------ |
    | `pos`        | $(N, 3)$ | float32 | Source view                                |
    | `normal`     | $(N, 3)$ | float32 | Source normals                             |
    | `ref_pos`    | $(M, 3)$ | float32 | Reference view                             |
    | `ref_normal` | $(M, 3)$ | float32 | Reference normals                          |
    | `transform`  | $(4, 4)$ | float32 | Transform of the source onto the reference |
    | `label`      | scalar   | int64   | Index of the class in `classes`            |

    Args:
        root: Root directory where the dataset should be stored.
        train: If `True`, pairs the training shapes, otherwise the test shapes.
        classes: The class names to include in the dataset. If `"all"`, all classes are included.
        asymmetric: Whether the rotationally symmetric classes are dropped.
        num_points: Number of points $N$ of each view.
        rotation_magnitude: Maximum Euler angle about each axis, in degrees.
        translation_magnitude: Maximum translation along each axis.
        keep_ratio: Fraction of the shape points each view keeps.
        noise_magnitude: Clipping value of the Gaussian jitter; `None` adds no noise.
        noise_std: Standard deviation of the Gaussian jitter.
        deterministic: Whether sample $i$ is drawn from a generator seeded with $i$; defaults to `not train`.
        transform: A function/transform that takes in a dictionary containing the data and returns a transformed
            version.
        download: If `True`, downloads the dataset from the internet and puts it in `root`.
        force_download: If `True`, forces to download the dataset from the internet, even if it is already downloaded.
        force_process: Accepted for contract parity; the HDF5 shards are used directly, so this is a no-op.
        show_progress: If `True`, displays a progress bar of the download and loading.

    Example:
        ```{.python notest}
        from torch_pointcloud.datasets import ModelNet40Registration

        dataset = ModelNet40Registration(root="data", train=False, download=True)
        sample = dataset[0]
        sample["pos"].shape, sample["ref_pos"].shape, sample["transform"].shape
        ```
    """

    def __init__(
        self,
        root: PathLike,
        *,
        train: bool = True,
        classes: Union[str, Sequence[str]] = "all",
        asymmetric: bool = True,
        num_points: int = 717,
        rotation_magnitude: float = 45.0,
        translation_magnitude: float = 0.5,
        keep_ratio: float = 0.7,
        noise_magnitude: Optional[float] = 0.05,
        noise_std: float = 0.01,
        deterministic: Optional[bool] = None,
        transform: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
        download: bool = False,
        force_download: bool = False,
        force_process: bool = False,
        show_progress: bool = True,
    ) -> None:
        names = MODELNET40_CLASSES if classes == "all" else ensure_tuple(classes)
        unknown = sorted(set(names) - set(MODELNET40_CLASSES))
        if unknown:
            raise ValueError(f"Unknown ModelNet40 classes {unknown}.")

        super().__init__(
            root,
            train=train,
            transform=transform,
            download=download,
            force_download=force_download,
            force_process=force_process,
            show_progress=show_progress,
        )
        self.num_points = num_points
        self.rotation_magnitude = rotation_magnitude
        self.translation_magnitude = translation_magnitude
        self.keep_ratio = keep_ratio
        self.noise_magnitude = noise_magnitude
        self.noise_std = noise_std
        self.deterministic = (not train) if deterministic is None else deterministic
        self.classes = tuple(sorted(names))

        labels = [MODELNET40_CLASSES.index(name) for name in self.classes]
        kept = [label for label in labels if not asymmetric or label in MODELNET40_ASYMMETRIC_LABELS]
        self.indices = np.flatnonzero(np.isin(self.labels, kept))

    @property
    def name(self) -> str:
        """The pairs are drawn from the `ModelNet40Hdf5` shards, so the two datasets share their directory."""
        return ModelNet40Hdf5.__name__

    def sample_pair(
        self, points: np.ndarray, normals: np.ndarray, rng: np.random.RandomState
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        r"""Two partial views of a shape with their normals, and the transform of the source onto the reference.

        Returns:
            The source points and normals, the reference points and normals, and the $(4, 4)$ transform.
        """
        points = points - points.mean(axis=0)
        ref_points = points / np.max(np.linalg.norm(points, axis=1))

        angles = rng.rand(3) * np.pi * self.rotation_magnitude / 180.0
        translation = rng.uniform(-self.translation_magnitude, self.translation_magnitude, 3)
        transform = np.eye(4)
        transform[:3, :3] = _euler_zyx_rotation(angles)
        transform[:3, 3] = translation
        src_points = (ref_points - translation) @ transform[:3, :3]
        src_normals = normals @ transform[:3, :3]

        views = []
        for view, view_normals in ((ref_points, normals), (src_points, src_normals)):
            num_kept = int(np.floor(view.shape[0] * self.keep_ratio + 0.5))
            phi, theta = rng.uniform(0.0, 2 * np.pi), rng.uniform(0.0, np.pi)
            plane = np.array([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)])
            kept = np.argsort(-(view @ plane))[:num_kept]
            views.append((view[kept], view_normals[kept]))
        subsets = [rng.permutation(view.shape[0])[: self.num_points] for view, _ in views]
        views = [(view[subset], view_normals[subset]) for (view, view_normals), subset in zip(views, subsets)]
        if self.noise_magnitude is not None:
            views = [
                (
                    view
                    + np.clip(
                        rng.normal(scale=self.noise_std, size=view.shape), -self.noise_magnitude, self.noise_magnitude
                    ),
                    view_normals,
                )
                for view, view_normals in views
            ]
        orders = [rng.permutation(view.shape[0]) for view, _ in views]
        (ref_view, ref_normals), (src_view, src_view_normals) = [
            (view[order], view_normals[order]) for (view, view_normals), order in zip(views, orders)
        ]
        return src_view, src_view_normals, ref_view, ref_normals, transform

    @override
    def __getitem__(self, index: int) -> Dict[str, Any]:
        shape_index = int(self.indices[index])
        seed = index if self.deterministic else int(torch.randint(0, 2**31 - 1, (1,)).item())
        rng = np.random.RandomState(seed)
        src_points, src_normals, ref_points, ref_normals, transform = self.sample_pair(
            self.pos[shape_index].astype(np.float64), self.normal[shape_index].astype(np.float64), rng
        )
        label = self.classes.index(MODELNET40_CLASSES[int(self.labels[shape_index])])
        data: Dict[str, Any] = {
            DataKeys.POS: torch.from_numpy(src_points.astype(np.float32)),
            DataKeys.NORMAL: torch.from_numpy(src_normals.astype(np.float32)),
            DataKeys.REF_POS: torch.from_numpy(ref_points.astype(np.float32)),
            DataKeys.REF_NORMAL: torch.from_numpy(ref_normals.astype(np.float32)),
            DataKeys.TRANSFORM: torch.from_numpy(transform.astype(np.float32)),
            DataKeys.LABEL: torch.tensor(label, dtype=torch.long),
        }
        if self.transform is not None:
            data = self.transform(data)
        return data

    @override
    def __len__(self) -> int:
        return len(self.indices)
