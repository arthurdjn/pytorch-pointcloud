r"""XCube ShapeNet voxel dataset.

Pre-voxelized ShapeNet shapes (chair, car, airplane) with per-voxel normals, released with
:arxiv: [XCube](https://arxiv.org/abs/2312.03806) at resolutions $128^3$ (coarse stage) and $512^3$
(fine stage). Download the raw data from
:github: [xrenaa/XCube-Shapenet-Dataset](https://huggingface.co/datasets/xrenaa/XCube-Shapenet-Dataset).

The raw `.pkl` files contain grids pickled by the 2024 fvdb build XCube was developed against.
`load_xcube_shape` reads one shape without that dependency (a stub unpickler plus a NanoVDB file-header
shim readable by fvdb-core); `process()` runs it over a split and caches plain `ijk` / `normal` arrays
per shape.
"""

import io
import pickle
import struct
import tempfile
import types
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Literal, Optional, Tuple, TypedDict, Union

import numpy as np
import torch
from torch import Tensor
from tqdm import tqdm
from typing_extensions import override

from torch_pointcloud.utils.conversion import ensure_tuple
from torch_pointcloud.utils.data import DataKeys
from torch_pointcloud.utils.imports import _FVDB_GITHUB_URL, optional_import
from torch_pointcloud.utils.types import PathLike

from .pointcloud import PointCloudDataset

if TYPE_CHECKING:
    from fvdb import GridBatch
else:
    GridBatch, _ = optional_import("fvdb", "GridBatch", url=_FVDB_GITHUB_URL)

TransformLike = Callable[[Dict[str, Any]], Dict[str, Any]]
XCubeShapeNetCategory = Literal["Airplane", "Car", "Chair"]


class XCubeShapeNetData(TypedDict):
    ijk: Tensor
    normal: Tensor


class _StubState:
    state: Any = None

    def __setstate__(self, state: Any) -> None:
        self.state = state


class _XCubeUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str) -> Any:
        if module.startswith("fvdb"):
            return type(name, (_StubState,), {})
        if module.partition(".")[0] not in ("torch", "numpy", "collections"):
            raise pickle.UnpicklingError(f"Unpickling {module}.{name} is not allowed in XCube shape files.")
        return super().find_class(module, name)


_PICKLE_MODULE = types.ModuleType("_xcube_pickle")
_PICKLE_MODULE.Unpickler = _XCubeUnpickler  # type: ignore[attr-defined]


def _read_xcube_pickle(path: PathLike) -> Tuple[np.ndarray, np.ndarray]:
    """Extract the serialized grid buffer and the per-voxel normals from a raw XCube `.pkl` file."""
    with open(path, "rb") as f:
        data = torch.load(f, map_location="cpu", weights_only=False, pickle_module=_PICKLE_MODULE)
    grid_buffer = data["points"].state.numpy()
    jdata, _, _ = data["normals"].state
    return grid_buffer, jdata.numpy()


def _grid_buffer_to_nanovdb(buffer: np.ndarray) -> bytes:
    r"""Wrap the in-memory NanoVDB grid of an old-fvdb `GridBatch` pickle into a `.nvdb` file image.

    The old serialization is a small header (batch and grid metadata) followed by the raw in-memory
    NanoVDB grid. A NanoVDB *file* needs a `FileHeader` and one `FileMetaData` per grid, whose fields are
    all recoverable from the `GridData` / `TreeData` / `RootData` structs at fixed offsets.
    """
    raw = buffer.tobytes()
    offset = raw.find(b"NanoVDB")
    if offset < 0:
        raise ValueError("Not an XCube fvdb grid buffer (no NanoVDB magic found).")
    grid = raw[offset:]
    grid_size = struct.unpack_from("<Q", grid, 32)[0]
    grid = grid[:grid_size]

    version = struct.unpack_from("<I", grid, 16)[0]
    grid_class = struct.unpack_from("<I", grid, 632)[0]
    grid_type = struct.unpack_from("<I", grid, 636)[0]
    world_bbox = struct.unpack_from("<6d", grid, 560)
    voxel_size = struct.unpack_from("<3d", grid, 608)
    node_count = struct.unpack_from("<3I", grid, 672 + 32)
    tile_count = struct.unpack_from("<3I", grid, 672 + 44)
    voxel_count = struct.unpack_from("<Q", grid, 672 + 56)[0]
    index_bbox = struct.unpack_from("<6i", grid, 672 + 64)

    header = struct.pack("<QIHH", struct.unpack_from("<Q", grid, 0)[0], version, 1, 0)
    metadata = struct.pack("<QQQQ", grid_size, grid_size, 0, voxel_count)
    metadata += struct.pack("<II", grid_type, grid_class)
    metadata += struct.pack("<6d", *world_bbox)
    metadata += struct.pack("<6i", *index_bbox)
    metadata += struct.pack("<3d", *voxel_size)
    metadata += struct.pack("<I", 0)
    metadata += struct.pack("<4I", node_count[0], node_count[1], node_count[2], 1)
    metadata += struct.pack("<3I", *tile_count)
    metadata += struct.pack("<HH", 0, 0)
    metadata += struct.pack("<I", version)
    return header + metadata + grid


def _grid_buffer_to_ijk(buffer: np.ndarray) -> np.ndarray:
    with tempfile.NamedTemporaryFile(suffix=".nvdb") as f:
        f.write(_grid_buffer_to_nanovdb(buffer))
        f.flush()
        grid = GridBatch.from_nanovdb(f.name)[0]
    ijk: np.ndarray = grid.ijk.jdata.cpu().numpy()
    return ijk


def load_xcube_shape(file_path: PathLike) -> XCubeShapeNetData:
    r"""Load one pre-voxelized XCube ShapeNet shape from a raw release `.pkl` file.

    The release pickles were produced by the 2024 fvdb build XCube was developed against, so they
    reference `fvdb.GridBatch` / `fvdb.JaggedTensor` classes that fvdb-core cannot deserialize. This
    reader extracts their raw pickled state with a stub unpickler (the original fvdb is not required) and
    rebuilds the grid by wrapping its in-memory NanoVDB buffer in a `.nvdb` file image that fvdb-core
    parses.

    Args:
        file_path: Path to a raw `<model_id>.pkl` shape.

    Returns:
        The shape's integer voxel coordinates `ijk` and unit per-voxel `normal`. World coordinates are
        $\text{pos} = (\text{ijk} + 0.5) \cdot \text{voxel\_size}$ with $\text{voxel\_size} = 1.28 /
        \text{resolution}$.

    Shape:
        - `ijk`: $(N, 3)$
        - `normal`: $(N, 3)$

    Example:
        ```python
        from torch_pointcloud.datasets.shapenet import load_xcube_shape

        data = load_xcube_shape("data/XCubeShapeNet/raw/128/03001627/1006be65e7bc937e9141f9b58470d646.pkl")
        pos = (data["ijk"] + 0.5) * (1.28 / 128)
        ```
    """
    grid_buffer, normal = _read_xcube_pickle(file_path)
    ijk = _grid_buffer_to_ijk(grid_buffer)
    return XCubeShapeNetData(ijk=torch.from_numpy(ijk), normal=torch.from_numpy(normal))


class XCubeShapeNet(PointCloudDataset):
    r"""XCube pre-voxelized ShapeNet shapes with per-voxel normals.

    Each sample is one shape voxelized at `resolution` per axis: voxel centers as `pos` and unit
    normals as `normal`. Coordinates live in $[-0.64, 0.64]$ (voxel size $1.28 / \text{resolution}$,
    grid origin at half a voxel).

    Expected raw layout (the Hugging Face release):

    ```text
    <root>/XCubeShapeNet/raw/
        128/<synset_id>/<model_id>.pkl  + train.lst / val.lst / test.lst
        512/<synset_id>/<model_id>.pkl  + train.lst / val.lst / test.lst
    ```

    Args:
        root: Dataset root directory.
        split: One of `"train"`, `"val"`, `"test"`.
        resolution: Voxel resolution per axis, $128$ or $512$.
        categories: Categories to expose, defaults to all three.
        transform: Callable applied to each sample in `__getitem__`.
        voxel_size: Edge length of one voxel; defaults to $1.28 / \text{resolution}$.
        force_process: Re-extract raw pickles even if the processed cache exists.
        show_progress: Show a progress bar during processing.

    Example:
        ```python
        dataset = XCubeShapeNet("data", split="test", resolution=128, categories="Chair")
        sample = dataset[0]
        sample[DataKeys.POS].shape, sample[DataKeys.NORMAL].shape
        ```
    """

    data_url = "https://huggingface.co/datasets/xrenaa/XCube-Shapenet-Dataset"

    category_ids: Dict[XCubeShapeNetCategory, str] = {
        "Airplane": "02691156",
        "Car": "02958343",
        "Chair": "03001627",
    }

    def __init__(
        self,
        root: PathLike,
        *,
        split: Literal["train", "val", "test"],
        resolution: Literal[128, 512] = 128,
        categories: Optional[Union[List[XCubeShapeNetCategory], XCubeShapeNetCategory]] = None,
        transform: Optional[TransformLike] = None,
        voxel_size: Optional[float] = None,
        force_process: bool = False,
        show_progress: bool = True,
    ) -> None:
        super().__init__(root)
        if split not in ("train", "val", "test"):
            raise ValueError(f"Invalid split: {split!r}. Must be one of 'train', 'val', 'test'.")
        if resolution not in (128, 512):
            raise ValueError(f"Invalid resolution: {resolution!r}. Must be 128 or 512.")

        self.split = split
        self.resolution = resolution
        self.categories = ensure_tuple(categories or self.category_ids.keys())
        self.transform = transform
        self.voxel_size = voxel_size if voxel_size is not None else 1.28 / resolution
        self.show_progress = show_progress

        for category in self.categories:
            if category not in self.category_ids:
                raise KeyError(f"Unknown {self.__class__.__name__} category: {category!r}")

        self.process(force=force_process, show_progress=show_progress)
        self.files = self._index_files()

    def _split_models(self, synset_id: str) -> List[str]:
        """Models of the split that exist on disk (a few `.lst` entries are absent from the release)."""
        split_file = Path(self.raw_dir, str(self.resolution), synset_id, f"{self.split}.lst")
        models = [m for m in split_file.read_text().splitlines() if m]
        raw_dir = Path(self.raw_dir, str(self.resolution), synset_id)
        processed_dir = Path(self.processed_dir, str(self.resolution), synset_id)
        available = [m for m in models if (raw_dir / f"{m}.pkl").exists() or (processed_dir / f"{m}.npz").exists()]
        if len(available) < len(models):
            warnings.warn(
                f"{len(models) - len(available)} of {len(models)} shapes listed in {split_file.name!r} for "
                f"synset {synset_id} are missing from the release and were skipped."
            )
        return available

    def _index_files(self) -> List[Tuple[XCubeShapeNetCategory, Path]]:
        files: List[Tuple[XCubeShapeNetCategory, Path]] = []
        for category in self.categories:
            synset_id = self.category_ids[category]
            for model in self._split_models(synset_id):
                files.append((category, Path(self.processed_dir, str(self.resolution), synset_id, f"{model}.npz")))
        return files

    @override
    def raw_files_exist(self) -> bool:
        for category in self.categories:
            synset_dir = Path(self.raw_dir, str(self.resolution), self.category_ids[category])
            if not (synset_dir / f"{self.split}.lst").exists():
                return False
        return True

    @override
    def processed_files_exist(self) -> bool:
        if not self.raw_files_exist():
            return False
        return all(path.exists() for _, path in self._index_files())

    def download(self) -> None:
        raise RuntimeError(
            f"{self.__class__.__name__} does not support automatic download. "
            f"Download the data from {self.data_url!r} and extract it under {self.raw_dir!r}."
        )

    def process(self, force: bool = False, show_progress: bool = True) -> None:
        if not self.raw_files_exist():
            self.download()
        if self.processed_files_exist() and not force:
            return

        for category in self.categories:
            synset_id = self.category_ids[category]
            raw_dir = Path(self.raw_dir, str(self.resolution), synset_id)
            out_dir = Path(self.processed_dir, str(self.resolution), synset_id)
            out_dir.mkdir(parents=True, exist_ok=True)
            models = self._split_models(synset_id)
            for model in tqdm(models, desc=f"Processing {category} ({self.split})", disable=not show_progress):
                out_path = out_dir / f"{model}.npz"
                if out_path.exists() and not force:
                    continue
                data = load_xcube_shape(raw_dir / f"{model}.pkl")
                buffer = io.BytesIO()
                np.savez_compressed(
                    buffer,
                    ijk=data["ijk"].numpy().astype(np.int16),
                    normal=data["normal"].numpy().astype(np.float32),
                )
                out_path.write_bytes(buffer.getvalue())

    @override
    def __len__(self) -> int:
        return len(self.files)

    @override
    def __getitem__(self, index: int) -> Dict[str, Any]:
        category, path = self.files[index]
        arrays = np.load(path)
        ijk = torch.from_numpy(arrays["ijk"].astype(np.float32))
        pos: Tensor = (ijk + 0.5) * self.voxel_size
        data: Dict[str, Any] = {
            DataKeys.POS: pos,
            DataKeys.NORMAL: torch.from_numpy(arrays["normal"]),
            DataKeys.CATEGORY: torch.tensor(list(self.category_ids).index(category), dtype=torch.long),
        }
        if self.transform is not None:
            data = self.transform(data)
        return data

    @override
    def extra_repr(self) -> str:
        return f"Split: {self.split}\nResolution: {self.resolution}\nCategories: {', '.join(self.categories)}"
