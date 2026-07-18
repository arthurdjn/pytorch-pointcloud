# mypy: disable-error-code="arg-type"
import io
import pickle
from collections import OrderedDict
from pathlib import Path
from typing import Callable

import numpy as np
import pytest
import torch
import torch._utils

from torch_pointcloud.datasets import XCubeShapeNet
from torch_pointcloud.datasets.shapenet import _StubState, _XCubeUnpickler, _read_xcube_pickle, load_xcube_shape
from torch_pointcloud.utils.data import DataKeys
from torch_pointcloud.utils.imports import _FVDB_AVAILABLE


def test_read_xcube_pickle_release_format(datasets_dir_factory: Callable[..., Path]) -> None:
    """The restricted unpickler still reads the release-format pickles (this step does not need fvdb)."""
    datasets_dir = datasets_dir_factory("XCubeShapeNet/raw/**/*")
    file_path = datasets_dir / "XCubeShapeNet" / "raw" / "128" / "03001627" / "dummy0.pkl"
    grid_buffer, normal = _read_xcube_pickle(file_path)

    assert isinstance(grid_buffer, np.ndarray)
    assert isinstance(normal, np.ndarray)
    assert normal.ndim == 2 and normal.shape[1] == 3


def test_xcube_unpickler_accepts_exact_release_globals() -> None:
    unpickler = _XCubeUnpickler(io.BytesIO(b""))
    assert unpickler.find_class("collections", "OrderedDict") is OrderedDict
    assert unpickler.find_class("torch._utils", "_rebuild_tensor_v2") is torch._utils._rebuild_tensor_v2
    for name in ("GridBatch", "JaggedTensor"):
        stub = unpickler.find_class("fvdb._Cpp", name)
        assert issubclass(stub, _StubState)


@pytest.mark.parametrize(
    "module,name",
    [
        ("os", "system"),
        ("builtins", "eval"),
        ("torch.serialization", "load"),
        ("numpy", "ndarray"),
        ("collections", "Counter"),
        ("fvdb", "GridBatch"),
        ("fvdb_evil", "GridBatch"),
        ("fvdb._Cpp", "SparseGrid"),
    ],
)
def test_xcube_unpickler_rejects_disallowed_globals(module: str, name: str) -> None:
    unpickler = _XCubeUnpickler(io.BytesIO(b""))
    with pytest.raises(pickle.UnpicklingError, match="not allowed"):
        unpickler.find_class(module, name)


@pytest.mark.skipif(not _FVDB_AVAILABLE, reason="fvdb is not installed")
def test_load_xcube_shape(datasets_dir_factory: Callable[..., Path]) -> None:
    datasets_dir = datasets_dir_factory("XCubeShapeNet/raw/**/*")
    file_path = datasets_dir / "XCubeShapeNet" / "raw" / "128" / "03001627" / "dummy0.pkl"
    data = load_xcube_shape(file_path)

    ijk = data["ijk"]
    normal = data["normal"]
    assert ijk.shape == normal.shape
    assert ijk.shape[1] == 3
    # Voxel coordinates are integers and normals are unit length.
    assert torch.allclose(ijk, ijk.round())
    assert torch.allclose(normal.norm(dim=1), torch.ones(len(normal)), atol=1e-3)


@pytest.mark.skipif(not _FVDB_AVAILABLE, reason="fvdb is not installed")
@pytest.mark.parametrize("split", ["train", "val", "test"])
def test_xcube_shapenet_process_and_load(split: str, datasets_dir_factory: Callable[..., Path]) -> None:
    data_dir = datasets_dir_factory("XCubeShapeNet/raw/**/*")
    dataset = XCubeShapeNet(data_dir, split=split, resolution=128, categories="Chair", show_progress=False)
    assert len(dataset) == 1

    sample = dataset[0]
    pos = sample[DataKeys.POS]
    normal = sample[DataKeys.NORMAL]
    assert pos.dtype == torch.float32 and normal.dtype == torch.float32
    assert pos.shape == normal.shape
    assert pos.shape[1] == 3
    assert sample[DataKeys.CATEGORY].item() == 2

    # Positions are voxel centers: pos = (ijk + 0.5) * voxel_size with integer ijk.
    ijk = pos / dataset.voxel_size - 0.5
    assert torch.allclose(ijk, ijk.round(), atol=1e-4)
    assert torch.allclose(normal.norm(dim=1), torch.ones(len(normal)), atol=1e-3)


@pytest.mark.skipif(not _FVDB_AVAILABLE, reason="fvdb is not installed")
def test_xcube_shapenet_processed_cache_reused(datasets_dir_factory: Callable[..., Path]) -> None:
    data_dir = datasets_dir_factory("XCubeShapeNet/raw/**/*")
    dataset = XCubeShapeNet(data_dir, split="test", resolution=128, categories="Chair", show_progress=False)
    cache = Path(dataset.processed_dir, "128", "03001627", "dummy0.npz")
    assert cache.exists()
    mtime = cache.stat().st_mtime
    again = XCubeShapeNet(data_dir, split="test", resolution=128, categories="Chair", show_progress=False)
    assert cache.stat().st_mtime == mtime
    assert torch.equal(dataset[0][DataKeys.POS], again[0][DataKeys.POS])


def test_xcube_shapenet_invalid_arguments(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Invalid split"):
        XCubeShapeNet(tmp_path, split="bad", resolution=128)
    with pytest.raises(ValueError, match="Invalid resolution"):
        XCubeShapeNet(tmp_path, split="test", resolution=256)


def test_xcube_shapenet_missing_raw_data(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="automatic download"):
        XCubeShapeNet(tmp_path, split="test", resolution=128, categories="Chair")
