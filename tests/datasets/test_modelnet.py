# mypy: disable-error-code="arg-type,call-overload"
import functools
import hashlib
import io
import os
import pickle
import tarfile
import zipfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterator, Tuple, Type, Union
from unittest.mock import Mock, patch

import h5py
import numpy as np
import pytest
import torch

import torch_pointcloud.transforms as T
from torch_pointcloud.datasets import ModelNet10, ModelNet40, ModelNet40Hdf5, ModelNetNormalResampled
from torch_pointcloud.datasets.modelnet import (
    load_modelnet_data,
    load_modelnet_normal_resampled_data,
)

ModelNet10NormalResampled = functools.partial(ModelNetNormalResampled, variant="10")
ModelNet10NormalResampled.__name__ = "ModelNetNormalResampled"  # type: ignore[attr-defined]
ModelNet40NormalResampled = functools.partial(ModelNetNormalResampled, variant="40")
ModelNet40NormalResampled.__name__ = "ModelNetNormalResampled"  # type: ignore[attr-defined]
ModelNetDataset = Union[ModelNet10, ModelNet40]


def _zip_bytes(member: str) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as zip_file:
        zip_file.writestr(member, "data")
    return buffer.getvalue()


def _targz_bytes(member: str) -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tar_file:
        info = tarfile.TarInfo(member)
        content = b"data"
        info.size = len(content)
        tar_file.addfile(info, io.BytesIO(content))
    return buffer.getvalue()


@contextmanager
def _serve_download(content: bytes) -> Iterator[Mock]:
    """Patch the network so that download_url receives `content` as the remote file."""
    with patch("torch_pointcloud.datasets.utils.urlopen") as mock_urlopen:
        response = Mock()
        response.read.side_effect = [content, b""]
        response.length = len(content)
        mock_urlopen.return_value.__enter__.return_value = response
        yield mock_urlopen


class _MarkerPayload:
    """Pickle payload whose deserialization would run a shell command."""

    def __init__(self, command: str) -> None:
        self.command = command

    def __reduce__(self) -> Tuple[Any, ...]:
        return (os.system, (self.command,))


@pytest.mark.parametrize("dataset_name", ["ModelNet10", "ModelNet40"])
def test_load_modelnet_data(datasets_dir: Path, dataset_name: str) -> None:
    """Test that the modelnet data is loaded correctly"""
    file_path = datasets_dir / dataset_name / "raw" / "chair" / "train" / "chair_0001.off"
    data = load_modelnet_data(file_path, 0)

    assert isinstance(data["pos"], torch.Tensor)
    assert isinstance(data["face"], torch.Tensor)
    assert isinstance(data["label"], torch.Tensor)

    assert data["label"].item() == 0
    # Loose ndim/feature-dim assertions: the fixture is regenerated at ~1024
    # vertices today, but the loader contract is independent of vertex count.
    assert data["pos"].ndim == 2 and data["pos"].shape[1] == 3
    assert data["face"].ndim == 2 and data["face"].shape[1] == 3


@pytest.mark.parametrize("dataset_name", ["ModelNetNormalResampled"])
def test_load_modelnet_normal_resampled_data(datasets_dir: Path, dataset_name: str) -> None:
    """Test that the modelnet normal resampled data is loaded correctly"""
    file_path = datasets_dir / dataset_name / "raw" / "chair" / "chair_0001.txt"
    data = load_modelnet_normal_resampled_data(file_path, 0)

    assert isinstance(data["pos"], torch.Tensor)
    assert isinstance(data["normal"], torch.Tensor)
    assert isinstance(data["label"], torch.Tensor)

    # Loose ndim/feature-dim assertions: the fixture is regenerated at ~1024
    # points today, but the loader contract is independent of point count.
    assert data["pos"].ndim == 2 and data["pos"].shape[1] == 3
    assert data["normal"].shape == data["pos"].shape
    assert data["label"].item() == 0


@pytest.mark.parametrize("dataset_cls", [ModelNet10, ModelNet40, ModelNet10NormalResampled, ModelNet40NormalResampled])
def test_modelnet_dataset_not_found(tmp_path: Path, dataset_cls: Type[ModelNetDataset]) -> None:
    """Raises an error if the dataset is not found"""
    with pytest.raises(RuntimeError, match="Dataset not found"):
        _ = dataset_cls(root=tmp_path, show_progress=False)


@pytest.mark.parametrize("dataset_cls", [ModelNet10, ModelNet40, ModelNet10NormalResampled, ModelNet40NormalResampled])
def test_modelnet_dataset_invalid_split(dataset_cls: Type[ModelNetDataset]) -> None:
    """Raises an error if the split is invalid or not supported"""
    with pytest.raises(ValueError, match="Invalid split"):
        _ = dataset_cls(root="not-found", split="bogus", show_progress=False)


@pytest.mark.parametrize("dataset_cls", [ModelNet10, ModelNet40, ModelNet10NormalResampled, ModelNet40NormalResampled])
@pytest.mark.parametrize("split", ["train", "test"])
def test_modelnet_dataset_raw_files_exist(
    datasets_dir_factory: Callable[..., Path],
    dataset_cls: Type[ModelNetDataset],
    split: str,
) -> None:
    """Test that the raw files exist"""
    datasets_dir = datasets_dir_factory(f"{dataset_cls.__name__}/raw/**/*")
    dataset = dataset_cls(root=datasets_dir, split=split, show_progress=False)
    assert dataset.raw_files_exist()


@pytest.mark.parametrize("dataset_cls", [ModelNet10, ModelNet40, ModelNet10NormalResampled, ModelNet40NormalResampled])
@pytest.mark.parametrize("split", ["train", "test"])
def test_modelnet_dataset_raw_files_not_exist(dataset_cls: Type[ModelNetDataset], split: str) -> None:
    """Test that the raw files do not exist"""
    with pytest.raises(RuntimeError, match="Dataset not found"):
        _ = dataset_cls(root="not-found", split=split, show_progress=False)


@pytest.mark.parametrize("dataset_cls", [ModelNet10, ModelNet40, ModelNet10NormalResampled, ModelNet40NormalResampled])
@pytest.mark.parametrize("split", ["train", "test"])
def test_modelnet_dataset_processed_files_exist(
    datasets_dir_factory: Callable[..., Path], dataset_cls: Type[ModelNetDataset], split: str
) -> None:
    """Test that the processed files exist"""
    datasets_dir = datasets_dir_factory(f"{dataset_cls.__name__}/processed/**/*")
    dataset = dataset_cls(root=datasets_dir, split=split, show_progress=False)
    assert dataset.processed_files_exist()


@pytest.mark.parametrize("dataset_cls", [ModelNet10, ModelNet40, ModelNet10NormalResampled, ModelNet40NormalResampled])
@pytest.mark.parametrize("split", ["train", "test"])
def test_modelnet_dataset_processed_files_not_exist(dataset_cls: Type[ModelNetDataset], split: str) -> None:
    """Test that the processed files do not exist"""
    with pytest.raises(RuntimeError, match="Dataset not found"):
        _ = dataset_cls(root="not-found", split=split, show_progress=False)


@pytest.mark.parametrize("dataset_cls", [ModelNet10, ModelNet40])
@pytest.mark.parametrize("split", ["train", "test"])
@patch("torch_pointcloud.datasets.modelnet.load_modelnet_data")
def test_modelnet_dataset_already_processed(
    mock_load: Mock,
    datasets_dir_factory: Callable[..., Path],
    dataset_cls: Type[ModelNetDataset],
    split: str,
) -> None:
    """Test that the dataset is loaded correctly for different splits"""
    mock_load.side_effect = load_modelnet_data
    datasets_dir = datasets_dir_factory(f"{dataset_cls.__name__}/processed/**/*")

    dataset = dataset_cls(root=datasets_dir, split=split, show_progress=False)
    assert len(dataset) > 0
    _ = list(dataset)

    assert mock_load.call_count == 0


@pytest.mark.parametrize("dataset_cls", [ModelNet10NormalResampled, ModelNet40NormalResampled])
@pytest.mark.parametrize("split", ["train", "test"])
@patch("torch_pointcloud.datasets.modelnet.load_modelnet_normal_resampled_data")
def test_modelnet_normal_resampled_dataset_already_processed(
    mock_load: Mock,
    datasets_dir_factory: Callable[..., Path],
    dataset_cls: Type[ModelNetDataset],
    split: str,
) -> None:
    """Test that the dataset is loaded correctly for different splits"""
    mock_load.side_effect = load_modelnet_normal_resampled_data
    datasets_dir = datasets_dir_factory(f"{dataset_cls.__name__}/processed/**/*")

    dataset = dataset_cls(root=datasets_dir, split=split, show_progress=False)
    assert len(dataset) > 0
    _ = list(dataset)

    assert mock_load.call_count == 0


@pytest.mark.parametrize("dataset_cls", [ModelNet10, ModelNet40])
@pytest.mark.parametrize("split", ["train", "test"])
@patch("torch_pointcloud.datasets.modelnet.load_modelnet_data")
def test_modelnet_dataset_process_split(
    mock_load: Mock,
    datasets_dir_factory: Callable[..., Path],
    dataset_cls: Type[ModelNetDataset],
    split: str,
) -> None:
    """Test that the dataset is processed correctly for different splits
    when the processed data does not already exist"""
    mock_load.side_effect = load_modelnet_data
    datasets_dir = datasets_dir_factory(f"{dataset_cls.__name__}/raw/**/*")

    dataset = dataset_cls(root=datasets_dir, split=split, show_progress=False)
    assert len(dataset) > 0
    _ = list(dataset)

    assert mock_load.call_count == len(dataset)


@pytest.mark.parametrize("dataset_cls", [ModelNet10NormalResampled, ModelNet40NormalResampled])
@pytest.mark.parametrize("split", ["train", "test"])
@patch("torch_pointcloud.datasets.modelnet.load_modelnet_normal_resampled_data")
def test_modelnet_normal_resampled_dataset_process_split(
    mock_load: Mock,
    datasets_dir_factory: Callable[..., Path],
    dataset_cls: Type[ModelNetDataset],
    split: str,
) -> None:
    """Test that the dataset is processed correctly for different splits
    when the processed data does not already exist"""
    mock_load.side_effect = load_modelnet_normal_resampled_data
    datasets_dir = datasets_dir_factory(f"{dataset_cls.__name__}/raw/**/*")

    dataset = dataset_cls(root=datasets_dir, split=split, show_progress=False)
    assert len(dataset) > 0
    _ = list(dataset)

    assert mock_load.call_count == len(dataset)


@pytest.mark.parametrize("dataset_cls", [ModelNet10, ModelNet40])
@pytest.mark.parametrize("split", ["train", "test"])
@patch("torch_pointcloud.datasets.modelnet.load_modelnet_data")
def test_modelnet_dataset_process_split_forced(
    mock_load: Mock,
    datasets_dir_factory: Callable[..., Path],
    dataset_cls: Type[ModelNetDataset],
    split: str,
) -> None:
    """Test that the dataset is processed correctly for different splits
    regardless of whether the processed data already exists"""
    mock_load.side_effect = load_modelnet_data
    datasets_dir = datasets_dir_factory(f"{dataset_cls.__name__}/**/*", symlinks=False)

    dataset = dataset_cls(root=datasets_dir, split=split, show_progress=False, force_process=True)
    assert len(dataset) > 0
    _ = list(dataset)

    assert mock_load.call_count == len(dataset)


@pytest.mark.parametrize("dataset_cls", [ModelNet10NormalResampled, ModelNet40NormalResampled])
@pytest.mark.parametrize("split", ["train", "test"])
@patch("torch_pointcloud.datasets.modelnet.load_modelnet_normal_resampled_data")
def test_modelnet_normal_resampled_dataset_process_split_forced(
    mock_load: Mock,
    datasets_dir_factory: Callable[..., Path],
    dataset_cls: Type[ModelNetDataset],
    split: str,
) -> None:
    """Test that the dataset is processed correctly for different splits
    regardless of whether the processed data already exists"""
    mock_load.side_effect = load_modelnet_normal_resampled_data
    datasets_dir = datasets_dir_factory(f"{dataset_cls.__name__}/**/*", symlinks=False)

    dataset = dataset_cls(root=datasets_dir, split=split, show_progress=False, force_process=True)
    assert len(dataset) > 0
    _ = list(dataset)

    assert mock_load.call_count == len(dataset)


def test_modelnet_download_redownloads_corrupt_resource(
    datasets_dir_factory: Callable[..., Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A cached archive that fails the checksum is re-downloaded (overwritten) instead of being reused."""
    datasets_dir = datasets_dir_factory("ModelNet10/processed/**/*")
    dataset = ModelNet10(root=datasets_dir, show_progress=False)
    archive = _zip_bytes("ModelNet10/dummy.off")
    monkeypatch.setattr(dataset, "md5", hashlib.md5(archive).hexdigest())
    resource_path = Path(dataset.raw_dir, dataset.resource)
    resource_path.parent.mkdir(parents=True, exist_ok=True)
    resource_path.write_bytes(b"corrupt")

    with _serve_download(archive) as mock_urlopen:
        dataset.download()

    assert mock_urlopen.called
    assert Path(dataset.raw_dir, "dummy.off").exists()
    assert not resource_path.exists()


def test_modelnet_download_raises_when_redownload_still_corrupt(datasets_dir_factory: Callable[..., Path]) -> None:
    """If the re-downloaded archive still fails the checksum, download() raises with both hashes."""
    datasets_dir = datasets_dir_factory("ModelNet10/processed/**/*")
    dataset = ModelNet10(root=datasets_dir, show_progress=False)
    resource_path = Path(dataset.raw_dir, dataset.resource)
    resource_path.parent.mkdir(parents=True, exist_ok=True)
    resource_path.write_bytes(b"corrupt")

    with _serve_download(b"still corrupt") as mock_urlopen:
        with pytest.raises(RuntimeError, match="MD5 hash mismatch") as excinfo:
            dataset.download()

    assert mock_urlopen.called
    assert dataset.md5 in str(excinfo.value)
    assert hashlib.md5(b"still corrupt").hexdigest() in str(excinfo.value)


def test_modelnet_force_download_overwrites_valid_resource(
    datasets_dir_factory: Callable[..., Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """force re-downloads the archive even when a valid one is already on disk."""
    datasets_dir = datasets_dir_factory("ModelNet10/raw/**/*")
    dataset = ModelNet10(root=datasets_dir, show_progress=False)
    archive = _zip_bytes("ModelNet10/dummy.off")
    monkeypatch.setattr(dataset, "md5", hashlib.md5(archive).hexdigest())
    resource_path = Path(dataset.raw_dir, dataset.resource)
    resource_path.write_bytes(archive)

    with _serve_download(archive) as mock_urlopen:
        dataset.download(force=True)

    assert mock_urlopen.called


def test_modelnet_normal_resampled_download_redownloads_corrupt_resource(
    datasets_dir_factory: Callable[..., Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A cached archive that fails the checksum is re-downloaded (overwritten) instead of being reused."""
    datasets_dir = datasets_dir_factory("ModelNetNormalResampled/processed/**/*")
    dataset = ModelNet10NormalResampled(root=datasets_dir, show_progress=False)
    archive = _targz_bytes("modelnet40_normal_resampled/dummy.txt")
    monkeypatch.setattr(dataset, "md5", hashlib.md5(archive).hexdigest())
    resource_path = Path(dataset.raw_dir, dataset.resource)
    resource_path.parent.mkdir(parents=True, exist_ok=True)
    resource_path.write_bytes(b"corrupt")

    with _serve_download(archive) as mock_urlopen:
        dataset.download()

    assert mock_urlopen.called
    assert Path(dataset.raw_dir, "dummy.txt").exists()
    assert not resource_path.exists()


def test_modelnet_normal_resampled_download_raises_when_redownload_still_corrupt(
    datasets_dir_factory: Callable[..., Path],
) -> None:
    """If the re-downloaded archive still fails the checksum, download() raises with both hashes."""
    datasets_dir = datasets_dir_factory("ModelNetNormalResampled/processed/**/*")
    dataset = ModelNet10NormalResampled(root=datasets_dir, show_progress=False)
    resource_path = Path(dataset.raw_dir, dataset.resource)
    resource_path.parent.mkdir(parents=True, exist_ok=True)
    resource_path.write_bytes(b"corrupt")

    with _serve_download(b"still corrupt") as mock_urlopen:
        with pytest.raises(RuntimeError, match="MD5 hash mismatch") as excinfo:
            dataset.download()

    assert mock_urlopen.called
    assert dataset.md5 in str(excinfo.value)
    assert hashlib.md5(b"still corrupt").hexdigest() in str(excinfo.value)


def test_modelnet_normal_resampled_force_download_overwrites_valid_resource(
    datasets_dir_factory: Callable[..., Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """force re-downloads the archive even when a valid one is already on disk."""
    datasets_dir = datasets_dir_factory("ModelNetNormalResampled/raw/**/*")
    dataset = ModelNet10NormalResampled(root=datasets_dir, show_progress=False)
    archive = _targz_bytes("modelnet40_normal_resampled/dummy.txt")
    monkeypatch.setattr(dataset, "md5", hashlib.md5(archive).hexdigest())
    resource_path = Path(dataset.raw_dir, dataset.resource)
    resource_path.write_bytes(archive)

    with _serve_download(archive) as mock_urlopen:
        dataset.download(force=True)

    assert mock_urlopen.called


def test_modelnet_normal_resampled_download_strips_archive_root_dir(
    datasets_dir_factory: Callable[..., Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Extraction strips the archive's root directory, named after the resource minus its '.tar.gz' suffix."""
    datasets_dir = datasets_dir_factory("ModelNetNormalResampled/processed/**/*")
    dataset = ModelNet10NormalResampled(root=datasets_dir, show_progress=False)
    archive = _targz_bytes("modelnet40_normal_resampled/dummy.txt")
    monkeypatch.setattr(dataset, "md5", hashlib.md5(archive).hexdigest())

    with _serve_download(archive):
        dataset.download()

    assert Path(dataset.raw_dir, "dummy.txt").exists()
    assert not Path(dataset.raw_dir, "modelnet40_normal_resampled").exists()


def test_modelnet_normal_resampled_stale_pickle_cache_raises(
    datasets_dir_factory: Callable[..., Path], tmp_path: Path
) -> None:
    """An old pickle-format cache is rejected with a force_process hint before unpickling anything."""
    datasets_dir = datasets_dir_factory("ModelNetNormalResampled/processed/**/*", symlinks=False)
    cache_path = Path(datasets_dir, "ModelNetNormalResampled", "processed", "modelnet10_test.dat")
    marker = tmp_path / "marker"
    cache_path.write_bytes(pickle.dumps(_MarkerPayload(f"touch {marker}")))

    with pytest.raises(RuntimeError, match="force_process=True"):
        _ = ModelNet10NormalResampled(root=datasets_dir, split="test", show_progress=False)

    assert not marker.exists()


def test_modelnet_normal_resampled_processed_cache_roundtrip(datasets_dir_factory: Callable[..., Path]) -> None:
    """The cache written by process() reloads under weights_only with identical tensors."""
    datasets_dir = datasets_dir_factory("ModelNetNormalResampled/raw/**/*")
    processed = ModelNet10NormalResampled(root=datasets_dir, split="train", show_progress=False)
    reloaded = ModelNet10NormalResampled(root=datasets_dir, split="train", show_progress=False)

    assert len(reloaded) == len(processed) > 0
    for sample, resample in zip(processed.data, reloaded.data):
        assert sorted(sample) == sorted(resample)
        for key in sample:
            assert isinstance(resample[key], torch.Tensor)
            assert torch.equal(sample[key], resample[key])


@pytest.mark.parametrize(
    "dataset_cls,classes_a,classes_b",
    [
        (ModelNet10, "chair", "table"),
        (ModelNet40, "airplane", "car"),
        (ModelNet10NormalResampled, "chair", "table"),
        (ModelNet40NormalResampled, "airplane", "car"),
    ],
)
def test_modelnet_dataset_cache_meta_mismatch_raises(
    datasets_dir_factory: Callable[..., Path],
    dataset_cls: Type[ModelNetDataset],
    classes_a: str,
    classes_b: str,
) -> None:
    """A processed cache written with different classes raises instead of silently loading the wrong data."""
    datasets_dir = datasets_dir_factory(f"{dataset_cls.__name__}/raw/**/*")
    _ = dataset_cls(root=datasets_dir, classes=classes_a, show_progress=False)

    with pytest.raises(RuntimeError, match="force_process=True"):
        _ = dataset_cls(root=datasets_dir, classes=classes_b, show_progress=False)

    dataset = dataset_cls(root=datasets_dir, classes=classes_b, show_progress=False, force_process=True)
    assert dataset.classes == (classes_b,)


def test_modelnet_dataset_cache_meta_pre_transform_mismatch_raises(
    datasets_dir_factory: Callable[..., Path],
) -> None:
    """A processed cache written without a pre_transform raises when one is requested later."""
    datasets_dir = datasets_dir_factory("ModelNet10/raw/**/*")
    _ = ModelNet10(root=datasets_dir, show_progress=False)

    with pytest.raises(RuntimeError, match="force_process=True"):
        _ = ModelNet10(root=datasets_dir, pre_transform=Mock(side_effect=lambda data: data), show_progress=False)


def test_modelnet_dataset_cache_meta_records_transform_params(
    datasets_dir_factory: Callable[..., Path],
) -> None:
    """The cache metadata includes transform parameters, so the same class with different params raises."""
    datasets_dir = datasets_dir_factory("ModelNet10/raw/**/*")
    _ = ModelNet10(root=datasets_dir, pre_transform=T.RandomSample(keys="pos", num_samples=32), show_progress=False)

    reloaded = ModelNet10(
        root=datasets_dir, pre_transform=T.RandomSample(keys="pos", num_samples=32), show_progress=False
    )
    assert len(reloaded) > 0

    with pytest.raises(RuntimeError, match="force_process=True"):
        _ = ModelNet10(root=datasets_dir, pre_transform=T.RandomSample(keys="pos", num_samples=64), show_progress=False)


def test_modelnet_dataset_force_download_implies_download(
    datasets_dir_factory: Callable[..., Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """`force_download=True` triggers the download even when `download` is left False."""
    datasets_dir = datasets_dir_factory("ModelNet10/raw/**/*")
    mock = Mock()
    monkeypatch.setattr(ModelNet10, "download", mock)

    _ = ModelNet10(root=datasets_dir, force_download=True, show_progress=False)
    mock.assert_called_once_with(force=True)


def test_modelnet_normal_resampled_force_download_implies_download(
    datasets_dir_factory: Callable[..., Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """`force_download=True` triggers the download even when `download` is left False."""
    datasets_dir = datasets_dir_factory("ModelNetNormalResampled/raw/**/*")
    mock = Mock()
    monkeypatch.setattr(ModelNetNormalResampled, "download", mock)

    _ = ModelNet10NormalResampled(root=datasets_dir, force_download=True, show_progress=False)
    mock.assert_called_once_with(force=True)


@pytest.mark.parametrize(
    "dataset_cls,resource",
    [
        (ModelNet10, "ModelNet10.zip"),
        (ModelNet10NormalResampled, "modelnet40_normal_resampled.tar.gz"),
    ],
)
def test_modelnet_dataset_leftover_archive_detected(
    datasets_dir_factory: Callable[..., Path],
    dataset_cls: Type[ModelNetDataset],
    resource: str,
) -> None:
    """A leftover archive marks an interrupted extraction, so the partial raw tree is not processed."""
    datasets_dir = datasets_dir_factory(f"{dataset_cls.__name__}/raw/**/*")
    archive_path = datasets_dir / dataset_cls.__name__ / "raw" / resource
    archive_path.write_bytes(b"partial")

    with pytest.raises(RuntimeError, match="Dataset not found"):
        _ = dataset_cls(root=datasets_dir, show_progress=False)

    archive_path.unlink()
    dataset = dataset_cls(root=datasets_dir, show_progress=False)
    assert len(dataset) > 0


def test_modelnet_download_reextracts_leftover_archive(
    datasets_dir_factory: Callable[..., Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A valid archive left by an interrupted extraction is re-extracted without re-downloading."""
    datasets_dir = datasets_dir_factory("ModelNet10/processed/**/*")
    dataset = ModelNet10(root=datasets_dir, show_progress=False)
    archive = _zip_bytes("ModelNet10/dummy.off")
    monkeypatch.setattr(dataset, "md5", hashlib.md5(archive).hexdigest())
    resource_path = Path(dataset.raw_dir, dataset.resource)
    resource_path.parent.mkdir(parents=True, exist_ok=True)
    resource_path.write_bytes(archive)

    with _serve_download(archive) as mock_urlopen:
        dataset.download()

    assert not mock_urlopen.called
    assert Path(dataset.raw_dir, "dummy.off").exists()
    assert not resource_path.exists()


def test_modelnet_dataset_interrupted_cache_write_reprocesses(
    datasets_dir_factory: Callable[..., Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A crash during the cache write leaves no processed file, so the next construction reprocesses."""
    datasets_dir = datasets_dir_factory("ModelNet10/raw/**/*")

    def interrupted_save(obj: Any, path: Any, **kwargs: Any) -> None:
        Path(path).write_bytes(b"partial")
        raise RuntimeError("interrupted")

    monkeypatch.setattr(torch, "save", interrupted_save)
    with pytest.raises(RuntimeError, match="interrupted"):
        _ = ModelNet10(root=datasets_dir, show_progress=False)

    assert not (datasets_dir / "ModelNet10" / "processed" / "train.pt").exists()

    monkeypatch.undo()
    dataset = ModelNet10(root=datasets_dir, show_progress=False)
    assert len(dataset) > 0


def test_modelnet_dataset_getitem_returns_shallow_copy(datasets_dir_factory: Callable[..., Path]) -> None:
    """User edits on a returned sample dict never reach the in-memory cache."""
    datasets_dir = datasets_dir_factory("ModelNet10/processed/**/*")
    dataset = ModelNet10(root=datasets_dir, show_progress=False)

    sample = dataset[0]
    assert sample is not dataset.data[0]
    sample["extra"] = 1
    assert "extra" not in dataset[0]


@pytest.mark.parametrize(
    "dataset_cls,category",
    [
        (ModelNet10, "chair"),
        (ModelNet40, "airplane"),
        (ModelNet10NormalResampled, "chair"),
        (ModelNet40NormalResampled, "airplane"),
    ],
)
def test_modelnet_dataset_legacy_cache_without_meta_loads(
    datasets_dir_factory: Callable[..., Path],
    dataset_cls: Type[ModelNetDataset],
    category: str,
) -> None:
    """A processed cache from before cache metadata existed is accepted as-is."""
    datasets_dir = datasets_dir_factory(f"{dataset_cls.__name__}/processed/**/*")
    dataset = dataset_cls(root=datasets_dir, classes=category, show_progress=False)
    assert not list(Path(dataset.processed_dir).glob("*.meta.json"))
    assert len(dataset) > 0


@pytest.mark.parametrize("dataset_cls", [ModelNet10, ModelNet40, ModelNet10NormalResampled, ModelNet40NormalResampled])
def test_modelnet_dataset_progress(
    datasets_dir_factory: Callable[..., Path],
    dataset_cls: Type[ModelNetDataset],
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Test that the dataset is processed correctly with a progress bar"""
    datasets_dir = datasets_dir_factory(f"{dataset_cls.__name__}/raw/**/*")

    dataset = dataset_cls(root=datasets_dir, show_progress=True)
    assert len(dataset) > 0
    captured = capsys.readouterr()
    assert "Processing" in captured.err
    assert captured.out == ""


@pytest.mark.parametrize("dataset_cls", [ModelNet10, ModelNet40, ModelNet10NormalResampled, ModelNet40NormalResampled])
def test_modelnet_dataset_without_progress(
    datasets_dir_factory: Callable[..., Path],
    dataset_cls: Type[ModelNetDataset],
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Test that the dataset is processed correctly without a progress bar"""
    datasets_dir = datasets_dir_factory(f"{dataset_cls.__name__}/raw/**/*")

    dataset = dataset_cls(root=datasets_dir, show_progress=False)
    assert len(dataset) > 0
    captured = capsys.readouterr()
    assert captured.err == ""
    assert captured.out == ""


@pytest.mark.parametrize("dataset_cls", [ModelNet10, ModelNet40, ModelNet10NormalResampled, ModelNet40NormalResampled])
def test_modelnet_dataset_progress_with_cached_processed(
    datasets_dir_factory: Callable[..., Path],
    dataset_cls: Type[ModelNetDataset],
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Test that no progress bar is shown if the processed dataset already exists"""
    datasets_dir = datasets_dir_factory(f"{dataset_cls.__name__}/processed/**/*")

    dataset = dataset_cls(root=datasets_dir, show_progress=True)
    assert len(dataset) > 0
    captured = capsys.readouterr()
    assert captured.err == ""
    assert captured.out == ""


@pytest.mark.parametrize(
    "dataset_cls,category",
    [
        (ModelNet10, "chair"),
        (ModelNet40, "airplane"),
        (ModelNet10NormalResampled, "chair"),
        (ModelNet40NormalResampled, "airplane"),
    ],
)
def test_modelnet_dataset_category(
    datasets_dir_factory: Callable[..., Path],
    dataset_cls: Type[ModelNetDataset],
    category: str,
) -> None:
    """Test that the dataset is loaded correctly for a specific category"""
    datasets_dir = datasets_dir_factory(f"{dataset_cls.__name__}/raw/**/*")

    dataset = dataset_cls(root=datasets_dir, classes=category, show_progress=False)
    assert len(dataset) > 0
    assert dataset.classes == (category,)


@pytest.mark.parametrize(
    "dataset_cls,categories",
    [
        (ModelNet10, ["chair", "table"]),
        (ModelNet40, ["airplane", "car"]),
        (ModelNet10NormalResampled, ["chair", "table"]),
        (ModelNet40NormalResampled, ["airplane", "car"]),
    ],
)
def test_modelnet_dataset_categories(
    datasets_dir_factory: Callable[..., Path],
    dataset_cls: Type[ModelNetDataset],
    categories: list[str],
) -> None:
    """Test that the dataset is loaded correctly for multiple categories"""
    datasets_dir = datasets_dir_factory(f"{dataset_cls.__name__}/raw/**/*")

    dataset = dataset_cls(root=datasets_dir, classes=categories, show_progress=False)
    assert len(dataset) > 0
    assert len(dataset.classes) == len(categories)
    assert all(category in dataset.classes for category in categories)


@pytest.mark.parametrize("dataset_cls", [ModelNet10, ModelNet40, ModelNet10NormalResampled, ModelNet40NormalResampled])
def test_modelnet_dataset_pre_transform(
    datasets_dir_factory: Callable[..., Path], dataset_cls: Type[ModelNetDataset]
) -> None:
    """Test that the dataset is transformed correctly before being processed"""
    datasets_dir = datasets_dir_factory(f"{dataset_cls.__name__}/raw/**/*")

    pre_transform = Mock(side_effect=lambda x: x)
    dataset = dataset_cls(root=datasets_dir, pre_transform=pre_transform, show_progress=False)
    assert pre_transform.call_count == len(dataset)


@pytest.mark.parametrize("dataset_cls", [ModelNet10, ModelNet40, ModelNet10NormalResampled, ModelNet40NormalResampled])
def test_modelnet_dataset_pre_filter(
    datasets_dir_factory: Callable[..., Path], dataset_cls: Type[ModelNetDataset]
) -> None:
    """Test that the dataset is filtered correctly before being processed"""
    datasets_dir = datasets_dir_factory(f"{dataset_cls.__name__}/raw/**/*")

    pre_filter = Mock(side_effect=lambda x: True)
    dataset = dataset_cls(root=datasets_dir, pre_filter=pre_filter, show_progress=False)
    assert pre_filter.call_count == len(dataset)


@pytest.mark.parametrize("dataset_cls", [ModelNet10, ModelNet40, ModelNet10NormalResampled, ModelNet40NormalResampled])
def test_modelnet_dataset_transform(
    datasets_dir_factory: Callable[..., Path], dataset_cls: Type[ModelNetDataset]
) -> None:
    """Test that the dataset is transformed correctly after being processed"""
    datasets_dir = datasets_dir_factory(f"{dataset_cls.__name__}/processed/**/*")

    transform = Mock(side_effect=lambda data: data)
    dataset = dataset_cls(root=datasets_dir, transform=transform, show_progress=False)
    _ = list(dataset)
    assert transform.call_count == len(dataset)


def _write_modelnet40_hdf5_shard(path: Path, labels: list[int], num_points: int = 32, seed: int = 0) -> None:
    """Write a tiny synthetic `ply_data_*.h5` shard with the release's keys, shapes and dtypes."""
    rng = np.random.default_rng(seed)
    pos = rng.standard_normal((len(labels), num_points, 3)).astype(np.float32)
    normal = rng.standard_normal((len(labels), num_points, 3)).astype(np.float32)
    with h5py.File(path, "w") as f:
        f.create_dataset("data", data=pos)
        f.create_dataset("normal", data=normal)
        f.create_dataset("label", data=np.asarray(labels, dtype=np.uint8).reshape(-1, 1))


def _write_modelnet40_hdf5_raw(raw_dir: Path) -> None:
    """Fabricate a tiny raw tree: two train shards, one test shard, and the release-style file lists."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    _write_modelnet40_hdf5_shard(raw_dir / "ply_data_train0.h5", labels=[0, 1], seed=0)
    _write_modelnet40_hdf5_shard(raw_dir / "ply_data_train1.h5", labels=[2], seed=1)
    _write_modelnet40_hdf5_shard(raw_dir / "ply_data_test0.h5", labels=[3, 4], seed=2)
    (raw_dir / "train_files.txt").write_text(
        "data/modelnet40_ply_hdf5_2048/ply_data_train0.h5\ndata/modelnet40_ply_hdf5_2048/ply_data_train1.h5\n"
    )
    (raw_dir / "test_files.txt").write_text("data/modelnet40_ply_hdf5_2048/ply_data_test0.h5\n")


def _modelnet40_hdf5_zip_bytes(tmp_path: Path) -> bytes:
    """Zip the tiny raw tree under the release's `modelnet40_ply_hdf5_2048/` root directory."""
    stage = tmp_path / "modelnet40_ply_hdf5_2048_stage"
    _write_modelnet40_hdf5_raw(stage)
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as zip_file:
        for file_path in sorted(stage.rglob("*")):
            if file_path.is_file():
                zip_file.write(file_path, Path("modelnet40_ply_hdf5_2048") / file_path.relative_to(stage))
    return buffer.getvalue()


def test_modelnet40_hdf5_dataset_not_found() -> None:
    """Raises an error if the dataset is not found"""
    with pytest.raises(RuntimeError, match="Dataset not found"):
        _ = ModelNet40Hdf5(root="not-found", show_progress=False)


def test_modelnet40_hdf5_dataset_invalid_split(datasets_dir_factory: Callable[..., Path]) -> None:
    """Raises an error if the split is invalid or not supported"""
    datasets_dir = datasets_dir_factory("ModelNet40Hdf5/**/*")
    with pytest.raises(ValueError, match="Invalid split"):
        _ = ModelNet40Hdf5(root=datasets_dir, split="bogus", show_progress=False)


@pytest.mark.parametrize("split,expected_labels", [("train", [0, 1, 2]), ("test", [3, 4])])
def test_modelnet40_hdf5_dataset_loads_shards_in_list_order(
    datasets_dir_factory: Callable[..., Path], split: str, expected_labels: list[int]
) -> None:
    """Shards are concatenated in the split file-list order, with the release's `(N, 1)` labels squeezed."""
    datasets_dir = datasets_dir_factory("ModelNet40Hdf5/**/*")
    _write_modelnet40_hdf5_raw(datasets_dir / "ModelNet40Hdf5" / "raw")

    dataset = ModelNet40Hdf5(root=datasets_dir, split=split, show_progress=False)
    assert len(dataset) == len(expected_labels)
    assert [dataset[i]["label"].item() for i in range(len(dataset))] == expected_labels

    sample = dataset[0]
    assert sample["pos"].shape == (32, 3) and sample["pos"].dtype == torch.float32
    assert sample["normal"].shape == (32, 3) and sample["normal"].dtype == torch.float32
    assert sample["label"].shape == () and sample["label"].dtype == torch.int64


def test_modelnet40_hdf5_dataset_transform_called(datasets_dir_factory: Callable[..., Path]) -> None:
    """The transform is applied to every sample at `__getitem__` time."""
    datasets_dir = datasets_dir_factory("ModelNet40Hdf5/**/*")
    _write_modelnet40_hdf5_raw(datasets_dir / "ModelNet40Hdf5" / "raw")

    transform = Mock(side_effect=lambda data: data)
    dataset = ModelNet40Hdf5(root=datasets_dir, split="test", transform=transform, show_progress=False)
    _ = list(dataset)
    assert transform.call_count == len(dataset)


def test_modelnet40_hdf5_getitem_returns_detached_copy(datasets_dir_factory: Callable[..., Path]) -> None:
    """In-place user edits on a returned sample never reach the in-memory cache."""
    datasets_dir = datasets_dir_factory("ModelNet40Hdf5/**/*")
    _write_modelnet40_hdf5_raw(datasets_dir / "ModelNet40Hdf5" / "raw")

    dataset = ModelNet40Hdf5(root=datasets_dir, split="test", show_progress=False)
    sample = dataset[0]
    original = sample["pos"].clone()
    sample["pos"] += 1.0
    assert torch.equal(dataset[0]["pos"], original)


def test_modelnet40_hdf5_download_extracts_archive(
    tmp_path: Path, datasets_dir_factory: Callable[..., Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """`download=True` fetches the archive, strips its root directory into `raw/`, and removes the archive."""
    datasets_dir = datasets_dir_factory("ModelNet40Hdf5/**/*")
    archive = _modelnet40_hdf5_zip_bytes(tmp_path)
    monkeypatch.setattr(ModelNet40Hdf5, "md5", hashlib.md5(archive).hexdigest())

    with _serve_download(archive) as mock_urlopen:
        dataset = ModelNet40Hdf5(root=datasets_dir, split="train", download=True, show_progress=False)

    assert mock_urlopen.called
    assert len(dataset) == 3
    assert Path(dataset.raw_dir, "ply_data_train0.h5").exists()
    assert not Path(dataset.raw_dir, "modelnet40_ply_hdf5_2048").exists()
    assert not Path(dataset.raw_dir, dataset.resource).exists()


def test_modelnet40_hdf5_download_raises_when_redownload_still_corrupt(
    datasets_dir_factory: Callable[..., Path],
) -> None:
    """If the re-downloaded archive still fails the checksum, download() raises with both hashes."""
    datasets_dir = datasets_dir_factory("ModelNet40Hdf5/**/*")
    _write_modelnet40_hdf5_raw(datasets_dir / "ModelNet40Hdf5" / "raw")
    dataset = ModelNet40Hdf5(root=datasets_dir, split="train", show_progress=False)
    resource_path = Path(dataset.raw_dir, dataset.resource)
    resource_path.write_bytes(b"corrupt")

    with _serve_download(b"still corrupt") as mock_urlopen:
        with pytest.raises(RuntimeError, match="MD5 hash mismatch") as excinfo:
            dataset.download()

    assert mock_urlopen.called
    assert dataset.md5 in str(excinfo.value)
    assert hashlib.md5(b"still corrupt").hexdigest() in str(excinfo.value)


def test_modelnet40_hdf5_leftover_archive_detected(datasets_dir_factory: Callable[..., Path]) -> None:
    """A leftover archive marks an interrupted extraction, so the partial raw tree is not loaded."""
    datasets_dir = datasets_dir_factory("ModelNet40Hdf5/**/*")
    raw_dir = datasets_dir / "ModelNet40Hdf5" / "raw"
    _write_modelnet40_hdf5_raw(raw_dir)
    archive_path = raw_dir / ModelNet40Hdf5.resource
    archive_path.write_bytes(b"partial")

    with pytest.raises(RuntimeError, match="Dataset not found"):
        _ = ModelNet40Hdf5(root=datasets_dir, show_progress=False)

    archive_path.unlink()
    dataset = ModelNet40Hdf5(root=datasets_dir, show_progress=False)
    assert len(dataset) > 0


def test_modelnet40_hdf5_force_download_implies_download(
    datasets_dir_factory: Callable[..., Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """`force_download=True` triggers the download even when `download` is left False."""
    datasets_dir = datasets_dir_factory("ModelNet40Hdf5/**/*")
    _write_modelnet40_hdf5_raw(datasets_dir / "ModelNet40Hdf5" / "raw")
    mock = Mock()
    monkeypatch.setattr(ModelNet40Hdf5, "download", mock)

    _ = ModelNet40Hdf5(root=datasets_dir, force_download=True, show_progress=False)
    mock.assert_called_once_with(force=True)
