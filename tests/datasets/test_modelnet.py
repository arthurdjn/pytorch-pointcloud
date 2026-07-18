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

import pytest
import torch

from torch_pointcloud.datasets import ModelNet10, ModelNet40, ModelNetNormalResampled
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
@pytest.mark.parametrize("train", [True, False])
def test_modelnet_dataset_raw_files_exist(
    datasets_dir_factory: Callable[..., Path],
    dataset_cls: Type[ModelNetDataset],
    train: bool,
) -> None:
    """Test that the raw files exist"""
    datasets_dir = datasets_dir_factory(f"{dataset_cls.__name__}/raw/**/*")
    dataset = dataset_cls(root=datasets_dir, train=train, show_progress=False)
    assert dataset.raw_files_exist()


@pytest.mark.parametrize("dataset_cls", [ModelNet10, ModelNet40, ModelNet10NormalResampled, ModelNet40NormalResampled])
@pytest.mark.parametrize("train", [True, False])
def test_modelnet_dataset_raw_files_not_exist(dataset_cls: Type[ModelNetDataset], train: bool) -> None:
    """Test that the raw files do not exist"""
    with pytest.raises(RuntimeError, match="Dataset not found"):
        _ = dataset_cls(root="not-found", train=train, show_progress=False)


@pytest.mark.parametrize("dataset_cls", [ModelNet10, ModelNet40, ModelNet10NormalResampled, ModelNet40NormalResampled])
@pytest.mark.parametrize("train", [True, False])
def test_modelnet_dataset_processed_files_exist(
    datasets_dir_factory: Callable[..., Path], dataset_cls: Type[ModelNetDataset], train: bool
) -> None:
    """Test that the processed files exist"""
    datasets_dir = datasets_dir_factory(f"{dataset_cls.__name__}/processed/**/*")
    dataset = dataset_cls(root=datasets_dir, train=train, show_progress=False)
    assert dataset.processed_files_exist()


@pytest.mark.parametrize("dataset_cls", [ModelNet10, ModelNet40, ModelNet10NormalResampled, ModelNet40NormalResampled])
@pytest.mark.parametrize("train", [True, False])
def test_modelnet_dataset_processed_files_not_exist(dataset_cls: Type[ModelNetDataset], train: bool) -> None:
    """Test that the processed files do not exist"""
    with pytest.raises(RuntimeError, match="Dataset not found"):
        _ = dataset_cls(root="not-found", train=train, show_progress=False)


@pytest.mark.parametrize("dataset_cls", [ModelNet10, ModelNet40])
@pytest.mark.parametrize("train", [True, False])
@patch("torch_pointcloud.datasets.modelnet.load_modelnet_data")
def test_modelnet_dataset_already_processed(
    mock_load: Mock,
    datasets_dir_factory: Callable[..., Path],
    dataset_cls: Type[ModelNetDataset],
    train: bool,
) -> None:
    """Test that the dataset is loaded correctly for different splits"""
    mock_load.side_effect = load_modelnet_data
    datasets_dir = datasets_dir_factory(f"{dataset_cls.__name__}/processed/**/*")

    dataset = dataset_cls(root=datasets_dir, train=train, show_progress=False)
    assert len(dataset) > 0
    _ = list(dataset)

    assert mock_load.call_count == 0


@pytest.mark.parametrize("dataset_cls", [ModelNet10NormalResampled, ModelNet40NormalResampled])
@pytest.mark.parametrize("train", [True, False])
@patch("torch_pointcloud.datasets.modelnet.load_modelnet_normal_resampled_data")
def test_modelnet_normal_resampled_dataset_already_processed(
    mock_load: Mock,
    datasets_dir_factory: Callable[..., Path],
    dataset_cls: Type[ModelNetDataset],
    train: bool,
) -> None:
    """Test that the dataset is loaded correctly for different splits"""
    mock_load.side_effect = load_modelnet_normal_resampled_data
    datasets_dir = datasets_dir_factory(f"{dataset_cls.__name__}/processed/**/*")

    dataset = dataset_cls(root=datasets_dir, train=train, show_progress=False)
    assert len(dataset) > 0
    _ = list(dataset)

    assert mock_load.call_count == 0


@pytest.mark.parametrize("dataset_cls", [ModelNet10, ModelNet40])
@pytest.mark.parametrize("train", [True, False])
@patch("torch_pointcloud.datasets.modelnet.load_modelnet_data")
def test_modelnet_dataset_process_split(
    mock_load: Mock,
    datasets_dir_factory: Callable[..., Path],
    dataset_cls: Type[ModelNetDataset],
    train: bool,
) -> None:
    """Test that the dataset is processed correctly for different splits
    when the processed data does not already exist"""
    mock_load.side_effect = load_modelnet_data
    datasets_dir = datasets_dir_factory(f"{dataset_cls.__name__}/raw/**/*")

    dataset = dataset_cls(root=datasets_dir, train=train, show_progress=False)
    assert len(dataset) > 0
    _ = list(dataset)

    assert mock_load.call_count == len(dataset)


@pytest.mark.parametrize("dataset_cls", [ModelNet10NormalResampled, ModelNet40NormalResampled])
@pytest.mark.parametrize("train", [True, False])
@patch("torch_pointcloud.datasets.modelnet.load_modelnet_normal_resampled_data")
def test_modelnet_normal_resampled_dataset_process_split(
    mock_load: Mock,
    datasets_dir_factory: Callable[..., Path],
    dataset_cls: Type[ModelNetDataset],
    train: bool,
) -> None:
    """Test that the dataset is processed correctly for different splits
    when the processed data does not already exist"""
    mock_load.side_effect = load_modelnet_normal_resampled_data
    datasets_dir = datasets_dir_factory(f"{dataset_cls.__name__}/raw/**/*")

    dataset = dataset_cls(root=datasets_dir, train=train, show_progress=False)
    assert len(dataset) > 0
    _ = list(dataset)

    assert mock_load.call_count == len(dataset)


@pytest.mark.parametrize("dataset_cls", [ModelNet10, ModelNet40])
@pytest.mark.parametrize("train", [True, False])
@patch("torch_pointcloud.datasets.modelnet.load_modelnet_data")
def test_modelnet_dataset_process_split_forced(
    mock_load: Mock,
    datasets_dir_factory: Callable[..., Path],
    dataset_cls: Type[ModelNetDataset],
    train: bool,
) -> None:
    """Test that the dataset is processed correctly for different splits
    regardless of whether the processed data already exists"""
    mock_load.side_effect = load_modelnet_data
    datasets_dir = datasets_dir_factory(f"{dataset_cls.__name__}/**/*", symlinks=False)

    dataset = dataset_cls(root=datasets_dir, train=train, show_progress=False, force_process=True)
    assert len(dataset) > 0
    _ = list(dataset)

    assert mock_load.call_count == len(dataset)


@pytest.mark.parametrize("dataset_cls", [ModelNet10NormalResampled, ModelNet40NormalResampled])
@pytest.mark.parametrize("train", [True, False])
@patch("torch_pointcloud.datasets.modelnet.load_modelnet_normal_resampled_data")
def test_modelnet_normal_resampled_dataset_process_split_forced(
    mock_load: Mock,
    datasets_dir_factory: Callable[..., Path],
    dataset_cls: Type[ModelNetDataset],
    train: bool,
) -> None:
    """Test that the dataset is processed correctly for different splits
    regardless of whether the processed data already exists"""
    mock_load.side_effect = load_modelnet_normal_resampled_data
    datasets_dir = datasets_dir_factory(f"{dataset_cls.__name__}/**/*", symlinks=False)

    dataset = dataset_cls(root=datasets_dir, train=train, show_progress=False, force_process=True)
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
        _ = ModelNet10NormalResampled(root=datasets_dir, show_progress=False)

    assert not marker.exists()


def test_modelnet_normal_resampled_processed_cache_roundtrip(datasets_dir_factory: Callable[..., Path]) -> None:
    """The cache written by process() reloads under weights_only with identical tensors."""
    datasets_dir = datasets_dir_factory("ModelNetNormalResampled/raw/**/*")
    processed = ModelNet10NormalResampled(root=datasets_dir, train=True, show_progress=False)
    reloaded = ModelNet10NormalResampled(root=datasets_dir, train=True, show_progress=False)

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
