# mypy: disable-error-code="arg-type,call-overload"
import hashlib
import io
import zipfile
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Iterator
from unittest.mock import Mock, patch

import pytest

from torch_pointcloud.datasets import ScanObjectNN


def _zip_bytes(member: str) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as zip_file:
        zip_file.writestr(member, "data")
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


@pytest.mark.parametrize("split", ["main", "split1", "split2", "split3", "split4"])
@pytest.mark.parametrize(
    "variant", [None, "augmented25_norot", "augmented25rot", "augmentedrot", "augmentedrot_scale75"]
)
@pytest.mark.parametrize("background", [True, False])
@pytest.mark.parametrize("train", [True, False])
def test_scanobjectnn_dataset_not_found(split: str, variant: str, background: bool, train: bool) -> None:
    """Raises an error if the dataset is not found"""
    with pytest.raises(RuntimeError, match="Dataset not found"):
        _ = ScanObjectNN(
            root="not-found",
            split=split,
            variant=variant,
            background=background,
            train=train,
            show_progress=False,
        )


@pytest.mark.parametrize("split", ["main", "split1", "split2", "split3", "split4"])
@pytest.mark.parametrize(
    "variant", [None, "augmented25_norot", "augmented25rot", "augmentedrot", "augmentedrot_scale75"]
)
@pytest.mark.parametrize("background", [True, False])
@pytest.mark.parametrize("train", [True, False])
def test_scanobjectnn_dataset_raw_files_exist(
    datasets_dir_factory: Callable[..., Path], split: str, variant: str, background: bool, train: bool
) -> None:
    """Test that the raw files exist"""
    datasets_dir = datasets_dir_factory("ScanObjectNN/raw/**/*")

    dataset = ScanObjectNN(
        root=datasets_dir,
        split=split,
        variant=variant,
        background=background,
        train=train,
        show_progress=False,
    )
    assert dataset.raw_files_exist()
    assert len(dataset) > 0


@pytest.mark.parametrize("split", ["main", "split1", "split2", "split3", "split4"])
@pytest.mark.parametrize(
    "variant", [None, "augmented25_norot", "augmented25rot", "augmentedrot", "augmentedrot_scale75"]
)
@pytest.mark.parametrize("background", [True, False])
@pytest.mark.parametrize("train", [True, False])
def test_scanobjectnn_dataset_raw_files_not_exist(split: str, variant: str, background: bool, train: bool) -> None:
    """Test that the raw files do not exist"""

    with pytest.raises(RuntimeError, match="Dataset not found"):
        _ = ScanObjectNN(
            root="not-found",
            split=split,
            variant=variant,
            background=background,
            train=train,
            show_progress=False,
        )


@pytest.mark.parametrize("split", ["main", "split1", "split2", "split3", "split4"])
@pytest.mark.parametrize(
    "variant", [None, "augmented25_norot", "augmented25rot", "augmentedrot", "augmentedrot_scale75"]
)
@pytest.mark.parametrize("background", [True, False])
@pytest.mark.parametrize("train", [True, False])
def test_scanobjectnn_dataset_processed_files_exist(
    datasets_dir_factory: Callable[..., Path], split: str, variant: str, background: bool, train: bool
) -> None:
    """Test that the processed files exist"""
    datasets_dir = datasets_dir_factory("ScanObjectNN/processed/**/*")

    dataset = ScanObjectNN(
        root=datasets_dir,
        split=split,
        variant=variant,
        background=background,
        train=train,
        show_progress=False,
    )
    assert dataset.processed_files_exist()
    assert len(dataset) > 0


@pytest.mark.parametrize("split", ["main", "split1", "split2", "split3", "split4"])
@pytest.mark.parametrize(
    "variant", [None, "augmented25_norot", "augmented25rot", "augmentedrot", "augmentedrot_scale75"]
)
@pytest.mark.parametrize("background", [True, False])
@pytest.mark.parametrize("train", [True, False])
def test_scanobjectnn_dataset_processed_files_not_exist(
    split: str, variant: str, background: bool, train: bool
) -> None:
    """Test that the processed files do not exist"""
    with pytest.raises(RuntimeError, match="Dataset not found"):
        _ = ScanObjectNN(
            root="not-found",
            split=split,
            variant=variant,
            background=background,
            train=train,
            show_progress=False,
        )


@pytest.mark.parametrize("split", ["main", "split1", "split2", "split3", "split4"])
@pytest.mark.parametrize(
    "variant", [None, "augmented25_norot", "augmented25rot", "augmentedrot", "augmentedrot_scale75"]
)
@pytest.mark.parametrize("background", [True, False])
@pytest.mark.parametrize("train", [True, False])
def test_scanobjectnn_dataset_force_process(
    datasets_dir_factory: Callable[..., Path], split: str, variant: str, background: bool, train: bool
) -> None:
    """Test that the dataset is processed correctly for different splits regardless of whether the processed data already exists"""
    datasets_dir = datasets_dir_factory("ScanObjectNN/**/*", symlinks=False)

    dataset = ScanObjectNN(
        root=datasets_dir,
        split=split,
        variant=variant,
        background=background,
        train=train,
        show_progress=False,
        force_process=True,
    )
    assert len(dataset) > 0


def test_scanobjectnn_download_redownloads_corrupt_resource(
    datasets_dir_factory: Callable[..., Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A cached archive that fails the checksum is re-downloaded (overwritten) instead of being reused."""
    datasets_dir = datasets_dir_factory("ScanObjectNN/processed/**/*")
    dataset = ScanObjectNN(root=datasets_dir, show_progress=False)
    archive = _zip_bytes("h5_files/dummy.h5")
    monkeypatch.setattr(dataset, "md5", hashlib.md5(archive).hexdigest())
    resource_path = Path(dataset.raw_dir, dataset.resource)
    resource_path.parent.mkdir(parents=True, exist_ok=True)
    resource_path.write_bytes(b"corrupt")

    with _serve_download(archive) as mock_urlopen:
        dataset.download(show_progress=False)

    assert mock_urlopen.called
    assert Path(dataset.raw_dir, "dummy.h5").exists()
    assert not resource_path.exists()


def test_scanobjectnn_download_raises_when_redownload_still_corrupt(
    datasets_dir_factory: Callable[..., Path],
) -> None:
    """If the re-downloaded archive still fails the checksum, download() raises with both hashes."""
    datasets_dir = datasets_dir_factory("ScanObjectNN/processed/**/*")
    dataset = ScanObjectNN(root=datasets_dir, show_progress=False)
    resource_path = Path(dataset.raw_dir, dataset.resource)
    resource_path.parent.mkdir(parents=True, exist_ok=True)
    resource_path.write_bytes(b"corrupt")

    with _serve_download(b"still corrupt") as mock_urlopen:
        with pytest.raises(RuntimeError, match="MD5 hash mismatch") as excinfo:
            dataset.download(show_progress=False)

    assert mock_urlopen.called
    assert dataset.md5 in str(excinfo.value)
    assert hashlib.md5(b"still corrupt").hexdigest() in str(excinfo.value)


def test_scanobjectnn_force_download_overwrites_valid_resource(
    datasets_dir_factory: Callable[..., Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """force re-downloads the archive even when a valid one is already on disk."""
    datasets_dir = datasets_dir_factory("ScanObjectNN/raw/**/*")
    dataset = ScanObjectNN(root=datasets_dir, show_progress=False)
    archive = _zip_bytes("h5_files/dummy.h5")
    monkeypatch.setattr(dataset, "md5", hashlib.md5(archive).hexdigest())
    resource_path = Path(dataset.raw_dir, dataset.resource)
    resource_path.write_bytes(archive)

    with _serve_download(archive) as mock_urlopen:
        dataset.download(force=True, show_progress=False)

    assert mock_urlopen.called


@pytest.mark.parametrize("split", ["invalid"])
@pytest.mark.parametrize("variant", ["invalid"])
def test_scanobjectnn_dataset_invalid_split(
    datasets_dir_factory: Callable[..., Path], split: str, variant: str
) -> None:
    """Raises an error if the split is invalid or not supported"""
    datasets_dir = datasets_dir_factory("ScanObjectNN/**/*")

    with pytest.raises(ValueError):
        _ = ScanObjectNN(root=datasets_dir, split=split, variant=variant, show_progress=False)


@pytest.mark.parametrize("label", list(ScanObjectNN.original_classes))
def test_scanobjectnn_dataset_labels(datasets_dir_factory: Callable[..., Path], label: str) -> None:
    """Test that the dataset is loaded correctly for a specific category"""
    datasets_dir = datasets_dir_factory("ScanObjectNN/processed/**/*")

    dataset = ScanObjectNN(root=datasets_dir, split="main", train=False, classes=[label], show_progress=False)  # type: ignore[list-item]
    assert len(dataset) > 0
    assert len(dataset.classes) == 1
    assert dataset.classes[0] == label
    assert dataset.class_to_idx[label] == 0


def test_scanobjectnn_dataset_transform_called(datasets_dir_factory: Callable[..., Path]) -> None:
    """Test that the dataset is transformed correctly after being processed"""
    datasets_dir = datasets_dir_factory("ScanObjectNN/processed/**/*")

    transform = Mock(side_effect=lambda data: data)
    dataset = ScanObjectNN(root=datasets_dir, split="main", train=False, transform=transform, show_progress=False)
    _ = list(dataset)
    assert transform.call_count == len(dataset)
