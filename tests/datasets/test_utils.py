import io
import tarfile
import tempfile
import zipfile
from pathlib import Path
from typing import Generator
from unittest.mock import Mock, patch

import pytest

from torch_pointcloud.datasets.utils import (
    compute_hash,
    download_url,
    extract_tar,
    extract_zip,
    is_hash_valid,
    urltailname,
)


def create_zip(out_dir: Path, nested: bool = False) -> Path:
    zip_path = out_dir / "test.zip"
    with tempfile.TemporaryDirectory(prefix="tmp_", dir=out_dir) as tmp_dir:
        file_path = Path(tmp_dir) / "file.txt"
        file_path.write_text("content")

        with zipfile.ZipFile(zip_path, "w") as zip_file:
            out_path = "files/file.txt" if nested else "file.txt"
            zip_file.write(file_path, out_path)

    return zip_path


def create_tar(out_dir: Path, symlink: bool = False) -> Path:
    tar_path = out_dir / "test.tar.gz"
    with tarfile.open(tar_path, "w:gz") as tar_file:
        info = tarfile.TarInfo("files/file.txt")
        content = b"content"
        info.size = len(content)
        tar_file.addfile(info, io.BytesIO(content))

        if symlink:
            link = tarfile.TarInfo("files/link.txt")
            link.type = tarfile.SYMTYPE
            link.linkname = "missing-target.txt"
            tar_file.addfile(link)

    return tar_path


@pytest.fixture
def mock_response() -> Mock:
    response = Mock()
    response.read.side_effect = [b"content", b""]  # Return content once, then empty
    response.length = 7  # len("content")
    return response


@pytest.fixture
def mock_urlopen(mock_response: Mock) -> Generator[Mock, None, None]:
    with patch("torch_pointcloud.datasets.utils.urlopen") as mock:
        mock.return_value.__enter__.return_value = mock_response
        yield mock


def test_urltailname() -> None:
    """Test that the URL tail name is correctly extracted."""
    assert urltailname("https://example.com/file.zip") == "file.zip"
    assert urltailname("https://example.com/path/to/file.zip") == "file.zip"
    assert urltailname("https://example.com/my%20file.zip") == "my file.zip"


def test_download_url_default(mock_urlopen: Mock, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that the file is downloaded at the current working directory with the URL file name."""
    monkeypatch.chdir(tmp_path)

    path = download_url("https://example.com/file.txt", show_progress=False)
    assert path == "file.txt"
    assert Path(path).exists()
    assert Path(path).read_text() == "content"


def test_download_url_custom_path(mock_urlopen: Mock, tmp_path: Path) -> None:
    """Test that the file is downloaded to a custom path."""
    path = download_url("https://example.com/file.txt", tmp_path / "custom.txt", show_progress=False)
    assert path == str(tmp_path / "custom.txt")
    assert Path(path).exists()
    assert Path(path).read_text() == "content"


def test_download_url_existing_file_is_kept(mock_urlopen: Mock, tmp_path: Path) -> None:
    """Test that an existing file is returned as-is when overwrite is not requested."""
    file_path = tmp_path / "file.txt"
    file_path.write_text("old content")

    path = download_url("https://example.com/file.txt", file_path, show_progress=False)
    assert not mock_urlopen.called
    assert Path(path).read_text() == "old content"


def test_download_url_overwrite_existing_file(mock_urlopen: Mock, tmp_path: Path) -> None:
    """Test that an existing file is re-downloaded when overwrite is requested."""
    file_path = tmp_path / "file.txt"
    file_path.write_text("old content")

    path = download_url("https://example.com/file.txt", file_path, show_progress=False, overwrite=True)
    assert mock_urlopen.called
    assert Path(path).read_text() == "content"


def test_download_no_progress(mock_urlopen: Mock, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Test that the show_progress bar is not displayed when downloading a file."""
    _ = download_url("https://example.com/file.txt", tmp_path / "file.txt", show_progress=False)
    captured = capsys.readouterr()
    assert "Downloading" not in captured.err


def test_download_url_with_progress(mock_urlopen: Mock, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Test that the show_progress bar is displayed when downloading a file."""
    _ = download_url("https://example.com/file.txt", tmp_path / "file.txt", show_progress=True)
    captured = capsys.readouterr()
    assert "Downloading" in captured.err


def test_extract_zip(tmp_path: Path) -> None:
    """Test that the zip file is extracted to the correct directory."""
    zip_path = create_zip(tmp_path)
    result = extract_zip(zip_path, tmp_path, show_progress=False)
    assert Path(result).exists()
    assert Path(result, "file.txt").exists()
    assert len(list(Path(result).glob("**/*.txt"))) == 1


def test_extract_zip_nested_relative_to(tmp_path: Path) -> None:
    """Test that the zip file is extracted relative to a specific directory."""
    zip_path = create_zip(tmp_path, nested=True)
    result = extract_zip(zip_path, tmp_path, relative_to="files", show_progress=False)
    assert Path(result).exists()
    assert Path(result, "file.txt").exists()
    assert len(list(Path(result).glob("**/*.txt"))) == 1


def test_extract_zip_no_progress(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Test that the show_progress bar is not displayed when extracting a zip file."""
    zip_path = create_zip(tmp_path)
    _ = extract_zip(zip_path, tmp_path, show_progress=False)
    captured = capsys.readouterr()
    assert "Extracting" not in captured.err


def test_extract_zip_with_progress(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Test that the show_progress bar is displayed when extracting a zip file."""
    zip_path = create_zip(tmp_path)
    _ = extract_zip(zip_path, tmp_path, show_progress=True)
    captured = capsys.readouterr()
    assert "Extracting" in captured.err


def test_extract_tar(tmp_path: Path) -> None:
    """Test that the tar file is extracted to the correct directory."""
    tar_path = create_tar(tmp_path)
    result = extract_tar(tar_path, tmp_path / "out", show_progress=False)
    assert Path(result, "files", "file.txt").read_text() == "content"


def test_extract_tar_relative_to(tmp_path: Path) -> None:
    """Test that the tar file is extracted relative to a specific directory."""
    tar_path = create_tar(tmp_path)
    result = extract_tar(tar_path, tmp_path / "out", relative_to="files", show_progress=False)
    assert Path(result, "file.txt").read_text() == "content"


def test_extract_tar_skips_non_regular_members(tmp_path: Path) -> None:
    """Test that symlink members (which have no extractable file object) are skipped instead of crashing."""
    tar_path = create_tar(tmp_path, symlink=True)
    result = extract_tar(tar_path, tmp_path / "out", show_progress=False)
    assert Path(result, "files", "file.txt").read_text() == "content"
    assert not Path(result, "files", "link.txt").exists()


@pytest.fixture
def test_content_file(tmp_path: Path) -> Path:
    test_file = tmp_path / "test.txt"
    test_file.write_text("test content")
    return test_file


@pytest.mark.parametrize(
    "hash_type, expected_hash",
    [
        ("md5", "9473fdd0d880a43c21b7778d34872157"),
        ("sha1", "1eebdf4fdc9fc7bf283031b93f9aef3338de9052"),
        ("sha256", "6ae8a75555209fd6c44157c0aed8016e763ff435a19cf186f76863140143ff72"),
        (
            "sha512",
            "0cbf4caef38047bba9a24e621a961484e5d2a92176a859e7eb27df343dd34eb98d538a6c5f4da1ce302ec250b821cc001e46cc97a704988297185a4df7e99602",
        ),
    ],
)
def test_compute_hash(test_content_file: Path, hash_type: str, expected_hash: str) -> None:
    """Test that the file hash is computed correctly for every supported hash type."""
    assert compute_hash(test_content_file, hash_type) == expected_hash  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "hash_type, expected_hash",
    [
        ("md5", "9473fdd0d880a43c21b7778d34872157"),
        ("sha1", "1eebdf4fdc9fc7bf283031b93f9aef3338de9052"),
        ("sha256", "6ae8a75555209fd6c44157c0aed8016e763ff435a19cf186f76863140143ff72"),
        (
            "sha512",
            "0cbf4caef38047bba9a24e621a961484e5d2a92176a859e7eb27df343dd34eb98d538a6c5f4da1ce302ec250b821cc001e46cc97a704988297185a4df7e99602",
        ),
    ],
)
def test_is_hash_valid(test_content_file: Path, hash_type: str, expected_hash: str) -> None:
    """Test that hash validation works correctly."""
    assert is_hash_valid(test_content_file, expected_hash, hash_type)  # type: ignore[arg-type]


@pytest.mark.parametrize("hash_type", ["md5", "sha1", "sha256", "sha512"])
def test_is_hash_valid_wrong_hash(test_content_file: Path, hash_type: str) -> None:
    """Test that hash validation fails when the hash is incorrect."""
    assert not is_hash_valid(test_content_file, "wrong_hash", hash_type)  # type: ignore[arg-type]


def test_is_hash_valid_unsupported_type(test_content_file: Path) -> None:
    """Test that unsupported hash types return False."""
    assert not is_hash_valid(test_content_file, "some_hash", "unsupported_type")  # type: ignore[arg-type]


def test_is_hash_valid_nonexistent_file() -> None:
    """Test that validation fails for nonexistent files."""
    assert not is_hash_valid("nonexistent_file.txt", "some_hash", "md5")


def test_is_hash_valid_none_hash() -> None:
    """Test that validation is skipped when expected hash is None."""
    assert is_hash_valid("nonexistent_file.txt", None)
