import os
import tempfile
import zipfile
from pathlib import Path
from typing import Generator
from unittest.mock import Mock, patch

import pytest

from torch_pointcloud.datasets.utils import download_url, extract_zip, urltailname


def create_zip(out_dir: Path, nested: bool = False) -> Path:
    zip_path = out_dir / "test.zip"
    with tempfile.TemporaryDirectory(prefix="tmp_", dir=out_dir) as tmp_dir:
        file_path = Path(tmp_dir) / "file.txt"
        file_path.write_text("content")

        with zipfile.ZipFile(zip_path, "w") as zip_file:
            out_path = "files/file.txt" if nested else "file.txt"
            zip_file.write(file_path, out_path)

    return zip_path


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
    # Test basic cases
    assert urltailname("https://example.com/file.zip") == "file.zip"
    # Test with a path
    assert urltailname("https://example.com/path/to/file.zip") == "file.zip"
    # Test with URL encoded characters
    assert urltailname("https://example.com/my%20file.zip") == "my file.zip"


def test_download_url_default(mock_urlopen: Mock, tmp_path: Path) -> None:
    """Test that the file is downloaded at the current working directory with the URL file name."""
    os.chdir(tmp_path)

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
