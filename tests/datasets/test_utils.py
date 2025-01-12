import os
from pathlib import Path
from typing import Generator
from unittest.mock import Mock, patch

import pytest

from torch_pointcloud.datasets.utils import download_url, urltailname


def test_urltailname() -> None:
    # Test basic cases
    assert urltailname("https://example.com/file.zip") == "file.zip"
    # Test with a path
    assert urltailname("https://example.com/path/to/file.zip") == "file.zip"
    # Test with URL encoded characters
    assert urltailname("https://example.com/my%20file.zip") == "my file.zip"


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


def test_download_url_default(mock_urlopen: Mock, tmp_path: Path) -> None:
    os.chdir(tmp_path)

    path = download_url("https://example.com/file.txt", progress=False)
    assert path == "file.txt"
    assert Path(path).exists()
    assert Path(path).read_text() == "content"


def test_download_url_custom_path(mock_urlopen: Mock, tmp_path: Path) -> None:
    path = download_url("https://example.com/file.txt", tmp_path / "custom.txt", progress=False)
    assert path == str(tmp_path / "custom.txt")
    assert Path(path).exists()
    assert Path(path).read_text() == "content"


def test_download_no_progress(mock_urlopen: Mock, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    _ = download_url("https://example.com/file.txt", tmp_path / "file.txt", progress=False)
    captured = capsys.readouterr()
    assert "Downloading" not in captured.err


def test_download_url_with_progress(mock_urlopen: Mock, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    _ = download_url("https://example.com/file.txt", tmp_path / "file.txt", progress=True)
    captured = capsys.readouterr()
    assert "Downloading" in captured.err
