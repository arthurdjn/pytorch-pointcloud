import io
import os
import subprocess
import sys
import tarfile
import tempfile
import threading
import time
import zipfile
from http.client import HTTPMessage
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from ssl import SSLContext
from typing import Generator
from unittest.mock import Mock, patch
from urllib.error import URLError
from urllib.request import HTTPRedirectHandler

import pytest

from torch_pointcloud.datasets.utils import (
    atomic_path,
    check_terms_accepted,
    compute_hash,
    download_url,
    extract_tar,
    extract_zip,
    is_hash_valid,
    remove_stale_temporaries,
    urlsize,
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


def test_download_url_passes_timeout(mock_urlopen: Mock, tmp_path: Path) -> None:
    """The timeout is forwarded to `urlopen` so a stalled server raises instead of hanging."""
    download_url("https://example.com/file.txt", tmp_path / "file.txt", show_progress=False, timeout=5.0)
    assert mock_urlopen.call_args.kwargs["timeout"] == 5.0


def test_download_url_sends_headers_without_forwarding_them_on_redirect(mock_urlopen: Mock, tmp_path: Path) -> None:
    download_url(
        "https://example.com/file.txt",
        tmp_path / "file.txt",
        show_progress=False,
        headers={"Authorization": "Bearer t"},
    )
    request = mock_urlopen.call_args.args[0]
    assert request.get_header("Authorization") == "Bearer t"
    assert request.get_header("User-agent") == "torch_pointcloud"

    handler = HTTPRedirectHandler()
    redirected = handler.redirect_request(request, io.BytesIO(), 302, "Found", HTTPMessage(), "https://cdn.example/x")
    assert redirected is not None and redirected.get_header("Authorization") is None
    assert redirected.get_header("User-agent") == "torch_pointcloud"


def test_download_url_failure_removes_the_partial_file(mock_urlopen: Mock, mock_response: Mock, tmp_path: Path) -> None:
    mock_response.read.side_effect = [b"cont", OSError("connection reset")]
    with pytest.raises(OSError, match="connection reset"):
        download_url("https://example.com/file.txt", tmp_path / "file.txt", show_progress=False)
    assert not list(tmp_path.iterdir())


def test_download_url_concurrent_downloads_do_not_share_a_partial_file(tmp_path: Path) -> None:
    file_path = tmp_path / "file.txt"

    def read_inner(*args: object) -> bytes:
        return next(inner_chunks)

    def read_outer(*args: object) -> bytes:
        chunk = next(outer_chunks)
        if chunk == b"tent":
            # A second download of the same file starts and finishes while the first one is half written.
            with patch("torch_pointcloud.datasets.utils.urlopen") as inner_urlopen:
                inner_urlopen.return_value.__enter__.return_value = Mock(read=read_inner, length=7)
                download_url("https://example.com/file.txt", file_path, overwrite=True, show_progress=False)
            assert file_path.read_bytes() == b"content"
        return chunk

    inner_chunks = iter([b"content", b""])
    outer_chunks = iter([b"con", b"tent", b""])
    with patch("torch_pointcloud.datasets.utils.urlopen") as outer_urlopen:
        outer_urlopen.return_value.__enter__.return_value = Mock(read=read_outer, length=7)
        download_url("https://example.com/file.txt", file_path, show_progress=False)

    assert file_path.read_bytes() == b"content"
    assert [path.name for path in tmp_path.iterdir()] == ["file.txt"]


def test_download_url_incomplete_keeps_the_local_file_when_offline(tmp_path: Path) -> None:
    file_path = tmp_path / "file.txt"
    file_path.write_bytes(b"content")

    with patch("torch_pointcloud.datasets.utils.urlopen", side_effect=URLError("network unreachable")):
        result = download_url("https://example.com/file.txt", file_path, overwrite="incomplete", show_progress=False)

    assert result == file_path.as_posix()
    assert file_path.read_bytes() == b"content"


def test_download_url_from_concurrent_processes(tmp_path: Path) -> None:
    num_processes = 3
    content = bytes(range(256)) * 4096
    all_downloads_started = threading.Barrier(num_processes)

    class HalfwayGatedHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            self.send_response(200)
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content[: len(content) // 2])
            self.wfile.flush()
            all_downloads_started.wait(timeout=120)
            self.wfile.write(content[len(content) // 2 :])

        def log_message(self, format: str, *args: object) -> None:
            pass

    script = (
        "import sys; from torch_pointcloud.datasets.utils import download_url; "
        "download_url(sys.argv[1], sys.argv[2], show_progress=False, chunk_size=4096, timeout=120.0)"
    )
    file_path = tmp_path / "cache" / "file.bin"
    with ThreadingHTTPServer(("127.0.0.1", 0), HalfwayGatedHandler) as server:
        threading.Thread(target=server.serve_forever, daemon=True).start()
        url = f"http://127.0.0.1:{server.server_port}/file.bin"
        command = [sys.executable, "-c", script, url, str(file_path)]
        processes = [subprocess.Popen(command, stderr=subprocess.DEVNULL) for _ in range(num_processes)]
        exit_codes = [process.wait() for process in processes]
        server.shutdown()

    assert exit_codes == [0] * num_processes
    assert file_path.read_bytes() == content
    assert [path.name for path in file_path.parent.iterdir()] == ["file.bin"]


def test_atomic_path_replaces_a_file_and_keeps_it_on_failure(tmp_path: Path) -> None:
    path = tmp_path / "data.txt"
    with atomic_path(path) as tmp_file:
        tmp_file.write_text("first")
        assert not path.exists()
    assert path.read_text() == "first"

    with pytest.raises(RuntimeError, match="interrupted"):
        with atomic_path(path) as tmp_file:
            tmp_file.write_text("second")
            raise RuntimeError("interrupted")
    assert path.read_text() == "first"
    assert [child.name for child in tmp_path.iterdir()] == ["data.txt"]


def test_atomic_path_concurrent_directory_writers(tmp_path: Path) -> None:
    path = tmp_path / "scene"
    with atomic_path(path) as first_dir, atomic_path(path) as second_dir:
        assert first_dir != second_dir
        first_dir.mkdir()
        second_dir.mkdir()
        (first_dir / "pos.npy").write_text("first")
        (second_dir / "pos.npy").write_text("second")

    assert (path / "pos.npy").read_text() in ("first", "second")
    assert [child.name for child in tmp_path.iterdir()] == ["scene"]


def test_atomic_path_replaces_a_directory_holding_an_undeletable_entry(tmp_path: Path) -> None:
    path = tmp_path / "scene"
    locked_dir = path / "locked"
    locked_dir.mkdir(parents=True)
    (locked_dir / "old.npy").write_text("old")
    (path / "pos.npy").write_text("old")
    locked_dir.chmod(0o500)
    try:
        with atomic_path(path) as tmp_dir:
            tmp_dir.mkdir()
            (tmp_dir / "pos.npy").write_text("new")
    finally:
        for leftover in tmp_path.rglob("locked"):
            leftover.chmod(0o700)

    assert [child.name for child in path.iterdir()] == ["pos.npy"]
    assert (path / "pos.npy").read_text() == "new"


def test_atomic_path_raises_when_the_directory_cannot_be_replaced(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "scene"
    path.mkdir()
    (path / "pos.npy").write_text("old")
    replace = Path.replace

    def failing_replace(self: Path, target: Path) -> Path:
        if self == path:
            raise PermissionError("locked")
        return replace(self, target)

    monkeypatch.setattr(Path, "replace", failing_replace)
    with pytest.raises(PermissionError, match="locked"):
        with atomic_path(path) as tmp_dir:
            tmp_dir.mkdir()
            (tmp_dir / "pos.npy").write_text("new")

    assert (path / "pos.npy").read_text() == "old"
    assert [child.name for child in tmp_path.iterdir()] == ["scene"]


def test_remove_stale_temporaries_keeps_recent_and_final_entries(tmp_path: Path) -> None:
    stale_dir = tmp_path / "scene.0a1b.tmp"
    stale_dir.mkdir()
    (stale_dir / "pos.npy").write_text("partial")
    stale_file = tmp_path / "meta.json.0a1b.tmp"
    stale_file.write_text("partial")
    live_dir = tmp_path / "scene.2c3d.tmp"
    live_dir.mkdir()
    final_dir = tmp_path / "scene"
    final_dir.mkdir()
    two_days_ago = time.time() - 2 * 24 * 3600
    for old_path in (stale_dir, stale_file, final_dir):
        os.utime(old_path, (two_days_ago, two_days_ago))

    remove_stale_temporaries(tmp_path)
    assert sorted(child.name for child in tmp_path.iterdir()) == ["scene", "scene.2c3d.tmp"]


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


def test_is_hash_valid_none_hash_warns_and_passes() -> None:
    """A missing expected hash skips validation but warns that the file is not verified."""
    with pytest.warns(UserWarning, match="No checksum provided"):
        assert is_hash_valid("nonexistent_file.txt", None)


def test_urlsize_returns_content_length(mock_urlopen: Mock, mock_response: Mock) -> None:
    """The remote size is read from the content-length header of a HEAD request."""
    mock_response.headers = {"content-length": "1024"}
    assert urlsize("https://example.com/file.zip") == 1024


def test_urlsize_returns_none_without_content_length(mock_urlopen: Mock, mock_response: Mock) -> None:
    """A response without a content-length header yields None, not a bogus size of 0."""
    mock_response.headers = {}
    assert urlsize("https://example.com/file.zip") is None


def test_urlsize_builds_ssl_context_from_ca_arguments(mock_urlopen: Mock, mock_response: Mock) -> None:
    """CA arguments become an `SSLContext` passed as `context`, never forwarded to `urlopen` directly."""
    mock_response.headers = {"content-length": "7"}
    assert urlsize("https://example.com/file.zip", cadefault=True) == 7

    kwargs = mock_urlopen.call_args.kwargs
    assert "cafile" not in kwargs
    assert "capath" not in kwargs
    assert "cadefault" not in kwargs
    assert isinstance(kwargs["context"], SSLContext)


def test_download_url_incomplete_keeps_matching_size(mock_urlopen: Mock, mock_response: Mock, tmp_path: Path) -> None:
    """overwrite='incomplete' keeps an existing file whose size matches the remote size."""
    mock_response.headers = {"content-length": "7"}
    file_path = tmp_path / "file.txt"
    file_path.write_text("content")

    result = download_url("https://example.com/file.txt", file_path, overwrite="incomplete", show_progress=False)

    assert Path(result).read_text() == "content"
    assert mock_urlopen.call_count == 1  # the HEAD request only


def test_download_url_incomplete_redownloads_mismatched_size(
    mock_urlopen: Mock, mock_response: Mock, tmp_path: Path
) -> None:
    """overwrite='incomplete' re-downloads an existing file whose size differs from the remote size."""
    mock_response.headers = {"content-length": "3"}
    file_path = tmp_path / "file.txt"
    file_path.write_text("partial-old")

    result = download_url("https://example.com/file.txt", file_path, overwrite="incomplete", show_progress=False)

    assert Path(result).read_text() == "content"
    assert mock_urlopen.call_count == 2


def test_download_url_incomplete_unknown_size_keeps_file(
    mock_urlopen: Mock, mock_response: Mock, tmp_path: Path
) -> None:
    """overwrite='incomplete' keeps the existing file when the remote size is unknown."""
    mock_response.headers = {}
    file_path = tmp_path / "file.txt"
    file_path.write_text("existing")

    result = download_url("https://example.com/file.txt", file_path, overwrite="incomplete", show_progress=False)

    assert Path(result).read_text() == "existing"
    assert mock_urlopen.call_count == 1  # the HEAD request only


def test_check_terms_accepted_passes_without_prompt(monkeypatch: pytest.MonkeyPatch) -> None:
    prompt = Mock(side_effect=AssertionError("must not prompt"))
    monkeypatch.setattr("builtins.input", prompt)
    check_terms_accepted(True, "ScanObjectNN", "https://example.org/terms")
    prompt.assert_not_called()


@pytest.mark.parametrize("answer", ["y", "Y", "yes", " Yes "])
def test_check_terms_accepted_prompt_accepts(monkeypatch: pytest.MonkeyPatch, answer: str) -> None:
    prompt = Mock(return_value=answer)
    monkeypatch.setattr("builtins.input", prompt)
    check_terms_accepted(False, "ScanObjectNN", "https://example.org/terms")
    assert "https://example.org/terms" in prompt.call_args.args[0]


@pytest.mark.parametrize("answer", ["", "n", "no", "maybe"])
def test_check_terms_accepted_prompt_refuses(monkeypatch: pytest.MonkeyPatch, answer: str) -> None:
    monkeypatch.setattr("builtins.input", lambda prompt: answer)
    with pytest.raises(RuntimeError, match="terms-of-use"):
        check_terms_accepted(False, "ScanObjectNN", "https://example.org/terms")


@pytest.mark.parametrize("error", [EOFError, OSError])
def test_check_terms_accepted_without_terminal_raises(monkeypatch: pytest.MonkeyPatch, error: type) -> None:
    monkeypatch.setattr("builtins.input", Mock(side_effect=error))
    with pytest.raises(RuntimeError, match="accept_terms=True"):
        check_terms_accepted(False, "ScanObjectNN", "https://example.org/terms")
