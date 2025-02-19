import hashlib
import shutil
import ssl
import zipfile
from pathlib import Path
from ssl import SSLContext
from typing import Callable, Dict, Literal, Optional, Union
from urllib.parse import unquote, urlparse
from urllib.request import Request, urlopen

from tqdm import tqdm

from torch_pointcloud.utils.types import PathLike

HashType = Literal["md5", "sha1", "sha256", "sha512"]
SUPPORTED_HASH_TYPES: Dict[HashType, Callable] = {
    "md5": hashlib.md5,
    "sha1": hashlib.sha1,
    "sha256": hashlib.sha256,
    "sha512": hashlib.sha512,
}

USER_AGENT = "torch_pointcloud"


def urltailname(url: str) -> str:
    """Get the name of the last segment of a URL.

    Args:
        url: The URL to get the basename from.

    Returns:
        The (decoded) name of the last segment of the URL.

    Examples:
        >>> urltailname("https://example.com/file.zip")
        "file.zip"
        >>> urltailname("https://example.com/path/to/my%20file.zip")
        "my file.zip"
    """
    return unquote(urlparse(url).path.split("/")[-1])


def urlsize(
    url: str,
    *,
    timeout: Optional[float] = None,
    cafile: Optional[str] = None,
    capath: Optional[str] = None,
    cadefault: bool = False,
    context: Optional[SSLContext] = None,
) -> int:
    """Get the size of a URL.

    Args:
        url: The URL to get the size of.
        **kwargs: Additional arguments to pass to the `requests.head` function.

    Returns:
        The size of the URL in bytes.

    Examples:
        >>> urlsize("https://example.com/file.zip")
        1024
    """
    req = Request(url, method="HEAD")
    with urlopen(req, timeout=timeout, cafile=cafile, capath=capath, cadefault=cadefault, context=context) as f:
        return int(f.headers.get("content-length", 0))


def download_url(
    url: str,
    file_path: PathLike = "",
    chunk_size: int = 1024 * 32,
    description: str = "Downloading",
    show_progress: bool = True,
    overwrite: Union[bool, Literal["incomplete"]] = False,
) -> str:
    """Download a file from a URL to a local path.

    Args:
        url: The URL to download the file from.
        file_path: The local path to save the file to (including the file name).
            If not provided, the file will be saved in the current working directory with the name taken from `Path
        chunk_size: The size of the chunks to download in bytes.
        description: The description to display in the progress bar.
        show_progress: Whether to display a progress bar.
        overwrite: Whether to overwrite the file if it already exists. If `True`, the local file will be overwritten
            even if it already exists. If `'incomplete'`, the local file will be overwritten if it already exists and its
            size does not match the expected size. If `False`, the local file will not be overwritten if it already
            exists.

    Returns:
        The local path to the downloaded file.

    Examples:
        >>> download_url("https://example.com/file.zip")
        "file.zip"
        >>> download_url("https://example.com/my%20file.zip", "my_file.zip", show_progress=False)
        "my_file.zip"
    """
    file_path = Path(file_path if file_path else urltailname(url))
    file_path.parent.mkdir(parents=True, exist_ok=True)

    if file_path.exists() and not overwrite:
        return file_path.as_posix()
    if file_path.exists() and overwrite == "incomplete" and file_path.stat().st_size == urlsize(url):
        return file_path.as_posix()

    context = ssl._create_unverified_context()
    with urlopen(Request(url, headers={"User-Agent": USER_AGENT}), context=context) as response:
        with open(file_path, "wb") as fh:
            with tqdm(
                total=response.length,
                desc=description,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
                disable=not show_progress,
            ) as pbar:
                while chunk := response.read(chunk_size):
                    fh.write(chunk)
                    pbar.update(len(chunk))

    return file_path.as_posix()


def extract_zip(zip_path: PathLike, out_dir: PathLike, relative_to: PathLike = "", show_progress: bool = True) -> str:
    """Extract a zip file to a directory.

    Args:
        zip_path: The path to the zip file to extract.
        out_dir: The directory to extract the zip file to.
        relative_to: If provided, extract the zip file relative to this directory.
            This is useful when the zip file contains nested directories but
            you want to extract the files on a specific level
            (e.g. for a nested zip `A.zip` containing files under directory `A`,
            then you will get `A/A/*.png` when extracting, but setting `relative_to="A"`
            will extract the files to `A/*.png` only).
        show_progress: Whether to display a progress bar.

    Returns:
        The path to the extracted directory.

    Examples:
        >>> extract_zip("A.zip", "A")
        "A"
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        members = zip_ref.namelist()
        for member in tqdm(members, total=len(members), desc="Extracting", disable=not show_progress):
            if member.endswith("/"):
                continue

            member_path = Path(member)
            if relative_to and member_path.is_relative_to(relative_to):
                member_path = member_path.relative_to(relative_to)

            out_path = Path(out_dir) / member_path
            out_path.parent.mkdir(parents=True, exist_ok=True)

            with zip_ref.open(member) as source, open(out_path, "wb") as dest:
                shutil.copyfileobj(source, dest)

    return out_dir.as_posix()


# Adapted from https://github.com/Project-MONAI/MONAI/blob/df1ba5d1e6aa9a0a1744b7ae3ff37ca114cec7bb/monai/apps/utils.py
def is_hash_valid(file_path: PathLike, expected_hash: Optional[str] = None, hash_type: HashType = "md5") -> bool:
    """Check if the hash of a file matches the expected hash.

    Args:
        file_path: The path to the file to check the hash of.
        expected_hash: The expected hash of the file.
        hash_type: The type of hash to use for the comparison.

    Returns:
        True if the hash of the file matches the expected hash, False otherwise.

    Examples:
        >>> is_hash_valid("file.zip", "f7f6b4e3a3e0f8e4e3e3e3e3e3e3", "md5")
        False
        >>> is_hash_valid("file.zip", "f7f6b4e3a3e0f8e4e3e3e3e3e3e3", "sha256")
        True
    """

    if expected_hash is None:
        return True

    hash_fn = SUPPORTED_HASH_TYPES.get(hash_type)
    if hash_fn is None:
        return False

    actual_hash = hash_fn()

    try:
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                actual_hash.update(chunk)
    except Exception:
        return False

    calculated_hash = actual_hash.hexdigest()

    return calculated_hash == expected_hash
