import hashlib
import json
import shutil
import tarfile
import zipfile
from pathlib import Path
from ssl import SSLContext, create_default_context
from typing import Any, Callable, Dict, Literal, Optional, Union
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
        'file.zip'
        >>> urltailname("https://example.com/path/to/my%20file.zip")
        'my file.zip'
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
) -> Optional[int]:
    """Get the size of a URL.

    Args:
        url: The URL to get the size of.
        timeout: Optional timeout in seconds for the request.
        cafile: Optional path to a CA file.
        capath: Optional path to a directory with CA certificates.
        cadefault: Whether to use the default CA store.
        context: Optional `SSLContext` for the request. Built from `cafile` / `capath` / `cadefault`
            when not provided.

    Returns:
        The size of the URL in bytes, or `None` if the response has no `content-length` header.

    Examples:
        >>> urlsize("https://example.com/file.zip")  # doctest: +SKIP
        1024
    """
    if context is None and (cafile or capath or cadefault):
        context = create_default_context(cafile=cafile, capath=capath)
    req = Request(url, method="HEAD")
    with urlopen(req, timeout=timeout, context=context) as f:
        size = f.headers.get("content-length")
        return int(size) if size is not None else None


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
            size does not match the expected size (when the remote size is unknown, the local file is kept). If `False`,
            the local file will not be overwritten if it already exists.

    Returns:
        The local path to the downloaded file.

    Examples:
        >>> download_url("https://example.com/file.zip")  # doctest: +SKIP
        "file.zip"
        >>> download_url("https://example.com/my%20file.zip", "my_file.zip", show_progress=False)  # doctest: +SKIP
        "my_file.zip"
    """
    file_path = Path(file_path if file_path else urltailname(url))
    file_path.parent.mkdir(parents=True, exist_ok=True)

    if file_path.exists() and not overwrite:
        return file_path.as_posix()
    if file_path.exists() and overwrite == "incomplete":
        expected_size = urlsize(url)
        if expected_size is None or file_path.stat().st_size == expected_size:
            return file_path.as_posix()

    part_path = file_path.with_name(file_path.name + ".part")
    with urlopen(Request(url, headers={"User-Agent": USER_AGENT})) as response:
        with open(part_path, "wb") as fh:
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
    part_path.replace(file_path)

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
        >>> extract_zip("A.zip", "A")  # doctest: +SKIP
        "A"
    """
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        members = zip_ref.namelist()
        for member in tqdm(members, total=len(members), desc="Extracting", disable=not show_progress):
            if member.endswith("/"):
                continue

            member_path = Path(member)
            if relative_to and member_path.is_relative_to(relative_to):
                member_path = member_path.relative_to(relative_to)

            dst_path = Path(out_dir, member_path).resolve()
            if not dst_path.is_relative_to(out_dir):
                continue

            dst_path.parent.mkdir(parents=True, exist_ok=True)

            with zip_ref.open(member) as source, open(dst_path, "wb") as dest:
                shutil.copyfileobj(source, dest)

    return out_dir.as_posix()


def extract_tar(
    tar_path: PathLike,
    dst_dir: PathLike,
    /,
    relative_to: PathLike = "",
    show_progress: bool = True,
) -> str:
    """Extract a tar file to a directory.

    Args:
        tar_path: The path to the tar file to extract.
        dst_dir: The directory to extract the tar file to.
        relative_to: If provided, extract the tar file relative to this directory.
        show_progress: Whether to display a progress bar.
    """
    dst_dir = Path(dst_dir).resolve()
    dst_dir.mkdir(parents=True, exist_ok=True)

    with tarfile.open(tar_path, "r:gz") as tar_ref:
        members = tar_ref.getmembers()

        for member in tqdm(members, total=len(members), desc="Extracting", disable=not show_progress):
            if not member.isfile():
                continue

            member_path = Path(member.name)
            if relative_to and member_path.is_relative_to(relative_to):
                member_path = member_path.relative_to(relative_to)

            dst_path = Path(dst_dir, member_path).resolve()
            if not dst_path.is_relative_to(dst_dir):
                continue

            dst_path.parent.mkdir(parents=True, exist_ok=True)
            with tar_ref.extractfile(member) as src, open(dst_path, "wb") as dst:  # type: ignore[union-attr]
                shutil.copyfileobj(src, dst)

    return dst_dir.as_posix()


def compute_hash(file_path: PathLike, hash_type: HashType = "md5") -> str:
    """Compute the hash of a file.

    Args:
        file_path: The path to the file to hash.
        hash_type: The type of hash to compute.

    Returns:
        The hex digest of the file's hash.

    Examples:
        >>> compute_hash("file.zip")  # doctest: +SKIP
        '9473fdd0d880a43c21b7778d34872157'
    """
    file_hash = SUPPORTED_HASH_TYPES[hash_type]()

    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            file_hash.update(chunk)

    return str(file_hash.hexdigest())


def check_cache_meta(meta_path: PathLike, meta: Dict[str, Any]) -> None:
    """Validate a processed cache against the parameters it was built with.

    Compares the JSON metadata stored next to a processed cache with the metadata the current
    constructor parameters would produce, and raises a `RuntimeError` on mismatch so a stale cache
    is never silently served. A missing metadata file (legacy cache) is accepted as-is.

    Args:
        meta_path: Path to the cache `meta.json` file.
        meta: The metadata the requested parameters would produce.

    Examples:
        >>> check_cache_meta("data/ModelNet10/processed/train.meta.json", {"classes": ["chair"]})  # doctest: +SKIP
    """
    meta_path = Path(meta_path)
    if not meta_path.exists():
        return
    cached_meta = json.loads(meta_path.read_text())
    if cached_meta != meta:
        raise RuntimeError(
            f"Processed cache at {meta_path.parent.as_posix()!r} was created with different parameters "
            f"(cached: {cached_meta}, requested: {meta}). Pass force_process=True to regenerate it."
        )


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
        >>> is_hash_valid("file.zip", "f7f6b4e3a3e0f8e4e3e3e3e3e3e3", "md5")  # doctest: +SKIP
        False
        >>> is_hash_valid("file.zip", "f7f6b4e3a3e0f8e4e3e3e3e3e3e3", "sha256")  # doctest: +SKIP
        True
    """

    if expected_hash is None:
        return True

    if hash_type not in SUPPORTED_HASH_TYPES:
        return False

    try:
        return compute_hash(file_path, hash_type) == expected_hash
    except Exception:
        return False
