import ssl
from pathlib import Path
from urllib.parse import unquote, urlparse
from urllib.request import Request, urlopen

from tqdm import tqdm

from torch_pointcloud.utils.types import PathLike

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


def download_url(
    url: str,
    file_path: PathLike = "",
    chunk_size: int = 1024 * 32,
    description: str = "Downloading",
    progress: bool = True,
) -> str:
    """Download a file from a URL to a local path.

    Args:
        url: The URL to download the file from.
        file_path: The local path to save the file to (including the file name).
            If not provided, the file will be saved in the current working directory with the name taken from `Path
        chunk_size: The size of the chunks to download in bytes.
        description: The description to display in the progress bar.
        progress: Whether to display a progress bar.

    Returns:
        The local path to the downloaded file.

    Examples:
        >>> download_url("https://example.com/file.zip")
        "file.zip"
        >>> download_url("https://example.com/my%20file.zip", "my_file.zip", progress=False)
        "my_file.zip"
    """
    file_path = Path(file_path if file_path else urltailname(url))
    file_path.parent.mkdir(parents=True, exist_ok=True)

    context = ssl._create_unverified_context()
    with urlopen(Request(url, headers={"User-Agent": USER_AGENT}), context=context) as response:
        with open(file_path, "wb") as fh:
            with tqdm(
                total=response.length,
                desc=description,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
                disable=not progress,
            ) as pbar:
                while chunk := response.read(chunk_size):
                    fh.write(chunk)
                    pbar.update(len(chunk))

    return file_path.as_posix()


def extract_zip(zip_path: PathLike, out_dir: PathLike, relative_to: PathLike = "", progress: bool = True) -> str:
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
        progress: Whether to display a progress bar.

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
        for member in tqdm(members, total=len(members), desc="Extracting", disable=not progress):
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
