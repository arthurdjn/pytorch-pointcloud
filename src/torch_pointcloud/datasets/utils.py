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
            if progress:
                with tqdm(
                    total=response.length,
                    desc=description,
                    unit="B",
                    unit_scale=True,
                    unit_divisor=1024,
                ) as pbar:
                    while chunk := response.read(chunk_size):
                        fh.write(chunk)
                        pbar.update(len(chunk))
            else:
                while chunk := response.read(chunk_size):
                    fh.write(chunk)

    return file_path.as_posix()
