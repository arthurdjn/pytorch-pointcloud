import shutil
import ssl
import zipfile
from pathlib import Path
from urllib.request import Request, urlopen

from tqdm import tqdm

from torch_pointcloud.utils.types import PATH_LIKE

USER_AGENT = "torch_pointcloud"


def download_file(out_path: PATH_LIKE, url: str, chunk_size: int = 1024 * 32, description: str = "Downloading") -> str:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    context = ssl._create_unverified_context()
    with urlopen(Request(url, headers={"User-Agent": USER_AGENT}), context=context) as response:
        with open(out_path, "wb") as fh:
            with tqdm(total=response.length, desc=description, unit="B", unit_scale=True, unit_divisor=1024) as pbar:
                while chunk := response.read(chunk_size):
                    fh.write(chunk)
                    pbar.update(len(chunk))

    return out_path.as_posix()


def extract_zip(zip_path: PATH_LIKE, out_dir: PATH_LIKE, relative_to: PATH_LIKE = "") -> str:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        for member in zip_ref.namelist():
            if member.endswith("/"):
                continue

            member_path = Path(member)
            if member_path.is_relative_to(relative_to):
                member_path = member_path.relative_to(relative_to)

            out_path = Path(out_dir) / member_path
            out_path.parent.mkdir(parents=True, exist_ok=True)

            with zip_ref.open(member) as source, open(out_path, "wb") as dest:
                shutil.copyfileobj(source, dest)

    return out_dir.as_posix()
