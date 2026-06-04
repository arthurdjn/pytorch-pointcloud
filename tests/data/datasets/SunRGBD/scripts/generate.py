"""Generate a tiny SUN RGB-D fixture by subsampling real scenes.

SUN RGB-D's loader streams from the official zips (metadata + split out of `SUNRGBDtoolbox.zip`,
depth/RGB out of `SUNRGBD.zip`) without ever extracting them, so the fixture ships *tiny subset
zips* in `raw/` rather than loose files. Two commands, mirroring the other datasets:

- `raw`: writes a small `SUNRGBDtoolbox.zip` (the real 0.16 MB `allsplit.mat`, so `read_split`'s
  full-count assert passes, plus a `SUNRGBDMeta.mat` rebuilt with only the kept scenes) and a small
  `SUNRGBD.zip` (only those scenes' depth/RGB PNG members, copied from the real release).
- `process`: runs the unchanged loader on that raw fixture and subsamples each cloud to
  `--num-points`, writing `processed/<split>/` (so processed is literally `process(raw)`).

The default source is `$TORCH_POINTCLOUD_DATA_DIR/SunRGBD` (override with `--src-dir`).

Usage:
    uv run --no-sync python scripts/generate.py raw
    uv run --no-sync python scripts/generate.py process
"""

import io
import shutil
import zipfile
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Any, List

import numpy as np
import scipy.io as sio
import torch
from tqdm import tqdm

from torch_pointcloud.config import DATA_DIR
from torch_pointcloud.datasets import SunRGBD
from torch_pointcloud.datasets.sunrgbd import (
    SUNRGBD_RELEASE_ZIP,
    SUNRGBD_TOOLBOX_ZIP,
    TOOLBOX_META_MEMBER,
    TOOLBOX_SPLIT_MEMBER,
    _box_list,
    rebase_sequence,
)
from torch_pointcloud.utils.data import DataKeys

BOX_FIELDS = ("classname", "centroid", "coeffs", "basis", "orientation")
META_FIELDS = ("sequenceName", "depthpath", "rgbpath", "K", "Rtilt", "groundtruth3DBB")
SPLIT_KEYS = {"train": "alltrain", "val": "alltest"}


def main() -> None:
    args = parse_args()
    if args.command == "raw":
        generate_raw(args)
    elif args.command == "process":
        generate_processed(args)


def parse_args() -> Namespace:
    parser = ArgumentParser(description="Generate SUN RGB-D test data by subsampling real scenes.")
    common = ArgumentParser(add_help=False)
    common.add_argument(
        "dst_dir",
        type=str,
        nargs="?",
        default=str(Path(__file__).resolve().parent.parent),
        help="Output SunRGBD fixture directory (default: the fixture dir next to this script).",
    )
    common.add_argument("--splits", type=str, nargs="+", default=["train", "val"], help="Splits to generate.")
    common.add_argument("--num-scenes", type=int, default=3, help="Number of scenes kept per split.")

    subparsers = parser.add_subparsers(dest="command", required=True)
    raw = subparsers.add_parser("raw", parents=[common], help="Write the tiny subset zips into raw/.")
    raw.add_argument(
        "--src-dir",
        type=str,
        default=str(Path(DATA_DIR) / "SunRGBD"),
        help="Real SunRGBD directory holding raw/ (default: $TORCH_POINTCLOUD_DATA_DIR/SunRGBD).",
    )
    process = subparsers.add_parser("process", parents=[common], help="Process raw/ into processed/.")
    process.add_argument("--num-points", type=int, default=2048, help="Number of points kept per scene.")
    process.add_argument("--seed", type=int, default=42, help="Random seed for point subsampling.")
    return parser.parse_args()


def generate_raw(args: Namespace) -> None:
    src = Path(args.src_dir)
    dst_raw = Path(args.dst_dir) / "raw"
    dst_raw.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(src / "raw" / SUNRGBD_TOOLBOX_ZIP) as z:
        allsplit_bytes = z.read(TOOLBOX_SPLIT_MEMBER)
        meta = sio.loadmat(io.BytesIO(z.read(TOOLBOX_META_MEMBER)), struct_as_record=False, squeeze_me=True)
        split = sio.loadmat(io.BytesIO(allsplit_bytes), struct_as_record=False, squeeze_me=True)

    by_seq = {rebase_sequence(str(e.sequenceName)): e for e in meta["SUNRGBDMeta"]}
    entries = []
    for sp in args.splits:
        present = [s for s in (rebase_sequence(str(p)) for p in split[SPLIT_KEYS[sp]]) if s in by_seq]
        entries.extend(by_seq[s] for s in present[: args.num_scenes])

    with zipfile.ZipFile(dst_raw / SUNRGBD_TOOLBOX_ZIP, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr(TOOLBOX_SPLIT_MEMBER, allsplit_bytes)
        z.writestr(TOOLBOX_META_MEMBER, _dump_meta(entries))

    with (
        zipfile.ZipFile(_resolve_release(src)) as zin,
        zipfile.ZipFile(dst_raw / SUNRGBD_RELEASE_ZIP, "w", zipfile.ZIP_DEFLATED) as zout,
    ):
        for e in tqdm(entries, total=len(entries), desc="Copying PNGs"):
            for path in (str(e.depthpath), str(e.rgbpath)):
                member = f"SUNRGBD/{rebase_sequence(path)}"
                zout.writestr(member, zin.read(member))
    print(f"Wrote {len(entries)} scenes into {dst_raw}")


def generate_processed(args: Namespace) -> None:
    dst = Path(args.dst_dir)
    generator = torch.Generator().manual_seed(args.seed)
    for split in args.splits:
        shutil.rmtree(dst / "processed" / split, ignore_errors=True)
        dataset = SunRGBD(root=dst.parent, split=split, force_process=True, show_progress=False)
        for scene_dir in dataset.processed_files:
            _subsample_scene(scene_dir, args.num_points, generator)
        print(f"Processed {len(dataset.processed_files)} {split} scenes")


def _resolve_release(src: Path) -> Path:
    for candidate in (src / "raw" / SUNRGBD_RELEASE_ZIP, src / SUNRGBD_RELEASE_ZIP):
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Could not find {SUNRGBD_RELEASE_ZIP!r} under {src!r} or its raw/ subdirectory.")


def _dump_meta(entries: List[Any]) -> bytes:
    """Rebuild a `SUNRGBDMeta` struct array holding only the kept scenes, as `.mat` bytes."""
    cells = np.empty((len(entries),), dtype=object)
    for i, entry in enumerate(entries):
        boxes = _box_list(entry.groundtruth3DBB)
        gt: Any = np.zeros((0, 0))
        if boxes:
            gt = np.empty((len(boxes),), dtype=object)
            for j, box in enumerate(boxes):
                gt[j] = {field: _field_value(box, field) for field in BOX_FIELDS}
        cells[i] = {field: _field_value(entry, field) for field in META_FIELDS[:-1]}
        cells[i]["groundtruth3DBB"] = gt
    buffer = io.BytesIO()
    sio.savemat(buffer, {"SUNRGBDMeta": cells})
    return buffer.getvalue()


def _field_value(obj: Any, field: str) -> Any:
    value = getattr(obj, field)
    if field in ("classname", "sequenceName", "depthpath", "rgbpath"):
        return str(value)
    return np.asarray(value, dtype=np.float64)


def _subsample_scene(scene_dir: Path, num_points: int, generator: torch.Generator) -> None:
    """Subsample the per-point `.npy` arrays to `num_points` in place; boxes are kept verbatim."""
    n = int(np.load(scene_dir / f"{DataKeys.POS}.npy").shape[0])
    if n >= num_points:
        idx = torch.randperm(n, generator=generator)[:num_points].numpy()
    else:
        idx = torch.randint(0, n, (num_points,), generator=generator).numpy()
    for name in (DataKeys.POS, DataKeys.COLOR):
        np.save(scene_dir / f"{name}.npy", np.load(scene_dir / f"{name}.npy")[idx])


if __name__ == "__main__":
    main()
