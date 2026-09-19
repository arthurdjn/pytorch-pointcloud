"""Generate a tiny synthetic SUN RGB-D fixture in the layout of the official release.

SUN RGB-D's loader streams from the official zips (metadata + split out of `SUNRGBDtoolbox.zip`,
depth/RGB out of `SUNRGBD.zip`) without ever extracting them, so the fixture ships *tiny zips* in
`raw/` rather than loose files. No file of the official release is read: the images, cameras and
boxes are drawn from a seeded generator. Two commands, mirroring the other datasets:

- `raw`: writes a small `SUNRGBDtoolbox.zip` (an `allsplit.mat` and a `SUNRGBDMeta.mat` listing only
  the generated scenes) and a small `SUNRGBD.zip` (those scenes' depth PNG and RGB JPEG members).
- `process`: runs the unchanged loader on that raw fixture and subsamples each cloud to
  `--num-points`, writing `processed/<split>/` (so processed is literally `process(raw)`).

Usage:
    uv run --no-sync python scripts/generate.py raw
    uv run --no-sync python scripts/generate.py process
"""

import io
import shutil
import zipfile
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Any, Dict

import numpy as np
import scipy.io as sio
import torch
from PIL import Image

from torch_pointcloud.datasets import SunRGBD
from torch_pointcloud.datasets.sunrgbd import (
    SUNRGBD_CLASSES,
    SUNRGBD_RELEASE_ZIP,
    SUNRGBD_TOOLBOX_ZIP,
    TOOLBOX_META_MEMBER,
    TOOLBOX_SPLIT_MEMBER,
)
from torch_pointcloud.utils.data import DataKeys

SPLIT_KEYS = {"train": "alltrain", "val": "alltest"}


def main() -> None:
    args = parse_args()
    if args.command == "raw":
        generate_raw(args)
    elif args.command == "process":
        generate_processed(args)


def parse_args() -> Namespace:
    parser = ArgumentParser(description="Generate synthetic SUN RGB-D test data.")
    common = ArgumentParser(add_help=False)
    common.add_argument(
        "dst_dir",
        type=str,
        nargs="?",
        default=str(Path(__file__).resolve().parent.parent),
        help="Output SunRGBD fixture directory (default: the fixture dir next to this script).",
    )
    common.add_argument("--splits", type=str, nargs="+", default=["train", "val"], help="Splits to generate.")
    common.add_argument("--num-scenes", type=int, default=3, help="Number of scenes per split.")

    subparsers = parser.add_subparsers(dest="command", required=True)
    raw = subparsers.add_parser("raw", parents=[common], help="Write the tiny subset zips into raw/.")
    raw.add_argument("--seed", type=int, default=42, help="Random seed for the generated scenes.")
    process = subparsers.add_parser("process", parents=[common], help="Process raw/ into processed/.")
    process.add_argument("--num-points", type=int, default=2048, help="Number of points kept per scene.")
    process.add_argument("--seed", type=int, default=42, help="Random seed for point subsampling.")
    return parser.parse_args()


def generate_raw(args: Namespace) -> None:
    dst_raw = Path(args.dst_dir) / "raw"
    dst_raw.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    height, width = 72, 96

    split: Dict[str, Any] = {}
    entries = np.empty((len(args.splits) * args.num_scenes,), dtype=object)
    members: Dict[str, bytes] = {}
    for s, sp in enumerate(args.splits):
        sequences = [f"kv1/synthetic/{sp}_{i:04d}" for i in range(args.num_scenes)]
        split[SPLIT_KEYS[sp]] = np.array([f"/n/fs/sun3d/data/SUNRGBD/{seq}" for seq in sequences], dtype=object)

        for i, seq in enumerate(sequences):
            # A back wall above a floor ramp, a few closer rectangles, and holes of invalid zero depth.
            depth_mm = np.full((height, width), rng.integers(2500, 4000), dtype=np.uint16)
            depth_mm[height // 2 :] = np.linspace(depth_mm[0, 0], 1000, height - height // 2)[:, None]
            for _ in range(3):
                top, left = rng.integers(0, height // 2), rng.integers(0, width // 2)
                depth_mm[top : top + height // 4, left : left + width // 4] = rng.integers(1200, 2400)
            depth_mm[rng.random((height, width)) < 0.1] = 0
            rgb = np.clip(rng.integers(0, 256, size=3) + rng.integers(-20, 21, size=(height, width, 3)), 0, 255)

            # The last scene of each split has no box, the first one also holds a box of a class the loader drops.
            num_boxes = 0 if i == args.num_scenes - 1 else int(rng.integers(1, 5))
            class_names = list(rng.choice(SUNRGBD_CLASSES, size=num_boxes)) + (["lamp"] if i == 0 else [])
            gt: Any = np.zeros((0, 0))
            if class_names:
                gt = np.empty((len(class_names),), dtype=object)
                for j, class_name in enumerate(class_names):
                    theta = rng.uniform(-np.pi, np.pi)
                    cos, sin = np.cos(theta), np.sin(theta)
                    gt[j] = {
                        "classname": str(class_name),
                        "centroid": rng.uniform((-1.5, 1.5, -1.0), (1.5, 3.5, 0.0)),
                        "coeffs": rng.uniform(0.2, 0.8, size=3),
                        "basis": np.array([[cos, sin, 0.0], [-sin, cos, 0.0], [0.0, 0.0, 1.0]]),
                        "orientation": np.array([cos, sin, 0.0]),
                    }

            tilt = rng.uniform(0.0, 0.4)
            entries[s * args.num_scenes + i] = {
                "sequenceName": f"SUNRGBD/{seq}",
                "depthpath": f"/n/fs/sun3d/data/SUNRGBD/{seq}/depth/0000001.png",
                "rgbpath": f"/n/fs/sun3d/data/SUNRGBD/{seq}/image/0000001.jpg",
                "K": np.array([[0.72 * width, 0.0, width / 2], [0.0, 0.72 * width, height / 2], [0.0, 0.0, 1.0]]),
                "Rtilt": np.array(
                    [[1.0, 0.0, 0.0], [0.0, np.cos(tilt), np.sin(tilt)], [0.0, -np.sin(tilt), np.cos(tilt)]]
                ),
                "groundtruth3DBB": gt,
            }
            # The release stores depth rotated left by 3 bits, see `decode_depth`.
            members[f"SUNRGBD/{seq}/depth/0000001.png"] = _encode_image(depth_mm << 3, "PNG")
            members[f"SUNRGBD/{seq}/image/0000001.jpg"] = _encode_image(rgb.astype(np.uint8), "JPEG")

    with zipfile.ZipFile(dst_raw / SUNRGBD_TOOLBOX_ZIP, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr(TOOLBOX_SPLIT_MEMBER, _dump_mat(split))
        z.writestr(TOOLBOX_META_MEMBER, _dump_mat({"SUNRGBDMeta": entries}))

    with zipfile.ZipFile(dst_raw / SUNRGBD_RELEASE_ZIP, "w", zipfile.ZIP_DEFLATED) as z:
        for member, content in members.items():
            z.writestr(member, content)
    print(f"Wrote {len(entries)} scenes into {dst_raw}")


def generate_processed(args: Namespace) -> None:
    dst = Path(args.dst_dir)
    generator = torch.Generator().manual_seed(args.seed)
    for split in args.splits:
        shutil.rmtree(dst / "processed" / split, ignore_errors=True)
        dataset = SunRGBD(root=dst.parent, train=split == "train", force_process=True, show_progress=False)
        for scene_dir in dataset.processed_files:
            _subsample_scene(scene_dir, args.num_points, generator)
        # The committed cache stays unmarked: the tests read it as the legacy layout without a completion marker.
        (dst / "processed" / split / "meta.json").unlink()
        print(f"Processed {len(dataset.processed_files)} {split} scenes")


def _dump_mat(variables: Dict[str, Any]) -> bytes:
    buffer = io.BytesIO()
    sio.savemat(buffer, variables)
    return buffer.getvalue()


def _encode_image(image: np.ndarray, image_format: str) -> bytes:
    buffer = io.BytesIO()
    Image.fromarray(image).save(buffer, format=image_format)
    return buffer.getvalue()


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
