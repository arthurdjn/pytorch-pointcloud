"""Generate a tiny Paris-Lille-3D fixture by subsampling real `.ply` scans.

For each PLY file shipped with the 10-class benchmark
(`Lille1_1`, `Lille1_2`, `Lille2`, `Paris`), we keep `--num-points` randomly
sampled vertices and preserve the original schema
(`x`, `y`, `z`, `reflectance`, `class`).

The default source directory is `$TORCH_POINTCLOUD_DATA_DIR/ParisLille3D/raw`.

Usage:
    uv run --no-sync python scripts/generate.py raw ./raw
"""

from argparse import ArgumentParser, Namespace
from pathlib import Path

import numpy as np
import plyfile

from torch_pointcloud.config import DATA_DIR

DEFAULT_FILES = ("Lille1_1.ply", "Lille1_2.ply", "Lille2.ply", "Paris.ply")


def main() -> None:
    args = parse_args()
    if args.command == "raw":
        generate_raw(args)


def parse_args() -> Namespace:
    parser = ArgumentParser(description="Generate Paris-Lille-3D test data by subsampling real .ply files.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    raw_parser = subparsers.add_parser("raw", help="Generate raw test data")
    raw_parser.add_argument("dst_dir", type=str, help="Path to output raw directory.")
    raw_parser.add_argument(
        "--src-dir",
        type=str,
        default=str(Path(DATA_DIR) / "ParisLille3D" / "raw"),
        help="Source ParisLille3D raw directory (default: $TORCH_POINTCLOUD_DATA_DIR/ParisLille3D/raw).",
    )
    raw_parser.add_argument("--num-points", type=int, default=1024, help="Vertices kept per PLY after subsampling.")
    raw_parser.add_argument(
        "--files",
        nargs="+",
        default=list(DEFAULT_FILES),
        help="PLY file names to subsample (relative to --src-dir).",
    )
    raw_parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser.parse_args()


def generate_raw(args: Namespace) -> None:
    src_root = Path(args.src_dir)
    dst_root = Path(args.dst_dir)
    if not src_root.exists():
        raise FileNotFoundError(
            f"Source ParisLille3D raw directory not found: {src_root!r}. "
            f"Set --src-dir or TORCH_POINTCLOUD_DATA_DIR."
        )

    rng = np.random.default_rng(args.seed)
    dst_root.mkdir(parents=True, exist_ok=True)

    for fname in args.files:
        src = src_root / fname
        if not src.exists():
            raise FileNotFoundError(f"Missing source PLY: {src!r}")

        plydata = plyfile.PlyData.read(src.as_posix())
        v = plydata["vertex"].data
        n = v.shape[0]
        keep_n = min(args.num_points, n)
        indices = rng.choice(n, size=keep_n, replace=False)
        indices.sort()
        new_v = v[indices].copy()

        vertex_element = plyfile.PlyElement.describe(new_v, "vertex")
        dst = dst_root / fname
        plyfile.PlyData([vertex_element], text=False).write(dst.as_posix())
        print(f"  {dst}  (vertices={keep_n})")


if __name__ == "__main__":
    main()
