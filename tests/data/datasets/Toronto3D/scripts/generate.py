"""Generate a tiny synthetic Toronto-3D fixture in the layout of the original release.

Each tile holds `--num-points` vertices in the CloudCompare-export schema
(`x, y, z, red, green, blue, scalar_Intensity, scalar_GPSTime, scalar_ScanAngleRank, scalar_Label`).
No scan of the original release is read: every tile is a street in raw UTM coordinates drawn from a
seeded generator, where each of the 9 classes sits in its own height range above the road.

Usage:
    uv run --no-sync python scripts/generate.py raw ./raw
"""

from argparse import ArgumentParser, Namespace
from pathlib import Path

import numpy as np
import plyfile

from torch_pointcloud.datasets.toronto3d import TORONTO3D_UTM_OFFSET

DEFAULT_FILES = ("L001.ply", "L002.ply", "L003.ply", "L004.ply")


def main() -> None:
    args = parse_args()
    if args.command == "raw":
        generate_raw(args)


def parse_args() -> Namespace:
    parser = ArgumentParser(description="Generate synthetic Toronto-3D test data.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    raw_parser = subparsers.add_parser("raw", help="Generate raw test data")
    raw_parser.add_argument("dst_dir", type=str, help="Path to output raw directory.")
    raw_parser.add_argument("--num-points", type=int, default=1024, help="Vertices per PLY.")
    raw_parser.add_argument(
        "--files",
        nargs="+",
        default=list(DEFAULT_FILES),
        help="PLY file names to write.",
    )
    raw_parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser.parse_args()


def generate_raw(args: Namespace) -> None:
    dst_root = Path(args.dst_dir)
    rng = np.random.default_rng(args.seed)
    dst_root.mkdir(parents=True, exist_ok=True)

    # Lowest and highest height above the road per class, in `TORONTO3D_CLASSES` order.
    height_range = np.array(
        [[0.0, 3.0], [0.0, 0.05], [0.0, 0.05], [0.5, 8.0], [0.0, 20.0], [7.0, 9.0], [0.0, 8.0], [0.0, 1.6], [0.0, 2.0]]
    )
    dtype = [
        ("x", "f8"),
        ("y", "f8"),
        ("z", "f8"),
        ("red", "u1"),
        ("green", "u1"),
        ("blue", "u1"),
        ("scalar_Intensity", "f4"),
        ("scalar_GPSTime", "f4"),
        ("scalar_ScanAngleRank", "f4"),
        ("scalar_Label", "f4"),
    ]

    for k, fname in enumerate(args.files):
        vertex = np.empty(args.num_points, dtype=dtype)
        label = rng.integers(0, len(height_range), size=args.num_points)
        # Consecutive 250 m tiles of one street, 60 m wide, 140 m above sea level.
        vertex["x"] = TORONTO3D_UTM_OFFSET[0] + rng.uniform(0.0, 60.0, size=args.num_points)
        vertex["y"] = TORONTO3D_UTM_OFFSET[1] + 250.0 * k + rng.uniform(0.0, 250.0, size=args.num_points)
        vertex["z"] = 140.0 + rng.uniform(height_range[label, 0], height_range[label, 1])
        for channel in ("red", "green", "blue"):
            vertex[channel] = rng.integers(0, 256, size=args.num_points)
        vertex["scalar_Intensity"] = rng.integers(0, 256, size=args.num_points)
        vertex["scalar_GPSTime"] = np.sort(rng.uniform(324000.0, 325000.0, size=args.num_points))
        vertex["scalar_ScanAngleRank"] = rng.integers(-30, 31, size=args.num_points)
        vertex["scalar_Label"] = label

        dst = dst_root / fname
        plyfile.PlyData([plyfile.PlyElement.describe(vertex, "vertex")], text=False).write(dst.as_posix())
        print(f"  {dst}  (vertices={args.num_points})")


if __name__ == "__main__":
    main()
