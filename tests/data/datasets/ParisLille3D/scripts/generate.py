"""Generate a tiny synthetic Paris-Lille-3D fixture in the layout of the original release.

For each PLY file of the 10-class benchmark (`Lille1_1`, `Lille1_2`, `Lille2`, `Paris`), we write
`--num-points` vertices in the original schema (`x`, `y`, `z`, `reflectance`, `class`). No scan of the
original release is read: every tile is a street drawn from a seeded generator, where each of the 10
classes sits in its own height range above the road.

Usage:
    uv run --no-sync python scripts/generate.py raw ./raw
"""

from argparse import ArgumentParser, Namespace
from pathlib import Path

import numpy as np
import plyfile

DEFAULT_FILES = ("Lille1_1.ply", "Lille1_2.ply", "Lille2.ply", "Paris.ply")


def main() -> None:
    args = parse_args()
    if args.command == "raw":
        generate_raw(args)


def parse_args() -> Namespace:
    parser = ArgumentParser(description="Generate synthetic Paris-Lille-3D test data.")
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

    # Lowest and highest height above the road per class, in `PARISLILLE3D_CLASSES` order.
    height_range = np.array(
        [
            [0.0, 3.0],
            [0.0, 0.05],
            [0.0, 20.0],
            [0.0, 6.0],
            [0.0, 1.0],
            [0.0, 1.2],
            [0.0, 1.0],
            [0.0, 1.8],
            [0.0, 1.6],
            [0.5, 8.0],
        ]
    )
    dtype = [("x", "f4"), ("y", "f4"), ("z", "f4"), ("reflectance", "u1"), ("class", "i4")]

    for k, fname in enumerate(args.files):
        vertex = np.empty(args.num_points, dtype=dtype)
        label = rng.integers(0, len(height_range), size=args.num_points)
        # Consecutive 300 m tiles of one street, 40 m wide, 30 m above sea level.
        vertex["x"] = rng.uniform(0.0, 40.0, size=args.num_points)
        vertex["y"] = 300.0 * k + rng.uniform(0.0, 300.0, size=args.num_points)
        vertex["z"] = 30.0 + rng.uniform(height_range[label, 0], height_range[label, 1])
        vertex["reflectance"] = rng.integers(0, 256, size=args.num_points)
        vertex["class"] = label

        dst = dst_root / fname
        plyfile.PlyData([plyfile.PlyElement.describe(vertex, "vertex")], text=False).write(dst.as_posix())
        print(f"  {dst}  (vertices={args.num_points})")


if __name__ == "__main__":
    main()
