"""Generate a tiny synthetic DALES fixture in the layout of the LAS release.

The release extracts to `dales_las/{train,test}/<tile>.las`: one LAS 1.2 file per 500 x 500 m tile, in UTM zone 10N
coordinates, with the class id in the classification field. No tile of the release is read: every point is drawn from
a seeded generator, each class in its own height range above the ground.

Usage:
    uv run --no-sync python scripts/generate.py raw ./raw
"""

from argparse import ArgumentParser, Namespace
from pathlib import Path

import laspy
import numpy as np

TILES = {"train": ("5080_54435", "5190_54400"), "test": ("5080_54470", "5100_54440")}
TILE_SIZE = 500.0
# Class ids of the release and the height range above the ground of each.
HEIGHT_RANGE = {
    0: (0.0, 30.0),
    1: (0.0, 0.1),
    2: (0.5, 20.0),
    3: (0.0, 1.6),
    4: (0.0, 3.5),
    5: (8.0, 20.0),
    6: (0.0, 1.8),
    7: (0.0, 10.0),
    8: (3.0, 15.0),
}


def main() -> None:
    args = parse_args()
    if args.command == "raw":
        generate_raw(args)


def parse_args() -> Namespace:
    parser = ArgumentParser(description="Generate synthetic DALES test data.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    raw_parser = subparsers.add_parser("raw", help="Generate raw test data")
    raw_parser.add_argument("dst_dir", type=str, help="Path to output raw directory.")
    raw_parser.add_argument("--num-points", type=int, default=1024, help="Points per tile.")
    raw_parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser.parse_args()


def generate_tile(rng: np.random.Generator, num_points: int, name: str) -> laspy.LasData:
    # The tile name is its south-west corner in hectometres: easting 508.0 km, northing 5443.5 km.
    easting, northing = (100.0 * float(part) for part in name.split("_"))
    header = laspy.LasHeader(point_format=0, version="1.2")
    header.offsets = np.array([easting, northing, 0.0])
    header.scales = np.array([0.01, 0.01, 0.01])
    las = laspy.LasData(header)

    codes = np.array(list(HEIGHT_RANGE))
    code = rng.choice(codes, size=num_points)
    low, high = np.array([HEIGHT_RANGE[int(c)] for c in code]).T
    las.x = easting + rng.uniform(0.0, TILE_SIZE, size=num_points)
    las.y = northing + rng.uniform(0.0, TILE_SIZE, size=num_points)
    las.z = 60.0 + rng.uniform(low, high)
    las.classification = code.astype(np.uint8)
    number_of_returns = rng.integers(1, 5, size=num_points)
    las.number_of_returns = number_of_returns.astype(np.uint8)
    las.return_number = rng.integers(1, number_of_returns + 1).astype(np.uint8)
    return las


def generate_raw(args: Namespace) -> None:
    dst_root = Path(args.dst_dir)
    rng = np.random.default_rng(args.seed)

    for split, names in TILES.items():
        split_dir = dst_root / "dales_las" / split
        split_dir.mkdir(parents=True, exist_ok=True)
        for name in names:
            dst = split_dir / f"{name}.las"
            generate_tile(rng, args.num_points, name).write(dst.as_posix())
            print(f"  {dst}  (points={args.num_points})")


if __name__ == "__main__":
    main()
