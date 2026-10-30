"""Generate a tiny synthetic GridNet-HD fixture in the layout of the Hub release.

The release stores one LAS 1.2 file per zone under `<zone>/lidar/<zone>.las`, in EPSG:32632 coordinates, with the
annotated class id in an `int32` extra field `ground_truth` that the test zones do not carry, and lists the zones of
each split in `split.json`. No zone of the release is read: every point is drawn from a seeded generator, each
annotated class in its own height range above the ground.

Usage:
    uv run --no-sync python scripts/generate.py raw ./raw
"""

import json
from argparse import ArgumentParser, Namespace
from pathlib import Path

import laspy
import numpy as np

SPLIT = {"train": ["t1z6a"], "val": ["t1z8"], "test": ["t1z4"]}
# South-west corner of each zone in EPSG:32632 metres and its altitude.
CORNERS = {
    "t1z6a": (383_500.0, 5_100_000.0, 1_900.0),
    "t1z8": (382_568.0, 5_096_806.0, 2_020.0),
    "t1z4": (383_195.0, 5_101_080.0, 1_822.0),
}
ZONE_SIZE = 200.0
# Annotated class ids of the release and the height range above the ground of each.
HEIGHT_RANGE = {
    0: (20.0, 45.0),
    1: (15.0, 40.0),
    2: (10.0, 35.0),
    3: (0.0, 10.0),
    4: (0.0, 5.0),
    5: (25.0, 45.0),
    6: (30.0, 45.0),
    7: (28.0, 44.0),
    8: (24.0, 40.0),
    9: (22.0, 38.0),
    10: (20.0, 36.0),
    11: (18.0, 34.0),
    12: (0.0, 2.0),
    13: (0.0, 1.0),
    14: (3.0, 25.0),
    15: (0.5, 3.0),
    16: (0.0, 0.5),
    17: (0.0, 0.3),
    18: (0.0, 0.2),
    19: (0.0, 0.1),
    20: (0.0, 0.05),
    21: (0.0, 8.0),
    22: (0.0, 1.0),
    255: (0.0, 45.0),
}


def main() -> None:
    args = parse_args()
    if args.command == "raw":
        generate_raw(args)


def parse_args() -> Namespace:
    parser = ArgumentParser(description="Generate synthetic GridNet-HD test data.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    raw_parser = subparsers.add_parser("raw", help="Generate raw test data")
    raw_parser.add_argument("dst_dir", type=str, help="Path to output raw directory.")
    raw_parser.add_argument("--num-points", type=int, default=1024, help="Points per zone.")
    raw_parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser.parse_args()


def generate_zone(rng: np.random.Generator, num_points: int, name: str, labelled: bool) -> laspy.LasData:
    easting, northing, altitude = CORNERS[name]
    header = laspy.LasHeader(point_format=3, version="1.2")
    header.offsets = np.array([easting, northing, altitude])
    header.scales = np.array([0.0001, 0.0001, 0.0001])
    if labelled:
        header.add_extra_dim(laspy.ExtraBytesParams(name="ground_truth", type=np.int32))
    las = laspy.LasData(header)

    codes = np.array(list(HEIGHT_RANGE))
    code = rng.choice(codes, size=num_points)
    low, high = np.array([HEIGHT_RANGE[int(c)] for c in code]).T
    las.x = easting + rng.uniform(0.0, ZONE_SIZE, size=num_points)
    las.y = northing + rng.uniform(0.0, ZONE_SIZE, size=num_points)
    las.z = altitude + rng.uniform(low, high)
    las.intensity = rng.integers(0, 160, size=num_points).astype(np.uint16)
    number_of_returns = rng.integers(1, 4, size=num_points)
    las.number_of_returns = number_of_returns.astype(np.uint8)
    las.return_number = rng.integers(1, number_of_returns + 1).astype(np.uint8)
    las.red, las.green, las.blue = (rng.integers(0, 256, size=num_points).astype(np.uint16) << 8 for _ in range(3))
    if labelled:
        las["ground_truth"] = code.astype(np.int32)
    return las


def generate_raw(args: Namespace) -> None:
    dst_root = Path(args.dst_dir)
    dst_root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    with open(dst_root / "split.json", "w") as f:
        json.dump(SPLIT, f, indent=4)
    for split, names in SPLIT.items():
        for name in names:
            dst = dst_root / name / "lidar" / f"{name}.las"
            dst.parent.mkdir(parents=True, exist_ok=True)
            generate_zone(rng, args.num_points, name, labelled=split != "test").write(dst.as_posix())
            print(f"  {dst}  (points={args.num_points}, labelled={split != 'test'})")


if __name__ == "__main__":
    main()
