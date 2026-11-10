"""Generate a tiny synthetic FRACTAL fixture in the layout of the release.

The test split ships as 10 zip archives `test-NN.zip`, each holding `test/NN/<patch>.las`: one 50 x 50 m patch of
`--num-points` points in LAS 1.4 point format 8 (RGB + near-infrared), in Lambert-93 coordinates. No patch of the
release is read: every point is drawn from a seeded generator, each LiDAR HD class in its own height range. The
patches are uncompressed LAS so that reading them needs `laspy` alone.

Usage:
    uv run --no-sync python scripts/generate.py raw ./raw
"""

import io
import zipfile
from argparse import ArgumentParser, Namespace
from pathlib import Path

import laspy
import numpy as np

NUM_ARCHIVES = 10
# LiDAR HD classification codes and the height range above the ground of each.
HEIGHT_RANGE = {
    1: (0.0, 5.0),
    2: (0.0, 0.1),
    3: (0.0, 0.5),
    4: (0.5, 1.5),
    5: (1.5, 15.0),
    6: (3.0, 12.0),
    9: (0.0, 0.05),
    17: (5.0, 6.0),
    64: (0.0, 4.0),
    65: (-20.0, 60.0),
    66: (0.0, 0.05),
}


def main() -> None:
    args = parse_args()
    if args.command == "raw":
        generate_raw(args)


def parse_args() -> Namespace:
    parser = ArgumentParser(description="Generate synthetic FRACTAL test data.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    raw_parser = subparsers.add_parser("raw", help="Generate raw test data")
    raw_parser.add_argument("dst_dir", type=str, help="Path to output raw directory.")
    raw_parser.add_argument("--num-points", type=int, default=1024, help="Points per patch.")
    raw_parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser.parse_args()


def generate_patch(rng: np.random.Generator, num_points: int, x0: float, y0: float) -> bytes:
    header = laspy.LasHeader(point_format=8, version="1.4")
    header.offsets = np.array([0.0, 5000000.0, 0.0])
    header.scales = np.array([0.001, 0.001, 0.001])
    las = laspy.LasData(header)

    codes = np.array(list(HEIGHT_RANGE))
    code = rng.choice(codes, size=num_points)
    low, high = np.array([HEIGHT_RANGE[int(c)] for c in code]).T
    las.x = x0 + rng.uniform(0.0, 50.0, size=num_points)
    las.y = y0 + rng.uniform(0.0, 50.0, size=num_points)
    las.z = 120.0 + rng.uniform(low, high)
    las.classification = code.astype(np.uint8)
    las.intensity = rng.integers(0, 4000, size=num_points).astype(np.uint16)
    number_of_returns = rng.integers(1, 4, size=num_points)
    las.number_of_returns = number_of_returns.astype(np.uint8)
    las.return_number = rng.integers(1, number_of_returns + 1).astype(np.uint8)
    for channel in ("red", "green", "blue", "nir"):
        las[channel] = (rng.integers(0, 256, size=num_points) * 256).astype(np.uint16)

    buffer = io.BytesIO()
    las.write(buffer)
    return buffer.getvalue()


def generate_raw(args: Namespace) -> None:
    dst_root = Path(args.dst_dir)
    rng = np.random.default_rng(args.seed)
    dst_root.mkdir(parents=True, exist_ok=True)

    for i in range(NUM_ARCHIVES):
        # One patch per archive, on a 50 m grid of southern France.
        x0, y0 = 743050.0 + 50.0 * i, 6246650.0
        name = f"test/{i:02d}/TEST-{int(x0) // 1000:04d}_{int(y0) // 1000 + 1:04d}-{i:09d}.las"
        dst = dst_root / f"test-{i:02d}.zip"
        with zipfile.ZipFile(dst, "w", compression=zipfile.ZIP_STORED) as archive:
            # A fixed timestamp keeps the archives byte-identical across runs.
            member = zipfile.ZipInfo(name, date_time=(2024, 5, 2, 0, 0, 0))
            archive.writestr(member, generate_patch(rng, args.num_points, x0, y0))
        print(f"  {dst}  ({name}, points={args.num_points})")


if __name__ == "__main__":
    main()
