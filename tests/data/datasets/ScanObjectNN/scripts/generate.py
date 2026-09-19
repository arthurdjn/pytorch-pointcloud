"""Generate a tiny synthetic ScanObjectNN fixture in the layout of the original release.

For each (partition, background, split, variant) combination, we write an `.h5` archive of `--num-objects`
objects with `--num-points` points each. No archive of the original release is read: every object is a noisy
ellipsoid drawn from a seeded generator, standing on a floor patch in the archives that keep the background.

Usage:
    uv run --no-sync python scripts/generate.py raw ./raw
    uv run --no-sync python scripts/generate.py process ./raw
"""

from argparse import ArgumentParser, Namespace
from pathlib import Path

import h5py
import numpy as np

from torch_pointcloud.datasets import ScanObjectNN
from torch_pointcloud.datasets.scanobjectnn import (
    SCANOBJECTNN_CLASSES,
    SCANOBJECTNN_PARTITIONS,
    SCANOBJECTNN_VARIANTS,
)

NUM_CLASSES = len(SCANOBJECTNN_CLASSES)

SPLIT_DIRS: list[str] = []
for _partition in SCANOBJECTNN_PARTITIONS:
    for _bg in (True, False):
        name = _partition
        if _partition == "main":
            name += "_split"
        if not _bg:
            name += "_nobg"
        SPLIT_DIRS.append(name)

FILE_STEMS: list[str] = []
for _split in ("train", "test"):
    prefix = "training" if _split == "train" else "test"
    FILE_STEMS.append(f"{prefix}_objectdataset")
    for _variant in SCANOBJECTNN_VARIANTS:
        FILE_STEMS.append(f"{prefix}_objectdataset_{_variant}")


def main() -> None:
    args = parse_args()

    if args.command == "raw":
        generate_raw(args)
    elif args.command == "process":
        generate_processed(args)


def parse_args() -> Namespace:
    parser = ArgumentParser(description="Generate synthetic ScanObjectNN test data.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    raw_parser = subparsers.add_parser("raw", help="Generate raw test data")
    raw_parser.add_argument("dst_dir", type=str, help="Path to output raw data directory")
    raw_parser.add_argument("--num-points", type=int, default=1024, help="Points per object.")
    raw_parser.add_argument(
        "--num-objects",
        type=int,
        default=2,
        help="Objects per file (except the full-coverage label-test file, which always has 15).",
    )
    raw_parser.add_argument("--seed", type=int, default=42, help="Random seed.")

    process_parser = subparsers.add_parser("process", help="Process raw data into final format")
    process_parser.add_argument("raw_dir", type=str, help="Path to raw data directory")

    return parser.parse_args()


# Files that must contain one example per class so per-class unit tests pass.
# All others only need a couple of examples (enough for batch_size=2 forwards).
_FULL_COVERAGE_FILES: set[str] = {
    "main_split_nobg/test_objectdataset.h5",
}


def generate_raw(args: Namespace) -> None:
    rng = np.random.default_rng(args.seed)

    for split_dir in SPLIT_DIRS:
        out_dir = Path(args.dst_dir, split_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        for stem in FILE_STEMS:
            if f"{split_dir}/{stem}.h5" in _FULL_COVERAGE_FILES:
                labels = np.arange(NUM_CLASSES, dtype=np.int64)
            else:
                labels = rng.integers(0, NUM_CLASSES, size=args.num_objects)

            direction = rng.normal(size=(len(labels), args.num_points, 3))
            direction /= np.linalg.norm(direction, axis=2, keepdims=True)
            pos = rng.uniform(0.2, 0.8, size=(len(labels), 1, 3)) * direction
            pos += rng.normal(0.0, 0.01, size=pos.shape)
            if not split_dir.endswith("_nobg"):
                # A quarter of the points lie on the floor the object stands on.
                num_floor = args.num_points // 4
                pos[:, :num_floor, :2] = rng.uniform(-1.0, 1.0, size=(len(labels), num_floor, 2))
                pos[:, :num_floor, 2] = pos[:, :, 2].min(axis=1, keepdims=True)

            dst_h5 = out_dir / f"{stem}.h5"
            with h5py.File(dst_h5, "w") as f:
                f.create_dataset("data", data=pos.astype(np.float32))
                f.create_dataset("label", data=labels)
            print(f"  {dst_h5}  (objects={len(labels)}, points={args.num_points})")

    print("Done!")


def generate_processed(args: Namespace) -> None:
    raw_dir = Path(args.raw_dir)
    root = raw_dir.resolve().parent.parent.as_posix()

    for partition in SCANOBJECTNN_PARTITIONS:
        for variant in list(SCANOBJECTNN_VARIANTS) + [None]:
            for background in (True, False):
                for train in (True, False):
                    _ = ScanObjectNN(
                        root=root,
                        train=train,
                        partition=partition,
                        variant=variant,
                        background=background,
                        force_process=True,
                    )


if __name__ == "__main__":
    main()
