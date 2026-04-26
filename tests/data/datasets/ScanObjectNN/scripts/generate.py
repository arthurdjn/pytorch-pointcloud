from argparse import ArgumentParser, Namespace
from pathlib import Path

import h5py
import numpy as np

from torch_pointcloud.datasets import ScanObjectNN
from torch_pointcloud.datasets.scanobjectnn import (
    SCANOBJECTNN_CLASSES,
    SCANOBJECTNN_SPLITS,
    SCANOBJECTNN_VARIANTS,
)

NUM_CLASSES = len(SCANOBJECTNN_CLASSES)

SPLIT_DIRS: list[str] = []
for _split in SCANOBJECTNN_SPLITS:
    for _bg in (True, False):
        name = _split
        if _split == "main":
            name += "_split"
        if not _bg:
            name += "_nobg"
        SPLIT_DIRS.append(name)

FILE_STEMS: list[str] = []
for _train in (True, False):
    prefix = "training" if _train else "test"
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
    parser = ArgumentParser(description="Generate ScanObjectNN test data")
    subparsers = parser.add_subparsers(dest="command", required=True)

    raw_parser = subparsers.add_parser("raw", help="Generate raw test data")
    raw_parser.add_argument("dst_dir", type=str, help="Path to output raw data directory")
    raw_parser.add_argument("--num-points", type=int, default=10, help="Number of points per object")
    raw_parser.add_argument("--num-objects-per-class", type=int, default=1, help="Number of objects per class")
    raw_parser.add_argument("--seed", type=int, default=42, help="Random seed")

    process_parser = subparsers.add_parser("process", help="Process raw data into final format")
    process_parser.add_argument("raw_dir", type=str, help="Path to raw data directory")

    return parser.parse_args()


def generate_raw(args: Namespace) -> None:
    np.random.seed(args.seed)

    n = NUM_CLASSES * args.num_objects_per_class
    labels = np.tile(np.arange(NUM_CLASSES, dtype=np.int64), args.num_objects_per_class)

    for split_dir in SPLIT_DIRS:
        out_dir = Path(args.dst_dir, split_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        for stem in FILE_STEMS:
            pos = np.random.uniform(-1, 1, (n, args.num_points, 3)).astype(np.float32)
            h5_path = out_dir / f"{stem}.h5"
            with h5py.File(h5_path, "w") as f:
                f.create_dataset("data", data=pos)
                f.create_dataset("label", data=labels)
            print(f"  {h5_path}")

    print("Done!")


def generate_processed(args: Namespace) -> None:
    raw_dir = Path(args.raw_dir)
    root = raw_dir.resolve().parent.parent.as_posix()

    for split in SCANOBJECTNN_SPLITS:
        for variant in list(SCANOBJECTNN_VARIANTS) + [None]:
            for background in (True, False):
                for train in (True, False):
                    _ = ScanObjectNN(
                        root=root,
                        split=split,
                        variant=variant,
                        background=background,
                        train=train,
                        force_process=True,
                    )


if __name__ == "__main__":
    main()
