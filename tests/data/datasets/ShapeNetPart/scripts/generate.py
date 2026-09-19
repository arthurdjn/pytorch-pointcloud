"""Generate a tiny synthetic ShapeNetPart fixture in the layout of the original release.

For each ShapeNetPart category, we write `--max-objects` per-object `.txt` files of `--max-points` rows
(position / normal / segment columns) and the shuffled split file lists that reference them. No file of the
original release is read: every object is a stack of ellipsoids drawn from a seeded generator, one per part
label of its category, with exact surface normals.

Usage:
    uv run --no-sync python scripts/generate.py raw ./raw
    uv run --no-sync python scripts/generate.py process ./raw
"""

import json
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict, List

import numpy as np

from torch_pointcloud.datasets import ShapeNetPart


def main() -> None:
    args = parse_args()

    if args.command == "raw":
        generate_raw(args)
    elif args.command == "process":
        generate_processed(args)


def parse_args() -> Namespace:
    parser = ArgumentParser(description="Generate synthetic ShapeNetPart test data.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    raw_parser = subparsers.add_parser("raw", help="Generate raw test data")
    raw_parser.add_argument("dst_dir", type=str, help="Path to output raw data")
    raw_parser.add_argument("--max-points", type=int, default=1024, help="Maximum number of points per object.")
    raw_parser.add_argument("--max-objects", type=int, default=4, help="Maximum number of objects per category.")
    raw_parser.add_argument("--seed", type=int, default=42, help="Random seed.")

    process_parser = subparsers.add_parser("process", help="Process raw data into final format")
    process_parser.add_argument("raw_dir", type=str, help="Path to raw data directory")

    return parser.parse_args()


def generate_raw(args: Namespace) -> None:
    dst_data_dir = Path(args.dst_dir)
    rng = np.random.default_rng(args.seed)

    # The first half of a category's objects goes to train, the rest alternates between val and test.
    splits: Dict[str, List[str]] = {"train": [], "val": [], "test": []}
    for category, category_id in ShapeNetPart.category_ids.items():
        parts = ShapeNetPart.seg_ids[category]
        for k in range(args.max_objects):
            # One ellipsoid per part, stacked along the up axis inside the unit sphere.
            segment = np.repeat(parts, -(-args.max_points // len(parts)))[: args.max_points]
            level = (segment - parts[0] + 0.5) / len(parts) - 0.5
            direction = rng.normal(size=(args.max_points, 3))
            direction /= np.linalg.norm(direction, axis=1, keepdims=True)
            radius = rng.uniform(0.1, 0.4, size=(len(parts), 3))[segment - parts[0]]
            radius[:, 1] = 0.5 / len(parts)
            pos = radius * direction
            pos[:, 1] += level
            normal = direction / radius
            normal /= np.linalg.norm(normal, axis=1, keepdims=True)

            data_id = rng.bytes(16).hex()
            out_path = dst_data_dir / category_id / f"{data_id}.txt"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            np.savetxt(out_path, np.concatenate([pos, normal, segment[:, None]], axis=1), delimiter=" ", fmt="%.6f")

            split = "train" if k < args.max_objects // 2 else ("val", "test")[k % 2]
            splits[split].append(f"shape_data/{category_id}/{data_id}")

    for split, data_ids in splits.items():
        out_path = dst_data_dir / "train_test_split" / f"shuffled_{split}_file_list.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump([data_ids[i] for i in rng.permutation(len(data_ids))], f)


def generate_processed(args: Namespace) -> None:
    raw_dir = Path(args.raw_dir)
    root = raw_dir.resolve().parent.parent.as_posix()

    for split in ("train", "val", "test"):
        _ = ShapeNetPart(root=root, split=split, show_progress=True, force_process=True)


if __name__ == "__main__":
    main()
