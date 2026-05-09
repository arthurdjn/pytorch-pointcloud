"""Generate a tiny ShapeNetPart fixture by subsampling real per-object `.txt` files.

For each ShapeNetPart category, we keep the first `--max-objects` raw files and
subsample each to `--max-points` rows (preserving the position / normal / segment
columns). Splits are derived from the original shuffled file lists, restricted to
the kept ids.

The default source directory is `$TORCH_POINTCLOUD_DATA_DIR/ShapeNetPart/raw`.

Usage:
    uv run --no-sync python scripts/generate.py raw ./raw
    uv run --no-sync python scripts/generate.py process ./raw
"""

import json
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import List

import numpy as np

from torch_pointcloud.config import DATA_DIR
from torch_pointcloud.datasets import ShapeNetPart


def main() -> None:
    args = parse_args()

    if args.command == "raw":
        generate_raw(args)
    elif args.command == "process":
        generate_processed(args)


def parse_args() -> Namespace:
    parser = ArgumentParser(description="Generate ShapeNetPart test data by subsampling real .txt files.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    raw_parser = subparsers.add_parser("raw", help="Generate raw test data")
    raw_parser.add_argument("dst_dir", type=str, help="Path to output raw data")
    raw_parser.add_argument(
        "--src-dir",
        type=str,
        default=str(Path(DATA_DIR) / "ShapeNetPart" / "raw"),
        help="Source ShapeNetPart raw directory (default: $TORCH_POINTCLOUD_DATA_DIR/ShapeNetPart/raw).",
    )
    raw_parser.add_argument("--max-points", type=int, default=1024, help="Maximum number of points per object.")
    raw_parser.add_argument("--max-objects", type=int, default=4, help="Maximum number of objects per category.")
    raw_parser.add_argument("--seed", type=int, default=42, help="Random seed.")

    process_parser = subparsers.add_parser("process", help="Process raw data into final format")
    process_parser.add_argument("raw_dir", type=str, help="Path to raw data directory")

    return parser.parse_args()


def generate_raw(args: Namespace) -> None:
    src_data_dir = Path(args.src_dir)
    dst_data_dir = Path(args.dst_dir)
    if not src_data_dir.exists():
        raise FileNotFoundError(
            f"Source ShapeNetPart raw directory not found: {src_data_dir!r}. "
            f"Set --src-dir or TORCH_POINTCLOUD_DATA_DIR."
        )

    rng = np.random.default_rng(args.seed)

    filtered_ids: List[str] = []
    for category_dir in sorted(p for p in src_data_dir.iterdir() if p.is_dir() and p.name.isdigit()):
        txt_paths = sorted(category_dir.rglob("*.txt"))
        for txt_path in txt_paths[: args.max_objects]:
            data = np.loadtxt(txt_path, delimiter=" ")
            if data.size == 0:
                continue
            num_keep = min(args.max_points, data.shape[0])
            indices = rng.choice(data.shape[0], size=num_keep, replace=False)
            indices.sort()
            data = data[indices]

            out_path = dst_data_dir / txt_path.relative_to(src_data_dir)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            np.savetxt(out_path, data, delimiter=" ", fmt="%.6f")

            data_id = f"shape_data/{txt_path.parent.name}/{txt_path.stem}"
            filtered_ids.append(data_id)

    for split in ("train", "val", "test"):
        generate_split(src_data_dir, dst_data_dir, filtered_ids, split)


def generate_split(src_data_dir: Path, dst_data_dir: Path, filtered_ids: List[str], split: str) -> None:
    src_split = src_data_dir / "train_test_split" / f"shuffled_{split}_file_list.json"
    with open(src_split, "r") as f:
        file_list = json.load(f)

    filtered_data_files = [name for name in file_list if name in filtered_ids]

    out_path = dst_data_dir / "train_test_split" / f"shuffled_{split}_file_list.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(filtered_data_files, f)


def generate_processed(args: Namespace) -> None:
    raw_dir = Path(args.raw_dir)
    root = raw_dir.resolve().parent.parent.as_posix()

    for split in ("train", "val", "test"):
        _ = ShapeNetPart(root=root, split=split, show_progress=True, force_process=True)


if __name__ == "__main__":
    main()
