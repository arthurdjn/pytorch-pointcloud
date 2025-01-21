import json
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import List

import numpy as np

from torch_pointcloud.datasets import ShapeNetPart


def main() -> None:
    args = parse_args()

    if args.command == "raw":
        generate_raw(args)
    elif args.command == "process":
        generate_processed(args)


def parse_args() -> Namespace:
    parser = ArgumentParser(description="Generate ShapeNetPart data for testing")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Raw data generation command
    raw_parser = subparsers.add_parser("raw", help="Generate raw test data")
    raw_parser.add_argument("src_dir", type=str, help="Path to source raw data")
    raw_parser.add_argument("dst_dir", type=str, help="Path to output raw data")
    raw_parser.add_argument("--max-points", type=int, default=10, help="Maximum number of points per object")
    raw_parser.add_argument("--max-objects", type=int, default=4, help="Maximum number of objects per category")

    # Process command
    process_parser = subparsers.add_parser("process", help="Process raw data into final format")
    process_parser.add_argument("raw_dir", type=str, help="Path to raw data directory")

    return parser.parse_args()


def generate_raw(args: Namespace) -> None:
    src_data_dir = Path(args.src_dir)
    dst_data_dir = Path(args.dst_dir)
    max_points = args.max_points
    max_objects = args.max_objects

    # Contains IDs of the generated data
    filtered_ids: List[str] = []

    for file_path in Path(src_data_dir).iterdir():
        if not file_path.is_dir():
            continue

        txt_paths = sorted(file_path.rglob("**/*.txt"))
        for txt_path in txt_paths[:max_objects]:
            data = np.loadtxt(txt_path, delimiter=" ")
            data = data[:max_points, :]
            out_path = Path(dst_data_dir) / txt_path.relative_to(src_data_dir)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            np.savetxt(out_path, data, delimiter=" ")

            # Store the data ID in the same format as in ShapeNetPart
            data_id = f"shape_data/{txt_path.parent.name}/{txt_path.stem}"
            filtered_ids.append(data_id)

    # Generate associated train_test_split.json file
    generate_split(src_data_dir, dst_data_dir, filtered_ids, "train")
    generate_split(src_data_dir, dst_data_dir, filtered_ids, "val")
    generate_split(src_data_dir, dst_data_dir, filtered_ids, "test")


def generate_split(src_data_dir: Path, dst_data_dir: Path, filtered_ids: List[str], split: str) -> None:
    with open(f"{src_data_dir}/train_test_split/shuffled_{split}_file_list.json", "r") as f:
        file_list = json.load(f)

    filtered_data_files = []
    for file_name in file_list:
        if file_name in filtered_ids:
            filtered_data_files.append(file_name)

    out_path = Path(dst_data_dir, "train_test_split", f"shuffled_{split}_file_list.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(filtered_data_files, f)


def generate_processed(args: Namespace) -> None:
    raw_dir = Path(args.raw_dir)
    root = raw_dir.resolve().parent.parent.as_posix()

    for split in ["train", "val", "test"]:
        _ = ShapeNetPart(root=root, split=split, progress=True, process=True)


if __name__ == "__main__":
    main()
