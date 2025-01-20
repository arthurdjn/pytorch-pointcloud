import json
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import List

import numpy as np


def main() -> None:
    args = parse_args()

    src_data_dir = Path(args.src)
    dst_data_dir = Path(args.dst)
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

            # Store the data ID in the same format as in ShapeNet
            data_id = f"shape_data/{txt_path.parent.name}/{txt_path.stem}"
            filtered_ids.append(data_id)

    # Generate associated train_test_split.json file
    generate_split(src_data_dir, dst_data_dir, filtered_ids, "train")
    generate_split(src_data_dir, dst_data_dir, filtered_ids, "val")
    generate_split(src_data_dir, dst_data_dir, filtered_ids, "test")


def parse_args() -> Namespace:
    parser = ArgumentParser(description="Generate ShapeNet data for testing")
    parser.add_argument("src", type=str, help="Path to source raw data")
    parser.add_argument("dst", type=str, help="Path to output raw data")
    parser.add_argument("--max-points", type=int, default=10, help="Maximum number of points per object")
    parser.add_argument("--max-objects", type=int, default=4, help="Maximum number of objects per category")
    return parser.parse_args()


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


if __name__ == "__main__":
    main()
