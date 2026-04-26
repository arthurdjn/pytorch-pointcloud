from argparse import ArgumentParser, Namespace
from pathlib import Path

import numpy as np

from torch_pointcloud.datasets import S3DIS


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
    raw_parser.add_argument("--max-rooms", type=int, default=2, help="Maximum number of rooms per area")

    # Process command
    process_parser = subparsers.add_parser("process", help="Process raw data into final format")
    process_parser.add_argument("raw_dir", type=str, help="Path to raw data directory")

    return parser.parse_args()


def generate_raw(args: Namespace) -> None:
    np.random.seed(42)
    src_data_dir = Path(args.src_dir)
    dst_data_dir = Path(args.dst_dir)
    max_points = args.max_points
    max_rooms = args.max_rooms

    # Use all areas
    areas = ["Area_1", "Area_2", "Area_3", "Area_4", "Area_5", "Area_6"]

    for area in areas:
        # Subsample alignment angles
        alignment_angles_path = Path(src_data_dir, area, f"{area}_alignmentAngle.txt")
        with open(alignment_angles_path, "r") as f:
            lines = f.readlines()
        out_lines = [line for line in lines if line.startswith("#")]
        out_lines += [line for line in lines if not line.startswith("#")][:max_rooms]

        out_alignment_angles_path = Path(dst_data_dir) / alignment_angles_path.relative_to(src_data_dir)
        out_alignment_angles_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_alignment_angles_path, "w") as f:
            f.writelines(out_lines)

        # Subsample room data
        room_dirs = sorted([path for path in Path(src_data_dir, area).iterdir() if path.is_dir()])
        for room_dir in room_dirs[:max_rooms]:
            room_data = []
            annotation_paths = sorted(Path(room_dir, "Annotations").glob("*.txt"))
            for annotation_path in annotation_paths:
                data = np.loadtxt(annotation_path)
                np.random.shuffle(data)
                data = data[:max_points]
                room_data.append(data)
                out_annotation_path = Path(dst_data_dir) / annotation_path.relative_to(src_data_dir)
                out_annotation_path.parent.mkdir(parents=True, exist_ok=True)
                np.savetxt(out_annotation_path, data)

            # Concatenate room data just like in the original data
            out_data_path = Path(dst_data_dir) / room_dir.relative_to(src_data_dir) / f"{room_dir.name}.txt"
            np.savetxt(out_data_path, np.concatenate(room_data))


def generate_processed(args: Namespace) -> None:
    raw_dir = Path(args.raw_dir)
    root = raw_dir.resolve().parent.parent.as_posix()

    _ = S3DIS(
        root=root,
        areas=["Area_1", "Area_2", "Area_3", "Area_4", "Area_5", "Area_6"],
        progress=True,
        force_process=True,
    )


if __name__ == "__main__":
    main()
