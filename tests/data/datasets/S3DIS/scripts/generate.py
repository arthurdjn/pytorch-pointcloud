"""Generate a tiny S3DIS fixture by subsampling real per-annotation `.txt` files.

For each Area, we keep `--max-rooms` rooms and subsample each annotation to
`--max-points-per-annotation` points. The output respects the original directory
layout (Area_X/RoomName/Annotations/{class}_N.txt + room-level concatenated file +
alignmentAngle.txt) so the standard S3DIS loader can read it unchanged.

The default source directory is `$TORCH_POINTCLOUD_DATA_DIR/S3DIS/raw`.

Usage:
    uv run --no-sync python scripts/generate.py raw ./raw
    uv run --no-sync python scripts/generate.py process ./raw
"""

from argparse import ArgumentParser, Namespace
from pathlib import Path

import numpy as np

from torch_pointcloud.config import DATA_DIR
from torch_pointcloud.datasets import S3DIS


def main() -> None:
    args = parse_args()

    if args.command == "raw":
        generate_raw(args)
    elif args.command == "process":
        generate_processed(args)


def parse_args() -> Namespace:
    parser = ArgumentParser(description="Generate S3DIS test data by subsampling real annotation files.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    raw_parser = subparsers.add_parser("raw", help="Generate raw test data")
    raw_parser.add_argument("dst_dir", type=str, help="Path to output raw data")
    raw_parser.add_argument(
        "--src-dir",
        type=str,
        default=str(Path(DATA_DIR) / "S3DIS" / "raw"),
        help="Source S3DIS raw directory (default: $TORCH_POINTCLOUD_DATA_DIR/S3DIS/raw).",
    )
    raw_parser.add_argument(
        "--max-points-per-annotation",
        type=int,
        default=64,
        help="Maximum number of points kept per annotation file.",
    )
    raw_parser.add_argument("--max-rooms", type=int, default=2, help="Maximum number of rooms per area.")
    raw_parser.add_argument("--seed", type=int, default=42, help="Random seed.")

    process_parser = subparsers.add_parser("process", help="Process raw data into final format")
    process_parser.add_argument("raw_dir", type=str, help="Path to raw data directory")

    return parser.parse_args()


def generate_raw(args: Namespace) -> None:
    src_data_dir = Path(args.src_dir)
    dst_data_dir = Path(args.dst_dir)
    if not src_data_dir.exists():
        raise FileNotFoundError(
            f"Source S3DIS raw directory not found: {src_data_dir!r}. Set --src-dir or TORCH_POINTCLOUD_DATA_DIR."
        )

    rng = np.random.default_rng(args.seed)
    max_points = args.max_points_per_annotation
    max_rooms = args.max_rooms

    areas = ["Area_1", "Area_2", "Area_3", "Area_4", "Area_5", "Area_6"]

    for area in areas:
        # Subsample alignment angles
        alignment_angles_path = src_data_dir / area / f"{area}_alignmentAngle.txt"
        with open(alignment_angles_path, "r") as f:
            lines = f.readlines()
        out_lines = [line for line in lines if line.startswith("#")]
        out_lines += [line for line in lines if not line.startswith("#")][:max_rooms]

        out_alignment_angles_path = dst_data_dir / alignment_angles_path.relative_to(src_data_dir)
        out_alignment_angles_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_alignment_angles_path, "w") as f:
            f.writelines(out_lines)

        # Subsample room data
        room_dirs = sorted(p for p in (src_data_dir / area).iterdir() if p.is_dir())
        for room_dir in room_dirs[:max_rooms]:
            room_data = []
            annotation_paths = sorted((room_dir / "Annotations").glob("*.txt"))
            for annotation_path in annotation_paths:
                data = np.loadtxt(annotation_path, dtype=np.float32)
                if data.size == 0:
                    continue
                num_keep = min(max_points, data.shape[0])
                indices = rng.choice(data.shape[0], size=num_keep, replace=False)
                indices.sort()
                data = data[indices]
                room_data.append(data)

                out_annotation_path = dst_data_dir / annotation_path.relative_to(src_data_dir)
                out_annotation_path.parent.mkdir(parents=True, exist_ok=True)
                np.savetxt(out_annotation_path, data, fmt="%.3f")

            out_room_path = dst_data_dir / room_dir.relative_to(src_data_dir) / f"{room_dir.name}.txt"
            np.savetxt(out_room_path, np.concatenate(room_data), fmt="%.3f")
        print(f"  {area}: kept {min(max_rooms, len(room_dirs))} rooms with up to {max_points} pts/annotation")


def generate_processed(args: Namespace) -> None:
    raw_dir = Path(args.raw_dir)
    root = raw_dir.resolve().parent.parent.as_posix()

    _ = S3DIS(
        root=root,
        areas=["Area_1", "Area_2", "Area_3", "Area_4", "Area_5", "Area_6"],
        show_progress=True,
        force_process=True,
    )


if __name__ == "__main__":
    main()
