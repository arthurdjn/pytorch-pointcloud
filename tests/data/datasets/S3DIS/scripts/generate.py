"""Generate a tiny synthetic S3DIS fixture in the layout of the original release.

Each room is a box of walls holding one axis-aligned box per annotation, so every class of `S3DIS_CLASSES` appears in
every room. The output respects the original directory layout (Area_X/RoomName/Annotations/{class}_N.txt + room-level
concatenated file + alignmentAngle.txt) so the standard S3DIS loader can read it unchanged. No file of the original
release is read: the points are drawn from a seeded generator.

Usage:
    uv run --no-sync python scripts/generate.py raw ./raw
    uv run --no-sync python scripts/generate.py process ./raw
"""

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
    parser = ArgumentParser(description="Generate synthetic S3DIS test data.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    raw_parser = subparsers.add_parser("raw", help="Generate raw test data")
    raw_parser.add_argument("dst_dir", type=str, help="Path to output raw data")
    raw_parser.add_argument(
        "--points-per-annotation",
        type=int,
        default=64,
        help="Number of points drawn per annotation file.",
    )
    raw_parser.add_argument("--seed", type=int, default=42, help="Random seed.")

    process_parser = subparsers.add_parser("process", help="Process raw data into final format")
    process_parser.add_argument("raw_dir", type=str, help="Path to raw data directory")

    return parser.parse_args()


def generate_raw(args: Namespace) -> None:
    dst_data_dir = Path(args.dst_dir)
    rng = np.random.default_rng(args.seed)
    num_points = args.points_per_annotation

    rooms = {
        "Area_1": ("WC_1", "conferenceRoom_1"),
        "Area_2": ("WC_1", "WC_2"),
        "Area_3": ("WC_1", "WC_2"),
        "Area_4": ("WC_1", "WC_2"),
        "Area_5": ("WC_1", "WC_2"),
        "Area_6": ("conferenceRoom_1", "copyRoom_1"),
    }
    # Annotation name, box min and box max, as fractions of the room extent. `stairs` is not an S3DIS class:
    # the original release contains such annotations and the loader folds them into `clutter`.
    boxes = [
        ("ceiling_1", (0.0, 0.0, 1.0), (1.0, 1.0, 1.0)),
        ("floor_1", (0.0, 0.0, 0.0), (1.0, 1.0, 0.0)),
        ("wall_1", (0.0, 0.0, 0.0), (1.0, 0.0, 1.0)),
        ("wall_2", (0.0, 1.0, 0.0), (1.0, 1.0, 1.0)),
        ("wall_3", (0.0, 0.0, 0.0), (0.0, 1.0, 1.0)),
        ("wall_4", (1.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
        ("beam_1", (0.0, 0.45, 0.9), (1.0, 0.55, 1.0)),
        ("column_1", (0.0, 0.0, 0.0), (0.1, 0.1, 1.0)),
        ("window_1", (0.3, 0.0, 0.4), (0.6, 0.0, 0.8)),
        ("door_1", (1.0, 0.4, 0.0), (1.0, 0.6, 0.7)),
        ("table_1", (0.4, 0.4, 0.0), (0.6, 0.6, 0.25)),
        ("chair_1", (0.3, 0.45, 0.0), (0.38, 0.55, 0.3)),
        ("chair_2", (0.62, 0.45, 0.0), (0.7, 0.55, 0.3)),
        ("sofa_1", (0.1, 0.8, 0.0), (0.5, 0.95, 0.3)),
        ("bookcase_1", (0.7, 0.9, 0.0), (0.95, 1.0, 0.7)),
        ("board_1", (0.0, 0.3, 0.4), (0.0, 0.7, 0.7)),
        ("clutter_1", (0.8, 0.1, 0.0), (0.9, 0.2, 0.1)),
        ("clutter_2", (0.45, 0.45, 0.25), (0.55, 0.55, 0.3)),
        ("stairs_1", (0.75, 0.3, 0.0), (0.95, 0.5, 0.2)),
    ]

    for area, room_names in rooms.items():
        area_dir = dst_data_dir / area
        area_dir.mkdir(parents=True, exist_ok=True)

        lines = [
            f"## Global alignment angle per disjoint space in {area} ##\n",
            "## Disjoint Space Name Global Alignment Angle ##\n",
        ]
        lines += [f"{room_name} {rng.choice([0, 90, 180, 270])}\n" for room_name in room_names]
        with open(area_dir / f"{area}_alignmentAngle.txt", "w") as f:
            f.writelines(lines)

        for room_name in room_names:
            annotations_dir = area_dir / room_name / "Annotations"
            annotations_dir.mkdir(parents=True, exist_ok=True)
            origin = rng.uniform(-30.0, 30.0, size=3)
            extent = rng.uniform((3.0, 3.0, 2.5), (8.0, 8.0, 3.5))

            room_data = []
            for name, box_min, box_max in boxes:
                pos = origin + extent * rng.uniform(box_min, box_max, size=(num_points, 3))
                color = np.clip(rng.integers(0, 256, size=3) + rng.integers(-10, 11, size=(num_points, 3)), 0, 255)
                data = np.concatenate([pos, color], axis=1)
                room_data.append(data)
                np.savetxt(annotations_dir / f"{name}.txt", data, fmt="%.3f")

            np.savetxt(area_dir / room_name / f"{room_name}.txt", np.concatenate(room_data), fmt="%.3f")
        print(f"  {area}: wrote {len(room_names)} rooms with {num_points} pts/annotation")


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
