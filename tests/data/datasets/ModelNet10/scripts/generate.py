"""Generate a tiny synthetic ModelNet (10 or 40) fixture in the layout of the original release.

For each class, the script writes `--max-objects` train and `--max-objects` test `.off` meshes. No mesh
of the original release is read: every object is a closed triangle mesh (a sphere stretched along its axes
and rippled along its height) drawn from a seeded generator, so vertex normals are defined everywhere.

Usage:
    uv run --no-sync python scripts/generate.py raw ./raw --variant 40
    uv run --no-sync python scripts/generate.py process ./raw --variant 40
"""

from argparse import ArgumentParser, Namespace
from pathlib import Path

import numpy as np

from torch_pointcloud.datasets import ModelNet10, ModelNet40
from torch_pointcloud.datasets.modelnet import MODELNET10_CLASSES, MODELNET40_CLASSES


def main() -> None:
    args = parse_args()
    if args.command == "raw":
        generate_raw(args)
    elif args.command == "process":
        generate_processed(args)


def parse_args() -> Namespace:
    parser = ArgumentParser(description="Generate synthetic ModelNet test data.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    raw_parser = subparsers.add_parser("raw", help="Generate raw test data")
    raw_parser.add_argument("dst_dir", type=str, help="Path to output raw data")
    raw_parser.add_argument(
        "--variant",
        type=str,
        choices=["10", "40"],
        default="40",
        help="Which ModelNet variant to generate (10 or 40).",
    )
    raw_parser.add_argument("--max-points", type=int, default=256, help="Vertices per mesh.")
    raw_parser.add_argument("--max-objects", type=int, default=2, help="Files per class per split.")
    raw_parser.add_argument("--seed", type=int, default=42, help="Random seed.")

    process_parser = subparsers.add_parser("process", help="Process raw data into final format")
    process_parser.add_argument("raw_dir", type=str, help="Path to raw data directory")
    process_parser.add_argument("--variant", type=str, choices=["10", "40"], default="40")

    return parser.parse_args()


def generate_raw(args: Namespace) -> None:
    dst_root = Path(args.dst_dir)
    rng = np.random.default_rng(args.seed)
    classes = MODELNET10_CLASSES if args.variant == "10" else MODELNET40_CLASSES

    # A latitude / longitude grid over the unit sphere, closed along the longitude.
    resolution = int(np.sqrt(args.max_points))
    latitude, longitude = np.meshgrid(
        np.linspace(0.05, np.pi - 0.05, resolution),
        np.linspace(0.0, 2.0 * np.pi, resolution, endpoint=False),
        indexing="ij",
    )
    i, j = np.meshgrid(np.arange(resolution - 1), np.arange(resolution), indexing="ij")
    corner, right = (i * resolution + j).ravel(), (i * resolution + (j + 1) % resolution).ravel()
    face = np.concatenate(
        [
            np.stack([corner, corner + resolution, right], axis=1),
            np.stack([right, corner + resolution, right + resolution], axis=1),
        ]
    )

    for cls in classes:
        for k in range(2 * args.max_objects):
            split = "train" if k < args.max_objects else "test"
            ripple = 1.0 + rng.uniform(0.0, 0.3) * np.sin(rng.integers(1, 5) * latitude)
            radius = rng.uniform(10.0, 100.0, size=3)
            pos = np.stack(
                [
                    radius[0] * ripple * np.sin(latitude) * np.cos(longitude),
                    radius[1] * ripple * np.sin(latitude) * np.sin(longitude),
                    radius[2] * np.cos(latitude),
                ],
                axis=-1,
            ).reshape(-1, 3)

            dst_path = dst_root / cls / split / f"{cls}_{k + 1:04d}.off"
            dst_path.parent.mkdir(parents=True, exist_ok=True)
            with open(dst_path, "w") as f:
                f.write("OFF\n")
                f.write(f"{pos.shape[0]} {face.shape[0]} 0\n")
                for v in pos:
                    f.write(f"{v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
                for tri in face:
                    f.write(f"3 {int(tri[0])} {int(tri[1])} {int(tri[2])}\n")
        print(f"  {cls}: wrote {args.max_objects} files/split, {pos.shape[0]} vertices each")


def generate_processed(args: Namespace) -> None:
    raw_dir = Path(args.raw_dir)
    root = raw_dir.resolve().parent.parent.as_posix()

    cls = ModelNet10 if args.variant == "10" else ModelNet40
    for train in (True, False):
        dataset = cls(root=root, train=train, show_progress=True, force_process=True)
    # The committed cache stays without metadata: the tests read it as a cache from before metadata existed.
    for meta_path in Path(dataset.processed_dir).glob("*.meta.json"):
        meta_path.unlink()


if __name__ == "__main__":
    main()
