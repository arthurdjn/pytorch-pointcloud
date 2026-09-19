"""Generate a tiny synthetic ModelNetNormalResampled fixture in the layout of the original release.

The raw release ships pre-resampled point clouds as `.txt` files with `(x, y, z, nx, ny, nz)` per row. We
write `--max-objects` files per class per split, `--max-points` rows each, with the split lists, the class
name lists and the file list that reference them. No file of the original release is read: every object is
an ellipsoid inside the unit sphere drawn from a seeded generator, with its exact surface normals.

Usage:
    uv run --no-sync python scripts/generate.py raw ./raw
    uv run --no-sync python scripts/generate.py process ./raw
"""

from argparse import ArgumentParser, Namespace
from pathlib import Path

import numpy as np

from torch_pointcloud.datasets import ModelNetNormalResampled
from torch_pointcloud.datasets.modelnet import MODELNET10_CLASSES, MODELNET40_CLASSES

VARIANTS = ("10", "40")
SPLITS = ("train", "test")


def main() -> None:
    args = parse_args()
    if args.command == "raw":
        generate_raw(args)
    elif args.command == "process":
        generate_processed(args)


def parse_args() -> Namespace:
    parser = ArgumentParser(description="Generate synthetic ModelNetNormalResampled test data.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    raw_parser = subparsers.add_parser("raw", help="Generate raw test data")
    raw_parser.add_argument("dst_dir", type=str, help="Path to output raw data")
    raw_parser.add_argument("--max-points", type=int, default=1024, help="Points per object.")
    raw_parser.add_argument("--max-objects", type=int, default=2, help="Files per class per split.")
    raw_parser.add_argument("--seed", type=int, default=42, help="Random seed.")

    process_parser = subparsers.add_parser("process", help="Process raw data into final format")
    process_parser.add_argument("raw_dir", type=str, help="Path to raw data directory")

    return parser.parse_args()


def generate_raw(args: Namespace) -> None:
    dst_root = Path(args.dst_dir)
    dst_root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    for variant, classes in zip(VARIANTS, (MODELNET10_CLASSES, MODELNET40_CLASSES)):
        (dst_root / f"modelnet{variant}_shape_names.txt").write_text("\n".join(classes) + "\n")
        for s, split in enumerate(SPLITS):
            ids = [f"{cls}_{s * args.max_objects + k + 1:04d}" for cls in classes for k in range(args.max_objects)]
            (dst_root / f"modelnet{variant}_{split}.txt").write_text("\n".join(ids) + "\n")

    # ModelNet10 is a subset of ModelNet40, so every object is written once.
    stems = [f"{cls}_{k + 1:04d}" for cls in MODELNET40_CLASSES for k in range(len(SPLITS) * args.max_objects)]
    for stem in stems:
        direction = rng.normal(size=(args.max_points, 3))
        direction /= np.linalg.norm(direction, axis=1, keepdims=True)
        radius = rng.uniform(0.2, 1.0, size=3)
        radius /= radius.max()
        normal = direction / radius
        normal /= np.linalg.norm(normal, axis=1, keepdims=True)

        dst_txt = dst_root / stem.rsplit("_", 1)[0] / f"{stem}.txt"
        dst_txt.parent.mkdir(parents=True, exist_ok=True)
        np.savetxt(dst_txt, np.concatenate([radius * direction, normal], axis=1), fmt="%.6f", delimiter=",")

    lines = sorted(f"{stem.rsplit('_', 1)[0]}/{stem}.txt" for stem in stems)
    (dst_root / "filelist.txt").write_text("\n".join(lines) + "\n")

    print(f"  Wrote {len(stems)} unique objects, {args.max_points} points each.")


def generate_processed(args: Namespace) -> None:
    raw_dir = Path(args.raw_dir)
    root = raw_dir.resolve().parent.parent.as_posix()

    for variant in VARIANTS:
        for train in (True, False):
            dataset = ModelNetNormalResampled(
                root=root,
                variant=variant,  # type: ignore[arg-type]
                train=train,
                show_progress=True,
                force_process=True,
            )
    # The committed cache stays without metadata: the tests read it as a cache from before metadata existed.
    for meta_path in Path(dataset.processed_dir).glob("*.meta.json"):
        meta_path.unlink()


if __name__ == "__main__":
    main()
