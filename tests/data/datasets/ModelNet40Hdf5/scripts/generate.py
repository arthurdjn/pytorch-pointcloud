"""Generate a tiny synthetic `modelnet40_ply_hdf5_2048` fixture in the layout of the original release.

The script writes two train shards and one test shard of `--num-objects` objects with `--num-points` points
each, plus the release-style `train_files.txt` / `test_files.txt` lists. No shard of the original release is
read: every object is a noisy ellipsoid on the unit sphere drawn from a seeded generator, with unit normals.

Usage:
    uv run --no-sync python scripts/generate.py ./raw
"""

from argparse import ArgumentParser, Namespace
from pathlib import Path

import h5py
import numpy as np

from torch_pointcloud.datasets.modelnet import MODELNET40_CLASSES

RELEASE_DIR = "data/modelnet40_ply_hdf5_2048"


def main() -> None:
    args = parse_args()
    raw_dir = Path(args.raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    shards = {"train": ["ply_data_train0.h5", "ply_data_train1.h5"], "test": ["ply_data_test0.h5"]}
    for split, names in shards.items():
        for name in names:
            write_shard(raw_dir / name, rng, num_objects=args.num_objects, num_points=args.num_points)
        (raw_dir / f"{split}_files.txt").write_text("".join(f"{RELEASE_DIR}/{name}\n" for name in names))
    (raw_dir / "shape_names.txt").write_text("".join(f"{name}\n" for name in MODELNET40_CLASSES))


def write_shard(path: Path, rng: np.random.Generator, *, num_objects: int, num_points: int) -> None:
    """Write one shard with the release's `data` / `normal` / `label` datasets."""
    directions = rng.normal(size=(num_objects, num_points, 3)).astype(np.float32)
    directions /= np.linalg.norm(directions, axis=-1, keepdims=True)
    radii = rng.uniform(0.4, 1.0, size=(num_objects, 1, 3)).astype(np.float32)
    pos = directions * radii
    pos /= np.linalg.norm(pos, axis=-1, keepdims=True).max(axis=1, keepdims=True)
    labels = rng.integers(0, len(MODELNET40_CLASSES), size=(num_objects, 1), dtype=np.uint8)
    with h5py.File(path, "w") as f:
        f.create_dataset("data", data=pos)
        f.create_dataset("normal", data=directions)
        f.create_dataset("label", data=labels)


def parse_args() -> Namespace:
    parser = ArgumentParser(description="Generate synthetic ModelNet40 HDF5 test data.")
    parser.add_argument("raw_dir", help="Directory receiving the shards and file lists (the dataset's `raw/`).")
    parser.add_argument("--num-objects", type=int, default=4, help="Objects per shard.")
    parser.add_argument("--num-points", type=int, default=2048, help="Points per object.")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


if __name__ == "__main__":
    main()
