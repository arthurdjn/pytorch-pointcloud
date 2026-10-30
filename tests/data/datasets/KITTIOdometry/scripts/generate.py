"""Generate a tiny synthetic KITTI odometry fixture in the layout of the velodyne release and the GeoTransformer pair lists.

    raw/sequences/<seq>/velodyne/{frame:06d}.bin     # float32 (N, 4) = (x, y, z, intensity)
    raw/metadata/{train,val,test}.pkl                 # list of pair records (seq_id, frame0, frame1, transform, ...)

No file of the real release is read. Every sequence is a seeded road (a ground plane and two walls) and every scan a
rigidly moved, cropped copy of it, so the written transforms align the scans exactly.

Usage:
    uv run --no-sync python scripts/generate.py raw ./raw
"""

import pickle
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from scipy.spatial.transform import Rotation

# One sequence per split, with the scans of each and the pairs the metadata lists, `(reference, source)`.
SEQUENCES: Dict[str, Tuple[str, int, Tuple[Tuple[int, int], ...]]] = {
    "train": ("00", 3, ((0, 1), (1, 2), (0, 2))),
    "val": ("06", 2, ((0, 1),)),
    "test": ("08", 3, ((0, 1), (1, 2), (0, 2))),
}


def main() -> None:
    args = parse_args()
    if args.command == "raw":
        generate_raw(args)


def parse_args() -> Namespace:
    parser = ArgumentParser(description="Generate synthetic KITTI odometry test data.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    raw_parser = subparsers.add_parser("raw", help="Generate raw test data")
    raw_parser.add_argument("dst_dir", type=str, help="Path to output raw directory.")
    raw_parser.add_argument("--num-points", type=int, default=1500, help="Points sampled per surface of a road.")
    raw_parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser.parse_args()


def road(rng: np.random.Generator, num_points: int) -> np.ndarray:
    """A 60 m stretch of road: the ground and a wall on each side, sampled on a 10 cm grid."""
    ground = np.stack([rng.uniform(0, 60, num_points), rng.uniform(-8, 8, num_points), np.zeros(num_points)], axis=1)
    left = np.stack([rng.uniform(0, 60, num_points), np.full(num_points, -8.0), rng.uniform(0, 4, num_points)], axis=1)
    right = np.stack([rng.uniform(0, 60, num_points), np.full(num_points, 8.0), rng.uniform(0, 4, num_points)], axis=1)
    points = np.concatenate([ground, left, right])
    return np.round(points / 0.1) * 0.1


def scans(rng: np.random.Generator, num_points: int, num_scans: int) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Scans of one sequence, each a crop of the road in its own frame, with the pose of every scan.

    The crops are subsets of one sampling of the road, so the overlapping points of two scans coincide exactly once
    aligned.
    """
    scene = road(rng, num_points)
    scene = scene[rng.permutation(len(scene))[: 2 * num_points]]
    clouds, poses = [], []
    for i in range(num_scans):
        keep = scene[:, 0] > i * 12.0  # overlapping crops along the road
        crop = scene[keep]
        pose = np.eye(4)
        pose[:3, :3] = Rotation.from_euler("zyx", [rng.uniform(-0.3, 0.3), 0.0, 0.0]).as_matrix()
        pose[:3, 3] = [12.0 * i + rng.uniform(-1, 1), rng.uniform(-1, 1), 0.0]
        clouds.append((crop - pose[:3, 3]) @ pose[:3, :3])  # the scan in its own frame: pose @ local = world
        poses.append(pose)
    return clouds, poses


def pair_transform(poses: List[np.ndarray], ref: int, src: int) -> np.ndarray:
    """Transform of scan `src` onto scan `ref`: `ref_local = T @ src_local`."""
    return np.linalg.inv(poses[ref]) @ poses[src]


def write_scans(dst: Path, rng: np.random.Generator, clouds: List[np.ndarray]) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for i, cloud in enumerate(clouds):
        intensity = rng.uniform(0, 1, (len(cloud), 1))
        np.concatenate([cloud, intensity], axis=1).astype(np.float32).tofile(dst / f"{i:06d}.bin")


def generate_raw(args: Namespace) -> None:
    rng = np.random.default_rng(args.seed)
    dst = Path(args.dst_dir)
    metadata = dst / "metadata"
    metadata.mkdir(parents=True, exist_ok=True)

    for split, (sequence, num_scans, pairs) in SEQUENCES.items():
        clouds, poses = scans(rng, args.num_points, num_scans)
        write_scans(dst / "sequences" / sequence / "velodyne", rng, clouds)
        records = []
        for ref, src in pairs:
            records.append(
                {
                    "seq_id": int(sequence),
                    "frame0": np.int64(ref),
                    "frame1": src,
                    "transform": pair_transform(poses, ref, src),
                    "pcd0": f"downsampled/{sequence}/{ref:06d}.npy",
                    "pcd1": f"downsampled/{sequence}/{src:06d}.npy",
                }
            )
        with open(metadata / f"{split}.pkl", "wb") as f:
            pickle.dump(records, f)


if __name__ == "__main__":
    main()
