"""Generate a tiny synthetic SemanticKITTI fixture in the layout of the original release.

The output mirrors the real layout:

    raw/sequences/{seq}/velodyne/{frame:06d}.bin    # float32 (N, 4) = (x, y, z, intensity)
    raw/sequences/{seq}/labels/{frame:06d}.label    # uint32 (N,)   = (instance << 16) | semantic_id

The test split (sequences 11-21 in the real dataset) has velodyne but no `labels/`,
matching the real release where test labels are withheld.

No scan of the original release is read: every sweep is cast from a seeded generator, one ray per
(beam, azimuth) pair of a spinning sensor, against a road, a facade and a parked car.

Usage:
    uv run --no-sync python scripts/generate.py raw ./raw
"""

from argparse import ArgumentParser, Namespace
from pathlib import Path

import numpy as np

from torch_pointcloud.datasets.semantickitti import SEMANTIC_KITTI_SEQUENCES_PER_SPLIT


def generate(args: Namespace) -> None:
    rng = np.random.default_rng(args.seed)
    out_root = Path(args.dst_dir)

    # Keeping it tiny (1 sequence per split, 2 frames per sequence) is enough to exercise enumeration
    # and per-split routing.
    plan = {
        "train": (SEMANTIC_KITTI_SEQUENCES_PER_SPLIT["train"][0], 2),  # seq 00
        "val": (SEMANTIC_KITTI_SEQUENCES_PER_SPLIT["val"][0], 2),  # seq 08
        "test": (SEMANTIC_KITTI_SEQUENCES_PER_SPLIT["test"][0], 2),  # seq 11 (no labels)
    }

    num_beams = 16
    azimuth, elevation = np.meshgrid(
        np.linspace(-np.pi, np.pi, args.num_points // num_beams, endpoint=False),
        np.deg2rad(np.linspace(-24.8, -0.5, num_beams)),
    )
    azimuth, elevation = azimuth.ravel(), elevation.ravel()

    for split, (seq, num_frames) in plan.items():
        dst_seq_dir = out_root / "sequences" / seq
        for frame in range(num_frames):
            # Horizontal distance at which each ray stops: on the road 1.73 m below the sensor, on a facade,
            # or on a car parked in a narrow azimuth sector.
            road = 1.73 / np.tan(-elevation)
            facade = np.full_like(road, rng.uniform(8.0, 40.0))
            car_azimuth = rng.uniform(-np.pi, np.pi - 0.4)
            car = np.where((azimuth > car_azimuth) & (azimuth < car_azimuth + 0.4), rng.uniform(4.0, 7.0), np.inf)
            distance = np.stack([road, facade, car])
            hit = distance.argmin(axis=0)
            distance = distance.min(axis=0)

            pos = np.stack(
                [distance * np.cos(azimuth), distance * np.sin(azimuth), distance * np.tan(elevation)], axis=1
            )
            pos += rng.normal(0.0, 0.02, size=pos.shape)
            scan = np.concatenate([pos, rng.uniform(0.0, 1.0, size=(len(pos), 1))], axis=1)
            (dst_seq_dir / "velodyne").mkdir(parents=True, exist_ok=True)
            scan.astype(np.float32).tofile(dst_seq_dir / "velodyne" / f"{frame:06d}.bin")

            if split != "test":
                semantic = np.array([40, 50, 10], dtype=np.uint32)[hit]  # road, building, car
                instance = (hit == 2).astype(np.uint32)
                (dst_seq_dir / "labels").mkdir(parents=True, exist_ok=True)
                ((instance << 16) | semantic).tofile(dst_seq_dir / "labels" / f"{frame:06d}.label")

        print(f"generated {split:>5}: sequences/{seq} ({num_frames} frames, {args.num_points} pts each)")


def parse_args() -> Namespace:
    parser = ArgumentParser(description="Generate a tiny synthetic SemanticKITTI test fixture.")
    parser.add_argument("command", choices=["raw"], help="What to generate (only 'raw' is supported).")
    parser.add_argument("dst_dir", type=str, help="Output directory (e.g. ./raw).")
    parser.add_argument("--num-points", type=int, default=1024, help="Points per scan.")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "raw":
        generate(args)


if __name__ == "__main__":
    main()
