"""Generate a tiny synthetic SemanticKITTI fixture for unit tests.

The output mirrors the real layout:

    raw/sequences/{seq}/velodyne/{frame:06d}.bin    # float32 (N, 4) = (x, y, z, intensity)
    raw/sequences/{seq}/labels/{frame:06d}.label    # uint32 (N,)   = (instance << 16) | semantic_id

The test split (sequences 11–21 in the real dataset) has velodyne but no `labels/`,
matching the real release where test labels are withheld.

Usage:
    uv run python scripts/generate.py raw ./raw

After running, scenes are tiny (~50 points each, a few KB total) so the fixture
can live in git and CI runs cheaply.
"""

from argparse import ArgumentParser, Namespace
from pathlib import Path

import numpy as np

from torch_pointcloud.datasets.semantickitti import (
    SEMANTIC_KITTI_LABEL_NAMES,
    SEMANTIC_KITTI_SEQUENCES_PER_SPLIT,
)


def _write_scan(velodyne_dir: Path, frame_id: str, num_points: int, rng: np.random.Generator) -> None:
    """Write a velodyne `.bin` (float32, (N, 4) = xyz + intensity)."""
    velodyne_dir.mkdir(parents=True, exist_ok=True)
    pos = rng.uniform(-50.0, 50.0, size=(num_points, 3)).astype(np.float32)
    intensity = rng.uniform(0.0, 1.0, size=(num_points, 1)).astype(np.float32)
    scan = np.concatenate([pos, intensity], axis=1)
    out = velodyne_dir / f"{frame_id}.bin"
    out.write_bytes(scan.tobytes())


def _write_label(labels_dir: Path, frame_id: str, num_points: int, rng: np.random.Generator) -> None:
    """Write a `.label` file. Lower 16 bits = semantic id, upper 16 bits = instance id."""
    labels_dir.mkdir(parents=True, exist_ok=True)
    valid_ids = np.array(list(SEMANTIC_KITTI_LABEL_NAMES.keys()), dtype=np.uint32)
    semantic = rng.choice(valid_ids, size=num_points)
    instance = rng.integers(0, 4, size=num_points, dtype=np.uint32)
    label = (instance << 16) | semantic
    out = labels_dir / f"{frame_id}.label"
    out.write_bytes(label.astype(np.uint32).tobytes())


def generate(args: Namespace) -> None:
    rng = np.random.default_rng(args.seed)
    out_root = Path(args.dst_dir)

    # Pick a few sequences from each split. Keeping it tiny — 1 sequence per split,
    # 2 frames per sequence — is enough to exercise enumeration and per-split routing.
    plan = {
        "train": (SEMANTIC_KITTI_SEQUENCES_PER_SPLIT["train"][0], 2),  # seq 00
        "val": (SEMANTIC_KITTI_SEQUENCES_PER_SPLIT["val"][0], 2),  # seq 08
        "test": (SEMANTIC_KITTI_SEQUENCES_PER_SPLIT["test"][0], 2),  # seq 11 (no labels)
    }

    for split, (seq, num_frames) in plan.items():
        seq_dir = out_root / "sequences" / seq
        velodyne_dir = seq_dir / "velodyne"
        labels_dir = seq_dir / "labels"
        for i in range(num_frames):
            frame_id = f"{i:06d}"
            _write_scan(velodyne_dir, frame_id, args.num_points, rng)
            if split != "test":
                _write_label(labels_dir, frame_id, args.num_points, rng)
        print(f"generated {split:>5}: sequences/{seq} ({num_frames} frames, {args.num_points} pts each)")


def parse_args() -> Namespace:
    parser = ArgumentParser(description="Generate a tiny SemanticKITTI test fixture.")
    parser.add_argument("command", choices=["raw"], help="What to generate (only 'raw' is supported).")
    parser.add_argument("dst_dir", type=str, help="Output directory (e.g. ./raw).")
    parser.add_argument("--num-points", type=int, default=50, help="Points per scan.")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "raw":
        generate(args)


if __name__ == "__main__":
    main()
