"""Generate a tiny SemanticKITTI fixture by subsampling real raw scans.

The output mirrors the real layout:

    raw/sequences/{seq}/velodyne/{frame:06d}.bin    # float32 (N, 4) = (x, y, z, intensity)
    raw/sequences/{seq}/labels/{frame:06d}.label    # uint32 (N,)   = (instance << 16) | semantic_id

The test split (sequences 11-21 in the real dataset) has velodyne but no `labels/`,
matching the real release where test labels are withheld.

The default source directory is `$TORCH_POINTCLOUD_DATA_DIR/SemanticKITTI/raw`.

Usage:
    uv run --no-sync python scripts/generate.py raw ./raw
    uv run --no-sync python scripts/generate.py raw ./raw --src-dir /path/to/SemanticKITTI/raw
"""

from argparse import ArgumentParser, Namespace
from pathlib import Path

import numpy as np

from torch_pointcloud.config import DATA_DIR
from torch_pointcloud.datasets.semantickitti import SEMANTIC_KITTI_SEQUENCES_PER_SPLIT


def _subsample_scan(src_bin: Path, dst_bin: Path, indices: np.ndarray) -> None:
    scan = np.fromfile(src_bin, dtype=np.float32).reshape(-1, 4)
    dst_bin.parent.mkdir(parents=True, exist_ok=True)
    scan[indices].astype(np.float32).tofile(dst_bin)


def _subsample_label(src_label: Path, dst_label: Path, indices: np.ndarray) -> None:
    labels = np.fromfile(src_label, dtype=np.uint32)
    dst_label.parent.mkdir(parents=True, exist_ok=True)
    labels[indices].astype(np.uint32).tofile(dst_label)


def generate(args: Namespace) -> None:
    rng = np.random.default_rng(args.seed)
    src_root = Path(args.src_dir)
    out_root = Path(args.dst_dir)

    if not src_root.exists():
        raise FileNotFoundError(
            f"Source SemanticKITTI raw directory not found: {src_root!r}. "
            f"Set --src-dir or TORCH_POINTCLOUD_DATA_DIR to a folder containing 'sequences/'."
        )

    # Pick a few sequences from each split. Keeping it tiny — 1 sequence per split,
    # 2 frames per sequence — is enough to exercise enumeration and per-split routing.
    plan = {
        "train": (SEMANTIC_KITTI_SEQUENCES_PER_SPLIT["train"][0], 2),  # seq 00
        "val": (SEMANTIC_KITTI_SEQUENCES_PER_SPLIT["val"][0], 2),  # seq 08
        "test": (SEMANTIC_KITTI_SEQUENCES_PER_SPLIT["test"][0], 2),  # seq 11 (no labels)
    }

    for split, (seq, num_frames) in plan.items():
        src_seq_dir = src_root / "sequences" / seq
        src_velodyne = src_seq_dir / "velodyne"
        src_labels = src_seq_dir / "labels"

        if not src_velodyne.exists():
            raise FileNotFoundError(f"Missing velodyne directory for sequence {seq!r}: {src_velodyne!r}")

        bin_paths = sorted(src_velodyne.glob("*.bin"))[:num_frames]
        if not bin_paths:
            raise FileNotFoundError(f"No .bin scans found in {src_velodyne!r}")

        dst_seq_dir = out_root / "sequences" / seq
        for src_bin in bin_paths:
            scan = np.fromfile(src_bin, dtype=np.float32).reshape(-1, 4)
            num_total = scan.shape[0]
            num_keep = min(args.num_points, num_total)
            indices = rng.choice(num_total, size=num_keep, replace=False)
            indices.sort()

            dst_bin = dst_seq_dir / "velodyne" / src_bin.name
            _subsample_scan(src_bin, dst_bin, indices)

            if split != "test":
                src_label = src_labels / f"{src_bin.stem}.label"
                if not src_label.exists():
                    raise FileNotFoundError(f"Missing label file for {src_bin.name!r}: {src_label!r}")
                dst_label = dst_seq_dir / "labels" / f"{src_bin.stem}.label"
                _subsample_label(src_label, dst_label, indices)

        print(f"generated {split:>5}: sequences/{seq} ({len(bin_paths)} frames, {args.num_points} pts each)")


def parse_args() -> Namespace:
    parser = ArgumentParser(description="Generate a tiny SemanticKITTI test fixture by subsampling real scans.")
    parser.add_argument("command", choices=["raw"], help="What to generate (only 'raw' is supported).")
    parser.add_argument("dst_dir", type=str, help="Output directory (e.g. ./raw).")
    parser.add_argument(
        "--src-dir",
        type=str,
        default=str(Path(DATA_DIR) / "SemanticKITTI" / "raw"),
        help="Source SemanticKITTI raw directory (default: $TORCH_POINTCLOUD_DATA_DIR/SemanticKITTI/raw).",
    )
    parser.add_argument("--num-points", type=int, default=1024, help="Points per scan after subsampling.")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "raw":
        generate(args)


if __name__ == "__main__":
    main()
