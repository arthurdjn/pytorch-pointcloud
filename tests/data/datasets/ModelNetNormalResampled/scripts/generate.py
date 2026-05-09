"""Generate a tiny ModelNetNormalResampled fixture by subsampling real point clouds.

The raw release ships pre-resampled point clouds (~10k points each) as `.txt`
files with `(x, y, z, nx, ny, nz)` per row. We keep `--max-objects` files per
class per split, subsample each to `--max-points` rows, and trim the split lists
to the kept ids so the dataset class can locate them.

The default source directory is `$TORCH_POINTCLOUD_DATA_DIR/ModelNetNormalResampled/raw`.

Usage:
    uv run --no-sync python scripts/generate.py raw ./raw
    uv run --no-sync python scripts/generate.py process ./raw
"""

from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import List

import numpy as np

from torch_pointcloud.config import DATA_DIR
from torch_pointcloud.datasets import ModelNetNormalResampled

VARIANTS = ("10", "40")
SPLITS = ("train", "test")


def main() -> None:
    args = parse_args()
    if args.command == "raw":
        generate_raw(args)
    elif args.command == "process":
        generate_processed(args)


def parse_args() -> Namespace:
    parser = ArgumentParser(description="Generate ModelNetNormalResampled test data by subsampling real .txt files.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    raw_parser = subparsers.add_parser("raw", help="Generate raw test data")
    raw_parser.add_argument("dst_dir", type=str, help="Path to output raw data")
    raw_parser.add_argument(
        "--src-dir",
        type=str,
        default=str(Path(DATA_DIR) / "ModelNetNormalResampled" / "raw"),
        help="Source ModelNetNormalResampled raw directory.",
    )
    raw_parser.add_argument("--max-points", type=int, default=1024, help="Points kept per object after subsampling.")
    raw_parser.add_argument("--max-objects", type=int, default=2, help="Files kept per class per split.")
    raw_parser.add_argument("--seed", type=int, default=42, help="Random seed.")

    process_parser = subparsers.add_parser("process", help="Process raw data into final format")
    process_parser.add_argument("raw_dir", type=str, help="Path to raw data directory")

    return parser.parse_args()


def generate_raw(args: Namespace) -> None:
    src_root = Path(args.src_dir)
    dst_root = Path(args.dst_dir)
    if not src_root.exists():
        raise FileNotFoundError(
            f"Source ModelNetNormalResampled raw directory not found: {src_root!r}. "
            f"Set --src-dir or TORCH_POINTCLOUD_DATA_DIR."
        )

    rng = np.random.default_rng(args.seed)

    # Per-class kept ids by split (stem only, e.g. "airplane_0001"). Used below to
    # rewrite the split files so they only reference files that we actually shipped.
    kept_ids_by_split: dict[str, List[str]] = {split: [] for split in SPLITS}

    # The split files are the source of truth for which file goes in which split,
    # so iterate them and keep `max-objects` per class per split.
    for variant in VARIANTS:
        for split in SPLITS:
            src_split = src_root / f"modelnet{variant}_{split}.txt"
            with open(src_split, "r") as f:
                ids = [line.strip() for line in f if line.strip()]
            counts: dict[str, int] = {}
            kept = []
            for stem in ids:
                # Class name = everything before the last underscore.
                cls = "_".join(stem.split("_")[:-1])
                if counts.get(cls, 0) >= args.max_objects:
                    continue
                counts[cls] = counts.get(cls, 0) + 1
                kept.append(stem)
            kept_ids_by_split[split].extend(kept)

            dst_split = dst_root / f"modelnet{variant}_{split}.txt"
            dst_split.parent.mkdir(parents=True, exist_ok=True)
            with open(dst_split, "w") as f:
                f.write("\n".join(kept) + "\n")

    # Copy class-name lists and full filelist (small; helps downstream tooling).
    for fname in ("modelnet10_shape_names.txt", "modelnet40_shape_names.txt"):
        src = src_root / fname
        if src.exists():
            (dst_root / fname).write_text(src.read_text())

    # Subsample every retained file once, dedup across variants/splits.
    seen: set[str] = set()
    for split, kept in kept_ids_by_split.items():
        for stem in kept:
            if stem in seen:
                continue
            seen.add(stem)
            cls = "_".join(stem.split("_")[:-1])
            src_txt = src_root / cls / f"{stem}.txt"
            if not src_txt.exists():
                raise FileNotFoundError(f"Missing source file: {src_txt!r}")
            data = np.loadtxt(src_txt, delimiter=",")
            num_keep = min(args.max_points, data.shape[0])
            indices = rng.choice(data.shape[0], size=num_keep, replace=False)
            indices.sort()
            data = data[indices]

            dst_txt = dst_root / cls / f"{stem}.txt"
            dst_txt.parent.mkdir(parents=True, exist_ok=True)
            np.savetxt(dst_txt, data, fmt="%.6f", delimiter=",")

    # Rewrite filelist.txt to only the files we kept.
    dst_filelist = dst_root / "filelist.txt"
    lines = sorted(f"{'_'.join(s.split('_')[:-1])}/{s}.txt" for s in seen)
    dst_filelist.write_text("\n".join(lines) + "\n")

    print(f"  Kept {len(seen)} unique objects, ~{args.max_points} points each.")


def generate_processed(args: Namespace) -> None:
    raw_dir = Path(args.raw_dir)
    root = raw_dir.resolve().parent.parent.as_posix()

    for variant in VARIANTS:
        for train in (True, False):
            _ = ModelNetNormalResampled(
                root=root,
                variant=variant,  # type: ignore[arg-type]
                train=train,
                show_progress=True,
                force_process=True,
            )


if __name__ == "__main__":
    main()
