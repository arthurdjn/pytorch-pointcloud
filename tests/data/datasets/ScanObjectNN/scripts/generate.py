"""Generate a tiny ScanObjectNN fixture by subsampling real `.h5` archives.

For each (split, background, train, variant) combination, we keep one real example
per class (15 classes) and randomly subsample its points down to `--num-points`.

The default source directory is `$TORCH_POINTCLOUD_DATA_DIR/ScanObjectNN/raw`.

Usage:
    uv run --no-sync python scripts/generate.py raw ./raw
    uv run --no-sync python scripts/generate.py process ./raw
"""

from argparse import ArgumentParser, Namespace
from pathlib import Path

import h5py
import numpy as np

from torch_pointcloud.config import DATA_DIR
from torch_pointcloud.datasets import ScanObjectNN
from torch_pointcloud.datasets.scanobjectnn import (
    SCANOBJECTNN_CLASSES,
    SCANOBJECTNN_SPLITS,
    SCANOBJECTNN_VARIANTS,
)

NUM_CLASSES = len(SCANOBJECTNN_CLASSES)

SPLIT_DIRS: list[str] = []
for _split in SCANOBJECTNN_SPLITS:
    for _bg in (True, False):
        name = _split
        if _split == "main":
            name += "_split"
        if not _bg:
            name += "_nobg"
        SPLIT_DIRS.append(name)

FILE_STEMS: list[str] = []
for _train in (True, False):
    prefix = "training" if _train else "test"
    FILE_STEMS.append(f"{prefix}_objectdataset")
    for _variant in SCANOBJECTNN_VARIANTS:
        FILE_STEMS.append(f"{prefix}_objectdataset_{_variant}")


def main() -> None:
    args = parse_args()

    if args.command == "raw":
        generate_raw(args)
    elif args.command == "process":
        generate_processed(args)


def parse_args() -> Namespace:
    parser = ArgumentParser(description="Generate ScanObjectNN test data by subsampling real .h5 files.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    raw_parser = subparsers.add_parser("raw", help="Generate raw test data")
    raw_parser.add_argument("dst_dir", type=str, help="Path to output raw data directory")
    raw_parser.add_argument(
        "--src-dir",
        type=str,
        default=str(Path(DATA_DIR) / "ScanObjectNN" / "raw"),
        help="Source ScanObjectNN raw directory (default: $TORCH_POINTCLOUD_DATA_DIR/ScanObjectNN/raw).",
    )
    raw_parser.add_argument("--num-points", type=int, default=1024, help="Points per object after subsampling.")
    raw_parser.add_argument(
        "--num-objects",
        type=int,
        default=2,
        help="Objects per file (except the full-coverage label-test file, which always has 15).",
    )
    raw_parser.add_argument("--seed", type=int, default=42, help="Random seed.")

    process_parser = subparsers.add_parser("process", help="Process raw data into final format")
    process_parser.add_argument("raw_dir", type=str, help="Path to raw data directory")

    return parser.parse_args()


def _pick_one_per_class(labels: np.ndarray) -> np.ndarray:
    """Return indices of the first occurrence of each class id in `labels`."""
    picked = []
    for cls in range(NUM_CLASSES):
        matches = np.flatnonzero(labels == cls)
        if matches.size == 0:
            continue
        picked.append(int(matches[0]))
    return np.asarray(picked, dtype=np.int64)


# Files that must contain one example per class so per-class unit tests pass.
# All others only need a couple of examples (enough for batch_size=2 forwards).
_FULL_COVERAGE_FILES: set[str] = {
    "main_split_nobg/test_objectdataset.h5",
}


def generate_raw(args: Namespace) -> None:
    src_root = Path(args.src_dir)
    if not src_root.exists():
        raise FileNotFoundError(
            f"Source ScanObjectNN raw directory not found: {src_root!r}. "
            f"Set --src-dir or TORCH_POINTCLOUD_DATA_DIR."
        )

    rng = np.random.default_rng(args.seed)

    for split_dir in SPLIT_DIRS:
        out_dir = Path(args.dst_dir, split_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        for stem in FILE_STEMS:
            src_h5 = src_root / split_dir / f"{stem}.h5"
            if not src_h5.exists():
                # Some variants exist only for a subset of splits; fall back to the
                # first available source file so the fixture preserves the layout.
                fallback = _find_fallback(src_root, stem)
                if fallback is None:
                    raise FileNotFoundError(
                        f"Could not find any source .h5 for stem {stem!r} under {src_root!r}."
                    )
                src_h5 = fallback

            with h5py.File(src_h5, "r") as f:
                src_pos = f["data"][:]  # (N, 2048, 3)
                src_labels = f["label"][:]  # (N,)

            relpath = f"{split_dir}/{stem}.h5"
            if relpath in _FULL_COVERAGE_FILES:
                indices = _pick_one_per_class(src_labels.astype(np.int64))
            else:
                indices = np.arange(min(args.num_objects, src_pos.shape[0]), dtype=np.int64)
            pos = src_pos[indices]
            labels = src_labels[indices].astype(np.int64)

            num_pts = min(args.num_points, pos.shape[1])
            point_idx = rng.choice(pos.shape[1], size=num_pts, replace=False)
            point_idx.sort()
            pos = pos[:, point_idx, :].astype(np.float32)

            dst_h5 = out_dir / f"{stem}.h5"
            with h5py.File(dst_h5, "w") as f:
                f.create_dataset("data", data=pos)
                f.create_dataset("label", data=labels)
            print(f"  {dst_h5}  (objects={len(labels)}, points={num_pts})")

    print("Done!")


def _find_fallback(src_root: Path, stem: str) -> Path | None:
    for candidate in src_root.rglob(f"{stem}.h5"):
        return candidate
    return None


def generate_processed(args: Namespace) -> None:
    raw_dir = Path(args.raw_dir)
    root = raw_dir.resolve().parent.parent.as_posix()

    for split in SCANOBJECTNN_SPLITS:
        for variant in list(SCANOBJECTNN_VARIANTS) + [None]:
            for background in (True, False):
                for train in (True, False):
                    _ = ScanObjectNN(
                        root=root,
                        split=split,
                        variant=variant,
                        background=background,
                        train=train,
                        force_process=True,
                    )


if __name__ == "__main__":
    main()
