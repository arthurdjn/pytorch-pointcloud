"""Generate a tiny Semantic3D fixture.

For each scene we produce two ASCII files matching the real release format:

- `<scene>.txt`      — `x y z intensity r g b` rows (whitespace-separated)
- `<scene>.labels`   — class id rows (only for train scenes; held-out test scenes
                        have no labels file in the real release)

If `--src-dir` points at a real Semantic3D raw directory we subsample its scenes
in place; otherwise the script falls back to generating structured synthetic
data with the right schema (random points in a 50 m bounding box, RGB in 0-255,
labels uniformly sampled from the eight benchmark classes).

Usage:
    uv run --no-sync python scripts/generate.py raw ./raw
    uv run --no-sync python scripts/generate.py raw ./raw --src-dir /path/to/Semantic3D/raw
"""

from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Iterable, Optional

import numpy as np

from torch_pointcloud.config import DATA_DIR
from torch_pointcloud.datasets.semantic3d import SEMANTIC3D_CLASSES

# A single scene per split keeps the fixture tiny (~75 KB total at 1024 points).
_DEFAULT_TRAIN_SCENES = ("bildstein_station1_xyz_intensity_rgb",)
_DEFAULT_TEST_SCENES = ("MarketplaceFeldkirch_Station4_rgb_intensity-reduced",)


def main() -> None:
    args = parse_args()
    if args.command == "raw":
        generate_raw(args)


def parse_args() -> Namespace:
    parser = ArgumentParser(description="Generate Semantic3D test data (subsampled real or synthetic).")
    subparsers = parser.add_subparsers(dest="command", required=True)

    raw_parser = subparsers.add_parser("raw", help="Generate raw test data")
    raw_parser.add_argument("dst_dir", type=str, help="Path to output raw directory.")
    raw_parser.add_argument(
        "--src-dir",
        type=str,
        default=str(Path(DATA_DIR) / "Semantic3D" / "raw"),
        help="Source Semantic3D raw directory (default: $TORCH_POINTCLOUD_DATA_DIR/Semantic3D/raw).",
    )
    raw_parser.add_argument("--num-points", type=int, default=1024, help="Points kept per scene.")
    raw_parser.add_argument(
        "--train-scenes",
        nargs="+",
        default=list(_DEFAULT_TRAIN_SCENES),
        help="Train scene names (without .txt/.labels suffix).",
    )
    raw_parser.add_argument(
        "--test-scenes",
        nargs="+",
        default=list(_DEFAULT_TEST_SCENES),
        help="Held-out test scene names (no .labels file is written).",
    )
    raw_parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser.parse_args()


def generate_raw(args: Namespace) -> None:
    src_root = Path(args.src_dir)
    dst_root = Path(args.dst_dir)
    rng = np.random.default_rng(args.seed)
    dst_root.mkdir(parents=True, exist_ok=True)

    use_real = src_root.exists()
    if use_real:
        print(f"Subsampling real Semantic3D scenes from {src_root}.")
    else:
        print(f"No source data at {src_root}; generating synthetic Semantic3D fixture.")

    for scene in args.train_scenes:
        _emit_scene(scene, src_root, dst_root, args.num_points, rng, with_labels=True, use_real=use_real)
    for scene in args.test_scenes:
        _emit_scene(scene, src_root, dst_root, args.num_points, rng, with_labels=False, use_real=use_real)


def _emit_scene(
    scene: str,
    src_root: Path,
    dst_root: Path,
    num_points: int,
    rng: np.random.Generator,
    with_labels: bool,
    use_real: bool,
) -> None:
    src_txt = src_root / f"{scene}.txt"
    src_labels = src_root / f"{scene}.labels"

    if use_real and src_txt.exists():
        arr = np.loadtxt(src_txt.as_posix())  # (N, 7) = xyz + intensity + rgb
        n = arr.shape[0]
        keep_n = min(num_points, n)
        indices = rng.choice(n, size=keep_n, replace=False)
        indices.sort()
        arr = arr[indices]
        labels: Optional[np.ndarray]
        if with_labels and src_labels.exists():
            labels = np.loadtxt(src_labels.as_posix(), dtype=np.int64)[indices]
        else:
            labels = None
    else:
        # Synthetic fallback: structured-but-random points in a 50 m cube.
        keep_n = num_points
        pos = rng.uniform(-25.0, 25.0, size=(keep_n, 3)).astype(np.float64)
        intensity = rng.uniform(-0.5, 0.5, size=(keep_n, 1)).astype(np.float64)
        rgb = rng.integers(0, 256, size=(keep_n, 3)).astype(np.float64)
        arr = np.concatenate([pos, intensity, rgb], axis=1)
        labels = rng.integers(1, len(SEMANTIC3D_CLASSES), size=(keep_n,), dtype=np.int64) if with_labels else None

    dst_txt = dst_root / f"{scene}.txt"
    np.savetxt(dst_txt.as_posix(), arr, fmt=("%.6f", "%.6f", "%.6f", "%.6f", "%d", "%d", "%d"))
    print(f"  {dst_txt}  ({keep_n} points)")

    if labels is not None:
        dst_labels = dst_root / f"{scene}.labels"
        np.savetxt(dst_labels.as_posix(), labels, fmt="%d")
        print(f"  {dst_labels}  ({keep_n} labels)")


def _resolve(scene: str, root: Path, ext: str) -> Path:
    return root / f"{scene}.{ext}"


def _all_scenes(scenes: Iterable[str]) -> Iterable[str]:
    return tuple(scenes)


if __name__ == "__main__":
    main()
