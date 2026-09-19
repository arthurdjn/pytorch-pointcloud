"""Generate a tiny synthetic Semantic3D fixture in the layout of the original release.

For each scene we produce two ASCII files matching the real release format:

- `<scene>.txt`:      `x y z intensity r g b` rows (whitespace-separated)
- `<scene>.labels`:   class id rows (only for train scenes; held-out test scenes
                        have no labels file in the real release)

No scan of the original release is read: the scenes are drawn from a seeded generator (random points in a
50 m bounding box, RGB in 0-255, labels uniformly sampled from the eight benchmark classes).

Usage:
    uv run --no-sync python scripts/generate.py raw ./raw
"""

from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Iterable

import numpy as np

from torch_pointcloud.datasets.semantic3d import SEMANTIC3D_CLASSES

# A single scene per split keeps the fixture tiny (~75 KB total at 1024 points).
_DEFAULT_TRAIN_SCENES = ("bildstein_station1_xyz_intensity_rgb",)
_DEFAULT_TEST_SCENES = ("MarketplaceFeldkirch_Station4_rgb_intensity-reduced",)


def main() -> None:
    args = parse_args()
    if args.command == "raw":
        generate_raw(args)


def parse_args() -> Namespace:
    parser = ArgumentParser(description="Generate synthetic Semantic3D test data.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    raw_parser = subparsers.add_parser("raw", help="Generate raw test data")
    raw_parser.add_argument("dst_dir", type=str, help="Path to output raw directory.")
    raw_parser.add_argument("--num-points", type=int, default=1024, help="Points per scene.")
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
    dst_root = Path(args.dst_dir)
    rng = np.random.default_rng(args.seed)
    dst_root.mkdir(parents=True, exist_ok=True)

    for scene in args.train_scenes:
        _emit_scene(scene, dst_root, args.num_points, rng, with_labels=True)
    for scene in args.test_scenes:
        _emit_scene(scene, dst_root, args.num_points, rng, with_labels=False)


def _emit_scene(scene: str, dst_root: Path, num_points: int, rng: np.random.Generator, with_labels: bool) -> None:
    # Structured-but-random points in a 50 m cube.
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
