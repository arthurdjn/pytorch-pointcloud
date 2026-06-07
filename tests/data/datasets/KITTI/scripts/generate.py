"""Generate a tiny KITTI fixture by subsampling real frames from the object split.

The output mirrors the real KITTI object layout under `raw/`:

    raw/training/velodyne/{frame}.bin   # float32 (N, 4) = (x, y, z, intensity)
    raw/training/calib/{frame}.txt      # P0-P3, R0_rect, Tr_velo_to_cam, Tr_imu_to_velo (verbatim)
    raw/training/label_2/{frame}.txt    # type trunc occ alpha bbox(4) dims(h,w,l) loc(x,y,z) ry (verbatim)

The calibration and labels are copied verbatim from the real release; only the point cloud is
subsampled. The first frames that contain at least one detection-class object are kept, so each
shipped frame yields a non-empty box set. `image_2/` is intentionally not shipped (the real PNGs are
~800 KB each); the FOV filter is covered by the unit tests instead.

The default source directory is `$TORCH_POINTCLOUD_DATA_DIR/KITTI/raw`.

Usage:
    uv run --no-sync python scripts/generate.py
    uv run --no-sync python scripts/generate.py --src-dir /path/to/KITTI
"""

import shutil
from argparse import ArgumentParser, Namespace
from pathlib import Path

import numpy as np

from torch_pointcloud.config import DATA_DIR
from torch_pointcloud.datasets.kitti import KITTI_DETECTION_CLASSES


def _has_foreground(label_file: Path) -> bool:
    return any(line.split(" ")[0] in KITTI_DETECTION_CLASSES for line in label_file.read_text().splitlines())


def _select_frames(src_split: Path, num_frames: int) -> list[str]:
    frames: list[str] = []
    for bin_path in sorted((src_split / "velodyne").glob("*.bin")):
        if _has_foreground(src_split / "label_2" / f"{bin_path.stem}.txt"):
            frames.append(bin_path.stem)
        if len(frames) == num_frames:
            break
    return frames


def generate(args: Namespace) -> None:
    rng = np.random.default_rng(args.seed)
    src_split = Path(args.src_dir) / args.split
    dst_split = Path(args.dst_dir) / "raw" / args.split
    if not (src_split / "velodyne").is_dir():
        raise FileNotFoundError(f"Source KITTI split not found: {src_split / 'velodyne'!r}. Set --src-dir.")

    for sub in ("velodyne", "calib", "label_2"):
        (dst_split / sub).mkdir(parents=True, exist_ok=True)

    frames = _select_frames(src_split, args.num_frames)
    if not frames:
        raise RuntimeError(f"No frames with a detection-class object found under {src_split!r}.")

    for frame in frames:
        scan = np.fromfile(src_split / "velodyne" / f"{frame}.bin", dtype=np.float32).reshape(-1, 4)
        num_keep = min(args.num_points, scan.shape[0])
        indices = rng.choice(scan.shape[0], size=num_keep, replace=False)
        indices.sort()
        scan[indices].astype(np.float32).tofile(dst_split / "velodyne" / f"{frame}.bin")
        shutil.copy(src_split / "calib" / f"{frame}.txt", dst_split / "calib" / f"{frame}.txt")
        shutil.copy(src_split / "label_2" / f"{frame}.txt", dst_split / "label_2" / f"{frame}.txt")

    print(f"generated {len(frames)} frames ({args.num_points} pts each) into {dst_split}: {frames}")


def parse_args() -> Namespace:
    parser = ArgumentParser(description="Generate a tiny KITTI test fixture by subsampling real frames.")
    parser.add_argument(
        "dst_dir",
        type=str,
        nargs="?",
        default=str(Path(__file__).resolve().parent.parent),
        help="Output KITTI fixture directory (default: the fixture dir next to this script).",
    )
    parser.add_argument(
        "--src-dir",
        type=str,
        default=str(Path(DATA_DIR) / "KITTI" / "raw"),
        help="Source KITTI directory holding the split (default: $TORCH_POINTCLOUD_DATA_DIR/KITTI/raw).",
    )
    parser.add_argument("--split", type=str, default="training", help="KITTI object split to read.")
    parser.add_argument("--num-frames", type=int, default=3, help="Number of frames to keep.")
    parser.add_argument("--num-points", type=int, default=1024, help="Points per frame after subsampling.")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed.")
    return parser.parse_args()


def main() -> None:
    generate(parse_args())


if __name__ == "__main__":
    main()
