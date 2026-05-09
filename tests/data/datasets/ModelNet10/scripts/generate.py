"""Generate a tiny ModelNet (10 or 40) fixture by subsampling real `.off` meshes.

For each class, the script keeps `--max-objects` train and `--max-objects` test
files, subsampling each mesh to roughly `--max-points` vertices while preserving
face connectivity (vertex normals need triangles).

The default source directory is `$TORCH_POINTCLOUD_DATA_DIR/ModelNet{variant}/raw`
where `{variant}` is 10 or 40 depending on `--variant`.

Usage:
    uv run --no-sync python scripts/generate.py raw ./raw --variant 40
    uv run --no-sync python scripts/generate.py process ./raw --variant 40
"""

from argparse import ArgumentParser, Namespace
from pathlib import Path

import numpy as np

from torch_pointcloud.config import DATA_DIR
from torch_pointcloud.datasets import ModelNet10, ModelNet40
from torch_pointcloud.utils.io import load_off


def main() -> None:
    args = parse_args()
    if args.command == "raw":
        generate_raw(args)
    elif args.command == "process":
        generate_processed(args)


def parse_args() -> Namespace:
    parser = ArgumentParser(description="Generate ModelNet test data by subsampling real .off meshes.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    raw_parser = subparsers.add_parser("raw", help="Generate raw test data")
    raw_parser.add_argument("dst_dir", type=str, help="Path to output raw data")
    raw_parser.add_argument(
        "--variant",
        type=str,
        choices=["10", "40"],
        default="40",
        help="Which ModelNet variant to subsample (10 or 40).",
    )
    raw_parser.add_argument(
        "--src-dir",
        type=str,
        default=None,
        help="Source ModelNet raw directory (default: $TORCH_POINTCLOUD_DATA_DIR/ModelNet{variant}/raw).",
    )
    raw_parser.add_argument("--max-points", type=int, default=1024, help="Vertices kept per mesh after subsampling.")
    raw_parser.add_argument("--max-objects", type=int, default=2, help="Files kept per class per split.")
    raw_parser.add_argument("--seed", type=int, default=42, help="Random seed.")

    process_parser = subparsers.add_parser("process", help="Process raw data into final format")
    process_parser.add_argument("raw_dir", type=str, help="Path to raw data directory")
    process_parser.add_argument("--variant", type=str, choices=["10", "40"], default="40")

    return parser.parse_args()


def _resolve_src_dir(args: Namespace) -> Path:
    if args.src_dir:
        return Path(args.src_dir)
    return Path(DATA_DIR) / f"ModelNet{args.variant}" / "raw"


def _subsample_off(src_path: Path, dst_path: Path, max_points: int, rng: np.random.Generator) -> None:
    """Subsample an OFF file's vertices while keeping a connected face set."""
    pos, face = load_off(src_path)
    n_vertices = pos.shape[0]
    n_faces = face.shape[0]
    if n_vertices == 0 or n_faces == 0:
        raise ValueError(f"Empty mesh: {src_path!r}")

    target_faces = min(n_faces, max(1, max_points * 2 // 3))
    face_indices = rng.choice(n_faces, size=target_faces, replace=False)
    face_indices.sort()
    selected_faces = face[face_indices]

    keep_indices = np.unique(selected_faces.ravel())
    if keep_indices.size > max_points:
        keep_indices = rng.choice(keep_indices, size=max_points, replace=False)
        keep_indices.sort()
        old_to_new_partial = np.full(n_vertices, -1, dtype=np.int64)
        old_to_new_partial[keep_indices] = np.arange(keep_indices.size, dtype=np.int64)
        valid = (old_to_new_partial[selected_faces] >= 0).all(axis=1)
        selected_faces = selected_faces[valid]

    new_pos = pos[keep_indices]
    old_to_new = np.full(n_vertices, -1, dtype=np.int64)
    old_to_new[keep_indices] = np.arange(keep_indices.size, dtype=np.int64)
    remapped = old_to_new[selected_faces]
    if remapped.size == 0:
        remapped = np.array(
            [[0, min(1, keep_indices.size - 1), min(2, keep_indices.size - 1)]],
            dtype=np.int64,
        )

    dst_path.parent.mkdir(parents=True, exist_ok=True)
    with open(dst_path, "w") as f:
        f.write("OFF\n")
        f.write(f"{new_pos.shape[0]} {remapped.shape[0]} 0\n")
        for v in new_pos:
            f.write(f"{v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for tri in remapped:
            f.write(f"3 {int(tri[0])} {int(tri[1])} {int(tri[2])}\n")


def generate_raw(args: Namespace) -> None:
    src_root = _resolve_src_dir(args)
    if not src_root.exists():
        raise FileNotFoundError(
            f"Source ModelNet{args.variant} raw directory not found: {src_root!r}. "
            f"Set --src-dir or TORCH_POINTCLOUD_DATA_DIR."
        )
    dst_root = Path(args.dst_dir)
    rng = np.random.default_rng(args.seed)

    classes = sorted(p.name for p in src_root.iterdir() if p.is_dir())
    for cls in classes:
        for split in ("train", "test"):
            src_split_dir = src_root / cls / split
            if not src_split_dir.exists():
                continue
            off_paths = sorted(src_split_dir.glob("*.off"))[: args.max_objects]
            for src_off in off_paths:
                dst_off = dst_root / cls / split / src_off.name
                _subsample_off(src_off, dst_off, args.max_points, rng)
        print(f"  {cls}: kept up to {args.max_objects} files/split, ~{args.max_points} vertices each")


def generate_processed(args: Namespace) -> None:
    raw_dir = Path(args.raw_dir)
    root = raw_dir.resolve().parent.parent.as_posix()

    cls = ModelNet10 if args.variant == "10" else ModelNet40
    for train in (True, False):
        _ = cls(root=root, train=train, show_progress=True, force_process=True)


if __name__ == "__main__":
    main()
