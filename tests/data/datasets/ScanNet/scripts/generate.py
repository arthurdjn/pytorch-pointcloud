"""Generate a tiny ScanNet fixture by subsampling real scenes.

For each scene picked from `metadata/scannetv2_{split}.txt`, we keep `--num-points`
vertices from `{scene_id}_vh_clean_2.ply`, propagate the subsample to the
`segs.json` (per-vertex segment id), and rewrite the faces array to keep only
triangles whose three vertices were all retained. The aggregation JSON and
metadata `.txt` are copied verbatim — they reference `objectId` / segment ids
that survive subsampling.

If the official split file references scene IDs that aren't present in
`--src-dir`, the script falls back to whichever scene IDs *are* present and
writes a fixture-local split file matching the kept scenes.

The default source directory is `$TORCH_POINTCLOUD_DATA_DIR/ScanNet/raw`.

Usage:
    uv run --no-sync python scripts/generate.py raw ./raw --version v2 --split train
    uv run --no-sync python scripts/generate.py process ./raw --version v2 --split train
"""

import json
import shutil
import warnings
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import List

import numpy as np
import plyfile
from tqdm import tqdm

from torch_pointcloud.config import DATA_DIR
from torch_pointcloud.datasets import ScanNet, ScanNet20


def main() -> None:
    args = parse_args()

    if args.ignore_warnings:
        warnings.filterwarnings("ignore")

    if args.command == "raw":
        generate_raw(args)
    elif args.command == "process":
        generate_processed(args)


def parse_args() -> Namespace:
    parser = ArgumentParser(description="Generate ScanNet test data by subsampling real scenes.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    raw_parser = subparsers.add_parser("raw", help="Generate raw test data")
    raw_parser.add_argument("dst_dir", type=str, help="Path to output raw data")
    raw_parser.add_argument(
        "--src-dir",
        type=str,
        default=str(Path(DATA_DIR) / "ScanNet" / "raw"),
        help="Source ScanNet raw directory (default: $TORCH_POINTCLOUD_DATA_DIR/ScanNet/raw).",
    )
    raw_parser.add_argument("--num-points", type=int, default=1024, help="Number of vertices kept per scene.")
    raw_parser.add_argument("--num-scenes", type=int, default=5, help="Number of scenes per split/version.")
    raw_parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    raw_parser.add_argument("--version", type=str, default="v2", help="ScanNet version (v1 or v2).")
    raw_parser.add_argument("--split", type=str, default="train", help="Split (train, val, test).")
    raw_parser.add_argument("--ignore-warnings", action="store_true", help="Ignore warnings.")

    process_parser = subparsers.add_parser("process", help="Process raw data into final format")
    process_parser.add_argument("raw_dir", type=str, help="Path to raw data directory")
    process_parser.add_argument("--version", type=str, default="v2", help="ScanNet version (v1 or v2).")
    process_parser.add_argument("--split", type=str, default="train", help="Split to process.")
    process_parser.add_argument("--ignore-warnings", action="store_true", help="Ignore warnings.")
    return parser.parse_args()


def generate_raw(args: Namespace) -> None:
    rng = np.random.default_rng(args.seed)
    src_root = Path(args.src_dir)
    dst_root = Path(args.dst_dir)
    if not src_root.exists():
        raise FileNotFoundError(
            f"Source ScanNet raw directory not found: {src_root!r}. "
            f"Set --src-dir or TORCH_POINTCLOUD_DATA_DIR."
        )

    # The source ships only v2 scans; we use the same vertex data for v1 too.
    src_scans_root = src_root / "v2" / "scans"
    if not src_scans_root.exists():
        raise FileNotFoundError(f"Missing v2 scans directory: {src_scans_root!r}")

    # Copy the version-specific labels file from wherever the source keeps it.
    labels_filename = "scannetv2-labels.combined.tsv" if args.version == "v2" else "scannet-labels.combined.tsv"
    src_labels = _find_labels_file(src_root, labels_filename)
    dst_labels = dst_root / args.version / "tasks" / labels_filename
    dst_labels.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src_labels, dst_labels)

    # Decide which scenes to ship. Prefer scenes from the official split file that
    # exist locally; otherwise fall back to the first `num_scenes` scenes on disk.
    src_split = src_root / "metadata" / f"scannetv2_{args.split}.txt"
    candidate_ids: List[str] = []
    if src_split.exists():
        with open(src_split) as f:
            candidate_ids = [line.strip() for line in f if line.strip()]
    available = {p.name for p in src_scans_root.iterdir() if p.is_dir()}
    kept_ids = [sid for sid in candidate_ids if sid in available][: args.num_scenes]
    if not kept_ids:
        # Fallback: no split-listed scenes are present; pick whatever we have.
        kept_ids = sorted(available)[: args.num_scenes]

    # Write the fixture's own split file listing the kept scenes.
    dst_split = dst_root / "metadata" / f"scannetv2_{args.split}.txt"
    dst_split.parent.mkdir(parents=True, exist_ok=True)
    with open(dst_split, "w") as f:
        f.write("\n".join(kept_ids) + "\n")

    for scene_id in tqdm(kept_ids, total=len(kept_ids), desc=f"Generating {args.version}/{args.split}"):
        src_scene_dir = src_scans_root / scene_id
        dst_scene_dir = dst_root / args.version / "scans" / scene_id
        dst_scene_dir.mkdir(parents=True, exist_ok=True)

        keep_indices = _subsample_scene(src_scene_dir, dst_scene_dir, scene_id, args.num_points, rng)
        _subsample_segs(src_scene_dir, dst_scene_dir, scene_id, keep_indices)
        # Test scenes are released without aggregation/segs in the real release.
        # But we already wrote segs (vertex-to-segment) above; only skip aggregation
        # for test if it isn't present in the source.
        _copy_aggregation(src_scene_dir, dst_scene_dir, scene_id, optional=(args.split == "test"))
        _copy_metadata(src_scene_dir, dst_scene_dir, scene_id)


def _find_labels_file(src_root: Path, filename: str) -> Path:
    for candidate in (
        src_root / "v2" / "tasks" / filename,
        src_root / "metadata" / filename,
        src_root / "v1" / "tasks" / filename,
    ):
        if candidate.exists():
            return candidate
    # Cross-version fallback: ship the v2 file named as v1.
    cross = src_root / "v2" / "tasks" / "scannetv2-labels.combined.tsv"
    if cross.exists():
        return cross
    raise FileNotFoundError(f"Could not find labels file {filename!r} under {src_root!r}.")


def _subsample_scene(
    src_scene_dir: Path,
    dst_scene_dir: Path,
    scene_id: str,
    num_points: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Subsample the `_vh_clean_2.ply` mesh, returning the kept vertex indices.

    We sample triangles first and keep their unique vertices, which preserves the
    mesh connectivity needed for `vertex_normals` (random vertex sampling would
    leave most faces with at least one dropped corner and produce an empty face
    array).
    """
    src_ply = src_scene_dir / f"{scene_id}_vh_clean_2.ply"
    if not src_ply.exists():
        raise FileNotFoundError(f"Missing PLY: {src_ply!r}")

    with open(src_ply, "rb") as f:
        plydata = plyfile.PlyData.read(f)
    src_vertex = plydata["vertex"].data
    src_face = plydata["face"].data

    n_vertices = src_vertex.shape[0]
    src_face_indices = np.stack([np.asarray(face, dtype=np.int64) for face in src_face["vertex_indices"]], axis=0)
    n_faces = src_face_indices.shape[0]

    # Pick enough faces so the unique-vertex set has roughly `num_points` entries.
    # `num_points * 2 // 3` is an empirical heuristic (≈3 verts/face minus shared).
    target_faces = min(n_faces, max(1, num_points * 2 // 3))
    face_indices = rng.choice(n_faces, size=target_faces, replace=False)
    face_indices.sort()
    selected_faces = src_face_indices[face_indices]

    keep_indices = np.unique(selected_faces.ravel())
    if keep_indices.size > num_points:
        keep_indices = rng.choice(keep_indices, size=num_points, replace=False)
        keep_indices.sort()
        # Filter faces again so all three corners survive.
        old_to_new_partial = np.full(n_vertices, -1, dtype=np.int64)
        old_to_new_partial[keep_indices] = np.arange(keep_indices.size, dtype=np.int64)
        valid = (old_to_new_partial[selected_faces] >= 0).all(axis=1)
        selected_faces = selected_faces[valid]

    new_vertex = src_vertex[keep_indices]
    old_to_new = np.full(n_vertices, -1, dtype=np.int64)
    old_to_new[keep_indices] = np.arange(keep_indices.size, dtype=np.int64)
    remapped = old_to_new[selected_faces]
    if remapped.size == 0:
        # Pathological case: keep at least one degenerate face referencing the
        # first three vertices so downstream readers don't choke on an empty array.
        remapped = np.array([[0, min(1, keep_indices.size - 1), min(2, keep_indices.size - 1)]], dtype=np.int64)

    new_face_data = np.empty(remapped.shape[0], dtype=src_face.dtype)
    new_face_data["vertex_indices"] = [tuple(row) for row in remapped]

    vertex_element = plyfile.PlyElement.describe(new_vertex, "vertex")
    face_element = plyfile.PlyElement.describe(new_face_data, "face")

    dst_ply = dst_scene_dir / f"{scene_id}_vh_clean_2.ply"
    plyfile.PlyData([vertex_element, face_element], text=False).write(str(dst_ply))
    return keep_indices


def _subsample_segs(
    src_scene_dir: Path,
    dst_scene_dir: Path,
    scene_id: str,
    keep_indices: np.ndarray,
) -> None:
    src_segs = src_scene_dir / f"{scene_id}_vh_clean_2.0.010000.segs.json"
    if not src_segs.exists():
        return
    with open(src_segs, "r") as f:
        segs = json.load(f)

    src_seg_indices = np.asarray(segs["segIndices"], dtype=np.int64)
    new_seg_indices = src_seg_indices[keep_indices].tolist()
    segs["segIndices"] = new_seg_indices

    dst_segs = dst_scene_dir / f"{scene_id}_vh_clean_2.0.010000.segs.json"
    with open(dst_segs, "w") as f:
        json.dump(segs, f)


def _copy_aggregation(src_scene_dir: Path, dst_scene_dir: Path, scene_id: str, optional: bool = False) -> None:
    src_agg = src_scene_dir / f"{scene_id}.aggregation.json"
    if not src_agg.exists():
        # Some scenes ship the aggregation under `_vh_clean.aggregation.json`. Either is fine.
        alt = src_scene_dir / f"{scene_id}_vh_clean.aggregation.json"
        if alt.exists():
            src_agg = alt
        elif optional:
            return
        else:
            raise FileNotFoundError(f"Missing aggregation file for scene {scene_id!r}: {src_agg!r}")
    dst_agg = dst_scene_dir / f"{scene_id}.aggregation.json"
    shutil.copyfile(src_agg, dst_agg)


def _copy_metadata(src_scene_dir: Path, dst_scene_dir: Path, scene_id: str) -> None:
    src_meta = src_scene_dir / f"{scene_id}.txt"
    dst_meta = dst_scene_dir / f"{scene_id}.txt"
    shutil.copyfile(src_meta, dst_meta)


def generate_processed(args: Namespace) -> None:
    raw_dir = Path(args.raw_dir)
    root = raw_dir.resolve().parent.parent.as_posix()

    _ = ScanNet(
        root=root,
        version=args.version,
        split=args.split,
        show_progress=True,
        force_process=True,
    )
    # Also build the ScanNet20 cache (lives under `processed_20/`) so pretrained
    # tests using `ScanNet20` find the data without re-running raw subsampling.
    _ = ScanNet20(
        root=root,
        version=args.version,
        split=args.split,
        show_progress=True,
        force_process=True,
    )


if __name__ == "__main__":
    main()
