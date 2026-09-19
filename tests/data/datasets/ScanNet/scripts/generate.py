"""Generate a tiny synthetic ScanNet fixture in the layout of the original release.

Each scene is a room of triangulated planar patches, one per object, so the loader can estimate vertex
normals from the faces. A scene ships the four files of the release: the `_vh_clean_2.ply` mesh, the
`segs.json` over-segmentation (per-vertex segment id), the `aggregation.json` grouping segments into labelled
objects, and the metadata `.txt` holding the `axisAlignment` matrix. One patch belongs to no object, so every
scene holds unlabelled vertices next to the 0-based `objectId` 0. No scan of the original release is read: the
scenes are drawn from a seeded generator.

The labels file under `{version}/tasks/` is not written by this script.

Usage:
    uv run --no-sync python scripts/generate.py raw ./raw --version v2 --split train
    uv run --no-sync python scripts/generate.py process ./raw --version v2 --split train
"""

import json
import warnings
from argparse import ArgumentParser, Namespace
from pathlib import Path

import numpy as np
import plyfile

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
    parser = ArgumentParser(description="Generate synthetic ScanNet test data.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    raw_parser = subparsers.add_parser("raw", help="Generate raw test data")
    raw_parser.add_argument("dst_dir", type=str, help="Path to output raw data")
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
    dst_root = Path(args.dst_dir)
    scene_ids = {
        "train": ["scene0191_00", "scene0191_01", "scene0191_02", "scene0119_00", "scene0230_00"],
        "val": ["scene0568_00", "scene0568_01", "scene0568_02", "scene0304_00", "scene0488_00"],
        "test": ["scene0000_00", "scene0000_01", "scene0000_02", "scene0001_00", "scene0001_01"],
    }
    # Seeded per split and not per version, so v1 and v2 hold the same scenes like the real releases do.
    rng = np.random.default_rng([args.seed, list(scene_ids).index(args.split)])

    dst_split = dst_root / "metadata" / f"scannetv2_{args.split}.txt"
    dst_split.parent.mkdir(parents=True, exist_ok=True)
    with open(dst_split, "w") as f:
        f.write("\n".join(scene_ids[args.split]) + "\n")

    for scene_id in scene_ids[args.split]:
        scene_dir = dst_root / args.version / "scans" / scene_id
        scene_dir.mkdir(parents=True, exist_ok=True)
        _write_scene(scene_dir, scene_id, rng)
    print(f"  {args.version}/{args.split}: wrote {len(scene_ids[args.split])} scenes")


def _write_scene(scene_dir: Path, scene_id: str, rng: np.random.Generator) -> None:
    # `raw_category` names of the labels file; `couch` and `refrigerator` differ from their `nyu40class`.
    furniture = [
        "chair",
        "couch",
        "table",
        "door",
        "window",
        "bookshelf",
        "cabinet",
        "bed",
        "desk",
        "curtain",
        "refrigerator",
        "toilet",
        "sink",
        "bathtub",
        "picture",
        "counter",
        "shower curtain",
    ]
    labels = ["wall", "wall", "floor", *rng.choice(furniture, size=8, replace=False), None]
    size_x, size_y = rng.uniform(4.0, 8.0, size=2)

    # One planar patch per object, as an origin and two edge vectors. The last patch belongs to no object.
    patches = [
        ((0.0, 0.0, 0.0), (size_x, 0.0, 0.0), (0.0, 0.0, 2.5)),
        ((0.0, 0.0, 0.0), (0.0, size_y, 0.0), (0.0, 0.0, 2.5)),
        ((0.0, 0.0, 0.0), (size_x, 0.0, 0.0), (0.0, size_y, 0.0)),
    ]
    for _ in labels[3:]:
        corner = (*rng.uniform((0.0, 0.0), (size_x - 1.0, size_y - 1.0)), rng.uniform(0.3, 1.2))
        patches.append((corner, (rng.uniform(0.4, 1.0), 0.0, 0.0), (0.0, rng.uniform(0.4, 1.0), 0.0)))

    resolution = 8
    u, v = np.meshgrid(np.linspace(0.0, 1.0, resolution), np.linspace(0.0, 1.0, resolution), indexing="ij")
    i, j = np.meshgrid(np.arange(resolution - 1), np.arange(resolution - 1), indexing="ij")
    corners = (i * resolution + j).ravel()
    patch_faces = np.concatenate(
        [
            np.stack([corners, corners + resolution, corners + 1], axis=1),
            np.stack([corners + 1, corners + resolution, corners + resolution + 1], axis=1),
        ]
    )

    pos, color, faces, seg_indices, seg_groups = [], [], [], [], []
    segment_ids = rng.choice(30000, size=2 * len(patches), replace=False)
    for k, (label, (origin, edge_u, edge_v)) in enumerate(zip(labels, patches)):
        grid = np.asarray(origin) + u[..., None] * np.asarray(edge_u) + v[..., None] * np.asarray(edge_v)
        # Scanner-like noise: perfectly flat patches make the pretrained snapshots sensitive to kernel round-off.
        pos.append(grid.reshape(-1, 3) + rng.normal(0.0, 0.01, size=(resolution**2, 3)))
        color.append(np.clip(rng.integers(0, 256, size=3) + rng.integers(-10, 11, size=(resolution**2, 3)), 0, 255))
        faces.append(patch_faces + k * resolution**2)
        # Every patch is over-segmented into two segments.
        segments = segment_ids[2 * k : 2 * k + 2]
        seg_indices.append(np.repeat(segments, resolution**2 // 2))
        if label is not None:
            seg_groups.append({"id": k, "objectId": k, "segments": segments.tolist(), "label": str(label)})

    # The mesh is stored in the scanner frame: `axisAlignment` maps it back onto the axis-aligned room.
    theta = rng.uniform(-np.pi, np.pi)
    alignment = np.eye(4)
    alignment[:2, :2] = [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
    alignment[:3, 3] = rng.uniform(-4.0, 4.0, size=3)
    raw_pos = (np.concatenate(pos) - alignment[:3, 3]) @ alignment[:3, :3]

    vertex = np.empty(
        len(raw_pos),
        dtype=[("x", "f4"), ("y", "f4"), ("z", "f4"), ("red", "u1"), ("green", "u1"), ("blue", "u1"), ("alpha", "u1")],
    )
    vertex["x"], vertex["y"], vertex["z"] = raw_pos.T
    vertex["red"], vertex["green"], vertex["blue"] = np.concatenate(color).T
    vertex["alpha"] = 255
    face = np.empty(len(faces) * len(patch_faces), dtype=[("vertex_indices", "i4", (3,))])
    face["vertex_indices"] = np.concatenate(faces)
    plyfile.PlyData(
        [plyfile.PlyElement.describe(vertex, "vertex"), plyfile.PlyElement.describe(face, "face")],
        text=False,
    ).write(str(scene_dir / f"{scene_id}_vh_clean_2.ply"))

    segs_name = f"{scene_id}_vh_clean_2.0.010000.segs.json"
    with open(scene_dir / segs_name, "w") as f:
        json.dump({"sceneId": scene_id, "segIndices": np.concatenate(seg_indices).tolist()}, f)
    with open(scene_dir / f"{scene_id}.aggregation.json", "w") as f:
        json.dump(
            {
                "sceneId": f"scannet.{scene_id}",
                "appId": "Aggregator.v2",
                "segGroups": seg_groups,
                "segmentsFile": f"scannet.{segs_name}",
            },
            f,
        )
    with open(scene_dir / f"{scene_id}.txt", "w") as f:
        f.write(f"axisAlignment = {' '.join(f'{value:.6f}' for value in alignment.ravel())} \n")
        f.write("colorHeight = 968\ncolorWidth = 1296\nfx_color = 1170.187988\nsceneType = Misc.\n")


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
    # The tiling tests read the val split without the axis alignment (lives under `processed_noalign_20/`).
    if args.split == "val":
        _ = ScanNet20(
            root=root,
            version=args.version,
            split=args.split,
            use_axis_alignment=False,
            show_progress=True,
            force_process=True,
        )

    # v1 and v2 are processed into the same directories, and a completion marker would pin them to one version.
    for meta_path in Path(root, "ScanNet").glob(f"processed*/{args.split}/meta.json"):
        meta_path.unlink()


if __name__ == "__main__":
    main()
