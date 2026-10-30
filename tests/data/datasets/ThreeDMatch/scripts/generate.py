"""Generate a tiny synthetic 3DMatch fixture in the layout of the PREDATOR release and the GeoTransformer pair lists.

    raw/indoor/test/<scene>/cloud_bin_{i}.pth          # legacy torch pickle of a float64 (N, 3) array
    raw/indoor/train/<scene>/cloud_bin_{i}.pth
    raw/metadata/{train,val}.pkl                        # list of pair records (rotation, translation, overlap, ...)
    raw/metadata/benchmarks/{3DMatch,3DLoMatch}/<scene>/{gt.log,gt.info,gt_overlap.log}
    processed/                                          # the test-split cache `ThreeDMatch` builds from `raw/`

No file of the real release is read. Every scene is a seeded room (a floor and two walls) and every fragment a
rigidly moved, cropped copy of it, so the written transforms align the fragments exactly.

Usage:
    uv run --no-sync python scripts/generate.py raw ./raw
    uv run --no-sync python scripts/generate.py process ./raw
"""

import pickle
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from torch_pointcloud.datasets import ThreeDMatch

TEST_SCENES = ("7-scenes-redkitchen", "sun3d-hotel_umd-maryland_hotel3")
TRAIN_SCENES = ("7-scenes-chess",)
NUM_FRAGMENTS = 3


def main() -> None:
    args = parse_args()
    if args.command == "raw":
        generate_raw(args)
    elif args.command == "process":
        generate_processed(args)


def parse_args() -> Namespace:
    parser = ArgumentParser(description="Generate synthetic 3DMatch test data.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    raw_parser = subparsers.add_parser("raw", help="Generate raw test data")
    raw_parser.add_argument("dst_dir", type=str, help="Path to output raw directory.")
    raw_parser.add_argument("--num-points", type=int, default=1200, help="Points sampled per surface of a scene.")
    raw_parser.add_argument("--seed", type=int, default=42, help="Random seed.")

    process_parser = subparsers.add_parser("process", help="Build the processed test-split cache from raw data")
    process_parser.add_argument("raw_dir", type=str, help="Path to the raw directory.")
    return parser.parse_args()


def room(rng: np.random.Generator, num_points: int) -> np.ndarray:
    """A 3 m room: a floor and two walls, sampled on a 2.5 cm grid like the voxelized release."""
    floor = np.stack([rng.uniform(0, 3, num_points), rng.uniform(0, 3, num_points), np.zeros(num_points)], axis=1)
    wall_x = np.stack([np.zeros(num_points), rng.uniform(0, 3, num_points), rng.uniform(0, 2.5, num_points)], axis=1)
    wall_y = np.stack([rng.uniform(0, 3, num_points), np.zeros(num_points), rng.uniform(0, 2.5, num_points)], axis=1)
    points = np.concatenate([floor, wall_x, wall_y])
    return np.round(points / 0.025) * 0.025


def fragments(rng: np.random.Generator, num_points: int) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Fragments of one scene, each a crop of the room in its own frame, with the pose of every fragment.

    The crops are subsets of one sampling of the room, so the overlapping points of two fragments coincide exactly
    once aligned.
    """
    scene = room(rng, num_points)
    scene = scene[rng.permutation(len(scene))[: 2 * num_points]]
    clouds, poses = [], []
    for i in range(NUM_FRAGMENTS):
        keep = scene[:, 0] + scene[:, 1] > i * 0.6  # overlapping crops along the diagonal
        crop = scene[keep]
        pose = np.eye(4)
        pose[:3, :3] = Rotation.from_euler("zyx", rng.uniform(-0.3, 0.3, 3)).as_matrix()
        pose[:3, 3] = rng.uniform(-0.5, 0.5, 3)
        clouds.append((crop - pose[:3, 3]) @ pose[:3, :3])  # the fragment in its own frame: pose @ local = world
        poses.append(pose)
    return clouds, poses


def pair_transform(poses: List[np.ndarray], ref: int, src: int) -> np.ndarray:
    """Transform of fragment `src` onto fragment `ref`: `ref_local = T @ src_local`."""
    return np.linalg.inv(poses[ref]) @ poses[src]


def write_fragments(dst: Path, clouds: List[np.ndarray], poses: List[np.ndarray]) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for i, (cloud, pose) in enumerate(zip(clouds, poses)):
        torch.save(cloud.astype(np.float64), dst / f"cloud_bin_{i}.pth", _use_new_zipfile_serialization=False)
        rows = "\n".join("\t".join(f"{v: .8e}" for v in row) for row in pose)
        (dst / f"cloud_bin_{i}.info.txt").write_text(f"{dst.name}\t seq-01\t {i}\t {i + 49}\t\n{rows}\n")


def write_benchmark(dst: Path, poses: List[np.ndarray], overlaps: Dict[Tuple[int, int], float]) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    log, info, overlap_log = [], [], []
    for (ref, src), overlap in overlaps.items():
        transform = pair_transform(poses, ref, src)
        log.append(f"{ref}\t {src}\t {NUM_FRAGMENTS}\t")
        log += ["\t".join(f"{v: .8e}" for v in row) + "\t" for row in transform]
        information = np.diag([5000.0, 5000.0, 5000.0, 2000.0, 2000.0, 2000.0])
        info.append(f"{ref}\t {src}\t {NUM_FRAGMENTS}\t")
        info += ["\t".join(f"{v: .8e}" for v in row) + "\t" for row in information]
        overlap_log.append(f"{ref},{src},{overlap:.4f}")
    (dst / "gt.log").write_text("\n".join(log) + "\n")
    (dst / "gt.info").write_text("\n".join(info) + "\n")
    (dst / "gt_overlap.log").write_text("\n".join(overlap_log) + "\n")


def generate_raw(args: Namespace) -> None:
    rng = np.random.default_rng(args.seed)
    dst = Path(args.dst_dir)
    metadata = dst / "metadata"

    for scene in TEST_SCENES:
        clouds, poses = fragments(rng, args.num_points)
        write_fragments(dst / "indoor" / "test" / scene, clouds, poses)
        # 3DMatch pairs overlap above 30%, 3DLoMatch pairs between 10% and 30%.
        write_benchmark(metadata / "benchmarks" / "3DMatch" / scene, poses, {(0, 1): 0.65, (1, 2): 0.6, (0, 2): 0.35})
        write_benchmark(metadata / "benchmarks" / "3DLoMatch" / scene, poses, {(0, 2): 0.2})

    scene_poses = {}
    for scene in TRAIN_SCENES:
        clouds, poses = fragments(rng, args.num_points)
        write_fragments(dst / "indoor" / "train" / scene, clouds, poses)
        scene_poses[scene] = poses

    # The val pairs are the PREDATOR validation scenes; the fixture reuses the train scene with fewer pairs.
    for split, pairs in (("train", ((0, 1), (1, 2), (0, 2))), ("val", ((0, 1),))):
        records = []
        for scene, poses in scene_poses.items():
            for ref, src in pairs:
                transform = pair_transform(poses, ref, src)
                records.append(
                    {
                        "overlap": np.float64(0.5),
                        "pcd0": f"train/{scene}/cloud_bin_{ref}.pth",
                        "pcd1": f"train/{scene}/cloud_bin_{src}.pth",
                        "rotation": transform[:3, :3],
                        "translation": transform[:3, 3],
                        "scene_name": scene,
                        "frag_id0": ref,
                        "frag_id1": src,
                    }
                )
        metadata.mkdir(parents=True, exist_ok=True)
        with open(metadata / f"{split}.pkl", "wb") as f:
            pickle.dump(records, f)


def generate_processed(args: Namespace) -> None:
    raw_dir = Path(args.raw_dir).resolve()
    dataset = ThreeDMatch(root=raw_dir.parent.parent, split="test", show_progress=False, force_process=True)
    # The committed cache stays without metadata: the tests read it as a cache from before metadata existed.
    for meta_path in Path(dataset.processed_dir).glob("*.meta.json"):
        meta_path.unlink()


if __name__ == "__main__":
    main()
