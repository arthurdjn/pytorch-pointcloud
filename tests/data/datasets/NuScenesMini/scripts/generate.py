"""Generate a tiny nuScenes-mini fixture by subsampling real keyframes.

The output mirrors the extracted `v1.0-mini`:

    raw/v1.0-mini/*.json              # ego_pose, calibrated_sensor, category, instance,
                                      # sample_annotation, sample_data (consistent subset, verbatim)
    raw/samples/LIDAR_TOP/*.pcd.bin   # float32 (N, 5) = (x, y, z, intensity, ring) keyframes
    raw/sweeps/LIDAR_TOP/*.pcd.bin    # float32 (N, 5) prior sweeps

The metadata tables the loader reads are subset to a consistent slice: a few LIDAR keyframes, the
prior sweeps reachable along their `prev` chains, and the ego poses, sensor calibration, annotations,
instances and categories those records reference. Only the point clouds are subsampled; every record
is copied verbatim. Keyframes are chosen with the fewest annotations (so the fixture stays tiny)
among those with a full enough sweep chain and at least one detection-class object.

The default source directory is `$TORCH_POINTCLOUD_DATA_DIR/NuScenesMini`.

Usage:
    uv run --no-sync python scripts/generate.py
    uv run --no-sync python scripts/generate.py --src-dir /path/to/NuScenesMini
"""

import json
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from torch_pointcloud.config import DATA_DIR
from torch_pointcloud.datasets.nuscenes import _CATEGORY_TO_DETECTION


def _load(meta_dir: Path, name: str) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = json.loads((meta_dir / f"{name}.json").read_text())
    return records


def _chain(keyframe: Dict[str, Any], sd_by_token: Dict[str, Dict[str, Any]], sweeps: int) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    current: Dict[str, Any] | None = keyframe
    while current is not None and len(records) < sweeps:
        records.append(current)
        prev = current["prev"]
        current = sd_by_token.get(prev) if prev else None
    return records


def generate(args: Namespace) -> None:
    rng = np.random.default_rng(args.seed)
    src_raw = Path(args.src_dir) / "raw"
    src_meta = src_raw / args.version
    if not src_meta.is_dir():
        raise FileNotFoundError(f"Source nuScenes metadata not found: {src_meta!r}. Set --src-dir.")

    sample_data = _load(src_meta, "sample_data")
    sd_by_token = {r["token"]: r for r in sample_data}
    ego_by_token = {r["token"]: r for r in _load(src_meta, "ego_pose")}
    cal_by_token = {r["token"]: r for r in _load(src_meta, "calibrated_sensor")}
    annotations = _load(src_meta, "sample_annotation")
    inst_by_token = {r["token"]: r for r in _load(src_meta, "instance")}
    cat_by_token = {r["token"]: r for r in _load(src_meta, "category")}

    anns_by_sample: Dict[str, List[Dict[str, Any]]] = {}
    for ann in annotations:
        anns_by_sample.setdefault(ann["sample_token"], []).append(ann)

    def detection_count(sample_token: str) -> int:
        names = (
            cat_by_token[inst_by_token[a["instance_token"]]["category_token"]]["name"]
            for a in anns_by_sample.get(sample_token, [])
        )
        return sum(name in _CATEGORY_TO_DETECTION for name in names)

    keyframes = [r for r in sample_data if r["is_key_frame"] and "LIDAR_TOP" in r["filename"]]
    candidates = [
        kf
        for kf in keyframes
        if len(_chain(kf, sd_by_token, args.sweeps)) == args.sweeps and detection_count(kf["sample_token"]) >= 1
    ]
    candidates.sort(key=lambda kf: (len(anns_by_sample.get(kf["sample_token"], [])), kf["token"]))
    chosen = candidates[: args.num_keyframes]
    if not chosen:
        raise RuntimeError("No keyframe with a full sweep chain and a detection-class object was found.")

    kept_sd: Dict[str, Dict[str, Any]] = {}
    for kf in chosen:
        for record in _chain(kf, sd_by_token, args.sweeps):
            kept_sd[record["token"]] = record

    kept_ego = {r["ego_pose_token"] for r in kept_sd.values()}
    kept_cal = {r["calibrated_sensor_token"] for r in kept_sd.values()}
    kept_files = {r["filename"] for r in kept_sd.values()}
    sample_tokens = {kf["sample_token"] for kf in chosen}
    kept_anns = [a for a in annotations if a["sample_token"] in sample_tokens]
    kept_inst = {a["instance_token"] for a in kept_anns}
    kept_cat = {inst_by_token[t]["category_token"] for t in kept_inst}

    dst_raw = Path(args.dst_dir) / "raw"
    dst_meta = dst_raw / args.version
    dst_meta.mkdir(parents=True, exist_ok=True)
    tables = {
        "ego_pose": [ego_by_token[t] for t in sorted(kept_ego)],
        "calibrated_sensor": [cal_by_token[t] for t in sorted(kept_cal)],
        "category": [cat_by_token[t] for t in sorted(kept_cat)],
        "instance": [inst_by_token[t] for t in sorted(kept_inst)],
        "sample_annotation": kept_anns,
        "sample_data": list(kept_sd.values()),
    }
    for name, records in tables.items():
        (dst_meta / f"{name}.json").write_text(json.dumps(records, indent=2))

    for filename in sorted(kept_files):
        scan = np.fromfile(src_raw / filename, dtype=np.float32).reshape(-1, 5)
        num_keep = min(args.num_points, scan.shape[0])
        indices = rng.choice(scan.shape[0], size=num_keep, replace=False)
        indices.sort()
        out_path = dst_raw / filename
        out_path.parent.mkdir(parents=True, exist_ok=True)
        scan[indices].astype(np.float32).tofile(out_path)

    print(f"generated {len(chosen)} keyframes, {len(kept_files)} scans ({args.num_points} pts each) into {dst_raw}")


def parse_args() -> Namespace:
    parser = ArgumentParser(description="Generate a tiny nuScenes-mini test fixture by subsampling real keyframes.")
    parser.add_argument(
        "dst_dir",
        type=str,
        nargs="?",
        default=str(Path(__file__).resolve().parent.parent),
        help="Output NuScenesMini fixture directory (default: the fixture dir next to this script).",
    )
    parser.add_argument(
        "--src-dir",
        type=str,
        default=str(Path(DATA_DIR) / "NuScenesMini"),
        help="Source NuScenesMini directory holding raw/ (default: $TORCH_POINTCLOUD_DATA_DIR/NuScenesMini).",
    )
    parser.add_argument("--version", type=str, default="v1.0-mini", help="Metadata version directory.")
    parser.add_argument("--num-keyframes", type=int, default=2, help="Number of LIDAR keyframes to keep.")
    parser.add_argument("--sweeps", type=int, default=3, help="LIDAR clouds per keyframe (keyframe + prior sweeps).")
    parser.add_argument("--num-points", type=int, default=1024, help="Points per scan after subsampling.")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed.")
    return parser.parse_args()


def main() -> None:
    generate(parse_args())


if __name__ == "__main__":
    main()
