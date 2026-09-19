"""Generate a tiny synthetic nuScenes-mini fixture in the layout of the `v1.0-mini` release.

The output mirrors the extracted `v1.0-mini` layout under `raw/`:

    raw/v1.0-mini/*.json              # ego_pose, calibrated_sensor, category, instance, attribute,
                                      # sample, sample_annotation, sample_data, scene
    raw/samples/LIDAR_TOP/*.pcd.bin   # float32 (N, 5) = (x, y, z, intensity, ring) keyframes
    raw/sweeps/LIDAR_TOP/*.pcd.bin    # float32 (N, 5) prior sweeps

No file of the real release is read. One scene named after a `mini_val` scene holds `--num-keyframes`
keyframes, each preceded by its prior sweeps along one `prev` chain. The ego vehicle drives straight past a
moving car, a standing pedestrian, a barrier (a class without attribute) and an animal (a category outside
the detection classes). Every sweep is cast from a seeded generator (one ray per (beam, azimuth) pair of a
spinning sensor) against a road, a facade and the annotated objects, so `num_lidar_pts` counts real hits.

Usage:
    uv run --no-sync python scripts/generate.py
"""

import json
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from torch_pointcloud.datasets.nuscenes import NUSCENES_ATTRIBUTES, _pose_matrix


def generate(args: Namespace) -> None:
    rng = np.random.default_rng(args.seed)
    dst_raw = Path(args.dst_dir) / "raw"
    dst_meta = dst_raw / args.version
    dst_meta.mkdir(parents=True, exist_ok=True)

    def token() -> str:
        return rng.bytes(16).hex()

    attributes = [{"token": token(), "name": name, "description": ""} for name in NUSCENES_ATTRIBUTES]
    attribute_token = {record["name"]: record["token"] for record in attributes}
    calibrated_sensor: Dict[str, Any] = {
        "token": token(),
        "sensor_token": token(),
        "translation": [0.94, 0.0, 1.84],
        "rotation": [float(np.sqrt(0.5)), 0.0, 0.0, -float(np.sqrt(0.5))],
        "camera_intrinsic": [],
    }
    scene: Dict[str, Any] = {"token": token(), "log_token": "", "name": "scene-0103", "description": ""}

    # Category, attribute, size (width, length, height), start position and velocity in the global frame.
    objects = [
        ("vehicle.car", "vehicle.moving", (1.8, 4.5, 1.6), (1012.0, 2003.0), (4.0, 0.0)),
        ("human.pedestrian.adult", "pedestrian.standing", (0.6, 0.7, 1.8), (1008.0, 1996.0), (0.0, 0.0)),
        ("movable_object.barrier", None, (2.5, 0.5, 1.0), (1015.0, 2002.0), (0.0, 0.0)),
        ("animal", None, (0.4, 0.8, 0.5), (1010.0, 2005.0), (0.0, 0.0)),
    ]
    categories = [{"token": token(), "name": name, "description": ""} for name, *_ in objects]
    instances: List[Dict[str, Any]] = [
        {"token": token(), "category_token": category["token"], "nbr_annotations": args.num_keyframes}
        for category in categories
    ]

    num_beams = 16
    azimuth, elevation = np.meshgrid(
        np.linspace(-np.pi, np.pi, args.num_points // num_beams, endpoint=False),
        np.deg2rad(np.linspace(-24.8, -0.5, num_beams)),
    )
    ring = np.repeat(np.arange(num_beams), args.num_points // num_beams)
    azimuth, elevation = azimuth.ravel(), elevation.ravel()
    facade = rng.uniform(30.0, 60.0)

    samples: List[Dict[str, Any]] = []
    sample_data: List[Dict[str, Any]] = []
    ego_poses: List[Dict[str, Any]] = []
    annotations: List[Dict[str, Any]] = []
    for k in range(args.num_keyframes):
        sample = {"token": token(), "timestamp": 0, "prev": "", "next": "", "scene_token": scene["token"]}
        if samples:
            sample["prev"], samples[-1]["next"] = samples[-1]["token"], sample["token"]
        samples.append(sample)

        # The keyframe closes a chain of prior sweeps recorded every 50 ms; keyframes are 500 ms apart.
        for s in reversed(range(args.sweeps)):
            is_key_frame = s == 0
            elapsed = 0.5 * k - 0.05 * s
            timestamp = 1_600_000_000_000_000 + round(elapsed * 1e6)
            ego_translation = [1000.0 + 5.0 * elapsed, 2000.0, 0.0]
            ego_rotation = [1.0, 0.0, 0.0, 0.0]
            ego_pose = {"token": token(), "timestamp": timestamp, "rotation": ego_rotation}
            ego_pose["translation"] = ego_translation
            ego_poses.append(ego_pose)
            sensor_from_global = np.linalg.inv(
                _pose_matrix(ego_translation, ego_rotation)
                @ _pose_matrix(calibrated_sensor["translation"], calibrated_sensor["rotation"])
            )

            # Horizontal distance at which each ray stops: on the road 1.84 m below the sensor, on a facade, or
            # on an object when the ray crosses its azimuth sector below its top.
            stops = [1.84 / np.tan(-elevation), np.full_like(elevation, facade)]
            for _, _, (width, length, height), start, velocity in objects:
                center = np.asarray(start) + np.asarray(velocity) * elapsed
                center = (sensor_from_global @ [*center, 0.5 * height, 1.0])[:3]
                distance = np.hypot(center[0], center[1])
                offset = np.angle(np.exp(1j * (azimuth - np.arctan2(center[1], center[0]))))
                crosses = np.abs(offset) < np.arctan(0.5 * max(width, length) / distance)
                crosses &= distance * np.tan(elevation) < center[2] + 0.5 * height
                stops.append(np.where(crosses, distance, np.inf))
            hit = np.argmin(stops, axis=0)
            distance = np.min(stops, axis=0)

            pos = np.stack([distance * np.cos(azimuth), distance * np.sin(azimuth), distance * np.tan(elevation)], 1)
            pos += rng.normal(0.0, 0.02, size=pos.shape)
            scan = np.concatenate([pos, rng.uniform(0.0, 255.0, size=(len(pos), 1)), ring[:, None]], axis=1)
            filename = f"{'samples' if is_key_frame else 'sweeps'}/LIDAR_TOP/synthetic__LIDAR_TOP__{timestamp}.pcd.bin"
            (dst_raw / filename).parent.mkdir(parents=True, exist_ok=True)
            scan.astype(np.float32).tofile(dst_raw / filename)

            record = {
                "token": ego_pose["token"],
                "sample_token": sample["token"],
                "ego_pose_token": ego_pose["token"],
                "calibrated_sensor_token": calibrated_sensor["token"],
                "timestamp": timestamp,
                "fileformat": "pcd",
                "is_key_frame": is_key_frame,
                "height": 0,
                "width": 0,
                "filename": filename,
                "prev": "",
                "next": "",
            }
            if sample_data:
                record["prev"], sample_data[-1]["next"] = sample_data[-1]["token"], record["token"]
            sample_data.append(record)

            if not is_key_frame:
                continue
            sample["timestamp"] = timestamp
            for i, (_, attribute, size, start, velocity) in enumerate(objects):
                center = np.asarray(start) + np.asarray(velocity) * elapsed
                annotation = {
                    "token": token(),
                    "sample_token": sample["token"],
                    "instance_token": instances[i]["token"],
                    "visibility_token": "4",
                    "attribute_tokens": [attribute_token[attribute]] if attribute else [],
                    "translation": [*center.tolist(), 0.5 * size[2]],
                    "size": list(size),
                    "rotation": [1.0, 0.0, 0.0, 0.0],
                    "prev": "",
                    "next": "",
                    "num_lidar_pts": int((hit == i + 2).sum()),
                    "num_radar_pts": 0,
                }
                if k > 0:
                    previous = annotations[-len(objects)]
                    annotation["prev"], previous["next"] = previous["token"], annotation["token"]
                else:
                    instances[i]["first_annotation_token"] = annotation["token"]
                instances[i]["last_annotation_token"] = annotation["token"]
                annotations.append(annotation)

    scene["nbr_samples"] = len(samples)
    scene["first_sample_token"], scene["last_sample_token"] = samples[0]["token"], samples[-1]["token"]
    tables = {
        "attribute": attributes,
        "calibrated_sensor": [calibrated_sensor],
        "category": categories,
        "ego_pose": ego_poses,
        "instance": instances,
        "sample": samples,
        "sample_annotation": annotations,
        "sample_data": sample_data,
        "scene": [scene],
    }
    for name, records in tables.items():
        (dst_meta / f"{name}.json").write_text(json.dumps(records, indent=2))

    print(f"generated {len(samples)} keyframes, {len(sample_data)} scans ({args.num_points} pts each) into {dst_raw}")


def parse_args() -> Namespace:
    parser = ArgumentParser(description="Generate a tiny synthetic nuScenes-mini test fixture.")
    parser.add_argument(
        "dst_dir",
        type=str,
        nargs="?",
        default=str(Path(__file__).resolve().parent.parent),
        help="Output NuScenesMini fixture directory (default: the fixture dir next to this script).",
    )
    parser.add_argument("--version", type=str, default="v1.0-mini", help="Metadata version directory.")
    parser.add_argument("--num-keyframes", type=int, default=2, help="Number of LIDAR keyframes.")
    parser.add_argument("--sweeps", type=int, default=3, help="LIDAR clouds per keyframe (keyframe + prior sweeps).")
    parser.add_argument("--num-points", type=int, default=1024, help="Points per scan.")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed.")
    return parser.parse_args()


def main() -> None:
    generate(parse_args())


if __name__ == "__main__":
    main()
