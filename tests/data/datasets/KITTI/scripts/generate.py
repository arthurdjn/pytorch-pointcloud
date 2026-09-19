"""Generate a tiny synthetic KITTI fixture in the layout of the object split.

The output mirrors the real KITTI object layout under `raw/`:

    raw/training/velodyne/{frame}.bin   # float32 (N, 4) = (x, y, z, intensity)
    raw/training/calib/{frame}.txt      # P0-P3, R0_rect, Tr_velo_to_cam, Tr_imu_to_velo
    raw/training/label_2/{frame}.txt    # type trunc occ alpha bbox(4) dims(h,w,l) loc(x,y,z) ry
    raw/training/image_2/{frame}.png    # blank PNG carrying the frame's (height, width) header

No file of the real release is read. The calibration is a seeded near-identity rig, the annotations are
written by this script, and every sweep is cast from a seeded generator (one ray per (beam, azimuth) pair
of a spinning sensor) against a road, a facade and the annotated objects, so the boxes hold points. Every
frame holds at least one detection-class object, next to `DontCare` rows and non-detection classes. Each
frame ships a minimal all-black PNG (enough for the image-header reads; the FOV filter itself is covered
by the unit tests).

Usage:
    uv run --no-sync python scripts/generate.py
"""

import struct
import zlib
from argparse import ArgumentParser, Namespace
from pathlib import Path

import numpy as np

from torch_pointcloud.datasets.kitti import load_kitti_calib, rect_to_lidar


def _write_blank_png(dst: Path, height: int, width: int) -> None:
    """Write a minimal all-black grayscale PNG whose IHDR carries the given dimensions."""

    def chunk(tag: bytes, payload: bytes) -> bytes:
        return struct.pack(">I", len(payload)) + tag + payload + struct.pack(">I", zlib.crc32(tag + payload))

    ihdr = struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0)
    idat = zlib.compress(b"\x00" * ((width + 1) * height), 9)
    dst.write_bytes(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IDAT", idat) + chunk(b"IEND", b""))


def generate(args: Namespace) -> None:
    rng = np.random.default_rng(args.seed)
    dst_split = Path(args.dst_dir) / "raw" / args.split
    for sub in ("velodyne", "calib", "label_2", "image_2"):
        (dst_split / sub).mkdir(parents=True, exist_ok=True)

    # Image (height, width) and annotation rows per frame, in the camera frame of the label format.
    frames = {
        "000000": (
            (370, 1224),
            ["Pedestrian 0.00 0 -0.20 700.00 140.00 800.00 304.92 1.80 0.50 1.00 2.00 1.50 8.00 0.00"],
        ),
        "000001": (
            (375, 1242),
            [
                "Truck 0.00 0 -1.57 600.00 150.00 630.00 190.00 2.80 2.60 12.00 0.50 1.50 60.00 -1.57",
                "Car 0.00 0 1.85 380.00 180.00 420.00 200.00 1.60 1.80 3.70 -16.00 2.40 50.00 1.57",
                "Cyclist 0.00 3 -1.65 670.00 160.00 690.00 190.00 1.80 0.60 2.00 4.50 1.30 40.00 -1.55",
                "DontCare -1 -1 -10 500.00 170.00 590.00 190.00 -1 -1 -1 -1000 -1000 -1000 -10",
            ],
        ),
        "000002": (
            (375, 1242),
            [
                "Misc 0.00 0 -1.82 800.00 160.00 990.00 320.00 1.60 1.50 2.40 3.20 1.60 8.50 -1.47",
                "Car 0.00 0 -1.67 650.00 190.00 700.00 220.00 1.40 1.60 4.40 3.20 2.30 30.00 -1.58",
            ],
        ),
    }

    num_beams = 16
    azimuth, elevation = np.meshgrid(
        np.linspace(-np.pi, np.pi, args.num_points // num_beams, endpoint=False),
        np.deg2rad(np.linspace(-24.8, -0.5, num_beams)),
    )
    azimuth, elevation = azimuth.ravel(), elevation.ravel()

    for frame, ((height, width), rows) in frames.items():
        focal, (roll, pitch, yaw) = rng.uniform(700.0, 730.0), rng.normal(0.0, 0.01, size=3)
        rotation_x = [[1.0, 0.0, 0.0], [0.0, np.cos(roll), -np.sin(roll)], [0.0, np.sin(roll), np.cos(roll)]]
        rotation_y = [[np.cos(pitch), 0.0, np.sin(pitch)], [0.0, 1.0, 0.0], [-np.sin(pitch), 0.0, np.cos(pitch)]]
        rotation_z = [[np.cos(yaw), -np.sin(yaw), 0.0], [np.sin(yaw), np.cos(yaw), 0.0], [0.0, 0.0, 1.0]]
        projection = np.array([[focal, 0.0, width / 2, 0.0], [0.0, focal, height / 2, 0.0], [0.0, 0.0, 1.0, 0.0]])
        matrices = {f"P{i}": projection.copy() for i in range(4)}
        for i, baseline in enumerate((0.0, -0.54, 0.06, -0.48)):
            matrices[f"P{i}"][0, 3] = focal * baseline
        matrices["R0_rect"] = np.array(rotation_x) @ np.array(rotation_y) @ np.array(rotation_z)
        # The camera looks along the LiDAR x axis: x_cam = -y_lidar, y_cam = -z_lidar, z_cam = x_lidar.
        matrices["Tr_velo_to_cam"] = np.array([[0.0, -1.0, 0.0, 0.0], [0.0, 0.0, -1.0, -0.08], [1.0, 0.0, 0.0, -0.27]])
        matrices["Tr_imu_to_velo"] = np.array([[1.0, 0.0, 0.0, -0.81], [0.0, 1.0, 0.0, 0.32], [0.0, 0.0, 1.0, -0.8]])
        calib_file = dst_split / "calib" / f"{frame}.txt"
        calib_file.write_text(
            "".join(
                f"{key}: {' '.join(f'{value:.12e}' for value in matrix.ravel())}\n" for key, matrix in matrices.items()
            )
        )
        (dst_split / "label_2" / f"{frame}.txt").write_text("\n".join(rows) + "\n")
        _write_blank_png(dst_split / "image_2" / f"{frame}.png", height, width)

        # Horizontal distance at which each ray stops: on the road 1.73 m below the sensor, on a facade, or on
        # an annotated object when the ray crosses its azimuth sector below its top.
        stops = [1.73 / np.tan(-elevation), np.full_like(elevation, rng.uniform(40.0, 70.0))]
        calib = load_kitti_calib(calib_file)
        for row in rows:
            fields = row.split(" ")
            if fields[0] == "DontCare":
                continue
            object_height, object_width, object_length = (float(value) for value in fields[8:11])
            center = rect_to_lidar(np.array([[float(value) for value in fields[11:14]]]), calib)[0]
            distance = np.hypot(center[0], center[1])
            sector = np.arctan(0.5 * max(object_width, object_length) / distance)
            offset = np.angle(np.exp(1j * (azimuth - np.arctan2(center[1], center[0]))))
            crosses = (np.abs(offset) < sector) & (distance * np.tan(elevation) < center[2] + object_height)
            stops.append(np.where(crosses, distance, np.inf))
        distance = np.min(stops, axis=0)

        pos = np.stack([distance * np.cos(azimuth), distance * np.sin(azimuth), distance * np.tan(elevation)], axis=1)
        pos += rng.normal(0.0, 0.02, size=pos.shape)
        scan = np.concatenate([pos, rng.uniform(0.0, 1.0, size=(len(pos), 1))], axis=1)
        scan.astype(np.float32).tofile(dst_split / "velodyne" / f"{frame}.bin")

    print(f"generated {len(frames)} frames ({args.num_points} pts each) into {dst_split}: {list(frames)}")


def parse_args() -> Namespace:
    parser = ArgumentParser(description="Generate a tiny synthetic KITTI test fixture.")
    parser.add_argument(
        "dst_dir",
        type=str,
        nargs="?",
        default=str(Path(__file__).resolve().parent.parent),
        help="Output KITTI fixture directory (default: the fixture dir next to this script).",
    )
    parser.add_argument("--split", type=str, default="training", help="KITTI object split to write.")
    parser.add_argument("--num-points", type=int, default=1024, help="Points per frame.")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed.")
    return parser.parse_args()


def main() -> None:
    generate(parse_args())


if __name__ == "__main__":
    main()
