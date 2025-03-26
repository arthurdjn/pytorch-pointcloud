import json
import textwrap
import warnings
from argparse import ArgumentParser, Namespace
from pathlib import Path

import numpy as np
import plyfile
from tqdm import tqdm

from torch_pointcloud.datasets import ScanNet
from torch_pointcloud.datasets.utils import download_url
from torch_pointcloud.utils.types import PathLike


def main() -> None:
    args = parse_args()

    if args.ignore_warnings:
        warnings.filterwarnings("ignore")

    if args.command == "raw":
        generate_raw(args)
    elif args.command == "process":
        generate_processed(args)


def parse_args() -> Namespace:
    parser = ArgumentParser(description="Generate ScanNet data for testing")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Raw data generation command
    raw_parser = subparsers.add_parser("raw", help="Generate raw test data")
    raw_parser.add_argument("dst_dir", type=str, help="Path to output raw data")
    raw_parser.add_argument("--num-points", type=int, default=10, help="Number of points in a scene")
    raw_parser.add_argument("--num-scenes", type=int, default=5, help="Number of scenes")
    raw_parser.add_argument("--num-segments", type=int, default=20, help="Number of segments in a scene")
    raw_parser.add_argument("--seed", type=int, default=42, help="Random seed")
    raw_parser.add_argument("--version", type=str, default="v2", help="Version of ScanNet to generate data for")
    raw_parser.add_argument("--split", type=str, default="train", help="Split to generate data for")
    raw_parser.add_argument(
        "--classes",
        type=str,
        nargs="+",
        default=["wall", "floor", "chair", "table", "desk", "bed", "bookshelf"],
        help="Classes to include in the data",
    )
    raw_parser.add_argument("--ignore-warnings", action="store_true", help="Ignore warnings")

    # Process command
    process_parser = subparsers.add_parser("process", help="Process raw data into final format")
    process_parser.add_argument("raw_dir", type=str, help="Path to raw data directory")
    process_parser.add_argument("--version", type=str, default="v2", help="Version of ScanNet to generate data for")
    process_parser.add_argument("--split", type=str, default="train", help="Split to process")
    process_parser.add_argument("--ignore-warnings", action="store_true", help="Ignore warnings")
    return parser.parse_args()


def generate_raw(args: Namespace) -> None:
    np.random.seed(args.seed)

    # Download the labels
    labels_filename = f"scannet{'v2' if args.version == 'v2' else ''}-labels.combined.tsv"
    labels_path = Path(args.dst_dir, args.version, "tasks", labels_filename)
    url = f"http://kaldir.vc.in.tum.de/scannet/{args.version}/tasks/{labels_filename}"
    download_url(url, labels_path)

    # Download the split file
    split_path = Path(args.dst_dir, "metadata", f"scannetv2_{args.split}.txt")
    url = f"https://raw.githubusercontent.com/facebookresearch/votenet/master/scannet/meta_data/scannetv2_{args.split}.txt"
    download_url(url, split_path)

    # Get the scene IDs associated to the split
    with open(split_path, "r") as f:
        scene_ids = f.readlines()
        scene_ids = sorted([line.strip() for line in scene_ids])
        scene_ids = scene_ids[: args.num_scenes]

    # Generate the dummy scenes
    for scene_id in tqdm(scene_ids, total=len(scene_ids), desc="Generating"):
        scene_dir = Path(args.dst_dir, args.version, "scans", scene_id)
        scene_dir.mkdir(parents=True, exist_ok=True)

        _generate_scene_points(scene_dir, scene_id, num_points=args.num_points)
        _generate_scene_segmentation(scene_dir, scene_id, num_segments=args.num_segments, num_points=args.num_points)
        _generate_scene_aggregation(scene_dir, scene_id, num_segments=args.num_segments, classes=args.classes)
        _generate_scene_metadata(scene_dir, scene_id, with_axis_alignment=args.version == "v2")


def generate_processed(args: Namespace) -> None:
    raw_dir = Path(args.raw_dir)
    root = raw_dir.resolve().parent.parent.as_posix()

    _ = ScanNet(
        root=root,
        version=args.version,
        with_unk=True,
        classes="all",
        split=args.split,
        show_progress=True,
        force_process=True,
    )


def _generate_scene_points(scene_dir: PathLike, scene_id: str, num_points: int = 1000) -> None:
    points = np.random.uniform(-5, 5, (num_points, 3)).astype(np.float32)
    colors = np.random.randint(0, 256, (num_points, 3)).astype(np.uint8)
    normals = np.random.uniform(-1, 1, (num_points, 3)).astype(np.float32)
    normals = normals / np.linalg.norm(normals, axis=1, keepdims=True)

    # Generate random faces (triangles)
    num_faces = num_points * 2  # Arbitrary number of faces
    faces = np.random.randint(0, num_points, (num_faces, 3))

    # Create PLY file
    vertex_data = np.empty(
        num_points,
        dtype=[
            ("x", "f4"),
            ("y", "f4"),
            ("z", "f4"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
            ("nx", "f4"),
            ("ny", "f4"),
            ("nz", "f4"),
        ],
    )

    vertex_data["x"] = points[:, 0]
    vertex_data["y"] = points[:, 1]
    vertex_data["z"] = points[:, 2]
    vertex_data["red"] = colors[:, 0]
    vertex_data["green"] = colors[:, 1]
    vertex_data["blue"] = colors[:, 2]
    vertex_data["nx"] = normals[:, 0]
    vertex_data["ny"] = normals[:, 1]
    vertex_data["nz"] = normals[:, 2]

    # Create face data
    face_data = np.empty(num_faces, dtype=[("vertex_indices", "i4", (3,))])
    face_data["vertex_indices"] = faces

    # Create PLY elements
    vertex_element = plyfile.PlyElement.describe(vertex_data, "vertex")
    face_element = plyfile.PlyElement.describe(face_data, "face")

    # Save PLY file with both vertices and faces
    ply_path = Path(scene_dir, f"{scene_id}_vh_clean_2.ply")
    plyfile.PlyData([vertex_element, face_element], text=False).write(str(ply_path))


def _generate_scene_segmentation(
    scene_dir: PathLike,
    scene_id: str,
    num_segments: int = 20,
    num_points: int = 1000,
) -> None:
    seg_indices = np.random.randint(0, num_segments, num_points).tolist()
    segs_data = {
        "params": {
            "kThresh": "0.0001",
            "segMinVerts": "20",
            "minPoints": "750",
            "maxPoints": "30000",
            "thinThresh": "0.05",
            "flatThresh": "0.001",
            "minLength": "0.02",
            "maxLength": "1",
        },
        "sceneId": scene_id,
        "segIndices": seg_indices,
    }

    segs_path = Path(scene_dir, f"{scene_id}_vh_clean_2.0.010000.segs.json")
    with open(segs_path, "w") as f:
        json.dump(segs_data, f)


def _generate_scene_aggregation(
    scene_dir: PathLike,
    scene_id: str,
    classes: list[str],
    num_segments: int = 20,
    num_objects: int = 5,
) -> None:
    seg_groups = []
    for obj_idx in range(num_objects):
        num_obj_segments = np.random.randint(1, num_segments)
        segments = np.random.choice(range(num_segments), num_obj_segments, replace=False).tolist()
        label = np.random.choice(classes)
        seg_groups.append({"id": obj_idx, "objectId": obj_idx, "segments": segments, "label": label})

    agg_data = {
        "sceneId": f"scannet.{scene_id}",
        "appId": "Aggregator.v2",
        "segGroups": seg_groups,
        "segmentsFile": f"scannet.{scene_id}_vh_clean_2.0.010000.segs.json",
    }

    agg_path = Path(scene_dir, f"{scene_id}.aggregation.json")
    with open(agg_path, "w") as f:
        json.dump(agg_data, f)


def _generate_scene_metadata(scene_dir: PathLike, scene_id: str, with_axis_alignment: bool = True) -> None:
    metadata_path = Path(scene_dir, f"{scene_id}.txt")

    metadata = ""

    if with_axis_alignment:
        metadata += "axisAlignment = 1.0 0.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 0.0 1.0\n"

    metadata += "colorHeight = 100\n"
    metadata += "colorToDepthExtrinsics = 1.0 0.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 0.0 1.0\n"
    metadata += "colorWidth = 100\n"
    metadata += "depthHeight = 100\n"
    metadata += "depthWidth = 100\n"
    metadata += "fx_color = 1.0\n"
    metadata += "fx_depth = 1.0\n"
    metadata += "fy_color = 1.0\n"
    metadata += "fy_depth = 1.0\n"
    metadata += "mx_color = 50.0\n"
    metadata += "mx_depth = 50.0\n"
    metadata += "my_color = 50.0\n"
    metadata += "my_depth = 50.0\n"
    metadata += "numColorFrames = 1\n"
    metadata += "numDepthFrames = 1\n"
    metadata += "numIMUmeasurements = 1\n"
    metadata += "sceneType = Apartment\n"

    with open(metadata_path, "w") as f:
        f.write(textwrap.dedent(metadata.strip()))


if __name__ == "__main__":
    main()
