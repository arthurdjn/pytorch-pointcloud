from collections import defaultdict
from functools import cached_property
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, TypedDict
from urllib.parse import urljoin
from urllib.request import urlopen

import numpy as np
import pandas as pd
import torch
from plyfile import PlyData
from torch import Tensor
from torch.utils.data import Dataset

from torch_pointcloud.utils.geometry import axis_aligned_bounding_box
from torch_pointcloud.utils.io import load_json
from torch_pointcloud.utils.types import PATH_LIKE

from .utils import download_file

UNK_LABEL = "<unk>"
UNK_TARGET = -1


def align_points(xyz: np.ndarray, m: np.ndarray) -> np.ndarray:
    """Align the points coordinates using the axis alignment matrix.

    Args:
        xyz: The points of shape (N,3), in XYZ format.
        m: The axis alignment matrix (4,4).

    Returns:
        The aligned points coordinates of shape (N,3).
    """
    N, _ = xyz.shape
    out_xyz = np.ones((N, 4))
    out_xyz[:, :3] = xyz[:, :3]
    out_xyz = np.dot(out_xyz, m.T)  # (N,4)
    return out_xyz[:, :3]


class ScannetData(TypedDict, total=False):
    xyz: Tensor
    rgb: Tensor
    label: Tensor
    instance_label: Tensor
    sem_label: Tensor
    scan_id: str


class Scannet(Dataset):

    data_url = "http://kaldir.vc.in.tum.de/scannet/"
    resources = [
        "{version}/scans.txt",  # list of all scan ids
        "{version}/tasks/scannet-labels.combined.tsv",  # v1 raw labels
        "{version}/tasks/scannetv2-labels.combined.tsv",  # v2 raw labels
        "{version}/scans/{scan_id}/{scan_id}.aggregation.json",
        "v1/scans/{scan_id}/{scan_id}.sens",  # NOTE: The `.sens` file from the v2 version is the same as the v1
        "{version}/scans/{scan_id}/{scan_id}.txt",
        "{version}/scans/{scan_id}/{scan_id}_vh_clean.ply",
        "{version}/scans/{scan_id}/{scan_id}_vh_clean_2.0.010000.segs.json",
        "{version}/scans/{scan_id}/{scan_id}_vh_clean_2.ply",
        "{version}/scans/{scan_id}/{scan_id}_vh_clean.segs.json",
        "{version}/scans/{scan_id}/{scan_id}_vh_clean.aggregation.json",
        "{version}/scans/{scan_id}/{scan_id}_vh_clean_2.labels.ply",
        "{version}/scans/{scan_id}/{scan_id}_2d-instance.zip",
        "{version}/scans/{scan_id}/{scan_id}_2d-instance-filt.zip",
        "{version}/scans/{scan_id}/{scan_id}_2d-label.zip",
        "{version}/scans/{scan_id}/{scan_id}_2d-label-filt.zip",
    ]
    meta_url = "https://raw.githubusercontent.com/facebookresearch/votenet/master/scannet/meta_data/"
    meta_resources = [
        "scannetv2-labels.combined.tsv",
        "scannetv2_train.txt",
        "scannetv2_test.txt",
        "scannetv2_val.txt",
    ]

    # TODO: Add overall cluster sizes and heading labels (see votenet)

    def __init__(
        self,
        root: PATH_LIKE,
        version: Literal["v1", "v2"],
        split: Literal["train", "val", "test"],
        transform: Optional[Callable[[Dict[str, Tensor]], Dict[str, Tensor]]] = None,
        target_transform: Optional[Callable[[Dict[str, Tensor]], Dict[str, Tensor]]] = None,
        transforms: Optional[Callable[[Dict[str, Tensor]], Dict[str, Tensor]]] = None,
        download: bool = False,
        unk_label: str = UNK_LABEL,
        unk_target: int = UNK_TARGET,
    ) -> None:
        super().__init__()
        assert version in ["v1", "v2"], "ScanNet version must be either 'v1' or 'v2'"
        assert split in ["train", "val", "test"], "ScanNet split must be either 'train', 'val' or 'test'"

        self.root = Path(root).as_posix()
        self.version = version
        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        self.transforms = transforms
        self.unk_label = unk_label
        self.unk_target = unk_target

        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError("Dataset not found. You can use download=True to download it")

        if not Path(self.processed_dir, f"{self.split}.pt").exists():
            self.process()

        self.data = self._load_processed_data()

    @property
    def data_dir(self) -> str:
        return Path(self.root, f"{self.__class__.__name__}{self.version}").as_posix()

    @property
    def raw_dir(self) -> str:
        return Path(self.data_dir, "raw").as_posix()

    @property
    def processed_dir(self) -> str:
        return Path(self.data_dir, "processed").as_posix()

    @cached_property
    def classes(self) -> List[str]:
        return []

    def download(self) -> None:
        if self._check_exists():
            return

        raw_dir = Path(self.raw_dir)

        # 1. Download the metadata
        for resource_path in self.meta_resources:
            url = urljoin(self.meta_url, resource_path)
            file_name = Path(resource_path).name
            out_path = raw_dir / "metadata" / file_name
            download_file(out_path, url, description=f"Downloading Metadata {file_name!r}")

        # 2. Download associated labels (resource index 1 or 2)
        resource_path = self.resources[1] if self.version == "v1" else self.resources[2]
        resource_path = resource_path.format(version=self.version)
        url = urljoin(self.data_url, resource_path)

        file_name = Path(resource_path).name
        out_path = Path(raw_dir) / file_name
        download_file(out_path, url, description=f"Downloading Labels {file_name!r}")

        # 3. Download raw scans (all resources starting from index 3)
        # First, get all scan ids
        resource_path = self.resources[0].format(version=self.version)
        url = urljoin(self.data_url, resource_path)
        with urlopen(url) as f:
            scan_ids = [line.decode("utf-8").strip() for line in f]

        # Download all resources per scan
        for i, scan_id in enumerate(scan_ids):
            for resource_path in self.resources[3:]:
                resource_path = resource_path.format(version=self.version, scan_id=scan_id)
                url = urljoin(self.data_url, resource_path)

                file_name = Path(resource_path).name
                out_path = raw_dir / "scans" / scan_id / file_name
                download_file(out_path, url, description=f"Downloading Scans [{i+1}/{len(scan_ids)}] {file_name!r}")

    def process(self) -> None:
        raise NotImplementedError("Process method not implemented")

    def _process_scan(self, scan_id: str) -> ScannetData:
        label_path = Path(self.raw_dir, "metadata", "scannetv2-labels.combined.tsv")
        scan_mesh_path = Path(self.raw_dir, "scans", scan_id, f"{scan_id}_vh_clean_2.ply")
        scan_meta_path = Path(self.raw_dir, "scans", scan_id, f"{scan_id}.txt")
        scan_agg_path = Path(self.raw_dir, "scans", scan_id, f"{scan_id}.aggregation.json")
        scan_seg_path = Path(self.raw_dir, "scans", scan_id, f"{scan_id}_vh_clean_2.0.010000.segs.json")

        # Process the RGB points
        scan_meta = self._load_scan_metadata(scan_meta_path)
        xyz, rgb = self._load_scan_points(scan_mesh_path)
        xyz = align_points(xyz, scan_meta["axisAlignment"])

        if not scan_agg_path.exists() or not scan_seg_path.exists():
            return {"xyz": xyz, "rgb": rgb}

        # Create mappings from object_id to label / segments
        aggregation_data = load_json(scan_agg_path)
        object_id_to_label = {}
        object_id_to_segments = defaultdict(list)
        for seg_group in aggregation_data["segGroups"]:
            object_id = seg_group["objectId"]
            label = seg_group["label"]
            segments = seg_group["segments"]
            object_id_to_label[object_id] = label
            object_id_to_segments[object_id].extend(segments)

        # Process the segments
        seg_data = load_json(scan_seg_path)
        segment_to_vertices = defaultdict(list)
        num_vertices = len(seg_data["segIndices"])
        for idx, seg_id in enumerate(seg_data["segIndices"]):
            segment_to_vertices[seg_id].append(idx)

        # Sanity checks
        assert len(xyz) == num_vertices, "Invalid number of vertices in the point cloud."

        # Create a mapping from label "raw_category" to id "nyu40id"
        label_df = pd.read_csv(label_path, sep="\t")
        label_to_label_id = dict(zip(label_df["raw_category"], label_df["nyu40id"]))
        object_id_to_label_id = {object_id: label_to_label_id[label] for object_id, label in object_id_to_label.items()}

        # Create the targets associated to each points (semantic labels between [-1, num_classes-1])
        label_ids = np.zeros(shape=(num_vertices), dtype=np.uint32) + self.unk_target
        object_ids = np.zeros(shape=(num_vertices), dtype=np.uint32) + self.unk_target
        bboxes = np.zeros((len(object_id_to_segments), 7))

        for object_id, segments in object_id_to_segments.items():
            label_id = object_id_to_label_id[object_id]
            for segment in segments:
                vertices = segment_to_vertices[segment]
                label_ids[vertices] = label_id
                object_ids[vertices] = object_id

            selected_xyz = xyz[object_ids == object_id]
            bbox = axis_aligned_bounding_box(selected_xyz)
            bboxes[object_id, :] = np.concatenate([bbox, [label_id]])

        # TODO !
        # ! Mask / filter some classes

        # TODO: Add heading and size labels

        return {"xyz": xyz, "rgb": rgb, "semantic": label_ids, "instance": object_ids, "bboxes": bboxes}

    @staticmethod
    def _load_scan_points(file_path: PATH_LIKE) -> Tuple[np.ndarray, np.ndarray]:
        with open(file_path, "rb") as f:
            plydata = PlyData.read(f)
            vertex = plydata["vertex"]
            xyz = np.column_stack([vertex.data["x"], vertex.data["y"], vertex.data["z"]])
            rgb = np.column_stack([vertex.data["red"], vertex.data["green"], vertex.data["blue"]])
        return xyz, rgb

    @staticmethod
    def _load_scan_metadata(file_path: PATH_LIKE) -> Dict[str, Any]:
        meta = {}
        with open(file_path) as f:
            lines = f.readlines()

        for line in lines:
            raw_key, raw_value = line.strip().split(" = ")
            key: str = raw_key.strip()
            value: Any = raw_value.strip()
            if key in ["axisAlignment", "colorToDepthExtrinsics"]:
                value = [float(x) for x in value.split(" ")]
                value = np.array(value).reshape((4, 4))
            elif key in [
                "colorHeight",
                "colorWidth",
                "depthHeight",
                "depthWidth",
                "numColorFrames",
                "numDepthFrames",
                "numIMUmeasurements",
            ]:
                value = int(value)
            elif key in [
                "fx_color",
                "fx_depth",
                "fy_color",
                "fy_depth",
                "mx_color",
                "mx_depth",
                "my_color",
                "my_depth",
            ]:
                value = float(value)
            elif key in ["sceneType"]:
                value = value
            else:
                continue
            meta[key] = value
        return meta

    def _load_processed_data(self) -> List[Dict[str, Tensor]]:
        file_path = Path(self.processed_dir, f"{self.split}.pt")
        return torch.load(file_path)

    def _check_exists(self) -> bool:
        return Path(self.raw_dir).exists()

    def __getitem__(self, index: int) -> Any:
        data = self.data[index]
        if self.transforms is not None:
            data = self.transforms(data)
        return data

    def __len__(self) -> int:
        return len(self.data)
