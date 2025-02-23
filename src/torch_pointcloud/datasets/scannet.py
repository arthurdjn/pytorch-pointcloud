"""
The ScanNet dataset as described in the paper
[ScanNet: Richly-annotated 3D Reconstructions of Indoor Scenes](https://arxiv.org/abs/1702.04405).

"""

import warnings
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, Sequence, Tuple, TypedDict, Union
from urllib.parse import urljoin
from urllib.request import urlopen

import numpy as np
import pandas as pd
import plyfile
import torch
from tqdm import tqdm
from typing_extensions import NotRequired, override

from torch_pointcloud.utils.geometry import transform_points, vertex_normals
from torch_pointcloud.utils.io import load_json
from torch_pointcloud.utils.types import PathLike

from .pointcloud import PointCloudDataset
from .utils import download_url

UNK_CLS = "<unk>"
UNK_IDX = -1


class ScanNetData(TypedDict):
    points: torch.Tensor
    colors: torch.Tensor
    normals: torch.Tensor
    instances: NotRequired[torch.Tensor]
    labels: NotRequired[torch.Tensor]
    scene: NotRequired[str]


def load_scannet_scene_mesh(file_path: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """Load a ScanNet PLY file and return the vertices and faces.

    Args:
        file_path: The path to the PLY file.

    Returns:
        The vertices and faces.

    Examples:
        >>> vertices, faces = load_ply("data/ScanNet/raw/v2/scans/scene0000_00/scene0000_00_vh_clean_2.ply")
    """
    with open(file_path, "rb") as f:
        plydata = plyfile.PlyData.read(f)

    vertices = np.array([tuple(vertex) for vertex in plydata["vertex"].data], dtype=np.float32)
    faces = np.stack(plydata["face"].data["vertex_indices"], axis=0)
    return torch.from_numpy(vertices), torch.from_numpy(faces).long()


def load_scannet_scene_metadata(file_path: str) -> Dict[str, Any]:
    """Load a ScanNet metadata file and return the metadata.

    Args:
        file_path: The path to the metadata file.

    Returns:
        The metadata.

    Examples:
        >>> metadata = load_scannet_scene_metadata("data/ScanNet/raw/v2/scans/scene0000_00/scene0000_00.txt")
    """
    meta = {}
    with open(file_path) as f:
        lines = f.readlines()

    for line in lines:
        raw_key, raw_value = line.strip().split(" = ")
        key: str = raw_key.strip()
        value: Any = raw_value.strip()
        if key in ["axisAlignment", "colorToDepthExtrinsics"]:
            value = [float(x) for x in value.split(" ")]
            value = torch.tensor(value, dtype=torch.float32).reshape((4, 4))
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


def load_scannet_labels(file_path: PathLike) -> pd.DataFrame:
    """Load the ScanNet labels CSV file as a `pandas.DataFrame` object.

    Args:
        file_path: Path to the labels CSV file, usually located in the `raw` directory
            as `data/ScanNet/raw/metadata/scannetv2-labels.combined.tsv`

    Returns:
        The labels as a `pandas.DataFrame` object.

    Examples:
        >>> file_path = "data/ScanNet/raw/metadata/scannetv2-labels.combined.tsv"
        >>> labels = load_scannet_labels(file_path)
    """
    return pd.read_csv(file_path, sep="\t")


def select_scannet_classes(
    labels: pd.DataFrame,
    name: str,
    sort_by: Optional[str] = None,
    values: Union[Sequence[str], Literal["all"]] = "all",
) -> List[Any]:
    """Select the classes to load from the labels.

    Args:
        labels: The labels as a `pandas.DataFrame` object.
        name: The name of the column in the labels to select the classes from.
        sort_by: The column to sort the labels by.
        values: The values to select from the labels.

    Returns:
        The selected classes.

    Examples:
        >>> labels = load_scannet_labels("data/ScanNet/raw/metadata/scannetv2-labels.combined.tsv")
        >>> classes = select_scannet_classes(labels, "raw_category", sort_by="id", values=["wall", "floor"])
        >>> nyu40classes = select_scannet_classes(labels, "nyu40class", sort_by="nyu40id", values="all")
    """
    if sort_by is not None:
        labels = labels.sort_values(sort_by)

    original_values = labels[name].unique().tolist()

    if values == "all":
        return original_values
    elif isinstance(values, (list, tuple)):
        if not all(c in original_values for c in values):
            missing_values = ", ".join(set(values) - set(original_values))
            warnings.warn(
                f"Some values are not present in the labels {name!r}, "
                f"ignoring them: {missing_values}. "
                f"If you want to load all values, use 'all' instead."
            )
            values = [c for c in values if c in original_values]

        return list(values)

    else:
        raise ValueError(
            f"Invalid values, expected 'all' or a sequence of strings associated to {name!r}, "
            f"but got {type(values).__name__}."
        )


def load_scannet_scene(
    mesh_path: PathLike,
    meta_path: Optional[PathLike] = None,
    aggregation_path: Optional[PathLike] = None,
    segments_path: Optional[PathLike] = None,
    label_to_idx: Optional[Dict[str, int]] = None,
    scene_id: Optional[str] = None,
) -> ScanNetData:
    """Load a ScanNet scene and return the parsed points, colors, normals, instances, and labels
    in a dictionary format.

    Args:
        mesh_path: Path to the raw mesh file, usually saved as `data/ScanNet/raw/v2/scans/{scan_id}/{scan_id}.ply`.
        meta_path: Path to the metadata file, usually saved as `data/ScanNet/raw/v2/scans/{scan_id}/{scan_id}.txt`.
        aggregation_path: Path to the aggregation file, usually saved as `data/ScanNet/raw/v2/scans/{scan_id}/{scan_id}.aggregation.json`.
        segments_path: Path to the segments file, usually saved as `data/ScanNet/raw/v2/scans/{scan_id}/{scan_id}.segs.json`.
        label_to_idx: A dictionary mapping object labels to contiguous positive indices. The labels correspond to the `raw_category` column
            in the labels CSV file, or to the `label` key in the aggregation JSON file.
            This mapping is used to map object labels to their associated target indices.

    Returns:
        The loaded scene.

    Examples:
        >>> mesh_path = "data/ScanNet/raw/v2/scans/scene0000_00/scene0000_00.ply"
        >>> meta_path = "data/ScanNet/raw/v2/scans/scene0000_00/scene0000_00.txt"
        >>> aggregation_path = "data/ScanNet/raw/v2/scans/scene0000_00/scene0000_00.aggregation.json"
        >>> segments_path = "data/ScanNet/raw/v2/scans/scene0000_00/scene0000_00.segs.json"
        >>> labels_path = "data/ScanNet/raw/metadata/scannetv2-labels.combined.tsv"
        >>> labels = load_scannet_labels(labels_path)
        >>> label_to_idx = {label: idx for idx, label in enumerate(labels["raw_category"].unique())}
        >>> scene = load_scannet_scene(mesh_path, meta_path, aggregation_path, segments_path, label_to_idx)
    """
    label_to_idx = label_to_idx or {}

    # Load the points
    vertices, faces = load_scannet_scene_mesh(mesh_path)
    points, colors = vertices[:, :3], vertices[:, 3:6]
    normals = vertex_normals(points, faces)

    # Optionally transform the points with the axis alignment matrix
    metadata = load_scannet_scene_metadata(meta_path) if meta_path else {}
    if "axisAlignment" in metadata:
        # The axis alignment matrix is a 4x4 matrix that transforms the points
        # that is provided in the v2 version of the dataset
        points = transform_points(points, metadata["axisAlignment"])

    if not aggregation_path or not segments_path:
        # If no aggregation or segments are provided,
        # return the points and colors
        return {"points": points, "colors": colors, "normals": normals}

    # Create mappings from object_id to label / segments
    aggregation_data = load_json(aggregation_path)
    object_id_to_label = {}
    object_id_to_segments = defaultdict(list)
    for seg_group in aggregation_data["segGroups"]:
        object_id = seg_group["objectId"]
        # The label corresponds to the `category` (v1) or `raw_category` (v2)
        # column in the labels CSV file
        label = seg_group["label"]
        segments = seg_group["segments"]
        object_id_to_label[object_id] = label
        object_id_to_segments[object_id].extend(segments)

    # Process the segments
    segments_data = load_json(segments_path)
    segment_to_vertices = defaultdict(list)
    num_vertices = len(segments_data["segIndices"])
    for idx, seg_id in enumerate(segments_data["segIndices"]):
        segment_to_vertices[seg_id].append(idx)

    # Sanity checks
    assert len(points) == num_vertices, "Invalid number of vertices in the point cloud."

    # Create the targets associated to each points (semantic labels between [-1, num_classes-1])
    instances = torch.full((num_vertices,), -1, dtype=torch.int32)
    labels = torch.full((num_vertices,), -1, dtype=torch.int32)

    for object_id, segments in object_id_to_segments.items():
        label = object_id_to_label[object_id]
        label_idx = label_to_idx.get(label, -1)

        for segment in segments:
            vertices = torch.tensor(segment_to_vertices[segment], dtype=torch.int32)
            instances[vertices] = object_id
            labels[vertices] = label_idx

    # Filter out unlabelled points
    mask = labels != -1
    points = points[mask]
    colors = colors[mask]
    normals = normals[mask]
    instances = instances[mask]
    labels = labels[mask]

    data: ScanNetData = {
        "points": points,
        "colors": colors,
        "normals": normals,
        "instances": instances,
        "labels": labels,
    }

    if scene_id:
        data["scene"] = scene_id

    return data


class ScanNet(PointCloudDataset):
    """The ScanNet dataset as described in the paper
    [ScanNet: Richly-annotated 3D Reconstructions of Indoor Scenes](https://arxiv.org/abs/1702.04405).
    This dataset contains 2.5M views in 1513 scans acquired in 707 distinct spaces.
    Each scan is annotated with 3D camera poses, meshes, object segmentation, and scene semantics for
    a total of 36,000 annotated object instances.

    The dataset is available in two versions:

    - `v1`: The original dataset with 1,513 scans.
    - `v2`: Improved annotation coverage to ~90% (from 63% in v1),
        with 100 more scans for test.

    Note:
        It is recommended to use the `v2` version, as it contains more annotated object instances.
        The `v1` version is kept for backward compatibility.

    Note:
        By default, the labels are taken from the `nyu40class` column in the labels CSV file,
        and the `nyu40id` column is used to sort the labels. Note than the `class_to_idx` property
        returns a dictionary mapping the class name to the contiguous index, and indices
        may not correspond to the `nyu40id` values.

        In most cases, if you set `classes="all"` and `with_unk=True`, the labels will be contiguous and
        the `class_to_idx` property will match the `nyu40id` (or more generally, the `label_id` column) values.

    Args:
        root: The root directory of the dataset.
        version: The version of the dataset to use.
        split: The split dataset to load, one of `train`, `val`, or `test`.
        classes: The classes to load.
        label_name: The name of the label column in the labels CSV file.
        label_id: The id of the label column in the labels CSV file.
        with_unk: Whether to include the unknown class in the classes.
        transform: A callable that transforms the data when retrieved from the dataset.
        pre_transform: Used to transform the data before saving it in the processed directory.
        pre_filter: Used to filter the data before saving it in the processed directory.
        download: Whether to download the raw data.
        force_download: Whether to force the download of the raw data.
        force_process: Whether to force the processing of the raw data.
        show_progress: Whether to show a progress bar during processing.

    Example:
        Assuming you have downloaded the raw dataset from http://kaldir.vc.in.tum.de/scannet/,
        and extracted it under `data/ScanNet/raw`, you can load the dataset as follows:

        ```python
        from torch_pointcloud.datasets import ScanNet

        dataset = ScanNet(
            root="data/ScanNet/raw",
            version="v2",
            train=True,
        )
        ```

        To load only the "wall" and "floor" classes, you can do:

        ```python
        dataset = ScanNet(
            root="data/ScanNet/raw",
            version="v2",
            train=True,
            classes=["wall", "floor"],
            with_unk=True,  # All other classes will be mapped to the unknown class
        )
        ```

        By default, the labels are taken from the `nyu40class` column in the labels CSV file,
        and the `nyu40id` column is used to map the labels to contiguous indices.
        You can change this by setting the `label_name` and `label_id` arguments.

        For example, to use the `raw_category` column and the `id` column, you can do:

        ```python
        dataset = ScanNet(
            root="data/ScanNet/raw",
            version="v2",
            train=True,
            label_name="raw_category",
            label_id="id",
        )
        ```
    """

    data_url = "http://kaldir.vc.in.tum.de/scannet/"
    meta_url = "https://raw.githubusercontent.com/facebookresearch/votenet/master/scannet/meta_data/"
    label_resources = [
        "v1/tasks/scannet-labels.combined.tsv",  # v1 raw labels
        "v2/tasks/scannetv2-labels.combined.tsv",  # v2 raw labels
    ]

    scan_ids_resource = "{version}/scans.txt"
    scan_resources = [
        # "v1/scans/{scan_id}/{scan_id}.sens",  # NOTE: The `.sens` file from the v2 version is the same as the v1
        "{version}/scans/{scan_id}/{scan_id}.aggregation.json",  # File used during processing
        "{version}/scans/{scan_id}/{scan_id}.txt",  # File used during processing
        "{version}/scans/{scan_id}/{scan_id}_vh_clean_2.0.010000.segs.json",  # File used during processing
        "{version}/scans/{scan_id}/{scan_id}_vh_clean_2.ply",  # File used during processing
        # "{version}/scans/{scan_id}/{scan_id}_vh_clean.ply",
        # "{version}/scans/{scan_id}/{scan_id}_vh_clean.segs.json",
        # "{version}/scans/{scan_id}/{scan_id}_vh_clean.aggregation.json",
        # "{version}/scans/{scan_id}/{scan_id}_vh_clean_2.labels.ply",
        # "{version}/scans/{scan_id}/{scan_id}_2d-instance.zip",
        # "{version}/scans/{scan_id}/{scan_id}_2d-instance-filt.zip",
        # "{version}/scans/{scan_id}/{scan_id}_2d-label.zip",
        # "{version}/scans/{scan_id}/{scan_id}_2d-label-filt.zip",
    ]
    test_scan_ids_resource = "v2/scans_test.txt"
    test_scan_resources = [
        # "v2/scans/{scan_id}/{scan_id}.sens",
        "v2/scans/{scan_id}/{scan_id}.txt",
        "v2/scans/{scan_id}/{scan_id}_vh_clean_2.ply",
        # "v2/scans/{scan_id}/{scan_id}_vh_clean.ply",
    ]
    meta_resources = [
        "scannetv2-labels.combined.tsv",
        "scannetv2_train.txt",
        "scannetv2_test.txt",
        "scannetv2_val.txt",
    ]

    def __init__(
        self,
        root: str,
        version: Literal["v1", "v2"] = "v2",
        split: Literal["train", "test", "val"] = "train",
        classes: Union[Sequence[str], Literal["all"]] = "all",
        label_name: str = "nyu40class",
        label_id: str = "nyu40id",
        with_unk: Optional[bool] = None,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None,
        download: bool = False,
        force_download: bool = False,
        force_process: bool = False,
        show_progress: bool = True,
    ) -> None:
        super().__init__(root)
        if split not in ["train", "val", "test"]:
            raise ValueError(f"Invalid split {split!r}, expected one of 'train', 'val' or 'test'.")

        self.version = version
        self.split = split
        self.label_name = label_name
        self.label_id = label_id
        self.transform = transform
        self.pre_transform = pre_transform
        self.pre_filter = pre_filter
        self.show_progress = show_progress

        if download or force_download:
            self.download(force=force_download)

        # Get associated raw labels from the CSV file
        resource_path = self.label_resources[int(self.version == "v2")]
        resource_path = resource_path.format(version=self.version)
        labels_path = Path(self.raw_dir, resource_path)
        if not labels_path.exists():
            raise FileNotFoundError(
                f"Labels file not found at {labels_path!r}. Make sure to download the labels from {self.data_url}."
            )

        self.labels = load_scannet_labels(labels_path)

        # Select desired classes (with unknown class if specified)
        self.classes = select_scannet_classes(self.labels, self.label_name, sort_by=self.label_id, values=classes)
        if with_unk:
            self.classes.insert(0, UNK_CLS)

        self.process(force=force_process)

        self.data = self._load_processed_data()

    @property
    def class_to_idx(self) -> Dict[str, int]:
        return {cls: idx for idx, cls in enumerate(self.classes)}

    def raw_files_exist(self) -> bool:
        scans_dir = Path(self.raw_dir, self.version if self.split in ["train", "val"] else "v2", "scans")

        if not scans_dir.exists():
            return False

        # Check that there is at least one scene directory
        scene_dirs = list(scans_dir.glob("scene*"))
        if len(scene_dirs) == 0:
            return False

        return True

    def processed_files_exist(self) -> bool:
        # Checks that the processed files exist for the specified split
        return len(list(Path(self.processed_dir, self.split).glob("*.pt"))) > 0

    def download(self, force: bool = False) -> None:
        if self.raw_files_exist() and not force:
            return

        # Download the metadata, to get train / val / test splits
        # to reproduce SOTA benchmarks
        for resource_path in self.meta_resources:
            url = urljoin(self.meta_url, resource_path)
            file_name = Path(resource_path).name
            out_path = Path(self.raw_dir, "metadata", file_name)
            download_url(
                url,
                out_path,
                description=f"Downloading metadata {file_name!r}",
                show_progress=self.show_progress,
                overwrite=True if force else "incomplete",
            )

        # Download associated labels and categories
        resource_path = self.label_resources[int(self.version == "v2")]
        resource_path = resource_path.format(version=self.version)
        url = urljoin(self.data_url, resource_path)
        out_path = Path(self.raw_dir, resource_path)
        download_url(
            url,
            out_path,
            description=f"Downloading labels {Path(out_path).name!r}",
            show_progress=self.show_progress,
            overwrite=True if force else "incomplete",
        )

        # Search for all available scans
        is_train_val = self.split in ["train", "val"]
        resource_path = self.scan_ids_resource if is_train_val else self.test_scan_ids_resource
        resource_path = resource_path.format(version=self.version)
        url = urljoin(self.data_url, resource_path)
        with urlopen(url) as f:
            scan_ids = [line.decode("utf-8").strip() for line in f]

        # Download all resources per scan
        resources = self.scan_resources if is_train_val else self.test_scan_resources
        for i, scan_id in enumerate(scan_ids):
            for resource_path in resources:
                resource_path = resource_path.format(version=self.version, scan_id=scan_id)
                url = urljoin(self.data_url, resource_path)
                out_path = Path(self.raw_dir, resource_path)
                file_name = Path(resource_path).name
                download_url(
                    url,
                    out_path,
                    description=f"Downloading scans [{i + 1}/{len(scan_ids)}] {file_name!r}",
                    show_progress=self.show_progress,
                    overwrite=True if force else "incomplete",
                )

    def process(self, force: bool = False) -> None:
        if self.processed_files_exist() and not force:
            return
        elif not self.raw_files_exist():
            raise RuntimeError(
                f"Dataset not found at {self.root!r}. "
                f"You can download the raw dataset from {self.data_url!r}, "
                f"and extract it under {self.raw_dir!r}."
            )

        raw_dir = Path(self.raw_dir)

        # Create the mapping between object labels (also named "raw_category" in the CSV labels) and indices
        # NOTE: indices must be contiguous positive integers to be ready to use for training purposes
        class_to_idx = self.class_to_idx
        unk_idx = class_to_idx.get(UNK_CLS, -1)
        label_col = "raw_category" if self.version == "v2" else "category"
        label_to_id = dict(zip(self.labels[label_col], self.labels[self.label_id]))
        label_to_idx = {label: class_to_idx.get(label, unk_idx) for label in label_to_id.keys()}

        # Look for the associated scene IDs for the specified split
        is_train_val = self.split in ["train", "val"]
        scans_dir = Path(raw_dir, self.version if is_train_val else "v2", "scans")
        split_file = raw_dir / "metadata" / f"scannetv2_{self.split}.txt"
        with open(split_file) as f:
            scene_ids = sorted([line.strip() for line in f])

        # Process each scene
        for scene_id in tqdm(scene_ids, desc="Processing", total=len(scene_ids)):
            meta_path = next(scans_dir.glob(f"{scene_id}/{scene_id}.txt"), None)
            mesh_path = next(scans_dir.glob(f"{scene_id}/{scene_id}_vh_clean_2.ply"), None)
            aggregation_path = next(scans_dir.glob(f"{scene_id}/{scene_id}.aggregation.json"), None)
            segments_path = next(scans_dir.glob(f"{scene_id}/{scene_id}_vh_clean_2.0.010000.segs.json"), None)

            if not mesh_path:
                warnings.warn(
                    f"Scene {scene_id!r} is missing a mesh file. "
                    f"Make sure the scene has a {scene_id}_vh_clean_2.ply file.",
                    category=RuntimeWarning,
                )
                continue

            try:
                # If for some reason a scene cannot be loaded (or corrupted), skip it
                data = load_scannet_scene(
                    mesh_path=mesh_path,
                    meta_path=meta_path,
                    aggregation_path=aggregation_path,
                    segments_path=segments_path,
                    label_to_idx=label_to_idx,
                    scene_id=scene_id,
                )
            except Exception as e:
                warnings.warn(f"Error loading scene {scene_id!r}: {e!r}. Skipping...", category=RuntimeWarning)
                continue

            if self.pre_filter is not None and not self.pre_filter(data):
                continue

            if self.pre_transform is not None:
                data = self.pre_transform(data)

            out_path = Path(self.processed_dir, self.split, f"{scene_id}.pt")
            out_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(data, out_path)

    def _load_processed_data(self) -> Any:
        data_list = []
        for path in Path(self.processed_dir, self.split).glob("*.pt"):
            data = torch.load(path, weights_only=True)
            if isinstance(data, dict):
                data_list.append(data)
            else:
                data_list.extend(data)
        return data_list

    @override
    def __getitem__(self, index: int) -> Dict[str, Any]:
        data = self.data[index]
        if self.transform is not None:
            data = self.transform(data)
        return data

    @override
    def __len__(self) -> int:
        return len(self.data)
