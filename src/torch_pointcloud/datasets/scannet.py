"""
The ScanNet dataset as described in the paper
[ScanNet: Richly-annotated 3D Reconstructions of Indoor Scenes](https://arxiv.org/abs/1702.04405).

"""

import math
import warnings
from collections import defaultdict
from functools import cached_property
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, Sequence, Tuple, TypedDict, Union
from urllib.parse import urljoin
from urllib.request import urlopen

import numpy as np
import pandas as pd
import plyfile
import torch
from torch import Tensor
from tqdm import tqdm
from typing_extensions import NotRequired, override

import torch_pointcloud.transforms as T
from torch_pointcloud.utils.data import DataKeys
from torch_pointcloud.utils.geometry import transform_points, vertex_normals
from torch_pointcloud.utils.io import load_json, load_safetensors, save_safetensors
from torch_pointcloud.utils.misc import parallel_map
from torch_pointcloud.utils.types import PathLike

from .pointcloud import PointCloudDataset
from .utils import download_url

SCANNET_UNK_CLS = "<unk>"
SCANNET_UNK_IDX = 0

SCANNET20_LABELS = [SCANNET_UNK_IDX, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39]
SCANNET200_LABELS = [
    SCANNET_UNK_IDX,
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    11,
    13,
    14,
    15,
    16,
    17,
    18,
    19,
    21,
    22,
    23,
    24,
    26,
    27,
    28,
    29,
    31,
    32,
    33,
    34,
    35,
    36,
    38,
    39,
    40,
    41,
    42,
    44,
    45,
    46,
    47,
    48,
    49,
    50,
    51,
    52,
    54,
    55,
    56,
    57,
    58,
    59,
    62,
    63,
    64,
    65,
    66,
    67,
    68,
    69,
    70,
    71,
    72,
    73,
    74,
    75,
    76,
    77,
    78,
    79,
    80,
    82,
    84,
    86,
    87,
    88,
    89,
    90,
    93,
    95,
    96,
    97,
    98,
    99,
    100,
    101,
    102,
    103,
    104,
    105,
    106,
    107,
    110,
    112,
    115,
    116,
    118,
    120,
    121,
    122,
    125,
    128,
    130,
    131,
    132,
    134,
    136,
    138,
    139,
    140,
    141,
    145,
    148,
    154,
    155,
    156,
    157,
    159,
    161,
    163,
    165,
    166,
    168,
    169,
    170,
    177,
    180,
    185,
    188,
    191,
    193,
    195,
    202,
    208,
    213,
    214,
    221,
    229,
    230,
    232,
    233,
    242,
    250,
    261,
    264,
    276,
    283,
    286,
    300,
    304,
    312,
    323,
    325,
    331,
    342,
    356,
    370,
    392,
    395,
    399,
    408,
    417,
    488,
    540,
    562,
    570,
    572,
    581,
    609,
    748,
    776,
    1156,
    1163,
    1164,
    1165,
    1166,
    1167,
    1168,
    1169,
    1170,
    1171,
    1172,
    1173,
    1174,
    1175,
    1176,
    1178,
    1179,
    1180,
    1181,
    1182,
    1183,
    1184,
    1185,
    1186,
    1187,
    1188,
    1189,
    1190,
    1191,
]


class ScanNetData(TypedDict):
    pos: Tensor
    color: Tensor
    normal: Tensor
    instance: NotRequired[Tensor]
    segment: NotRequired[Tensor]
    scene: NotRequired[str]


def load_scannet_scene_mesh(file_path: PathLike) -> Tuple[Tensor, Tensor]:
    """Load a ScanNet PLY file and return the vertices and face.

    Args:
        file_path: The path to the PLY file.

    Returns:
        The vertices and face.

    Examples:
        >>> vertices, face = load_ply("data/ScanNet/raw/v2/scans/scene0000_00/scene0000_00_vh_clean_2.ply")
    """
    with open(file_path, "rb") as f:
        plydata = plyfile.PlyData.read(f)

    vertices = np.array([tuple(vertex) for vertex in plydata["vertex"].data], dtype=np.float32)
    face = np.stack(plydata["face"].data["vertex_indices"], axis=0)
    return torch.from_numpy(vertices), torch.from_numpy(face).long()


def load_scannet_scene_metadata(meta_path: PathLike, /) -> Dict[str, Any]:
    """Load a ScanNet metadata file and return the metadata.

    Args:
        meta_path: The path to the metadata file, usually saved as `data/ScanNet/raw/v2/scans/{scan_id}/{scan_id}.txt`.

    Returns:
        The metadata.

    Examples:
        >>> meta = load_scannet_scene_meta("data/ScanNet/raw/v2/scans/scene0000_00/scene0000_00.txt")
        >>> meta.keys()
        dict_keys(['axisAlignment', 'colorToDepthExtrinsics', 'colorHeight', 'colorWidth', 'depthHeight', 'depthWidth',
         'fx_color', 'fy_color', 'mx_color', 'my_color', 'numColorFrames', 'numDepthFrames', 'numIMUmeasurements',
         'sceneType'])
    """
    with open(meta_path) as f:
        lines = f.readlines()

    meta: Dict[str, Any] = {}
    for line in lines:
        if "=" not in line:
            continue

        key, val = line.split("=", 1)
        key, val = key.strip(), val.strip()

        if key in ["axisAlignment", "colorToDepthExtrinsics"]:
            matrix = np.fromstring(val, sep=" ", dtype=np.float32).reshape(4, 4)
            meta[key] = torch.from_numpy(matrix)
        elif val.isdigit():
            meta[key] = int(val)
        elif val.replace(".", "").isdigit():
            meta[key] = float(val)
        else:
            meta[key] = val

    return meta


def load_scannet_scene_aggregation_and_segs(
    aggregation_path: PathLike,
    segs_path: PathLike,
    label_to_idx: Optional[Dict[str, int]] = None,
) -> Tuple[Tensor, Optional[Tensor]]:
    """Read per-vertex instance ids and semantic labels from aggregation + segments.

    Args:
        aggregation_path: Path to the aggregation JSON file.
        segments_path: Path to the segments JSON file.
        label_to_idx: Optional mapping from `raw_category` string to NYU40 id
            (or any integer label). Built from the TSV with e.g.
            `{row["raw_category"]: int(row["nyu40id"]) for _, row in df.iterrows()}`.
            If provided, per-vertex semantic labels are returned.
            Unrecognized categories map to 0 (unlabeled).

    Returns:
        The instance and labels.

    Examples:
        >>> scene_dir = "data/ScanNet/raw/v2/scans/scene0000_00"
        >>> instance, labels = load_scannet_scene_aggregation_and_segs(
        ...     f"{scene_dir}/scene0000_00.aggregation.json",
        ...     f"{scene_dir}/scene0000_00.segs.json",
        ...     label_to_idx={
        ...         "chair": 1,
        ...         "floor": 2,
        ...         "wall": 3,
        ...         ...,
        ...     },
        ... )
    """
    aggregation = load_json(aggregation_path)
    segments = load_json(segs_path)

    seg_indices = np.array(segments["segIndices"])
    num_vertices = len(seg_indices)

    # segment id -> list of vertex indices
    seg_to_verts: Dict[int, list[int]] = defaultdict(list)
    for vi, seg_id in enumerate(seg_indices):
        seg_to_verts[seg_id].append(vi)

    instance = np.full(num_vertices, SCANNET_UNK_IDX, dtype=np.int32)
    labels = np.full(num_vertices, SCANNET_UNK_IDX, dtype=np.int32) if label_to_idx is not None else None

    for group in aggregation["segGroups"]:
        object_id = group["objectId"]
        raw_label = group["label"]
        label = label_to_idx.get(raw_label, 0) if label_to_idx is not None else raw_label

        for seg_id in group["segments"]:
            for vi in seg_to_verts.get(seg_id, []):
                instance[vi] = object_id
                if labels is not None:
                    labels[vi] = label

    return (
        torch.from_numpy(instance),
        torch.from_numpy(labels) if labels is not None else None,
    )


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
    use_axis_alignment: bool = True,
) -> ScanNetData:
    """Load a ScanNet scene and return the parsed points, color, normal, instance, and labels
    in a dictionary format.

    Args:
        mesh_path: Path to the raw mesh file, usually saved as `data/ScanNet/raw/v2/scans/{scan_id}/{scan_id}.ply`.
        meta_path: Path to the metadata file, usually saved as `data/ScanNet/raw/v2/scans/{scan_id}/{scan_id}.txt`.
        aggregation_path: Path to the aggregation file, usually saved as `data/ScanNet/raw/v2/scans/{scan_id}/{scan_id}.aggregation.json`.
        segments_path: Path to the segments file, usually saved as `data/ScanNet/raw/v2/scans/{scan_id}/{scan_id}.segs.json`.
        label_to_idx: A dictionary mapping object labels to contiguous positive indices. The labels correspond to the `raw_category` column
            in the labels CSV file, or to the `label` key in the aggregation JSON file.
            This mapping is used to map object labels to their associated target indices.
        use_axis_alignment: Whether to apply the axis alignment transformation from the
            scene metadata.  Set to ``False`` to keep the raw PLY coordinates.

    Returns:
        The loaded scene.

    Examples:
        >>> labels_path = "data/ScanNet/raw/metadata/scannetv2-labels.combined.tsv"
        >>> labels = load_scannet_labels(labels_path)
        >>> label_to_idx = {label: idx for idx, label in enumerate(labels["raw_category"].unique())}
        >>> scene_dir = "data/ScanNet/raw/v2/scans/scene0000_00"
        >>> scene = load_scannet_scene(
        ...     mesh_path=f"{scene_dir}/scene0000_00_vh_clean_2.ply",
        ...     meta_path=f"{scene_dir}/scene0000_00.txt",
        ...     aggregation_path=f"{scene_dir}/scene0000_00.aggregation.json",
        ...     segments_path=f"{scene_dir}/scene0000_00.segs.json",
        ...     label_to_idx=label_to_idx,
        ... )
        >>> scene
        {'points': tensor([[...]]), 'color': tensor([[...]]), 'normal': tensor([[...]]),
         'instance': tensor([...]), 'labels': tensor([...])}}
    """
    label_to_idx = label_to_idx or {}

    # Load the points
    vertices, face = load_scannet_scene_mesh(mesh_path)
    pos, color = vertices[:, :3], vertices[:, 3:6]

    # Optionally transform the points with the axis alignment matrix
    if use_axis_alignment:
        metadata = load_scannet_scene_metadata(meta_path) if meta_path else {}
        if "axisAlignment" in metadata:
            pos = transform_points(pos, metadata["axisAlignment"])

    normal = vertex_normals(pos, face)

    data: ScanNetData = {
        "pos": pos,
        "color": color,
        "normal": normal,
    }

    if aggregation_path and segments_path:
        instance, segment = load_scannet_scene_aggregation_and_segs(
            aggregation_path,
            segments_path,
            label_to_idx=label_to_idx,
        )
        data["instance"] = instance
        if segment is not None:
            data["segment"] = segment

    if scene_id:
        data["scene"] = scene_id

    return data


def tile_scannet_scene(
    scene: Dict[str, Any],
    block_size: float = 1.5,
    block_stride: float = 0.75,
    num_nodes: int = 8192,
    min_num_nodes: int = 100,
    scene_index: Optional[int] = None,
) -> List[Dict[str, Any]]:
    r"""Split a single ScanNet scene dict into fixed-size spatial blocks.

    Sweeps a $\text{block\_size} \times \text{block\_size}$ window (full Z extent) over
    the scene with the given stride, matching the tiling procedure used in the
    DGCNN ScanNet evaluation protocol.

    Args:
        scene: Dict with at least ``DataKeys.POS`` (float32, $(N, 3)$).
            All other tensors with a leading dimension of $N$ are sliced in parallel.
        block_size: Side length of each square block in meters.
        block_stride: Step size for the sliding window in meters.
        num_nodes: Fixed number of nodes per block.
        min_num_nodes: Minimum number of raw nodes for a block to be kept.
        scene_index: If provided, each block will include ``"scene_index"`` and
            ``"num_scene_points"`` entries.

    Returns:
        List of dicts, one per retained block.  Each block has exactly
        ``num_nodes`` nodes and an extra ``"scene_max"`` key with the scene-level
        coordinate maxima (useful for downstream normalization).
    """
    pos = scene[DataKeys.POS]
    N = pos.shape[0]

    pos_min = pos.min(dim=0).values
    pos_max = pos.max(dim=0).values

    num_block_x = max(math.ceil((pos_max[0].item() - pos_min[0].item() - block_size) / block_stride) + 1, 1)
    num_block_y = max(math.ceil((pos_max[1].item() - pos_min[1].item() - block_size) / block_stride) + 1, 1)

    x = pos[:, 0]
    y = pos[:, 1]

    blocks: List[Dict[str, Any]] = []
    for i in range(num_block_x):
        for j in range(num_block_y):
            s_x = pos_min[0].item() + i * block_stride
            e_x = min(s_x + block_size, pos_max[0].item())
            s_x = e_x - block_size
            s_y = pos_min[1].item() + j * block_stride
            e_y = min(s_y + block_size, pos_max[1].item())
            s_y = e_y - block_size

            mask = (x >= s_x - 1e-8) & (x <= e_x + 1e-8) & (y >= s_y - 1e-8) & (y <= e_y + 1e-8)
            indices = mask.nonzero(as_tuple=True)[0]
            n = indices.numel()
            if n < min_num_nodes:
                continue

            if n >= num_nodes:
                chosen = indices[torch.randperm(n)[:num_nodes]]
            else:
                chosen = indices[torch.randint(0, n, (num_nodes,))]

            block: Dict[str, Any] = {}
            for key, val in scene.items():
                if isinstance(val, Tensor) and val.shape[0] == N:
                    block[key] = val[chosen].clone()
                else:
                    block[key] = val

            block["scene_max"] = pos_max
            block["block_center"] = torch.tensor([s_x + block_size / 2.0, s_y + block_size / 2.0, 0.0], dtype=pos.dtype)
            block["point_indices"] = chosen
            if scene_index is not None:
                block["scene_index"] = scene_index
                block["num_scene_points"] = N
            blocks.append(block)

    return blocks


class ScanNet(PointCloudDataset):
    """The ScanNet dataset as described in the paper
    [ScanNet: Richly-annotated 3D Reconstructions of Indoor Scenes](https://arxiv.org/abs/1702.04405).
    This dataset contains 2.5M views in 1513 scans acquired in 707 distinct spaces.
    Each scan is annotated with 3D camera poses, meshes, object segmentation, and scene semantics for
    a total of 36,000 annotated object instance.

    The dataset is available in two versions:

    - `v1`: The original dataset with 1,513 scans.
    - `v2`: Improved annotation coverage to ~90% (from 63% in v1),
        with 100 more scans for test.

    Note:
        It is recommended to use the `v2` version, as it contains more annotated object instance.
        The `v1` version is kept for backward compatibility.

    Note:
        By default, the labels are taken from the `nyu40class` column in the labels CSV file,
        and the `nyu40id` column is used to sort the labels. Note than the `class_to_idx` property
        returns a dictionary mapping the class name to the contiguous index, and indices
        may not correspond to the `nyu40id` values.

        In most cases, the loaded labels are contiguous; see `class_to_idx` for the mapping from
        class name to index (indices may not match raw `nyu40id` values in the source files).

    Args:
        root: The root directory of the dataset.
        version: The version of the dataset to use.
        split: The split to load, one of `train`, `val`, or `test`.
        label_name: The name of the label column in the labels CSV file.
        label_id: The name of the id column in the labels CSV file.
        use_axis_alignment: If True, apply ScanNet's axis-alignment transform to the mesh.
        block_size: If set, split each scene into ground-plane blocks of this size (meters) for training.
        block_stride: Stride between adjacent blocks when `block_size` is set.
        num_nodes: Number of points sampled per block (when `block_size` is set) or per scene.
        min_num_nodes: Skip blocks with fewer than this many points.
        transform: A callable that transforms the data when retrieved from the dataset.
        download: Whether to download the raw data.
        force_download: Whether to force the download of the raw data.
        force_process: Whether to force the processing of the raw data.
        show_progress: Whether to show a progress bar during processing.
        num_workers: Worker processes for preprocessing, or `None` for sequential processing.

    Example:
        Assuming you have downloaded the raw dataset from http://kaldir.vc.in.tum.de/scannet/,
        and extracted it under `data/ScanNet/raw`, you can load the dataset as follows:

        ```python
        from torch_pointcloud.datasets import ScanNet

        dataset = ScanNet(
            root="data/ScanNet/raw",
            version="v2",
            split="train",
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

    unk_cls = "<unk>"
    unk_idx = 0

    data_url = "http://kaldir.vc.in.tum.de/scannet/"
    meta_url = "https://raw.githubusercontent.com/facebookresearch/votenet/master/scannet/meta_data/"
    label_resources = [
        "v1/tasks/scannet-labels.combined.tsv",  # v1 raw labels
        "v2/tasks/scannetv2-labels.combined.tsv",  # v2 raw labels
    ]
    scan_ids_resource = "{version}/scans.txt"
    scan_resources = [
        # "v1/scans/{scan_id}/{scan_id}.sens",  # NOTE: The `.sens` file from the v2 version is the same as the v1
        "{version}/scans/{scan_id}/{scan_id}.aggregation.json",
        "{version}/scans/{scan_id}/{scan_id}.txt",
        "{version}/scans/{scan_id}/{scan_id}_vh_clean_2.0.010000.segs.json",
        "{version}/scans/{scan_id}/{scan_id}_vh_clean_2.ply",
        # "{version}/scans/{scan_id}/{scan_id}_vh_clean.ply",
        # "{version}/scans/{scan_id}/{scan_id}_vh_clean.segs.json",
        "{version}/scans/{scan_id}/{scan_id}_vh_clean.aggregation.json",
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
        root: PathLike,
        version: Literal["v1", "v2"] = "v2",
        split: Literal["train", "test", "val"] = "train",
        label_name: str = "nyu40class",
        label_id: str = "nyu40id",
        use_axis_alignment: bool = True,
        block_size: Optional[float] = None,
        block_stride: float = 0.75,
        num_nodes: int = 8192,
        min_num_nodes: int = 100,
        transform: Optional[Callable] = None,
        download: bool = False,
        force_download: bool = False,
        force_process: bool = False,
        show_progress: bool = True,
        num_workers: Optional[int] = None,
    ) -> None:
        super().__init__(root)
        if split not in ["train", "val", "test"]:
            raise ValueError(f"Invalid split {split!r}, expected one of 'train', 'val' or 'test'.")

        self.version = version
        self.split = split
        self.label_name = label_name
        self.label_id = label_id
        self.use_axis_alignment = use_axis_alignment
        self.transform = transform
        self.show_progress = show_progress

        if download or force_download:
            self.download(force=force_download)

        self.process(force=force_process, num_workers=num_workers, show_progress=show_progress)
        self.load(
            show_progress=show_progress,
            block_size=block_size,
            block_stride=block_stride,
            num_nodes=num_nodes,
            min_num_nodes=min_num_nodes,
        )

    @cached_property
    def labels(self) -> pd.DataFrame:
        resource_path = self.label_resources[int(self.version == "v2")]
        resource_path = resource_path.format(version=self.version)
        labels_path = Path(self.raw_dir, resource_path)
        return load_scannet_labels(labels_path)

    @cached_property
    def classes(self) -> List[str]:
        if self.labels is None:
            classes = []
        else:
            df = self.labels.sort_values(self.label_id)
            classes = df[self.label_name].unique().tolist()

        classes.insert(0, self.unk_cls)
        return classes

    @cached_property
    def class_to_idx(self) -> Dict[str, int]:
        return {cls: idx for idx, cls in enumerate(self.classes)}

    def raw_files_exist(self) -> bool:
        # Check that the labels file exists
        label_resource = self.label_resources[int(self.version == "v2")].format(version=self.version)
        labels_path = Path(self.raw_dir, label_resource)
        if not labels_path.exists():
            return False

        # Check that the scans directory exists
        version_dir = self.version if self.split in ["train", "val"] else "v2"
        scans_dir = Path(self.raw_dir, version_dir, "scans")
        if not scans_dir.exists():
            return False

        # Check that there is at least one scene directory
        scene_dirs = list(scans_dir.glob("scene*"))
        if len(scene_dirs) == 0:
            return False

        return True

    @property
    def processed_dir(self) -> str:
        base = Path(self.data_dir, "processed")
        if not self.use_axis_alignment:
            base = Path(str(base) + "_noalign")
        return base.absolute().as_posix()

    @property
    def processed_files(self) -> List[Path]:
        return sorted(list(Path(self.processed_dir, self.split).glob("*.safetensors")))

    def processed_files_exist(self) -> bool:
        return len(self.processed_files) > 0

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

    def process(self, force: bool = False, num_workers: Optional[int] = None, show_progress: bool = True) -> None:
        if self.processed_files_exist() and not force:
            return
        if not self.raw_files_exist():
            raise RuntimeError(
                f"Dataset not found at {self.root!r}. "
                f"You can download the raw dataset from {self.data_url!r}, "
                f"and extract it under {self.raw_dir!r}."
            )

        raw_dir = Path(self.raw_dir)

        # Create the mapping between object labels (also named "raw_category" in the CSV labels) and indices
        raw_col = "raw_category" if self.version == "v2" else "category"

        # Two-step mapping: raw_category -> self.label_name (e.g. nyu40class) -> contiguous index.
        # A direct raw_category lookup in class_to_idx is wrong when label_name != label_col
        # (e.g. label_name="nyu40class"): "couch" would miss "sofa", "fridge" would miss
        # "refrigerator", etc. Iterating the TSV rows provides the correct intermediate mapping.
        label_to_idx = dict(zip(self.labels[raw_col].values, self.labels[self.label_id].values))
        label_to_idx[self.unk_cls] = self.unk_idx

        # Look for the associated scene IDs for the specified split
        is_train_val = self.split in ["train", "val"]
        scans_dir = Path(raw_dir, self.version if is_train_val else "v2", "scans")
        split_file = raw_dir / "metadata" / f"scannetv2_{self.split}.txt"
        with open(split_file) as f:
            scene_ids = sorted([line.strip() for line in f])

        def process_scene(scene_id: str) -> None:
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
                return

            try:
                data = load_scannet_scene(
                    mesh_path=mesh_path,
                    meta_path=meta_path,
                    aggregation_path=aggregation_path,
                    segments_path=segments_path,
                    label_to_idx=label_to_idx,
                    scene_id=scene_id,
                    use_axis_alignment=self.use_axis_alignment,
                )
            except Exception as e:
                warnings.warn(f"Error loading scene {scene_id!r}: {e!r}. Skipping...", category=RuntimeWarning)
                return

            dst_path = Path(self.processed_dir, self.split, f"{scene_id}.safetensors")
            dst_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(data, dst_path)
            save_safetensors(dst_path, data)

        parallel_map(
            process_scene,
            scene_ids,
            num_workers=num_workers,
            total=len(scene_ids),
            desc="Processing",
            show_progress=show_progress,
        )

    def load(
        self,
        block_size: Optional[float] = None,
        block_stride: float = 0.75,
        num_nodes: int = 8192,
        min_num_nodes: int = 100,
        show_progress: bool = True,
    ) -> None:
        self.data: List[Dict[str, Any]] = []
        self.scene_boundaries: List[int] = []
        for scene_idx, path in tqdm(
            enumerate(self.processed_files),
            desc="Loading",
            total=len(self.processed_files),
            disable=not show_progress,
        ):
            scene = load_safetensors(path)
            if block_size is not None and block_size > 0:
                blocks = tile_scannet_scene(
                    scene,
                    block_size=block_size,
                    block_stride=block_stride,
                    num_nodes=num_nodes,
                    min_num_nodes=min_num_nodes,
                    scene_index=scene_idx,
                )
                self.data.extend(blocks)
            else:
                self.data.append(scene)
            self.scene_boundaries.append(len(self.data))

    @override
    def __getitem__(self, index: int) -> Dict[str, Any]:
        data = self.data[index]
        if self.transform is not None:
            data = self.transform(data)
        return data

    @override
    def __len__(self) -> int:
        return len(self.data)


class ScanNet20(ScanNet):
    def __init__(
        self,
        root: PathLike,
        version: Literal["v1", "v2"] = "v2",
        split: Literal["train", "test", "val"] = "train",
        use_axis_alignment: bool = True,
        block_size: Optional[float] = None,
        block_stride: float = 0.75,
        num_nodes: int = 8192,
        min_num_nodes: int = 100,
        transform: Optional[Callable] = None,
        download: bool = False,
        force_download: bool = False,
        force_process: bool = False,
        show_progress: bool = True,
        num_workers: Optional[int] = None,
    ) -> None:
        self.relabel = T.Relabel(keys=DataKeys.SEGMENT, labels=SCANNET20_LABELS)
        super().__init__(
            root=root,
            version=version,
            split=split,
            label_name="nyu40class",
            label_id="nyu40id",
            use_axis_alignment=use_axis_alignment,
            block_size=block_size,
            block_stride=block_stride,
            num_nodes=num_nodes,
            min_num_nodes=min_num_nodes,
            transform=transform,
            download=download,
            force_download=force_download,
            force_process=force_process,
            show_progress=show_progress,
            num_workers=num_workers,
        )

    @override
    @property
    def name(self) -> str:
        return "ScanNet"

    @override
    @property
    def processed_dir(self) -> str:
        return super().processed_dir + "_20"

    @override
    def load(
        self,
        block_size: Optional[float] = None,
        block_stride: float = 0.75,
        num_nodes: int = 8192,
        min_num_nodes: int = 100,
        show_progress: bool = True,
    ) -> None:
        self.data: List[Dict[str, Any]] = []
        self.scene_boundaries: List[int] = []
        for scene_idx, path in tqdm(
            enumerate(self.processed_files),
            desc="Loading",
            total=len(self.processed_files),
            disable=not show_progress,
        ):
            scene = load_safetensors(path)
            scene = self.relabel(scene)
            if block_size is not None and block_size > 0:
                blocks = tile_scannet_scene(
                    scene,
                    block_size=block_size,
                    block_stride=block_stride,
                    num_nodes=num_nodes,
                    min_num_nodes=min_num_nodes,
                    scene_index=scene_idx,
                )
                self.data.extend(blocks)
            else:
                self.data.append(scene)
            self.scene_boundaries.append(len(self.data))


class ScanNet200(ScanNet):
    def __init__(
        self,
        root: str,
        version: Literal["v1", "v2"] = "v2",
        split: Literal["train", "test", "val"] = "train",
        use_axis_alignment: bool = True,
        block_size: Optional[float] = None,
        block_stride: float = 0.75,
        num_nodes: int = 8192,
        min_num_nodes: int = 100,
        transform: Optional[Callable] = None,
        download: bool = False,
        force_download: bool = False,
        force_process: bool = False,
        show_progress: bool = True,
        num_workers: Optional[int] = None,
    ) -> None:
        self.relabel = T.Relabel(keys=DataKeys.SEGMENT, labels=SCANNET200_LABELS)
        super().__init__(
            root=root,
            version=version,
            split=split,
            label_name="raw",
            label_id="id",
            use_axis_alignment=use_axis_alignment,
            block_size=block_size,
            block_stride=block_stride,
            num_nodes=num_nodes,
            min_num_nodes=min_num_nodes,
            transform=transform,
            download=download,
            force_download=force_download,
            force_process=force_process,
            show_progress=show_progress,
            num_workers=num_workers,
        )

    @override
    @property
    def name(self) -> str:
        return "ScanNet"

    @override
    @property
    def processed_dir(self) -> str:
        return super().processed_dir + "_200"

    @override
    def load(
        self,
        block_size: Optional[float] = None,
        block_stride: float = 0.75,
        num_nodes: int = 8192,
        min_num_nodes: int = 100,
        show_progress: bool = True,
    ) -> None:
        self.data: List[Dict[str, Any]] = []
        self.scene_boundaries: List[int] = []
        for scene_idx, path in tqdm(
            enumerate(self.processed_files),
            desc="Loading",
            total=len(self.processed_files),
            disable=not show_progress,
        ):
            scene = load_safetensors(path)
            scene = self.relabel(scene)
            if block_size is not None and block_size > 0:
                blocks = tile_scannet_scene(
                    scene,
                    block_size=block_size,
                    block_stride=block_stride,
                    num_nodes=num_nodes,
                    min_num_nodes=min_num_nodes,
                    scene_index=scene_idx,
                )
                self.data.extend(blocks)
            else:
                self.data.append(scene)
            self.scene_boundaries.append(len(self.data))
