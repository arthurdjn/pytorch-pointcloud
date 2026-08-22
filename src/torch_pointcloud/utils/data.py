"""Data loading: standard sample keys, packed-batch collation, and the point cloud data loader."""

import functools
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from torch_pointcloud.utils.conversion import ensure_tuple
from torch_pointcloud.utils.imports import _OCNN_GITHUB_URL, optional_import
from torch_pointcloud.utils.types import KeyCollection, StrEnum

if TYPE_CHECKING:
    import ocnn
    from ocnn.octree import Octree, Points


ocnn, _OCNN_AVAILABLE = optional_import("ocnn", url=_OCNN_GITHUB_URL)
Octree, _ = optional_import("ocnn.octree", "Octree", url=_OCNN_GITHUB_URL)
Points, _ = optional_import("ocnn.octree", "Points", url=_OCNN_GITHUB_URL)


class DataKeys(StrEnum):
    """Standard keys of a sample dict, shared by the datasets, transforms and models.

    Members are plain strings, so a key is usable wherever a literal is (e.g. `data[DataKeys.POS]` and
    `data["pos"]` address the same entry).
    """

    # General keys (PyG convention)
    X = "x"
    POS = "pos"
    POS_GRID = "pos_grid"
    POS_VOXEL = "pos_voxel"
    VOXEL = "voxel"
    VOXEL_NUM_POINTS = "voxel_num_points"
    COLOR = "color"
    NORMAL = "normal"
    FACE = "face"
    SEGMENT = "segment"
    SEMANTIC = "semantic"
    INSTANCE = "instance"
    SUPERPOINT = "superpoint"
    INTENSITY = "intensity"
    REFLECTANCE = "reflectance"
    CATEGORY = "category"
    LABEL = "label"
    BATCH = "batch"
    INVERSE = "inverse"
    BOX = "box"
    BATCH_BOX = "batch_box"
    VELOCITY = "velocity"
    NUM_POINTS = "num_points"
    ATTRIBUTE = "attribute"
    TRUNCATION = "truncation"
    OCCLUSION = "occlusion"
    BBOX_HEIGHT = "bbox_height"
    CALIB = "calib"
    IMAGE_SHAPE = "image_shape"
    FRAME = "frame"
    TIMESTAMP = "timestamp"
    GPS_TIME = "gps_time"
    TOKEN = "token"
    CONDITION = "condition"
    NORM_POS = "norm_pos"
    # Sample-identity keys
    NAME = "name"
    SCENE = "scene"
    SEQUENCE = "sequence"
    # Tiling keys (fixed-size block datasets)
    ROOM_MAX = "room_max"
    SCENE_MAX = "scene_max"
    BLOCK_CENTER = "block_center"
    POINT_INDICES = "point_indices"
    SCENE_INDEX = "scene_index"
    NUM_SCENE_POINTS = "num_scene_points"
    # Octree-based keys (OCNN convention)
    OCTREE = "octree"
    POINTS = "points"
    BOX_MASK = "box_mask"


def _tails_equal(tensors: List[Tensor]) -> bool:
    if tensors[0].ndim < 1:
        return False

    tail = tensors[0].shape[1:]
    return all(t.ndim >= 1 and t.shape[1:] == tail for t in tensors)


def _collate_value(values: List[Any]) -> Any:
    first = values[0]

    if isinstance(first, Tensor):
        if first.ndim == 0:
            return torch.stack(values)
        if _tails_equal(values):
            return torch.cat(values, dim=0)
        return list(values)

    if isinstance(first, (bool, int, float)):
        return torch.tensor(values)

    if _OCNN_AVAILABLE and isinstance(first, Points):
        return ocnn.octree.merge_points(values)

    if _OCNN_AVAILABLE and isinstance(first, Octree):
        octree = ocnn.octree.merge_octrees(values)
        octree.construct_all_neigh()
        return octree

    return list(values)


def _leading_size(value: Any) -> int:
    if isinstance(value, Tensor):
        return value.shape[0] if value.ndim >= 1 else 1
    if isinstance(value, (str, bytes)):
        return 1
    if hasattr(value, "__len__"):
        return len(value)
    return 1


def collate(
    data_list: List[Dict[str, Any]],
    batch_from: str = DataKeys.POS,
    batch_key: str = DataKeys.BATCH,
    stack_keys: Optional[KeyCollection] = None,
    cat_keys: Optional[KeyCollection] = None,
) -> Dict[str, Any]:
    r"""Collate a list of point-cloud sample dicts into one batched dict.

    By default every per-point tensor is concatenated PyG-style along dim 0 (packed), scalars are
    stacked, and a per-point `batch_key` index is synthesized from `batch_from`. Two extra knobs say
    how specific keys collate instead:

    - `stack_keys`: stack to a new leading batch dim ($(M, \cdot) \to (B, M, \cdot)$, $(N, \cdot) \to (B, N, \cdot)$)
      rather than concatenating. Used for fixed-size per-scene ground truth (the VoteNet loss consumes
      dense $(B, M, \cdot)$ targets, which a plain cat would flatten).
    - `cat_keys`: keep these packed (cat) but additionally emit a `batch_<key>` scene index mirroring
      `batch_key`. Used for ragged per-scene ground truth such as `box` $(K, 8)$ -> `batch_box` $(K,)$.

    Every key must be present in every sample; a key missing from a sample raises a `ValueError`.
    `stack_keys` / `cat_keys` entries absent from all samples are ignored. A key may appear in only
    one of `stack_keys` / `cat_keys`; overlapping entries raise a `ValueError`.

    Args:
        data_list: List of sample dicts.
        batch_from: Key whose leading dimension defines the per-point batch index.
        batch_key: Output key for the per-point batch index.
        stack_keys: Keys collated by stacking to a leading batch dim instead of concatenating.
        cat_keys: Packed keys that additionally emit a `batch_<key>` per-element scene index.

    Returns:
        A single batched dict.
    """
    stack_keys = ensure_tuple(stack_keys)
    cat_keys = ensure_tuple(cat_keys)
    overlap = sorted((set(stack_keys) & set(cat_keys)) - {None})
    if overlap:
        raise ValueError(f"Keys cannot be in both `stack_keys` and `cat_keys`: {overlap}.")
    if not data_list:
        return {}

    stacked = [k for k in stack_keys if all(k in d for d in data_list)]
    keys = list(dict.fromkeys(k for d in data_list for k in d))
    out: Dict[str, Any] = {}
    for k in keys:
        for i, d in enumerate(data_list):
            if k not in d:
                raise ValueError(f"Cannot collate key {k!r}: missing from sample {i}.")
        values = [d[k] for d in data_list]
        out[k] = torch.stack(values, dim=0) if k in stacked else _collate_value(values)

    for src, dst in ((batch_from, batch_key), *((k, f"batch_{k}") for k in cat_keys)):
        if all(src in d for d in data_list):
            lengths = [_leading_size(d[src]) for d in data_list]
            first = data_list[0][src]
            device = first.device if torch.is_tensor(first) else None
            out[dst] = torch.cat([torch.full((n,), i, dtype=torch.long, device=device) for i, n in enumerate(lengths)])

    return out


class PointCloudDataLoader(DataLoader):
    r"""`DataLoader` that batches point clouds with the packed-batch `collate` by default.

    Wraps `torch.utils.data.DataLoader`, defaulting `collate_fn` to
    [`collate`][torch_pointcloud.utils.data.collate]. How keys collate is set by the spec arguments
    (`batch_from` / `batch_key` for the per-point index, `stack_keys` for dense per-scene ground
    truth, `cat_keys` for ragged per-scene ground truth). These are supplied by the caller, never
    read off the dataset: transforms rewrite the key set downstream of the dataset (a `box` key may
    be derived from an object by a transform), so only the code building the loader knows which keys
    must stack or cat. Passing `collate_fn=...` via the usual `DataLoader` kwarg overrides the spec.

    Args:
        dataset: The dataset to load from.
        batch_from: Key whose leading dimension defines the per-point batch index.
        batch_key: Output key for the per-point batch index.
        stack_keys: Keys collated by stacking to a leading batch dim instead of concatenating.
        cat_keys: Packed keys that additionally emit a `batch_<key>` per-element scene index.
        **kwargs: Forwarded to `torch.utils.data.DataLoader` (`batch_size`, `shuffle`, `collate_fn`, ...).
    """

    def __init__(
        self,
        dataset: Dataset,
        *,
        batch_from: str = DataKeys.POS,
        batch_key: str = DataKeys.BATCH,
        stack_keys: Optional[Sequence[str]] = None,
        cat_keys: Optional[Sequence[str]] = None,
        **kwargs: Any,
    ) -> None:
        collate_fn = functools.partial(
            collate,
            batch_from=batch_from,
            batch_key=batch_key,
            stack_keys=stack_keys,
            cat_keys=cat_keys,
        )
        kwargs.setdefault("collate_fn", collate_fn)
        super().__init__(dataset, **kwargs)
