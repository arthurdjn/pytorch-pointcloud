"""Transforms that build an octree and its features."""

from typing import Any, Dict, Optional

from torch_pointcloud.ops.octree import build_octree
from torch_pointcloud.utils.conversion import ensure_tuple_size
from torch_pointcloud.utils.types import KeyCollection

from .base import DictTransform

__all__ = [
    "BuildOctree",
    "OctreeFeatures",
]


class BuildOctree(DictTransform):
    """Build an octree from positions stored in a dictionary.

    === "Object"

        ![BuildOctree on an object](../../assets/transforms/build_octree.png)

    === "Scene"

        ![BuildOctree on a room](../../assets/transforms/build_octree_scene.png)

    Args:
        pos_key: Key holding the point positions.
        dst_octree_key: Key under which the octree is stored.
        depth: Octree depth.
        full_depth: Full depth of the octree.
        batch_size: Batch size.
        normal_key: Key holding surface normals.
        feature_key: Key holding point features.
        label_key: Key holding per-point labels.
        batch_key: Key holding batch indices.
        dst_points_key: Key under which the octree points are stored.
    """

    def __init__(
        self,
        *,
        pos_key: str,
        dst_octree_key: str,
        depth: int,
        full_depth: int = 2,
        batch_size: int = 1,
        normal_key: str | None = None,
        feature_key: str | None = None,
        label_key: str | None = None,
        batch_key: str | None = None,
        dst_points_key: str | None = None,
    ) -> None:
        super().__init__([], False)
        if dst_points_key is not None and dst_points_key == dst_octree_key:
            raise ValueError(f"`dst_points_key` and `dst_octree_key` must be different, got {dst_points_key!r}.")

        self.pos_key = pos_key
        self.depth = depth
        self.dst_octree_key = dst_octree_key
        self.full_depth = full_depth
        self.batch_size = batch_size
        self.normal_key = normal_key
        self.feature_key = feature_key
        self.label_key = label_key
        self.batch_key = batch_key
        self.dst_points_key = dst_points_key

    def transform(self, data: Dict[str, Any]) -> Dict[str, Any]:
        data = dict(data)

        pos = data[self.pos_key]
        normal = data[self.normal_key] if self.normal_key is not None else None
        features = data[self.feature_key] if self.feature_key is not None else None
        batch_id = data[self.batch_key] if self.batch_key is not None else None
        labels = data[self.label_key] if self.label_key is not None else None

        octree, points = build_octree(
            pos=pos,
            normal=normal,
            x=features,
            batch=batch_id,
            labels=labels,
            depth=self.depth,
            full_depth=self.full_depth,
            batch_size=self.batch_size,
            return_points=True,
        )

        data[self.dst_octree_key] = octree
        if self.dst_points_key is not None:
            data[self.dst_points_key] = points

        return data


class OctreeFeatures(DictTransform):
    """Extract per-node features from an octree via `octree.get_input_feature`.

    === "Object"

        ![OctreeFeatures on an object](../../assets/transforms/octree_features.png)

    === "Scene"

        ![OctreeFeatures on a room](../../assets/transforms/octree_features_scene.png)

    Args:
        keys: Keys holding `Octree` instances to extract features from.
        features_type: Feature spec passed to `octree.get_input_feature` (e.g.
            `"ND"` for normals + depth, `"NDFP"` for normals + depth + features
            + position).
        nempty: If `True`, return features only for non-empty nodes; otherwise
            include empty-node padding.
        dst_keys: Where to store the extracted feature tensors. Defaults to `keys`.
        allow_missing_keys: If `True`, silently skip absent keys.
    """

    def __init__(
        self,
        keys: KeyCollection,
        features_type: str,
        nempty: bool = False,
        dst_keys: Optional[KeyCollection] = None,
        allow_missing_keys: bool = False,
    ):
        super().__init__(keys, allow_missing_keys)
        self.dst_keys = ensure_tuple_size(dst_keys or self.keys, len(self.keys))
        self.features_type = features_type
        self.nempty = nempty

    def transform(self, data: Dict[str, Any]) -> Dict[str, Any]:
        data = dict(data)
        for key, dst_key in self.iter_keys(data, self.dst_keys):
            octree = data[key]
            data[dst_key] = octree.get_input_feature(self.features_type, self.nempty)
        return data
