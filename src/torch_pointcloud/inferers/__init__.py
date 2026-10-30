"""Test-time inference strategies for segmentation and detection: tiling, windows, refinement and augmentation."""

import typing
import warnings

from .inferer import DetectionInferer, Inferer, SegmentationInferer
from .knn_window import KNNWindowSegmentationInferer, knn_window_segmentation_inference
from .part_refinement import PartRefinementInferer, part_refinement_inference
from .potential_sphere import PotentialSphereSegmentationInferer, potential_sphere_segmentation_inference
from .simple import SimpleInferer
from .sliding_window import SlidingWindowSegmentationInferer, sliding_window_segmentation_inference
from .tta import TTADetectionInferer, TTASegmentationInferer, simple_tta_transforms
from .voxel_partition import VoxelPartitionSegmentationInferer

__all__ = [
    "DetectionInferer",
    "Inferer",
    "KNNWindowSegmentationInferer",
    "knn_window_segmentation_inference",
    "PartRefinementInferer",
    "part_refinement_inference",
    "PotentialSphereSegmentationInferer",
    "potential_sphere_segmentation_inference",
    "SegmentationInferer",
    "SimpleInferer",
    "SlidingWindowSegmentationInferer",
    "sliding_window_segmentation_inference",
    "TTADetectionInferer",
    "TTASegmentationInferer",
    "simple_tta_transforms",
    "VoxelPartitionSegmentationInferer",
]

_RENAMED = {
    "KNNWindowInferer": "KNNWindowSegmentationInferer",
    "knn_window_inference": "knn_window_segmentation_inference",
    "PotentialSphereInferer": "PotentialSphereSegmentationInferer",
    "potential_sphere_inference": "potential_sphere_segmentation_inference",
    "SlidingWindowInferer": "SlidingWindowSegmentationInferer",
    "sliding_window_inference": "sliding_window_segmentation_inference",
    "TTAInferer": "TTASegmentationInferer",
    "VoxelPartitionInferer": "VoxelPartitionSegmentationInferer",
}


def __getattr__(name: str) -> typing.Any:
    if name in _RENAMED:
        warnings.warn(
            f"`{name}` is deprecated since 0.0.19 and will be removed in 0.0.20; use `{_RENAMED[name]}` instead.",
            FutureWarning,
            stacklevel=2,
        )
        return globals()[_RENAMED[name]]

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
