"""Inference strategies for scenes larger than one forward pass: tiling, windows, and refinement."""

from .inferer import DetectionInferer, Inferer, SegmentationInferer
from .knn_window import (
    KNNWindowInferer,
    KNNWindowSegmentationInferer,
    knn_window_inference,
    knn_window_segmentation_inference,
)
from .part_refinement import PartRefinementInferer, part_refinement_inference
from .potential_sphere import (
    PotentialSphereInferer,
    PotentialSphereSegmentationInferer,
    potential_sphere_inference,
    potential_sphere_segmentation_inference,
)
from .simple import SimpleInferer
from .sliding_window import (
    SlidingWindowInferer,
    SlidingWindowSegmentationInferer,
    sliding_window_inference,
    sliding_window_segmentation_inference,
)
from .tta import TTADetectionInferer, TTAInferer, TTASegmentationInferer, simple_tta_transforms
from .voxel_partition import VoxelPartitionInferer, VoxelPartitionSegmentationInferer

__all__ = [
    "DetectionInferer",
    "Inferer",
    "KNNWindowInferer",
    "KNNWindowSegmentationInferer",
    "knn_window_inference",
    "knn_window_segmentation_inference",
    "PartRefinementInferer",
    "part_refinement_inference",
    "PotentialSphereInferer",
    "PotentialSphereSegmentationInferer",
    "potential_sphere_inference",
    "potential_sphere_segmentation_inference",
    "SegmentationInferer",
    "SimpleInferer",
    "SlidingWindowInferer",
    "SlidingWindowSegmentationInferer",
    "sliding_window_inference",
    "sliding_window_segmentation_inference",
    "TTADetectionInferer",
    "TTAInferer",
    "TTASegmentationInferer",
    "simple_tta_transforms",
    "VoxelPartitionInferer",
    "VoxelPartitionSegmentationInferer",
]
