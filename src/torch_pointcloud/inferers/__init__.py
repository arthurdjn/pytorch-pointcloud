"""Inference strategies for scenes larger than one forward pass: tiling, windows, and refinement."""

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
