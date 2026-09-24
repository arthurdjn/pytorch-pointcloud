"""Evaluation metrics for classification, segmentation, detection and instance segmentation."""

from .classification import accuracy, confusion_matrix
from .detection import box_average_precision, box_matches
from .instance_segmentation import instance_average_precision, instance_matches
from .kitti import kitti_average_precision
from .nuscenes import nuscenes_detection_metrics
from .segmentation import intersection_over_union, part_intersection_over_union, part_mean_intersection_over_union

__all__ = [
    "accuracy",
    "box_average_precision",
    "box_matches",
    "confusion_matrix",
    "instance_average_precision",
    "instance_matches",
    "intersection_over_union",
    "kitti_average_precision",
    "nuscenes_detection_metrics",
    "part_intersection_over_union",
    "part_mean_intersection_over_union",
]
