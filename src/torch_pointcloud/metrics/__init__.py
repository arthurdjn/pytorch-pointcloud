"""Evaluation metrics for classification, segmentation, detection, instance segmentation and registration."""

from .classification import accuracy, confusion_matrix
from .detection import box_average_precision, box_matches
from .instance_segmentation import instance_average_precision, instance_matches
from .kitti import kitti_average_precision
from .nuscenes import nuscenes_detection_metrics
from .registration import (
    alignment_error,
    feature_matching_recall,
    inlier_ratio,
    registration_recall,
    root_mean_square_alignment_error,
    rotation_error,
    translation_error,
)
from .segmentation import intersection_over_union, part_intersection_over_union, part_mean_intersection_over_union

__all__ = [
    "accuracy",
    "alignment_error",
    "box_average_precision",
    "box_matches",
    "confusion_matrix",
    "feature_matching_recall",
    "inlier_ratio",
    "instance_average_precision",
    "instance_matches",
    "intersection_over_union",
    "kitti_average_precision",
    "nuscenes_detection_metrics",
    "part_intersection_over_union",
    "part_mean_intersection_over_union",
    "registration_recall",
    "root_mean_square_alignment_error",
    "rotation_error",
    "translation_error",
]
