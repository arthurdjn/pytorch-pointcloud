"""Functional API of the transforms: the tensor functions of every themed module."""

from .augmentation import (
    color_auto_contrast,
    color_grayscale,
    color_jitter,
    color_shift,
    flip_boxes,
    flip_vectors,
    random_color_drop,
    random_color_jitter,
    random_elastic_distortion,
    random_jitter,
    rotate_boxes,
    scale_boxes,
    translate_boxes,
)
from .box import angle_to_class, class_to_angle, class_to_size, points_in_oriented_box
from .geometry import axis_min_offset, estimate_normals, quantize, rotate_vectors, rotation_matrix, shift
from .masking import apply_mask, bounding_box, box_mask, cube_mask, remove_near_origin, sphere_mask
from .mixing import laser_mix_masks, polar_mix_masks
from .sampling import (
    farthest_point_sample,
    random_dropout_mask,
    random_sample,
    random_sample_face_vertices,
    shuffle_indices,
)
from .scaling import minimal_enclosing_ball, normalize, rescale
from .utils import absolute, relabel
from .voxelization import divisible_pad, split_batch

__all__ = [
    "absolute",
    "angle_to_class",
    "apply_mask",
    "axis_min_offset",
    "bounding_box",
    "box_mask",
    "class_to_angle",
    "class_to_size",
    "color_auto_contrast",
    "color_grayscale",
    "color_jitter",
    "color_shift",
    "cube_mask",
    "divisible_pad",
    "estimate_normals",
    "farthest_point_sample",
    "flip_boxes",
    "flip_vectors",
    "laser_mix_masks",
    "minimal_enclosing_ball",
    "normalize",
    "points_in_oriented_box",
    "polar_mix_masks",
    "quantize",
    "random_color_drop",
    "random_color_jitter",
    "random_dropout_mask",
    "random_elastic_distortion",
    "random_jitter",
    "random_sample",
    "random_sample_face_vertices",
    "relabel",
    "remove_near_origin",
    "rescale",
    "rotate_boxes",
    "rotate_vectors",
    "rotation_matrix",
    "scale_boxes",
    "shift",
    "translate_boxes",
    "shuffle_indices",
    "sphere_mask",
    "split_batch",
]
