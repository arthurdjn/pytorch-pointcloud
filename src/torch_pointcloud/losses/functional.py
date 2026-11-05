"""Functional API of the losses: the tensor function of every themed module, re-exported here in the spirit of
`torch.nn.functional`. Tensors in, tensors out, no state, no reduction unless the docstring says so."""

from .assignment import assignment_negative_log_likelihood
from .centerpoint import center_regression_loss, iou_prediction_loss
from .chamfer import chamfer_distance
from .circle import weighted_circle_loss
from .corner import corner_loss
from .focal import gaussian_focal_loss, one_hot_foreground, poly1_focal_loss, sigmoid_focal_loss
from .kpconv import kpconv_deform_regularizer
from .lovasz import lovasz_softmax
from .matching import hungarian_match, hungarian_match_batched
from .tnet import tnet_orthogonality_regularizer

__all__ = [
    "assignment_negative_log_likelihood",
    "center_regression_loss",
    "chamfer_distance",
    "corner_loss",
    "gaussian_focal_loss",
    "hungarian_match",
    "hungarian_match_batched",
    "iou_prediction_loss",
    "kpconv_deform_regularizer",
    "lovasz_softmax",
    "one_hot_foreground",
    "poly1_focal_loss",
    "sigmoid_focal_loss",
    "tnet_orthogonality_regularizer",
    "weighted_circle_loss",
]
