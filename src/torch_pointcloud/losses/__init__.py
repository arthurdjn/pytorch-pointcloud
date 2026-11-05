r"""Training criteria.

Generic losses are plain `(input, target)` modules, each next to its tensor function, which
[`functional`][torch_pointcloud.losses.functional] re-exports. The detection losses are named after the head (or
model) whose output they read and assign their targets from the packed `box` / `label` / `batch_box` ground truth;
their `forward(output, data)` returns a dict with the scalar `loss` to optimize plus its detached named terms
(`<term>_loss`). The generic losses return a tensor; a training loop accepts either.
"""

from . import functional
from .anchor import AnchorHeadLoss, MultiGroupAnchorHeadLoss
from .centerpoint import CenterHeadLoss, VoxelNeXtHeadLoss, center_regression_loss, iou_prediction_loss
from .chamfer import ChamferDistance, chamfer_distance
from .corner import CornerLoss, corner_loss
from .fcaf3d import FCAF3DLoss
from .focal import (
    GaussianFocalLoss,
    Poly1FocalLoss,
    SigmoidFocalLoss,
    gaussian_focal_loss,
    one_hot_foreground,
    poly1_focal_loss,
    sigmoid_focal_loss,
)
from .iou import BoxIoULoss, box_iou_loss
from .kpconv import KPConvDeformRegularizer, kpconv_deform_regularizer
from .lovasz import LovaszLoss, lovasz_softmax
from .matching import hungarian_match, hungarian_match_batched
from .pointrcnn import PointRCNNLoss
from .sum import SumLoss
from .threedetr import ThreeDETRLoss
from .tnet import TNetOrthogonalityRegularizer, tnet_orthogonality_regularizer
from .tr3d import TR3DLoss
from .transfusion import TransFusionHeadLoss
from .votenet import VoteNetLoss

__all__ = [
    "AnchorHeadLoss",
    "BoxIoULoss",
    "CenterHeadLoss",
    "ChamferDistance",
    "CornerLoss",
    "FCAF3DLoss",
    "GaussianFocalLoss",
    "Poly1FocalLoss",
    "KPConvDeformRegularizer",
    "LovaszLoss",
    "MultiGroupAnchorHeadLoss",
    "PointRCNNLoss",
    "SigmoidFocalLoss",
    "SumLoss",
    "TNetOrthogonalityRegularizer",
    "TR3DLoss",
    "ThreeDETRLoss",
    "TransFusionHeadLoss",
    "VoteNetLoss",
    "VoxelNeXtHeadLoss",
    "box_iou_loss",
    "center_regression_loss",
    "chamfer_distance",
    "corner_loss",
    "functional",
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
]
