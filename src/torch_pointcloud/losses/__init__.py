"""Training criteria for detection, segmentation, and generative models."""

from .anchor import AnchorLoss, MultiHeadAnchorLoss
from .centerpoint import CenterPointLoss, SparseCenterPointLoss
from .chamfer import chamfer_distance
from .kpconv import KPConvDeformRegularizer, kpconv_deform_regularizer
from .lovasz import LovaszLoss
from .pointrcnn import PointRCNNLoss
from .sum import SumLoss
from .threedetr import ThreeDETRLoss
from .transfusion import TransFusionLoss
from .votenet import VoteNetLoss

__all__ = [
    "AnchorLoss",
    "CenterPointLoss",
    "KPConvDeformRegularizer",
    "ThreeDETRLoss",
    "LovaszLoss",
    "MultiHeadAnchorLoss",
    "PointRCNNLoss",
    "SparseCenterPointLoss",
    "SumLoss",
    "TransFusionLoss",
    "VoteNetLoss",
    "chamfer_distance",
    "kpconv_deform_regularizer",
]
