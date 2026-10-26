"""Model architectures, the `create_model` factory, and the pretrained weight registry."""

from ._base import (
    ClassificationModel,
    DetectionModel,
    PartSegmentationModel,
    PretrainingModel,
    SemanticSegmentationModel,
)
from ._registry import (
    DeprecatedModelDict,
    ModelDict,
    ModelInfoDict,
    Task,
    WeightsDict,
    create_model,
    deprecated_model,
    list_models,
    model_info,
    register_model,
)
from .centerpoint import CenterPointDetection, CenterPointPillarsDetection
from .concerto import ConcertoSegmentation
from .dgcnn import DGCNNClassification, DGCNNPartSegmentation, DGCNNSegmentation
from .kpconv import KPFCNNClassification, KPFCNNSegmentation
from .lion import LIONDetection
from .octformer import OctFormerClassification, OctFormerSegmentation
from .parta2 import PartA2Detection
from .point_bert import PointBERTClassification, PointBERTDiscreteVAE, PointBERTPretraining
from .point_m2ae import PointM2AEClassification, PointM2AEPartSegmentation, PointM2AEPretraining
from .point_mae import PointMAEClassification, PointMAEPartSegmentation, PointMAEPretraining
from .point_mamba import PointMambaClassification, PointMambaPretraining
from .point_transformer import PointTransformerClassification, PointTransformerSegmentation
from .point_transformer_v2 import PointTransformerV2Classification, PointTransformerV2Segmentation
from .point_transformer_v3 import PointTransformerV3Classification, PointTransformerV3Segmentation
from .pointcnn import PointCNNClassification, PointCNNSegmentation
from .pointconv import PointConvDensityClassification
from .pointgpt import PointGPTClassification, PointGPTPretraining
from .pointmlp import PointMLPClassification, PointMLPSegmentation
from .pointnet import PointNetClassification, PointNetSegmentation
from .pointnet2 import PointNet2Classification, PointNet2PartSegmentation, PointNet2Segmentation
from .pointnext import PointNeXtClassification, PointNeXtPartSegmentation, PointNeXtSegmentation
from .pointpillars import PointPillarsDetection, PointPillarsMultiHeadDetection
from .pointrcnn import PointRCNNDetection
from .pv_rcnn import PVRCNNDetection
from .pvcnn import PVCNNClassification, PVCNNPartSegmentation, PVCNNSegmentation
from .pvcnn2 import PVCNN2Classification, PVCNN2Segmentation
from .randlanet import RandLANetClassification, RandLANetSegmentation
from .second import SECONDDetection, SECONDMultiHeadDetection
from .sonata import SonataSegmentation
from .spformer_unet import SPFormerUNetSegmentation
from .sphereformer import SphereFormerSegmentation
from .spunet import SparseUNetSegmentation
from .spvcnn import SPVCNNClassification, SPVCNNSegmentation
from .threedetr import ThreeDETRDetection
from .utonia import UtoniaSegmentation
from .votenet import VoteNetDetection
from .voxel_mamba import VoxelMambaDetection
from .voxelnext import VoxelNeXtDetection

__all__ = [
    "ClassificationModel",
    "DetectionModel",
    "PartA2Detection",
    "PartSegmentationModel",
    "PretrainingModel",
    "SemanticSegmentationModel",
    "DeprecatedModelDict",
    "ModelDict",
    "ModelInfoDict",
    "Task",
    "WeightsDict",
    "create_model",
    "deprecated_model",
    "list_models",
    "model_info",
    "register_model",
    "CenterPointDetection",
    "CenterPointPillarsDetection",
    "ConcertoSegmentation",
    "DGCNNClassification",
    "DGCNNPartSegmentation",
    "DGCNNSegmentation",
    "KPFCNNClassification",
    "KPFCNNSegmentation",
    "LIONDetection",
    "OctFormerClassification",
    "OctFormerSegmentation",
    "PointBERTClassification",
    "PointBERTDiscreteVAE",
    "PointBERTPretraining",
    "PointM2AEClassification",
    "PointM2AEPartSegmentation",
    "PointM2AEPretraining",
    "PointMAEClassification",
    "PointMAEPartSegmentation",
    "PointMAEPretraining",
    "PointMambaClassification",
    "PointMambaPretraining",
    "PointTransformerClassification",
    "PointTransformerSegmentation",
    "PointTransformerV2Classification",
    "PointTransformerV2Segmentation",
    "PointTransformerV3Classification",
    "PointTransformerV3Segmentation",
    "PointCNNClassification",
    "PointCNNSegmentation",
    "PointConvDensityClassification",
    "PointGPTClassification",
    "PointGPTPretraining",
    "PointMLPClassification",
    "PointMLPSegmentation",
    "PointNetClassification",
    "PointNetSegmentation",
    "PointNet2Classification",
    "PointNet2PartSegmentation",
    "PointNet2Segmentation",
    "PointNeXtClassification",
    "PointNeXtPartSegmentation",
    "PointNeXtSegmentation",
    "PointPillarsDetection",
    "PointPillarsMultiHeadDetection",
    "PointRCNNDetection",
    "PVRCNNDetection",
    "PVCNNClassification",
    "PVCNNPartSegmentation",
    "PVCNNSegmentation",
    "PVCNN2Classification",
    "PVCNN2Segmentation",
    "RandLANetClassification",
    "RandLANetSegmentation",
    "SECONDDetection",
    "SECONDMultiHeadDetection",
    "SonataSegmentation",
    "SPFormerUNetSegmentation",
    "SphereFormerSegmentation",
    "SparseUNetSegmentation",
    "SPVCNNClassification",
    "SPVCNNSegmentation",
    "ThreeDETRDetection",
    "UtoniaSegmentation",
    "VoteNetDetection",
    "VoxelMambaDetection",
    "VoxelNeXtDetection",
]
