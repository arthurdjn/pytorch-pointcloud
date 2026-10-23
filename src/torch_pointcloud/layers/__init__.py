"""Reusable neural network blocks shared across the model architectures."""

from ._modules import (
    ModuleLike,
    ModuleName,
    ModuleRegistryDict,
    RegisteredModuleLike,
    create_module,
)
from .act import create_act
from .affine import Affine, affine
from .anchors import (
    AnchorGroupHead,
    AnchorHead,
    AnchorHeadMultiOutput,
    AnchorHeadOutput,
    MultiGroupAnchorHead,
    prediction_branch,
)
from .bev_backbone import BEVBackbone, BEVResidualBackbone, ResidualBlock2d
from .conv2d_blocks import Conv2dBlock
from .conv3d_blocks import Conv3dBlock
from .dropouts import (
    DropoutLike,
    DropoutName,
    DropPath,
    create_dropout,
    drop_path,
)
from .fps import FPS
from .geometric_affine import GeometricAffineConv, NormalizeType
from .grid_pool import GridPool
from .grouped_vector_attention import GroupedVectorAttention
from .kpconv_blocks import (
    KPConv,
    KPConvBlock,
    KPConvGroupNormBlock,
    KPResidualBlock,
    KPResidualGroupNormBlock,
    LinearGroupNormBlock,
    create_kernel_points,
)
from .linear_blocks import LinearBlock
from .norms import PackedGroupNorm, create_norm
from .octree_attention import OctreeAttention, OctreeRelativePositionEncoding, OctreeT
from .octree_blocks import OctreeConvBlock, OctreeDeconvBlock
from .pdnorm import PDNorm
from .point_patch_embed import PointPatchEmbed
from .point_transformer_conv import PointTransformerConv
from .pointconv_blocks import (
    PointConv,
    PointConvDensity,
    PointConvDensityGlobalSetAbstraction,
    PointConvDensitySetAbstraction,
    PointConvGlobalSetAbstraction,
    PointConvSetAbstraction,
)
from .pointnet2_blocks import (
    PointNet2Conv,
    PointNet2FeaturePropagation,
    PointNet2GlobalSetAbstraction,
    PointNet2SetAbstraction,
)
from .pointnext_blocks import (
    PointNeXtConv,
    PointNeXtResidualBlock,
    PointNeXtSetAbstraction,
)
from .pools import (
    AdaptivePoolLike,
    AdaptivePoolName,
    CatPool,
    MaxPool,
    MeanPool,
    MinPool,
    MulPool,
    PoolLike,
    PoolName,
    SumPool,
    create_adaptive_pool,
    create_pool,
)
from .pvcnn_blocks import PVConv, SE3d, Voxelization
from .randlanet_blocks import (
    AttentivePooling,
    LocalFeatureAggregation,
    LocalSpatialEncoding,
    RandLANetResidualBlock,
)
from .rope import Point3DRoPE
from .serialized_attention import (
    PatchLayout,
    RelativePositionalEncoding,
    SerializedAttention,
    SerializedAttentionRoPE,
    SerializedAttentionRPE,
    patch_layout,
)
from .serialized_pool import SerializedPool, SerializedUpsample
from .spconv_blocks import (
    SparseBasicBlock,
    SparseConvBlock,
    SparseModule,
    SparseResidualBlock,
    SubMConv3dBlock,
    SubMConv3dResidualBlock,
)
from .tnet import DynamicTNet, TNet
from .torchsparse_blocks import TorchSparseConvBlock, TorchSparseResidualBlock
from .transformer import Attention, TransformerBlock, TransformerDecoderLayer, TransformerEncoderLayer
from .vfe import DynamicMeanVFE, PillarFeatureLayer
from .view import View
from .voxel_grid_pool import VoxelGridPool
from .xconv import XConv

__all__ = [
    "ModuleLike",
    "ModuleName",
    "ModuleRegistryDict",
    "RegisteredModuleLike",
    "create_module",
    "create_act",
    "Affine",
    "affine",
    "AnchorGroupHead",
    "AnchorHead",
    "AnchorHeadMultiOutput",
    "AnchorHeadOutput",
    "MultiGroupAnchorHead",
    "prediction_branch",
    "BEVBackbone",
    "BEVResidualBackbone",
    "ResidualBlock2d",
    "Conv2dBlock",
    "Conv3dBlock",
    "DropoutLike",
    "DropoutName",
    "DropPath",
    "create_dropout",
    "drop_path",
    "FPS",
    "GeometricAffineConv",
    "NormalizeType",
    "GridPool",
    "GroupedVectorAttention",
    "KPConv",
    "KPConvBlock",
    "KPConvGroupNormBlock",
    "KPResidualBlock",
    "KPResidualGroupNormBlock",
    "LinearGroupNormBlock",
    "create_kernel_points",
    "LinearBlock",
    "PackedGroupNorm",
    "create_norm",
    "OctreeAttention",
    "OctreeRelativePositionEncoding",
    "OctreeT",
    "OctreeConvBlock",
    "OctreeDeconvBlock",
    "PDNorm",
    "PointPatchEmbed",
    "PointTransformerConv",
    "PointConv",
    "PointConvDensity",
    "PointConvDensityGlobalSetAbstraction",
    "PointConvDensitySetAbstraction",
    "PointConvGlobalSetAbstraction",
    "PointConvSetAbstraction",
    "PointNet2Conv",
    "PointNet2FeaturePropagation",
    "PointNet2GlobalSetAbstraction",
    "PointNet2SetAbstraction",
    "PointNeXtConv",
    "PointNeXtResidualBlock",
    "PointNeXtSetAbstraction",
    "AdaptivePoolLike",
    "AdaptivePoolName",
    "CatPool",
    "MaxPool",
    "MeanPool",
    "MinPool",
    "MulPool",
    "PoolLike",
    "PoolName",
    "SumPool",
    "create_adaptive_pool",
    "create_pool",
    "PVConv",
    "SE3d",
    "Voxelization",
    "AttentivePooling",
    "LocalFeatureAggregation",
    "LocalSpatialEncoding",
    "RandLANetResidualBlock",
    "Point3DRoPE",
    "PatchLayout",
    "RelativePositionalEncoding",
    "SerializedAttention",
    "SerializedAttentionRoPE",
    "SerializedAttentionRPE",
    "patch_layout",
    "SerializedPool",
    "SerializedUpsample",
    "SparseBasicBlock",
    "SparseConvBlock",
    "SparseModule",
    "SparseResidualBlock",
    "SubMConv3dBlock",
    "SubMConv3dResidualBlock",
    "DynamicTNet",
    "TNet",
    "TorchSparseConvBlock",
    "TorchSparseResidualBlock",
    "Attention",
    "TransformerBlock",
    "TransformerDecoderLayer",
    "TransformerEncoderLayer",
    "DynamicMeanVFE",
    "PillarFeatureLayer",
    "View",
    "VoxelGridPool",
    "XConv",
]
