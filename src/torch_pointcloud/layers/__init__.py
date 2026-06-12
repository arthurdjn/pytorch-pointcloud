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
    AnchorHeadMulti,
    AnchorHeadMultiOutput,
    AnchorHeadOutput,
    AnchorHeadSingle,
    MultiGroupSingleHead,
    generate_anchors,
    limit_period,
    separate_branch,
)
from .bev_backbone import BaseBEVBackbone
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
from .heads import create_cls_head, create_seg_head
from .layer_container import LayerContainer
from .linear_blocks import LinearBlock
from .norms import create_norm
from .octree_attention import RPE, OctreeAttention, OctreeT
from .octree_blocks import OctreeConvBlock, OctreeDeconvBlock
from .pointconv import PointConv, PointConvDensity
from .pointnet2_blocks import (
    FPModule,
    GlobalSAModule,
    PointNet2Conv,
    PointNet2FeaturePropagation,
    PointNet2GlobalSetAbstraction,
    PointNet2SetAbstraction,
    SAModule,
    ensure_msg_list,
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
    LogSoftmaxPool,
    MaxPool,
    MeanPool,
    MinPool,
    MulPool,
    PoolLike,
    PoolName,
    SoftmaxPool,
    SumPool,
    create_adaptive_pool,
    create_pool,
)
from .pvcnn_blocks import PVConv
from .reshape import Reshape
from .rope import Point3DRoPE
from .serialized_attention import (
    RelativePositionalEncoding,
    SerializedAttention,
    SerializedAttentionRoPE,
    SerializedAttentionRPE,
)
from .serialized_pool import SerializedPool, SerializedUpsample
from .spconv_blocks import SubMConv3dBlock
from .tnet import DynamicTNet, TNet
from .transformer import Attention, TransformerBlock
from .view import View
from .xconv import XConv

# NOTE: `pointconv_sa` is intentionally not re-exported here because it imports
# `LinearBlock` from `torch_pointcloud.models.pointmlp`, creating a layers->models
# cycle. Use `from torch_pointcloud.layers.pointconv_sa import ...` until that
# cycle is broken (planned for the next refactor pass).
