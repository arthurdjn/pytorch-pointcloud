"""LitePT semantic segmentation models.

{{ paper("2512.13689") }}
"""

from typing import TYPE_CHECKING, Any, Callable, Dict, List, Literal, Optional, Sequence, Tuple, Union, overload

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing_extensions import NotRequired

import torch_pointcloud.transforms as T
from torch_pointcloud.datasets.nuscenes import NUSCENES_LIDARSEG_CLASSES, NUSCENES_LIDARSEG_LEARNING_MAP
from torch_pointcloud.datasets.scannet import SCANNET20_CLASSES
from torch_pointcloud.layers.act import create_act
from torch_pointcloud.layers.dropouts import DropPath
from torch_pointcloud.layers.grid_pool import GridPool
from torch_pointcloud.layers.norms import create_norm
from torch_pointcloud.layers.serialized_attention import PatchLayout, SerializedAttentionRoPE
from torch_pointcloud.layers.serialized_pool import SerializedUpsample
from torch_pointcloud.layers.spconv_blocks import SubMConv3dBlock
from torch_pointcloud.models._base import SemanticSegmentationModel
from torch_pointcloud.models._registry import WeightsDict, register_model
from torch_pointcloud.models.point_transformer_v3 import serialize
from torch_pointcloud.ops.serialization import SerializationOrder
from torch_pointcloud.utils.conversion import convert_to_spconv_tensor, ensure_tuple, ensure_tuple_size
from torch_pointcloud.utils.data import DataKeys
from torch_pointcloud.utils.imports import _SPCONV_GITHUB_URL, optional_import
from torch_pointcloud.utils.state_dict import transform_state_dict
from torch_pointcloud.utils.types import OptTensor, PooledFeaturesDict, ValueCollection

if TYPE_CHECKING:
    import spconv.pytorch as spconv


spconv, _ = optional_import("spconv.pytorch", url=_SPCONV_GITHUB_URL)


_STRUCTURED3D_CLASSES = (
    "wall",
    "floor",
    "cabinet",
    "bed",
    "chair",
    "sofa",
    "table",
    "door",
    "window",
    "picture",
    "desk",
    "shelves",
    "curtain",
    "dresser",
    "pillow",
    "mirror",
    "ceiling",
    "refrigerator",
    "television",
    "nightstand",
    "sink",
    "lamp",
    "otherstructure",
    "otherfurniture",
    "otherprop",
)


class LitePTFeaturesDict(PooledFeaturesDict):
    r"""`PooledFeaturesDict` of a LitePT encoder stage, carrying the serialization of the stages that attend.

    Attributes:
        serialized_order: Permutation sorting the stage's points along each order, shape $(L, N_s)$. Only set for a
            stage with attention blocks: the convolution stages never serialize.
        serialized_inverse: Inverse of `serialized_order`, shape $(L, N_s)$.
    """

    serialized_order: NotRequired[Tensor]
    serialized_inverse: NotRequired[Tensor]


class LitePTBlock(nn.Module):
    r"""LitePT block: a sparse-convolution residual, or a layer norm, then optionally pre-normed serialized attention
    with 3D rotary position embedding and an MLP.

    The early (fine) stages of LitePT keep only the residual sparse convolution of a Point Transformer V3 block
    (`conv=True`, `attention=False`); the late (coarse) stages keep only its attention and MLP, preceded by a layer
    norm in place of the convolution (`conv=False`, `attention=True`). The attention rotates $Q$ and $K$ with
    `Point3DRoPE` indexed by the integer grid coordinates of the stage.

    Args:
        channels: Feature width.
        num_heads: Number of attention heads.
        patch_size: Number of points per attention patch.
        mlp_ratio: Hidden-to-input width ratio of the MLP.
        qkv_bias: Whether the QKV projection has a bias.
        qk_scale: Attention logit scale; `None` uses $1 / \sqrt{\text{head\_dim}}$.
        attn_drop: Dropout on the attention weights.
        proj_drop: Dropout after the attention and MLP projections.
        drop_path: Stochastic depth rate of the attention and MLP branches.
        act: Activation of the MLP, passed to `create_act`.
        act_kwargs: Extra keyword arguments for the activation.
        cpe_indice_key: spconv index key shared by the convolutions of one stage.
        use_flash_attn: Whether attention runs through flash attention.
        upcast_attn: Whether the manual attention path computes the logits in fp32.
        upcast_softmax: Whether the manual attention path computes the softmax in fp32.
        rope_base: Frequency base of the rotary position embedding.
        conv: Whether the block starts with the residual sparse convolution; otherwise with a layer norm.
        attention: Whether the block has the attention and MLP branches.

    Shape:
        - Input: $(N, C)$ features, their `SparseConvTensor`, $(N, 3)$ integer grid coordinates, $(N,)$ batch.
        - Output: $(N, C)$ features and the `SparseConvTensor` holding them.
    """

    def __init__(
        self,
        channels: int,
        num_heads: int,
        patch_size: int = 1024,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_scale: Optional[float] = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        drop_path: float = 0.0,
        act: Union[str, Callable] = "gelu",
        act_kwargs: Optional[Dict[str, Any]] = None,
        cpe_indice_key: Optional[str] = None,
        use_flash_attn: bool = True,
        upcast_attn: bool = False,
        upcast_softmax: bool = False,
        rope_base: float = 100.0,
        conv: bool = True,
        attention: bool = True,
    ) -> None:
        if not conv and not attention:
            raise ValueError(
                "A LitePT block needs its convolution, its attention, or both; got conv=False, attention=False."
            )

        super().__init__()
        act_kwargs = act_kwargs or {}
        self.channels = channels
        self.use_conv = conv
        self.use_attention = attention

        if conv:
            self.cpe_conv = spconv.SubMConv3d(
                channels,
                channels,
                kernel_size=3,
                bias=True,
                indice_key=cpe_indice_key,
            )
            self.cpe_proj = nn.Linear(channels, channels)
            self.cpe_norm: nn.Module = create_norm("layer_norm", channels, mode="node") or nn.Identity()
        else:
            self.norm0: nn.Module = create_norm("layer_norm", channels, mode="node") or nn.Identity()

        if attention:
            self.norm1: nn.Module = create_norm("layer_norm", channels, mode="node") or nn.Identity()
            self.attn = SerializedAttentionRoPE(
                channels=channels,
                num_heads=num_heads,
                patch_size=patch_size,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                attn_drop=attn_drop,
                proj_drop=proj_drop,
                use_flash_attn=use_flash_attn,
                upcast_attn=upcast_attn,
                upcast_softmax=upcast_softmax,
                rope_base=rope_base,
            )
            self.norm2: nn.Module = create_norm("layer_norm", channels, mode="node") or nn.Identity()
            self.mlp = nn.Sequential(
                nn.Linear(channels, int(channels * mlp_ratio)),
                create_act(act, **act_kwargs) or nn.Identity(),
                nn.Dropout(proj_drop),
                nn.Linear(int(channels * mlp_ratio), channels),
                nn.Dropout(proj_drop),
            )
            self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(
        self,
        x: Tensor,
        x_sparse: Any,
        pos_grid: Tensor,
        batch: Tensor,
        serialized_order: OptTensor = None,
        serialized_inverse: OptTensor = None,
        patch_layout: Optional[PatchLayout] = None,
    ) -> Tuple[Tensor, Any]:
        if self.use_conv:
            x_sparse = self.cpe_conv(x_sparse)
            x_conv = self.cpe_proj(x_sparse.features)
            x = x + self.cpe_norm(x_conv)
        else:
            x = self.norm0(x)

        if self.use_attention:
            shortcut = x
            x = self.norm1(x)
            x = self.attn(
                x,
                pos_grid,
                batch,
                serialized_order=serialized_order,
                serialized_inverse=serialized_inverse,
                pos=pos_grid.float(),
                patch_layout=patch_layout,
            )
            x = shortcut + self.drop_path(x)

            shortcut = x
            x = self.norm2(x)
            x = self.mlp(x)
            x = shortcut + self.drop_path(x)

        x_sparse = x_sparse.replace_feature(x)
        return x, x_sparse


def _run_blocks(
    blocks: nn.ModuleList,
    x: Tensor,
    pos_grid: Tensor,
    batch: Tensor,
    serialized_order: OptTensor,
    serialized_inverse: OptTensor,
) -> Tensor:
    """Run the `LitePTBlock` units of one stage, cycling the attention blocks through the serialization orders."""
    x_sparse = convert_to_spconv_tensor(x, pos_grid, batch)
    layout: Optional[PatchLayout] = None
    attention_blocks = [block for block in blocks if isinstance(block, LitePTBlock) and block.use_attention]
    if attention_blocks:
        if serialized_order is None or serialized_inverse is None:
            raise ValueError("A stage with attention blocks needs `serialized_order` and `serialized_inverse`.")

        layout = attention_blocks[0].attn.patch_layout(batch)
    for i, block in enumerate(blocks):
        order_idx = i % len(serialized_order) if serialized_order is not None else 0
        x, x_sparse = block(
            x,
            x_sparse,
            pos_grid,
            batch,
            serialized_order=serialized_order[order_idx] if serialized_order is not None else None,
            serialized_inverse=serialized_inverse[order_idx] if serialized_inverse is not None else None,
            patch_layout=layout,
        )
    return x


class LitePTEncoderBlock(nn.Module):
    """One encoder stage: an optional grid pooling, then `depth` `LitePTBlock` units.

    A stage with attention blocks serializes its points along the serialization orders after pooling, and its
    consecutive blocks cycle through those orders; a convolution-only stage never serializes.
    """

    def __init__(
        self,
        channels: int,
        depth: int,
        num_heads: int,
        patch_size: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_scale: Optional[float] = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        drop_path: ValueCollection[float] = 0.0,
        act: Union[str, Callable] = "gelu",
        act_kwargs: Optional[Dict[str, Any]] = None,
        use_flash_attn: bool = True,
        upcast_attn: bool = False,
        upcast_softmax: bool = False,
        rope_base: float = 100.0,
        conv: bool = True,
        attention: bool = False,
        cpe_indice_key: Optional[str] = None,
        downsample: Optional[GridPool] = None,
        serialization_orders: Sequence[SerializationOrder] = ("z", "z-trans", "hilbert", "hilbert-trans"),
        shuffle_serialization_orders: bool = False,
    ) -> None:
        super().__init__()
        self.channels = channels
        self.use_attention = attention
        self.downsample = downsample
        self.serialization_orders = ensure_tuple(serialization_orders)
        self.shuffle_serialization_orders = shuffle_serialization_orders
        drop_path = ensure_tuple_size(drop_path, size=depth)

        self.blocks = nn.ModuleList()
        for i in range(depth):
            self.blocks.append(
                LitePTBlock(
                    channels=channels,
                    num_heads=num_heads,
                    patch_size=patch_size,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    attn_drop=attn_drop,
                    proj_drop=proj_drop,
                    drop_path=drop_path[i],
                    act=act,
                    act_kwargs=act_kwargs,
                    cpe_indice_key=cpe_indice_key,
                    use_flash_attn=use_flash_attn,
                    upcast_attn=upcast_attn,
                    upcast_softmax=upcast_softmax,
                    rope_base=rope_base,
                    conv=conv,
                    attention=attention,
                )
            )

    def forward(
        self,
        x: Tensor,
        pos_grid: Tensor,
        batch: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor, OptTensor, OptTensor, OptTensor]:
        inverse: OptTensor = None
        if self.downsample is not None:
            x, pos_grid, batch, inverse, _ = self.downsample(x, pos_grid, batch)

        serialized_order: OptTensor = None
        serialized_inverse: OptTensor = None
        if self.use_attention and len(self.blocks):
            _, serialized_order, serialized_inverse = serialize(
                pos_grid,
                batch,
                orders=self.serialization_orders,
                shuffle=self.shuffle_serialization_orders and self.training,
            )

        x = _run_blocks(self.blocks, x, pos_grid, batch, serialized_order, serialized_inverse)
        return x, pos_grid, batch, inverse, serialized_order, serialized_inverse


class LitePTDecoderBlock(nn.Module):
    """One decoder stage: an upsampling onto the skip resolution, then `depth` `LitePTBlock` units.

    The blocks reuse the serialization of the encoder stage they upsample to; when that stage did not attend, a
    decoder stage with attention serializes the skip points itself.
    """

    def __init__(
        self,
        channels: int,
        depth: int,
        num_heads: int,
        patch_size: int,
        upsample: SerializedUpsample,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_scale: Optional[float] = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        drop_path: ValueCollection[float] = 0.0,
        act: Union[str, Callable] = "gelu",
        act_kwargs: Optional[Dict[str, Any]] = None,
        use_flash_attn: bool = True,
        upcast_attn: bool = False,
        upcast_softmax: bool = False,
        rope_base: float = 100.0,
        conv: bool = True,
        attention: bool = False,
        cpe_indice_key: Optional[str] = None,
        serialization_orders: Sequence[SerializationOrder] = ("z", "z-trans", "hilbert", "hilbert-trans"),
        shuffle_serialization_orders: bool = False,
    ) -> None:
        super().__init__()
        self.channels = channels
        self.use_attention = attention
        self.upsample = upsample
        self.serialization_orders = ensure_tuple(serialization_orders)
        self.shuffle_serialization_orders = shuffle_serialization_orders
        drop_path = ensure_tuple_size(drop_path, size=depth)

        self.blocks = nn.ModuleList()
        for i in range(depth):
            self.blocks.append(
                LitePTBlock(
                    channels=channels,
                    num_heads=num_heads,
                    patch_size=patch_size,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    attn_drop=attn_drop,
                    proj_drop=proj_drop,
                    drop_path=drop_path[i],
                    act=act,
                    act_kwargs=act_kwargs,
                    cpe_indice_key=cpe_indice_key,
                    use_flash_attn=use_flash_attn,
                    upcast_attn=upcast_attn,
                    upcast_softmax=upcast_softmax,
                    rope_base=rope_base,
                    conv=conv,
                    attention=attention,
                )
            )

    def forward(
        self,
        x: Tensor,
        x_skip: Tensor,
        pos_grid_skip: Tensor,
        batch_skip: Tensor,
        inverse: Tensor,
        serialized_order_skip: OptTensor = None,
        serialized_inverse_skip: OptTensor = None,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        x = self.upsample(x, x_skip, inverse)
        if self.use_attention and len(self.blocks) and serialized_order_skip is None:
            _, serialized_order_skip, serialized_inverse_skip = serialize(
                pos_grid_skip,
                batch_skip,
                orders=self.serialization_orders,
                shuffle=self.shuffle_serialization_orders and self.training,
            )

        x = _run_blocks(self.blocks, x, pos_grid_skip, batch_skip, serialized_order_skip, serialized_inverse_skip)
        return x, pos_grid_skip, batch_skip


class LitePTEncoder(nn.Module):
    r"""LitePT encoder backbone.

    A sparse-convolution stem, then stages separated by grid pooling: the fine stages run residual sparse
    convolutions only, the coarse stages serialized attention with 3D rotary position embedding only.

    Args:
        in_channels: Number of input channels.
        serialization_orders: Space-filling curves the attention stages serialize along.
        shuffle_serialization_orders: Whether to shuffle the orders at every forward in training.
        strides: Grid pooling stride between consecutive stages.
        encoder_depths: Number of blocks per stage.
        encoder_channels: Feature width per stage.
        encoder_num_heads: Attention heads per stage.
        encoder_patch_size: Attention patch size per stage.
        encoder_conv: Whether the blocks of each stage start with the residual sparse convolution.
        encoder_attention: Whether the blocks of each stage have the attention and MLP branches.
        norm: Normalization of the stem and poolings, passed to `create_norm`.
        act: Activation, passed to `create_act`.
        act_kwargs: Extra keyword arguments for the activation.
        norm_kwargs: Extra keyword arguments for the normalization.
        mlp_ratio: Hidden-to-input width ratio of the block MLPs.
        qkv_bias: Whether the QKV projections have a bias.
        qk_scale: Attention logit scale; `None` uses $1 / \sqrt{\text{head\_dim}}$.
        attn_drop: Dropout on the attention weights.
        proj_drop: Dropout after the attention and MLP projections.
        drop_path: Stochastic depth rate, increasing linearly over the blocks.
        rope_base: Frequency base of the rotary position embedding.
        use_flash_attn: Whether attention runs through flash attention, which requires `flash-attn` and a CUDA
            device; pass `use_flash_attn=False` to run without it.
        upcast_attn: Whether the manual attention path computes the logits in fp32.
        upcast_softmax: Whether the manual attention path computes the softmax in fp32.

    Shape:
        - Input: $(N, \text{in\_channels})$ features, $(N, 3)$ integer grid coordinates, $(N,)$ batch.
        - Output: $(N_L, C_L)$ features of the coarsest stage with their grid coordinates and batch.
    """

    def __init__(
        self,
        in_channels: int = 6,
        serialization_orders: Sequence[SerializationOrder] = ("z", "z-trans", "hilbert", "hilbert-trans"),
        shuffle_serialization_orders: bool = True,
        strides: Sequence[int] = (2, 2, 2, 2),
        encoder_depths: Sequence[int] = (2, 2, 2, 6, 2),
        encoder_channels: Sequence[int] = (36, 72, 144, 252, 504),
        encoder_num_heads: Sequence[int] = (2, 4, 8, 14, 28),
        encoder_patch_size: Sequence[int] = (1024, 1024, 1024, 1024, 1024),
        encoder_conv: Sequence[bool] = (True, True, True, False, False),
        encoder_attention: Sequence[bool] = (False, False, False, True, True),
        norm: Union[str, Callable] = "batch_norm",
        act: Union[str, Callable] = "gelu",
        act_kwargs: Optional[Dict[str, Any]] = None,
        norm_kwargs: Optional[Dict[str, Any]] = None,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_scale: Optional[float] = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        drop_path: float = 0.3,
        rope_base: float = 100.0,
        use_flash_attn: bool = True,
        upcast_attn: bool = False,
        upcast_softmax: bool = False,
    ) -> None:
        super().__init__()
        depths = ensure_tuple(encoder_depths)
        n = len(depths)
        channels = ensure_tuple_size(encoder_channels, size=n)
        num_heads = ensure_tuple_size(encoder_num_heads, size=n)
        patch_sizes = ensure_tuple_size(encoder_patch_size, size=n)
        convs = ensure_tuple_size(encoder_conv, size=n)
        attentions = ensure_tuple_size(encoder_attention, size=n)
        strides = ensure_tuple_size(strides, size=n - 1)
        drop_paths = torch.split(torch.linspace(0, drop_path, sum(depths)), list(depths))

        self.in_channels = in_channels
        self.serialization_orders = ensure_tuple(serialization_orders)
        self.shuffle_serialization_orders = shuffle_serialization_orders

        self.stem = SubMConv3dBlock(
            in_channels,
            channels[0],
            kernel_size=5,
            padding=1,
            norm=norm,
            act=act,
            act_kwargs=act_kwargs,
            norm_kwargs=norm_kwargs,
            bias=False,
        )

        self.blocks = nn.ModuleList()
        for i in range(n):
            downsample: Optional[GridPool] = None
            if i > 0:
                downsample = GridPool(
                    in_channels=channels[i - 1],
                    out_channels=channels[i],
                    stride=strides[i - 1],
                    act=act,
                    norm=norm,
                    act_kwargs=act_kwargs,
                    norm_kwargs=norm_kwargs,
                )

            self.blocks.append(
                LitePTEncoderBlock(
                    channels=channels[i],
                    depth=depths[i],
                    num_heads=num_heads[i],
                    patch_size=patch_sizes[i],
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    attn_drop=attn_drop,
                    proj_drop=proj_drop,
                    drop_path=drop_paths[i].tolist(),
                    act=act,
                    act_kwargs=act_kwargs,
                    use_flash_attn=use_flash_attn,
                    upcast_attn=upcast_attn,
                    upcast_softmax=upcast_softmax,
                    rope_base=rope_base,
                    conv=convs[i],
                    attention=attentions[i],
                    cpe_indice_key=f"stage{i}",
                    downsample=downsample,
                    serialization_orders=self.serialization_orders,
                    shuffle_serialization_orders=shuffle_serialization_orders,
                )
            )

    @property
    def embedding_dim(self) -> int:
        """Feature dimension $C$ of the encoder output."""
        last = self.blocks[-1]
        assert isinstance(last, LitePTEncoderBlock)
        return last.channels

    @overload
    def forward(
        self,
        x: Tensor,
        pos_grid: Tensor,
        batch: Tensor,
        *,
        return_intermediates: Literal[True],
    ) -> Tuple[Tensor, Tensor, Tensor, List[LitePTFeaturesDict]]: ...

    @overload
    def forward(
        self,
        x: Tensor,
        pos_grid: Tensor,
        batch: Tensor,
        return_intermediates: Literal[False] = False,
    ) -> Tuple[Tensor, Tensor, Tensor]: ...

    def forward(
        self,
        x: Tensor,
        pos_grid: Tensor,
        batch: Tensor,
        return_intermediates: bool = False,
    ) -> Any:
        x = self.stem(x, pos_grid, batch)

        serialized_order: OptTensor = None
        serialized_inverse: OptTensor = None
        intermediates: List[LitePTFeaturesDict] = []
        for i, block in enumerate(self.blocks):
            intermediate: LitePTFeaturesDict = {"x": x, "pos_grid": pos_grid, "batch": batch}
            if serialized_order is not None and serialized_inverse is not None:
                intermediate["serialized_order"] = serialized_order
                intermediate["serialized_inverse"] = serialized_inverse

            x, pos_grid, batch, inverse, serialized_order, serialized_inverse = block(x, pos_grid, batch)

            if return_intermediates and i > 0:
                assert inverse is not None
                intermediate["pooling_inverse"] = inverse
                intermediates.append(intermediate)

        if return_intermediates:
            return x, pos_grid, batch, intermediates
        return x, pos_grid, batch


class LitePTDecoder(nn.Module):
    r"""LitePT decoder: grid unpooling through the encoder skips, with optional blocks at every resolution.

    Args:
        encoder_channels: Feature width of every encoder stage, to size the skip projections.
        decoder_depths: Number of blocks per stage, from the coarsest to the finest resolution.
        decoder_channels: Feature width per stage.
        decoder_num_heads: Attention heads per stage.
        decoder_patch_size: Attention patch size per stage.
        decoder_conv: Whether the blocks of each stage start with the residual sparse convolution.
        decoder_attention: Whether the blocks of each stage have the attention and MLP branches.
        serialization_orders: Space-filling curves an attention stage serializes along when its encoder skip did not.
        shuffle_serialization_orders: Whether to shuffle those orders at every forward in training.
        norm: Normalization of the unpooling projections, passed to `create_norm`.
        act: Activation, passed to `create_act`.
        act_kwargs: Extra keyword arguments for the activation.
        norm_kwargs: Extra keyword arguments for the normalization.
        mlp_ratio: Hidden-to-input width ratio of the block MLPs.
        qkv_bias: Whether the QKV projections have a bias.
        qk_scale: Attention logit scale; `None` uses $1 / \sqrt{\text{head\_dim}}$.
        attn_drop: Dropout on the attention weights.
        proj_drop: Dropout after the attention and MLP projections.
        drop_path: Stochastic depth rate, increasing linearly over the blocks.
        rope_base: Frequency base of the rotary position embedding.
        use_flash_attn: Whether attention runs through flash attention.
        upcast_attn: Whether the manual attention path computes the logits in fp32.
        upcast_softmax: Whether the manual attention path computes the softmax in fp32.

    Shape:
        - Input: $(N_L, C_L)$ coarsest encoder features and the encoder intermediates.
        - Output: $(N, C_0)$ features at the finest resolution with their grid coordinates and batch.
    """

    def __init__(
        self,
        encoder_channels: Sequence[int] = (36, 72, 144, 252, 504),
        decoder_depths: Sequence[int] = (0, 0, 0, 0),
        decoder_channels: Sequence[int] = (252, 144, 72, 72),
        decoder_num_heads: Sequence[int] = (14, 8, 4, 4),
        decoder_patch_size: Sequence[int] = (1024, 1024, 1024, 1024),
        decoder_conv: Sequence[bool] = (False, False, False, False),
        decoder_attention: Sequence[bool] = (False, False, False, False),
        serialization_orders: Sequence[SerializationOrder] = ("z", "z-trans", "hilbert", "hilbert-trans"),
        shuffle_serialization_orders: bool = True,
        norm: Union[str, Callable] = "batch_norm",
        act: Union[str, Callable] = "gelu",
        act_kwargs: Optional[Dict[str, Any]] = None,
        norm_kwargs: Optional[Dict[str, Any]] = None,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_scale: Optional[float] = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        drop_path: float = 0.3,
        rope_base: float = 100.0,
        use_flash_attn: bool = True,
        upcast_attn: bool = False,
        upcast_softmax: bool = False,
    ) -> None:
        super().__init__()
        depths = ensure_tuple(decoder_depths)
        n = len(depths)
        encoder_channels = ensure_tuple_size(encoder_channels, size=n + 1)
        channels = (encoder_channels[-1],) + ensure_tuple_size(decoder_channels, size=n)
        skip_channels = tuple(encoder_channels[:-1])[::-1]
        num_heads = ensure_tuple_size(decoder_num_heads, size=n)
        patch_sizes = ensure_tuple_size(decoder_patch_size, size=n)
        convs = ensure_tuple_size(decoder_conv, size=n)
        attentions = ensure_tuple_size(decoder_attention, size=n)
        drop_paths = torch.split(torch.linspace(0, drop_path, sum(depths)), list(depths))[::-1]

        self.blocks = nn.ModuleList()
        for i in range(n):
            upsample = SerializedUpsample(
                in_channels=channels[i],
                skip_channels=skip_channels[i],
                out_channels=channels[i + 1],
                norm=norm,
                act=act,
                act_kwargs=act_kwargs,
                norm_kwargs=norm_kwargs,
            )
            self.blocks.append(
                LitePTDecoderBlock(
                    channels=channels[i + 1],
                    depth=depths[i],
                    num_heads=num_heads[i],
                    patch_size=patch_sizes[i],
                    upsample=upsample,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    attn_drop=attn_drop,
                    proj_drop=proj_drop,
                    drop_path=drop_paths[i].tolist()[::-1],
                    act=act,
                    act_kwargs=act_kwargs,
                    use_flash_attn=use_flash_attn,
                    upcast_attn=upcast_attn,
                    upcast_softmax=upcast_softmax,
                    rope_base=rope_base,
                    conv=convs[i],
                    attention=attentions[i],
                    cpe_indice_key=f"stage{i}",
                    serialization_orders=serialization_orders,
                    shuffle_serialization_orders=shuffle_serialization_orders,
                )
            )

    @property
    def out_channels(self) -> int:
        """Feature dimension $C$ of the decoder output."""
        last = self.blocks[-1]
        assert isinstance(last, LitePTDecoderBlock)
        return last.channels

    def forward(self, x: Tensor, intermediates: List[LitePTFeaturesDict]) -> Tuple[Tensor, Tensor, Tensor]:
        for block, skip in zip(self.blocks, reversed(intermediates)):
            x, pos_grid, batch = block(
                x,
                skip["x"],
                skip["pos_grid"],
                skip["batch"],
                skip["pooling_inverse"],
                serialized_order_skip=skip.get("serialized_order"),
                serialized_inverse_skip=skip.get("serialized_inverse"),
            )
        return x, pos_grid, batch


class LitePTSegmentation(SemanticSegmentationModel):
    r"""LitePT for semantic segmentation.

    Based on the paper :arxiv: [LitePT: Lighter Yet Stronger Point Transformer](https://arxiv.org/abs/2512.13689)
    by Yuanwen Yue, Damien Robert, Jianyuan Wang, Sunghwan Hong, Jan Dirk Wegner, Christian Rupprecht, Konrad
    Schindler.

    A Point Transformer V3 encoder-decoder whose fine stages keep only the residual sparse convolutions and whose
    coarse stages keep only serialized attention, with the learned position embedding replaced by a 3D rotary
    embedding of the grid coordinates. The decoder unpools through the encoder skips without blocks by default.

    Args:
        in_channels: Number of input channels.
        num_classes: Number of output classes; `0` drops the head and returns the decoder features.
        serialization_orders: Space-filling curves the attention stages serialize along.
        shuffle_serialization_orders: Whether to shuffle the orders at every forward in training.
        strides: Grid pooling stride between consecutive encoder stages.
        encoder_depths: Number of blocks per encoder stage.
        encoder_channels: Feature width per encoder stage.
        encoder_num_heads: Attention heads per encoder stage.
        encoder_patch_size: Attention patch size per encoder stage.
        encoder_conv: Whether the blocks of each encoder stage start with the residual sparse convolution.
        encoder_attention: Whether the blocks of each encoder stage have the attention and MLP branches.
        decoder_depths: Number of blocks per decoder stage, from the coarsest to the finest resolution.
        decoder_channels: Feature width per decoder stage.
        decoder_num_heads: Attention heads per decoder stage.
        decoder_patch_size: Attention patch size per decoder stage.
        decoder_conv: Whether the blocks of each decoder stage start with the residual sparse convolution.
        decoder_attention: Whether the blocks of each decoder stage have the attention and MLP branches.
        norm: Normalization of the stem, poolings and unpoolings, passed to `create_norm`.
        act: Activation, passed to `create_act`.
        mlp_ratio: Hidden-to-input width ratio of the block MLPs.
        qkv_bias: Whether the QKV projections have a bias.
        qk_scale: Attention logit scale; `None` uses $1 / \sqrt{\text{head\_dim}}$.
        attn_drop: Dropout on the attention weights.
        proj_drop: Dropout after the attention and MLP projections.
        drop_path: Stochastic depth rate, increasing linearly over the blocks.
        rope_base: Frequency base of the rotary position embedding.
        use_flash_attn: Whether attention runs through flash attention. The registered configurations construct
            with `use_flash_attn=True`, which requires `flash-attn` and a CUDA device; pass `use_flash_attn=False`
            to run without it (the sparse convolutions still need a `spconv` build matching the device).
        upcast_attn: Whether the manual attention path computes the logits in fp32.
        upcast_softmax: Whether the manual attention path computes the softmax in fp32.
        dropout: Dropout on the per-point features entering the head.
        act_kwargs: Extra keyword arguments for the activation.
        norm_kwargs: Extra keyword arguments for the normalization.

    Example:
        ```python
        import torch

        from torch_pointcloud.models import LitePTSegmentation

        model = LitePTSegmentation(in_channels=6, num_classes=20, use_flash_attn=False).cuda()
        x = torch.randn(1000, 6).cuda()
        pos_grid = torch.randint(0, 100, (1000, 3)).cuda()
        batch = torch.zeros(1000, dtype=torch.long).cuda()
        logits = model(x, pos_grid, batch)  # (1000, 20)
        ```
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        serialization_orders: Sequence[SerializationOrder] = ("z", "z-trans", "hilbert", "hilbert-trans"),
        shuffle_serialization_orders: bool = True,
        strides: Sequence[int] = (2, 2, 2, 2),
        encoder_depths: Sequence[int] = (2, 2, 2, 6, 2),
        encoder_channels: Sequence[int] = (36, 72, 144, 252, 504),
        encoder_num_heads: Sequence[int] = (2, 4, 8, 14, 28),
        encoder_patch_size: Sequence[int] = (1024, 1024, 1024, 1024, 1024),
        encoder_conv: Sequence[bool] = (True, True, True, False, False),
        encoder_attention: Sequence[bool] = (False, False, False, True, True),
        decoder_depths: Sequence[int] = (0, 0, 0, 0),
        decoder_channels: Sequence[int] = (252, 144, 72, 72),
        decoder_num_heads: Sequence[int] = (14, 8, 4, 4),
        decoder_patch_size: Sequence[int] = (1024, 1024, 1024, 1024),
        decoder_conv: Sequence[bool] = (False, False, False, False),
        decoder_attention: Sequence[bool] = (False, False, False, False),
        norm: Union[str, Callable] = "batch_norm",
        act: Union[str, Callable] = "gelu",
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_scale: Optional[float] = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        drop_path: float = 0.3,
        rope_base: float = 100.0,
        use_flash_attn: bool = True,
        upcast_attn: bool = False,
        upcast_softmax: bool = False,
        dropout: float = 0.0,
        act_kwargs: Optional[Dict[str, Any]] = None,
        norm_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(in_channels=in_channels, num_classes=num_classes)
        self.serialization_orders = serialization_orders
        self.shuffle_serialization_orders = shuffle_serialization_orders
        self.strides = strides
        self.encoder_depths = encoder_depths
        self.encoder_channels = encoder_channels
        self.encoder_num_heads = encoder_num_heads
        self.encoder_patch_size = encoder_patch_size
        self.encoder_conv = encoder_conv
        self.encoder_attention = encoder_attention
        self.decoder_depths = decoder_depths
        self.decoder_channels = decoder_channels
        self.decoder_num_heads = decoder_num_heads
        self.decoder_patch_size = decoder_patch_size
        self.decoder_conv = decoder_conv
        self.decoder_attention = decoder_attention
        self.norm = norm
        self.act = act
        self.mlp_ratio = mlp_ratio
        self.qkv_bias = qkv_bias
        self.qk_scale = qk_scale
        self.attn_drop = attn_drop
        self.proj_drop = proj_drop
        self.drop_path = drop_path
        self.rope_base = rope_base
        self.use_flash_attn = use_flash_attn
        self.upcast_attn = upcast_attn
        self.upcast_softmax = upcast_softmax
        self.dropout = dropout
        self.act_kwargs = act_kwargs
        self.norm_kwargs = norm_kwargs

        self.encoder = self.configure_encoder()
        self.decoder = self.configure_decoder()
        self.head = self.configure_head()

    def configure_encoder(self) -> LitePTEncoder:
        """Build the `LitePTEncoder` backbone."""
        return LitePTEncoder(
            in_channels=self.in_channels,
            serialization_orders=self.serialization_orders,
            shuffle_serialization_orders=self.shuffle_serialization_orders,
            strides=self.strides,
            encoder_depths=self.encoder_depths,
            encoder_channels=self.encoder_channels,
            encoder_num_heads=self.encoder_num_heads,
            encoder_patch_size=self.encoder_patch_size,
            encoder_conv=self.encoder_conv,
            encoder_attention=self.encoder_attention,
            norm=self.norm,
            act=self.act,
            act_kwargs=self.act_kwargs,
            norm_kwargs=self.norm_kwargs,
            mlp_ratio=self.mlp_ratio,
            qkv_bias=self.qkv_bias,
            qk_scale=self.qk_scale,
            attn_drop=self.attn_drop,
            proj_drop=self.proj_drop,
            drop_path=self.drop_path,
            rope_base=self.rope_base,
            use_flash_attn=self.use_flash_attn,
            upcast_attn=self.upcast_attn,
            upcast_softmax=self.upcast_softmax,
        )

    def configure_decoder(self) -> LitePTDecoder:
        """Build the `LitePTDecoder` unpooling the coarsest features back through the encoder skips."""
        return LitePTDecoder(
            encoder_channels=self.encoder_channels,
            decoder_depths=self.decoder_depths,
            decoder_channels=self.decoder_channels,
            decoder_num_heads=self.decoder_num_heads,
            decoder_patch_size=self.decoder_patch_size,
            decoder_conv=self.decoder_conv,
            decoder_attention=self.decoder_attention,
            serialization_orders=self.serialization_orders,
            shuffle_serialization_orders=self.shuffle_serialization_orders,
            norm=self.norm,
            act=self.act,
            act_kwargs=self.act_kwargs,
            norm_kwargs=self.norm_kwargs,
            mlp_ratio=self.mlp_ratio,
            qkv_bias=self.qkv_bias,
            qk_scale=self.qk_scale,
            attn_drop=self.attn_drop,
            proj_drop=self.proj_drop,
            drop_path=self.drop_path,
            rope_base=self.rope_base,
            use_flash_attn=self.use_flash_attn,
            upcast_attn=self.upcast_attn,
            upcast_softmax=self.upcast_softmax,
        )

    @property
    def num_features(self) -> int:
        """Channel count $C$ of the per-point decoder features entering the head."""
        return self.decoder.out_channels

    def configure_head(self) -> nn.Module:
        if self.num_classes == 0:
            return nn.Identity()
        return nn.Linear(self.num_features, self.num_classes)

    def reset_classifier(self, num_classes: int, **kwargs: Any) -> None:
        """Resets the segmentation head with new parameters.

        Note:
            To set an empty segmentation head, use `num_classes=0`.

        Args:
            num_classes: Number of output classes.
            **kwargs: Additional keyword arguments to pass to the segmentation head.
        """
        self.num_classes = num_classes
        self.head = self.configure_head()

    @overload
    def forward_features(
        self,
        x: Tensor,
        pos_grid: Tensor,
        batch: Tensor,
        return_intermediates: Literal[True],
    ) -> Tuple[Tensor, Tensor, Tensor, List[LitePTFeaturesDict]]: ...

    @overload
    def forward_features(
        self,
        x: Tensor,
        pos_grid: Tensor,
        batch: Tensor,
        return_intermediates: Literal[False] = False,
    ) -> Tuple[Tensor, Tensor, Tensor]: ...

    def forward_features(
        self,
        x: Tensor,
        pos_grid: Tensor,
        batch: Tensor,
        return_intermediates: bool = False,
    ) -> Any:
        if return_intermediates:
            return self.encoder.forward(x, pos_grid, batch, return_intermediates=True)
        return self.encoder.forward(x, pos_grid, batch, return_intermediates=False)

    def forward_decoder(self, x: Tensor, intermediates: List[LitePTFeaturesDict]) -> Tuple[Tensor, Tensor, Tensor]:
        return self.decoder.forward(x, intermediates)

    def forward_head(self, x: Tensor, pre_logits: bool = False) -> Tensor:
        if self.dropout:
            x = F.dropout(x, p=float(self.dropout), training=self.training)
        return x if pre_logits else self.head(x)

    def forward(self, x: Tensor, pos_grid: Tensor, batch: Tensor) -> Tensor:
        r"""Forward pass of the LitePT segmentation network.

        Args:
            x: Per-point features of shape $(N, C)$.
            pos_grid: Integer grid coordinates of shape $(N, 3)$, used for the sparse convolutions, the serialization
                and the rotary position embedding.
            batch: Batch indices for each point of shape $(N,)$.

        Returns:
            Per-point segmentation logits of shape $(N, \text{num\_classes})$.
        """
        x, _, _, intermediates = self.forward_features(x, pos_grid, batch, return_intermediates=True)
        x, _, _ = self.forward_decoder(x, intermediates)
        return self.forward_head(x)


_BLOCK_CHECKPOINT_MAPPING = {
    "block{j}.conv.0.{param}": "blocks.{j}.cpe_conv.{param}",
    "block{j}.conv.1.{param}": "blocks.{j}.cpe_proj.{param}",
    "block{j}.conv.2.{param}": "blocks.{j}.cpe_norm.{param}",
    "block{j}.norm0.0.{param}": "blocks.{j}.norm0.{param}",
    "block{j}.norm1.0.{param}": "blocks.{j}.norm1.{param}",
    "block{j}.norm2.0.{param}": "blocks.{j}.norm2.{param}",
    "block{j}.attn.{layer}.{param}": "blocks.{j}.attn.{layer}.{param}",
    "block{j}.mlp.0.fc1.{param}": "blocks.{j}.mlp.0.{param}",
    "block{j}.mlp.0.fc2.{param}": "blocks.{j}.mlp.3.{param}",
}


def segmentation_checkpoint_filter_fn(state_dict: Dict[str, Any], model: nn.Module) -> Dict[str, Any]:
    """Rename the keys of a released LitePT training checkpoint (a `backbone.` encoder and decoder with a `seg_head`,
    saved from a distributed `module.`) to the `LitePTSegmentation` ones, as the `pretrained_filter_fn` of a
    `WeightsDict`, and add the rotary frequency buffers the released attention does not store.

    The released decoder names its stages after the encoder stage they upsample to, so the deepest runs first;
    `model` numbers them in execution order: `dec{i}` of `n` stages becomes `decoder.blocks.{n - 1 - i}`.

    Args:
        state_dict: State dict keyed as in the released checkpoint file.
        model: `LitePTSegmentation` the state dict is loaded into.

    Returns:
        The state dict keyed as `model` expects.

    Example:
        ```{.python notest}
        from torch_pointcloud.models import create_model
        from torch_pointcloud.models.litept import segmentation_checkpoint_filter_fn
        from torch_pointcloud.utils.state_dict import read_state_dict

        model = create_model("litept-small.scannet20.prs-eth", task="semantic-segmentation")
        state_dict = read_state_dict("exp/scannet/semseg-litept-small-v1m1/model/model_best.pth")
        model.load_state_dict(segmentation_checkpoint_filter_fn(state_dict, model))
        ```
    """
    assert isinstance(model, LitePTSegmentation)
    state_dict = {key.removeprefix("module."): value for key, value in state_dict.items()}
    mapping = {
        "seg_head.{param}": "head.{param}",
        "backbone.embedding.stem.conv.{param}": "encoder.stem.stem.{param}",
        "backbone.embedding.stem.norm.{param}": "encoder.stem.norm.module.{param}",
        "backbone.enc.enc{i}.down.proj.{param}": "encoder.blocks.{i}.downsample.proj.{param}",
        "backbone.enc.enc{i}.down.norm.0.{param}": "encoder.blocks.{i}.downsample.norm.module.{param}",
    }
    for source, target in _BLOCK_CHECKPOINT_MAPPING.items():
        mapping[f"backbone.enc.enc{{i}}.{source}"] = f"encoder.blocks.{{i}}.{target}"
    num_stages = len(model.decoder.blocks)
    for i in range(num_stages):
        stage = f"decoder.blocks.{num_stages - 1 - i}"
        mapping[f"backbone.dec.dec{i}.up.proj.0.{{param}}"] = f"{stage}.upsample.proj.{{param}}"
        mapping[f"backbone.dec.dec{i}.up.proj.1.{{param}}"] = f"{stage}.upsample.norm.module.{{param}}"
        mapping[f"backbone.dec.dec{i}.up.proj_skip.0.{{param}}"] = f"{stage}.upsample.proj_skip.{{param}}"
        mapping[f"backbone.dec.dec{i}.up.proj_skip.1.{{param}}"] = f"{stage}.upsample.norm_skip.module.{{param}}"
        for source, target in _BLOCK_CHECKPOINT_MAPPING.items():
            mapping[f"backbone.dec.dec{i}.{source}"] = f"{stage}.{target}"
    state_dict = transform_state_dict(state_dict, mapping)
    # The released attention computes its rotary frequencies on the fly; `Point3DRoPE` keeps them in a buffer.
    for key, value in model.state_dict().items():
        if key.endswith("attn.rope.inv_freq"):
            state_dict[key] = value
    return state_dict


def _litept_seg_hparams(
    num_classes: int,
    in_channels: int,
    encoder_depths: Sequence[int],
    encoder_channels: Sequence[int],
    encoder_num_heads: Sequence[int],
    decoder_channels: Sequence[int],
    decoder_num_heads: Sequence[int],
) -> Dict[str, Any]:
    """The `LitePTSegmentation` hparams shared by the released checkpoints: four serialization orders, three
    convolution stages then two attention stages with patches of 1024 points, a block-free decoder and BatchNorm
    with `eps=1e-3` in the stem, poolings and unpoolings. The sizes differ by depth, width and head count only.
    """
    return dict(
        in_channels=in_channels,
        num_classes=num_classes,
        serialization_orders=("z", "z-trans", "hilbert", "hilbert-trans"),
        shuffle_serialization_orders=True,
        strides=(2, 2, 2, 2),
        encoder_depths=encoder_depths,
        encoder_channels=encoder_channels,
        encoder_num_heads=encoder_num_heads,
        encoder_patch_size=(1024, 1024, 1024, 1024, 1024),
        encoder_conv=(True, True, True, False, False),
        encoder_attention=(False, False, False, True, True),
        decoder_depths=(0, 0, 0, 0),
        decoder_channels=decoder_channels,
        decoder_num_heads=decoder_num_heads,
        decoder_patch_size=(1024, 1024, 1024, 1024),
        decoder_conv=(False, False, False, False),
        decoder_attention=(False, False, False, False),
        norm="batch_norm",
        norm_kwargs={"eps": 1e-3, "momentum": 0.01},
        act="gelu",
        mlp_ratio=4.0,
        drop_path=0.3,
        rope_base=100.0,
        use_flash_attn=True,
    )


def _litept_indoor_transforms(relabel_labels: Optional[Sequence[int]] = None) -> T.Compose:
    """Color + normal feature pipeline of the released indoor checkpoints, on a 2 cm grid.

    `relabel_labels` shifts the dataset's $1..K$ class labels (0 reserved for unknown) down to $0..K-1$ and sends
    everything else to the ignore index (`range(1, 21)` for ScanNet20); `None` keeps labels that are already 0-based.
    """
    steps: List[Any] = [
        T.Shift(keys=DataKeys.POS, method="bbox", axes=[0, 1]),
        T.Shift(keys=DataKeys.POS, method="min", axes=[2]),
        # The released weights were trained on colors in $[0, 1]$.
        T.Normalize(keys=DataKeys.COLOR, mean=[0.0, 0.0, 0.0], std=[255.0, 255.0, 255.0]),
        T.Cat(keys=[DataKeys.COLOR, DataKeys.NORMAL], dst_key=DataKeys.X, dim=1),
    ]
    if relabel_labels is not None:
        steps.append(T.Relabel(keys=DataKeys.SEGMENT, labels=relabel_labels, default=-1, allow_missing_keys=True))
    steps += [
        T.CopyItems(
            keys=[DataKeys.POS, DataKeys.SEGMENT],
            dst_keys=[DataKeys.ORIGIN_POS, DataKeys.ORIGIN_SEGMENT],
            allow_missing_keys=True,
        ),
        T.Voxelize(
            pos_key=DataKeys.POS,
            pos_reduce="first",
            dst_pos_grid_key=DataKeys.POS_GRID,
            keys=[DataKeys.X, DataKeys.SEGMENT, DataKeys.COLOR, DataKeys.NORMAL, DataKeys.INSTANCE],
            reduce="first",
            size=0.02,
            method="fnv",
            allow_missing_keys=True,
            dst_inverse_key=DataKeys.INVERSE,
        ),
    ]
    return T.Compose(steps)


_LITEPT_NUSCENES_TRANSFORM = T.Compose(
    [
        T.Relabel(
            keys=DataKeys.SEGMENT,
            labels=NUSCENES_LIDARSEG_LEARNING_MAP,
            default=-1,
            allow_missing_keys=True,
        ),
        # The released weights were trained on intensities in $[0, 1]$.
        T.Normalize(keys=DataKeys.INTENSITY, mean=[0.0], std=[255.0]),
        T.Cat(keys=[DataKeys.POS, DataKeys.INTENSITY], dst_key=DataKeys.X, dim=1),
        T.CopyItems(
            keys=[DataKeys.POS, DataKeys.SEGMENT],
            dst_keys=[DataKeys.ORIGIN_POS, DataKeys.ORIGIN_SEGMENT],
            allow_missing_keys=True,
        ),
        T.Voxelize(
            pos_key=DataKeys.POS,
            pos_reduce="first",
            dst_pos_grid_key=DataKeys.POS_GRID,
            keys=[DataKeys.X, DataKeys.SEGMENT, DataKeys.INTENSITY],
            reduce="first",
            size=0.05,
            method="fnv",
            allow_missing_keys=True,
            dst_inverse_key=DataKeys.INVERSE,
        ),
    ]
)


@register_model(
    "litept-small.scannet20.prs-eth",
    task="semantic-segmentation",
    input_keys=("x", "pos_grid", "batch"),
    weights=WeightsDict(
        url="hf://prs-eth/LitePT/resolve/a8e76e92efbb2061639f5c683968bc5d248ee002/scannet-semseg-litept-small-v1m1/model/model_best.pth",
        pretrained_filter_fn=segmentation_checkpoint_filter_fn,
        dataset="scannet20",
        metrics={"mIoU": 76.54, "OA": 91.42},
        classes=SCANNET20_CLASSES,
        author="prs-eth",
        license="MIT",
    ),
    transform=_litept_indoor_transforms(range(1, 21)),
    hparams=_litept_seg_hparams(
        num_classes=20,
        in_channels=6,
        encoder_depths=(2, 2, 2, 6, 2),
        encoder_channels=(36, 72, 144, 252, 504),
        encoder_num_heads=(2, 4, 8, 14, 28),
        decoder_channels=(252, 144, 72, 72),
        decoder_num_heads=(14, 8, 4, 4),
    ),
)
def litept_small_scannet20(**hparams: Any) -> LitePTSegmentation:
    return LitePTSegmentation(**hparams)


@register_model(
    "litept-small.nuscenes.prs-eth",
    task="semantic-segmentation",
    input_keys=("x", "pos_grid", "batch"),
    weights=WeightsDict(
        url="hf://prs-eth/LitePT/resolve/a8e76e92efbb2061639f5c683968bc5d248ee002/nuscenes-semseg-litept-small-v1m1/model/model_best.pth",
        pretrained_filter_fn=segmentation_checkpoint_filter_fn,
        dataset="nuscenes",
        metrics={"mIoU": 82.01, "OA": 95.02, "simple/mIoU": 80.47, "simple/OA": 94.68},
        classes=NUSCENES_LIDARSEG_CLASSES,
        author="prs-eth",
        license="MIT",
    ),
    transform=_LITEPT_NUSCENES_TRANSFORM,
    hparams=_litept_seg_hparams(
        num_classes=16,
        in_channels=4,
        encoder_depths=(2, 2, 2, 6, 2),
        encoder_channels=(36, 72, 144, 252, 504),
        encoder_num_heads=(2, 4, 8, 14, 28),
        decoder_channels=(252, 144, 72, 72),
        decoder_num_heads=(14, 8, 4, 4),
    ),
)
def litept_small_nuscenes(**hparams: Any) -> LitePTSegmentation:
    return LitePTSegmentation(**hparams)


@register_model(
    "litept-large.nuscenes.prs-eth",
    task="semantic-segmentation",
    input_keys=("x", "pos_grid", "batch"),
    weights=WeightsDict(
        url="hf://prs-eth/LitePT/resolve/a8e76e92efbb2061639f5c683968bc5d248ee002/nuscenes-semseg-litept-large-v1m1/model/model_best.pth",
        pretrained_filter_fn=segmentation_checkpoint_filter_fn,
        dataset="nuscenes",
        metrics={"simple/mIoU": 81.12, "simple/OA": 94.68},
        classes=NUSCENES_LIDARSEG_CLASSES,
        author="prs-eth",
        license="MIT",
    ),
    transform=_LITEPT_NUSCENES_TRANSFORM,
    hparams=_litept_seg_hparams(
        num_classes=16,
        in_channels=4,
        encoder_depths=(3, 3, 3, 12, 3),
        encoder_channels=(72, 144, 288, 576, 864),
        encoder_num_heads=(4, 8, 16, 32, 48),
        decoder_channels=(576, 288, 144, 72),
        decoder_num_heads=(32, 16, 8, 4),
    ),
)
def litept_large_nuscenes(**hparams: Any) -> LitePTSegmentation:
    return LitePTSegmentation(**hparams)


@register_model(
    "litept-small.structured3d.prs-eth",
    task="semantic-segmentation",
    input_keys=("x", "pos_grid", "batch"),
    weights=WeightsDict(
        url="hf://prs-eth/LitePT/resolve/a8e76e92efbb2061639f5c683968bc5d248ee002/structured3d-semseg-litept-small-v1m1/model/model_best.pth",
        pretrained_filter_fn=segmentation_checkpoint_filter_fn,
        dataset="structured3d",
        classes=_STRUCTURED3D_CLASSES,
        author="prs-eth",
        license="MIT",
    ),
    transform=_litept_indoor_transforms(),
    hparams=_litept_seg_hparams(
        num_classes=25,
        in_channels=6,
        encoder_depths=(2, 2, 2, 6, 2),
        encoder_channels=(36, 72, 144, 252, 504),
        encoder_num_heads=(2, 4, 8, 14, 28),
        decoder_channels=(252, 144, 72, 72),
        decoder_num_heads=(14, 8, 4, 4),
    ),
)
def litept_small_structured3d(**hparams: Any) -> LitePTSegmentation:
    return LitePTSegmentation(**hparams)


@register_model(
    "litept-base.structured3d.prs-eth",
    task="semantic-segmentation",
    input_keys=("x", "pos_grid", "batch"),
    weights=WeightsDict(
        url="hf://prs-eth/LitePT/resolve/a8e76e92efbb2061639f5c683968bc5d248ee002/structured3d-semseg-litept-base-v1m1/model/model_best.pth",
        pretrained_filter_fn=segmentation_checkpoint_filter_fn,
        dataset="structured3d",
        classes=_STRUCTURED3D_CLASSES,
        author="prs-eth",
        license="MIT",
    ),
    transform=_litept_indoor_transforms(),
    hparams=_litept_seg_hparams(
        num_classes=25,
        in_channels=6,
        encoder_depths=(3, 3, 3, 12, 3),
        encoder_channels=(54, 108, 216, 432, 576),
        encoder_num_heads=(3, 6, 12, 24, 32),
        decoder_channels=(432, 216, 108, 72),
        decoder_num_heads=(24, 12, 6, 4),
    ),
)
def litept_base_structured3d(**hparams: Any) -> LitePTSegmentation:
    return LitePTSegmentation(**hparams)


@register_model(
    "litept-large.structured3d.prs-eth",
    task="semantic-segmentation",
    input_keys=("x", "pos_grid", "batch"),
    weights=WeightsDict(
        url="hf://prs-eth/LitePT/resolve/a8e76e92efbb2061639f5c683968bc5d248ee002/structured3d-semseg-litept-large-v1m1/model/model_best.pth",
        pretrained_filter_fn=segmentation_checkpoint_filter_fn,
        dataset="structured3d",
        classes=_STRUCTURED3D_CLASSES,
        author="prs-eth",
        license="MIT",
    ),
    transform=_litept_indoor_transforms(),
    hparams=_litept_seg_hparams(
        num_classes=25,
        in_channels=6,
        encoder_depths=(3, 3, 3, 12, 3),
        encoder_channels=(72, 144, 288, 576, 864),
        encoder_num_heads=(4, 8, 16, 32, 48),
        decoder_channels=(576, 288, 144, 72),
        decoder_num_heads=(32, 16, 8, 4),
    ),
)
def litept_large_structured3d(**hparams: Any) -> LitePTSegmentation:
    return LitePTSegmentation(**hparams)
