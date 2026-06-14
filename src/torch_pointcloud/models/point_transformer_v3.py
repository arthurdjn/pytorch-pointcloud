from typing import TYPE_CHECKING, Any, Callable, Dict, List, Literal, Optional, Sequence, Tuple, Union, overload

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_pointcloud.layers.act import create_act
from torch_pointcloud.layers.norms import create_norm

import torch_pointcloud.transforms as T
from torch_pointcloud.layers import PoolLike, create_pool
from torch_pointcloud.layers.dropouts import DropPath
from torch_pointcloud.layers.grid_pool import GridPool
from torch_pointcloud.layers.linear_blocks import LinearBlock
from torch_pointcloud.layers.serialized_attention import (
    SerializedAttention,
    SerializedAttentionRoPE,
    SerializedAttentionRPE,
)
from torch_pointcloud.layers.serialized_pool import SerializedPool, SerializedUpsample
from torch_pointcloud.layers.spconv_blocks import SubMConv3dBlock
from torch_pointcloud.models._base import ClassificationModel, SegmentationModel
from torch_pointcloud.models._registry import register_model
from torch_pointcloud.utils.conversion import convert_to_spconv_tensor, ensure_tuple, ensure_tuple_size
from torch_pointcloud.utils.data import DataKeys
from torch_pointcloud.utils.imports import optional_import
from torch_pointcloud.utils.serialization import SerializationOrder, serialize_coords
from torch_pointcloud.utils.types import OptTensor, ValueCollection

if TYPE_CHECKING:
    import spconv.pytorch as spconv


spconv, _ = optional_import("spconv.pytorch")

AttentionKind = Literal["default", "rpe", "rope"]


def _build_attention(
    attn_kind: AttentionKind,
    *,
    channels: int,
    num_heads: int,
    patch_size: int,
    qkv_bias: bool,
    qk_scale: Optional[float],
    attn_drop: float,
    proj_drop: float,
    use_flash_attn: bool,
    upcast_attn: bool,
    upcast_softmax: bool,
    rope_base: float,
) -> nn.Module:
    if attn_kind == "default":
        return SerializedAttention(
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
        )
    if attn_kind == "rpe":
        return SerializedAttentionRPE(
            channels=channels,
            num_heads=num_heads,
            patch_size=patch_size,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            upcast_attn=upcast_attn,
            upcast_softmax=upcast_softmax,
        )
    if attn_kind == "rope":
        return SerializedAttentionRoPE(
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
    raise ValueError(f"Unknown attention kind {attn_kind!r}; expected 'default', 'rpe', or 'rope'.")


def serialize(
    pos_grid: Tensor,
    batch: Tensor,
    orders: Sequence[SerializationOrder],
    shuffle: bool = False,
) -> Tuple[Tensor, Tensor, Tensor]:
    if shuffle:
        perm = torch.randperm(len(orders))
        orders = [orders[i] for i in perm]

    depth = int(pos_grid.max()).bit_length()
    serialized_code = torch.stack([serialize_coords(pos_grid, batch, depth=depth, order=order) for order in orders])
    serialized_order = torch.argsort(serialized_code, dim=1)
    serialized_inverse = torch.argsort(serialized_order, dim=1)
    return serialized_code, serialized_order, serialized_inverse


class Block(nn.Module):
    def __init__(
        self,
        channels: int,
        num_heads: int,
        patch_size: int = 48,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_scale: Optional[float] = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        drop_path: float = 0.0,
        norm: Union[str, Callable] = "layer_norm",
        act: Union[str, Callable] = "gelu",
        act_kwargs: Optional[Dict[str, Any]] = None,
        norm_kwargs: Optional[Dict[str, Any]] = None,
        cpe_indice_key: Optional[str] = None,
        attn_kind: AttentionKind = "default",
        use_flash_attn: bool = True,
        upcast_attn: bool = True,
        upcast_softmax: bool = True,
        rope_base: float = 10.0,
        legacy: bool = False,
    ):
        super().__init__()
        self.legacy = legacy
        act_kwargs = act_kwargs or {}
        norm_kwargs = norm_kwargs or {}

        self.cpe_conv = spconv.SubMConv3d(
            channels,
            channels,
            kernel_size=3,
            bias=True,
            indice_key=cpe_indice_key,
        )
        self.cpe_proj = nn.Linear(channels, channels)
        self.cpe_norm = create_norm("layer_norm", channels, mode="node") or nn.Identity()

        self.norm1 = create_norm("layer_norm", channels, mode="node") or nn.Identity()
        self.attn = _build_attention(
            attn_kind,
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

        self.norm2 = create_norm("layer_norm", channels, mode="node") or nn.Identity()
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
        pos: OptTensor = None,
    ) -> Tuple[Tensor, Any]:
        shortcut = x
        x_sparse = self.cpe_conv(x_sparse)
        x = self.cpe_proj(x_sparse.features)
        x = self.cpe_norm(x)
        x = shortcut = shortcut + x

        # Attention branch
        x = self.norm1(x)
        x = self.attn(
            x,
            pos_grid,
            batch,
            serialized_order=serialized_order,
            serialized_inverse=serialized_inverse,
            pos=pos,
        )
        x = self.drop_path(x)
        x = shortcut + x

        # MLP branch
        shortcut = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = self.drop_path(x)
        x_sparse = x_sparse.replace_feature(x if self.legacy else shortcut + x)
        x = shortcut + x

        return x, x_sparse


class EncoderBlock(nn.Module):
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
        norm: Union[str, Callable] = "layer_norm",
        act: Union[str, Callable] = "gelu",
        act_kwargs: Optional[Dict[str, Any]] = None,
        norm_kwargs: Optional[Dict[str, Any]] = None,
        attn_kind: AttentionKind = "default",
        use_flash_attn: bool = True,
        upcast_attn: bool = False,
        upcast_softmax: bool = False,
        cpe_indice_key: Optional[str] = None,
        downsample: Optional[nn.Module] = None,
        serialization_orders: Optional[Sequence[SerializationOrder]] = None,
        shuffle_serialization_orders: bool = False,
        rope_base: float = 10.0,
        legacy: bool = False,
    ):
        super().__init__()
        self.downsample = downsample
        self.serialization_orders = serialization_orders
        self.shuffle_serialization_orders = shuffle_serialization_orders
        drop_path = ensure_tuple_size(drop_path, size=depth)

        self.blocks = nn.ModuleList()
        for i in range(depth):
            self.blocks.append(
                Block(
                    channels=channels,
                    num_heads=num_heads,
                    patch_size=patch_size,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    attn_drop=attn_drop,
                    proj_drop=proj_drop,
                    drop_path=drop_path[i],
                    norm=norm,
                    act=act,
                    act_kwargs=act_kwargs,
                    norm_kwargs=norm_kwargs,
                    cpe_indice_key=cpe_indice_key,
                    attn_kind=attn_kind,
                    use_flash_attn=use_flash_attn,
                    upcast_attn=upcast_attn,
                    upcast_softmax=upcast_softmax,
                    rope_base=rope_base,
                    legacy=legacy,
                )
            )

    @overload
    def forward(
        self,
        x: OptTensor,
        pos_grid: Tensor,
        batch: Tensor,
        serialized_code: Tensor,
        serialized_order: Tensor,
        serialized_inverse: Tensor,
        return_inverse: Literal[True] = True,
        pos: OptTensor = None,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, OptTensor]: ...

    @overload
    def forward(
        self,
        x: OptTensor,
        pos_grid: Tensor,
        batch: Tensor,
        serialized_code: Tensor,
        serialized_order: Tensor,
        serialized_inverse: Tensor,
        return_inverse: Literal[False] = False,
        pos: OptTensor = None,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, OptTensor]: ...

    def forward(
        self,
        x: OptTensor,
        pos_grid: Tensor,
        batch: Tensor,
        serialized_code: Tensor,
        serialized_order: Tensor,
        serialized_inverse: Tensor,
        return_inverse: bool = False,
        pos: OptTensor = None,
    ) -> Any:
        if not serialized_code.shape == serialized_order.shape == serialized_inverse.shape:
            raise ValueError(
                "`serialized_code`, `serialized_order` and `serialized_inverse` "
                f"must have the same shape. Got {serialized_code.shape}, "
                f"{serialized_order.shape} and {serialized_inverse.shape} respectively."
            )

        num_serializations = len(serialized_code)
        inverse: OptTensor = None

        if self.downsample is not None:
            if isinstance(self.downsample, GridPool):
                assert self.serialization_orders is not None, "serialization_orders must be provided for grid pooling"
                x, pos_grid, batch, inverse, pos = self.downsample(x, pos_grid, batch, pos=pos)
                serialized_code, serialized_order, serialized_inverse = serialize(
                    pos_grid,
                    batch,
                    orders=self.serialization_orders,
                    shuffle=self.shuffle_serialization_orders and self.training,
                )
            else:
                x, pos_grid, batch, serialized_code, inverse = self.downsample(
                    x,
                    pos_grid,
                    batch,
                    serialized_code,
                    return_inverse=True,
                )
                serialized_order = torch.argsort(serialized_code, dim=1)
                serialized_inverse = torch.argsort(serialized_order, dim=1)

        assert x is not None
        x_sparse = convert_to_spconv_tensor(x, pos_grid, batch)
        for i, block in enumerate(self.blocks):
            order_idx = i % num_serializations
            x, x_sparse = block(
                x,
                x_sparse,
                pos_grid,
                batch,
                serialized_order=serialized_order[order_idx],
                serialized_inverse=serialized_inverse[order_idx],
                pos=pos,
            )

        if return_inverse:
            return x, pos_grid, batch, serialized_code, serialized_order, serialized_inverse, inverse, pos
        return x, pos_grid, batch, serialized_code, serialized_order, serialized_inverse, pos


class DecoderBlock(nn.Module):
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
        norm: Union[str, Callable] = "batch_norm",
        act: Union[str, Callable] = "gelu",
        act_kwargs: Optional[Dict[str, Any]] = None,
        norm_kwargs: Optional[Dict[str, Any]] = None,
        attn_kind: AttentionKind = "default",
        use_flash_attn: bool = True,
        upcast_attn: bool = False,
        upcast_softmax: bool = False,
        cpe_indice_key: Optional[str] = None,
        upsample: Optional[SerializedUpsample] = None,
        rope_base: float = 10.0,
        legacy: bool = False,
    ):
        super().__init__()
        self.upsample = upsample
        drop_path = ensure_tuple_size(drop_path, size=depth)

        self.blocks = nn.ModuleList()
        for i in range(depth):
            self.blocks.append(
                Block(
                    channels=channels,
                    num_heads=num_heads,
                    patch_size=patch_size,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    attn_drop=attn_drop,
                    proj_drop=proj_drop,
                    drop_path=drop_path[i],
                    norm=norm,
                    act=act,
                    act_kwargs=act_kwargs,
                    norm_kwargs=norm_kwargs,
                    cpe_indice_key=cpe_indice_key,
                    attn_kind=attn_kind,
                    use_flash_attn=use_flash_attn,
                    upcast_attn=upcast_attn,
                    upcast_softmax=upcast_softmax,
                    rope_base=rope_base,
                    legacy=legacy,
                )
            )

    def forward(
        self,
        x: Tensor,
        x_skip: Tensor,
        pos_grid_skip: Tensor,
        batch_skip: Tensor,
        serialized_order_skip: Tensor,
        serialized_inverse_skip: Tensor,
        inverse: OptTensor = None,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        if not serialized_order_skip.shape == serialized_inverse_skip.shape:
            raise ValueError(
                "`serialized_order_skip` and `serialized_inverse_skip` "
                f"must have the same shape. Got {serialized_order_skip.shape} "
                f"and {serialized_inverse_skip.shape} respectively."
            )

        num_serializations = len(serialized_order_skip)

        cpe_seed = x
        if self.upsample is not None:
            if inverse is None:
                raise ValueError("`inverse` must be provided when `upsample` module is set.")

            # Reproduce Pointcept's SerializedUnpooling: the next block's xCPE convolves only the projected
            # skip branch. Present in every release, so it is unconditional (unlike the block write-back).
            x, cpe_seed = self.upsample(x, x_skip, inverse, return_intermediate=True)

        x_sparse = convert_to_spconv_tensor(cpe_seed, pos_grid_skip, batch_skip)
        for i, block in enumerate(self.blocks):
            order_idx = i % num_serializations
            x, x_sparse = block(
                x,
                x_sparse,
                pos_grid_skip,
                batch_skip,
                serialized_order=serialized_order_skip[order_idx],
                serialized_inverse=serialized_inverse_skip[order_idx],
            )

        return x, pos_grid_skip, batch_skip


class PointTransformerV3Encoder(nn.Module):
    """Point Transformer V3 encoder backbone.

    Encoder-only backbone for feature extraction from 3D point clouds.
    Supports both sparse convolution (PTV3 Mode 1) and linear (Sonata / Mode 2)
    embedding stems, and both serialized (code-space) and grid-based pooling.

    Args:
        in_channels: Number of input channels.
        serialization_orders: Serialization orders for attention.
        shuffle_serialization_orders: Shuffle orders each forward pass.
        strides: Downsampling strides between encoder stages.
        encoder_depths: Number of blocks per encoder stage.
        encoder_channels: Feature channels per encoder stage.
        encoder_num_head: Attention heads per encoder stage.
        encoder_patch_size: Patch size per encoder stage.
        norm: Normalization layer type.
        act: Activation function type.
        mlp_ratio: MLP expansion ratio.
        qkv_bias: Use bias in QKV projection.
        qk_scale: Custom QK scaling factor.
        attn_drop: Attention dropout rate.
        proj_drop: Projection dropout rate.
        drop_path: Drop path rate.
        attn_kind: Attention variant: `"default"` (vanilla, PT-V3 / Sonata / Concerto),
            `"rpe"` (PT-V3 with learned relative position bias), or `"rope"`
            (Utonia, 3D rotary position embedding on `Q`, `K`).
        use_flash_attn: Use Flash Attention.
        upcast_attn: Upcast attention to fp32.
        upcast_softmax: Upcast softmax to fp32.
        pooling: Pooling strategy — `"serialized"` (code-space bit-shift) or
            `"grid"` (grid-coordinate clustering).
        stem_type: How to embed raw features — `"sparse_conv"` (SubMConv3d stem) or
            `"linear"` (linear projection).
        rope_base: RoPE frequency base. Only used when `attn_kind="rope"`.
        act_kwargs: Optional keyword arguments for the activation factory.
        norm_kwargs: Optional keyword arguments for the normalization factory.
        bias: Whether the stem and blocks use learnable bias where applicable.
        legacy: Reproduce the Pointcept v1.5.1 block xCPE bug (the block output was not written back to
            the sparse tensor the next block convolves; fixed in v1.5.2). The released weights need
            `legacy=True`; leave `False` (default) for new training.

    Inputs:
        x: Float tensor of shape $(N, \\text{in\\_channels})$.
        pos_grid: Int tensor of shape $(N, 3)$ with voxel-grid coordinates.
        batch: Long tensor of shape $(N,)$.

    Outputs:
        Encoded features at the deepest encoder level.
    """

    def __init__(
        self,
        in_channels: int = 6,
        serialization_orders: Sequence[SerializationOrder] = ("hilbert", "hilbert-trans"),
        shuffle_serialization_orders: bool = True,
        strides: Sequence[int] = (2, 2, 2, 2),
        encoder_depths: Sequence[int] = (2, 2, 2, 6, 2),
        encoder_channels: Sequence[int] = (32, 64, 128, 256, 512),
        encoder_num_head: Sequence[int] = (2, 4, 8, 16, 32),
        encoder_patch_size: Sequence[int] = (48, 48, 48, 48, 48),
        act: Union[str, Callable] = "gelu",
        norm: Union[str, Callable] = "batch_norm",
        act_kwargs: Optional[Dict[str, Any]] = None,
        norm_kwargs: Optional[Dict[str, Any]] = None,
        bias: bool = True,
        mlp_ratio: float = 4,
        qkv_bias: bool = True,
        qk_scale: Optional[float] = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        drop_path: float = 0.3,
        attn_kind: AttentionKind = "default",
        use_flash_attn: bool = True,
        upcast_attn: bool = False,
        upcast_softmax: bool = False,
        pooling: str = "serialized",
        stem_type: str = "sparse_conv",
        rope_base: float = 10.0,
        legacy: bool = False,
    ):
        in_channels = in_channels if in_channels > 0 else 3
        super().__init__()
        self.in_channels = in_channels
        self.serialization_orders = ensure_tuple(serialization_orders)
        self.shuffle_serialization_orders = shuffle_serialization_orders
        self.attn_kind = attn_kind
        self.legacy = legacy

        self.stem = self.configure_stem(
            in_channels=in_channels,
            out_channels=encoder_channels[0],
            norm=norm,
            act=act,
            act_kwargs=act_kwargs,
            norm_kwargs=norm_kwargs,
            bias=bias,
            stem_type=stem_type,
        )
        self.blocks = self.configure_blocks(
            depths=encoder_depths,
            channels=encoder_channels,
            num_heads=encoder_num_head,
            patch_sizes=encoder_patch_size,
            strides=strides,
            mlp_ratio=mlp_ratio,
            norm=norm,
            act=act,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            drop_path=drop_path,
            attn_kind=attn_kind,
            use_flash_attn=use_flash_attn,
            upcast_attn=upcast_attn,
            upcast_softmax=upcast_softmax,
            pooling=pooling,
            serialization_orders=self.serialization_orders,
            shuffle_serialization_orders=shuffle_serialization_orders,
            act_kwargs=act_kwargs,
            norm_kwargs=norm_kwargs,
            rope_base=rope_base,
            legacy=legacy,
        )

    @property
    def embedding_dim(self) -> int:
        return self.blocks[-1].blocks[-1].mlp[0].in_features  # type: ignore[index, union-attr]

    def configure_stem(
        self,
        in_channels: int,
        out_channels: int,
        norm: Union[str, Callable],
        act: Union[str, Callable],
        act_kwargs: Optional[Dict[str, Any]] = None,
        norm_kwargs: Optional[Dict[str, Any]] = None,
        bias: bool = True,
        stem_type: str = "sparse_conv",
    ) -> nn.Module:
        if stem_type == "linear":
            return LinearBlock(
                in_channels,
                out_channels,
                bias=bias,
                act=act,
                act_kwargs=act_kwargs,
                norm=norm,
                norm_kwargs=norm_kwargs,
            )

        return SubMConv3dBlock(
            in_channels,
            out_channels,
            kernel_size=5,
            padding=1,
            norm=norm,
            act=act,
            act_kwargs=act_kwargs,
            norm_kwargs=norm_kwargs,
            bias=False,
        )

    def configure_blocks(
        self,
        depths: Sequence[int],
        channels: Sequence[int],
        num_heads: Sequence[int],
        patch_sizes: Sequence[int],
        strides: Sequence[int],
        mlp_ratio: float = 4.0,
        bias: bool = True,
        norm: Union[str, Callable] = "batch_norm",
        act: Union[str, Callable] = "gelu",
        qkv_bias: bool = True,
        qk_scale: Optional[float] = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        drop_path: float = 0.0,
        attn_kind: AttentionKind = "default",
        use_flash_attn: bool = True,
        upcast_attn: bool = False,
        upcast_softmax: bool = False,
        pooling: str = "serialized",
        serialization_orders: Optional[Sequence[SerializationOrder]] = None,
        shuffle_serialization_orders: bool = False,
        act_kwargs: Optional[Dict[str, Any]] = None,
        norm_kwargs: Optional[Dict[str, Any]] = None,
        rope_base: float = 10.0,
        legacy: bool = False,
    ) -> nn.ModuleList:
        depths = ensure_tuple(depths)
        n = len(depths)
        channels = ensure_tuple_size(channels, size=n)
        num_heads = ensure_tuple_size(num_heads, size=n)
        patch_sizes = ensure_tuple_size(patch_sizes, size=n)
        strides = ensure_tuple_size(strides, size=n - 1)
        drop_paths = torch.split(torch.linspace(0, drop_path, sum(depths)), list(depths))
        use_grid_pool = pooling == "grid"

        blocks = nn.ModuleList()
        for i in range(n):
            downsample: Optional[nn.Module] = None
            if i > 0:
                Pool = GridPool if use_grid_pool else SerializedPool
                downsample = Pool(
                    in_channels=channels[i - 1],
                    out_channels=channels[i],
                    stride=strides[i - 1],
                    bias=bias,
                    act=act,
                    norm=norm,
                    act_kwargs=act_kwargs,
                    norm_kwargs=norm_kwargs,
                )

            block = EncoderBlock(
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
                norm=norm,
                act=act,
                act_kwargs=act_kwargs,
                norm_kwargs=norm_kwargs,
                cpe_indice_key=f"stage{i}",
                attn_kind=attn_kind,
                use_flash_attn=use_flash_attn,
                upcast_attn=upcast_attn,
                upcast_softmax=upcast_softmax,
                downsample=downsample,
                serialization_orders=serialization_orders if use_grid_pool else None,
                shuffle_serialization_orders=shuffle_serialization_orders if use_grid_pool else False,
                rope_base=rope_base,
                legacy=legacy,
            )

            blocks.append(block)
        return blocks

    @overload
    def forward(
        self,
        x: OptTensor,
        pos_grid: Tensor,
        batch: Tensor,
        return_intermediates: Literal[True],
        pos: OptTensor = None,
    ) -> Tuple[Tensor, Tensor, Tensor, List[Dict[str, Tensor]]]: ...

    @overload
    def forward(
        self,
        x: OptTensor,
        pos_grid: Tensor,
        batch: Tensor,
        return_intermediates: Literal[False] = False,
        pos: OptTensor = None,
    ) -> Tuple[Tensor, Tensor, Tensor]: ...

    def forward(
        self,
        x: OptTensor,
        pos_grid: Tensor,
        batch: Tensor,
        return_intermediates: bool = False,
        pos: OptTensor = None,
    ) -> Any:
        x = x if x is not None else pos_grid.float()

        serialized_code, serialized_order, serialized_inverse = serialize(
            pos_grid,
            batch,
            orders=self.serialization_orders,
            shuffle=self.shuffle_serialization_orders and self.training,
        )

        x = self.stem(x, pos_grid, batch)

        intermediates = []
        for i, block in enumerate(self.blocks):
            intermediate = {
                "x": x,
                "pos_grid": pos_grid,
                "batch": batch,
                "serialized_code": serialized_code,
                "serialized_order": serialized_order,
                "serialized_inverse": serialized_inverse,
            }

            (
                x,
                pos_grid,
                batch,
                serialized_code,
                serialized_order,
                serialized_inverse,
                inverse,
                pos,
            ) = block(
                x,
                pos_grid,
                batch,
                serialized_code=serialized_code,
                serialized_order=serialized_order,
                serialized_inverse=serialized_inverse,
                return_inverse=True,
                pos=pos,
            )

            if i > 0:
                intermediate["inverse"] = inverse
                intermediates.append(intermediate)

        if return_intermediates:
            return x, pos_grid, batch, intermediates
        return x, pos_grid, batch


class PointTransformerV3Decoder(nn.Module):
    """Point Transformer V3 decoder with skip connections.

    Decoder backbone that upsamples encoder features using skip connections
    from intermediate encoder stages. Used for dense prediction tasks like
    semantic segmentation.

    Args:
        encoder_channels: Channel sequence from the encoder (needed to derive
            skip-connection channels).
        decoder_depths: Number of blocks per decoder stage.
        decoder_channels: Feature channels per decoder stage.
        decoder_num_head: Attention heads per decoder stage.
        decoder_patch_size: Patch size per decoder stage.
        norm: Normalization layer type.
        act: Activation function type.
        mlp_ratio: MLP expansion ratio.
        qkv_bias: Use bias in QKV projection.
        qk_scale: Custom QK scaling factor.
        attn_drop: Attention dropout rate.
        proj_drop: Projection dropout rate.
        drop_path: Drop path rate.
        attn_kind: Attention variant (`"default"`, `"rpe"`, or `"rope"`).
        use_flash_attn: Use Flash Attention.
        upcast_attn: Upcast attention to fp32.
        upcast_softmax: Upcast softmax to fp32.
        rope_base: RoPE frequency base. Only used when `attn_kind="rope"`.

    Inputs:
        x: Encoded features at the deepest encoder level.
        intermediates: List of dicts from the encoder, each containing skip
            features, positions, batch indices, serialization tensors, and
            pooling inverse indices.

    Outputs:
        Decoded features at the shallowest decoder level.
    """

    def __init__(
        self,
        encoder_channels: Sequence[int] = (32, 64, 128, 256, 512),
        decoder_depths: Sequence[int] = (2, 2, 2, 2),
        decoder_channels: Sequence[int] = (256, 128, 64, 64),
        decoder_num_head: Sequence[int] = (16, 8, 4, 4),
        decoder_patch_size: Sequence[int] = (48, 48, 48, 48),
        norm: Union[str, Callable] = "batch_norm",
        act: Union[str, Callable] = "gelu",
        mlp_ratio: float = 4,
        qkv_bias: bool = True,
        qk_scale: Optional[float] = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        drop_path: float = 0.3,
        attn_kind: AttentionKind = "default",
        use_flash_attn: bool = True,
        upcast_attn: bool = False,
        upcast_softmax: bool = False,
        act_kwargs: Optional[Dict[str, Any]] = None,
        norm_kwargs: Optional[Dict[str, Any]] = None,
        rope_base: float = 10.0,
        legacy: bool = False,
    ):
        super().__init__()
        self.blocks = self.configure_blocks(
            depths=decoder_depths,
            channels=[encoder_channels[-1]] + list(decoder_channels),
            skip_channels=list(encoder_channels[:-1])[::-1],
            num_heads=decoder_num_head,
            patch_sizes=decoder_patch_size,
            mlp_ratio=mlp_ratio,
            norm=norm,
            act=act,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            drop_path=drop_path,
            attn_kind=attn_kind,
            use_flash_attn=use_flash_attn,
            upcast_attn=upcast_attn,
            upcast_softmax=upcast_softmax,
            act_kwargs=act_kwargs,
            norm_kwargs=norm_kwargs,
            rope_base=rope_base,
            legacy=legacy,
        )

    @property
    def out_channels(self) -> int:
        return self.blocks[-1].blocks[-1].mlp[0].in_features  # type: ignore[index, union-attr]

    def configure_blocks(
        self,
        depths: Sequence[int],
        channels: Sequence[int],
        skip_channels: Sequence[int],
        num_heads: Sequence[int],
        patch_sizes: Sequence[int],
        mlp_ratio: float = 4.0,
        norm: Union[str, Callable] = "batch_norm",
        act: Union[str, Callable] = "gelu",
        qkv_bias: bool = True,
        qk_scale: Optional[float] = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        drop_path: float = 0.0,
        attn_kind: AttentionKind = "default",
        use_flash_attn: bool = True,
        upcast_attn: bool = False,
        upcast_softmax: bool = False,
        act_kwargs: Optional[Dict[str, Any]] = None,
        norm_kwargs: Optional[Dict[str, Any]] = None,
        rope_base: float = 10.0,
        legacy: bool = False,
    ) -> nn.ModuleList:
        depths = ensure_tuple(depths)
        n = len(depths)
        channels = ensure_tuple_size(channels, size=n + 1)
        skip_channels = ensure_tuple_size(skip_channels, size=n)
        num_heads = ensure_tuple_size(num_heads, size=n)
        patch_sizes = ensure_tuple_size(patch_sizes, size=n)
        drop_paths = torch.split(torch.linspace(0, drop_path, sum(depths)), list(depths))[::-1]

        blocks = nn.ModuleList()
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

            block = DecoderBlock(
                channels=channels[i + 1],
                depth=depths[i],
                num_heads=num_heads[i],
                patch_size=patch_sizes[i],
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                attn_drop=attn_drop,
                proj_drop=proj_drop,
                drop_path=drop_paths[i].tolist()[::-1],
                norm=norm,
                act=act,
                act_kwargs=act_kwargs,
                norm_kwargs=norm_kwargs,
                cpe_indice_key=f"stage{i}",
                attn_kind=attn_kind,
                use_flash_attn=use_flash_attn,
                upcast_attn=upcast_attn,
                upcast_softmax=upcast_softmax,
                upsample=upsample,
                rope_base=rope_base,
                legacy=legacy,
            )

            blocks.append(block)
        return blocks

    def forward(self, x: Tensor, intermediates: List[Dict[str, Tensor]]) -> Tuple[Tensor, Tensor, Tensor]:
        for block, intermediate in zip(self.blocks, reversed(intermediates)):
            intermediate.pop("serialized_code", None)
            intermediate = {f"{k}_skip" if k != "inverse" else k: v for k, v in intermediate.items()}
            x, pos_grid, batch = block(x, **intermediate)
        return x, pos_grid, batch


class PointTransformerV3Classification(ClassificationModel):
    """PyTorch implementation of the Point Transformer V3 model, as described in the paper
    :arxiv: [Point Transformer V3: Simpler, Faster, Stronger](https://arxiv.org/abs/2312.10035)
    by Xiaoyang Wu, Li Jiang, Peng-Shuai Wang, Zhijian Liu, Xihui Liu, Yu Qiao, Wanli Ouyang, Tong He, Hengshuang Zhao.

    This implementation is based on the original implementation from :github: [Pointcept](https://github.com/Pointcept/Pointcept).

    Important:
        This model requires `spconv`, `torch-scatter` to be installed.
        It is also recommended to install `flash-attn` for faster attention.
        In addition, it is recommended to install `ocnn` if you want to use more serialization orders.

    Args:
        in_channels: Number of input channels (corresponding to the number of features).
        num_classes: Number of output classes.
        serialization_orders: Serialization orders to use for the `PointTransformerV3Encoder`.
        shuffle_serialization_orders: Whether to shuffle the serialization orders each step.
        strides: Downsampling strides between encoder stages.
        encoder_depths: Number of encoder blocks per stage.
        encoder_channels: Number of channels per stage.
        encoder_num_head: Number of attention heads per stage.
        encoder_patch_size: Patch size per stage.
        norm: Normalization layer to use.
        act: Activation function to use.
        mlp_ratio: MLP hidden dimension ratio inside each block.
        qkv_bias: Whether to use bias in the QKV projection.
        qk_scale: Scaling factor for the QK matrix.
        attn_drop: Dropout rate for the attention.
        proj_drop: Dropout rate for the output projection of each block.
        drop_path: Stochastic depth rate.
        attn_kind: Attention variant: `"default"`, `"rpe"`, or `"rope"`.
        use_flash_attn: Whether to use flash attention.
        upcast_attn: Whether to upcast the attention to fp32.
        upcast_softmax: Whether to upcast the softmax in fp32.
        rope_base: RoPE frequency base. Only used when `attn_kind="rope"`.
        dropout: Dropout rate before the classification head.
        global_pool: How to pool point features to a batch-level vector (`"max"`, `"mean"`, *etc.*).
        pooling: Pooling between encoder stages (`"serialized"` or `"grid"`).
        stem_type: Encoder stem: `"sparse_conv"` or `"linear"`.
        act_kwargs: Optional keyword arguments for the activation factory.
        norm_kwargs: Optional keyword arguments for the normalization factory.

    Inputs:
        x: Float tensor of shape $(N, in_channels)$.
        pos_grid: Int tensor of shape $(N, 3)$ with voxel-grid coordinates.
        batch: Long tensor of shape $(N,)$.

    Outputs:
        logits: Float tensor of shape $(N, num_classes)$.
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        serialization_orders: Sequence[SerializationOrder] = ("hilbert", "hilbert-trans"),
        shuffle_serialization_orders: bool = True,
        strides: Sequence[int] = (2, 2, 2, 2),
        encoder_depths: Sequence[int] = (2, 2, 2, 6, 2),
        encoder_channels: Sequence[int] = (32, 64, 128, 256, 512),
        encoder_num_head: Sequence[int] = (2, 4, 8, 16, 32),
        encoder_patch_size: Sequence[int] = (48, 48, 48, 48, 48),
        norm: Union[str, Callable] = "batch_norm",
        act: Union[str, Callable] = "gelu",
        mlp_ratio: float = 4,
        qkv_bias: bool = True,
        qk_scale: Optional[float] = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        drop_path: float = 0.3,
        attn_kind: AttentionKind = "default",
        use_flash_attn: bool = True,
        upcast_attn: bool = False,
        upcast_softmax: bool = False,
        rope_base: float = 10.0,
        dropout: float = 0.0,
        global_pool: PoolLike = "max",
        pooling: str = "serialized",
        stem_type: str = "sparse_conv",
        act_kwargs: Optional[Dict[str, Any]] = None,
        norm_kwargs: Optional[Dict[str, Any]] = None,
        legacy: bool = False,
    ):
        super().__init__(in_channels=in_channels, num_classes=num_classes)
        self.encoder = PointTransformerV3Encoder(
            in_channels=in_channels,
            serialization_orders=serialization_orders,
            shuffle_serialization_orders=shuffle_serialization_orders,
            strides=strides,
            encoder_depths=encoder_depths,
            encoder_channels=encoder_channels,
            encoder_num_head=encoder_num_head,
            encoder_patch_size=encoder_patch_size,
            norm=norm,
            act=act,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            drop_path=drop_path,
            attn_kind=attn_kind,
            use_flash_attn=use_flash_attn,
            upcast_attn=upcast_attn,
            upcast_softmax=upcast_softmax,
            rope_base=rope_base,
            pooling=pooling,
            stem_type=stem_type,
            act_kwargs=act_kwargs,
            norm_kwargs=norm_kwargs,
            legacy=legacy,
        )
        self.dropout = dropout
        self.global_pool = create_pool(global_pool)
        self.head = nn.Identity() if self.num_classes == 0 else nn.Linear(self.embedding_dim, self.num_classes)

    @property
    def embedding_dim(self) -> int:
        return self.encoder.embedding_dim

    def reset_classifier(self, num_classes: int, global_pool: PoolLike = "max", **kwargs: Any) -> None:
        """Resets the classification head with new parameters.

        Note:
            To set an empty classification head, use `num_classes=0`.

        Args:
            num_classes: Number of output classes.
            global_pool: Pooling method to aggregate point features ("max" or "mean").
            **kwargs: Additional keyword arguments to pass to the classification head.
        """
        self.num_classes = num_classes
        self.global_pool = create_pool(global_pool)
        self.head = nn.Identity() if self.num_classes == 0 else nn.Linear(self.embedding_dim, self.num_classes)

    @overload
    def forward_features(
        self,
        x: OptTensor,
        pos_grid: Tensor,
        batch: Tensor,
        return_intermediates: Literal[True],
    ) -> Tuple[Tensor, Tensor, Tensor, List[Dict[str, Tensor]]]: ...

    @overload
    def forward_features(
        self,
        x: OptTensor,
        pos_grid: Tensor,
        batch: Tensor,
        return_intermediates: Literal[False] = False,
    ) -> Tuple[Tensor, Tensor, Tensor]: ...

    def forward_features(
        self,
        x: OptTensor,
        pos_grid: Tensor,
        batch: Tensor,
        return_intermediates: bool = False,
    ) -> Any:
        if return_intermediates:
            return self.encoder.forward(x, pos_grid, batch, return_intermediates=True)
        return self.encoder.forward(x, pos_grid, batch, return_intermediates=False)

    def forward_head(self, x: Tensor, batch: Tensor, pre_logits: bool = False) -> Tensor:
        """Forward pass of the classification head from pre-pooling features.

        Args:
            x: Pre-pooling features of shape $(N, embedding\\_dim)$.
            batch: Batch indices for each point of shape $(N,)$.
            pre_logits: Whether to return pre-logits. Defaults to False.

        Returns:
            Classification logits of shape $(B, num\\_classes)$.
        """
        x = self.global_pool(x, batch)
        if self.dropout:
            x = F.dropout(x, p=float(self.dropout), training=self.training)
        return x if pre_logits else self.head(x)

    def forward(self, x: OptTensor, pos_grid: Tensor, batch: Tensor) -> Tensor:
        """Forward pass of the PointNet classification network.

        Args:
            x: Additional point features of shape $(N, C)$.
            pos_grid: Integer grid coordinates of shape $(N, 3)$. The encoder uses
                these to derive the Z-order / Hilbert serialisation index, so they
                must be voxel indices, not float positions.
            batch: Batch indices for each point of shape $(N,)$.

        Returns:
            Classification logits of shape $(B, num\\_classes)$.
        """
        x, _pos_grid, batch = self.forward_features(x, pos_grid, batch)
        return self.forward_head(x, batch)


class PointTransformerV3Segmentation(SegmentationModel):
    """PyTorch implementation of the Point Transformer V3 model for segmentation tasks.

    Based on the paper :arxiv: [Point Transformer V3: Simpler, Faster, Stronger](https://arxiv.org/abs/2312.10035)
    by Xiaoyang Wu, Li Jiang, Peng-Shuai Wang, Zhijian Liu, Xihui Liu, Yu Qiao, Wanli Ouyang, Tong He, Hengshuang Zhao.

    This segmentation variant uses an encoder-decoder architecture with skip connections.

    Args:
        in_channels: Number of input channels.
        num_classes: Number of output classes for segmentation.
        serialization_orders: Serialization orders to use for the encoder.
        strides: Strides for the downsampling operations.
        encoder_depths: Number of encoder blocks for each stage.
        encoder_channels: Number of channels for each encoder block.
        encoder_num_head: Number of attention heads for each encoder block.
        encoder_patch_size: Patch size for each encoder block.
        decoder_depths: Number of decoder blocks for each stage.
        decoder_channels: Number of channels for each decoder block.
        decoder_num_head: Number of attention heads for each decoder block.
        decoder_patch_size: Patch size for each decoder block.
        norm: Normalization layer to use.
        act: Activation function to use.
        mlp_ratio: Ratio of the hidden dimension to the input dimension.
        qkv_bias: Whether to use bias in the QKV projection.
        qk_scale: Scaling factor for the QK matrix.
        attn_drop: Dropout rate for the attention.
        proj_drop: Dropout rate for the projection.
        drop_path: Dropout rate for the drop path.
        shuffle_serialization_orders: Whether to shuffle the serialization orders.
        attn_kind: Attention variant: `"default"`, `"rpe"`, or `"rope"`.
        rope_base: RoPE frequency base. Only used when `attn_kind="rope"`.
        use_flash_attn: Whether to use flash attention.
        upcast_attn: Whether to upcast the attention.
        upcast_softmax: Whether to upcast the softmax.
        dropout: Dropout on the per-point logits.
        pooling: Inter-stage pooling (`"serialized"` or `"grid"`).
        stem_type: Encoder stem (`"sparse_conv"` or `"linear"`).
        act_kwargs: Optional keyword arguments for the activation factory.
        norm_kwargs: Optional keyword arguments for the normalization factory.
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        serialization_orders: Sequence[SerializationOrder] = ("hilbert", "hilbert-trans"),
        strides: Sequence[int] = (2, 2, 2, 2),
        encoder_depths: Sequence[int] = (2, 2, 2, 6, 2),
        encoder_channels: Sequence[int] = (32, 64, 128, 256, 512),
        encoder_num_head: Sequence[int] = (2, 4, 8, 16, 32),
        encoder_patch_size: Sequence[int] = (48, 48, 48, 48, 48),
        decoder_depths: Sequence[int] = (2, 2, 2, 2),
        decoder_channels: Sequence[int] = (256, 128, 64, 64),
        decoder_num_head: Sequence[int] = (16, 8, 4, 4),
        decoder_patch_size: Sequence[int] = (48, 48, 48, 48),
        norm: Union[str, Callable] = "batch_norm",
        act: Union[str, Callable] = "gelu",
        mlp_ratio: float = 4,
        qkv_bias: bool = True,
        qk_scale: Optional[float] = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        drop_path: float = 0.3,
        shuffle_serialization_orders: bool = True,
        attn_kind: AttentionKind = "default",
        use_flash_attn: bool = True,
        upcast_attn: bool = False,
        upcast_softmax: bool = False,
        rope_base: float = 10.0,
        dropout: float = 0.0,
        pooling: str = "serialized",
        stem_type: str = "sparse_conv",
        act_kwargs: Optional[Dict[str, Any]] = None,
        norm_kwargs: Optional[Dict[str, Any]] = None,
        legacy: bool = False,
    ) -> None:
        super().__init__(in_channels=in_channels, num_classes=num_classes)
        self.encoder = PointTransformerV3Encoder(
            in_channels=in_channels,
            serialization_orders=serialization_orders,
            shuffle_serialization_orders=shuffle_serialization_orders,
            strides=strides,
            encoder_depths=encoder_depths,
            encoder_channels=encoder_channels,
            encoder_num_head=encoder_num_head,
            encoder_patch_size=encoder_patch_size,
            norm=norm,
            act=act,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            drop_path=drop_path,
            attn_kind=attn_kind,
            use_flash_attn=use_flash_attn,
            upcast_attn=upcast_attn,
            upcast_softmax=upcast_softmax,
            rope_base=rope_base,
            pooling=pooling,
            stem_type=stem_type,
            act_kwargs=act_kwargs,
            norm_kwargs=norm_kwargs,
            legacy=legacy,
        )
        self.decoder = PointTransformerV3Decoder(
            encoder_channels=encoder_channels,
            decoder_depths=decoder_depths,
            decoder_channels=decoder_channels,
            decoder_num_head=decoder_num_head,
            decoder_patch_size=decoder_patch_size,
            norm=norm,
            act=act,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            drop_path=drop_path,
            attn_kind=attn_kind,
            use_flash_attn=use_flash_attn,
            upcast_attn=upcast_attn,
            upcast_softmax=upcast_softmax,
            rope_base=rope_base,
            act_kwargs=act_kwargs,
            norm_kwargs=norm_kwargs,
            legacy=legacy,
        )
        self.dropout = dropout
        self.head = nn.Identity() if self.num_classes == 0 else nn.Linear(self.out_channels, self.num_classes)

    @property
    def embedding_dim(self) -> int:
        return self.encoder.embedding_dim

    @property
    def out_channels(self) -> int:
        return self.decoder.out_channels

    @overload
    def forward_features(
        self,
        x: OptTensor,
        pos_grid: Tensor,
        batch: Tensor,
        return_intermediates: Literal[True],
    ) -> Tuple[Tensor, Tensor, Tensor, List[Dict[str, Tensor]]]: ...

    @overload
    def forward_features(
        self,
        x: OptTensor,
        pos_grid: Tensor,
        batch: Tensor,
        return_intermediates: Literal[False] = False,
    ) -> Tuple[Tensor, Tensor, Tensor]: ...

    def forward_features(
        self,
        x: OptTensor,
        pos_grid: Tensor,
        batch: Tensor,
        return_intermediates: bool = False,
    ) -> Any:
        if return_intermediates:
            return self.encoder.forward(x, pos_grid, batch, return_intermediates=True)
        return self.encoder.forward(x, pos_grid, batch, return_intermediates=False)

    def forward_decoder(self, x: Tensor, intermediates: List[Dict[str, Tensor]]) -> Tuple[Tensor, Tensor, Tensor]:
        return self.decoder.forward(x, intermediates)

    def forward_head(self, x: Tensor, pre_logits: bool = False) -> Tensor:
        if self.dropout:
            x = F.dropout(x, p=float(self.dropout), training=self.training)
        return x if pre_logits else self.head(x)

    def forward(self, x: Tensor, pos_grid: Tensor, batch: Tensor) -> Tensor:
        x, _, _, intermediates = self.forward_features(x, pos_grid, batch, return_intermediates=True)
        x, _, _ = self.forward_decoder(x, intermediates)
        return self.forward_head(x)


def _ptv3_seg_hparams(num_classes: int, attn_kind: AttentionKind = "default", patch_size: int = 1024) -> Dict[str, Any]:
    """Shared PT-v3m1 segmentation hparams (ScanNet / ScanNet200 / S3DIS share the architecture).

    `legacy=True` reproduces the released v1.5.1 checkpoints. The S3DIS variant uses `attn_kind="rpe"` with
    `patch_size=128` (the relative-position table size is derived from the patch size). RPE attention computes
    the explicit attention matrix, so flash attention is disabled.
    """
    return dict(
        in_channels=6,
        num_classes=num_classes,
        serialization_orders=("z", "z-trans", "hilbert", "hilbert-trans"),
        shuffle_serialization_orders=True,
        strides=(2, 2, 2, 2),
        encoder_depths=(2, 2, 2, 6, 2),
        encoder_channels=(32, 64, 128, 256, 512),
        encoder_num_head=(2, 4, 8, 16, 32),
        encoder_patch_size=(patch_size,) * 5,
        decoder_depths=(2, 2, 2, 2),
        decoder_channels=(256, 128, 64, 64),
        decoder_num_head=(16, 8, 4, 4),
        decoder_patch_size=(patch_size,) * 4,
        mlp_ratio=4,
        drop_path=0.3,
        attn_kind=attn_kind,
        use_flash_attn=attn_kind != "rpe",
        pooling="serialized",
        stem_type="sparse_conv",
        norm="batch_norm",
        legacy=True,
    )


def _ptv3_seg_transforms(relabel_labels: Optional[Sequence[int]] = None, estimate_normals: bool = False) -> T.Compose:
    """Color+normal feature pipeline shared by the PT-v3m1 segmentation models.

    `relabel_labels` shifts the dataset's $1..K$ class labels (0 reserved for unknown) down to $0..K-1$ and
    sends everything else to the ignore index: `range(1, 21)` for ScanNet20, `range(1, 201)` for ScanNet200.
    Pass `None` when labels are already 0-based (S3DIS).

    Set `estimate_normals=True` for datasets without normals (S3DIS): normals are estimated by local PCA, which
    approximates (does not equal) the mesh normals the released weights were trained on.
    """
    steps: List[Any] = [
        T.Shift(keys=DataKeys.POS, method="bbox", axes=[0, 1]),
        T.Shift(keys=DataKeys.POS, method="min", axes=[2]),
        T.Divide(keys=DataKeys.COLOR, divisor=255),
    ]
    if estimate_normals:
        steps.append(T.EstimateNormals(keys=DataKeys.POS, normal_key=DataKeys.NORMAL, orient_to_centroid=True))
    steps += [
        T.Cat(keys=[DataKeys.COLOR, DataKeys.NORMAL], dst_key=DataKeys.X, dim=1),
        T.Voxelize(
            pos_key=DataKeys.POS,
            pos_reduce="grid",
            keys=[DataKeys.X, DataKeys.SEGMENT],
            reduce=["first", "first"],
            size=0.02,
            method="fnv",
        ),
        T.CopyItems(keys=DataKeys.POS, names=DataKeys.POS_GRID),
    ]
    if relabel_labels is not None:
        steps.append(T.Relabel(keys=DataKeys.SEGMENT, labels=relabel_labels, default=-1))
    return T.Compose(steps)


@register_model(
    "ptv3-base.scannet20",
    task="segmentation",
    weights="hf://torch-pointcloud/ptv3/ptv3-base.scannet20.pth",
    transforms=_ptv3_seg_transforms(range(1, 21)),
    hparams=_ptv3_seg_hparams(20),
)
def ptv3_base_scannet20(**hparams: Any) -> PointTransformerV3Segmentation:
    return PointTransformerV3Segmentation(**hparams)


@register_model(
    "ptv3-base.scannet200",
    task="segmentation",
    weights="hf://torch-pointcloud/ptv3/ptv3-base.scannet200.pth",
    transforms=_ptv3_seg_transforms(range(1, 201)),
    hparams=_ptv3_seg_hparams(200),
)
def ptv3_base_scannet200(**hparams: Any) -> PointTransformerV3Segmentation:
    return PointTransformerV3Segmentation(**hparams)


@register_model(
    "ptv3-base.s3dis-area5",
    task="segmentation",
    weights="hf://torch-pointcloud/ptv3/ptv3-base.s3dis-area5.pth",
    transforms=_ptv3_seg_transforms(None, estimate_normals=True),
    hparams=_ptv3_seg_hparams(13, attn_kind="rpe", patch_size=128),
)
def ptv3_base_s3dis_area5(**hparams: Any) -> PointTransformerV3Segmentation:
    return PointTransformerV3Segmentation(**hparams)
