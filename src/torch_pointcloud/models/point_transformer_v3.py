"""Point Transformer V3 classification and segmentation models.

{{ paper("2312.10035") }}
"""

from typing import TYPE_CHECKING, Any, Callable, Dict, List, Literal, Optional, Sequence, Tuple, Union, overload

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

import torch_pointcloud.transforms as T
from torch_pointcloud.datasets.s3dis import S3DIS_CLASSES
from torch_pointcloud.datasets.scannet import SCANNET20_CLASSES
from torch_pointcloud.layers import PoolLike, create_pool
from torch_pointcloud.layers.act import create_act
from torch_pointcloud.layers.dropouts import DropPath
from torch_pointcloud.layers.grid_pool import GridPool
from torch_pointcloud.layers.linear_blocks import LinearBlock
from torch_pointcloud.layers.norms import create_norm
from torch_pointcloud.layers.serialized_attention import (
    SerializedAttention,
    SerializedAttentionRoPE,
    SerializedAttentionRPE,
)
from torch_pointcloud.layers.serialized_pool import SerializedPool, SerializedUpsample
from torch_pointcloud.layers.spconv_blocks import SubMConv3dBlock
from torch_pointcloud.models._base import ClassificationModel, SegmentationModel
from torch_pointcloud.models._registry import WeightsDict, register_model
from torch_pointcloud.utils.conversion import convert_to_spconv_tensor, ensure_tuple, ensure_tuple_size
from torch_pointcloud.utils.data import DataKeys
from torch_pointcloud.utils.imports import _SPCONV_GITHUB_URL, optional_import
from torch_pointcloud.utils.serialization import SerializationOrder, serialize_coords
from torch_pointcloud.utils.types import OptTensor, ValueCollection

if TYPE_CHECKING:
    import spconv.pytorch as spconv


spconv, _ = optional_import("spconv.pytorch", url=_SPCONV_GITHUB_URL)

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
    """Encode voxel-grid coordinates along one or more space-filling curves and sort the points by each code.

    Args:
        pos_grid: Non-negative integer grid coordinates of shape $(N, 3)$.
        batch: Per-point batch index of shape $(N,)$.
        orders: The $L$ space-filling curves to encode along, one code row per order.
        shuffle: Whether to permute `orders` before encoding, so consecutive blocks pick different curves.

    Returns:
        The serialization codes, the permutation sorting the points by code, and its inverse, each of
        shape $(L, N)$.

    Raises:
        ValueError: If `pos_grid` holds a negative coordinate, which would silently wrap around to a valid code.
    """
    if shuffle:
        perm = torch.randperm(len(orders))
        orders = [orders[i] for i in perm]

    if pos_grid.numel() and bool(pos_grid.min() < 0):
        raise ValueError(
            "Grid coordinates must be non-negative for serialization: negative values silently wrap around to "
            "valid codes. Shift by the per-axis minimum, as `Voxelize` does."
        )
    # An all-zero grid (single-voxel scene) has bit_length 0, which the encoders reject.
    depth = max(int(pos_grid.max()).bit_length(), 1)
    serialized_code = torch.stack([serialize_coords(pos_grid, batch, depth=depth, order=order) for order in orders])
    serialized_order = torch.argsort(serialized_code, dim=1)
    serialized_inverse = torch.argsort(serialized_order, dim=1)
    return serialized_code, serialized_order, serialized_inverse


def _resolve_condition(
    condition: Union[str, Sequence[str], None],
    default: Optional[str],
    conditions: Optional[Sequence[str]],
) -> Optional[str]:
    """Reduce a per-batch condition to a single name, falling back to `default`.

    Multi-dataset batches are single-domain, so a collated `condition` is a length-$B$ list of the
    same string; take its first entry. A `None` argument falls back to the model's constructor
    `condition` (single-dataset fine-tune or benchmark).

    Args:
        condition: A single condition name, a per-sample sequence of names, or `None`.
        default: Fallback condition name used when `condition` is `None`.
        conditions: Condition names the model was built with (`pdnorm_conditions`), or `None` when
            the model uses plain norms.

    Returns:
        The resolved condition name, or `None` when the model was built without conditions.

    Raises:
        ValueError: If a condition is given but the model was built without `pdnorm_conditions`, or
            if the model was built with `pdnorm_conditions` and no condition resolves.
    """
    resolved = condition if condition is not None else default
    if resolved is not None and not isinstance(resolved, str):
        resolved = resolved[0]
    if resolved is not None and conditions is None:
        raise ValueError(
            f"Got condition={resolved!r} but the model was built without conditional norms; "
            "construct it with `pdnorm_conditions=[...]` to enable per-dataset conditions."
        )
    if resolved is None and conditions is not None:
        raise ValueError(
            f"The model was built with pdnorm_conditions={list(conditions)!r}; pass `condition=` at "
            "forward time or set the constructor `condition` default."
        )
    return resolved


class PointTransformerV3Block(nn.Module):
    """Transformer block over serialized patches: an xCPE sparse-convolution residual, then pre-normed
    patch attention and an MLP.
    """

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
        # PTv3 blocks always use graph LayerNorm, so only the condition list rides `norm_kwargs`.
        conditions = norm_kwargs.get("conditions")
        self.cpe_norm: nn.Module = (
            create_norm("layer_norm", channels, conditions=conditions, mode="node") or nn.Identity()
        )
        self.norm1: nn.Module = create_norm("layer_norm", channels, conditions=conditions, mode="node") or nn.Identity()
        self.norm2: nn.Module = create_norm("layer_norm", channels, conditions=conditions, mode="node") or nn.Identity()
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
        condition: Optional[str] = None,
    ) -> Tuple[Tensor, Any]:
        norm_kwargs = {} if condition is None else {"condition": condition}
        shortcut = x
        x_sparse = self.cpe_conv(x_sparse)
        x = self.cpe_proj(x_sparse.features)
        x = self.cpe_norm(x, **norm_kwargs)
        x = shortcut = shortcut + x

        # Attention branch
        x = self.norm1(x, **norm_kwargs)
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
        x = self.norm2(x, **norm_kwargs)
        x = self.mlp(x)
        x = self.drop_path(x)
        x_sparse = x_sparse.replace_feature(x if self.legacy else shortcut + x)
        x = shortcut + x

        return x, x_sparse


class PointTransformerV3EncoderBlock(nn.Module):
    """One encoder stage: an optional pooling downsampling, then `depth` `PointTransformerV3Block` units.

    Consecutive blocks cycle through the available serialization orders, so each attends over a
    differently ordered patch partition. Grid pooling re-serializes the pooled cloud, while
    serialized pooling derives the coarser codes by bit-shifting the finer ones.
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
                PointTransformerV3Block(
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
        condition: Optional[str] = None,
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
        condition: Optional[str] = None,
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
        condition: Optional[str] = None,
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
                x, pos_grid, batch, inverse, pos = self.downsample(x, pos_grid, batch, pos=pos, condition=condition)
                serialized_code, serialized_order, serialized_inverse = serialize(
                    pos_grid,
                    batch,
                    orders=self.serialization_orders,
                    shuffle=self.shuffle_serialization_orders and self.training,
                )
            else:
                x, pos_grid, batch, serialized_code, inverse, pos = self.downsample(
                    x,
                    pos_grid,
                    batch,
                    serialized_code,
                    return_inverse=True,
                    pos=pos,
                    condition=condition,
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
                condition=condition,
            )

        if return_inverse:
            return x, pos_grid, batch, serialized_code, serialized_order, serialized_inverse, inverse, pos
        return x, pos_grid, batch, serialized_code, serialized_order, serialized_inverse, pos


class PointTransformerV3DecoderBlock(nn.Module):
    """One decoder stage: an optional upsampling onto the skip resolution, then `depth` `PointTransformerV3Block` units
    cycling through the skip's serialization orders.
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
                PointTransformerV3Block(
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
        pos_skip: OptTensor = None,
        condition: Optional[str] = None,
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

            # The next block's xCPE convolves only the projected skip branch. The reference does this in
            # every release, so it is unconditional (unlike the block write-back).
            x, cpe_seed = self.upsample(x, x_skip, inverse, return_intermediate=True, condition=condition)

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
                pos=pos_skip,
                condition=condition,
            )

        return x, pos_grid_skip, batch_skip


class PointTransformerV3Encoder(nn.Module):
    r"""Point Transformer V3 encoder backbone.

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
        encoder_num_heads: Attention heads per encoder stage.
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
            (Utonia, 3D rotary position embedding on `Q`, `K`). The `"rope"` variant requires the
            real-valued `pos` argument at forward time.
        use_flash_attn: Use Flash Attention. The registered configurations construct with
            `use_flash_attn=True`, which requires `flash-attn` and a CUDA device; pass
            `use_flash_attn=False` to run without it (the xCPE sparse convolution still needs a
            `spconv` build matching the device; the standard CUDA wheel cannot run on CPU).
        upcast_attn: Upcast attention to fp32.
        upcast_softmax: Upcast softmax to fp32.
        pooling: Pooling strategy: `"serialized"` (code-space bit-shift) or
            `"grid"` (grid-coordinate clustering).
        stem_type: How to embed raw features: `"sparse_conv"` (SubMConv3d stem) or
            `"linear"` (linear projection).
        rope_base: RoPE frequency base. Only used when `attn_kind="rope"`.
        act_kwargs: Optional keyword arguments for the activation factory.
        norm_kwargs: Optional keyword arguments for the normalization factory.
        bias: Whether the stem and blocks use learnable bias where applicable.
        legacy: Reproduce the reference implementation's v1.5.1 block xCPE bug (the block output was not written back to
            the sparse tensor the next block convolves; fixed in v1.5.2). The released weights need
            `legacy=True`; leave `False` (default) for new training.

    Inputs:
        x: Float tensor of shape $(N, \text{in\_channels})$.
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
        encoder_num_heads: Sequence[int] = (2, 4, 8, 16, 32),
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
            num_heads=encoder_num_heads,
            patch_sizes=encoder_patch_size,
            strides=strides,
            mlp_ratio=mlp_ratio,
            bias=bias,
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
        """Feature dimension $C$ of the encoder output."""
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
        """Build the embedding stem, either a `LinearBlock` or a `SubMConv3dBlock`."""
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
        """Build the `PointTransformerV3EncoderBlock` stages, giving every stage but the first a pooling downsampling."""
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

            block = PointTransformerV3EncoderBlock(
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
        condition: Optional[str] = None,
    ) -> Tuple[Tensor, Tensor, Tensor, List[Dict[str, Tensor]]]: ...

    @overload
    def forward(
        self,
        x: OptTensor,
        pos_grid: Tensor,
        batch: Tensor,
        return_intermediates: Literal[False] = False,
        pos: OptTensor = None,
        condition: Optional[str] = None,
    ) -> Tuple[Tensor, Tensor, Tensor]: ...

    def forward(
        self,
        x: OptTensor,
        pos_grid: Tensor,
        batch: Tensor,
        return_intermediates: bool = False,
        pos: OptTensor = None,
        condition: Optional[str] = None,
    ) -> Any:
        x = x if x is not None else pos_grid.float()

        serialized_code, serialized_order, serialized_inverse = serialize(
            pos_grid,
            batch,
            orders=self.serialization_orders,
            shuffle=self.shuffle_serialization_orders and self.training,
        )

        x = self.stem(x, pos_grid, batch, condition=condition)

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
            if pos is not None:
                intermediate["pos"] = pos

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
                condition=condition,
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
        decoder_num_heads: Attention heads per decoder stage.
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
        decoder_num_heads: Sequence[int] = (16, 8, 4, 4),
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
            num_heads=decoder_num_heads,
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
        """Feature dimension $C$ of the decoder output."""
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
        """Build the `PointTransformerV3DecoderBlock` stages, giving every stage an upsampling onto its skip resolution."""
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

            block = PointTransformerV3DecoderBlock(
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

    def forward(
        self, x: Tensor, intermediates: List[Dict[str, Tensor]], condition: Optional[str] = None
    ) -> Tuple[Tensor, Tensor, Tensor]:
        for block, intermediate in zip(self.blocks, reversed(intermediates)):
            skip_kwargs = {
                f"{k}_skip" if k != "inverse" else k: v for k, v in intermediate.items() if k != "serialized_code"
            }
            x, pos_grid, batch = block(x, **skip_kwargs, condition=condition)
        return x, pos_grid, batch


class PointTransformerV3Classification(ClassificationModel):
    r"""PyTorch implementation of the Point Transformer V3 model, as described in the paper
    :arxiv: [Point Transformer V3: Simpler, Faster, Stronger](https://arxiv.org/abs/2312.10035)
    by Xiaoyang Wu, Li Jiang, Peng-Shuai Wang, Zhijian Liu, Xihui Liu, Yu Qiao, Wanli Ouyang, Tong He, Hengshuang Zhao.

    This implementation is based on the original implementation from :github: [Pointcept](https://github.com/Pointcept/Pointcept).

    Important:
        This model requires `spconv`, `torch-scatter` to be installed.
        It is also recommended to install `flash-attn` for faster attention. The registered
        configurations construct with `use_flash_attn=True`, which requires `flash-attn` and a CUDA
        device; pass `use_flash_attn=False` to run without it. The xCPE sparse convolution still
        needs a `spconv` build matching the device; the standard CUDA wheel cannot run on CPU.
        In addition, it is recommended to install `ocnn` if you want to use more serialization orders.

    Args:
        in_channels: Number of input channels (corresponding to the number of features).
        num_classes: Number of output classes.
        serialization_orders: Serialization orders to use for the `PointTransformerV3Encoder`.
        shuffle_serialization_orders: Whether to shuffle the serialization orders each step.
        strides: Downsampling strides between encoder stages.
        encoder_depths: Number of encoder blocks per stage.
        encoder_channels: Number of channels per stage.
        encoder_num_heads: Number of attention heads per stage.
        encoder_patch_size: Patch size per stage.
        norm: Normalization layer to use.
        act: Activation function to use.
        mlp_ratio: MLP hidden dimension ratio inside each block.
        qkv_bias: Whether to use bias in the QKV projection.
        qk_scale: Scaling factor for the QK matrix.
        attn_drop: Dropout rate for the attention.
        proj_drop: Dropout rate for the output projection of each block.
        drop_path: Stochastic depth rate.
        attn_kind: Attention variant: `"default"`, `"rpe"`, or `"rope"`. The `"rope"` variant
            requires the real-valued `pos` argument at forward time.
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
        x: Float tensor of shape $(N, \text{in\_channels})$.
        pos_grid: Int tensor of shape $(N, 3)$ with voxel-grid coordinates.
        batch: Long tensor of shape $(N,)$.
        pos: Float tensor of shape $(N, 3)$ with metric coordinates. Required when
            `attn_kind="rope"`.

    Outputs:
        logits: Float tensor of shape $(N, \text{num\_classes})$.
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
        encoder_num_heads: Sequence[int] = (2, 4, 8, 16, 32),
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
        pdnorm_conditions: Optional[Sequence[str]] = None,
        condition: Optional[str] = None,
    ):
        super().__init__(in_channels=in_channels, num_classes=num_classes)
        self.condition = condition
        self.pdnorm_conditions = pdnorm_conditions
        norm_kwargs = norm_kwargs or {}
        if pdnorm_conditions is not None:
            norm_kwargs = {**norm_kwargs, "conditions": pdnorm_conditions}
        self.serialization_orders = serialization_orders
        self.shuffle_serialization_orders = shuffle_serialization_orders
        self.strides = strides
        self.encoder_depths = encoder_depths
        self.encoder_channels = encoder_channels
        self.encoder_num_heads = encoder_num_heads
        self.encoder_patch_size = encoder_patch_size
        self.norm = norm
        self.act = act
        self.act_kwargs = act_kwargs
        self.norm_kwargs = norm_kwargs
        self.mlp_ratio = mlp_ratio
        self.qkv_bias = qkv_bias
        self.qk_scale = qk_scale
        self.attn_drop = attn_drop
        self.proj_drop = proj_drop
        self.drop_path = drop_path
        self.attn_kind = attn_kind
        self.use_flash_attn = use_flash_attn
        self.upcast_attn = upcast_attn
        self.upcast_softmax = upcast_softmax
        self.rope_base = rope_base
        self.pooling = pooling
        self.stem_type = stem_type
        self.legacy = legacy
        self.dropout = dropout

        self.encoder = self.configure_encoder()
        self.global_pool = create_pool(global_pool)
        self.head = self.configure_head()

    def configure_encoder(self) -> PointTransformerV3Encoder:
        """Build the `PointTransformerV3Encoder` backbone."""
        return PointTransformerV3Encoder(
            in_channels=self.in_channels,
            serialization_orders=self.serialization_orders,
            shuffle_serialization_orders=self.shuffle_serialization_orders,
            strides=self.strides,
            encoder_depths=self.encoder_depths,
            encoder_channels=self.encoder_channels,
            encoder_num_heads=self.encoder_num_heads,
            encoder_patch_size=self.encoder_patch_size,
            norm=self.norm,
            act=self.act,
            mlp_ratio=self.mlp_ratio,
            qkv_bias=self.qkv_bias,
            qk_scale=self.qk_scale,
            attn_drop=self.attn_drop,
            proj_drop=self.proj_drop,
            drop_path=self.drop_path,
            attn_kind=self.attn_kind,
            use_flash_attn=self.use_flash_attn,
            upcast_attn=self.upcast_attn,
            upcast_softmax=self.upcast_softmax,
            rope_base=self.rope_base,
            pooling=self.pooling,
            stem_type=self.stem_type,
            act_kwargs=self.act_kwargs,
            norm_kwargs=self.norm_kwargs,
            legacy=self.legacy,
        )

    @property
    def num_features(self) -> int:
        """Feature dimension $C$ of the encoder output."""
        return self.encoder.embedding_dim

    def configure_head(self) -> nn.Module:
        if self.num_classes == 0:
            return nn.Identity()
        return nn.Linear(self.num_features, self.num_classes)

    def reset_classifier(self, num_classes: int, global_pool: Optional[PoolLike] = None, **kwargs: Any) -> None:
        """Resets the classification head with new parameters.

        Note:
            To set an empty classification head, use `num_classes=0`.

        Args:
            num_classes: Number of output classes.
            global_pool: Pooling method to aggregate point features ("max" or "mean").
                `None` keeps the current pooling.
            **kwargs: Additional keyword arguments to pass to the classification head.
        """
        self.num_classes = num_classes
        if global_pool is not None:
            self.global_pool = create_pool(global_pool)
        self.head = self.configure_head()

    @overload
    def forward_features(
        self,
        x: OptTensor,
        pos_grid: Tensor,
        batch: Tensor,
        return_intermediates: Literal[True],
        pos: OptTensor = None,
        condition: Optional[str] = None,
    ) -> Tuple[Tensor, Tensor, Tensor, List[Dict[str, Tensor]]]: ...

    @overload
    def forward_features(
        self,
        x: OptTensor,
        pos_grid: Tensor,
        batch: Tensor,
        return_intermediates: Literal[False] = False,
        pos: OptTensor = None,
        condition: Optional[str] = None,
    ) -> Tuple[Tensor, Tensor, Tensor]: ...

    def forward_features(
        self,
        x: OptTensor,
        pos_grid: Tensor,
        batch: Tensor,
        return_intermediates: bool = False,
        pos: OptTensor = None,
        condition: Optional[str] = None,
    ) -> Any:
        if return_intermediates:
            return self.encoder.forward(x, pos_grid, batch, return_intermediates=True, pos=pos, condition=condition)
        return self.encoder.forward(x, pos_grid, batch, return_intermediates=False, pos=pos, condition=condition)

    def forward_head(self, x: Tensor, batch: Tensor, pre_logits: bool = False) -> Tensor:
        r"""Forward pass of the classification head from pre-pooling features.

        Args:
            x: Pre-pooling features of shape $(N, \text{embedding\_dim})$.
            batch: Batch indices for each point of shape $(N,)$.
            pre_logits: Whether to return pre-logits. Defaults to False.

        Returns:
            Classification logits of shape $(B, \text{num\_classes})$.
        """
        x = self.global_pool(x, batch)
        if self.dropout:
            x = F.dropout(x, p=float(self.dropout), training=self.training)
        return x if pre_logits else self.head(x)

    def forward(
        self,
        x: OptTensor,
        pos_grid: Tensor,
        batch: Tensor,
        condition: Union[str, Sequence[str], None] = None,
        pos: OptTensor = None,
    ) -> Tensor:
        r"""Forward pass of the Point Transformer V3 classification network.

        Args:
            x: Additional point features of shape $(N, C)$.
            pos_grid: Integer grid coordinates of shape $(N, 3)$. The encoder uses
                these to derive the Z-order / Hilbert serialization index, so they
                must be voxel indices, not float positions.
            batch: Batch indices for each point of shape $(N,)$.
            condition: Optional per-batch condition selecting the PDNorm inner norms.
            pos: Real-valued metric positions of shape $(N, 3)$. Required when
                `attn_kind="rope"`; ignored otherwise.

        Returns:
            Classification logits of shape $(B, \text{num\_classes})$.
        """
        x, _pos_grid, batch = self.forward_features(
            x, pos_grid, batch, pos=pos, condition=_resolve_condition(condition, self.condition, self.pdnorm_conditions)
        )
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
        encoder_num_heads: Number of attention heads for each encoder block.
        encoder_patch_size: Patch size for each encoder block.
        decoder_depths: Number of decoder blocks for each stage.
        decoder_channels: Number of channels for each decoder block.
        decoder_num_heads: Number of attention heads for each decoder block.
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
        attn_kind: Attention variant: `"default"`, `"rpe"`, or `"rope"`. The `"rope"` variant
            requires the real-valued `pos` argument at forward time.
        rope_base: RoPE frequency base. Only used when `attn_kind="rope"`.
        use_flash_attn: Whether to use flash attention. The registered configurations construct
            with `use_flash_attn=True`, which requires `flash-attn` and a CUDA device; pass
            `use_flash_attn=False` to run without it (the xCPE sparse convolution still needs a
            `spconv` build matching the device; the standard CUDA wheel cannot run on CPU).
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
        encoder_num_heads: Sequence[int] = (2, 4, 8, 16, 32),
        encoder_patch_size: Sequence[int] = (48, 48, 48, 48, 48),
        decoder_depths: Sequence[int] = (2, 2, 2, 2),
        decoder_channels: Sequence[int] = (256, 128, 64, 64),
        decoder_num_heads: Sequence[int] = (16, 8, 4, 4),
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
        pdnorm_conditions: Optional[Sequence[str]] = None,
        condition: Optional[str] = None,
    ) -> None:
        super().__init__(in_channels=in_channels, num_classes=num_classes)
        self.condition = condition
        self.pdnorm_conditions = pdnorm_conditions
        norm_kwargs = norm_kwargs or {}
        if pdnorm_conditions is not None:
            norm_kwargs = {**norm_kwargs, "conditions": pdnorm_conditions}
        self.serialization_orders = serialization_orders
        self.shuffle_serialization_orders = shuffle_serialization_orders
        self.strides = strides
        self.encoder_depths = encoder_depths
        self.encoder_channels = encoder_channels
        self.encoder_num_heads = encoder_num_heads
        self.encoder_patch_size = encoder_patch_size
        self.decoder_depths = decoder_depths
        self.decoder_channels = decoder_channels
        self.decoder_num_heads = decoder_num_heads
        self.decoder_patch_size = decoder_patch_size
        self.norm = norm
        self.act = act
        self.act_kwargs = act_kwargs
        self.norm_kwargs = norm_kwargs
        self.mlp_ratio = mlp_ratio
        self.qkv_bias = qkv_bias
        self.qk_scale = qk_scale
        self.attn_drop = attn_drop
        self.proj_drop = proj_drop
        self.drop_path = drop_path
        self.attn_kind = attn_kind
        self.use_flash_attn = use_flash_attn
        self.upcast_attn = upcast_attn
        self.upcast_softmax = upcast_softmax
        self.rope_base = rope_base
        self.pooling = pooling
        self.stem_type = stem_type
        self.legacy = legacy
        self.dropout = dropout

        self.encoder = self.configure_encoder()
        self.decoder = self.configure_decoder()
        self.head = self.configure_head()

    def configure_encoder(self) -> PointTransformerV3Encoder:
        """Build the `PointTransformerV3Encoder` backbone."""
        return PointTransformerV3Encoder(
            in_channels=self.in_channels,
            serialization_orders=self.serialization_orders,
            shuffle_serialization_orders=self.shuffle_serialization_orders,
            strides=self.strides,
            encoder_depths=self.encoder_depths,
            encoder_channels=self.encoder_channels,
            encoder_num_heads=self.encoder_num_heads,
            encoder_patch_size=self.encoder_patch_size,
            norm=self.norm,
            act=self.act,
            mlp_ratio=self.mlp_ratio,
            qkv_bias=self.qkv_bias,
            qk_scale=self.qk_scale,
            attn_drop=self.attn_drop,
            proj_drop=self.proj_drop,
            drop_path=self.drop_path,
            attn_kind=self.attn_kind,
            use_flash_attn=self.use_flash_attn,
            upcast_attn=self.upcast_attn,
            upcast_softmax=self.upcast_softmax,
            rope_base=self.rope_base,
            pooling=self.pooling,
            stem_type=self.stem_type,
            act_kwargs=self.act_kwargs,
            norm_kwargs=self.norm_kwargs,
            legacy=self.legacy,
        )

    def configure_decoder(self) -> PointTransformerV3Decoder:
        """Build the `PointTransformerV3Decoder` upsampling the coarsest features back through the encoder skips."""
        return PointTransformerV3Decoder(
            encoder_channels=self.encoder_channels,
            decoder_depths=self.decoder_depths,
            decoder_channels=self.decoder_channels,
            decoder_num_heads=self.decoder_num_heads,
            decoder_patch_size=self.decoder_patch_size,
            norm=self.norm,
            act=self.act,
            mlp_ratio=self.mlp_ratio,
            qkv_bias=self.qkv_bias,
            qk_scale=self.qk_scale,
            attn_drop=self.attn_drop,
            proj_drop=self.proj_drop,
            drop_path=self.drop_path,
            attn_kind=self.attn_kind,
            use_flash_attn=self.use_flash_attn,
            upcast_attn=self.upcast_attn,
            upcast_softmax=self.upcast_softmax,
            rope_base=self.rope_base,
            act_kwargs=self.act_kwargs,
            norm_kwargs=self.norm_kwargs,
            legacy=self.legacy,
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
        x: OptTensor,
        pos_grid: Tensor,
        batch: Tensor,
        return_intermediates: Literal[True],
        pos: OptTensor = None,
        condition: Optional[str] = None,
    ) -> Tuple[Tensor, Tensor, Tensor, List[Dict[str, Tensor]]]: ...

    @overload
    def forward_features(
        self,
        x: OptTensor,
        pos_grid: Tensor,
        batch: Tensor,
        return_intermediates: Literal[False] = False,
        pos: OptTensor = None,
        condition: Optional[str] = None,
    ) -> Tuple[Tensor, Tensor, Tensor]: ...

    def forward_features(
        self,
        x: OptTensor,
        pos_grid: Tensor,
        batch: Tensor,
        return_intermediates: bool = False,
        pos: OptTensor = None,
        condition: Optional[str] = None,
    ) -> Any:
        if return_intermediates:
            return self.encoder.forward(x, pos_grid, batch, return_intermediates=True, pos=pos, condition=condition)
        return self.encoder.forward(x, pos_grid, batch, return_intermediates=False, pos=pos, condition=condition)

    def forward_decoder(
        self, x: Tensor, intermediates: List[Dict[str, Tensor]], condition: Optional[str] = None
    ) -> Tuple[Tensor, Tensor, Tensor]:
        return self.decoder.forward(x, intermediates, condition=condition)

    def forward_head(self, x: Tensor, pre_logits: bool = False) -> Tensor:
        if self.dropout:
            x = F.dropout(x, p=float(self.dropout), training=self.training)
        return x if pre_logits else self.head(x)

    def forward(
        self,
        x: Tensor,
        pos_grid: Tensor,
        batch: Tensor,
        condition: Union[str, Sequence[str], None] = None,
        pos: OptTensor = None,
    ) -> Tensor:
        r"""Forward pass of the Point Transformer V3 segmentation network.

        Args:
            x: Per-point features of shape $(N, C)$.
            pos_grid: Integer grid coordinates of shape $(N, 3)$ used for serialization.
            batch: Batch indices for each point of shape $(N,)$.
            condition: Optional per-batch condition selecting the PDNorm inner norms.
            pos: Real-valued metric positions of shape $(N, 3)$. Required when
                `attn_kind="rope"`; ignored otherwise.

        Returns:
            Per-point segmentation logits of shape $(N, \text{num\_classes})$.
        """
        resolved = _resolve_condition(condition, self.condition, self.pdnorm_conditions)
        x, _, _, intermediates = self.forward_features(
            x, pos_grid, batch, return_intermediates=True, pos=pos, condition=resolved
        )
        x, _, _ = self.forward_decoder(x, intermediates, condition=resolved)
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
        encoder_num_heads=(2, 4, 8, 16, 32),
        encoder_patch_size=(patch_size,) * 5,
        decoder_depths=(2, 2, 2, 2),
        decoder_channels=(256, 128, 64, 64),
        decoder_num_heads=(16, 8, 4, 4),
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
    steps.append(T.Cat(keys=[DataKeys.COLOR, DataKeys.NORMAL], dst_key=DataKeys.X, dim=1))
    if relabel_labels is not None:
        steps.append(T.Relabel(keys=DataKeys.SEGMENT, labels=relabel_labels, default=-1))
    steps += [
        T.CopyItems(
            keys=[DataKeys.POS, DataKeys.SEGMENT],
            names=[DataKeys.ORIGIN_POS, DataKeys.ORIGIN_SEGMENT],
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


@register_model(
    "ptv3-base.scannet20.pointcept",
    task="segmentation",
    weights=WeightsDict(
        url="hf://torch-pointcloud/ptv3/ptv3-base.scannet20.pointcept.safetensors",
        dataset="scannet20",
        metrics={"mIoU": 76.29},
        classes=SCANNET20_CLASSES,
        author="pointcept",
        license="MIT",
    ),
    transform=_ptv3_seg_transforms(range(1, 21)),
    hparams=_ptv3_seg_hparams(20),
)
def ptv3_base_scannet20(**hparams: Any) -> PointTransformerV3Segmentation:
    return PointTransformerV3Segmentation(**hparams)


@register_model(
    "ptv3-base.scannet200.pointcept",
    task="segmentation",
    weights=WeightsDict(
        url="hf://torch-pointcloud/ptv3/ptv3-base.scannet200.pointcept.safetensors",
        dataset="scannet200",
        metrics={"mIoU": 33.42},
        author="pointcept",
        license="MIT",
    ),
    transform=_ptv3_seg_transforms(range(1, 201)),
    hparams=_ptv3_seg_hparams(200),
)
def ptv3_base_scannet200(**hparams: Any) -> PointTransformerV3Segmentation:
    return PointTransformerV3Segmentation(**hparams)


@register_model(
    "ptv3-base.s3dis-area5.pointcept",
    task="segmentation",
    weights=WeightsDict(
        url="hf://torch-pointcloud/ptv3/ptv3-base.s3dis-area5.pointcept.safetensors",
        dataset="s3dis-area5",
        metrics={"mIoU": 32.06},
        classes=S3DIS_CLASSES,
        author="pointcept",
        license="MIT",
    ),
    transform=_ptv3_seg_transforms(None, estimate_normals=True),
    hparams=_ptv3_seg_hparams(13, attn_kind="rpe", patch_size=128),
)
def ptv3_base_s3dis_area5(**hparams: Any) -> PointTransformerV3Segmentation:
    return PointTransformerV3Segmentation(**hparams)
