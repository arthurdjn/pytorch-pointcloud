from typing import TYPE_CHECKING, Any, Callable, Dict, List, Literal, Optional, Sequence, Tuple, Union, overload

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn.resolver import activation_resolver, normalization_resolver

import torch_pointcloud.transforms as T
from torch_pointcloud.layers import PoolLike, create_cls_head, create_pool
from torch_pointcloud.layers.dropouts import DropPath
from torch_pointcloud.layers.grid_pool import GridPool
from torch_pointcloud.layers.linear_blocks import LinearBlock
from torch_pointcloud.layers.serialized_attention import SerializedAttention
from torch_pointcloud.layers.serialized_pool import SerializedPool, SerializedUpsample
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


def serialize(
    pos: Tensor,
    batch: Tensor,
    orders: Sequence[SerializationOrder],
    shuffle: bool = False,
) -> Tuple[Tensor, Tensor, Tensor]:
    if shuffle:
        perm = torch.randperm(len(orders))
        orders = [orders[i] for i in perm]

    depth = int(pos.max()).bit_length()
    serialized_code = torch.stack([serialize_coords(pos, batch, depth=depth, order=order) for order in orders])
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
        use_rpe: bool = False,
        use_flash_attn: bool = True,
        upcast_attention: bool = True,
        upcast_softmax: bool = True,
    ):
        super().__init__()
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
        self.cpe_norm = normalization_resolver(norm, channels, **norm_kwargs)

        self.norm1 = normalization_resolver(norm, channels, **norm_kwargs)
        self.attn = SerializedAttention(
            channels=channels,
            patch_size=patch_size,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            use_rpe=use_rpe,
            use_flash_attn=use_flash_attn,
            upcast_attention=upcast_attention,
            upcast_softmax=upcast_softmax,
        )

        self.norm2 = normalization_resolver(norm, channels, **norm_kwargs)
        self.mlp = nn.Sequential(
            nn.Linear(channels, int(channels * mlp_ratio)),
            activation_resolver(act, **act_kwargs),
            nn.Dropout(proj_drop),
            nn.Linear(int(channels * mlp_ratio), channels),
            nn.Dropout(proj_drop),
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(
        self,
        x: Tensor,
        pos: Tensor,
        batch: Tensor,
        serialized_order: OptTensor = None,
        serialized_inverse: OptTensor = None,
    ) -> Any:
        # Conv + Skip connection
        shortcut = x
        sparse_x = convert_to_spconv_tensor(x, pos, batch)
        sparse_x = self.cpe_conv(sparse_x)
        x = sparse_x.features
        x = self.cpe_proj(x)
        x = self.cpe_norm(x)

        # NOTE: Skip connection and save the new shortcut
        x = shortcut = shortcut + x

        # Attention branch
        x = self.norm1(x)
        x = self.attn(
            x,
            pos,
            batch,
            serialized_order=serialized_order,
            serialized_inverse=serialized_inverse,
        )
        x = self.drop_path(x)
        x = shortcut + x

        # MLP branch
        shortcut = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = self.drop_path(x)
        x = shortcut + x

        return x


class SubMConv3dBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        padding: int,
        norm: Union[str, Callable, None] = None,
        act: Union[str, Callable, None] = None,
        act_kwargs: Optional[Dict[str, Any]] = None,
        norm_kwargs: Optional[Dict[str, Any]] = None,
        bias: bool = True,
        stem_indice_key: Optional[str] = None,
    ):
        super().__init__()
        norm_kwargs = norm_kwargs or {}
        act_kwargs = act_kwargs or {}

        self.stem = spconv.SubMConv3d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=bias,
            indice_key=stem_indice_key,
        )
        self.norm = normalization_resolver(norm, out_channels, **norm_kwargs) if norm is not None else None
        self.act = activation_resolver(act, **act_kwargs) if act is not None else None

    def forward(
        self,
        x: Tensor,
        pos: Tensor,
        batch: Tensor,
    ) -> Tensor:
        sparse_x = convert_to_spconv_tensor(x, pos, batch)
        sparse_x = self.stem(sparse_x)

        x = sparse_x.features
        if self.norm is not None:
            x = self.norm(x)
        if self.act is not None:
            x = self.act(x)

        return x


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
        use_rpe: bool = False,
        use_flash_attn: bool = True,
        upcast_attention: bool = False,
        upcast_softmax: bool = False,
        cpe_indice_key: Optional[str] = None,
        downsample: Optional[nn.Module] = None,
        serialization_orders: Optional[Sequence[SerializationOrder]] = None,
        shuffle_serialization_orders: bool = False,
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
                    use_rpe=use_rpe,
                    use_flash_attn=use_flash_attn,
                    upcast_attention=upcast_attention,
                    upcast_softmax=upcast_softmax,
                )
            )

    @overload
    def forward(
        self,
        x: OptTensor,
        pos: Tensor,
        batch: Tensor,
        serialized_code: Tensor,
        serialized_order: Tensor,
        serialized_inverse: Tensor,
        return_inverse: Literal[True] = True,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]: ...

    @overload
    def forward(
        self,
        x: OptTensor,
        pos: Tensor,
        batch: Tensor,
        serialized_code: Tensor,
        serialized_order: Tensor,
        serialized_inverse: Tensor,
        return_inverse: Literal[False] = False,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]: ...

    def forward(
        self,
        x: OptTensor,
        pos: Tensor,
        batch: Tensor,
        serialized_code: Tensor,
        serialized_order: Tensor,
        serialized_inverse: Tensor,
        return_inverse: bool = False,
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
                x, pos, batch, inverse = self.downsample(x, pos, batch)
                serialized_code, serialized_order, serialized_inverse = serialize(
                    pos,
                    batch,
                    orders=self.serialization_orders,
                    shuffle=self.shuffle_serialization_orders,
                )
            else:
                x, pos, batch, serialized_code, inverse = self.downsample(
                    x,
                    pos,
                    batch,
                    serialized_code,
                    return_inverse=True,
                )
                serialized_order = torch.argsort(serialized_code, dim=1)
                serialized_inverse = torch.argsort(serialized_order, dim=1)

        for i, block in enumerate(self.blocks):
            order_idx = i % num_serializations
            x = block(
                x,
                pos,
                batch,
                serialized_order=serialized_order[order_idx],
                serialized_inverse=serialized_inverse[order_idx],
            )

        if return_inverse:
            return x, pos, batch, serialized_code, serialized_order, serialized_inverse, inverse
        return x, pos, batch, serialized_code, serialized_order, serialized_inverse


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
        norm: Union[str, Callable] = "batch_norm1d",
        act: Union[str, Callable] = "gelu",
        act_kwargs: Optional[Dict[str, Any]] = None,
        norm_kwargs: Optional[Dict[str, Any]] = None,
        use_rpe: bool = False,
        use_flash_attn: bool = True,
        upcast_attention: bool = False,
        upcast_softmax: bool = False,
        cpe_indice_key: Optional[str] = None,
        upsample: Optional[SerializedUpsample] = None,
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
                    use_rpe=use_rpe,
                    use_flash_attn=use_flash_attn,
                    upcast_attention=upcast_attention,
                    upcast_softmax=upcast_softmax,
                )
            )

    def forward(
        self,
        x: Tensor,
        x_skip: Tensor,
        pos_skip: Tensor,
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

        if self.upsample is not None:
            if inverse is None:
                raise ValueError("`inverse` must be provided when `upsample` module is set.")

            x = self.upsample(x, x_skip, inverse)

        for i, block in enumerate(self.blocks):
            order_idx = i % num_serializations
            x = block(
                x,
                pos_skip,
                batch_skip,
                serialized_order=serialized_order_skip[order_idx],
                serialized_inverse=serialized_inverse_skip[order_idx],
            )

        return x, pos_skip, batch_skip


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
        use_rpe: Use relative positional encoding.
        use_flash_attn: Use Flash Attention.
        upcast_attention: Upcast attention to fp32.
        upcast_softmax: Upcast softmax to fp32.
        pooling: Pooling strategy — `"serialized"` (code-space bit-shift) or
            `"grid"` (grid-coordinate clustering).
        embedding: Embedding type — `"sparse_conv"` (SubMConv3d stem) or
            `"linear"` (linear projection).

    Inputs:
        x: Float tensor of shape $(N, \\text{in\\_channels})$.
        pos: Int tensor of shape $(N, 3)$.
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
        norm: Union[str, Callable] = "batch_norm1d",
        act_kwargs: Optional[Dict[str, Any]] = None,
        norm_kwargs: Optional[Dict[str, Any]] = None,
        bias: bool = True,
        mlp_ratio: float = 4,
        qkv_bias: bool = True,
        qk_scale: Optional[float] = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        drop_path: float = 0.3,
        use_rpe: bool = False,
        use_flash_attn: bool = True,
        upcast_attention: bool = False,
        upcast_softmax: bool = False,
        pooling: str = "serialized",
        stem_type: str = "sparse_conv",
    ):
        in_channels = in_channels if in_channels > 0 else 3
        super().__init__()
        self.in_channels = in_channels
        self.serialization_orders = ensure_tuple(serialization_orders)
        self.shuffle_serialization_orders = shuffle_serialization_orders

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
            use_rpe=use_rpe,
            use_flash_attn=use_flash_attn,
            upcast_attention=upcast_attention,
            upcast_softmax=upcast_softmax,
            pooling=pooling,
            serialization_orders=self.serialization_orders,
            shuffle_serialization_orders=shuffle_serialization_orders,
            act_kwargs=act_kwargs,
            norm_kwargs=norm_kwargs,
        )

    @property
    def embedding_dim(self) -> int:
        return self.encoder[-1].blocks[-1].mlp[0].in_features  # type: ignore[index, union-attr]

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
        norm: Union[str, Callable] = "batch_norm1d",
        act: Union[str, Callable] = "gelu",
        qkv_bias: bool = True,
        qk_scale: Optional[float] = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        drop_path: float = 0.0,
        use_rpe: bool = False,
        use_flash_attn: bool = True,
        upcast_attention: bool = False,
        upcast_softmax: bool = False,
        pooling: str = "serialized",
        serialization_orders: Optional[Sequence[SerializationOrder]] = None,
        shuffle_serialization_orders: bool = False,
        act_kwargs: Optional[Dict[str, Any]] = None,
        norm_kwargs: Optional[Dict[str, Any]] = None,
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
                use_rpe=use_rpe,
                use_flash_attn=use_flash_attn,
                upcast_attention=upcast_attention,
                upcast_softmax=upcast_softmax,
                downsample=downsample,
                serialization_orders=serialization_orders if use_grid_pool else None,
                shuffle_serialization_orders=shuffle_serialization_orders if use_grid_pool else False,
            )

            blocks.append(block)
        return blocks

    @overload
    def forward(
        self,
        x: OptTensor,
        pos: Tensor,
        batch: Tensor,
        return_intermediates: Literal[True],
    ) -> Tuple[Tensor, Tensor, Tensor, List[Dict[str, Tensor]]]: ...

    @overload
    def forward(
        self,
        x: OptTensor,
        pos: Tensor,
        batch: Tensor,
        return_intermediates: Literal[False] = False,
    ) -> Tuple[Tensor, Tensor, Tensor]: ...

    def forward(
        self,
        x: OptTensor,
        pos: Tensor,
        batch: Tensor,
        return_intermediates: bool = False,
    ) -> Any:
        x = x if x is not None else pos.float()

        serialized_code, serialized_order, serialized_inverse = serialize(
            pos,
            batch,
            orders=self.serialization_orders,
            shuffle=self.shuffle_serialization_orders,
        )

        x = self.stem(x, pos, batch)

        intermediates = []
        for i, block in enumerate(self.blocks):
            intermediate = {
                "x": x,
                "pos": pos,
                "batch": batch,
                "serialized_code": serialized_code,
                "serialized_order": serialized_order,
                "serialized_inverse": serialized_inverse,
            }

            (
                x,
                pos,
                batch,
                serialized_code,
                serialized_order,
                serialized_inverse,
                inverse,
            ) = block(
                x,
                pos,
                batch,
                serialized_code=serialized_code,
                serialized_order=serialized_order,
                serialized_inverse=serialized_inverse,
                return_inverse=True,
            )

            if i > 0:
                intermediate["inverse"] = inverse
                intermediates.append(intermediate)

        if return_intermediates:
            return x, pos, batch, intermediates
        return x, pos, batch


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
        use_rpe: Use relative positional encoding.
        use_flash_attn: Use Flash Attention.
        upcast_attention: Upcast attention to fp32.
        upcast_softmax: Upcast softmax to fp32.

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
        norm: Union[str, Callable] = "batch_norm1d",
        act: Union[str, Callable] = "gelu",
        mlp_ratio: float = 4,
        qkv_bias: bool = True,
        qk_scale: Optional[float] = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        drop_path: float = 0.3,
        use_rpe: bool = False,
        use_flash_attn: bool = True,
        upcast_attention: bool = False,
        upcast_softmax: bool = False,
        act_kwargs: Optional[Dict[str, Any]] = None,
        norm_kwargs: Optional[Dict[str, Any]] = None,
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
            use_rpe=use_rpe,
            use_flash_attn=use_flash_attn,
            upcast_attention=upcast_attention,
            upcast_softmax=upcast_softmax,
            act_kwargs=act_kwargs,
            norm_kwargs=norm_kwargs,
        )

    @property
    def out_channels(self) -> int:
        return self.stages[-1].blocks[-1].mlp[0].in_features  # type: ignore[index, union-attr]

    def configure_blocks(
        self,
        depths: Sequence[int],
        channels: Sequence[int],
        skip_channels: Sequence[int],
        num_heads: Sequence[int],
        patch_sizes: Sequence[int],
        mlp_ratio: float = 4.0,
        norm: Union[str, Callable] = "batch_norm1d",
        act: Union[str, Callable] = "gelu",
        qkv_bias: bool = True,
        qk_scale: Optional[float] = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        drop_path: float = 0.0,
        use_rpe: bool = False,
        use_flash_attn: bool = True,
        upcast_attention: bool = False,
        upcast_softmax: bool = False,
        act_kwargs: Optional[Dict[str, Any]] = None,
        norm_kwargs: Optional[Dict[str, Any]] = None,
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
                use_rpe=use_rpe,
                use_flash_attn=use_flash_attn,
                upcast_attention=upcast_attention,
                upcast_softmax=upcast_softmax,
                upsample=upsample,
            )

            blocks.append(block)
        return blocks

    def forward(self, x: Tensor, intermediates: List[Dict[str, Tensor]]) -> Tuple[Tensor, Tensor, Tensor]:
        for block, intermediate in zip(self.blocks, reversed(intermediates)):
            intermediate.pop("serialized_code", None)
            intermediate = {f"{k}_skip" if k != "inverse" else k: v for k, v in intermediate.items()}
            x, pos, batch = block(x, **intermediate)
        return x, pos, batch


class PointTransformerV3Classification(ClassificationModel):
    """PyTorch implementation of the Point Transformer V3 model, as described in the paper
    [Point Transformer V3: Simpler, Faster, Stronger](https://arxiv.org/abs/2312.10035)
    by Xiaoyang Wu, Li Jiang, Peng-Shuai Wang, Zhijian Liu, Xihui Liu, Yu Qiao, Wanli Ouyang, Tong He, Hengshuang Zhao.

    This implementation is based on the original implementation from [Pointcept](https://github.com/Pointcept/Pointcept).

    Important:
        This model requires `spconv`, `torch-scatter` to be installed.
        It is also recommended to install `flash-attn` for faster attention.
        In addition, it is recommended to install `ocnn` if you want to use more serialization orders.

    Args:
        in_channels: Number of input channels (corresponding to the number of features).
        num_classes: Number of output classes.
        serialization_orders: Serialization orders to use for the PointTransformerV3 encoder.
        stride: Stride for the downsampling operations.
        enc_depths: Number of encoder blocks for each stage.
        enc_channels: Number of channels for each encoder block.
        enc_num_head: Number of attention heads for each encoder block.
        enc_patch_size: Patch size for each encoder block.
        norm: Normalization layer to use.
        act: Activation function to use.
        mlp_ratio: Ratio of the hidden dimension to the input dimension.
        qkv_bias: Whether to use bias in the QKV projection.
        qk_scale: Scaling factor for the QK matrix.
        attn_drop: Dropout rate for the attention.
        proj_drop: Dropout rate for the projection.
        drop_path: Dropout rate for the drop path.
        shuffle_serialization_orders: Whether to shuffle the serialization orders.
        use_rpe: Whether to use relative positional encoding.
        use_flash_attn: Whether to use flash attention.
        upcast_attention: Whether to upcast the attention.
        upcast_softmax: Whether to upcast the softmax.
        dropout: Dropout rate for the dropout.
        global_pool: Pooling method to aggregate point features ("max" or "mean").

    Inputs:
        x: Float tensor of shape $(N, in_channels)$.
        pos: Int tensor of shape $(N, 3)$.
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
        norm: Union[str, Callable] = "batch_norm1d",
        act: Union[str, Callable] = "gelu",
        mlp_ratio: float = 4,
        qkv_bias: bool = True,
        qk_scale: Optional[float] = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        drop_path: float = 0.3,
        use_rpe: bool = False,
        use_flash_attn: bool = True,
        upcast_attention: bool = False,
        upcast_softmax: bool = False,
        dropout: float = 0.0,
        global_pool: PoolLike = "max",
        pooling: str = "serialized",
        stem_type: str = "sparse_conv",
        act_kwargs: Optional[Dict[str, Any]] = None,
        norm_kwargs: Optional[Dict[str, Any]] = None,
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
            use_rpe=use_rpe,
            use_flash_attn=use_flash_attn,
            upcast_attention=upcast_attention,
            upcast_softmax=upcast_softmax,
            pooling=pooling,
            stem_type=stem_type,
            act_kwargs=act_kwargs,
            norm_kwargs=norm_kwargs,
        )
        self.dropout = dropout
        self.global_pool = create_pool(global_pool)
        self.head = create_cls_head(num_features=self.embedding_dim, num_classes=self.num_classes)

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
        self.head = create_cls_head(num_features=self.embedding_dim, num_classes=self.num_classes, **kwargs)

    @overload
    def forward_features(
        self,
        x: OptTensor,
        pos: Tensor,
        batch: Tensor,
        return_intermediates: Literal[True],
    ) -> Tuple[Tensor, Tensor, Tensor, List[Dict[str, Tensor]]]: ...

    @overload
    def forward_features(
        self,
        x: OptTensor,
        pos: Tensor,
        batch: Tensor,
        return_intermediates: Literal[False] = False,
    ) -> Tuple[Tensor, Tensor, Tensor]: ...

    def forward_features(
        self,
        x: OptTensor,
        pos: Tensor,
        batch: Tensor,
        return_intermediates: bool = False,
    ) -> Any:
        if return_intermediates:
            return self.encoder.forward(x, pos, batch, return_intermediates=True)
        return self.encoder.forward(x, pos, batch, return_intermediates=False)

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

    def forward(self, x: OptTensor, pos: Tensor, batch: Tensor) -> Tensor:
        """Forward pass of the PointNet classification network.

        Args:
            x: Additional point features of shape $(N, C)$.
            pos: Grid coordinates of shape $(N, 3)$.
            batch: Batch indices for each point of shape $(N,)$.

        Returns:
            Classification logits of shape $(B, num\\_classes)$.
        """
        x, pos, batch = self.forward_features(x, pos, batch)
        return self.forward_head(x, batch)


class PointTransformerV3Segmentation(SegmentationModel):
    """PyTorch implementation of the Point Transformer V3 model for segmentation tasks.

    Based on the paper [Point Transformer V3: Simpler, Faster, Stronger](https://arxiv.org/abs/2312.10035)
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
        use_rpe: Whether to use relative positional encoding.
        use_flash_attn: Whether to use flash attention.
        upcast_attention: Whether to upcast the attention.
        upcast_softmax: Whether to upcast the softmax.
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
        norm: Union[str, Callable] = "batch_norm1d",
        act: Union[str, Callable] = "gelu",
        mlp_ratio: float = 4,
        qkv_bias: bool = True,
        qk_scale: Optional[float] = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        drop_path: float = 0.3,
        shuffle_serialization_orders: bool = True,
        use_rpe: bool = False,
        use_flash_attn: bool = True,
        upcast_attention: bool = False,
        upcast_softmax: bool = False,
        dropout: float = 0.0,
        pooling: str = "serialized",
        stem_type: str = "sparse_conv",
        act_kwargs: Optional[Dict[str, Any]] = None,
        norm_kwargs: Optional[Dict[str, Any]] = None,
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
            use_rpe=use_rpe,
            use_flash_attn=use_flash_attn,
            upcast_attention=upcast_attention,
            upcast_softmax=upcast_softmax,
            pooling=pooling,
            stem_type=stem_type,
            act_kwargs=act_kwargs,
            norm_kwargs=norm_kwargs,
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
            use_rpe=use_rpe,
            use_flash_attn=use_flash_attn,
            upcast_attention=upcast_attention,
            upcast_softmax=upcast_softmax,
            act_kwargs=act_kwargs,
            norm_kwargs=norm_kwargs,
        )
        self.dropout = dropout
        self.head = create_cls_head(num_features=self.out_channels, num_classes=self.num_classes)

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
        pos: Tensor,
        batch: Tensor,
        return_intermediates: Literal[True],
    ) -> Tuple[Tensor, Tensor, Tensor, List[Dict[str, Tensor]]]: ...

    @overload
    def forward_features(
        self,
        x: OptTensor,
        pos: Tensor,
        batch: Tensor,
        return_intermediates: Literal[False] = False,
    ) -> Tuple[Tensor, Tensor, Tensor]: ...

    def forward_features(
        self,
        x: OptTensor,
        pos: Tensor,
        batch: Tensor,
        return_intermediates: bool = False,
    ) -> Any:
        if return_intermediates:
            return self.encoder.forward(x, pos, batch, return_intermediates=True)
        return self.encoder.forward(x, pos, batch, return_intermediates=False)

    def forward_decoder(self, x: Tensor, intermediates: List[Dict[str, Tensor]]) -> Tuple[Tensor, Tensor, Tensor]:
        return self.decoder.forward(x, intermediates)

    def forward_head(self, x: Tensor, pre_logits: bool = False) -> Tensor:
        if self.dropout:
            x = F.dropout(x, p=float(self.dropout), training=self.training)
        return x if pre_logits else self.head(x)

    def forward(self, x: Tensor, pos: Tensor, batch: Tensor) -> Tensor:
        x, _, _, intermediates = self.forward_features(x, pos, batch, return_intermediates=True)
        x, _, _ = self.forward_decoder(x, intermediates)
        return self.forward_head(x)


@register_model(
    "sonata",
    task="base",
    weights="hf://torch-pointcloud/sonata/sonata.pth",
    transforms=T.Compose(
        [
            T.CenterShift(keys=DataKeys.POS, apply_z=True),
            T.GridSample(
                pos_key=DataKeys.POS,
                feature_keys=[DataKeys.COLOR, DataKeys.NORMAL],
                grid_size=0.02,
            ),
            T.Divide(keys=DataKeys.COLOR, divisor=255),
            T.Cat(keys=[DataKeys.POS, DataKeys.COLOR, DataKeys.NORMAL], dst_key=DataKeys.X, dim=1),
        ]
    ),
    hparams=dict(
        in_channels=9,
        serialization_orders=("z", "z-trans", "hilbert", "hilbert-trans"),
        shuffle_serialization_orders=True,
        strides=(2, 2, 2, 2),
        encoder_depths=(3, 3, 3, 12, 3),
        encoder_channels=(48, 96, 192, 384, 512),
        encoder_num_head=(3, 6, 12, 24, 32),
        encoder_patch_size=(1024, 1024, 1024, 1024, 1024),
        norm="layer_norm",
        act="gelu",
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.3,
        use_rpe=False,
        use_flash_attn=True,
        upcast_attention=False,
        upcast_softmax=False,
        pooling="grid",
        stem_type="linear",
        norm_kwargs={"mode": "node"},
    ),
)
def sonata(**hparams: Any) -> PointTransformerV3Encoder:
    return PointTransformerV3Encoder(**hparams)
