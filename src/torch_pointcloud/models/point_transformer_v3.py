import math
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional, Sequence, Tuple, overload

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from torch_pointcloud.layers import ActLike, NormLike, PoolLike, create_act, create_cls_head, create_norm, create_pool
from torch_pointcloud.layers.dropouts import DropPath
from torch_pointcloud.transforms.functional import divisible_pad, split_batch
from torch_pointcloud.utils.conversion import batch_to_offset, ensure_tuple, ensure_tuple_size
from torch_pointcloud.utils.imports import optional_import
from torch_pointcloud.utils.serialization import SerializationOrder, serialize_coords
from torch_pointcloud.utils.types import OptTensor, ValueCollection

if TYPE_CHECKING:
    import flash_attn
    import spconv.pytorch as spconv
    import torch_scatter


flash_attn, _FLASH_ATTN_AVAILABLE = optional_import("flash_attn")
torch_scatter, _ = optional_import("torch_scatter")
spconv, _ = optional_import("spconv.pytorch")


class RelativePositionalEncoding(nn.Module):
    def __init__(self, patch_size: int, num_heads: int) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.coords_boundary = int(math.pow(4 * patch_size, 1 / 3) * 2)
        self.rpe_num = 2 * self.coords_boundary + 1
        self.rpe_table = nn.Parameter(torch.zeros(3 * self.rpe_num, num_heads))
        nn.init.trunc_normal_(self.rpe_table, std=0.02)

    def forward(self, relative_coords: Tensor) -> Tensor:
        clamped_coords = relative_coords.clamp(-self.coords_boundary, self.coords_boundary)
        positive_indices = clamped_coords + self.coords_boundary
        dim_strides = torch.arange(3, device=relative_coords.device) * self.rpe_num

        idx = positive_indices + dim_strides

        encodings = self.rpe_table.index_select(0, idx.reshape(-1))
        encodings = encodings.view(idx.shape + (-1,)).sum(3)
        encodings = encodings.permute(0, 3, 1, 2)

        return encodings

    def extra_repr(self) -> str:
        return f"patch_size={self.patch_size}, num_heads={self.num_heads}"


class SerializedAttention(nn.Module):
    def __init__(
        self,
        channels: int,
        num_heads: int,
        patch_size: int,
        qkv_bias: bool = True,
        qk_scale: Optional[float] = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        with_rpe: bool = False,
        with_flash_attn: bool = True,
        upcast_attention: bool = True,
        upcast_softmax: bool = True,
    ):
        super().__init__()
        assert channels % num_heads == 0
        self.channels = channels
        self.num_heads = num_heads
        self.scale = qk_scale or (channels // num_heads) ** -0.5
        self.upcast_attention = upcast_attention
        self.upcast_softmax = upcast_softmax
        self.with_rpe = with_rpe
        self.with_flash_attn = with_flash_attn
        self.patch_size = patch_size
        self.attn_drop = attn_drop
        self.proj_drop = proj_drop

        if with_flash_attn:
            if not _FLASH_ATTN_AVAILABLE:
                raise ImportError(flash_attn)
            elif with_rpe:
                raise ValueError("Relative positional encoding is not supported with Flash Attention.")
            elif upcast_attention:
                raise ValueError("Upcasting attention is not supported with Flash Attention.")
            elif upcast_softmax:
                raise ValueError("Upcasting softmax is not supported with Flash Attention.")

        self.qkv = torch.nn.Linear(channels, channels * 3, bias=qkv_bias)
        self.proj = torch.nn.Linear(channels, channels)
        self.softmax = torch.nn.Softmax(dim=-1)
        self.rpe = RelativePositionalEncoding(patch_size, num_heads) if self.with_rpe else None

    def _forward_default_attn(self, qkv: Tensor, grid_coords: OptTensor, patch_size: int) -> Tensor:
        K, H, C = patch_size, self.num_heads, self.channels

        # Encode and reshape qkv: (N', K, 3, H, C') -> (3, N', H, K, C')
        q, k, v = qkv.reshape(-1, K, 3, H, C // H).permute(2, 0, 3, 1, 4).unbind(dim=0)

        if self.upcast_attention:
            q = q.float()
            k = k.float()

        attn = (q * self.scale) @ k.transpose(-2, -1)  # (N', H, K, K)

        if self.with_rpe:
            if self.rpe is None:
                raise RuntimeError(
                    "`rpe` must be provided when `with_rpe` is True. "
                    "Please check the model configuration or reinitialize the model."
                )

            if grid_coords is None:
                raise ValueError("`grid_coords` must be provided when `with_rpe` is True")

            grid_coords = grid_coords.reshape(-1, K, 3)
            relative_coords = grid_coords.unsqueeze(2) - grid_coords.unsqueeze(1)
            attn = attn + self.rpe(relative_coords)

        if self.upcast_softmax:
            attn = attn.float()

        attn = self.softmax(attn)
        attn = F.dropout(attn, p=self.attn_drop, training=self.training).to(qkv.dtype)

        feat = (attn @ v).transpose(1, 2).reshape(-1, C)
        return feat

    def _forward_flash_attn(self, qkv: Tensor, batch: Tensor) -> Tensor:
        H, C = self.num_heads, self.channels

        patch_idxs = split_batch(batch, self.patch_size)
        offset = batch_to_offset(patch_idxs)
        # NOTE: The first element of `cu_seqlens` is always 0, and should be int32 to work with `flash-attn`
        cu_seqlens = torch.cat([torch.tensor([0], device=batch.device, dtype=torch.int), offset.int()])

        feat = flash_attn.flash_attn_varlen_qkvpacked_func(
            qkv.half().reshape(-1, 3, H, C // H),
            cu_seqlens,
            max_seqlen=self.patch_size,
            dropout_p=self.attn_drop if self.training else 0,
            softmax_scale=self.scale,
        )

        return feat.reshape(-1, C).to(qkv.dtype)

    def forward(
        self,
        features: Tensor,
        grid_coords: OptTensor,
        batch: Tensor,
        serialized_order: OptTensor = None,
        serialized_inverse: OptTensor = None,
    ) -> Any:
        # NOTE: For default attention (i.e. without Flash Attention), we use the patch size
        # as the minimum between the batch sizes and the specified patch size
        patch_size: int = (
            self.patch_size  # type: ignore[assignment]
            if self.with_flash_attn
            else min(torch.bincount(batch).min().item(), self.patch_size)  # fmt: skip
        )

        # Only pad batches larger than the patch size
        padded_indices, unpadded_indices, padded_batch = divisible_pad(
            batch,
            patch_size,
            return_inverse=True,
            mode="above",
        )

        order = serialized_order[padded_indices] if serialized_order is not None else padded_indices
        inverse = unpadded_indices[serialized_inverse] if serialized_inverse is not None else unpadded_indices

        # Apply attention
        qkv = self.qkv(features)[order]
        if self.with_flash_attn:
            features = self._forward_flash_attn(qkv, padded_batch)
        else:
            if grid_coords is not None:
                grid_coords = grid_coords[order]
            features = self._forward_default_attn(qkv, grid_coords, patch_size)

        features = features[inverse]

        # Head projection
        features = self.proj(features)
        features = F.dropout(features, p=self.proj_drop, training=self.training)
        return features


def to_sparse_conv_tensor(
    features: Tensor,
    grid_coords: Tensor,
    batch: Tensor,
    spatial_shape: Optional[Sequence[int]] = None,
    padding: int = 96,
) -> "spconv.SparseConvTensor":
    if spatial_shape is None:
        spatial_shape = torch.add(torch.max(grid_coords, dim=0).values, padding).tolist()

    return spconv.SparseConvTensor(
        features=features,
        indices=torch.cat([batch.unsqueeze(-1).int(), grid_coords.int()], dim=1).contiguous(),
        spatial_shape=spatial_shape,
        batch_size=batch[-1].item() + 1,
    )


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
        norm: NormLike = "layer_norm",
        act: ActLike = "gelu",
        cpe_indice_key: Optional[str] = None,
        with_rpe: bool = False,
        with_flash_attn: bool = True,
        upcast_attention: bool = True,
        upcast_softmax: bool = True,
    ):
        super().__init__()
        self.cpe_conv = spconv.SubMConv3d(
            channels,
            channels,
            kernel_size=3,
            bias=True,
            indice_key=cpe_indice_key,
        )
        self.cpe_proj = nn.Linear(channels, channels)
        self.cpe_norm = create_norm(norm, channels)

        self.norm1 = create_norm(norm, channels)
        self.attn = SerializedAttention(
            channels=channels,
            patch_size=patch_size,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            with_rpe=with_rpe,
            with_flash_attn=with_flash_attn,
            upcast_attention=upcast_attention,
            upcast_softmax=upcast_softmax,
        )

        self.norm2 = create_norm(norm, channels)
        self.mlp = nn.Sequential(
            nn.Linear(channels, int(channels * mlp_ratio)),
            create_act(act),
            nn.Dropout(proj_drop),
            nn.Linear(int(channels * mlp_ratio), channels),
            nn.Dropout(proj_drop),
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(
        self,
        features: Tensor,
        grid_coords: Tensor,
        batch: Tensor,
        serialized_order: OptTensor = None,
        serialized_inverse: OptTensor = None,
    ) -> Any:
        # Conv + Skip connection
        shortcut = features
        sparse_features = to_sparse_conv_tensor(features, grid_coords, batch)
        sparse_features = self.cpe_conv(sparse_features)
        features = sparse_features.features
        features = self.cpe_proj(features)
        features = self.cpe_norm(features)

        # NOTE: Skip connection and save the new shortcut
        features = shortcut = shortcut + features

        # Attention branch
        features = self.norm1(features)
        features = self.attn(
            features,
            grid_coords,
            batch,
            serialized_order=serialized_order,
            serialized_inverse=serialized_inverse,
        )
        features = self.drop_path(features)
        features = shortcut + features

        # MLP branch
        shortcut = features
        features = self.norm2(features)
        features = self.mlp(features)
        features = self.drop_path(features)
        features = shortcut + features

        return features


class SerializedPooling(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 2,
        norm: Optional[NormLike] = None,
        act: Optional[ActLike] = None,
        reduce: str = "max",
    ):
        super().__init__()
        if reduce not in ["sum", "mean", "min", "max"]:
            raise ValueError(f"Invalid reduce operation: {reduce}. Must be one of: 'sum', 'mean', 'min', 'max'.")
        if stride != 2 ** (math.ceil(stride) - 1).bit_length():
            raise ValueError(f"Invalid stride: {stride}. Must be a power of 2.")

        self.stride = stride
        self.reduce = reduce
        self.proj = nn.Linear(in_channels, out_channels)
        self.norm = create_norm(norm, out_channels) if norm is not None else None
        self.act = create_act(act) if act is not None else None

    @overload
    def forward(
        self,
        features: Tensor,
        grid_coords: Tensor,
        batch: Tensor,
        serialized_code: Tensor,
        return_pooling_inverse: Literal[True] = True,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]: ...

    @overload
    def forward(
        self,
        features: Tensor,
        grid_coords: Tensor,
        batch: Tensor,
        serialized_code: Tensor,
        return_pooling_inverse: Literal[False] = False,
    ) -> Tuple[Tensor, Tensor, Tensor]: ...

    def forward(
        self,
        features: Tensor,
        grid_coords: Tensor,
        batch: Tensor,
        serialized_code: Tensor,
        return_pooling_inverse: bool = False,
    ) -> Tuple[Tensor, ...]:
        pooling_depth = (math.ceil(self.stride) - 1).bit_length()
        pooled_code = serialized_code >> (pooling_depth * 3)
        _, cluster, counts = torch.unique(pooled_code[0], sorted=True, return_inverse=True, return_counts=True)

        # Sort by cluster for segment_csr
        _, indices = torch.sort(cluster)
        idx_ptr = torch.cat([counts.new_zeros(1), torch.cumsum(counts, dim=0)])
        head_indices = indices[idx_ptr[:-1]]

        # Pool features, positions and batch indices
        features = torch_scatter.segment_csr(self.proj(features)[indices], idx_ptr, reduce="max")
        grid_coords = grid_coords[head_indices] >> pooling_depth
        batch = batch[head_indices]
        pooled_code = pooled_code[:, head_indices]

        if self.norm:
            features = self.norm(features)
        if self.act:
            features = self.act(features)

        if return_pooling_inverse:
            return features, grid_coords, batch, pooled_code, cluster
        return features, grid_coords, batch, pooled_code


class SerializedUnpooling(nn.Module):
    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        norm: Optional[NormLike] = None,
        act: Optional[ActLike] = None,
    ):
        super().__init__()
        self.proj = nn.Linear(in_channels, out_channels)
        self.proj_skip = nn.Linear(skip_channels, out_channels)

        self.norm = create_norm(norm, out_channels) if norm is not None else None
        self.norm_skip = create_norm(norm, out_channels) if norm is not None else None

        self.act = create_act(act) if act is not None else None
        self.act_skip = create_act(act) if act is not None else None

    def forward(self, features: Tensor, skip_features: Tensor, pooling_inverse: Tensor) -> Tensor:
        features = self.proj(features)
        if self.norm is not None:
            features = self.norm(features)
        if self.act is not None:
            features = self.act(features)

        skip_features = self.proj_skip(skip_features)
        if self.norm_skip is not None:
            skip_features = self.norm_skip(skip_features)
        if self.act_skip is not None:
            skip_features = self.act_skip(skip_features)

        return skip_features + features[pooling_inverse]


class Embedding(nn.Module):
    def __init__(
        self,
        in_channels: int,
        embedding_dim: int,
        norm: Optional[NormLike] = None,
        act: Optional[ActLike] = None,
        stem_indice_key: Optional[str] = None,
    ):
        super().__init__()

        self.stem = spconv.SubMConv3d(
            in_channels,
            embedding_dim,
            kernel_size=5,
            padding=1,
            bias=False,
            indice_key=stem_indice_key,
        )
        self.norm = create_norm(norm, embedding_dim) if norm is not None else None
        self.act = create_act(act) if act is not None else None

    def forward(
        self,
        features: Tensor,
        grid_coords: Tensor,
        batch: Tensor,
    ) -> Tensor:
        sparse_features = to_sparse_conv_tensor(features, grid_coords, batch)
        sparse_features = self.stem(sparse_features)

        features = sparse_features.features
        if self.norm is not None:
            features = self.norm(features)
        if self.act is not None:
            features = self.act(features)

        return features


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
        norm: NormLike = "layer_norm",
        act: ActLike = "gelu",
        with_rpe: bool = False,
        with_flash_attn: bool = True,
        upcast_attention: bool = False,
        upcast_softmax: bool = False,
        cpe_indice_key: Optional[str] = None,
        downsample: Optional[SerializedPooling] = None,
    ):
        super().__init__()
        self.downsample = downsample
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
                    cpe_indice_key=cpe_indice_key,
                    with_rpe=with_rpe,
                    with_flash_attn=with_flash_attn,
                    upcast_attention=upcast_attention,
                    upcast_softmax=upcast_softmax,
                )
            )

    @overload
    def forward(
        self,
        features: OptTensor,
        grid_coords: Tensor,
        batch: Tensor,
        serialized_code: Tensor,
        serialized_order: Tensor,
        serialized_inverse: Tensor,
        return_pooling_inverse: Literal[True] = True,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]: ...

    @overload
    def forward(
        self,
        features: OptTensor,
        grid_coords: Tensor,
        batch: Tensor,
        serialized_code: Tensor,
        serialized_order: Tensor,
        serialized_inverse: Tensor,
        return_pooling_inverse: Literal[False] = False,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]: ...

    def forward(
        self,
        features: OptTensor,
        grid_coords: Tensor,
        batch: Tensor,
        serialized_code: Tensor,
        serialized_order: Tensor,
        serialized_inverse: Tensor,
        return_pooling_inverse: bool = False,
    ) -> Any:
        if not serialized_code.shape == serialized_order.shape == serialized_inverse.shape:
            raise ValueError(
                "`serialized_code`, `serialized_order` and `serialized_inverse` "
                f"must have the same shape. Got {serialized_code.shape}, "
                f"{serialized_order.shape} and {serialized_inverse.shape} respectively."
            )

        num_serializations = len(serialized_code)
        pooling_inverse: OptTensor = None

        if self.downsample is not None:
            features, grid_coords, batch, serialized_code, pooling_inverse = self.downsample(
                features,
                grid_coords,
                batch,
                serialized_code,
                return_pooling_inverse=True,
            )

            serialized_order = torch.argsort(serialized_code, dim=1)
            serialized_inverse = torch.argsort(serialized_order, dim=1)

        for i, block in enumerate(self.blocks):
            order_idx = i % num_serializations
            features = block(
                features,
                grid_coords,
                batch,
                serialized_order=serialized_order[order_idx],
                serialized_inverse=serialized_inverse[order_idx],
            )

        if return_pooling_inverse:
            return features, grid_coords, batch, serialized_code, serialized_order, serialized_inverse, pooling_inverse
        return features, grid_coords, batch, serialized_code, serialized_order, serialized_inverse


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
        norm: NormLike = "batch_norm1d",
        act: ActLike = "gelu",
        with_rpe: bool = False,
        with_flash_attn: bool = True,
        upcast_attention: bool = False,
        upcast_softmax: bool = False,
        cpe_indice_key: Optional[str] = None,
        upsample: Optional[SerializedUnpooling] = None,
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
                    cpe_indice_key=cpe_indice_key,
                    with_rpe=with_rpe,
                    with_flash_attn=with_flash_attn,
                    upcast_attention=upcast_attention,
                    upcast_softmax=upcast_softmax,
                )
            )

    def forward(
        self,
        features: Tensor,
        skip_features: Tensor,
        skip_grid_coords: Tensor,
        skip_batch: Tensor,
        skip_serialized_order: Tensor,
        skip_serialized_inverse: Tensor,
        pooling_inverse: OptTensor = None,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        if not skip_serialized_order.shape == skip_serialized_inverse.shape:
            raise ValueError(
                "`skip_serialized_order` and `skip_serialized_inverse` "
                f"must have the same shape. Got {skip_serialized_order.shape} "
                f"and {skip_serialized_inverse.shape} respectively."
            )

        num_serializations = len(skip_serialized_order)

        if self.upsample is not None:
            if pooling_inverse is None:
                raise ValueError("`pooling_inverse` must be provided when `upsample` module is set.")

            features = self.upsample(features, skip_features, pooling_inverse)

        for i, block in enumerate(self.blocks):
            order_idx = i % num_serializations
            features = block(
                features,
                skip_grid_coords,
                skip_batch,
                serialized_order=skip_serialized_order[order_idx],
                serialized_inverse=skip_serialized_inverse[order_idx],
            )

        return features, skip_grid_coords, skip_batch


def serialize(
    grid_coords: Tensor,
    batch: Tensor,
    orders: Sequence[SerializationOrder],
    shuffle: bool = False,
) -> Tuple[Tensor, Tensor, Tensor]:
    if shuffle:
        perm = torch.randperm(len(orders))
        orders = [orders[i] for i in perm]

    depth = int(grid_coords.max()).bit_length()
    serialized_code = torch.stack([serialize_coords(grid_coords, batch, depth=depth, order=order) for order in orders])
    serialized_order = torch.argsort(serialized_code, dim=1)
    serialized_inverse = torch.argsort(serialized_order, dim=1)
    return serialized_code, serialized_order, serialized_inverse


def create_encoder_blocks(
    depths: Sequence[int],
    channels: Sequence[int],
    num_heads: Sequence[int],
    patch_sizes: Sequence[int],
    strides: Sequence[int],
    mlp_ratio: float = 4.0,
    norm: NormLike = "batch_norm1d",
    act: ActLike = "gelu",
    qkv_bias: bool = True,
    qk_scale: Optional[float] = None,
    attn_drop: float = 0.0,
    proj_drop: float = 0.0,
    drop_path: float = 0.0,
    with_rpe: bool = False,
    with_flash_attn: bool = True,
    upcast_attention: bool = False,
    upcast_softmax: bool = False,
) -> nn.ModuleList:
    depths = ensure_tuple(depths)
    n = len(depths)
    channels = ensure_tuple_size(channels, size=n, extra_msg="Encoder length `channels` != `depths`.")
    num_heads = ensure_tuple_size(num_heads, size=n, extra_msg="Encoder length `num_heads` != `depths`.")
    patch_sizes = ensure_tuple_size(patch_sizes, size=n, extra_msg="Encoder length `patch_sizes` != `depths`.")
    strides = ensure_tuple_size(strides, size=n - 1, extra_msg="Encoder length `strides` != `depths` - 1.")

    # Pre-compute the drop paths for each encoder block.
    # For example, if the drop path is 0.3, and the depths are (2, 3, 4),
    # then the drop paths for each block, at each stage, are:
    # - block 0: [0.0000, 0.0375]
    # - block 1: [0.0750, 0.1125, 0.1500]
    # - block 2: [0.1875, 0.2250, 0.2625, 0.3000]
    drop_paths = torch.split(torch.linspace(0, drop_path, sum(depths)), list(depths))

    blocks = nn.ModuleList()
    for i in range(n):
        downsample: Optional[SerializedPooling] = None
        if i > 0:
            downsample = SerializedPooling(
                in_channels=channels[i - 1],
                out_channels=channels[i],
                stride=strides[i - 1],
                norm=norm,
                act=act,
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
            norm="layer_norm",
            act=act,
            cpe_indice_key=f"stage{i}",
            with_rpe=with_rpe,
            with_flash_attn=with_flash_attn,
            upcast_attention=upcast_attention,
            upcast_softmax=upcast_softmax,
            downsample=downsample,
        )
        blocks.append(block)
    return blocks


def create_decoder_blocks(
    depths: Sequence[int],
    channels: Sequence[int],
    skip_channels: Sequence[int],
    num_heads: Sequence[int],
    patch_sizes: Sequence[int],
    mlp_ratio: float = 4.0,
    norm: NormLike = "batch_norm1d",
    act: ActLike = "gelu",
    qkv_bias: bool = True,
    qk_scale: Optional[float] = None,
    attn_drop: float = 0.0,
    proj_drop: float = 0.0,
    drop_path: float = 0.0,
    with_rpe: bool = False,
    with_flash_attn: bool = True,
    upcast_attention: bool = False,
    upcast_softmax: bool = False,
) -> nn.ModuleList:
    depths = ensure_tuple(depths)
    n = len(depths)
    channels = ensure_tuple_size(channels, size=n + 1, extra_msg="Decoder length `channels` != `depths` + 1.")
    skip_channels = ensure_tuple_size(skip_channels, size=n, extra_msg="Decoder length `skip_channels` != `depths`.")
    num_heads = ensure_tuple_size(num_heads, size=n, extra_msg="Decoder length `num_heads` != `depths`.")
    patch_sizes = ensure_tuple_size(patch_sizes, size=n, extra_msg="Decoder length `patch_sizes` != `depths`.")

    # Pre-compute the drop paths for each (decoder) block.
    # The drop path is the same as the encoder block, but in reverse order.
    # For example, if the drop path is 0.3, and the depths are (4, 3, 2),
    # then the drop paths for each block at each stage are:
    # - block 0: [0.3000, 0.2625, 0.2250, 0.1875]
    # - block 1: [0.1500, 0.1125, 0.0750]
    # - block 2: [0.0375, 0.0000]
    drop_paths = torch.split(torch.linspace(0, drop_path, sum(depths)), list(depths))[::-1]

    blocks = nn.ModuleList()
    for i in range(n):
        upsample = SerializedUnpooling(
            in_channels=channels[i],
            skip_channels=skip_channels[i],
            out_channels=channels[i + 1],
            norm=norm,
            act=act,
        )

        # NOTE: For decoder blocks, the drop paths should be in reverse order (i.e. higher -> lower within each block)
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
            norm="layer_norm",
            act=act,
            cpe_indice_key=f"stage{i}",
            with_rpe=with_rpe,
            with_flash_attn=with_flash_attn,
            upcast_attention=upcast_attention,
            upcast_softmax=upcast_softmax,
            upsample=upsample,
        )
        blocks.append(block)
    return blocks


class PointTransformerV3Classification(nn.Module):
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
        with_rpe: Whether to use relative positional encoding.
        with_flash_attn: Whether to use flash attention.
        upcast_attention: Whether to upcast the attention.
        upcast_softmax: Whether to upcast the softmax.
        dropout: Dropout rate for the dropout.
        global_pool: Pooling method to aggregate point features ("max" or "mean").

    Inputs:
        features: Float tensor of shape $(N, in_channels)$.
        grid_coords: Int tensor of shape $(N, 3)$.
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
        stride: Sequence[int] = (2, 2, 2, 2),
        encoder_depths: Sequence[int] = (2, 2, 2, 6, 2),
        encoder_channels: Sequence[int] = (32, 64, 128, 256, 512),
        encoder_num_head: Sequence[int] = (2, 4, 8, 16, 32),
        encoder_patch_size: Sequence[int] = (48, 48, 48, 48, 48),
        norm: NormLike = "batch_norm1d",
        act: ActLike = "gelu",
        mlp_ratio: float = 4,
        qkv_bias: bool = True,
        qk_scale: Optional[float] = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        drop_path: float = 0.3,
        with_rpe: bool = False,
        with_flash_attn: bool = True,
        upcast_attention: bool = False,
        upcast_softmax: bool = False,
        dropout: float = 0.0,
        global_pool: PoolLike = "max",
    ):
        super().__init__()
        self.in_channels = in_channels if in_channels > 0 else 3
        self.num_classes = num_classes
        self.serialization_orders = ensure_tuple(serialization_orders)
        self.shuffle_serialization_orders = shuffle_serialization_orders

        self.embedding = Embedding(in_channels=self.in_channels, embedding_dim=encoder_channels[0], norm=norm, act=act)
        self.encoder = self.configure_encoder_blocks(
            depths=encoder_depths,
            channels=encoder_channels,
            num_heads=encoder_num_head,
            patch_sizes=encoder_patch_size,
            strides=stride,
            mlp_ratio=mlp_ratio,
            norm=norm,
            act=act,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            drop_path=drop_path,
            with_rpe=with_rpe,
            with_flash_attn=with_flash_attn,
            upcast_attention=upcast_attention,
            upcast_softmax=upcast_softmax,
        )

        self.dropout = dropout
        self.global_pool = create_pool(global_pool)
        self.head = create_cls_head(num_features=self.embedding_dim, num_classes=self.num_classes)

    @property
    def embedding_dim(self) -> int:
        return self.encoder[-1].blocks[-1].mlp[0].in_features  # type: ignore[index, union-attr]

    def configure_encoder_blocks(self, *args: Any, **kwargs: Any) -> nn.ModuleList:
        return create_encoder_blocks(*args, **kwargs)

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
        features: OptTensor,
        grid_coords: Tensor,
        batch: Tensor,
        return_intermediates: Literal[True],
    ) -> Tuple[Tensor, Tensor, Tensor, List[Dict[str, Tensor]]]: ...

    @overload
    def forward_features(
        self,
        features: OptTensor,
        grid_coords: Tensor,
        batch: Tensor,
        return_intermediates: Literal[False] = False,
    ) -> Tuple[Tensor, Tensor, Tensor]: ...

    def forward_features(
        self,
        features: OptTensor,
        grid_coords: Tensor,
        batch: Tensor,
        return_intermediates: bool = False,
    ) -> Any:
        """Forward pass of the PointTransformerV3 encoder, returning pre-pooling features.

        Args:
            features: Additional point features of shape $(N, features_dim)$.
            grid_coords: Grid coordinates of shape $(N, 3)$.
            batch: Batch indices for each point of shape $(N,)$.

        Returns:
            Pre-pooling features of shape $(N, mlp2_dims[-1])$ where $N$ is the batch size.
        """
        features = features if features is not None else grid_coords.float()

        # Serialize the grid coordinates for each serialization order (e.g. "z", "z-trans", etc.)
        # NOTE: For faster processing, we pre-compute the serialized code, order and inverse.
        # These variables will be reused in each blocks and updated after each pooling operation.
        serialized_code, serialized_order, serialized_inverse = serialize(
            grid_coords,
            batch,
            orders=self.serialization_orders,
            shuffle=self.shuffle_serialization_orders,
        )

        features = self.embedding(features, grid_coords, batch)

        intermediates = []
        for i, block in enumerate(self.encoder):
            intermediate = {
                "features": features,
                "grid_coords": grid_coords,
                "batch": batch,
                "serialized_code": serialized_code,
                "serialized_order": serialized_order,
                "serialized_inverse": serialized_inverse,
            }

            (
                features,
                grid_coords,
                batch,
                serialized_code,
                serialized_order,
                serialized_inverse,
                pooling_inverse,
            ) = block(
                features,
                grid_coords,
                batch,
                serialized_code=serialized_code,
                serialized_order=serialized_order,
                serialized_inverse=serialized_inverse,
                return_pooling_inverse=True,
            )

            if i > 0:
                intermediate["pooling_inverse"] = pooling_inverse
                intermediates.append(intermediate)

        if return_intermediates:
            return features, grid_coords, batch, intermediates
        return features, grid_coords, batch

    def forward_head(self, x: Tensor, batch: Tensor, pre_logits: bool = False) -> Tensor:
        """Forward pass of the classification head from pre-pooling features.

        Args:
            x: Pre-pooling features of shape $(N, embedding_dim)$.
            batch: Batch indices for each point of shape $(N,)$.
            pre_logits: Whether to return pre-logits. Defaults to False.

        Returns:
            Classification logits of shape $(B, num_classes)$.
        """
        x = self.global_pool(x, batch)
        if self.dropout:
            x = F.dropout(x, p=float(self.dropout), training=self.training)
        return x if pre_logits else self.head(x)

    def forward(self, features: OptTensor, grid_coords: Tensor, batch: Tensor) -> Tensor:
        """Forward pass of the PointNet classification network.

        Args:
            features: Additional point features of shape $(N, features_dim)$.
            grid_coords: Grid coordinates of shape $(N, 3)$.
            batch: Batch indices for each point of shape $(N,)$.

        Returns:
            Classification logits of shape $(B, num_classes)$.
        """
        features, grid_coords, batch = self.forward_features(features, grid_coords, batch)
        return self.forward_head(features, batch)


class PointTransformerV3Segmentation(nn.Module):
    """PyTorch implementation of the Point Transformer V3 model for segmentation tasks.

    Based on the paper [Point Transformer V3: Simpler, Faster, Stronger](https://arxiv.org/abs/2312.10035)
    by Xiaoyang Wu, Li Jiang, Peng-Shuai Wang, Zhijian Liu, Xihui Liu, Yu Qiao, Wanli Ouyang, Tong He, Hengshuang Zhao.

    This segmentation variant uses an encoder-decoder architecture with skip connections.

    Args:
        in_channels: Number of input channels.
        num_classes: Number of output classes for segmentation.
        serialization_orders: Serialization orders to use for the encoder.
        stride: Stride for the downsampling operations.
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
        with_rpe: Whether to use relative positional encoding.
        with_flash_attn: Whether to use flash attention.
        upcast_attention: Whether to upcast the attention.
        upcast_softmax: Whether to upcast the softmax.
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        serialization_orders: Sequence[str] = ("hilbert", "hilbert-trans"),
        strides: Sequence[int] = (2, 2, 2, 2),
        encoder_depths: Sequence[int] = (2, 2, 2, 6, 2),
        encoder_channels: Sequence[int] = (32, 64, 128, 256, 512),
        encoder_num_head: Sequence[int] = (2, 4, 8, 16, 32),
        encoder_patch_size: Sequence[int] = (48, 48, 48, 48, 48),
        decoder_depths: Sequence[int] = (2, 2, 2, 2),
        decoder_channels: Sequence[int] = (256, 128, 64, 64),
        decoder_num_head: Sequence[int] = (16, 8, 4, 4),
        decoder_patch_size: Sequence[int] = (48, 48, 48, 48),
        norm: NormLike = "batch_norm1d",
        act: ActLike = "gelu",
        mlp_ratio: float = 4,
        qkv_bias: bool = True,
        qk_scale: Optional[float] = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        drop_path: float = 0.3,
        shuffle_serialization_orders: bool = True,
        with_rpe: bool = False,
        with_flash_attn: bool = True,
        upcast_attention: bool = False,
        upcast_softmax: bool = False,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.in_channels = in_channels if in_channels > 0 else 1
        self.num_classes = num_classes
        self.serialization_orders = ensure_tuple(serialization_orders)
        self.shuffle_serialization_orders = shuffle_serialization_orders

        self.embedding = Embedding(in_channels=self.in_channels, embedding_dim=encoder_channels[0], norm=norm, act=act)
        self.encoder = self.configure_encoder_blocks(
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
            with_rpe=with_rpe,
            with_flash_attn=with_flash_attn,
            upcast_attention=upcast_attention,
            upcast_softmax=upcast_softmax,
        )
        self.decoder = self.configure_decoder_blocks(
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
            with_rpe=with_rpe,
            with_flash_attn=with_flash_attn,
            upcast_attention=upcast_attention,
            upcast_softmax=upcast_softmax,
        )

        self.dropout = dropout
        self.head = create_cls_head(num_features=self.upsampling_dim, num_classes=self.num_classes)

    @property
    def embedding_dim(self) -> int:
        return self.encoder[-1].blocks[-1].mlp[0].in_features  # type: ignore[index, union-attr]

    @property
    def upsampling_dim(self) -> int:
        return self.decoder[-1].blocks[-1].mlp[0].in_features  # type: ignore[index, union-attr]

    def configure_encoder_blocks(self, *args: Any, **kwargs: Any) -> nn.ModuleList:
        return create_encoder_blocks(*args, **kwargs)

    def configure_decoder_blocks(self, *args: Any, **kwargs: Any) -> nn.ModuleList:
        return create_decoder_blocks(*args, **kwargs)

    @overload
    def forward_features(
        self,
        features: OptTensor,
        grid_coords: Tensor,
        batch: Tensor,
        return_intermediates: Literal[True],
    ) -> Tuple[Tensor, Tensor, Tensor, List[Dict[str, Tensor]]]: ...

    @overload
    def forward_features(
        self,
        features: OptTensor,
        grid_coords: Tensor,
        batch: Tensor,
        return_intermediates: Literal[False] = False,
    ) -> Tuple[Tensor, Tensor, Tensor]: ...

    def forward_features(
        self,
        features: OptTensor,
        grid_coords: Tensor,
        batch: Tensor,
        return_intermediates: bool = False,
    ) -> Any:
        """Forward pass of the PointTransformerV3 encoder, returning pre-pooling features.

        Args:
            features: Additional point features of shape $(N, features_dim)$.
            grid_coords: Grid coordinates of shape $(N, 3)$.
            batch: Batch indices for each point of shape $(N,)$.

        Returns:
            Pre-pooling features of shape $(N, mlp2_dims[-1])$ where $N$ is the batch size.
        """
        # features = features if features is not None else grid_coords.float()
        features = features if features is not None else torch.ones(grid_coords.shape[0], 1).to(grid_coords.device)

        # Serialize the grid coordinates for each serialization order (e.g. "z", "z-trans", etc.)
        # NOTE: For faster processing, we pre-compute the serialized code, order and inverse.
        # These variables will be reused in each blocks and updated after each pooling operation.
        serialized_code, serialized_order, serialized_inverse = serialize(
            grid_coords,
            batch,
            orders=self.serialization_orders,
            shuffle=self.shuffle_serialization_orders,
        )

        features = self.embedding(features, grid_coords, batch)

        intermediates = []
        for i, block in enumerate(self.encoder):
            intermediate = {
                "features": features,
                "grid_coords": grid_coords,
                "batch": batch,
                "serialized_code": serialized_code,
                "serialized_order": serialized_order,
                "serialized_inverse": serialized_inverse,
            }

            (
                features,
                grid_coords,
                batch,
                serialized_code,
                serialized_order,
                serialized_inverse,
                pooling_inverse,
            ) = block(
                features,
                grid_coords,
                batch,
                serialized_code=serialized_code,
                serialized_order=serialized_order,
                serialized_inverse=serialized_inverse,
                return_pooling_inverse=True,
            )

            if i > 0:
                intermediate["pooling_inverse"] = pooling_inverse
                intermediates.append(intermediate)

        if return_intermediates:
            return features, grid_coords, batch, intermediates
        return features, grid_coords, batch

    def forward_decoder(
        self,
        features: Tensor,
        intermediates: List[Dict[str, Tensor]],
    ) -> Tuple[Tensor, Tensor, Tensor]:
        for block, intermediate in zip(self.decoder, reversed(intermediates)):
            intermediate.pop("serialized_code", None)
            intermediate = {f"skip_{k}" if k != "pooling_inverse" else k: v for k, v in intermediate.items()}
            features, grid_coords, batch = block(features, **intermediate)
        return features, grid_coords, batch

    def forward_head(self, features: Tensor, pre_logits: bool = False) -> Tensor:
        if self.dropout:
            features = F.dropout(features, p=float(self.dropout), training=self.training)
        return features if pre_logits else self.head(features)

    def forward(self, features: Tensor, grid_coords: Tensor, batch: Tensor) -> Tensor:
        features, _, _, intermediates = self.forward_features(features, grid_coords, batch, return_intermediates=True)
        features, _, _ = self.forward_decoder(features, intermediates)
        return self.forward_head(features)
