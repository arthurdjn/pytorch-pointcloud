"""
PyTorch implementation of the Point Transformer V3 model, as described in the paper
[Point Transformer V3: Simpler, Faster, Stronger](https://arxiv.org/abs/2312.10035)
by Xiaoyang Wu, Li Jiang, Peng-Shuai Wang, Zhijian Liu, Xihui Liu, Yu Qiao, Wanli Ouyang, Tong He, Hengshuang Zhao.

This implementation is based on the original implementation from [Pointcept](https://github.com/Pointcept/Pointcept),
with model variants and pretrained weights available from Hugging Face.

```python
from torch_pointcloud import create_segmentation_model

# Original model trained by the authors of Pointcept
model = create_segmentation_model("hf:pointcept/point-transformer-v3m1-base")
```
"""

import math
import warnings
from functools import partial
from typing import TYPE_CHECKING, Any, NamedTuple, Optional, Sequence, Tuple, TypedDict, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing_extensions import Unpack

from torch_pointcloud.layers import ActLike, NormLike, create_act, create_norm, linear_block
from torch_pointcloud.layers.drop import DropPath
from torch_pointcloud.transforms.functional import divisible_pad, split_batch
from torch_pointcloud.utils.conversion import batch_to_offset, ensure_tuple
from torch_pointcloud.utils.imports import OptionalImportError, optional_import
from torch_pointcloud.utils.serialization import serialize_grid_coords

if TYPE_CHECKING:
    import flash_attn
    import spconv.pytorch as spconv
    import torch_scatter


flash_attn, _FLASH_ATTN_AVAILABLE = optional_import("flash_attn", url="https://github.com/Dao-AILab/flash-attention")
torch_scatter, _ = optional_import("torch_scatter", url="https://github.com/rusty1s/pytorch_scatter")
spconv, _ = optional_import("spconv.pytorch", url="https://github.com/traveller59/spconv")


def mlp_block(
    in_features: int,
    hidden_features: Optional[int] = None,
    out_features: Optional[int] = None,
    act: ActLike = "gelu",
    dropout: float = 0.0,
) -> nn.Module:
    out_features = out_features or in_features
    hidden_features = hidden_features or in_features
    return nn.Sequential(
        nn.Linear(in_features, hidden_features),
        create_act(act),
        nn.Dropout(dropout),
        nn.Linear(hidden_features, out_features),
        nn.Dropout(dropout),
    )


class RelativePositionalEncoding(nn.Module):
    """
    Relative Positional Encoding for 3D point clouds.

    This module computes positional encodings based on the relative positions
    between points in 3D space. It creates a learnable lookup table that maps
    relative coordinates to attention biases for multi-head attention.

    Args:
        patch_size: Number of points in each attention patch
        num_heads: Number of attention heads
    """

    def __init__(self, patch_size: int, num_heads: int) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.coords_boundary = int(math.pow(4 * patch_size, 1 / 3) * 2)
        self.rpe_num = 2 * self.coords_boundary + 1
        self.rpe_table = nn.Parameter(torch.zeros(3 * self.rpe_num, num_heads))
        nn.init.trunc_normal_(self.rpe_table, std=0.02)

    def forward(self, relative_coords: Tensor) -> Tensor:
        """Compute the relative positional encoding for given relative coordinates.

        Args:
            relative_coords: Relative coordinates between points, shape $(N, K, K, 3)$
                where $N$ is batch size, $K$ is number of points per patch

        Returns:
            Positional encoding tensor of shape $(N, num_heads, K, K)$
        """
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
        enable_rpe: bool = False,
        enable_flash: bool = True,
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
        self.enable_rpe = enable_rpe
        self.enable_flash = enable_flash
        self.patch_size = patch_size
        self.attn_drop = attn_drop
        self.proj_drop = proj_drop

        if enable_flash:
            if not _FLASH_ATTN_AVAILABLE:
                raise OptionalImportError(flash_attn)

            if enable_rpe:
                warnings.warn(
                    "Relative positional encoding is not supported with Flash Attention. "  # fmt: skip
                    "Setting `enable_rpe` to `False`."
                )
                self.enable_rpe = False

            if upcast_attention:
                warnings.warn(
                    "Upcasting attention is not supported with Flash Attention. "  # fmt: skip
                    "Setting `upcast_attention` to `False`."
                )
                self.upcast_attention = False

            if upcast_softmax:
                warnings.warn(
                    "Upcasting softmax is not supported with Flash Attention. "  # fmt: skip
                    "Setting `upcast_softmax` to `False`."
                )
                self.upcast_softmax = False

        self.qkv = torch.nn.Linear(channels, channels * 3, bias=qkv_bias)
        self.proj = torch.nn.Linear(channels, channels)
        self.softmax = torch.nn.Softmax(dim=-1)
        self.rpe = RelativePositionalEncoding(patch_size, num_heads) if self.enable_rpe else None

    def _forward_default_attn(self, qkv: Tensor, *, grid_coords: Optional[Tensor] = None, patch_size: int) -> Tensor:
        K, H, C = patch_size, self.num_heads, self.channels

        # Encode and reshape qkv: (N', K, 3, H, C') -> (3, N', H, K, C')
        q, k, v = qkv.reshape(-1, K, 3, H, C // H).permute(2, 0, 3, 1, 4).unbind(dim=0)

        if self.upcast_attention:
            q = q.float()
            k = k.float()

        attn = (q * self.scale) @ k.transpose(-2, -1)  # (N', H, K, K)

        if self.enable_rpe:
            if self.rpe is None:
                raise RuntimeError(
                    "`rpe` must be provided when `enable_rpe` is True. "
                    "Please check the model configuration or reinitialize the model."
                )

            if grid_coords is None:
                raise ValueError("`grid_coords` must be provided when `enable_rpe` is True")

            grid_coords = grid_coords.reshape(-1, K, 3)
            relative_coords = grid_coords.unsqueeze(2) - grid_coords.unsqueeze(1)
            attn = attn + self.rpe(relative_coords)

        if self.upcast_softmax:
            attn = attn.float()

        attn = self.softmax(attn)
        attn = F.dropout(attn, p=self.attn_drop, training=self.training).to(qkv.dtype)

        feat = (attn @ v).transpose(1, 2).reshape(-1, C)
        return feat

    def _forward_flash_attn(self, qkv: Tensor, batch_idxs: Tensor) -> Tensor:
        H, C = self.num_heads, self.channels

        patch_idxs = split_batch(batch_idxs, self.patch_size)
        offset = batch_to_offset(patch_idxs)
        # NOTE: The first element of `cu_seqlens` is always 0, and should be int32 to work with `flash-attn`
        cu_seqlens = torch.cat([torch.tensor([0], device=batch_idxs.device, dtype=torch.int32), offset.to(torch.int32)])

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
        batch_idx: Tensor,
        grid_coords: Optional[Tensor] = None,
        serialized_order: Optional[Tensor] = None,
        serialized_inverse: Optional[Tensor] = None,
    ) -> Any:
        # NOTE: For default attention (i.e. without Flash Attention), we use the patch size
        # as the minimum between the batch sizes and the specified patch size
        patch_size: int = (
            self.patch_size  # type: ignore[assignment]
            if self.enable_flash
            else min(torch.bincount(batch_idx).min().item(), self.patch_size)  # fmt: skip
        )

        # Only pad batches larger than the patch size
        padded_idxs, unpadded_idxs, padded_batch_idxs = divisible_pad(
            batch_idx,
            patch_size,
            return_inverse=True,
            mode="above",
        )

        order = serialized_order[padded_idxs] if serialized_order is not None else padded_idxs
        inverse = serialized_inverse[unpadded_idxs] if serialized_inverse is not None else unpadded_idxs

        # Apply attention
        qkv = self.qkv(features)[order]
        if self.enable_flash:
            features = self._forward_flash_attn(qkv, padded_batch_idxs)
        else:
            if grid_coords is not None:
                grid_coords = grid_coords[order]
            features = self._forward_default_attn(qkv, grid_coords=grid_coords, patch_size=patch_size)

        features = features[inverse]

        # Head projection
        features = self.proj(features)
        features = F.dropout(features, p=self.proj_drop, training=self.training)
        return features


def to_sparse_conv_tensor(
    features: Tensor,
    grid_coords: Tensor,
    batch_idx: Tensor,
    spatial_shape: Optional[Sequence[int]] = None,
    padding: int = 96,
) -> "spconv.SparseConvTensor":
    if spatial_shape is None:
        spatial_shape = torch.add(torch.max(grid_coords, dim=0).values, padding).tolist()

    return spconv.SparseConvTensor(
        features=features,
        indices=torch.cat([batch_idx.unsqueeze(-1).int(), grid_coords.int()], dim=1).contiguous(),
        spatial_shape=spatial_shape,
        batch_size=batch_idx[-1].item() + 1,
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
        norm_layer: NormLike = nn.LayerNorm,
        act_layer: ActLike = "gelu",
        pre_norm: bool = True,
        spconv_indice_key: Optional[str] = None,
        enable_rpe: bool = False,
        enable_flash: bool = True,
        upcast_attention: bool = True,
        upcast_softmax: bool = True,
    ):
        super().__init__()
        self.conv = spconv.SubMConv3d(
            channels,
            channels,
            kernel_size=3,
            bias=True,
            indice_key=spconv_indice_key,
        )
        self.mlp1 = nn.Linear(channels, channels)
        self.attn = SerializedAttention(
            channels=channels,
            patch_size=patch_size,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            enable_rpe=enable_rpe,
            enable_flash=enable_flash,
            upcast_attention=upcast_attention,
            upcast_softmax=upcast_softmax,
        )
        self.mlp2 = mlp_block(
            in_features=channels,
            hidden_features=int(channels * mlp_ratio),
            out_features=channels,
            act=act_layer,
            dropout=proj_drop,
        )
        self.pre_norm = pre_norm
        self.norm = create_norm(norm_layer, channels)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(
        self,
        features: Tensor,
        grid_coords: Tensor,
        batch_idx: Tensor,
        *,
        serialized_order: Optional[Tensor] = None,
        serialized_inverse: Optional[Tensor] = None,
        sparse_features: "Optional[spconv.SparseConvTensor]" = None,
        return_sparse_features: bool = False,
    ) -> Any:
        if sparse_features is None:
            # Get the sparse features if not provided
            sparse_features = to_sparse_conv_tensor(features, grid_coords, batch_idx)

        # Conv + Skip connection
        shortcut = features
        sparse_features = self.conv(sparse_features)
        features = sparse_features.features
        features = self.mlp1(features)
        features = self.norm(features)

        # NOTE: Skip connection and save the new shortcut
        features = shortcut = shortcut + features

        if self.pre_norm:
            features = self.norm(features)

        # Attention + Skip connection
        features = self.attn(
            features,
            batch_idx,
            grid_coords=grid_coords,
            serialized_order=serialized_order,
            serialized_inverse=serialized_inverse,
        )
        features = self.drop_path(features)
        features = shortcut + features

        if not self.pre_norm:
            features = self.norm(features)

        # MLP + Skip connection
        shortcut = features

        if self.pre_norm:
            features = self.norm(features)

        features = self.mlp2(features)
        features = self.drop_path(features)
        features = shortcut + features

        if not self.pre_norm:
            features = self.norm(features)

        sparse_features = sparse_features.replace_feature(features)

        if return_sparse_features:
            return features, sparse_features
        return features


# features: Tensor,
# coords: Tensor,
# grid_coords: Tensor,
# batch_idx: Tensor,
# serialized_code: Tensor,
# serialized_order: Tensor,
# serialized_inverse: Tensor,
# serialized_depth: int,


class SerializedPointData(TypedDict, total=False):
    features: Tensor
    coords: Tensor
    grid_coords: Tensor
    serialized_code: Tensor
    serialized_order: Tensor
    serialized_inverse: Tensor
    serialized_depth: int
    batch_idx: Tensor


class SerializedPooling(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 2,
        norm: Optional[NormLike] = None,
        act: Optional[ActLike] = None,
        reduce: str = "max",
        shuffle_orders: bool = True,
    ):
        super().__init__()
        if reduce not in ["sum", "mean", "min", "max"]:
            raise ValueError(f"Invalid reduce operation: {reduce}. Must be one of: 'sum', 'mean', 'min', 'max'.")
        if stride != 2 ** (math.ceil(stride) - 1).bit_length():
            raise ValueError(f"Invalid stride: {stride}. Must be a power of 2.")

        self.stride = stride
        self.reduce = reduce
        self.shuffle_orders = shuffle_orders
        self.proj = nn.Linear(in_channels, out_channels)
        self.norm = create_norm(norm, out_channels) if norm is not None else None
        self.act = create_act(act) if act is not None else None

    def forward(
        self,
        features: Tensor,
        coords: Tensor,
        grid_coords: Tensor,
        serialized_code: Tensor,
        serialized_order: Tensor,
        serialized_inverse: Tensor,
        serialized_depth: int,
        batch_idx: Tensor,
    ) -> Any:
        pooling_depth = (math.ceil(self.stride) - 1).bit_length()
        if pooling_depth > serialized_depth:
            pooling_depth = 0

        code = serialized_code >> pooling_depth * 3
        _, cluster, counts = torch.unique(
            code[0],
            sorted=True,
            return_inverse=True,
            return_counts=True,
        )
        # indices of point sorted by cluster, for torch_scatter.segment_csr
        _, indices = torch.sort(cluster)
        # index pointer for sorted point, for torch_scatter.segment_csr
        idx_ptr = torch.cat([counts.new_zeros(1), torch.cumsum(counts, dim=0)])
        # head_indices of each cluster, for reduce attr e.g. code, batch
        head_indices = indices[idx_ptr[:-1]]

        # generate down code, order, inverse
        code = code[:, head_indices]
        order = torch.argsort(code)
        inverse = torch.zeros_like(order).scatter_(
            dim=1,
            index=order,
            src=torch.arange(0, code.shape[1], device=order.device).repeat(code.shape[0], 1),
        )

        if self.shuffle_orders:
            perm = torch.randperm(code.shape[0])
            code = code[perm]
            order = order[perm]
            inverse = inverse[perm]

        # collect information
        point = dict(
            features=torch_scatter.segment_csr(self.proj(features)[indices], idx_ptr, reduce=self.reduce),
            coords=torch_scatter.segment_csr(coords[indices], idx_ptr, reduce="mean"),
            grid_coords=grid_coords[head_indices] >> pooling_depth,
            serialized_code=code,
            serialized_order=order,
            serialized_inverse=inverse,
            serialized_depth=serialized_depth - pooling_depth,
            batch_idx=batch_idx[head_indices],
        )

        if self.traceable:
            point["pooling_inverse"] = cluster
            point["pooling_parent"] = point

        if self.norm is not None:
            point["features"] = self.norm(point["features"])
        if self.act is not None:
            point["features"] = self.act(point["features"])

        # NOTE: point.sparsify() is used to compute the sparse features, so not a big deal...
        # return features, coords, grid_coords, serialized_code, serialized_order, serialized_inverse, batch_idx
        return point


# class SerializedUnpooling(nn.Module):
#     def __init__(
#         self,
#         in_channels: int,
#         skip_channels: int,
#         out_channels: int,
#         norm: Optional[NormLike] = None,
#         act: Optional[ActLike] = None,
#     ):
#         super().__init__()
#         self.proj = linear_block(
#             in_features=in_channels,
#             out_features=out_channels,
#             norm=norm,
#             act=act,
#             order="lnad",
#         )
#         self.proj_skip = linear_block(
#             in_features=skip_channels,
#             out_features=out_channels,
#             norm=norm,
#             act=act,
#             order="lnad",
#         )

#     def forward(self, point: Any) -> Any:
#         assert "pooling_parent" in point.keys()
#         assert "pooling_inverse" in point.keys()
#         parent = point.pop("pooling_parent")
#         inverse = point.pop("pooling_inverse")

#         point = self.proj(point)
#         parent = self.proj_skip(parent)

#         parent.feat = parent.feat + point.feat[inverse]

#         return parent


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
        batch_idx: Tensor,
        return_sparse_features: bool = False,
    ) -> Union[Tensor, Tuple[Tensor, "spconv.SparseConvTensor"]]:
        sparse_features = to_sparse_conv_tensor(features, grid_coords, batch_idx)
        sparse_features = self.stem(sparse_features)

        features = sparse_features.features
        if self.norm is not None:
            features = self.norm(features)
        if self.act is not None:
            features = self.act(features)

        if return_sparse_features:
            return features, sparse_features
        return features


class EncoderBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        num_blocks: int = 2,
        num_heads: int = 2,
        patch_size: int = 48,
        mlp_ratio: float = 4,
        qkv_bias: bool = True,
        qk_scale: Optional[float] = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        drop_path: float = 0.3,
        norm_layer: NormLike = nn.LayerNorm,
        act_layer: ActLike = nn.GELU,
        serialization_order: Optional[Union[str, Sequence[str]]] = ("z", "z-trans"),
        pre_norm: bool = True,
        enable_rpe: bool = False,
        enable_flash: bool = True,
        upcast_attention: bool = False,
        upcast_softmax: bool = False,
        spconv_indice_key: Optional[str] = None,
        pooling: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.serialization_order = ensure_tuple(serialization_order)
        self.pooling = pooling
        self.blocks = nn.ModuleList()

        for i in range(num_blocks):
            self.blocks.add_module(
                f"block{i}",
                Block(
                    channels=channels,
                    num_heads=num_heads,
                    patch_size=patch_size,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    attn_drop=attn_drop,
                    proj_drop=proj_drop,
                    drop_path=drop_path,
                    norm_layer=norm_layer,
                    act_layer=act_layer,
                    pre_norm=pre_norm,
                    spconv_indice_key=spconv_indice_key,
                    enable_rpe=enable_rpe,
                    enable_flash=enable_flash,
                    upcast_attention=upcast_attention,
                    upcast_softmax=upcast_softmax,
                ),
            )

    def forward(
        self,
        features: Tensor,
        coords: Tensor,
        grid_coords: Tensor,
        batch_idx: Tensor,
        serialized_code: Tensor,
        serialized_order: Tensor,
        serialized_inverse: Tensor,
        serialized_depth: int,
    ) -> Any:
        if self.pooling is not None:
            features, coords, grid_coords, serialized_code, serialized_order, serialized_inverse, batch_idx = (
                self.pooling(
                    features=features,
                    coords=coords,
                    grid_coords=grid_coords,
                    batch_idx=batch_idx,
                    serialized_code=serialized_code,
                    serialized_depth=serialized_depth,
                )
            )

        sparse_features = to_sparse_conv_tensor(features, grid_coords, batch_idx)
        for i, block in enumerate(self.blocks):
            order_idx = i % len(self.serialization_order)
            features, sparse_features = block(
                features=features,
                grid_coords=grid_coords,
                sparse_features=sparse_features,
                batch_idx=batch_idx,
                serialized_order=serialized_order[order_idx],
                serialized_inverse=serialized_inverse[order_idx],
                return_sparse_features=True,
            )

        return features, coords, grid_coords, serialized_code, serialized_order, serialized_inverse, batch_idx


class PointTransformerV3Classification(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        serialization_orders: Sequence[str] = ("z", "z-trans"),
        stride: Sequence[int] = (2, 2, 2, 2),
        enc_depths: Sequence[int] = (2, 2, 2, 6, 2),
        enc_channels: Sequence[int] = (32, 64, 128, 256, 512),
        enc_num_head: Sequence[int] = (2, 4, 8, 16, 32),
        enc_patch_size: Sequence[int] = (48, 48, 48, 48, 48),
        mlp_ratio: float = 4,
        qkv_bias: bool = True,
        qk_scale: Optional[float] = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        drop_path: float = 0.3,
        pre_norm: bool = True,
        shuffle_orders: bool = True,
        enable_rpe: bool = False,
        enable_flash: bool = True,
        upcast_attention: bool = False,
        upcast_softmax: bool = False,
    ):
        super().__init__()
        self.serialization_orders = ensure_tuple(serialization_orders)
        self.shuffle_orders = shuffle_orders
        self.num_stages = len(enc_depths)
        assert self.num_stages == len(stride) + 1
        assert self.num_stages == len(enc_depths)
        assert self.num_stages == len(enc_channels)
        assert self.num_stages == len(enc_num_head)
        assert self.num_stages == len(enc_patch_size)

        norm_layer = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)
        act_layer = nn.GELU

        self.embedding = Embedding(
            in_channels=in_channels,
            embedding_dim=enc_channels[0],
            norm=norm_layer,
            act=act_layer,
        )

        enc_drop_path = [x.item() for x in torch.linspace(0, drop_path, sum(enc_depths))]
        self.encoder = nn.ModuleList()

        for i in range(self.num_stages):
            partial_block = partial(
                EncoderBlock,
                channels=enc_channels[i],
                num_blocks=enc_depths[i],
                num_heads=enc_num_head[i],
                patch_size=enc_patch_size[i],
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                attn_drop=attn_drop,
                proj_drop=proj_drop,
                drop_path=enc_drop_path[i],
                pre_norm=pre_norm,
                norm_layer=norm_layer,
                act_layer=act_layer,
                enable_rpe=enable_rpe,
                enable_flash=enable_flash,
                upcast_attention=upcast_attention,
                upcast_softmax=upcast_softmax,
            )

            if i > 0:
                pooling = SerializedPooling(
                    in_channels=enc_channels[i - 1],
                    out_channels=enc_channels[i],
                    stride=stride[i - 1],
                    norm=norm_layer,
                    act=act_layer,
                )
                block = partial_block(pooling=pooling)
            else:
                block = partial_block()

            self.encoder.append(block)

        self.head = nn.Linear(enc_channels[-1], num_classes)

    def serialize_grid_coords(self, grid_coords: Tensor, batch_idx: Tensor) -> Any:
        serialized_depth = int(grid_coords.max()).bit_length()
        serialized_orders = []
        serialized_codes = []
        serialized_inverses = []

        # Shuffle randomly the serialization orders if desired, for better generalization
        order_idxs = (
            torch.randperm(len(self.serialization_orders))
            if self.shuffle_orders
            else torch.arange(len(self.serialization_orders))
        )

        for order_idx in order_idxs:
            serialized_code, serialized_order, serialized_inverse = serialize_grid_coords(
                grid_coords,
                batch_idx,
                depth=serialized_depth,
                order=self.serialization_orders[order_idx],
                shuffle_orders=self.shuffle_orders,
            )
            serialized_orders.append(serialized_order)
            serialized_codes.append(serialized_code)
            serialized_inverses.append(serialized_inverse)

        return (
            torch.stack(serialized_orders),
            torch.stack(serialized_codes),
            torch.stack(serialized_inverses),
            serialized_depth,
        )

    def forward(self, coords: Tensor, grid_coords: Tensor, features: Tensor, batch_idx: Tensor) -> Any:
        serialized_orders, serialized_codes, serialized_inverses, serialized_depth = self.serialize_grid_coords(
            grid_coords=grid_coords,
            batch_idx=batch_idx,
        )

        features, sparse_features = self.embedding(features, grid_coords, batch_idx, return_sparse_features=True)
        for enc_block in self.encoder:
            # TODO: out = enc_block(**out)
            features, coords, grid_coords, batch_idx = enc_block(
                features=features,
                coords=coords,
                grid_coords=grid_coords,
                batch_idx=batch_idx,
                serialized_order=serialized_orders,
                serialized_code=serialized_codes,
                serialized_inverse=serialized_inverses,
                serialized_depth=serialized_depth,
            )

        # PSEUDO CODE:
        # b0_out = self.block0()
        # for i in range(1, self.num_stages):
        #     bi_out = self.block{i}(
        #         features=features,
        #         coords=coords,
        #         grid_coords=grid_coords,
        #         batch_idx=batch_idx,
        #     )

        # for dec_block in self.decoder:
        #     out = dec_block(
        #         features=features,
        #         coords=coords,
        #         grid_coords=grid_coords,
        #         batch_idx=batch_idx,
        #     )
        return features, coords, grid_coords, batch_idx
