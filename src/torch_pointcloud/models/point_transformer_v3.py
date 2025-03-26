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
from typing import TYPE_CHECKING, Any, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from torch_pointcloud.layers import (
    ActLike,
    NormLike,
    create_act,
    create_cls_head,
    create_norm,
    create_pool,
    create_seg_head,
)
from torch_pointcloud.layers.drop import DropPath
from torch_pointcloud.transforms.functional import divisible_pad, split_batch
from torch_pointcloud.utils.conversion import batch_to_offset, ensure_tuple
from torch_pointcloud.utils.imports import OptionalImportError, optional_import
from torch_pointcloud.utils.serialization import serialize_coords
from torch_pointcloud.utils.types import OptTensor

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

    def _forward_default_attn(self, qkv: Tensor, grid_coords: OptTensor, patch_size: int) -> Tensor:
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
            if self.enable_flash
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
        if self.enable_flash:
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
        norm: NormLike = "layer_norm",
        act: ActLike = "gelu",
        cpe_indice_key: Optional[str] = None,
        enable_rpe: bool = False,
        enable_flash: bool = True,
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
            enable_rpe=enable_rpe,
            enable_flash=enable_flash,
            upcast_attention=upcast_attention,
            upcast_softmax=upcast_softmax,
        )

        self.norm2 = create_norm(norm, channels)
        self.mlp = mlp_block(
            in_features=channels,
            hidden_features=int(channels * mlp_ratio),
            out_features=channels,
            act=act,
            dropout=proj_drop,
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
        shuffle_orders: bool = True,
    ):
        super().__init__()
        if reduce not in ["sum", "mean", "min", "max"]:
            raise ValueError(
                f"Invalid reduce operaIntTensor, tion: {reduce}. Must be one of: 'sum', 'mean', 'min', 'max'."
            )
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
        grid_coords: Tensor,
        batch: Tensor,
        serialized_code: OptTensor = None,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        # Generate serialization code if not provided
        if serialized_code is None:
            serialized_code = serialize_coords(grid_coords, batch)

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

        return features, grid_coords, batch, pooled_code


def random_permutation(*tensors: Tensor) -> Union[Tensor, Tuple[Tensor, ...]]:
    perm = torch.randperm(tensors[0].shape[0])
    outputs = tuple(tensor[perm] for tensor in tensors)
    return outputs if len(outputs) > 1 else outputs[0]


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
    ) -> Tensor:
        sparse_features = to_sparse_conv_tensor(features, grid_coords, batch_idx)
        sparse_features = self.stem(sparse_features)

        features = sparse_features.features
        if self.norm is not None:
            features = self.norm(features)
        if self.act is not None:
            features = self.act(features)

        return features


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
        norm: NormLike = "batch_norm1d",
        act: ActLike = "gelu",
        mlp_ratio: float = 4,
        qkv_bias: bool = True,
        qk_scale: Optional[float] = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        drop_path: float = 0.3,
        shuffle_orders: bool = True,
        enable_rpe: bool = False,
        enable_flash: bool = True,
        upcast_attention: bool = False,
        upcast_softmax: bool = False,
        dropout: float = 0.0,
        global_pool: str = "max",
    ):
        super().__init__()
        self.serialization_orders = ensure_tuple(serialization_orders)
        self.shuffle_orders = shuffle_orders
        self.num_stages = len(enc_depths)

        enc_depth = len(enc_depths)
        if not (enc_depth == len(stride) + 1 == len(enc_channels) == len(enc_num_head) == len(enc_patch_size)):
            raise ValueError(
                "The number of stages must be equal to the length of `stride + 1`, "
                "the length of `enc_depths`, the length of `enc_channels`, the length of `enc_num_head`, "
                f"and the length of `enc_patch_size`. Got enc_depths={enc_depths}, stride={stride}, "
                f"enc_channels={enc_channels}, enc_num_head={enc_num_head}, enc_patch_size={enc_patch_size}"
            )

        self.embedding = Embedding(in_channels=in_channels, embedding_dim=enc_channels[0], norm=norm, act=act)

        down_blocks = self.configure_downsample_blocks(
            stride=stride,
            enc_depths=enc_depths,
            enc_channels=enc_channels,
            enc_num_head=enc_num_head,
            enc_patch_size=enc_patch_size,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            drop_path=drop_path,
            shuffle_orders=shuffle_orders,
            enable_rpe=enable_rpe,
            enable_flash=enable_flash,
            upcast_attention=upcast_attention,
            upcast_softmax=upcast_softmax,
            norm=norm,
            act=act,
        )
        for d, down in enumerate(down_blocks):
            self.add_module(f"down{d}", down)

        self.embedding_dim = enc_channels[-1]
        self.dropout = dropout
        self.global_pool = create_pool(global_pool)
        self.head = create_cls_head(num_features=self.embedding_dim, num_classes=num_classes)

    def configure_downsample_blocks(
        self,
        *,
        stride: Sequence[int],
        enc_depths: Sequence[int],
        enc_channels: Sequence[int],
        enc_num_head: Sequence[int],
        enc_patch_size: Sequence[int],
        mlp_ratio: float,
        qkv_bias: bool,
        qk_scale: Optional[float],
        attn_drop: float,
        proj_drop: float,
        drop_path: float,
        shuffle_orders: bool,
        enable_rpe: bool,
        enable_flash: bool,
        upcast_attention: bool,
        upcast_softmax: bool,
        norm: NormLike,
        act: ActLike,
    ) -> List[nn.ModuleList]:
        enc_drop_paths = torch.split(torch.linspace(0, drop_path, sum(enc_depths)), list(enc_depths))
        enc_blocks = []
        for d in range(len(enc_depths)):
            blocks = nn.ModuleList()
            if d > 0:
                pooling = SerializedPooling(
                    in_channels=enc_channels[d - 1],
                    out_channels=enc_channels[d],
                    stride=stride[d - 1],
                    norm=norm,
                    act=act,
                    shuffle_orders=shuffle_orders,
                )
                blocks.append(pooling)

            for i in range(enc_depths[d]):
                block = Block(
                    channels=enc_channels[d],
                    num_heads=enc_num_head[d],
                    patch_size=enc_patch_size[d],
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    attn_drop=attn_drop,
                    proj_drop=proj_drop,
                    drop_path=enc_drop_paths[d][i].item(),
                    norm="layer_norm",
                    act=act,
                    cpe_indice_key=f"stage{d}",
                    enable_rpe=enable_rpe,
                    enable_flash=enable_flash,
                    upcast_attention=upcast_attention,
                    upcast_softmax=upcast_softmax,
                )
                blocks.append(block)
            enc_blocks.append(blocks)
        return enc_blocks

    def reset_classifier(self, num_classes: int, global_pool: str = "max", **kwargs: Any) -> None:
        """Resets the classification head with new parameters.

        Note:
            To set an empty classification head, use `num_classes=0`.

        Args:
            num_classes: Number of output classes.
            global_pool: Pooling method to aggregate point features ("max" or "mean").
            **kwargs: Additional keyword arguments to pass to the classification head.
        """
        self.num_classes = num_classes
        self.global_pool = create_pool(global_pool) if isinstance(global_pool, str) else global_pool
        self.head = create_cls_head(num_features=self.embedding_dim, num_classes=self.num_classes, **kwargs)

    def serialize_coords(self, grid_coords: Tensor, batch_idx: Tensor, depth: int) -> Tensor:
        serialization_orders = self.serialization_orders
        if self.shuffle_orders:
            perm = torch.randperm(len(serialization_orders))
            serialization_orders = [serialization_orders[i] for i in perm]

        serialized_codes = [
            serialize_coords(
                grid_coords,
                batch_idx,
                depth=depth,
                order=serialization_order,
            )
            for serialization_order in serialization_orders
        ]

        return torch.stack(serialized_codes)

    def forward_features(
        self,
        features: OptTensor,
        grid_coords: Tensor,
        batch: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Forward pass of the PointTransformerV3 encoder, returning pre-pooling features.

        Args:
            grid_coords: Grid coordinates of shape $(N, 3)$.
            features: Additional point features of shape $(N, features_dim)$.
            batch: Batch indices for each point of shape $(N,)$.

        Returns:
            Pre-pooling features of shape $(N, mlp2_dims[-1])$ where $N$ is the batch size.
        """
        features = features if features is not None else grid_coords.float()

        # Serialize the grid coordinates for each serialization order (e.g. "z", "z-trans", etc.)
        # NOTE: For faster processing, we pre-compute the serialized code, order and inverse.
        # These variables will be reused in each blocks and updated after each pooling operation.
        serialized_depth = int(grid_coords.max()).bit_length()
        serialized_code = self.serialize_coords(grid_coords, batch, depth=serialized_depth)
        serialized_order = torch.argsort(serialized_code, dim=1)
        serialized_inverse = torch.argsort(serialized_order, dim=1)

        # NOTE: It is important that the outputs of the layers / blocks reuse the same
        # variable names, so that we can easily forward the inputs to the next layer / block.
        features = self.embedding(features, grid_coords, batch)

        # Forward encoder blocks (pooling + blocks)
        order_idx = 0
        for d in range(self.num_stages):
            downi = self.get_submodule(f"down{d}")
            assert isinstance(downi, nn.ModuleList)  # Sanity check

            for layer in downi:
                if isinstance(layer, SerializedPooling):
                    # Apply pooling and update serialized code, order and inverse
                    features, grid_coords, batch, serialized_code = layer(features, grid_coords, batch, serialized_code)
                    serialized_order = torch.argsort(serialized_code, dim=1)
                    serialized_inverse = torch.argsort(serialized_order, dim=1)

                else:
                    # Block pass, using a specific serialization order
                    # NOTE: For better regularization, we use different serialization orders for each block.
                    curr_order = order_idx % len(self.serialization_orders)
                    order_idx += 1
                    features = layer(
                        features,
                        grid_coords,
                        batch,
                        serialized_order=serialized_order[curr_order],
                        serialized_inverse=serialized_inverse[curr_order],
                    )

        return features, grid_coords, batch

    def forward_head(self, features: Tensor, batch: Tensor, pre_logits: bool = False) -> Tensor:
        """Forward pass of the classification head from pre-pooling features.

        Args:
            x: Pre-pooling features of shape $(N, embedding_dim)$.
            batch: Batch indices for each point of shape $(N,)$.
            pre_logits: Whether to return pre-logits. Defaults to False.

        Returns:
            Classification logits of shape $(B, num_classes)$.
        """
        features = self.global_pool(features, batch)
        if self.dropout:
            features = F.dropout(features, p=float(self.dropout), training=self.training)
        return features if pre_logits else self.head(features)

    def forward(self, features: Tensor, grid_coords: OptTensor, batch: Tensor) -> Tensor:
        """Forward pass of the PointNet classification network.

        Args:
            coords: Point coordinates of shape $(N, coords_dim)$.
            features: Additional point features of shape $(N, features_dim)$.
            batch_idxs: Batch indices for each point of shape $(N,)$.

        Returns:
            Classification logits of shape $(B, num_classes)$.
        """
        features, grid_coords, batch = self.forward_features(features, grid_coords, batch)
        return self.forward_head(features, batch)
