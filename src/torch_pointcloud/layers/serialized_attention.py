"""Serialized attention variants used by Point Transformer V3 and descendants."""

import math
from typing import TYPE_CHECKING, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from torch_pointcloud.layers.rope import Point3DRoPE
from torch_pointcloud.transforms.functional import divisible_pad, split_batch
from torch_pointcloud.utils.conversion import batch_to_offset
from torch_pointcloud.utils.imports import optional_import
from torch_pointcloud.utils.types import OptTensor

if TYPE_CHECKING:
    import flash_attn


flash_attn, _FLASH_ATTN_AVAILABLE = optional_import("flash_attn")


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


def _flash_attend_qkv(
    qkv_packed: Tensor,
    padded_batch: Tensor,
    patch_size: int,
    scale: float,
    attn_drop: float,
    training: bool,
) -> Tensor:
    """Variable-length flash attention over fixed-size patches.

    Wraps `flash_attn.flash_attn_varlen_qkvpacked_func` with the per-batch
    `cu_seqlens` derivation that all variants share.
    """
    patch_idxs = split_batch(padded_batch, patch_size)
    offset = batch_to_offset(patch_idxs)
    # cu_seqlens must start at 0 and be int32 for flash-attn
    cu_seqlens = torch.cat([torch.tensor([0], device=padded_batch.device, dtype=torch.int), offset.int()])
    return flash_attn.flash_attn_varlen_qkvpacked_func(
        qkv_packed,
        cu_seqlens,
        max_seqlen=patch_size,
        dropout_p=attn_drop if training else 0,
        softmax_scale=scale,
    )


class SerializedAttention(nn.Module):
    r"""Vanilla serialized attention from
    :arxiv: [Point Transformer V3](https://arxiv.org/abs/2312.10035).

    No positional information is added inside attention itself — relative
    structure comes from the conditional position embedding (CPE) applied
    around each block.
    """

    def __init__(
        self,
        channels: int,
        num_heads: int,
        patch_size: int,
        qkv_bias: bool = True,
        qk_scale: Optional[float] = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        use_flash_attn: bool = True,
        upcast_attention: bool = True,
        upcast_softmax: bool = True,
    ) -> None:
        super().__init__()
        if channels % num_heads != 0:
            raise ValueError(f"channels ({channels}) must be divisible by num_heads ({num_heads}).")
        if use_flash_attn:
            if not _FLASH_ATTN_AVAILABLE:
                raise ImportError(
                    "`flash_attn` is required when `use_flash_attn=True`. Install with `pip install flash-attn`."
                )
            if upcast_attention:
                raise ValueError("Upcasting attention is not supported with Flash Attention.")
            if upcast_softmax:
                raise ValueError("Upcasting softmax is not supported with Flash Attention.")
        self.channels = channels
        self.num_heads = num_heads
        self.patch_size = patch_size
        self.scale = qk_scale or (channels // num_heads) ** -0.5
        self.attn_drop = attn_drop
        self.proj_drop = proj_drop
        self.use_flash_attn = use_flash_attn
        self.upcast_attention = upcast_attention
        self.upcast_softmax = upcast_softmax

        self.qkv = nn.Linear(channels, channels * 3, bias=qkv_bias)
        self.proj = nn.Linear(channels, channels)

    def forward(
        self,
        x: Tensor,
        pos_grid: OptTensor,
        batch: Tensor,
        serialized_order: OptTensor = None,
        serialized_inverse: OptTensor = None,
        pos: OptTensor = None,
    ) -> Tensor:
        H, C = self.num_heads, self.channels
        patch_size = (
            self.patch_size if self.use_flash_attn else min(int(torch.bincount(batch).min().item()), self.patch_size)
        )

        padded_indices, unpadded_indices, padded_batch = divisible_pad(
            batch, patch_size, mode="above", pad_fill="replicate", return_inverse=True
        )
        order = serialized_order[padded_indices] if serialized_order is not None else padded_indices
        inverse = unpadded_indices[serialized_inverse] if serialized_inverse is not None else unpadded_indices
        qkv = self.qkv(x)[order]

        if self.use_flash_attn:
            qkv_packed = qkv.half().reshape(-1, 3, H, C // H)
            feat = _flash_attend_qkv(qkv_packed, padded_batch, patch_size, self.scale, self.attn_drop, self.training)
            feat = feat.reshape(-1, C).to(qkv.dtype)
        else:
            K = patch_size
            q, k, v = qkv.reshape(-1, K, 3, H, C // H).permute(2, 0, 3, 1, 4).unbind(dim=0)
            if self.upcast_attention:
                q = q.float()
                k = k.float()
            attn = (q * self.scale) @ k.transpose(-2, -1)
            if self.upcast_softmax:
                attn = attn.float()
            attn = attn.softmax(dim=-1)
            attn = F.dropout(attn, p=self.attn_drop, training=self.training).to(qkv.dtype)
            feat = (attn @ v).transpose(1, 2).reshape(-1, C)

        feat = feat[inverse]
        feat = self.proj(feat)
        return F.dropout(feat, p=self.proj_drop, training=self.training)


class SerializedAttentionRPE(nn.Module):
    r"""Serialized attention with the relative position bias from PT-V3.

    Adds a learned per-head bias indexed by the integer voxel-grid offset
    between query and key inside each patch. Flash Attention does not support
    arbitrary attention biases, so this variant always uses the manual softmax
    path.
    """

    def __init__(
        self,
        channels: int,
        num_heads: int,
        patch_size: int,
        qkv_bias: bool = True,
        qk_scale: Optional[float] = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        upcast_attention: bool = True,
        upcast_softmax: bool = True,
    ) -> None:
        super().__init__()
        if channels % num_heads != 0:
            raise ValueError(f"channels ({channels}) must be divisible by num_heads ({num_heads}).")
        self.channels = channels
        self.num_heads = num_heads
        self.patch_size = patch_size
        self.scale = qk_scale or (channels // num_heads) ** -0.5
        self.attn_drop = attn_drop
        self.proj_drop = proj_drop
        self.use_flash_attn = False
        self.upcast_attention = upcast_attention
        self.upcast_softmax = upcast_softmax

        self.qkv = nn.Linear(channels, channels * 3, bias=qkv_bias)
        self.proj = nn.Linear(channels, channels)
        self.rpe = RelativePositionalEncoding(patch_size, num_heads)

    def forward(
        self,
        x: Tensor,
        pos_grid: OptTensor,
        batch: Tensor,
        serialized_order: OptTensor = None,
        serialized_inverse: OptTensor = None,
        pos: OptTensor = None,
    ) -> Tensor:
        if pos_grid is None:
            raise ValueError("`pos_grid` must be provided for SerializedAttentionRPE.")

        H, C = self.num_heads, self.channels
        patch_size = min(int(torch.bincount(batch).min().item()), self.patch_size)

        padded_indices, unpadded_indices, _ = divisible_pad(
            batch, patch_size, mode="above", pad_fill="replicate", return_inverse=True
        )
        order = serialized_order[padded_indices] if serialized_order is not None else padded_indices
        inverse = unpadded_indices[serialized_inverse] if serialized_inverse is not None else unpadded_indices
        qkv = self.qkv(x)[order]
        pos_grid_ordered = pos_grid[order]

        K = patch_size
        q, k, v = qkv.reshape(-1, K, 3, H, C // H).permute(2, 0, 3, 1, 4).unbind(dim=0)
        if self.upcast_attention:
            q = q.float()
            k = k.float()
        attn = (q * self.scale) @ k.transpose(-2, -1)

        pos_grid_ordered = pos_grid_ordered.reshape(-1, K, 3)
        relative_coords = pos_grid_ordered.unsqueeze(2) - pos_grid_ordered.unsqueeze(1)
        attn = attn + self.rpe(relative_coords)

        if self.upcast_softmax:
            attn = attn.float()
        attn = attn.softmax(dim=-1)
        attn = F.dropout(attn, p=self.attn_drop, training=self.training).to(qkv.dtype)
        feat = (attn @ v).transpose(1, 2).reshape(-1, C)

        feat = feat[inverse]
        feat = self.proj(feat)
        return F.dropout(feat, p=self.proj_drop, training=self.training)


class SerializedAttentionRoPE(nn.Module):
    r"""Serialized attention with 3D rotary position embedding from
    :arxiv: [Utonia](https://arxiv.org/abs/2603.03283).

    Rotates $Q$, $K$ via [`Point3DRoPE`](rope.md) using the real-valued metric
    position of each token. Flash Attention is supported and uses bfloat16
    (matching upstream Utonia's reference implementation).
    """

    def __init__(
        self,
        channels: int,
        num_heads: int,
        patch_size: int,
        qkv_bias: bool = True,
        qk_scale: Optional[float] = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        use_flash_attn: bool = True,
        upcast_attention: bool = True,
        upcast_softmax: bool = True,
        rope_base: float = 10.0,
    ) -> None:
        super().__init__()
        if channels % num_heads != 0:
            raise ValueError(f"channels ({channels}) must be divisible by num_heads ({num_heads}).")
        if use_flash_attn:
            if not _FLASH_ATTN_AVAILABLE:
                raise ImportError(
                    "`flash_attn` is required when `use_flash_attn=True`. Install with `pip install flash-attn`."
                )
            if upcast_attention:
                raise ValueError("Upcasting attention is not supported with Flash Attention.")
            if upcast_softmax:
                raise ValueError("Upcasting softmax is not supported with Flash Attention.")
        self.channels = channels
        self.num_heads = num_heads
        self.patch_size = patch_size
        self.scale = qk_scale or (channels // num_heads) ** -0.5
        self.attn_drop = attn_drop
        self.proj_drop = proj_drop
        self.use_flash_attn = use_flash_attn
        self.upcast_attention = upcast_attention
        self.upcast_softmax = upcast_softmax

        self.qkv = nn.Linear(channels, channels * 3, bias=qkv_bias)
        self.proj = nn.Linear(channels, channels)
        self.rope = Point3DRoPE(head_dim=channels // num_heads, base=rope_base)

    def forward(
        self,
        x: Tensor,
        pos_grid: OptTensor,
        batch: Tensor,
        serialized_order: OptTensor = None,
        serialized_inverse: OptTensor = None,
        pos: OptTensor = None,
    ) -> Tensor:
        if pos is None:
            raise ValueError("`pos` must be provided for SerializedAttentionRoPE.")

        H, C = self.num_heads, self.channels
        patch_size = (
            self.patch_size if self.use_flash_attn else min(int(torch.bincount(batch).min().item()), self.patch_size)
        )

        padded_indices, unpadded_indices, padded_batch = divisible_pad(
            batch, patch_size, mode="above", pad_fill="replicate", return_inverse=True
        )
        order = serialized_order[padded_indices] if serialized_order is not None else padded_indices
        inverse = unpadded_indices[serialized_inverse] if serialized_inverse is not None else unpadded_indices
        qkv = self.qkv(x)[order]
        pos_ordered = pos[padded_indices][order]

        q, k, v = qkv.reshape(-1, 3, H, C // H).unbind(dim=1)
        q, k = self.rope(q, k, pos_ordered)

        if self.use_flash_attn:
            qkv_packed = torch.stack([q, k, v], dim=1).to(torch.bfloat16)
            feat = _flash_attend_qkv(qkv_packed, padded_batch, patch_size, self.scale, self.attn_drop, self.training)
            feat = feat.reshape(-1, C).to(qkv.dtype)
        else:
            K = patch_size
            q = q.reshape(-1, K, H, C // H).permute(0, 2, 1, 3)
            k = k.reshape(-1, K, H, C // H).permute(0, 2, 1, 3)
            v = v.reshape(-1, K, H, C // H).permute(0, 2, 1, 3)
            if self.upcast_attention:
                q = q.float()
                k = k.float()
            attn = (q * self.scale) @ k.transpose(-2, -1)
            if self.upcast_softmax:
                attn = attn.float()
            attn = attn.softmax(dim=-1)
            attn = F.dropout(attn, p=self.attn_drop, training=self.training).to(qkv.dtype)
            feat = (attn @ v).transpose(1, 2).reshape(-1, C)

        feat = feat[inverse]
        feat = self.proj(feat)
        return F.dropout(feat, p=self.proj_drop, training=self.training)
