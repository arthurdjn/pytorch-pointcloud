import math
from typing import TYPE_CHECKING, Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

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


class SerializedAttention(nn.Module):
    r"""
    Serialized attention layer, introduced in the paper
    [Point Transformer V3: Simpler, Faster, Stronger](https://arxiv.org/abs/2312.10035).
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
        use_rpe: bool = False,
        use_flash_attn: bool = True,
        upcast_attention: bool = True,
        upcast_softmax: bool = True,
    ):
        super().__init__()
        if use_flash_attn:
            if not _FLASH_ATTN_AVAILABLE:
                raise ImportError(flash_attn)
            elif use_rpe:
                raise ValueError("Relative positional encoding is not supported with Flash Attention.")
            elif upcast_attention:
                raise ValueError("Upcasting attention is not supported with Flash Attention.")
            elif upcast_softmax:
                raise ValueError("Upcasting softmax is not supported with Flash Attention.")

        assert channels % num_heads == 0
        self.channels = channels
        self.num_heads = num_heads
        self.scale = qk_scale or (channels // num_heads) ** -0.5
        self.upcast_attention = upcast_attention
        self.upcast_softmax = upcast_softmax
        self.use_rpe = use_rpe
        self.use_flash_attn = use_flash_attn
        self.patch_size = patch_size
        self.attn_drop = attn_drop
        self.proj_drop = proj_drop

        self.qkv = nn.Linear(channels, channels * 3, bias=qkv_bias)
        self.proj = nn.Linear(channels, channels)
        self.softmax = nn.Softmax(dim=-1)
        self.rpe = RelativePositionalEncoding(patch_size, num_heads) if self.use_rpe else None

    def _forward_default_attn(self, qkv: Tensor, pos: OptTensor, patch_size: int) -> Tensor:
        K, H, C = patch_size, self.num_heads, self.channels

        # Encode and reshape qkv: (N', K, 3, H, C') -> (3, N', H, K, C')
        q, k, v = qkv.reshape(-1, K, 3, H, C // H).permute(2, 0, 3, 1, 4).unbind(dim=0)

        if self.upcast_attention:
            q = q.float()
            k = k.float()

        attn = (q * self.scale) @ k.transpose(-2, -1)  # (N', H, K, K)

        if self.use_rpe:
            if self.rpe is None:
                raise RuntimeError(
                    "`rpe` must be provided when `use_rpe` is True. "
                    "Please check the model configuration or reinitialize the model."
                )

            if pos is None:
                raise ValueError("`pos` must be provided when `use_rpe` is True")

            pos = pos.reshape(-1, K, 3)
            relative_coords = pos.unsqueeze(2) - pos.unsqueeze(1)
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
        x: Tensor,
        pos: OptTensor,
        batch: Tensor,
        serialized_order: OptTensor = None,
        serialized_inverse: OptTensor = None,
    ) -> Any:
        patch_size = self.patch_size
        # NOTE: For default attention (i.e. without Flash Attention), we use the patch size
        # as the minimum between the batch sizes and the specified patch size
        if not self.use_flash_attn:
            patch_size = min(int(torch.bincount(batch).min().item()), self.patch_size)

        # Only pad batches larger than the patch size
        padded_indices, unpadded_indices, padded_batch = divisible_pad(
            batch,
            patch_size,
            mode="above",
            pad_fill="replicate",
            return_inverse=True,
        )

        order = serialized_order[padded_indices] if serialized_order is not None else padded_indices
        inverse = unpadded_indices[serialized_inverse] if serialized_inverse is not None else unpadded_indices

        # Apply attention
        qkv = self.qkv(x)[order]
        if self.use_flash_attn:
            x = self._forward_flash_attn(qkv, padded_batch)
        else:
            if pos is not None:
                pos = pos[order]
            x = self._forward_default_attn(qkv, pos, patch_size)

        x = x[inverse]

        # Head projection
        x = self.proj(x)
        x = F.dropout(x, p=self.proj_drop, training=self.training)
        return x
