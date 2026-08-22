"""SphereFormer segmentation model.

{{ paper("2303.12766") }}
"""

from collections import OrderedDict
from functools import partial
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

import torch_pointcloud.transforms as T
from torch_pointcloud.layers import SparseModule, SparseResidualBlock
from torch_pointcloud.layers.act import create_act
from torch_pointcloud.layers.dropouts import DropPath
from torch_pointcloud.layers.norms import create_norm
from torch_pointcloud.models._base import SegmentationModel
from torch_pointcloud.models._registry import register_model
from torch_pointcloud.utils.data import DataKeys
from torch_pointcloud.utils.imports import (
    _SPCONV_GITHUB_URL,
    _SPTR_GITHUB_URL,
    _TORCH_SCATTER_GITHUB_URL,
    optional_import,
)

if TYPE_CHECKING:
    import spconv.pytorch as spconv
    from spconv.core import ConvAlgo
    from spconv.pytorch import SparseConvTensor
    from torch_scatter import scatter_mean


spconv, _ = optional_import("spconv.pytorch", url=_SPCONV_GITHUB_URL)
SparseConvTensor, _ = optional_import("spconv.pytorch", "SparseConvTensor", url=_SPCONV_GITHUB_URL)
ConvAlgo, _ = optional_import("spconv.core", "ConvAlgo", url=_SPCONV_GITHUB_URL)
scatter_mean, _ = optional_import("torch_scatter", "scatter_mean", url=_TORCH_SCATTER_GITHUB_URL)
sptr, _SPTR_AVAILABLE = optional_import("sptr", url=_SPTR_GITHUB_URL)


def cart2sphere(pos: Tensor) -> Tensor:
    r"""Map Cartesian coordinates to spherical coordinates $(\theta, \phi, r)$ (degrees, degrees, meters).

    The azimuth $\theta = \operatorname{atan2}(y, x)$ and polar angle $\phi = \operatorname{atan2}(\sqrt{x^2+y^2}, z)$
    are returned in degrees (with $\theta$ shifted to $[0, 360)$); $r = \sqrt{x^2+y^2+z^2}$ is the radius.

    Args:
        pos: Cartesian coordinates.

    Returns:
        The spherical coordinates $(\theta, \phi, r)$.

    Shape:
        - Input: $(N, 3)$
        - Output: $(N, 3)$

    Example:
        ```pycon
        >>> import torch
        >>> from torch_pointcloud.models.sphereformer import cart2sphere
        >>> sphere = cart2sphere(torch.randn(8, 3))
        >>> sphere.shape
        torch.Size([8, 3])

        ```
    """
    x, y, z = pos[:, 0], pos[:, 1], pos[:, 2]
    theta = (torch.atan2(y, x) + np.pi) * 180 / np.pi
    phi = torch.atan2(torch.sqrt(x**2 + y**2), z) * 180 / np.pi
    r = torch.sqrt(x**2 + y**2 + z**2)
    return torch.stack([theta, phi, r], dim=-1)


def exponential_split(
    pos: Tensor,
    index_query: Tensor,
    index_key: Tensor,
    relative_position_index: Tensor,
    radial_split_exponent: float = 0.0125,
    offset: int = 24,
) -> Tensor:
    r"""Quantize the radial relative position $r_q - r_k$ into a signed, exponentially-growing bin index.

    Reproduces the reference radial split: bins are symmetric around $0$, double in width every two steps
    (`[0, a)`, `[a, 2a)`, `[2a, 4a)`, `[4a, 6a)`, `[6a, 10a)`, ... with $a$ the base bin width), and the sign
    of $r_q - r_k$ selects the positive or negative half. The returned index is shifted by `offset` (half the
    number of rows of the radial relative-position table, the reference `quant_size_scale`) so it indexes the
    table without going negative, and clamped to $[0, 2 \cdot \text{offset} - 1]$: radial gaps beyond the
    outermost bins fall into those bins instead of overflowing the table. The signed bin index overwrites the
    third (radial) column of `relative_position_index` in place, matching the `split_func` contract of the
    `sptr` kernel.

    Args:
        pos: Spherical coordinates whose third column is the radius $r$.
        index_query: Per-pair query indices.
        index_key: Per-pair key indices.
        relative_position_index: Per-pair, per-axis relative-position table indices whose radial column is
            replaced with the signed exponential bin index.
        radial_split_exponent: Base bin width of the radial exponential split.
        offset: Non-negative shift applied to the signed bin index; the radial table has
            $2 \cdot \text{offset}$ rows.

    Returns:
        The updated `relative_position_index` with its radial column set to the signed bin index.

    Shape:
        - `pos`: $(N, 3)$
        - `index_query`, `index_key`: $(M,)$
        - `relative_position_index`: $(M, 3)$
        - Output: $(M, 3)$
    """
    r = pos[:, 2]
    rel_pos = r[index_query.long()] - r[index_key.long()]
    rel_pos_abs = rel_pos.abs()
    flag_float = (rel_pos >= 0).float()
    idx = 2 * torch.floor(torch.log((rel_pos_abs + 2 * radial_split_exponent) / radial_split_exponent) / np.log(2)) - 2
    idx = idx + ((3 * (2 ** (idx // 2)) - 2) * radial_split_exponent <= rel_pos_abs).float()
    idx = idx * (2 * flag_float - 1) + (flag_float - 1)
    relative_position_index[:, 2] = (idx.long() + offset).clamp_(0, 2 * offset - 1)
    return relative_position_index


class WindowedRelPosAttention(SparseModule):
    r"""Block-diagonal windowed multi-head self-attention with contextual relative-position encoding.

    Runs two attentions in parallel and concatenates their heads: the first half of the heads attend within
    cubic (Cartesian) windows, the second half within radial (spherical) windows. Within each window every
    point attends to all others; scores are $q \cdot k$ plus a learnable relative-position bias on both query
    and key, softmax-normalized per query, and the value is augmented with its own relative-position bias before
    the weighted sum. The windowed attention is computed by the `sptr` CUDA kernel
    (`sptr.sparse_self_attention` with `pe_type="contextual"`, `rel_query=rel_key=rel_value=True`), mirroring the
    reference `SparseMultiheadSASphereConcat`.

    Args:
        embed_dim: Token dimension.
        num_heads: Number of attention heads (split evenly between cubic and spherical branches).
        window_size: Cubic window size, of shape $(3,)$.
        window_size_sphere: Spherical window size $(\theta, \phi, r)$, of shape $(3,)$.
        quant_size: Cubic relative-position quantization size, of shape $(3,)$.
        quant_size_sphere: Spherical relative-position quantization size, of shape $(3,)$.
        radial_split_exponent: Base bin width for the radial exponential split.
        qkv_bias: Whether the fused QKV projection uses a bias.
        qk_scale: Optional override for the $1/\sqrt{d}$ attention scale.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        window_size: Tensor,
        window_size_sphere: Tensor,
        quant_size: Tensor,
        quant_size_sphere: Tensor,
        radial_split_exponent: float = 0.0125,
        qkv_bias: bool = True,
        qk_scale: Optional[float] = None,
    ) -> None:
        super().__init__()
        if not _SPTR_AVAILABLE:
            raise ImportError(
                f"Optional module `sptr` is required to use `WindowedRelPosAttention`. "
                f"Install it from {_SPTR_GITHUB_URL}."
            )
        if num_heads < 2:
            raise ValueError(f"`num_heads` must be at least 2 (one cubic and one spherical head), got {num_heads}.")
        if embed_dim % num_heads != 0:
            raise ValueError(f"`embed_dim` ({embed_dim}) must be divisible by `num_heads` ({num_heads}).")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        head_dim = embed_dim // num_heads
        self.head_dim = head_dim
        self.scale = qk_scale or head_dim**-0.5
        self.radial_split_exponent = radial_split_exponent

        self.window_size = window_size.detach().cpu().numpy().astype(np.float32)
        self.window_size_sphere = window_size_sphere.detach().cpu().numpy().astype(np.float32)
        self.quant_size = quant_size.detach().cpu().numpy().astype(np.float32)
        self.quant_size_sphere = quant_size_sphere.detach().cpu().numpy().astype(np.float32)

        self.num_heads_cubic = num_heads // 2
        self.num_heads_sphere = num_heads - self.num_heads_cubic

        quant_grid_length = int((float(window_size[0]) + 1e-4) / float(quant_size[0]))
        self.quant_grid_length = quant_grid_length
        quant_grid_length_sphere = int((float(window_size_sphere[0]) + 1e-4) / float(quant_size_sphere[0]))
        self.quant_grid_length_sphere = quant_grid_length_sphere

        self.relative_pos_query_table = nn.Parameter(
            torch.zeros(2 * quant_grid_length - 1, 3, self.num_heads_cubic, head_dim)
        )
        self.relative_pos_key_table = nn.Parameter(
            torch.zeros(2 * quant_grid_length - 1, 3, self.num_heads_cubic, head_dim)
        )
        self.relative_pos_value_table = nn.Parameter(
            torch.zeros(2 * quant_grid_length - 1, 3, self.num_heads_cubic, head_dim)
        )
        self.relative_pos_query_table_sphere = nn.Parameter(
            torch.zeros(2 * quant_grid_length_sphere, 3, self.num_heads_sphere, head_dim)
        )
        self.relative_pos_key_table_sphere = nn.Parameter(
            torch.zeros(2 * quant_grid_length_sphere, 3, self.num_heads_sphere, head_dim)
        )
        self.relative_pos_value_table_sphere = nn.Parameter(
            torch.zeros(2 * quant_grid_length_sphere, 3, self.num_heads_sphere, head_dim)
        )

        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for table in (
            self.relative_pos_query_table,
            self.relative_pos_key_table,
            self.relative_pos_value_table,
            self.relative_pos_query_table_sphere,
            self.relative_pos_key_table_sphere,
            self.relative_pos_value_table_sphere,
        ):
            nn.init.trunc_normal_(table, std=0.02)

    def forward(self, x: Tensor, pos: Tensor, batch: Tensor) -> Tensor:
        num_points, channels = x.shape
        qkv = self.qkv(x).reshape(num_points, 3, self.num_heads, self.head_dim).permute(1, 0, 2, 3).contiguous()
        query, key, value = qkv[0], qkv[1], qkv[2]
        query = query * self.scale

        pos_sphere = cart2sphere(pos)

        index_0, index_0_offsets, n_max, index_1, index_1_offsets, sort_idx = sptr.get_indices_params(
            pos, batch, self.window_size, False
        )
        out_cubic = sptr.sparse_self_attention(
            query[:, : self.num_heads_cubic].contiguous().float(),
            key[:, : self.num_heads_cubic].contiguous().float(),
            value[:, : self.num_heads_cubic].contiguous().float(),
            pos.float(),
            index_0.int(),
            index_0_offsets.int(),
            n_max,
            index_1.int(),
            index_1_offsets.int(),
            sort_idx,
            self.window_size,
            False,
            pe_type="contextual",
            rel_query=True,
            rel_key=True,
            rel_value=True,
            quant_size=self.quant_size,
            quant_grid_length=self.quant_grid_length,
            relative_pos_query_table=self.relative_pos_query_table.float(),
            relative_pos_key_table=self.relative_pos_key_table.float(),
            relative_pos_value_table=self.relative_pos_value_table.float(),
        )

        (
            index_0_sphere,
            index_0_offsets_sphere,
            n_max_sphere,
            index_1_sphere,
            index_1_offsets_sphere,
            sort_idx_sphere,
        ) = sptr.get_indices_params(pos_sphere, batch, self.window_size_sphere, False)
        out_sphere = sptr.sparse_self_attention(
            query[:, self.num_heads_cubic :].contiguous().float(),
            key[:, self.num_heads_cubic :].contiguous().float(),
            value[:, self.num_heads_cubic :].contiguous().float(),
            pos_sphere.float(),
            index_0_sphere.int(),
            index_0_offsets_sphere.int(),
            n_max_sphere,
            index_1_sphere.int(),
            index_1_offsets_sphere.int(),
            sort_idx_sphere,
            self.window_size_sphere,
            False,
            pe_type="contextual",
            rel_query=True,
            rel_key=True,
            rel_value=True,
            quant_size=self.quant_size_sphere,
            quant_grid_length=self.quant_grid_length_sphere,
            relative_pos_query_table=self.relative_pos_query_table_sphere.float(),
            relative_pos_key_table=self.relative_pos_key_table_sphere.float(),
            relative_pos_value_table=self.relative_pos_value_table_sphere.float(),
            split_func=partial(
                exponential_split,
                radial_split_exponent=self.radial_split_exponent,
                offset=self.quant_grid_length_sphere,
            ),
        )

        out = torch.cat([out_cubic, out_sphere], 1).reshape(num_points, channels)
        return self.proj(out)


class SphereFormerBlock(nn.Module):
    r"""Pre-norm transformer block wrapping `WindowedRelPosAttention` with an MLP, as in the reference.

    Applies `x = x + attn(LN(x))` then `x = x + mlp(LN(x))`, with a GELU MLP of ratio `mlp_ratio` and an
    optional stochastic-depth `drop_path` on each residual branch.

    Args:
        embed_dim: Token dimension.
        num_heads: Number of attention heads.
        window_size: Cubic window size, of shape $(3,)$.
        window_size_sphere: Spherical window size, of shape $(3,)$.
        quant_size: Cubic relative-position quantization size, of shape $(3,)$.
        quant_size_sphere: Spherical relative-position quantization size, of shape $(3,)$.
        radial_split_exponent: Base bin width for the radial exponential split.
        mlp_ratio: Hidden-dim multiplier for the MLP.
        drop_path: Stochastic-depth rate.
        qkv_bias: Whether the QKV projection uses a bias.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        window_size: Tensor,
        window_size_sphere: Tensor,
        quant_size: Tensor,
        quant_size_sphere: Tensor,
        radial_split_exponent: float = 0.0125,
        mlp_ratio: float = 4.0,
        drop_path: float = 0.0,
        qkv_bias: bool = True,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = WindowedRelPosAttention(
            embed_dim,
            num_heads=num_heads,
            window_size=window_size,
            window_size_sphere=window_size_sphere,
            quant_size=quant_size,
            quant_size_sphere=quant_size_sphere,
            radial_split_exponent=radial_split_exponent,
            qkv_bias=qkv_bias,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = nn.LayerNorm(embed_dim)
        mlp_channels = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_channels),
            nn.GELU(),
            nn.Linear(mlp_channels, embed_dim),
        )

    def forward(self, x: Tensor, pos: Tensor, batch: Tensor) -> Tensor:
        x = x + self.drop_path(self.attn(self.norm1(x), pos, batch))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


def _scale_window(window_size: Tensor, quant_size: Tensor, scale: float, scale_last: bool) -> Tuple[Tensor, Tensor]:
    factors = window_size.new_tensor([scale, scale, scale if scale_last else 1.0])
    return window_size * factors, quant_size * factors


class SphereFormerUBlock(nn.Module):
    r"""Recursive UNet block: sparse residual blocks + windowed attention, then a downsample/upsample branch.

    Each level runs `block_reps` sparse residual blocks, an optional `SphereFormerBlock`, then (for non-leaf
    levels) a strided sparse convolution into the next-deeper `SphereFormerUBlock`, an inverse convolution
    back, a skip concatenation, and `block_reps` tail residual blocks. The cubic and spherical window sizes are
    scaled by `window_size_scale` at every deeper level, mirroring the reference.

    Args:
        planes: Channel count of this level and all deeper levels.
        block_reps: Number of residual blocks before (and after) the recursive branch.
        window_size: Cubic window size at this level, of shape $(3,)$.
        window_size_sphere: Spherical window size at this level, of shape $(3,)$.
        quant_size: Cubic quantization size at this level, of shape $(3,)$.
        quant_size_sphere: Spherical quantization size at this level, of shape $(3,)$.
        head_dim: Per-head dimension (sets `num_heads = planes[0] // head_dim`).
        window_size_scale: Pair `(cubic_scale, sphere_scale)` applied per deeper level.
        drop_path: Per-level stochastic-depth rates (indexed by level).
        radial_split_exponent: Base bin width for the radial exponential split.
        indice_key_id: spconv indice-key id for this level.
        sphere_layers: Levels (`indice_key_id`) that get a `SphereFormerBlock`.
        norm: Normalization layer name / callable.
        norm_kwargs: Extra keyword arguments for the normalisation layer.
        act: Activation name / callable.
        act_kwargs: Extra keyword arguments for the activation.
    """

    def __init__(
        self,
        planes: Sequence[int],
        block_reps: int,
        window_size: Tensor,
        window_size_sphere: Tensor,
        quant_size: Tensor,
        quant_size_sphere: Tensor,
        head_dim: int = 16,
        window_size_scale: Tuple[float, float] = (2.0, 2.0),
        drop_path: Sequence[float] = (0.0,),
        radial_split_exponent: float = 0.0125,
        indice_key_id: int = 1,
        sphere_layers: Sequence[int] = (1, 2, 3, 4, 5),
        norm: Union[str, Callable, None] = "batch_norm",
        norm_kwargs: Optional[Dict[str, Any]] = None,
        act: Union[str, Callable, None] = "relu",
        act_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()
        self.planes = tuple(planes)
        self.indice_key_id = indice_key_id
        self.sphere_layers = tuple(sphere_layers)

        blocks = OrderedDict(
            (
                f"block{i}",
                SparseResidualBlock(
                    planes[0],
                    planes[0],
                    indice_key=f"subm{indice_key_id}",
                    norm=norm,
                    norm_kwargs=norm_kwargs,
                    act=act,
                    act_kwargs=act_kwargs,
                ),
            )
            for i in range(block_reps)
        )
        self.blocks = spconv.SparseSequential(blocks)

        self.transformer_block: Optional[SphereFormerBlock] = None
        if indice_key_id in self.sphere_layers:
            if planes[0] % head_dim != 0:
                raise ValueError(f"`planes[0]` ({planes[0]}) must be divisible by `head_dim` ({head_dim}).")
            self.transformer_block = SphereFormerBlock(
                planes[0],
                num_heads=planes[0] // head_dim,
                window_size=window_size,
                window_size_sphere=window_size_sphere,
                quant_size=quant_size,
                quant_size_sphere=quant_size_sphere,
                radial_split_exponent=radial_split_exponent,
                drop_path=float(drop_path[0]),
            )

        if len(planes) > 1:
            self.conv = spconv.SparseSequential(
                create_norm(norm, planes[0], **(norm_kwargs or {})) or nn.Identity(),
                create_act(act, **(act_kwargs or {})) or nn.Identity(),
                spconv.SparseConv3d(
                    planes[0],
                    planes[1],
                    kernel_size=2,
                    stride=2,
                    bias=False,
                    indice_key=f"spconv{indice_key_id}",
                    algo=ConvAlgo.Native,
                ),
            )

            scale_cubic, scale_sphere = window_size_scale
            window_next, quant_next = _scale_window(window_size, quant_size, scale_cubic, scale_last=True)
            window_sphere_next, quant_sphere_next = _scale_window(
                window_size_sphere, quant_size_sphere, scale_sphere, scale_last=False
            )

            self.unet = SphereFormerUBlock(
                planes[1:],
                block_reps,
                window_next,
                window_sphere_next,
                quant_next,
                quant_sphere_next,
                head_dim=head_dim,
                window_size_scale=window_size_scale,
                drop_path=drop_path[1:],
                radial_split_exponent=radial_split_exponent,
                indice_key_id=indice_key_id + 1,
                sphere_layers=sphere_layers,
                norm=norm,
                norm_kwargs=norm_kwargs,
                act=act,
                act_kwargs=act_kwargs,
            )

            self.deconv = spconv.SparseSequential(
                create_norm(norm, planes[1], **(norm_kwargs or {})) or nn.Identity(),
                create_act(act, **(act_kwargs or {})) or nn.Identity(),
                spconv.SparseInverseConv3d(
                    planes[1],
                    planes[0],
                    kernel_size=2,
                    bias=False,
                    indice_key=f"spconv{indice_key_id}",
                    algo=ConvAlgo.Native,
                ),
            )

            blocks_tail = OrderedDict(
                (
                    f"block{i}",
                    SparseResidualBlock(
                        planes[0] * (2 - i),
                        planes[0],
                        indice_key=f"subm{indice_key_id}",
                        norm=norm,
                        norm_kwargs=norm_kwargs,
                        act=act,
                        act_kwargs=act_kwargs,
                    ),
                )
                for i in range(block_reps)
            )
            self.blocks_tail = spconv.SparseSequential(blocks_tail)

    def forward(self, x: "SparseConvTensor", pos: Tensor, batch: Tensor) -> "SparseConvTensor":
        out = self.blocks(x)

        if self.transformer_block is not None:
            out = out.replace_feature(self.transformer_block(out.features, pos, batch))

        identity = SparseConvTensor(out.features, out.indices, out.spatial_shape, out.batch_size)

        if len(self.planes) > 1:
            out_decoder = self.conv(out)

            indice_pairs = out_decoder.indice_dict[f"spconv{self.indice_key_id}"].indice_pairs
            pair_in, pair_out = indice_pairs[0], indice_pairs[1]
            valid = pair_in != -1
            pair_in, pair_out = pair_in[valid].long(), pair_out[valid].long()
            pos_next = scatter_mean(pos[pair_in], index=pair_out, dim=0)
            batch_next = scatter_mean(batch.float()[pair_in], index=pair_out, dim=0)

            out_decoder = self.unet(out_decoder, pos_next, batch_next.long())
            out_decoder = self.deconv(out_decoder)
            out = out.replace_feature(torch.cat([identity.features, out_decoder.features], dim=1))
            out = self.blocks_tail(out)

        return out


class SphereFormerSegmentation(SegmentationModel):
    r"""SphereFormer semantic-segmentation model, as described in the paper
    :arxiv: [Spherical Transformer for LiDAR-based 3D Recognition](https://arxiv.org/abs/2303.12766).

    A sparse-convolution UNet32 backbone with a cubic + radial windowed self-attention block at every stage.
    The windowed attention is computed by the `sptr` CUDA kernel (an optional dependency), as in the reference.
    Inputs follow the packed convention: point features `x`, integer voxel-grid coordinates `pos_grid`,
    real-valued coordinates `pos` (used by the attention), and a per-point `batch` index. The output is
    per-point class logits.

    Args:
        in_channels: Input feature channels.
        num_classes: Number of semantic classes.
        base_channels: Stem / level-0 channel count $m$.
        layers: Per-level channel counts (length = number of UNet levels).
        block_reps: Residual blocks per level (before and after the recursive branch).
        head_dim: Per-head dimension for the windowed attention.
        window_size: Base cubic window size (`voxel_size * patch_size * window`), of shape $(3,)$.
        window_size_sphere: Base spherical window size $(\theta, \phi, r)$, of shape $(3,)$.
        quant_size: Base cubic quantization size, of shape $(3,)$.
        quant_size_sphere: Base spherical quantization size, of shape $(3,)$.
        window_size_scale: Pair `(cubic_scale, sphere_scale)` applied per deeper level.
        sphere_layers: Levels (1-indexed) that receive a windowed-attention block.
        radial_split_exponent: Base bin width for the radial exponential split.
        drop_path: Maximum stochastic-depth rate (linearly spread across levels).
        min_spatial_shape: Per-axis lower bound on the inferred sparse spatial shape.
        norm: Normalization layer name / callable.
        norm_kwargs: Extra keyword arguments for the normalisation layer.
        act: Activation name / callable.
        act_kwargs: Extra keyword arguments for the activation.

    Example:
        ```pycon
        >>> import torch
        >>> from torch_pointcloud.models import create_model
        >>> model = create_model("sphereformer.semantickitti", task="segmentation").eval()  # doctest: +SKIP
        >>> pos = torch.rand(1000, 3) * 10  # doctest: +SKIP
        >>> pos_grid = (pos / 0.05).floor().long()  # doctest: +SKIP
        >>> x = torch.cat([pos, torch.rand(1000, 1)], dim=1)  # doctest: +SKIP
        >>> batch = torch.zeros(1000, dtype=torch.long)  # doctest: +SKIP
        >>> logits = model(x, pos, pos_grid, batch)  # doctest: +SKIP
        >>> logits.shape  # doctest: +SKIP
        torch.Size([1000, 19])

        ```
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        *,
        base_channels: int = 32,
        layers: Sequence[int] = (32, 64, 128, 256, 256),
        block_reps: int = 2,
        head_dim: int = 16,
        window_size: Sequence[float] = (0.3, 0.3, 0.3),
        window_size_sphere: Sequence[float] = (2.0, 2.0, 80.0),
        quant_size: Sequence[float] = (0.0125, 0.0125, 0.0125),
        quant_size_sphere: Sequence[float] = (2.0 / 24, 2.0 / 24, 80.0 / 24),
        window_size_scale: Tuple[float, float] = (2.0, 1.5),
        sphere_layers: Sequence[int] = (1, 2, 3, 4, 5),
        radial_split_exponent: float = 0.0125,
        drop_path: float = 0.0,
        min_spatial_shape: int = 128,
        norm: Union[str, Callable, None] = "batch_norm",
        norm_kwargs: Optional[Dict[str, Any]] = None,
        act: Union[str, Callable, None] = "relu",
        act_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(in_channels=in_channels, num_classes=num_classes)
        self.base_channels = base_channels
        self.layers = tuple(layers)
        self.min_spatial_shape = min_spatial_shape
        if norm_kwargs is None:
            norm_kwargs = {"eps": 1e-4, "momentum": 0.1}

        self.input_conv = spconv.SparseSequential(
            spconv.SubMConv3d(in_channels, base_channels, kernel_size=3, padding=1, bias=False, indice_key="subm1")
        )

        drop_paths = [float(v) for v in torch.linspace(0, drop_path, len(self.layers) + 2)]

        self.unet = SphereFormerUBlock(
            self.layers,
            block_reps=block_reps,
            window_size=torch.as_tensor(window_size, dtype=torch.float32),
            window_size_sphere=torch.as_tensor(window_size_sphere, dtype=torch.float32),
            quant_size=torch.as_tensor(quant_size, dtype=torch.float32),
            quant_size_sphere=torch.as_tensor(quant_size_sphere, dtype=torch.float32),
            head_dim=head_dim,
            window_size_scale=window_size_scale,
            drop_path=drop_paths,
            radial_split_exponent=radial_split_exponent,
            indice_key_id=1,
            sphere_layers=sphere_layers,
            norm=norm,
            norm_kwargs=norm_kwargs,
            act=act,
            act_kwargs=act_kwargs,
        )

        self.output_layer = spconv.SparseSequential(
            create_norm(norm, base_channels, **norm_kwargs) or nn.Identity(),
            create_act(act, **(act_kwargs or {})) or nn.Identity(),
        )

        self.head: nn.Module = self.configure_head()
        self.reset_parameters()

    def configure_head(self) -> nn.Module:
        if self.num_classes <= 0:
            return nn.Identity()
        return nn.Linear(self.base_channels, self.num_classes)

    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.BatchNorm1d):
                nn.init.constant_(module.weight, 1.0)
                nn.init.constant_(module.bias, 0.0)

    def reset_classifier(self, num_classes: int) -> None:
        self.num_classes = num_classes
        self.head = self.configure_head()

    def forward_features(
        self,
        x: Tensor,
        pos: Tensor,
        pos_grid: Tensor,
        batch: Tensor,
    ) -> "SparseConvTensor":
        indices = torch.cat([batch.unsqueeze(-1), pos_grid], dim=1).int()
        spatial_shape = np.clip((pos_grid.max(0).values + 1).cpu().numpy(), self.min_spatial_shape, None)
        batch_size = int(batch.max().item()) + 1
        sparse_x = SparseConvTensor(x, indices, spatial_shape.tolist(), batch_size)
        sparse_x = self.input_conv(sparse_x)
        return self.unet(sparse_x, pos, batch)

    def forward_decoder(self, x: "SparseConvTensor") -> "SparseConvTensor":
        return self.output_layer(x)

    def forward_head(self, x: "SparseConvTensor", pre_logits: bool = False) -> Tensor:
        return x.features if pre_logits else self.head(x.features)

    def forward(self, x: Tensor, pos: Tensor, pos_grid: Tensor, batch: Tensor) -> Tensor:
        sparse_x = self.forward_features(x, pos, pos_grid, batch)
        sparse_x = self.forward_decoder(sparse_x)
        return self.forward_head(sparse_x)


@register_model(
    "sphereformer.semantickitti",
    task="segmentation",
    # The original pretrained weights are no longer downloadable (the authors' CUHK OneDrive links are dead,
    # see dvlab-research/SphereFormer issue #78), so the architecture is registered without pretrained weights.
    weights=None,
    transform=T.Compose(
        [
            T.Relabel(
                keys=DataKeys.SEGMENT,
                labels={
                    10: 0,
                    252: 0,
                    11: 1,
                    15: 2,
                    18: 3,
                    258: 3,
                    20: 4,
                    259: 4,
                    30: 5,
                    254: 5,
                    31: 6,
                    253: 6,
                    32: 7,
                    255: 7,
                    40: 8,
                    44: 9,
                    48: 10,
                    49: 11,
                    50: 12,
                    51: 13,
                    70: 14,
                    71: 15,
                    72: 16,
                    80: 17,
                    81: 18,
                },
                default=255,
            ),
            T.Cat(keys=[DataKeys.POS, DataKeys.INTENSITY], dst_key=DataKeys.X, dim=1),
            T.Voxelize(
                pos_key=DataKeys.POS,
                pos_reduce="mean",
                keys=[DataKeys.X],
                reduce=["mean"],
                size=0.05,
                grid_pos_key=DataKeys.POS_GRID,
                dst_inverse_key=DataKeys.INVERSE,
            ),
        ]
    ),
    hparams=dict(
        in_channels=4,
        num_classes=19,
        base_channels=32,
        layers=(32, 64, 128, 256, 256),
        block_reps=2,
        head_dim=16,
        window_size=(0.3, 0.3, 0.3),
        window_size_sphere=(2.0, 2.0, 80.0),
        quant_size=(0.0125, 0.0125, 0.0125),
        quant_size_sphere=(2.0 / 24, 2.0 / 24, 80.0 / 24),
        window_size_scale=(2.0, 1.5),
        sphere_layers=(1, 2, 3, 4, 5),
        radial_split_exponent=0.0125,
        norm_kwargs={"eps": 1e-4, "momentum": 0.1},
    ),
)
def sphereformer_semantickitti(**hparams: Any) -> SphereFormerSegmentation:
    return SphereFormerSegmentation(**hparams)


@register_model(
    "sphereformer.nuscenes",
    task="segmentation",
    weights=None,
    transform=T.Compose(
        [
            T.Relabel(
                keys=DataKeys.SEGMENT,
                labels={
                    9: 0,
                    14: 1,
                    15: 2,
                    16: 2,
                    17: 3,
                    18: 4,
                    21: 5,
                    2: 6,
                    3: 6,
                    4: 6,
                    6: 6,
                    12: 7,
                    22: 8,
                    23: 9,
                    24: 10,
                    25: 11,
                    26: 12,
                    27: 13,
                    28: 14,
                    30: 15,
                },
                default=255,
            ),
            T.Cat(keys=[DataKeys.POS, DataKeys.INTENSITY], dst_key=DataKeys.X, dim=1),
            T.Voxelize(
                pos_key=DataKeys.POS,
                pos_reduce="mean",
                keys=[DataKeys.X],
                reduce=["mean"],
                size=0.1,
                grid_pos_key=DataKeys.POS_GRID,
                dst_inverse_key=DataKeys.INVERSE,
            ),
        ]
    ),
    hparams=dict(
        in_channels=4,
        num_classes=16,
        base_channels=32,
        layers=(32, 64, 128, 256, 256),
        block_reps=2,
        head_dim=16,
        window_size=(0.6, 0.6, 0.6),
        window_size_sphere=(2.0, 2.0, 120.0),
        quant_size=(0.025, 0.025, 0.025),
        quant_size_sphere=(2.0 / 24, 2.0 / 24, 120.0 / 24),
        window_size_scale=(2.0, 2.0),
        sphere_layers=(1, 2, 3, 4, 5),
        radial_split_exponent=0.0125,
        norm_kwargs={"eps": 1e-4, "momentum": 0.1},
    ),
)
def sphereformer_nuscenes(**hparams: Any) -> SphereFormerSegmentation:
    return SphereFormerSegmentation(**hparams)
