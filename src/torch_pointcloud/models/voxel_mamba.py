from functools import partial
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.nn import MLP

import torch_pointcloud.transforms as T
from torch_pointcloud.layers.act import create_act
from torch_pointcloud.layers.conv2d_blocks import Conv2dBlock
from torch_pointcloud.layers.norms import create_norm
from torch_pointcloud.utils.box3d import nms3d
from torch_pointcloud.utils.data import DataKeys
from torch_pointcloud.utils.hilbert import encode as hilbert_encode
from torch_pointcloud.utils.imports import optional_import
from torch_pointcloud.utils.types import Detection3D, OptTensor

from ._base import DetectionModel
from ._registry import register_model

if TYPE_CHECKING:
    import spconv.pytorch as spconv
    from mamba_ssm.modules.block import Block
    from torch_scatter import scatter_max, scatter_mean

spconv, _IS_SPCONV_AVAILABLE = optional_import("spconv.pytorch")
Block, _ = optional_import("mamba_ssm.modules.block", "Block")
Mamba, _ = optional_import("mamba_ssm", "Mamba")
RMSNorm, _ = optional_import("mamba_ssm.ops.triton.layer_norm", "RMSNorm")
scatter_max, _ = optional_import("torch_scatter", "scatter_max")
scatter_mean, _ = optional_import("torch_scatter", "scatter_mean")

_SparseModule: Any = spconv.SparseModule if _IS_SPCONV_AVAILABLE else nn.Module


def build_hilbert_template(rank: int, z_max: int, device: Union[str, torch.device] = "cpu") -> Tensor:
    r"""Build the flat Hilbert-curve lookup table used to serialize voxels (the reference's `curve_template`).

    A cube of side $N = 2^\text{rank}$ is enumerated in $(z, y, x)$ order and each voxel is mapped to
    its Hilbert-curve position, then the table is truncated to $N \cdot N \cdot z_\max$ entries (the
    curve only needs to cover voxels up to $z_\max$ in the height axis). The table is indexed by the
    flat coordinate $z \cdot N \cdot N + y \cdot N + x$ to read a voxel's position along the curve.

    This reproduces the reference template (`tools/hilbert_curves/create_hilbert_curve_template.py`)
    bit-exactly via [`hilbert.encode`][torch_pointcloud.utils.hilbert.encode], avoiding a 260 MB
    precomputed-weight download.

    Args:
        rank: Number of bits per dimension; the cube side is $N = 2^\text{rank}$.
        z_max: Height extent the curve must cover; the table keeps the first $N \cdot N \cdot z_\max$ entries.
        device: Device the template is built on.

    Returns:
        A `long` tensor of shape $(N \cdot N \cdot z_\max,)$ giving each voxel's position on the curve.

    Shape:
        - Output: $(N \cdot N \cdot z_\max,)$

    Example:
        >>> template = build_hilbert_template(rank=7, z_max=9)
        >>> template.shape
        torch.Size([147456])
    """
    n = 1 << rank
    chunks: List[Tensor] = []
    for z0 in range(z_max):
        y = torch.arange(n, device=device).view(n, 1).expand(n, n)
        x = torch.arange(n, device=device).view(1, n).expand(n, n)
        z = torch.full((n, n), z0, device=device)
        coords_zyx = torch.stack([z.reshape(-1), y.reshape(-1), x.reshape(-1)], dim=1)
        chunks.append(hilbert_encode(coords_zyx, num_dims=3, num_bits=rank).long())
    return torch.cat(chunks)


@torch.no_grad()
def hilbert_serialize(
    template: Tensor,
    coords: Tensor,
    batch_size: int,
    hilbert_spatial_size: Tuple[int, int, int],
    shift: int,
) -> Tuple[List[Tensor], List[Tensor]]:
    r"""Per-scene voxel orderings along the Hilbert curve (the reference's `get_hilbert_index_3d_mamba_lite`).

    Each voxel's flat coordinate (after a constant `shift` on every axis) indexes `template` to read
    its Hilbert position; sorting those positions within a scene yields the forward ordering, and
    sorting the forward ordering yields the inverse that scatters Mamba outputs back to voxel order.

    Args:
        template: Flat Hilbert lookup table from `build_hilbert_template`, shape $(\cdot,)$.
        coords: Voxel coordinates $(N, 4)$ as $(\text{batch}, z, y, x)$.
        batch_size: Number of scenes $B$ in the batch.
        hilbert_spatial_size: Curve grid $(z, y, x)$ used to flatten coordinates.
        shift: Constant offset added to $z$, $y$ and $x$ before indexing the table.

    Returns:
        `(forward, inverse)`, each a length-$B$ list of `long` index tensors.

    Shape:
        - coords: $(N, 4)$
    """
    _, size_y, size_x = hilbert_spatial_size
    x = coords[:, 3] + shift
    y = coords[:, 2] + shift
    z = coords[:, 1] + shift
    flat = (z * size_y * size_x + y * size_x + x).long()
    hilbert_inds = template[flat].long()

    forward: List[Tensor] = []
    inverse: List[Tensor] = []
    for i in range(batch_size):
        mask = coords[:, 0] == i
        order = torch.argsort(hilbert_inds[mask])
        forward.append(order)
        inverse.append(torch.argsort(order))
    return forward, inverse


class PFNLayer(nn.Module):
    r"""Pillar feature net layer (the reference's `PFNLayerV2`): linear + norm + ReLU with a per-voxel max-pool.

    Non-final layers halve their output width and concatenate the pooled feature back onto every
    point; the final layer returns the pooled per-voxel feature directly.

    Args:
        in_channels: Input feature channels.
        out_channels: Target output channels (halved internally for non-final layers).
        last: Whether this is the final PFN layer.

    Shape:
        - Input: $(N, C_\text{in})$ point features and $(N,)$ voxel index.
        - Output: $(N, C')$ for non-final layers, $(M, C_\text{out})$ for the final layer.
    """

    def __init__(self, in_channels: int, out_channels: int, last: bool) -> None:
        super().__init__()
        self.last = last
        out_dim = out_channels if last else out_channels // 2
        self.mlp = MLP(
            [in_channels, out_dim],
            act="relu",
            norm="batch_norm",
            norm_kwargs=dict(eps=1e-3, momentum=0.01),
            bias=False,
            plain_last=False,
        )

    def forward(self, inputs: Tensor, unq_inv: Tensor) -> Tensor:
        x = self.mlp(inputs)
        x_max = scatter_max(x, unq_inv, dim=0)[0]
        if self.last:
            return x_max
        return torch.cat([x, x_max[unq_inv]], dim=1)


class DynamicMeanVFE(nn.Module):
    r"""Dynamic mean voxel feature encoder for the Voxel Mamba detector.

    Points are assigned to voxels on the fly (no fixed points-per-voxel), augmented with the
    per-voxel cluster-mean offset and voxel-center offset, then encoded by a stack of
    linear + norm + ReLU PFN layers whose per-voxel max-pool produces one feature vector per voxel.

    Reference implementation: :github:
    [gwenzhang/Voxel-Mamba](https://github.com/gwenzhang/Voxel-Mamba) (`DynamicVoxelVFE`).

    Args:
        in_channels: Raw point feature channels including xyz (e.g. $5$ for Waymo $x, y, z, \text{intensity}, \text{elongation}$).
        num_filters: Output width of each PFN layer; the last entry is the voxel feature dim.
        voxel_size: Voxel size $(v_x, v_y, v_z)$.
        point_cloud_range: Range $(x_\min, y_\min, z_\min, x_\max, y_\max, z_\max)$.
        grid_size: Voxel grid extent $(n_x, n_y, n_z)$.

    Shape:
        - Input: $(N, C_\text{in})$ point features and $(N,)$ batch index.
        - Output: $(M, C_\text{out})$ voxel features and $(M, 4)$ voxel coords.
    """

    voxel_size: Tensor
    point_cloud_range: Tensor
    grid_size: Tensor

    def __init__(
        self,
        in_channels: int,
        num_filters: Sequence[int],
        voxel_size: Sequence[float],
        point_cloud_range: Sequence[float],
        grid_size: Sequence[int],
    ) -> None:
        super().__init__()
        feat_channels = in_channels + 6
        widths = [feat_channels, *num_filters]
        self.pfn_layers = nn.ModuleList(
            PFNLayer(widths[i], widths[i + 1], last=i >= len(widths) - 2) for i in range(len(widths) - 1)
        )
        self.out_channels = num_filters[-1]

        self.register_buffer("voxel_size", torch.tensor(voxel_size, dtype=torch.float32), persistent=False)
        self.register_buffer(
            "point_cloud_range", torch.tensor(point_cloud_range, dtype=torch.float32), persistent=False
        )
        self.register_buffer("grid_size", torch.tensor(list(grid_size), dtype=torch.long), persistent=False)

        self.scale_xyz = grid_size[0] * grid_size[1] * grid_size[2]
        self.scale_yz = grid_size[1] * grid_size[2]
        self.scale_z = grid_size[2]

    def forward(self, pos: Tensor, x: OptTensor, batch: Tensor) -> Tuple[Tensor, Tensor]:
        point_feats = pos if x is None else torch.cat([pos, x], dim=1)

        coords = torch.floor((pos - self.point_cloud_range[:3]) / self.voxel_size).int()
        mask = ((coords >= 0) & (coords < self.grid_size)).all(dim=1)
        coords = coords[mask]
        pos = pos[mask]
        point_feats = point_feats[mask]
        batch = batch[mask]

        merge = batch.int() * self.scale_xyz + coords[:, 0] * self.scale_yz + coords[:, 1] * self.scale_z + coords[:, 2]
        unq_indices, unq_inv, _ = torch.unique(merge, return_inverse=True, return_counts=True, dim=0)

        points_mean = scatter_mean(pos, unq_inv, dim=0)
        f_cluster = pos - points_mean[unq_inv]
        center = coords.to(pos.dtype) * self.voxel_size + (self.voxel_size / 2 + self.point_cloud_range[:3])
        f_center = pos - center

        features = torch.cat([point_feats, f_cluster, f_center], dim=1)
        for pfn in self.pfn_layers:
            features = pfn(features, unq_inv)

        unq_indices = unq_indices.int()
        voxel_indices = torch.stack(
            [
                torch.div(unq_indices, self.scale_xyz, rounding_mode="floor"),
                torch.div(unq_indices % self.scale_xyz, self.scale_yz, rounding_mode="floor"),
                torch.div(unq_indices % self.scale_yz, self.scale_z, rounding_mode="floor"),
                unq_indices % self.scale_z,
            ],
            dim=1,
        )
        voxel_indices = voxel_indices[:, [0, 3, 2, 1]]
        return features, voxel_indices


def _make_sparse_block(
    in_channels: int,
    out_channels: int,
    kernel_size: Union[int, Tuple[int, ...]],
    *,
    stride: Union[int, Tuple[int, ...]] = 1,
    padding: Union[int, Tuple[int, ...]] = 0,
    indice_key: str,
    conv_type: str,
    act: Union[str, Callable, None] = "relu",
    act_kwargs: Optional[Dict[str, Any]] = None,
    norm: Union[str, Callable, None] = "batch_norm",
    norm_kwargs: Optional[Dict[str, Any]] = None,
) -> "spconv.SparseSequential":
    if conv_type == "subm":
        conv = spconv.SubMConv3d(
            in_channels,
            out_channels,
            kernel_size,
            bias=False,
            indice_key=indice_key,
        )
    elif conv_type == "spconv":
        conv = spconv.SparseConv3d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
            indice_key=indice_key,
        )
    elif conv_type == "inverseconv":
        conv = spconv.SparseInverseConv3d(
            in_channels,
            out_channels,
            kernel_size,
            bias=False,
            indice_key=indice_key,
        )
    else:
        raise ValueError(f"Unknown conv_type {conv_type!r}.")
    return spconv.SparseSequential(
        conv,
        create_norm(norm, out_channels, dim=1, **(norm_kwargs or {})),
        create_act(act, **(act_kwargs or {})),
    )


class SparseResidualBlock(_SparseModule):
    r"""Submanifold residual block (the reference's `Sparse1ConvBlock`): one $3\times3\times3$ subm conv + skip.

    Args:
        channels: Input and output channels.
        indice_key: Shared submanifold indice key.
        act: Activation type or callable.
        act_kwargs: Extra activation arguments.
        norm: Normalization type or callable.
        norm_kwargs: Extra normalization arguments.
    """

    def __init__(
        self,
        channels: int,
        *,
        indice_key: str,
        act: Union[str, Callable, None] = "relu",
        act_kwargs: Optional[Dict[str, Any]] = None,
        norm: Union[str, Callable, None] = "batch_norm",
        norm_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()
        self.conv1 = spconv.SubMConv3d(channels, channels, 3, padding=1, bias=True, indice_key=indice_key)
        self.bn1 = create_norm(norm, channels, dim=1, **(norm_kwargs or {}))
        self.act = create_act(act, **(act_kwargs or {}))

    def forward(self, x: "spconv.SparseConvTensor") -> "spconv.SparseConvTensor":
        out = self.conv1(x)
        feat = out.features
        if self.bn1 is not None:
            feat = self.bn1(feat)
        feat = feat + x.features
        if self.act is not None:
            feat = self.act(feat)
        return out.replace_feature(feat)


class DownSparse(nn.Module):
    r"""Downsampling stage (the reference's `DownSp`): optional strided sparse conv + residual subm blocks.

    Args:
        channels: Channels (constant through the stage).
        kernel_size: Kernel size of the leading strided conv.
        stride: Stride of the leading conv; if $1$ the leading conv is an identity.
        num_blocks: Number of trailing residual subm blocks.
        indice_key: Base indice key for the stage.
        act: Activation type or callable.
        act_kwargs: Extra activation arguments.
        norm: Normalization type or callable.
        norm_kwargs: Extra normalization arguments.
    """

    def __init__(
        self,
        channels: int,
        kernel_size: int,
        stride: int,
        num_blocks: int,
        *,
        indice_key: str,
        act: Union[str, Callable, None] = "relu",
        act_kwargs: Optional[Dict[str, Any]] = None,
        norm: Union[str, Callable, None] = "batch_norm",
        norm_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()
        block_kwargs: Dict[str, Any] = dict(act=act, act_kwargs=act_kwargs, norm=norm, norm_kwargs=norm_kwargs)
        blocks: List[nn.Module] = []
        if stride > 1:
            blocks.append(
                _make_sparse_block(
                    channels,
                    channels,
                    kernel_size,
                    stride=stride,
                    padding=kernel_size // 2,
                    indice_key=f"spconv_{indice_key}",
                    conv_type="spconv",
                    **block_kwargs,
                )
            )
        else:
            blocks.append(nn.Identity())
        for _ in range(num_blocks):
            blocks.append(SparseResidualBlock(channels, indice_key=indice_key, **block_kwargs))
        self.blocks = spconv.SparseSequential(*blocks)

    def forward(self, x: "spconv.SparseConvTensor") -> "spconv.SparseConvTensor":
        return self.blocks(x)


class DSB(nn.Module):
    r"""Dual-scale State Space Models block of Voxel Mamba.

    A voxel sparse tensor is encoded at a high-resolution and a downsampled scale. The
    high-resolution scale runs a **backward** (sequence-flipped) Mamba pass and the low-resolution
    scale a **forward** Mamba pass, each over the voxels serialized along a Hilbert curve (the
    group-free sequence). The low-resolution scale is fused back through an inverse (or submanifold)
    sparse conv plus both high-resolution skips.

    Args:
        d_model: Voxel feature channels.
        down_kernel_size: Kernel size per scale (high, low).
        down_stride: Stride per scale (high, low).
        num_down: Residual-block count per scale (high, low).
        indice_key: Base indice key for this block.
        downsample_lvl: Hilbert template key for the low-resolution scale.
        down_resolution: If `True`, fuse with an inverse conv; else with a subm conv.
        norm_epsilon: LayerNorm epsilon for the Mamba output norms.
        rms_norm: Use `RMSNorm` instead of `nn.LayerNorm` inside the Mamba blocks.
        fused_add_norm: Use the fused add+norm kernel inside the Mamba blocks.
        residual_in_fp32: Keep the Mamba residual stream in fp32.
        act: Activation of the sparse conv blocks.
        act_kwargs: Extra activation arguments.
        norm: Normalization of the sparse conv blocks.
        norm_kwargs: Extra normalization arguments.
    """

    def __init__(
        self,
        d_model: int,
        *,
        down_kernel_size: Sequence[int],
        down_stride: Sequence[int],
        num_down: Sequence[int],
        indice_key: str,
        downsample_lvl: str,
        down_resolution: bool,
        norm_epsilon: float,
        rms_norm: bool,
        fused_add_norm: bool,
        residual_in_fp32: bool,
        act: Union[str, Callable, None] = "relu",
        act_kwargs: Optional[Dict[str, Any]] = None,
        norm: Union[str, Callable, None] = "batch_norm",
        norm_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()
        mamba_kwargs: Dict[str, Any] = dict(
            norm_epsilon=norm_epsilon,
            rms_norm=rms_norm,
            residual_in_fp32=residual_in_fp32,
            fused_add_norm=fused_add_norm,
        )
        self.mamba_forward = _create_mamba_block(d_model, layer_idx=0, **mamba_kwargs)
        self.mamba_backward = _create_mamba_block(d_model, layer_idx=1, **mamba_kwargs)

        conv_kwargs: Dict[str, Any] = dict(act=act, act_kwargs=act_kwargs, norm=norm, norm_kwargs=norm_kwargs)
        self.encoder_high = DownSparse(
            d_model, down_kernel_size[0], down_stride[0], num_down[0], indice_key=f"{indice_key}_0", **conv_kwargs
        )
        self.encoder_low = DownSparse(
            d_model, down_kernel_size[1], down_stride[1], num_down[1], indice_key=f"{indice_key}_1", **conv_kwargs
        )
        self.decoder = _make_sparse_block(
            d_model,
            d_model,
            down_kernel_size[1],
            indice_key=f"spconv_{indice_key}_1" if down_resolution else f"{indice_key}_1",
            conv_type="inverseconv" if down_resolution else "subm",
            **conv_kwargs,
        )
        self.decoder_norm = create_norm(norm, d_model, dim=1, **(norm_kwargs or {}))

        self.downsample_lvl = downsample_lvl
        self.norm = nn.LayerNorm(d_model, eps=norm_epsilon)
        self.norm_back = nn.LayerNorm(d_model, eps=norm_epsilon)

    def forward(
        self,
        voxel_features: Tensor,
        voxel_indices: Tensor,
        batch_size: int,
        spatial_shape: List[int],
        curve_template: Dict[str, Tensor],
        hilbert_spatial_size: Dict[str, Tuple[int, int, int]],
        pos_embed: nn.Module,
        stage: int,
    ) -> Tuple[Tensor, Tensor]:
        x = spconv.SparseConvTensor(
            features=voxel_features,
            indices=voxel_indices.int(),
            spatial_shape=spatial_shape,
            batch_size=batch_size,
        )
        x_high = self.encoder_high(x)
        x_low = self.encoder_low(x_high)

        forward_high, inverse_high = hilbert_serialize(
            curve_template["curve_template_rank9"],
            x_high.indices,
            batch_size,
            hilbert_spatial_size["curve_template_rank9"],
            shift=stage,
        )
        forward_low, inverse_low = hilbert_serialize(
            curve_template[self.downsample_lvl],
            x_low.indices,
            batch_size,
            hilbert_spatial_size[self.downsample_lvl],
            shift=stage,
        )

        feats_low = x_low.features + pos_embed(_pos_embed_coords(x_low.indices, x_low.spatial_shape))
        out_low = torch.zeros_like(feats_low)
        for i in range(batch_size):
            mask = x_low.indices[:, 0] == i
            seq = feats_low[mask][forward_low[i]][None]
            out_low[mask] = self.mamba_forward(seq, None)[0].squeeze(0)[inverse_low[i]]
        x_low_mamba = x_low.replace_feature(self.norm(out_low))

        feats_high = x_high.features + pos_embed(_pos_embed_coords(x_high.indices, x_high.spatial_shape))
        out_high = torch.zeros_like(feats_high)
        for i in range(batch_size):
            mask = x_high.indices[:, 0] == i
            seq = feats_high[mask][forward_high[i]][None].flip(1)
            out_high[mask] = self.mamba_backward(seq, None)[0].squeeze(0).flip(0)[inverse_high[i]]
        x_high_mamba = x_high.replace_feature(self.norm_back(out_high))

        x = self.decoder(x_low_mamba)
        x = x.replace_feature(x.features + x_high_mamba.features + x_high.features)
        if self.decoder_norm is not None:
            x = x.replace_feature(self.decoder_norm(x.features))
        return x.features, x.indices


def _pos_embed_coords(coords: Tensor, spatial_shape: List[int]) -> Tensor:
    out = torch.zeros((coords.shape[0], 9), device=coords.device, dtype=torch.float32)
    out[:, 0] = coords[:, 1] / spatial_shape[0]
    out[:, 1:3] = torch.div(coords[:, 2:], 12, rounding_mode="floor") / (spatial_shape[1] // 12 + 1)
    out[:, 3:5] = (coords[:, 2:] % 12) / 12.0
    out[:, 5:7] = torch.div(coords[:, 2:] + 6, 12, rounding_mode="floor") / (spatial_shape[1] // 12 + 1)
    out[:, 7:9] = ((coords[:, 2:] + 6) % 12) / 12.0
    return out


def _create_mamba_block(
    d_model: int,
    *,
    layer_idx: int,
    norm_epsilon: float,
    rms_norm: bool,
    fused_add_norm: bool,
    residual_in_fp32: bool,
) -> nn.Module:
    r"""Build one Mamba `Block` (pre-norm residual + Mamba mixer), matching the reference `create_block`."""
    mixer_cls = partial(Mamba, layer_idx=layer_idx)
    norm_cls = partial(RMSNorm if rms_norm else nn.LayerNorm, eps=norm_epsilon)
    block = Block(
        d_model,
        mixer_cls,
        mlp_cls=nn.Identity,
        norm_cls=norm_cls,
        fused_add_norm=fused_add_norm,
        residual_in_fp32=residual_in_fp32,
    )
    block.layer_idx = layer_idx
    return block


class VoxelMambaBackbone(nn.Module):
    r"""Group-free Voxel Mamba sparse 3D backbone (the reference's `Voxel_Mamba_Waymo`).

    A stack of [`DSB`][torch_pointcloud.models.voxel_mamba.DSB] blocks serializes voxels along
    multi-scale Hilbert curves and applies bidirectional Mamba (state-space) blocks, with periodic
    sparse downsampling of the height axis. There is no windowing or grouping: the whole scene is one
    sequence.

    Args:
        d_model: Voxel feature channels.
        grid_size: Voxel grid extent $(n_x, n_y, n_z)$.
        num_stage: Number of `DSB` blocks per stage.
        num_down: Per-stage residual-block counts for the two `conv_encoder` scales.
        down_stride: Per-stage strides for the two `conv_encoder` scales.
        down_kernel_size: Per-stage kernel sizes for the two `conv_encoder` scales.
        down_resolution: Per-stage flag selecting inverse-conv (vs subm) fusion.
        downsample_lvl: Per-stage Hilbert template key for the low-resolution scale.
        extra_down: Block index after which the final height-compression conv runs.
        norm_epsilon: LayerNorm epsilon for the Mamba output norms.
        rms_norm: Use `RMSNorm` inside the Mamba blocks.
        fused_add_norm: Use the fused add+norm kernel inside the Mamba blocks.
        residual_in_fp32: Keep the Mamba residual stream in fp32.
    """

    _template_rank9: Tensor
    _template_rank8: Tensor
    _template_rank7: Tensor

    def __init__(
        self,
        d_model: int,
        grid_size: Sequence[int],
        *,
        num_stage: Sequence[int] = (2, 2, 2),
        num_down: Sequence[Sequence[int]] = ((0, 1), (0, 1), (0, 1)),
        down_stride: Sequence[Sequence[int]] = ((1, 1), (1, 2), (1, 4)),
        down_kernel_size: Sequence[Sequence[int]] = ((3, 3), (3, 3), (3, 5)),
        down_resolution: Sequence[bool] = (False, True, True),
        downsample_lvl: Sequence[str] = ("curve_template_rank9", "curve_template_rank8", "curve_template_rank7"),
        extra_down: int = 5,
        norm_epsilon: float = 1e-5,
        rms_norm: bool = True,
        fused_add_norm: bool = True,
        residual_in_fp32: bool = True,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_stage = list(num_stage)
        self.extra_down = extra_down
        self.sparse_shape = [int(grid_size[2]) + 1, int(grid_size[1]), int(grid_size[0])]
        conv_kwargs: Dict[str, Any] = dict(norm="batch_norm", norm_kwargs=dict(eps=1e-3, momentum=0.01))

        self.hilbert_spatial_size: Dict[str, Tuple[int, int, int]] = {}
        for rank, z_max in ((9, 41), (8, 17), (7, 9)):
            self.register_buffer(f"_template_rank{rank}", build_hilbert_template(rank, z_max), persistent=False)
            side = 1 << rank
            self.hilbert_spatial_size[f"curve_template_rank{rank}"] = (1, side, side)

        blocks: List[nn.Module] = []
        for i, count in enumerate(num_stage):
            for ns in range(count):
                blocks.append(
                    DSB(
                        d_model,
                        down_kernel_size=down_kernel_size[i],
                        down_stride=down_stride[i],
                        num_down=num_down[i],
                        indice_key=f"stem{i}_layer{ns}",
                        downsample_lvl=downsample_lvl[i],
                        down_resolution=down_resolution[i],
                        norm_epsilon=norm_epsilon,
                        rms_norm=rms_norm,
                        fused_add_norm=fused_add_norm,
                        residual_in_fp32=residual_in_fp32,
                        **conv_kwargs,
                    )
                )
        self.block_list = nn.ModuleList(blocks)

        self.down_z_list = nn.ModuleList(
            _make_sparse_block(
                d_model,
                d_model,
                (3, 1, 1),
                stride=(2, 1, 1),
                indice_key=f"downz_{i}",
                conv_type="spconv",
                **conv_kwargs,
            )
            for i in range(len(num_stage))
        )
        self.conv_out = _make_sparse_block(
            d_model,
            d_model,
            (3, 1, 1),
            stride=(2, 1, 1),
            indice_key="final_conv_out",
            conv_type="spconv",
            **conv_kwargs,
        )
        self.pos_embed = MLP([9, d_model, d_model], act="relu", norm="batch_norm", plain_last=True)
        self.out_channels = d_model

    def forward(self, voxel_features: Tensor, voxel_indices: Tensor, batch_size: int) -> Tuple[Tensor, Tensor]:
        curve_template = {
            "curve_template_rank9": self._template_rank9,
            "curve_template_rank8": self._template_rank8,
            "curve_template_rank7": self._template_rank7,
        }
        spatial_shape = self.sparse_shape
        for i, block in enumerate(self.block_list):
            voxel_features, voxel_indices = block(
                voxel_features,
                voxel_indices,
                batch_size,
                spatial_shape,
                curve_template,
                self.hilbert_spatial_size,
                self.pos_embed,
                i,
            )
            if i > 0 and i % 2 == 1:
                xd = spconv.SparseConvTensor(
                    features=voxel_features,
                    indices=voxel_indices.int(),
                    spatial_shape=spatial_shape,
                    batch_size=batch_size,
                )
                if i == self.extra_down:
                    xd = self.conv_out(xd)
                xd = self.down_z_list[i // 2](xd)
                voxel_features, voxel_indices, spatial_shape = xd.features, xd.indices, xd.spatial_shape
        return voxel_features, voxel_indices


class BasicBlock2d(nn.Module):
    r"""Residual 2D conv block (the reference's `BasicBlock`) of the BEV residual backbone.

    Args:
        in_channels: Input channels.
        out_channels: Output channels.
        stride: Stride of the first conv (and the optional projection shortcut).
        downsample: Add a $1\times1$ projection shortcut to match channels / stride.
        norm: Normalization type or callable.
        norm_kwargs: Extra normalization arguments.
        act: Activation type or callable.
        act_kwargs: Extra activation arguments.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        stride: int = 1,
        downsample: bool = False,
        norm: Union[str, Callable, None] = "batch_norm",
        norm_kwargs: Optional[Dict[str, Any]] = None,
        act: Union[str, Callable, None] = "relu",
        act_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()
        block_kwargs: Dict[str, Any] = dict(norm=norm, norm_kwargs=norm_kwargs, act=act, act_kwargs=act_kwargs)
        self.conv1 = Conv2dBlock(in_channels, out_channels, 3, stride=stride, padding=1, **block_kwargs)
        self.conv2 = Conv2dBlock(out_channels, out_channels, 3, padding=1, act=None, norm=norm, norm_kwargs=norm_kwargs)

        self.act = create_act(act, **(act_kwargs or {}))
        self.downsample = (
            Conv2dBlock(
                in_channels,
                out_channels,
                1,
                stride=stride,
                padding=0,
                act=None,
                norm=norm,
                norm_kwargs=norm_kwargs,
            )
            if downsample
            else None
        )

    def forward(self, x: Tensor) -> Tensor:
        identity = x if self.downsample is None else self.downsample(x)
        out = self.conv2(self.conv1(x))
        out = out + identity
        return out if self.act is None else self.act(out)


class BaseBEVResBackbone(nn.Module):
    r"""Residual SSD-style 2D BEV backbone used by Voxel Mamba.

    Like the anchor-detector BEV backbone but each level is a stack of residual
    [`BasicBlock2d`][torch_pointcloud.models.voxel_mamba.BasicBlock2d]s; level outputs are upsampled
    and concatenated.

    Args:
        input_channels: Channels of the input BEV feature map.
        layer_nums: Residual blocks (beyond the strided one) per level.
        layer_strides: Downsample stride of the leading block, per level.
        num_filters: Channel width per level.
        upsample_strides: Upsample factor per level.
        num_upsample_filters: Channels of each upsampled level.
        norm: Normalization type or callable.
        norm_kwargs: Extra normalization arguments.
        act: Activation type or callable.
        act_kwargs: Extra activation arguments.
    """

    def __init__(
        self,
        input_channels: int,
        layer_nums: Sequence[int],
        layer_strides: Sequence[int],
        num_filters: Sequence[int],
        upsample_strides: Sequence[float],
        num_upsample_filters: Sequence[int],
        *,
        norm: Union[str, Callable, None] = "batch_norm",
        norm_kwargs: Optional[Dict[str, Any]] = None,
        act: Union[str, Callable, None] = "relu",
        act_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()
        block_kwargs: Dict[str, Any] = dict(norm=norm, norm_kwargs=norm_kwargs, act=act, act_kwargs=act_kwargs)
        num_levels = len(layer_nums)
        c_in_list = [input_channels, *num_filters[:-1]]
        self.blocks = nn.ModuleList()
        self.deblocks = nn.ModuleList()
        for idx in range(num_levels):
            cur: List[nn.Module] = [
                BasicBlock2d(
                    c_in_list[idx],
                    num_filters[idx],
                    stride=layer_strides[idx],
                    downsample=True,
                    **block_kwargs,
                )
            ]
            for _ in range(layer_nums[idx]):
                cur.append(BasicBlock2d(num_filters[idx], num_filters[idx], **block_kwargs))
            self.blocks.append(nn.Sequential(*cur))
            stride = upsample_strides[idx]
            if stride >= 1:
                self.deblocks.append(
                    Conv2dBlock(
                        num_filters[idx],
                        num_upsample_filters[idx],
                        int(stride),
                        stride=int(stride),
                        padding=0,
                        transposed=True,
                        **block_kwargs,
                    )
                )
            else:
                down = int(round(1 / stride))
                self.deblocks.append(
                    Conv2dBlock(
                        num_filters[idx], num_upsample_filters[idx], down, stride=down, padding=0, **block_kwargs
                    )
                )
        self.num_bev_features = sum(num_upsample_filters)

    def forward(self, spatial_features: Tensor) -> Tensor:
        ups = []
        x = spatial_features
        for block, deblock in zip(self.blocks, self.deblocks):
            x = block(x)
            ups.append(deblock(x))
        return torch.cat(ups, dim=1) if len(ups) > 1 else ups[0]


def _head_branch(in_channels: int, out_channels: int, num_layers: int, block_kwargs: Dict[str, Any]) -> nn.Sequential:
    hidden = [Conv2dBlock(in_channels, in_channels, 3, padding=1, **block_kwargs) for _ in range(num_layers - 1)]
    return nn.Sequential(*hidden, nn.Conv2d(in_channels, out_channels, 3, stride=1, padding=1, bias=True))


class SeparateHead(nn.Module):
    r"""Per-attribute conv head of the center head.

    One small stack of $3\times3$ convs per box attribute, applied to the shared BEV features:
    `center` $(2)$, `center_z` $(1)$, `dim` $(3)$, `rot` $(2)$, `iou` $(1)$ and a class `heatmap`.
    The branch widths are fixed by the box parametrization; only the number of classes and the conv
    depth are configurable.

    Args:
        in_channels: Input channels.
        num_classes: Number of classes predicted by the heatmap branch.
        num_layers: Number of convs per branch.
        act: Activation of the hidden conv blocks.
        act_kwargs: Extra activation arguments.
        norm: Normalization of the hidden conv blocks.
        norm_kwargs: Extra normalization arguments.
        bias: Whether the hidden convs carry a bias.

    Shape:
        - Input: $(B, C_\text{in}, H, W)$.
        - Output: dict of $(B, C_\text{attr}, H, W)$ tensors keyed by attribute.
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        num_layers: int = 2,
        act: Union[str, Callable, None] = "relu",
        act_kwargs: Optional[Dict[str, Any]] = None,
        norm: Union[str, Callable, None] = "batch_norm",
        norm_kwargs: Optional[Dict[str, Any]] = None,
        bias: bool = False,
    ) -> None:
        super().__init__()
        block_kwargs: Dict[str, Any] = dict(
            act=act, act_kwargs=act_kwargs, norm=norm, norm_kwargs=norm_kwargs, bias=bias
        )
        self.center = _head_branch(in_channels, 2, num_layers, block_kwargs)
        self.center_z = _head_branch(in_channels, 1, num_layers, block_kwargs)
        self.dim = _head_branch(in_channels, 3, num_layers, block_kwargs)
        self.rot = _head_branch(in_channels, 2, num_layers, block_kwargs)
        self.iou = _head_branch(in_channels, 1, num_layers, block_kwargs)
        self.heatmap = _head_branch(in_channels, num_classes, num_layers, block_kwargs)

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        return {
            "center": self.center(x),
            "center_z": self.center_z(x),
            "dim": self.dim(x),
            "rot": self.rot(x),
            "iou": self.iou(x),
            "heatmap": self.heatmap(x),
        }


class CenterHead(nn.Module):
    r"""Center-based detection head producing per-pixel heatmaps and box regressions.

    A shared $3\times3$ conv reduces the BEV features, then a
    [`SeparateHead`][torch_pointcloud.models.voxel_mamba.SeparateHead] regresses the per-attribute
    maps (one shared head over all classes here).

    Reference implementation: :github:
    [open-mmlab/OpenPCDet](https://github.com/open-mmlab/OpenPCDet) (`CenterHead`).

    Args:
        in_channels: Channels of the input BEV feature map.
        num_classes: Number of foreground classes.
        shared_conv_channels: Channels of the shared conv before the separate heads.
        num_head_layers: Number of convs per separate-head branch.
        bn_eps: BatchNorm epsilon.
        bn_momentum: BatchNorm momentum.
        bias: Whether convs preceding a norm carry a bias.
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        *,
        shared_conv_channels: int = 64,
        num_head_layers: int = 2,
        bn_eps: float = 1e-3,
        bn_momentum: float = 0.01,
        bias: bool = False,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        norm_kwargs = {"eps": bn_eps, "momentum": bn_momentum}
        self.shared_conv = Conv2dBlock(
            in_channels,
            shared_conv_channels,
            3,
            padding=1,
            bias=bias,
            act="relu",
            norm="batch_norm",
            norm_kwargs=norm_kwargs,
        )
        self.prediction_head = SeparateHead(
            shared_conv_channels,
            num_classes,
            num_layers=num_head_layers,
            norm="batch_norm",
            norm_kwargs=norm_kwargs,
            bias=bias,
        )

    def forward(self, spatial_features_2d: Tensor) -> Dict[str, Tensor]:
        return self.prediction_head(self.shared_conv(spatial_features_2d))


def _topk(scores: Tensor, k: int) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    b, c, h, w = scores.shape
    topk_scores, topk_inds = torch.topk(scores.flatten(2, 3), k)
    topk_inds = topk_inds % (h * w)
    topk_ys = torch.div(topk_inds, w, rounding_mode="floor").float()
    topk_xs = (topk_inds % w).int().float()
    topk_score, topk_ind = torch.topk(topk_scores.view(b, -1), k)
    topk_classes = torch.div(topk_ind, k, rounding_mode="floor")
    topk_inds = _gather_feat(topk_inds.view(b, -1, 1), topk_ind).view(b, k)
    topk_ys = _gather_feat(topk_ys.view(b, -1, 1), topk_ind).view(b, k)
    topk_xs = _gather_feat(topk_xs.view(b, -1, 1), topk_ind).view(b, k)
    return topk_score, topk_inds, topk_classes, torch.stack([topk_xs, topk_ys], dim=-1)


def _gather_feat(feat: Tensor, ind: Tensor) -> Tensor:
    dim = feat.size(2)
    ind = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
    return feat.gather(1, ind)


class VoxelMambaDetection(DetectionModel):
    r"""Voxel Mamba: group-free state-space 3D object detector (packed point format).

    Reference: :arxiv:
    [Zhang et al., 2024](https://arxiv.org/abs/2406.10700). Reference implementation: :github:
    [gwenzhang/Voxel-Mamba](https://github.com/gwenzhang/Voxel-Mamba) (built on OpenPCDet + DSVT).

    Voxels are serialized into a single Hilbert-curve sequence and processed by bidirectional Mamba
    (state-space) blocks (no windowing / grouping), then scattered to a BEV map, refined by a 2D
    residual backbone, and decoded by a center-based head.

    Args:
        in_channels: Raw point feature channels including xyz (e.g. $5$ for Waymo).
        num_classes: Number of foreground classes.
        voxel_size: Voxel size $(v_x, v_y, v_z)$.
        point_cloud_range: Range $(x_\min, y_\min, z_\min, x_\max, y_\max, z_\max)$.
        d_model: Voxel feature channels of the Mamba backbone.
        vfe_num_filters: PFN widths of the dynamic mean VFE.
        layer_nums: 2D backbone residual-block counts per level.
        layer_strides: 2D backbone downsample strides per level.
        num_filters: 2D backbone channel widths per level.
        upsample_strides: 2D backbone upsample factors per level.
        num_upsample_filters: 2D backbone upsample channels per level.
        shared_conv_channels: Channels of the head's shared conv.
        rms_norm: Use `RMSNorm` inside the Mamba blocks.
        fused_add_norm: Use the fused add+norm kernel inside the Mamba blocks.
        norm_epsilon: LayerNorm epsilon for the Mamba output norms.
    """

    def __init__(
        self,
        in_channels: int = 5,
        num_classes: int = 3,
        *,
        voxel_size: Sequence[float] = (0.32, 0.32, 0.1875),
        point_cloud_range: Sequence[float] = (-74.88, -74.88, -2.0, 74.88, 74.88, 4.0),
        d_model: int = 128,
        vfe_num_filters: Sequence[int] = (128, 128),
        layer_nums: Sequence[int] = (1, 2, 2),
        layer_strides: Sequence[int] = (1, 2, 2),
        num_filters: Sequence[int] = (128, 128, 256),
        upsample_strides: Sequence[float] = (1, 2, 4),
        num_upsample_filters: Sequence[int] = (128, 128, 128),
        shared_conv_channels: int = 64,
        rms_norm: bool = True,
        fused_add_norm: bool = True,
        norm_epsilon: float = 1e-5,
    ) -> None:
        super().__init__(in_channels=in_channels, num_classes=num_classes)
        self.voxel_size = tuple(voxel_size)
        self.point_cloud_range = tuple(point_cloud_range)
        grid = [int(round((point_cloud_range[i + 3] - point_cloud_range[i]) / voxel_size[i])) for i in range(3)]
        self.grid_size: Tuple[int, int, int] = (grid[0], grid[1], grid[2])
        self.feature_map_stride = 1

        self.vfe = DynamicMeanVFE(in_channels, vfe_num_filters, voxel_size, point_cloud_range, self.grid_size)
        self.backbone_3d = VoxelMambaBackbone(
            d_model,
            self.grid_size,
            rms_norm=rms_norm,
            fused_add_norm=fused_add_norm,
            norm_epsilon=norm_epsilon,
        )
        self.bev_channels = d_model
        self.backbone = BaseBEVResBackbone(
            self.bev_channels,
            layer_nums,
            layer_strides,
            num_filters,
            upsample_strides,
            num_upsample_filters,
            norm_kwargs={"eps": 1e-3, "momentum": 0.01},
        )
        self.head = CenterHead(
            self.backbone.num_bev_features,
            num_classes,
            shared_conv_channels=shared_conv_channels,
        )

    def forward_features(self, x: OptTensor, pos: Tensor, batch: Tensor) -> Tensor:
        batch_size = int(batch.max().item()) + 1
        voxel_features, voxel_indices = self.vfe(pos, x, batch)
        voxel_features, voxel_indices = self.backbone_3d(voxel_features, voxel_indices, batch_size)
        bev = self._scatter_bev(voxel_features, voxel_indices, batch_size)
        return self.backbone(bev)

    def _scatter_bev(self, features: Tensor, coords: Tensor, batch_size: int) -> Tensor:
        nx, ny = self.grid_size[0], self.grid_size[1]
        bev = features.new_zeros((batch_size, self.bev_channels, ny * nx))
        flat = (coords[:, 2] * nx + coords[:, 3]).long()
        for b in range(batch_size):
            mask = coords[:, 0] == b
            bev[b, :, flat[mask]] = features[mask].t()
        return bev.view(batch_size, self.bev_channels, ny, nx)

    def forward(self, x: OptTensor, pos: Tensor, batch: Tensor) -> Dict[str, Tensor]:
        return self.head(self.forward_features(x, pos, batch))

    @torch.no_grad()
    def decode(
        self,
        out: Dict[str, Tensor],
        *,
        score_threshold: float = 0.1,
        nms_iou: float = 0.7,
        k: int = 500,
        iou_rectifier: Sequence[float] = (0.68, 0.71, 0.65),
    ) -> Detection3D:
        r"""Decode center-head predictions into packed detections.

        Peaks of the (sigmoid) heatmap give candidate centers; box attributes are gathered at those
        peaks, mapped to world coordinates, thresholded by score, rescored by the predicted IoU
        ($s^{1 - r_c} \cdot \text{iou}^{r_c}$ with a per-class rectifier $r_c$, as in the reference),
        and reduced by per-class 3D NMS.

        Args:
            out: The dict returned by `forward`.
            score_threshold: Minimum (pre-rectification) heatmap score to keep a box.
            nms_iou: IoU threshold of the per-class 3D NMS.
            k: Number of heatmap peaks gathered per scene.
            iou_rectifier: Per-class IoU-rectification exponent (the reference Waymo values).

        Returns:
            Packed detections `{"boxes": (K, 7), "scores": (K,), "labels": (K,), "batch": (K,)}` (PyG layout).
        """
        heatmap = out["heatmap"].sigmoid()
        batch_size = heatmap.shape[0]
        scores, inds, classes, centers = _topk(heatmap, k)

        center = _transpose_gather(out["center"], inds)
        center_z = _transpose_gather(out["center_z"], inds)
        dim = _transpose_gather(out["dim"], inds).exp()
        rot = _transpose_gather(out["rot"], inds)
        angle = torch.atan2(rot[..., 1], rot[..., 0])
        iou = torch.clamp((_transpose_gather(out["iou"], inds) + 1) * 0.5, min=0, max=1.0).squeeze(-1)

        xs = (centers[..., 0:1] + center[..., 0:1]) * self.feature_map_stride * self.voxel_size[
            0
        ] + self.point_cloud_range[0]
        ys = (centers[..., 1:2] + center[..., 1:2]) * self.feature_map_stride * self.voxel_size[
            1
        ] + self.point_cloud_range[1]
        boxes = torch.cat([xs, ys, center_z, dim, angle.unsqueeze(-1)], dim=-1)

        out_boxes, out_scores, out_labels, out_batch = [], [], [], []
        for b in range(batch_size):
            keep = scores[b] > score_threshold
            scene_boxes, scene_scores, scene_labels = boxes[b][keep], scores[b][keep], classes[b][keep]
            rectifier = scene_scores.new_tensor(iou_rectifier)[scene_labels.long()]
            scene_scores = scene_scores.pow(1 - rectifier) * iou[b][keep].pow(rectifier)
            idx = nms3d(scene_boxes, scene_scores, scene_labels, nms_iou)
            out_boxes.append(scene_boxes[idx])
            out_scores.append(scene_scores[idx])
            out_labels.append(scene_labels[idx])
            out_batch.append(torch.full((idx.numel(),), b, dtype=torch.long, device=heatmap.device))

        return {
            "boxes": torch.cat(out_boxes),
            "scores": torch.cat(out_scores),
            "labels": torch.cat(out_labels),
            "batch": torch.cat(out_batch),
        }


def _transpose_gather(feat: Tensor, ind: Tensor) -> Tensor:
    feat = feat.permute(0, 2, 3, 1).contiguous().view(feat.size(0), -1, feat.size(1))
    return _gather_feat(feat, ind)


@register_model(
    "voxel-mamba-gwenzhang.waymo",
    task="detection",
    # No public trained weights for Voxel Mamba: the Waymo checkpoint is license-gated and the nuScenes
    # model was never released, so the architecture is registered without pretrained weights.
    weights=None,
    transforms=T.Compose([T.Cat(keys=[DataKeys.INTENSITY, "elongation"], dst_key=DataKeys.X, dim=1)]),
    hparams=dict(
        in_channels=5,
        num_classes=3,
        voxel_size=(0.32, 0.32, 0.1875),
        point_cloud_range=(-74.88, -74.88, -2.0, 74.88, 74.88, 4.0),
        d_model=128,
        vfe_num_filters=(128, 128),
        layer_nums=(1, 2, 2),
        layer_strides=(1, 2, 2),
        num_filters=(128, 128, 256),
        upsample_strides=(1, 2, 4),
        num_upsample_filters=(128, 128, 128),
        shared_conv_channels=64,
    ),
)
def voxel_mamba_waymo(**hparams: Any) -> VoxelMambaDetection:
    return VoxelMambaDetection(**hparams)
