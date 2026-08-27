"""Voxel-Mamba detection model.

{{ paper("2406.10700") }}
"""

from functools import partial
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Sequence, Tuple, TypedDict, Union

import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.nn import MLP

import torch_pointcloud.transforms as T
from torch_pointcloud.layers import SparseConvBlock, SubMConv3dResidualBlock
from torch_pointcloud.layers.anchors import separate_branch
from torch_pointcloud.layers.bev_backbone import BaseBEVResBackbone
from torch_pointcloud.layers.conv2d_blocks import Conv2dBlock
from torch_pointcloud.layers.norms import create_norm
from torch_pointcloud.layers.vfe import DynamicMeanVFE
from torch_pointcloud.utils.data import DataKeys
from torch_pointcloud.utils.heatmap import transpose_gather
from torch_pointcloud.utils.hilbert import encode as hilbert_encode
from torch_pointcloud.utils.imports import _MAMBA_SSM_GITHUB_URL, _SPCONV_GITHUB_URL, optional_import
from torch_pointcloud.utils.types import Detection3D, OptTensor

from ._base import DetectionModel
from ._registry import register_model

if TYPE_CHECKING:
    import spconv.pytorch as spconv
    from mamba_ssm.modules.block import Block

spconv, _ = optional_import("spconv.pytorch", url=_SPCONV_GITHUB_URL)
Block, _ = optional_import("mamba_ssm.modules.block", "Block", url=_MAMBA_SSM_GITHUB_URL)
Mamba, _ = optional_import("mamba_ssm", "Mamba", url=_MAMBA_SSM_GITHUB_URL)
RMSNorm, _ = optional_import("mamba_ssm.ops.triton.layer_norm", "RMSNorm", url=_MAMBA_SSM_GITHUB_URL)


def build_hilbert_template(rank: int, z_max: int, device: Union[str, torch.device] = "cpu") -> Tensor:
    r"""Build the flat Hilbert-curve lookup table used to serialize voxels (the reference's `curve_template`).

    A cube of side $N = 2^\text{rank}$ is enumerated in $(z, y, x)$ order and each voxel is mapped to
    its Hilbert-curve position, then the table is truncated to $N \cdot N \cdot z_\max$ entries (the
    curve only needs to cover voxels up to $z_\max$ in the height axis). The table is indexed by the
    flat coordinate $z \cdot N \cdot N + y \cdot N + x$ to read a voxel's position along the curve.

    This reproduces the reference template (`tools/hilbert_curves/create_hilbert_curve_template.py`)
    bit-exactly via `hilbert.encode`, avoiding a 260 MB
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
        ```pycon
        >>> template = build_hilbert_template(rank=7, z_max=9)
        >>> template.shape
        torch.Size([147456])

        ```
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
    voxel_indices: Tensor,
    batch_size: int,
    rank: int,
    shift: int,
) -> Tuple[List[Tensor], List[Tensor]]:
    r"""Per-scene voxel orderings along the Hilbert curve (the reference's `get_hilbert_index_3d_mamba_lite`).

    Each voxel's flat coordinate (after a constant `shift` on every axis) indexes `template` to read
    its Hilbert position; sorting those positions within a scene yields the forward ordering, and
    sorting the forward ordering yields the inverse that scatters Mamba outputs back to voxel order.

    Args:
        template: Flat Hilbert lookup table from `build_hilbert_template`, shape $(\cdot,)$.
        voxel_indices: Voxel coordinates $(N, 4)$ as $(\text{batch}, z, y, x)$.
        batch_size: Number of scenes $B$ in the batch.
        rank: Rank the template was built with; the curve grid side is $2^\text{rank}$.
        shift: Constant offset added to $z$, $y$ and $x$ before indexing the table.

    Returns:
        `(forward, inverse)`, each a length-$B$ list of `long` index tensors.

    Shape:
        - voxel_indices: $(N, 4)$
    """
    side = 1 << rank
    x = voxel_indices[:, 3] + shift
    y = voxel_indices[:, 2] + shift
    z = voxel_indices[:, 1] + shift
    flat = (z * side * side + y * side + x).long()
    hilbert_inds = template[flat].long()

    forward: List[Tensor] = []
    inverse: List[Tensor] = []
    for i in range(batch_size):
        mask = voxel_indices[:, 0] == i
        order = torch.argsort(hilbert_inds[mask])
        forward.append(order)
        inverse.append(torch.argsort(order))
    return forward, inverse


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
                SparseConvBlock(
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
            blocks.append(SubMConv3dResidualBlock(channels, indice_key=indice_key, **block_kwargs))
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
        downsample_rank: Hilbert template rank for the low-resolution scale.
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
        downsample_rank: int,
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
        self.decoder = SparseConvBlock(
            d_model,
            d_model,
            down_kernel_size[1],
            indice_key=f"spconv_{indice_key}_1" if down_resolution else f"{indice_key}_1",
            conv_type="inverseconv" if down_resolution else "subm",
            **conv_kwargs,
        )
        self.decoder_norm = create_norm(norm, d_model, dim=1, **(norm_kwargs or {}))

        self.downsample_rank = downsample_rank
        self.norm = nn.LayerNorm(d_model, eps=norm_epsilon)
        self.norm_back = nn.LayerNorm(d_model, eps=norm_epsilon)

    def forward(
        self,
        voxel_features: Tensor,
        voxel_indices: Tensor,
        batch_size: int,
        spatial_shape: List[int],
        templates: Dict[int, Tensor],
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

        forward_high, inverse_high = hilbert_serialize(templates[9], x_high.indices, batch_size, rank=9, shift=stage)
        forward_low, inverse_low = hilbert_serialize(
            templates[self.downsample_rank],
            x_low.indices,
            batch_size,
            rank=self.downsample_rank,
            shift=stage,
        )

        feats_low = x_low.features + pos_embed(_pos_embed_input(x_low.indices, x_low.spatial_shape))
        out_low = torch.zeros_like(feats_low)
        for i in range(batch_size):
            mask = x_low.indices[:, 0] == i
            seq = feats_low[mask][forward_low[i]][None]
            out_low[mask] = self.mamba_forward(seq, None)[0].squeeze(0)[inverse_low[i]]
        x_low_mamba = x_low.replace_feature(self.norm(out_low))

        feats_high = x_high.features + pos_embed(_pos_embed_input(x_high.indices, x_high.spatial_shape))
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


def _pos_embed_input(voxel_indices: Tensor, spatial_shape: List[int]) -> Tensor:
    out = torch.zeros((voxel_indices.shape[0], 9), device=voxel_indices.device, dtype=torch.float32)
    out[:, 0] = voxel_indices[:, 1] / spatial_shape[0]
    out[:, 1:3] = torch.div(voxel_indices[:, 2:], 12, rounding_mode="floor") / (spatial_shape[1] // 12 + 1)
    out[:, 3:5] = (voxel_indices[:, 2:] % 12) / 12.0
    out[:, 5:7] = torch.div(voxel_indices[:, 2:] + 6, 12, rounding_mode="floor") / (spatial_shape[1] // 12 + 1)
    out[:, 7:9] = ((voxel_indices[:, 2:] + 6) % 12) / 12.0
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
        downsample_rank: Per-stage Hilbert template rank for the low-resolution scale.
        extra_down: Block index after which the final height-compression conv runs.
        norm_epsilon: LayerNorm epsilon for the Mamba output norms.
        rms_norm: Use `RMSNorm` inside the Mamba blocks.
        fused_add_norm: Use the fused add+norm kernel inside the Mamba blocks.
        residual_in_fp32: Keep the Mamba residual stream in fp32.
    """

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
        downsample_rank: Sequence[int] = (9, 8, 7),
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

        for rank, z_max in ((9, 41), (8, 17), (7, 9)):
            self.register_buffer(f"_template_rank{rank}", build_hilbert_template(rank, z_max), persistent=False)

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
                        downsample_rank=downsample_rank[i],
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
            SparseConvBlock(
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
        self.conv_out = SparseConvBlock(
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
        templates = {rank: self.get_buffer(f"_template_rank{rank}") for rank in (9, 8, 7)}
        spatial_shape = self.sparse_shape
        for i, block in enumerate(self.block_list):
            voxel_features, voxel_indices = block(
                voxel_features,
                voxel_indices,
                batch_size,
                spatial_shape,
                templates,
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


class CenterHeadOutput(TypedDict):
    r"""Raw dense center-head maps over the BEV feature grid.

    Attributes:
        center: Sub-cell BEV center offset, shape $(B, 2, H, W)$.
        center_z: Absolute box height, shape $(B, 1, H, W)$.
        dim: Log box size, shape $(B, 3, H, W)$.
        rot: $(\cos\theta, \sin\theta)$, shape $(B, 2, H, W)$.
        iou: IoU-rectification prediction in $[-1, 1]$, shape $(B, 1, H, W)$.
        heatmap: Per-class center logits, shape $(B, C, H, W)$.
    """

    center: Tensor
    center_z: Tensor
    dim: Tensor
    rot: Tensor
    iou: Tensor
    heatmap: Tensor


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
        branch_kwargs: Dict[str, Any] = dict(
            num_middle_conv=num_layers - 1,
            num_middle_filter=in_channels,
            act=act,
            act_kwargs=act_kwargs,
            norm=norm,
            norm_kwargs=norm_kwargs,
            bias=bias,
        )
        self.center = separate_branch(in_channels, 2, **branch_kwargs)
        self.center_z = separate_branch(in_channels, 1, **branch_kwargs)
        self.dim = separate_branch(in_channels, 3, **branch_kwargs)
        self.rot = separate_branch(in_channels, 2, **branch_kwargs)
        self.iou = separate_branch(in_channels, 1, **branch_kwargs)
        self.heatmap = separate_branch(in_channels, num_classes, **branch_kwargs)

    def forward(self, x: Tensor) -> CenterHeadOutput:
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

    def forward(self, spatial_features_2d: Tensor) -> CenterHeadOutput:
        return self.prediction_head(self.shared_conv(spatial_features_2d))


class VoxelMambaDetection(DetectionModel):
    r"""Voxel Mamba: group-free state-space 3D object detector (packed point format).

    Reference: :arxiv:
    [Zhang et al., 2024](https://arxiv.org/abs/2406.10700). Reference implementation: :github:
    [gwenzhang/Voxel-Mamba](https://github.com/gwenzhang/Voxel-Mamba) (built on DSVT).

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
        self.d_model = d_model
        self.vfe_num_filters = vfe_num_filters
        self.layer_nums = layer_nums
        self.layer_strides = layer_strides
        self.num_filters = num_filters
        self.upsample_strides = upsample_strides
        self.num_upsample_filters = num_upsample_filters
        self.shared_conv_channels = shared_conv_channels
        self.rms_norm = rms_norm
        self.fused_add_norm = fused_add_norm
        self.norm_epsilon = norm_epsilon

        self.vfe = self.configure_vfe()
        self.backbone_3d = self.configure_backbone_3d()
        self.bev_channels = d_model
        self.backbone = self.configure_backbone()
        self.head = self.configure_head()

    def configure_vfe(self) -> DynamicMeanVFE:
        """Build the dynamic mean voxel feature encoder."""
        return DynamicMeanVFE(
            self.in_channels, self.vfe_num_filters, self.voxel_size, self.point_cloud_range, self.grid_size
        )

    def configure_backbone_3d(self) -> VoxelMambaBackbone:
        """Build the Hilbert-serialized Mamba voxel backbone."""
        return VoxelMambaBackbone(
            self.d_model,
            self.grid_size,
            rms_norm=self.rms_norm,
            fused_add_norm=self.fused_add_norm,
            norm_epsilon=self.norm_epsilon,
        )

    def configure_backbone(self) -> BaseBEVResBackbone:
        """Build the residual 2D BEV backbone."""
        return BaseBEVResBackbone(
            self.bev_channels,
            self.layer_nums,
            self.layer_strides,
            self.num_filters,
            self.upsample_strides,
            self.num_upsample_filters,
            norm_kwargs={"eps": 1e-3, "momentum": 0.01},
        )

    def configure_head(self) -> CenterHead:
        """Build the center-based detection head."""
        return CenterHead(
            self.backbone.num_bev_features,
            self.num_classes,
            shared_conv_channels=self.shared_conv_channels,
        )

    def forward_features(self, x: OptTensor, pos: Tensor, batch: Tensor) -> Tensor:
        batch_size = int(batch.max().item()) + 1
        voxel_features, voxel_indices = self.vfe(pos, x, batch)
        voxel_features, voxel_indices = self.backbone_3d(voxel_features, voxel_indices, batch_size)
        bev = self._scatter_bev(voxel_features, voxel_indices, batch_size)
        return self.backbone(bev)

    def _scatter_bev(self, x: Tensor, voxel_indices: Tensor, batch_size: int) -> Tensor:
        nx, ny = self.grid_size[0], self.grid_size[1]
        bev = x.new_zeros((batch_size, self.bev_channels, ny * nx))
        flat = (voxel_indices[:, 2] * nx + voxel_indices[:, 3]).long()
        for b in range(batch_size):
            mask = voxel_indices[:, 0] == b
            bev[b, :, flat[mask]] = x[mask].t()
        return bev.view(batch_size, self.bev_channels, ny, nx)

    def forward(self, x: OptTensor, pos: Tensor, batch: Tensor) -> CenterHeadOutput:
        return self.head(self.forward_features(x, pos, batch))

    @torch.no_grad()
    def decode(
        self,
        out: CenterHeadOutput,
        *,
        score_threshold: float = 0.0,
        top_k: int = 500,
        iou_rectifier: Sequence[float] = (0.68, 0.71, 0.65),
    ) -> Detection3D:
        r"""Decode center-head predictions into raw candidate detections (no NMS).

        Peaks of the (sigmoid) heatmap give candidate centers; box attributes are gathered at those
        peaks, mapped to world coordinates, and rescored by the predicted IoU
        ($s^{1 - r_c} \cdot \text{iou}^{r_c}$ with a per-class rectifier $r_c$, as in the reference). The
        full candidate set is returned; the evaluation pipeline applies score thresholding and per-class
        3D NMS via the `torch_pointcloud.utils.box3d` utilities.

        Args:
            out: A `CenterHeadOutput` from `forward`.
            score_threshold: Minimum (pre-rectification) heatmap score to keep a peak; the non-filtering
                $0$ default returns every peak (the reference protocol filters at $0.1$).
            top_k: Number of heatmap peaks gathered per scene.
            iou_rectifier: Per-class IoU-rectification exponent, one entry per class (the default holds
                the reference Waymo 3-class values).

        Returns:
            Packed candidate detections `{"boxes": (K, 7), "scores": (K,), "labels": (K,), "batch": (K,)}`
            (PyG layout).
        """
        if len(iou_rectifier) != self.num_classes:
            raise ValueError(
                f"`iou_rectifier` must have one entry per class ({self.num_classes}), got {len(iou_rectifier)}."
            )
        heatmap = out["heatmap"].sigmoid()
        batch_size, _, h, w = heatmap.shape
        # One global top-k over the flat heatmap equals the reference's per-class-then-global two-stage top-k.
        scores, inds = torch.topk(heatmap.flatten(1), top_k)
        classes = torch.div(inds, h * w, rounding_mode="floor")
        inds = inds % (h * w)
        xs = (inds % w).float()
        ys = torch.div(inds, w, rounding_mode="floor").float()

        center = transpose_gather(out["center"], inds)
        center_z = transpose_gather(out["center_z"], inds)
        dim = transpose_gather(out["dim"], inds).exp()
        rot = transpose_gather(out["rot"], inds)
        angle = torch.atan2(rot[..., 1], rot[..., 0])
        iou = torch.clamp((transpose_gather(out["iou"], inds) + 1) * 0.5, min=0, max=1.0).squeeze(-1)

        xs = (xs + center[..., 0]) * self.feature_map_stride * self.voxel_size[0] + self.point_cloud_range[0]
        ys = (ys + center[..., 1]) * self.feature_map_stride * self.voxel_size[1] + self.point_cloud_range[1]
        boxes = torch.cat([xs.unsqueeze(-1), ys.unsqueeze(-1), center_z, dim, angle.unsqueeze(-1)], dim=-1)

        out_boxes, out_scores, out_labels, out_batch = [], [], [], []
        for b in range(batch_size):
            keep = scores[b] > score_threshold
            scene_boxes, scene_scores, scene_labels = boxes[b][keep], scores[b][keep], classes[b][keep]
            rectifier = scene_scores.new_tensor(iou_rectifier)[scene_labels.long()]
            out_boxes.append(scene_boxes)
            out_scores.append(scene_scores.pow(1 - rectifier) * iou[b][keep].pow(rectifier))
            out_labels.append(scene_labels)
            out_batch.append(torch.full((scene_boxes.shape[0],), b, dtype=torch.long, device=heatmap.device))

        return {
            "boxes": torch.cat(out_boxes),
            "scores": torch.cat(out_scores),
            "labels": torch.cat(out_labels),
            "batch": torch.cat(out_batch),
        }


@register_model(
    "voxel-mamba.waymo",
    task="detection",
    # No public trained weights for Voxel Mamba: the Waymo checkpoint is license-gated and the nuScenes
    # model was never released, so the architecture is registered without pretrained weights.
    weights=None,
    transform=T.Compose([T.Cat(keys=[DataKeys.INTENSITY, "elongation"], dst_key=DataKeys.X, dim=1)]),
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
