"""LION detection model.

{{ paper("2407.18232") }}
"""

import math
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn import MLP
from torch_geometric.utils import scatter

import torch_pointcloud.transforms as T
from torch_pointcloud.datasets.nuscenes import NUSCENES_DETECTION_CLASSES
from torch_pointcloud.layers.bev_backbone import BEVResidualBackbone
from torch_pointcloud.layers.transfusion_head import TransFusionHead, TransFusionHeadOutput
from torch_pointcloud.layers.vfe import DynamicMeanVFE
from torch_pointcloud.utils.data import DataKeys
from torch_pointcloud.utils.imports import (
    _MAMBA_SSM_GITHUB_URL,
    _SPCONV_GITHUB_URL,
    optional_import,
)
from torch_pointcloud.utils.types import Detection3D, OptTensor

from ._base import DetectionModel
from ._registry import WeightsDict, register_model

if TYPE_CHECKING:
    import spconv.pytorch as spconv
    from mamba_ssm.ops.selective_scan_interface import mamba_inner_fn

spconv, _IS_SPCONV_AVAILABLE = optional_import("spconv.pytorch", url=_SPCONV_GITHUB_URL)
mamba_inner_fn, _ = optional_import(
    "mamba_ssm.ops.selective_scan_interface", "mamba_inner_fn", url=_MAMBA_SSM_GITHUB_URL
)


class BiMamba(nn.Module):
    r"""Bidirectional Mamba mixer used by the LION linear group RNN (`Mamba` in LION's vendored ops).

    The input window-group sequence is projected to the Mamba inner width and scanned both forward and
    (sequence-flipped) backward by two independent selective-scan branches with separate input convs,
    $\Delta$ projections and state matrices ($A$, $A_b$, $D$, $D_b$); the two scans are summed (after
    un-flipping the backward branch) and projected back to $d_\text{model}$. This is the operator the
    paper denotes as the linear RNN over each spatially grouped window.

    Args:
        d_model: Feature channels of the mixer input/output.
        d_state: SSM state width $N$.
        d_conv: Depthwise causal-conv kernel width.
        expand: Inner-width expansion factor; $d_\text{inner} = \text{expand} \cdot d_\text{model}$.

    Shape:
        - Input: $(B, L, d_\text{model})$
        - Output: $(B, L, d_\text{model})$
    """

    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(expand * d_model)
        self.dt_rank = math.ceil(d_model / 16)

        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        self.conv1d = nn.Conv1d(
            self.d_inner, self.d_inner, kernel_size=d_conv, groups=self.d_inner, padding=d_conv - 1, bias=True
        )
        self.act = nn.SiLU()
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + d_state * 2, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)
        self.A_log = nn.Parameter(torch.zeros(self.d_inner, d_state))
        self.D = nn.Parameter(torch.ones(self.d_inner))

        self.conv1d_b = nn.Conv1d(
            self.d_inner, self.d_inner, kernel_size=d_conv, groups=self.d_inner, padding=d_conv - 1, bias=True
        )
        self.x_proj_b = nn.Linear(self.d_inner, self.dt_rank + d_state * 2, bias=False)
        self.dt_proj_b = nn.Linear(self.dt_rank, self.d_inner, bias=True)
        self.A_b_log = nn.Parameter(torch.zeros(self.d_inner, d_state))
        self.D_b = nn.Parameter(torch.ones(self.d_inner))

        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        self.register_buffer("_eye", torch.eye(self.d_inner), persistent=False)

    def _scan(
        self,
        xz: Tensor,
        conv1d: nn.Conv1d,
        x_proj: nn.Linear,
        dt_proj: nn.Linear,
        a_log: Tensor,
        d: Tensor,
    ) -> Tensor:
        return mamba_inner_fn(
            xz,
            conv1d.weight,
            conv1d.bias,
            x_proj.weight,
            dt_proj.weight,
            self._eye,
            None,
            -torch.exp(a_log.float()),
            None,
            None,
            d.float(),
            delta_bias=dt_proj.bias.float(),
            delta_softplus=True,
        )

    def forward(self, hidden_states: Tensor) -> Tensor:
        batch, seqlen, _ = hidden_states.shape
        h_flat = hidden_states.permute(2, 0, 1).reshape(self.d_model, batch * seqlen)
        xz = (self.in_proj.weight @ h_flat).view(self.d_inner * 2, batch, seqlen).permute(1, 0, 2)

        out = self._scan(xz, self.conv1d, self.x_proj, self.dt_proj, self.A_log, self.D)
        out_b = self._scan(xz.flip([-1]), self.conv1d_b, self.x_proj_b, self.dt_proj_b, self.A_b_log, self.D_b)
        return F.linear(out + out_b.flip([1]), self.out_proj.weight, self.out_proj.bias)


class MambaBlock(nn.Module):
    r"""Post-norm residual wrapper around [`BiMamba`][torch_pointcloud.models.lion.BiMamba] (`Block`).

    Computes $x + \text{LayerNorm}(\text{BiMamba}(x))$ (post-norm residual, the LION default).

    Args:
        d_model: Feature channels.
        d_state: SSM state width passed to `BiMamba`.
        d_conv: Causal-conv kernel width passed to `BiMamba`.
        expand: Inner-width expansion passed to `BiMamba`.
    """

    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2) -> None:
        super().__init__()
        self.mamba = BiMamba(d_model, d_state=d_state, d_conv=d_conv, expand=expand)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: Tensor) -> Tensor:
        return x + self.norm(self.mamba(x))


@torch.no_grad()
def window_partition(
    pos: Tensor,
    sparse_shape: Sequence[int],
    window_shape: Sequence[int],
    shift: bool,
) -> Tuple[Tensor, Tensor, Tensor]:
    r"""Assign each voxel to a 3D window and return its in-window offset (`get_window_coors_shift_v2`).

    Each voxel is assigned to a 3D window of size `window_shape` (optionally shifted by half a window,
    Swin-style), yielding two flat per-voxel window keys: one ordering windows $x$-major
    (`win_index_x`) and one $y$-major (`win_index_y`). The within-window $(z, y, x)$ offset is
    returned for the intra-window order.

    Args:
        pos: Voxel indices $(N, 4)$ as $(\text{batch}, z, y, x)$.
        sparse_shape: Grid extent $(z, y, x)$.
        window_shape: Window size $(w_x, w_y, w_z)$.
        shift: Whether to offset windows by half their size.

    Returns:
        `(win_index_x, win_index_y, offsets)` of shapes $(N,)$, $(N,)$, $(N, 3)$.
    """
    sparse_shape_z, sparse_shape_y, sparse_shape_x = sparse_shape
    win_shape_x, win_shape_y, win_shape_z = window_shape

    if shift:
        shift_x, shift_y, shift_z = win_shape_x // 2, win_shape_y // 2, win_shape_z // 2
    else:
        shift_x, shift_y, shift_z = 0, 0, 0

    max_num_win_x = int(np.ceil((sparse_shape_x / win_shape_x)) + 1)
    max_num_win_y = int(np.ceil((sparse_shape_y / win_shape_y)) + 1)
    max_num_win_z = int(np.ceil((sparse_shape_z / win_shape_z)) + 1)
    max_num_win_per_sample = max_num_win_x * max_num_win_y * max_num_win_z

    x = pos[:, 3] + shift_x
    y = pos[:, 2] + shift_y
    z = pos[:, 1] + shift_z

    win_x = x // win_shape_x
    win_y = y // win_shape_y
    win_z = z // win_shape_z

    offset_x = x % win_shape_x
    offset_y = y % win_shape_y
    offset_z = z % win_shape_z

    win_index_x = (
        pos[:, 0] * max_num_win_per_sample + win_x * max_num_win_y * max_num_win_z + win_y * max_num_win_z + win_z
    )
    win_index_y = (
        pos[:, 0] * max_num_win_per_sample + win_y * max_num_win_x * max_num_win_z + win_x * max_num_win_z + win_z
    )
    offsets = torch.stack([offset_z, offset_y, offset_x], dim=-1)
    return win_index_x, win_index_y, offsets


class FlattenedWindowMapping(nn.Module):
    r"""Window grouping / serialization for the linear group RNN (`FlattenedWindowMapping`).

    Builds, for a voxel set, the index maps that (a) pad each scene up to a multiple of `group_size`
    (`flat2win` / `win2flat`) and (b) sort voxels into space-filling window order along the $x$- and
    $y$-major directions, so the Mamba operator can run over fixed-length contiguous groups.

    Args:
        window_shape: Window size $(w_x, w_y, w_z)$.
        group_size: Sequence length of each Mamba group.
        shift: Whether windows are shifted by half their size.
    """

    def __init__(self, window_shape: Sequence[int], group_size: int, shift: bool) -> None:
        super().__init__()
        self.window_shape = list(window_shape)
        self.group_size = group_size
        self.shift = shift

    def forward(self, pos: Tensor, batch_size: int, sparse_shape: Sequence[int]) -> Dict[str, Tensor]:
        pos = pos.long()
        # bincount keeps a zero count for batch elements with no voxels, which torch.unique would drop.
        num_per_batch = torch.bincount(pos[:, 0], minlength=batch_size)
        batch_start_indices = F.pad(torch.cumsum(num_per_batch, dim=0), (1, 0))
        num_per_batch_p = (
            torch.div(
                batch_start_indices[1:] - batch_start_indices[:-1] + self.group_size - 1,
                self.group_size,
                rounding_mode="trunc",
            )
            * self.group_size
        )
        batch_start_indices_p = F.pad(torch.cumsum(num_per_batch_p, dim=0), (1, 0))
        flat2win = torch.arange(int(batch_start_indices_p[-1]), device=pos.device)
        win2flat = torch.arange(int(batch_start_indices[-1]), device=pos.device)

        for i in range(batch_size):
            if num_per_batch[i] != num_per_batch_p[i]:
                bias_index = batch_start_indices_p[i] - batch_start_indices[i]
                flat2win[
                    batch_start_indices_p[i + 1]
                    - self.group_size
                    + (num_per_batch[i] % self.group_size) : batch_start_indices_p[i + 1]
                ] = (
                    flat2win[
                        batch_start_indices_p[i + 1]
                        - 2 * self.group_size
                        + (num_per_batch[i] % self.group_size) : batch_start_indices_p[i + 1] - self.group_size
                    ]
                    if (batch_start_indices_p[i + 1] - batch_start_indices_p[i]) - self.group_size != 0
                    else win2flat[batch_start_indices[i] : batch_start_indices[i + 1]].repeat(
                        int((batch_start_indices_p[i + 1] - batch_start_indices_p[i]) // num_per_batch[i]) + 1
                    )[: self.group_size - (num_per_batch[i] % self.group_size)]
                    + bias_index
                )
            win2flat[batch_start_indices[i] : batch_start_indices[i + 1]] += (
                batch_start_indices_p[i] - batch_start_indices[i]
            )
            flat2win[batch_start_indices_p[i] : batch_start_indices_p[i + 1]] -= (
                batch_start_indices_p[i] - batch_start_indices[i]
            )

        mappings: Dict[str, Tensor] = {"flat2win": flat2win, "win2flat": win2flat}
        win_index_x, win_index_y, offsets = window_partition(pos, sparse_shape, self.window_shape, self.shift)
        wx, wy, wz = self.window_shape
        vx = win_index_x * wx * wy * wz
        vx = vx + offsets[..., 2] * wy * wz + offsets[..., 1] * wz + offsets[..., 0]
        vy = win_index_y * wx * wy * wz
        vy = vy + offsets[..., 1] * wx * wz + offsets[..., 2] * wz + offsets[..., 0]
        mappings["x"] = torch.sort(vx)[1]
        mappings["y"] = torch.sort(vy)[1]
        return mappings


class PatchMerging3D(nn.Module):
    r"""Voxel-generation / down-scaling step of LION (`PatchMerging3D`).

    A submanifold conv (LayerNorm + GELU) refines features, then voxels are merged onto a coarser grid
    by summing features that fall into the same down-scaled cell. When `diffusion` is set, the top
    `diff_scale` fraction of voxels (ranked by mean activation) are densified by spawning zero-feature
    neighbors before merging, the 3D voxel-generation that lets the linear RNN reach empty space.

    Args:
        dim: Input feature channels.
        out_dim: Output channels of the post-merge norm; `-1` keeps `dim`.
        down_scale: Per-axis merge factor $(s_x, s_y, s_z)$.
        diffusion: Enable the voxel-generation (densification) step.
        diff_scale: Fraction of voxels expanded when `diffusion` is set.
    """

    def __init__(
        self,
        dim: int,
        out_dim: int = -1,
        down_scale: Sequence[int] = (2, 2, 2),
        diffusion: bool = False,
        diff_scale: float = 0.2,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.sub_conv = spconv.SparseSequential(
            spconv.SubMConv3d(dim, dim, 3, bias=False, indice_key="subm"),
            nn.LayerNorm(dim),
            nn.GELU(),
        )
        self.norm = nn.LayerNorm(dim if out_dim == -1 else out_dim)
        self.down_scale = list(down_scale)
        self.diffusion = diffusion
        self.diff_scale = diff_scale

    def forward(
        self,
        x: "spconv.SparseConvTensor",
        pos_shift: int = 1,
        diffusion_scale: int = 4,
    ) -> Tuple["spconv.SparseConvTensor", Tensor]:
        if diffusion_scale not in (2, 4):
            raise ValueError(f"`diffusion_scale` must be 2 or 4, got {diffusion_scale}.")

        x = self.sub_conv(x)
        d, h, w = x.spatial_shape
        down_scale = self.down_scale

        if self.diffusion:
            x_feat_att = x.features.mean(-1)
            batch_size = int(x.indices[:, 0].max()) + 1
            selected_diffusion_feats_list = [x.features.clone()]
            selected_diffusion_pos_list = [x.indices.clone()]
            for i in range(batch_size):
                mask = x.indices[:, 0] == i
                valid_num = int(mask.sum())
                k = int(valid_num * self.diff_scale)
                _, indices = torch.topk(x_feat_att[mask], k)
                selected_pos_copy = x.indices[mask][indices].clone()
                n_sel = selected_pos_copy.shape[0]
                selected_pos_expand = selected_pos_copy.repeat(diffusion_scale, 1)
                selected_feats_expand = x.features[mask][indices].repeat(diffusion_scale, 1) * 0.0

                selected_pos_expand[n_sel * 0 : n_sel * 1, 3:4] = (selected_pos_copy[:, 3:4] - pos_shift).clamp(
                    min=0, max=w - 1
                )
                selected_pos_expand[n_sel * 0 : n_sel * 1, 2:3] = (selected_pos_copy[:, 2:3] + pos_shift).clamp(
                    min=0, max=h - 1
                )
                selected_pos_expand[n_sel * 0 : n_sel * 1, 1:2] = selected_pos_copy[:, 1:2].clamp(min=0, max=d - 1)
                selected_pos_expand[n_sel : n_sel * 2, 3:4] = (selected_pos_copy[:, 3:4] + pos_shift).clamp(
                    min=0, max=w - 1
                )
                selected_pos_expand[n_sel : n_sel * 2, 2:3] = (selected_pos_copy[:, 2:3] + pos_shift).clamp(
                    min=0, max=h - 1
                )
                selected_pos_expand[n_sel : n_sel * 2, 1:2] = selected_pos_copy[:, 1:2].clamp(min=0, max=d - 1)
                if diffusion_scale == 4:
                    selected_pos_expand[n_sel * 2 : n_sel * 3, 3:4] = (selected_pos_copy[:, 3:4] - pos_shift).clamp(
                        min=0, max=w - 1
                    )
                    selected_pos_expand[n_sel * 2 : n_sel * 3, 2:3] = (selected_pos_copy[:, 2:3] - pos_shift).clamp(
                        min=0, max=h - 1
                    )
                    selected_pos_expand[n_sel * 2 : n_sel * 3, 1:2] = selected_pos_copy[:, 1:2].clamp(min=0, max=d - 1)
                    selected_pos_expand[n_sel * 3 : n_sel * 4, 3:4] = (selected_pos_copy[:, 3:4] + pos_shift).clamp(
                        min=0, max=w - 1
                    )
                    selected_pos_expand[n_sel * 3 : n_sel * 4, 2:3] = (selected_pos_copy[:, 2:3] - pos_shift).clamp(
                        min=0, max=h - 1
                    )
                    selected_pos_expand[n_sel * 3 : n_sel * 4, 1:2] = selected_pos_copy[:, 1:2].clamp(min=0, max=d - 1)
                selected_diffusion_pos_list.append(selected_pos_expand)
                selected_diffusion_feats_list.append(selected_feats_expand)
            pos = torch.cat(selected_diffusion_pos_list)
            final_diffusion_feats = torch.cat(selected_diffusion_feats_list)
        else:
            pos = x.indices.clone()
            final_diffusion_feats = x.features.clone()

        pos[:, 3:4] = pos[:, 3:4] // down_scale[0]
        pos[:, 2:3] = pos[:, 2:3] // down_scale[1]
        pos[:, 1:2] = pos[:, 1:2] // down_scale[2]

        scale_xyz = (
            (x.spatial_shape[0] // down_scale[2])
            * (x.spatial_shape[1] // down_scale[1])
            * (x.spatial_shape[2] // down_scale[0])
        )
        scale_yz = (x.spatial_shape[0] // down_scale[2]) * (x.spatial_shape[1] // down_scale[1])
        scale_z = x.spatial_shape[0] // down_scale[2]

        merge_indices = pos[:, 0].int() * scale_xyz + pos[:, 3] * scale_yz + pos[:, 2] * scale_z + pos[:, 1]
        new_sparse_shape = [math.ceil(x.spatial_shape[i] / down_scale[2 - i]) for i in range(3)]
        unq_indices, unq_inv = torch.unique(merge_indices, return_inverse=True)
        x_merge = scatter(final_diffusion_feats, unq_inv, dim=0, reduce="sum")

        unq_indices = unq_indices.int()
        voxel_indices = torch.stack(
            [
                torch.div(unq_indices, scale_xyz, rounding_mode="floor"),
                torch.div(unq_indices % scale_xyz, scale_yz, rounding_mode="floor"),
                torch.div(unq_indices % scale_yz, scale_z, rounding_mode="floor"),
                unq_indices % scale_z,
            ],
            dim=1,
        )
        voxel_indices = voxel_indices[:, [0, 3, 2, 1]]
        x_merge = self.norm(x_merge)
        out = spconv.SparseConvTensor(
            features=x_merge,
            indices=voxel_indices.int(),
            spatial_shape=new_sparse_shape,
            batch_size=x.batch_size,
        )
        return out, unq_inv


class PatchExpanding3D(nn.Module):
    r"""Scatter-back step pairing a [`PatchMerging3D`][torch_pointcloud.models.lion.PatchMerging3D] (`PatchExpanding3D`).

    Gathers the merged features back to the finer voxel layout (via the merge inverse map) and adds
    them onto the skip features at that scale.

    Args:
        dim: Feature channels (unused; kept to mirror the reference signature).
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(
        self,
        x: "spconv.SparseConvTensor",
        up_x: "spconv.SparseConvTensor",
        unq_inv: Tensor,
    ) -> "spconv.SparseConvTensor":
        _, c = x.features.shape
        x_copy = torch.gather(x.features, 0, unq_inv.unsqueeze(1).repeat(1, c))
        return up_x.replace_feature(up_x.features + x_copy)


class LIONLayer(nn.Module):
    r"""One linear group RNN layer: window-group serialize + Mamba per direction (`LIONLayer`).

    For each serialization direction (`x`, `y`) the voxels are gathered into fixed-length groups by
    [`FlattenedWindowMapping`][torch_pointcloud.models.lion.FlattenedWindowMapping], passed through a
    [`MambaBlock`][torch_pointcloud.models.lion.MambaBlock], and scattered back to voxel order.

    Args:
        dim: Feature channels.
        window_shape: Window size $(w_x, w_y, w_z)$.
        group_size: Mamba group length.
        direction: Serialization directions (e.g. `("x", "y")`).
        shift: Whether windows are shifted by half their size.
        d_state: SSM state width.
        d_conv: Causal-conv kernel width.
        expand: Inner-width expansion factor.
    """

    def __init__(
        self,
        dim: int,
        window_shape: Sequence[int],
        group_size: int,
        direction: Sequence[str],
        shift: bool,
        d_state: int,
        d_conv: int,
        expand: int,
    ) -> None:
        super().__init__()
        self.direction = list(direction)
        self.group_size = group_size
        self.blocks = nn.ModuleList(MambaBlock(dim, d_state=d_state, d_conv=d_conv, expand=expand) for _ in direction)
        self.window_partition = FlattenedWindowMapping(window_shape, group_size, shift)

    def forward(self, x: "spconv.SparseConvTensor") -> "spconv.SparseConvTensor":
        mappings = self.window_partition(x.indices, x.batch_size, x.spatial_shape)
        for i, block in enumerate(self.blocks):
            indices = mappings[self.direction[i]]
            x_features = x.features[indices][mappings["flat2win"]]
            x_features = x_features.view(-1, self.group_size, x.features.shape[-1])
            x_features = block(x_features)
            x.features[indices] = x_features.view(-1, x_features.shape[-1])[mappings["win2flat"]]
        return x


class LIONBlock(nn.Module):
    r"""Hierarchical encoder/decoder stage of the LION backbone (`LIONBlock`).

    A `depth`-deep stack alternating [`LIONLayer`][torch_pointcloud.models.lion.LIONLayer] (with a
    learned position embedding) and a down-scaling
    [`PatchMerging3D`][torch_pointcloud.models.lion.PatchMerging3D] on the way down, then a matching
    decoder that scatters features back with
    [`PatchExpanding3D`][torch_pointcloud.models.lion.PatchExpanding3D].

    Args:
        dim: Feature channels.
        depth: Encoder/decoder depth.
        down_scales: Per-level merge factor $(s_x, s_y, s_z)$.
        window_shape: Window size $(w_x, w_y, w_z)$.
        group_size: Mamba group length.
        direction: Serialization directions.
        shift: Whether the second layer per level uses shifted windows.
        d_state: SSM state width.
        d_conv: Causal-conv kernel width.
        expand: Inner-width expansion factor.
    """

    def __init__(
        self,
        dim: int,
        depth: int,
        down_scales: Sequence[Sequence[int]],
        window_shape: Sequence[int],
        group_size: int,
        direction: Sequence[str],
        shift: bool,
        d_state: int,
        d_conv: int,
        expand: int,
    ) -> None:
        super().__init__()
        self.down_scales = [list(s) for s in down_scales]
        shifts = [False, shift]
        mamba_kwargs: Dict[str, Any] = dict(d_state=d_state, d_conv=d_conv, expand=expand)

        self.encoder = nn.ModuleList()
        self.downsample_list = nn.ModuleList()
        self.pos_emb_list = nn.ModuleList()
        for idx in range(depth):
            self.encoder.append(LIONLayer(dim, window_shape, group_size, direction, shifts[idx], **mamba_kwargs))
            self.pos_emb_list.append(MLP([3, dim, dim], act="relu", norm="batch_norm", plain_last=True))
            self.downsample_list.append(PatchMerging3D(dim, dim, down_scale=down_scales[idx]))

        self.decoder = nn.ModuleList()
        self.decoder_norm = nn.ModuleList()
        self.upsample_list = nn.ModuleList()
        for idx in range(depth):
            self.decoder.append(LIONLayer(dim, window_shape, group_size, direction, shifts[idx], **mamba_kwargs))
            self.decoder_norm.append(nn.LayerNorm(dim))
            self.upsample_list.append(PatchExpanding3D(dim))

    def forward(self, x: "spconv.SparseConvTensor") -> "spconv.SparseConvTensor":
        features: List["spconv.SparseConvTensor"] = []
        index: List[Tensor] = []
        for idx, enc in enumerate(self.encoder):
            win_z, win_y, win_x = x.spatial_shape
            pos_x = (x.indices[:, 3] - win_x / 2) / win_x * 2 * math.pi
            pos_y = (x.indices[:, 2] - win_y / 2) / win_y * 2 * math.pi
            pos_z = (x.indices[:, 1] - win_z / 2) / win_z * 2 * math.pi
            pos_emb = self.pos_emb_list[idx](torch.stack((pos_x, pos_y, pos_z), dim=-1))
            x = x.replace_feature(pos_emb + x.features)
            x = enc(x)
            features.append(x)
            x, unq_inv = self.downsample_list[idx](x)
            index.append(unq_inv)

        for i, (dec, norm, up_x, unq_inv) in enumerate(
            zip(self.decoder, self.decoder_norm, features[::-1], index[::-1])
        ):
            x = dec(x)
            x = self.upsample_list[i](x, up_x, unq_inv)
            x = x.replace_feature(norm(x.features))
        return x


class LION3DBackbone(nn.Module):
    r"""LION sparse 3D backbone: hierarchical linear group RNN over voxels (`LION3DBackboneOneStride`).

    Four [`LIONBlock`][torch_pointcloud.models.lion.LIONBlock] stages, each followed by a height
    down-scaling [`PatchMerging3D`][torch_pointcloud.models.lion.PatchMerging3D] (diffusion enabled),
    end in a single [`LIONLayer`][torch_pointcloud.models.lion.LIONLayer] over the compressed grid.
    The output keeps the planar resolution (one-stride) and a height of 2 for BEV folding.

    Args:
        grid_size: Voxel grid extent $(n_x, n_y, n_z)$.
        channels: Backbone feature channels.
        num_layers: Number of `LIONBlock` stages.
        depths: Per-stage encoder/decoder depth.
        layer_down_scales: Per-stage, per-depth merge factors.
        window_shape: Per-stage window size $(w_x, w_y, w_z)$.
        group_size: Per-stage Mamba group length.
        direction: Serialization directions.
        diffusion: Enable voxel-generation in the height-merge steps.
        diff_scale: Fraction of voxels expanded by diffusion.
        shift: Whether shifted windows are used.
        d_state: SSM state width.
        d_conv: Causal-conv kernel width.
        expand: Inner-width expansion factor.
    """

    def __init__(
        self,
        grid_size: Sequence[int],
        *,
        channels: int = 128,
        num_layers: int = 4,
        depths: Sequence[int] = (2, 2, 2, 2),
        layer_down_scales: Sequence[Sequence[Sequence[int]]] = (
            ((2, 2, 2), (2, 2, 2)),
            ((2, 2, 2), (2, 2, 2)),
            ((2, 2, 2), (2, 2, 2)),
            ((2, 2, 2), (2, 2, 2)),
        ),
        window_shape: Sequence[Sequence[int]] = ((13, 13, 32), (13, 13, 16), (13, 13, 8), (13, 13, 4)),
        group_size: Sequence[int] = (4096, 2048, 1024, 512),
        direction: Sequence[str] = ("x", "y"),
        diffusion: bool = True,
        diff_scale: float = 0.2,
        shift: bool = True,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
    ) -> None:
        super().__init__()
        if not (num_layers == len(depths) == len(layer_down_scales) == len(window_shape) == len(group_size)):
            raise ValueError(
                f"`depths`, `layer_down_scales`, `window_shape` and `group_size` must all have length"
                f" `num_layers` ({num_layers}), got {len(depths)}, {len(layer_down_scales)},"
                f" {len(window_shape)} and {len(group_size)}."
            )

        self.sparse_shape = list(grid_size[::-1])
        layer_dim = [channels] * num_layers
        block_kwargs: Dict[str, Any] = dict(d_state=d_state, d_conv=d_conv, expand=expand)
        merge_kwargs: Dict[str, Any] = dict(diffusion=diffusion, diff_scale=diff_scale)

        self.linear_1 = LIONBlock(
            layer_dim[0],
            depths[0],
            layer_down_scales[0],
            window_shape[0],
            group_size[0],
            direction,
            shift,
            **block_kwargs,
        )
        self.dow1 = PatchMerging3D(layer_dim[0], layer_dim[0], down_scale=[1, 1, 2], **merge_kwargs)
        self.linear_2 = LIONBlock(
            layer_dim[1],
            depths[1],
            layer_down_scales[1],
            window_shape[1],
            group_size[1],
            direction,
            shift,
            **block_kwargs,
        )
        self.dow2 = PatchMerging3D(layer_dim[1], layer_dim[1], down_scale=[1, 1, 2], **merge_kwargs)
        self.linear_3 = LIONBlock(
            layer_dim[2],
            depths[2],
            layer_down_scales[2],
            window_shape[2],
            group_size[2],
            direction,
            shift,
            **block_kwargs,
        )
        self.dow3 = PatchMerging3D(layer_dim[2], layer_dim[3], down_scale=[1, 1, 2], **merge_kwargs)
        self.linear_4 = LIONBlock(
            layer_dim[3],
            depths[3],
            layer_down_scales[3],
            window_shape[3],
            group_size[3],
            direction,
            shift,
            **block_kwargs,
        )
        self.dow4 = PatchMerging3D(layer_dim[3], layer_dim[3], down_scale=[1, 1, 2], **merge_kwargs)
        self.linear_out = LIONLayer(layer_dim[3], [13, 13, 2], 256, ["x", "y"], shift, **block_kwargs)
        self.num_point_features = channels

    def forward(self, voxel_features: Tensor, voxel_indices: Tensor, batch_size: int) -> "spconv.SparseConvTensor":
        x = spconv.SparseConvTensor(
            features=voxel_features,
            indices=voxel_indices.int(),
            spatial_shape=self.sparse_shape,
            batch_size=batch_size,
        )
        x = self.linear_1(x)
        x1, _ = self.dow1(x)
        x = self.linear_2(x1)
        x2, _ = self.dow2(x)
        x = self.linear_3(x2)
        x3, _ = self.dow3(x)
        x = self.linear_4(x3)
        x4, _ = self.dow4(x)
        return self.linear_out(x4)


class LIONDetection(DetectionModel):
    r"""LION: linear group RNN (Mamba) 3D object detector (packed point format).

    Reference: :arxiv: [LION: Linear Group RNN for 3D Object Detection in Point Clouds](https://arxiv.org/abs/2407.18232) (Liu et al., 2024). Reference implementation:
    :github: [happinesslz/LION](https://github.com/happinesslz/LION).

    Points are encoded into voxels by a dynamic mean VFE, processed by a hierarchical sparse backbone
    that serializes voxels into spatially grouped windows and runs a bidirectional Mamba operator
    (the linear group RNN), with periodic 3D voxel-generation / height-merging steps. The one-stride
    output is folded to a dense BEV map, refined by a residual 2D backbone, and decoded by a
    query-based [`TransFusionHead`][torch_pointcloud.layers.transfusion_head.TransFusionHead].

    Args:
        in_channels: Raw point feature channels including xyz (5 for nuScenes $x, y, z, \text{intensity}, \Delta t$).
        num_classes: Number of foreground classes (10 for nuScenes).
        voxel_size: Voxel size $(v_x, v_y, v_z)$.
        point_cloud_range: Range $(x_\min, y_\min, z_\min, x_\max, y_\max, z_\max)$.
        channels: Backbone / VFE feature channels.
        vfe_num_filters: PFN widths of the dynamic mean VFE.
        depths: Per-stage encoder/decoder depth of the 3D backbone.
        window_shape: Per-stage window size $(w_x, w_y, w_z)$.
        group_size: Per-stage Mamba group length.
        diffusion: Enable voxel-generation in the height-merge steps.
        diff_scale: Fraction of voxels expanded by diffusion.
        layer_nums: 2D backbone residual-block counts per level.
        layer_strides: 2D backbone downsample strides per level.
        num_filters: 2D backbone channel widths per level.
        upsample_strides: 2D backbone upsample factors per level.
        num_upsample_filters: 2D backbone upsample channels per level.
        feature_map_stride: BEV stride of the head.
        local_max_classes: Crowded small-object class indices passed to the head (nuScenes pedestrian /
            traffic-cone); see [`TransFusionHead`][torch_pointcloud.layers.transfusion_head.TransFusionHead].
        post_center_range: Box-center range kept at decode; `None` keeps every box.
        d_state: SSM state width.
        d_conv: Causal-conv kernel width.
        expand: Inner-width expansion factor of the Mamba operator.
    """

    def __init__(
        self,
        in_channels: int = 5,
        num_classes: int = 10,
        *,
        voxel_size: Sequence[float] = (0.3, 0.3, 0.25),
        point_cloud_range: Sequence[float] = (-54.0, -54.0, -5.0, 54.0, 54.0, 3.0),
        channels: int = 128,
        vfe_num_filters: Sequence[int] = (128, 128),
        depths: Sequence[int] = (2, 2, 2, 2),
        window_shape: Sequence[Sequence[int]] = ((13, 13, 32), (13, 13, 16), (13, 13, 8), (13, 13, 4)),
        group_size: Sequence[int] = (4096, 2048, 1024, 512),
        diffusion: bool = True,
        diff_scale: float = 0.2,
        layer_nums: Sequence[int] = (1, 2, 2),
        layer_strides: Sequence[int] = (1, 2, 2),
        num_filters: Sequence[int] = (128, 128, 256),
        upsample_strides: Sequence[float] = (0.5, 1, 2),
        num_upsample_filters: Sequence[int] = (128, 128, 128),
        feature_map_stride: int = 2,
        local_max_classes: Sequence[int] = (),
        post_center_range: Optional[Sequence[float]] = None,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
    ) -> None:
        super().__init__(in_channels=in_channels, num_classes=num_classes)
        self.voxel_size = tuple(voxel_size)
        self.point_cloud_range = tuple(point_cloud_range)
        grid = [int(round((point_cloud_range[i + 3] - point_cloud_range[i]) / voxel_size[i])) for i in range(3)]
        self.grid_size: Tuple[int, int, int] = (grid[0], grid[1], grid[2])
        self.channels = channels
        self.vfe_num_filters = vfe_num_filters
        self.depths = depths
        self.window_shape = window_shape
        self.group_size = group_size
        self.diffusion = diffusion
        self.diff_scale = diff_scale
        self.layer_nums = layer_nums
        self.layer_strides = layer_strides
        self.num_filters = num_filters
        self.upsample_strides = upsample_strides
        self.num_upsample_filters = num_upsample_filters
        self.feature_map_stride = feature_map_stride
        self.local_max_classes = local_max_classes
        self.post_center_range = post_center_range
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand

        self.vfe = self.configure_vfe()
        self.backbone_3d = self.configure_backbone_3d()
        self.backbone = self.configure_backbone()
        self.head = self.configure_head()

    def configure_vfe(self) -> DynamicMeanVFE:
        """Build the dynamic mean voxel feature encoder."""
        return DynamicMeanVFE(
            self.in_channels, self.vfe_num_filters, self.voxel_size, self.point_cloud_range, self.grid_size
        )

    def configure_backbone_3d(self) -> LION3DBackbone:
        """Build the sparse 3D linear group RNN backbone."""
        return LION3DBackbone(
            self.grid_size,
            channels=self.channels,
            depths=self.depths,
            window_shape=self.window_shape,
            group_size=self.group_size,
            diffusion=self.diffusion,
            diff_scale=self.diff_scale,
            d_state=self.d_state,
            d_conv=self.d_conv,
            expand=self.expand,
        )

    def configure_backbone(self) -> BEVResidualBackbone:
        """Build the residual 2D BEV backbone."""
        return BEVResidualBackbone(
            self.channels * 2,
            self.layer_nums,
            self.layer_strides,
            self.num_filters,
            self.upsample_strides,
            self.num_upsample_filters,
            norm_kwargs={"eps": 1e-3, "momentum": 0.01},
        )

    @property
    def num_features(self) -> int:
        """Channel count $C$ of the BEV feature map entering the head."""
        return self.backbone.num_bev_features

    def configure_head(self) -> TransFusionHead:
        """Build the query-based TransFusion detection head."""
        return TransFusionHead(
            self.backbone.num_bev_features,
            self.num_classes,
            self.grid_size,
            self.point_cloud_range,
            self.voxel_size,
            feature_map_stride=self.feature_map_stride,
            dropout=0.0,
            bias=True,
            use_iou=True,
            query_radius=20,
            iou_rectifier=0.5,
            height_code="top",
            local_max_classes=self.local_max_classes,
            post_center_range=self.post_center_range,
        )

    def forward_features(self, x: OptTensor, pos: Tensor, batch: Tensor) -> Tensor:
        batch_size = int(batch.max().item()) + 1
        voxel_features, voxel_indices = self.vfe(x, pos, batch)
        encoded = self.backbone_3d(voxel_features, voxel_indices, batch_size)
        dense = encoded.dense()
        b, c, d, h, w = dense.shape
        bev = dense.view(b, c * d, h, w)
        return self.backbone(bev)

    def forward_head(self, features: Tensor) -> TransFusionHeadOutput:
        return self.head(features)

    def forward(self, x: OptTensor, pos: Tensor, batch: Tensor) -> TransFusionHeadOutput:
        features = self.forward_features(x, pos, batch)
        return self.forward_head(features)

    @torch.no_grad()
    def decode(self, out: TransFusionHeadOutput) -> Detection3D:
        r"""Decode a forward output into raw candidate detections (see `TransFusionHead.decode`)."""
        return self.head.decode(out)


def checkpoint_filter_fn(state_dict: Dict[str, Any], model: nn.Module) -> Dict[str, Any]:
    r"""Transpose the head convolution kernels of a released LION checkpoint, as the `pretrained_filter_fn` of a
    `WeightsDict`.

    The released head convolves the BEV map with $x$ along its rows, while
    [`TransFusionHead`][torch_pointcloud.layers.transfusion_head.TransFusionHead] takes the $(y, x)$ layout of the
    backbone, so the spatial axes of each head convolution kernel are swapped.

    Args:
        state_dict: State dict of the released checkpoint.
        model: Model the state dict is loaded into.

    Returns:
        The state dict with the head convolution kernels transposed.
    """
    return {
        key: value.transpose(-1, -2).contiguous() if key.startswith("head.") and value.dim() == 4 else value
        for key, value in state_dict.items()
    }


@register_model(
    "lion-mamba.nuscenes.zhe-liu",
    task="detection",
    weights=WeightsDict(
        url="hf://torch-pointcloud/lion-mamba.nuscenes.zhe-liu/resolve/cf347e7237553182ab3818056d3d7bb8843811d3/model.safetensors",
        pretrained_filter_fn=checkpoint_filter_fn,
        dataset="nuscenes",
        metrics={"mAP": 68.82, "NDS": 72.33},
        classes=NUSCENES_DETECTION_CLASSES,
        author="zhe-liu",
        license="Apache-2.0",
    ),
    transform=T.Compose([T.Cat(keys=[DataKeys.INTENSITY, DataKeys.TIMESTAMP], dst_key=DataKeys.X, dim=1)]),
    hparams=dict(
        in_channels=5,
        num_classes=10,
        voxel_size=(0.3, 0.3, 0.25),
        point_cloud_range=(-54.0, -54.0, -5.0, 54.0, 54.0, 3.0),
        channels=128,
        vfe_num_filters=(128, 128),
        depths=(2, 2, 2, 2),
        window_shape=((13, 13, 32), (13, 13, 16), (13, 13, 8), (13, 13, 4)),
        group_size=(4096, 2048, 1024, 512),
        diffusion=True,
        diff_scale=0.2,
        layer_nums=(1, 2, 2),
        layer_strides=(1, 2, 2),
        num_filters=(128, 128, 256),
        upsample_strides=(0.5, 1, 2),
        num_upsample_filters=(128, 128, 128),
        feature_map_stride=2,
        local_max_classes=(8, 9),
        post_center_range=(-61.2, -61.2, -10.0, 61.2, 61.2, 10.0),
        d_state=16,
        d_conv=4,
        expand=2,
    ),
)
def lion_mamba_nuscenes(**hparams: Any) -> LIONDetection:
    return LIONDetection(**hparams)
