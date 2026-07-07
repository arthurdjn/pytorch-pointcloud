import math
from typing import TYPE_CHECKING, Any, Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn import MLP
from torch_geometric.nn.dense.linear import Linear

import torch_pointcloud.transforms as T
from torch_pointcloud.layers.bev_backbone import BaseBEVResBackbone
from torch_pointcloud.layers.vfe import DynamicMeanVFE
from torch_pointcloud.utils.data import DataKeys
from torch_pointcloud.utils.imports import (
    _MAMBA_SSM_GITHUB_URL,
    _SPCONV_GITHUB_URL,
    _TORCH_SCATTER_GITHUB_URL,
    optional_import,
)
from torch_pointcloud.utils.types import Detection3D, OptTensor

from ._base import DetectionModel
from ._registry import register_model

if TYPE_CHECKING:
    import spconv.pytorch as spconv
    from mamba_ssm.ops.selective_scan_interface import mamba_inner_fn
    from torch_scatter import scatter_add

spconv, _IS_SPCONV_AVAILABLE = optional_import("spconv.pytorch", url=_SPCONV_GITHUB_URL)
mamba_inner_fn, _ = optional_import(
    "mamba_ssm.ops.selective_scan_interface", "mamba_inner_fn", url=_MAMBA_SSM_GITHUB_URL
)
scatter_add, _ = optional_import("torch_scatter", "scatter_add", url=_TORCH_SCATTER_GITHUB_URL)


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
        self, xz: Tensor, conv1d: nn.Conv1d, x_proj: nn.Linear, dt_proj: nn.Linear, a_log: Tensor, d: Tensor
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
    pos: Tensor, sparse_shape: Sequence[int], window_shape: Sequence[int], shift: bool
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
        _, num_per_batch = torch.unique(pos[:, 0], sorted=False, return_counts=True)
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
                        (batch_start_indices_p[i + 1] - batch_start_indices_p[i]) // num_per_batch[i] + 1
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
    neighbours before merging, the 3D voxel-generation that lets the linear RNN reach empty space.

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
        assert diffusion_scale in (2, 4)
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
        unq_indices, unq_inv = torch.unique(merge_indices, return_inverse=True, return_counts=False, dim=0)
        x_merge = scatter_add(final_diffusion_feats, unq_inv, dim=0)

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
        self, x: "spconv.SparseConvTensor", up_x: "spconv.SparseConvTensor", unq_inv: Tensor
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
        assert num_layers == len(depths) == len(layer_down_scales) == len(window_shape) == len(group_size)
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


class SeparateHeadTransfusion(nn.Module):
    r"""Per-attribute prediction head of TransFusion (`SeparateHead_Transfusion`).

    One small MLP per box attribute, applied to the per-query features: `center` $(2)$, `height`
    $(1)$, `dim` $(3)$, `rot` $(2)$, `vel` $(2)$, `iou` $(1)$ and a class `heatmap`. The reference's
    $1 \times 1$ convolutions over $(B, C, Q)$ are equivalent to linear layers over the flattened
    query dim, so each branch is a plain `MLP`. The branch widths are fixed by the box
    parametrization; only the number of classes and the depths are configurable.

    Args:
        in_channels: Input feature channels.
        head_channels: Hidden channels of the per-attribute MLP.
        num_classes: Number of classes predicted by the heatmap branch.
        num_layers: Number of layers per box-attribute branch.
        num_heatmap_layers: Number of layers of the heatmap branch.
        init_bias: Bias initialization for the heatmap output layer.
        bias: Whether hidden layers carry a bias.

    Shape:
        - Input: $(B, C_\text{in}, Q)$ per-query features.
        - Output: dict of $(B, C_\text{attr}, Q)$ tensors keyed by attribute.
    """

    def __init__(
        self,
        in_channels: int,
        head_channels: int,
        num_classes: int,
        num_layers: int = 2,
        num_heatmap_layers: int = 2,
        init_bias: float = -2.19,
        bias: bool = False,
    ) -> None:
        super().__init__()
        hidden = [head_channels] * (num_layers - 1)
        layer_bias = [bias] * (num_layers - 1) + [True]
        factory_kwargs: Dict[str, Any] = dict(act="relu", norm="batch_norm", plain_last=True)
        self.center = MLP([in_channels, *hidden, 2], bias=layer_bias, **factory_kwargs)
        self.height = MLP([in_channels, *hidden, 1], bias=layer_bias, **factory_kwargs)
        self.dim = MLP([in_channels, *hidden, 3], bias=layer_bias, **factory_kwargs)
        self.rot = MLP([in_channels, *hidden, 2], bias=layer_bias, **factory_kwargs)
        self.vel = MLP([in_channels, *hidden, 2], bias=layer_bias, **factory_kwargs)
        self.iou = MLP([in_channels, *hidden, 1], bias=layer_bias, **factory_kwargs)
        heatmap_hidden = [head_channels] * (num_heatmap_layers - 1)
        heatmap_bias = [bias] * (num_heatmap_layers - 1) + [True]
        self.heatmap = MLP([in_channels, *heatmap_hidden, num_classes], bias=heatmap_bias, **factory_kwargs)
        final_lin = self.heatmap.lins[-1]
        assert isinstance(final_lin, Linear) and final_lin.bias is not None
        final_lin.bias.data.fill_(init_bias)

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        B, _, Q = x.shape
        flat = x.transpose(1, 2).reshape(B * Q, -1)
        outputs = {
            "center": self.center(flat),
            "height": self.height(flat),
            "dim": self.dim(flat),
            "rot": self.rot(flat),
            "vel": self.vel(flat),
            "iou": self.iou(flat),
            "heatmap": self.heatmap(flat),
        }
        return {name: out.reshape(B, Q, -1).transpose(1, 2) for name, out in outputs.items()}


class TransformerDecoderLayer(nn.Module):
    r"""Single TransFusion decoder layer: self-attn + cross-attn + FFN (`TransformerDecoderLayer`).

    Args:
        embed_dim: Model channels.
        num_heads: Number of attention heads.
        mlp_dim: FFN hidden width.
        dropout: Dropout probability.
        activation: FFN activation (`relu`/`gelu`).
        self_posembed: Position embedding applied to the flattened query positions (self-attention).
        cross_posembed: Position embedding applied to the flattened key positions (cross-attention).
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        mlp_dim: int,
        dropout: float,
        activation: str,
        self_posembed: nn.Module,
        cross_posembed: nn.Module,
    ) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.linear1 = nn.Linear(embed_dim, mlp_dim)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(mlp_dim, embed_dim)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.norm3 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu
        self.self_posembed = self_posembed
        self.cross_posembed = cross_posembed

    def forward(
        self, query: Tensor, key: Tensor, query_pos: Tensor, key_pos: Tensor, key_padding_mask: OptTensor = None
    ) -> Tensor:
        b, num_query, _ = query_pos.shape
        query_pos_embed = self.self_posembed(query_pos.reshape(b * num_query, -1)).reshape(b, num_query, -1)
        query_pos_embed = query_pos_embed.transpose(0, 1)
        b, num_key, _ = key_pos.shape
        key_pos_embed = self.cross_posembed(key_pos.reshape(b * num_key, -1)).reshape(b, num_key, -1)
        key_pos_embed = key_pos_embed.transpose(0, 1)
        query = query.permute(2, 0, 1)
        key = key.permute(2, 0, 1)

        q = k = v = query + query_pos_embed
        query2 = self.self_attn(q, k, value=v)[0]
        query = query + self.dropout1(query2)
        query = self.norm1(query)

        query2 = self.multihead_attn(
            query=query + query_pos_embed,
            key=key + key_pos_embed,
            value=key + key_pos_embed,
            key_padding_mask=key_padding_mask,
        )[0]
        query = query + self.dropout2(query2)
        query = self.norm2(query)

        query2 = self.linear2(self.dropout(self.activation(self.linear1(query))))
        query = query + self.dropout3(query2)
        query = self.norm3(query)
        return query.permute(1, 2, 0)


class BasicBlock2D(nn.Module):
    r"""Conv2d + BN + ReLU block used by the TransFusion heatmap head (`BasicBlock2D`).

    Args:
        in_channels: Input channels.
        out_channels: Output channels.
        kernel_size: Conv kernel size.
        padding: Conv padding.
        bias: Whether the conv carries a bias.
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, padding: int, bias: bool) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        return self.relu(self.bn(self.conv(x)))


class TransFusionHead(nn.Module):
    r"""Query-based TransFusion detection head (`TransFusionHead`).

    A shared conv produces a BEV feature map and a dense class heatmap; the top `num_proposals` heatmap
    peaks initialize object queries that attend (self- then local-cross-attention) to BEV features via
    a [`TransformerDecoderLayer`][torch_pointcloud.models.lion.TransformerDecoderLayer]. Per-query box
    attributes are regressed by a
    [`SeparateHeadTransfusion`][torch_pointcloud.models.lion.SeparateHeadTransfusion]. `decode` rescores
    by IoU and applies per-task circular NMS (nuScenes ped/cone) to yield final boxes.

    Reference implementation: :github:
    [mit-han-lab/bevfusion](https://github.com/mit-han-lab/bevfusion) (`TransFusionHead`).

    Args:
        input_channels: BEV feature channels feeding the shared conv.
        num_classes: Number of foreground classes.
        grid_size: Voxel grid extent $(n_x, n_y, n_z)$.
        point_cloud_range: Range $(x_\min, y_\min, z_\min, x_\max, y_\max, z_\max)$.
        voxel_size: Voxel size $(v_x, v_y, v_z)$.
        feature_map_stride: BEV stride relating feature pixels to metric coordinates.
        hidden_channel: Query / decoder feature width.
        num_proposals: Number of object queries.
        num_heads: Decoder attention heads.
        nms_kernel_size: Heatmap local-max pooling kernel.
        ffn_channel: Decoder FFN width.
        dropout: Decoder dropout.
        bn_momentum: BatchNorm momentum override.
        activation: Decoder FFN activation.
        num_heatmap_layers: Layers in the heatmap branch of the prediction head.
        query_radius: Half-width of the local cross-attention window.
        iou_rectifier: Per-class exponent blending heatmap score with predicted IoU.
        nms_radius: Per-task circular-NMS radius (only the `local_max_classes` use $> 0$).
        local_max_classes: Crowded small-object class indices (nuScenes pedestrian / traffic-cone): their
            heatmap peaks skip the kernel local-max NMS in `predict` and each gets its own circular NMS
            task in `decode`. Empty by default so the head stays agnostic to the label set.
        post_center_range: Box-center range filter applied at decode.
    """

    query_offset_x: Tensor
    query_offset_y: Tensor
    bev_pos: Tensor

    def __init__(
        self,
        input_channels: int,
        num_classes: int,
        grid_size: Sequence[int],
        point_cloud_range: Sequence[float],
        voxel_size: Sequence[float],
        *,
        feature_map_stride: int = 2,
        hidden_channel: int = 128,
        num_proposals: int = 200,
        num_heads: int = 8,
        nms_kernel_size: int = 3,
        ffn_channel: int = 256,
        dropout: float = 0.0,
        bn_momentum: float = 0.1,
        activation: str = "relu",
        num_heatmap_layers: int = 2,
        query_radius: int = 20,
        iou_rectifier: float = 0.5,
        nms_radius: float = 0.175,
        local_max_classes: Sequence[int] = (),
        post_center_range: Sequence[float] = (-61.2, -61.2, -10.0, 61.2, 61.2, 10.0),
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.grid_size = tuple(int(g) for g in grid_size)
        self.point_cloud_range = tuple(float(p) for p in point_cloud_range)
        self.voxel_size = tuple(float(v) for v in voxel_size)
        self.feature_map_stride = feature_map_stride
        self.num_proposals = num_proposals
        self.nms_kernel_size = nms_kernel_size
        self.iou_rectifier = iou_rectifier
        self.nms_radius = nms_radius
        self.local_max_classes = tuple(int(c) for c in local_max_classes)
        self.post_center_range = list(post_center_range)
        self.code_size = 10

        query_range = torch.arange(-query_radius, query_radius + 1)
        qx, qy = torch.meshgrid(query_range, query_range, indexing="ij")
        self.register_buffer("query_offset_x", qx, persistent=False)
        self.register_buffer("query_offset_y", qy, persistent=False)

        self.shared_conv = nn.Conv2d(input_channels, hidden_channel, kernel_size=3, padding=1)
        self.heatmap_head = nn.Sequential(
            BasicBlock2D(hidden_channel, hidden_channel, kernel_size=3, padding=1, bias=True),
            nn.Conv2d(hidden_channel, num_classes, kernel_size=3, padding=1),
        )
        self.class_encoding = nn.Linear(num_classes, hidden_channel)
        self.decoder = TransformerDecoderLayer(
            hidden_channel,
            num_heads,
            ffn_channel,
            dropout,
            activation,
            self_posembed=MLP([2, hidden_channel, hidden_channel], act="relu", norm="batch_norm", plain_last=True),
            cross_posembed=MLP([2, hidden_channel, hidden_channel], act="relu", norm="batch_norm", plain_last=True),
        )
        self.prediction_head = SeparateHeadTransfusion(
            hidden_channel, 64, num_classes, num_heatmap_layers=num_heatmap_layers, bias=True
        )

        for m in self.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                m.momentum = bn_momentum

        x_size = self.grid_size[0] // feature_map_stride
        y_size = self.grid_size[1] // feature_map_stride
        bx, by = torch.meshgrid(
            torch.linspace(0, x_size - 1, x_size), torch.linspace(0, y_size - 1, y_size), indexing="ij"
        )
        bev_pos = torch.cat([(bx + 0.5)[None], (by + 0.5)[None]], dim=0)[None]
        self.register_buffer("bev_pos", bev_pos.view(1, 2, -1).permute(0, 2, 1), persistent=False)

    def predict(self, inputs: Tensor) -> Dict[str, Tensor]:
        batch_size = inputs.shape[0]
        lidar_feat = self.shared_conv(inputs)
        lidar_feat_flatten = lidar_feat.view(batch_size, lidar_feat.shape[1], -1)
        bev_pos = self.bev_pos.repeat(batch_size, 1, 1)

        dense_heatmap = self.heatmap_head(lidar_feat)
        heatmap = dense_heatmap.detach().sigmoid()
        x_grid, y_grid = heatmap.shape[-2:]
        padding = self.nms_kernel_size // 2
        local_max = torch.zeros_like(heatmap)
        local_max_inner = F.max_pool2d(heatmap, kernel_size=self.nms_kernel_size, stride=1, padding=0)
        local_max[:, :, padding:(-padding), padding:(-padding)] = local_max_inner
        for cls_idx in self.local_max_classes:
            local_max[:, cls_idx] = F.max_pool2d(heatmap[:, cls_idx], kernel_size=1, stride=1, padding=0)
        heatmap = heatmap * (heatmap == local_max)
        heatmap = heatmap.view(batch_size, heatmap.shape[1], -1)

        top_proposals = heatmap.view(batch_size, -1).argsort(dim=-1, descending=True)[..., : self.num_proposals]
        top_proposals_class = top_proposals // heatmap.shape[-1]
        top_proposals_index = top_proposals % heatmap.shape[-1]
        query_feat = lidar_feat_flatten.gather(
            index=top_proposals_index[:, None, :].expand(-1, lidar_feat_flatten.shape[1], -1), dim=-1
        )

        one_hot = F.one_hot(top_proposals_class, num_classes=self.num_classes).float()
        query_cat_encoding = self.class_encoding(one_hot).transpose(1, 2)
        query_feat = query_feat + query_cat_encoding
        query_pos = bev_pos.gather(
            index=top_proposals_index[:, None, :].permute(0, 2, 1).expand(-1, -1, bev_pos.shape[-1]), dim=1
        )

        top_proposals_x = top_proposals_index // x_grid
        top_proposals_y = top_proposals_index % y_grid
        top_proposals_key_x = top_proposals_x[:, :, None, None] + self.query_offset_x[None, None]
        top_proposals_key_y = top_proposals_y[:, :, None, None] + self.query_offset_y[None, None]
        top_proposals_key_index = top_proposals_key_x.view(
            batch_size, top_proposals_key_x.shape[1], -1
        ) * x_grid + top_proposals_key_y.view(batch_size, top_proposals_key_y.shape[1], -1)
        key_mask = (top_proposals_key_index < 0) + (top_proposals_key_index >= (x_grid * y_grid))
        top_proposals_key_index = torch.clamp(top_proposals_key_index, min=0, max=x_grid * y_grid - 1)
        num_proposals = top_proposals_key_index.shape[1]
        key_feat = lidar_feat_flatten.gather(
            index=top_proposals_key_index.view(batch_size, 1, -1).expand(-1, lidar_feat_flatten.shape[1], -1), dim=-1
        )
        key_feat = key_feat.view(batch_size, lidar_feat_flatten.shape[1], num_proposals, -1)
        key_pos = bev_pos.gather(
            index=top_proposals_key_index.view(batch_size, 1, -1).permute(0, 2, 1).expand(-1, -1, bev_pos.shape[-1]),
            dim=1,
        )
        key_pos = key_pos.view(batch_size, num_proposals, -1, bev_pos.shape[-1])
        key_feat = key_feat.permute(0, 2, 1, 3).reshape(batch_size * num_proposals, lidar_feat_flatten.shape[1], -1)
        key_pos = key_pos.view(-1, key_pos.shape[2], key_pos.shape[-1])
        key_padding_mask = key_mask.view(-1, key_mask.shape[-1])

        query_feat_t = query_feat.permute(0, 2, 1).reshape(batch_size * num_proposals, -1, 1)
        query_pos_t = query_pos.view(-1, 1, query_pos.shape[-1])
        query_feat_t = self.decoder(query_feat_t, key_feat, query_pos_t, key_pos, key_padding_mask)
        query_feat = query_feat_t.reshape(batch_size, num_proposals, -1).permute(0, 2, 1)

        res_layer = self.prediction_head(query_feat)
        res_layer["center"] = res_layer["center"] + query_pos.permute(0, 2, 1)
        res_layer["query_heatmap_score"] = heatmap.gather(
            index=top_proposals_index[:, None, :].expand(-1, self.num_classes, -1), dim=-1
        )
        res_layer["query_labels"] = top_proposals_class
        res_layer["dense_heatmap"] = dense_heatmap
        return res_layer

    def forward(self, spatial_features_2d: Tensor) -> Dict[str, Tensor]:
        feats = spatial_features_2d.permute(0, 1, 3, 2).contiguous()
        return self.predict(feats)

    def _decode_bbox(
        self,
        heatmap: Tensor,
        rot: Tensor,
        dim: Tensor,
        center: Tensor,
        height: Tensor,
        vel: Tensor,
    ) -> List[Dict[str, Tensor]]:
        post_center_range = torch.tensor(self.post_center_range, device=heatmap.device, dtype=torch.float32)
        final_preds = heatmap.max(1, keepdim=False).indices
        final_scores = heatmap.max(1, keepdim=False).values

        center = center.clone()
        center[:, 0] = center[:, 0] * self.feature_map_stride * self.voxel_size[0] + self.point_cloud_range[0]
        center[:, 1] = center[:, 1] * self.feature_map_stride * self.voxel_size[1] + self.point_cloud_range[1]
        dim = dim.exp()
        height = height - dim[:, 2:3, :] * 0.5
        rots, rotc = rot[:, 0:1, :], rot[:, 1:2, :]
        rot = torch.atan2(rots, rotc)
        final_box_preds = torch.cat([center, height, dim, rot, vel], dim=1).permute(0, 2, 1)

        mask = (final_box_preds[..., :3] >= post_center_range[:3]).all(2)
        mask &= (final_box_preds[..., :3] <= post_center_range[3:]).all(2)
        preds: List[Dict[str, Tensor]] = []
        for i in range(heatmap.shape[0]):
            cmask = mask[i]
            preds.append(
                {
                    "pred_boxes": final_box_preds[i, cmask],
                    "pred_scores": final_scores[i, cmask],
                    "pred_labels": final_preds[i, cmask],
                    "cmask": cmask,
                }
            )
        return preds

    @torch.no_grad()
    def decode(self, preds_dicts: Dict[str, Tensor]) -> Detection3D:
        r"""Decode raw head predictions into raw candidate detections (no NMS).

        Multiplies the sigmoid query scores by the gathered dense-heatmap score, recovers oriented
        boxes, rescores each by predicted IoU (`iou_rectifier`), and filters by the post-center range.
        The full candidate set is returned; the evaluation pipeline applies the per-task circular NMS
        (on the `local_max_classes`, e.g. the nuScenes pedestrian / traffic-cone) via the
        `torch_pointcloud.utils.box3d` utilities (see the benchmark example).

        Args:
            preds_dicts: The dict returned by `forward` / `predict`.

        Returns:
            Packed candidate detections `{"boxes": (K, 7), "scores": (K,), "labels": (K,), "batch": (K,)}`
            (PyG layout).
        """
        batch_size = preds_dicts["heatmap"].shape[0]
        batch_score = preds_dicts["heatmap"].sigmoid()
        one_hot = F.one_hot(preds_dicts["query_labels"], num_classes=self.num_classes).permute(0, 2, 1)
        batch_score = batch_score * preds_dicts["query_heatmap_score"] * one_hot
        batch_iou = (preds_dicts["iou"] + 1) * 0.5

        preds = self._decode_bbox(
            batch_score,
            preds_dicts["rot"],
            preds_dicts["dim"],
            preds_dicts["center"],
            preds_dicts["height"],
            preds_dicts["vel"],
        )

        out_boxes, out_scores, out_labels, out_batch = [], [], [], []
        for i in range(batch_size):
            boxes3d = preds[i]["pred_boxes"]
            scores = preds[i]["pred_scores"]
            labels = preds[i]["pred_labels"]
            cmask = preds[i]["cmask"]
            pred_iou = torch.clamp(batch_iou[i][0][cmask], min=0, max=1.0)
            rectifier = scores.new_full((self.num_classes,), self.iou_rectifier)
            out_boxes.append(boxes3d[:, :7])
            out_scores.append(torch.pow(scores, 1 - rectifier[labels]) * torch.pow(pred_iou, rectifier[labels]))
            out_labels.append(labels.long())
            out_batch.append(torch.full((scores.shape[0],), i, dtype=torch.long, device=scores.device))

        return {
            "boxes": torch.cat(out_boxes),
            "scores": torch.cat(out_scores),
            "labels": torch.cat(out_labels),
            "batch": torch.cat(out_batch),
        }


class LIONDetection(DetectionModel):
    r"""LION: linear group RNN (Mamba) 3D object detector (packed point format).

    Reference: :arxiv: [Liu et al., 2024](https://arxiv.org/abs/2407.18232). Reference implementation:
    :github: [happinesslz/LION](https://github.com/happinesslz/LION) (built on OpenPCDet).

    Points are encoded into voxels by a dynamic mean VFE, processed by a hierarchical sparse backbone
    that serializes voxels into spatially grouped windows and runs a bidirectional Mamba operator
    (the linear group RNN), with periodic 3D voxel-generation / height-merging steps. The one-stride
    output is folded to a dense BEV map, refined by a residual 2D backbone, and decoded by a
    query-based [`TransFusionHead`][torch_pointcloud.models.lion.TransFusionHead].

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
            traffic-cone); see [`TransFusionHead`][torch_pointcloud.models.lion.TransFusionHead].
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
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
    ) -> None:
        super().__init__(in_channels=in_channels, num_classes=num_classes)
        self.voxel_size = tuple(voxel_size)
        self.point_cloud_range = tuple(point_cloud_range)
        grid = [int(round((point_cloud_range[i + 3] - point_cloud_range[i]) / voxel_size[i])) for i in range(3)]
        self.grid_size: Tuple[int, int, int] = (grid[0], grid[1], grid[2])

        self.vfe = DynamicMeanVFE(in_channels, vfe_num_filters, voxel_size, point_cloud_range, self.grid_size)
        self.backbone_3d = LION3DBackbone(
            self.grid_size,
            channels=channels,
            depths=depths,
            window_shape=window_shape,
            group_size=group_size,
            diffusion=diffusion,
            diff_scale=diff_scale,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )
        bev_input_channels = channels * 2
        self.backbone = BaseBEVResBackbone(
            bev_input_channels,
            layer_nums,
            layer_strides,
            num_filters,
            upsample_strides,
            num_upsample_filters,
            norm_kwargs={"eps": 1e-3, "momentum": 0.01},
        )
        self.head = TransFusionHead(
            self.backbone.num_bev_features,
            num_classes,
            self.grid_size,
            point_cloud_range,
            voxel_size,
            feature_map_stride=feature_map_stride,
            local_max_classes=local_max_classes,
        )

    def forward_features(self, x: OptTensor, pos: Tensor, batch: Tensor) -> Tensor:
        batch_size = int(batch.max().item()) + 1
        voxel_features, voxel_indices = self.vfe(pos, x, batch)
        encoded = self.backbone_3d(voxel_features, voxel_indices, batch_size)
        dense = encoded.dense()
        b, c, d, h, w = dense.shape
        bev = dense.view(b, c * d, h, w)
        return self.backbone(bev)

    def forward(self, x: OptTensor, pos: Tensor, batch: Tensor) -> Dict[str, Tensor]:
        return self.head(self.forward_features(x, pos, batch))

    @torch.no_grad()
    def decode(self, out: Dict[str, Tensor]) -> Detection3D:
        r"""Decode a forward output into raw candidate detections (see `TransFusionHead.decode`)."""
        return self.head.decode(out)


@register_model(
    "lion-mamba-happinesslz.nuscenes",
    task="detection",
    weights="hf://torch-pointcloud/lion/lion-mamba-happinesslz.nuscenes.pt",
    transforms=T.Compose([T.Cat(keys=[DataKeys.INTENSITY, DataKeys.TIMESTAMP], dst_key=DataKeys.X, dim=1)]),
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
        d_state=16,
        d_conv=4,
        expand=2,
    ),
)
def lion_mamba_nuscenes(**hparams: Any) -> LIONDetection:
    return LIONDetection(**hparams)
