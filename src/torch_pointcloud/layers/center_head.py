r"""Dense center-based detection head shared by the BEV detectors (CenterPoint, Voxel Mamba).

The training targets come from [`draw_heatmap_targets`][torch_pointcloud.ops.heatmap.draw_heatmap_targets] and
the loss is [`CenterHeadLoss`][torch_pointcloud.losses.centerpoint.CenterHeadLoss].
"""

from typing import Any, Callable, Dict, List, Optional, Sequence, TypedDict, Union

import torch
import torch.nn as nn
from torch import Tensor
from typing_extensions import NotRequired

from torch_pointcloud.layers.anchors import prediction_branch
from torch_pointcloud.layers.conv2d_blocks import Conv2dBlock
from torch_pointcloud.ops.heatmap import transpose_gather
from torch_pointcloud.utils.types import Detection3D


class CenterHeadOutput(TypedDict):
    r"""Raw dense center-head maps over the BEV feature grid, one entry per class group.

    Attributes:
        heatmap: Per-group class logits, each of shape $(B, n_g, H, W)$ for the $n_g$ classes of the group.
        center: Per-group sub-cell BEV center offset, each $(B, 2, H, W)$.
        center_z: Per-group absolute box height, each $(B, 1, H, W)$.
        dim: Per-group log box size, each $(B, 3, H, W)$.
        rot: Per-group $(\cos\theta, \sin\theta)$, each $(B, 2, H, W)$.
        vel: Per-group BEV velocity, each $(B, 2, H, W)$; only when built with `use_velocity`.
        iou: Per-group IoU-rectification prediction in $[-1, 1]$, each $(B, 1, H, W)$; only when built with
            `use_iou`.
    """

    heatmap: List[Tensor]
    center: List[Tensor]
    center_z: List[Tensor]
    dim: List[Tensor]
    rot: List[Tensor]
    vel: NotRequired[List[Tensor]]
    iou: NotRequired[List[Tensor]]


class SeparateHead(nn.Module):
    r"""Per-attribute conv branches of the dense center head, for one class group.

    One stack of $3\times3$ convs per box attribute over the shared BEV features: `center` $(2)$,
    `center_z` $(1)$, `dim` $(3)$, `rot` $(2)$, the optional `vel` $(2)$ and `iou` $(1)$, and the class
    `heatmap`. The branch widths are fixed by the box parametrization; only the class count, the conv
    depth and the optional branches are configurable.

    Args:
        in_channels: Input channels, also the width of the hidden convs.
        num_classes: Number of classes predicted by the heatmap branch.
        num_layers: Number of convs per branch.
        use_velocity: Add the BEV velocity branch.
        use_iou: Add the IoU-rectification branch.
        init_bias: Bias of the heatmap output conv at initialization.
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
        *,
        num_layers: int = 2,
        use_velocity: bool = False,
        use_iou: bool = False,
        init_bias: float = -2.19,
        act: Union[str, Callable, None] = "relu",
        act_kwargs: Optional[Dict[str, Any]] = None,
        norm: Union[str, Callable, None] = "batch_norm",
        norm_kwargs: Optional[Dict[str, Any]] = None,
        bias: bool = False,
    ) -> None:
        super().__init__()
        self.init_bias = init_bias
        branch_kwargs: Dict[str, Any] = dict(
            num_middle_conv=num_layers - 1,
            num_middle_filter=in_channels,
            act=act,
            act_kwargs=act_kwargs,
            norm=norm,
            norm_kwargs=norm_kwargs,
            bias=bias,
        )
        self.center = prediction_branch(in_channels, 2, **branch_kwargs)
        self.center_z = prediction_branch(in_channels, 1, **branch_kwargs)
        self.dim = prediction_branch(in_channels, 3, **branch_kwargs)
        self.rot = prediction_branch(in_channels, 2, **branch_kwargs)
        self.vel: Optional[nn.Sequential] = None
        if use_velocity:
            self.vel = prediction_branch(in_channels, 2, **branch_kwargs)
        self.iou: Optional[nn.Sequential] = None
        if use_iou:
            self.iou = prediction_branch(in_channels, 1, **branch_kwargs)
        self.heatmap = prediction_branch(in_channels, num_classes, **branch_kwargs)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        r"""Start the heatmap logits at `init_bias`, the other branches at Kaiming-normal weights and zero biases."""
        for name, branch in self.named_children():
            assert isinstance(branch, nn.Sequential)
            if name == "heatmap":
                final = branch[-1]
                assert isinstance(final, nn.Conv2d) and final.bias is not None
                nn.init.constant_(final.bias, self.init_bias)
                continue

            for module in branch.modules():
                if isinstance(module, nn.Conv2d):
                    nn.init.kaiming_normal_(module.weight)
                    if module.bias is not None:
                        nn.init.constant_(module.bias, 0.0)

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        out = {
            "heatmap": self.heatmap(x),
            "center": self.center(x),
            "center_z": self.center_z(x),
            "dim": self.dim(x),
            "rot": self.rot(x),
        }
        if self.vel is not None:
            out["vel"] = self.vel(x)
        if self.iou is not None:
            out["iou"] = self.iou(x)
        return out


class CenterHead(nn.Module):
    r"""Dense center-based detection head: a shared conv, then one `SeparateHead` per class group.

    Reference: :arxiv: [Center-based 3D Object Detection and Tracking](https://arxiv.org/abs/2006.11275) (Yin et al., 2021).

    The peaks of each group's heatmap are the object centers; the box attributes are read at those cells. A
    shared $3\times3$ conv reduces the BEV features, then one
    [`SeparateHead`][torch_pointcloud.layers.center_head.SeparateHead] per group regresses the per-attribute
    maps. `decode` selects the top-$K$ peaks of every group and scene, recovers an oriented box (and velocity)
    per peak and maps the group-local class to its global index.

    Args:
        in_channels: Channels of the input BEV feature map.
        class_groups: Zero-based class-index groups, one separate head each (e.g. `[[0], [1, 2], ...]`).
        feature_map_stride: Stride from the voxel grid to the BEV feature map.
        voxel_size: Voxel size $(v_x, v_y, v_z)$.
        point_cloud_range: Range $(x_\min, y_\min, z_\min, x_\max, y_\max, z_\max)$.
        shared_conv_channels: Channels of the shared conv feeding every separate head.
        num_head_layers: Number of convs per separate-head branch.
        use_velocity: Predict the BEV velocity of every box.
        use_iou: Predict the IoU-rectification score of every box.
        init_bias: Bias of the heatmap output convs at initialization.
        act: Activation of the shared conv and the hidden head convs.
        act_kwargs: Extra activation arguments.
        norm: Normalization of the shared conv and the hidden head convs.
        norm_kwargs: Extra normalization arguments.
        bias: Whether the convs preceding a norm carry a bias.
    """

    def __init__(
        self,
        in_channels: int,
        class_groups: Sequence[Sequence[int]],
        *,
        feature_map_stride: int,
        voxel_size: Sequence[float],
        point_cloud_range: Sequence[float],
        shared_conv_channels: int = 64,
        num_head_layers: int = 2,
        use_velocity: bool = False,
        use_iou: bool = False,
        init_bias: float = -2.19,
        act: Union[str, Callable, None] = "relu",
        act_kwargs: Optional[Dict[str, Any]] = None,
        norm: Union[str, Callable, None] = "batch_norm",
        norm_kwargs: Optional[Dict[str, Any]] = None,
        bias: bool = False,
    ) -> None:
        super().__init__()
        self.class_groups = [list(group) for group in class_groups]
        self.num_classes = sum(len(group) for group in self.class_groups)
        self.feature_map_stride = feature_map_stride
        self.voxel_size = tuple(voxel_size)
        self.point_cloud_range = tuple(point_cloud_range)

        self.shared_conv = Conv2dBlock(
            in_channels,
            shared_conv_channels,
            3,
            padding=1,
            bias=bias,
            act=act,
            act_kwargs=act_kwargs,
            norm=norm,
            norm_kwargs=norm_kwargs,
        )
        self.heads = nn.ModuleList(
            [
                SeparateHead(
                    shared_conv_channels,
                    len(group),
                    num_layers=num_head_layers,
                    use_velocity=use_velocity,
                    use_iou=use_iou,
                    init_bias=init_bias,
                    act=act,
                    act_kwargs=act_kwargs,
                    norm=norm,
                    norm_kwargs=norm_kwargs,
                    bias=bias,
                )
                for group in self.class_groups
            ]
        )

    def forward(self, x: Tensor) -> CenterHeadOutput:
        x = self.shared_conv(x)
        preds: List[Dict[str, Tensor]] = [head(x) for head in self.heads]
        out: CenterHeadOutput = {
            "heatmap": [p["heatmap"] for p in preds],
            "center": [p["center"] for p in preds],
            "center_z": [p["center_z"] for p in preds],
            "dim": [p["dim"] for p in preds],
            "rot": [p["rot"] for p in preds],
        }
        if "vel" in preds[0]:
            out["vel"] = [p["vel"] for p in preds]
        if "iou" in preds[0]:
            out["iou"] = [p["iou"] for p in preds]
        return out

    @torch.no_grad()
    def decode(
        self,
        out: CenterHeadOutput,
        *,
        top_k: int = 500,
        score_threshold: float = 0.0,
        iou_rectifier: Optional[Sequence[float]] = None,
    ) -> Detection3D:
        r"""Decode the raw head maps into raw candidate detections (no NMS).

        For every class group and scene, the top-$K$ cells of the (sigmoid) heatmap over all its classes give the
        candidate centers; the box attributes are gathered at those cells and mapped to world coordinates. With
        `iou_rectifier`, a head built with `use_iou` rescores every candidate by its predicted IoU
        ($s^{1 - r_c} \cdot \text{iou}^{r_c}$ with the per-class exponent $r_c$). The full candidate set is
        returned; the evaluation pipeline applies score thresholding and per-class 3D NMS via the
        `torch_pointcloud.ops.box3d` utilities.

        Args:
            out: A `CenterHeadOutput` from `forward`.
            top_k: Heatmap peaks gathered per group and scene.
            score_threshold: Minimum heatmap score (before rectification) to keep a peak; the non-filtering $0$
                default returns every peak.
            iou_rectifier: Per-class IoU-rectification exponent, one entry per global class, for a head built
                with `use_iou`.

        Returns:
            Packed candidate detections `{"boxes": (K, 7), "scores": (K,), "labels": (K,), "batch": (K,)}`
            (PyG layout), plus `"velocity"` $(K, 2)$ from a head built with `use_velocity`.
        """
        if iou_rectifier is not None and len(iou_rectifier) != self.num_classes:
            raise ValueError(
                f"`iou_rectifier` must have one entry per class ({self.num_classes}), got {len(iou_rectifier)}."
            )
        if iou_rectifier is not None and "iou" not in out:
            raise ValueError("`iou_rectifier` needs the `iou` maps of a head built with `use_iou`.")

        vx, vy, _ = self.voxel_size
        out_boxes, out_scores, out_labels, out_batch, out_velocity = [], [], [], [], []
        for group_idx, group in enumerate(self.class_groups):
            heatmap = out["heatmap"][group_idx].sigmoid()
            batch_size, _, height, width = heatmap.shape
            # One top-k over the flat (class, cell) map equals a per-class top-k followed by a global one.
            scores, inds = torch.topk(heatmap.flatten(1), min(top_k, heatmap[0].numel()))
            classes = torch.div(inds, height * width, rounding_mode="floor")
            inds = inds % (height * width)
            xs = (inds % width).float()
            ys = torch.div(inds, width, rounding_mode="floor").float()

            center = transpose_gather(out["center"][group_idx], inds)
            center_z = transpose_gather(out["center_z"][group_idx], inds)
            dim = transpose_gather(out["dim"][group_idx], inds).exp()
            rot = transpose_gather(out["rot"][group_idx], inds)
            angle = torch.atan2(rot[..., 1], rot[..., 0])

            xs = (xs + center[..., 0]) * self.feature_map_stride * vx + self.point_cloud_range[0]
            ys = (ys + center[..., 1]) * self.feature_map_stride * vy + self.point_cloud_range[1]
            boxes = torch.cat([xs.unsqueeze(-1), ys.unsqueeze(-1), center_z, dim, angle.unsqueeze(-1)], dim=-1)
            labels = torch.as_tensor(group, dtype=torch.long, device=heatmap.device)[classes]

            rescored = scores
            if iou_rectifier is not None:
                iou = torch.clamp((transpose_gather(out["iou"][group_idx], inds) + 1) * 0.5, min=0, max=1).squeeze(-1)
                rectifier = scores.new_tensor(iou_rectifier)[labels]
                rescored = scores.pow(1 - rectifier) * iou.pow(rectifier)
            velocity = transpose_gather(out["vel"][group_idx], inds) if "vel" in out else None

            for b in range(batch_size):
                keep = scores[b] > score_threshold
                out_boxes.append(boxes[b][keep])
                out_scores.append(rescored[b][keep])
                out_labels.append(labels[b][keep])
                out_batch.append(torch.full((int(keep.sum()),), b, dtype=torch.long, device=heatmap.device))
                if velocity is not None:
                    out_velocity.append(velocity[b][keep])

        detections: Detection3D = {
            "boxes": torch.cat(out_boxes),
            "scores": torch.cat(out_scores),
            "labels": torch.cat(out_labels),
            "batch": torch.cat(out_batch),
        }
        if "vel" in out:
            detections["velocity"] = torch.cat(out_velocity)
        return detections
