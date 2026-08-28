"""VoxelNeXt detection model.

{{ paper("2303.11301") }}
"""

from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Sequence, Tuple, TypedDict, Union

import torch
import torch.nn as nn
from torch import Tensor

import torch_pointcloud.transforms as T
from torch_pointcloud.datasets.nuscenes import NUSCENES_DETECTION_CLASSES
from torch_pointcloud.layers import SparseConvBlock
from torch_pointcloud.layers.act import create_act
from torch_pointcloud.layers.norms import create_norm
from torch_pointcloud.utils.data import DataKeys
from torch_pointcloud.utils.imports import _SPCONV_GITHUB_URL, optional_import
from torch_pointcloud.utils.types import Detection3D

from ._base import DetectionModel
from ._registry import WeightsDict, register_model
from .second import SparseBasicBlock

if TYPE_CHECKING:
    import spconv.pytorch as spconv

spconv, _ = optional_import("spconv.pytorch", url=_SPCONV_GITHUB_URL)


class VoxelNeXtHeadOutput(TypedDict):
    r"""Raw per-voxel predictions of the fully sparse VoxelNeXt head.

    Each entry is a list with one tensor per class group. `voxel_indices` are the sparse positions
    $(\text{batch}, y, x)$ of the BEV feature map that every prediction row is anchored to.

    Attributes:
        hm: Per-group classification logits, each of shape $(V, n_g)$ for $n_g$ classes in the group.
        center: Per-group BEV center offset, each $(V, 2)$.
        center_z: Per-group absolute box height, each $(V, 1)$.
        dim: Per-group log box size, each $(V, 3)$.
        rot: Per-group $(\cos\theta, \sin\theta)$, each $(V, 2)$.
        vel: Per-group BEV velocity, each $(V, 2)$.
        voxel_indices: Sparse BEV indices $(V, 3)$ with columns $(\text{batch}, y, x)$.
    """

    hm: List[Tensor]
    center: List[Tensor]
    center_z: List[Tensor]
    dim: List[Tensor]
    rot: List[Tensor]
    vel: List[Tensor]
    voxel_indices: Tensor


class VoxelResBackbone8xVoxelNeXt(nn.Module):
    r"""Fully sparse residual voxel backbone (`VoxelResBackBone8xVoxelNeXt`), $8\times$ BEV stride.

    Extends the SECOND residual backbone with two extra downsampling stages (`conv5`, `conv6`) whose
    outputs are folded back onto the stage-4 sparse tensor (their indices rescaled by $2$ and $4$), so
    the receptive field grows without densifying. The merged 3D sparse tensor is then collapsed along
    height into a 2D BEV sparse tensor (`bev_out`), refined by a 2D `conv_out` + `shared_conv`, and
    returned as a 2D `spconv.SparseConvTensor` for the fully sparse head (no dense BEV map).

    Args:
        in_channels: Input voxel feature channels ($5$ for nuScenes $x, y, z, \text{intensity}, \Delta t$).
        channels: Per-stage channel widths $(c_1, \ldots, c_5)$ for `conv1`-`conv6`.
        out_channels: Output channels of the 2D `conv_out` / `shared_conv`.
        act: Activation type or callable.
        act_kwargs: Extra activation arguments.
        norm: Normalization type or callable.
        norm_kwargs: Extra normalization arguments.
    """

    def __init__(
        self,
        in_channels: int,
        *,
        channels: Sequence[int] = (16, 32, 64, 128, 128),
        out_channels: int = 128,
        act: Union[str, Callable, None] = "relu",
        act_kwargs: Optional[Dict[str, Any]] = None,
        norm: Union[str, Callable, None] = "batch_norm",
        norm_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()
        block_kwargs: Dict[str, Any] = dict(act=act, act_kwargs=act_kwargs, norm=norm, norm_kwargs=norm_kwargs)
        c1, c2, c3, c4, c5 = channels

        self.conv_input = spconv.SparseSequential(
            spconv.SubMConv3d(in_channels, c1, 3, padding=1, bias=False, indice_key="subm1"),
            create_norm(norm, c1, dim=1, **(norm_kwargs or {})),
            create_act(act, **(act_kwargs or {})),
        )
        self.conv1 = nn.ModuleList(
            [SparseBasicBlock(c1, "res1", **block_kwargs), SparseBasicBlock(c1, "res1", **block_kwargs)]
        )
        self.conv2 = nn.ModuleList(
            [
                SparseConvBlock(
                    c1, c2, 3, stride=2, padding=1, indice_key="spconv2", conv_type="spconv", **block_kwargs
                ),
                SparseBasicBlock(c2, "res2", **block_kwargs),
                SparseBasicBlock(c2, "res2", **block_kwargs),
            ]
        )
        self.conv3 = nn.ModuleList(
            [
                SparseConvBlock(
                    c2, c3, 3, stride=2, padding=1, indice_key="spconv3", conv_type="spconv", **block_kwargs
                ),
                SparseBasicBlock(c3, "res3", **block_kwargs),
                SparseBasicBlock(c3, "res3", **block_kwargs),
            ]
        )
        self.conv4 = nn.ModuleList(
            [
                SparseConvBlock(
                    c3, c4, 3, stride=2, padding=1, indice_key="spconv4", conv_type="spconv", **block_kwargs
                ),
                SparseBasicBlock(c4, "res4", **block_kwargs),
                SparseBasicBlock(c4, "res4", **block_kwargs),
            ]
        )
        self.conv5 = nn.ModuleList(
            [
                SparseConvBlock(
                    c4, c5, 3, stride=2, padding=1, indice_key="spconv5", conv_type="spconv", **block_kwargs
                ),
                SparseBasicBlock(c5, "res5", **block_kwargs),
                SparseBasicBlock(c5, "res5", **block_kwargs),
            ]
        )
        self.conv6 = nn.ModuleList(
            [
                SparseConvBlock(
                    c5, c5, 3, stride=2, padding=1, indice_key="spconv6", conv_type="spconv", **block_kwargs
                ),
                SparseBasicBlock(c5, "res6", **block_kwargs),
                SparseBasicBlock(c5, "res6", **block_kwargs),
            ]
        )
        self.conv_out = spconv.SparseSequential(
            spconv.SparseConv2d(c4, out_channels, 3, stride=1, padding=1, bias=False, indice_key="spconv_down2"),
            create_norm(norm, out_channels, dim=1, **(norm_kwargs or {})),
            create_act(act, **(act_kwargs or {})),
        )
        # The reference `shared_conv` (and the whole head) use a plain `nn.BatchNorm1d` (default
        # eps 1e-5), unlike the 3D conv stages whose norm carries the reference eps 1e-3 override.
        self.shared_conv = spconv.SparseSequential(
            spconv.SubMConv2d(out_channels, out_channels, 3, stride=1, padding=1, bias=True),
            create_norm(norm, out_channels, dim=1),
            create_act(act, **(act_kwargs or {})),
        )
        self.out_channels = out_channels

    def bev_out(self, x_conv: "spconv.SparseConvTensor") -> "spconv.SparseConvTensor":
        """Collapses the height axis by summing the features of every voxel sharing the same BEV cell."""
        features_cat = x_conv.features
        indices_cat = x_conv.indices[:, [0, 2, 3]]
        spatial_shape = x_conv.spatial_shape[1:]

        indices_unique, inverse = torch.unique(indices_cat, dim=0, return_inverse=True)
        features_unique = features_cat.new_zeros((indices_unique.shape[0], features_cat.shape[1]))
        features_unique.index_add_(0, inverse, features_cat)

        return spconv.SparseConvTensor(
            features=features_unique,
            indices=indices_unique,
            spatial_shape=spatial_shape,
            batch_size=x_conv.batch_size,
        )

    @staticmethod
    def _run_stage(stage: nn.ModuleList, x: "spconv.SparseConvTensor") -> "spconv.SparseConvTensor":
        for module in stage:
            x = module(x)
        return x

    def forward(self, x: "spconv.SparseConvTensor") -> "spconv.SparseConvTensor":
        x = self.conv_input(x)
        x_conv1 = self._run_stage(self.conv1, x)
        x_conv2 = self._run_stage(self.conv2, x_conv1)
        x_conv3 = self._run_stage(self.conv3, x_conv2)
        x_conv4 = self._run_stage(self.conv4, x_conv3)
        x_conv5 = self._run_stage(self.conv5, x_conv4)
        x_conv6 = self._run_stage(self.conv6, x_conv5)

        x_conv5.indices[:, 1:] *= 2
        x_conv6.indices[:, 1:] *= 4
        x_conv4 = x_conv4.replace_feature(torch.cat([x_conv4.features, x_conv5.features, x_conv6.features]))
        x_conv4.indices = torch.cat([x_conv4.indices, x_conv5.indices, x_conv6.indices])

        out = self.bev_out(x_conv4)
        out = self.conv_out(out)
        return self.shared_conv(out)


class VoxelNeXtSeparateHead(nn.Module):
    r"""Per-group sparse regression head (`SeparateHead`).

    One small sparse 2D conv stack per box attribute (`hm`, `center`, ...). Every attribute has
    `num_conv` $- 1$ hidden `SubMConv2d` blocks (kernel `head_kernel_size`) followed by a $1\times1$
    `SubMConv2d` projecting to the attribute's output channels. Runs directly on the BEV sparse tensor
    and returns per-voxel feature tensors (no dense map).

    Args:
        in_channels: Shared-conv feature channels feeding every attribute stack.
        head_dict: Mapping attribute name -> `{"out_channels": int, "num_conv": int}`.
        head_kernel_size: Kernel size of the hidden `SubMConv2d` blocks.
        use_bias: Whether the hidden conv carries a bias (`USE_BIAS_BEFORE_NORM`).
        act: Activation type or callable.
        act_kwargs: Extra activation arguments.
        norm: Normalization type or callable.
        norm_kwargs: Extra normalization arguments.
    """

    def __init__(
        self,
        in_channels: int,
        head_dict: Dict[str, Dict[str, int]],
        *,
        head_kernel_size: int,
        use_bias: bool,
        act: Union[str, Callable, None] = "relu",
        act_kwargs: Optional[Dict[str, Any]] = None,
        norm: Union[str, Callable, None] = "batch_norm",
        norm_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()
        self.head_names = list(head_dict.keys())
        for name, cfg in head_dict.items():
            out_channels = cfg["out_channels"]
            num_conv = cfg["num_conv"]
            layers: List[nn.Module] = []
            for _ in range(num_conv - 1):
                layers.append(
                    spconv.SparseSequential(
                        spconv.SubMConv2d(
                            in_channels,
                            in_channels,
                            head_kernel_size,
                            padding=head_kernel_size // 2,
                            bias=use_bias,
                            indice_key=name,
                        ),
                        create_norm(norm, in_channels, dim=1, **(norm_kwargs or {})),
                        create_act(act, **(act_kwargs or {})),
                    )
                )
            layers.append(spconv.SubMConv2d(in_channels, out_channels, 1, bias=True, indice_key=name + "out"))
            self.add_module(name, nn.Sequential(*layers))

    def forward(self, x: "spconv.SparseConvTensor") -> Dict[str, Tensor]:
        return {name: self.get_submodule(name)(x).features for name in self.head_names}


class VoxelNeXtHead(nn.Module):
    r"""Fully sparse multi-group detection head (`VoxelNeXtHead`).

    A stack of [`VoxelNeXtSeparateHead`][torch_pointcloud.models.voxelnext.VoxelNeXtSeparateHead]s,
    one per class group, predicting CenterPoint-style attributes directly on the BEV sparse voxels.
    `decode` performs per-group top-$K$ voxel selection, recovers oriented boxes from the sparse
    indices, then per-group 3D NMS.

    Args:
        in_channels: Shared-conv feature channels feeding each separate head.
        class_groups: Class-index groups (0-based), one per separate head.
        head_dict: Per-attribute config shared by every group (`hm` is appended per group).
        head_kernel_size: Kernel size of the hidden head convs.
        num_hm_conv: Number of convs in the classification (`hm`) stack.
        use_bias: Whether hidden head convs carry a bias.
        feature_map_stride: BEV stride relating sparse indices to metric coordinates.
        voxel_size: Voxel size $(v_x, v_y, v_z)$.
        point_cloud_range: Range $(x_\min, y_\min, z_\min, x_\max, y_\max, z_\max)$.
        act: Activation type or callable.
        act_kwargs: Extra activation arguments.
        norm: Normalization type or callable.
        norm_kwargs: Extra normalization arguments.
    """

    def __init__(
        self,
        in_channels: int,
        class_groups: Sequence[Sequence[int]],
        *,
        head_dict: Dict[str, Dict[str, int]],
        head_kernel_size: int,
        num_hm_conv: int,
        use_bias: bool,
        feature_map_stride: int,
        voxel_size: Sequence[float],
        point_cloud_range: Sequence[float],
        act: Union[str, Callable, None] = "relu",
        act_kwargs: Optional[Dict[str, Any]] = None,
        norm: Union[str, Callable, None] = "batch_norm",
        norm_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()
        self.class_groups = [list(group) for group in class_groups]
        self.feature_map_stride = feature_map_stride
        self.voxel_size = tuple(voxel_size)
        self.point_cloud_range = tuple(point_cloud_range)

        self.heads_list = nn.ModuleList()
        for group in self.class_groups:
            group_head_dict = {name: dict(cfg) for name, cfg in head_dict.items()}
            group_head_dict["hm"] = {"out_channels": len(group), "num_conv": num_hm_conv}
            self.heads_list.append(
                VoxelNeXtSeparateHead(
                    in_channels,
                    group_head_dict,
                    head_kernel_size=head_kernel_size,
                    use_bias=use_bias,
                    act=act,
                    act_kwargs=act_kwargs,
                    norm=norm,
                    norm_kwargs=norm_kwargs,
                )
            )

    def forward(self, x: "spconv.SparseConvTensor") -> VoxelNeXtHeadOutput:
        preds: List[Dict[str, Tensor]] = [head(x) for head in self.heads_list]
        return {
            "hm": [p["hm"] for p in preds],
            "center": [p["center"] for p in preds],
            "center_z": [p["center_z"] for p in preds],
            "dim": [p["dim"] for p in preds],
            "rot": [p["rot"] for p in preds],
            "vel": [p["vel"] for p in preds],
            "voxel_indices": x.indices,
        }

    def _decode_group(
        self,
        out: VoxelNeXtHeadOutput,
        group_idx: int,
        batch_size: int,
        top_k: int,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        voxel_indices = out["voxel_indices"]
        batch_idx = voxel_indices[:, 0]
        spatial_indices = voxel_indices[:, 1:].float()

        hm = out["hm"][group_idx].sigmoid()
        center = out["center"][group_idx]
        center_z = out["center_z"][group_idx]
        dim = out["dim"][group_idx].exp()
        rot = out["rot"][group_idx]
        vel = out["vel"][group_idx]
        angle = torch.atan2(rot[:, 1:2], rot[:, 0:1])

        vx, vy, _ = self.voxel_size
        xs = (spatial_indices[:, 1:2] + center[:, 0:1]) * self.feature_map_stride * vx + self.point_cloud_range[0]
        ys = (spatial_indices[:, 0:1] + center[:, 1:2]) * self.feature_map_stride * vy + self.point_cloud_range[1]
        boxes = torch.cat([xs, ys, center_z, dim, angle, vel], dim=1)

        global_classes = torch.as_tensor(self.class_groups[group_idx], device=hm.device)
        out_boxes, out_scores, out_labels, out_batch = [], [], [], []
        for b in range(batch_size):
            mask = batch_idx == b
            scene_hm = hm[mask]
            if scene_hm.numel() == 0:
                continue

            k = min(top_k, scene_hm.shape[0])
            flat_scores, flat_inds = scene_hm.reshape(-1).topk(k)
            voxel_inds = flat_inds // scene_hm.shape[1]
            class_inds = flat_inds % scene_hm.shape[1]
            out_boxes.append(boxes[mask][voxel_inds])
            out_scores.append(flat_scores)
            out_labels.append(global_classes[class_inds])
            out_batch.append(torch.full((k,), b, dtype=torch.long, device=hm.device))

        if not out_boxes:
            # spconv voxel indices are int32; labels and batch must stay int64 like the non-empty path.
            empty = boxes.new_zeros((0, boxes.shape[1]))
            return empty, hm.new_zeros(0), global_classes.new_zeros(0), global_classes.new_zeros(0)
        return torch.cat(out_boxes), torch.cat(out_scores), torch.cat(out_labels), torch.cat(out_batch)

    @torch.no_grad()
    def decode(self, out: VoxelNeXtHeadOutput, *, batch_size: int, top_k: int = 500) -> Detection3D:
        r"""Decode raw sparse head outputs into raw candidate detections (no score threshold or NMS).

        Selects the top-$K$ scoring voxels per group and scene and recovers an oriented box, score and
        label per candidate, along with the predicted BEV velocity $(v_x, v_y)$ under `velocity`. The
        full candidate set is returned; the evaluation pipeline applies score thresholding and per-class
        3D NMS via the `torch_pointcloud.utils.box3d` utilities (see the benchmark example).

        Args:
            out: A `VoxelNeXtHeadOutput` from `forward`.
            batch_size: Number of scenes $B$ in the batch.
            top_k: Per-group, per-scene voxel cap.

        Returns:
            Packed candidate detections `{"boxes": (K, 7), "scores": (K,), "labels": (K,), "batch": (K,)}`
            (PyG layout), plus `"velocity"` $(K, 2)$.
        """
        all_boxes, all_scores, all_labels, all_batch = [], [], [], []
        for group_idx in range(len(self.heads_list)):
            boxes, scores, labels, batch = self._decode_group(out, group_idx, batch_size, top_k)
            all_boxes.append(boxes)
            all_scores.append(scores)
            all_labels.append(labels)
            all_batch.append(batch)

        boxes_all = torch.cat(all_boxes)
        return {
            "boxes": boxes_all[:, :7],
            "scores": torch.cat(all_scores),
            "labels": torch.cat(all_labels),
            "batch": torch.cat(all_batch),
            "velocity": boxes_all[:, 7:9],
        }


class VoxelNeXtDetection(DetectionModel):
    r"""VoxelNeXt fully sparse 3D object detector (packed point format).

    Reference: :arxiv: [Chen et al., 2023](https://arxiv.org/abs/2303.11301).
    Reference implementation: :github: [dvlab-research/VoxelNeXt](https://github.com/dvlab-research/VoxelNeXt)
    (ported via :github: [open-mmlab/OpenPCDet](https://github.com/open-mmlab/OpenPCDet),
    `cbgs_voxel0075_voxelnext`). A residual sparse 3D backbone
    ([`VoxelResBackbone8xVoxelNeXt`][torch_pointcloud.models.voxelnext.VoxelResBackbone8xVoxelNeXt])
    collapses to a 2D BEV sparse tensor that a fully sparse multi-group head
    ([`VoxelNeXtHead`][torch_pointcloud.models.voxelnext.VoxelNeXtHead]) predicts boxes on directly,
    with no dense bird's-eye-view map. Input points carry 5 features ($x, y, z, \text{intensity},
    \Delta t$).

    Args:
        in_channels: Raw point feature channels including xyz (5 for nuScenes).
        num_classes: Number of foreground classes (10 for nuScenes).
        voxel_size: Voxel size $(v_x, v_y, v_z)$.
        point_cloud_range: Range $(x_\min, y_\min, z_\min, x_\max, y_\max, z_\max)$.
        head_class_groups: Class-index groups, one per separate head (e.g. `[[0], [1, 2], ...]`).
        feature_map_stride: BEV feature-map stride of the head.
        channels: Per-stage channel widths of the 3D backbone.
        shared_conv_channels: Output channels of the backbone's 2D shared conv (head input).
        head_kernel_size: Kernel size of the hidden head convs.
        num_hm_conv: Number of convs in the classification (`hm`) stack.
        use_bias_before_norm: Whether hidden head convs carry a bias before their norm.
        act: Activation type or callable for the backbone and head.
        act_kwargs: Extra activation arguments.
        norm: Normalization type or callable for the backbone and head.
        norm_kwargs: Extra normalization arguments.
    """

    def __init__(
        self,
        in_channels: int = 5,
        num_classes: int = 10,
        *,
        voxel_size: Sequence[float] = (0.075, 0.075, 0.2),
        point_cloud_range: Sequence[float] = (-54.0, -54.0, -5.0, 54.0, 54.0, 3.0),
        head_class_groups: Sequence[Sequence[int]],
        feature_map_stride: int,
        channels: Sequence[int] = (16, 32, 64, 128, 128),
        shared_conv_channels: int = 128,
        head_kernel_size: int = 1,
        num_hm_conv: int = 2,
        use_bias_before_norm: bool = True,
        act: Union[str, Callable, None] = "relu",
        act_kwargs: Optional[Dict[str, Any]] = None,
        norm: Union[str, Callable, None] = "batch_norm",
        norm_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(in_channels=in_channels, num_classes=num_classes)
        self.voxel_size = tuple(voxel_size)
        self.point_cloud_range = tuple(point_cloud_range)

        grid = [int(round((point_cloud_range[i + 3] - point_cloud_range[i]) / voxel_size[i])) for i in range(3)]
        self.grid_size: Tuple[int, int, int] = (grid[0], grid[1], grid[2])
        # spconv spatial shape is (z, y, x) with an extra +1 on z (matches the reference `grid[::-1] + [1, 0, 0]`).
        self.sparse_shape: List[int] = [grid[2] + 1, grid[1], grid[0]]
        self.head_class_groups = head_class_groups
        self.feature_map_stride = feature_map_stride
        self.channels = channels
        self.shared_conv_channels = shared_conv_channels
        self.head_kernel_size = head_kernel_size
        self.num_hm_conv = num_hm_conv
        self.use_bias_before_norm = use_bias_before_norm
        self.act = act
        self.act_kwargs = act_kwargs
        self.norm = norm
        self.norm_kwargs = norm_kwargs

        self.backbone_3d = self.configure_backbone_3d()
        self.head = self.configure_head()

    def configure_backbone_3d(self) -> VoxelResBackbone8xVoxelNeXt:
        """Build the fully sparse residual voxel backbone."""
        return VoxelResBackbone8xVoxelNeXt(
            self.in_channels,
            channels=self.channels,
            out_channels=self.shared_conv_channels,
            act=self.act,
            act_kwargs=self.act_kwargs,
            norm=self.norm,
            norm_kwargs=self.norm_kwargs,
        )

    @property
    def num_features(self) -> int:
        """Channel count $C$ of the sparse voxel features entering the head."""
        return self.backbone_3d.out_channels

    def configure_head(self) -> VoxelNeXtHead:
        """Build the fully sparse multi-group detection head."""
        head_dict: Dict[str, Dict[str, int]] = {
            "center": {"out_channels": 2, "num_conv": 2},
            "center_z": {"out_channels": 1, "num_conv": 2},
            "dim": {"out_channels": 3, "num_conv": 2},
            "rot": {"out_channels": 2, "num_conv": 2},
            "vel": {"out_channels": 2, "num_conv": 2},
        }
        # The head uses a plain `nn.BatchNorm1d` (default eps), so the 3D-conv `norm_kwargs` eps
        # override is intentionally not threaded into it (matches the reference `SeparateHead`).
        return VoxelNeXtHead(
            self.shared_conv_channels,
            self.head_class_groups,
            head_dict=head_dict,
            head_kernel_size=self.head_kernel_size,
            num_hm_conv=self.num_hm_conv,
            use_bias=self.use_bias_before_norm,
            feature_map_stride=self.feature_map_stride,
            voxel_size=self.voxel_size,
            point_cloud_range=self.point_cloud_range,
            act=self.act,
            act_kwargs=self.act_kwargs,
            norm=self.norm,
        )

    def forward_features(
        self, voxels: Tensor, pos_voxel: Tensor, voxel_num_points: Tensor, batch: Tensor
    ) -> "spconv.SparseConvTensor":
        voxel_indices = torch.cat([batch.view(-1, 1).to(pos_voxel), pos_voxel], dim=1)
        batch_size = int(batch.max().item()) + 1

        normalizer = torch.clamp_min(voxel_num_points.view(-1, 1), min=1.0).type_as(voxels)
        voxel_features = voxels.sum(dim=1) / normalizer

        sparse_tensor = spconv.SparseConvTensor(
            features=voxel_features,
            indices=voxel_indices.int(),
            spatial_shape=self.sparse_shape,
            batch_size=batch_size,
        )
        return self.backbone_3d(sparse_tensor)

    def forward_head(self, features: "spconv.SparseConvTensor") -> VoxelNeXtHeadOutput:
        return self.head(features)

    def forward(
        self,
        voxels: Tensor,
        pos_voxel: Tensor,
        voxel_num_points: Tensor,
        batch: Tensor,
    ) -> VoxelNeXtHeadOutput:
        features = self.forward_features(voxels, pos_voxel, voxel_num_points, batch)
        return self.forward_head(features)

    @torch.no_grad()
    def decode(self, out: VoxelNeXtHeadOutput, *, top_k: int = 500) -> Detection3D:
        r"""Decode a forward output into raw candidate detections (see `VoxelNeXtHead.decode`)."""
        batch_size = int(out["voxel_indices"][:, 0].max().item()) + 1 if out["voxel_indices"].numel() else 0
        return self.head.decode(out, batch_size=batch_size, top_k=top_k)


@register_model(
    "voxelnext.nuscenes.openpcdet",
    task="detection",
    weights=WeightsDict(
        url="hf://torch-pointcloud/voxelnext.nuscenes.openpcdet/resolve/main/model.safetensors",
        dataset="nuscenes",
        classes=NUSCENES_DETECTION_CLASSES,
        author="openpcdet",
        license="Apache-2.0",
    ),
    transform=T.Compose(
        [
            T.Cat(keys=[DataKeys.INTENSITY, "timestamp"], dst_key=DataKeys.X, dim=1),
            T.HardVoxelize(
                pos_key=DataKeys.POS,
                feat_key=DataKeys.X,
                voxel_size=(0.075, 0.075, 0.2),
                point_cloud_range=(-54.0, -54.0, -5.0, 54.0, 54.0, 3.0),
                max_num_points=10,
                max_num_voxels=160000,
            ),
        ]
    ),
    hparams=dict(
        in_channels=5,
        num_classes=10,
        voxel_size=(0.075, 0.075, 0.2),
        point_cloud_range=(-54.0, -54.0, -5.0, 54.0, 54.0, 3.0),
        # nuScenes 10-class order: car, truck, construction_vehicle, bus, trailer, barrier,
        # motorcycle, bicycle, pedestrian, traffic_cone.
        head_class_groups=[[0], [1, 2], [3, 4], [5], [6, 7], [8, 9]],
        feature_map_stride=8,
        head_kernel_size=1,
        norm_kwargs={"eps": 1e-3, "momentum": 0.01},
    ),
)
def voxelnext_openpcdet_nuscenes(**hparams: Any) -> VoxelNeXtDetection:
    return VoxelNeXtDetection(**hparams)
