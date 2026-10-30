"""FCAF3D detection model.

{{ paper("2112.00322") }}
"""

import inspect
import math
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Sequence, Tuple, TypedDict, Union

import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.nn.pool.select.topk import topk

import torch_pointcloud.transforms as T
from torch_pointcloud.datasets.s3dis import S3DIS_DETECTION_CLASSES
from torch_pointcloud.datasets.scannet import SCANNET_DETECTION_CLASSES
from torch_pointcloud.datasets.sunrgbd import SUNRGBD_CLASSES
from torch_pointcloud.layers.act import create_act
from torch_pointcloud.layers.norms import create_norm
from torch_pointcloud.layers.spconv_blocks import SparseBasicBlock, SparseConvBlock, StridedSparseConv3d
from torch_pointcloud.ops.voxelization import trilinear_sparse_devoxelize, voxelize_to_spconv_tensor
from torch_pointcloud.utils.data import DataKeys
from torch_pointcloud.utils.imports import _SPCONV_GITHUB_URL, optional_import
from torch_pointcloud.utils.types import Detection3D

from ._base import DetectionModel
from ._registry import WeightsDict, register_model

if TYPE_CHECKING:
    import spconv.pytorch as spconv
    from spconv.pytorch import SparseConvTensor
    from spconv.pytorch.functional import sparse_add

spconv, _ = optional_import("spconv.pytorch", url=_SPCONV_GITHUB_URL)
sparse_add, _ = optional_import("spconv.pytorch.functional", "sparse_add", url=_SPCONV_GITHUB_URL)


class FCAF3DOutput(TypedDict):
    r"""FCAF3D raw output: one prediction per sparse location of every pyramid level (packed layout).

    Rows are ordered level by level, from the finest level.

    Attributes:
        cls_logits: Class logits, shape $(M, C)$.
        centerness: Centerness logit, shape $(M, 1)$.
        boxes: Predicted boxes $(c_x, c_y, c_z, d_x, d_y, d_z, \theta)$, heading $0$ for axis-aligned boxes, shape
            $(M, 7)$.
        pos: Location of the prediction, in the units of the input positions, shape $(M, 3)$.
        batch: Per-prediction scene index, shape $(M,)$.
        level: Pyramid level of the prediction, shape $(M,)$.
    """

    cls_logits: Tensor
    centerness: Tensor
    boxes: Tensor
    pos: Tensor
    batch: Tensor
    level: Tensor


class FCAF3DEncoder(nn.Module):
    r"""Residual encoder over sparse voxels: a strided stem and stages of residual blocks, each halving the resolution.

    The stem is a strided $3 \times 3 \times 3$ convolution, a normalization, an activation and a $2 \times 2 \times 2$
    max pooling. Every stage then opens with a strided `SparseBasicBlock` and stacks submanifold ones. All the strided
    layers keep one output voxel per occupied parent cell (see `StridedSparseConv3d`).

    Args:
        in_channels: Number of input voxel feature channels.
        stem_channels: Output channels of the stem.
        channels: Output channels of every stage.
        depths: Number of residual blocks in every stage.
        stem_norm: Normalization of the stem, passed to `create_norm`. A norm whose `forward` takes `batch` (e.g.
            `instance_norm`) normalizes every scene separately.
        stem_norm_kwargs: Extra keyword arguments for the stem normalization.
        act: Activation passed to `create_act`.
        act_kwargs: Extra keyword arguments for the activation.
        norm: Normalization of the residual blocks, passed to `create_norm`.
        norm_kwargs: Extra keyword arguments for the residual block normalization.

    Example:
        ```python
        import spconv.pytorch as spconv
        import torch
        from torch_pointcloud.models.fcaf3d import FCAF3DEncoder

        encoder = FCAF3DEncoder(3, 64, channels=(64, 128), depths=(2, 2)).cuda()
        indices = torch.randint(0, 64, (1000, 4), dtype=torch.int32, device="cuda")
        indices[:, 0] = 0
        indices = torch.unique(indices, dim=0)
        x = spconv.SparseConvTensor(torch.randn(len(indices), 3, device="cuda"), indices, [64, 64, 64], 1)
        stages = encoder(x)  # strides 8 and 16
        ```
    """

    def __init__(
        self,
        in_channels: int,
        stem_channels: int,
        *,
        channels: Sequence[int],
        depths: Sequence[int],
        stem_norm: Union[str, Callable, None] = "batch_norm",
        stem_norm_kwargs: Optional[Dict[str, Any]] = None,
        act: Union[str, Callable, None] = "relu",
        act_kwargs: Optional[Dict[str, Any]] = None,
        norm: Union[str, Callable, None] = "batch_norm",
        norm_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        if len(channels) != len(depths):
            raise ValueError(f"channels and depths must have the same length, got {len(channels)} and {len(depths)}.")

        super().__init__()
        self.in_channels = in_channels
        self.stem_channels = stem_channels
        self.channels = tuple(channels)
        self.depths = tuple(depths)

        self.conv1 = StridedSparseConv3d(in_channels, stem_channels, kernel_size=3, stride=2)
        self.norm1 = create_norm(stem_norm, stem_channels, **(stem_norm_kwargs or {})) or nn.Identity()
        self.supports_norm_batch = "batch" in inspect.signature(self.norm1.forward).parameters
        self.act = create_act(act, **(act_kwargs or {})) or nn.Identity()
        self.pool = spconv.SparseMaxPool3d(2, 2)

        self.stages = nn.ModuleList()
        width = stem_channels
        for i, (out_channels, depth) in enumerate(zip(channels, depths)):
            blocks = [
                SparseBasicBlock(
                    width if j == 0 else out_channels,
                    out_channels,
                    stride=2 if j == 0 else 1,
                    indice_key=f"subm{i}",
                    act=act,
                    act_kwargs=act_kwargs,
                    norm=norm,
                    norm_kwargs=norm_kwargs,
                )
                for j in range(depth)
            ]
            self.stages.append(spconv.SparseSequential(*blocks))
            width = out_channels

    @property
    def strides(self) -> Tuple[int, ...]:
        """Total stride of every stage output with respect to the input voxels."""
        return tuple(2 ** (i + 3) for i in range(len(self.stages)))

    def forward(self, x: "SparseConvTensor") -> List["SparseConvTensor"]:
        x = self.conv1(x)
        if self.supports_norm_batch:
            features = self.norm1(x.features, x.indices[:, 0].long(), x.batch_size)
        else:
            features = self.norm1(x.features)
        x = x.replace_feature(self.act(features))
        x = self.pool(x)

        stages = []
        for stage in self.stages:
            x = stage(x)
            stages.append(x)
        return stages


class FCAF3DHead(nn.Module):
    r"""FCAF3D neck and head: a pruned generative feature pyramid with anchor-free predictions at every level.

    From the coarsest encoder stage down, a transposed convolution generates the $2 \times 2 \times 2$ children of every
    voxel, the encoder stage of that resolution is added on the union of both voxel sets, and only the
    `max_voxels_per_scene` voxels of every scene whose parents score highest are kept. Every level then predicts, per
    voxel, class logits, a centerness logit and the box regression; the class logits of a level, maxed over the
    classes and interpolated at the voxels of the next finer level, are the scores of its pruning. The neck and the
    head cannot be separated, as pruning a level needs the predictions of the level above.

    Args:
        in_channels: Channels of every encoder stage, from the finest.
        channels: Channels of the per-level features entering the predictions.
        num_classes: Number of object classes.
        num_box_outputs: Box regression channels: $6$ face distances, or $8$ with the heading code.
        max_voxels_per_scene: Voxels kept per scene at every level but the coarsest.
        act: Activation passed to `create_act`.
        act_kwargs: Extra keyword arguments for the activation.
        norm: Normalization passed to `create_norm`.
        norm_kwargs: Extra keyword arguments for the normalization.

    Example:
        ```python
        import spconv.pytorch as spconv
        import torch
        from torch_pointcloud.models.fcaf3d import FCAF3DEncoder, FCAF3DHead

        channels = (64, 128, 256, 512)
        encoder = FCAF3DEncoder(3, 64, channels=channels, depths=(1, 1, 1, 1)).cuda().eval()
        head = FCAF3DHead(channels, 128, 18, num_box_outputs=6, max_voxels_per_scene=100000).cuda().eval()
        indices = torch.randint(0, 256, (20000, 4), dtype=torch.int32, device="cuda")
        indices[:, 0] = 0
        indices = torch.unique(indices, dim=0)
        x = spconv.SparseConvTensor(torch.randn(len(indices), 3, device="cuda"), indices, [256, 256, 256], 1)
        with torch.no_grad():
            out = head(encoder(x))
        out["cls_logits"].shape  # one row per voxel of every level, from the finest
        ```
    """

    def __init__(
        self,
        in_channels: Sequence[int],
        channels: int,
        num_classes: int,
        *,
        num_box_outputs: int,
        max_voxels_per_scene: int,
        act: Union[str, Callable, None] = "elu",
        act_kwargs: Optional[Dict[str, Any]] = None,
        norm: Union[str, Callable, None] = "batch_norm",
        norm_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        if num_box_outputs not in (6, 8):
            raise ValueError(f"num_box_outputs must be 6 or 8, got {num_box_outputs}.")

        super().__init__()
        self.in_channels = tuple(in_channels)
        self.channels = channels
        self.num_box_outputs = num_box_outputs
        self.max_voxels_per_scene = max_voxels_per_scene
        block_kwargs: Dict[str, Any] = dict(act=act, act_kwargs=act_kwargs, norm=norm, norm_kwargs=norm_kwargs)

        self.up_blocks = nn.ModuleList(
            spconv.SparseSequential(
                SparseConvBlock(
                    in_channels[i + 1],
                    in_channels[i],
                    2,
                    stride=2,
                    indice_key=None,
                    conv_type="transposeconv",
                    **block_kwargs,
                ),
                SparseConvBlock(in_channels[i], in_channels[i], 3, indice_key=f"up{i}", **block_kwargs),
            )
            for i in range(len(in_channels) - 1)
        )
        self.out_blocks = nn.ModuleList(
            SparseConvBlock(in_channels[i], channels, 3, indice_key=f"out{i}", **block_kwargs)
            for i in range(len(in_channels))
        )
        self.center = nn.Linear(channels, 1, bias=False)
        self.box = nn.Linear(channels, num_box_outputs, bias=False)
        self.scales = nn.Parameter(torch.ones(len(in_channels)))
        for layer in (self.center, self.box):
            nn.init.normal_(layer.weight, std=0.01)
        self.reset_classifier(num_classes)

    def reset_classifier(self, num_classes: int) -> None:
        r"""Replace the class branch with one of `num_classes` outputs, initialized to a $1\%$ foreground prior."""
        self.num_classes = num_classes
        self.cls = nn.Linear(self.channels, num_classes)
        nn.init.normal_(self.cls.weight, std=0.01)
        nn.init.constant_(self.cls.bias, -math.log(99))

    def prune(self, x: "SparseConvTensor", scores: "SparseConvTensor") -> "SparseConvTensor":
        r"""Keep the `max_voxels_per_scene` voxels of every scene of `x` scoring highest under the coarser `scores`."""
        with torch.no_grad():
            indices = x.indices.long()
            coarse = scores.indices.long()
            score = trilinear_sparse_devoxelize(
                scores.features,
                coarse[:, 1:],
                coarse[:, 0],
                indices[:, 1:].to(scores.features.dtype) / 2,
                indices[:, 0],
            )
            keep = topk(score.view(-1), self.max_voxels_per_scene, indices[:, 0])
            mask = torch.zeros(len(indices), dtype=torch.bool, device=indices.device)
            mask[keep] = True
        pruned = x.replace_feature(x.features[mask])
        pruned.indices = x.indices[mask]
        # The cached index pairs describe the voxels before pruning (as in spconv's `select_by_index`).
        pruned.indice_dict = {}
        return pruned

    def forward(self, stages: Sequence["SparseConvTensor"]) -> Dict[str, Tensor]:
        """Predict at every pyramid level.

        Args:
            stages: Encoder stages, from the finest, with the channels of `in_channels`.

        Returns:
            The packed `cls_logits`, `centerness` and `box` predictions, the `[batch, x, y, z]` voxel `indices` of
            each level's grid and the `level` of every row, from the finest level.
        """
        num_levels = len(stages)
        outputs: List[Dict[str, Tensor]] = []
        x = stages[-1]
        scores: Optional["SparseConvTensor"] = None
        for i in reversed(range(num_levels)):
            if i < num_levels - 1:
                assert scores is not None
                x = self.up_blocks[i](x)
                x = sparse_add(stages[i], x)
                x = self.prune(x, scores)

            out = self.out_blocks[i](x)
            features = out.features
            cls = self.cls(features)
            box = self.box(features)
            distances = torch.exp(box[:, :6] * self.scales[i])
            outputs.append(
                {
                    "cls_logits": cls,
                    "centerness": self.center(features),
                    "box": torch.cat([distances, box[:, 6:]], dim=1),
                    "indices": out.indices,
                    "level": torch.full((len(features),), i, dtype=torch.long, device=features.device),
                }
            )
            scores = spconv.SparseConvTensor(
                cls.max(dim=1, keepdim=True).values,
                out.indices,
                out.spatial_shape,
                out.batch_size,
            )

        outputs = outputs[::-1]
        return {key: torch.cat([output[key] for output in outputs]) for key in outputs[0]}


class FCAF3DDetection(DetectionModel):
    r"""FCAF3D fully convolutional anchor-free 3D object detector (packed point format).

    Reference: :arxiv: [FCAF3D: Fully Convolutional Anchor-Free 3D Object Detection](https://arxiv.org/abs/2112.00322)
    (Rukhovich et al., 2022).

    The points are voxelized at `voxel_size` (the first point of a voxel gives its features) and encoded by a
    [`FCAF3DEncoder`][torch_pointcloud.models.fcaf3d.FCAF3DEncoder] into four stages of strides $8$ to
    $64$, which [`FCAF3DHead`][torch_pointcloud.models.fcaf3d.FCAF3DHead] turns into a pruned pyramid of per-voxel
    predictions. Every voxel regresses the distances to the faces of its box, and for oriented boxes a heading code
    of its aspect ratio that a quarter turn leaves invariant.

    !!! note
        Run the model in float32, or in float16 after `model.half()` with float16 features `x`. Under
        `torch.autocast`, the sparse convolutions of the encoder fail in spconv's kernel tuner.

    Args:
        in_channels: Number of point feature channels (e.g. $3$ for RGB).
        num_classes: Number of object classes.
        voxel_size: Edge length of the input voxels, in the units of `pos`.
        oriented_boxes: Whether the boxes have a heading; axis-aligned boxes have heading $0$.
        max_voxels_per_scene: Voxels kept per scene at every pyramid level but the coarsest.
        stem_channels: Output channels of the encoder stem.
        encoder_channels: Output channels of every encoder stage.
        encoder_depths: Number of residual blocks in every encoder stage.
        head_channels: Channels of the per-level features entering the predictions.
        stem_norm: Normalization of the encoder stem.
        stem_norm_kwargs: Extra keyword arguments for the stem normalization; `None` gives the affine instance norm
            of the released weights, $\epsilon = 10^{-8}$.
        norm: Normalization of the encoder blocks and of the head.
        norm_kwargs: Extra keyword arguments for the normalization.

    Example:
        ```python
        import torch
        from torch_pointcloud.models import create_model

        model = create_model("fcaf3d.scannet.mmdet3d", task="detection", pretrained=True).cuda().eval()
        pos = torch.rand(20000, 3, device="cuda") * 4
        x = torch.rand(20000, 3, device="cuda")
        batch = torch.zeros(20000, dtype=torch.long, device="cuda")
        detections = model.decode(model(x, pos, batch))
        ```
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        *,
        voxel_size: float,
        oriented_boxes: bool,
        max_voxels_per_scene: int,
        stem_channels: int = 64,
        encoder_channels: Sequence[int] = (64, 128, 256, 512),
        encoder_depths: Sequence[int] = (3, 4, 6, 3),
        head_channels: int = 128,
        stem_norm: Union[str, Callable, None] = "instance_norm",
        stem_norm_kwargs: Optional[Dict[str, Any]] = None,
        norm: Union[str, Callable, None] = "batch_norm",
        norm_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(in_channels=in_channels, num_classes=num_classes)
        self.voxel_size = voxel_size
        self.oriented_boxes = oriented_boxes
        self.max_voxels_per_scene = max_voxels_per_scene
        self.stem_channels = stem_channels
        self.encoder_channels = tuple(encoder_channels)
        self.encoder_depths = tuple(encoder_depths)
        self.head_channels = head_channels
        self.stem_norm = stem_norm
        self.stem_norm_kwargs = stem_norm_kwargs if stem_norm_kwargs is not None else {"eps": 1e-8, "affine": True}
        self.norm = norm
        self.norm_kwargs = norm_kwargs

        self.encoder = self.configure_encoder()
        self.head = self.configure_head()

    def configure_encoder(self) -> FCAF3DEncoder:
        """Build the sparse residual encoder."""
        return FCAF3DEncoder(
            self.in_channels,
            self.stem_channels,
            channels=self.encoder_channels,
            depths=self.encoder_depths,
            stem_norm=self.stem_norm,
            stem_norm_kwargs=self.stem_norm_kwargs,
            norm=self.norm,
            norm_kwargs=self.norm_kwargs,
        )

    def configure_head(self) -> FCAF3DHead:
        """Build the pruned feature pyramid and its predictions."""
        return FCAF3DHead(
            self.encoder_channels,
            self.head_channels,
            self.num_classes,
            num_box_outputs=8 if self.oriented_boxes else 6,
            max_voxels_per_scene=self.max_voxels_per_scene,
            norm=self.norm,
            norm_kwargs=self.norm_kwargs,
        )

    @property
    def num_features(self) -> int:
        """Channel count $C$ of the coarsest encoder stage."""
        return self.encoder_channels[-1]

    @property
    def strides(self) -> Tuple[int, ...]:
        """Total stride of every pyramid level with respect to the input voxels."""
        return self.encoder.strides

    def reset_classifier(self, num_classes: int) -> None:
        self.num_classes = num_classes
        self.head.reset_classifier(num_classes)

    def forward_features(self, x: Tensor, pos: Tensor, batch: Tensor) -> Tuple[List["SparseConvTensor"], Tensor]:
        r"""Voxelize the points and encode them.

        Args:
            x: Point features $(N, C_\text{in})$.
            pos: Point positions $(N, 3)$.
            batch: Per-point scene index $(N,)$.

        Returns:
            The encoder stages, from the finest, and the `origin` $(3,)$ of their voxel grid: voxel $v$ of a stage of
            stride $s$ covers the input voxels from $s v + \text{origin}$.
        """
        sparse, origin = voxelize_to_spconv_tensor(x, pos, batch, self.voxel_size, self.encoder.strides[-1])
        stages = self.encoder(sparse)
        return stages, origin

    def forward_head(self, stages: Sequence["SparseConvTensor"], origin: Tensor) -> FCAF3DOutput:
        r"""Predict boxes from the encoder stages.

        Args:
            stages: Encoder stages from `forward_features`, from the finest.
            origin: Voxel-grid origin from `forward_features`, shape $(3,)$.

        Returns:
            The raw per-voxel predictions of every pyramid level.
        """
        out = self.head(stages)
        level = out["level"]
        strides = torch.tensor(self.strides, device=level.device)[level].unsqueeze(1)
        indices = out["indices"].long()
        pos = (indices[:, 1:] * strides + origin) * self.voxel_size

        return {
            "cls_logits": out["cls_logits"],
            "centerness": out["centerness"],
            "boxes": self.decode_boxes(out["box"], pos),
            "pos": pos,
            "batch": indices[:, 0],
            "level": level,
        }

    def forward(self, x: Tensor, pos: Tensor, batch: Tensor) -> FCAF3DOutput:
        stages, origin = self.forward_features(x, pos, batch)
        return self.forward_head(stages, origin)

    def decode_boxes(self, box: Tensor, pos: Tensor) -> Tensor:
        r"""Turn the box regression of every location into a box.

        The regression is the distances from the location to the faces of its box $(x_-, x_+, y_-, y_+, z_-, z_+)$,
        followed for oriented boxes by the heading code $(\ln q \sin 2\theta, \ln q \cos 2\theta)$ of the box aspect
        ratio $q = d_y / d_x$, which describes $(d_x, d_y, \theta)$ and $(d_y, d_x, \theta + \pi / 2)$ alike. An
        oriented box keeps the center and the footprint $d_x + d_y$ of the axis-aligned box of its face distances.

        Args:
            box: Box regression, shape $(M, 6)$ or $(M, 8)$.
            pos: Locations, shape $(M, 3)$.

        Returns:
            The boxes $(c_x, c_y, c_z, d_x, d_y, d_z, \theta)$, heading $0$ for axis-aligned boxes, shape $(M, 7)$.
        """
        center = pos + (box[:, 1:6:2] - box[:, 0:6:2]) / 2
        size = box[:, 0:6:2] + box[:, 1:6:2]
        heading = torch.zeros_like(center[:, :1])
        if self.oriented_boxes:
            footprint = size[:, 0] + size[:, 1]
            ratio = torch.exp(torch.sqrt(box[:, 6] ** 2 + box[:, 7] ** 2))
            width = footprint / (1 + ratio)
            size = torch.stack([width, width * ratio, size[:, 2]], dim=1)
            heading = 0.5 * torch.atan2(box[:, 6:7], box[:, 7:8])
        return torch.cat([center, size, heading], dim=1)

    def decode(self, out: FCAF3DOutput) -> Detection3D:
        r"""Score the boxes of a forward output (no score threshold, top-k or NMS).

        Every location is scored per class by its class probability times its centerness, and labelled by its best
        class. The authors' evaluation keeps the $1000$ best `scores` of every scene and level, then every (box,
        class) pair whose class score exceeds $0.01$, suppressed per class at a bird's-eye-view IoU of $0.5$ (`nms3d`
        with `rotated=True`).

        Args:
            out: The raw output of `forward`.

        Returns:
            Packed detections `{"boxes", "scores", "labels", "batch", "class_scores"}` with one row per output row,
            the per-class scores in `class_scores`.

        Shape:
            - boxes: $(M, 7)$
            - scores / labels / batch: $(M,)$
            - class_scores: $(M, C)$
        """
        class_scores = out["cls_logits"].sigmoid() * out["centerness"].sigmoid()
        scores, labels = class_scores.max(dim=1)
        return {
            "boxes": out["boxes"],
            "scores": scores,
            "labels": labels,
            "batch": out["batch"],
            "class_scores": class_scores,
        }


@register_model(
    "fcaf3d.scannet.mmdet3d",
    task="detection",
    weights=WeightsDict(
        url="hf://torch-pointcloud/fcaf3d.scannet.mmdet3d/resolve/7e2f790ff8a07439ace573f926dff71e5d5aa2b0/model.safetensors",
        dataset="scannet",
        metrics={"mAP@0.25": 69.55, "mAP@0.5": 55.10},
        classes=SCANNET_DETECTION_CLASSES,
        author="mmdet3d",
        license="Apache-2.0",
    ),
    transform=T.Compose(
        [
            T.RandomSample(
                keys=[DataKeys.POS, DataKeys.COLOR, DataKeys.NORMAL, DataKeys.SEGMENT, DataKeys.INSTANCE],
                num_samples=100000,
                allow_missing_keys=True,
            ),
            T.Normalize(keys=DataKeys.COLOR, mean=[0.0, 0.0, 0.0], std=[255.0, 255.0, 255.0]),
            T.Cat(keys=[DataKeys.COLOR], dst_key=DataKeys.X, dim=1),
        ]
    ),
    hparams=dict(
        in_channels=3,
        num_classes=18,
        voxel_size=0.01,
        oriented_boxes=False,
        max_voxels_per_scene=100000,
    ),
)
def fcaf3d_scannet_mmdet3d(**hparams: Any) -> FCAF3DDetection:
    return FCAF3DDetection(**hparams)


@register_model(
    "fcaf3d.sunrgbd.mmdet3d",
    task="detection",
    weights=WeightsDict(
        url="hf://torch-pointcloud/fcaf3d.sunrgbd.mmdet3d/resolve/17c47353b0e75a84b9c08a4fed2e27995b1d81b1/model.safetensors",
        dataset="sunrgbd",
        metrics={"mAP@0.25": 64.70, "mAP@0.5": 47.93},
        classes=SUNRGBD_CLASSES,
        author="mmdet3d",
        license="Apache-2.0",
    ),
    transform=T.Compose(
        [
            T.RandomSample(
                keys=[DataKeys.POS, DataKeys.COLOR, DataKeys.NORMAL, DataKeys.SEGMENT, DataKeys.INSTANCE],
                num_samples=100000,
                allow_missing_keys=True,
            ),
            T.Normalize(keys=DataKeys.COLOR, mean=[0.0, 0.0, 0.0], std=[255.0, 255.0, 255.0]),
            T.Cat(keys=[DataKeys.COLOR], dst_key=DataKeys.X, dim=1),
        ]
    ),
    hparams=dict(
        in_channels=3,
        num_classes=10,
        voxel_size=0.01,
        oriented_boxes=True,
        max_voxels_per_scene=100000,
    ),
)
def fcaf3d_sunrgbd_mmdet3d(**hparams: Any) -> FCAF3DDetection:
    return FCAF3DDetection(**hparams)


@register_model(
    "fcaf3d.s3dis-area5.mmdet3d",
    task="detection",
    weights=WeightsDict(
        url="hf://torch-pointcloud/fcaf3d.s3dis-area5.mmdet3d/resolve/9f8af3ec43ebcd2ee45afad7030db8cdb37b4e3d/model.safetensors",
        dataset="s3dis-area5",
        metrics={"mAP@0.25": 66.31, "mAP@0.5": 46.71},
        classes=S3DIS_DETECTION_CLASSES,
        author="mmdet3d",
        license="Apache-2.0",
    ),
    transform=T.Compose(
        [
            T.RandomSample(
                keys=[DataKeys.POS, DataKeys.COLOR, DataKeys.NORMAL, DataKeys.SEGMENT, DataKeys.INSTANCE],
                num_samples=100000,
                allow_missing_keys=True,
            ),
            T.Normalize(keys=DataKeys.COLOR, mean=[0.0, 0.0, 0.0], std=[255.0, 255.0, 255.0]),
            T.Cat(keys=[DataKeys.COLOR], dst_key=DataKeys.X, dim=1),
        ]
    ),
    hparams=dict(
        in_channels=3,
        num_classes=5,
        voxel_size=0.01,
        oriented_boxes=False,
        max_voxels_per_scene=100000,
    ),
)
def fcaf3d_s3dis_area5_mmdet3d(**hparams: Any) -> FCAF3DDetection:
    return FCAF3DDetection(**hparams)
