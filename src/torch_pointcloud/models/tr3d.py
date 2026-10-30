"""TR3D detection model.

{{ paper("2302.02858") }}
"""

import math
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Sequence, Tuple, TypedDict, Union

import torch
import torch.nn as nn
from torch import Tensor

import torch_pointcloud.transforms as T
from torch_pointcloud.datasets.s3dis import S3DIS_DETECTION_CLASSES
from torch_pointcloud.datasets.scannet import SCANNET_DETECTION_CLASSES
from torch_pointcloud.datasets.sunrgbd import SUNRGBD_CLASSES
from torch_pointcloud.layers.spconv_blocks import SparseConvBlock
from torch_pointcloud.ops.voxelization import voxelize_to_spconv_tensor
from torch_pointcloud.utils.data import DataKeys
from torch_pointcloud.utils.imports import _SPCONV_GITHUB_URL, optional_import
from torch_pointcloud.utils.types import Detection3D

from ._base import DetectionModel
from ._registry import WeightsDict, register_model
from .fcaf3d import FCAF3DEncoder

if TYPE_CHECKING:
    import spconv.pytorch as spconv
    from spconv.pytorch import SparseConvTensor
    from spconv.pytorch.functional import sparse_add

spconv, _ = optional_import("spconv.pytorch", url=_SPCONV_GITHUB_URL)
sparse_add, _ = optional_import("spconv.pytorch.functional", "sparse_add", url=_SPCONV_GITHUB_URL)


class TR3DOutput(TypedDict):
    r"""TR3D raw output: one prediction per sparse location of every pyramid level (packed layout).

    Rows are ordered level by level, from the finest level.

    Attributes:
        cls_logits: Class logits, shape $(M, C)$.
        boxes: Predicted boxes $(c_x, c_y, c_z, d_x, d_y, d_z, \theta)$, heading $0$ for axis-aligned boxes, shape
            $(M, 7)$.
        pos: Location of the prediction, in the units of the input positions, shape $(M, 3)$.
        batch: Per-prediction scene index, shape $(M,)$.
        level: Pyramid level of the prediction, shape $(M,)$.
    """

    cls_logits: Tensor
    boxes: Tensor
    pos: Tensor
    batch: Tensor
    level: Tensor


class TR3DNeck(nn.Module):
    r"""Sparse feature pyramid of TR3D: one output level per input stage but the coarsest.

    From the coarsest stage down, a strided transposed $3 \times 3 \times 3$ convolution generates the finer voxels
    around every voxel, the stage of that resolution is added on the union of both voxel sets, and a lateral block
    then an output block follow. All blocks are convolution, normalization and activation.

    Args:
        in_channels: Channels of every input stage, from the finest.
        out_channels: Channels of every output level.
        act: Activation passed to `create_act`.
        act_kwargs: Extra keyword arguments for the activation.
        norm: Normalization passed to `create_norm`.
        norm_kwargs: Extra keyword arguments for the normalization.

    Example:
        ```python
        import spconv.pytorch as spconv
        import torch
        from torch_pointcloud.models.fcaf3d import FCAF3DEncoder
        from torch_pointcloud.models.tr3d import TR3DNeck

        encoder = FCAF3DEncoder(3, 64, channels=(64, 128, 128, 128), depths=(1, 1, 1, 1)).cuda().eval()
        neck = TR3DNeck((128, 128, 128), 128).cuda().eval()
        indices = torch.randint(0, 256, (20000, 4), dtype=torch.int32, device="cuda")
        indices[:, 0] = 0
        indices = torch.unique(indices, dim=0)
        x = spconv.SparseConvTensor(torch.randn(len(indices), 3, device="cuda"), indices, [256, 256, 256], 1)
        with torch.no_grad():
            levels = neck(encoder(x)[1:])  # strides 16 and 32
        ```
    """

    def __init__(
        self,
        in_channels: Sequence[int],
        out_channels: int,
        *,
        act: Union[str, Callable, None] = "relu",
        act_kwargs: Optional[Dict[str, Any]] = None,
        norm: Union[str, Callable, None] = "batch_norm",
        norm_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()
        self.in_channels = tuple(in_channels)
        self.out_channels = out_channels
        block_kwargs: Dict[str, Any] = dict(act=act, act_kwargs=act_kwargs, norm=norm, norm_kwargs=norm_kwargs)
        num_levels = len(in_channels) - 1

        self.up_blocks = nn.ModuleList(
            SparseConvBlock(
                in_channels[i + 1],
                in_channels[i],
                3,
                stride=2,
                padding=1,
                indice_key=None,
                conv_type="transposeconv",
                **block_kwargs,
            )
            for i in range(num_levels)
        )
        self.lateral_blocks = nn.ModuleList(
            SparseConvBlock(in_channels[i], in_channels[i], 3, indice_key=f"lateral{i}", **block_kwargs)
            for i in range(num_levels)
        )
        self.out_blocks = nn.ModuleList(
            SparseConvBlock(in_channels[i], out_channels, 3, indice_key=f"lateral{i}", **block_kwargs)
            for i in range(num_levels)
        )

    def forward(self, stages: Sequence["SparseConvTensor"]) -> List["SparseConvTensor"]:
        x = stages[-1]
        levels = []
        for i in reversed(range(len(stages) - 1)):
            x = self.up_blocks[i](x).shadow_copy()
            x.spatial_shape = stages[i].spatial_shape
            x = sparse_add(stages[i], x)
            x = self.lateral_blocks[i](x)
            levels.append(self.out_blocks[i](x))
        return levels[::-1]


class TR3DDetection(DetectionModel):
    r"""TR3D fully convolutional 3D object detector (packed point format).

    Reference: :arxiv: [TR3D: Towards Real-Time Indoor 3D Object Detection](https://arxiv.org/abs/2302.02858)
    (Rukhovich et al., 2023).

    The points are voxelized at `voxel_size` (the first point of a voxel gives its features) and encoded by a
    [`FCAF3DEncoder`][torch_pointcloud.models.fcaf3d.FCAF3DEncoder] into four stages of strides $8$ to
    $64$. A [`TR3DNeck`][torch_pointcloud.models.tr3d.TR3DNeck] over the last three stages yields two levels, of
    strides $16$ and $32$, where every voxel predicts class logits, the offset to its box center and the box size,
    and for oriented boxes a heading code of its aspect ratio that a quarter turn leaves invariant.

    !!! note
        Run the model in float32, or in float16 after `model.half()` with float16 features `x`. Under
        `torch.autocast`, the sparse convolutions of the encoder fail in spconv's kernel tuner.

    Args:
        in_channels: Number of point feature channels (e.g. $3$ for RGB).
        num_classes: Number of object classes.
        voxel_size: Edge length of the input voxels, in the units of `pos`.
        oriented_boxes: Whether the boxes have a heading; axis-aligned boxes have heading $0$.
        stem_channels: Output channels of the encoder stem.
        encoder_channels: Output channels of every encoder stage.
        encoder_depths: Number of residual blocks in every encoder stage.
        neck_channels: Channels of every neck level.
        stem_norm: Normalization of the encoder stem.
        stem_norm_kwargs: Extra keyword arguments for the stem normalization.
        norm: Normalization of the encoder blocks and of the neck.
        norm_kwargs: Extra keyword arguments for the normalization.

    Example:
        ```python
        import torch
        from torch_pointcloud.models import create_model

        model = create_model("tr3d.scannet.mmdet3d", task="detection", pretrained=True).cuda().eval()
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
        stem_channels: int = 64,
        encoder_channels: Sequence[int] = (64, 128, 128, 128),
        encoder_depths: Sequence[int] = (3, 4, 6, 3),
        neck_channels: int = 128,
        stem_norm: Union[str, Callable, None] = "batch_norm",
        stem_norm_kwargs: Optional[Dict[str, Any]] = None,
        norm: Union[str, Callable, None] = "batch_norm",
        norm_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(in_channels=in_channels, num_classes=num_classes)
        self.voxel_size = voxel_size
        self.oriented_boxes = oriented_boxes
        self.stem_channels = stem_channels
        self.encoder_channels = tuple(encoder_channels)
        self.encoder_depths = tuple(encoder_depths)
        self.neck_channels = neck_channels
        self.stem_norm = stem_norm
        self.stem_norm_kwargs = stem_norm_kwargs
        self.norm = norm
        self.norm_kwargs = norm_kwargs

        self.encoder = self.configure_encoder()
        self.neck = self.configure_neck()
        self.box = nn.Linear(neck_channels, 8 if oriented_boxes else 6)
        self.cls = self.configure_head()
        nn.init.normal_(self.box.weight, std=0.01)

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

    def configure_neck(self) -> TR3DNeck:
        """Build the feature pyramid over the last three encoder stages."""
        return TR3DNeck(self.encoder_channels[1:], self.neck_channels, norm=self.norm, norm_kwargs=self.norm_kwargs)

    def configure_head(self) -> nn.Linear:
        """Build the classification branch."""
        cls = nn.Linear(self.neck_channels, self.num_classes)
        nn.init.normal_(cls.weight, std=0.01)
        nn.init.constant_(cls.bias, -math.log(99))
        return cls

    @property
    def num_features(self) -> int:
        """Channel count $C$ of the coarsest encoder stage."""
        return self.encoder_channels[-1]

    @property
    def strides(self) -> Tuple[int, ...]:
        """Total stride of every pyramid level with respect to the input voxels."""
        return self.encoder.strides[1:-1]

    def reset_classifier(self, num_classes: int) -> None:
        self.num_classes = num_classes
        self.cls = self.configure_head()

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

    def forward_head(self, stages: Sequence["SparseConvTensor"], origin: Tensor) -> TR3DOutput:
        r"""Predict boxes from the encoder stages.

        Args:
            stages: Encoder stages from `forward_features`, from the finest.
            origin: Voxel-grid origin from `forward_features`, shape $(3,)$.

        Returns:
            The raw per-voxel predictions of every pyramid level.
        """
        outputs: Dict[str, List[Tensor]] = {"cls_logits": [], "box": [], "pos": [], "batch": [], "level": []}
        for i, level in enumerate(self.neck(stages[1:])):
            features = level.features
            indices = level.indices.long()
            outputs["cls_logits"].append(self.cls(features))
            outputs["box"].append(self.box(features))
            outputs["pos"].append((indices[:, 1:] * self.strides[i] + origin) * self.voxel_size)
            outputs["batch"].append(indices[:, 0])
            outputs["level"].append(torch.full_like(indices[:, 0], i))
        box, pos = torch.cat(outputs["box"]), torch.cat(outputs["pos"])

        return {
            "cls_logits": torch.cat(outputs["cls_logits"]),
            "boxes": self.decode_boxes(box, pos),
            "pos": pos,
            "batch": torch.cat(outputs["batch"]),
            "level": torch.cat(outputs["level"]),
        }

    def forward(self, x: Tensor, pos: Tensor, batch: Tensor) -> TR3DOutput:
        stages, origin = self.forward_features(x, pos, batch)
        return self.forward_head(stages, origin)

    def decode_boxes(self, box: Tensor, pos: Tensor) -> Tensor:
        r"""Turn the box regression of every location into a box.

        The regression is the offset from the location to the box center and the log box size, followed for oriented
        boxes by the heading code $(\ln q \sin 2\theta, \ln q \cos 2\theta)$ of the box aspect ratio $q = d_y / d_x$,
        which describes $(d_x, d_y, \theta)$ and $(d_y, d_x, \theta + \pi / 2)$ alike. An oriented box keeps the
        footprint $d_x + d_y$ of its regressed size, and its height is the sum of the last two regressed sizes, the
        decoding the released oriented weights were trained with.

        Args:
            box: Box regression, shape $(M, 6)$ or $(M, 8)$.
            pos: Locations, shape $(M, 3)$.

        Returns:
            The boxes $(c_x, c_y, c_z, d_x, d_y, d_z, \theta)$, heading $0$ for axis-aligned boxes, shape $(M, 7)$.
        """
        center = pos + box[:, :3]
        size = torch.exp(box[:, 3:6])
        heading = torch.zeros_like(center[:, :1])
        if self.oriented_boxes:
            footprint = size[:, 0] + size[:, 1]
            ratio = torch.exp(torch.sqrt(box[:, 6] ** 2 + box[:, 7] ** 2))
            width = footprint / (1 + ratio)
            size = torch.stack([width, width * ratio, size[:, 1] + size[:, 2]], dim=1)
            heading = 0.5 * torch.atan2(box[:, 6:7], box[:, 7:8])
        return torch.cat([center, size, heading], dim=1)

    def decode(self, out: TR3DOutput) -> Detection3D:
        r"""Score the boxes of a forward output (no score threshold, top-k or NMS).

        Every location is scored per class by its class probability and labelled by its best class. The authors'
        evaluation keeps the $1000$ best `scores` of every scene, then every (box, class) pair whose class score
        exceeds $0.01$, suppressed per class at a bird's-eye-view IoU of $0.5$ (`nms3d` with `rotated=True`).

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
        class_scores = out["cls_logits"].sigmoid()
        scores, labels = class_scores.max(dim=1)
        return {
            "boxes": out["boxes"],
            "scores": scores,
            "labels": labels,
            "batch": out["batch"],
            "class_scores": class_scores,
        }


@register_model(
    "tr3d.scannet.mmdet3d",
    task="detection",
    weights=WeightsDict(
        url="hf://torch-pointcloud/tr3d.scannet.mmdet3d/resolve/0000000000000000000000000000000000000000/model.safetensors",
        dataset="scannet",
        metrics={"mAP@0.25": 71.99, "mAP@0.5": 58.56},
        classes=SCANNET_DETECTION_CLASSES,
        author="mmdet3d",
        license="Apache-2.0",
    ),
    transform=T.Compose(
        [
            T.Normalize(keys=DataKeys.COLOR, mean=[0.0, 0.0, 0.0], std=[255.0, 255.0, 255.0]),
            T.Cat(keys=[DataKeys.COLOR], dst_key=DataKeys.X, dim=1),
        ]
    ),
    hparams=dict(
        in_channels=3,
        num_classes=18,
        voxel_size=0.01,
        oriented_boxes=False,
    ),
)
def tr3d_scannet_mmdet3d(**hparams: Any) -> TR3DDetection:
    return TR3DDetection(**hparams)


@register_model(
    "tr3d.sunrgbd.mmdet3d",
    task="detection",
    weights=WeightsDict(
        url="hf://torch-pointcloud/tr3d.sunrgbd.mmdet3d/resolve/0000000000000000000000000000000000000000/model.safetensors",
        dataset="sunrgbd",
        metrics={"mAP@0.25": 66.67, "mAP@0.5": 50.37},
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
    ),
)
def tr3d_sunrgbd_mmdet3d(**hparams: Any) -> TR3DDetection:
    return TR3DDetection(**hparams)


@register_model(
    "tr3d.s3dis-area5.mmdet3d",
    task="detection",
    weights=WeightsDict(
        url="hf://torch-pointcloud/tr3d.s3dis-area5.mmdet3d/resolve/0000000000000000000000000000000000000000/model.safetensors",
        dataset="s3dis-area5",
        metrics={"mAP@0.25": 71.26, "mAP@0.5": 48.29},
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
    ),
)
def tr3d_s3dis_area5_mmdet3d(**hparams: Any) -> TR3DDetection:
    return TR3DDetection(**hparams)
