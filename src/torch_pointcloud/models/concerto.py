from typing import Any, Callable, Dict, List, Literal, Optional, Sequence, Tuple, Union, overload

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

import torch_pointcloud.transforms as T
from torch_pointcloud.models._base import SegmentationModel
from torch_pointcloud.datasets.scannet import SCANNET20_CLASSES
from torch_pointcloud.models._registry import WeightsDict, register_model
from torch_pointcloud.models.point_transformer_v3 import AttentionKind, PointTransformerV3Encoder
from torch_pointcloud.utils.data import DataKeys
from torch_pointcloud.utils.serialization import SerializationOrder
from torch_pointcloud.utils.types import OptTensor


class ConcertoSegmentation(SegmentationModel):
    r"""Concerto linear-probing segmentation model.

    Linear-probe variant from
    :arxiv: [Concerto: Joint 2D-3D Self-Supervised Learning](https://arxiv.org/abs/2510.23607)
    (Zhang et al., NeurIPS 2025). The pretrained PT-V3 encoder is run with
    intermediate features captured at every encoder stage; those features are
    successively unpooled through the saved pooling inverses, concatenated with
    each parent stage, then projected by a single linear segmentation head.

    Architecturally identical to Sonata's linear-probe head, only the encoder
    scale differs (Concerto-large uses $C = (64, 128, 256, 512, 768)$ versus
    Sonata's $(48, 96, 192, 384, 512)$).
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        serialization_orders: Sequence[SerializationOrder] = ("z", "z-trans", "hilbert", "hilbert-trans"),
        shuffle_serialization_orders: bool = True,
        strides: Sequence[int] = (2, 2, 2, 2),
        encoder_depths: Sequence[int] = (3, 3, 3, 12, 3),
        encoder_channels: Sequence[int] = (64, 128, 256, 512, 768),
        encoder_num_head: Sequence[int] = (4, 8, 16, 32, 48),
        encoder_patch_size: Sequence[int] = (1024, 1024, 1024, 1024, 1024),
        norm: Union[str, Callable] = "layer_norm",
        act: Union[str, Callable] = "gelu",
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_scale: Optional[float] = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        drop_path: float = 0.3,
        attn_kind: AttentionKind = "default",
        use_flash_attn: bool = True,
        upcast_attn: bool = False,
        upcast_softmax: bool = False,
        dropout: float = 0.0,
        pooling: str = "grid",
        stem_type: str = "linear",
        act_kwargs: Optional[Dict[str, Any]] = None,
        norm_kwargs: Optional[Dict[str, Any]] = None,
        legacy: bool = False,
    ):
        super().__init__(in_channels=in_channels, num_classes=num_classes)
        self.encoder_channels = encoder_channels
        self.encoder = PointTransformerV3Encoder(
            in_channels=in_channels,
            serialization_orders=serialization_orders,
            shuffle_serialization_orders=shuffle_serialization_orders,
            strides=strides,
            encoder_depths=encoder_depths,
            encoder_channels=encoder_channels,
            encoder_num_head=encoder_num_head,
            encoder_patch_size=encoder_patch_size,
            norm=norm,
            act=act,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            drop_path=drop_path,
            attn_kind=attn_kind,
            use_flash_attn=use_flash_attn,
            upcast_attn=upcast_attn,
            upcast_softmax=upcast_softmax,
            pooling=pooling,
            stem_type=stem_type,
            act_kwargs=act_kwargs,
            norm_kwargs=norm_kwargs,
            legacy=legacy,
        )
        self.dropout = dropout
        self.head = nn.Linear(self.embedding_dim, num_classes)

    @property
    def embedding_dim(self) -> int:
        return sum(self.encoder_channels)

    def reset_classifier(self, num_classes: int) -> None:
        self.num_classes = num_classes
        self.head = nn.Linear(self.embedding_dim, num_classes)

    @overload
    def forward_features(
        self,
        x: OptTensor,
        pos_grid: Tensor,
        batch: Tensor,
        return_intermediates: Literal[True],
    ) -> Tuple[Tensor, Tensor, Tensor, List[Dict[str, Tensor]]]: ...

    @overload
    def forward_features(
        self,
        x: OptTensor,
        pos_grid: Tensor,
        batch: Tensor,
        return_intermediates: Literal[False] = False,
    ) -> Tuple[Tensor, Tensor, Tensor]: ...

    def forward_features(
        self,
        x: OptTensor,
        pos_grid: Tensor,
        batch: Tensor,
        return_intermediates: bool = False,
    ) -> Any:
        if return_intermediates:
            return self.encoder.forward(x, pos_grid, batch, return_intermediates=True)
        return self.encoder.forward(x, pos_grid, batch, return_intermediates=False)

    def forward_decoder(self, x: Tensor, intermediates: List[Dict[str, Tensor]]) -> Tuple[Tensor, Tensor, Tensor]:
        pos_grid = batch = None
        for intermediate in reversed(intermediates):
            inverse = intermediate["inverse"]
            x = torch.cat([intermediate["x"], x[inverse]], dim=-1)
            pos_grid = intermediate["pos_grid"]
            batch = intermediate["batch"]

        if pos_grid is None or batch is None:
            raise ValueError("Concerto segmentation requires encoder intermediates for feature unpooling.")
        return x, pos_grid, batch

    def forward_head(self, x: Tensor, pre_logits: bool = False) -> Tensor:
        if self.dropout:
            x = F.dropout(x, p=float(self.dropout), training=self.training)
        return x if pre_logits else self.head(x)

    def forward(self, x: Tensor, pos_grid: Tensor, batch: Tensor) -> Tensor:
        x, _, _, intermediates = self.forward_features(x, pos_grid, batch, return_intermediates=True)
        x, _, _ = self.forward_decoder(x, intermediates)
        return self.forward_head(x)


_CONCERTO_TRANSFORMS = T.Compose(
    [
        T.Shift(keys=DataKeys.POS, method="bbox", axes=[0, 1]),  # XY: bbox midrange
        T.Shift(keys=DataKeys.POS, method="min", axes=[2]),  # Z: min
        T.Divide(keys=DataKeys.COLOR, divisor=255),
        T.Cat(keys=[DataKeys.POS, DataKeys.COLOR, DataKeys.NORMAL], dst_key=DataKeys.X, dim=1),
        T.Voxelize(
            pos_key=DataKeys.POS,
            pos_reduce="grid",
            keys=[DataKeys.X],
            reduce=["first"],
            size=0.02,
            method="fnv",
        ),
        T.CopyItems(keys=DataKeys.POS, names=DataKeys.POS_GRID),
    ]
)

_CONCERTO_SEG_TRANSFORMS = T.Compose(
    [
        T.Shift(keys=DataKeys.POS, method="bbox", axes=[0, 1]),  # XY: bbox midrange
        T.Shift(keys=DataKeys.POS, method="min", axes=[2]),  # Z: min
        T.Divide(keys=DataKeys.COLOR, divisor=255),
        T.Cat(keys=[DataKeys.POS, DataKeys.COLOR, DataKeys.NORMAL], dst_key=DataKeys.X, dim=1),
        T.Voxelize(
            pos_key=DataKeys.POS,
            pos_reduce="grid",
            keys=[DataKeys.X, DataKeys.SEGMENT],
            reduce=["first", "first"],
            size=0.02,
            method="fnv",
        ),
        T.CopyItems(keys=DataKeys.POS, names=DataKeys.POS_GRID),
        T.Relabel(keys=DataKeys.SEGMENT, labels=range(1, 21), default=-1),
    ]
)


def _concerto_encoder_hparams(
    encoder_depths: Sequence[int],
    encoder_channels: Sequence[int],
    encoder_num_head: Sequence[int],
) -> Dict[str, Any]:
    return dict(
        in_channels=9,
        serialization_orders=("z", "z-trans", "hilbert", "hilbert-trans"),
        shuffle_serialization_orders=True,
        strides=(2, 2, 2, 2),
        encoder_depths=encoder_depths,
        encoder_channels=encoder_channels,
        encoder_num_head=encoder_num_head,
        encoder_patch_size=(1024, 1024, 1024, 1024, 1024),
        norm="layer_norm",
        act="gelu",
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.3,
        use_flash_attn=True,
        upcast_attn=False,
        upcast_softmax=False,
        pooling="grid",
        stem_type="linear",
        norm_kwargs={"mode": "node"},
    )


@register_model(
    "concerto-tiny.pointcept",
    task="base",
    weights=WeightsDict(
        url="hf://torch-pointcloud/concerto/concerto-tiny.pointcept.safetensors",
        author="pointcept",
        license="CC-BY-NC-4.0",
    ),
    transform=_CONCERTO_TRANSFORMS,
    hparams=_concerto_encoder_hparams(
        encoder_depths=(1, 1, 1, 3, 1),
        encoder_channels=(16, 32, 64, 128, 256),
        encoder_num_head=(1, 2, 4, 8, 16),
    ),
)
def concerto_tiny(**hparams: Any) -> PointTransformerV3Encoder:
    return PointTransformerV3Encoder(**hparams)


@register_model(
    "concerto-small.pointcept",
    task="base",
    weights=WeightsDict(
        url="hf://torch-pointcloud/concerto/concerto-small.pointcept.safetensors",
        author="pointcept",
        license="CC-BY-NC-4.0",
    ),
    transform=_CONCERTO_TRANSFORMS,
    hparams=_concerto_encoder_hparams(
        encoder_depths=(2, 2, 2, 6, 2),
        encoder_channels=(32, 64, 128, 256, 512),
        encoder_num_head=(2, 4, 8, 16, 32),
    ),
)
def concerto_small(**hparams: Any) -> PointTransformerV3Encoder:
    return PointTransformerV3Encoder(**hparams)


@register_model(
    "concerto-base.pointcept",
    task="base",
    weights=WeightsDict(
        url="hf://torch-pointcloud/concerto/concerto-base.pointcept.safetensors",
        author="pointcept",
        license="CC-BY-NC-4.0",
    ),
    transform=_CONCERTO_TRANSFORMS,
    hparams=_concerto_encoder_hparams(
        encoder_depths=(3, 3, 3, 12, 3),
        encoder_channels=(48, 96, 192, 384, 512),
        encoder_num_head=(3, 6, 12, 24, 32),
    ),
)
def concerto_base(**hparams: Any) -> PointTransformerV3Encoder:
    return PointTransformerV3Encoder(**hparams)


@register_model(
    "concerto-large.pointcept",
    task="base",
    weights=WeightsDict(
        url="hf://torch-pointcloud/concerto/concerto-large.pointcept.safetensors",
        author="pointcept",
        license="CC-BY-NC-4.0",
    ),
    transform=_CONCERTO_TRANSFORMS,
    hparams=_concerto_encoder_hparams(
        encoder_depths=(3, 3, 3, 12, 3),
        encoder_channels=(64, 128, 256, 512, 768),
        encoder_num_head=(4, 8, 16, 32, 48),
    ),
)
def concerto_large(**hparams: Any) -> PointTransformerV3Encoder:
    return PointTransformerV3Encoder(**hparams)


@register_model(
    "concerto-large-lp.scannet20.pointcept",
    task="segmentation",
    weights=WeightsDict(
        url="hf://torch-pointcloud/concerto/concerto-large-lp.scannet20.pointcept.safetensors",
        dataset="scannet20",
        metrics={"mIoU": 77.68},
        classes=SCANNET20_CLASSES,
        author="pointcept",
        license="CC-BY-NC-4.0",
    ),
    transform=_CONCERTO_SEG_TRANSFORMS,
    hparams=dict(
        num_classes=20,
        **_concerto_encoder_hparams(
            encoder_depths=(3, 3, 3, 12, 3),
            encoder_channels=(64, 128, 256, 512, 768),
            encoder_num_head=(4, 8, 16, 32, 48),
        ),
    ),
)
def concerto_large_lp_scannet20(**hparams: Any) -> ConcertoSegmentation:
    return ConcertoSegmentation(**hparams)
