"""Utonia pretrained encoder and linear-probing segmentation model.

{{ paper("2603.03283") }}
"""

from typing import Any, Callable, Dict, List, Literal, Optional, Sequence, Tuple, Union, overload

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

import torch_pointcloud.transforms as T
from torch_pointcloud.datasets.scannet import SCANNET20_CLASSES
from torch_pointcloud.models._base import SegmentationModel
from torch_pointcloud.models._registry import WeightsDict, register_model
from torch_pointcloud.models.point_transformer_v3 import PointTransformerV3Encoder
from torch_pointcloud.utils.data import DataKeys
from torch_pointcloud.utils.serialization import SerializationOrder
from torch_pointcloud.utils.types import OptTensor


class UtoniaSegmentation(SegmentationModel):
    r"""Utonia linear-probing segmentation model.

    Linear-probe variant from
    :arxiv: [Utonia: Toward One Encoder for All Point Clouds](https://arxiv.org/abs/2603.03283)
    (ICML 2026). Architecturally similar to Sonata / Concerto's
    linear-probe head, with one key change: every attention layer adds a 3D
    rotary position embedding ([`Point3DRoPE`](../layers/rope.md)) on top of
    `(q, k)`, indexed by the real-valued metric position rather than the
    integer voxel grid. The position is mean-pooled at every encoder stage so
    each level operates at its natural scale.

    Note:
        The default (and registered) configuration enables flash attention (`use_flash_attn=True`),
        which requires the `flash-attn` package and a CUDA device; pass `use_flash_attn=False` to
        run without it. The xCPE sparse convolution still needs a `spconv` build matching the
        device; the standard CUDA wheel cannot run the forward on CPU.
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        serialization_orders: Sequence[SerializationOrder] = ("z", "z-trans", "hilbert", "hilbert-trans"),
        shuffle_serialization_orders: bool = True,
        strides: Sequence[int] = (2, 2, 2, 2),
        encoder_depths: Sequence[int] = (3, 3, 3, 12, 3),
        encoder_channels: Sequence[int] = (54, 108, 216, 432, 576),
        encoder_num_heads: Sequence[int] = (3, 6, 12, 24, 32),
        encoder_patch_size: Sequence[int] = (1024, 1024, 1024, 1024, 1024),
        norm: Union[str, Callable] = "layer_norm",
        act: Union[str, Callable] = "gelu",
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_scale: Optional[float] = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        drop_path: float = 0.3,
        use_flash_attn: bool = True,
        upcast_attn: bool = False,
        upcast_softmax: bool = False,
        rope_base: float = 10.0,
        dropout: float = 0.0,
        pooling: str = "grid",
        stem_type: str = "linear",
        act_kwargs: Optional[Dict[str, Any]] = None,
        norm_kwargs: Optional[Dict[str, Any]] = None,
        legacy: bool = False,
    ):
        super().__init__(in_channels=in_channels, num_classes=num_classes)
        # PyG's LayerNorm defaults to graph mode, which normalizes across the whole packed batch and
        # leaks features between samples; node mode matches nn.LayerNorm semantics.
        if norm == "layer_norm":
            norm_kwargs = {"mode": "node", **(norm_kwargs or {})}
        self.serialization_orders = serialization_orders
        self.shuffle_serialization_orders = shuffle_serialization_orders
        self.strides = strides
        self.encoder_depths = encoder_depths
        self.encoder_channels = encoder_channels
        self.encoder_num_heads = encoder_num_heads
        self.encoder_patch_size = encoder_patch_size
        self.norm = norm
        self.act = act
        self.mlp_ratio = mlp_ratio
        self.qkv_bias = qkv_bias
        self.qk_scale = qk_scale
        self.attn_drop = attn_drop
        self.proj_drop = proj_drop
        self.drop_path = drop_path
        self.use_flash_attn = use_flash_attn
        self.upcast_attn = upcast_attn
        self.upcast_softmax = upcast_softmax
        self.rope_base = rope_base
        self.dropout = dropout
        self.pooling = pooling
        self.stem_type = stem_type
        self.act_kwargs = act_kwargs
        self.norm_kwargs = norm_kwargs
        self.legacy = legacy

        self.encoder = self.configure_encoder()
        self.head = self.configure_head()

    def configure_encoder(self) -> PointTransformerV3Encoder:
        """Build the `PointTransformerV3Encoder` backbone with rotary position embeddings."""
        return PointTransformerV3Encoder(
            in_channels=self.in_channels,
            serialization_orders=self.serialization_orders,
            shuffle_serialization_orders=self.shuffle_serialization_orders,
            strides=self.strides,
            encoder_depths=self.encoder_depths,
            encoder_channels=self.encoder_channels,
            encoder_num_heads=self.encoder_num_heads,
            encoder_patch_size=self.encoder_patch_size,
            norm=self.norm,
            act=self.act,
            mlp_ratio=self.mlp_ratio,
            qkv_bias=self.qkv_bias,
            qk_scale=self.qk_scale,
            attn_drop=self.attn_drop,
            proj_drop=self.proj_drop,
            drop_path=self.drop_path,
            use_flash_attn=self.use_flash_attn,
            upcast_attn=self.upcast_attn,
            upcast_softmax=self.upcast_softmax,
            pooling=self.pooling,
            stem_type=self.stem_type,
            act_kwargs=self.act_kwargs,
            norm_kwargs=self.norm_kwargs,
            attn_kind="rope",
            rope_base=self.rope_base,
            legacy=self.legacy,
        )

    @property
    def num_features(self) -> int:
        """Channel count $C$ entering the head: every encoder stage unpooled and concatenated."""
        return sum(self.encoder_channels)

    def configure_head(self) -> nn.Module:
        return (
            nn.Identity()
            if self.num_classes == 0
            else nn.Linear(self.num_features, self.num_classes).train(self.training)
        )

    def reset_classifier(self, num_classes: int) -> None:
        self.num_classes = num_classes
        self.head = self.configure_head()

    @overload
    def forward_features(
        self,
        x: OptTensor,
        pos_grid: Tensor,
        batch: Tensor,
        return_intermediates: Literal[True],
        pos: OptTensor = None,
    ) -> Tuple[Tensor, Tensor, Tensor, List[Dict[str, Tensor]]]: ...

    @overload
    def forward_features(
        self,
        x: OptTensor,
        pos_grid: Tensor,
        batch: Tensor,
        return_intermediates: Literal[False] = False,
        pos: OptTensor = None,
    ) -> Tuple[Tensor, Tensor, Tensor]: ...

    def forward_features(
        self,
        x: OptTensor,
        pos_grid: Tensor,
        batch: Tensor,
        return_intermediates: bool = False,
        pos: OptTensor = None,
    ) -> Any:
        if return_intermediates:
            return self.encoder.forward(x, pos_grid, batch, return_intermediates=True, pos=pos)
        return self.encoder.forward(x, pos_grid, batch, return_intermediates=False, pos=pos)

    def forward_decoder(self, x: Tensor, intermediates: List[Dict[str, Tensor]]) -> Tuple[Tensor, Tensor, Tensor]:
        pos_grid = batch = None
        for intermediate in reversed(intermediates):
            inverse = intermediate["inverse"]
            x = torch.cat([intermediate["x"], x[inverse]], dim=-1)
            pos_grid = intermediate["pos_grid"]
            batch = intermediate["batch"]

        if pos_grid is None or batch is None:
            raise ValueError("Utonia segmentation requires encoder intermediates for feature unpooling.")
        return x, pos_grid, batch

    def forward_head(self, x: Tensor, pre_logits: bool = False) -> Tensor:
        if self.dropout:
            x = F.dropout(x, p=float(self.dropout), training=self.training)
        return x if pre_logits else self.head(x)

    def forward(self, x: Tensor, pos: Tensor, pos_grid: Tensor, batch: Tensor) -> Tensor:
        """Forward pass.

        Args:
            x: Per-point features of shape $(N, C)$.
            pos: Real-valued metric positions of shape $(N, 3)$ used by 3D RoPE.
            pos_grid: Integer voxel-grid coordinates of shape $(N, 3)$ used by the
                encoder for serialization and sparse convolutions.
            batch: Per-point batch index of shape $(N,)$.
        """
        x, _, _, intermediates = self.forward_features(x, pos_grid, batch, return_intermediates=True, pos=pos)
        x, _, _ = self.forward_decoder(x, intermediates)
        return self.forward_head(x)


_UTONIA_TRANSFORMS = T.Compose(
    [
        T.Scale(keys=DataKeys.POS, scale=0.5),
        T.Shift(keys=DataKeys.POS, method="bbox", axes=[0, 1]),  # XY: bbox midrange
        T.Shift(keys=DataKeys.POS, method="min", axes=[2]),  # Z: min
        T.Divide(keys=DataKeys.COLOR, divisor=255),
        T.Cat(keys=[DataKeys.POS, DataKeys.COLOR, DataKeys.NORMAL], dst_key=DataKeys.X, dim=1),
        T.CopyItems(
            keys=[DataKeys.POS, DataKeys.SEGMENT],
            names=[DataKeys.ORIGIN_POS, DataKeys.ORIGIN_SEGMENT],
            allow_missing_keys=True,
        ),
        T.Voxelize(
            pos_key=DataKeys.POS,
            pos_reduce="first",
            dst_pos_grid_key=DataKeys.POS_GRID,
            keys=[DataKeys.X, DataKeys.SEGMENT, DataKeys.COLOR, DataKeys.NORMAL, DataKeys.INSTANCE],
            reduce="first",
            size=0.01,
            method="fnv",
            allow_missing_keys=True,
            dst_inverse_key=DataKeys.INVERSE,
        ),
    ]
)

_UTONIA_SEG_TRANSFORMS = T.Compose(
    [
        T.Scale(keys=DataKeys.POS, scale=0.5),
        T.Shift(keys=DataKeys.POS, method="bbox", axes=[0, 1]),  # XY: bbox midrange
        T.Shift(keys=DataKeys.POS, method="min", axes=[2]),  # Z: min
        T.Divide(keys=DataKeys.COLOR, divisor=255),
        T.Cat(keys=[DataKeys.POS, DataKeys.COLOR, DataKeys.NORMAL], dst_key=DataKeys.X, dim=1),
        T.Relabel(keys=DataKeys.SEGMENT, labels=range(1, 21), default=-1),
        T.CopyItems(
            keys=[DataKeys.POS, DataKeys.SEGMENT],
            names=[DataKeys.ORIGIN_POS, DataKeys.ORIGIN_SEGMENT],
            allow_missing_keys=True,
        ),
        T.Voxelize(
            pos_key=DataKeys.POS,
            pos_reduce="first",
            dst_pos_grid_key=DataKeys.POS_GRID,
            keys=[DataKeys.X, DataKeys.SEGMENT, DataKeys.COLOR, DataKeys.NORMAL, DataKeys.INSTANCE],
            reduce="first",
            size=0.01,
            method="fnv",
            allow_missing_keys=True,
            dst_inverse_key=DataKeys.INVERSE,
        ),
    ]
)


def _utonia_encoder_hparams() -> Dict[str, Any]:
    return dict(
        in_channels=9,
        serialization_orders=("z", "z-trans", "hilbert", "hilbert-trans"),
        shuffle_serialization_orders=True,
        strides=(2, 2, 2, 2),
        encoder_depths=(3, 3, 3, 12, 3),
        encoder_channels=(54, 108, 216, 432, 576),
        encoder_num_heads=(3, 6, 12, 24, 32),
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
        attn_kind="rope",
        rope_base=10.0,
    )


@register_model(
    "utonia.pretrain.pointcept",
    task="base",
    weights=WeightsDict(
        url="hf://torch-pointcloud/utonia.pretrain.pointcept/resolve/main/model.safetensors",
        author="pointcept",
        license="CC-BY-NC-4.0",
    ),
    transform=_UTONIA_TRANSFORMS,
    hparams=_utonia_encoder_hparams(),
)
def utonia(**hparams: Any) -> PointTransformerV3Encoder:
    return PointTransformerV3Encoder(**hparams)


@register_model(
    "utonia-lp.scannet20.pointcept",
    task="segmentation",
    weights=WeightsDict(
        url="hf://torch-pointcloud/utonia-lp.scannet20.pointcept/resolve/main/model.safetensors",
        dataset="scannet20",
        metrics={"mIoU": 77.70},
        classes=SCANNET20_CLASSES,
        author="pointcept",
        license="CC-BY-NC-4.0",
    ),
    transform=_UTONIA_SEG_TRANSFORMS,
    hparams=dict(
        num_classes=20,
        in_channels=9,
        serialization_orders=("z", "z-trans", "hilbert", "hilbert-trans"),
        shuffle_serialization_orders=True,
        strides=(2, 2, 2, 2),
        encoder_depths=(3, 3, 3, 12, 3),
        encoder_channels=(54, 108, 216, 432, 576),
        encoder_num_heads=(3, 6, 12, 24, 32),
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
        rope_base=10.0,
    ),
)
def utonia_scannet20(**hparams: Any) -> UtoniaSegmentation:
    return UtoniaSegmentation(**hparams)
