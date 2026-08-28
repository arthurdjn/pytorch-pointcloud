"""PointConv classification model.

{{ paper("1811.07246") }}
"""

from typing import Any, Callable, Dict, List, Literal, NamedTuple, Optional, Sequence, Tuple, Type, Union, overload

import torch.nn as nn
from torch import Tensor
from torch_geometric.nn import MLP
from torch_geometric.typing import OptTensor

import torch_pointcloud.transforms as T
from torch_pointcloud.datasets.modelnet import MODELNET40_CLASSES
from torch_pointcloud.layers import (
    FPS,
    PointConvDensityGlobalSetAbstraction,
    PointConvDensitySetAbstraction,
    PoolLike,
    create_pool,
)
from torch_pointcloud.utils.conversion import ensure_list
from torch_pointcloud.utils.data import DataKeys

from ._base import ClassificationModel
from ._registry import WeightsDict, register_model


class PointConvIntermediate(NamedTuple):
    """Input features and point cloud of one set-abstraction stage, recorded before it downsamples."""

    x: Tensor
    pos: Tensor
    batch: Tensor


class PointConvDensityEncoder(nn.Module):
    """Stack of density-reweighted set-abstraction stages, each sampling centroids with FPS.

    When `global_pool` is given, the last stage becomes a `PointConvDensityGlobalSetAbstraction` and
    collapses the cloud to one feature vector per sample.
    """

    def __init__(
        self,
        in_channels: int,
        channels: Sequence[Sequence[int]],
        num_neighbors: Sequence[int],
        bandwidths: Sequence[float],
        ratios: Sequence[float],
        weight_channels: Union[Sequence[int]] = (8, 8),
        density_channels: Union[Sequence[int]] = (16, 8),
        spatial_dim: int = 3,
        act: Union[str, Callable, None] = "relu",
        act_kwargs: Optional[Dict[str, Any]] = None,
        act_first: bool = False,
        norm: Union[str, Callable, None] = "batch_norm",
        norm_kwargs: Optional[Dict[str, Any]] = None,
        bias: bool = True,
        global_pool: Optional[PoolLike] = None,
    ):
        super().__init__()
        num_layers = len(channels)

        self.layers = nn.ModuleList()
        for i in range(num_layers):
            layer_type: Type[nn.Module] = PointConvDensitySetAbstraction
            kwargs: Dict[str, Any] = {
                "num_neighbors": num_neighbors[i],
                "downsample": FPS(ratios[i]) if ratios[i] > 0.0 else None,
            }

            if i == num_layers - 1 and global_pool is not None:
                layer_type = PointConvDensityGlobalSetAbstraction
                kwargs = {"pool": global_pool}

            layer = layer_type(
                in_channels=in_channels,
                bandwidth=bandwidths[i],
                channels=channels[i],
                density_channels=density_channels,
                weight_channels=weight_channels,
                act=act,
                act_kwargs=act_kwargs,
                act_first=act_first,
                norm=norm,
                norm_kwargs=norm_kwargs,
                bias=bias,
                spatial_dim=spatial_dim,
                **kwargs,
            )

            self.layers.append(layer)
            in_channels = channels[i][-1]

    @overload
    def forward(
        self,
        x: OptTensor,
        pos: Tensor,
        batch: Tensor,
        return_intermediates: Literal[True],
    ) -> Tuple[Tensor, Tensor, Tensor, List[Any]]: ...

    @overload
    def forward(
        self,
        x: OptTensor,
        pos: Tensor,
        batch: Tensor,
        return_intermediates: Literal[False] = False,
    ) -> Tuple[Tensor, Tensor, Tensor]: ...

    def forward(
        self,
        x: OptTensor,
        pos: Tensor,
        batch: Tensor,
        return_intermediates: bool = False,
    ) -> Union[Tuple[Tensor, Tensor, Tensor], Tuple[Tensor, Tensor, Tensor, List[Any]]]:
        intermediates = []
        for layer in self.layers:
            if return_intermediates:
                intermediates.append(PointConvIntermediate(x, pos, batch))

            x, pos, batch = layer(x, pos, batch)

        if return_intermediates:
            return x, pos, batch, intermediates
        return x, pos, batch


class PointConvDensityClassification(ClassificationModel):
    r"""PointConv classification model with density re-weighting from
    :arxiv: [PointConv: Deep Convolutional Networks on 3D Point Clouds](https://arxiv.org/abs/1811.07246)
    by Wenxuan Wu, Zhongang Qi, Li Fuxin.

    Each set-abstraction stage samples centroids with farthest point sampling, estimates a kernel
    density per point, and applies a density-reweighted continuous convolution over each $k$-NN
    neighborhood. The last stage pools globally, so the classification head operates on one feature
    vector per sample.

    Shape:
        - Input: features of shape $(N, \text{in\_channels})$ (optional), points of shape
          $(N, \text{spatial\_dim})$, batch indices of shape $(N,)$.
        - Output: logits of shape $(B, \text{num\_classes})$.
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        *,
        channels: Sequence[Sequence[int]] = ([64, 64, 128], [128, 128, 256], [256, 512, 1024]),
        num_neighbors: Sequence[int] = (32, 64, 1024),
        bandwidths: Sequence[float] = (0.1, 0.2, 0.4),
        ratios: Sequence[float] = (0.5, 0.25, 0.0),
        density_channels: Sequence[int] = (16, 8),
        weight_channels: Sequence[int] = (8, 8),
        act: Union[str, Callable, None] = "relu",
        act_kwargs: Optional[Dict[str, Any]] = None,
        act_first: bool = False,
        norm: Union[str, Callable, None] = "batch_norm",
        norm_kwargs: Optional[Dict[str, Any]] = None,
        bias: bool = True,
        dropout: float = 0.5,
        global_pool: PoolLike = "mean",
        head_channels: Sequence[int] = (512, 256),
    ):
        super().__init__(in_channels=in_channels, num_classes=num_classes)
        self.channels = channels
        self.num_neighbors = ensure_list(num_neighbors)
        self.bandwidths = ensure_list(bandwidths)
        self.ratios = ensure_list(ratios)
        self.density_channels = ensure_list(density_channels)
        self.weight_channels = ensure_list(weight_channels)
        self.act = act
        self.act_kwargs = act_kwargs
        self.act_first = act_first
        self.norm = norm
        self.norm_kwargs = norm_kwargs
        self.bias = bias
        self.dropout = dropout
        self.global_pool = global_pool
        self.head_channels = ensure_list(head_channels)

        self.encoder = self.configure_encoder()
        self.head = self.configure_head()

    def configure_encoder(self) -> PointConvDensityEncoder:
        """Build the `PointConvDensityEncoder` backbone."""
        return PointConvDensityEncoder(
            in_channels=self.in_channels,
            channels=self.channels,
            num_neighbors=self.num_neighbors,
            bandwidths=self.bandwidths,
            ratios=self.ratios,
            density_channels=self.density_channels,
            weight_channels=self.weight_channels,
            act=self.act,
            act_kwargs=self.act_kwargs,
            act_first=self.act_first,
            norm=self.norm,
            norm_kwargs=self.norm_kwargs,
            bias=self.bias,
            global_pool=self.global_pool,
        )

    @property
    def num_features(self) -> int:
        """Feature dimension $C$ of the encoder output."""
        return self.encoder.layers[-1].fc.stem.out_features  # type: ignore[return-value,union-attr]

    def configure_head(self) -> nn.Module:
        if self.num_classes == 0:
            return nn.Identity()
        channels = [self.num_features] + ensure_list(self.head_channels) + [self.num_classes]
        return MLP(
            channels,
            act=self.act,
            act_kwargs=self.act_kwargs,
            act_first=self.act_first,
            norm=self.norm,
            norm_kwargs=self.norm_kwargs,
            bias=self.bias,
            dropout=self.dropout,
            plain_last=True,
        )

    def reset_classifier(self, num_classes: int, global_pool: Optional[PoolLike] = None, **kwargs: Any) -> None:
        self.num_classes = num_classes
        if global_pool is not None:
            self.encoder.layers[-1].pool = create_pool(global_pool)
        self.head = self.configure_head()

    @overload
    def forward_features(
        self,
        x: OptTensor,
        pos: Tensor,
        batch: Tensor,
        return_intermediates: Literal[True],
    ) -> Tuple[Tensor, Tensor, Tensor, List[Any]]: ...

    @overload
    def forward_features(
        self,
        x: OptTensor,
        pos: Tensor,
        batch: Tensor,
        return_intermediates: Literal[False] = False,
    ) -> Tuple[Tensor, Tensor, Tensor]: ...

    def forward_features(
        self,
        x: OptTensor,
        pos: Tensor,
        batch: Tensor,
        return_intermediates: bool = False,
    ) -> Any:
        return self.encoder(x, pos, batch, return_intermediates=return_intermediates)

    def forward_head(self, x: Tensor, batch: Tensor, pre_logits: bool = False) -> Tensor:
        # NOTE: In PointConv, the global pooling is performed in the encoder.
        return x if pre_logits else self.head(x)

    def forward(self, x: OptTensor, pos: Tensor, batch: Tensor) -> Tensor:
        x, _, batch = self.forward_features(x, pos, batch)
        return self.forward_head(x, batch)


@register_model("pointconv-density-base", task="classification")
def pointconv_density_clf(in_channels: int, num_classes: int, **kwargs: Any) -> PointConvDensityClassification:
    hparams: Dict[str, Any] = dict(
        channels=[[64, 64, 128], [128, 128, 256], [256, 512, 1024]],
        ratios=[0.5, 0.25, 0.125],
        num_neighbors=[32, 64, 128],
        bandwidths=[0.1, 0.2, 0.4],
        head_channels=[512, 256],
        act="relu",
        act_first=False,
        norm="batch_norm",
        bias=True,
        global_pool="mean",
        dropout=0.7,
    )
    hparams.update(kwargs)
    return PointConvDensityClassification(in_channels=in_channels, num_classes=num_classes, **hparams)


@register_model(
    "pointconv-density-base.modelnet40.wenxuan-wu",
    task="classification",
    weights=WeightsDict(
        url="hf://torch-pointcloud/pointconv/pointconv-density-base.modelnet40.wenxuan-wu.safetensors",
        dataset="modelnet40",
        metrics={"OA": 92.30},
        classes=MODELNET40_CLASSES,
        author="wenxuan-wu",
        license="MIT",
    ),
    transform=T.Compose(
        [
            T.Rescale(keys=DataKeys.POS),
            T.CopyItems(keys=DataKeys.POS, names=DataKeys.ORIGIN_POS),
            T.FarthestPointSample(
                pos_key=DataKeys.POS,
                keys=[DataKeys.NORMAL],
                num_samples=1024,
                random_start=False,
                dst_index_key=DataKeys.INDEX,
            ),
            T.CopyItems(keys=DataKeys.NORMAL, names=DataKeys.X),
        ]
    ),
    hparams=dict(
        in_channels=3,
        num_classes=40,
        channels=[[64, 64, 128], [128, 128, 256], [256, 512, 1024]],
        ratios=[0.5, 0.25, 0.125],
        num_neighbors=[32, 64, 128],
        bandwidths=[0.1, 0.2, 0.4],
        head_channels=[512, 256],
        act="relu",
        act_first=False,
        norm="batch_norm",
        bias=True,
        global_pool="mean",
        dropout=0.7,
    ),
)
def pointconv_density_modelnet40_clf(**hparams: Any) -> PointConvDensityClassification:
    return PointConvDensityClassification(**hparams)
