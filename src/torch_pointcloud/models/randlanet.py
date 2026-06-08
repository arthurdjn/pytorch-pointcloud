from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Literal,
    NamedTuple,
    Optional,
    Sequence,
    Tuple,
    Union,
    overload,
)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn import MLP
from torch_geometric.nn.resolver import activation_resolver

import torch_pointcloud.transforms as T
from torch_pointcloud.layers import PoolLike, create_cls_head, create_pool
from torch_pointcloud.layers.pointnet2_blocks import PointNet2FeaturePropagation
from torch_pointcloud.utils.cluster import knn, knn_graph
from torch_pointcloud.utils.conversion import ensure_list, ensure_tuple_size
from torch_pointcloud.utils.data import DataKeys
from torch_pointcloud.utils.imports import optional_import
from torch_pointcloud.utils.ops import decimate_indices, softmax
from torch_pointcloud.utils.types import OptTensor

from ._base import ClassificationModel, SegmentationModel
from ._registry import register_model

if TYPE_CHECKING:
    from torch_scatter import scatter_add, scatter_max


scatter_add, _ = optional_import("torch_scatter", "scatter_add")
scatter_max, _ = optional_import("torch_scatter", "scatter_max")


class RandLANetIntermediate(NamedTuple):
    x: Tensor
    pos: Tensor
    batch: Tensor


def random_max_pool(
    features: Tensor,
    pos: Tensor,
    batch: Tensor,
    factor: int,
    num_neighbors: int,
    generator: Optional[torch.Generator] = None,
) -> Tuple[Tensor, Tensor, Tensor]:
    decim_idx, decim_batch = decimate_indices(batch, factor, generator=generator)
    pos_decim = pos[decim_idx]
    edge_index = knn(pos, pos_decim, num_neighbors, batch_x=batch, batch_y=decim_batch)
    pooled, _ = scatter_max(features[edge_index[1]], edge_index[0], dim=0, dim_size=pos_decim.size(0))
    return pooled, pos_decim, decim_batch


class LocalSpatialEncoding(nn.Module):
    """Per-edge spatial encoding MLP (RandLA-Net Section 3.2).

    Wraps a single `Linear+norm+act` block that lifts an input feature to `out_channels`.
    Used twice per `LocalFeatureAggregation`: first on the raw 10-channel relative
    positional encoding, then on its output.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        act: Union[str, Callable, None],
        act_kwargs: Optional[Dict[str, Any]] = None,
        act_first: bool = False,
        norm: Union[str, Callable, None],
        norm_kwargs: Optional[Dict[str, Any]] = None,
        bias: bool = False,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.mlp = MLP(
            channel_list=[in_channels, out_channels],
            act=act,
            act_kwargs=act_kwargs,
            act_first=act_first,
            norm=norm,
            norm_kwargs=norm_kwargs,
            bias=bias,
            plain_last=False,
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.mlp(x)


class AttentivePooling(nn.Module):
    """Attention-weighted aggregation of pre-gathered edge features (RandLA-Net Section 3.3).

    Given $(E, C)$ edge features and a per-edge destination index, learn per-edge
    attention scores via a no-bias linear layer, softmax-normalize them across the
    neighbors of each destination point, sum the score-weighted features per
    destination, then project the result with a `Linear+norm+act` block.

    Args:
        in_channels: Channels of each edge feature.
        out_channels: Output channels after the post-aggregation MLP.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        act: Union[str, Callable, None],
        act_kwargs: Optional[Dict[str, Any]] = None,
        act_first: bool = False,
        norm: Union[str, Callable, None],
        norm_kwargs: Optional[Dict[str, Any]] = None,
        bias: bool = False,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.fc = nn.Linear(in_channels, in_channels, bias=False)
        self.mlp = MLP(
            channel_list=[in_channels, out_channels],
            act=act,
            act_kwargs=act_kwargs,
            act_first=act_first,
            norm=norm,
            norm_kwargs=norm_kwargs,
            bias=bias,
            plain_last=False,
        )

    def forward(self, edge_features: Tensor, dst_idx: Tensor, num_dst: int) -> Tensor:
        att_scores = softmax(self.fc(edge_features), dst_idx)
        weighted = att_scores * edge_features
        out = scatter_add(weighted, dst_idx, dim=0, dim_size=num_dst)
        return self.mlp(out)


class LocalFeatureAggregation(nn.Module):
    """Local Feature Aggregation module (RandLA-Net Section 3.4).

    Stacks two `LocalSpatialEncoding` + `AttentivePooling` units to progressively grow
    the receptive field, doubling a per-point feature of dim $d_\text{out} / 2$ to
    $d_\text{out}$. Mirrors the *LocSE + Attentive Pooling* "dilated" combination in
    Fig. 3 of the paper. The 10-channel relative positional encoding follows the
    original :github: [QingyongHu/RandLA-Net](https://github.com/QingyongHu/RandLA-Net) channel
    order (`cat([rel_dist, rel_xyz, xyz_i, xyz_j], dim=-1)`) so pretrained weights load
    without permuting the first 1x1 kernel. The second LSE re-projects the output of
    the first LSE to match the original `building_block`.
    """

    def __init__(
        self,
        d_out: int,
        *,
        act: Union[str, Callable, None],
        act_kwargs: Optional[Dict[str, Any]] = None,
        act_first: bool = False,
        norm: Union[str, Callable, None],
        norm_kwargs: Optional[Dict[str, Any]] = None,
        bias: bool = False,
    ) -> None:
        super().__init__()
        if d_out % 2 != 0:
            raise ValueError(f"`d_out` must be even, got {d_out}.")
        self.d_out = d_out

        mlp_kwargs: Dict[str, Any] = dict(
            act=act,
            act_kwargs=act_kwargs,
            act_first=act_first,
            norm=norm,
            norm_kwargs=norm_kwargs,
            bias=bias,
        )
        self.lse1 = LocalSpatialEncoding(10, d_out // 2, **mlp_kwargs)
        self.att_pooling_1 = AttentivePooling(d_out, d_out // 2, **mlp_kwargs)
        self.lse2 = LocalSpatialEncoding(d_out // 2, d_out // 2, **mlp_kwargs)
        self.att_pooling_2 = AttentivePooling(d_out, d_out, **mlp_kwargs)

    def forward(self, x: Tensor, pos: Tensor, edge_index: Tensor) -> Tensor:
        src_idx, dst_idx = edge_index
        num_dst = x.size(0)

        # Per-edge 10-channel relative positional encoding (RandLA-Net Section 3.2).
        pos_i = pos[dst_idx]
        pos_j = pos[src_idx]
        rel_xyz = pos_i - pos_j
        rel_dist = torch.linalg.norm(rel_xyz, dim=1, keepdim=True)
        rel = torch.cat([rel_dist, rel_xyz, pos_i, pos_j], dim=1)  # (E, 10)

        f_pos1 = self.lse1(rel)  # (E, d_out//2)
        f_neighbors = x[src_idx]  # (E, d_out//2)
        edge_feats = torch.cat([f_neighbors, f_pos1], dim=1)  # (E, d_out)
        x = self.att_pooling_1(edge_feats, dst_idx, num_dst)  # (N, d_out//2)

        f_pos2 = self.lse2(f_pos1)  # (E, d_out//2)
        f_neighbors = x[src_idx]
        edge_feats = torch.cat([f_neighbors, f_pos2], dim=1)
        x = self.att_pooling_2(edge_feats, dst_idx, num_dst)  # (N, d_out)
        return x


class DilatedResidualBlock(nn.Module):
    """RandLA-Net dilated residual block (Fig. 3 of the paper).

    Maps `d_in` channels to $2 \\cdot d_\\text{out}$ via a residual path of
    `MLP -> LocalFeatureAggregation -> MLP` plus a parallel `Linear+norm` shortcut.
    The sum is activated by the configured activation. `mlp2` and `shortcut` skip the
    activation (paper-mandated: `Conv2d(activation=False)` upstream); the configured
    activation is applied once after the residual sum.

    Args:
        d_in: Number of input channels.
        d_out: "Configuration" channel count; the block actually outputs $2 \\cdot d_\\text{out}$.
        num_neighbors: Number of neighbors for the local feature aggregation.
    """

    def __init__(
        self,
        d_in: int,
        d_out: int,
        num_neighbors: int,
        *,
        act: Union[str, Callable, None],
        act_kwargs: Optional[Dict[str, Any]] = None,
        act_first: bool = False,
        norm: Union[str, Callable, None],
        norm_kwargs: Optional[Dict[str, Any]] = None,
        bias: bool = False,
    ) -> None:
        super().__init__()
        if d_out % 2 != 0:
            raise ValueError(f"`d_out` must be even, got {d_out}.")
        self.num_neighbors = num_neighbors
        self.d_in = d_in
        self.d_out = d_out
        self.out_channels = 2 * d_out

        mlp_kwargs: Dict[str, Any] = dict(
            act=act,
            act_kwargs=act_kwargs,
            act_first=act_first,
            norm=norm,
            norm_kwargs=norm_kwargs,
            bias=bias,
        )
        self.mlp1 = MLP(channel_list=[d_in, d_out // 2], plain_last=False, **mlp_kwargs)
        self.lfa = LocalFeatureAggregation(d_out, **mlp_kwargs)
        # `mlp2` and `shortcut` use `act=None` (paper) — activation is applied once
        # after the residual sum.
        self.mlp2 = MLP(
            channel_list=[d_out, 2 * d_out],
            act=None,
            norm=norm,
            norm_kwargs=norm_kwargs,
            bias=bias,
            plain_last=False,
        )
        self.shortcut = MLP(
            channel_list=[d_in, 2 * d_out],
            act=None,
            norm=norm,
            norm_kwargs=norm_kwargs,
            bias=bias,
            plain_last=False,
        )
        self.act = activation_resolver(act, **(act_kwargs or {}))

    def forward(self, x: Tensor, pos: Tensor, batch: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        edge_index = knn_graph(pos, self.num_neighbors, batch=batch, loop=True)
        shortcut = self.shortcut(x)
        x = self.mlp1(x)
        x = self.lfa(x, pos, edge_index)
        x = self.mlp2(x)
        return self.act(x + shortcut), pos, batch


class RandLANetEncoder(nn.Module):
    r"""Stack of `DilatedResidualBlock` units interleaved with random-sampling
    K-NN max-pool decimation (RandLA-Net Section 3 / Fig. 2).

    Each encoder block doubles its $d_\text{out}^\text{config}$ to produce a
    $2 \cdot d_\text{out}^\text{config}$ channel feature; the per-stage decimation
    ratio then sub-samples the cloud by that factor.

    When `return_intermediates=True` is passed to `forward`, the encoder returns the
    bottleneck features plus the per-stage skip features in fine-to-coarse order:
    `intermediates[0]` is block 0's PRE-decimation output (used as the full-resolution
    skip in `RandLANetDecoder`); subsequent entries are the POST-decimation outputs of
    blocks $0 \ldots N-2$. The decoder consumes them in reverse.

    Args:
        in_channels: Number of input channels.
        encoder_channels: Output channels of each dilated residual block (must be even).
        decimation: Decimation factor between consecutive encoder blocks. Either a
            single `int` or a per-block sequence of length $N$.
        num_neighbors: Number of neighbors for the K-NN graph in each block. Either a
            single `int` or a per-block sequence of length $N$.
    """

    def __init__(
        self,
        in_channels: int,
        encoder_channels: Sequence[int],
        decimation: Union[int, Sequence[int]] = 4,
        num_neighbors: Union[int, Sequence[int]] = 16,
        *,
        act: Union[str, Callable, None],
        act_kwargs: Optional[Dict[str, Any]] = None,
        act_first: bool = False,
        norm: Union[str, Callable, None],
        norm_kwargs: Optional[Dict[str, Any]] = None,
        bias: bool = False,
    ) -> None:
        super().__init__()
        num_blocks = len(encoder_channels)
        extra_msg = f"`{{param}}` must be a sequence of the same length as the number of blocks {num_blocks}."
        self.decimation: Tuple[int, ...] = ensure_tuple_size(
            decimation, num_blocks, extra_msg=extra_msg.format(param="decimation")
        )
        num_neighbors_t: Tuple[int, ...] = ensure_tuple_size(
            num_neighbors, num_blocks, extra_msg=extra_msg.format(param="num_neighbors")
        )

        block_kwargs: Dict[str, Any] = dict(
            act=act,
            act_kwargs=act_kwargs,
            act_first=act_first,
            norm=norm,
            norm_kwargs=norm_kwargs,
            bias=bias,
        )
        self.blocks = nn.ModuleList()
        for out_channels, k in zip(encoder_channels, num_neighbors_t):
            if out_channels % 2 != 0:
                raise ValueError(
                    f"Each entry of `encoder_channels` must be even (the block doubles d_out internally), "
                    f"got {out_channels}."
                )
            # encoder_channels[i] is the BLOCK OUTPUT (== 2 * d_out_config), matching the
            # upstream paper convention where each block ends with a (d_out * 2) channel feature.
            self.blocks.append(DilatedResidualBlock(in_channels, out_channels // 2, num_neighbors=k, **block_kwargs))
            in_channels = out_channels
        self.out_channels = encoder_channels[-1]

    @overload
    def forward(
        self,
        x: Tensor,
        pos: Tensor,
        batch: Tensor,
        return_intermediates: Literal[True],
    ) -> Tuple[Tensor, Tensor, Tensor, List[RandLANetIntermediate]]: ...

    @overload
    def forward(
        self,
        x: Tensor,
        pos: Tensor,
        batch: Tensor,
        return_intermediates: Literal[False] = False,
    ) -> Tuple[Tensor, Tensor, Tensor]: ...

    def forward(
        self,
        x: Tensor,
        pos: Tensor,
        batch: Tensor,
        return_intermediates: bool = False,
    ) -> Any:
        intermediates: List[RandLANetIntermediate] = []
        for i, block in enumerate(self.blocks):
            assert isinstance(block, DilatedResidualBlock)  # For type checking
            x, pos, batch = block(x, pos, batch)
            if return_intermediates and i == 0:
                # Block 0's pre-decimation output is the only full-resolution skip;
                # the decoder consumes it last to upsample back to the input resolution.
                intermediates.append(RandLANetIntermediate(x=x, pos=pos, batch=batch))

            generator: Optional[torch.Generator] = None
            if not self.training:
                # Stable seed derived from input so eval is reproducible per-input
                generator = torch.Generator(device=batch.device)
                generator.manual_seed(int(batch.numel()))

            x, pos, batch = random_max_pool(
                x,
                pos,
                batch,
                factor=self.decimation[i],
                num_neighbors=block.num_neighbors,
                generator=generator,
            )

            if return_intermediates and i < len(self.blocks) - 1:
                intermediates.append(RandLANetIntermediate(x=x, pos=pos, batch=batch))

        if return_intermediates:
            return x, pos, batch, intermediates
        return x, pos, batch


class RandLANetDecoder(nn.Module):
    r"""Stack of `PointNet2FeaturePropagation` blocks specialized to RandLA-Net.

    Each block does 1-NN upsampling from the deeper resolution to its `pos_skip`
    resolution, concatenates with the encoder skip features, then applies a single
    linear+norm+act projection. The first block consumes the bottleneck features
    and the deepest encoder skip; subsequent blocks each cut the resolution by a
    stage of `decimation` until the finest (full-resolution) skip is reached.

    Note:
        Upstream RandLA-Net cats `[skip, interp]` while
        `PointNet2FeaturePropagation` cats `[interp, skip]`. To preserve weight
        equivalence with the upstream checkpoint, the conversion utilities
        (`convert_randlanet_state_dict`, `convert_open3d_randlanet_state_dict`)
        swap the first linear layer's column blocks per FP block.

    Args:
        in_channels: Channels at the bottleneck (input to the first FP).
        skip_channels: Per-stage encoder skip channels in coarse-to-fine order.
        fp_channels: Per-stage decoder output channels (same length as `skip_channels`).
        act: Activation passed to each `PointNet2FeaturePropagation` MLP.
        act_kwargs: Activation kwargs.
        act_first: If `True`, activation is applied before normalization.
        norm: Normalization passed to each `PointNet2FeaturePropagation` MLP.
        norm_kwargs: Normalization kwargs.
        bias: Whether to use bias in the MLP layers.

    Note:
        No paper-specific defaults are baked in here; callers (typically the
        registered model factory) supply them.
    """

    def __init__(
        self,
        in_channels: int,
        skip_channels: Sequence[int],
        fp_channels: Sequence[int],
        *,
        act: Union[str, Callable, None],
        act_kwargs: Optional[Dict[str, Any]] = None,
        act_first: bool = False,
        norm: Union[str, Callable, None],
        norm_kwargs: Optional[Dict[str, Any]] = None,
        bias: bool = False,
    ) -> None:
        super().__init__()
        if len(skip_channels) != len(fp_channels):
            raise ValueError(
                f"`skip_channels` ({len(skip_channels)}) and `fp_channels` ({len(fp_channels)}) must match."
            )
        self.fp_blocks = nn.ModuleList()
        for skip, out in zip(skip_channels, fp_channels):
            block = PointNet2FeaturePropagation(
                channels=[skip + in_channels, out],
                k=1,
                act=act,
                act_kwargs=act_kwargs,
                act_first=act_first,
                norm=norm,
                norm_kwargs=norm_kwargs,
                bias=bias,
                plain_last=False,
            )
            self.fp_blocks.append(block)
            in_channels = out

    def forward(
        self,
        x: Tensor,
        pos: Tensor,
        batch: Tensor,
        intermediates: List[RandLANetIntermediate],
    ) -> Tuple[Tensor, Tensor, Tensor]:
        for block, skip in zip(self.fp_blocks, reversed(intermediates)):
            x, pos, batch = block(x, pos, batch, skip.x, skip.pos, skip.batch)
        return x, pos, batch


class RandLANetClassification(ClassificationModel):
    """RandLA-Net classification model from
    :arxiv: [RandLA-Net: Efficient Semantic Segmentation of Large-Scale Point Clouds](https://arxiv.org/abs/1911.11236)
    by Qingyong Hu, Bo Yang, Linhai Xie, Stefano Rosa, Yulan Guo, Zhihua Wang, Niki Trigoni, Andrew Markham.

    Random sampling for downsampling and dilated residual blocks with local feature aggregation;
    point features are pooled globally after the encoder for classification.

    Args:
        in_channels: Number of input channels.
        num_classes: Number of classes.
        stem_channels: Number of channels in the stem MLP. Set to `None` to skip the stem.
        encoder_channels: Output channels of each dilated residual block (must be even).
        decimation: Decimation factor between consecutive encoder blocks.
        num_neighbors: Number of neighbors for the kNN graph in each block.
        aggr_channels: Optional channels for the aggregation MLP applied before global pooling.
        dropout: Dropout rate before the classification head.
        global_pool: Global pooling operation.
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        *,
        stem_channels: Optional[int] = 8,
        encoder_channels: Sequence[int],
        decimation: Union[int, Sequence[int]] = 4,
        num_neighbors: Union[int, Sequence[int]] = 16,
        aggr_channels: Optional[Union[int, Sequence[int]]] = None,
        dropout: float = 0.0,
        global_pool: PoolLike = "max",
        act: Union[str, Callable, None] = "relu",
        act_kwargs: Optional[Dict[str, Any]] = None,
        act_first: bool = False,
        norm: Union[str, Callable, None] = "batch_norm",
        norm_kwargs: Optional[Dict[str, Any]] = None,
        bias: bool = False,
    ) -> None:
        super().__init__(in_channels=in_channels, num_classes=num_classes)
        mlp_kwargs: Dict[str, Any] = dict(
            act=act,
            act_kwargs=act_kwargs,
            act_first=act_first,
            norm=norm,
            norm_kwargs=norm_kwargs,
            bias=bias,
        )

        self.stem: Optional[nn.Module] = None
        if stem_channels:
            self.stem = MLP(
                channel_list=[in_channels, stem_channels],
                plain_last=False,
                **mlp_kwargs,
            )
            in_channels = stem_channels

        self.encoder = RandLANetEncoder(
            in_channels=in_channels,
            encoder_channels=encoder_channels,
            decimation=decimation,
            num_neighbors=num_neighbors,
            **mlp_kwargs,
        )

        in_channels = encoder_channels[-1]
        aggr_channels = ensure_list(aggr_channels, none_as_empty=True)
        self.aggr: Optional[nn.Module] = None
        if aggr_channels:
            self.aggr = MLP(
                channel_list=[in_channels, aggr_channels],
                plain_last=False,
                **mlp_kwargs,
            )

        self.embedding_dim = aggr_channels[-1] if aggr_channels else encoder_channels[-1]
        self.global_pool = create_pool(global_pool)
        self.dropout = dropout
        self.head = create_cls_head(self.embedding_dim, num_classes)

    def reset_classifier(self, num_classes: int, global_pool: PoolLike = "max", **kwargs: Any) -> None:
        self.num_classes = num_classes
        self.global_pool = create_pool(global_pool)
        self.head = create_cls_head(num_features=self.embedding_dim, num_classes=self.num_classes, **kwargs)

    @overload
    def forward_features(
        self,
        x: OptTensor,
        pos: Tensor,
        batch: Tensor,
        return_intermediates: Literal[True],
    ) -> Tuple[Tensor, Tensor, Tensor, List[RandLANetIntermediate]]: ...

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
        x = x if x is not None else pos
        if self.stem is not None:
            x = self.stem(x)

        x, pos, batch, intermediates = self.encoder(x, pos, batch, return_intermediates=True)

        if self.aggr is not None:
            x = self.aggr(x)

        if return_intermediates:
            return x, pos, batch, intermediates
        return x, pos, batch

    def forward_head(self, x: Tensor, batch: Tensor, pre_logits: bool = False) -> Tensor:
        x = self.global_pool(x, batch)
        if self.dropout:
            x = F.dropout(x, p=float(self.dropout), training=self.training)
        return x if pre_logits else self.head(x)

    def forward(self, x: OptTensor, pos: Tensor, batch: Tensor) -> Tensor:
        x, _, batch = self.forward_features(x, pos, batch)
        return self.forward_head(x, batch, pre_logits=False)


class RandLANetSegmentation(SegmentationModel):
    """RandLA-Net segmentation model from
    :arxiv: [RandLA-Net: Efficient Semantic Segmentation of Large-Scale Point Clouds](https://arxiv.org/abs/1911.11236)
    by Qingyong Hu, Bo Yang, Linhai Xie, Stefano Rosa, Yulan Guo, Zhihua Wang, Niki Trigoni, Andrew Markham.

    Encoder uses random sampling between dilated residual blocks; decoder uses 1-NN
    nearest-neighbor interpolation with concatenation skips and a single 1x1 linear+BN+act
    per stage. The skip used at full resolution is the pre-decimation output of the first
    encoder block.

    Args:
        in_channels: Number of input channels.
        num_classes: Number of classes.
        stem_channels: Number of channels in the stem MLP. Set to `None` to skip the stem.
        encoder_channels: Output channels of each dilated residual block (must be even).
        fp_channels: Per-stage decoder channels (one list per upsampling step).
        head_channels: Hidden channels of the segmentation head MLP.
        decimation: Decimation factor between consecutive encoder blocks.
        num_neighbors: Number of neighbors for the kNN graph in each block.
        aggr_channels: Channels for the bottleneck MLP between the encoder and the decoder
            (the upstream "decoder_0").
        dropout: Dropout rate inside the segmentation head MLP.
        act: Activation type for both the decoder FP MLPs and the segmentation head MLP
            (string passed to PyG's `activation_resolver`, or a `Callable` / `nn.Module`).
        act_kwargs: Keyword arguments forwarded to the activation.
        act_first: If `True`, apply activation before normalization (PyG's MLP
            `act_first` semantics) — `Linear → Act → Norm → Dropout` instead of the
            default `Linear → Norm → Act → Dropout`.
        norm: Normalization type for the decoder FP MLPs and the head MLP.
        norm_kwargs: Keyword arguments forwarded to the normalization layers.
        bias: Whether the decoder/head hidden linear layers carry an explicit bias. The
            final head layer always uses `bias=True` since it has no normalization.
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        *,
        stem_channels: Optional[int] = 8,
        encoder_channels: Sequence[int],
        fp_channels: Sequence[int],
        head_channels: Sequence[int] = (64, 32),
        decimation: Union[int, Sequence[int]] = 4,
        num_neighbors: Union[int, Sequence[int]] = 16,
        aggr_channels: Optional[Union[int, Sequence[int]]] = None,
        dropout: float = 0.5,
        act: Union[str, Callable, None] = "relu",
        act_kwargs: Optional[Dict[str, Any]] = None,
        act_first: bool = False,
        norm: Union[str, Callable, None] = "batch_norm",
        norm_kwargs: Optional[Dict[str, Any]] = None,
        bias: bool = False,
    ) -> None:
        super().__init__(in_channels=in_channels, num_classes=num_classes)
        self.act = act
        self.act_kwargs = act_kwargs
        self.act_first = act_first
        self.norm = norm
        self.norm_kwargs = norm_kwargs
        self.bias = bias

        mlp_kwargs: Dict[str, Any] = dict(
            act=act,
            act_kwargs=act_kwargs,
            act_first=act_first,
            norm=norm,
            norm_kwargs=norm_kwargs,
            bias=bias,
        )

        self.stem: Optional[nn.Module] = None
        if stem_channels:
            self.stem = MLP(
                channel_list=[in_channels, stem_channels],
                plain_last=False,
                **mlp_kwargs,
            )
            in_channels = stem_channels

        self.encoder = RandLANetEncoder(
            in_channels=in_channels,
            encoder_channels=encoder_channels,
            decimation=decimation,
            num_neighbors=num_neighbors,
            **mlp_kwargs,
        )

        # Skip channels in forward (fine-to-coarse) order:
        # `block 0 PRE-decim`, `block 0 POST-decim`, `block 1 POST-decim`, ..., `block N-2 POST-decim`.
        # Block 0's pre-decimation output is reused as the last (full-resolution) skip
        # so the decoder can upsample back to the input resolution.
        skip_channels: List[int] = [encoder_channels[0]] + list(encoder_channels[:-1])

        in_channels = encoder_channels[-1]
        aggr_channels = ensure_list(aggr_channels, none_as_empty=True)
        self.aggr: Optional[nn.Module] = None
        if aggr_channels:
            self.aggr = MLP(
                channel_list=[in_channels, *aggr_channels],
                plain_last=False,
                **mlp_kwargs,
            )

        decoder_in = aggr_channels[-1] if aggr_channels else encoder_channels[-1]
        self.decoder = RandLANetDecoder(
            in_channels=decoder_in,
            skip_channels=skip_channels[::-1],
            fp_channels=fp_channels,
            **mlp_kwargs,
        )

        self.embedding_dim = fp_channels[-1]
        self.head_channels = ensure_list(head_channels, none_as_empty=True)
        self.dropout = dropout
        self.head = self.configure_head()

    def configure_head(self) -> nn.Module:
        # Per-point seg head: hidden layers carry `Linear(bias=`self.bias`)+norm+act+Dropout`,
        # the final `plain_last` layer is a bare `Linear(bias=True)` — its bias is
        # meaningful since it sees no normalization.
        n_hidden = len(self.head_channels)
        return MLP(
            channel_list=[self.embedding_dim, *self.head_channels, self.num_classes],
            act=self.act,
            act_kwargs=self.act_kwargs,
            act_first=self.act_first,
            norm=self.norm,
            norm_kwargs=self.norm_kwargs,
            bias=[self.bias] * n_hidden + [True],
            dropout=self.dropout,
            plain_last=True,
        )

    def reset_classifier(self, num_classes: int) -> None:
        self.num_classes = num_classes
        self.head = self.configure_head()

    @overload
    def forward_features(
        self,
        x: OptTensor,
        pos: Tensor,
        batch: Tensor,
        return_intermediates: Literal[True],
    ) -> Tuple[Tensor, Tensor, Tensor, List[RandLANetIntermediate]]: ...

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
        x = x if x is not None else pos
        if self.stem is not None:
            x = self.stem(x)

        if return_intermediates:
            x, pos, batch, intermediates = self.encoder(x, pos, batch, return_intermediates=True)
        else:
            x, pos, batch = self.encoder(x, pos, batch)

        if self.aggr is not None:
            x = self.aggr(x)

        if return_intermediates:
            return x, pos, batch, intermediates
        return x, pos, batch

    def forward_decoder(
        self,
        x: Tensor,
        pos: Tensor,
        batch: Tensor,
        intermediates: List[RandLANetIntermediate],
    ) -> Tensor:
        x, _, _ = self.decoder(x, pos, batch, intermediates)
        return x

    def forward_head(self, x: Tensor, pre_logits: bool = False) -> Tensor:
        return x if pre_logits else self.head(x)

    def forward(self, x: OptTensor, pos: Tensor, batch: Tensor) -> Tensor:
        x, pos, batch, intermediates = self.forward_features(x, pos, batch, return_intermediates=True)
        x = self.forward_decoder(x, pos, batch, intermediates)
        return self.forward_head(x)


@register_model(
    "randlanet-tsunghanwu.semantickitti",
    task="segmentation",
    weights="hf://torch-pointcloud/randlanet/randlanet-tsunghanwu.semantickitti.pt",
    transforms=T.Compose(
        [
            T.Relabel(
                keys=DataKeys.SEGMENT,
                labels={
                    10: 0,  # car
                    252: 0,  # moving-car
                    11: 1,  # bicycle
                    15: 2,  # motorcycle
                    18: 3,  # truck
                    258: 3,  # moving-truck
                    20: 4,  # other-vehicle
                    259: 4,  # moving-other-vehicle
                    30: 5,  # person
                    254: 5,  # moving-person
                    31: 6,  # bicyclist
                    253: 6,  # moving-bicyclist
                    32: 7,  # motorcyclist
                    255: 7,  # moving-motorcyclist
                    40: 8,  # road
                    44: 9,  # parking
                    48: 10,  # sidewalk
                    49: 11,  # other-ground
                    50: 12,  # building
                    51: 13,  # fence
                    70: 14,  # vegetation
                    71: 15,  # trunk
                    72: 16,  # terrain
                    80: 17,  # pole
                    81: 18,  # traffic-sign
                },
                default=255,
            ),
            T.Voxelize(
                pos_key=DataKeys.POS,
                pos_reduce="mean",
                keys=[DataKeys.SEGMENT],
                reduce=["first"],
                size=0.06,
            ),
        ]
    ),
    hparams=dict(
        in_channels=3,
        num_classes=19,
        stem_channels=8,
        encoder_channels=[32, 128, 256, 512],
        fp_channels=[256, 128, 32, 32],
        head_channels=[64, 32],
        aggr_channels=512,
        decimation=4,
        num_neighbors=16,
        dropout=0.5,
        act="leaky_relu",
        act_kwargs={"negative_slope": 0.2},
        norm="batch_norm",
        bias=False,
    ),
)
def randlanet_tsunghanwu_semantickitti_seg(**hparams: Any) -> RandLANetSegmentation:
    model = RandLANetSegmentation(**hparams)
    # Upstream `tsunghan-wu/RandLA-Net-pytorch` uses BN `eps=1e-6` (vs PyTorch's 1e-5)
    # and `momentum=0.99` (BNMomentumScheduler default, vs PyTorch's 0.1). The eps
    # affects per-channel scaling at inference and so must match the training-time value
    # to reproduce the upstream logits.
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            m.eps = 1e-6
            m.momentum = 0.99

    return model
