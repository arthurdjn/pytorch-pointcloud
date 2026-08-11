from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import torch
from torch import Tensor, nn
from torch_geometric.nn import MLP

import torch_pointcloud.transforms as T
from torch_pointcloud.datasets.modelnet import MODELNET40_CLASSES
from torch_pointcloud.datasets.scanobjectnn import SCANOBJECTNN_CLASSES
from torch_pointcloud.layers import FPModule, PointPatchEmbed, TransformerBlock
from torch_pointcloud.layers.act import create_act
from torch_pointcloud.utils.cluster import group
from torch_pointcloud.utils.data import DataKeys
from torch_pointcloud.utils.imports import _TORCH_SCATTER_GITHUB_URL, optional_import
from torch_pointcloud.utils.types import OptTensor

from ._base import BaseModel, ClassificationModel, SegmentationModel
from ._registry import WeightsDict, register_model

if TYPE_CHECKING:
    from torch_scatter import scatter

scatter, _ = optional_import("torch_scatter", "scatter", url=_TORCH_SCATTER_GITHUB_URL)


class EncoderBlock(nn.Module):
    r"""A stack of transformer blocks that adds the positional embedding before every block.

    Args:
        embed_dim: The number of channels.
        depth: The number of transformer blocks.
        num_heads: The number of attention heads.
        mlp_ratio: The feed-forward expansion ratio.
        qkv_bias: Whether to add a bias to the query / key / value projection.
        dropout: The dropout rate for the MLP and attention output projections.
        attn_drop_rate: The attention dropout rate.
        drop_path: The stochastic depth rate, either a scalar or a per-block list.
        act: The activation function used in the feed-forward MLP.
        norm: The normalization applied before attention and the MLP.

    Shape:
        - Input: $(B, N, C)$
        - Output: $(B, N, C)$
    """

    def __init__(
        self,
        embed_dim: int,
        depth: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        dropout: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path: Union[float, List[float]] = 0.0,
        act: Union[str, Callable, None] = "gelu",
        norm: Union[str, Callable, None] = nn.LayerNorm,
    ):
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    dropout=dropout,
                    attn_dropout=attn_drop_rate,
                    drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                    act=act,
                    norm=norm,
                )
                for i in range(depth)
            ]
        )

    def forward(self, x: Tensor, pos: Tensor, mask: OptTensor = None) -> Tensor:
        for block in self.blocks:
            x = block(x + pos, mask)
        return x


def local_attention_mask(mask: Tensor) -> Tensor:
    r"""Convert a boolean neighbor mask into the additive pre-softmax bias of the shared attention.

    Entries that are `True` (a point pair outside the local radius, or a padded / cross-sample pair)
    are pushed to a large negative value so they vanish after the softmax. The result is broadcast over
    the attention heads.

    Args:
        mask: Boolean / float mask of shape $(B, N, N)$ where non-zero marks a forbidden pair.

    Returns:
        The additive attention bias of shape $(B, 1, N, N)$.

    Shape:
        - `mask`: $(B, N, N)$
        - Output: $(B, 1, N, N)$
    """
    return (mask * -100000.0).unsqueeze(1)


def local_att_mask(pos: Tensor, radius: float, dist: OptTensor = None) -> Tuple[Tensor, Tensor]:
    r"""Compute the boolean local-attention mask of a center set from a pairwise-distance threshold.

    A pair of centers is masked (`True`) when their Euclidean distance is at least `radius`. The
    pairwise-distance matrix is recomputed only when the cached `dist` does not match the current
    number of centers, so consecutive stages with the same token count share it.

    Args:
        pos: Center coordinates of shape $(B, N, 3)$.
        radius: The local-attention radius.
        dist: An optional cached pairwise-distance matrix from a previous call.

    Returns:
        A tuple `(mask, dist)` with the boolean mask of shape $(B, N, N)$ and the pairwise-distance
        matrix of shape $(B, N, N)$.

    Shape:
        - `pos`: $(B, N, 3)$
        - `mask`: $(B, N, N)$
        - `dist`: $(B, N, N)$
    """
    with torch.no_grad():
        if dist is None or dist.shape[1] != pos.shape[1]:
            dist = torch.cdist(pos, pos, p=2)
        mask = dist >= radius
    return mask, dist


def dense_centers_to_packed(centers: Tensor) -> Tuple[Tensor, Tensor]:
    r"""Flatten a densified center set $(B, G, 3)$ into packed coordinates and a batch index.

    Every sample carries the same number of centers $G$, so the packed batch index is simply each
    sample id repeated $G$ times.

    Args:
        centers: Densified center coordinates of shape $(B, G, 3)$.

    Returns:
        A tuple `(pos, batch)` with `pos` of shape $(B \cdot G, 3)$ and `batch` of shape $(B \cdot G,)$.

    Shape:
        - `centers`: $(B, G, 3)$
        - `pos`: $(B \cdot G, 3)$
        - `batch`: $(B \cdot G,)$
    """
    b, g, _ = centers.shape
    pos = centers.reshape(b * g, 3)
    batch = torch.arange(b, device=centers.device).repeat_interleave(g)
    return pos, batch


class ConvResBlock1d(nn.Module):
    r"""A residual block used by the pre-training feature-propagation extraction.

    Computes $\text{act}(W_2(\text{act}(W_1 x)) + x)$ where each linear layer is followed by a
    normalization. The residual addition prevents expressing this as a single plain-last `MLP`, so the
    inner `net1` (`linear -> norm -> act`) and `net2` (`linear -> norm`) chains are built as `MLP` and the
    residual add is kept in `forward`. A shared $1 \times 1$ convolution over $(B, C, N)$ is equivalent to
    a `MLP` over the flattened feature dim.

    Args:
        channels: The number of input and output channels.
        act: The activation function.
        act_kwargs: Extra keyword arguments for the activation.
        norm: The normalization function.
        norm_kwargs: Extra keyword arguments for the normalization.

    Shape:
        - Input: $(B \cdot N, C)$
        - Output: $(B \cdot N, C)$
    """

    def __init__(
        self,
        channels: int,
        act: Union[str, Callable, None] = "gelu",
        act_kwargs: Optional[Dict[str, Any]] = None,
        norm: Union[str, Callable, None] = "batch_norm",
        norm_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        self.act = create_act(act, **(act_kwargs or {})) or nn.Identity()
        self.net1 = MLP(
            [channels, channels], act=act, act_kwargs=act_kwargs, norm=norm, norm_kwargs=norm_kwargs, plain_last=False
        )
        self.net2 = MLP([channels, channels], act=None, norm=norm, norm_kwargs=norm_kwargs, plain_last=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.act(self.net2(self.net1(x)) + x)


class FeaturePropagation(nn.Module):
    r"""Interpolate features from a coarse set to a fine set, fuse and refine them.

    Each fine point gathers the three nearest coarse points with the shared packed `FPModule`
    interpolation, optionally concatenates the fine features, then applies a fuse `MLP` and a residual
    extraction stack. Used by the pre-training decoder. The interpolation matches the reference
    inverse-distance-squared weighting ($1 / (d^2 + 10^{-8})$) up to the $\sim 10^{-6}$ difference between
    the direct $\lVert a - b \rVert^2$ distance and the reference algebraic expansion.

    Args:
        in_channels: The number of input channels (fine + coarse features).
        out_channels: The number of output channels.
        blocks: The number of residual blocks in the extraction stack.
        act: The activation function.
        act_kwargs: Extra keyword arguments for the activation.
        norm: The normalization function.
        norm_kwargs: Extra keyword arguments for the normalization.

    Shape:
        - `centers_fine`: $(B, N, 3)$
        - `centers_coarse`: $(B, S, 3)$
        - `points1`: $(B, N, C_1)$ or `None`
        - `points2`: $(B, S, C_2)$
        - Output: $(B, N, C_\text{out})$
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        blocks: int = 1,
        act: Union[str, Callable, None] = "gelu",
        act_kwargs: Optional[Dict[str, Any]] = None,
        norm: Union[str, Callable, None] = "batch_norm",
        norm_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        self.fp = FPModule(
            in_channels=in_channels,
            channels=[out_channels],
            k=3,
            act=act,
            act_kwargs=act_kwargs,
            norm=norm,
            norm_kwargs=norm_kwargs,
            bias=True,
            weighting="squared",
            eps=1e-8,
        )
        self.extraction = nn.ModuleList(
            [
                ConvResBlock1d(out_channels, act=act, act_kwargs=act_kwargs, norm=norm, norm_kwargs=norm_kwargs)
                for _ in range(blocks)
            ]
        )

    def forward(self, centers_fine: Tensor, centers_coarse: Tensor, points1: OptTensor, points2: Tensor) -> Tensor:
        b, n, _ = centers_fine.shape
        pos, batch = dense_centers_to_packed(centers_coarse)
        pos_skip, batch_skip = dense_centers_to_packed(centers_fine)
        x = points2.reshape(pos.size(0), -1)
        x_skip = None if points1 is None else points1.reshape(pos_skip.size(0), -1)

        x, _, _ = self.fp(x, pos, batch, x_skip, pos_skip, batch_skip)
        for block in self.extraction:
            x = block(x)
        return x.reshape(b, n, -1)


class HierarchicalEncoder(nn.Module):
    r"""Multi-scale hierarchical transformer encoder of Point-M2AE.

    Each stage embeds (or merges) tokens, computes a local-attention mask from the stage centers and
    runs a transformer block stack. The forward pass operates on the visible (unmasked) tokens; in
    `eval` mode no masking is applied. The encoder consumes the precomputed multi-scale grouping
    (neighborhoods, centers and neighbor indices) so the same grouping can be shared with the decoder.

    Args:
        encoder_depths: The number of transformer blocks per stage.
        encoder_dims: The channel width per stage.
        local_radius: The local-attention radius per stage (non-positive disables the mask).
        num_heads: The number of attention heads.
        drop_path: The maximum stochastic depth rate (linearly scaled across blocks).
        with_norms: Whether to apply a per-stage output LayerNorm (disabled for the ModelNet40 / ScanObjectNN finetune heads).
        in_channels: The number of per-point feature channels concatenated to the coordinates at the first stage ($0$ for coordinates only).
        token_local_channels: Hidden widths of the first-stage token embedder's per-point MLP (later stages derive their widths from `encoder_dims`).
        token_global_channels: Hidden widths of the first-stage token embedder's per-group MLP.
        act: The activation function used in the token embedders and transformer blocks.
        act_kwargs: Extra keyword arguments for the activation.
        norm: The normalization function used in the token embedders.
        norm_kwargs: Extra keyword arguments for the normalization.
    """

    def __init__(
        self,
        encoder_depths: Sequence[int],
        encoder_dims: Sequence[int],
        local_radius: Sequence[float],
        num_heads: int,
        drop_path: float = 0.1,
        with_norms: bool = True,
        in_channels: int = 0,
        token_local_channels: Sequence[int] = (128, 256),
        token_global_channels: Sequence[int] = (512,),
        act: Union[str, Callable, None] = "relu",
        act_kwargs: Optional[Dict[str, Any]] = None,
        norm: Union[str, Callable, None] = "batch_norm",
        norm_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        self.encoder_depths = list(encoder_depths)
        self.encoder_dims = list(encoder_dims)
        self.local_radius = list(local_radius)
        self.with_norms = with_norms

        self.token_embed = nn.ModuleList()
        self.encoder_pos_embeds = nn.ModuleList()
        for i in range(len(self.encoder_dims)):
            if i == 0:
                stage_channels = 3 + in_channels
                local_channels = token_local_channels
                global_channels = token_global_channels
            else:
                stage_channels = self.encoder_dims[i - 1]
                local_channels = (stage_channels, stage_channels)
                global_channels = (self.encoder_dims[i],)
            self.token_embed.append(
                PointPatchEmbed(
                    embed_dim=self.encoder_dims[i],
                    in_channels=stage_channels,
                    local_channels=local_channels,
                    global_channels=global_channels,
                    act=act,
                    act_kwargs=act_kwargs,
                    norm=norm,
                    norm_kwargs=norm_kwargs,
                )
            )
            self.encoder_pos_embeds.append(MLP([3, self.encoder_dims[i], self.encoder_dims[i]], act="gelu", norm=None))

        self.encoder_blocks = nn.ModuleList()
        dpr = [x.item() for x in torch.linspace(0, drop_path, sum(self.encoder_depths))]
        depth_count = 0
        for i in range(len(self.encoder_depths)):
            self.encoder_blocks.append(
                EncoderBlock(
                    embed_dim=self.encoder_dims[i],
                    depth=self.encoder_depths[i],
                    drop_path=dpr[depth_count : depth_count + self.encoder_depths[i]],
                    num_heads=num_heads,
                )
            )
            depth_count += self.encoder_depths[i]

        if with_norms:
            self.encoder_norms = nn.ModuleList([nn.LayerNorm(d) for d in self.encoder_dims])

    def forward(
        self,
        neighborhoods: List[Tensor],
        centers: List[Tensor],
        idxs: List[Tensor],
        return_stages: bool = False,
    ) -> Union[Tensor, List[Tensor]]:
        x_vis_list: List[Tensor] = []
        xyz_dist: OptTensor = None
        x_vis = torch.empty(0)
        for i in range(len(centers)):
            if i == 0:
                group_input_tokens = self.token_embed[i](neighborhoods[0])
            else:
                b, g1, _ = x_vis.shape
                b, g2, k2, _ = neighborhoods[i].shape
                x_vis_neighborhoods = x_vis.reshape(b * g1, -1)[idxs[i], :].reshape(b, g2, k2, -1)
                group_input_tokens = self.token_embed[i](x_vis_neighborhoods)

            if self.local_radius[i] > 0:
                mask_vis_att, xyz_dist = local_att_mask(centers[i], self.local_radius[i], xyz_dist)
                attn_mask: OptTensor = local_attention_mask(mask_vis_att.float())
            else:
                attn_mask = None

            pos = self.encoder_pos_embeds[i](centers[i])
            x_vis = self.encoder_blocks[i](group_input_tokens, pos, attn_mask)
            x_vis_list.append(x_vis)

        if self.with_norms:
            for i in range(len(x_vis_list)):
                x_vis_list[i] = self.encoder_norms[i](x_vis_list[i])

        if return_stages:
            return x_vis_list
        return x_vis_list[-1]


def multi_scale_group(
    pos: Tensor,
    batch: Tensor,
    num_groups: Sequence[int],
    group_sizes: Sequence[int],
    random_start: bool = False,
) -> Tuple[List[Tensor], List[Tensor], List[Tensor]]:
    r"""Build the multi-scale grouping by repeatedly applying FPS + KNN on the previous-stage centers.

    Args:
        pos: Packed coordinates of shape $(N, 3)$.
        batch: Per-point batch index of shape $(N,)$.
        num_groups: The number of centers per stage.
        group_sizes: The neighborhood size per stage.

    Returns:
        A tuple `(neighborhoods, centers, idxs)` of per-stage tensors. `centers[i]` is $(B, G_i, 3)$,
        `neighborhoods[i]` is $(B, G_i, k_i, 3)$ and `idxs[i]` is the flat neighbor index into stage $i-1$.
    """
    neighborhoods: List[Tensor] = []
    centers: List[Tensor] = []
    idxs: List[Tensor] = []
    batch_size = int(batch.max().item()) + 1
    for i in range(len(num_groups)):
        if i == 0:
            neighborhood, center, idx = group(
                pos, batch, num_groups[i], group_sizes[i], random_start=random_start, return_indices=True
            )
        else:
            prev = centers[i - 1]
            prev_pos = prev.reshape(-1, 3)
            prev_batch = torch.arange(batch_size, device=pos.device).repeat_interleave(prev.size(1))
            neighborhood, center, idx = group(
                prev_pos,
                prev_batch,
                num_groups[i],
                group_sizes[i],
                random_start=random_start,
                return_indices=True,
            )

        neighborhoods.append(neighborhood)
        centers.append(center)
        idxs.append(idx)
    return neighborhoods, centers, idxs


class PointM2AEClassification(ClassificationModel):
    r"""Implementation of the Point-M2AE classification model.

    :arxiv: [Point-M2AE: Multi-scale Masked Autoencoders for Hierarchical Point Cloud Pre-training](https://arxiv.org/abs/2205.14401).
    This implementation is adapted from the official repository :github: [ZrrSkywalker/Point-M2AE](https://github.com/ZrrSkywalker/Point-M2AE).

    The model groups the input cloud at multiple scales, encodes it with the hierarchical
    transformer encoder and pools the finest-stage tokens into a feature passed to an MLP head.
    The pooling matches the reference: ModelNet40 sums the token mean and max, while ScanObjectNN
    concatenates the mean over all tokens with the max over the tokens after the first one.

    Args:
        in_channels: The number of per-point feature channels concatenated to the coordinates ($0$ for coordinates only).
        num_classes: The number of output classes.
        group_sizes: The neighborhood size per stage.
        num_groups: The number of centers per stage.
        encoder_depths: The number of transformer blocks per stage.
        encoder_dims: The channel width per stage.
        token_local_channels: Hidden widths of the first-stage token embedder's per-point MLP.
        token_global_channels: Hidden widths of the first-stage token embedder's per-group MLP.
        local_radius: The local-attention radius per stage.
        num_heads: The number of attention heads.
        drop_path: The maximum stochastic depth rate.
        concat_pooling: Use the ScanObjectNN concat pooling when `True`, the ModelNet40 sum pooling otherwise.
        dropout: The dropout rate in the classification head.
        head_channels: The hidden widths of the classification head.

    Shape:
        - `pos`: $(N, 3)$
        - `batch`: $(N,)$
        - Output: $(B, C)$ where $C$ is the number of classes.
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        *,
        group_sizes: Sequence[int] = (16, 8, 8),
        num_groups: Sequence[int] = (512, 256, 64),
        encoder_depths: Sequence[int] = (5, 5, 5),
        encoder_dims: Sequence[int] = (96, 192, 384),
        token_local_channels: Sequence[int] = (128, 256),
        token_global_channels: Sequence[int] = (512,),
        local_radius: Sequence[float] = (0.32, 0.64, 1.28),
        num_heads: int = 6,
        drop_path: float = 0.1,
        concat_pooling: bool = False,
        dropout: float = 0.5,
        head_channels: Sequence[int] = (256, 256),
    ):
        super().__init__(in_channels=in_channels, num_classes=num_classes)
        self.group_sizes = list(group_sizes)
        self.num_groups = list(num_groups)
        self.encoder_dims = list(encoder_dims)
        self.feat_dim = encoder_dims[-1]
        self.concat_pooling = concat_pooling
        self.dropout = dropout
        self.head_channels = list(head_channels)

        self.h_encoder = HierarchicalEncoder(
            encoder_depths=encoder_depths,
            encoder_dims=encoder_dims,
            local_radius=local_radius,
            num_heads=num_heads,
            drop_path=drop_path,
            with_norms=False,
            in_channels=in_channels,
            token_local_channels=token_local_channels,
            token_global_channels=token_global_channels,
        )

        self.norm = nn.LayerNorm(self.feat_dim)
        self.head = self.configure_head()

    def configure_head(self) -> nn.Module:
        if self.num_classes == 0:
            return nn.Identity()
        head_in = self.feat_dim * 2 if self.concat_pooling else self.feat_dim
        return MLP(
            [head_in, *self.head_channels, self.num_classes],
            act="relu",
            norm="batch_norm",
            dropout=self.dropout,
            plain_last=True,
        )

    def reset_classifier(self, num_classes: int, global_pool: Any = None, **kwargs: Any) -> None:
        if global_pool is not None:
            raise ValueError(
                f"{self.__class__.__name__} pools with a fixed mean / max scheme selected by `concat_pooling`; "
                "`global_pool` is not configurable."
            )
        self.num_classes = num_classes
        self.head = self.configure_head()

    def forward_features(self, x: OptTensor, pos: Tensor, batch: Tensor) -> Tensor:
        neighborhoods, centers, idxs = multi_scale_group(
            pos,
            batch,
            self.num_groups,
            self.group_sizes,
            random_start=self.training,
        )
        if x is not None:
            extra = x[idxs[0]].reshape(*neighborhoods[0].shape[:3], -1)
            neighborhoods[0] = torch.cat([neighborhoods[0], extra], dim=-1)
        x_vis = self.h_encoder(neighborhoods, centers, idxs)
        return self.norm(x_vis)

    def forward_head(self, x: Tensor, pre_logits: bool = False) -> Tensor:
        if self.concat_pooling:
            x_cat = torch.cat((x.mean(1), x[:, 1:].max(1)[0]), dim=1)
        else:
            x_cat = x.mean(1) + x.max(1)[0]

        return x_cat if pre_logits else self.head(x_cat)

    def forward(self, x: OptTensor, pos: Tensor, batch: Tensor) -> Tensor:
        x_vis = self.forward_features(x, pos, batch)
        return self.forward_head(x_vis)


class PointM2AESegmentation(SegmentationModel):
    r"""Implementation of the Point-M2AE part-segmentation model.

    :arxiv: [Point-M2AE: Multi-scale Masked Autoencoders for Hierarchical Point Cloud Pre-training](https://arxiv.org/abs/2205.14401).
    This implementation is adapted from the official repository :github: [ZrrSkywalker/Point-M2AE](https://github.com/ZrrSkywalker/Point-M2AE).

    Per-stage encoder features are propagated back to the full-resolution cloud, concatenated with
    a global feature and the one-hot object label, and decoded into per-point logits. Every sample
    in the packed batch must contain the same number of points; a ragged batch raises `ValueError`.

    Args:
        in_channels: The number of per-point feature channels concatenated to the coordinates ($0$ for coordinates only).
        num_classes: The number of part-segmentation classes.
        num_categories: The number of object categories (for the one-hot label embedding).
        group_sizes: The neighborhood size per stage.
        num_groups: The number of centers per stage.
        encoder_depths: The number of transformer blocks per stage.
        encoder_dims: The channel width per stage.
        token_local_channels: Hidden widths of the first-stage token embedder's per-point MLP.
        token_global_channels: Hidden widths of the first-stage token embedder's per-group MLP.
        local_radius: The local-attention radius per stage.
        num_heads: The number of attention heads.

    Shape:
        - `pos`: $(N, 3)$
        - `batch`: $(N,)$
        - `category`: $(B, \text{num\_categories})$ one-hot object label
        - Output: $(N, C)$ where $C$ is the number of part classes.
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        *,
        num_categories: int = 16,
        group_sizes: Sequence[int] = (16, 8, 8),
        num_groups: Sequence[int] = (512, 256, 64),
        encoder_depths: Sequence[int] = (5, 5, 5),
        encoder_dims: Sequence[int] = (96, 192, 384),
        token_local_channels: Sequence[int] = (128, 256),
        token_global_channels: Sequence[int] = (512,),
        local_radius: Sequence[float] = (0.32, 0.64, 1.28),
        num_heads: int = 6,
    ):
        super().__init__(in_channels=in_channels, num_classes=num_classes)
        self.group_sizes = list(group_sizes)
        self.num_groups = list(num_groups)
        self.encoder_dims = list(encoder_dims)
        self.num_categories = num_categories
        self.embed_dim = encoder_dims[-1]

        self.h_encoder = HierarchicalEncoder(
            encoder_depths=encoder_depths,
            encoder_dims=encoder_dims,
            local_radius=local_radius,
            num_heads=num_heads,
            with_norms=True,
            in_channels=in_channels,
            token_local_channels=token_local_channels,
            token_global_channels=token_global_channels,
        )

        self.label_conv = MLP(
            [num_categories, 64],
            act="leaky_relu",
            act_kwargs=dict(negative_slope=0.2),
            norm="batch_norm",
            bias=False,
            plain_last=False,
        )

        self.propagations = nn.ModuleList(
            [
                FPModule(
                    in_channels=encoder_dims[i] + 3,
                    channels=[self.embed_dim * 4, 1024],
                    k=3,
                    act="relu",
                    norm="batch_norm",
                    bias=True,
                    weighting="squared",
                    eps=1e-8,
                )
                for i in range(3)
            ]
        )

        self.head = self.configure_head()

    def configure_head(self) -> nn.Module:
        if self.num_classes == 0:
            return nn.Identity()
        return MLP(
            [3 * 1024 * 2 + 64, 1024, 512, 256, self.num_classes],
            act="relu",
            norm="batch_norm",
            dropout=[0.5, 0.0, 0.0, 0.0],
            bias=True,
            plain_last=True,
        )

    def reset_classifier(self, num_classes: int, **kwargs: Any) -> None:
        self.num_classes = num_classes
        self.head = self.configure_head()

    def forward_features(self, x: OptTensor, pos: Tensor, batch: Tensor) -> Tuple[List[Tensor], List[Tensor]]:
        neighborhoods, centers, idxs = multi_scale_group(
            pos,
            batch,
            self.num_groups,
            self.group_sizes,
            random_start=self.training,
        )
        if x is not None:
            extra = x[idxs[0]].reshape(*neighborhoods[0].shape[:3], -1)
            neighborhoods[0] = torch.cat([neighborhoods[0], extra], dim=-1)
        x_vis_list = self.h_encoder(neighborhoods, centers, idxs, return_stages=True)
        return x_vis_list, centers

    def forward_decoder(self, x_vis_list: List[Tensor], centers: List[Tensor], pos: Tensor, batch: Tensor) -> Tensor:
        batch_size = int(batch.max().item()) + 1
        counts = batch.bincount()
        if bool((counts != counts[0]).any()):
            raise ValueError(
                f"{self.__class__.__name__} requires the same number of points per sample, got per-sample "
                f"counts from {int(counts.min())} to {int(counts.max())}."
            )
        num_points = pos.size(0) // batch_size
        feats: List[Tensor] = []
        for i in range(len(x_vis_list)):
            center_pos, center_batch = dense_centers_to_packed(centers[i])
            x_groups = x_vis_list[i].reshape(center_pos.size(0), -1)
            f, _, _ = self.propagations[i](x_groups, center_pos, center_batch, pos, pos, batch)
            feats.append(f.reshape(batch_size, num_points, -1).transpose(1, 2))
        return torch.cat(feats, dim=1)

    def forward_head(self, x: Tensor, category: Tensor, pre_logits: bool = False) -> Tensor:
        B = x.size(0)
        N = x.size(2)
        x_max = torch.max(x, 2)[0]
        x_avg = torch.mean(x, 2)
        x_max_feature = x_max.view(B, -1).unsqueeze(-1).repeat(1, 1, N)
        x_avg_feature = x_avg.view(B, -1).unsqueeze(-1).repeat(1, 1, N)
        cls_label_feature = self.label_conv(category).unsqueeze(-1).repeat(1, 1, N)
        x_global_feature = torch.cat((x_max_feature + x_avg_feature, cls_label_feature), 1)

        x = torch.cat((x_global_feature, x), 1)
        x = x.transpose(1, 2).reshape(B * N, -1)
        if pre_logits:
            return x
        x = self.head(x)
        return x.reshape(B, N, -1).transpose(1, 2)

    def forward(self, x: OptTensor, pos: Tensor, batch: Tensor, category: Tensor) -> Tensor:
        batch_size = int(batch.max().item()) + 1
        num_points = pos.size(0) // batch_size

        x_vis_list, centers = self.forward_features(x, pos, batch)
        x = self.forward_decoder(x_vis_list, centers, pos, batch)
        logits = self.forward_head(x, category)
        return logits.permute(0, 2, 1).reshape(batch_size * num_points, -1)


class PointM2AEMaskedAutoEncoder(BaseModel):
    r"""Implementation of the Point-M2AE pre-training model.

    :arxiv: [Point-M2AE: Multi-scale Masked Autoencoders for Hierarchical Point Cloud Pre-training](https://arxiv.org/abs/2205.14401).
    This implementation is adapted from the official repository :github: [ZrrSkywalker/Point-M2AE](https://github.com/ZrrSkywalker/Point-M2AE).

    The model masks tokens at the coarsest stage, back-propagates the mask through the multi-scale
    grouping, encodes the visible tokens with the hierarchical encoder and reconstructs the masked
    local neighborhoods with a hierarchical decoder and a reconstruction head.

    Args:
        in_channels: The number of input channels (unused; coordinates drive the grouping).
        group_sizes: The neighborhood size per stage.
        num_groups: The number of centers per stage.
        mask_ratio: The fraction of coarsest-stage tokens to mask.
        encoder_depths: The number of encoder blocks per stage.
        encoder_dims: The encoder channel width per stage.
        token_local_channels: Hidden widths of the first-stage token embedder's per-point MLP.
        token_global_channels: Hidden widths of the first-stage token embedder's per-group MLP.
        local_radius: The local-attention radius per stage (disabled during pre-training).
        decoder_depths: The number of decoder blocks per stage.
        decoder_dims: The decoder channel width per stage.
        decoder_up_blocks: The number of residual blocks in each feature-propagation stage.
        num_heads: The number of attention heads.
        drop_path: The maximum stochastic depth rate.

    Shape:
        - `pos`: $(N, 3)$
        - `batch`: $(N,)$
        - Output: a tuple `(pred, target)` of reconstructed and ground-truth neighborhoods, each $(L, k_0, 3)$.
    """

    def __init__(
        self,
        in_channels: int,
        *,
        group_sizes: Sequence[int] = (16, 8, 8),
        num_groups: Sequence[int] = (512, 256, 64),
        mask_ratio: float = 0.8,
        encoder_depths: Sequence[int] = (5, 5, 5),
        encoder_dims: Sequence[int] = (96, 192, 384),
        token_local_channels: Sequence[int] = (128, 256),
        token_global_channels: Sequence[int] = (512,),
        local_radius: Sequence[float] = (0.32, 0.64, 1.28),
        decoder_depths: Sequence[int] = (1, 1),
        decoder_dims: Sequence[int] = (384, 192),
        decoder_up_blocks: Sequence[int] = (1, 1),
        num_heads: int = 6,
        drop_path: float = 0.1,
    ):
        super().__init__(in_channels=in_channels)
        self.group_sizes = list(group_sizes)
        self.num_groups = list(num_groups)
        self.mask_ratio = mask_ratio
        self.decoder_dims = list(decoder_dims)
        self.decoder_depths = list(decoder_depths)
        self.decoder_up_blocks = list(decoder_up_blocks)

        self.h_encoder = HierarchicalEncoderMAE(
            encoder_depths=encoder_depths,
            encoder_dims=encoder_dims,
            local_radius=local_radius,
            num_heads=num_heads,
            mask_ratio=mask_ratio,
            drop_path=drop_path,
            token_local_channels=token_local_channels,
            token_global_channels=token_global_channels,
        )

        self.mask_token = nn.Parameter(torch.zeros(1, self.decoder_dims[0]))
        nn.init.trunc_normal_(self.mask_token, std=0.02)

        self.h_decoder = nn.ModuleList()
        self.decoder_pos_embeds = nn.ModuleList()
        self.token_prop = nn.ModuleList()
        dpr = [x.item() for x in torch.linspace(0, drop_path, sum(self.decoder_depths))]
        depth_count = 0
        for i in range(len(self.decoder_dims)):
            self.h_decoder.append(
                EncoderBlock(
                    embed_dim=self.decoder_dims[i],
                    depth=self.decoder_depths[i],
                    drop_path=dpr[depth_count : depth_count + self.decoder_depths[i]],
                    num_heads=num_heads,
                )
            )
            depth_count += self.decoder_depths[i]
            self.decoder_pos_embeds.append(MLP([3, self.decoder_dims[i], self.decoder_dims[i]], act="gelu", norm=None))
            if i > 0:
                self.token_prop.append(
                    FeaturePropagation(
                        self.decoder_dims[i] + self.decoder_dims[i - 1],
                        self.decoder_dims[i],
                        blocks=self.decoder_up_blocks[i - 1],
                    )
                )
        self.decoder_norm = nn.LayerNorm(self.decoder_dims[-1])
        self.rec_head = nn.Linear(self.decoder_dims[-1], 3 * self.group_sizes[0])

    def forward(self, x: OptTensor, pos: Tensor, batch: Tensor) -> Tuple[Tensor, Tensor]:
        neighborhoods, centers, idxs = multi_scale_group(
            pos,
            batch,
            self.num_groups,
            self.group_sizes,
            random_start=self.training,
        )

        x_vis_list, mask_vis_list, masks = self.h_encoder(neighborhoods, centers, idxs)

        centers = centers[::-1]
        neighborhoods = neighborhoods[::-1]
        x_vis_list = x_vis_list[::-1]
        mask_vis_list = mask_vis_list[::-1]
        masks = masks[::-1]

        center_0 = torch.empty(0, device=pos.device)
        x_full = torch.empty(0, device=pos.device)
        pos_full = torch.empty(0, device=pos.device)
        for i in range(len(self.decoder_dims)):
            center = centers[i]
            if i == 0:
                x_full, mask = x_vis_list[i], masks[i]
                B, _, C = x_full.shape
                center_0 = torch.cat((center[~mask].reshape(B, -1, 3), center[mask].reshape(B, -1, 3)), dim=1)
                pos_emd_vis = self.decoder_pos_embeds[i](center[~mask]).reshape(B, -1, C)
                pos_emd_mask = self.decoder_pos_embeds[i](center[mask]).reshape(B, -1, C)
                pos_full = torch.cat([pos_emd_vis, pos_emd_mask], dim=1)
                _, num_mask, _ = pos_emd_mask.shape
                mask_token = self.mask_token.expand(B, num_mask, -1)
                x_full = torch.cat([x_full, mask_token], dim=1)
            else:
                x_vis = x_vis_list[i]
                bool_vis_pos = ~masks[i]
                mask_vis = mask_vis_list[i]
                B, N, _ = center.shape
                _, _, C = x_vis.shape
                x_full_en = torch.zeros(B, N, C, device=pos.device)
                x_full_en[bool_vis_pos] = x_vis[mask_vis]

                anchor = center_0 if i == 1 else centers[i - 1]
                x_full = self.token_prop[i - 1](center, anchor, x_full_en, x_full)
                pos_full = self.decoder_pos_embeds[i](center)

            x_full = self.h_decoder[i](x_full, pos_full)

        x_full = self.decoder_norm(x_full)
        B, N, C = x_full.shape
        x_rec = x_full[masks[-2]].reshape(-1, C)
        L = x_rec.size(0)

        pred = self.rec_head(x_rec).reshape(L, -1, 3)
        target = neighborhoods[-2][masks[-2]].reshape(L, -1, 3)
        return pred, target


class HierarchicalEncoderMAE(nn.Module):
    r"""Hierarchical encoder variant used for pre-training, with multi-scale back-propagated masking.

    The coarsest-stage mask is sampled and propagated to finer stages through the neighbor indices,
    so a fine token is visible only if it contributes to a visible coarse token. Visible tokens are
    packed per sample to the longest visible length and processed with a per-stage local-attention
    mask.

    Args:
        encoder_depths: The number of transformer blocks per stage.
        encoder_dims: The channel width per stage.
        local_radius: The local-attention radius per stage.
        num_heads: The number of attention heads.
        mask_ratio: The fraction of coarsest-stage tokens to mask.
        drop_path: The maximum stochastic depth rate.
        token_local_channels: Hidden widths of the first-stage token embedder's per-point MLP (later stages derive their widths from `encoder_dims`).
        token_global_channels: Hidden widths of the first-stage token embedder's per-group MLP.
        act: The activation function used in the token embedders and transformer blocks.
        act_kwargs: Extra keyword arguments for the activation.
        norm: The normalization function used in the token embedders.
        norm_kwargs: Extra keyword arguments for the normalization.
    """

    def __init__(
        self,
        encoder_depths: Sequence[int],
        encoder_dims: Sequence[int],
        local_radius: Sequence[float],
        num_heads: int,
        mask_ratio: float,
        drop_path: float = 0.1,
        token_local_channels: Sequence[int] = (128, 256),
        token_global_channels: Sequence[int] = (512,),
        act: Union[str, Callable, None] = "relu",
        act_kwargs: Optional[Dict[str, Any]] = None,
        norm: Union[str, Callable, None] = "batch_norm",
        norm_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        if not 0.0 < mask_ratio < 1.0:
            raise ValueError(f"`mask_ratio` must be in (0, 1), got {mask_ratio}.")
        self.encoder_depths = list(encoder_depths)
        self.encoder_dims = list(encoder_dims)
        self.local_radius = list(local_radius)
        self.mask_ratio = mask_ratio

        self.token_embed = nn.ModuleList()
        self.encoder_pos_embeds = nn.ModuleList()
        for i in range(len(self.encoder_dims)):
            if i == 0:
                stage_channels = 3
                local_channels = token_local_channels
                global_channels = token_global_channels
            else:
                stage_channels = self.encoder_dims[i - 1]
                local_channels = (stage_channels, stage_channels)
                global_channels = (self.encoder_dims[i],)
            self.token_embed.append(
                PointPatchEmbed(
                    embed_dim=self.encoder_dims[i],
                    in_channels=stage_channels,
                    local_channels=local_channels,
                    global_channels=global_channels,
                    act=act,
                    act_kwargs=act_kwargs,
                    norm=norm,
                    norm_kwargs=norm_kwargs,
                )
            )
            self.encoder_pos_embeds.append(MLP([3, self.encoder_dims[i], self.encoder_dims[i]], act="gelu", norm=None))

        self.encoder_blocks = nn.ModuleList()
        dpr = [x.item() for x in torch.linspace(0, drop_path, sum(self.encoder_depths))]
        depth_count = 0
        for i in range(len(self.encoder_depths)):
            self.encoder_blocks.append(
                EncoderBlock(
                    embed_dim=self.encoder_dims[i],
                    depth=self.encoder_depths[i],
                    drop_path=dpr[depth_count : depth_count + self.encoder_depths[i]],
                    num_heads=num_heads,
                )
            )
            depth_count += self.encoder_depths[i]
        self.encoder_norms = nn.ModuleList([nn.LayerNorm(d) for d in self.encoder_dims])

    def rand_mask(self, center: Tensor) -> Tensor:
        B, G, _ = center.shape
        num_mask = int(self.mask_ratio * G)
        overall_mask = torch.zeros(B, G, device=center.device)
        for i in range(B):
            perm = torch.randperm(G, device=center.device)
            overall_mask[i, perm[:num_mask]] = 1
        return overall_mask.to(torch.bool)

    def forward(
        self,
        neighborhoods: List[Tensor],
        centers: List[Tensor],
        idxs: List[Tensor],
        bool_masked_pos_top: OptTensor = None,
    ) -> Tuple[List[Tensor], List[Tensor], List[Tensor]]:
        bool_masked_pos: List[Tensor] = []
        if bool_masked_pos_top is None:
            bool_masked_pos.append(self.rand_mask(centers[-1]))
        else:
            bool_masked_pos.append(bool_masked_pos_top)

        for i in range(len(neighborhoods) - 1, 0, -1):
            b, g, k, _ = neighborhoods[i].shape
            idx = idxs[i].reshape(b * g, -1)
            idx_masked = ~(bool_masked_pos[-1].reshape(-1).unsqueeze(-1)) * idx
            idx_masked = idx_masked.reshape(-1).long()
            masked_pos = torch.ones(b * centers[i - 1].shape[1], device=idx.device).scatter(0, idx_masked, 0).bool()
            bool_masked_pos.append(masked_pos.reshape(b, centers[i - 1].shape[1]))

        bool_masked_pos.reverse()
        x_vis_list: List[Tensor] = []
        mask_vis_list: List[Tensor] = []
        xyz_dist: OptTensor = None
        x_vis = torch.empty(0)
        for i in range(len(centers)):
            if i == 0:
                group_input_tokens = self.token_embed[i](neighborhoods[0])
            else:
                b, g1, _ = x_vis.shape
                b, g2, k2, _ = neighborhoods[i].shape
                x_vis_neighborhoods = x_vis.reshape(b * g1, -1)[idxs[i], :].reshape(b, g2, k2, -1)
                group_input_tokens = self.token_embed[i](x_vis_neighborhoods)

            bool_vis_pos = ~(bool_masked_pos[i])
            batch_size, seq_len, C = group_input_tokens.size()

            vis_tokens_len = bool_vis_pos.long().sum(dim=1)
            max_tokens_len = int(torch.max(vis_tokens_len).item())
            x_vis = torch.zeros(batch_size, max_tokens_len, C, device=group_input_tokens.device)
            masked_center = torch.zeros(batch_size, max_tokens_len, 3, device=group_input_tokens.device)
            mask_vis = torch.ones(batch_size, max_tokens_len, max_tokens_len, device=group_input_tokens.device)

            for bz in range(batch_size):
                vis_tokens = group_input_tokens[bz][bool_vis_pos[bz]]
                x_vis[bz][0 : vis_tokens_len[bz]] = vis_tokens
                vis_centers = centers[i][bz][bool_vis_pos[bz]]
                masked_center[bz][0 : vis_tokens_len[bz]] = vis_centers
                mask_vis[bz][0 : vis_tokens_len[bz], 0 : vis_tokens_len[bz]] = 0

            if self.local_radius[i] > 0:
                mask_radius, xyz_dist = local_att_mask(masked_center, self.local_radius[i], xyz_dist)
                mask_vis_att = mask_radius * mask_vis
            else:
                mask_vis_att = mask_vis

            pos = self.encoder_pos_embeds[i](masked_center)
            x_vis = self.encoder_blocks[i](x_vis, pos, local_attention_mask(mask_vis_att))
            x_vis_list.append(x_vis)
            mask_vis_list.append(~(mask_vis[:, :, 0].bool()))

            if i != len(centers) - 1:
                group_input_tokens[bool_vis_pos] = x_vis[~(mask_vis[:, :, 0].bool())]
                x_vis = group_input_tokens

        for i in range(len(x_vis_list)):
            x_vis_list[i] = self.encoder_norms[i](x_vis_list[i])

        return x_vis_list, mask_vis_list, bool_masked_pos


@register_model(
    "point-m2ae-base.modelnet40.renrui-zhang",
    task="classification",
    weights=WeightsDict(
        url="hf://torch-pointcloud/point-m2ae/point-m2ae-base.modelnet40.renrui-zhang.safetensors",
        dataset="modelnet40",
        metrics={"OA": 92.87},
        classes=MODELNET40_CLASSES,
        author="renrui-zhang",
        license="MIT",
    ),
    transform=T.Compose(
        [
            T.Rescale(keys=DataKeys.POS),
            T.FarthestPointSample(pos_key=DataKeys.POS, num_samples=1024, random_start=False),
        ]
    ),
    hparams=dict(
        in_channels=0,
        num_classes=40,
        group_sizes=(16, 8, 8),
        num_groups=(512, 256, 64),
        encoder_depths=(5, 5, 5),
        encoder_dims=(96, 192, 384),
        token_local_channels=(128, 256),
        token_global_channels=(512,),
        local_radius=(0.32, 0.64, 1.28),
        num_heads=6,
        drop_path=0.1,
        concat_pooling=False,
        dropout=0.5,
        head_channels=(256, 256),
    ),
)
def point_m2ae_base_modelnet40(**kwargs: Any) -> PointM2AEClassification:
    return PointM2AEClassification(**kwargs)


@register_model(
    "point-m2ae-base.scanobjectnn-hardest.renrui-zhang",
    task="classification",
    weights=WeightsDict(
        url="hf://torch-pointcloud/point-m2ae/point-m2ae-base.scanobjectnn-hardest.renrui-zhang.safetensors",
        dataset="scanobjectnn-hardest",
        metrics={"OA": 86.54},
        classes=SCANOBJECTNN_CLASSES,
        author="renrui-zhang",
        license="MIT",
    ),
    transform=T.Compose([T.FarthestPointSample(pos_key=DataKeys.POS, num_samples=2048, random_start=False)]),
    hparams=dict(
        in_channels=0,
        num_classes=15,
        group_sizes=(32, 16, 16),
        num_groups=(512, 256, 64),
        encoder_depths=(5, 5, 5),
        encoder_dims=(96, 192, 384),
        token_local_channels=(128, 256),
        token_global_channels=(512,),
        local_radius=(0.32, 0.64, 1.28),
        num_heads=6,
        drop_path=0.1,
        concat_pooling=True,
        dropout=0.5,
        head_channels=(256, 256),
    ),
)
def point_m2ae_base_scanobjectnn_hardest(**kwargs: Any) -> PointM2AEClassification:
    return PointM2AEClassification(**kwargs)


@register_model(
    "point-m2ae-base.scanobjectnn-objbg.renrui-zhang",
    task="classification",
    weights=WeightsDict(
        url="hf://torch-pointcloud/point-m2ae/point-m2ae-base.scanobjectnn-objbg.renrui-zhang.safetensors",
        dataset="scanobjectnn-objbg",
        metrics={"OA": 91.22},
        classes=SCANOBJECTNN_CLASSES,
        author="renrui-zhang",
        license="MIT",
    ),
    transform=T.Compose([T.FarthestPointSample(pos_key=DataKeys.POS, num_samples=2048, random_start=False)]),
    hparams=dict(
        in_channels=0,
        num_classes=15,
        group_sizes=(32, 16, 16),
        num_groups=(512, 256, 64),
        encoder_depths=(5, 5, 5),
        encoder_dims=(96, 192, 384),
        token_local_channels=(128, 256),
        token_global_channels=(512,),
        local_radius=(0.32, 0.64, 1.28),
        num_heads=6,
        drop_path=0.1,
        concat_pooling=True,
        dropout=0.5,
        head_channels=(256, 256),
    ),
)
def point_m2ae_base_scanobjectnn_objbg(**kwargs: Any) -> PointM2AEClassification:
    return PointM2AEClassification(**kwargs)


@register_model(
    "point-m2ae-base.shapenetpart.renrui-zhang",
    task="segmentation",
    weights=WeightsDict(
        url="hf://torch-pointcloud/point-m2ae/point-m2ae-base.shapenetpart.renrui-zhang.safetensors",
        dataset="shapenetpart",
        author="renrui-zhang",
        license="MIT",
    ),
    transform=T.Compose(
        [
            T.Rescale(keys=DataKeys.POS, method="centroid"),
            T.FarthestPointSample(pos_key=DataKeys.POS, keys=("segment",), num_samples=2048, random_start=False),
            T.OneHot(keys="category", num_classes=16),
        ]
    ),
    hparams=dict(
        in_channels=0,
        num_classes=50,
        num_categories=16,
        group_sizes=(16, 8, 8),
        num_groups=(512, 256, 64),
        encoder_depths=(5, 5, 5),
        encoder_dims=(96, 192, 384),
        token_local_channels=(128, 256),
        token_global_channels=(512,),
        local_radius=(0.32, 0.64, 1.28),
        num_heads=6,
    ),
)
def point_m2ae_base_shapenetpart(**kwargs: Any) -> PointM2AESegmentation:
    return PointM2AESegmentation(**kwargs)


@register_model(
    "point-m2ae-base.pretrain.renrui-zhang",
    task="base",
    weights=WeightsDict(
        url="hf://torch-pointcloud/point-m2ae/point-m2ae-base.pretrain.renrui-zhang.safetensors",
        dataset="shapenet55",
        author="renrui-zhang",
        license="MIT",
    ),
    hparams=dict(
        in_channels=0,
        group_sizes=(16, 8, 8),
        num_groups=(512, 256, 64),
        mask_ratio=0.8,
        encoder_depths=(5, 5, 5),
        encoder_dims=(96, 192, 384),
        token_local_channels=(128, 256),
        token_global_channels=(512,),
        local_radius=(0.32, 0.64, 1.28),
        decoder_depths=(1, 1),
        decoder_dims=(384, 192),
        decoder_up_blocks=(1, 1),
        num_heads=6,
        drop_path=0.1,
    ),
)
def point_m2ae_base_pretrain(**kwargs: Any) -> PointM2AEMaskedAutoEncoder:
    return PointM2AEMaskedAutoEncoder(**kwargs)
