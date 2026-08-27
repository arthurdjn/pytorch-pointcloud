r"""3DETR: an end-to-end transformer detector for 3D point clouds.

{{ paper("2109.08141") }}

Reference: :arxiv: [Misra et al., 2021](https://arxiv.org/abs/2109.08141).
Reference implementation: :github: [facebookresearch/3detr](https://github.com/facebookresearch/3detr).
"""

import math
from typing import Any, Callable, Dict, List, Optional, Tuple, TypedDict, Union

import torch
import torch.nn as nn
from torch import Tensor

import torch_pointcloud.transforms as T
from torch_pointcloud.datasets.scannet import SCANNET_DETECTION_CLASSES
from torch_pointcloud.datasets.sunrgbd import SUNRGBD_CLASSES
from torch_pointcloud.layers import create_act, create_norm
from torch_pointcloud.layers.pointnet2_blocks import SAModule
from torch_pointcloud.utils.cluster import fps
from torch_pointcloud.utils.data import DataKeys
from torch_pointcloud.utils.types import Detection3D, OptTensor

from ._base import DetectionModel
from ._registry import WeightsDict, register_model


def _to_dense(x: Tensor, batch: Tensor, num_points: int) -> Tensor:
    r"""Reshape a packed per-scene tensor with equal counts into a dense $(B, P, C)$ batch.

    3DETR operates on fixed-size point sets (every scene is sampled to the same count), so the packed
    boundary tensor densifies without padding.

    Args:
        x: Packed features, shape $(B \cdot P, C)$.
        batch: Per-row scene index, shape $(B \cdot P,)$.
        num_points: Points per scene $P$.

    Returns:
        Dense features, shape $(B, P, C)$.
    """
    batch_size = int(batch.max().item()) + 1 if batch.numel() else 0
    return x.view(batch_size, num_points, x.size(-1))


class DETR3DOutput(TypedDict):
    r"""Decoded 3DETR predictions for a batch of $B$ scenes with $Q$ queries each (last decoder layer)."""

    sem_cls_logits: Tensor
    center_unnormalized: Tensor
    size_unnormalized: Tensor
    angle_logits: Tensor
    angle_residual: Tensor
    angle_continuous: Tensor
    objectness_prob: Tensor
    sem_cls_prob: Tensor


class DETR3DTrainOutput(DETR3DOutput, total=False):
    r"""Training-mode 3DETR output: the eval `DETR3DOutput` plus the per-decoder-layer head outputs.

    `aux_outputs` holds one dict per decoder layer (the last entry mirrors the top-level eval fields),
    each carrying the normalized and unnormalized head quantities the set-prediction loss consumes, and
    `point_cloud_dims` is the per-scene $(\text{lo}, \text{hi})$ min-max extent used to normalize centers
    and sizes. These extra keys are present only when the model is in training mode; the eval forward
    returns exactly the `DETR3DOutput` keys.
    """

    aux_outputs: List[Dict[str, Tensor]]
    point_cloud_dims: Tuple[Tensor, Tensor]


class PointnetSAModuleVotes(nn.Module):
    r"""Set-abstraction tokenizer mirroring 3DETR's `PointnetSAModuleVotes` (single-scale, max pool).

    Wraps [`SAModule`][torch_pointcloud.layers.pointnet2_blocks.SAModule] with the reference settings:
    farthest-point-sampled centroids, a ball query that normalizes the relative position by the radius and
    concatenates it before the grouped features (`pos_first`), and a shared MLP whose first width already
    accounts for the $3$ position channels. Returns the sampling index so the encoder can trace tokens back
    to the input.

    Args:
        in_channels: Input feature channels per point (excluding xyz).
        channels: Shared-MLP widths after the input layer, e.g. `[64, 128, 256]`.
        num_points: Number of farthest-point-sampled centroids.
        radius: Ball-query radius.
        num_neighbors: Ball-query neighbor cap.
        act: Activation type or callable.
        act_kwargs: Extra activation arguments.
        norm: Normalization type or callable.
        norm_kwargs: Extra normalization arguments.
    """

    def __init__(
        self,
        in_channels: int,
        channels: List[int],
        *,
        num_points: int,
        radius: float,
        num_neighbors: int,
        act: Union[str, Callable, None] = "relu",
        act_kwargs: Optional[Dict[str, Any]] = None,
        norm: Union[str, Callable, None] = "batch_norm",
        norm_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()
        self.num_points = num_points
        self.out_channels = channels[-1]
        self.sa = SAModule(
            in_channels=in_channels,
            channels=list(channels),
            num_points=num_points,
            radii=radius,
            num_neighbors=num_neighbors,
            use_pos=True,
            normalize_pos=True,
            pos_first=True,
            sort_neighbors=True,
            pool="max",
            bias=False,
            act=act,
            act_kwargs=act_kwargs,
            norm=norm,
            norm_kwargs=norm_kwargs,
        )

    def forward(
        self,
        x: OptTensor,
        pos: Tensor,
        batch: Tensor,
        idx: OptTensor = None,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        if idx is None:
            idx = fps(pos, batch, num_nodes=self.num_points, random_start=self.training)
        if x is None:
            x = pos.new_zeros((pos.size(0), 0))
        x_out, pos_out, batch_out = self.sa(x, pos, batch, idx)
        return x_out, pos_out, batch_out, idx


class TransformerEncoderLayer(nn.Module):
    r"""Pre-norm transformer encoder layer with positional embeddings added to the attention query/key.

    Mirrors the reference 3DETR encoder layer (`normalize_before=True`): self-attention over the tokens
    plus a feed-forward block, each wrapped in a residual with the layer norm applied to the input.

    Args:
        embed_dim: Token embedding dimension.
        num_heads: Number of attention heads.
        mlp_dim: Hidden width of the feed-forward block.
        dropout: Dropout probability.
        act: Activation type or callable for the feed-forward block.
        act_kwargs: Extra activation arguments.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        mlp_dim: int,
        dropout: float,
        *,
        act: Union[str, Callable, None] = "relu",
        act_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.linear1 = nn.Linear(embed_dim, mlp_dim)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(mlp_dim, embed_dim)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = create_act(act, **(act_kwargs or {})) or nn.ReLU()

    def forward(self, src: Tensor, src_mask: OptTensor = None, pos: OptTensor = None) -> Tensor:
        src2 = self.norm1(src)
        q = k = src2 if pos is None else src2 + pos
        src2 = self.self_attn(q, k, value=src2, attn_mask=src_mask)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src


class TransformerDecoderLayer(nn.Module):
    r"""Pre-norm transformer decoder layer with self-attention, cross-attention and a feed-forward block.

    Mirrors the reference 3DETR decoder layer (`normalize_before=True`). Query positions are added to the
    self-attention query/key and to the cross-attention query; encoder positions are added to the
    cross-attention key.

    Args:
        embed_dim: Token embedding dimension.
        num_heads: Number of attention heads.
        mlp_dim: Hidden width of the feed-forward block.
        dropout: Dropout probability.
        act: Activation type or callable for the feed-forward block.
        act_kwargs: Extra activation arguments.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        mlp_dim: int,
        dropout: float,
        *,
        act: Union[str, Callable, None] = "relu",
        act_kwargs: Optional[Dict[str, Any]] = None,
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
        self.activation = create_act(act, **(act_kwargs or {})) or nn.ReLU()

    def forward(
        self,
        tgt: Tensor,
        memory: Tensor,
        pos: Tensor,
        query_pos: Tensor,
    ) -> Tensor:
        tgt2 = self.norm1(tgt)
        q = k = tgt2 + query_pos
        tgt2 = self.self_attn(q, k, value=tgt2)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(
            query=tgt2 + query_pos,
            key=memory + pos,
            value=memory,
        )[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt


class TransformerEncoder(nn.Module):
    r"""Stack of `num_layers` identical pre-norm encoder layers (no final norm), per 3DETR's `vanilla`.

    Args:
        embed_dim: Token embedding dimension.
        num_heads: Number of attention heads.
        mlp_dim: Hidden width of the feed-forward block.
        num_layers: Number of stacked encoder layers.
        dropout: Dropout probability.
        act: Activation type or callable.
        act_kwargs: Extra activation arguments.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        mlp_dim: int,
        num_layers: int,
        dropout: float,
        *,
        act: Union[str, Callable, None] = "relu",
        act_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            TransformerEncoderLayer(embed_dim, num_heads, mlp_dim, dropout, act=act, act_kwargs=act_kwargs)
            for _ in range(num_layers)
        )

    def forward(self, src: Tensor, pos: Tensor) -> Tuple[Tensor, Tensor]:
        output = src
        for layer in self.layers:
            output = layer(output)
        return pos, output


class MaskedTransformerEncoder(nn.Module):
    r"""3DETR-m encoder: pre-norm layers with radius self-attention masks and one interim downsampling.

    The first layer attends within `masking_radius[0]`, then a set-abstraction layer halves the token
    count, and the remaining layers attend within the later radii. Mirrors the reference
    `MaskedTransformerEncoder`.

    Args:
        embed_dim: Token embedding dimension.
        num_heads: Number of attention heads.
        mlp_dim: Hidden width of the feed-forward block.
        num_layers: Number of stacked encoder layers (must equal `len(masking_radius)`).
        dropout: Dropout probability.
        masking_radius: Per-layer attention radius (a value $\le 0$ disables masking for that layer).
        interim_downsampling: Set-abstraction layer applied after the first encoder layer.
        act: Activation type or callable.
        act_kwargs: Extra activation arguments.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        mlp_dim: int,
        num_layers: int,
        dropout: float,
        *,
        masking_radius: List[float],
        interim_downsampling: PointnetSAModuleVotes,
        act: Union[str, Callable, None] = "relu",
        act_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()
        if len(masking_radius) != num_layers:
            raise ValueError(f"`masking_radius` must have {num_layers} entries, got {len(masking_radius)}.")
        self.layers = nn.ModuleList(
            TransformerEncoderLayer(embed_dim, num_heads, mlp_dim, dropout, act=act, act_kwargs=act_kwargs)
            for _ in range(num_layers)
        )
        self.masking_radius = masking_radius
        self.interim_downsampling = interim_downsampling

    @staticmethod
    def compute_mask(pos: Tensor, radius: float) -> Tensor:
        r"""Builds the boolean attention mask hiding token pairs farther apart than `radius`.

        Args:
            pos: Dense token positions, shape $(B, P, 3)$.
            radius: Attention radius.

        Returns:
            Mask of shape $(B, P, P)$, `True` where attention is blocked.
        """
        with torch.no_grad():
            dist = torch.cdist(pos, pos, p=2)
            mask = dist >= radius
        return mask

    def forward(self, src: Tensor, pos: Tensor) -> Tuple[Tensor, Tensor]:
        output = src
        batch_size = pos.size(0)
        for idx, layer in enumerate(self.layers):
            assert isinstance(layer, TransformerEncoderLayer)
            mask: OptTensor = None
            if self.masking_radius[idx] > 0:
                bool_mask = self.compute_mask(pos, self.masking_radius[idx])
                num_heads = layer.num_heads
                n = bool_mask.size(1)
                mask = bool_mask.unsqueeze(1).repeat(1, num_heads, 1, 1).view(batch_size * num_heads, n, n)
            output = layer(output, src_mask=mask)

            if idx == 0:
                tokens = output.size(0)
                packed = output.permute(1, 0, 2).reshape(batch_size * tokens, -1)
                pos_packed = pos.reshape(batch_size * tokens, 3)
                token_batch = torch.arange(batch_size, device=pos.device).repeat_interleave(tokens)
                x_ds, pos_ds, _, _ = self.interim_downsampling(packed, pos_packed, token_batch)
                new_tokens = self.interim_downsampling.num_points
                output = x_ds.view(batch_size, new_tokens, -1).permute(1, 0, 2)
                pos = pos_ds.view(batch_size, new_tokens, 3)
        return pos, output


class TransformerDecoder(nn.Module):
    r"""Stack of `num_layers` pre-norm decoder layers with a shared final norm applied to every output.

    Mirrors the reference 3DETR decoder with `return_intermediate=True`: each layer's output is normed and
    collected so the box heads can be applied per layer (only the last is kept at eval).

    Args:
        embed_dim: Token embedding dimension.
        num_heads: Number of attention heads.
        mlp_dim: Hidden width of the feed-forward block.
        num_layers: Number of stacked decoder layers.
        dropout: Dropout probability.
        act: Activation type or callable.
        act_kwargs: Extra activation arguments.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        mlp_dim: int,
        num_layers: int,
        dropout: float,
        *,
        act: Union[str, Callable, None] = "relu",
        act_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            TransformerDecoderLayer(embed_dim, num_heads, mlp_dim, dropout, act=act, act_kwargs=act_kwargs)
            for _ in range(num_layers)
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(
        self,
        tgt: Tensor,
        memory: Tensor,
        pos: Tensor,
        query_pos: Tensor,
    ) -> Tensor:
        output = tgt
        intermediate = []
        for layer in self.layers:
            output = layer(output, memory, pos=pos, query_pos=query_pos)
            intermediate.append(self.norm(output))
        return torch.stack(intermediate)


class PositionEmbeddingFourier(nn.Module):
    r"""Fixed Fourier-feature positional embedding (Tancik et al.) over normalized coordinates.

    Mirrors 3DETR's `PositionEmbeddingCoordsSine(pos_type="fourier")`: coordinates are min-max normalized
    to $[0, 1]$ against the per-scene point-cloud range, scaled by $2\pi$, projected by a fixed Gaussian
    matrix and mapped through sine/cosine.

    Args:
        d_pos: Output embedding dimension (must be even).
        d_in: Input coordinate dimension.
    """

    gauss_b: Tensor

    def __init__(self, d_pos: int, d_in: int = 3) -> None:
        super().__init__()
        if d_pos % 2 != 0:
            raise ValueError(f"`d_pos` must be even, got {d_pos}.")
        self.d_pos = d_pos
        self.register_buffer("gauss_b", torch.empty(d_in, d_pos // 2).normal_())

    @torch.no_grad()
    def forward(self, pos: Tensor, input_range: Tuple[Tensor, Tensor]) -> Tensor:
        bsize, npoints, d_in = pos.shape
        d_out = self.d_pos // 2
        lo, hi = input_range
        diff = (hi - lo).unsqueeze(1)
        coord = (pos - lo.unsqueeze(1)) / diff
        coord = coord * (2 * math.pi)
        proj = torch.mm(coord.reshape(-1, d_in), self.gauss_b[:, :d_out]).view(bsize, npoints, d_out)
        embed = torch.cat([proj.sin(), proj.cos()], dim=2).permute(0, 2, 1)
        return embed


class GenericConvMLP(nn.Module):
    r"""$1\times1$-conv MLP over $(B, C, N)$ tokens, mirroring 3DETR's `GenericMLP` (head / projection block).

    Each hidden layer is a `Conv1d` optionally followed by batch norm, activation and dropout; the output
    layer is a bare `Conv1d` (optionally with norm and activation). The reference uses this both for the
    encoder-to-decoder projection (norm + activation on the output) and for every box head (dropout, no
    output norm).

    Args:
        in_channels: Input channel count.
        hidden_channels: Hidden layer widths.
        out_channels: Output channel count.
        norm: Hidden-layer normalization type or callable (applied to each hidden layer).
        act: Activation type or callable.
        act_kwargs: Extra activation arguments.
        dropout: Dropout probability after each hidden layer (0 disables).
        hidden_bias: Whether hidden convolutions carry a bias.
        out_bias: Whether the output convolution carries a bias.
        out_use_norm: Whether to apply the hidden norm to the output as well.
        out_use_act: Whether to apply the activation to the output as well.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: List[int],
        out_channels: int,
        *,
        norm: Union[str, Callable, None] = None,
        act: Union[str, Callable, None] = "relu",
        act_kwargs: Optional[Dict[str, Any]] = None,
        dropout: float = 0.0,
        hidden_bias: bool = False,
        out_bias: bool = True,
        out_use_norm: bool = False,
        out_use_act: bool = False,
    ) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        prev = in_channels
        for width in hidden_channels:
            layers.append(nn.Conv1d(prev, width, 1, bias=hidden_bias))
            hidden_norm = create_norm(norm, width, dim=1)
            if hidden_norm is not None:
                layers.append(hidden_norm)
            layers.append(create_act(act, **(act_kwargs or {})) or nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(p=dropout))
            prev = width
        layers.append(nn.Conv1d(prev, out_channels, 1, bias=out_bias))
        if out_use_norm:
            out_norm = create_norm(norm, out_channels, dim=1)
            if out_norm is not None:
                layers.append(out_norm)
        if out_use_act:
            layers.append(create_act(act, **(act_kwargs or {})) or nn.ReLU())
        self.layers = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.layers(x)


class DETR3DDetection(DetectionModel):
    r"""3DETR end-to-end transformer 3D object detector (packed point format).

    Reference: :arxiv: [Misra et al., 2021](https://arxiv.org/abs/2109.08141).
    Reference implementation: :github: [facebookresearch/3detr](https://github.com/facebookresearch/3detr).

    A set-abstraction tokenizer downsamples the cloud to a fixed set of point tokens, a transformer encoder
    (vanilla, or masked with one interim downsampling for 3DETR-m) refines them, a fixed number of object
    queries are farthest-point-sampled from the encoder tokens, and a transformer decoder attends them
    against the encoder memory. Five MLP heads decode each query into a class, center, size and heading.
    Box centers and sizes are predicted in a per-scene min-max normalized frame and unnormalized against the
    point-cloud extent.

    Args:
        in_channels: Input feature channels per point excluding xyz ($0$ for xyz-only, $3$ for RGB).
        num_classes: Number of semantic classes (the class head adds one background slot).
        num_angle_bin: Heading-angle bins ($1$ for axis-aligned ScanNet, $12$ for oriented SUN RGB-D).
        num_queries: Number of object queries (decoded boxes) per scene.
        preenc_npoints: Token count after the set-abstraction tokenizer.
        encoder_type: `"vanilla"` (encoder keeps `preenc_npoints` tokens) or `"masked"` (3DETR-m: radius
            attention masks plus one interim downsampling to $\text{preenc\_npoints} // 2$).
        encoder_embed_dim: Encoder token dimension.
        encoder_num_heads: Encoder attention heads.
        encoder_feedforward_channels: Encoder feed-forward width.
        encoder_depth: Encoder layers.
        encoder_dropout: Encoder dropout.
        decoder_embed_dim: Decoder token dimension.
        decoder_num_heads: Decoder attention heads.
        decoder_feedforward_channels: Decoder feed-forward width.
        decoder_depth: Decoder layers.
        decoder_dropout: Decoder dropout.
        mlp_dropout: Dropout inside each box head.
        preenc_radius: Tokenizer ball-query radius.
        preenc_nsample: Tokenizer ball-query neighbor cap.
        act: Activation type or callable.
        act_kwargs: Extra activation arguments.
        norm: Normalization type or callable for the convolutional blocks (tokenizer, projection, heads).
        norm_kwargs: Extra normalization arguments.
    """

    num_angle_bin: int
    num_queries: int
    preenc_npoints: int
    encoder_embed_dim: int
    decoder_embed_dim: int

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        *,
        num_angle_bin: int,
        num_queries: int,
        preenc_npoints: int = 2048,
        encoder_type: str = "vanilla",
        encoder_embed_dim: int = 256,
        encoder_num_heads: int = 4,
        encoder_feedforward_channels: int = 128,
        encoder_depth: int = 3,
        encoder_dropout: float = 0.1,
        decoder_embed_dim: int = 256,
        decoder_num_heads: int = 4,
        decoder_feedforward_channels: int = 256,
        decoder_depth: int = 8,
        decoder_dropout: float = 0.1,
        mlp_dropout: float = 0.3,
        preenc_radius: float = 0.2,
        preenc_nsample: int = 64,
        act: Union[str, Callable, None] = "relu",
        act_kwargs: Optional[Dict[str, Any]] = None,
        norm: Union[str, Callable, None] = "batch_norm",
        norm_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(in_channels=in_channels, num_classes=num_classes)
        if encoder_type not in ("vanilla", "masked"):
            raise ValueError(f"Unknown `encoder_type` {encoder_type!r}, expected 'vanilla' or 'masked'.")

        self.num_angle_bin = num_angle_bin
        self.num_queries = num_queries
        self.preenc_npoints = preenc_npoints
        self.encoder_type = encoder_type
        self.encoder_embed_dim = encoder_embed_dim
        self.encoder_num_heads = encoder_num_heads
        self.encoder_feedforward_channels = encoder_feedforward_channels
        self.encoder_depth = encoder_depth
        self.encoder_dropout = encoder_dropout
        self.decoder_embed_dim = decoder_embed_dim
        self.decoder_num_heads = decoder_num_heads
        self.decoder_feedforward_channels = decoder_feedforward_channels
        self.decoder_depth = decoder_depth
        self.decoder_dropout = decoder_dropout
        self.mlp_dropout = mlp_dropout
        self.preenc_radius = preenc_radius
        self.preenc_nsample = preenc_nsample
        self.act = act
        self.act_kwargs = act_kwargs
        self.norm = norm
        self.norm_kwargs = norm_kwargs

        self.pre_encoder = self.configure_pre_encoder()
        self.encoder = self.configure_encoder()
        self.encoder_to_decoder_projection = self.configure_encoder_to_decoder_projection()
        self.pos_embedding = self.configure_pos_embedding()
        self.query_projection = self.configure_query_projection()
        self.decoder = self.configure_decoder()
        self.mlp_heads = self.configure_mlp_heads()

    def configure_pre_encoder(self) -> PointnetSAModuleVotes:
        """Build the set-abstraction tokenizer."""
        return PointnetSAModuleVotes(
            self.in_channels,
            [64, 128, self.encoder_embed_dim],
            num_points=self.preenc_npoints,
            radius=self.preenc_radius,
            num_neighbors=self.preenc_nsample,
            act=self.act,
            act_kwargs=self.act_kwargs,
            norm=self.norm,
            norm_kwargs=self.norm_kwargs,
        )

    def configure_encoder(self) -> nn.Module:
        """Build the transformer encoder (masked with one interim downsampling for 3DETR-m)."""
        if self.encoder_type == "masked":
            interim = PointnetSAModuleVotes(
                self.encoder_embed_dim,
                [256, 256, self.encoder_embed_dim],
                num_points=self.preenc_npoints // 2,
                radius=0.4,
                num_neighbors=32,
                act=self.act,
                act_kwargs=self.act_kwargs,
                norm=self.norm,
                norm_kwargs=self.norm_kwargs,
            )
            return MaskedTransformerEncoder(
                self.encoder_embed_dim,
                self.encoder_num_heads,
                self.encoder_feedforward_channels,
                num_layers=3,
                dropout=self.encoder_dropout,
                masking_radius=[0.16, 0.64, 1.44],
                interim_downsampling=interim,
                act=self.act,
                act_kwargs=self.act_kwargs,
            )
        return TransformerEncoder(
            self.encoder_embed_dim,
            self.encoder_num_heads,
            self.encoder_feedforward_channels,
            num_layers=self.encoder_depth,
            dropout=self.encoder_dropout,
            act=self.act,
            act_kwargs=self.act_kwargs,
        )

    def configure_encoder_to_decoder_projection(self) -> GenericConvMLP:
        """Build the projection from encoder tokens to the decoder dimension."""
        if self.encoder_type == "masked":
            proj_hidden = [self.encoder_embed_dim]
        else:
            proj_hidden = [self.encoder_embed_dim, self.encoder_embed_dim]
        return GenericConvMLP(
            self.encoder_embed_dim,
            proj_hidden,
            self.decoder_embed_dim,
            norm=self.norm,
            act=self.act,
            act_kwargs=self.act_kwargs,
            hidden_bias=False,
            out_bias=False,
            out_use_norm=True,
            out_use_act=True,
        )

    def configure_pos_embedding(self) -> PositionEmbeddingFourier:
        """Build the Fourier position embedding."""
        return PositionEmbeddingFourier(d_pos=self.decoder_embed_dim, d_in=3)

    def configure_query_projection(self) -> GenericConvMLP:
        """Build the query projection."""
        return GenericConvMLP(
            self.decoder_embed_dim,
            [self.decoder_embed_dim],
            self.decoder_embed_dim,
            norm=None,
            act=self.act,
            act_kwargs=self.act_kwargs,
            hidden_bias=True,
            out_bias=True,
            out_use_act=True,
        )

    def configure_decoder(self) -> TransformerDecoder:
        """Build the transformer decoder."""
        return TransformerDecoder(
            self.decoder_embed_dim,
            self.decoder_num_heads,
            self.decoder_feedforward_channels,
            num_layers=self.decoder_depth,
            dropout=self.decoder_dropout,
            act=self.act,
            act_kwargs=self.act_kwargs,
        )

    def configure_mlp_heads(self) -> nn.ModuleDict:
        """Build the per-query class, center, size and heading heads."""

        def head(out_channels: int) -> GenericConvMLP:
            return GenericConvMLP(
                self.decoder_embed_dim,
                [self.decoder_embed_dim, self.decoder_embed_dim],
                out_channels,
                norm=self.norm,
                act=self.act,
                act_kwargs=self.act_kwargs,
                dropout=self.mlp_dropout,
                hidden_bias=False,
                out_bias=True,
            )

        mlp_heads = nn.ModuleDict()
        mlp_heads["sem_cls_head"] = head(self.num_classes + 1)
        mlp_heads["center_head"] = head(3)
        mlp_heads["size_head"] = head(3)
        mlp_heads["angle_cls_head"] = head(self.num_angle_bin)
        mlp_heads["angle_residual_head"] = head(self.num_angle_bin)
        return mlp_heads

    def reset_classifier(self, num_classes: int) -> None:
        self.num_classes = num_classes
        head = self.mlp_heads["sem_cls_head"]
        assert isinstance(head, GenericConvMLP)
        last = head.layers[-1]
        assert isinstance(last, nn.Conv1d)
        head.layers[-1] = nn.Conv1d(self.decoder_embed_dim, num_classes + 1, 1).to(last.weight.device)

    def _point_cloud_dims(self, pos: Tensor, batch: Tensor) -> Tuple[Tensor, Tensor]:
        if batch.numel() == 0:
            raise ValueError("`DETR3DDetection` requires a non-empty point cloud.")
        counts = batch.bincount()
        if bool((counts != counts[0]).any()):
            raise ValueError(
                f"`DETR3DDetection` requires the same number of points per scene, got counts {counts.tolist()}."
            )
        dense = _to_dense(pos, batch, int(counts[0]))
        return dense.amin(dim=1), dense.amax(dim=1)

    def run_encoder(
        self,
        x: OptTensor,
        pos: Tensor,
        batch: Tensor,
        idx: OptTensor = None,
    ) -> Tuple[Tensor, Tensor]:
        r"""Tokenizes the point cloud with the pre-encoder and runs the masked transformer encoder.

        Args:
            x: Packed point features, shape $(N, C)$, or `None` to use the positions.
            pos: Packed point positions, shape $(N, 3)$.
            batch: Per-point scene index, shape $(N,)$.
            idx: Optional pre-computed sampling indices for the pre-encoder.

        Returns:
            The token positions of shape $(B, P, 3)$ and the token features of shape $(P, B, C)$.
        """
        x_tok, pos_tok, batch_tok, _ = self.pre_encoder(x, pos, batch, idx)
        num_tok = self.preenc_npoints
        enc_xyz = _to_dense(pos_tok, batch_tok, num_tok)
        enc_features = _to_dense(x_tok, batch_tok, num_tok).permute(1, 0, 2)
        enc_xyz, enc_features = self.encoder(enc_features, enc_xyz)
        return enc_xyz, enc_features

    def get_query_embeddings(
        self,
        enc_xyz: Tensor,
        point_cloud_dims: Tuple[Tensor, Tensor],
        query_idx: OptTensor = None,
    ) -> Tuple[Tensor, Tensor]:
        r"""Samples the query positions among the encoder tokens and embeds them into decoder queries.

        Args:
            enc_xyz: Encoder token positions, shape $(B, P, 3)$.
            point_cloud_dims: Per-scene minimum and maximum corners, used to normalize the positional embedding.
            query_idx: Optional pre-computed query indices. Farthest point sampling is used when `None`.

        Returns:
            The query positions of shape $(B, Q, 3)$ and the query embeddings of shape $(B, C, Q)$.
        """
        batch_size, num_enc = enc_xyz.shape[:2]
        if query_idx is None:
            flat = enc_xyz.reshape(batch_size * num_enc, 3)
            token_batch = torch.arange(batch_size, device=enc_xyz.device).repeat_interleave(num_enc)
            sampled = fps(flat, token_batch, num_nodes=self.num_queries, random_start=self.training)
            query_idx = (
                sampled - torch.arange(batch_size, device=enc_xyz.device).repeat_interleave(self.num_queries) * num_enc
            ).view(batch_size, self.num_queries)
        query_xyz = torch.gather(enc_xyz, 1, query_idx.unsqueeze(-1).expand(-1, -1, 3))
        pos_embed = self.pos_embedding(query_xyz, point_cloud_dims)
        query_embed = self.query_projection(pos_embed)
        return query_xyz, query_embed

    def forward_features(
        self,
        x: OptTensor,
        pos: Tensor,
        batch: Tensor,
        idx: OptTensor = None,
    ) -> Tuple[Tensor, Tensor]:
        enc_xyz, enc_features = self.run_encoder(x, pos, batch, idx)
        enc_features = self.encoder_to_decoder_projection(enc_features.permute(1, 2, 0)).permute(2, 0, 1)
        return enc_xyz, enc_features

    def forward_head(
        self,
        enc_xyz: Tensor,
        enc_features: Tensor,
        point_cloud_dims: Tuple[Tensor, Tensor],
        query_idx: OptTensor = None,
    ) -> DETR3DOutput:
        query_xyz, query_embed = self.get_query_embeddings(enc_xyz, point_cloud_dims, query_idx)
        enc_pos = self.pos_embedding(enc_xyz, point_cloud_dims).permute(2, 0, 1)
        query_embed = query_embed.permute(2, 0, 1)
        tgt = torch.zeros_like(query_embed)
        box_features = self.decoder(tgt, enc_features, pos=enc_pos, query_pos=query_embed)
        return self._decode_heads(query_xyz, point_cloud_dims, box_features)

    def _decode_layer(
        self,
        query_xyz: Tensor,
        point_cloud_dims: Tuple[Tensor, Tensor],
        cls_logits: Tensor,
        center_offset: Tensor,
        size_normalized: Tensor,
        angle_logits: Tensor,
        angle_residual_normalized: Tensor,
    ) -> Dict[str, Tensor]:
        r"""Decode one decoder layer's raw head tensors into the full set of box quantities.

        Builds both the normalized frame (center / size in the per-scene min-max box, consumed by the
        set-prediction loss) and the unnormalized frame (metric boxes, consumed by `decode`). The eval
        forward keeps only the `DETR3DOutput` subset of the last layer; the extra normalized fields are
        used by the training loss.
        """
        angle_residual = angle_residual_normalized * (math.pi / angle_residual_normalized.shape[-1])
        lo, hi = point_cloud_dims
        scene_scale = (hi - lo).clamp(min=1e-1)
        center_unnormalized = query_xyz + center_offset
        center_normalized = (center_unnormalized - lo.unsqueeze(1)) / (hi - lo).unsqueeze(1)
        size_unnormalized = size_normalized * scene_scale.unsqueeze(1)
        angle_continuous = self._angle_from_logits(angle_logits, angle_residual)

        cls_prob = cls_logits.softmax(dim=-1)
        objectness_prob = 1 - cls_prob[..., -1]

        return {
            "sem_cls_logits": cls_logits,
            "center_normalized": center_normalized,
            "center_unnormalized": center_unnormalized,
            "size_normalized": size_normalized,
            "size_unnormalized": size_unnormalized,
            "angle_logits": angle_logits,
            "angle_residual": angle_residual,
            "angle_residual_normalized": angle_residual_normalized,
            "angle_continuous": angle_continuous,
            "objectness_prob": objectness_prob,
            "sem_cls_prob": cls_prob[..., :-1],
        }

    def _decode_heads(
        self,
        query_xyz: Tensor,
        point_cloud_dims: Tuple[Tensor, Tensor],
        box_features: Tensor,
    ) -> DETR3DOutput:
        num_layers, num_queries, batch_size = box_features.shape[:3]
        feats = box_features.permute(0, 2, 3, 1).reshape(num_layers * batch_size, self.decoder_embed_dim, num_queries)

        cls_logits = (
            self.mlp_heads["sem_cls_head"](feats).transpose(1, 2).reshape(num_layers, batch_size, num_queries, -1)
        )
        center_offset = (self.mlp_heads["center_head"](feats).sigmoid().transpose(1, 2) - 0.5).reshape(
            num_layers, batch_size, num_queries, -1
        )
        size_normalized = (
            self.mlp_heads["size_head"](feats)
            .sigmoid()
            .transpose(1, 2)
            .reshape(num_layers, batch_size, num_queries, -1)
        )
        angle_logits = (
            self.mlp_heads["angle_cls_head"](feats).transpose(1, 2).reshape(num_layers, batch_size, num_queries, -1)
        )
        angle_residual_normalized = (
            self.mlp_heads["angle_residual_head"](feats)
            .transpose(1, 2)
            .reshape(num_layers, batch_size, num_queries, -1)
        )

        last = self._decode_layer(
            query_xyz,
            point_cloud_dims,
            cls_logits[-1],
            center_offset[-1],
            size_normalized[-1],
            angle_logits[-1],
            angle_residual_normalized[-1],
        )
        output: DETR3DOutput = {
            "sem_cls_logits": last["sem_cls_logits"],
            "center_unnormalized": last["center_unnormalized"],
            "size_unnormalized": last["size_unnormalized"],
            "angle_logits": last["angle_logits"],
            "angle_residual": last["angle_residual"],
            "angle_continuous": last["angle_continuous"],
            "objectness_prob": last["objectness_prob"],
            "sem_cls_prob": last["sem_cls_prob"],
        }
        if not self.training:
            return output

        aux_outputs = [
            self._decode_layer(
                query_xyz,
                point_cloud_dims,
                cls_logits[layer],
                center_offset[layer],
                size_normalized[layer],
                angle_logits[layer],
                angle_residual_normalized[layer],
            )
            for layer in range(num_layers)
        ]
        train_output: DETR3DTrainOutput = {
            "sem_cls_logits": output["sem_cls_logits"],
            "center_unnormalized": output["center_unnormalized"],
            "size_unnormalized": output["size_unnormalized"],
            "angle_logits": output["angle_logits"],
            "angle_residual": output["angle_residual"],
            "angle_continuous": output["angle_continuous"],
            "objectness_prob": output["objectness_prob"],
            "sem_cls_prob": output["sem_cls_prob"],
            "aux_outputs": aux_outputs,
            "point_cloud_dims": point_cloud_dims,
        }
        return train_output

    def _angle_from_logits(self, angle_logits: Tensor, angle_residual: Tensor) -> Tensor:
        if self.num_angle_bin == 1:
            return (angle_logits * 0 + angle_residual * 0).squeeze(-1).clamp(min=0)
        angle_per_cls = 2 * math.pi / self.num_angle_bin
        pred_cls = angle_logits.argmax(dim=-1).detach()
        angle = angle_per_cls * pred_cls + angle_residual.gather(2, pred_cls.unsqueeze(-1)).squeeze(-1)
        angle[angle > math.pi] = angle[angle > math.pi] - 2 * math.pi
        return angle

    def forward(self, x: OptTensor, pos: Tensor, batch: Tensor) -> DETR3DOutput:
        point_cloud_dims = self._point_cloud_dims(pos, batch)
        enc_xyz, enc_features = self.forward_features(x, pos, batch)
        return self.forward_head(enc_xyz, enc_features, point_cloud_dims)

    @torch.no_grad()
    def decode(self, out: DETR3DOutput) -> Detection3D:
        r"""Decode a forward output into raw per-query detections (no NMS, threshold, or filtering).

        Builds one oriented box per query, scores it by objectness, and labels it by the argmax semantic
        class. The angle head predicts negated angles, so the decoded heading is the negated
        `angle_continuous`, matching the dataset / metric $(c_x, c_y, c_z, d_x, d_y, d_z, \theta)$
        counter-clockwise convention. The result is the full unfiltered query set; the evaluation pipeline
        applies point-count filtering, NMS, score thresholding, and the indoor per-class expansion (driven
        by the returned `class_probs`) via the `torch_pointcloud.utils.box3d` utilities, reproducing 3DETR's
        `APCalculator` test protocol (`exact_eval=True`).

        Args:
            out: A `DETR3DOutput` from `forward`.

        Returns:
            Packed queries `{"boxes", "scores", "labels", "batch", "class_probs"}` (PyG layout), where the
            per-query score is objectness, the label is the argmax semantic class, and `class_probs` holds
            the semantic-class probabilities.

        Shape:
            - boxes: $(B \cdot Q, 7)$
            - scores / labels / batch: $(B \cdot Q,)$
            - class_probs: $(B \cdot Q, C)$
        """
        batch_size, num_queries = out["center_unnormalized"].shape[:2]
        boxes = torch.cat(
            [out["center_unnormalized"], out["size_unnormalized"], -out["angle_continuous"].unsqueeze(-1)], dim=-1
        )
        objectness = out["objectness_prob"]
        class_probs = out["sem_cls_prob"]
        batch = torch.arange(batch_size, device=boxes.device).repeat_interleave(num_queries)
        return {
            "boxes": boxes.reshape(-1, 7),
            "scores": objectness.reshape(-1),
            "labels": class_probs.argmax(dim=-1).reshape(-1),
            "batch": batch,
            "class_probs": class_probs.reshape(-1, class_probs.size(-1)),
        }


_SCANNET_TRANSFORM = T.Compose(
    [
        T.CopyItems(
            keys=[DataKeys.POS, DataKeys.SEGMENT],
            names=[DataKeys.ORIGIN_POS, DataKeys.ORIGIN_SEGMENT],
            allow_missing_keys=True,
        ),
        T.RandomSample(
            keys=[DataKeys.POS, DataKeys.COLOR, DataKeys.NORMAL, DataKeys.SEGMENT, DataKeys.INSTANCE],
            num_samples=40000,
            allow_missing_keys=True,
            dst_index_key=DataKeys.INDEX,
        ),
    ]
)

_SUNRGBD_TRANSFORM = T.Compose(
    [
        T.CopyItems(keys=DataKeys.POS, names=DataKeys.ORIGIN_POS),
        T.RandomSample(
            keys=[DataKeys.POS, DataKeys.COLOR],
            num_samples=20000,
            allow_missing_keys=True,
            dst_index_key=DataKeys.INDEX,
        ),
    ]
)


@register_model(
    "3detr-m.scannet.fair",
    task="detection",
    weights=WeightsDict(
        url="hf://torch-pointcloud/3detr/3detr-m.scannet.fair.safetensors",
        dataset="scannet",
        classes=SCANNET_DETECTION_CLASSES,
        author="fair",
        license="Apache-2.0",
    ),
    transform=_SCANNET_TRANSFORM,
    hparams=dict(
        in_channels=0,
        num_classes=18,
        num_angle_bin=1,
        num_queries=256,
        encoder_type="masked",
        encoder_dropout=0.3,
    ),
)
def detr3d_m_scannet(**hparams: Any) -> DETR3DDetection:
    return DETR3DDetection(**hparams)


@register_model(
    "3detr.scannet.fair",
    task="detection",
    weights=WeightsDict(
        url="hf://torch-pointcloud/3detr/3detr.scannet.fair.safetensors",
        dataset="scannet",
        classes=SCANNET_DETECTION_CLASSES,
        author="fair",
        license="Apache-2.0",
    ),
    transform=_SCANNET_TRANSFORM,
    hparams=dict(
        in_channels=0,
        num_classes=18,
        num_angle_bin=1,
        num_queries=256,
        encoder_type="vanilla",
    ),
)
def detr3d_scannet(**hparams: Any) -> DETR3DDetection:
    return DETR3DDetection(**hparams)


@register_model(
    "3detr.sunrgbd.fair",
    task="detection",
    weights=WeightsDict(
        url="hf://torch-pointcloud/3detr/3detr.sunrgbd.fair.safetensors",
        dataset="sunrgbd",
        classes=SUNRGBD_CLASSES,
        author="fair",
        license="Apache-2.0",
    ),
    transform=_SUNRGBD_TRANSFORM,
    hparams=dict(
        in_channels=0,
        num_classes=10,
        num_angle_bin=12,
        num_queries=128,
        encoder_type="vanilla",
    ),
)
def detr3d_sunrgbd(**hparams: Any) -> DETR3DDetection:
    return DETR3DDetection(**hparams)
