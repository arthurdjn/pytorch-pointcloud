"""GeoTransformer point cloud registration model.

{{ paper("2202.06688") }}
"""

import math
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple, TypedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn import MLP
from typing_extensions import NotRequired

import torch_pointcloud.transforms as T
from torch_pointcloud.layers.kpconv_blocks import KPConvGroupNormBlock, KPResidualGroupNormBlock, LinearGroupNormBlock
from torch_pointcloud.layers.voxel_grid_pool import VoxelGridPool
from torch_pointcloud.ops.cluster import knn
from torch_pointcloud.ops.geometry import transform_points
from torch_pointcloud.ops.registration import (
    node_correspondences,
    pairwise_squared_distance,
    point_to_node_partition,
    weighted_procrustes,
)
from torch_pointcloud.utils.data import DataKeys
from torch_pointcloud.utils.types import FeaturesDict, OptTensor, Registration3D

from ._base import RegistrationModel
from ._registry import WeightsDict, register_model

BlockType = Literal["self", "cross"]


class GeoTransformerOutput(TypedDict):
    r"""Raw output of `GeoTransformerRegistration.forward`, packed over the $B$ pairs of the batch.

    The superpoints (coarse points) of every pair are indexed locally, within that pair's source or reference
    cloud, and `node_corr_batch` says which pair a superpoint correspondence and its patches belong to.

    Attributes:
        src_pos_c: Source superpoints, shape $(M_s, 3)$.
        src_batch_c: Pair index of every source superpoint, shape $(M_s,)$.
        src_feats_c: Unit-norm source superpoint features after the geometric transformer, shape $(M_s, C)$.
        ref_pos_c: Reference superpoints, shape $(M_r, 3)$.
        ref_batch_c: Pair index of every reference superpoint, shape $(M_r,)$.
        ref_feats_c: Unit-norm reference superpoint features, shape $(M_r, C)$.
        ref_node_corr_indices: Reference superpoint of every superpoint correspondence, shape $(P,)$.
        src_node_corr_indices: Source superpoint of every superpoint correspondence, shape $(P,)$.
        node_corr_scores: Matching score of every superpoint correspondence, shape $(P,)$.
        node_corr_batch: Pair index of every superpoint correspondence, shape $(P,)$.
        ref_patch_points: Points of the reference patch of every correspondence, shape $(P, K, 3)$.
        src_patch_points: Points of the source patch of every correspondence, shape $(P, K, 3)$.
        ref_patch_masks: Validity of the reference patch slots, shape $(P, K)$.
        src_patch_masks: Validity of the source patch slots, shape $(P, K)$.
        matching_scores: Log optimal-transport assignment between the two patches of every correspondence, with
            a dustbin row and column, shape $(P, K + 1, K + 1)$.
        gt_node_corr_indices: Ground-truth `(reference, source)` superpoint correspondences, shape $(G, 2)$; present
            when `forward` received the ground-truth transform.
        gt_node_corr_overlaps: Overlap ratio of every ground-truth superpoint correspondence, shape $(G,)$.
        gt_node_corr_batch: Pair index of every ground-truth superpoint correspondence, shape $(G,)$.
    """

    src_pos_c: Tensor
    src_batch_c: Tensor
    src_feats_c: Tensor
    ref_pos_c: Tensor
    ref_batch_c: Tensor
    ref_feats_c: Tensor
    ref_node_corr_indices: Tensor
    src_node_corr_indices: Tensor
    node_corr_scores: Tensor
    node_corr_batch: Tensor
    ref_patch_points: Tensor
    src_patch_points: Tensor
    ref_patch_masks: Tensor
    src_patch_masks: Tensor
    matching_scores: Tensor
    gt_node_corr_indices: NotRequired[Tensor]
    gt_node_corr_overlaps: NotRequired[Tensor]
    gt_node_corr_batch: NotRequired[Tensor]


def sinusoidal_embedding(indices: Tensor, dim: int) -> Tensor:
    r"""Sinusoidal embedding of continuous indices, sine and cosine interleaved per frequency.

    Args:
        indices: Values to embed, any shape $(\ldots)$.
        dim: Embedding width $D$, even.

    Returns:
        The embeddings, shape $(\ldots, D)$.

    Example:
        ```pycon
        >>> import torch
        >>> from torch_pointcloud.models.geotransformer import sinusoidal_embedding
        >>> sinusoidal_embedding(torch.zeros(2, 3), dim=4).shape
        torch.Size([2, 3, 4])

        ```
    """
    if dim % 2 != 0:
        raise ValueError(f"`dim` must be even, got {dim}.")

    div_term = torch.exp(torch.arange(0, dim, 2, device=indices.device).float() * (-math.log(10000.0) / dim))
    omegas = indices.unsqueeze(-1).unsqueeze(-1) * div_term.view(1, -1, 1)  # (..., D / 2, 1)
    embeddings = torch.cat([torch.sin(omegas), torch.cos(omegas)], dim=-1)  # (..., D / 2, 2)
    return embeddings.view(*indices.shape, dim)


class GeometricStructureEmbedding(nn.Module):
    r"""Geometric structure embedding of a superpoint set: pair-wise distances and triplet-wise angles.

    For every pair of superpoints $(i, j)$ the embedding sums a projection of the sinusoidal embedding of their
    distance $d_{ij} / \sigma_d$ and a projection of the sinusoidal embedding of the angles between
    $\vec{p_i p_j}$ and the vectors from $p_i$ to its $k$ nearest superpoints, scaled by $\sigma_a$ and reduced
    over the $k$ neighbors.

    Args:
        embed_dim: Embedding width $C$.
        sigma_d: Distance temperature, in the unit of the coordinates.
        sigma_a: Angle temperature, in degrees.
        angle_k: Number of nearest superpoints $k$ the angles are measured against.
        reduction_a: Reduction of the angular embedding over the $k$ neighbors, `"max"` or `"mean"`.

    Shape:
        - Input: $(B, N, 3)$ superpoints.
        - Output: $(B, N, N, C)$.
    """

    def __init__(
        self,
        embed_dim: int,
        sigma_d: float,
        sigma_a: float,
        angle_k: int,
        reduction_a: Literal["max", "mean"] = "max",
    ) -> None:
        super().__init__()
        if reduction_a not in ("max", "mean"):
            raise ValueError(f"Unsupported `reduction_a` {reduction_a!r}, expected 'max' or 'mean'.")

        self.embed_dim = embed_dim
        self.sigma_d = sigma_d
        self.sigma_a = sigma_a
        self.factor_a = 180.0 / (sigma_a * math.pi)
        self.angle_k = angle_k
        self.reduction_a = reduction_a
        self.proj_d = nn.Linear(embed_dim, embed_dim)
        self.proj_a = nn.Linear(embed_dim, embed_dim)

    @torch.no_grad()
    def embedding_indices(self, points: Tensor) -> Tuple[Tensor, Tensor]:
        r"""Distance indices $(B, N, N)$ and angular indices $(B, N, N, k)$ of a superpoint set."""
        batch_size, num_points, _ = points.shape
        dist_map = torch.sqrt(pairwise_squared_distance(points, points))  # (B, N, N)
        d_indices = dist_map / self.sigma_d

        k = min(self.angle_k, num_points - 1)
        knn_indices = dist_map.topk(k=k + 1, dim=2, largest=False)[1][:, :, 1:]  # (B, N, k)
        knn_indices = knn_indices.unsqueeze(3).expand(batch_size, num_points, k, 3)
        expanded_points = points.unsqueeze(1).expand(batch_size, num_points, num_points, 3)
        knn_points = torch.gather(expanded_points, dim=2, index=knn_indices)  # (B, N, k, 3)
        ref_vectors = knn_points - points.unsqueeze(2)  # (B, N, k, 3)
        anc_vectors = points.unsqueeze(1) - points.unsqueeze(2)  # (B, N, N, 3)
        ref_vectors = ref_vectors.unsqueeze(2).expand(batch_size, num_points, num_points, k, 3)
        anc_vectors = anc_vectors.unsqueeze(3).expand(batch_size, num_points, num_points, k, 3)
        sin_values = torch.linalg.norm(torch.cross(ref_vectors, anc_vectors, dim=-1), dim=-1)  # (B, N, N, k)
        cos_values = torch.sum(ref_vectors * anc_vectors, dim=-1)
        a_indices = torch.atan2(sin_values, cos_values) * self.factor_a
        return d_indices, a_indices

    def forward(self, points: Tensor) -> Tensor:
        d_indices, a_indices = self.embedding_indices(points)
        d_embeddings = self.proj_d(sinusoidal_embedding(d_indices, self.embed_dim))
        a_embeddings = self.proj_a(sinusoidal_embedding(a_indices, self.embed_dim))
        if self.reduction_a == "max":
            a_embeddings = a_embeddings.max(dim=3)[0]
        else:
            a_embeddings = a_embeddings.mean(dim=3)
        return d_embeddings + a_embeddings


class GeometricAttention(nn.Module):
    r"""Multi-head attention over dense token sets, with an optional relative positional term.

    With `pos_embed`, the logits are $(q_i \cdot k_j + q_i \cdot p_{ij}) / \sqrt{d}$ where $p_{ij}$ is the
    projected pair-wise embedding, the geometric self-attention of GeoTransformer; without it the layer is plain
    scaled dot-product cross-attention.

    Args:
        embed_dim: Token width $C$, a multiple of `num_heads`.
        num_heads: Number of attention heads.
        relative_pos: Whether the layer projects and attends to a pair-wise positional embedding.

    Shape:
        - Input: $(B, N, C)$ queries, $(B, M, C)$ keys and values, an optional $(B, N, M, C)$ positional embedding
          and an optional $(B, M)$ key mask (`True` where a key is ignored).
        - Output: $(B, N, C)$.
    """

    def __init__(self, embed_dim: int, num_heads: int, relative_pos: bool = False) -> None:
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError(f"`embed_dim` ({embed_dim}) must be a multiple of `num_heads` ({num_heads}).")

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.proj_q = nn.Linear(embed_dim, embed_dim)
        self.proj_k = nn.Linear(embed_dim, embed_dim)
        self.proj_v = nn.Linear(embed_dim, embed_dim)
        self.proj_p: Optional[nn.Linear] = nn.Linear(embed_dim, embed_dim) if relative_pos else None

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        pos_embed: OptTensor = None,
        key_masks: OptTensor = None,
    ) -> Tensor:
        batch_size, num_queries, _ = query.shape
        num_keys = key.size(1)
        q = self.proj_q(query).view(batch_size, num_queries, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.proj_k(key).view(batch_size, num_keys, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.proj_v(value).view(batch_size, num_keys, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.einsum("bhnc,bhmc->bhnm", q, k)
        if self.proj_p is not None:
            if pos_embed is None:
                raise ValueError("The relative positional attention needs `pos_embed`.")

            p = self.proj_p(pos_embed).view(batch_size, num_queries, num_keys, self.num_heads, self.head_dim)
            scores = scores + torch.einsum("bhnc,bnmhc->bhnm", q, p)
        scores = scores / self.head_dim**0.5
        if key_masks is not None:
            scores = scores.masked_fill(key_masks.unsqueeze(1).unsqueeze(1), float("-inf"))
        scores = F.softmax(scores, dim=-1)

        out = torch.matmul(scores, v)  # (B, H, N, C / H)
        return out.transpose(1, 2).reshape(batch_size, num_queries, self.embed_dim)


class GeometricTransformerLayer(nn.Module):
    r"""Post-norm transformer layer of GeoTransformer: attention, residual and norm, then a two-layer MLP, residual
    and norm.

    Args:
        embed_dim: Token width $C$.
        num_heads: Number of attention heads.
        relative_pos: Whether the attention takes the pair-wise positional embedding (the self-attention layers).

    Shape:
        - Input: $(B, N, C)$ tokens, $(B, M, C)$ memory tokens, an optional $(B, N, M, C)$ positional embedding and
          an optional $(B, M)$ memory mask.
        - Output: $(B, N, C)$.
    """

    def __init__(self, embed_dim: int, num_heads: int, relative_pos: bool = False) -> None:
        super().__init__()
        self.attention = GeometricAttention(embed_dim, num_heads, relative_pos=relative_pos)
        self.linear = nn.Linear(embed_dim, embed_dim)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.mlp = MLP([embed_dim, embed_dim * 2, embed_dim], act="relu", norm=None, plain_last=True)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(
        self,
        x: Tensor,
        memory: Tensor,
        pos_embed: OptTensor = None,
        memory_masks: OptTensor = None,
    ) -> Tensor:
        hidden = self.attention(x, memory, memory, pos_embed=pos_embed, key_masks=memory_masks)
        x = self.norm1(x + self.linear(hidden))
        return self.norm2(x + self.mlp(x))


class GeometricTransformer(nn.Module):
    r"""Geometric transformer: geometric self-attention and plain cross-attention alternating over the superpoints
    of the two clouds.

    Args:
        in_channels: Width of the superpoint features entering the transformer.
        embed_dim: Token width of the attention layers.
        out_channels: Width of the output superpoint features.
        num_heads: Number of attention heads.
        blocks: Sequence of `"self"` and `"cross"` layers.
        sigma_d: Distance temperature of the geometric structure embedding.
        sigma_a: Angle temperature of the geometric structure embedding, in degrees.
        angle_k: Number of nearest superpoints of the angular embedding.
        reduction_a: Reduction of the angular embedding, `"max"` or `"mean"`.

    Shape:
        - Input: $(B, N, 3)$ reference superpoints, $(B, M, 3)$ source superpoints, $(B, N, C_\text{in})$ and
          $(B, M, C_\text{in})$ features, optional $(B, N)$ and $(B, M)$ masks.
        - Output: $(B, N, C_\text{out})$ and $(B, M, C_\text{out})$ features.
    """

    def __init__(
        self,
        in_channels: int,
        embed_dim: int,
        out_channels: int,
        num_heads: int,
        blocks: Sequence[BlockType],
        sigma_d: float,
        sigma_a: float,
        angle_k: int,
        reduction_a: Literal["max", "mean"] = "max",
    ) -> None:
        super().__init__()
        for block in blocks:
            if block not in ("self", "cross"):
                raise ValueError(f"Unsupported block type {block!r}, expected 'self' or 'cross'.")

        self.blocks: Tuple[BlockType, ...] = tuple(blocks)
        self.embedding = GeometricStructureEmbedding(embed_dim, sigma_d, sigma_a, angle_k, reduction_a=reduction_a)
        self.in_proj = nn.Linear(in_channels, embed_dim)
        self.layers = nn.ModuleList(
            [GeometricTransformerLayer(embed_dim, num_heads, relative_pos=block == "self") for block in self.blocks]
        )
        self.out_proj = nn.Linear(embed_dim, out_channels)

    def forward(
        self,
        ref_points: Tensor,
        src_points: Tensor,
        ref_feats: Tensor,
        src_feats: Tensor,
        ref_masks: OptTensor = None,
        src_masks: OptTensor = None,
    ) -> Tuple[Tensor, Tensor]:
        ref_embeddings = self.embedding(ref_points)
        src_embeddings = self.embedding(src_points)
        ref_feats = self.in_proj(ref_feats)
        src_feats = self.in_proj(src_feats)
        for block, layer in zip(self.blocks, self.layers):
            if block == "self":
                ref_feats = layer(ref_feats, ref_feats, pos_embed=ref_embeddings, memory_masks=ref_masks)
                src_feats = layer(src_feats, src_feats, pos_embed=src_embeddings, memory_masks=src_masks)
            else:
                ref_feats = layer(ref_feats, src_feats, memory_masks=src_masks)
                src_feats = layer(src_feats, ref_feats, memory_masks=ref_masks)
        return self.out_proj(ref_feats), self.out_proj(src_feats)


class LogOptimalTransport(nn.Module):
    r"""Sinkhorn optimal transport in log space with a learnable dustbin score (the SuperGlue layer).

    The score matrix is padded with a dustbin row and column holding the learnable `alpha`, masked rows and columns
    are sent to $-\infty$, and `num_iterations` Sinkhorn normalizations spread a uniform mass over the valid rows and
    columns.

    Args:
        num_iterations: Number of Sinkhorn iterations.
        inf: Value standing for infinity in the masked entries.

    Shape:
        - Input: $(B, M, N)$ scores and optional $(B, M)$ row and $(B, N)$ column masks (`True` where valid).
        - Output: $(B, M + 1, N + 1)$ log assignment.
    """

    def __init__(self, num_iterations: int, inf: float = 1e12) -> None:
        super().__init__()
        self.num_iterations = num_iterations
        self.inf = inf
        self.alpha = nn.Parameter(torch.tensor(1.0))

    def log_sinkhorn_normalization(self, scores: Tensor, log_mu: Tensor, log_nu: Tensor) -> Tensor:
        u, v = torch.zeros_like(log_mu), torch.zeros_like(log_nu)
        for _ in range(self.num_iterations):
            u = log_mu - torch.logsumexp(scores + v.unsqueeze(1), dim=2)
            v = log_nu - torch.logsumexp(scores + u.unsqueeze(2), dim=1)
        return scores + u.unsqueeze(2) + v.unsqueeze(1)

    def forward(self, scores: Tensor, row_masks: OptTensor = None, col_masks: OptTensor = None) -> Tensor:
        batch_size, num_row, num_col = scores.shape
        device = scores.device
        if row_masks is None:
            row_masks = torch.ones(batch_size, num_row, dtype=torch.bool, device=device)
        if col_masks is None:
            col_masks = torch.ones(batch_size, num_col, dtype=torch.bool, device=device)

        padded_row_masks = torch.zeros(batch_size, num_row + 1, dtype=torch.bool, device=device)
        padded_row_masks[:, :num_row] = ~row_masks
        padded_col_masks = torch.zeros(batch_size, num_col + 1, dtype=torch.bool, device=device)
        padded_col_masks[:, :num_col] = ~col_masks
        padded_score_masks = padded_row_masks.unsqueeze(2) | padded_col_masks.unsqueeze(1)

        padded_col = self.alpha.expand(batch_size, num_row, 1)
        padded_row = self.alpha.expand(batch_size, 1, num_col + 1)
        padded_scores = torch.cat([torch.cat([scores, padded_col], dim=-1), padded_row], dim=1)
        padded_scores = padded_scores.masked_fill(padded_score_masks, -self.inf)

        num_valid_row = row_masks.float().sum(1)
        num_valid_col = col_masks.float().sum(1)
        norm = -torch.log(num_valid_row + num_valid_col)  # (B,)

        log_mu = torch.empty(batch_size, num_row + 1, device=device)
        log_mu[:, :num_row] = norm.unsqueeze(1)
        log_mu[:, num_row] = torch.log(num_valid_col) + norm
        log_mu[padded_row_masks] = -self.inf

        log_nu = torch.empty(batch_size, num_col + 1, device=device)
        log_nu[:, :num_col] = norm.unsqueeze(1)
        log_nu[:, num_col] = torch.log(num_valid_row) + norm
        log_nu[padded_col_masks] = -self.inf

        outputs = self.log_sinkhorn_normalization(padded_scores, log_mu, log_nu)
        return outputs - norm.unsqueeze(1).unsqueeze(2)

    def extra_repr(self) -> str:
        return f"num_iterations={self.num_iterations}"


class KPConvFPN(nn.Module):
    r"""KPConv feature pyramid of the registration backbones: a stem stage at the input resolution, grid-subsampled
    encoder stages, and nearest-neighbor upsampling decoders back to the fine stage.

    The first stage is a `KPConvGroupNormBlock` followed by one residual block; every further stage grid-subsamples
    the points, then runs a strided residual block and two residual blocks at the doubled radius. The decoders
    upsample the coarsest features to each finer stage, concatenate that stage's encoder features and mix them with
    a unary block; the last decoder is a plain linear layer and defines the fine stage. Neighborhoods are the
    `max_num_neighbors` nearest points within each stage's radius, so the clouds of a batch never mix.

    Args:
        in_channels: Number of input feature channels.
        stem_channels: Output width of the stem convolution.
        encoder_channels: Output width of each stage.
        decoder_channels: Output width of each decoder, from the second coarsest stage down to the fine stage.
        voxel_sizes: Grid size of each subsampling, one per stage transition.
        radii: Neighborhood radius of each stage.
        kp_sigma: Kernel extent of each stage.
        max_num_neighbors: Neighbor limit of each stage.
        kernel_size: Number of kernel points of every convolution.
        num_groups: Number of channel groups of the normalizations.

    Inputs:
        x: Point features of shape $(N, \text{in\_channels})$.
        pos: Point coordinates of shape $(N, 3)$.
        cloud: Index of the cloud every point belongs to, shape $(N,)$, sorted; the neighborhoods stay within it.
        batch: Index of the batch element every point belongs to, shape $(N,)$; the normalizations are computed
            per element.

    Outputs:
        The fine stage and the coarsest stage as `FeaturesDict` holding `x`, `pos` and the `cloud` index as `batch`.
    """

    def __init__(
        self,
        in_channels: int,
        *,
        stem_channels: int,
        encoder_channels: Sequence[int],
        decoder_channels: Sequence[int],
        voxel_sizes: Sequence[float],
        radii: Sequence[float],
        kp_sigma: Sequence[float],
        max_num_neighbors: Sequence[int],
        kernel_size: int = 15,
        num_groups: int = 32,
    ) -> None:
        super().__init__()
        num_stages = len(encoder_channels)
        if len(voxel_sizes) != num_stages - 1:
            raise ValueError("Expected `voxel_sizes` to have one entry less than `encoder_channels`.")
        if len(radii) != num_stages or len(kp_sigma) != num_stages or len(max_num_neighbors) != num_stages:
            raise ValueError("Expected `radii`, `kp_sigma` and `max_num_neighbors` to match `encoder_channels`.")
        if not 1 <= len(decoder_channels) < num_stages:
            raise ValueError("Expected between one and `len(encoder_channels) - 1` decoders.")

        self.in_channels = in_channels
        self.stem_channels = stem_channels
        self.encoder_channels = tuple(encoder_channels)
        self.decoder_channels = tuple(decoder_channels)
        self.voxel_sizes = tuple(voxel_sizes)
        self.radii = tuple(radii)
        self.kp_sigma = tuple(kp_sigma)
        self.max_num_neighbors = tuple(max_num_neighbors)
        self.kernel_size = kernel_size
        self.num_groups = num_groups

        self.pools = nn.ModuleList([VoxelGridPool(grid_size=size, reduce="max") for size in self.voxel_sizes])
        self.stages = nn.ModuleList()
        stem = KPConvGroupNormBlock(in_channels, stem_channels, kernel_size, radii[0], kp_sigma[0], num_groups)
        first = KPResidualGroupNormBlock(
            stem_channels, encoder_channels[0], kernel_size, radii[0], kp_sigma[0], num_groups
        )
        self.stages.append(nn.ModuleList([stem, first]))
        for i in range(1, num_stages):
            strided = KPResidualGroupNormBlock(
                encoder_channels[i - 1],
                encoder_channels[i - 1],
                kernel_size,
                radii[i - 1],
                kp_sigma[i - 1],
                num_groups,
                strided=True,
                max_num_neighbors=max_num_neighbors[i - 1],
            )
            blocks = [
                KPResidualGroupNormBlock(
                    encoder_channels[i - 1] if j == 0 else encoder_channels[i],
                    encoder_channels[i],
                    kernel_size,
                    radii[i],
                    kp_sigma[i],
                    num_groups,
                )
                for j in range(2)
            ]
            self.stages.append(nn.ModuleList([strided, *blocks]))

        self.decoders = nn.ModuleList()
        in_channels = encoder_channels[-1]
        for i, out_channels in enumerate(self.decoder_channels):
            skip_channels = encoder_channels[num_stages - 2 - i]
            if i < len(self.decoder_channels) - 1:
                self.decoders.append(LinearGroupNormBlock(in_channels + skip_channels, out_channels, num_groups))
            else:
                self.decoders.append(nn.Linear(in_channels + skip_channels, out_channels))
            in_channels = out_channels

    @property
    def fine_stage(self) -> int:
        """Index of the stage the last decoder outputs at."""
        return len(self.encoder_channels) - 1 - len(self.decoder_channels)

    def neighbors(
        self,
        pos: Tensor,
        cloud: Tensor,
        query_pos: Tensor,
        query_cloud: Tensor,
        radius: float,
        k: int,
    ) -> Tensor:
        r"""`[source, target]` edges from the $k$ nearest points of `pos` within `radius` of every query point."""
        row, col = knn(pos, query_pos, k=k, batch_x=cloud, batch_y=query_cloud)
        within = torch.linalg.norm(query_pos[row] - pos[col], dim=-1) < radius
        return torch.stack([col[within], row[within]], dim=0)

    def forward(self, x: Tensor, pos: Tensor, cloud: Tensor, batch: Tensor) -> Tuple[FeaturesDict, FeaturesDict]:
        num_stages = len(self.stages)
        feats: List[Tensor] = []
        positions: List[Tensor] = [pos]
        clouds: List[Tensor] = [cloud]
        batches: List[Tensor] = [batch]

        for i, blocks in enumerate(self.stages):
            assert isinstance(blocks, nn.ModuleList)
            if i > 0:
                _, pos, cloud, inverse = self.pools[i - 1](pos, pos, cloud, return_inverse=True)
                batch = torch.zeros_like(cloud).scatter_(0, inverse, batch)
                positions.append(pos)
                clouds.append(cloud)
                batches.append(batch)
                edge_index = self.neighbors(
                    positions[i - 1], clouds[i - 1], pos, cloud, self.radii[i - 1], self.max_num_neighbors[i - 1]
                )
                x = blocks[0](x, (positions[i - 1], pos), edge_index, (batches[i - 1], batch))
                rest = list(blocks)[1:]
            else:
                rest = list(blocks)

            edge_index = self.neighbors(pos, cloud, pos, cloud, self.radii[i], self.max_num_neighbors[i])
            for block in rest:
                x = block(x, pos, edge_index, batch)
            feats.append(x)

        coarse: FeaturesDict = {"x": x, "pos": positions[-1], "batch": clouds[-1]}
        latent = x
        for i, decoder in enumerate(self.decoders):
            stage = num_stages - 2 - i
            row, col = knn(
                positions[stage + 1], positions[stage], k=1, batch_x=clouds[stage + 1], batch_y=clouds[stage]
            )
            within = (
                torch.linalg.norm(positions[stage][row] - positions[stage + 1][col], dim=-1) < self.radii[stage + 1]
            )
            nearest = torch.full((positions[stage].size(0),), latent.size(0), dtype=torch.long, device=pos.device)
            nearest[row[within]] = col[within]
            upsampled = torch.cat([latent, latent.new_zeros(1, latent.size(1))], dim=0)[nearest]
            latent = torch.cat([upsampled, feats[stage]], dim=1)
            if isinstance(decoder, LinearGroupNormBlock):
                latent = decoder(latent, batches[stage])
            else:
                latent = decoder(latent)

        fine: FeaturesDict = {"x": latent, "pos": positions[self.fine_stage], "batch": clouds[self.fine_stage]}
        return fine, coarse


class GeoTransformerRegistration(RegistrationModel):
    r"""GeoTransformer for pairwise rigid registration, as described in the paper
    :arxiv: [Geometric Transformer for Fast and Robust Point Cloud Registration](https://arxiv.org/abs/2202.06688)
    by Zheng Qin, Hao Yu, Changjian Wang, Yulan Guo, Yuxing Peng, Kai Xu.

    A `KPConvFPN` backbone encodes both clouds; its coarsest points are the superpoints, whose features go through
    the `GeometricTransformer` (geometric self-attention within a cloud, cross-attention between the two) and are
    matched with a dual-normalized score to pick `num_correspondences` superpoint pairs. Every superpoint carries a
    patch of its `num_points_in_patch` nearest fine points; the fine features of the two patches of a superpoint
    pair are matched by Sinkhorn optimal transport, and `decode` turns the resulting point correspondences into
    the rigid transform with the local-to-global registration: one weighted Procrustes fit per patch pair, the
    one with the most inliers selected, then refined on the inliers of all correspondences. No RANSAC is involved.

    Note:
        The implementation is based on the original paper and the authors' code
        :github: [GeoTransformer](https://github.com/qinzheng93/GeoTransformer).

    Args:
        in_channels: Number of input feature channels; the registered checkpoints read a constant feature.
        stem_channels: Output width of the backbone stem.
        encoder_channels: Output width of each backbone stage.
        decoder_channels: Output width of each backbone decoder; the last one is the fine feature width.
        voxel_sizes: Grid size of each backbone subsampling.
        radii: Neighborhood radius of each backbone stage.
        kp_sigma: Kernel extent of each backbone stage.
        max_num_neighbors: Neighbor limit of each backbone stage.
        kernel_size: Number of kernel points of every convolution.
        num_groups: Number of channel groups of the backbone normalizations.
        embed_dim: Token width of the geometric transformer.
        coarse_channels: Width of the superpoint features the transformer outputs.
        num_heads: Number of attention heads.
        blocks: Sequence of `"self"` and `"cross"` transformer layers.
        sigma_d: Distance temperature of the geometric structure embedding.
        sigma_a: Angle temperature of the geometric structure embedding, in degrees.
        angle_k: Number of nearest superpoints of the angular embedding.
        reduction_a: Reduction of the angular embedding, `"max"` or `"mean"`.
        num_points_in_patch: Number of fine points $K$ in the patch of a superpoint.
        matching_radius: Distance under which two fine points match when building the ground-truth superpoint
            correspondences from the ground-truth transform.
        num_correspondences: Number of superpoint correspondences kept at inference.
        dual_normalization: Whether the superpoint matching scores are normalized over both rows and columns.
        num_targets: Number of ground-truth superpoint correspondences sampled at training.
        overlap_threshold: Minimum overlap of a ground-truth superpoint correspondence sampled at training.
        num_sinkhorn_iterations: Number of Sinkhorn iterations of the optimal transport.
        topk: Number of candidates kept per row and per column of a patch assignment in `decode`.
        acceptance_radius: Inlier distance of the local-to-global registration.
        mutual: Whether a point correspondence must be among the top candidates of both its row and its column.
        confidence_threshold: Minimum assignment probability of a point correspondence.
        correspondence_threshold: Minimum number of point correspondences for a patch pair to propose a transform.
        correspondence_limit: Maximum number of point correspondences scored against each proposal; `None` keeps
            them all.
        num_refinement_steps: Number of global Procrustes refinements of the selected transform.
    """

    def __init__(
        self,
        in_channels: int = 1,
        *,
        stem_channels: int = 64,
        encoder_channels: Sequence[int] = (128, 256, 512, 1024),
        decoder_channels: Sequence[int] = (512, 256),
        voxel_sizes: Sequence[float] = (0.05, 0.1, 0.2),
        radii: Sequence[float] = (0.0625, 0.125, 0.25, 0.5),
        kp_sigma: Sequence[float] = (0.05, 0.1, 0.2, 0.4),
        max_num_neighbors: Sequence[int] = (38, 36, 36, 38),
        kernel_size: int = 15,
        num_groups: int = 32,
        embed_dim: int = 256,
        coarse_channels: int = 256,
        num_heads: int = 4,
        blocks: Sequence[BlockType] = ("self", "cross", "self", "cross", "self", "cross"),
        sigma_d: float = 0.2,
        sigma_a: float = 15.0,
        angle_k: int = 3,
        reduction_a: Literal["max", "mean"] = "max",
        num_points_in_patch: int = 64,
        matching_radius: float = 0.05,
        num_correspondences: int = 256,
        dual_normalization: bool = True,
        num_targets: int = 128,
        overlap_threshold: float = 0.1,
        num_sinkhorn_iterations: int = 100,
        topk: int = 3,
        acceptance_radius: float = 0.1,
        mutual: bool = True,
        confidence_threshold: float = 0.05,
        correspondence_threshold: int = 3,
        correspondence_limit: Optional[int] = None,
        num_refinement_steps: int = 5,
    ) -> None:
        super().__init__(in_channels)
        self.num_points_in_patch = num_points_in_patch
        self.matching_radius = matching_radius
        self.num_correspondences = num_correspondences
        self.dual_normalization = dual_normalization
        self.num_targets = num_targets
        self.overlap_threshold = overlap_threshold
        self.topk = topk
        self.acceptance_radius = acceptance_radius
        self.mutual = mutual
        self.confidence_threshold = confidence_threshold
        self.correspondence_threshold = correspondence_threshold
        self.correspondence_limit = correspondence_limit
        self.num_refinement_steps = num_refinement_steps

        self.backbone = KPConvFPN(
            in_channels,
            stem_channels=stem_channels,
            encoder_channels=encoder_channels,
            decoder_channels=decoder_channels,
            voxel_sizes=voxel_sizes,
            radii=radii,
            kp_sigma=kp_sigma,
            max_num_neighbors=max_num_neighbors,
            kernel_size=kernel_size,
            num_groups=num_groups,
        )
        self.transformer = GeometricTransformer(
            self.backbone.encoder_channels[-1],
            embed_dim,
            coarse_channels,
            num_heads,
            blocks,
            sigma_d,
            sigma_a,
            angle_k,
            reduction_a=reduction_a,
        )
        self.optimal_transport = LogOptimalTransport(num_sinkhorn_iterations)

    @property
    def num_features(self) -> int:
        """Width $C$ of the fine point features of `forward_features`."""
        return self.backbone.decoder_channels[-1]

    def forward_features(
        self,
        x: OptTensor,
        pos: Tensor,
        batch: Tensor,
        ref_x: OptTensor,
        ref_pos: Tensor,
        ref_batch: Tensor,
    ) -> Tuple[FeaturesDict, FeaturesDict]:
        r"""Run the backbone on the source and reference clouds of the batch.

        Args:
            x: Source features of shape $(N, C)$; `None` uses a constant feature of `in_channels` ones.
            pos: Source coordinates of shape $(N, 3)$.
            batch: Pair index of every source point, shape $(N,)$.
            ref_x: Reference features of shape $(M, C)$, or `None`.
            ref_pos: Reference coordinates of shape $(M, 3)$.
            ref_batch: Pair index of every reference point, shape $(M,)$.

        Returns:
            The fine and the coarse `FeaturesDict` of the $2B$ clouds packed together, the sources first
            ($0 \ldots B-1$) then the references ($B \ldots 2B-1$), each holding `x`, `pos` and that cloud index as
            `batch`.
        """
        batch_size = int(batch.max()) + 1 if batch.numel() else 0
        if x is None:
            x = pos.new_ones(pos.size(0), self.in_channels)
        if ref_x is None:
            ref_x = ref_pos.new_ones(ref_pos.size(0), self.in_channels)

        cloud = torch.cat([batch, ref_batch + batch_size])
        pair = torch.cat([batch, ref_batch])
        return self.backbone(torch.cat([x, ref_x]), torch.cat([pos, ref_pos]), cloud, pair)

    def superpoint_matching(
        self,
        ref_feats: Tensor,
        src_feats: Tensor,
        ref_masks: Tensor,
        src_masks: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        r"""Top `num_correspondences` superpoint pairs by dual-normalized feature similarity, ignoring empty patches."""
        ref_indices = torch.nonzero(ref_masks, as_tuple=True)[0]
        src_indices = torch.nonzero(src_masks, as_tuple=True)[0]
        scores = torch.exp(-pairwise_squared_distance(ref_feats[ref_indices], src_feats[src_indices], normalized=True))
        if self.dual_normalization:
            ref_scores = scores / scores.sum(dim=1, keepdim=True)
            src_scores = scores / scores.sum(dim=0, keepdim=True)
            scores = ref_scores * src_scores
        num_correspondences = min(self.num_correspondences, scores.numel())
        corr_scores, corr_indices = scores.view(-1).topk(k=num_correspondences, largest=True)
        ref_corr_indices = ref_indices[corr_indices // scores.size(1)]
        src_corr_indices = src_indices[corr_indices % scores.size(1)]
        return ref_corr_indices, src_corr_indices, corr_scores

    def forward_head(
        self,
        fine: FeaturesDict,
        coarse: FeaturesDict,
        transform: OptTensor = None,
    ) -> GeoTransformerOutput:
        r"""Match the superpoints of every pair, then the fine points of the matched patches.

        Args:
            fine: Fine-stage features of the $2B$ packed clouds, from `forward_features`.
            coarse: Coarse-stage features of the $2B$ packed clouds, from `forward_features`.
            transform: Ground-truth transform of every pair, shape $(B, 4, 4)$; required at training, where the
                patches to match are sampled among the ground-truth superpoint correspondences, and optional at
                evaluation, where it only adds the ground-truth correspondences to the output.

        Returns:
            The `GeoTransformerOutput` of the batch.
        """
        if self.training and transform is None:
            raise ValueError("Training needs the ground-truth `transform` of every pair to sample the patch pairs.")

        cloud_f, cloud_c = fine["batch"], coarse["batch"]
        batch_size = (int(cloud_c.max()) + 1) // 2
        collected: Dict[str, List[Tensor]] = {}

        for b in range(batch_size):
            src_f, ref_f = cloud_f == b, cloud_f == b + batch_size
            src_c, ref_c = cloud_c == b, cloud_c == b + batch_size
            ref_points_c, src_points_c = coarse["pos"][ref_c], coarse["pos"][src_c]
            ref_points_f, src_points_f = fine["pos"][ref_f], fine["pos"][src_f]
            ref_feats_f, src_feats_f = fine["x"][ref_f], fine["x"][src_f]

            _, ref_node_masks, ref_node_knn_indices, ref_node_knn_masks = point_to_node_partition(
                ref_points_f, ref_points_c, self.num_points_in_patch
            )
            _, src_node_masks, src_node_knn_indices, src_node_knn_masks = point_to_node_partition(
                src_points_f, src_points_c, self.num_points_in_patch
            )
            ref_padded_points_f = torch.cat([ref_points_f, torch.zeros_like(ref_points_f[:1])], dim=0)
            src_padded_points_f = torch.cat([src_points_f, torch.zeros_like(src_points_f[:1])], dim=0)
            ref_node_knn_points = ref_padded_points_f[ref_node_knn_indices]  # (M_r, K, 3)
            src_node_knn_points = src_padded_points_f[src_node_knn_indices]  # (M_s, K, 3)

            if transform is not None:
                gt_node_corr_indices, gt_node_corr_overlaps = node_correspondences(
                    ref_points_c,
                    src_points_c,
                    ref_node_knn_points,
                    src_node_knn_points,
                    transform[b],
                    self.matching_radius,
                    ref_masks=ref_node_masks,
                    src_masks=src_node_masks,
                    ref_knn_masks=ref_node_knn_masks,
                    src_knn_masks=src_node_knn_masks,
                )
                collected.setdefault("gt_node_corr_indices", []).append(gt_node_corr_indices)
                collected.setdefault("gt_node_corr_overlaps", []).append(gt_node_corr_overlaps)
                collected.setdefault("gt_node_corr_batch", []).append(
                    torch.full_like(gt_node_corr_overlaps, b, dtype=torch.long)
                )

            ref_feats_c, src_feats_c = self.transformer(
                ref_points_c.unsqueeze(0),
                src_points_c.unsqueeze(0),
                coarse["x"][ref_c].unsqueeze(0),
                coarse["x"][src_c].unsqueeze(0),
            )
            ref_feats_c = F.normalize(ref_feats_c.squeeze(0), p=2, dim=1)
            src_feats_c = F.normalize(src_feats_c.squeeze(0), p=2, dim=1)

            with torch.no_grad():
                ref_node_corr_indices, src_node_corr_indices, node_corr_scores = self.superpoint_matching(
                    ref_feats_c, src_feats_c, ref_node_masks, src_node_masks
                )
                if self.training:
                    assert transform is not None
                    keep = gt_node_corr_overlaps > self.overlap_threshold
                    targets, overlaps = gt_node_corr_indices[keep], gt_node_corr_overlaps[keep]
                    if targets.size(0) > self.num_targets:
                        sel = torch.randperm(targets.size(0), device=targets.device)[: self.num_targets]
                        targets, overlaps = targets[sel], overlaps[sel]
                    ref_node_corr_indices, src_node_corr_indices, node_corr_scores = (
                        targets[:, 0],
                        targets[:, 1],
                        overlaps,
                    )

            ref_node_corr_knn_indices = ref_node_knn_indices[ref_node_corr_indices]  # (P, K)
            src_node_corr_knn_indices = src_node_knn_indices[src_node_corr_indices]  # (P, K)
            ref_node_corr_knn_masks = ref_node_knn_masks[ref_node_corr_indices]  # (P, K)
            src_node_corr_knn_masks = src_node_knn_masks[src_node_corr_indices]  # (P, K)
            ref_padded_feats_f = torch.cat([ref_feats_f, torch.zeros_like(ref_feats_f[:1])], dim=0)
            src_padded_feats_f = torch.cat([src_feats_f, torch.zeros_like(src_feats_f[:1])], dim=0)
            ref_node_corr_knn_feats = ref_padded_feats_f[ref_node_corr_knn_indices]  # (P, K, C)
            src_node_corr_knn_feats = src_padded_feats_f[src_node_corr_knn_indices]  # (P, K, C)

            matching_scores = torch.einsum("bnd,bmd->bnm", ref_node_corr_knn_feats, src_node_corr_knn_feats)
            matching_scores = matching_scores / ref_feats_f.size(1) ** 0.5
            matching_scores = self.optimal_transport(matching_scores, ref_node_corr_knn_masks, src_node_corr_knn_masks)

            collected.setdefault("src_pos_c", []).append(src_points_c)
            collected.setdefault("src_batch_c", []).append(torch.full_like(src_points_c[:, 0], b, dtype=torch.long))
            collected.setdefault("src_feats_c", []).append(src_feats_c)
            collected.setdefault("ref_pos_c", []).append(ref_points_c)
            collected.setdefault("ref_batch_c", []).append(torch.full_like(ref_points_c[:, 0], b, dtype=torch.long))
            collected.setdefault("ref_feats_c", []).append(ref_feats_c)
            collected.setdefault("ref_node_corr_indices", []).append(ref_node_corr_indices)
            collected.setdefault("src_node_corr_indices", []).append(src_node_corr_indices)
            collected.setdefault("node_corr_scores", []).append(node_corr_scores)
            collected.setdefault("node_corr_batch", []).append(torch.full_like(node_corr_scores, b, dtype=torch.long))
            collected.setdefault("ref_patch_points", []).append(ref_node_knn_points[ref_node_corr_indices])
            collected.setdefault("src_patch_points", []).append(src_node_knn_points[src_node_corr_indices])
            collected.setdefault("ref_patch_masks", []).append(ref_node_corr_knn_masks)
            collected.setdefault("src_patch_masks", []).append(src_node_corr_knn_masks)
            collected.setdefault("matching_scores", []).append(matching_scores)

        output: Dict[str, Tensor] = {key: torch.cat(values, dim=0) for key, values in collected.items()}
        return output  # type: ignore[return-value]

    def forward(
        self,
        x: OptTensor,
        pos: Tensor,
        batch: Tensor,
        ref_x: OptTensor,
        ref_pos: Tensor,
        ref_batch: Tensor,
        transform: OptTensor = None,
    ) -> GeoTransformerOutput:
        r"""Encode and match a batch of source / reference pairs.

        Args:
            x: Source features of shape $(N, C)$; `None` uses a constant feature of `in_channels` ones.
            pos: Source coordinates of shape $(N, 3)$.
            batch: Pair index of every source point, shape $(N,)$.
            ref_x: Reference features of shape $(M, C)$, or `None`.
            ref_pos: Reference coordinates of shape $(M, 3)$.
            ref_batch: Pair index of every reference point, shape $(M,)$.
            transform: Ground-truth transform of every pair, shape $(B, 4, 4)$; required at training.

        Returns:
            The `GeoTransformerOutput` of the batch, decoded into transforms by `decode`.
        """
        fine, coarse = self.forward_features(x, pos, batch, ref_x, ref_pos, ref_batch)
        return self.forward_head(fine, coarse, transform=transform)

    def correspondence_matrix(self, score_mat: Tensor, ref_knn_masks: Tensor, src_knn_masks: Tensor) -> Tensor:
        r"""Point correspondences of every patch pair: the top-`topk` entries of each row and column above the
        confidence threshold, intersected when `mutual`, restricted to the valid patch slots."""
        mask_mat = ref_knn_masks.unsqueeze(2) & src_knn_masks.unsqueeze(1)
        batch_size, ref_length, src_length = score_mat.shape
        device = score_mat.device
        k = min(self.topk, ref_length, src_length)

        ref_topk_scores, ref_topk_indices = score_mat.topk(k=k, dim=2)  # (B, N, k)
        ref_score_mat = torch.zeros_like(score_mat).scatter_(2, ref_topk_indices, ref_topk_scores)
        ref_corr_mat = ref_score_mat > self.confidence_threshold

        src_topk_scores, src_topk_indices = score_mat.topk(k=k, dim=1)  # (B, k, M)
        src_score_mat = torch.zeros_like(score_mat).scatter_(1, src_topk_indices, src_topk_scores)
        src_corr_mat = src_score_mat > self.confidence_threshold

        corr_mat = (ref_corr_mat & src_corr_mat) if self.mutual else (ref_corr_mat | src_corr_mat)
        return corr_mat & mask_mat.to(device)

    def recompute_correspondence_scores(
        self,
        ref_corr_points: Tensor,
        src_corr_points: Tensor,
        corr_scores: Tensor,
        transform: Tensor,
    ) -> Tensor:
        aligned = transform_points(src_corr_points, transform)
        inliers = torch.linalg.norm(ref_corr_points - aligned, dim=1) < self.acceptance_radius
        return corr_scores * inliers.float()

    def local_to_global_registration(
        self,
        ref_knn_points: Tensor,
        src_knn_points: Tensor,
        score_mat: Tensor,
        corr_mat: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        r"""Local-to-global registration of one pair from the point correspondences of its patch pairs.

        Every patch pair with at least `correspondence_threshold` correspondences proposes a weighted Procrustes
        transform; the proposal with the most inliers over all correspondences seeds `num_refinement_steps`
        weighted Procrustes refinements on the inliers.

        Returns:
            The reference and source points of all correspondences $(C, 3)$, their scores $(C,)$ and the
            estimated transform $(4, 4)$.
        """
        batch_indices, ref_indices, src_indices = torch.nonzero(corr_mat, as_tuple=True)
        global_ref_corr_points = ref_knn_points[batch_indices, ref_indices]
        global_src_corr_points = src_knn_points[batch_indices, src_indices]
        global_corr_scores = score_mat[batch_indices, ref_indices, src_indices]

        if self.correspondence_limit is not None and global_corr_scores.size(0) > self.correspondence_limit:
            corr_scores, sel_indices = global_corr_scores.topk(k=self.correspondence_limit, largest=True)
            ref_corr_points = global_ref_corr_points[sel_indices]
            src_corr_points = global_src_corr_points[sel_indices]
        else:
            ref_corr_points, src_corr_points, corr_scores = (
                global_ref_corr_points,
                global_src_corr_points,
                global_corr_scores,
            )

        # `nonzero` is row-major, so the correspondences of one patch pair are consecutive.
        counts = torch.bincount(batch_indices, minlength=corr_mat.size(0))
        proposals = torch.nonzero(counts >= self.correspondence_threshold, as_tuple=True)[0]
        if proposals.numel() > 0:
            max_corr = int(counts[proposals].max())
            keep = torch.isin(batch_indices, proposals)
            local_batch = torch.searchsorted(proposals, batch_indices[keep])
            starts = torch.cumsum(counts, dim=0) - counts
            slot = torch.arange(batch_indices.numel(), device=batch_indices.device)[keep] - starts[batch_indices[keep]]
            batch_ref_corr_points = ref_knn_points.new_zeros(proposals.numel(), max_corr, 3)
            batch_src_corr_points = ref_knn_points.new_zeros(proposals.numel(), max_corr, 3)
            batch_corr_scores = global_corr_scores.new_zeros(proposals.numel(), max_corr)
            batch_ref_corr_points[local_batch, slot] = global_ref_corr_points[keep]
            batch_src_corr_points[local_batch, slot] = global_src_corr_points[keep]
            batch_corr_scores[local_batch, slot] = global_corr_scores[keep]
            batch_transforms = weighted_procrustes(batch_src_corr_points, batch_ref_corr_points, batch_corr_scores)
            batch_aligned = transform_points(src_corr_points.unsqueeze(0), batch_transforms)  # (P, C, 3)
            batch_residuals = torch.linalg.norm(ref_corr_points.unsqueeze(0) - batch_aligned, dim=2)
            batch_inlier_masks = batch_residuals < self.acceptance_radius
            best_index = batch_inlier_masks.sum(dim=1).argmax()
            cur_corr_scores = corr_scores * batch_inlier_masks[best_index].float()
        else:
            estimated_transform = weighted_procrustes(src_corr_points, ref_corr_points, corr_scores)
            cur_corr_scores = self.recompute_correspondence_scores(
                ref_corr_points, src_corr_points, corr_scores, estimated_transform
            )

        estimated_transform = weighted_procrustes(src_corr_points, ref_corr_points, cur_corr_scores)
        for _ in range(self.num_refinement_steps - 1):
            cur_corr_scores = self.recompute_correspondence_scores(
                ref_corr_points, src_corr_points, corr_scores, estimated_transform
            )
            estimated_transform = weighted_procrustes(src_corr_points, ref_corr_points, cur_corr_scores)

        return global_ref_corr_points, global_src_corr_points, global_corr_scores, estimated_transform

    @torch.no_grad()
    def decode(self, output: GeoTransformerOutput) -> Registration3D:
        r"""Estimate the transform of every pair with the local-to-global registration.

        Args:
            output: The `forward` output of a batch.

        Returns:
            A `Registration3D` with the `transform` $(B, 4, 4)$ of every pair and the packed point correspondences
            `src_points`, `ref_points`, `scores` and `batch`.
        """
        node_corr_batch = output["node_corr_batch"]
        ref_batch_c = output["ref_batch_c"]
        batch_size = int(ref_batch_c.max()) + 1 if ref_batch_c.numel() else 0
        transforms: List[Tensor] = []
        src_points: List[Tensor] = []
        ref_points: List[Tensor] = []
        scores: List[Tensor] = []
        batches: List[Tensor] = []
        for b in range(batch_size):
            pairs = node_corr_batch == b
            if not bool(pairs.any()):
                empty = output["ref_pos_c"].new_zeros(0, 3)
                transforms.append(torch.eye(4, dtype=empty.dtype, device=empty.device))
                src_points.append(empty)
                ref_points.append(empty)
                scores.append(empty[:, 0])
                batches.append(empty[:, 0].long())
                continue
            score_mat = torch.exp(output["matching_scores"][pairs][:, :-1, :-1])
            ref_masks, src_masks = output["ref_patch_masks"][pairs], output["src_patch_masks"][pairs]
            corr_mat = self.correspondence_matrix(score_mat, ref_masks, src_masks)
            score_mat = score_mat * corr_mat.float()
            ref_corr, src_corr, corr_scores, transform = self.local_to_global_registration(
                output["ref_patch_points"][pairs], output["src_patch_points"][pairs], score_mat, corr_mat
            )
            transforms.append(transform)
            src_points.append(src_corr)
            ref_points.append(ref_corr)
            scores.append(corr_scores)
            batches.append(torch.full_like(corr_scores, b, dtype=torch.long))

        return {
            "transform": torch.stack(transforms, dim=0),
            "src_points": torch.cat(src_points, dim=0),
            "ref_points": torch.cat(ref_points, dim=0),
            "scores": torch.cat(scores, dim=0),
            "batch": torch.cat(batches, dim=0),
        }


_REGISTRATION_INPUT_KEYS = (
    DataKeys.X,
    DataKeys.POS,
    DataKeys.BATCH,
    DataKeys.REF_X,
    DataKeys.REF_POS,
    DataKeys.BATCH_REF_POS,
)


@register_model(
    "geotransformer.3dmatch.zheng-qin",
    task="registration",
    input_keys=_REGISTRATION_INPUT_KEYS,
    weights=WeightsDict(
        url="hf://torch-pointcloud/geotransformer.3dmatch.zheng-qin/resolve/1f4ff32be914a2b6cc5a7cff38306e18092bc194/model.safetensors",
        dataset="3dmatch",
        metrics={
            "RR": 92.48,
            "FMR": 98.14,
            "IR": 70.93,
            "3DLoMatch/RR": 74.36,
            "3DLoMatch/FMR": 87.07,
            "3DLoMatch/IR": 43.54,
        },
        author="zheng-qin",
        license="MIT",
    ),
    hparams=dict(
        in_channels=1,
        stem_channels=64,
        encoder_channels=[128, 256, 512, 1024],
        decoder_channels=[512, 256],
        voxel_sizes=[0.05, 0.1, 0.2],
        radii=[0.0625, 0.125, 0.25, 0.5],
        kp_sigma=[0.05, 0.1, 0.2, 0.4],
        max_num_neighbors=[38, 36, 36, 38],
        embed_dim=256,
        sigma_d=0.2,
        num_points_in_patch=64,
        matching_radius=0.05,
        num_correspondences=256,
        topk=3,
        acceptance_radius=0.1,
    ),
)
def geotransformer_3dmatch(**hparams: Any) -> GeoTransformerRegistration:
    return GeoTransformerRegistration(**hparams)


@register_model(
    "geotransformer.kitti.zheng-qin",
    task="registration",
    input_keys=_REGISTRATION_INPUT_KEYS,
    transform=T.Compose(
        [
            T.Voxelize(pos_key=DataKeys.POS, pos_reduce="mean", size=0.3, method="grid"),
            T.Voxelize(pos_key=DataKeys.REF_POS, pos_reduce="mean", size=0.3, method="grid"),
        ]
    ),
    weights=WeightsDict(
        url="hf://torch-pointcloud/geotransformer.kitti.zheng-qin/resolve/087d8f64d8e29990f4d2b6ece451134223dc483a/model.safetensors",
        dataset="kitti",
        author="zheng-qin",
        license="MIT",
    ),
    hparams=dict(
        in_channels=1,
        stem_channels=64,
        encoder_channels=[128, 256, 512, 1024, 2048],
        decoder_channels=[1024, 512, 256],
        voxel_sizes=[0.6, 1.2, 2.4, 4.8],
        radii=[1.275, 2.55, 5.1, 10.2, 20.4],
        kp_sigma=[0.6, 1.2, 2.4, 4.8, 9.6],
        max_num_neighbors=[70, 68, 75, 75, 76],
        embed_dim=128,
        sigma_d=4.8,
        num_points_in_patch=128,
        matching_radius=0.6,
        num_correspondences=256,
        topk=2,
        acceptance_radius=0.6,
    ),
)
def geotransformer_kitti(**hparams: Any) -> GeoTransformerRegistration:
    return GeoTransformerRegistration(**hparams)


@register_model(
    "geotransformer.modelnet40.zheng-qin",
    task="registration",
    input_keys=_REGISTRATION_INPUT_KEYS,
    weights=WeightsDict(
        url="hf://torch-pointcloud/geotransformer.modelnet40.zheng-qin/resolve/dc8b40eeeae1b3b642a6e22a1eea9165d40f3177/model.safetensors",
        dataset="modelnet40",
        metrics={"RRE": 1.597, "RTE": 0.019, "RMSE": 0.018},
        author="zheng-qin",
        license="MIT",
    ),
    hparams=dict(
        in_channels=1,
        stem_channels=64,
        encoder_channels=[128, 256, 512],
        decoder_channels=[256, 256],
        voxel_sizes=[0.1, 0.2],
        radii=[0.125, 0.25, 0.5],
        kp_sigma=[0.1, 0.2, 0.4],
        max_num_neighbors=[31, 31, 33],
        embed_dim=256,
        sigma_d=0.2,
        num_points_in_patch=128,
        matching_radius=0.05,
        num_correspondences=128,
        topk=3,
        acceptance_radius=0.1,
    ),
)
def geotransformer_modelnet40(**hparams: Any) -> GeoTransformerRegistration:
    return GeoTransformerRegistration(**hparams)
