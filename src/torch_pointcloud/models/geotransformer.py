"""GeoTransformer point cloud registration model.

{{ paper("2202.06688") }}
"""

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple, TypedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn import MLP
from torch_geometric.utils import cumsum, degree, mask_to_index, to_dense_batch
from typing_extensions import NotRequired

import torch_pointcloud.transforms as T
from torch_pointcloud.layers.kpconv_blocks import KPConvBlock, KPResidualBlock
from torch_pointcloud.layers.norms import create_norm
from torch_pointcloud.layers.optimal_transport import LogOptimalTransport
from torch_pointcloud.layers.sinusoidal_embedding import sinusoidal_embedding
from torch_pointcloud.layers.voxel_grid_pool import VoxelGridPool
from torch_pointcloud.ops.cluster import dense_neighbors, gather_neighbors, knn
from torch_pointcloud.ops.registration import (
    local_to_global_registration,
    mutual_topk_correspondences,
    node_correspondences,
    pairwise_squared_distance,
    point_to_node_partition,
)
from torch_pointcloud.utils.data import DataKeys
from torch_pointcloud.utils.state_dict import transform_state_dict
from torch_pointcloud.utils.types import FeaturesDict, OptTensor, Registration3D

from ._base import RegistrationModel
from ._registry import WeightsDict, register_model


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
        gt_node_corr_indices: Ground-truth `(source, reference)` superpoint correspondences, shape $(G, 2)$; present
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


class GeometricStructureEmbedding(nn.Module):
    r"""Geometric structure embedding of a superpoint set: pair-wise distances and triplet-wise angles.

    For every pair of superpoints $(i, j)$ the embedding sums a projection of the sinusoidal embedding of their
    distance $d_{ij} / \sigma_d$ and a projection of the sinusoidal embedding of the angles between
    $\vec{p_i p_j}$ and the vectors from $p_i$ to its $k$ nearest superpoints, scaled by $\sigma_a$ and reduced
    over the $k$ neighbors. A set of fewer than `angle_k + 1` superpoints measures the angles against all the
    others, and a single superpoint has no angular term. Sets padded to a common size take a mask of their valid
    superpoints, and the padding takes no part in the angles of the valid ones.

    Args:
        embed_dim: Embedding width $C$.
        sigma_d: Distance temperature, in the unit of the coordinates.
        sigma_a: Angle temperature, in degrees.
        angle_k: Number of nearest superpoints $k$ the angles are measured against.

    Shape:
        - Input: $(B, N, 3)$ superpoints and an optional $(B, N)$ mask of the valid ones.
        - Output: $(B, N, N, C)$.
    """

    def __init__(
        self,
        embed_dim: int,
        sigma_d: float,
        sigma_a: float,
        angle_k: int,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.sigma_d = sigma_d
        self.factor_a = 180.0 / (sigma_a * math.pi)
        self.angle_k = angle_k
        self.proj_d = nn.Linear(embed_dim, embed_dim)
        self.proj_a = nn.Linear(embed_dim, embed_dim)

    @torch.no_grad()
    def embedding_indices(self, points: Tensor, masks: OptTensor = None) -> Tuple[Tensor, Tensor, Tensor]:
        r"""Distance indices $(B, N, N)$ and angular indices $(B, N, N, k)$ of a superpoint set, and whether each set
        has angle neighbors $(B,)$."""
        batch_size, num_points, _ = points.shape
        dist_map = torch.sqrt(pairwise_squared_distance(points, points))  # (B, N, N)
        d_indices = dist_map / self.sigma_d

        k = min(self.angle_k, num_points - 1)
        knn_dist_map = dist_map if masks is None else dist_map.masked_fill(~masks.unsqueeze(1), float("inf"))
        knn_indices = knn_dist_map.topk(k=k + 1, dim=2, largest=False)[1][:, :, 1:]  # (B, N, k)
        has_neighbors = torch.full((batch_size,), k > 0, dtype=torch.bool, device=points.device)
        if masks is not None and k > 0:
            # A padded neighbor repeats the nearest one, which leaves the max over the neighbors unchanged.
            knn_masks = torch.gather(masks.unsqueeze(1).expand(-1, num_points, -1), 2, knn_indices)
            knn_indices = torch.where(knn_masks, knn_indices, knn_indices[:, :, :1])
            has_neighbors = masks.sum(dim=1) > 1
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
        return d_indices, a_indices, has_neighbors

    def forward(self, points: Tensor, masks: OptTensor = None) -> Tensor:
        d_indices, a_indices, has_neighbors = self.embedding_indices(points, masks)
        d_embeddings = self.proj_d(sinusoidal_embedding(d_indices, self.embed_dim))
        if a_indices.size(3) == 0:
            return d_embeddings
        a_embeddings = self.proj_a(sinusoidal_embedding(a_indices, self.embed_dim)).amax(dim=3)
        if masks is None:
            return d_embeddings + a_embeddings
        # A padded set of a single superpoint has no angular term.
        return torch.addcmul(d_embeddings, a_embeddings, has_neighbors.view(-1, 1, 1, 1).to(a_embeddings.dtype))


class GeometricAttention(nn.Module):
    r"""Multi-head attention over dense token sets, with an optional relative positional term.

    With `pos_embed`, the logits are $(q_i \cdot k_j + q_i \cdot p_{ij}) / \sqrt{d}$ where $p_{ij}$ is the
    projected pair-wise embedding, the geometric self-attention of GeoTransformer; without it the layer is plain
    scaled dot-product cross-attention. A key mask leaves the padding of sets padded to a common size out of the
    softmax.

    Args:
        embed_dim: Token width $C$, a multiple of `num_heads`.
        num_heads: Number of attention heads.
        relative_pos: Whether the layer projects and attends to a pair-wise positional embedding.

    Shape:
        - Input: $(B, N, C)$ queries, $(B, M, C)$ keys and values, an optional $(B, N, M, C)$ positional embedding
          and an optional $(B, M)$ mask of the valid keys.
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
        key_mask: OptTensor = None,
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
        if key_mask is not None:
            scores = scores.masked_fill(~key_mask.view(batch_size, 1, 1, num_keys), float("-inf"))
        scores = F.softmax(scores / self.head_dim**0.5, dim=-1)

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
          an optional $(B, M)$ mask of the valid memory tokens.
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
        memory_mask: OptTensor = None,
    ) -> Tensor:
        hidden = self.attention(x, memory, memory, pos_embed=pos_embed, key_mask=memory_mask)
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
        depth: Number of transformer blocks, each a geometric self-attention layer followed by a cross-attention
            layer.
        sigma_d: Distance temperature of the geometric structure embedding.
        sigma_a: Angle temperature of the geometric structure embedding, in degrees.
        angle_k: Number of nearest superpoints of the angular embedding.

    Each cross-attention layer updates the reference features first, then the source features from the updated
    reference ones.

    Shape:
        - Input: $(B, N, 3)$ source superpoints, $(B, M, 3)$ reference superpoints, $(B, N, C_\text{in})$ and
          $(B, M, C_\text{in})$ features, and optional $(B, N)$ and $(B, M)$ masks of the valid superpoints when the
          sets of a cloud are padded to a common size.
        - Output: $(B, N, C_\text{out})$ source and $(B, M, C_\text{out})$ reference features.
    """

    def __init__(
        self,
        in_channels: int,
        embed_dim: int,
        out_channels: int,
        num_heads: int,
        depth: int,
        sigma_d: float,
        sigma_a: float,
        angle_k: int,
    ) -> None:
        super().__init__()
        if depth < 1:
            raise ValueError(f"`depth` must be positive, got {depth}.")

        self.embedding = GeometricStructureEmbedding(embed_dim, sigma_d, sigma_a, angle_k)
        self.in_proj = nn.Linear(in_channels, embed_dim)
        self.layers = nn.ModuleList(
            [GeometricTransformerLayer(embed_dim, num_heads, relative_pos=i % 2 == 0) for i in range(2 * depth)]
        )
        self.out_proj = nn.Linear(embed_dim, out_channels)

    def forward(
        self,
        src_points: Tensor,
        ref_points: Tensor,
        src_feats: Tensor,
        ref_feats: Tensor,
        src_masks: OptTensor = None,
        ref_masks: OptTensor = None,
    ) -> Tuple[Tensor, Tensor]:
        ref_embeddings = self.embedding(ref_points, ref_masks)
        src_embeddings = self.embedding(src_points, src_masks)
        ref_feats = self.in_proj(ref_feats)
        src_feats = self.in_proj(src_feats)
        for i, layer in enumerate(self.layers):
            if i % 2 == 0:
                ref_feats = layer(ref_feats, ref_feats, pos_embed=ref_embeddings, memory_mask=ref_masks)
                src_feats = layer(src_feats, src_feats, pos_embed=src_embeddings, memory_mask=src_masks)
            else:
                ref_feats = layer(ref_feats, src_feats, memory_mask=src_masks)
                src_feats = layer(src_feats, ref_feats, memory_mask=ref_masks)
        return self.out_proj(src_feats), self.out_proj(ref_feats)


class KPConvFPN(nn.Module):
    r"""KPConv feature pyramid of the registration backbones: a stem stage at the input resolution, grid-subsampled
    encoder stages, and nearest-neighbor upsampling decoders back to the fine stage.

    The first stage is a group-normalized `KPConvBlock` followed by one `KPResidualBlock`; every further stage
    grid-subsamples the points, then runs a strided residual block and two residual blocks at the doubled radius. The
    decoders upsample the coarsest features to each finer stage, concatenate that stage's encoder features and mix
    them with a unary `MLP`; the last decoder is a plain linear layer and defines the fine stage. Neighborhoods are the
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

        self.encoder_channels = tuple(encoder_channels)
        self.decoder_channels = tuple(decoder_channels)
        self.radii = tuple(radii)
        self.max_num_neighbors = tuple(max_num_neighbors)

        self.pools = nn.ModuleList([VoxelGridPool(grid_size=size, reduce="max") for size in voxel_sizes])
        self.stages = nn.ModuleList()
        stem = KPConvBlock(
            spatial_dim=3,
            in_channels=in_channels,
            out_channels=stem_channels,
            kernel_size=kernel_size,
            kp_radius=radii[0],
            kp_sigma=kp_sigma[0],
            act="leaky_relu",
            act_kwargs={"negative_slope": 0.1},
            norm="group_norm",
            norm_kwargs={"num_groups": num_groups},
            bias=True,
            normalize_by_positive_neighbors=True,
        )
        block_kwargs: Dict[str, Any] = dict(
            spatial_dim=3,
            kernel_size=kernel_size,
            act="leaky_relu",
            act_kwargs={"negative_slope": 0.1},
            norm="group_norm",
            norm_kwargs={"num_groups": num_groups},
            bias=True,
            normalize_by_positive_neighbors=True,
        )
        first = KPResidualBlock(
            in_channels=stem_channels,
            out_channels=encoder_channels[0],
            kp_radius=radii[0],
            kp_sigma=kp_sigma[0],
            **block_kwargs,
        )
        self.stages.append(nn.ModuleList([stem, first]))
        for i in range(1, num_stages):
            strided = KPResidualBlock(
                in_channels=encoder_channels[i - 1],
                out_channels=encoder_channels[i - 1],
                kp_radius=radii[i - 1],
                kp_sigma=kp_sigma[i - 1],
                strided=True,
                max_num_neighbors=max_num_neighbors[i - 1],
                **block_kwargs,
            )
            blocks = [
                KPResidualBlock(
                    in_channels=encoder_channels[i - 1] if j == 0 else encoder_channels[i],
                    out_channels=encoder_channels[i],
                    kp_radius=radii[i],
                    kp_sigma=kp_sigma[i],
                    **block_kwargs,
                )
                for j in range(2)
            ]
            self.stages.append(nn.ModuleList([strided, *blocks]))

        self.decoders = nn.ModuleList()
        in_channels = encoder_channels[-1]
        for i, out_channels in enumerate(self.decoder_channels):
            skip_channels = encoder_channels[num_stages - 2 - i]
            last = i == len(self.decoder_channels) - 1
            decoder = MLP(
                [in_channels + skip_channels, out_channels],
                act="leaky_relu",
                act_kwargs={"negative_slope": 0.1},
                norm=None if last else create_norm("group_norm", out_channels, num_groups=num_groups),
                plain_last=last,
            )
            self.decoders.append(decoder)
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
        num_clouds: int,
    ) -> Tensor:
        r"""`[source, target]` edges from the $k$ nearest points of `pos` within `radius` of every query point."""
        row, col = knn(pos, query_pos, k=k, batch_x=cloud, batch_y=query_cloud, batch_size=num_clouds)
        within = torch.linalg.norm(query_pos[row] - pos[col], dim=-1) < radius
        return torch.stack([col[within], row[within]], dim=0)

    def forward(self, x: Tensor, pos: Tensor, cloud: Tensor, batch: Tensor) -> Tuple[FeaturesDict, FeaturesDict]:
        num_stages = len(self.stages)
        num_clouds = int(cloud.max()) + 1
        batch_size = int(batch.max()) + 1
        feats: List[Tensor] = []
        positions: List[Tensor] = [pos]
        clouds: List[Tensor] = [cloud]
        batches: List[Tensor] = [batch]

        for i, modules in enumerate(self.stages):
            assert isinstance(modules, nn.ModuleList)
            blocks = list(modules)
            if i > 0:
                _, pos, cloud, inverse = self.pools[i - 1](pos, pos, cloud, return_inverse=True, batch_size=num_clouds)
                batch = torch.zeros_like(cloud).scatter_(0, inverse, batch)
                positions.append(pos)
                clouds.append(cloud)
                batches.append(batch)
                edge_index = self.neighbors(
                    positions[i - 1],
                    clouds[i - 1],
                    pos,
                    cloud,
                    self.radii[i - 1],
                    self.max_num_neighbors[i - 1],
                    num_clouds,
                )
                x = blocks.pop(0)(x, (positions[i - 1], pos), edge_index, (batches[i - 1], batch), batch_size)

            edge_index = self.neighbors(pos, cloud, pos, cloud, self.radii[i], self.max_num_neighbors[i], num_clouds)
            for block in blocks:
                x = block(x, pos, edge_index, batch, batch_size)
            feats.append(x)

        coarse: FeaturesDict = {"x": x, "pos": positions[-1], "batch": clouds[-1]}
        latent = x
        for i, decoder in enumerate(self.decoders):
            stage = num_stages - 2 - i
            edge_index = self.neighbors(
                positions[stage + 1],
                clouds[stage + 1],
                positions[stage],
                clouds[stage],
                self.radii[stage + 1],
                1,
                num_clouds,
            )
            nearest, _ = dense_neighbors(edge_index, latent.size(0), positions[stage].size(0), max_num_neighbors=1)
            upsampled = gather_neighbors(latent, nearest).squeeze(1)
            latent = decoder(torch.cat([upsampled, feats[stage]], dim=1), batches[stage], batch_size)

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

    The arguments in the unit of the coordinates (`voxel_sizes`, `radii`, `kp_sigma`, `sigma_d`, `matching_radius`,
    `acceptance_radius`) and the neighbor limits calibrated on the training data have no default; the registered
    checkpoints set them.

    Args:
        in_channels: Number of input feature channels; the registered checkpoints read a constant feature.
        voxel_sizes: Grid size of each backbone subsampling.
        radii: Neighborhood radius of each backbone stage.
        kp_sigma: Kernel extent of each backbone stage.
        max_num_neighbors: Neighbor limit of each backbone stage.
        sigma_d: Distance temperature of the geometric structure embedding.
        matching_radius: Distance under which two fine points match when building the ground-truth superpoint
            correspondences from the ground-truth transform.
        acceptance_radius: Inlier distance of the local-to-global registration.
        stem_channels: Output width of the backbone stem.
        encoder_channels: Output width of each backbone stage.
        decoder_channels: Output width of each backbone decoder; the last one is the fine feature width.
        kernel_size: Number of kernel points of every convolution.
        num_groups: Number of channel groups of the backbone normalizations.
        embed_dim: Token width of the geometric transformer.
        coarse_channels: Width of the superpoint features the transformer outputs.
        num_heads: Number of attention heads.
        depth: Number of transformer blocks, each a geometric self-attention layer followed by a cross-attention
            layer.
        sigma_a: Angle temperature of the geometric structure embedding, in degrees.
        angle_k: Number of nearest superpoints of the angular embedding.
        num_points_in_patch: Number of fine points $K$ in the patch of a superpoint.
        num_correspondences: Number of superpoint correspondences kept at inference.
        num_targets: Number of ground-truth superpoint correspondences sampled at training.
        overlap_threshold: Minimum overlap of a ground-truth superpoint correspondence sampled at training.
        num_sinkhorn_iterations: Number of Sinkhorn iterations of the optimal transport.
        topk: Number of candidates kept per row and per column of a patch assignment in `decode`.
        confidence_threshold: Minimum assignment probability of a point correspondence.
        correspondence_threshold: Minimum number of point correspondences for a patch pair to propose a transform.
        num_refinement_steps: Number of weighted Procrustes fits on the inliers of the selected transform.
    """

    def __init__(
        self,
        in_channels: int = 1,
        *,
        voxel_sizes: Sequence[float],
        radii: Sequence[float],
        kp_sigma: Sequence[float],
        max_num_neighbors: Sequence[int],
        sigma_d: float,
        matching_radius: float,
        acceptance_radius: float,
        stem_channels: int = 64,
        encoder_channels: Sequence[int] = (128, 256, 512, 1024),
        decoder_channels: Sequence[int] = (512, 256),
        kernel_size: int = 15,
        num_groups: int = 32,
        embed_dim: int = 256,
        coarse_channels: int = 256,
        num_heads: int = 4,
        depth: int = 3,
        sigma_a: float = 15.0,
        angle_k: int = 3,
        num_points_in_patch: int = 64,
        num_correspondences: int = 256,
        num_targets: int = 128,
        overlap_threshold: float = 0.1,
        num_sinkhorn_iterations: int = 100,
        topk: int = 3,
        confidence_threshold: float = 0.05,
        correspondence_threshold: int = 3,
        num_refinement_steps: int = 5,
    ) -> None:
        super().__init__(in_channels)
        self.voxel_sizes = tuple(voxel_sizes)
        self.radii = tuple(radii)
        self.kp_sigma = tuple(kp_sigma)
        self.max_num_neighbors = tuple(max_num_neighbors)
        self.sigma_d = sigma_d
        self.matching_radius = matching_radius
        self.acceptance_radius = acceptance_radius
        self.stem_channels = stem_channels
        self.encoder_channels = tuple(encoder_channels)
        self.decoder_channels = tuple(decoder_channels)
        self.kernel_size = kernel_size
        self.num_groups = num_groups
        self.embed_dim = embed_dim
        self.coarse_channels = coarse_channels
        self.num_heads = num_heads
        self.depth = depth
        self.sigma_a = sigma_a
        self.angle_k = angle_k
        self.num_points_in_patch = num_points_in_patch
        self.num_correspondences = num_correspondences
        self.num_targets = num_targets
        self.overlap_threshold = overlap_threshold
        self.num_sinkhorn_iterations = num_sinkhorn_iterations
        self.topk = topk
        self.confidence_threshold = confidence_threshold
        self.correspondence_threshold = correspondence_threshold
        self.num_refinement_steps = num_refinement_steps

        self.backbone = self.configure_backbone()
        self.transformer = self.configure_transformer()
        self.optimal_transport = self.configure_optimal_transport()

    def configure_backbone(self) -> KPConvFPN:
        """Build the `KPConvFPN` backbone shared by the two clouds."""
        return KPConvFPN(
            self.in_channels,
            stem_channels=self.stem_channels,
            encoder_channels=self.encoder_channels,
            decoder_channels=self.decoder_channels,
            voxel_sizes=self.voxel_sizes,
            radii=self.radii,
            kp_sigma=self.kp_sigma,
            max_num_neighbors=self.max_num_neighbors,
            kernel_size=self.kernel_size,
            num_groups=self.num_groups,
        )

    def configure_transformer(self) -> GeometricTransformer:
        """Build the `GeometricTransformer` on the coarsest backbone stage."""
        return GeometricTransformer(
            self.encoder_channels[-1],
            self.embed_dim,
            self.coarse_channels,
            self.num_heads,
            self.depth,
            self.sigma_d,
            self.sigma_a,
            self.angle_k,
        )

    def configure_optimal_transport(self) -> LogOptimalTransport:
        """Build the Sinkhorn layer matching the points of two patches."""
        return LogOptimalTransport(self.num_sinkhorn_iterations)

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
            batch: Pair index of every source point, shape $(N,)$, sorted.
            ref_x: Reference features of shape $(M, C)$, or `None`.
            ref_pos: Reference coordinates of shape $(M, 3)$.
            ref_batch: Pair index of every reference point, shape $(M,)$, sorted.

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
        src_feats: Tensor,
        ref_feats: Tensor,
        src_masks: Tensor,
        ref_masks: Tensor,
        src_batch: Tensor,
        ref_batch: Tensor,
        batch_size: int,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        r"""Top `num_correspondences` superpoint pairs of every pair of clouds by dual-normalized feature similarity,
        ignoring the superpoints with an empty patch.

        Args:
            src_feats: Unit-norm source superpoint features, shape $(M_s, C)$.
            ref_feats: Unit-norm reference superpoint features, shape $(M_r, C)$.
            src_masks: Source superpoints with a non-empty patch, shape $(M_s,)$.
            ref_masks: Reference superpoints with a non-empty patch, shape $(M_r,)$.
            src_batch: Pair of every source superpoint, shape $(M_s,)$, sorted.
            ref_batch: Pair of every reference superpoint, shape $(M_r,)$, sorted.
            batch_size: Number of pairs $B$.

        Returns:
            The source and the reference superpoint of every correspondence, indexed in `src_feats` and `ref_feats`,
            its score and its pair, shape $(P,)$ each, ordered by pair then decreasing score.
        """
        src_indices, ref_indices = mask_to_index(src_masks), mask_to_index(ref_masks)
        src_feats, src_valid = to_dense_batch(src_feats[src_indices], src_batch[src_indices], batch_size=batch_size)
        ref_feats, ref_valid = to_dense_batch(ref_feats[ref_indices], ref_batch[ref_indices], batch_size=batch_size)
        src_indices, _ = to_dense_batch(src_indices, src_batch[src_indices], batch_size=batch_size)
        ref_indices, _ = to_dense_batch(ref_indices, ref_batch[ref_indices], batch_size=batch_size)

        valid = ref_valid.unsqueeze(2) & src_valid.unsqueeze(1)  # (B, N_r, N_s)
        scores = torch.exp(-pairwise_squared_distance(ref_feats, src_feats, normalized=True)) * valid
        ref_scores = scores / scores.sum(dim=2, keepdim=True)
        src_scores = scores / scores.sum(dim=1, keepdim=True)
        scores = torch.where(valid, ref_scores * src_scores, -1.0)
        num_correspondences = min(self.num_correspondences, scores[0].numel())
        corr_scores, corr_indices = scores.flatten(1).topk(k=num_correspondences, dim=1, largest=True)
        ref_corr_indices, src_corr_indices = torch.unravel_index(corr_indices, scores.shape[1:])
        corr_batch = torch.arange(batch_size, device=scores.device).unsqueeze(1).expand_as(corr_indices)
        keep = corr_scores >= 0
        return (
            src_indices.gather(1, src_corr_indices)[keep],
            ref_indices.gather(1, ref_corr_indices)[keep],
            corr_scores[keep],
            corr_batch[keep],
        )

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

        cloud = coarse["batch"]
        num_clouds = int(cloud.max()) + 1
        batch_size = num_clouds // 2
        ptr = cumsum(degree(cloud, num_clouds, dtype=torch.long))
        num_src = int(ptr[batch_size])
        local = torch.arange(cloud.size(0), device=cloud.device) - ptr[cloud]  # index within the cloud

        _, node_masks, node_knn_indices, node_knn_masks = point_to_node_partition(
            fine["pos"], coarse["pos"], self.num_points_in_patch, fine["batch"], cloud, num_clouds
        )
        node_knn_points = gather_neighbors(fine["pos"], node_knn_indices)  # (M, K, 3)

        src_pos_c, ref_pos_c = coarse["pos"][:num_src], coarse["pos"][num_src:]
        src_batch_c, ref_batch_c = cloud[:num_src], cloud[num_src:] - batch_size
        src_local, ref_local = local[:num_src], local[num_src:]
        src_masks, ref_masks = node_masks[:num_src], node_masks[num_src:]
        src_knn_indices, ref_knn_indices = node_knn_indices[:num_src], node_knn_indices[num_src:]
        src_knn_masks, ref_knn_masks = node_knn_masks[:num_src], node_knn_masks[num_src:]
        src_knn_points, ref_knn_points = node_knn_points[:num_src], node_knn_points[num_src:]

        src_points, src_valid = to_dense_batch(src_pos_c, src_batch_c, batch_size=batch_size)
        ref_points, ref_valid = to_dense_batch(ref_pos_c, ref_batch_c, batch_size=batch_size)
        src_x, _ = to_dense_batch(coarse["x"][:num_src], src_batch_c, batch_size=batch_size)
        ref_x, _ = to_dense_batch(coarse["x"][num_src:], ref_batch_c, batch_size=batch_size)
        src_feats_c, ref_feats_c = self.transformer(
            src_points, ref_points, src_x, ref_x, src_masks=src_valid, ref_masks=ref_valid
        )
        src_feats_c = F.normalize(src_feats_c[src_valid], p=2, dim=1)
        ref_feats_c = F.normalize(ref_feats_c[ref_valid], p=2, dim=1)

        if transform is not None:
            gt_node_corr_indices, gt_node_corr_overlaps = node_correspondences(
                src_pos_c,
                ref_pos_c,
                src_knn_points,
                ref_knn_points,
                transform,
                self.matching_radius,
                src_masks=src_masks,
                ref_masks=ref_masks,
                src_knn_masks=src_knn_masks,
                ref_knn_masks=ref_knn_masks,
                src_batch=src_batch_c,
                ref_batch=ref_batch_c,
            )
            gt_node_corr_batch = src_batch_c[gt_node_corr_indices[:, 0]]

        with torch.no_grad():
            if self.training:
                keep = gt_node_corr_overlaps > self.overlap_threshold
                targets, node_corr_scores = gt_node_corr_indices[keep], gt_node_corr_overlaps[keep]
                node_corr_batch = gt_node_corr_batch[keep]
                # At most `num_targets` targets per pair, drawn at random, in their original order.
                order = torch.argsort(node_corr_batch + torch.rand_like(node_corr_scores))
                rank = torch.empty_like(order)
                rank[order] = torch.arange(order.size(0), device=order.device)
                first = cumsum(degree(node_corr_batch, batch_size, dtype=torch.long))[node_corr_batch]
                sampled = rank - first < self.num_targets
                targets, node_corr_scores = targets[sampled], node_corr_scores[sampled]
                node_corr_batch = node_corr_batch[sampled]
                src_node_corr_indices, ref_node_corr_indices = targets.unbind(dim=1)
            else:
                src_node_corr_indices, ref_node_corr_indices, node_corr_scores, node_corr_batch = (
                    self.superpoint_matching(
                        src_feats_c, ref_feats_c, src_masks, ref_masks, src_batch_c, ref_batch_c, batch_size
                    )
                )

        ref_node_corr_knn_indices = ref_knn_indices[ref_node_corr_indices]  # (P, K)
        src_node_corr_knn_indices = src_knn_indices[src_node_corr_indices]  # (P, K)
        ref_node_corr_knn_masks = ref_knn_masks[ref_node_corr_indices]  # (P, K)
        src_node_corr_knn_masks = src_knn_masks[src_node_corr_indices]  # (P, K)
        ref_node_corr_knn_feats = gather_neighbors(fine["x"], ref_node_corr_knn_indices)  # (P, K, C)
        src_node_corr_knn_feats = gather_neighbors(fine["x"], src_node_corr_knn_indices)  # (P, K, C)

        matching_scores = torch.einsum("bnd,bmd->bnm", ref_node_corr_knn_feats, src_node_corr_knn_feats)
        matching_scores = matching_scores / fine["x"].size(1) ** 0.5
        matching_scores = self.optimal_transport(matching_scores, ref_node_corr_knn_masks, src_node_corr_knn_masks)

        output: GeoTransformerOutput = {
            "src_pos_c": src_pos_c,
            "src_batch_c": src_batch_c,
            "src_feats_c": src_feats_c,
            "ref_pos_c": ref_pos_c,
            "ref_batch_c": ref_batch_c,
            "ref_feats_c": ref_feats_c,
            "ref_node_corr_indices": ref_local[ref_node_corr_indices],
            "src_node_corr_indices": src_local[src_node_corr_indices],
            "node_corr_scores": node_corr_scores,
            "node_corr_batch": node_corr_batch,
            "ref_patch_points": ref_knn_points[ref_node_corr_indices],
            "src_patch_points": src_knn_points[src_node_corr_indices],
            "ref_patch_masks": ref_node_corr_knn_masks,
            "src_patch_masks": src_node_corr_knn_masks,
            "matching_scores": matching_scores,
        }
        if transform is not None:
            output["gt_node_corr_indices"] = torch.stack(
                [src_local[gt_node_corr_indices[:, 0]], ref_local[gt_node_corr_indices[:, 1]]], dim=1
            )
            output["gt_node_corr_overlaps"] = gt_node_corr_overlaps
            output["gt_node_corr_batch"] = gt_node_corr_batch
        return output

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
            batch: Pair index of every source point, shape $(N,)$, sorted.
            ref_x: Reference features of shape $(M, C)$, or `None`.
            ref_pos: Reference coordinates of shape $(M, 3)$.
            ref_batch: Pair index of every reference point, shape $(M,)$, sorted.
            transform: Ground-truth transform of every pair, shape $(B, 4, 4)$; required at training.

        Returns:
            The `GeoTransformerOutput` of the batch, decoded into transforms by `decode`.
        """
        fine, coarse = self.forward_features(x, pos, batch, ref_x, ref_pos, ref_batch)
        return self.forward_head(fine, coarse, transform=transform)

    @torch.no_grad()
    def decode(self, output: GeoTransformerOutput) -> Registration3D:
        r"""Estimate the transform of every pair with the local-to-global registration.

        A pair without any point correspondence (no superpoint correspondence, or none of its patch assignments above
        `confidence_threshold`) gets the identity.

        Args:
            output: The `forward` output of a batch.

        Returns:
            A `Registration3D` with the `transform` $(B, 4, 4)$ of every pair and the packed point correspondences
            `src_points`, `ref_points`, `scores` and `batch`.
        """
        ref_batch_c = output["ref_batch_c"]
        batch_size = int(ref_batch_c.max()) + 1 if ref_batch_c.numel() else 0
        score_mat = torch.exp(output["matching_scores"][:, :-1, :-1])  # (P, K, K), reference rows
        corr_mat = mutual_topk_correspondences(
            score_mat,
            self.topk,
            self.confidence_threshold,
            row_mask=output["ref_patch_masks"],
            col_mask=output["src_patch_masks"],
        )
        patch, ref_slot, src_slot = torch.nonzero(corr_mat, as_tuple=True)
        src_points = output["src_patch_points"][patch, src_slot]
        ref_points = output["ref_patch_points"][patch, ref_slot]
        scores = score_mat[patch, ref_slot, src_slot]
        batch = output["node_corr_batch"][patch]
        transform = local_to_global_registration(
            src_points,
            ref_points,
            scores,
            patch,
            self.acceptance_radius,
            self.correspondence_threshold,
            self.num_refinement_steps,
            batch=batch,
            batch_size=batch_size,
        )
        return {
            "transform": transform,
            "src_points": src_points,
            "ref_points": ref_points,
            "scores": scores,
            "batch": batch,
        }


_UNARY_CHECKPOINT_MAPPING = {
    "backbone.stages.{i}.{j}.unary1.linear.{param}": "backbone.stages.{i}.{j}.unary1.lins.0.{param}",
    "backbone.stages.{i}.{j}.unary1.norm.{param}": "backbone.stages.{i}.{j}.unary1.norms.0.{param}",
    "backbone.stages.{i}.{j}.unary2.linear.{param}": "backbone.stages.{i}.{j}.unary2.lins.0.{param}",
    "backbone.stages.{i}.{j}.unary2.norm.{param}": "backbone.stages.{i}.{j}.unary2.norms.0.{param}",
    "backbone.stages.{i}.{j}.shortcut.linear.{param}": "backbone.stages.{i}.{j}.shortcut.lins.0.{param}",
    "backbone.stages.{i}.{j}.shortcut.norm.{param}": "backbone.stages.{i}.{j}.shortcut.norms.0.{param}",
    "backbone.decoders.{i}.linear.{param}": "backbone.decoders.{i}.lins.0.{param}",
    "backbone.decoders.{i}.norm.{param}": "backbone.decoders.{i}.norms.0.{param}",
    "backbone.decoders.{i}.{param}": "backbone.decoders.{i}.lins.0.{param}",
}


def checkpoint_filter_fn(state_dict: Dict[str, Any], model: nn.Module) -> Dict[str, Any]:
    """Rename the keys of the `torch-pointcloud` GeoTransformer files to the `GeoTransformerRegistration` ones, as the
    `pretrained_filter_fn` of a `WeightsDict`.

    The files name the linear layer and normalization of every unary layer of the backbone `linear` and `norm`, and
    the last decoder is a bare linear layer; the model builds them as single-layer PyG `MLP`s (`lins.0`, `norms.0`).

    Args:
        state_dict: State dict keyed as in the files.
        model: `GeoTransformerRegistration` the state dict is loaded into.

    Returns:
        The state dict keyed as `model` expects.
    """
    assert isinstance(model, GeoTransformerRegistration)
    return transform_state_dict(state_dict, _UNARY_CHECKPOINT_MAPPING)


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
        pretrained_filter_fn=checkpoint_filter_fn,
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
            T.Voxelize(pos_key=DataKeys.POS, pos_reduce="mean", size=0.3, origin="min"),
            T.Voxelize(pos_key=DataKeys.REF_POS, pos_reduce="mean", size=0.3, origin="min"),
        ]
    ),
    weights=WeightsDict(
        url="hf://torch-pointcloud/geotransformer.kitti.zheng-qin/resolve/087d8f64d8e29990f4d2b6ece451134223dc483a/model.safetensors",
        dataset="kitti",
        metrics={"RR": 99.82, "RRE": 0.236, "RTE": 6.33},
        author="zheng-qin",
        license="MIT",
        pretrained_filter_fn=checkpoint_filter_fn,
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
        pretrained_filter_fn=checkpoint_filter_fn,
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
