"""Point-Mamba classification and masked autoencoder pretraining models."""

from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Literal,
    Optional,
    Sequence,
    Tuple,
    Union,
    overload,
)

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch_geometric.nn import MLP

import torch_pointcloud.transforms as T
from torch_pointcloud.datasets.modelnet import MODELNET40_CLASSES
from torch_pointcloud.datasets.scanobjectnn import SCANOBJECTNN_CLASSES
from torch_pointcloud.layers.affine import Affine
from torch_pointcloud.layers.dropouts import DropPath
from torch_pointcloud.layers.pools import AdaptivePoolLike, PoolLike, create_adaptive_pool
from torch_pointcloud.utils.cluster import fps, knn, local_grid
from torch_pointcloud.utils.conversion import ensure_list
from torch_pointcloud.utils.data import DataKeys
from torch_pointcloud.utils.imports import (
    _MAMBA_SSM_GITHUB_URL,
    _TORCH_SCATTER_GITHUB_URL,
    optional_import,
)
from torch_pointcloud.utils.serialization import SerializationOrder, serialize_coords
from torch_pointcloud.utils.types import OptTensor

from ._base import BaseModel, ClassificationModel
from ._registry import WeightsDict, register_model

if TYPE_CHECKING:
    from mamba_ssm import Mamba
    from torch_scatter import scatter

Mamba, _ = optional_import("mamba_ssm", "Mamba", url=_MAMBA_SSM_GITHUB_URL)
scatter, _ = optional_import("torch_scatter", "scatter", url=_TORCH_SCATTER_GITHUB_URL)


RADIUS = 0.03162277660168379  # sqrt(1e-3)


def order_sort(pos_grid: Tensor, batch: Tensor, order: SerializationOrder) -> Tensor:
    """Sort points along a space-filling curve, so a state space model scans spatially close points in sequence.

    Args:
        pos_grid: Non-negative integer grid coordinates of shape $(N, 3)$.
        batch: Per-point batch index of shape $(N,)$.
        order: Space-filling curve to encode the coordinates along.

    Returns:
        The permutation of shape $(N,)$ that sorts the points by their serialization code.

    Raises:
        ValueError: If `pos_grid` holds a negative coordinate, which would silently wrap around to a valid code.
    """
    if pos_grid.numel() and bool(pos_grid.min() < 0):
        raise ValueError(
            "Grid coordinates must be non-negative for serialization: negative values silently wrap around to "
            "valid codes. Shift by the per-axis minimum, as `Voxelize` does."
        )
    # An all-zero grid (single-voxel scene) has bit_length 0, which the encoders reject.
    depth = max(int(pos_grid.max()).bit_length(), 1)
    serialized_code = serialize_coords(pos_grid, batch, depth=depth, order=order)
    serialized_order = torch.argsort(serialized_code)
    return serialized_order


class PointMambaBlock(nn.Module):
    __doc__ = (
        r"""Implementation of the PointMamba block as described in the paper :arxiv: [PointMamba: A Simple State Space Model for Point Cloud Analysis](https://arxiv.org/abs/2402.10739).
        This implementation is adapted from the official repository :github: [LMD0311/PointMamba](https://github.com/LMD0311/PointMamba).
        """
        r"""
        The `PointMambaBlock` is a residual block that consists of a normalization layer,
        a Mamba block, and a stochastic-depth (drop-path) layer:

        ```txt
        x -> Norm -> Mamba -> DropPath -> y
        |                                 ^
        +---------------------------------+
        ```

        Important:
            This module requires the `mamba_ssm` package to be installed.

        Args:
            channels: The number of input and output channels.
            d_state: The number of state channels.
            d_conv: The number of convolution channels.
            expand: The expansion factor for the hidden channels.
            drop_path: The stochastic-depth rate for the residual branch.

        Shapes:
            - Input: $(N, P, C)$ where $N$ is the number of batches, $P$ is the number of patches, and $C$ is the number of channels.
            - Output: $(N, P, C)$ where $N$ is the number of batches, $P$ is the number of patches, and $C$ is the number of channels.
        """
    )

    def __init__(
        self,
        channels: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        drop_path: float = 0.0,
    ):
        super().__init__()
        self.norm = nn.LayerNorm(channels)
        self.mamba = Mamba(d_model=channels, d_state=d_state, d_conv=d_conv, expand=expand, use_fast_path=True)
        self.drop_path = DropPath(drop_path)

    def forward(self, x: Tensor) -> Tensor:
        residual = x
        x = self.norm(x)
        x = self.mamba(x)
        return residual + self.drop_path(x)


class PointPatchEmbedding(nn.Module):
    __doc__ = (
        r"""
        Defines the patch-level encoding based on the paper :arxiv: [PointMamba: A Simple State Space Model for Point Cloud Analysis](https://arxiv.org/abs/2402.10739).
        This implementation is adapted from the official repository :github: [LMD0311/PointMamba](https://github.com/LMD0311/PointMamba).
        """
        r"""
        This module encodes features and positions to a patch-level embedding, 
        where patches are sampled using the farthest point sampling (FPS) method.
        Features and positions are encoded using two MLPs, 
        and the patch-level embedding is obtained by pooling the features and positions.
        
        Args:
            in_channels: The number of input channels.
            out_channels: The number of output channels.
            num_group: The number of patches to sample.
            group_size: The number of neighbors to consider for each patch.
            local_channels: Hidden widths of the per-point MLP (the derived input width is prepended).
            global_channels: Hidden widths of the per-patch MLP.
            act: The activation function to use.
            act_kwargs: The keyword arguments to pass to the activation function.
            act_first: Whether to apply the activation function before the normalization.
            norm: The normalization function to use.
            norm_kwargs: The keyword arguments to pass to the normalization function.
            bias: Whether to use bias in the MLPs.
            aggr: The neighbor aggregation to use.

        Shapes:
            - Input: $(N, C_{\text{in}})$ where $N$ is the number of nodes and $C_{\text{in}}$ is the number of input channels.
            - Output: $(P, C_{\text{out}})$ where $P$ is the number of patches and $C_{\text{out}}$ is the number of output channels.
        """
    )

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_group: int,
        group_size: int,
        spatial_dim: int = 3,
        local_channels: Sequence[int] = (128, 256),
        global_channels: Sequence[int] = (512,),
        act: Union[str, Callable, None] = "relu",
        act_kwargs: Optional[Dict[str, Any]] = None,
        act_first: bool = False,
        norm: Union[str, Callable, None] = "batch_norm",
        norm_kwargs: Optional[Dict[str, Any]] = None,
        bias: Union[bool, List[bool]] = True,
        aggr: PoolLike = "max",
    ):
        super().__init__()
        # The "layer_norm" string resolves to PyG's graph-mode LayerNorm, which normalizes across the
        # whole packed batch; the `nn.LayerNorm` class applies the intended per-point normalization.
        norm = nn.LayerNorm if norm == "layer_norm" else norm
        # Common parameters for all MLPs and associated blocks
        factory_kwargs = dict(
            act=act,
            act_kwargs=act_kwargs,
            act_first=act_first,
            norm=norm,
            norm_kwargs=norm_kwargs,
            bias=bias,
        )

        self.num_group = num_group
        self.group_size = group_size
        self.spatial_dim = spatial_dim

        # In case we do not provide input features (i.e. `in_channels==0`), we use relative positions as features.
        # Otherwise, we concatenate the input features of the centroids (local features),
        # neighbors (global features) and relative positions.
        in_channels = spatial_dim if in_channels == 0 else 2 * in_channels + spatial_dim
        self.local_mlp = MLP([in_channels, *local_channels], **factory_kwargs)
        self.global_mlp = MLP([2 * local_channels[-1], *global_channels, out_channels], **factory_kwargs)
        self.aggr = aggr

    @overload
    def forward(
        self,
        x: OptTensor,
        pos: Tensor,
        batch: Tensor,
        return_pos_rel: Literal[True],
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]: ...

    @overload
    def forward(
        self,
        x: OptTensor,
        pos: Tensor,
        batch: Tensor,
        return_pos_rel: Literal[False] = False,
    ) -> Tuple[Tensor, Tensor, Tensor]: ...

    def forward(self, x: OptTensor, pos: Tensor, batch: Tensor, return_pos_rel: bool = False) -> Any:
        idx_centroid = fps(pos, batch, num_nodes=self.num_group, random_start=self.training)
        pos_centroid = pos[idx_centroid]
        x_centroid = x[idx_centroid] if x is not None else torch.empty(0, device=pos.device)
        batch_centroid = batch[idx_centroid]
        num_centroids = pos_centroid.size(0)
        row, col = knn(pos, pos_centroid, self.group_size, batch_x=batch, batch_y=batch_centroid)

        pos_neighbor = pos[col]
        pos_rel = pos_neighbor - pos_centroid[row]
        x_neighbor = x[col] if x is not None else torch.empty(0, device=pos.device)

        x = torch.cat([x_centroid[row], x_neighbor, pos_rel], dim=-1) if x is not None else pos_rel
        x_local = self.local_mlp(x)
        x_max = scatter(x_local, row, dim=0, dim_size=num_centroids, reduce=self.aggr)

        x_cat = torch.cat([x_max[row], x_local], dim=1)
        x_final = self.global_mlp(x_cat)

        x_patch = scatter(x_final, row, dim=0, dim_size=num_centroids, reduce=self.aggr)

        if return_pos_rel:
            pos_rel = pos_rel.view(num_centroids, self.group_size, self.spatial_dim)
            return x_patch, pos_centroid, batch_centroid, pos_rel
        return x_patch, pos_centroid, batch_centroid


class PointMambaEncoder(nn.Module):
    __doc__ = (
        r"""Implementation of the PointMamba encoder as described in the paper :arxiv: [PointMamba: A Simple State Space Model for Point Cloud Analysis](https://arxiv.org/abs/2402.10739).
        This implementation is adapted from the official repository :github: [LMD0311/PointMamba](https://github.com/LMD0311/PointMamba).
        """
        r"""
        This encoder consists of a patch-level embedding, 
        a position embedding, and a Mamba block.
        
        Args:
            in_channels: The number of input channels.
            embed_dim: The number of output channels.
            depth: The number of Mamba blocks.
            num_group: The number of patches to sample.
            group_size: The number of neighbors to consider for each patch.
            drop_path: The maximum stochastic-depth rate, linearly scaled across the Mamba blocks.
            use_cls_token: Whether to use a class token.
            spatial_dim: The dimension of the spatial features.
            patch_local_channels: Hidden widths of the patch embedder's per-point MLP.
            patch_global_channels: Hidden widths of the patch embedder's per-patch MLP.
            pos_embed_channels: Hidden widths of the positional-embedding MLP.
            act: The activation function to use.
            act_kwargs: The keyword arguments to pass to the activation function.
            act_first: Whether to apply the activation function before the normalization.
            norm: The normalization function to use.
            norm_kwargs: The keyword arguments to pass to the normalization function.
            bias: Whether to use bias in the MLPs.

        Shapes:
            - Input: $(N, C_{\text{in}})$ where $N$ is the number of nodes and $C_{\text{in}}$ is the number of input channels.
            - Output: $(B, P, C_{\text{out}})$ where $B$ is the number of batches, $P$ is the number of patches,
                and $C_{\text{out}}$ is the number of output channels.
        """
    )

    def __init__(
        self,
        in_channels: int,
        embed_dim: int,
        depth: int,
        num_group: int,
        group_size: int,
        drop_path: float = 0.0,
        use_cls_token: bool = False,
        spatial_dim: int = 3,
        patch_local_channels: Sequence[int] = (128, 256),
        patch_global_channels: Sequence[int] = (512,),
        pos_embed_channels: Sequence[int] = (128,),
        act: Union[str, Callable, None] = "relu",
        act_kwargs: Optional[Dict[str, Any]] = None,
        act_first: bool = False,
        norm: Union[str, Callable, None] = "batch_norm",
        norm_kwargs: Optional[Dict[str, Any]] = None,
        bias: Union[bool, List[bool]] = True,
    ):
        super().__init__()
        # Common parameters for all MLPs and associated blocks
        factory_kwargs: Dict[str, Any] = dict(
            act=act,
            act_kwargs=act_kwargs,
            act_first=act_first,
            norm=norm,
            norm_kwargs=norm_kwargs,
            bias=bias,
        )

        self.embed_dim = embed_dim
        self.patch_embed = PointPatchEmbedding(
            in_channels=in_channels,
            out_channels=embed_dim,
            num_group=num_group,
            group_size=group_size,
            spatial_dim=spatial_dim,
            local_channels=patch_local_channels,
            global_channels=patch_global_channels,
            **factory_kwargs,
        )
        self.pos_embed = MLP([spatial_dim, *pos_embed_channels, self.embed_dim], act="gelu", norm=None, bias=bias)
        self.order_h = Affine(self.embed_dim)
        self.order_th = Affine(self.embed_dim)

        dpr = [x.item() for x in torch.linspace(0, drop_path, depth)]
        self.blocks = nn.ModuleList([PointMambaBlock(self.embed_dim, drop_path=dpr[i]) for i in range(depth)])
        self.norm_f = nn.LayerNorm(self.embed_dim)

        if use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
            self.pos_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        else:
            self.register_parameter("cls_token", None)
            self.register_parameter("pos_token", None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        pass

    @overload
    def forward(
        self,
        x: OptTensor,
        pos: Tensor,
        batch: Tensor,
        return_intermediates: Literal[True],
    ) -> Tuple[Tensor, List[Tensor]]: ...

    @overload
    def forward(
        self,
        x: OptTensor,
        pos: Tensor,
        batch: Tensor,
        return_intermediates: Literal[False] = False,
    ) -> Tensor: ...

    def forward(
        self,
        x: OptTensor,
        pos: Tensor,
        batch: Tensor,
        return_intermediates: bool = False,
    ) -> Any:
        patch_feat, patch_pos, patch_batch = self.patch_embed(x, pos, batch)
        pos_embed = self.pos_embed(patch_pos)
        B = int(patch_batch.max() + 1)

        # Sort by Hilbert curve
        patch_pos_grid = local_grid(patch_pos, batch=patch_batch, size=0.02)
        idx_h = order_sort(patch_pos_grid, patch_batch, order="hilbert")
        feat_h = patch_feat[idx_h]
        pos_h = pos_embed[idx_h]
        # Sort by Trans-Hilbert curve
        idx_th = order_sort(patch_pos_grid, patch_batch, order="hilbert-trans")
        feat_th = patch_feat[idx_th]
        pos_th = pos_embed[idx_th]

        # Densify to (B, N, C) because Mamba blocks expect dense inputs.
        # NOTE: The patch encoder samples a fixed number of patches / nodes per batch,
        # so we can safely view / reshape sparse tensors to dense tensors.
        feat_h_dense = feat_h.view(B, -1, self.embed_dim)
        pos_h_dense = pos_h.view(B, -1, self.embed_dim)
        feat_th_dense = feat_th.view(B, -1, self.embed_dim)
        pos_th_dense = pos_th.view(B, -1, self.embed_dim)

        # Apply order scales (affine transformations)
        feat_h_dense = self.order_h(feat_h_dense)
        feat_th_dense = self.order_th(feat_th_dense)

        # Construct input sequence for mamba blocks (order matters)
        if self.cls_token is not None and self.pos_token is not None:
            cls_token = self.cls_token.expand(B, -1, -1)
            pos_token = self.pos_token.expand(B, -1, -1)
            pos_seq = torch.cat([pos_h_dense, pos_th_dense, pos_token], dim=1)
            x_seq = torch.cat([feat_h_dense, feat_th_dense, cls_token], dim=1)
        else:
            pos_seq = torch.cat([pos_h_dense, pos_th_dense], dim=1)
            x_seq = torch.cat([feat_h_dense, feat_th_dense], dim=1)

        # Mamba backbone
        intermediates: List[Tensor] = []

        x_seq = x_seq + pos_seq
        for block in self.blocks:
            if return_intermediates:
                intermediates.append(x_seq)
            x_seq = block(x_seq)

        x_seq = self.norm_f(x_seq)

        if return_intermediates:
            return x_seq, intermediates[::-1]
        return x_seq


class PointMambaEncoderMAE(nn.Module):
    r"""PointMamba encoder for masked-autoencoder pre-training.

    Embeds patches, serializes them along a randomly chosen Hilbert or Trans-Hilbert curve, masks a
    fraction of the patch tokens, and processes only the visible tokens with the Mamba blocks. The
    forward pass returns a dict with the visible-token features, the densified patch positions and
    reconstruction targets, and the visible / masked token indices.

    Args:
        in_channels: The number of input channels.
        embed_dim: The number of token channels.
        depth: The number of Mamba blocks.
        num_group: The number of patches to sample.
        group_size: The number of neighbors to consider for each patch.
        mask_ratio: The fraction of patch tokens to mask.
        drop_path: The maximum stochastic-depth rate, linearly scaled across the Mamba blocks.
        spatial_dim: The dimension of the spatial features.
        patch_local_channels: Hidden widths of the patch embedder's per-point MLP.
        patch_global_channels: Hidden widths of the patch embedder's per-patch MLP.
        pos_embed_channels: Hidden widths of the positional-embedding MLP.
        act: The activation function to use.
        act_kwargs: The keyword arguments to pass to the activation function.
        act_first: Whether to apply the activation function before the normalization.
        norm: The normalization function to use.
        norm_kwargs: The keyword arguments to pass to the normalization function.
        bias: Whether to use bias in the MLPs.

    Shape:
        - Input: $(N, C_\text{in})$ features (or `None`), $(N, 3)$ coordinates, and a $(N,)$ batch index.
        - Output: a dict with `x_vis` $(B, V, C)$, `pos_dense` $(B, P, 3)$, `target_pos_dense`
          $(B, P, M, 3)$, `mask_idx` $(B, P - V)$, `vis_idx` $(B, V)$, and the number of patches $P$.
    """

    def __init__(
        self,
        in_channels: int,
        embed_dim: int,
        depth: int,
        num_group: int,
        group_size: int,
        mask_ratio: float,
        drop_path: float = 0.0,
        spatial_dim: int = 3,
        patch_local_channels: Sequence[int] = (128, 256),
        patch_global_channels: Sequence[int] = (512,),
        pos_embed_channels: Sequence[int] = (128,),
        act: Union[str, Callable, None] = "relu",
        act_kwargs: Optional[Dict[str, Any]] = None,
        act_first: bool = False,
        norm: Union[str, Callable, None] = "batch_norm",
        norm_kwargs: Optional[Dict[str, Any]] = None,
        bias: Union[bool, List[bool]] = True,
    ):
        super().__init__()
        factory_kwargs: Dict[str, Any] = dict(
            act=act,
            act_kwargs=act_kwargs,
            act_first=act_first,
            norm=norm,
            norm_kwargs=norm_kwargs,
            bias=bias,
        )

        self.embed_dim = embed_dim
        self.mask_ratio = mask_ratio
        self.patch_embed = PointPatchEmbedding(
            in_channels=in_channels,
            out_channels=embed_dim,
            num_group=num_group,
            group_size=group_size,
            spatial_dim=spatial_dim,
            local_channels=patch_local_channels,
            global_channels=patch_global_channels,
            **factory_kwargs,
        )

        # Encoder-specific positional embedding
        self.pos_embed = MLP([spatial_dim, *pos_embed_channels, self.embed_dim], act="gelu", norm=None, bias=bias)

        # Order scale indicators
        self.order_h = Affine(self.embed_dim)
        self.order_th = Affine(self.embed_dim)

        dpr = [x.item() for x in torch.linspace(0, drop_path, depth)]
        self.blocks = nn.ModuleList([PointMambaBlock(self.embed_dim, drop_path=dpr[i]) for i in range(depth)])
        self.norm_f = nn.LayerNorm(self.embed_dim)

    def forward(self, x: OptTensor, pos: Tensor, batch: Tensor) -> Dict[str, Any]:
        patch_feat, patch_pos, patch_batch, target_pos_rel = self.patch_embed(x, pos, batch, return_pos_rel=True)
        B = int(patch_batch.max() + 1)
        patch_pos_grid = local_grid(patch_pos, batch=patch_batch, size=0.02)

        # Random serialization
        use_trans = torch.rand(1, device=pos.device) < 0.5
        order_key: SerializationOrder = "hilbert-trans" if use_trans else "hilbert"
        order_scale = self.order_th if use_trans else self.order_h
        idx_sort = order_sort(patch_pos_grid, patch_batch, order=order_key)

        # Densify to (B, P, ...) for the Mamba blocks
        feat_dense = patch_feat[idx_sort].view(B, -1, self.embed_dim)
        pos_dense = patch_pos[idx_sort].view(B, -1, 3)
        target_pos_dense = target_pos_rel[idx_sort].view(B, -1, self.patch_embed.group_size, 3)

        # Apply associated random order scale (affine transformation)
        feat_dense = order_scale(feat_dense)

        # Mask random tokens
        P = feat_dense.size(1)
        seq_len = P - int(self.mask_ratio * P)
        noise = torch.rand(B, P, device=feat_dense.device)
        noise_idx = torch.argsort(noise, dim=1)
        vis_idx = noise_idx[:, :seq_len]
        vis_idx, _ = torch.sort(vis_idx, dim=1)
        mask_idx = noise_idx[:, seq_len:]
        mask_idx, _ = torch.sort(mask_idx, dim=1)

        # Gather visible tokens
        x_vis = torch.gather(feat_dense, 1, vis_idx.unsqueeze(-1).expand(-1, -1, self.embed_dim))

        # Get position embeddings for visible tokens
        pos_emb_all = self.pos_embed(patch_pos)
        pos_emb_sorted = pos_emb_all[idx_sort].view(B, -1, self.embed_dim)
        pos_emb_vis = torch.gather(pos_emb_sorted, 1, vis_idx.unsqueeze(-1).expand(-1, -1, self.embed_dim))

        # Mamba blocks (process only visible tokens)
        x_vis = x_vis + pos_emb_vis
        for block in self.blocks:
            x_vis = block(x_vis)

        x_vis = self.norm_f(x_vis)

        return {
            "x_vis": x_vis,
            "pos_dense": pos_dense,
            "target_pos_dense": target_pos_dense,
            "mask_idx": mask_idx,
            "vis_idx": vis_idx,
            "P": P,
        }


class PointMambaDecoderMAE(nn.Module):
    r"""PointMamba decoder for masked-autoencoder pre-training.

    Scatters the visible-token features into a full-length sequence of learnable mask tokens, adds
    the positional embedding of every patch center, and refines the sequence with a stack of Mamba
    blocks.

    Args:
        embed_dim: The number of token channels.
        depth: The number of Mamba blocks.
        drop_path: The maximum stochastic-depth rate, linearly scaled across the Mamba blocks.
        spatial_dim: The dimension of the spatial features.
        pos_embed_channels: Hidden widths of the positional-embedding MLP.
        bias: Whether to use bias in the MLPs.

    Shape:
        - Input: `x_vis` $(B, V, C)$, `pos_dense` $(B, P, 3)$, and `ids_keep` $(B, V)$.
        - Output: $(B, P, C)$.
    """

    def __init__(
        self,
        embed_dim: int,
        depth: int,
        drop_path: float,
        spatial_dim: int = 3,
        pos_embed_channels: Sequence[int] = (128,),
        bias: Union[bool, List[bool]] = True,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.pos_embed = MLP([spatial_dim, *pos_embed_channels, embed_dim], act="gelu", norm=None, bias=bias)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        dpr = [x.item() for x in torch.linspace(0, drop_path, depth)]
        self.blocks = nn.ModuleList([PointMambaBlock(embed_dim, drop_path=dpr[i]) for i in range(depth)])
        self.norm_f = nn.LayerNorm(embed_dim)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.trunc_normal_(self.mask_token, std=0.02)

    def forward(
        self,
        x_vis: Tensor,
        pos_dense: Tensor,
        ids_keep: Tensor,
    ) -> Tensor:
        B, P, *_ = pos_dense.shape

        pos_emb_all = self.pos_embed(pos_dense)

        x_full = self.mask_token.expand(B, P, -1).clone()
        scatter_indices = ids_keep.unsqueeze(-1).expand(-1, -1, self.embed_dim)
        x_full.scatter_(dim=1, index=scatter_indices, src=x_vis)

        x_full = x_full + pos_emb_all
        for block in self.blocks:
            x_full = block(x_full)

        x_full = self.norm_f(x_full)
        return x_full


class PointMambaClassification(ClassificationModel):
    __doc__ = (
        r"""Implementation of the PointMamba encoder as described in the paper :arxiv: [PointMamba: A Simple State Space Model for Point Cloud Analysis](https://arxiv.org/abs/2402.10739).
        This implementation is adapted from the official repository :github: [LMD0311/PointMamba](https://github.com/LMD0311/PointMamba).
        """
        r"""
        This classification model consists of a PointMamba encoder and a MLP classification head.

        Args:
            in_channels: The number of input channels.
            num_classes: The number of output classes.
            embed_dim: The number of output channels.
            depth: The number of Mamba blocks.
            num_group: The number of patches to sample.
            group_size: The number of neighbors to consider for each patch.
            drop_path: The maximum stochastic-depth rate, linearly scaled across the Mamba blocks.
            use_cls_token: Whether to use a class token.
            spatial_dim: The dimension of the spatial features.
            patch_local_channels: Hidden widths of the patch embedder's per-point MLP.
            patch_global_channels: Hidden widths of the patch embedder's per-patch MLP.
            pos_embed_channels: Hidden widths of the positional-embedding MLP.
            act: The activation function to use.
            act_kwargs: The keyword arguments to pass to the activation function.
            act_first: Whether to apply the activation function before the normalization.
            norm: The normalization function to use.
            norm_kwargs: The keyword arguments to pass to the normalization function.
            bias: Whether to use bias in the MLPs.
            dropout: The dropout rate applied to the pooled global feature before the head. The head
                itself uses a fixed $0.5$ dropout, matching the reference recipe.
            global_pool: The pooling function to use.
            head_channels: The number of channels in the head.

        Shapes:
            - Input: $(N, C_{\text{in}})$ where $N$ is the number of nodes and $C_{\text{in}}$ is the number of input channels.
            - Output: $(B, C)$ where $B$ is the number of batches and $C$ is the number of number of classes.
        """
    )

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        *,
        embed_dim: int = 384,
        depth: int = 12,
        num_group: int = 64,
        group_size: int = 32,
        drop_path: float = 0.1,
        use_cls_token: bool = False,
        spatial_dim: int = 3,
        patch_local_channels: Sequence[int] = (128, 256),
        patch_global_channels: Sequence[int] = (512,),
        pos_embed_channels: Sequence[int] = (128,),
        act: Union[str, Callable, None] = "relu",
        act_kwargs: Optional[Dict[str, Any]] = None,
        act_first: bool = False,
        norm: Union[str, Callable, None] = "batch_norm",
        norm_kwargs: Optional[Dict[str, Any]] = None,
        bias: Union[bool, List[bool]] = True,
        dropout: float = 0.5,
        global_pool: AdaptivePoolLike = "mean",
        head_channels: Optional[Union[int, Sequence[int]]] = None,
    ):
        super().__init__(in_channels=in_channels, num_classes=num_classes)
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_group = num_group
        self.group_size = group_size
        self.drop_path = drop_path
        self.use_cls_token = use_cls_token
        self.spatial_dim = spatial_dim
        self.patch_local_channels = patch_local_channels
        self.patch_global_channels = patch_global_channels
        self.pos_embed_channels = pos_embed_channels
        self.act = act
        self.act_kwargs = act_kwargs
        self.act_first = act_first
        # The "layer_norm" string resolves to PyG's graph-mode LayerNorm, which normalizes across the
        # whole batch; the `nn.LayerNorm` class applies the intended per-feature normalization.
        self.norm = nn.LayerNorm if norm == "layer_norm" else norm
        self.norm_kwargs = norm_kwargs
        self.bias = bias
        self.dropout = dropout
        self.head_channels = ensure_list(head_channels, none_as_empty=True)

        self.encoder = self.configure_encoder()
        self.global_pool = create_adaptive_pool(global_pool)
        self.head = self.configure_head()
        self.reset_parameters()

    def configure_encoder(self) -> PointMambaEncoder:
        """Build the `PointMambaEncoder` backbone."""
        return PointMambaEncoder(
            in_channels=self.in_channels,
            embed_dim=self.embed_dim,
            depth=self.depth,
            num_group=self.num_group,
            group_size=self.group_size,
            drop_path=self.drop_path,
            use_cls_token=self.use_cls_token,
            spatial_dim=self.spatial_dim,
            patch_local_channels=self.patch_local_channels,
            patch_global_channels=self.patch_global_channels,
            pos_embed_channels=self.pos_embed_channels,
            act=self.act,
            act_kwargs=self.act_kwargs,
            act_first=self.act_first,
            norm=self.norm,
            norm_kwargs=self.norm_kwargs,
            bias=self.bias,
        )

    def configure_head(self) -> nn.Module:
        if self.num_classes == 0:
            return nn.Identity()
        embed_dim = self.encoder.embed_dim
        if self.use_cls_token:
            embed_dim *= 2

        return MLP(
            [embed_dim, *self.head_channels, self.num_classes],
            dropout=0.5,  # the reference recipe pins the head dropout at 0.5
            act=self.act,
            act_kwargs=self.act_kwargs,
            act_first=self.act_first,
            norm=self.norm,
            norm_kwargs=self.norm_kwargs,
            bias=self.bias,
        )

    def reset_parameters(self) -> None:
        pass

    def reset_classifier(
        self,
        num_classes: int,
        global_pool: Optional[AdaptivePoolLike] = None,
        head_channels: Optional[Union[int, Sequence[int]]] = None,
        dropout: Optional[float] = None,
    ) -> None:
        self.num_classes = num_classes
        if global_pool is not None:
            self.global_pool = create_adaptive_pool(global_pool)
        if head_channels is not None:
            self.head_channels = ensure_list(head_channels)
        if dropout is not None:
            self.dropout = dropout
        self.head = self.configure_head()

    @overload
    def forward_features(
        self,
        x: OptTensor,
        pos: Tensor,
        batch: Tensor,
        return_intermediates: Literal[True],
    ) -> Tuple[Tensor, List[Tensor]]: ...

    @overload
    def forward_features(
        self,
        x: OptTensor,
        pos: Tensor,
        batch: Tensor,
        return_intermediates: Literal[False] = False,
    ) -> Tensor: ...

    def forward_features(self, x: OptTensor, pos: Tensor, batch: Tensor, return_intermediates: bool = False) -> Any:
        return self.encoder(x, pos, batch, return_intermediates=return_intermediates)

    def forward_head(self, x: Tensor, pre_logits: bool = False) -> Tensor:
        # Pool the patched tokens of the mamba blocks.
        # If using the cls token, concatenate the cls token to the pooled tokens.
        # NOTE: For the pooling operation, we need to transpose the pooled dimension last,
        # because all torch.nn.AdaptivePool1d operations expect the pooled dimension(s) to be last.
        if self.use_cls_token:
            cls_token, x_tokens = x[:, -1, :], x[:, :-1, :]
            x_pool = self.global_pool(x_tokens.transpose(1, 2)).squeeze(-1)
            x_global = torch.cat([cls_token, x_pool], dim=1)
        else:
            x_global = self.global_pool(x.transpose(1, 2)).squeeze(-1)

        if self.dropout > 0:
            x_global = F.dropout(x_global, p=self.dropout, training=self.training)

        return x_global if pre_logits else self.head(x_global)

    def forward(self, x: OptTensor, pos: Tensor, batch: Tensor) -> Tensor:
        x = self.forward_features(x, pos, batch)
        return self.forward_head(x)


class PointMambaMAE(BaseModel):
    r"""PointMamba masked-autoencoder pre-training model, as described in the paper :arxiv: [PointMamba: A Simple State Space Model for Point Cloud Analysis](https://arxiv.org/abs/2402.10739).
    This implementation is adapted from the official repository :github: [LMD0311/PointMamba](https://github.com/LMD0311/PointMamba).

    Masks a fraction of the serialized patch tokens, encodes the visible tokens with Mamba blocks,
    reconstructs the masked patches' relative coordinates with a Mamba decoder and a linear head, and
    returns the predicted and target patch coordinates for a set-to-set reconstruction objective such
    as `chamfer_distance` from `torch_pointcloud.losses`.

    Args:
        in_channels: The number of input channels.
        embed_dim: The number of token channels.
        encoder_depth: The number of encoder Mamba blocks.
        decoder_depth: The number of decoder Mamba blocks.
        num_group: The number of patches to sample.
        group_size: The number of neighbors to consider for each patch.
        mask_ratio: The fraction of patch tokens to mask.
        drop_path: The maximum stochastic-depth rate, linearly scaled across the Mamba blocks.
        spatial_dim: The dimension of the spatial features.
        patch_local_channels: Hidden widths of the patch embedder's per-point MLP.
        patch_global_channels: Hidden widths of the patch embedder's per-patch MLP.
        pos_embed_channels: Hidden widths of the positional-embedding MLPs.
        act: The activation function to use.
        act_kwargs: The keyword arguments to pass to the activation function.
        act_first: Whether to apply the activation function before the normalization.
        norm: The normalization function to use.
        norm_kwargs: The keyword arguments to pass to the normalization function.
        bias: Whether to use bias in the MLPs.

    Shape:
        - Input: $(N, C_\text{in})$ features (or `None`), $(N, 3)$ coordinates, and a $(N,)$ batch index.
        - Output: predicted and target patches, each of shape $(B \cdot M_\text{mask}, \text{group\_size}, 3)$.
    """

    def __init__(
        self,
        in_channels: int,
        *,
        embed_dim: int = 384,
        encoder_depth: int = 12,
        decoder_depth: int = 4,
        num_group: int = 64,
        group_size: int = 32,
        mask_ratio: float = 0.6,
        drop_path: float = 0.1,
        spatial_dim: int = 3,
        patch_local_channels: Sequence[int] = (128, 256),
        patch_global_channels: Sequence[int] = (512,),
        pos_embed_channels: Sequence[int] = (128,),
        act: Union[str, Callable, None] = "relu",
        act_kwargs: Optional[Dict[str, Any]] = None,
        act_first: bool = False,
        norm: Union[str, Callable, None] = "batch_norm",
        norm_kwargs: Optional[Dict[str, Any]] = None,
        bias: Union[bool, List[bool]] = True,
    ):
        super().__init__(in_channels=in_channels)
        self.embed_dim = embed_dim
        self.encoder_depth = encoder_depth
        self.decoder_depth = decoder_depth
        self.num_group = num_group
        self.group_size = group_size
        self.mask_ratio = mask_ratio
        self.drop_path = drop_path
        self.spatial_dim = spatial_dim
        self.patch_local_channels = patch_local_channels
        self.patch_global_channels = patch_global_channels
        self.pos_embed_channels = pos_embed_channels
        self.act = act
        self.act_kwargs = act_kwargs
        self.act_first = act_first
        self.norm = norm
        self.norm_kwargs = norm_kwargs
        self.bias = bias

        self.encoder = self.configure_encoder()
        self.decoder = self.configure_decoder()
        self.head = self.configure_head()
        self.reset_parameters()

    def reset_parameters(self) -> None:
        pass

    def configure_encoder(self) -> PointMambaEncoderMAE:
        """Build the `PointMambaEncoderMAE` backbone, which encodes only the visible patches."""
        return PointMambaEncoderMAE(
            in_channels=self.in_channels,
            embed_dim=self.embed_dim,
            depth=self.encoder_depth,
            num_group=self.num_group,
            group_size=self.group_size,
            mask_ratio=self.mask_ratio,
            drop_path=self.drop_path,
            spatial_dim=self.spatial_dim,
            patch_local_channels=self.patch_local_channels,
            patch_global_channels=self.patch_global_channels,
            pos_embed_channels=self.pos_embed_channels,
            act=self.act,
            act_kwargs=self.act_kwargs,
            act_first=self.act_first,
            norm=self.norm,
            norm_kwargs=self.norm_kwargs,
            bias=self.bias,
        )

    def configure_decoder(self) -> PointMambaDecoderMAE:
        """Build the `PointMambaDecoderMAE` that reconstructs the masked patches."""
        return PointMambaDecoderMAE(
            embed_dim=self.embed_dim,
            depth=self.decoder_depth,
            drop_path=self.drop_path,
            spatial_dim=self.spatial_dim,
            pos_embed_channels=self.pos_embed_channels,
            bias=self.bias,
        )

    def configure_head(self) -> nn.Module:
        return nn.Linear(self.embed_dim, 3 * self.group_size)

    def forward(self, x: OptTensor, pos: Tensor, batch: Tensor) -> Tuple[Tensor, Tensor]:
        out = self.encoder(x, pos, batch)
        x_vis, pos_dense, target_pos_dense, mask_idx, vis_idx = (
            out["x_vis"],
            out["pos_dense"],
            out["target_pos_dense"],
            out["mask_idx"],
            out["vis_idx"],
        )

        x_rec_dense = self.decoder(x_vis, pos_dense, vis_idx)

        gather_ids_feat = mask_idx.unsqueeze(-1).expand(-1, -1, self.embed_dim)
        x_rec_masked = torch.gather(x_rec_dense, 1, gather_ids_feat)

        pred_masked = self.head(x_rec_masked)

        pred_masked = pred_masked.view(x_rec_masked.shape[0], x_rec_masked.shape[1], self.group_size, 3)
        gather_ids_target = mask_idx.view(x_rec_masked.shape[0], -1, 1, 1).expand(-1, -1, self.group_size, 3)
        target_masked = torch.gather(target_pos_dense, 1, gather_ids_target)

        pred = pred_masked.view(-1, self.group_size, self.spatial_dim)
        target = target_masked.view(-1, self.group_size, self.spatial_dim)
        return pred, target


@register_model(
    "point-mamba-base.modelnet40.dingkang-liang",
    task="classification",
    weights=WeightsDict(
        url="hf://torch-pointcloud/point-mamba/point-mamba-base.modelnet40.dingkang-liang.safetensors",
        dataset="modelnet40",
        metrics={"OA": 93.64},
        classes=MODELNET40_CLASSES,
        author="dingkang-liang",
        license="Apache-2.0",
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
        embed_dim=384,
        depth=12,
        num_group=64,
        group_size=32,
        drop_path=0.1,
        use_cls_token=False,
        spatial_dim=3,
        patch_local_channels=(128, 256),
        patch_global_channels=(512,),
        pos_embed_channels=(128,),
        act="relu",
        act_kwargs=None,
        act_first=False,
        norm="batch_norm",
        norm_kwargs=None,
        bias=True,
        dropout=0.5,
        global_pool="mean",
        head_channels=(256, 256),
    ),
)
def point_mamba_base_modelnet40_clf(**kwargs: Any) -> PointMambaClassification:
    return PointMambaClassification(**kwargs)


@register_model(
    "point-mamba-base.scanobjectnn.dingkang-liang",
    task="classification",
    weights=WeightsDict(
        url="hf://torch-pointcloud/point-mamba/point-mamba-base.scanobjectnn.dingkang-liang.safetensors",
        dataset="scanobjectnn",
        classes=SCANOBJECTNN_CLASSES,
        author="dingkang-liang",
        license="Apache-2.0",
    ),
    transform=T.Compose(
        [
            T.RemoveNearOrigin(pos_key=DataKeys.POS, radius=RADIUS),
            T.FarthestPointSample(pos_key=DataKeys.POS, num_samples=2048, random_start=False),
        ]
    ),
    hparams=dict(
        in_channels=0,
        num_classes=15,
        embed_dim=384,
        depth=12,
        num_group=128,
        group_size=32,
        drop_path=0.5,
        use_cls_token=False,
        spatial_dim=3,
        patch_local_channels=(128, 256),
        patch_global_channels=(512,),
        pos_embed_channels=(128,),
        act="relu",
        act_kwargs=None,
        act_first=False,
        norm="batch_norm",
        norm_kwargs=None,
        bias=True,
        dropout=0.1,
        global_pool="mean",
        head_channels=(256, 256),
    ),
)
def point_mamba_base_scanobjectnn_clf(**kwargs: Any) -> PointMambaClassification:
    return PointMambaClassification(**kwargs)


@register_model(
    "point-mamba-base.scanobjectnn-nobg.dingkang-liang",
    task="classification",
    weights=WeightsDict(
        url="hf://torch-pointcloud/point-mamba/point-mamba-base.scanobjectnn-nobg.dingkang-liang.safetensors",
        dataset="scanobjectnn-nobg",
        classes=SCANOBJECTNN_CLASSES,
        author="dingkang-liang",
        license="Apache-2.0",
    ),
    transform=T.Compose(
        [
            T.RemoveNearOrigin(pos_key=DataKeys.POS, radius=RADIUS),
            T.FarthestPointSample(pos_key=DataKeys.POS, num_samples=2048, random_start=False),
        ]
    ),
    hparams=dict(
        in_channels=0,
        num_classes=15,
        embed_dim=384,
        depth=12,
        num_group=128,
        group_size=32,
        drop_path=0.5,
        use_cls_token=False,
        spatial_dim=3,
        patch_local_channels=(128, 256),
        patch_global_channels=(512,),
        pos_embed_channels=(128,),
        act="relu",
        act_kwargs=None,
        act_first=False,
        norm="batch_norm",
        norm_kwargs=None,
        bias=True,
        dropout=0.1,
        global_pool="mean",
        head_channels=(256, 256),
    ),
)
def point_mamba_base_scanobjectnn_nobg_clf(**kwargs: Any) -> PointMambaClassification:
    return PointMambaClassification(**kwargs)


@register_model(
    "point-mamba-base.scanobjectnn-augmentedrot-scale75.dingkang-liang",
    task="classification",
    weights=WeightsDict(
        url="hf://torch-pointcloud/point-mamba/point-mamba-base.scanobjectnn-augmentedrot-scale75.dingkang-liang.safetensors",
        dataset="scanobjectnn-augmentedrot-scale75",
        metrics={"OA": 89.28},
        classes=SCANOBJECTNN_CLASSES,
        author="dingkang-liang",
        license="Apache-2.0",
    ),
    transform=T.Compose(
        [
            T.RemoveNearOrigin(pos_key=DataKeys.POS, radius=RADIUS),
            T.FarthestPointSample(pos_key=DataKeys.POS, num_samples=2048, random_start=False),
        ]
    ),
    hparams=dict(
        in_channels=0,
        num_classes=15,
        embed_dim=384,
        depth=12,
        num_group=128,
        group_size=32,
        drop_path=0.5,
        use_cls_token=False,
        spatial_dim=3,
        patch_local_channels=(128, 256),
        patch_global_channels=(512,),
        pos_embed_channels=(128,),
        act="relu",
        act_kwargs=None,
        act_first=False,
        norm="batch_norm",
        norm_kwargs=None,
        bias=True,
        dropout=0.1,
        global_pool="mean",
        head_channels=(256, 256),
    ),
)
def point_mamba_base_scanobjectnn_augmentedrot_scale75_clf(**kwargs: Any) -> PointMambaClassification:
    return PointMambaClassification(**kwargs)


@register_model(
    "point-mamba-base.pretrain.dingkang-liang",
    task="base",
    weights=WeightsDict(
        url="hf://torch-pointcloud/point-mamba/point-mamba-base.pretrain.dingkang-liang.safetensors",
        dataset="shapenet55",
        author="dingkang-liang",
        license="Apache-2.0",
    ),
    hparams=dict(
        in_channels=0,
        embed_dim=384,
        encoder_depth=12,
        decoder_depth=4,
        num_group=64,
        group_size=32,
        mask_ratio=0.6,
        drop_path=0.1,
        spatial_dim=3,
        patch_local_channels=(128, 256),
        patch_global_channels=(512,),
        pos_embed_channels=(128,),
        act="relu",
        act_kwargs=None,
        act_first=False,
        norm="batch_norm",
        norm_kwargs=None,
        bias=True,
    ),
)
def point_mamba_base_pretrain(**kwargs: Any) -> PointMambaMAE:
    return PointMambaMAE(**kwargs)
