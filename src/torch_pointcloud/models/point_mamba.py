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

from torch_pointcloud.layers.affine import Affine
from torch_pointcloud.layers.pools import AdaptivePoolLike, PoolLike, create_adaptive_pool
from torch_pointcloud.utils.cluster import fps, local_grid
from torch_pointcloud.utils.conversion import ensure_list
from torch_pointcloud.utils.imports import optional_import
from torch_pointcloud.utils.serialization import SerializationOrder, serialize_coords
from torch_pointcloud.utils.types import OptTensor

from ._base import ClassificationModel
from ._registry import register_model

if TYPE_CHECKING:
    from mamba_ssm import Mamba
    from torch_cluster import knn
    from torch_scatter import scatter

Mamba, _ = optional_import("mamba_ssm", "Mamba")
knn, _ = optional_import("torch_cluster", "knn")
scatter, _ = optional_import("torch_scatter", "scatter")


PAPER_TITLE = "PointMamba: A Simple State Space Model for Point Cloud Analysis"
PAPER_URL = "https://arxiv.org/abs/2402.10739"
PAPER_AUTHORS = "Dingkang Liang, Xin Zhou, Wei Xu, Xingkui Zhu, Zhikang Zou, Xiaoqing Ye, Xiao Tan, Xiang Bai"
PAPER_CITATION = f"[{PAPER_TITLE}]({PAPER_URL}) by {PAPER_AUTHORS}"
OFFICIAL_REPO_URL = "https://github.com/LMD0311/PointMamba"


def order_sort(pos_grid: Tensor, batch: Tensor, order: SerializationOrder) -> Tensor:
    depth = int(pos_grid.max()).bit_length()
    serialized_code = serialize_coords(pos_grid, batch, depth=depth, order=order)
    serialized_order = torch.argsort(serialized_code)
    return serialized_order


class PointMambaBlock(nn.Module):
    __doc__ = (
        rf"""Implementation of the PointMamba block as described in the paper :arxiv: {PAPER_CITATION}.
        This implementation is adapted from the official repository :github: {OFFICIAL_REPO_URL}.
        """
        r"""
        The `PointMambaBlock` is a residual block that consists of a normalization layer, 
        a Mamba block, and a dropout layer:
        
        ```txt
        x -> Norm -> Mamba -> Dropout -> y
        |                                ^
        +--------------------------------+
        ```
        
        Important:
            This module requires the `mamba_ssm` package to be installed.
            
        Args:
            channels: The number of input and output channels.
            d_state: The number of state channels.
            d_conv: The number of convolution channels.
            expand: The expansion factor for the hidden channels.
            dropout: The dropout rate to use. If `None`, no dropout is applied.
            
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
        dropout: float = 0.0,
    ):
        super().__init__()
        self.norm = nn.LayerNorm(channels)
        self.mamba = Mamba(d_model=channels, d_state=d_state, d_conv=d_conv, expand=expand, use_fast_path=True)
        self.dropout = dropout

    def forward(self, x: Tensor) -> Tensor:
        residual = x
        x = self.norm(x)
        x = self.mamba(x)
        if self.dropout is not None:
            x = F.dropout(x, p=self.dropout, training=self.training)
        return x + residual


class PointPatchEmbedding(nn.Module):
    __doc__ = (
        rf"""
        Defines the patch-level encoding based on the paper :arxiv: {PAPER_CITATION}.
        This implementation is adapted from the official repository :github: {OFFICIAL_REPO_URL}.
        """
        r"""
        This module encodes features and positions to a patch-level embedding, 
        where patches are sampled using the farthest point sampling (FPS) method.
        Features and positions are encoded using two MLPs, 
        and the patch-level embedding is obtained by pooling the features and positions.
        
        Args:
            in_channels: The number of input channels.
            out_channels: The number of output channels.
            num_patches: The number of patches to sample.
            group_size: The number of neighbors to consider for each patch.
            act: The activation function to use.
            act_kwargs: The keyword arguments to pass to the activation function.
            act_first: Whether to apply the activation function before the normalization.
            norm: The normalization function to use.
            norm_kwargs: The keyword arguments to pass to the normalization function.
            bias: Whether to use bias in the MLPs.
            aggr: The pooling function to use.
        
        Shapes:
            - Input: $(N, C_{\text{in}})$ where $N$ is the number of nodes and $C_{\text{in}}$ is the number of input channels.
            - Output: $(P, C_{\text{out}})$ where $P$ is the number of patches and $C_{\text{out}}$ is the number of output channels.
        """
    )

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_patches: int,
        group_size: int,
        spatial_dim: int = 3,
        act: Union[str, Callable, None] = "relu",
        act_kwargs: Optional[Dict[str, Any]] = None,
        act_first: bool = False,
        norm: Union[str, Callable, None] = "batch_norm",
        norm_kwargs: Optional[Dict[str, Any]] = None,
        bias: Union[bool, List[bool]] = True,
        reduce: PoolLike = "max",
    ):
        super().__init__()
        # Common parameters for all MLPs and associated blocks
        factory_kwargs = dict(
            act=act,
            act_kwargs=act_kwargs,
            act_first=act_first,
            norm=norm,
            norm_kwargs=norm_kwargs,
            bias=bias,
        )

        self.num_patches = num_patches
        self.group_size = group_size

        # In case we do not provide input features (i.e. `in_channels==0`), we use relative positions as features.
        # Otherwise, we concatenate the input features of the centroids (local features),
        # neighbors (global features) and relative positions.
        in_channels = spatial_dim if in_channels == 0 else 2 * in_channels + spatial_dim
        self.mlp1 = MLP([in_channels, 128, 256], **factory_kwargs)
        self.mlp2 = MLP([512, 512, out_channels], **factory_kwargs)
        self.reduce = reduce

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
        idx_centroid = fps(pos, batch, num_nodes=self.num_patches)
        pos_centroid = pos[idx_centroid]
        x_centroid = x[idx_centroid] if x is not None else torch.empty(0, device=pos.device)
        batch_centroid = batch[idx_centroid]
        num_centroids = pos_centroid.size(0)
        row, col = knn(pos, pos_centroid, self.group_size, batch_x=batch, batch_y=batch_centroid)

        pos_neighbor = pos[col]
        pos_rel = pos_neighbor - pos_centroid[row]
        x_neighbor = x[col] if x is not None else torch.empty(0, device=pos.device)

        x = torch.cat([x_centroid[row], x_neighbor, pos_rel], dim=-1) if x is not None else pos_rel
        x_local = self.mlp1(x)
        x_max = scatter(x_local, row, dim=0, dim_size=num_centroids, reduce=self.reduce)

        x_cat = torch.cat([x_max[row], x_local], dim=1)
        x_final = self.mlp2(x_cat)

        x_patch = scatter(x_final, row, dim=0, dim_size=num_centroids, reduce=self.reduce)

        if return_pos_rel:
            return x_patch, pos_centroid, batch_centroid, pos_rel
        return x_patch, pos_centroid, batch_centroid


class PointMambaEncoder(nn.Module):
    __doc__ = (
        rf"""Implementation of the PointMamba encoder as described in the paper :arxiv: {PAPER_CITATION}.
        This implementation is adapted from the official repository :github: {OFFICIAL_REPO_URL}.
        """
        r"""
        This encoder consists of a patch-level embedding, 
        a position embedding, and a Mamba block.
        
        Args:
            in_channels: The number of input channels.
            embedding_dim: The number of output channels.
            depth: The number of Mamba blocks.
            num_patches: The number of patches to sample.
            group_size: The number of neighbors to consider for each patch.
            drop_path_rate: The dropout rate to use.
            use_cls_token: Whether to use a class token.
            spatial_dim: The dimension of the spatial features.
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
        embedding_dim: int,
        depth: int,
        num_patches: int,
        group_size: int,
        drop_path_rate: float = 0.0,
        use_cls_token: bool = False,
        spatial_dim: int = 3,
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

        self.embedding_dim = embedding_dim
        self.patch_embed = PointPatchEmbedding(
            in_channels=in_channels,
            out_channels=embedding_dim,
            num_patches=num_patches,
            group_size=group_size,
            spatial_dim=spatial_dim,
            **factory_kwargs,
        )
        self.pos_embed = MLP([spatial_dim, 128, self.embedding_dim], act="gelu", norm=None, bias=bias)
        self.order_h = Affine(self.embedding_dim)
        self.order_th = Affine(self.embedding_dim)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([PointMambaBlock(self.embedding_dim, dropout=dpr[i]) for i in range(depth)])
        self.norm_f = nn.LayerNorm(self.embedding_dim)

        if use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embedding_dim))
            self.pos_token = nn.Parameter(torch.zeros(1, 1, self.embedding_dim))
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
        feat_h_dense = feat_h.view(B, -1, self.embedding_dim)
        pos_h_dense = pos_h.view(B, -1, self.embedding_dim)
        feat_th_dense = feat_th.view(B, -1, self.embedding_dim)
        pos_th_dense = pos_th.view(B, -1, self.embedding_dim)

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
    def __init__(
        self,
        in_channels: int,
        embedding_dim: int,
        depth: int,
        num_patches: int,
        group_size: int,
        mask_ratio: float,
        drop_path_rate: float = 0.0,
        use_cls_token: bool = False,
        spatial_dim: int = 3,
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

        self.embedding_dim = embedding_dim
        self.mask_ratio = mask_ratio
        self.patch_embed = PointPatchEmbedding(
            in_channels=in_channels,
            out_channels=embedding_dim,
            num_patches=num_patches,
            group_size=group_size,
            spatial_dim=spatial_dim,
            **factory_kwargs,
        )

        self.pos_embed = MLP([spatial_dim, 128, self.embedding_dim], act="gelu", norm=None, bias=bias)
        self.order_h = Affine(self.embedding_dim)
        self.order_th = Affine(self.embedding_dim)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([PointMambaBlock(self.embedding_dim, dropout=dpr[i]) for i in range(depth)])
        self.norm_f = nn.LayerNorm(self.embedding_dim)

    def forward(self, x: OptTensor, pos: Tensor, batch: Tensor) -> Tuple[Tensor, Tensor, Dict[str, Any]]:
        # 1. Patch Embedding & GT Extraction
        # We explicitly ask for pos_rel (targets)
        patch_feat, patch_pos, patch_batch, pos_rel_gt = self.patch_embed(x, pos, batch, return_pos_rel=True)

        B = int(patch_batch.max() + 1)
        patch_pos_grid = local_grid(patch_pos, batch=patch_batch, size=0.02)

        # 2. Random Serialization (MAE Specific)
        # Randomly choose between Hilbert and Trans-Hilbert
        use_trans = torch.rand(1, device=pos.device) < 0.5
        order_key: SerializationOrder = "hilbert-trans" if use_trans else "hilbert"
        order_scale = self.order_th if use_trans else self.order_h
        idx_sort = order_sort(patch_pos_grid, patch_batch, order=order_key)

        # Sort all data to dense tensors (B, P, ...)
        feat_dense = patch_feat[idx_sort].view(B, -1, self.embedding_dim)
        pos_dense = patch_pos[idx_sort].view(B, -1, 3)
        target_dense = pos_rel_gt[idx_sort].view(B, -1, self.patch_embed.group_size, 3)

        # Apply Affine Transformation (Order Scale) to features
        feat_dense = order_scale(feat_dense)

        # 3. Masking
        P = feat_dense.size(1)
        num_mask = int(self.mask_ratio * P)
        len_keep = P - num_mask

        # Generate random mask indices
        noise = torch.rand(B, P, device=feat_dense.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        ids_keep = ids_shuffle[:, :len_keep]

        # Gather visible tokens
        x_vis = torch.gather(feat_dense, 1, ids_keep.unsqueeze(-1).expand(-1, -1, self.embedding_dim))

        # Get position embeddings for visible tokens (using base pos_embed)
        # Embed raw positions -> sort -> gather visible
        pos_emb_all = self.pos_embed(patch_pos)
        pos_emb_sorted = pos_emb_all[idx_sort].view(B, -1, self.embedding_dim)
        pos_emb_vis = torch.gather(pos_emb_sorted, 1, ids_keep.unsqueeze(-1).expand(-1, -1, self.embedding_dim))

        # Add Pos Embed to visible features
        x_vis = x_vis + pos_emb_vis

        # 4. Mamba Blocks (Process only visible)
        for block in self.blocks:
            x_vis = block(x_vis)
        x_vis = self.norm_f(x_vis)

        # Pack metadata needed by decoder
        mask_info = {
            "ids_restore": ids_restore,
            "pos_dense": pos_dense,
            "len_keep": len_keep,
            "num_mask": num_mask,
            "P": P,
        }

        return x_vis, target_dense, mask_info


class PointMambaClassification(ClassificationModel):
    __doc__ = (
        rf"""Implementation of the PointMamba encoder as described in the paper :arxiv: {PAPER_CITATION}.
        This implementation is adapted from the official repository :github: {OFFICIAL_REPO_URL}.
        """
        r"""
        This classification model consists of a PointMamba encoder and a MLP classification head.

        Args:
            in_channels: The number of input channels.
            num_classes: The number of output classes.
            embedding_dim: The number of output channels.
            depth: The number of Mamba blocks.
            num_patches: The number of patches to sample.
            group_size: The number of neighbors to consider for each patch.
            drop_path_rate: The dropout rate to use.
            use_cls_token: Whether to use a class token.
            spatial_dim: The dimension of the spatial features.
            act: The activation function to use.
            act_kwargs: The keyword arguments to pass to the activation function.
            act_first: Whether to apply the activation function before the normalization.
            norm: The normalization function to use.
            norm_kwargs: The keyword arguments to pass to the normalization function.
            bias: Whether to use bias in the MLPs.
            dropout: The dropout rate to use.
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
        embedding_dim: int = 384,
        depth: int = 12,
        num_patches: int = 64,
        group_size: int = 32,
        drop_path_rate: float = 0.1,
        use_cls_token: bool = False,
        spatial_dim: int = 3,
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
        self.embedding_dim = embedding_dim
        self.depth = depth
        self.num_patches = num_patches
        self.group_size = group_size
        self.drop_path_rate = drop_path_rate
        self.use_cls_token = use_cls_token
        self.spatial_dim = spatial_dim
        self.act = act
        self.act_kwargs = act_kwargs
        self.act_first = act_first
        self.norm = norm
        self.norm_kwargs = norm_kwargs
        self.bias = bias
        self.dropout = dropout
        self.head_channels = ensure_list(head_channels, none_as_empty=True)

        self.encoder = self.configure_encoder()
        self.global_pool = create_adaptive_pool(global_pool)
        self.head = self.configure_head()
        self.reset_parameters()

    def configure_encoder(self) -> PointMambaEncoder:
        return PointMambaEncoder(
            in_channels=self.in_channels,
            embedding_dim=self.embedding_dim,
            depth=self.depth,
            num_patches=self.num_patches,
            group_size=self.group_size,
            drop_path_rate=self.drop_path_rate,
            use_cls_token=self.use_cls_token,
            spatial_dim=self.spatial_dim,
            act=self.act,
            act_kwargs=self.act_kwargs,
            act_first=self.act_first,
            norm=self.norm,
            norm_kwargs=self.norm_kwargs,
            bias=self.bias,
        )

    def configure_head(self) -> nn.Module:
        head_channels = ensure_list(self.head_channels, none_as_empty=True)
        embedding_dim = self.encoder.embedding_dim
        if self.use_cls_token:
            embedding_dim *= 2

        return MLP(
            [embedding_dim, *head_channels, self.num_classes],
            dropout=self.dropout,
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
        global_pool: AdaptivePoolLike = "mean",
        head_channels: Optional[Union[int, Sequence[int]]] = None,
        dropout: float = 0.5,
    ) -> None:
        self.num_classes = num_classes
        self.head_channels = ensure_list(head_channels, none_as_empty=True)
        self.dropout = dropout
        self.global_pool = create_adaptive_pool(global_pool)
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

    def forward(self, x: Tensor, pos: Tensor, batch: Tensor) -> Tensor:
        x = self.forward_features(x, pos, batch)
        return self.forward_head(x)


class PointMambaMAE(nn.Module):
    def __init__(
        self,
        in_channels: int,
        *,
        embedding_dim: int = 384,
        decoder_dim: int = 384,
        depth: int = 12,
        decoder_depth: int = 4,
        num_patches: int = 64,
        group_size: int = 32,
        mask_ratio: float = 0.6,
        drop_path_rate: float = 0.1,
        spatial_dim: int = 3,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.decoder_dim = decoder_dim
        self.mask_ratio = mask_ratio
        self.group_size = group_size
        self.spatial_dim = spatial_dim


@register_model("point-mamba.original.modelnet40", task="classification")
def point_mamba_modelnet40_clf(in_channels: int = 0, num_classes: int = 40, **kwargs: Any) -> PointMambaClassification:
    hparams: Dict[str, Any] = dict(
        in_channels=in_channels,
        num_classes=num_classes,
        embedding_dim=384,
        depth=12,
        num_patches=64,
        group_size=32,
        drop_path_rate=0.1,
        use_cls_token=False,
        spatial_dim=3,
        act="relu",
        act_kwargs=None,
        act_first=True,
        norm="batch_norm",
        norm_kwargs=None,
        bias=False,
        dropout=0.5,
        global_pool="mean",
        head_channels=(256, 256),
    )
    hparams.update(kwargs)
    return PointMambaClassification(**hparams)
