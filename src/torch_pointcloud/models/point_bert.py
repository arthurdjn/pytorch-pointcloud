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
from torch_pointcloud.layers import Conv2dBlock, PointPatchEmbed, TransformerBlock, create_act, create_norm
from torch_pointcloud.utils.cluster import group, knn
from torch_pointcloud.utils.conversion import ensure_list
from torch_pointcloud.utils.imports import optional_import
from torch_pointcloud.utils.types import OptTensor

from ._base import BaseModel, ClassificationModel
from ._registry import register_model

if TYPE_CHECKING:
    from torch_scatter import scatter

scatter, _ = optional_import("torch_scatter", "scatter")


class PointBERTEncoder(nn.Module):
    r"""Point-BERT transformer backbone.

    Implements the backbone of :arxiv: [Point-BERT: Pre-training 3D Point Cloud Transformers with
    Masked Point Modeling](https://arxiv.org/abs/2111.14819), adapted from
    :github: [lulutang0608/Point-BERT](https://github.com/lulutang0608/Point-BERT).

    The backbone groups the cloud into patches (FPS + KNN), embeds each patch into a token with a
    mini-PointNet (`PointTokenizer`), bridges to the transformer dimension with a linear layer,
    prepends a class token, adds a learned positional embedding of the patch centers, and applies a
    standard pre-norm transformer encoder. Optional per-point features `x` are gathered per patch and
    concatenated to the centered coordinates before the tokenizer.

    Args:
        embed_dim: The transformer dimension $d$.
        depth: The number of transformer blocks.
        num_heads: The number of attention heads.
        num_group: The number of patches $G$.
        group_size: The number of points $M$ per patch.
        encoder_dims: The token-embedding dimension before the linear bridge.
        in_channels: The number of per-point feature channels concatenated to the coordinates ($0$ for coordinates only).
        token_local_channels: Hidden widths of the tokenizer's per-point MLP.
        token_global_channels: Hidden widths of the tokenizer's per-patch MLP.
        pos_embed_channels: Hidden widths of the positional-embedding MLP.
        drop_path_rate: The stochastic depth rate (identity at eval).
        spatial_dim: The number of spatial dimensions of the coordinates.
        act: The activation used in the transformer MLPs.
        act_kwargs: Keyword arguments for the activation.
        norm: The normalization used in the transformer blocks.
        norm_kwargs: Keyword arguments for the normalization.
        token_act: The activation used in the token encoder.
        token_act_kwargs: Keyword arguments for the token-encoder activation.
        token_norm: The normalization used in the token encoder.
        token_norm_kwargs: Keyword arguments for the token-encoder normalization.

    Shape:
        - Input: $(N, C)$ or `None`, $(N, 3)$ and $(N,)$.
        - Output: $(B, G + 1, d)$ (token 0 is the class token).

    Example:
        ```python
        import torch
        from torch_pointcloud.models.point_bert import PointBERTEncoder

        encoder = PointBERTEncoder(embed_dim=384, depth=12, num_heads=6)
        pos = torch.randn(2048, 3)
        batch = torch.cat([torch.zeros(1024), torch.ones(1024)]).long()
        out = encoder(None, pos, batch)
        print(out.shape)
        ```
    """

    def __init__(
        self,
        embed_dim: int = 384,
        depth: int = 12,
        num_heads: int = 6,
        num_group: int = 64,
        group_size: int = 32,
        encoder_dims: int = 256,
        in_channels: int = 0,
        token_local_channels: Sequence[int] = (128, 256),
        token_global_channels: Sequence[int] = (512,),
        pos_embed_channels: Sequence[int] = (128,),
        drop_path_rate: float = 0.1,
        spatial_dim: int = 3,
        act: Union[str, Callable, None] = "gelu",
        act_kwargs: Optional[Dict[str, Any]] = None,
        norm: Union[str, Callable, None] = nn.LayerNorm,
        norm_kwargs: Optional[Dict[str, Any]] = None,
        token_act: Union[str, Callable, None] = "relu",
        token_act_kwargs: Optional[Dict[str, Any]] = None,
        token_norm: Union[str, Callable, None] = "batch_norm",
        token_norm_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_heads = num_heads
        self.num_group = num_group
        self.group_size = group_size
        self.encoder_dims = encoder_dims

        self.encoder = PointPatchEmbed(
            embed_dim=encoder_dims,
            in_channels=spatial_dim + in_channels,
            local_channels=token_local_channels,
            global_channels=token_global_channels,
            act=token_act,
            act_kwargs=token_act_kwargs,
            norm=token_norm,
            norm_kwargs=token_norm_kwargs,
        )
        self.reduce_dim = nn.Linear(encoder_dims, embed_dim)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.cls_pos = nn.Parameter(torch.zeros(1, 1, embed_dim))

        self.pos_embed = MLP([spatial_dim, *pos_embed_channels, embed_dim], act="gelu", norm=None, plain_last=True)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    embed_dim,
                    num_heads=num_heads,
                    act=act,
                    act_kwargs=act_kwargs,
                    norm=norm,
                    norm_kwargs=norm_kwargs,
                )
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim)

    @overload
    def forward(
        self, x: OptTensor, pos: Tensor, batch: Tensor, return_intermediates: Literal[True]
    ) -> Tuple[Tensor, Tensor, List[Tensor]]: ...

    @overload
    def forward(
        self, x: OptTensor, pos: Tensor, batch: Tensor, return_intermediates: Literal[False] = False
    ) -> Tensor: ...

    def forward(self, x: OptTensor, pos: Tensor, batch: Tensor, return_intermediates: bool = False) -> Any:
        if x is None:
            neighborhood, center = group(pos, batch, self.num_group, self.group_size, random_start=self.training)
        else:
            neighborhood, center, neighbor_idx = group(
                pos, batch, self.num_group, self.group_size, random_start=self.training, return_indices=True
            )
            neighborhood = torch.cat([neighborhood, x[neighbor_idx].reshape(*neighborhood.shape[:3], -1)], dim=-1)

        tokens = self.encoder(neighborhood)
        tokens = self.reduce_dim(tokens)

        B = tokens.size(0)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        cls_pos = self.cls_pos.expand(B, -1, -1)
        pos_embed = self.pos_embed(center)

        x = torch.cat([cls_tokens, tokens], dim=1)
        pos_seq = torch.cat([cls_pos, pos_embed], dim=1)

        intermediates: List[Tensor] = []
        for block in self.blocks:
            x = block(x + pos_seq)
            if return_intermediates:
                intermediates.append(x)

        x = self.norm(x)
        if return_intermediates:
            return x, center, intermediates
        return x


class PointBERTClassification(ClassificationModel):
    r"""Point-BERT classification model.

    Implements the finetuning model of :arxiv: [Point-BERT: Pre-training 3D Point Cloud Transformers
    with Masked Point Modeling](https://arxiv.org/abs/2111.14819), adapted from
    :github: [lulutang0608/Point-BERT](https://github.com/lulutang0608/Point-BERT).

    A `PointBERTEncoder` backbone followed by a 2-layer MLP head. The global feature concatenates the
    class token with the max-pooled patch tokens, so the head input dimension is $2d$.

    Args:
        in_channels: The number of per-point feature channels concatenated to the coordinates ($0$ for coordinates only).
        num_classes: The number of output classes.
        embed_dim: The transformer dimension $d$.
        depth: The number of transformer blocks.
        num_heads: The number of attention heads.
        num_group: The number of patches $G$.
        group_size: The number of points $M$ per patch.
        encoder_dims: The token-embedding dimension before the linear bridge.
        token_local_channels: Hidden widths of the tokenizer's per-point MLP.
        token_global_channels: Hidden widths of the tokenizer's per-patch MLP.
        pos_embed_channels: Hidden widths of the positional-embedding MLP.
        drop_path_rate: The stochastic depth rate (identity at eval).
        spatial_dim: The number of spatial dimensions of the coordinates.
        act: The activation used in the transformer MLPs.
        act_kwargs: Keyword arguments for the activation.
        norm: The normalization used in the transformer blocks.
        norm_kwargs: Keyword arguments for the normalization.
        head_act: The activation used in the classification head.
        dropout: The dropout rate of the classification head.
        head_channels: The hidden width(s) of the classification head.

    Shape:
        - Input: $(N, 3)$ and $(N,)$.
        - Output: $(B, \text{num\_classes})$.

    Example:
        ```python
        import torch
        from torch_pointcloud.models.point_bert import PointBERTClassification

        model = PointBERTClassification(in_channels=0, num_classes=40)
        pos = torch.randn(2048, 3)
        batch = torch.cat([torch.zeros(1024), torch.ones(1024)]).long()
        logits = model(None, pos, batch)
        print(logits.shape)
        ```
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        *,
        embed_dim: int = 384,
        depth: int = 12,
        num_heads: int = 6,
        num_group: int = 64,
        group_size: int = 32,
        encoder_dims: int = 256,
        token_local_channels: Sequence[int] = (128, 256),
        token_global_channels: Sequence[int] = (512,),
        pos_embed_channels: Sequence[int] = (128,),
        drop_path_rate: float = 0.1,
        spatial_dim: int = 3,
        act: Union[str, Callable, None] = "gelu",
        act_kwargs: Optional[Dict[str, Any]] = None,
        norm: Union[str, Callable, None] = nn.LayerNorm,
        norm_kwargs: Optional[Dict[str, Any]] = None,
        head_act: Union[str, Callable, None] = "relu",
        dropout: float = 0.5,
        head_channels: Optional[Union[int, Sequence[int]]] = 256,
    ):
        super().__init__(in_channels=in_channels, num_classes=num_classes)
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_heads = num_heads
        self.num_group = num_group
        self.group_size = group_size
        self.encoder_dims = encoder_dims
        self.token_local_channels = token_local_channels
        self.token_global_channels = token_global_channels
        self.pos_embed_channels = pos_embed_channels
        self.drop_path_rate = drop_path_rate
        self.spatial_dim = spatial_dim
        self.act = act
        self.act_kwargs = act_kwargs
        self.norm = norm
        self.norm_kwargs = norm_kwargs
        self.head_act = head_act
        self.dropout = dropout
        self.head_channels = ensure_list(head_channels, none_as_empty=True)

        self.encoder = self.configure_encoder()
        self.head = self.configure_head()

    def configure_encoder(self) -> PointBERTEncoder:
        return PointBERTEncoder(
            embed_dim=self.embed_dim,
            depth=self.depth,
            num_heads=self.num_heads,
            num_group=self.num_group,
            group_size=self.group_size,
            encoder_dims=self.encoder_dims,
            in_channels=self.in_channels,
            token_local_channels=self.token_local_channels,
            token_global_channels=self.token_global_channels,
            pos_embed_channels=self.pos_embed_channels,
            drop_path_rate=self.drop_path_rate,
            spatial_dim=self.spatial_dim,
            act=self.act,
            act_kwargs=self.act_kwargs,
            norm=self.norm,
            norm_kwargs=self.norm_kwargs,
        )

    def configure_head(self) -> nn.Module:
        head_channels = ensure_list(self.head_channels, none_as_empty=True)
        return MLP(
            [self.embed_dim * 2, *head_channels, self.num_classes],
            act=self.head_act,
            norm=None,
            dropout=self.dropout,
            plain_last=True,
        )

    def reset_classifier(
        self,
        num_classes: int,
        head_channels: Optional[Union[int, Sequence[int]]] = 256,
        dropout: float = 0.5,
    ) -> None:
        self.num_classes = num_classes
        self.head_channels = ensure_list(head_channels, none_as_empty=True)
        self.dropout = dropout
        self.head = self.configure_head()

    @overload
    def forward_features(
        self, x: OptTensor, pos: Tensor, batch: Tensor, return_intermediates: Literal[True]
    ) -> Tuple[Tensor, Tensor, List[Tensor]]: ...

    @overload
    def forward_features(
        self, x: OptTensor, pos: Tensor, batch: Tensor, return_intermediates: Literal[False] = False
    ) -> Tensor: ...

    def forward_features(self, x: OptTensor, pos: Tensor, batch: Tensor, return_intermediates: bool = False) -> Any:
        return self.encoder(x, pos, batch, return_intermediates=return_intermediates)

    def forward_head(self, x: Tensor, pre_logits: bool = False) -> Tensor:
        global_feat = torch.cat([x[:, 0], x[:, 1:].max(dim=1)[0]], dim=-1)
        return global_feat if pre_logits else self.head(global_feat)

    def forward(self, x: OptTensor, pos: Tensor, batch: Tensor) -> Tensor:
        x = self.forward_features(x, pos, batch)
        return self.forward_head(x)


class PointBERTMaskedTransformer(BaseModel):
    r"""Point-BERT masked point modeling backbone (pretrain).

    Implements the masked transformer (`transformer_q`) of :arxiv: [Point-BERT: Pre-training 3D Point
    Cloud Transformers with Masked Point Modeling](https://arxiv.org/abs/2111.14819), adapted from
    :github: [lulutang0608/Point-BERT](https://github.com/lulutang0608/Point-BERT).

    It embeds patches, masks a fraction of the patch tokens with a learned mask token, runs the
    transformer, and exposes a token-classification head (`lm_head`, predicting dVAE codebook ids)
    and a contrastive class head (`cls_head`). The MoCo / cutmix machinery of the full pretraining
    objective is omitted; this module is the reusable encoder that downstream finetuning loads.

    Args:
        in_channels: The number of per-point feature channels concatenated to the coordinates ($0$ for coordinates only).
        embed_dim: The transformer dimension $d$.
        depth: The number of transformer blocks.
        num_heads: The number of attention heads.
        num_group: The number of patches $G$.
        group_size: The number of points $M$ per patch.
        encoder_dims: The token-embedding dimension before the linear bridge.
        token_local_channels: Hidden widths of the tokenizer's per-point MLP.
        token_global_channels: Hidden widths of the tokenizer's per-patch MLP.
        pos_embed_channels: Hidden widths of the positional-embedding MLP.
        num_tokens: The dVAE vocabulary size predicted by `lm_head`.
        cls_dim: The contrastive head output dimension.
        mask_ratio: Lower / upper bounds of the block-masking ratio.
        drop_path_rate: The stochastic depth rate (identity at eval).
        spatial_dim: The number of spatial dimensions of the coordinates.
        act: The activation used in the transformer MLPs and the contrastive head.
        act_kwargs: Keyword arguments for the activation.
        norm: The normalization used in the transformer blocks.
        norm_kwargs: Keyword arguments for the normalization.
        token_act: The activation used in the token encoder.
        token_act_kwargs: Keyword arguments for the token-encoder activation.
        token_norm: The normalization used in the token encoder.
        token_norm_kwargs: Keyword arguments for the token-encoder normalization.

    Shape:
        - Input: $(N, 3)$ and $(N,)$.
        - Output: a dict with `cls_feature` $(B, \text{cls\_dim})$ and `logits` $(B, G, \text{num\_tokens})$.

    Example:
        ```python
        import torch
        from torch_pointcloud.models.point_bert import PointBERTMaskedTransformer

        model = PointBERTMaskedTransformer(in_channels=0)
        pos = torch.randn(2048, 3)
        batch = torch.cat([torch.zeros(1024), torch.ones(1024)]).long()
        out = model(None, pos, batch)
        print(out["cls_feature"].shape, out["logits"].shape)
        ```
    """

    def __init__(
        self,
        in_channels: int,
        *,
        embed_dim: int = 384,
        depth: int = 12,
        num_heads: int = 6,
        num_group: int = 64,
        group_size: int = 32,
        encoder_dims: int = 256,
        token_local_channels: Sequence[int] = (128, 256),
        token_global_channels: Sequence[int] = (512,),
        pos_embed_channels: Sequence[int] = (128,),
        num_tokens: int = 8192,
        cls_dim: int = 512,
        mask_ratio: Tuple[float, float] = (0.25, 0.45),
        drop_path_rate: float = 0.1,
        spatial_dim: int = 3,
        act: Union[str, Callable, None] = "gelu",
        act_kwargs: Optional[Dict[str, Any]] = None,
        norm: Union[str, Callable, None] = nn.LayerNorm,
        norm_kwargs: Optional[Dict[str, Any]] = None,
        token_act: Union[str, Callable, None] = "relu",
        token_act_kwargs: Optional[Dict[str, Any]] = None,
        token_norm: Union[str, Callable, None] = "batch_norm",
        token_norm_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(in_channels=in_channels)
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_heads = num_heads
        self.num_group = num_group
        self.group_size = group_size
        self.encoder_dims = encoder_dims
        self.num_tokens = num_tokens
        self.cls_dim = cls_dim
        self.mask_ratio = mask_ratio
        self.drop_path_rate = drop_path_rate

        self.encoder = PointPatchEmbed(
            embed_dim=encoder_dims,
            in_channels=spatial_dim + in_channels,
            local_channels=token_local_channels,
            global_channels=token_global_channels,
            act=token_act,
            act_kwargs=token_act_kwargs,
            norm=token_norm,
            norm_kwargs=token_norm_kwargs,
        )
        self.reduce_dim = nn.Linear(encoder_dims, embed_dim)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.cls_pos = nn.Parameter(torch.zeros(1, 1, embed_dim))

        self.pos_embed = MLP([spatial_dim, *pos_embed_channels, embed_dim], act="gelu", norm=None, plain_last=True)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    embed_dim,
                    num_heads=num_heads,
                    act=act,
                    act_kwargs=act_kwargs,
                    norm=norm,
                    norm_kwargs=norm_kwargs,
                )
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.lm_head = nn.Linear(embed_dim, num_tokens)
        self.cls_head = MLP([embed_dim, cls_dim, cls_dim], act=act, act_kwargs=act_kwargs, norm=None, plain_last=True)

    def forward(self, x: OptTensor, pos: Tensor, batch: Tensor) -> Dict[str, Tensor]:
        if x is None:
            neighborhood, center = group(pos, batch, self.num_group, self.group_size, random_start=self.training)
        else:
            neighborhood, center, neighbor_idx = group(
                pos, batch, self.num_group, self.group_size, random_start=self.training, return_indices=True
            )
            neighborhood = torch.cat([neighborhood, x[neighbor_idx].reshape(*neighborhood.shape[:3], -1)], dim=-1)

        tokens = self.encoder(neighborhood)
        tokens = self.reduce_dim(tokens)

        B = tokens.size(0)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        cls_pos = self.cls_pos.expand(B, -1, -1)
        pos_embed = self.pos_embed(center)

        x_seq = torch.cat([cls_tokens, tokens], dim=1)
        pos_seq = torch.cat([cls_pos, pos_embed], dim=1)

        for block in self.blocks:
            x_seq = block(x_seq + pos_seq)
        x_seq = self.norm(x_seq)

        cls_feature = self.cls_head(x_seq[:, 0])
        logits = self.lm_head(x_seq[:, 1:])
        return {"cls_feature": cls_feature, "logits": logits}


class TokenDGCNN(nn.Module):
    r"""DGCNN feature-propagation block of the Point-BERT dVAE tokenizer.

    Implements the DGCNN of :arxiv: [Point-BERT: Pre-training 3D Point Cloud Transformers with Masked
    Point Modeling](https://arxiv.org/abs/2111.14819), adapted from
    :github: [lulutang0608/Point-BERT](https://github.com/lulutang0608/Point-BERT).

    Operates on the $G$ patch tokens treated as a point set in token-feature space: it stacks four
    EdgeConv layers (KNN with $k = 4$ over the patch centers, edge features, group-norm, leaky-relu,
    max over neighbors), concatenates the four levels, and projects to the output channel. The edge
    layers are genuine 2D convolutions over $(B, C, k, N)$ and use `Conv2dBlock`; the input
    projection and the final fusion are per-point shared `Linear` maps (the group-norm of the final
    fusion still reduces over channels and points on the $(B, C_\text{out}, G)$ layout). Used twice in
    the dVAE: before the codebook (`dgcnn_1`) and after (`dgcnn_2`).

    Args:
        in_channels: The input token-feature dimension.
        out_channels: The output token-feature dimension.
        act: The activation used in the edge layers and the final fusion.
        act_kwargs: Keyword arguments for the activation.
        norm: The normalization used in the edge layers and the final fusion.
        norm_kwargs: Keyword arguments for the normalization.

    Shape:
        - Input: features $(B, G, C_\text{in})$ and centers $(B, G, 3)$.
        - Output: $(B, G, C_\text{out})$.

    Example:
        ```python
        import torch
        from torch_pointcloud.models.point_bert import TokenDGCNN

        dgcnn = TokenDGCNN(in_channels=256, out_channels=8192)
        feat = torch.randn(2, 64, 256)
        center = torch.randn(2, 64, 3)
        out = dgcnn(feat, center)
        print(out.shape)
        ```
    """

    K = 4

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        act: Union[str, Callable, None] = "leaky_relu",
        act_kwargs: Optional[Dict[str, Any]] = None,
        norm: Union[str, Callable, None] = "group_norm",
        norm_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        act_kwargs = act_kwargs if act_kwargs is not None else {"negative_slope": 0.2}
        norm_kwargs = norm_kwargs if norm_kwargs is not None else {"num_groups": 4}
        edge_kwargs: Dict[str, Any] = dict(
            kernel_size=1, act=act, act_kwargs=act_kwargs, norm=norm, norm_kwargs=norm_kwargs, bias=False
        )

        self.input_trans = MLP([in_channels, 128], act=None, norm=None, plain_last=True)
        self.layer1 = Conv2dBlock(256, 256, **edge_kwargs)
        self.layer2 = Conv2dBlock(512, 512, **edge_kwargs)
        self.layer3 = Conv2dBlock(1024, 512, **edge_kwargs)
        self.layer4 = Conv2dBlock(1024, 1024, **edge_kwargs)
        self.layer5_conv = nn.Linear(2304, out_channels, bias=False)
        self.layer5_norm = create_norm(norm, out_channels, dim=2, **norm_kwargs)
        self.layer5_act = create_act(act, **act_kwargs)

    @staticmethod
    def _graph_feature(idx: Tensor, x: Tensor) -> Tensor:
        B, C, N = x.shape
        K = idx.size(-1)
        feat = x.transpose(2, 1).reshape(B * N, C)[idx.reshape(-1)].view(B, N, K, C).permute(0, 3, 1, 2)
        x_expand = x.view(B, C, N, 1).expand(-1, -1, -1, K)
        return torch.cat([feat - x_expand, x_expand], dim=1)

    def forward(self, f: Tensor, coor: Tensor) -> Tensor:
        B, N, _ = coor.shape
        batch = torch.arange(B, device=coor.device).repeat_interleave(N)
        coor_flat = coor.reshape(B * N, -1)
        idx = knn(coor_flat, coor_flat, self.K, batch_x=batch, batch_y=batch)[1].view(B, N, self.K)

        f = self.input_trans(f).transpose(1, 2)
        features = []
        f = self.layer1(self._graph_feature(idx, f)).max(dim=-1)[0]
        features.append(f)
        f = self.layer2(self._graph_feature(idx, f)).max(dim=-1)[0]
        features.append(f)
        f = self.layer3(self._graph_feature(idx, f)).max(dim=-1)[0]
        features.append(f)
        f = self.layer4(self._graph_feature(idx, f)).max(dim=-1)[0]
        features.append(f)

        f = torch.cat(features, dim=1)
        f = self.layer5_conv(f.transpose(-1, -2)).transpose(-1, -2)
        if self.layer5_norm is not None:
            f = self.layer5_norm(f)
        if self.layer5_act is not None:
            f = self.layer5_act(f)
        return f.transpose(-1, -2)


class FoldingDecoder(nn.Module):
    r"""Folding reconstruction decoder of the Point-BERT dVAE.

    Implements the folding decoder of :arxiv: [Point-BERT: Pre-training 3D Point Cloud Transformers
    with Masked Point Modeling](https://arxiv.org/abs/2111.14819), adapted from
    :github: [lulutang0608/Point-BERT](https://github.com/lulutang0608/Point-BERT).

    A per-token global feature is decoded to a coarse point set with a fully-connected MLP, then
    folded to a fine point set with a $1 \times 1$ convolution conditioned on a $2 \times 2$ grid.

    Args:
        in_channels: The per-token feature dimension fed to the decoder.
        num_fine: The number of fine points $N$ reconstructed per patch.
        act: The activation used in the coarse MLP and the folding convolution.
        act_kwargs: Keyword arguments for the activation.
        norm: The normalization used in the folding convolution.
        norm_kwargs: Keyword arguments for the normalization.

    Shape:
        - Input: $(B, G, C_\text{in})$.
        - Output: coarse $(B, G, N // 4, 3)$ and fine $(B, G, N, 3)$.

    Example:
        ```python
        import torch
        from torch_pointcloud.models.point_bert import FoldingDecoder

        decoder = FoldingDecoder(in_channels=256, num_fine=32)
        feature = torch.randn(2, 64, 256)
        coarse, fine = decoder(feature)
        print(coarse.shape, fine.shape)
        ```
    """

    folding_seed: Tensor

    def __init__(
        self,
        in_channels: int,
        num_fine: int,
        act: Union[str, Callable, None] = "relu",
        act_kwargs: Optional[Dict[str, Any]] = None,
        norm: Union[str, Callable, None] = "batch_norm",
        norm_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        self.num_fine = num_fine
        self.grid_size = 2
        self.num_coarse = num_fine // 4
        self.mlp = MLP(
            [in_channels, 1024, 1024, 3 * self.num_coarse], act=act, act_kwargs=act_kwargs, norm=None, plain_last=True
        )
        self.final_conv = MLP(
            [in_channels + 3 + 2, 512, 512, 3],
            act=act,
            act_kwargs=act_kwargs,
            norm=norm,
            norm_kwargs=norm_kwargs,
            plain_last=True,
        )
        a = (
            torch.linspace(-0.05, 0.05, steps=self.grid_size)
            .view(1, self.grid_size)
            .expand(self.grid_size, self.grid_size)
            .reshape(1, -1)
        )
        b = (
            torch.linspace(-0.05, 0.05, steps=self.grid_size)
            .view(self.grid_size, 1)
            .expand(self.grid_size, self.grid_size)
            .reshape(1, -1)
        )
        self.register_buffer("folding_seed", torch.cat([a, b], dim=0).view(1, 2, self.grid_size**2), persistent=False)

    def forward(self, feature_global: Tensor) -> Tuple[Tensor, Tensor]:
        B, G, C = feature_global.shape
        feature_global = feature_global.reshape(B * G, C)

        coarse = self.mlp(feature_global).reshape(B * G, self.num_coarse, 3)

        point_feat = coarse.unsqueeze(2).expand(-1, -1, self.grid_size**2, -1)
        point_feat = point_feat.reshape(B * G, self.num_fine, 3)

        seed = self.folding_seed.unsqueeze(2).expand(B * G, -1, self.num_coarse, -1)
        seed = seed.reshape(B * G, 2, self.num_fine).transpose(2, 1).to(feature_global.device)

        feature = feature_global.unsqueeze(1).expand(-1, self.num_fine, -1)
        feat = torch.cat([feature, seed, point_feat], dim=2)

        center = coarse.unsqueeze(2).expand(-1, -1, self.grid_size**2, -1)
        center = center.reshape(B * G, self.num_fine, 3)

        fine = self.final_conv(feat.reshape(B * G * self.num_fine, -1)).reshape(B * G, self.num_fine, 3) + center
        fine = fine.reshape(B, G, self.num_fine, 3)
        coarse = coarse.reshape(B, G, self.num_coarse, 3)
        return coarse, fine


class PointBERTDiscreteVAE(BaseModel):
    r"""Point-BERT discrete VAE point tokenizer.

    Implements the dVAE of :arxiv: [Point-BERT: Pre-training 3D Point Cloud Transformers with Masked
    Point Modeling](https://arxiv.org/abs/2111.14819), adapted from
    :github: [lulutang0608/Point-BERT](https://github.com/lulutang0608/Point-BERT).

    The dVAE produces the discrete point tokens used as the masked-modeling targets in Point-BERT
    pretraining: a mini-PointNet token encoder, a DGCNN that maps tokens to codebook logits, a
    learned codebook, a second DGCNN, and a folding decoder reconstructing the patches.

    Args:
        in_channels: The number of input channels ($0$, coordinates only).
        num_group: The number of patches $G$.
        group_size: The number of points $M$ per patch.
        encoder_dims: The mini-PointNet token-embedding dimension.
        token_local_channels: Hidden widths of the tokenizer's per-point MLP.
        token_global_channels: Hidden widths of the tokenizer's per-patch MLP.
        num_tokens: The codebook vocabulary size.
        tokens_dims: The codebook embedding dimension.
        decoder_dims: The dimension fed to the folding decoder.
        act: The activation used in the token encoder and the folding decoder.
        act_kwargs: Keyword arguments for the token / decoder activation.
        norm: The normalization used in the token encoder and the folding decoder.
        norm_kwargs: Keyword arguments for the token / decoder normalization.

    Shape:
        - Input: $(N, 3)$ and $(N,)$.
        - Output: a dict with `logits` $(B, G, \text{num\_tokens})$ and reconstructions.

    Example:
        ```python
        import torch
        from torch_pointcloud.models.point_bert import PointBERTDiscreteVAE

        model = PointBERTDiscreteVAE(in_channels=0)
        pos = torch.randn(2048, 3)
        batch = torch.cat([torch.zeros(1024), torch.ones(1024)]).long()
        out = model(None, pos, batch)
        print(out["logits"].shape, out["fine"].shape)
        ```
    """

    def __init__(
        self,
        in_channels: int,
        *,
        num_group: int = 64,
        group_size: int = 32,
        encoder_dims: int = 256,
        token_local_channels: Sequence[int] = (128, 256),
        token_global_channels: Sequence[int] = (512,),
        num_tokens: int = 8192,
        tokens_dims: int = 256,
        decoder_dims: int = 256,
        act: Union[str, Callable, None] = "relu",
        act_kwargs: Optional[Dict[str, Any]] = None,
        norm: Union[str, Callable, None] = "batch_norm",
        norm_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(in_channels=in_channels)
        self.num_group = num_group
        self.group_size = group_size
        self.encoder_dims = encoder_dims
        self.num_tokens = num_tokens
        self.tokens_dims = tokens_dims
        self.decoder_dims = decoder_dims

        self.encoder = PointPatchEmbed(
            embed_dim=encoder_dims,
            local_channels=token_local_channels,
            global_channels=token_global_channels,
            act=act,
            act_kwargs=act_kwargs,
            norm=norm,
            norm_kwargs=norm_kwargs,
        )
        self.dgcnn_1 = TokenDGCNN(in_channels=encoder_dims, out_channels=num_tokens)
        self.codebook = nn.Parameter(torch.zeros(num_tokens, tokens_dims))
        self.dgcnn_2 = TokenDGCNN(in_channels=tokens_dims, out_channels=decoder_dims)
        self.decoder = FoldingDecoder(
            in_channels=decoder_dims,
            num_fine=group_size,
            act=act,
            act_kwargs=act_kwargs,
            norm=norm,
            norm_kwargs=norm_kwargs,
        )

    def tokenize(self, pos: Tensor, batch: Tensor) -> Tensor:
        neighborhood, center = group(pos, batch, self.num_group, self.group_size, random_start=self.training)
        feat = self.encoder(neighborhood)
        logits = self.dgcnn_1(feat, center)
        return logits

    def forward(
        self, x: OptTensor, pos: Tensor, batch: Tensor, temperature: float = 1.0, hard: bool = False
    ) -> Dict[str, Tensor]:
        neighborhood, center = group(pos, batch, self.num_group, self.group_size, random_start=self.training)
        feat = self.encoder(neighborhood)
        logits = self.dgcnn_1(feat, center)
        soft_one_hot = F.gumbel_softmax(logits, tau=temperature, dim=2, hard=hard)
        sampled = torch.einsum("bgn,nc->bgc", soft_one_hot, self.codebook)
        feature = self.dgcnn_2(sampled, center)
        coarse, fine = self.decoder(feature)
        return {
            "logits": logits,
            "neighborhood": neighborhood,
            "center": center,
            "coarse": coarse,
            "fine": fine,
        }


def _modelnet_transforms(num_samples: int) -> Callable:
    return T.Compose(
        [
            T.Rescale(keys="pos", method="centroid"),
            T.FarthestPointSample(pos_key="pos", num_samples=num_samples, random_start=False),
        ]
    )


def _scanobjectnn_transforms() -> Callable:
    return T.Compose([T.FarthestPointSample(pos_key="pos", num_samples=1024, random_start=False)])


_CLS_HPARAMS = dict(
    in_channels=0,
    embed_dim=384,
    depth=12,
    num_heads=6,
    num_group=64,
    group_size=32,
    encoder_dims=256,
    token_local_channels=(128, 256),
    token_global_channels=(512,),
    pos_embed_channels=(128,),
    drop_path_rate=0.1,
    spatial_dim=3,
    act="gelu",
    act_kwargs=None,
    head_act="relu",
    dropout=0.5,
    head_channels=256,
)


@register_model(
    "point-bert-base.modelnet40",
    task="classification",
    weights="hf://torch-pointcloud/point-bert/point-bert-base.modelnet40.pth",
    transforms=_modelnet_transforms(1024),
    hparams=dict(num_classes=40, **_CLS_HPARAMS),
)
def point_bert_base_modelnet40(**kwargs: Any) -> PointBERTClassification:
    return PointBERTClassification(**kwargs)


@register_model(
    "point-bert-base.modelnet40-4k",
    task="classification",
    weights="hf://torch-pointcloud/point-bert/point-bert-base.modelnet40-4k.pth",
    transforms=_modelnet_transforms(4096),
    hparams=dict(num_classes=40, **{**_CLS_HPARAMS, "num_group": 256}),
)
def point_bert_base_modelnet40_4k(**kwargs: Any) -> PointBERTClassification:
    return PointBERTClassification(**kwargs)


@register_model(
    "point-bert-base.modelnet40-8k",
    task="classification",
    weights="hf://torch-pointcloud/point-bert/point-bert-base.modelnet40-8k.pth",
    transforms=_modelnet_transforms(8192),
    hparams=dict(num_classes=40, **{**_CLS_HPARAMS, "num_group": 512}),
)
def point_bert_base_modelnet40_8k(**kwargs: Any) -> PointBERTClassification:
    return PointBERTClassification(**kwargs)


@register_model(
    "point-bert-base.scanobjectnn-objonly",
    task="classification",
    weights="hf://torch-pointcloud/point-bert/point-bert-base.scanobjectnn-objonly.pth",
    transforms=_scanobjectnn_transforms(),
    hparams=dict(num_classes=40, **_CLS_HPARAMS),
)
def point_bert_base_scanobjectnn_objonly(**kwargs: Any) -> PointBERTClassification:
    return PointBERTClassification(**kwargs)


@register_model(
    "point-bert-base.scanobjectnn-objbg",
    task="classification",
    weights="hf://torch-pointcloud/point-bert/point-bert-base.scanobjectnn-objbg.pth",
    transforms=_scanobjectnn_transforms(),
    hparams=dict(num_classes=40, **_CLS_HPARAMS),
)
def point_bert_base_scanobjectnn_objbg(**kwargs: Any) -> PointBERTClassification:
    return PointBERTClassification(**kwargs)


@register_model(
    "point-bert-base.scanobjectnn-hardest",
    task="classification",
    weights="hf://torch-pointcloud/point-bert/point-bert-base.scanobjectnn-hardest.pth",
    transforms=_scanobjectnn_transforms(),
    hparams=dict(num_classes=40, **_CLS_HPARAMS),
)
def point_bert_base_scanobjectnn_hardest(**kwargs: Any) -> PointBERTClassification:
    return PointBERTClassification(**kwargs)


@register_model(
    "point-bert-base.pretrain",
    task="base",
    weights="hf://torch-pointcloud/point-bert/point-bert-base.pretrain.pth",
    hparams=dict(
        in_channels=0,
        embed_dim=384,
        depth=12,
        num_heads=6,
        num_group=64,
        group_size=32,
        encoder_dims=256,
        token_local_channels=(128, 256),
        token_global_channels=(512,),
        pos_embed_channels=(128,),
        num_tokens=8192,
        cls_dim=512,
        mask_ratio=(0.25, 0.45),
        drop_path_rate=0.1,
        spatial_dim=3,
        act="gelu",
        act_kwargs=None,
    ),
)
def point_bert_base_pretrain(**kwargs: Any) -> PointBERTMaskedTransformer:
    return PointBERTMaskedTransformer(**kwargs)


@register_model(
    "point-bert-base.dvae",
    task="base",
    weights="hf://torch-pointcloud/point-bert/point-bert-base.dvae.pth",
    hparams=dict(
        in_channels=0,
        num_group=64,
        group_size=32,
        encoder_dims=256,
        token_local_channels=(128, 256),
        token_global_channels=(512,),
        num_tokens=8192,
        tokens_dims=256,
        decoder_dims=256,
    ),
)
def point_bert_base_dvae(**kwargs: Any) -> PointBERTDiscreteVAE:
    return PointBERTDiscreteVAE(**kwargs)
