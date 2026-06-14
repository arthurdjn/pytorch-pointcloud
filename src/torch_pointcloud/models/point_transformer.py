from typing import Any, Callable, Dict, List, Literal, Optional, Sequence, Tuple, Union, overload

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn import MLP, MessagePassing, fps, knn, knn_graph, knn_interpolate
from torch_geometric.nn.inits import reset
from torch_pointcloud.layers.act import create_act
from torch_pointcloud.layers.norms import create_norm
from torch_geometric.typing import Adj, OptTensor, PairTensor, SparseTensor, torch_sparse
from torch_geometric.utils import add_self_loops, remove_self_loops, scatter, softmax
from typing_extensions import Unpack

from torch_pointcloud.layers import PoolLike, create_pool
from torch_pointcloud.utils.conversion import ensure_tuple, ensure_tuple_size
from torch_pointcloud.utils.types import MessagePassingParams


# Adapted from: https://github.com/pyg-team/pytorch_geometric/blob/master/torch_geometric/nn/conv/transformer_conv.py
class PointTransformerConv(MessagePassing):
    r"""The Point Transformer layer from the
    :arxiv: ["Point Transformer"](https://arxiv.org/abs/2012.09164) paper
    by Hengshuang Zhao, Li Jiang, Jiaya Jia, Philip Torr, Vladlen Koltun.

    Note:
        This implementation was adapted from the PyTorch Geometric library,
        and supports the `num_groups` parameter to behave like the original
        implementation.

    $$
        \mathbf{x}^{\prime}_i =  \sum_{j \in
        \mathcal{N}(i) \cup \{ i \}} \alpha_{i,j} \left(\mathbf{W}_3
        \mathbf{x}_j + \delta_{ij} \right),
    $$

    where the attention coefficients $\alpha_{i,j}$ and
    positional embedding $\delta_{ij}$ are computed as

    $$
        \alpha_{i,j}= \textrm{softmax} \left( \gamma_\mathbf{\Theta}
        (\mathbf{W}_1 \mathbf{x}_i - \mathbf{W}_2 \mathbf{x}_j +
        \delta_{i,j}) \right)
    $$

    and

    $$
        \delta_{i,j}= h_{\mathbf{\Theta}}(\mathbf{p}_i - \mathbf{p}_j),
    $$

    with $\gamma_\mathbf{\Theta}$ and $h_\mathbf{\Theta}$
    denoting neural networks, *i.e.* MLPs, and
    $\mathbf{P} \in \mathbb{R}^{N \times D}$ defines the position of
    each point.

    Args:
        in_channels (int or tuple): Size of each input sample, or `-1` to
            derive the size from the first input(s) to the forward method.
            A tuple corresponds to the sizes of source and target
            dimensionalities.
        out_channels (int): Size of each output sample.
        pos_nn (torch.nn.Module, optional): A neural network
            $h_\mathbf{\Theta}$ which maps relative spatial coordinates
            `pos_j - pos_i` of shape `[-1, 3]` to shape
            `[-1, out_channels]`.
            Will default to a `torch.nn.Linear` transformation if not
            further specified.
        attn_nn (torch.nn.Module, optional): A neural network
            $\gamma_\mathbf{\Theta}$ which maps transformed
            node features of shape `[-1, out_channels]`
            to shape `[-1, out_channels]`.
        add_self_loops: If `False`, do not add self-loops to the input graph.

    Shapes:
        - **input:**
          node features $(|\mathcal{V}|, F_{in})$ or
          $((|\mathcal{V_s}|, F_{s}), (|\mathcal{V_t}|, F_{t}))$
          if bipartite,
          positions $(|\mathcal{V}|, 3)$ or
          $((|\mathcal{V_s}|, 3), (|\mathcal{V_t}|, 3))$ if bipartite,
          edge indices $(2, |\mathcal{E}|)$
        - **output:** node features $(|\mathcal{V}|, F_{out})$ or
          $((|\mathcal{V}_t|, F_{out}))$ if bipartite
    """

    def __init__(
        self,
        in_channels: Union[int, Tuple[int, int]],
        out_channels: int,
        spatial_dim: int = 3,
        num_groups: int = 8,
        pos_nn: Optional[Callable[[Tensor], Tensor]] = None,
        attn_nn: Optional[Callable[[Tensor], Tensor]] = None,
        add_self_loops: bool = False,  # noqa: F811
        **kwargs: Unpack[MessagePassingParams],
    ) -> None:
        kwargs.setdefault("aggr", "add")
        super().__init__(**kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_groups = num_groups
        self.add_self_loops = add_self_loops

        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)

        # if no position encoding network provided, create default one
        # following original implementation
        self.pos_nn = pos_nn or nn.Sequential(
            nn.Linear(spatial_dim, spatial_dim),
            nn.BatchNorm1d(spatial_dim),
            nn.ReLU(inplace=True),
            nn.Linear(spatial_dim, out_channels),
        )

        # if no custom attention network provided, create default one
        # that outputs num_groups weights per edge
        self.attn_nn = attn_nn or nn.Sequential(
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Linear(out_channels, out_channels // num_groups),
            nn.BatchNorm1d(out_channels // num_groups),
            nn.ReLU(inplace=True),
            nn.Linear(out_channels // num_groups, out_channels // num_groups),
        )

        self.lin = nn.Linear(in_channels[0], out_channels, bias=False)
        self.lin_src = nn.Linear(in_channels[0], out_channels, bias=False)
        self.lin_dst = nn.Linear(in_channels[1], out_channels, bias=False)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        super().reset_parameters()
        reset(self.pos_nn)
        if self.attn_nn is not None:
            reset(self.attn_nn)

        self.lin.reset_parameters()
        self.lin_src.reset_parameters()
        self.lin_dst.reset_parameters()

    def forward(
        self,
        x: Union[Tensor, PairTensor],
        pos: Union[Tensor, PairTensor],
        edge_index: Adj,
    ) -> Tensor:
        if isinstance(x, Tensor):
            alpha = (self.lin_src(x), self.lin_dst(x))
            x = (self.lin(x), x)
        else:
            alpha = (self.lin_src(x[0]), self.lin_dst(x[1]))
            x = (self.lin(x[0]), x[1])

        if isinstance(pos, Tensor):
            pos = (pos, pos)

        if self.add_self_loops:
            if isinstance(edge_index, Tensor):
                edge_index, _ = remove_self_loops(edge_index)
                edge_index, _ = add_self_loops(edge_index, num_nodes=min(pos[0].size(0), pos[1].size(0)))
            elif isinstance(edge_index, SparseTensor):
                edge_index = torch_sparse.set_diag(edge_index)

        # propagate_type: (x: PairTensor, pos: PairTensor, alpha: PairTensor)
        out = self.propagate(edge_index, x=x, pos=pos, alpha=alpha)
        return out

    def message(
        self,
        x_j: Tensor,
        pos_i: Tensor,
        pos_j: Tensor,
        alpha_i: Tensor,
        alpha_j: Tensor,
        index: Tensor,
        ptr: OptTensor,
        size_i: Optional[int],
    ) -> Tensor:
        delta = self.pos_nn(pos_i - pos_j)  # (num_edges, out_channels)
        alpha = alpha_i - alpha_j + delta  # (num_edges, out_channels)

        alpha = self.attn_nn(alpha)  # (num_edges, out_channels // num_groups)
        alpha = softmax(alpha, index, ptr, size_i)

        # reshape value features for grouped attention
        x_v = x_j + delta  # (num_edges, out_channels)
        x_v = x_v.view(x_v.size(0), self.num_groups, -1)  # (num_edges, num_groups, out_channels // num_groups)
        # apply grouped attention weights
        alpha = alpha.unsqueeze(1)  # (num_edges, 1, out_channels // num_groups)
        x_v = x_v * alpha  # (num_edges, num_groups, out_channels // num_groups)
        # reshape back to original feature dimension
        x_v = x_v.view(x_v.size(0), -1)  # (num_edges, out_channels)

        return x_v

    def extra_repr(self) -> str:
        return f"in_channels={self.in_channels}, out_channels={self.out_channels}, num_groups={self.num_groups}"


# Adapted from: https://github.com/pyg-team/pytorch_geometric/blob/master/examples/point_transformer_classification.py
class PointTransformerBlock(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        spatial_dim: int = 3,
        num_groups: int = 8,
        add_self_loops: bool = False,
        act: Union[str, Callable, None] = "relu",
        act_first: bool = False,
        act_kwargs: Optional[Dict[str, Any]] = None,
        norm: Union[str, Callable, None] = "batch_norm",
        norm_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()
        act_kwargs = act_kwargs or {}
        norm_kwargs = norm_kwargs or {}
        kwargs = dict(
            act=act,
            act_first=act_first,
            act_kwargs=act_kwargs,
            norm=norm,
            norm_kwargs=norm_kwargs,
            plain_last=False,
        )

        self.act = create_act(act, **act_kwargs) or nn.Identity()
        self.lin1 = nn.Linear(in_channels, in_channels)
        self.norm1 = create_norm(norm, out_channels, **norm_kwargs) or nn.Identity()
        self.transformer = PointTransformerConv(
            in_channels,
            out_channels,
            spatial_dim=spatial_dim,
            num_groups=num_groups,
            pos_nn=MLP([spatial_dim, spatial_dim, out_channels], **kwargs),
            attn_nn=MLP([out_channels, out_channels // num_groups, out_channels // num_groups], **kwargs),
            add_self_loops=add_self_loops,
        )
        self.norm2 = create_norm(norm, out_channels, **norm_kwargs) or nn.Identity()
        self.lin3 = nn.Linear(out_channels, out_channels)
        self.norm3 = create_norm(norm, out_channels, **norm_kwargs) or nn.Identity()

    def forward(self, x: Tensor, pos: Tensor, edge_index: Adj) -> Tensor:
        shortcut = x
        x = self.act(self.norm1(self.lin1(x)))
        x = self.act(self.norm2(self.transformer(x, pos, edge_index)))
        x = self.act(self.norm3(self.lin3(x)) + shortcut)
        return x


# Adapted from: https://github.com/pyg-team/pytorch_geometric/blob/master/examples/point_transformer_classification.py
class PointTransformerTransitionDown(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_neighbors: int = 16,
        ratio: float = 0.25,
        act: Union[str, Callable, None] = "relu",
        act_kwargs: Optional[Dict[str, Any]] = None,
        act_first: bool = False,
        norm: Union[str, Callable, None] = "batch_norm",
        norm_kwargs: Optional[Dict[str, Any]] = None,
        pool: str = "max",
    ):
        super().__init__()
        kwargs = dict(
            act=act,
            act_first=act_first,
            act_kwargs=act_kwargs,
            norm=norm,
            norm_kwargs=norm_kwargs,
            plain_last=False,
        )

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_neighbors = num_neighbors
        self.ratio = ratio
        self.pool = pool
        self.mlp = MLP([in_channels, out_channels], **kwargs)

    def forward(self, x: Tensor, pos: Tensor, batch: OptTensor) -> Tuple[Tensor, Tensor, OptTensor]:
        id_clusters = fps(pos, ratio=self.ratio, batch=batch)
        sub_batch = batch[id_clusters] if batch is not None else None
        id_k_neighbor = knn(pos, pos[id_clusters], k=self.num_neighbors, batch_x=batch, batch_y=sub_batch)

        x = self.mlp(x)

        x_out = scatter(x[id_k_neighbor[1]], id_k_neighbor[0], dim=0, dim_size=id_clusters.size(0), reduce=self.pool)
        sub_pos = pos[id_clusters]
        return x_out, sub_pos, sub_batch

    def extra_repr(self) -> str:
        return f"in_channels={self.in_channels}, out_channels={self.out_channels}, num_neighbors={self.num_neighbors}, ratio={self.ratio}"

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.extra_repr()})"


# Adapted from: https://github.com/pyg-team/pytorch_geometric/blob/master/examples/point_transformer_classification.py
class PointTransformerTransitionUp(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        act: Union[str, Callable, None] = "relu",
        act_kwargs: Optional[Dict[str, Any]] = None,
        act_first: bool = False,
        norm: Union[str, Callable, None] = "batch_norm",
        norm_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        kwargs = dict(
            act=act,
            act_first=act_first,
            act_kwargs=act_kwargs,
            norm=norm,
            norm_kwargs=norm_kwargs,
            plain_last=False,
        )

        self.mlp = MLP([in_channels, out_channels], **kwargs)
        self.mlp_skip = MLP([out_channels, out_channels], **kwargs)

    def forward(
        self,
        x: Tensor,
        pos: Tensor,
        batch: OptTensor,
        x_skip: Tensor,
        pos_skip: Tensor,
        batch_skip: OptTensor,
    ) -> Tensor:
        x = self.mlp(x)
        x_interpolated = knn_interpolate(x, pos_x=pos, pos_y=pos_skip, k=3, batch_x=batch, batch_y=batch_skip)
        return self.mlp_skip(x_skip) + x_interpolated


class PointTransformerEncoderBlock(torch.nn.Module):
    def __init__(
        self,
        channels: int,
        depth: int,
        num_groups: int,
        num_neighbors: int,
        spatial_dim: int = 3,
        add_self_loops: bool = False,
        act: Union[str, Callable, None] = "relu",
        act_kwargs: Optional[Dict[str, Any]] = None,
        act_first: bool = False,
        norm: Union[str, Callable, None] = "batch_norm",
        norm_kwargs: Optional[Dict[str, Any]] = None,
        downsample: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.num_neighbors = num_neighbors
        self.downsample = downsample
        self.blocks = nn.ModuleList()
        for _ in range(depth):
            self.blocks.append(
                PointTransformerBlock(
                    in_channels=channels,
                    out_channels=channels,
                    num_groups=num_groups,
                    spatial_dim=spatial_dim,
                    add_self_loops=add_self_loops,
                    act=act,
                    act_kwargs=act_kwargs,
                    act_first=act_first,
                    norm=norm,
                    norm_kwargs=norm_kwargs,
                )
            )

    def forward(self, x: Tensor, pos: Tensor, batch: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        if self.downsample is not None:
            x, pos, batch = self.downsample(x, pos, batch)

        edge_index = knn_graph(pos, k=self.num_neighbors, batch=batch)
        for block in self.blocks:
            x = block(x, pos, edge_index)

        return x, pos, batch


class PointTransformerDecoderBlock(torch.nn.Module):
    def __init__(
        self,
        channels: int,
        depth: int,
        num_groups: int,
        num_neighbors: int,
        spatial_dim: int = 3,
        add_self_loops: bool = False,
        act: Union[str, Callable, None] = "relu",
        act_kwargs: Optional[Dict[str, Any]] = None,
        act_first: bool = False,
        norm: Union[str, Callable, None] = "batch_norm",
        norm_kwargs: Optional[Dict[str, Any]] = None,
        upsample: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.num_neighbors = num_neighbors
        self.upsample = upsample
        self.blocks = nn.ModuleList()
        for _ in range(depth):
            self.blocks.append(
                PointTransformerBlock(
                    in_channels=channels,
                    out_channels=channels,
                    num_groups=num_groups,
                    spatial_dim=spatial_dim,
                    add_self_loops=add_self_loops,
                    act=act,
                    act_kwargs=act_kwargs,
                    act_first=act_first,
                    norm=norm,
                    norm_kwargs=norm_kwargs,
                )
            )

    def forward(
        self,
        x: Tensor,
        pos: Tensor,
        batch: OptTensor,
        x_skip: Tensor,
        pos_skip: Tensor,
        batch_skip: OptTensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        if self.upsample is not None:
            x = self.upsample(x, pos, batch, x_skip, pos_skip, batch_skip)

        edge_index = knn_graph(pos_skip, k=self.num_neighbors, batch=batch_skip)
        for block in self.blocks:
            x = block(x, pos_skip, edge_index)
        return x, pos_skip, batch_skip


class PointTransformerEncoder(torch.nn.Module):
    def __init__(
        self,
        channels: Sequence[int],
        depths: Sequence[int],
        num_groups: Sequence[int],
        num_neighbors: Sequence[int],
        ratios: Sequence[float],
        spatial_dim: int = 3,
        add_self_loops: bool = False,
        act: Union[str, Callable, None] = "relu",
        act_kwargs: Optional[Dict[str, Any]] = None,
        act_first: bool = False,
        norm: Union[str, Callable, None] = "batch_norm",
        norm_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        depths = ensure_tuple(depths)
        n = len(depths)
        extra_msg = "Encoder length `{param}` != {size}."
        channels = ensure_tuple_size(channels, size=n, extra_msg=extra_msg.format(param="channels", size=n))
        num_groups = ensure_tuple_size(num_groups, size=n, extra_msg=extra_msg.format(param="num_groups", size=n))
        ratios = ensure_tuple_size(ratios, size=n - 1, extra_msg=extra_msg.format(param="ratios", size=n - 1))
        num_neighbors = ensure_tuple_size(
            num_neighbors,
            size=n,
            extra_msg=extra_msg.format(param="num_neighbors", size=n),
        )

        self.blocks = nn.ModuleList()
        for i in range(n):
            downsample: Optional[nn.Module] = None
            if i > 0:
                downsample = PointTransformerTransitionDown(
                    in_channels=channels[i - 1],
                    out_channels=channels[i],
                    num_neighbors=num_neighbors[i - 1],
                    ratio=ratios[i - 1],
                    act=act,
                    act_kwargs=act_kwargs,
                    act_first=act_first,
                    norm=norm,
                    norm_kwargs=norm_kwargs,
                )

            block = PointTransformerEncoderBlock(
                channels=channels[i],
                depth=depths[i],
                num_groups=num_groups[i],
                num_neighbors=num_neighbors[i],
                spatial_dim=spatial_dim,
                add_self_loops=add_self_loops,
                act=act,
                act_kwargs=act_kwargs,
                act_first=act_first,
                norm=norm,
                norm_kwargs=norm_kwargs,
                downsample=downsample,
            )
            self.blocks.append(block)

    @overload
    def forward(
        self,
        x: Tensor,
        pos: Tensor,
        batch: Tensor,
        return_intermediates: Literal[True],
    ) -> Tuple[Tensor, Tensor, Tensor, List[Dict[str, Tensor]]]: ...

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
        intermediates = []
        for block in self.blocks:
            if return_intermediates:
                intermediates.append({"features": x, "pos": pos, "batch": batch})
            x, pos, batch = block(x, pos, batch)

        if return_intermediates:
            return x, pos, batch, intermediates
        return x, pos, batch


class PointTransformerDecoder(torch.nn.Module):
    def __init__(
        self,
        channels: Sequence[int],
        # skip_channels: Sequence[int],
        depths: Sequence[int],
        num_groups: Sequence[int],
        num_neighbors: Sequence[int],
        spatial_dim: int = 3,
        add_self_loops: bool = False,
        act: Union[str, Callable, None] = "relu",
        act_kwargs: Optional[Dict[str, Any]] = None,
        act_first: bool = False,
        norm: Union[str, Callable, None] = "batch_norm",
        norm_kwargs: Optional[Dict[str, Any]] = None,
        upsample: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.upsample = upsample
        self.blocks = nn.ModuleList()

        depths = ensure_tuple(depths)
        n = len(depths)
        extra_msg = "Decoder length `{param}` != {size}."
        channels = ensure_tuple_size(channels, size=n + 1, extra_msg=extra_msg.format(param="channels", size=n + 1))
        num_groups = ensure_tuple_size(num_groups, size=n, extra_msg=extra_msg.format(param="num_groups", size=n))
        num_neighbors = ensure_tuple_size(
            num_neighbors,
            size=n,
            extra_msg=extra_msg.format(param="num_neighbors", size=n),
        )

        self.blocks = nn.ModuleList()
        for i in range(n):
            upsample = PointTransformerTransitionUp(
                in_channels=channels[i],
                out_channels=channels[i + 1],
                act=act,
                act_kwargs=act_kwargs,
                act_first=act_first,
                norm=norm,
                norm_kwargs=norm_kwargs,
            )

            block = PointTransformerDecoderBlock(
                channels=channels[i + 1],
                depth=depths[i],
                num_groups=num_groups[i],
                num_neighbors=num_neighbors[i],
                spatial_dim=spatial_dim,
                add_self_loops=add_self_loops,
                act=act,
                act_kwargs=act_kwargs,
                act_first=act_first,
                norm=norm,
                norm_kwargs=norm_kwargs,
                upsample=upsample,
            )
            self.blocks.append(block)

    def forward(
        self,
        x: Tensor,
        pos: Tensor,
        batch: Tensor,
        intermediates: List[Dict[str, Tensor]],
    ) -> Tuple[Tensor, Tensor, Tensor]:
        for block, intermediate in zip(self.blocks, reversed(intermediates)):
            x, pos, batch = block(x, pos, batch, intermediate["features"], intermediate["pos"], intermediate["batch"])
        return x, pos, batch


class PointTransformerClassification(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        *,
        encoder_channels: Sequence[int],
        encoder_depths: Sequence[int],
        encoder_num_groups: Sequence[int],
        encoder_num_neighbors: Sequence[int],
        ratios: Sequence[float],
        spatial_dim: int = 3,
        add_self_loops: bool = False,
        global_pool: PoolLike = "max",
        dropout: float = 0.0,
        act: Union[str, Callable, None] = "relu",
        act_kwargs: Optional[Dict[str, Any]] = None,
        act_first: bool = False,
        norm: Union[str, Callable, None] = "batch_norm",
        norm_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()
        # if in_channels is 0, we use positions as features
        self.in_channels = in_channels if in_channels > 0 else spatial_dim
        self.num_classes = num_classes
        self.embedding_dim = encoder_channels[-1]
        self.dropout = dropout

        self.embeddings = MLP([in_channels, encoder_channels[0]], plain_last=False)
        self.encoder = PointTransformerEncoder(
            channels=encoder_channels,
            depths=encoder_depths,
            num_groups=encoder_num_groups,
            num_neighbors=encoder_num_neighbors,
            ratios=ratios,
            spatial_dim=spatial_dim,
            add_self_loops=add_self_loops,
            act=act,
            act_kwargs=act_kwargs,
            act_first=act_first,
            norm=norm,
            norm_kwargs=norm_kwargs,
        )
        self.global_pool = create_pool(global_pool)
        self.head = nn.Identity() if num_classes == 0 else nn.Linear(self.embedding_dim, num_classes)

    def reset_classifier(self, num_classes: int, global_pool: PoolLike = "max", **kwargs: Any) -> None:
        self.num_classes = num_classes
        self.global_pool = create_pool(global_pool)
        self.head = nn.Identity() if self.num_classes == 0 else nn.Linear(self.embedding_dim, self.num_classes)

    @overload
    def forward_encoder(
        self,
        x: OptTensor,
        pos: Tensor,
        batch: Tensor,
        return_intermediates: Literal[True],
    ) -> Tuple[Tensor, Tensor, Tensor, List[Dict[str, Tensor]]]: ...

    @overload
    def forward_encoder(
        self,
        x: OptTensor,
        pos: Tensor,
        batch: Tensor,
        return_intermediates: Literal[False] = False,
    ) -> Tuple[Tensor, Tensor, Tensor]: ...

    def forward_encoder(
        self,
        x: OptTensor,
        pos: Tensor,
        batch: OptTensor = None,
        return_intermediates: bool = False,
    ) -> Any:
        x = x if x is not None else pos
        x = self.embeddings(x)
        return self.encoder(x, pos, batch, return_intermediates=return_intermediates)

    def forward_head(self, x: Tensor, batch: Tensor, pre_logits: bool = False) -> Tensor:
        x = self.global_pool(x, batch)
        if self.dropout:
            x = F.dropout(x, p=self.dropout, training=self.training)
        return x if pre_logits else self.head(x)

    def forward(self, x: OptTensor, pos: Tensor, batch: OptTensor = None) -> Tensor:
        x, _, batch = self.forward_encoder(x, pos, batch)
        return self.forward_head(x, batch)


class PointTransformerSegmentation(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        *,
        encoder_channels: Sequence[int],
        encoder_depths: Sequence[int],
        encoder_num_groups: Sequence[int],
        encoder_num_neighbors: Sequence[int],
        decoder_channels: Sequence[int],
        decoder_depths: Sequence[int],
        decoder_num_groups: Sequence[int],
        decoder_num_neighbors: Sequence[int],
        ratios: Sequence[float],
        spatial_dim: int = 3,
        add_self_loops: bool = False,
        dropout: float = 0.0,
        act: Union[str, Callable, None] = "relu",
        act_kwargs: Optional[Dict[str, Any]] = None,
        act_first: bool = False,
        norm: Union[str, Callable, None] = "batch_norm",
        norm_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()
        # if in_channels is 0, we use positions as features
        self.in_channels = in_channels if in_channels > 0 else spatial_dim
        self.num_classes = num_classes
        self.embedding_dim = decoder_channels[-1]
        self.dropout = dropout

        self.embeddings = MLP([self.in_channels, encoder_channels[0]], plain_last=False)
        self.encoder = PointTransformerEncoder(
            channels=encoder_channels,
            depths=encoder_depths,
            num_groups=encoder_num_groups,
            num_neighbors=encoder_num_neighbors,
            ratios=ratios,
            spatial_dim=spatial_dim,
            add_self_loops=add_self_loops,
            act=act,
            act_kwargs=act_kwargs,
            act_first=act_first,
            norm=norm,
            norm_kwargs=norm_kwargs,
        )
        self.decoder = PointTransformerDecoder(
            channels=[encoder_channels[-1]] + list(decoder_channels),
            depths=decoder_depths,
            num_groups=decoder_num_groups,
            num_neighbors=decoder_num_neighbors,
            spatial_dim=spatial_dim,
            add_self_loops=add_self_loops,
            act=act,
            act_kwargs=act_kwargs,
            act_first=act_first,
            norm=norm,
            norm_kwargs=norm_kwargs,
        )
        self.head = nn.Identity() if self.num_classes == 0 else nn.Linear(self.embedding_dim, self.num_classes)

    def reset_classifier(self, num_classes: int, **kwargs: Any) -> None:
        self.num_classes = num_classes
        self.head = nn.Identity() if self.num_classes == 0 else nn.Linear(self.embedding_dim, self.num_classes)

    @overload
    def forward_encoder(
        self,
        x: OptTensor,
        pos: Tensor,
        batch: Tensor,
        return_intermediates: Literal[True],
    ) -> Tuple[Tensor, Tensor, Tensor, List[Dict[str, Tensor]]]: ...

    @overload
    def forward_encoder(
        self,
        x: OptTensor,
        pos: Tensor,
        batch: Tensor,
        return_intermediates: Literal[False] = False,
    ) -> Tuple[Tensor, Tensor, Tensor]: ...

    def forward_encoder(
        self,
        x: OptTensor,
        pos: Tensor,
        batch: OptTensor = None,
        return_intermediates: bool = False,
    ) -> Any:
        x = x if x is not None else pos
        x = self.embeddings(x)
        return self.encoder(x, pos, batch, return_intermediates=return_intermediates)

    def forward_decoder(
        self,
        x: Tensor,
        pos: Tensor,
        batch: Tensor,
        intermediates: List[Dict[str, Tensor]],
    ) -> Tuple[Tensor, Tensor, Tensor]:
        return self.decoder(x, pos, batch, intermediates)

    def forward_head(self, x: Tensor, pre_logits: bool = False) -> Tensor:
        if self.dropout:
            x = F.dropout(x, p=self.dropout, training=self.training)
        return x if pre_logits else self.head(x)

    def forward(self, x: OptTensor, pos: Tensor, batch: OptTensor = None) -> Tensor:
        x, pos, batch, intermediates = self.forward_encoder(x, pos, batch, return_intermediates=True)
        x, _, _ = self.forward_decoder(x, pos, batch, intermediates)
        return self.forward_head(x)
