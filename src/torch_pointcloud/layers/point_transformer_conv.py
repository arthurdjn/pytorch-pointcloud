"""Point Transformer vector-attention convolution over a neighborhood graph."""

from typing import Callable, Optional, Tuple, Union

import torch.nn as nn
from torch import Tensor
from torch_geometric.nn import MessagePassing
from torch_geometric.nn.inits import reset
from torch_geometric.typing import Adj, OptTensor, PairTensor, SparseTensor, torch_sparse
from torch_geometric.utils import add_self_loops, remove_self_loops, softmax
from typing_extensions import Unpack

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
            `pos_j - pos_i` of shape $[-1, 3]$ to shape
            $[-1, \text{out\_channels}]$.
            Will default to a `torch.nn.Linear` transformation if not
            further specified.
        attn_nn (torch.nn.Module, optional): A neural network
            $\gamma_\mathbf{\Theta}$ which maps transformed
            node features of shape $[-1, \text{out\_channels}]$
            to shape $[-1, \text{out\_channels}]$.
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
