from typing import TYPE_CHECKING, Literal, Union

import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.nn import MessagePassing
from torch_geometric.nn.inits import reset
from torch_geometric.typing import Adj, SparseTensor
from torch_geometric.utils import add_self_loops, remove_self_loops
from typing_extensions import Unpack

from torch_pointcloud.utils.conversion import ensure_option
from torch_pointcloud.utils.imports import optional_import
from torch_pointcloud.utils.types import MessagePassingParams, OptTensor, PairTensor

if TYPE_CHECKING:
    import torch_sparse
    from torch_scatter import scatter_mean


torch_sparse, _ = optional_import("torch_sparse")
scatter_mean, _ = optional_import("torch_scatter", "scatter_mean")


NormalizeType = Literal["center", "anchor"]


class GeometricAffineConv(MessagePassing):
    def __init__(
        self,
        local_nn: nn.Module,
        channels: int,
        spatial_dim: int = 3,
        use_pos: bool = True,
        normalize: NormalizeType = "center",
        add_self_loops: bool = False,
        eps: float = 1e-5,
        **kwargs: Unpack[MessagePassingParams],
    ):
        kwargs.setdefault("aggr", "max")
        super().__init__(**kwargs)

        self.local_nn = local_nn
        self.channels = channels
        self.spatial_dim = spatial_dim
        self.use_pos = use_pos
        self.normalize = ensure_option(normalize, NormalizeType, name="normalize")
        self.add_self_loops = add_self_loops
        self.eps = eps

        msg_channels = channels + spatial_dim if use_pos else channels
        self.register_parameter("alpha", nn.Parameter(torch.ones(msg_channels)))
        self.register_parameter("beta", nn.Parameter(torch.zeros(msg_channels)))

    def reset_parameters(self) -> None:
        super().reset_parameters()
        reset(self.local_nn)

    def forward(
        self,
        x: Union[Tensor, PairTensor],
        pos: Union[Tensor, PairTensor],
        batch: Union[Tensor, PairTensor],
        edge_index: Adj,
    ) -> Tensor:
        x_src, x_dst = (x, x) if isinstance(x, Tensor) else x
        pos_src, pos_dst = (pos, pos) if isinstance(pos, Tensor) else pos
        if isinstance(batch, tuple) and len(batch) == 2:
            batch = batch[1] if self.flow == "source_to_target" else batch[0]

        if self.add_self_loops:
            if isinstance(edge_index, Tensor):
                edge_index, _ = remove_self_loops(edge_index)
                edge_index, _ = add_self_loops(edge_index, num_nodes=min(pos[0].size(0), pos[1].size(0)))
            elif isinstance(edge_index, SparseTensor):
                edge_index = torch_sparse.set_diag(edge_index)

        return self.propagate(
            edge_index,
            x=(x_src, x_dst),
            pos=(pos_src, pos_dst),
            batch=batch,
            size=(x_src.size(0), x_dst.size(0)),
        )

    def message(
        self,
        index: Tensor,
        x_i: Tensor,
        x_j: Tensor,
        pos_i: OptTensor,
        pos_j: OptTensor,
        batch: Tensor,
    ) -> Tensor:
        ensure_option(self.normalize, NormalizeType, name="normalize")

        msg_i, msg_j = x_i, x_j

        if self.use_pos:
            if pos_i is None or pos_j is None:
                raise ValueError("pos_i and pos_j must be provided if `use_pos` is True")

            msg_j = torch.cat([msg_j, pos_j], dim=1)
            msg_i = torch.cat([msg_i, pos_i], dim=1)

        if self.normalize == "anchor":
            mean = msg_i
        elif self.normalize == "center":
            local_mean = scatter_mean(msg_j, index, dim=0)
            mean = local_mean[index]

        msg = msg_j - mean
        edge_batch = batch[index]

        sq_msg_mean_c = msg.pow(2).mean(dim=-1)
        sigma_sq = scatter_mean(sq_msg_mean_c, edge_batch, dim=0)
        sigma = torch.sqrt(sigma_sq + self.eps)

        msg = msg / sigma[edge_batch].view(-1, 1)
        msg = self.alpha * msg + self.beta
        msg = torch.cat([msg, x_i], dim=1)

        return self.local_nn(msg)
