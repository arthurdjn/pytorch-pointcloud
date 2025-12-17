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
from torch_pointcloud.utils.types import MessagePassingParams, PairTensor

if TYPE_CHECKING:
    import torch_sparse
    from torch_scatter import scatter_add, scatter_mean


torch_sparse, _ = optional_import("torch_sparse")
scatter_add, _ = optional_import("torch_scatter", "scatter_add")
scatter_mean, _ = optional_import("torch_scatter", "scatter_mean")


def batch_scatter_std(
    x: Tensor,
    batch: Tensor,
    num_batches: int | None = None,
    unbiased: bool = False,
    eps: float = 1e-5,
) -> Tensor:
    """Compute the standard deviation per batch over all elements of $x$ belonging to that batch.

    Args:
        x: The input tensor of shape $(E, F)$.
        batch: The batch tensor of shape $(E,)$.
        num_batches: The number of batches. If not provided, it will be inferred from the batch tensor.
        unbiased: Whether to use the unbiased estimator of the variance.
        eps: The epsilon value to avoid division by zero.

    Returns:
        The standard deviation per batch of shape $(B,)$.
    """
    # NOTE: The below implementation benefits from scatter operations and minimizes
    # memory usage. A naive implementation could be done by flattening and broadcasting everything:
    #
    # ```python
    # E, F = x.shape
    # x_flat = x.reshape(-1)                                    # (E*F,)
    # batch_flat = batch.view(-1, 1).expand(-1, F).reshape(-1)  # (E*F,)
    # std_b = scatter_std(
    #     x_flat,
    #     batch_flat,
    #     dim=0,
    #     dim_size=num_batches,
    #     unbiased=unbiased,
    # )
    # ```
    #
    # However, this naive implementation is up to 100x slower for large tensors (100,000 points).

    if num_batches is None:
        num_batches = int(batch.max().item()) + 1 if batch.numel() > 0 else 1

    if x.numel() == 0:
        return x.new_ones(num_batches)

    E, F = x.shape
    dtype, device = x.dtype, x.device

    sum_e = x.sum(dim=1)  # (E,)
    sum_e2 = x.square().sum(dim=1)  # (E,)

    sum_g = scatter_add(sum_e, batch, dim=0, dim_size=num_batches)
    sum_g2 = scatter_add(sum_e2, batch, dim=0, dim_size=num_batches)

    count_e = torch.full((E,), float(F), device=device, dtype=dtype)
    count_g = scatter_add(count_e, batch, dim=0, dim_size=num_batches)

    mean_g = sum_g / (count_g + eps)
    var_g = sum_g2 / (count_g + eps) - mean_g.square()

    if unbiased:
        corr = count_g / torch.clamp(count_g - 1.0, min=1.0)
        var_g = var_g * corr

    return torch.sqrt(torch.clamp(var_g, min=0.0) + eps)


NormalizeType = Literal["center", "anchor"]


class GeometricAffineConv(MessagePassing):
    def __init__(
        self,
        local_nn: nn.Module,
        channels: int,
        spatial_dim: int = 3,
        normalize: NormalizeType = "center",
        add_self_loops: bool = True,
        eps: float = 1e-5,
        **kwargs: Unpack[MessagePassingParams],
    ):
        # NOTE: we use `node_dim=0` to ensure that the batch dimension is the first dimension
        #       this is important for the message passing API, otherwise we will have a IndexError.
        kwargs.setdefault("node_dim", 0)
        kwargs.setdefault("aggr", "max")
        super().__init__(**kwargs)

        self.local_nn = local_nn
        self.channels = channels
        self.normalize = ensure_option(normalize, NormalizeType, name="normalize")
        self.spatial_dim = spatial_dim
        self.add_self_loops = add_self_loops
        self.eps = eps

        self.register_parameter("affine_alpha", nn.Parameter(torch.ones(1, channels + self.spatial_dim)))
        self.register_parameter("affine_beta", nn.Parameter(torch.zeros(1, channels + self.spatial_dim)))
        self.reset_parameters()

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
        batch_src, batch_dst = (batch, batch) if isinstance(batch, Tensor) else batch

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
            batch=(batch_src, batch_dst),
            size=(x_src.size(0), x_dst.size(0)),
        )

    def message(
        self,
        x_j: Tensor,
        x_i: Tensor,
        pos_j: Tensor,
        pos_i: Tensor,
        batch_i: Tensor,
        index: Tensor,  # dst indices per edge (E,)
        size_i: int,  # N_dst (provided by propagate via size)
    ) -> Tensor:
        normalize = ensure_option(self.normalize, NormalizeType, name="normalize")

        neigh = torch.cat([x_j, pos_j], dim=-1)  # (E, C[+d])

        if normalize == "center":
            mean_per_dst = scatter_mean(neigh, index, dim=0, dim_size=size_i)  # (N_dst, F_neigh)
            mean = mean_per_dst[index]  # (E, F_neigh)
        elif normalize == "anchor":
            mean = torch.cat([x_i, pos_i], dim=-1)

        # Compute the difference between the neighbor features and the mean of dst nodes,
        # of shape $(E, F_neigh)$.
        diff = neigh - mean

        num_batches = int(batch_i.max().item()) + 1 if batch_i.numel() > 0 else 1
        # Compute the standard deviation per batch over all elements of $diff$ belonging to that batch.
        # std_b is the standard deviation per batch, of shape $(B,)$.
        std_b = batch_scatter_std(
            diff,
            batch_i,
            num_batches=num_batches,
            unbiased=True,
            eps=self.eps,
        )

        std = std_b[batch_i].unsqueeze(-1)  # (E, 1)
        neigh = diff / (std + self.eps)
        neigh = self.affine_alpha * neigh + self.affine_beta

        msg = torch.cat([neigh, x_i], dim=-1)  # (E, (C[+d]) + C)
        return self.local_nn(msg)

    def extra_repr(self) -> str:
        return f"{self.local_nn}, channels={self.channels}, spatial_dim={self.spatial_dim}, normalize={self.normalize}"

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.extra_repr()})"
