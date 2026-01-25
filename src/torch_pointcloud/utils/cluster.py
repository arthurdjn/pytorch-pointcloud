from typing import TYPE_CHECKING, List, Optional, Union

import torch
from torch import Tensor

from .imports import optional_import

if TYPE_CHECKING:
    import torch_cluster
    from torch_scatter import scatter_min

scatter_min, _ = optional_import("torch_scatter", name="scatter_min")
torch_cluster, _ = optional_import("torch_cluster")


def fps(
    src: Tensor,
    batch: Optional[Tensor] = None,
    ratio: Optional[Union[Tensor, float]] = None,
    num_nodes: Optional[Union[Tensor, float]] = None,
    random_start: bool = True,
    batch_size: Optional[int] = None,
    ptr: Optional[Union[Tensor, List[int]]] = None,
) -> Tensor:
    r"""A sampling algorithm from the paper [PointNet++: Deep Hierarchical Feature
    Learning on Point Sets in a Metric Space](https://arxiv.org/abs/1706.02413)
    by Qi et al., which iteratively samples the most distant point with regard
    to the rest points.

    This function is adapted from the `torch_cluster.fps` function and supports a sampling a
    fixed number of nodes with the `num_nodes` argument.

    Important:
        If provided, the `batch` tensor is expected to be sorted.

    Args:
        src: The source tensor to sample from of shape $(N, *)$.
        batch: The batch tensor to sample from of shape $(N,)$.
        ratio: The sampling ratio.
        num_nodes: The number of nodes to sample.
        random_start: Whether to start the sampling randomly.
        batch_size: The batch size.
        ptr: The pointer tensor to sample from.

    Returns:
        The sampled indices of shape $(M,)$.

    Examples:
        >>> import torch
        >>> from torch_pointcloud.utils.cluster import fps
        >>> src = torch.randn(100, 3)
        >>> batch = torch.cat([torch.zeros(50), torch.ones(50)]).long()
        >>> idx = fps(src, batch, num_nodes=10)
        >>> print(idx.shape)
        torch.Size([10])
    """
    if ratio is None and num_nodes is None:
        raise ValueError("Either `ratio` or `num_nodes` must be provided.")
    if ratio is not None and num_nodes is not None:
        raise ValueError("Only one of `ratio` or `num_nodes` can be provided.")
    if ratio is not None:
        return torch_cluster.fps(
            src,
            batch=batch,
            ratio=ratio,
            random_start=random_start,
            batch_size=batch_size,
            ptr=ptr,
        )

    if ptr is not None:
        ptr = torch.as_tensor(ptr, dtype=torch.long, device=src.device)
        node_counts = ptr[1:] - ptr[:-1]
        if batch_size is None:
            batch_size = node_counts.numel()
    else:
        if batch is not None:
            if batch_size is None:
                batch_size = int(batch.max()) + 1
            if src.size(0) != batch.numel():
                raise ValueError(f"Size of `src` ({src.size(0)}) must match size of `batch` ({batch.numel()}).")

            node_counts = batch.bincount(minlength=batch_size)
            ptr = torch.cat([torch.zeros(1, dtype=torch.long, device=src.device), node_counts.cumsum(0)])
        else:
            node_counts = torch.tensor([src.size(0)], device=src.device)
            batch_size = 1
            ptr = torch.tensor([0, src.size(0)], device=src.device)

    if isinstance(num_nodes, (int, float)):
        req_nodes = torch.full((batch_size,), int(num_nodes), dtype=torch.long, device=src.device)
    else:
        req_nodes = torch.as_tensor(num_nodes, dtype=torch.long, device=src.device)
        if req_nodes.ndim == 0:
            req_nodes = req_nodes.expand(batch_size)
        elif req_nodes.size(0) != batch_size:
            raise ValueError(f"Size of `num_nodes` ({req_nodes.size(0)}) must match batch size ({batch_size}).")

    req_ratios = req_nodes.float() / node_counts.float()
    idx = torch_cluster.fps(src, ratio=req_ratios, random_start=random_start, ptr=ptr)

    sampled_batch = torch.searchsorted(ptr, idx, right=True) - 1
    sampled_counts = sampled_batch.bincount(minlength=batch_size)
    offsets = torch.cat([torch.zeros(1, dtype=torch.long, device=src.device), sampled_counts.cumsum(0)])

    local_rank = torch.arange(idx.numel(), device=src.device) - offsets[sampled_batch]
    mask = local_rank < req_nodes[sampled_batch]
    return idx[mask]


def local_grid(src: Tensor, size: float, batch: Tensor | None = None) -> Tensor:
    r"""Applies local grid quantization to the source tensor as explained in the paper
    [TorchSparse++: Efficient node Cloud Engine](https://openaccess.thecvf.com/content/CVPR2023W/WAD/papers/Tang_TorchSparse_Efficient_node_Cloud_Engine_CVPRW_2023_paper.pdf)
    by Tang et al., which quantizes the source tensor to a local grid.

    Note:
        If a batch tensor is provided, the function will apply the quantization to each batch separately,
        ensuring the

    Args:
        src: The source tensor to quantize of shape $(N, *)$.
        size: The quantization size.
        batch: The associated batch tensor of shape $(N,)$.

    Returns:
        The quantized source tensor of shape $(N, *)$.

    Examples:
        >>> import torch
        >>> from torch_pointcloud.utils.cluster import local_grid
        >>> src = torch.randn(100, 3)
        >>> batch = torch.cat([torch.zeros(50), torch.ones(50)]).long()
        >>> src_grid = local_grid(src, size=1.0, batch=batch)
    """

    src_quantized = torch.div(src, size, rounding_mode="floor").long()

    # If the batch is not provided we compute the global shift the source tensor to the origin (faster)
    # Otherwise we compute the local shifts for each graph contained in the batch (slower)
    if batch is None:
        src_min, _ = src_quantized.min(0)
    else:
        src_min, _ = scatter_min(src_quantized, batch, dim=0)
        src_min = src_min[batch]

    return src_quantized - src_min
