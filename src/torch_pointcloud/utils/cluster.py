from typing import TYPE_CHECKING, List, Optional, Union

import torch
import torch.nn.functional as F
from torch import Tensor

from torch_pointcloud.config import FPS_RANDOM_START

from .imports import optional_import
from .types import OptTensor

if TYPE_CHECKING:
    import torch_cluster
    from torch_scatter import scatter_min

scatter_min, _ = optional_import("torch_scatter", name="scatter_min")
torch_cluster, _ = optional_import("torch_cluster")


def knn(
    x: Tensor,
    y: Tensor,
    k: int,
    batch_x: OptTensor = None,
    batch_y: OptTensor = None,
    cosine: bool = False,
    num_workers: int = 1,
    batch_size: Optional[int] = None,
) -> Tensor:
    r"""Find the $k$ nearest neighbors in $x$ for each point in $y$.
    This function is a wrapper around the `torch_cluster.knn` function, and supports the same arguments.
    However, in case the `batch_x` and `batch_y` tensors are provided, and the samples have the same number of nodes,
    this function uses a more efficient implementation that is significantly faster on GPU using `torch.cdist` + `topk`.

    Args:
        x: The source tensor to find the nearest neighbors of shape $(N, *)$.
        y: The target tensor to find the nearest neighbors of shape $(M, *)$.
        k: The number of nearest neighbors to find.
        batch_x: The batch tensor of the source tensor of shape $(N,)$.
        batch_y: The batch tensor of the target tensor of shape $(M,)$.
        cosine: Whether to use cosine distance.
        num_workers: The number of workers to use for the computation.
        batch_size: The batch size to use for the computation.

    Returns:
        The nearest neighbors of shape $(2, M*k)$.
    """

    def _torch_cluster_knn() -> Tensor:
        return torch_cluster.knn(
            x=x,
            y=y,
            k=k,
            batch_x=batch_x,
            batch_y=batch_y,
            cosine=cosine,
            num_workers=num_workers,
            batch_size=batch_size,
        )

    if batch_x is None or batch_y is None:
        return _torch_cluster_knn()

    counts_x = batch_x.bincount()
    counts_y = batch_y.bincount()

    if counts_x.numel() == 0 or not (counts_x[0] == counts_x).all() or not (counts_y[0] == counts_y).all():
        return _torch_cluster_knn()

    N_x = int(counts_x[0].item())
    N_y = int(counts_y[0].item())
    B = counts_x.numel()

    x_3d = x.view(B, N_x, -1)
    y_3d = y.view(B, N_y, -1)

    if cosine:
        x_3d = F.normalize(x_3d, dim=-1)
        y_3d = F.normalize(y_3d, dim=-1)

    dist = torch.cdist(y_3d, x_3d)  # (B, N_y, N_x)
    _, idx = dist.topk(k, dim=-1, largest=False)  # (B, N_y, k)

    offsets = torch.arange(B, device=x.device).view(B, 1, 1) * N_x
    src = (idx + offsets).reshape(-1)

    offsets_y = torch.arange(B, device=x.device).view(B, 1, 1) * N_y
    dst = (torch.arange(N_y, device=x.device).view(1, N_y, 1).expand(B, N_y, k) + offsets_y).reshape(-1)

    return torch.stack([dst, src], dim=0)


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
    random_start = random_start if FPS_RANDOM_START is None else FPS_RANDOM_START

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
