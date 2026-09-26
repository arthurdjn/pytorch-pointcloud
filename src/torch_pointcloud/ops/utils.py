"""Tensor operations on packed batches: safe division, softmax, permutations, decimation, and padding."""

from typing import Optional, Tuple, Union

import torch
from torch import Tensor
from torch_geometric.utils import scatter


def safe_divide(a: Tensor, b: Tensor, /, default: Union[float, Tensor] = float("nan")) -> Tensor:
    """Safely divide two tensors, returning a default value if the denominator is zero.

    !!! note
        If the inputs are not floating point numbers,
        they will be converted to floating point numbers (float32).

    Args:
        a: The numerator tensor.
        b: The denominator tensor.
        default: The default value to return if the denominator is zero.

    Returns:
        The result of the division.

    Example:
        ```pycon
        >>> safe_divide(torch.tensor([1.0, 2.0, 3.0]), torch.tensor([1.0, 0.0, 1.0]))
        tensor([1., nan, 3.])
        >>> safe_divide(torch.tensor([1.0, 2.0, 3.0]), torch.tensor([1.0, 0.0, 1.0]), default=0.0)
        tensor([1., 0., 3.])
        >>> safe_divide(torch.tensor([1, 2, 3]), torch.tensor([1, 0, 1]), default=torch.tensor([0, 0, 0]))
        tensor([1., 0., 3.])

        ```
    """
    if not isinstance(default, Tensor):
        default = torch.full(a.shape, default, device=a.device)

    a = a if torch.is_floating_point(a) else a.float()
    b = b if torch.is_floating_point(b) else b.float()
    default = default if torch.is_floating_point(default) else default.float()
    return torch.where(b != 0, a / b, default)


def softmax(x: Tensor, batch: Tensor, dim: int = 0) -> Tensor:
    """Apply softmax on a packed x tensor.
    The x tensor is expected to be of shape $(N, *)$,
    where $N$ is the number of nodes and $*$ is the feature size.
    The `batch` tensor must be of shape $(N,)$ and must be contiguous.

    Note:
        This function is adapted from the `torch_geometric` package.

    Args:
        x: The x tensor of shape $(N, *)$.
        batch: The batch tensor of shape $(N,)$.
        dim: The dimension along which to apply the softmax.

    Returns:
        The softmaxed tensor of shape $(N, *)$.
    """
    N = batch.max() + 1
    src_max = scatter(x.detach(), batch, dim, dim_size=N, reduce="max")
    out = x - src_max.index_select(dim, batch)
    out = out.exp()
    out_sum = scatter(out, batch, dim, dim_size=N, reduce="sum") + 1e-16
    out_sum = out_sum.index_select(dim, batch)

    return out / out_sum


def first_permutation(cluster: Tensor, num_clusters: Optional[int] = None) -> Tensor:
    r"""Index of the first occurrence of each cluster id in a consecutive cluster tensor.

    The permutation returned by `consecutive_cluster` picks a backend-dependent representative per
    cluster (the last occurrence on CPU, a nondeterministic one on CUDA). This helper always picks
    the first occurrence, so `tensor[first_permutation(cluster)]` is deterministic across devices.

    Args:
        cluster: Consecutive cluster indices of shape $(N,)$ with values in $[0, V)$.
        num_clusters: Number of clusters $V$. Inferred as `cluster.max() + 1` when `None`.

    Returns:
        Long tensor of shape $(V,)$ holding, per cluster id, the smallest index in `cluster` with that id.

    Example:
        ```pycon
        >>> cluster = torch.tensor([1, 0, 1, 2, 0])
        >>> first_permutation(cluster)
        tensor([1, 0, 3])

        ```
    """
    n = cluster.numel()
    if num_clusters is None:
        num_clusters = int(cluster.max().item()) + 1 if n > 0 else 0
    perm = torch.arange(n, device=cluster.device)
    first = torch.full((num_clusters,), n, dtype=torch.long, device=cluster.device)
    return first.scatter_reduce_(0, cluster, perm, reduce="amin")


@torch.no_grad()
def decimate_indices(
    batch: Tensor, factor: float, generator: Optional[torch.Generator] = None
) -> Tuple[Tensor, Tensor]:
    """Decimate indices from a packed batch index tensor.
    This function will return the decimated indices by a given factor along with the decimated batch indices.

    Note:
        This function is similar to the `decimation_indices` function in the `torch-geometric` package,
        except that this function uses the `batch` tensor instead of the `ptr` tensor representation.

    Args:
        batch: The packed batch index tensor.
        factor: The factor to decimate the indices by.
        generator: The generator to use for the random permutation. When given, it is reseeded from each
            sample's own point count, so a sample's decimation does not depend on the rest of the batch.

    Returns:
        The decimated indices and the decimated batch indices.

    Examples:
        ```pycon
        >>> batch = torch.tensor([0, 0, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 3])
        >>> decimate_indices(batch, 2)  # doctest: +SKIP
        (tensor([ 0,  4,  7,  6,  9, 10]), tensor([0, 1, 2, 2, 3, 3]))

        ```
    """
    if factor < 1:
        raise ValueError(
            f"The argument `factor` should be higher than (or equal to) 1 for downsampling, but got {factor}"
        )

    decim_indices = []
    decim_batch = []

    for i in torch.unique(batch).tolist():
        mask_i = batch == i
        size_i = int(mask_i.sum().item())

        # NOTE: Get at least one point to avoid empty decimation
        decim_size = max(1, int(size_i // factor))

        indices = torch.where(mask_i)[0]
        if generator is not None:
            # Reseed per sample: one generator drawn sequentially would make a sample's permutation
            # depend on how many draws the samples before it in the batch consumed.
            generator.manual_seed(size_i)

        perm = torch.randperm(size_i, device=batch.device, generator=generator)[:decim_size]

        # Decimate indices following the random permutation
        decim_indices.append(indices[perm])
        # Add batch indices for the decimated points
        decim_batch.append(torch.full((decim_size,), i, device=batch.device))

    return torch.cat(decim_indices), torch.cat(decim_batch)


def decimate(
    tensors: Tuple[Tensor, ...],
    batch: Tensor,
    factor: int,
    generator: Optional[torch.Generator] = None,
) -> Tuple[Tuple[Tensor, ...], Tensor]:
    """Decimates each input tensor by the given factor.
    This will return the decimated tensors along with the decimated batch indices.

    Note:
        This function is similar to the `decimate` function introduced in the `torch-geometric` RandLANet example.

    Args:
        tensors: A tuple of tensors to decimate.
        batch: The batch tensor of shape $(N,)$.
        factor: The factor to decimate the tensors by.
        generator: The generator to use for the random permutation.

    Returns:
        A tuple of decimated tensors and the decimated batch indices.

    Examples:
        ```pycon
        >>> tensors = (torch.randn(10, 3), torch.randn(10, 4))
        >>> batch = torch.tensor([0, 1, 1, 1, 2, 2, 2, 2, 3, 3])
        >>> decimate(tensors, batch, 2)  # doctest: +SKIP
        ((tensor([[-1.4570, -0.1023, -0.5992],
                [ 0.2408,  0.1325,  0.7642],
                [-0.2104, -1.4391,  0.5214],
                [ 1.6192,  1.4506,  0.2695],
                [ 0.3488,  0.9676, -0.4657]]),
        tensor([[-0.1933,  0.6526, -1.9006,  0.2286],
                [ 1.2888,  0.0523, -1.5469,  0.7567],
                [ 0.9442, -0.1849,  1.0608,  0.2083],
                [ 0.4788,  1.3537, -0.1593, -0.4249],
                [ 1.3065,  0.4598,  0.2618, -0.7599]])),
        tensor([0, 1, 2, 2, 3]))

        ```
    """
    idx_decim, batch_decim = decimate_indices(batch, factor, generator=generator)
    tensors_decim = tuple(tensor[idx_decim] for tensor in tensors)
    return tensors_decim, batch_decim


def pad_tail(tensor: Tensor, pad_size: int, dim: int, fill_value: float = 0) -> Tensor:
    r"""Pad the tail of a tensor with a fill value.

    Args:
        tensor: The tensor to pad.
        pad_size: The size of the padding that will be added to the tail of the tensor.
        dim: The dimension along which to pad the tensor.
        fill_value: The value to fill the padding with.

    Returns:
        The padded tensor.

    Examples:
        ```pycon
        >>> tensor = torch.tensor([1, 2, 3])
        >>> pad_tail(tensor, pad_size=2, dim=0, fill_value=0)
        tensor([1, 2, 3, 0, 0])

        >>> tensor = torch.tensor([[1, 2, 3], [4, 5, 6]])
        >>> pad_tail(tensor, pad_size=2, dim=0, fill_value=0)
        tensor([[1, 2, 3],
                [4, 5, 6],
                [0, 0, 0],
                [0, 0, 0]])

        >>> tensor = torch.tensor([[1, 2, 3], [4, 5, 6]])
        >>> pad_tail(tensor, pad_size=2, dim=1, fill_value=0)
        tensor([[1, 2, 3, 0, 0],
                [4, 5, 6, 0, 0]])

        ```
    """
    if pad_size < 0:
        raise ValueError(f"The padding size must be non-negative, but got {pad_size}.")
    elif pad_size == 0:
        return tensor

    tail_shape = list(tensor.shape)
    tail_shape[dim] = pad_size
    tail = tensor.new_full(tail_shape, fill_value)
    return torch.cat([tensor, tail], dim=dim)


def offset_index(index: Tensor, index_batch: Tensor, batch: Tensor) -> Tensor:
    r"""Offset per-element row indices into the packed row layout of a collated batch.

    `collate` concatenates a row map such as `inverse` or `index` as-is, so the entries of each batch element
    still address that element's own rows. This shifts every entry by the number of rows of the elements
    collated before it.

    Args:
        index: Per-element row indices.
        index_batch: Batch index of each entry of `index`, the `batch_<key>` tensor `collate` emits for `cat_keys`.
        batch: Batch index of the rows `index` addresses (`batch` for an `inverse` map, `batch_origin_pos` for
            an `index` map).

    Returns:
        The offset row indices.

    Shape:
        - `index`: $(M,)$
        - `index_batch`: $(M,)$
        - `batch`: $(N,)$
        - output: $(M,)$

    Example:
        ```python
        import torch
        from torch_pointcloud.ops.utils import offset_index

        inverse = torch.tensor([0, 1, 1, 0, 2, 2, 1])
        batch_inverse = torch.tensor([0, 0, 0, 1, 1, 1, 1])
        batch = torch.tensor([0, 0, 1, 1, 1])
        offset_index(inverse, batch_inverse, batch)  # tensor([0, 1, 1, 2, 4, 4, 3])
        ```
    """
    counts = torch.bincount(batch, minlength=int(index_batch.max()) + 1)
    offsets = torch.cumsum(counts, dim=0) - counts
    return index + offsets[index_batch]
