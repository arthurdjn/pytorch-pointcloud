from typing import Literal, Optional, Tuple, Union, overload

import torch
from torch import Tensor


@overload
def random_sample(
    tensor: Tensor,
    num_samples: int,
    return_indices: Literal[True] = True,
    seed: Optional[int] = None,
) -> Tensor: ...


@overload
def random_sample(
    tensor: Tensor,
    num_samples: int,
    return_indices: bool = False,
    seed: Optional[int] = None,
) -> Tensor: ...


def random_sample(
    tensor: Tensor,
    num_samples: int,
    return_indices: bool = False,
    seed: Optional[int] = None,
) -> Union[Tensor, Tuple[Tensor, Tensor]]:
    """Randomly sample a fixed number of values from a tensor.

    Note:
        The data is sampled uniformly from the tensor.

    Args:
        tensor: The input tensor.
        num_samples: The number of values to sample.
        return_indices: Whether to return the indices of the sampled values.
        seed: The seed for the random number generator.

    Returns:
        If `return_indices` is `True`, the function returns a tuple of the sampled values and their indices.
        Otherwise, it returns the sampled values.
    """
    rng = None
    if seed is not None:
        rng = torch.Generator(device=tensor.device)
        rng.manual_seed(seed)

    indices = torch.randint(0, tensor.size(0), (num_samples,), generator=rng)

    if return_indices:
        return tensor[indices], indices
    return tensor[indices]


@overload
def random_sample_face_vertices(
    vertices: Tensor,
    faces: Tensor,
    num_samples: int,
    return_normals: Literal[True] = True,
    return_indices: Literal[True] = True,
    seed: Optional[int] = None,
) -> Tuple[Tensor, Tensor, Tensor]: ...


@overload
def random_sample_face_vertices(
    vertices: Tensor,
    faces: Tensor,
    num_samples: int,
    return_normals: Literal[True] = True,
    return_indices: bool = False,
    seed: Optional[int] = None,
) -> Tuple[Tensor, Tensor]: ...


@overload
def random_sample_face_vertices(
    vertices: Tensor,
    faces: Tensor,
    num_samples: int,
    return_normals: bool = False,
    return_indices: bool = False,
    seed: Optional[int] = None,
) -> Tensor: ...


def random_sample_face_vertices(
    vertices: Tensor,
    faces: Tensor,
    num_samples: int,
    return_normals: bool = False,
    return_indices: bool = False,
    seed: Optional[int] = None,
) -> Union[Tensor, Tuple[Tensor, Tensor], Tuple[Tensor, Tensor, Tensor]]:
    """Randomly sample a fixed number of vertices from a 3D mesh (vertices, faces),
    using:

    Note:
        The data is sampled uniformly from the mesh.

    Args:
        vertices: The input tensor.
        faces: The input tensor.
        num_samples: The number of vertices to sample.
        return_normals: Whether to return the normals of the sampled vertices.
        return_indices: Whether to return the indices of the sampled vertices.
        seed: The seed for the random number generator.

    Returns:
        If `return_indices` is `True`, the function returns a tuple of the sampled vertices and their indices.
        Otherwise, it returns the sampled vertices.
        If `return_normals` is `True`, the function returns a tuple of the sampled vertices and their normals.
        Otherwise, it returns the sampled vertices.
    """
    rng = torch.Generator()
    if seed is not None:
        rng.manual_seed(seed)

    pos_max = vertices.abs().max()
    vertices = vertices / pos_max

    v01 = vertices[faces[:, 1]] - vertices[faces[:, 0]]
    v02 = vertices[faces[:, 2]] - vertices[faces[:, 0]]
    areas = v01.cross(v02, dim=1)
    areas = areas.norm(p=2, dim=1).abs() / 2

    probs = areas / areas.sum()
    samples = torch.multinomial(probs, num_samples, replacement=True, generator=rng)
    faces = faces[samples]

    frac = torch.rand(num_samples, 2, device=vertices.device, generator=rng)
    mask = frac.sum(dim=-1) > 1
    frac[mask] = 1 - frac[mask]

    v01 = vertices[faces[:, 1]] - vertices[faces[:, 0]]
    v02 = vertices[faces[:, 2]] - vertices[faces[:, 0]]

    if return_normals:
        normals = torch.nn.functional.normalize(v01.cross(v02, dim=1), p=2)

    vertices = vertices[faces[:, 0]]
    vertices += frac[:, :1] * v01
    vertices += frac[:, 1:] * v02
    vertices = vertices * pos_max

    if return_indices and return_normals:
        return vertices, normals, faces[:, 0]
    elif return_normals:
        return vertices, normals
    elif return_indices:
        return vertices, faces[:, 0]
    return vertices


def normalize_scale(points: Tensor, eps: float = 1e-8) -> Tensor:
    r"""Normalize the scale of a 3D tensor as follows:

    $$
    \mathbf{x} = \frac{\mathbf{x} - \mathbf{\mu}}{\max(\sqrt{\sum_{i=1}^3 x_i^2}, \epsilon)}
    $$

    Note:
        The data is normalized to have a unit scale.

    Args:
        points: The input tensor.
        eps: The epsilon value to avoid division by zero.

    Returns:
        The normalized tensor.
    """
    points -= points.mean(dim=-2, keepdim=True)
    points = points / (points.abs().max() + eps)
    return points


@torch.no_grad()
def divisible_pad(
    batch: Tensor,
    k: int,
    mode: Literal["below", "above", "all"] = "all",
    return_inverse: bool = False,
) -> Union[Tuple[Tensor, Tensor], Tuple[Tensor, Tensor, Tensor]]:
    """Pad the batch indices of a tensor to make them divisible by a given integer.

    Args:
        batch: The batch indices of the tensor.
        k: The integer to make the batch indices divisible by.
        mode: The mode to use for padding.
            - "below": Pad the batch indices to be below the given integer.
            - "above": Pad the batch indices to be above the given integer.
            - "all": Pad the batch indices to be divisible by the given integer.
        return_inverse: Whether to return the inverse of the padded indices.

    Returns:
        Returns a tuple of the indices to be padded, the padded batch indices.
        If `return_inverse` is `True`, the function returns a tuple of the indices to be padded, the inverse of the padded indices,
        and the padded batch indices.

    Examples:
        In its minimal setting, the function can be used to pad batches to make them divisible by a given integer.
        The below example shows how to pad batches of points to make them divisible by 5.
        >>> import torch
        >>> points = torch.randn(100, 3)
        >>> batch = torch.randint(0, 10, (100,)).sort().values
        >>> padded_idxs, padded_batch = divisible_pad(batch, k=5)
        >>> padded_points = points[padded_idxs]
        To revert the padding operation, you can use the inverse indices.
        >>> padded_idxs, inverse, padded_batch = divisible_pad(batch, k=5, return_inverse=True)
        >>> padded_points = points[padded_idxs]
        >>> original_points = points[inverse]  # Should be the same as `points`
        In some cases, you might want to pad batches only if they contain less than k points.
        You can achieve this by setting `mode` to `"below"`.
        > [!NOTE]
        > If you want to pad batches only if they contain more than k points,
        > you can set `mode` to `"above"`.
        >>> padded_idxs, padded_batch, inverse = divisible_pad(batch, k=5, mode="below", return_inverse=True)
        >>> padded_points = points[padded_idxs]
    """
    if mode not in ["below", "above", "all"]:
        raise ValueError(f"Unknown mode: {mode!r}. Expected 'below', 'above', or 'all'")

    device = batch.device

    # Get total (unique) batches and their counts
    # NOTE: using .unique() instead of .bincount() ensures that we can handle non-consecutive batch indices
    unique_batches, counts = torch.unique(batch, return_counts=True)
    num_batches = len(unique_batches)

    # Calculate required padding for each batch such that each batch is a multiple of k
    remainder = counts % k
    padding_needed = torch.zeros_like(remainder)

    if mode == "all":
        padding_needed[remainder > 0] = k - remainder[remainder > 0]
    elif mode == "below":
        mask = (counts < k) & (remainder > 0)
        padding_needed[mask] = k - remainder[mask]
    elif mode == "above":
        mask = (counts >= k) & (remainder > 0)
        padding_needed[mask] = k - remainder[mask]

    # Calculate new (padded) batch sizes with their starting indices
    # so that we can map original indices and batch to their padded counterparts
    new_batch_sizes = counts + padding_needed
    batch_start_idx = torch.cat([torch.tensor([0], device=device), torch.cumsum(counts, dim=0)[:-1]])
    new_batch_start_idx = torch.cat([torch.tensor([0], device=device), torch.cumsum(new_batch_sizes, dim=0)[:-1]])

    # Create indices and new batch tensors
    total_new_size = int(torch.sum(new_batch_sizes).item())
    indices = torch.zeros(total_new_size, dtype=torch.long, device=device)
    inverse_indices = torch.zeros(len(batch), dtype=torch.long, device=device)
    padded_batch = torch.zeros(total_new_size, dtype=batch.dtype, device=device)

    for i in range(num_batches):
        original_start = int(batch_start_idx[i].item())
        new_start = int(new_batch_start_idx[i].item())
        pad_size = int(padding_needed[i].item())
        batch_size = int(counts[i].item())

        # First, we can safely assign the first elements of the padded batch that do not require padding
        indices[new_start : new_start + batch_size] = torch.arange(original_start, original_start + batch_size)

        if pad_size > 0:
            # ...but pad with repeated values if needed (cycle through original indices)
            original_indices = torch.arange(original_start, original_start + batch_size)
            cycle_indices = original_indices[torch.arange(pad_size) % batch_size]
            indices[new_start + batch_size : new_start + batch_size + pad_size] = cycle_indices

        inverse_indices[original_start : original_start + batch_size] = torch.arange(new_start, new_start + batch_size)
        padded_batch[new_start : new_start + new_batch_sizes[i]] = unique_batches[i]

    if return_inverse:
        return indices, inverse_indices, padded_batch

    return indices, padded_batch


@torch.no_grad()
def split_batch(batch: Tensor, max_size: int) -> Tensor:
    """Split batches into multiple sub-batches of a given size.

    Note:
        The batch is only splitted if it is larger than the given size.
        If not, the batch is returned as is.

    Note:
        If you want to split batches smaller than the given size,
        you can use the `divisible_pad` function before splitting the batch.

    Args:
        batch: The batch indices of the points.
        max_size: The maximum size of the sub-batches.

    Returns:
        The sub-batch indices.

    Examples:
        >>> import torch
        >>> batch = torch.tensor([0, 0, 0, 1, 1, 1, 1, 2, 2, 3])
        >>> split_batch(batch, size=2)
        tensor([0, 0, 1, 2, 2, 3, 3, 4, 4, 5])
    """
    device = batch.device
    _, batch_counts = torch.unique(batch, return_counts=True)
    sub_counts = torch.div(batch_counts + max_size - 1, max_size, rounding_mode="floor")
    sub_offsets = torch.cumsum(torch.cat([torch.zeros(1, device=device, dtype=torch.long), sub_counts[:-1]]), dim=0)
    sub_idxs = torch.zeros_like(batch)

    offset = 0
    for i, batch_count in enumerate(batch_counts):
        idxs = slice(offset, offset + batch_count)
        # Get the relative sub-batch indices (starting from 0)
        relative_sub_idxs = torch.div(torch.arange(batch_count, device=device), max_size, rounding_mode="floor")
        # Assign the relative sub-batch indices,
        # making sure they are contiguous from already assigned sub-batches
        sub_idxs[idxs] = relative_sub_idxs + sub_offsets[i]
        offset += batch_count

    return sub_idxs
