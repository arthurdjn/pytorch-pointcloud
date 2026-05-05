from typing import Any, Literal, Optional, Tuple, Union, overload

import torch
from torch import Tensor

from torch_pointcloud.utils.cluster import fps


@overload
def random_sample(
    tensor: Tensor,
    num_samples: int,
    return_indices: Literal[True] = True,
    generator: Optional[torch.Generator] = None,
) -> Tensor: ...


@overload
def random_sample(
    tensor: Tensor,
    num_samples: int,
    return_indices: bool = False,
    generator: Optional[torch.Generator] = None,
) -> Tensor: ...


def random_sample(
    tensor: Tensor,
    num_samples: int,
    return_indices: bool = False,
    generator: Optional[torch.Generator] = None,
) -> Union[Tensor, Tuple[Tensor, Tensor]]:
    """Randomly sample a fixed number of values from a tensor.

    Note:
        The data is sampled uniformly from the tensor.

    Args:
        tensor: The input tensor.
        num_samples: The number of values to sample.
        return_indices: Whether to return the indices of the sampled values.
        generator: The generator for the random number generator.

    Returns:
        If `return_indices` is `True`, the function returns a tuple of the sampled values and their indices.
        Otherwise, it returns the sampled values.
    """
    indices = torch.randint(0, tensor.size(0), (num_samples,), generator=generator)

    if return_indices:
        return tensor[indices], indices
    return tensor[indices]


@overload
def random_sample_face_vertices(
    vertices: Tensor,
    face: Tensor,
    num_samples: int,
    return_normals: Literal[True],
    generator: Optional[torch.Generator] = None,
) -> Tuple[Tensor, Tensor]: ...


@overload
def random_sample_face_vertices(
    vertices: Tensor,
    face: Tensor,
    num_samples: int,
    return_normals: Literal[False] = False,
    generator: Optional[torch.Generator] = None,
) -> Tensor: ...


@overload
def random_sample_face_vertices(
    vertices: Tensor,
    face: Tensor,
    num_samples: int,
    return_normals: bool,
    generator: Optional[torch.Generator] = None,
) -> Union[Tensor, Tuple[Tensor, Tensor]]: ...


def random_sample_face_vertices(
    vertices: Tensor,
    face: Tensor,
    num_samples: int,
    return_normals: bool = False,
    generator: Optional[torch.Generator] = None,
) -> Union[Tensor, Tuple[Tensor, Tensor]]:
    """Randomly sample a fixed number of vertices from a 3D mesh (vertices, face),
    using:

    Note:
        The data is sampled uniformly from the mesh.

    Args:
        vertices: The input tensor.
        face: The input tensor.
        num_samples: The number of vertices to sample.
        return_normals: Whether to return the normal of the sampled vertices.
        generator: The generator for the random number generator.

    Returns:
        If `return_normals` is `True`, the function returns a tuple of the sampled vertices and their normal.
        Otherwise, it returns the sampled vertices.
    """
    pos_max = vertices.abs().max()
    vertices = vertices / pos_max

    v01 = vertices[face[:, 1]] - vertices[face[:, 0]]
    v02 = vertices[face[:, 2]] - vertices[face[:, 0]]
    areas = v01.cross(v02, dim=1)
    areas = areas.norm(p=2, dim=1).abs() / 2

    probs = areas / areas.sum()
    samples = torch.multinomial(probs, num_samples, replacement=True, generator=generator)
    face = face[samples]

    frac = torch.rand(num_samples, 2, device=vertices.device, generator=generator)
    mask = frac.sum(dim=-1) > 1
    frac[mask] = 1 - frac[mask]

    v01 = vertices[face[:, 1]] - vertices[face[:, 0]]
    v02 = vertices[face[:, 2]] - vertices[face[:, 0]]

    if return_normals:
        normal = torch.nn.functional.normalize(v01.cross(v02, dim=1), p=2)

    vertices = vertices[face[:, 0]]
    vertices += frac[:, :1] * v01
    vertices += frac[:, 1:] * v02
    vertices = vertices * pos_max

    if return_normals:
        return vertices, normal
    return vertices


def sample_farthest_points(
    pos: Tensor,
    num_samples: Optional[int] = None,
    ratio: Optional[float] = None,
    random_start: bool = False,
) -> Tensor:
    """Sample the farthest points from a tensor.
    This function is a wrapper around the `torch_pointcloud.utils.cluster.fps` function,
    and is provided here for convenience.

    See Also:
        `torch_pointcloud.utils.cluster.fps` for more details and advanced usage.

    Args:
        pos: The input tensor.
        num_samples: The number of points to sample.
        ratio: The ratio of points to sample.
        random_start: Whether to start the sampling from a random point.

    Returns:
        The indices of the sampled points.

    Examples:
        >>> import torch
        >>> from torch_pointcloud.transforms.functional import sample_farthest_points
        >>> pos = torch.randn(100, 3)
        >>> idx = sample_farthest_points(pos, num_samples=10)
        >>> print(idx.shape)
        torch.Size([10])
    """
    return fps(pos, num_nodes=num_samples, ratio=ratio, random_start=random_start)


def normalize_scale(
    points: Tensor,
    eps: float = 1e-6,
    method: Literal["centroid", "bbox", "linear"] = "centroid",
) -> Tensor:
    r"""Normalize the scale of a point set along the point dimension `dim=-2`.

    Args:
        points: Tensor of shape `(..., N, C)` with $C \geq 1$; min/max and means are over $N$.
        eps: Small constant added to the scale denominator for numerical stability.
        method:

            * `"centroid"` — subtract the mean over points, then divide by
              $\max(\max_i \|\mathbf{x}_i - \mathbf{\mu}\|_2, \epsilon)$ (max Euclidean distance
              from the centroid, clamped from below by `eps`).

            * `"bbox"` — subtract the axis-aligned bounding-box midpoint (midrange center),
              then divide by half the longest edge of that box plus $\epsilon$ (matches common
              ModelNet-style normalization):

              $$
              \mathbf{c} = \frac{\mathbf{x}_{\min} + \mathbf{x}_{\max}}{2}, \quad
              r = \frac{1}{2}\max_j (x_{\max,j} - x_{\min,j}) + \epsilon, \quad
              \mathbf{x} \leftarrow \frac{\mathbf{x} - \mathbf{c}}{r}
              $$

            * `"linear"` — subtract the centroid then divide by the longest axis-aligned
              span (matches Open3D-ML's `Augmentation.normalize` `linear` method, used
              by the published RandLA-Net Toronto-3D / Semantic3D checkpoints):

              $$
              \mathbf{x} \leftarrow \frac{\mathbf{x} - \boldsymbol{\mu}}{\max_j (x_{\max,j} - x_{\min,j}) + \epsilon}
              $$

    Returns:
        Normalized tensor, same shape as `points`.

    Raises:
        ValueError: If `method` is not `"centroid"`, `"bbox"`, or `"linear"`.
    """
    if method not in ["centroid", "bbox", "linear"]:
        raise ValueError(f"Invalid method: {method!r}. Expected 'centroid', 'bbox', or 'linear'.")

    if method == "bbox":
        bbmin = points.min(dim=-2).values
        bbmax = points.max(dim=-2).values
        center = (bbmin + bbmax) / 2
        radius = (bbmax - bbmin).max() / 2
        return (points - center) / (radius + eps)

    if method == "linear":
        bbmin = points.min(dim=-2).values
        bbmax = points.max(dim=-2).values
        scale = (bbmax - bbmin).max()
        centroid = points.mean(dim=-2, keepdim=True)
        return (points - centroid) / (scale + eps)

    centroid = points.mean(dim=-2, keepdim=True)
    points = points - centroid
    scale = torch.norm(points, dim=-1, keepdim=True).max()
    return points / (scale + eps)


@overload
def divisible_pad(
    batch: Tensor,
    k: int,
    mode: Literal["below", "above", "all"] = "all",
    pad_fill: Literal["cycle", "replicate"] = "cycle",
    return_inverse: Literal[False] = False,
) -> Tuple[Tensor, Tensor]: ...


@overload
def divisible_pad(
    batch: Tensor,
    k: int,
    mode: Literal["below", "above", "all"] = "all",
    pad_fill: Literal["cycle", "replicate"] = "cycle",
    return_inverse: Literal[True] = ...,
) -> Tuple[Tensor, Tensor, Tensor]: ...


@torch.no_grad()
def divisible_pad(
    batch: Tensor,
    k: int,
    mode: Literal["below", "above", "all"] = "all",
    pad_fill: Literal["cycle", "replicate"] = "cycle",
    return_inverse: bool = False,
) -> Union[Tuple[Tensor, Tensor], Tuple[Tensor, Tensor, Tensor]]:
    """Pad the batch indices of a tensor to make them divisible by a given integer.

    Consider a batch with three samples of sizes 2, 7, and 4, and `k=4`:

    ``text
    batch:  [0 0 | 1 1 1 1 1 1 1 | 2 2 2 2]
    size:     2          7            4
    ``

    **Mode** controls *which* batches get padded (`·` = padded slot):

    ``text
    mode="all"    [0 0 · · | 1 1 1 1 1 1 1 · | 2 2 2 2]
                     2→4          7→8             4 (ok)

    mode="below"  [0 0 · · | 1 1 1 1 1 1 1 | 2 2 2 2]
                     2→4  ↑        7 (≥k)       4 (ok)
                    only <k

    mode="above"  [0 0 | 1 1 1 1 1 1 1 · | 2 2 2 2]
                    2        7→8  ↑           4 (ok)
                  (<k)      only ≥k
    ``

    **Pad fill** controls *how* padded slots are filled.  Given batch 1
    with 7 elements (`A B C D E F G`) and `k=4`:

    ``text
    Original patches:  [A B C D] [E F G ·]
                        patch₀    patch₁ (incomplete)

    pad_fill="cycle"      → [A B C D] [E F G A]
      Cycles from the start                  ↑ wraps to A

    pad_fill="replicate"  → [A B C D] [E F G D]
      Copies from previous patch             ↑ same position as D
      at same offset
    ``

    When `batch_size < k` there is no previous patch, so `"replicate"`
    falls back to `"cycle"`:

    ``text
    batch 0 (size 2, k=4):  [A B · ·]
    pad_fill="cycle"      → [A B A B]
    pad_fill="replicate"  → [A B A B]   (same — no prior patch)
    ``

    Args:
        batch: The batch indices of the tensor.
        k: The integer to make the batch indices divisible by.
        mode: The mode to use for padding.
            - `"below"`: Pad only batches with fewer than `k` elements.
            - `"above"`: Pad only batches with `k` or more elements.
            - `"all"`: Pad all batches to be divisible by `k`.
        pad_fill: Strategy for filling padding slots.
            - `"cycle"`: Cycle through original indices from the start of
              the batch (`indices[0], indices[1], ...`).
            - `"replicate"`: Copy indices from the previous patch at the
              same relative offset.  When the last group of `k` elements is
              incomplete, the missing positions are filled with the
              corresponding positions from the preceding full group.  Falls
              back to `"cycle"` when there is no preceding group (i.e. the
              batch has fewer than `k` elements).
        return_inverse: Whether to return the inverse of the padded indices.

    Returns:
        Returns a tuple of `(indices, padded_batch)`.
        If `return_inverse` is `True`, returns `(indices, inverse_indices, padded_batch)`.
    """
    if mode not in ["below", "above", "all"]:
        raise ValueError(f"Unknown mode: {mode!r}. Expected 'below', 'above', or 'all'")
    if pad_fill not in ["cycle", "replicate"]:
        raise ValueError(f"Unknown pad_fill: {pad_fill!r}. Expected 'cycle' or 'replicate'")

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

        indices[new_start : new_start + batch_size] = torch.arange(original_start, original_start + batch_size)

        if pad_size > 0:
            if pad_fill == "replicate" and batch_size > k:
                rem = batch_size % k
                last_patch_start = new_start + batch_size - rem
                prev_patch_start = last_patch_start - k
                src_start = prev_patch_start + rem
                indices[new_start + batch_size : new_start + batch_size + pad_size] = indices[
                    src_start : src_start + pad_size
                ]
            else:
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


@overload
def remove_near_origin(pos: Tensor, radius: float, return_mask: Literal[True]) -> Tuple[Tensor, Tensor]: ...


@overload
def remove_near_origin(pos: Tensor, radius: float, return_mask: Literal[False] = False) -> Tensor: ...


@overload
def remove_near_origin(pos: Tensor, radius: float, return_mask: bool) -> Union[Tensor, Tuple[Tensor, Tensor]]: ...


def remove_near_origin(pos: Tensor, radius: float = 1e-3, return_mask: bool = False) -> Any:
    """Remove points that are within a given radius of the origin.

    Args:
        pos: The input tensor.
        radius: The radius of the sphere.

    Returns:
        The tensor with the points removed.
    """
    mask = pos.norm(dim=-1) > radius
    if return_mask:
        return pos[mask], mask
    return pos[mask]


def abs(x: Tensor, inplace: bool = False) -> Tensor:
    """Make the input tensor absolute.

    Args:
        x: The input tensor.

    Returns:
        The absolute tensor.

    Examples:
        >>> import torch
        >>> import torch_pointcloud.transforms.functional as F
        >>> x = torch.tensor([-1.0, 2.0, -3.0])
        >>> F.abs(x)
        tensor([1.0, 2.0, 3.0])
    """
    if inplace:
        x.abs_()
        return x

    return x.abs()


def bounding_box(x: Tensor, dim: int = 0) -> tuple[float, ...]:
    """Returns the min and max values along a given dimension.

    Args:
        x: The input tensor of shape (..., D, ...).
        dim: The dimension to compute bounds over.

    Returns:
        A tuple of (*min, *max) values.
    """
    bbmin = x.min(dim=dim).values.detach().cpu().tolist()
    bbmax = x.max(dim=dim).values.detach().cpu().tolist()
    return (*bbmin, *bbmax)


def inbox_mask(x: Tensor, bbox: tuple[float, ...], dim: int = -1) -> Tensor:
    """Create a mask for the input tensor that is within a given bounding box.

    Args:
        x: The input tensor.
        bbox: The bounding box.
        dim: The dimension to compute the mask over.

    Returns:
        The mask.
    """
    size = len(bbox)
    if not size == x.shape[dim] * 2:
        raise ValueError(f"Bounding box size mismatch, got {size} for dimension {dim} but expected {x.shape[dim] * 2}.")

    bbmin = torch.tensor(bbox[: size // 2], device=x.device, dtype=x.dtype)
    bbmax = torch.tensor(bbox[size // 2 :], device=x.device, dtype=x.dtype)
    return (x > bbmin).all(dim=dim) & (x < bbmax).all(dim=dim)


def apply_mask(x: Tensor, mask: Tensor) -> Tensor:
    """Apply a mask to a tensor.

    Args:
        x: The input tensor.
        mask: The mask.

    Returns:
        The tensor with the mask applied.

    Examples:
        >>> import torch
        >>> import torch_pointcloud.transforms.functional as F
        >>> x = torch.tensor([1.0, 2.0, 3.0])
        >>> mask = torch.tensor([True, False, True])
        >>> F.apply_mask(x, mask)
        tensor([1.0, 3.0])
    """
    return x[mask]
