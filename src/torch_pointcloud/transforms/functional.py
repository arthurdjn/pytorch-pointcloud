from typing import Any, Dict, Literal, Optional, Sequence, Tuple, Union, get_args, overload

import torch
from torch import Tensor

from torch_pointcloud.utils.cluster import fps

ShiftMethod = Literal["bbox", "centroid", "min"]

RescaleMethod = Literal["centroid", "bbox", "linear"]

PadMode = Literal["below", "above", "all"]

PadFill = Literal["cycle", "replicate"]


@overload
def random_sample(
    tensor: Tensor,
    num_samples: int,
    return_indices: Literal[True],
    replace: bool = False,
    generator: Optional[torch.Generator] = None,
) -> Tuple[Tensor, Tensor]: ...


@overload
def random_sample(
    tensor: Tensor,
    num_samples: int,
    return_indices: Literal[False] = False,
    replace: bool = False,
    generator: Optional[torch.Generator] = None,
) -> Tensor: ...


def random_sample(
    tensor: Tensor,
    num_samples: int,
    return_indices: bool = False,
    replace: bool = False,
    generator: Optional[torch.Generator] = None,
) -> Union[Tensor, Tuple[Tensor, Tensor]]:
    r"""Randomly sample a fixed number of values from a tensor.

    Note:
        The data is sampled uniformly along `dim=0`.

    Args:
        tensor: The input tensor of shape $(N, \ldots)$.
        num_samples: The number of values to sample.
        return_indices: Whether to return the indices of the sampled values.
        replace: If `True`, sample with replacement (duplicates allowed). If `False`,
            sample without replacement and raise `ValueError` when `num_samples > N`.
        generator: The generator for the random number generator.

    Returns:
        If `return_indices` is `True`, the function returns a tuple of the sampled values and their indices.
        Otherwise, it returns the sampled values.

    Raises:
        ValueError: If `replace=False` and `num_samples > tensor.size(0)`,
            or if `num_samples > 0` and the input is empty.
    """
    n = tensor.size(0)
    if num_samples == 0:
        indices = torch.empty(0, dtype=torch.long, device=tensor.device)
    elif n == 0:
        raise ValueError(f"Cannot sample {num_samples} values from an empty tensor (N=0).")
    elif replace:
        indices = torch.randint(0, n, (num_samples,), generator=generator, device=tensor.device)
    elif num_samples <= n:
        indices = torch.randperm(n, generator=generator, device=tensor.device)[:num_samples]
    elif num_samples > n:
        raise ValueError(
            f"Requested {num_samples} samples without replacement from a tensor of size {n}. "
            "Pass `replace=True` to allow duplicates."
        )

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


def farthest_point_sample(
    pos: Tensor,
    num_samples: Optional[int] = None,
    ratio: Optional[float] = None,
    random_start: bool = False,
) -> Tensor:
    """Farthest-point sampling (FPS) from a tensor of positions.

    Thin wrapper around `torch_pointcloud.utils.cluster.fps`, provided for
    convenience and naming symmetry with `random_sample`.

    See Also:
        `torch_pointcloud.utils.cluster.fps` for more details and advanced usage.

    Args:
        pos: The input tensor of shape $(N, D)$.
        num_samples: The number of points to sample.
        ratio: The ratio of points to sample.
        random_start: Whether to start the sampling from a random point.

    Returns:
        The indices of the sampled points.

    Examples:
        >>> import torch
        >>> from torch_pointcloud.transforms.functional import farthest_point_sample
        >>> pos = torch.randn(100, 3)
        >>> idx = farthest_point_sample(pos, num_samples=10)
        >>> print(idx.shape)
        torch.Size([10])
    """
    return fps(pos, num_nodes=num_samples, ratio=ratio, random_start=random_start)


def rescale(
    points: Tensor,
    eps: float = 1e-6,
    method: RescaleMethod = "centroid",
) -> Tensor:
    r"""Center a point set and rescale it to a unit extent.

    Operates along the point dimension `dim=-2`. Pairs a centering step with a
    scale-by-extent step that share the same statistics.

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
    if method not in get_args(RescaleMethod):
        raise ValueError(f"Invalid method: {method!r}. Expected one of {get_args(RescaleMethod)}.")

    if points.shape[-2] == 0:
        return points

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
    mode: PadMode = "all",
    pad_fill: PadFill = "cycle",
    return_inverse: Literal[False] = False,
) -> Tuple[Tensor, Tensor]: ...


@overload
def divisible_pad(
    batch: Tensor,
    k: int,
    mode: PadMode = "all",
    pad_fill: PadFill = "cycle",
    return_inverse: Literal[True] = ...,
) -> Tuple[Tensor, Tensor, Tensor]: ...


@torch.no_grad()
def divisible_pad(
    batch: Tensor,
    k: int,
    mode: PadMode = "all",
    pad_fill: PadFill = "cycle",
    return_inverse: bool = False,
) -> Union[Tuple[Tensor, Tensor], Tuple[Tensor, Tensor, Tensor]]:
    """Pad the batch indices of a tensor to make them divisible by a given integer.

    Consider a batch with three samples of sizes 2, 7, and 4, and `k=4`:

    ```text
    batch:  [0 0 | 1 1 1 1 1 1 1 | 2 2 2 2]
    size:     2          7            4
    ```

    **Mode** controls *which* batches get padded (`·` = padded slot):

    ```text
    mode="all"    [0 0 · · | 1 1 1 1 1 1 1 · | 2 2 2 2]
                     2→4          7→8             4 (ok)

    mode="below"  [0 0 · · | 1 1 1 1 1 1 1 | 2 2 2 2]
                     2→4  ↑        7 (≥k)       4 (ok)
                    only <k

    mode="above"  [0 0 | 1 1 1 1 1 1 1 · | 2 2 2 2]
                    2        7→8  ↑           4 (ok)
                  (<k)      only ≥k
    ```

    **Pad fill** controls *how* padded slots are filled.  Given batch 1
    with 7 elements (`A B C D E F G`) and `k=4`:

    ```text
    Original patches:  [A B C D] [E F G ·]
                        patch₀    patch₁ (incomplete)

    pad_fill="cycle"      → [A B C D] [E F G A]
      Cycles from the start                  ↑ wraps to A

    pad_fill="replicate"  → [A B C D] [E F G D]
      Copies from previous patch             ↑ same position as D
      at same offset
    ```

    When `batch_size < k` there is no previous patch, so `"replicate"`
    falls back to `"cycle"`:

    ```text
    batch 0 (size 2, k=4):  [A B · ·]
    pad_fill="cycle"      → [A B A B]
    pad_fill="replicate"  → [A B A B]   (same — no prior patch)
    ```

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
    if mode not in get_args(PadMode):
        raise ValueError(f"Unknown mode: {mode!r}. Expected one of {get_args(PadMode)}.")
    if pad_fill not in get_args(PadFill):
        raise ValueError(f"Unknown pad_fill: {pad_fill!r}. Expected one of {get_args(PadFill)}.")

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
    """Remove points that are within a given radius (L2) of the origin.

    Equivalent to inverting `sphere_mask(pos, center=0, radius=r)` and indexing.

    Args:
        pos: The input tensor of shape $(N, D)$.
        radius: The L2 radius (Euclidean distance) below which points are removed.
        return_mask: If `True`, also return the keep-mask.

    Returns:
        The filtered tensor; or `(filtered, mask)` if `return_mask=True`.
    """
    center = pos.new_zeros(pos.shape[-1])
    mask = ~sphere_mask(pos, center, radius, dim=-1)
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


def box_mask(x: Tensor, bbox: tuple[float, ...], dim: int = -1) -> Tensor:
    r"""Create a boolean mask for points inside an axis-aligned bounding box (AABB).

    Membership condition along `dim`:

    $$
    \text{bbmin}_j < x_j < \text{bbmax}_j \quad \forall j
    $$

    Args:
        x: The input tensor of shape `(..., D)` along `dim`.
        bbox: AABB as a flat tuple `(*bbmin, *bbmax)` of length `2 * D`.
        dim: The dimension to compute the mask over.

    Returns:
        The boolean mask, with `dim` reduced.

    Raises:
        ValueError: If `len(bbox) != 2 * x.shape[dim]`.
    """
    size = len(bbox)
    if not size == x.shape[dim] * 2:
        raise ValueError(f"Bounding box size mismatch, got {size} for dimension {dim} but expected {x.shape[dim] * 2}.")

    bbmin = torch.tensor(bbox[: size // 2], device=x.device, dtype=x.dtype)
    bbmax = torch.tensor(bbox[size // 2 :], device=x.device, dtype=x.dtype)
    return (x > bbmin).all(dim=dim) & (x < bbmax).all(dim=dim)


def cube_mask(
    x: Tensor,
    center: Union[Tensor, Sequence[float], float],
    radius: float,
    dim: int = -1,
) -> Tensor:
    r"""Create a boolean mask for points inside an axis-aligned cube (L∞ / Chebyshev ball).

    Membership condition along `dim`:

    $$
    \| x - c \|_{\infty} \leq r
    $$

    Geometrically, the L∞ ball of radius $r$ centered at $c$ is a hypercube
    with edge $2r$ aligned to the axes. Pair with `sphere_mask` (L2) and
    `box_mask` (explicit AABB).

    Args:
        x: The input tensor of shape `(..., D)` along `dim`.
        center: The center of the cube, shape `(D,)` or broadcastable.
        radius: The half-edge (radius) of the cube.
        dim: The dimension to reduce the per-axis comparison over.

    Returns:
        The boolean mask, with `dim` reduced.
    """
    center_t = torch.as_tensor(center, device=x.device, dtype=x.dtype)
    return (x - center_t).abs().amax(dim=dim) <= radius


def sphere_mask(
    x: Tensor,
    center: Union[Tensor, Sequence[float], float],
    radius: float,
    dim: int = -1,
) -> Tensor:
    r"""Create a boolean mask for points inside an L2 (Euclidean) ball.

    Membership condition along `dim`:

    $$
    \| x - c \|_2 \leq r
    $$

    Pair with `cube_mask` (L∞) and `box_mask` (explicit AABB).

    Args:
        x: The input tensor of shape `(..., D)` along `dim`.
        center: The center of the sphere, shape `(D,)` or broadcastable.
        radius: The radius of the sphere.
        dim: The dimension to compute the Euclidean norm over.

    Returns:
        The boolean mask, with `dim` reduced.
    """
    center_t = torch.as_tensor(center, device=x.device, dtype=x.dtype)
    return (x - center_t).norm(dim=dim) <= radius


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


def shift(
    x: Tensor,
    method: ShiftMethod,
    dim: int = 0,
    axes: Optional[Sequence[int]] = None,
) -> Tensor:
    r"""Subtract a data-driven offset from `x`.

    The offset is computed from `x` itself along the reduction dimension `dim`:

    | `method`     | Offset                                           |
    | ------------ | ------------------------------------------------ |
    | `"bbox"`     | Midrange: `(min + max) / 2`                      |
    | `"centroid"` | Mean across the reduced dimension                |
    | `"min"`      | Per-axis minimum (shifts to the positive octant) |

    When `axes` is given, only those axis-indices of the offset are non-zero,
    so axes not listed are left untouched. This is the composable knob for
    mixed-method shifts:

    ```python
    # Pointcept-style centering: XY by bbox-mid, Z by min
    x = F.shift(x, method="bbox", axes=[0, 1])
    x = F.shift(x, method="min",  axes=[2])
    ```

    The two calls touch disjoint axes, so they commute.

    Args:
        x: Input tensor.
        method: How the offset is computed. See the table.
        dim: The dimension to reduce over when computing the offset.
        axes: Last-dim axis indices to shift. `None` (default) shifts every axis.

    Returns:
        The shifted tensor, same shape as `x`. Returns `x` unchanged when
        `x.size(dim) == 0`.

    Raises:
        ValueError: If `method` is not one of `"bbox"`, `"centroid"`, `"min"`.
    """
    if method not in get_args(ShiftMethod):
        raise ValueError(f"Invalid method: {method!r}. Expected one of {get_args(ShiftMethod)}.")
    if x.size(dim) == 0:
        return x
    if method == "bbox":
        offset = (x.min(dim=dim).values + x.max(dim=dim).values) / 2
    elif method == "centroid":
        offset = x.mean(dim=dim)
    else:  # "min"
        offset = x.min(dim=dim).values
    if axes is not None:
        full_offset = torch.zeros_like(offset)
        axes_idx = torch.tensor(tuple(axes), device=offset.device, dtype=torch.long)
        full_offset.index_copy_(0, axes_idx, offset.index_select(0, axes_idx))
        offset = full_offset
    return x - offset


def axis_min_offset(x: Tensor, axis: int) -> Tensor:
    r"""Per-point offset from the minimum along a chosen coordinate axis.

    For positions of shape $(N, D)$ and an axis $a \in [0, D)$, returns a
    tensor of shape $(N, 1)$ whose entries are $x_{i, a} - \min_j x_{j, a}$.
    Useful for extracting "height above the local floor" as a per-point feature.

    Args:
        x: Input tensor of shape `(N, D)`.
        axis: Axis index in the last dimension.

    Returns:
        Tensor of shape `(N, 1)` with the same dtype as `x`. Returns an empty
        `(0, 1)` tensor when `x` is empty.
    """
    col = x[:, axis]
    if col.numel() == 0:
        return col.unsqueeze(-1).to(x.dtype)
    return (col - col.min()).unsqueeze(-1).to(x.dtype)


def normalize(
    x: Tensor,
    mean: Union[Tensor, Sequence[float], float],
    std: Union[Tensor, Sequence[float], float],
    eps: float = 1e-7,
) -> Tensor:
    r"""Per-channel standardization: $x' = (x - \mu) / \max(\sigma, \epsilon)$.

    Args:
        x: Input tensor. The last dimension is treated as the channel dim.
        mean: Per-channel mean(s). Broadcast against the last dimension.
        std: Per-channel standard deviation(s).
        eps: Lower bound on $\sigma$ to prevent division by zero.

    Returns:
        Standardized tensor, same shape as `x`.
    """
    mean_t = torch.as_tensor(mean, dtype=x.dtype, device=x.device)
    std_t = torch.as_tensor(std, dtype=x.dtype, device=x.device).clamp(min=eps)
    return (x - mean_t) / std_t


def relabel(
    labels: Tensor,
    mapping: Union[Sequence[int], Dict[int, int]],
    default: int = 0,
) -> Tensor:
    """Remap integer labels via a lookup table.

    `mapping` can be either:

    - a sequence of source values (1:1) — each value at index $i$ is mapped to $i$;
    - a `dict[int, int]` (general source → target) — supports N-to-1 merges
      (e.g. SemanticKITTI's `moving-car` and `car` both → 0).

    Source values not listed in `mapping` are set to `default`.

    Args:
        labels: Integer label tensor (any integer dtype). Output preserves dtype.
        mapping: Source-value listing (1:1) or explicit `{source: target}` dict (N:1).
        default: Value assigned to source values not listed in `mapping`.

    Returns:
        Remapped tensor with the same shape and dtype as `labels`.

    Raises:
        ValueError: If `mapping` is empty.
    """
    if isinstance(mapping, dict):
        table: Dict[int, int] = {int(k): int(v) for k, v in mapping.items()}
    else:
        table = {int(v): i for i, v in enumerate(mapping)}
    if not table:
        raise ValueError("relabel requires at least one source value in `mapping`.")
    sorted_sources = sorted(table.keys())
    src = torch.tensor(sorted_sources, dtype=torch.long, device=labels.device)
    tgt = torch.tensor([table[s] for s in sorted_sources], dtype=torch.long, device=labels.device)
    labels_long = labels.long()
    idx = torch.searchsorted(src, labels_long)
    idx_clamped = idx.clamp(max=src.numel() - 1)
    hit = src[idx_clamped] == labels_long
    dst = torch.full_like(labels_long, default)
    dst[hit] = tgt[idx_clamped[hit]]
    return dst.to(labels.dtype)
