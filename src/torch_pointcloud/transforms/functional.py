import math
from typing import Any, Dict, Literal, Optional, Sequence, Tuple, Union, get_args, overload

import torch
from torch import Tensor

from torch_pointcloud.utils.cluster import fps, knn

ShiftMethod = Literal["bbox", "centroid", "min"]

RescaleMethod = Literal["centroid", "bbox", "linear"]

PadMode = Literal["below", "above", "all"]

PadFill = Literal["cycle", "replicate", "random"]


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
            sample without replacement when $N \geq \text{num\_samples}$; when
            $\text{num\_samples} > N$ the draw falls back to replacement so the output
            always has `num_samples` rows.
        generator: The generator for the random number generator.

    Returns:
        If `return_indices` is `True`, the function returns a tuple of the sampled values and their indices.
        Otherwise, it returns the sampled values.

    Raises:
        ValueError: If `num_samples > 0` and the input is empty.
    """
    n = tensor.size(0)
    if num_samples == 0:
        indices = torch.empty(0, dtype=torch.long, device=tensor.device)
    elif n == 0:
        raise ValueError(f"Cannot sample {num_samples} values from an empty tensor (N=0).")
    elif replace or num_samples > n:
        indices = torch.randint(0, n, (num_samples,), generator=generator, device=tensor.device)
    else:
        indices = torch.randperm(n, generator=generator, device=tensor.device)[:num_samples]

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
        >>> idx = farthest_point_sample(pos, num_samples=10)  # doctest: +SKIP
        >>> print(idx.shape)  # doctest: +SKIP
        torch.Size([10])
    """
    return fps(pos, num_nodes=num_samples, ratio=ratio, random_start=random_start)


def estimate_normals(
    pos: Tensor, k: int = 16, batch: Optional[Tensor] = None, orient_to_centroid: bool = False
) -> Tensor:
    r"""Estimate per-point unit surface normals by local PCA.

    For each point the normal is the eigenvector of the smallest eigenvalue of the covariance of its $k$
    nearest neighbours, i.e. the direction of least variance (the local tangent-plane normal).

    PCA gives no canonical orientation. By default the sign is the arbitrary-but-deterministic sign returned by
    `torch.linalg.eigh`. With `orient_to_centroid`, each normal is flipped to point towards its cloud's
    centroid, which approximates the inward-facing orientation of meshes scanned from inside a room (S3DIS,
    ScanNet) and matters when the consuming model was trained on oriented normals.

    Args:
        pos: Point coordinates of shape $(N, 3)$.
        k: Number of nearest neighbours (the point itself included) used per local PCA. Must not exceed the
            number of points in the smallest cloud.
        batch: Optional $(N,)$ batch index so neighbours never cross cloud boundaries.
        orient_to_centroid: If `True`, flip each normal to point towards its cloud's centroid.

    Returns:
        Unit normals of shape $(N, 3)$.

    Raises:
        ValueError: If `pos` has fewer than `k` points.

    Shape:
        - Input: $(N, 3)$
        - Output: $(N, 3)$
    """
    num_points = pos.shape[0]
    if num_points < k:
        raise ValueError(f"estimate_normals requires at least k points for the k-NN PCA; got N={num_points}, k={k}.")
    neighbor_index = knn(pos, pos, k, batch_x=batch, batch_y=batch)[1].view(num_points, k)
    neighbors = pos[neighbor_index]
    centered = neighbors - neighbors.mean(dim=1, keepdim=True)
    covariance = centered.transpose(1, 2) @ centered / k
    _, eigenvectors = torch.linalg.eigh(covariance)
    normals = eigenvectors[..., 0]

    if orient_to_centroid:
        if batch is None:
            centroid = pos.mean(dim=0, keepdim=True)
        else:
            num_clouds = int(batch.max()) + 1
            counts = torch.zeros(num_clouds, device=pos.device, dtype=pos.dtype)
            counts.index_add_(0, batch, torch.ones(num_points, device=pos.device, dtype=pos.dtype))
            sums = torch.zeros(num_clouds, 3, device=pos.device, dtype=pos.dtype)
            sums.index_add_(0, batch, pos)
            centroid = (sums / counts.unsqueeze(1))[batch]
        flip = ((centroid - pos) * normals).sum(dim=-1, keepdim=True) < 0
        normals = torch.where(flip, -normals, normals)

    return normals


def rescale(
    points: Tensor,
    eps: float = 1e-6,
    method: RescaleMethod = "centroid",
) -> Tensor:
    r"""Center a point set and rescale it to a unit extent.

    Operates along the point dimension `dim=-2`. Pairs a centering step with a
    scale-by-extent step that share the same statistics. The scale denominator is a
    single statistic over **all** leading dimensions, so the input is treated as one
    point cloud: rescale packed batches per sample (pre-collate), never on
    concatenated clouds.

    Args:
        points: Tensor of shape $(\ldots, N, C)$ with $C \geq 1$; min/max and means are over $N$.
        eps: Small constant added to the scale denominator for numerical stability.
        method:

            * `"centroid"`: subtract the mean over points, then divide by
              $\max(\max_i \|\mathbf{x}_i - \mathbf{\mu}\|_2, \epsilon)$ (max Euclidean distance
              from the centroid, clamped from below by `eps`).

            * `"bbox"`: subtract the axis-aligned bounding-box midpoint (midrange center),
              then divide by half the longest edge of that box plus $\epsilon$ (matches common
              ModelNet-style normalization):

              $$
              \mathbf{c} = \frac{\mathbf{x}_{\min} + \mathbf{x}_{\max}}{2}, \quad
              r = \frac{1}{2}\max_j (x_{\max,j} - x_{\min,j}) + \epsilon, \quad
              \mathbf{x} \leftarrow \frac{\mathbf{x} - \mathbf{c}}{r}
              $$

            * `"linear"`: subtract the centroid then divide by the longest axis-aligned
              span (the convention used by the published RandLA-Net Toronto-3D /
              Semantic3D checkpoints):

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
    generator: Optional[torch.Generator] = None,
) -> Tuple[Tensor, Tensor]: ...


@overload
def divisible_pad(
    batch: Tensor,
    k: int,
    mode: PadMode = "all",
    pad_fill: PadFill = "cycle",
    return_inverse: Literal[True] = ...,
    generator: Optional[torch.Generator] = None,
) -> Tuple[Tensor, Tensor, Tensor]: ...


@torch.no_grad()
def divisible_pad(
    batch: Tensor,
    k: int,
    mode: PadMode = "all",
    pad_fill: PadFill = "cycle",
    return_inverse: bool = False,
    generator: Optional[torch.Generator] = None,
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

    pad_fill="random"     → [A B C D] [E F G ?]
      Random sample from the batch           ↑ uniform over {A..G}
    ```

    When `batch_size < k` there is no previous patch, so `"replicate"`
    falls back to `"cycle"`:

    ```text
    batch 0 (size 2, k=4):  [A B · ·]
    pad_fill="cycle"      → [A B A B]
    pad_fill="replicate"  → [A B A B]   (same, no prior patch)
    pad_fill="random"     → [A B ? ?]
    ```

    Args:
        batch: The batch indices of the tensor. Rows of the same batch must be contiguous (grouped, as
            produced by packed-batch collation); the batch values themselves may be non-consecutive.
            Interleaved orderings (e.g. `[0, 1, 0, 1]`) are not supported and silently mix samples.
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
            - `"random"`: Sample padded indices uniformly with replacement from
              within the batch's original indices. Consumes `generator` if given.
        return_inverse: Whether to return the inverse of the padded indices.
        generator: Optional `torch.Generator` for reproducibility (used only by
            `pad_fill="random"`).

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
            if pad_fill == "random":
                offsets = torch.randint(high=batch_size, size=(pad_size,), generator=generator, device=device)
                indices[new_start + batch_size : new_start + batch_size + pad_size] = original_start + offsets
            elif pad_fill == "replicate" and batch_size > k:
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
        >>> split_batch(batch, max_size=2)
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
        tensor([1., 2., 3.])
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


def box_mask(x: Tensor, bbox: tuple[float, ...], dim: int = -1, strict: bool = False) -> Tensor:
    r"""Create a boolean mask for points inside an axis-aligned bounding box (AABB).

    Membership condition along `dim` (default, boundary points included):

    $$
    \text{bbmin}_j \leq x_j \leq \text{bbmax}_j \quad \forall j
    $$

    With `strict=True` the inequalities are strict, so boundary points are excluded.

    Args:
        x: The input tensor of shape `(..., D)` along `dim`.
        bbox: AABB as a flat tuple `(*bbmin, *bbmax)` of length `2 * D`.
        dim: The dimension to compute the mask over.
        strict: If `True`, use strict inequalities (points exactly on the boundary are excluded).

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
    if strict:
        return (x > bbmin).all(dim=dim) & (x < bbmax).all(dim=dim)
    return (x >= bbmin).all(dim=dim) & (x <= bbmax).all(dim=dim)


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
        tensor([1., 3.])
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

    ```{.python notest}
    # Center XY at the bbox midpoint and Z at the minimum
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


def axis_min_offset(x: Tensor, axis: int, quantile: Optional[float] = None) -> Tensor:
    r"""Per-point offset from a floor reference along a chosen coordinate axis.

    For positions of shape $(N, D)$ and an axis $a \in [0, D)$, returns a
    tensor of shape $(N, 1)$ whose entries are $x_{i, a} - r$ where the floor
    reference $r$ is either the strict minimum $\min_j x_{j, a}$ (default) or, when
    `quantile` is given, the empirical quantile $Q_{q}(x_{\cdot, a})$. A small
    positive quantile (e.g. $q = 0.0099$, the `np.percentile(z, 0.99)` used by
    VoteNet) yields an outlier-robust floor estimate. Useful for extracting
    "height above the local floor" as a per-point feature.

    Args:
        x: Input tensor of shape `(N, D)`.
        axis: Axis index in the last dimension.
        quantile: Optional quantile $q \in [0, 1]$ for the floor reference. When
            `None`, the strict per-axis minimum is used (equivalent to $q = 0$).

    Returns:
        Tensor of shape `(N, 1)` with the same dtype as `x`. Returns an empty
        `(0, 1)` tensor when `x` is empty.
    """
    col = x[:, axis]
    if col.numel() == 0:
        return col.unsqueeze(-1).to(x.dtype)
    if quantile is None:
        ref = col.min()
    else:
        ref = torch.quantile(col.float(), quantile).to(col.dtype)
    return (col - ref).unsqueeze(-1).to(x.dtype)


def quantize(pos: Tensor, size: float) -> Tensor:
    r"""Integer voxel-grid coordinates of every point, without reducing the cloud.

    Each point maps to $\lfloor p / s \rfloor$ shifted so the per-axis minimum is $0$; points sharing a voxel
    get equal coordinates and every input row is kept. This is the coordinate a voxel-partition protocol feeds
    to a sparse model for each raw point (`Voxelize(pos_reduce="grid")` produces the same coordinates for the
    one representative it keeps per voxel).

    Args:
        pos: Point positions of shape $(N, D)$.
        size: Voxel side length in the units of `pos`.

    Returns:
        Long tensor of shape $(N, D)$ (empty input returns an empty $(0, D)$ tensor).

    Example:
        ```python
        import torch
        from torch_pointcloud.transforms import functional as F

        pos = torch.tensor([[0.0, 0.0, 0.0], [0.03, 0.0, 0.0], [0.05, 0.0, 0.0]])
        F.quantize(pos, size=0.02)  # tensor([[0, 0, 0], [1, 0, 0], [2, 0, 0]])
        ```
    """
    if size <= 0.0:
        raise ValueError(f"`size` must be > 0, got {size}.")

    pos_grid = torch.floor(pos / size).long()
    if pos_grid.shape[0] == 0:
        return pos_grid
    return pos_grid - pos_grid.min(dim=0).values


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
    if not torch.is_floating_point(x):
        x = x.float()
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

    - a sequence of source values (1:1): each value at index $i$ is mapped to $i$;
    - a `dict[int, int]` (general source → target): supports N-to-1 merges
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


def rotation_matrix(angle: float, axis: int = 2, device: Optional[torch.device] = None) -> Tensor:
    r"""3x3 rotation matrix for `angle` radians around an axis-aligned axis.

    Args:
        angle: Rotation angle in **radians**.
        axis: Axis index to rotate around (0=X, 1=Y, 2=Z).
        device: Output device. Defaults to CPU.

    Returns:
        Rotation matrix of shape `(3, 3)`.

    Raises:
        ValueError: If `axis` is not in `{0, 1, 2}`.
    """
    if axis not in (0, 1, 2):
        raise ValueError(f"axis must be 0, 1, or 2; got {axis}.")
    c = math.cos(angle)
    s = math.sin(angle)
    R = torch.eye(3, device=device, dtype=torch.float32)
    i, j = [(1, 2), (2, 0), (0, 1)][axis]
    R[i, i] = c
    R[j, j] = c
    R[i, j] = -s
    R[j, i] = s
    return R


def random_jitter(
    x: Tensor,
    sigma: float = 0.01,
    clip: Optional[float] = 0.05,
    generator: Optional[torch.Generator] = None,
) -> Tensor:
    """Add Gaussian noise to `x`, optionally clipped.

    Args:
        x: Input tensor.
        sigma: Standard deviation of the Gaussian noise.
        clip: If not `None`, clip the noise to `[-clip, clip]`.
        generator: Random generator for reproducibility.

    Returns:
        Jittered tensor with the same shape as `x`.
    """
    noise = torch.empty_like(x).normal_(mean=0.0, std=sigma, generator=generator)
    if clip is not None:
        noise = noise.clamp(min=-clip, max=clip)
    return x + noise


def random_dropout_mask(
    n: int,
    p_drop: float,
    device: Optional[torch.device] = None,
    generator: Optional[torch.Generator] = None,
) -> Tensor:
    """Return a boolean keep-mask of length `n` where each entry is kept with probability `1 - p_drop`.

    Args:
        n: Number of points.
        p_drop: Probability of dropping a point. Must be in $[0, 1)$.
        device: Output device.
        generator: Random generator for reproducibility.

    Returns:
        Boolean tensor of shape `(n,)`.

    Raises:
        ValueError: If `p_drop` is not in `[0, 1)`.
    """
    if not 0.0 <= p_drop < 1.0:
        raise ValueError(f"p_drop must be in [0, 1); got {p_drop}.")
    device = device or torch.device("cpu")
    rand = torch.rand(n, device=device, generator=generator)
    return rand >= p_drop


def shuffle_indices(
    n: int,
    device: Optional[torch.device] = None,
    generator: Optional[torch.Generator] = None,
) -> Tensor:
    """Return a random permutation of `[0, n)`.

    Args:
        n: Sequence length.
        device: Output device.
        generator: Random generator for reproducibility.

    Returns:
        Long tensor of shape `(n,)`.
    """
    device = device or torch.device("cpu")
    return torch.randperm(n, device=device, generator=generator)


def _color_max(color: Tensor, int_color: bool) -> float:
    """Resolve the color range maximum from the tensor dtype, validating the `int_color` flag."""
    if color.dtype == torch.uint8 or int_color:
        return 255.0
    if color.numel() > 0 and float(color.max()) > 1.0:
        raise ValueError(
            f"Float colors with `int_color=False` must lie in [0, 1], but got a maximum of {float(color.max()):.4g}. "
            "Pass `int_color=True` for [0, 255] float colors, or divide by 255 first."
        )
    return 1.0


def random_color_jitter(
    color: Tensor,
    brightness: float = 0.0,
    contrast: float = 0.0,
    saturation: float = 0.0,
    int_color: bool = False,
    generator: Optional[torch.Generator] = None,
) -> Tensor:
    """Jitter colors by brightness, contrast, and saturation, in that order.

    Each strength is a relative delta sampled uniformly from `[-x, x]` and
    applied multiplicatively (`out = x * factor`).

    Args:
        color: Color tensor of shape `(N, 3)`.
        brightness: Max relative brightness change. `0.2` means ±20%.
        contrast: Max relative contrast change.
        saturation: Max relative saturation change. Saturation moves toward
            (or away from) the per-channel grayscale luminance.
        int_color: If `True`, treat float colors as `[0, 255]` values; otherwise `[0, 1]`.
            `uint8` colors are always treated as `[0, 255]` regardless of the flag.
        generator: Random generator for reproducibility.

    Returns:
        Jittered colors with the same shape and dtype as `color`.

    Raises:
        ValueError: If `color` is a float tensor with values above 1 while `int_color=False`.
    """
    max_val = _color_max(color, int_color)
    out = color.float() / max_val

    if brightness > 0:
        b = torch.empty(1).uniform_(1 - brightness, 1 + brightness, generator=generator).item()
        out = out * b
    if contrast > 0:
        c = torch.empty(1).uniform_(1 - contrast, 1 + contrast, generator=generator).item()
        mean = out.mean(dim=0, keepdim=True)
        out = (out - mean) * c + mean
    if saturation > 0:
        s = torch.empty(1).uniform_(1 - saturation, 1 + saturation, generator=generator).item()
        # Luminance per point, broadcast across channels.
        gray = (out * torch.tensor([0.299, 0.587, 0.114], device=out.device)).sum(dim=-1, keepdim=True)
        out = (out - gray) * s + gray

    out = out.clamp(0.0, 1.0) * max_val
    return out.to(color.dtype)


def random_color_drop(
    color: Tensor,
    fill: float = 0.5,
    int_color: bool = False,
) -> Tensor:
    """Replace colors with a constant gray value (drops chromatic information).

    Args:
        color: Color tensor of shape `(N, 3)`.
        fill: Replacement value, expressed in the range implied by `int_color` (`[0, 1]` when
            `False`, `[0, 255]` when `True`). It is rescaled to the input's actual range when
            that differs, so the default `0.5` fills `127` on `uint8` colors.
        int_color: If `True`, treat float colors as `[0, 255]` values; otherwise `[0, 1]`.
            `uint8` colors are always treated as `[0, 255]` regardless of the flag.

    Returns:
        Tensor of the same shape and dtype as `color`, filled with the rescaled `fill`.

    Raises:
        ValueError: If `color` is a float tensor with values above 1 while `int_color=False`.
    """
    flag_max = 255.0 if int_color else 1.0
    return torch.full_like(color, fill * _color_max(color, int_color) / flag_max)


def color_grayscale(color: Tensor, int_color: bool = False) -> Tensor:
    """Convert RGB colors to grayscale using the BT.601 luminance weights.

    Args:
        color: Color tensor of shape `(N, 3)`.
        int_color: If `True`, treat colors as `[0, 255]` ints; otherwise `[0, 1]` floats.

    Returns:
        Tensor with the same shape and dtype as `color`, with R=G=B = luminance.
    """
    weights = torch.tensor([0.299, 0.587, 0.114], device=color.device)
    if int_color:
        lum = (color.float() * weights).sum(dim=-1, keepdim=True)
        return lum.expand_as(color).to(color.dtype)
    lum = (color * weights).sum(dim=-1, keepdim=True)
    return lum.expand_as(color).to(color.dtype)


def color_shift(color: Tensor, shift: Tensor, int_color: bool = False) -> Tensor:
    """Add a per-channel offset to colors, clamped to the valid color range.

    Args:
        color: Color tensor of shape `(N, 3)`.
        shift: Per-channel offset of shape `(3,)`, in the same range as the colors.
        int_color: If `True`, treat float colors as `[0, 255]` values; otherwise `[0, 1]`.
            `uint8` colors are always treated as `[0, 255]` regardless of the flag.

    Returns:
        Shifted colors with the same shape and dtype as `color`.

    Raises:
        ValueError: If `color` is a float tensor with values above 1 while `int_color=False`.
    """
    max_val = _color_max(color, int_color)
    out = color.float() + shift.to(color.device)
    return out.clamp(0.0, max_val).to(color.dtype)


def color_auto_contrast(color: Tensor, blend: float = 0.5, int_color: bool = False) -> Tensor:
    """Stretch per-cloud color range to the full `[0, max]` interval, then blend.

    For each channel, the min becomes 0 and the max becomes `max_val`. The
    output is then linearly blended with the original by `blend`
    (`blend=1.0` is the fully stretched version, `blend=0.0` is the input).

    Args:
        color: Color tensor of shape `(N, 3)`.
        blend: Blend weight in `[0, 1]`.
        int_color: If `True`, treat float colors as `[0, 255]` values; otherwise `[0, 1]`.
            `uint8` colors are always treated as `[0, 255]` regardless of the flag.

    Returns:
        Auto-contrast tensor with the same shape and dtype as `color`.

    Raises:
        ValueError: If `color` is a float tensor with values above 1 while `int_color=False`.
    """
    if color.shape[0] == 0:
        return color
    max_val = _color_max(color, int_color)
    out = color.float()
    lo = out.min(dim=0).values
    hi = out.max(dim=0).values
    scale = max_val / (hi - lo).clamp(min=1e-6)
    stretched = (out - lo) * scale
    blended = blend * stretched + (1.0 - blend) * out
    return blended.clamp(0.0, max_val).to(color.dtype)


def random_elastic_distortion(
    pos: Tensor,
    granularity: float = 0.2,
    magnitude: float = 0.4,
    generator: Optional[torch.Generator] = None,
) -> Tensor:
    """Apply a smooth random displacement field to `pos`.

    Implements the elastic distortion recipe common in sparse-voxel indoor
    segmentation pipelines: sample Gaussian noise on a coarse 3D grid (cells of
    side `granularity`), smooth it with two passes of a 3x3x3 mean filter,
    trilinear-interpolate the smoothed displacement at each point, and add it
    to the position. Net effect is a locally-coherent, low-frequency
    deformation that preserves nearby-point relationships.

    Args:
        pos: Input positions of shape `(N, 3)`.
        granularity: Size of the noise grid cells (in the same units as `pos`).
            Smaller values give higher-frequency distortion.
        magnitude: Standard deviation of the per-cell Gaussian noise (in the
            same units as `pos`). Larger values give stronger deformation.
        generator: Random generator for reproducibility.

    Returns:
        Distorted positions of shape `(N, 3)`.
    """
    if pos.shape[0] == 0:
        return pos
    if pos.shape[-1] != 3:
        raise ValueError(f"random_elastic_distortion expects shape (N, 3); got {tuple(pos.shape)}.")

    pos_min = pos.min(dim=0).values
    pos_max = pos.max(dim=0).values
    extent = (pos_max - pos_min).clamp(min=granularity)

    # Noise grid with node spacing `granularity` and one pad node on each side for safe interpolation
    grid_int = (extent / granularity).ceil().to(torch.long) + 3
    grid_x, grid_y, grid_z = (int(grid_int[i].item()) for i in range(3))

    # Sample noise on the coarse grid: (N, C, D, H, W) for grid_sample input
    noise = (
        torch.randn(
            1,
            3,
            grid_z,
            grid_y,
            grid_x,
            generator=generator,
            device=pos.device,
            dtype=torch.float32,
        )
        * magnitude
    )

    # Smooth via two passes of 3x3x3 mean filter
    for _ in range(2):
        noise = torch.nn.functional.avg_pool3d(noise, kernel_size=3, stride=1, padding=1)

    # Node j sits at pos_min + granularity * (j - 1), so one grid cell spans exactly `granularity`.
    # grid_sample's grid last dim is (x, y, z) which indexes (W, H, D) of the input.
    index = (pos - pos_min) / granularity + 1.0
    normalized = 2.0 * index / (grid_int.to(pos.dtype) - 1.0) - 1.0
    sample_coords = normalized.to(noise.dtype).view(1, 1, 1, -1, 3)

    displacement = torch.nn.functional.grid_sample(
        noise, sample_coords, mode="bilinear", padding_mode="zeros", align_corners=True
    )
    # displacement: (1, 3, 1, 1, N) -> (N, 3)
    displacement = displacement.squeeze(2).squeeze(2).squeeze(0).T
    return pos + displacement.to(pos.dtype)


def flip_boxes(boxes: Tensor, axis: int) -> Tensor:
    r"""Flip oriented 3D boxes along a spatial axis.

    Boxes are stored as $(K, 7)$ rows $[c_x, c_y, c_z, d_x, d_y, d_z, \theta]$ with full extents and heading
    in radians counterclockwise about $+z$ from $+x$. A flip negates the center component along `axis`. A
    flip along `axis` $0$ (the $yz$ plane) maps the heading to $\pi - \theta$; a flip along `axis` $1$ (the
    $xz$ plane) maps the heading to $-\theta$. Sizes are unchanged.

    Args:
        boxes: Box tensor of shape $(K, 7)$.
        axis: Center axis index to negate (0=X, 1=Y).

    Returns:
        The flipped box tensor of shape $(K, 7)$.
    """
    boxes = boxes.clone()
    boxes[:, axis] = -boxes[:, axis]
    if axis == 0:
        boxes[:, 6] = math.pi - boxes[:, 6]
    elif axis == 1:
        boxes[:, 6] = -boxes[:, 6]
    return boxes


def rotate_boxes(boxes: Tensor, rotation: Tensor, angle: float) -> Tensor:
    r"""Rotate oriented 3D boxes about the up axis.

    Box centers are rotated by `rotation` (`centers @ rotation.transpose(-1, -2)`) and the heading is
    incremented by `angle`, so a counterclockwise rotation about $+z$ keeps the counterclockwise heading
    aligned with the jointly rotated points. Sizes are unchanged.

    Args:
        boxes: Box tensor of shape $(K, 7)$ as $[c_x, c_y, c_z, d_x, d_y, d_z, \theta]$.
        rotation: A $3 \times 3$ rotation matrix rotating by `angle` counterclockwise about the $z$ axis.
        angle: Rotation angle in **radians**, added to the heading.

    Returns:
        The rotated box tensor of shape $(K, 7)$.
    """
    boxes = boxes.clone()
    boxes[:, 0:3] = boxes[:, 0:3] @ rotation.to(boxes).transpose(-1, -2)
    boxes[:, 6] = boxes[:, 6] + angle
    return boxes


def scale_boxes(boxes: Tensor, scale: Union[float, Tensor]) -> Tensor:
    r"""Scale oriented 3D boxes by an isotropic factor.

    Both centers and extents (columns $0$ to $6$) are multiplied by `scale`. Heading is unchanged.

    Args:
        boxes: Box tensor of shape $(K, 7)$.
        scale: Isotropic scalar factor applied to centers and sizes.

    Returns:
        The scaled box tensor of shape $(K, 7)$.
    """
    boxes = boxes.clone()
    factor = scale.to(boxes) if isinstance(scale, Tensor) else scale
    boxes[:, 0:6] = boxes[:, 0:6] * factor
    return boxes


def shift_boxes(boxes: Tensor, shift: Tensor) -> Tensor:
    r"""Translate oriented 3D boxes by a fixed offset.

    Centers (columns $0$ to $3$) are offset by `shift`. Sizes and heading are unchanged.

    Args:
        boxes: Box tensor of shape $(K, 7)$ as $[c_x, c_y, c_z, d_x, d_y, d_z, \theta]$.
        shift: Translation vector of shape $(3,)$.

    Returns:
        The shifted box tensor of shape $(K, 7)$.
    """
    boxes = boxes.clone()
    boxes[:, 0:3] = boxes[:, 0:3] + shift.to(boxes)
    return boxes


def flip_vectors(x: Tensor, axis: int) -> Tensor:
    r"""Flip a packed field of 3D vectors along a spatial axis.

    Negates component `axis` of every contiguous triple of the last dimension, so it handles both a plain
    $(N, 3)$ field (e.g. coordinates or normals) and a $(N, 3 G)$ field of $G$ tiled offsets (e.g. VoteNet
    vote offsets $(\text{center} - \text{point})$) alike.

    Args:
        x: Vector field of shape $(N, 3)$ or $(N, 3 G)$.
        axis: Axis index within each triple to negate.

    Returns:
        The flipped tensor with the same shape as `x`.
    """
    x = x.clone()
    x[..., axis::3] = -x[..., axis::3]
    return x


def rotate_vectors(x: Tensor, rotation: Tensor) -> Tensor:
    r"""Rotate a packed field of 3D vectors by a rotation matrix.

    Each contiguous triple of the last dimension rotates as a vector, so it handles both a plain $(N, 3)$
    field (e.g. coordinates or normals) and a $(N, 3 G)$ field of $G$ tiled offsets (e.g. VoteNet vote
    offsets) alike.

    Args:
        x: Vector field of shape $(N, 3)$ or $(N, 3 G)$.
        rotation: A $3 \times 3$ rotation matrix.

    Returns:
        The rotated tensor with the same shape as `x`.
    """
    triples = x.reshape(*x.shape[:-1], -1, 3)
    triples = triples @ rotation.to(x).transpose(-1, -2)
    return triples.reshape(x.shape)


def points_in_oriented_box(pos: Tensor, box: Tensor) -> Tensor:
    r"""Test which points lie inside a single oriented 3D box.

    The point offsets relative to the box center are rotated into the box frame by $-\theta$ about $+z$, then
    compared against the half-extents with an axis-aligned bounding-box test. The heading $\theta$ is in
    radians counterclockwise about $+z$. For a box with zero heading this reduces to a plain axis-aligned
    test.

    Args:
        pos: Coordinate tensor of shape $(N, 3)$.
        box: A single box of shape $(7,)$ as $[c_x, c_y, c_z, h_x, h_y, h_z, \theta]$ with **half**-extents.

    Returns:
        A boolean mask of shape $(N,)$ that is `True` for points inside the box.
    """
    center = box[0:3]
    half = box[3:6]
    heading = box[6]
    rotation = rotation_matrix(float(-heading), axis=2, device=pos.device).to(pos.dtype)
    local = (pos - center) @ rotation.transpose(-1, -2)
    return (local.abs() <= half).all(dim=1)


def angle_to_class(angle: Tensor, num_heading_bin: int) -> Tuple[Tensor, Tensor]:
    r"""Convert continuous heading angles to discrete bin classes and residuals.

    The range $[0, 2\pi)$ is split into `num_heading_bin` equal bins centered at
    $0, 1 \cdot (2\pi / N), \ldots, (N - 1) \cdot (2\pi / N)$. The returned class and residual satisfy
    $\text{class} \cdot (2\pi / N) + \text{residual} = \text{angle}$.

    Args:
        angle: Heading angles in radians of shape $(K,)$.
        num_heading_bin: Number of heading bins $N$.

    Returns:
        A tuple of the per-angle class indices (long, shape $(K,)$) and residual angles (shape $(K,)$).
    """
    two_pi = 2 * math.pi
    angle_per_class = two_pi / num_heading_bin
    angle = angle % two_pi
    shifted = (angle + angle_per_class / 2) % two_pi
    # The division can round up to exactly N when `shifted` sits a float ulp below 2 pi; clamp keeps the
    # class in range.
    cls = (shifted / angle_per_class).long().clamp(max=num_heading_bin - 1)
    residual = shifted - (cls.to(angle.dtype) * angle_per_class + angle_per_class / 2)
    return cls, residual


def class_to_angle(heading_class: Tensor, heading_residual: Tensor, num_heading_bin: int) -> Tensor:
    r"""Invert `angle_to_class`: recover continuous heading angles from bin classes and residuals.

    A single bin (`num_heading_bin == 1`, axis-aligned boxes) always decodes to a heading of $0$.

    Args:
        heading_class: Bin class indices (long) of shape $(K,)$.
        heading_residual: Per-angle residuals of shape $(K,)$.
        num_heading_bin: Number of heading bins $N$.

    Returns:
        The recovered heading angles of shape $(K,)$.
    """
    if num_heading_bin == 1:
        return torch.zeros_like(heading_residual)
    return heading_class.to(heading_residual.dtype) * (2 * math.pi / num_heading_bin) + heading_residual


def class_to_size(size_class: Tensor, size_residual: Tensor, mean_sizes: Tensor) -> Tensor:
    r"""Recover full box edge lengths from a size class index and residual (inverse of the size encoding).

    Args:
        size_class: Size class indices (long) of shape $(K,)$.
        size_residual: Per-axis residuals of shape $(K, 3)$.
        mean_sizes: Template sizes of shape $(C, 3)$ holding full edge lengths per class.

    Returns:
        The recovered full edge lengths of shape $(K, 3)$.
    """
    return mean_sizes.to(size_residual)[size_class.long()] + size_residual


def laser_mix_masks(
    pos: Tensor,
    other_pos: Tensor,
    num_areas: int,
    pitch_range: Tuple[float, float],
    generator: Optional[torch.Generator] = None,
) -> Tuple[Tensor, Tensor]:
    r"""Return keep-masks that swap alternating inclination (pitch) bands between two LiDAR scans.

    Each point's inclination is $\phi = \arctan2(z, \sqrt{x^2 + y^2})$ in degrees. The range
    `pitch_range` is split into `num_areas` equal bands; a random parity picks whether the even or
    odd bands are kept from the first scan, with the complementary bands kept from the second. The
    mixed scene is `torch.cat([pos[mask], other_pos[other_mask]])`, so the two masks tile the sky.

    Args:
        pos: Coordinates of the first scan of shape $(N, 3)$.
        other_pos: Coordinates of the second scan of shape $(M, 3)$.
        num_areas: Number of inclination bands to split `pitch_range` into.
        pitch_range: Inclination range `(min, max)` in degrees.
        generator: Random generator for reproducibility.

    Returns:
        A tuple `(mask, other_mask)` of boolean tensors of shape $(N,)$ and $(M,)$ that select the
        points kept from `pos` and from `other_pos` respectively.

    Shape:
        - `pos`: $(N, 3)$
        - `other_pos`: $(M, 3)$
        - output: $(N,)$ and $(M,)$

    Raises:
        ValueError: If `num_areas` is not positive.

    Example:
        ```python
        import torch
        from torch_pointcloud.transforms.functional import laser_mix_masks

        pos = torch.randn(100, 3)
        other = torch.randn(120, 3)
        g = torch.Generator().manual_seed(0)
        mask, other_mask = laser_mix_masks(pos, other, num_areas=4, pitch_range=(-25.0, 3.0), generator=g)
        mixed = torch.cat([pos[mask], other[other_mask]], dim=0)
        ```
    """
    if num_areas <= 0:
        raise ValueError(f"num_areas must be positive; got {num_areas}.")
    lo, hi = pitch_range
    edges = torch.linspace(lo, hi, num_areas + 1, device=pos.device)[1:-1]

    def bands(p: Tensor) -> Tensor:
        rho = torch.sqrt(p[:, 0] ** 2 + p[:, 1] ** 2)
        pitch = torch.rad2deg(torch.atan2(p[:, 2], rho))
        return torch.bucketize(pitch, edges.to(pitch))

    start = int(torch.randint(2, (1,), generator=generator).item())
    mask = (bands(pos) % 2) == start
    other_mask = (bands(other_pos) % 2) != start
    return mask, other_mask


def polar_mix_masks(
    pos: Tensor,
    other_pos: Tensor,
    generator: Optional[torch.Generator] = None,
) -> Tuple[Tensor, Tensor]:
    r"""Return keep-masks that swap a random azimuth half-sector between two LiDAR scans.

    Each point's azimuth is $\theta = \arctan2(y, x)$. A random start angle in $[-\pi, \pi)$ defines a
    half-circle sector $[\theta_0, \theta_0 + \pi)$ that wraps around the $\pm\pi$ seam, so half of the
    azimuth range is swapped regardless of the start angle. Points of the first scan outside the sector
    are kept, and points of the second scan inside the sector are added, so the mixed scene is
    `torch.cat([pos[mask], other_pos[other_mask]])`.

    Args:
        pos: Coordinates of the first scan of shape $(N, 3)$.
        other_pos: Coordinates of the second scan of shape $(M, 3)$.
        generator: Random generator for reproducibility.

    Returns:
        A tuple `(mask, other_mask)` of boolean tensors of shape $(N,)$ and $(M,)$ that select the
        points kept from `pos` and pasted from `other_pos` respectively.

    Shape:
        - `pos`: $(N, 3)$
        - `other_pos`: $(M, 3)$
        - output: $(N,)$ and $(M,)$

    Example:
        ```python
        import torch
        from torch_pointcloud.transforms.functional import polar_mix_masks

        pos = torch.randn(100, 3)
        other = torch.randn(120, 3)
        g = torch.Generator().manual_seed(0)
        mask, other_mask = polar_mix_masks(pos, other, generator=g)
        mixed = torch.cat([pos[mask], other[other_mask]], dim=0)
        ```
    """
    start = (torch.rand(1, generator=generator).item() * 2.0 - 1.0) * math.pi
    yaw = torch.atan2(pos[:, 1], pos[:, 0])
    other_yaw = torch.atan2(other_pos[:, 1], other_pos[:, 0])
    inside = (yaw - start) % (2 * math.pi) < math.pi
    other_inside = (other_yaw - start) % (2 * math.pi) < math.pi
    return ~inside, other_inside
