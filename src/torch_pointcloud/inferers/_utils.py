"""Shared inferer helpers: dict indexing, per-fragment transforms, chunk splitting, and Gaussian distance weighting."""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
from torch import Tensor


def index_select_dict(data: Dict[str, Any], idx: Tensor, n_points: int) -> Dict[str, Any]:
    """Pick `idx` rows from every tensor in `data` whose first dim equals `n_points`.

    Non-tensor entries and tensors with a different leading dimension (e.g. the
    `batch` index after re-keying, scalar metadata) are passed through untouched.
    """
    out: Dict[str, Any] = {}
    for key, value in data.items():
        if torch.is_tensor(value) and value.dim() > 0 and value.size(0) == n_points:
            out[key] = value[idx]
        else:
            out[key] = value
    return out


def apply_transform(
    sample: Dict[str, Any],
    transform: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]],
    *,
    pos_key: str,
    inverse_key: Optional[str],
) -> Tuple[Dict[str, Any], Optional[Tensor]]:
    r"""Run the per-fragment `transform` over `sample` and return it with its source-to-predictor row map.

    Any value already at `inverse_key` is dropped before `transform` runs, so a scene-level `inverse` never
    becomes the prior the fragment's own map composes through. The map the transform writes under `inverse_key`
    is popped from the returned sample, so it never reaches the predictor.

    Args:
        sample: Data dict of one fragment (block, sub-cloud, sphere).
        transform: Callable applied to `sample`, or `None` to leave it unchanged.
        pos_key: Dict key for the position tensor, whose row count defines the fragment size.
        inverse_key: Dict key under which a row-altering `transform` records its long index map of shape
            $(N_\text{source},)$ with values in $[0, N_\text{predictor})$. `None` when rows are preserved.

    Returns:
        The transformed sample and its inverse map, `None` when the transform wrote none. Predictions of shape
        $(N_\text{predictor}, C)$ index with the map back to the $N_\text{source}$ rows of `sample`.

    Raises:
        ValueError: If `transform` changes the row count without writing a map under `inverse_key`.
    """
    n_source = int(sample[pos_key].size(0))
    sample = {key: value for key, value in sample.items() if key != inverse_key}
    if transform is not None:
        sample = transform(sample)
    inverse_map = sample.pop(inverse_key, None) if inverse_key is not None else None
    n_predictor = int(sample[pos_key].size(0))
    if inverse_map is None and n_predictor != n_source:
        raise ValueError(
            f"`transform` changed the fragment's row count ({n_source} -> {n_predictor}) without recording an "
            f"index map. Make it write one (e.g. `dst_inverse_key`) and pass that key as `inverse_key`."
        )
    return sample, inverse_map


def check_batch_alignment(pos: Tensor, batch: Tensor, pos_key: str, batch_key: str) -> None:
    """Raise when the per-point batch index does not line up with the positions row for row.

    Inferers split a scene by `batch` and index `pos` with the result, so a shorter `batch` would leave
    the trailing points unpredicted instead of failing.
    """
    if batch.size(0) != pos.size(0):
        raise ValueError(
            f"`data[{batch_key!r}]` has {batch.size(0)} rows but `data[{pos_key!r}]` has {pos.size(0)}: the batch "
            f"index must be aligned to the positions. Collating with `batch_from` set to a voxelized key indexes "
            f"those voxels instead; pass that key to `cat_keys` and keep `batch` on the points."
        )


def split_chunks(n: int, max_size: Optional[int], rng: Optional[torch.Generator], device: torch.device) -> List[Tensor]:
    r"""Partition `range(n)` into index chunks of at most `max_size` points each.

    When `max_size` is `None` or `n` is within `max_size`, returns a single in-order
    chunk holding every index. Otherwise the indices are permuted with `rng` and
    sliced into $\lceil n / \text{max\_size} \rceil$ chunks. Chunk sizes sum to `n`;
    the last chunk may be smaller than `max_size`.

    Args:
        n: Number of indices to partition.
        max_size: Maximum points per chunk, or `None` for a single unsplit chunk.
        rng: Generator for the permutation, or `None` for the global generator. Only consumed when a split
            is needed.
        device: Device of the returned indices, which a given `rng` must live on.

    Returns:
        List of 1-D `long` index tensors on `device`. The chunks partition
        `range(n)` exactly: every index appears in one and only one chunk.
    """
    if max_size is None:
        return [torch.arange(n, device=device)]
    if n > max_size:
        perm = torch.randperm(n, generator=rng, device=device)
        return [perm[i : i + max_size] for i in range(0, n, max_size)]
    return [torch.arange(n, device=device)]


def gaussian_weights(distances: Tensor, sigma: Union[float, Tensor], eps: float = 1e-12) -> Tensor:
    r"""Gaussian falloff weights $w_i = \exp(-d_i^2 / 2\sigma^2)$.

    Args:
        distances: Per-point distances $d_i$ to a window or block center.
        sigma: Gaussian standard deviation. A scalar applies a single radius to every
            point; a `Tensor` broadcastable against `distances` (one radius per window
            row) applies per-row radii.
        eps: Lower bound on `sigma`, guarding against division by zero.

    Returns:
        Weights with the same shape as `distances`, each in $(0, 1]$.
    """
    safe_sigma = sigma.clamp_min(eps) if isinstance(sigma, Tensor) else max(sigma, eps)
    return torch.exp(-0.5 * (distances / safe_sigma) ** 2)
