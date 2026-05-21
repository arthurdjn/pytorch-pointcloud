r"""Sliding-window inference for large-scale point cloud segmentation.

Tiles the scene with axis-aligned cubic blocks and accumulates per-point
predictions across all blocks that contain each point. Adjacent blocks overlap
by a configurable fraction, reducing seam artefacts at block boundaries.

When `overlap=0.0` (default), blocks form a non-overlapping partition and each
point is predicted exactly once. Higher `overlap` values increase coverage
redundancy; points near block boundaries receive a weighted average of
predictions from all blocks that contain them.

Per-block pre-processing goes through the `transform` argument.

!!! warning "`block_size` is in the units of `data[pos_key]`, not always metres"

    The inferer tiles in whatever coordinate space `pos` is in at call time. If
    positions are voxel indices after upstream voxelization, `block_size` is a
    voxel count, not metres. A scene voxelized at $2\,\text{cm}$ tiled with
    `block_size=200` gives $4\,\text{m}$ blocks.
"""

import itertools
import math
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple

import torch
from torch import Tensor
from tqdm import tqdm

from torch_pointcloud.utils.data import DataKeys

from ._utils import gaussian_weights, index_select_dict, split_chunks
from .inferer import Inferer

WindowMode = Literal["constant", "gaussian"]


def _assign_point_blocks(
    pos: Tensor,
    *,
    block_size: float,
    overlap: float,
    mode: WindowMode,
    sigma_scale: float,
) -> Tuple[List[Tensor], List[Tensor]]:
    r"""Group point indices by the overlapping blocks that contain them.

    Tiles `pos` (one batch element, shape $(N, D)$) with cubic blocks of side
    `block_size` whose extents are spaced $\text{block\_size} \cdot (1 - \text{overlap})$
    apart. A point belongs to every block covering it: per axis the block index
    $i$ ranges over $[\,i_\text{hi} - K + 1,\; i_\text{hi}\,]$, where
    $i_\text{hi} = \lfloor (p - \text{lo}) / \text{step} \rfloor$ and
    $K = \lceil \text{block\_size} / \text{step} \rceil$. For `overlap=0`, $K = 1$
    and $i_\text{lo} = i_\text{hi}$, so each point lands in exactly one block.

    Args:
        pos: Point positions for a single batch element, shape $(N, D)$.
        block_size: Side length of each cubic block, in the units of `pos`.
        overlap: Fraction of `block_size` shared between adjacent blocks, in $[0, 1)$.
        mode: Per-point weight scheme. `"constant"` weights every point equally;
            `"gaussian"` weights by $\exp(-d^2 / 2\sigma^2)$ on the distance $d$ to
            the block center.
        sigma_scale: Gaussian sigma scale factor (only used when `mode="gaussian"`).

    Returns:
        `(point_groups, weight_groups)`: equal-length lists with one entry per
        non-empty block. `point_groups[j]` holds the local point indices of block
        $j$; `weight_groups[j]` holds their matching per-point blend weights.
    """
    device = pos.device
    n, n_dim = pos.size(0), pos.size(1)
    half = block_size / 2.0
    step = block_size * (1.0 - overlap)
    sigma = float(sigma_scale * half * (n_dim**0.5))

    lo = pos.amin(dim=0)
    K = math.ceil(block_size / step)
    n_per_dim = (((pos.amax(dim=0) - lo) / step).ceil().long() + 1).clamp_min(1)

    strides = torch.ones(n_dim, device=device, dtype=torch.long)
    for d in range(n_dim - 2, -1, -1):
        strides[d] = strides[d + 1] * int(n_per_dim[d + 1].item())

    i_hi = ((pos - lo) / step).floor().long().clamp(min=torch.zeros_like(n_per_dim), max=n_per_dim - 1)  # (N, D)
    i_lo = (i_hi - K + 1).clamp_min(0)  # (N, D)

    arange_n = torch.arange(n, device=device)
    block_flat_list: List[Tensor] = []
    point_id_list: List[Tensor] = []
    weight_list: List[Tensor] = []

    for offsets in itertools.product(range(K), repeat=n_dim):
        off = i_lo.new_tensor(offsets)
        i = i_lo + off  # (N, D)
        valid = (i <= i_hi).all(dim=1)
        if not valid.any():
            continue
        i_v = i[valid]
        if mode == "gaussian":
            centres_v = lo + half + i_v.to(pos.dtype) * step
            dist = torch.linalg.norm(pos[valid] - centres_v, dim=-1)
            w = gaussian_weights(dist, sigma)
        else:
            w = pos.new_ones(int(valid.sum()))
        block_flat_list.append((i_v * strides).sum(dim=1))
        point_id_list.append(arange_n[valid])
        weight_list.append(w)

    block_flat = torch.cat(block_flat_list)
    point_ids = torch.cat(point_id_list)
    weights = torch.cat(weight_list)

    sort_idx = block_flat.argsort(stable=True)
    _, counts = torch.unique_consecutive(block_flat[sort_idx], return_counts=True)
    sizes = counts.tolist()

    point_groups = point_ids[sort_idx].split(sizes)
    weight_groups = weights[sort_idx].split(sizes)
    return list(point_groups), list(weight_groups)


@torch.no_grad()
def sliding_window_inference(
    data: Dict[str, Any],
    *,
    predictor: Callable[[Dict[str, Any]], Tensor],
    block_size: float,
    overlap: float = 0.0,
    mode: WindowMode = "constant",
    sigma_scale: float = 0.125,
    roi_num_points: Optional[int] = None,
    softmax: bool = True,
    transform: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
    pos_key: str = DataKeys.POS,
    batch_key: str = DataKeys.BATCH,
    progress: bool = False,
    seed: Optional[int] = None,
) -> Tensor:
    r"""Sliding-window inference for large-scale point cloud segmentation.

    Places block centers on a regular grid with step
    $\text{block\_size} \cdot (1 - \text{overlap})$ and calls `predictor` once
    per non-empty block. Each point's predictions from all covering blocks are
    accumulated with distance-based weights and divided by total weight.

    With `overlap=0` and `mode="constant"`, each point lands in exactly one
    block and the weight division is a no-op.

    Args:
        data: Dict of per-point tensors. Must contain `pos` (shape $(N, D)$)
            and `batch` (shape $(N,)$). Extra per-point tensors are sliced to
            the active block automatically. Non-tensor entries and tensors with
            a different leading dim flow through unchanged.
        predictor: Callable taking a per-block data dict and returning logits
            of shape $(M, C_\text{out})$, where $M$ is the number of points in
            the block. The output channel count $C_\text{out}$ is inferred from
            the first call.
        block_size: Side length of each cubic block, in the same units as
            `data[pos_key]`.
        overlap: Fraction of `block_size` shared between adjacent blocks, in
            $[0, 1)$. `0.0` gives a strict non-overlapping partition;
            `0.5` means adjacent block centers are spaced half a block apart.
        mode: Per-point weight within each block. `"constant"` gives equal
            weight to all points in the block. `"gaussian"` weights by
            $\exp(-d^2 / 2\sigma^2)$ where $d$ is the distance to the block
            center and $\sigma = \text{sigma\_scale} \cdot \text{block\_size}
            \cdot \sqrt{D} / 2$.
        sigma_scale: Gaussian sigma scale factor. Only used when
            `mode="gaussian"`.
        roi_num_points: Optional cap on points per `predictor` call. Blocks
            exceeding this are split into random sub-batches; every point in
            the block is still predicted exactly once per block pass.
            `None` passes the whole block in one call.
        softmax: If `True`, softmax each block's logits before accumulating.
            Use `True` when averaging predictions across multiple blocks or
            TTA passes. Set `False` to accumulate raw logits.
        transform: Optional callable applied to each block's data dict before
            the predictor.
        pos_key: Dict key for the position tensor.
        batch_key: Dict key for the per-point batch index.
        progress: If `True`, show a `tqdm` progress bar per batch element.
        seed: RNG seed for sub-batch permutations when `roi_num_points` is set.

    Returns:
        Per-point output tensor of shape $(N, C_\text{out})$, containing a
        distance-weighted average of softmax probabilities when `softmax=True`
        or of raw logits when `softmax=False`.
    """
    if pos_key not in data:
        raise KeyError(f"`data` is missing the required key {pos_key!r}.")
    if batch_key not in data:
        raise KeyError(f"`data` is missing the required key {batch_key!r}.")
    if block_size <= 0.0:
        raise ValueError(f"`block_size` must be > 0, got {block_size}.")
    if not 0.0 <= overlap < 1.0:
        raise ValueError(f"`overlap` must be in [0, 1), got {overlap}.")
    if mode not in ("constant", "gaussian"):
        raise ValueError(f"`mode` must be 'constant' or 'gaussian', got {mode!r}.")
    if roi_num_points is not None and roi_num_points < 1:
        raise ValueError(f"`roi_num_points` must be >= 1 or None, got {roi_num_points}.")

    pos = data[pos_key]
    batch = data[batch_key]
    device = pos.device
    n_total = pos.size(0)
    output: Optional[Tensor] = None

    rng = torch.Generator(device=device)
    if seed is not None:
        rng.manual_seed(int(seed))

    for b in torch.unique(batch).tolist():
        idx_b = torch.where(batch == b)[0]
        n_b = int(idx_b.numel())
        if n_b == 0:
            continue
        pos_b = pos[idx_b]
        data_b = index_select_dict(data, idx_b, n_total)

        point_groups, weight_groups = _assign_point_blocks(
            pos_b, block_size=block_size, overlap=overlap, mode=mode, sigma_scale=sigma_scale
        )

        scores_b: Optional[Tensor] = None
        weights_b = torch.zeros(n_b, device=device, dtype=torch.float32)

        for point_ids, w in tqdm(
            zip(point_groups, weight_groups),
            total=len(point_groups),
            desc=f"batch {int(b)}",
            leave=False,
            disable=not progress,
        ):
            chunks = split_chunks(int(point_ids.numel()), roi_num_points, rng)

            for chunk_local in chunks:
                chunk_idx = point_ids[chunk_local]
                window = index_select_dict(data_b, chunk_idx, n_b)
                window[batch_key] = torch.zeros(chunk_idx.numel(), device=device, dtype=torch.long)
                if transform is not None:
                    window = transform(window)
                logits = predictor(window)

                if scores_b is None:
                    scores_b = torch.zeros(n_b, int(logits.size(-1)), device=device, dtype=torch.float32)

                preds = torch.softmax(logits, dim=-1) if softmax else logits
                chunk_w = w[chunk_local]
                scores_b.index_add_(0, chunk_idx, preds * chunk_w.unsqueeze(-1))
                weights_b.index_add_(0, chunk_idx, chunk_w)

        if scores_b is None:
            continue

        if output is None:
            output = torch.zeros(n_total, int(scores_b.size(1)), device=device, dtype=torch.float32)

        output[idx_b] = scores_b / weights_b.clamp_min(1e-6).unsqueeze(-1)

    if output is None:
        return pos.new_zeros((0, 0))
    return output


class SlidingWindowInferer(Inferer):
    r"""Sliding-window inferer for large-scale point cloud segmentation.

    Places block centers on a regular grid and accumulates per-point predictions
    across all blocks that contain each point, blended by distance-based weights.
    At `overlap=0` each point lands in exactly one block and the weight division
    is a no-op.

    All parameters are forwarded verbatim to `sliding_window_inference`.

    Example:
        ```python
        from torch_pointcloud.inferers import SlidingWindowInferer

        # Non-overlapping partition: one prediction per point.
        inferer = SlidingWindowInferer(block_size=6.0, overlap=0.0)

        # 25 % overlap with Gaussian blending at boundaries:
        inferer = SlidingWindowInferer(block_size=6.0, overlap=0.25, mode="gaussian")
        probs = inferer(data, predictor=lambda d: model(d["pos"], d["x"], d["batch"]))
        ```
    """

    def __init__(
        self,
        block_size: float,
        overlap: float = 0.0,
        mode: WindowMode = "constant",
        sigma_scale: float = 0.125,
        roi_num_points: Optional[int] = None,
        softmax: bool = True,
        transform: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
        pos_key: str = DataKeys.POS,
        batch_key: str = DataKeys.BATCH,
        progress: bool = False,
        seed: Optional[int] = None,
    ) -> None:
        self.block_size = block_size
        self.overlap = overlap
        self.mode = mode
        self.sigma_scale = sigma_scale
        self.roi_num_points = roi_num_points
        self.softmax = softmax
        self.transform = transform
        self.pos_key = pos_key
        self.batch_key = batch_key
        self.progress = progress
        self.seed = seed

    def forward(
        self,
        data: Dict[str, Any],
        predictor: Callable[[Dict[str, Any]], Tensor],
    ) -> Tensor:
        return sliding_window_inference(
            data,
            predictor=predictor,
            block_size=self.block_size,
            overlap=self.overlap,
            mode=self.mode,
            sigma_scale=self.sigma_scale,
            roi_num_points=self.roi_num_points,
            softmax=self.softmax,
            transform=self.transform,
            pos_key=self.pos_key,
            batch_key=self.batch_key,
            progress=self.progress,
            seed=self.seed,
        )
