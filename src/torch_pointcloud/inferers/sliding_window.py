r"""Sliding-window inference for large-scale point cloud segmentation.

Tiles the scene with axis-aligned cubic blocks and accumulates per-point
predictions across all blocks that contain each point. Adjacent blocks overlap
by a configurable fraction, reducing seam artifacts at block boundaries.

When `overlap=0.0` (default), blocks form a non-overlapping partition and each
point is predicted exactly once. Higher `overlap` values increase coverage
redundancy; points near block boundaries receive a weighted average of
predictions from all blocks that contain them.

Per-block pre-processing goes through the `transform` argument.

!!! warning "`block_size` is in the units of `data[pos_key]`, not always meters"

    The inferer tiles in whatever coordinate space `pos` is in at call time. If
    positions are voxel indices after upstream voxelization, `block_size` is a
    voxel count, not meters. A scene voxelized at $2\,\text{cm}$ tiled with
    `block_size=200` gives $4\,\text{m}$ blocks.
"""

import itertools
import math
from typing import Any, Callable, Dict, List, Literal, Optional, Sequence, Tuple

import torch
from torch import Tensor
from tqdm import tqdm

from torch_pointcloud.utils.data import DataKeys, collate

from ._utils import apply_transform, check_batch_alignment, gaussian_weights, index_select_dict, split_chunks
from .inferer import Inferer

WindowMode = Literal["constant", "gaussian"]
AggregateMode = Literal["mean", "max", "vote"]


def _assign_point_blocks(
    pos: Tensor,
    *,
    block_size: float,
    overlap: float,
    mode: WindowMode,
    sigma_scale: float,
    dims: Optional[Sequence[int]] = None,
    padding: float = 0.0,
) -> Tuple[List[Tensor], List[Tensor], List[Tensor]]:
    r"""Group point indices by the overlapping blocks that contain them.

    Tiles `pos` (one batch element, shape $(N, D)$) with cubic blocks of side
    `block_size` along the axes listed in `dims` (default: all of `pos`'s
    spatial axes). Adjacent blocks are spaced
    $\text{block\_size} \cdot (1 - \text{overlap})$ apart on each tiled axis.
    Untiled axes contribute a single full-span slab, so the per-block point set
    is the intersection of the tiled-axis blocks with the entire untiled extent
    of `pos`.

    Membership is containment: a point belongs to a block when it lies inside
    the block's extent, extended by `padding` (in the units of `pos`) on every
    tiled axis. With `padding > 0` points within a thin margin of a boundary
    fall into the neighboring block too.

    Args:
        pos: Point positions for a single batch element, shape $(N, D)$.
        block_size: Side length of each cubic block, in the units of `pos`.
        overlap: Fraction of `block_size` shared between adjacent blocks, in $[0, 1)$.
        mode: Per-point weight scheme. `"constant"` weights every point equally;
            `"gaussian"` weights by $\exp(-d^2 / 2\sigma^2)$ on the distance $d$ to
            the block center (computed across the tiled axes).
        sigma_scale: Gaussian sigma scale factor (only used when `mode="gaussian"`).
        dims: Axes (indices into `pos`'s last dim) to tile. `None` tiles every axis (current behavior).
        padding: Extra margin (in `pos` units) added to each block's extent on every
            tiled axis. Useful when a thin boundary of context should be included.

    Returns:
        `(point_groups, weight_groups, bbox_groups)`: equal-length lists with one
        entry per non-empty block. `point_groups[j]` holds the local point indices
        of block $j$; `weight_groups[j]` holds their per-point blend weights;
        `bbox_groups[j]` is the block's axis-aligned bounding box as a $(2D,)$
        tensor laid out $[\,\min_0, \ldots, \min_{D-1},\; \max_0, \ldots, \max_{D-1}\,]$.
        Tiled axes get the block's grid-defined extent; untiled axes get the full
        $(\min, \max)$ of `pos` along that axis.
    """
    device = pos.device
    n, n_dim = pos.size(0), pos.size(1)
    half = block_size / 2.0
    step = block_size * (1.0 - overlap)

    if dims is None:
        tile_axes = list(range(n_dim))
    else:
        tile_axes = list(dims)
    n_tiled = len(tile_axes)
    if n_tiled == 0:
        raise ValueError("`dims` must contain at least one axis.")

    pos_tiled = pos[:, tile_axes]
    sigma = float(sigma_scale * half * (n_tiled**0.5))

    lo_full = pos.amin(dim=0)
    hi_full = pos.amax(dim=0)
    lo = pos_tiled.amin(dim=0)
    K = math.ceil((block_size + 2.0 * padding) / step)
    n_per_dim = (((pos_tiled.amax(dim=0) - lo) / step).ceil().long() + 1).clamp_min(1)

    strides = torch.ones(n_tiled, device=device, dtype=torch.long)
    for d in range(n_tiled - 2, -1, -1):
        strides[d] = strides[d + 1] * int(n_per_dim[d + 1].item())

    i_hi = (
        ((pos_tiled - lo + padding) / step).floor().long().clamp(min=torch.zeros_like(n_per_dim), max=n_per_dim - 1)
    )  # (N, D_t)
    i_lo = (i_hi - K + 1).clamp_min(0)  # (N, D_t)

    arange_n = torch.arange(n, device=device)
    block_flat_list: List[Tensor] = []
    point_id_list: List[Tensor] = []
    weight_list: List[Tensor] = []
    center_idx_list: List[Tensor] = []  # block index in tiled-grid coords

    for offsets in itertools.product(range(K), repeat=n_tiled):
        off = i_lo.new_tensor(offsets)
        i = i_lo + off  # (N, D_t)
        valid = (i <= i_hi).all(dim=1)
        if not valid.any():
            continue

        block_lo = lo + i.to(pos.dtype) * step
        # Blocks are half-open at the top so a no-overlap tiling predicts every point exactly once and the
        # membership test agrees with the K = ceil(width / step) enumeration when width is a step multiple.
        # The float bound test alone is not enough: on grid-aligned coordinates `(p - lo) / step` can round
        # across an integer in fp32, landing `p` exactly on a bound of its own `i_hi` block on either side.
        # Membership of that block is forced by index identity so every point keeps at least one block.
        inside = ((pos_tiled >= block_lo - padding) & (pos_tiled < block_lo + block_size + padding)).all(dim=1)
        inside = inside | (i == i_hi).all(dim=1)
        valid = valid & inside
        if not valid.any():
            continue

        i_v = i[valid]
        if mode == "gaussian":
            centers_v_tiled = lo + half + i_v.to(pos.dtype) * step  # (M, D_t)
            dist = torch.linalg.norm(pos_tiled[valid] - centers_v_tiled, dim=-1)
            # exp underflows to exactly 0 in float32 beyond ~13 sigma; the floor keeps every
            # covered point at a nonzero blend weight so its predictions survive the division.
            w = gaussian_weights(dist, sigma).clamp_min(1e-12)
        else:
            # Blend weights are float even when `pos` holds integer grid coordinates.
            w = pos.new_ones(int(valid.sum()), dtype=torch.float32)
        block_flat_list.append((i_v * strides).sum(dim=1))
        point_id_list.append(arange_n[valid])
        weight_list.append(w)
        center_idx_list.append(i_v)

    block_flat = torch.cat(block_flat_list)
    point_ids = torch.cat(point_id_list)
    weights = torch.cat(weight_list)
    centers_idx_all = torch.cat(center_idx_list, dim=0)  # (sum_M, D_t)

    sort_idx = block_flat.argsort(stable=True)
    block_sorted = block_flat[sort_idx]
    _, first_idx = torch.unique_consecutive(block_sorted, return_inverse=True)
    unique_pos = (first_idx[1:] != first_idx[:-1]).nonzero(as_tuple=False).flatten() + 1
    unique_pos = torch.cat([torch.zeros(1, dtype=torch.long, device=device), unique_pos])
    _, counts = torch.unique_consecutive(block_sorted, return_counts=True)
    sizes = counts.tolist()

    block_lo_tiled = lo + centers_idx_all[sort_idx][unique_pos].to(pos.dtype) * step
    block_hi_tiled = block_lo_tiled + block_size
    bbox_lo_template = lo_full.clone()
    bbox_hi_template = hi_full.clone()
    bbox_groups: List[Tensor] = []
    for row_lo, row_hi in zip(block_lo_tiled, block_hi_tiled):
        b_lo = bbox_lo_template.clone()
        b_hi = bbox_hi_template.clone()
        for j, ax in enumerate(tile_axes):
            b_lo[ax] = row_lo[j]
            b_hi[ax] = row_hi[j]
        bbox_groups.append(torch.cat([b_lo, b_hi]))

    point_groups = point_ids[sort_idx].split(sizes)
    weight_groups = weights[sort_idx].split(sizes)
    return list(point_groups), list(weight_groups), bbox_groups


def _build_window(
    data_b: Dict[str, Any],
    point_ids: Tensor,
    bbox: Tensor,
    *,
    n_b: int,
    transform: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]],
    pos_key: str,
    batch_key: str,
    block_bbox_key: str,
    inverse_key: Optional[str],
) -> Tuple[Dict[str, Any], Optional[Tensor]]:
    """Slice one block out of `data_b` and run `transform` over it, returning it and its inverse map."""
    n_block = int(point_ids.numel())
    window = index_select_dict(data_b, point_ids, n_b)
    window[batch_key] = torch.zeros(n_block, device=point_ids.device, dtype=torch.long)
    window[block_bbox_key] = bbox
    return apply_transform(window, transform, pos_key=pos_key, inverse_key=inverse_key)


def _predict_packed(
    samples: List[Dict[str, Any]],
    *,
    predictor: Callable[[Dict[str, Any]], Tensor],
    sw_batch_size: int,
    apply_softmax: bool,
    pos_key: str,
    batch_key: str,
) -> Tensor:
    """Collate `samples` into packs of at most `sw_batch_size` elements, predict each pack, concatenate."""
    parts: List[Tensor] = []
    for start in range(0, len(samples), sw_batch_size):
        packed = collate(samples[start : start + sw_batch_size], batch_from=pos_key, batch_key=batch_key)
        logits = predictor(packed)
        parts.append(torch.softmax(logits, dim=-1) if apply_softmax else logits)
    return torch.cat(parts)


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
    sw_batch_size: int = 1,
    softmax: bool = False,
    aggregate: AggregateMode = "mean",
    transform: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
    dims: Optional[Sequence[int]] = None,
    padding: float = 0.0,
    pos_key: str = DataKeys.POS,
    batch_key: str = DataKeys.BATCH,
    block_bbox_key: str = DataKeys.BLOCK_BBOX,
    inverse_key: Optional[str] = None,
    progress: bool = False,
    seed: Optional[int] = None,
) -> Tensor:
    r"""Sliding-window inference for large-scale point cloud segmentation.

    Places block centers on a regular grid with step
    $\text{block\_size} \cdot (1 - \text{overlap})$ and calls `predictor` once
    per non-empty block. Each point's predictions from all covering blocks are
    combined by `aggregate`: a distance-weighted average (`"mean"`), the single
    most confident prediction (`"max"`), or a count of hard votes (`"vote"`).

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
            center across the tiled axes and $\sigma = \text{sigma\_scale}
            \cdot \text{block\_size} \cdot \sqrt{D_\text{tiled}} / 2$ with
            $D_\text{tiled}$ the number of tiled axes.
        sigma_scale: Gaussian sigma scale factor. Only used when
            `mode="gaussian"`.
        roi_num_points: Maximum points in one batch element. A block with more is split into random
            sub-batches, each predicted exactly once per block pass. `None` passes the whole block as
            one element. Use it when the architecture needs a bounded or fixed $N$: pair it with a
            `DivisiblePad`-style `transform` that pads each block to a multiple of `roi_num_points`
            and writes its source-to-padded index map under `inverse_key`, and every element is then
            exactly `roi_num_points` long.
        sw_batch_size: Number of elements collated into one `predictor` call. Purely a throughput
            knob: it changes the number of calls, never the output and never the size of an element.
            Set together with `roi_num_points`, a call holds at most `sw_batch_size` elements of at
            most `roi_num_points` points each. Small blocks are dominated by per-call overhead, where
            a larger value is several times faster. Output-neutrality assumes `predictor` respects the
            packed batch index; one that pools globally without it will differ at `sw_batch_size > 1`.
            Peak memory holds `sw_batch_size` transformed windows at once, so it scales with
            `block_size` rather than with `roi_num_points`.
        softmax: If `True`, softmax each block's logits before accumulating.
            Use `True` when averaging predictions across multiple blocks or
            TTA passes. Set `False` to accumulate raw logits. `"max"` and
            `"vote"` aggregation always read confidences off the softmax.
        aggregate: How the predictions of the blocks covering a point are
            combined. `"mean"`: distance-weighted average of the (softmax)
            predictions. `"max"`: winner-takes-all, each point keeps the
            prediction of the block that is most confident about it (the
            PVCNN / PointCNN scene merge). `"vote"`: each block casts one hard
            vote (its argmax) per point and the output holds the weighted vote
            fractions, so `argmax` is the majority label (the PointNet++ protocol).
        transform: Optional callable applied to each block's data dict before
            the predictor. The transform sees the whole block; if it changes the
            row count (pad, voxelize, ...) it must record a source-to-predictor
            index map under `inverse_key` so the inferer can gather predictions
            back to the original block points.
        dims: Axes (indices into `pos`'s last dim) to tile. `None` tiles
            every axis (cubic blocks). Pass `(0, 1)` for 2D tiling that leaves
            the third axis spanning the full scene height.
        padding: Extra margin (in `pos` units) extending each block's membership
            on every tiled axis. Useful for including a thin context guard band.
        pos_key: Dict key for the position tensor.
        batch_key: Dict key for the per-point batch index.
        block_bbox_key: Dict key under which the block bounding box is exposed to
            the `transform` callable.
        inverse_key: Dict key under which a row-altering `transform` records a
            source-to-predictor long index map of shape $(N_\text{block},)$ with
            values in $[0, N_\text{window})$, where $N_\text{block}$ is the
            pre-transform block size and $N_\text{window}$ is the post-transform
            size. When set, any scene-level value at this key is dropped from the
            window before `transform` runs, so a registered pipeline's `inverse`
            never becomes the prior the block map composes through; the inferer
            then pops the block map before calling the predictor and gathers
            predictions back to block-local rows. Leave `None` when the transform
            preserves row count, or when no transform is used.
        progress: If `True`, show a `tqdm` progress bar per batch element.
        seed: RNG seed for sub-batch permutations when `roi_num_points` is set. `None` draws from the global generator, so
            `torch.manual_seed` seeds the inferer together with the transforms.

    Returns:
        Per-point output tensor of shape $(N, C_\text{out})$: with
        `aggregate="mean"` a distance-weighted average of softmax probabilities
        when `softmax=True` or of raw logits when `softmax=False`; with
        `"max"` the most confident block's probabilities; with `"vote"` the
        per-class vote fractions. An empty scene ($N = 0$) returns
        a $(0, 0)$ tensor: the predictor is never called, so the channel count
        cannot be inferred.
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
    if aggregate not in ("mean", "max", "vote"):
        raise ValueError(f"`aggregate` must be 'mean', 'max' or 'vote', got {aggregate!r}.")
    if roi_num_points is not None and roi_num_points < 1:
        raise ValueError(f"`roi_num_points` must be >= 1 or None, got {roi_num_points}.")
    if sw_batch_size < 1:
        raise ValueError(f"`sw_batch_size` must be >= 1, got {sw_batch_size}.")
    if padding < 0.0:
        raise ValueError(f"`padding` must be >= 0, got {padding}.")

    pos = data[pos_key]
    batch = data[batch_key]
    check_batch_alignment(pos, batch, pos_key, batch_key)
    device = pos.device
    n_total = pos.size(0)
    output: Optional[Tensor] = None

    rng = None if seed is None else torch.Generator(device=device).manual_seed(int(seed))
    apply_softmax = softmax or aggregate != "mean"

    for b in torch.unique(batch).tolist():
        idx_b = torch.where(batch == b)[0]
        n_b = int(idx_b.numel())
        if n_b == 0:
            continue
        pos_b = pos[idx_b]
        data_b = index_select_dict(data, idx_b, n_total)

        point_groups, weight_groups, bbox_groups = _assign_point_blocks(
            pos_b,
            block_size=block_size,
            overlap=overlap,
            mode=mode,
            sigma_scale=sigma_scale,
            dims=dims,
            padding=padding,
        )

        scores_b: Optional[Tensor] = None
        weights_b = torch.zeros(n_b, device=device, dtype=torch.float32)
        confidence_b = torch.full((n_b,), -1.0, device=device, dtype=torch.float32)

        blocks = list(zip(point_groups, weight_groups, bbox_groups))
        for start in tqdm(
            range(0, len(blocks), sw_batch_size),
            total=math.ceil(len(blocks) / sw_batch_size),
            desc=f"batch {int(b)}",
            leave=False,
            disable=not progress,
        ):
            batch_blocks = blocks[start : start + sw_batch_size]
            prepared: List[Tuple[int, Optional[Tensor], List[Tensor]]] = []  # (n_window, inverse_map, chunks)
            samples: List[Dict[str, Any]] = []

            for point_ids, _, bbox in batch_blocks:
                window, inverse_map = _build_window(
                    data_b,
                    point_ids,
                    bbox,
                    n_b=n_b,
                    transform=transform,
                    pos_key=pos_key,
                    batch_key=batch_key,
                    block_bbox_key=block_bbox_key,
                    inverse_key=inverse_key,
                )
                n_window = int(window[pos_key].size(0))
                chunks = split_chunks(n_window, roi_num_points, rng, device)
                samples.extend(index_select_dict(window, chunk, n_window) for chunk in chunks)
                prepared.append((n_window, inverse_map, chunks))

            if not samples:
                continue

            preds = _predict_packed(
                samples,
                predictor=predictor,
                sw_batch_size=sw_batch_size,
                apply_softmax=apply_softmax,
                pos_key=pos_key,
                batch_key=batch_key,
            )

            num_classes = int(preds.size(-1))
            if scores_b is None:
                scores_b = torch.zeros(n_b, num_classes, device=device, dtype=torch.float32)
            chunk_preds = iter(preds.split([int(chunk.numel()) for _, _, chunks in prepared for chunk in chunks]))

            for (point_ids, w, _), (n_window, inverse_map, chunks) in zip(batch_blocks, prepared):
                block_preds = torch.zeros(n_window, num_classes, device=device, dtype=torch.float32)
                for chunk in chunks:
                    block_preds[chunk] = next(chunk_preds).to(block_preds)

                preds_at_block = block_preds if inverse_map is None else block_preds[inverse_map]

                if aggregate == "max":
                    confidence, _ = preds_at_block.max(dim=-1)
                    better = confidence > confidence_b[point_ids]
                    scores_b[point_ids[better]] = preds_at_block[better]
                    confidence_b[point_ids[better]] = confidence[better]
                    weights_b[point_ids] = 1.0
                    continue

                if aggregate == "vote":
                    preds_at_block = torch.nn.functional.one_hot(
                        preds_at_block.argmax(dim=-1), num_classes=num_classes
                    ).to(scores_b.dtype)

                scores_b.index_add_(0, point_ids, preds_at_block * w.unsqueeze(-1))
                weights_b.index_add_(0, point_ids, w)

        if scores_b is None:
            continue

        if output is None:
            output = torch.zeros(n_total, int(scores_b.size(1)), device=device, dtype=torch.float32)

        covered = weights_b > 0
        scores_b[covered] = scores_b[covered] / weights_b[covered].unsqueeze(-1)
        output[idx_b] = scores_b

    if output is None:
        return pos.new_zeros((0, 0))
    return output


class SlidingWindowInferer(Inferer):
    r"""Sliding-window inferer for large-scale point cloud segmentation.

    Places block centers on a regular grid and accumulates per-point predictions
    across all blocks that contain each point, blended by distance-based weights.
    At `overlap=0` each point lands in exactly one block and the weight division
    is a no-op.

    All parameters are forwarded verbatim to `sliding_window_inference`, except `seed`, which is offset by
    the number of calls the instance has made: repeated calls (e.g. `TTAInferer` votes) draw different random
    numbers, and a fresh instance replays the same sequence.

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
        sw_batch_size: int = 1,
        softmax: bool = False,
        aggregate: AggregateMode = "mean",
        transform: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
        dims: Optional[Sequence[int]] = None,
        padding: float = 0.0,
        pos_key: str = DataKeys.POS,
        batch_key: str = DataKeys.BATCH,
        block_bbox_key: str = DataKeys.BLOCK_BBOX,
        inverse_key: Optional[str] = None,
        progress: bool = False,
        seed: Optional[int] = None,
    ) -> None:
        self.block_size = block_size
        self.overlap = overlap
        self.mode = mode
        self.sigma_scale = sigma_scale
        self.roi_num_points = roi_num_points
        self.sw_batch_size = sw_batch_size
        self.softmax = softmax
        self.aggregate = aggregate
        self.transform = transform
        self.dims = dims
        self.padding = padding
        self.pos_key = pos_key
        self.batch_key = batch_key
        self.block_bbox_key = block_bbox_key
        self.inverse_key = inverse_key
        self.progress = progress
        self.seed = seed
        self._num_calls = 0

    def forward(
        self,
        data: Dict[str, Any],
        predictor: Callable[[Dict[str, Any]], Tensor],
    ) -> Tensor:
        seed = None if self.seed is None else self.seed + self._num_calls
        self._num_calls += 1
        return sliding_window_inference(
            data,
            predictor=predictor,
            block_size=self.block_size,
            overlap=self.overlap,
            mode=self.mode,
            sigma_scale=self.sigma_scale,
            roi_num_points=self.roi_num_points,
            sw_batch_size=self.sw_batch_size,
            softmax=self.softmax,
            aggregate=self.aggregate,
            transform=self.transform,
            dims=self.dims,
            padding=self.padding,
            pos_key=self.pos_key,
            batch_key=self.batch_key,
            block_bbox_key=self.block_bbox_key,
            inverse_key=self.inverse_key,
            progress=self.progress,
            seed=seed,
        )
