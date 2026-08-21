r"""KNN-window inference for large-scale point cloud segmentation.

Implements a coverage-driven iterative loop: each step selects the least-covered
point as a window centre, crops its $k$ nearest neighbours, runs the predictor on
that crop, and accumulates per-point predictions weighted by distance to the centre.
The loop ends once every point's coverage score exceeds a threshold.

Windows adapt to point density and naturally prioritise under-covered regions.
Because windows overlap, each point is typically predicted several times; overlapping
predictions are combined by a weighted mean or an exponential moving average (EMA).
"""

import warnings
from typing import Any, Callable, Dict, List, Literal, Optional

import torch
from torch import Tensor
from tqdm import tqdm

from torch_pointcloud.utils.data import DataKeys

from ._utils import gaussian_weights, index_select_dict
from .inferer import Inferer

WindowMode = Literal["constant", "gaussian"]
AggregateMode = Literal["weighted_mean", "ema"]


def _knn_centres(pos_src: Tensor, centres: Tensor, k: int) -> Tensor:
    r"""Per-centre $k$-nearest indices into `pos_src`.

    Uses `cdist + topk` directly instead of `torch_pointcloud.utils.cluster.knn`
    because the latter falls back to `torch_cluster.knn` for large source clouds,
    which has a hard $k \leq 100$ ceiling on CUDA. Since $M = \text{sw\_batch\_size}$
    is always small, the dense $(M, N)$ distance matrix is cheap.

    Returns:
        Indices tensor of shape $(M, k)$.
    """
    dist = torch.cdist(centres, pos_src)
    _, idx = dist.topk(k, dim=-1, largest=False)
    return idx


def _gaussian_window_weights(distances: Tensor, sigma_scale: float, eps: float = 1e-12) -> Tensor:
    r"""Per-window gaussian weights with radius $\sigma = \text{sigma\_scale} \cdot \max_i d_i$.

    Vectorised over a batched edge tensor of shape $(M, K)$ so that each row uses its
    own per-window radius. The falloff itself is delegated to `gaussian_weights`.
    """
    sigma = distances.amax(dim=-1, keepdim=True) * float(sigma_scale)
    return gaussian_weights(distances, sigma, eps)


@torch.no_grad()
def knn_window_inference(
    data: Dict[str, Any],
    *,
    predictor: Callable[[Dict[str, Any]], Tensor],
    roi_num_points: int = 65_536,
    sw_batch_size: int = 1,
    overlap: float = 0.5,
    mode: WindowMode = "constant",
    sigma_scale: float = 0.125,
    aggregate: AggregateMode = "weighted_mean",
    ema_smoothing: float = 0.95,
    softmax: bool = False,
    transform: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
    pos_key: str = DataKeys.POS,
    batch_key: str = DataKeys.BATCH,
    progress: bool = False,
    seed: Optional[int] = None,
) -> Tensor:
    r"""Iterative KNN-window inference for large-scale point cloud segmentation.

    Maintains a per-point coverage score initialised to small random noise. Each
    iteration selects the `sw_batch_size` least-covered points as window centres,
    crops their $k$ nearest neighbours, runs `predictor` on the packed crop, and
    accumulates per-point predictions weighted by distance to the centre. The loop
    ends once every point's coverage score exceeds `overlap`.

    `aggregate` controls how overlapping window predictions are combined:

    - `"weighted_mean"`: accumulates distance-weighted logits and divides by total
      weight at the end, producing a weighted-average logit tensor.
    - `"ema"`: per-update softmax EMA
      ($\text{new} = \alpha \cdot \text{old} + (1 - \alpha) \cdot \text{softmax}(\text{logits})$),
      outputting calibrated probabilities without a final softmax step.

    Args:
        data: Dict of per-point tensors. Must contain `pos` (shape $(N, D)$) and
            `batch` (shape $(N,)$). Any additional tensor whose first dim equals
            $N$ (e.g. `color`, `intensity`, `segment`) is automatically sliced
            to the active window. Non-tensor entries and tensors with a different
            leading dim (scalar metadata) flow through unchanged.
        predictor: Callable taking a per-window data dict and returning per-point
            logits of shape $(M, C_\text{out})$, where $M$ is the (packed) total
            point count of the windowed batch and the per-window batch index lives at
            `window[batch_key]`. The output channel count $C_\text{out}$ is inferred
            from the first call.
        roi_num_points: Number of points per window (the window size $k$).
        sw_batch_size: Number of windows packed into one `predictor` call.
            Higher values reduce launch overhead at the cost of more memory.
        overlap: Coverage threshold in $(0, 1)$. The loop stops once every point
            has accumulated at least `overlap` possibility mass. Higher values give
            more thorough coverage at the cost of more iterations. `0.0` is rejected:
            the initial coverage scores already satisfy a zero threshold, so no
            window would ever be predicted.
        mode: Distance weighting for each window. `"constant"` gives equal weight
            to every point; `"gaussian"` weights by $\exp(-d^2 / 2\sigma^2)$ with
            $\sigma = \text{sigma\_scale} \cdot \max_i d_i$. `"gaussian"` requires
            `aggregate="weighted_mean"`; EMA updates ignore distance weights.
        sigma_scale: Gaussian sigma scale factor (only used when `mode="gaussian"`).
        aggregate: How predictions from overlapping windows are combined.
            `"weighted_mean"`: weighted-average logits (divide by total weight at end).
            `"ema"`: softmax EMA
            ($\text{new} = \alpha \cdot \text{old} + (1 - \alpha) \cdot \text{softmax}(\text{logits})$);
            use `sw_batch_size=1` with EMA to match the reference evaluation protocol.
        ema_smoothing: EMA factor $\alpha \in [0, 1)$ used when `aggregate="ema"`.
        softmax: If `True`, softmax each window's logits before the weighted-mean
            accumulation, so overlapping windows average probabilities instead of
            raw logits. Ignored when `aggregate="ema"`, which always accumulates
            softmax probabilities.
        transform: Optional callable applied to each window's data dict before the
            predictor (typical example: `T.Shift(keys=DataKeys.POS, method="centroid")`).
        pos_key: Dict key for the position tensor.
        batch_key: Dict key for the per-point batch index.
        progress: If `True`, show a `tqdm` progress bar per batch element.
        seed: RNG seed for the per-point initial possibility scores.

    Returns:
        Per-point output tensor of shape $(N, C_\text{out})$. Aggregation produces
        logits when `aggregate="weighted_mean"` (probabilities with `softmax=True`)
        and probabilities when `aggregate="ema"`. An empty scene ($N = 0$) returns
        a $(0, 0)$ tensor: the predictor is never called, so the channel count
        cannot be inferred.
    """
    if pos_key not in data:
        raise KeyError(f"`data` is missing the required key {pos_key!r}.")
    if batch_key not in data:
        raise KeyError(f"`data` is missing the required key {batch_key!r}.")
    if not 0.0 < overlap < 1.0:
        raise ValueError(f"`overlap` must be in (0, 1), got {overlap}.")
    if mode not in ("constant", "gaussian"):
        raise ValueError(f"`mode` must be 'constant' or 'gaussian', got {mode!r}.")
    if aggregate not in ("weighted_mean", "ema"):
        raise ValueError(f"`aggregate` must be 'weighted_mean' or 'ema', got {aggregate!r}.")
    if sw_batch_size < 1:
        raise ValueError(f"`sw_batch_size` must be >= 1, got {sw_batch_size}.")
    if not 0.0 <= ema_smoothing < 1.0:
        raise ValueError(f"`ema_smoothing` must be in [0, 1), got {ema_smoothing}.")
    if aggregate == "ema" and mode == "gaussian":
        raise ValueError(
            "`mode='gaussian'` is incompatible with `aggregate='ema'`: EMA updates blend by `ema_smoothing`, not "
            "by per-point distance weights. Use `aggregate='weighted_mean'` or `mode='constant'`."
        )
    if aggregate == "ema" and sw_batch_size > 1:
        warnings.warn(
            "`aggregate='ema'` with `sw_batch_size > 1` applies per-window EMA updates sequentially; "
            "overlapping points are updated multiple times per step in an order-dependent way."
            "Use `sw_batch_size=1` for EMA aggregation.",
            stacklevel=2,
            category=UserWarning,
        )

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
        pos_b = pos[idx_b]
        data_b = index_select_dict(data, idx_b, n_total)

        k = min(roi_num_points, n_b)

        scores_b: Optional[Tensor] = None
        weights_b: Optional[Tensor] = None

        possibility = torch.rand(n_b, generator=rng, device=device, dtype=torch.float32) * 1e-3

        prev_covered = 0
        steps = 0
        with tqdm(total=n_b, desc=f"batch {int(b)}", leave=False, disable=not progress) as pbar:
            while True:
                min_p = float(possibility.min().item())
                if min_p >= overlap:
                    break

                sw = min(sw_batch_size, n_b)
                _, centre_local = torch.topk(possibility, sw, largest=False, sorted=False)
                centre_pos = pos_b[centre_local]
                local_idxs = _knn_centres(pos_b, centre_pos, k)
                flat_idxs = local_idxs.reshape(-1)
                per_window_dicts: List[Dict[str, Any]] = []
                for w_i in range(sw):
                    w_idx = local_idxs[w_i]
                    wd = index_select_dict(data_b, w_idx, n_b)
                    wd[batch_key] = torch.full((k,), w_i, device=device, dtype=torch.long)
                    if transform is not None:
                        wd = transform(wd)
                        if int(wd[pos_key].size(0)) != k:
                            raise ValueError(
                                f"`transform` must preserve each window's row count; got {k} -> "
                                f"{int(wd[pos_key].size(0))} rows."
                            )

                    per_window_dicts.append(wd)

                window_data: Dict[str, Any] = {}
                for key in per_window_dicts[0]:
                    values = [wd[key] for wd in per_window_dicts]
                    if torch.is_tensor(values[0]) and values[0].dim() > 0 and values[0].size(0) == k:
                        window_data[key] = torch.cat(values, dim=0)
                    else:
                        window_data[key] = values[0]

                window_logits = predictor(window_data)
                num_classes = int(window_logits.size(-1))
                window_logits = window_logits.reshape(sw, k, num_classes)

                if output is None:
                    output = torch.zeros(n_total, num_classes, device=device, dtype=torch.float32)
                if scores_b is None:
                    scores_b = torch.zeros(n_b, num_classes, device=device, dtype=torch.float32)
                    if aggregate == "weighted_mean":
                        weights_b = torch.zeros(n_b, device=device, dtype=torch.float32)

                distances = torch.linalg.norm(pos_b[local_idxs] - centre_pos.unsqueeze(1), dim=-1)
                if mode == "gaussian":
                    # exp underflows to exactly 0 in float32 beyond ~13 sigma; the floor keeps every
                    # windowed point at a nonzero blend weight so its predictions survive the division.
                    w = _gaussian_window_weights(distances, sigma_scale=sigma_scale).clamp_min(1e-12)
                else:
                    w = torch.ones_like(distances)

                if aggregate == "ema":
                    window_probs = torch.softmax(window_logits, dim=-1)
                    for w_i in range(sw):
                        idx_w = local_idxs[w_i]
                        scores_b[idx_w] = ema_smoothing * scores_b[idx_w] + (1.0 - ema_smoothing) * window_probs[w_i]
                else:
                    window_preds = torch.softmax(window_logits, dim=-1) if softmax else window_logits
                    weighted_preds = window_preds * w.unsqueeze(-1)
                    scores_b.index_add_(0, flat_idxs, weighted_preds.reshape(sw * k, num_classes))
                    weights_b.index_add_(0, flat_idxs, w.reshape(-1))  # type: ignore[union-attr]

                d_sq = distances.square()
                d_sq_max = d_sq.amax(dim=-1, keepdim=True).clamp_min(1e-12)
                delta = (1.0 - d_sq / d_sq_max).square()
                possibility.index_add_(0, flat_idxs, delta.reshape(-1))

                steps += 1
                covered = int((possibility >= overlap).sum().item())
                pbar.update(max(0, covered - prev_covered))
                pbar.set_postfix({"steps": steps, "min_p": f"{min_p:.3f}"})
                prev_covered = covered

        if scores_b is None or output is None:
            continue

        if aggregate == "ema":
            final = scores_b
        else:
            assert weights_b is not None
            weighted = weights_b > 0
            scores_b[weighted] = scores_b[weighted] / weights_b[weighted].unsqueeze(-1)
            final = scores_b

        output[idx_b] = final

    if output is None:
        return pos.new_zeros((0, 0))
    return output


class KNNWindowInferer(Inferer):
    """Iterative KNN-window inferer for large-scale point cloud segmentation.

    Maintains a per-point coverage score and iteratively crops KNN windows around
    the least-covered points until all points are covered. Reuse the same instance
    across scenes; compose with `TTAInferer` for multi-augmentation averaging.

    All parameters are forwarded verbatim to `knn_window_inference`.

    Example:
        ```python
        from torch_pointcloud.inferers import KNNWindowInferer

        # EMA aggregation: outputs calibrated probabilities directly.
        inferer = KNNWindowInferer(roi_num_points=65_536, overlap=0.5, aggregate="ema")
        probs = inferer(data, predictor=lambda d: model(d["pos"], d["pos"], d["batch"]))
        ```
    """

    def __init__(
        self,
        roi_num_points: int = 65_536,
        sw_batch_size: int = 1,
        overlap: float = 0.5,
        mode: WindowMode = "constant",
        sigma_scale: float = 0.125,
        aggregate: AggregateMode = "weighted_mean",
        ema_smoothing: float = 0.95,
        softmax: bool = False,
        transform: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
        pos_key: str = DataKeys.POS,
        batch_key: str = DataKeys.BATCH,
        progress: bool = False,
        seed: Optional[int] = None,
    ) -> None:
        self.roi_num_points = roi_num_points
        self.sw_batch_size = sw_batch_size
        self.overlap = overlap
        self.mode = mode
        self.sigma_scale = sigma_scale
        self.aggregate = aggregate
        self.ema_smoothing = ema_smoothing
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
        return knn_window_inference(
            data,
            predictor=predictor,
            roi_num_points=self.roi_num_points,
            sw_batch_size=self.sw_batch_size,
            overlap=self.overlap,
            mode=self.mode,
            sigma_scale=self.sigma_scale,
            aggregate=self.aggregate,
            ema_smoothing=self.ema_smoothing,
            softmax=self.softmax,
            transform=self.transform,
            pos_key=self.pos_key,
            batch_key=self.batch_key,
            progress=self.progress,
            seed=self.seed,
        )
