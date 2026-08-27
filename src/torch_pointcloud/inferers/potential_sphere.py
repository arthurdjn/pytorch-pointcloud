r"""Potential-driven sphere voting for large-scale point cloud segmentation.

The scene is covered by radius-defined spheres centered where the cloud has been seen the least, tracked by a
coarse grid of potentials; each sphere's softmax predictions are blended into the running per-point scores by
an exponential moving average until every region has been covered about `num_votes` times.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
from torch import Tensor
from torch_geometric.utils import scatter
from tqdm import tqdm

from torch_pointcloud.utils.data import DataKeys, collate
from torch_pointcloud.utils.ops import voxel_grid_fnv

from ._utils import check_batch_alignment, index_select_dict
from .inferer import Inferer


def _next_sphere(
    pos_b: Tensor,
    coarse_pos: Tensor,
    potentials: Tensor,
    rng: torch.Generator,
    radius: float,
    jitter: float,
) -> Tuple[Tensor, Tensor]:
    """Draw the next center, raise the potentials it covers and return its point indices and the center."""
    radius_sq = radius**2
    center = coarse_pos[int(potentials.argmin().item())]
    if jitter > 0.0:
        noise = torch.randn(center.shape, generator=rng, device=center.device, dtype=center.dtype) * jitter
        center = center + noise.clamp(-radius / 2.0, radius / 2.0)

    coarse_d_sq = (coarse_pos - center).square().sum(dim=-1)
    covered = coarse_d_sq < radius_sq
    potentials[covered] += (1.0 - coarse_d_sq[covered] / radius_sq).square()
    return torch.where((pos_b - center).square().sum(dim=-1) < radius_sq)[0], center


@torch.no_grad()
def potential_sphere_inference(
    data: Dict[str, Any],
    *,
    predictor: Callable[[Dict[str, Any]], Tensor],
    radius: float,
    num_votes: float = 10.0,
    potential_size: Optional[float] = None,
    jitter: Optional[float] = None,
    inner_ratio: float = 0.7,
    ema_smoothing: float = 0.95,
    sw_batch_size: int = 1,
    transform: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
    pos_key: str = DataKeys.POS,
    batch_key: str = DataKeys.BATCH,
    progress: bool = False,
    seed: Optional[int] = None,
) -> Tensor:
    r"""Potential-driven sphere voting for large-scale point cloud segmentation.

    The scene is covered by spheres of radius $r$ whose centers are chosen where the cloud has been seen the
    least: a coarse grid of *potentials* (one scalar per `potential_size` cell, initialized with a small random
    value) tracks coverage, each sphere is centered on the cell with the lowest potential (plus a Gaussian
    jitter) and raises the potentials of the cells it covers by the Tukey window $(1 - d^2 / r^2)^2$. The
    predictor runs on every sphere and its softmax probabilities are blended into the running per-point
    scores by an exponential moving average, restricted to the points within `inner_ratio` $\cdot r$ of the
    center where the sphere's context is complete. The loop stops once every cell's potential reaches
    `num_votes`, so each region has been predicted about that many times.

    This is the test protocol of :arxiv: [KPConv](https://arxiv.org/abs/1904.08889) (radius-defined input
    spheres, `test_smooth` EMA, potential sampling), and it composes with any model that consumes a packed
    sphere: the per-sphere `transform` sees the centered sphere dict and can add the reference's stochastic
    test-time augmentation and the model's feature stack. Points that no sphere reaches keep all-zero
    scores.

    Args:
        data: Dict of per-point tensors. Must contain `pos` (shape $(N, D)$) and `batch` (shape $(N,)$);
            extra per-point tensors are sliced to the active sphere automatically.
        predictor: Callable mapping a packed sphere dict to per-point logits of shape $(M, C)$.
        radius: Sphere radius, in the units of `pos`.
        num_votes: Potential threshold ending the loop, i.e. the number of times every region is covered
            (KPConv reports its numbers at the first multiple of $10$).
        potential_size: Cell size of the coarse potential grid. Defaults to `radius / 10`.
        jitter: Standard deviation of the Gaussian jitter added to each sphere center, clipped at
            `radius / 2`. Defaults to `radius / 10`; `0` disables it.
        inner_ratio: Fraction of `radius` inside which the sphere's predictions are kept.
        ema_smoothing: EMA factor $\alpha \in [0, 1)$ of the score update
            $\text{new} = \alpha \cdot \text{old} + (1 - \alpha) \cdot \text{softmax}(\text{logits})$.
        sw_batch_size: Number of spheres packed into one predictor call. Centers are still drawn one at a
            time with the potentials updated in between, as the reference sampler does.
        transform: Optional per-sphere callable applied to the centered sphere dict before the predictor.
            The transform must preserve the sphere's row count and keep positions centered on the sphere (the
            `inner_ratio` mask is evaluated on the transformed positions, as the reference does).
        pos_key: Dict key for the position tensor.
        batch_key: Dict key for the per-point batch index.
        progress: If `True`, show a `tqdm` progress bar per batch element.
        seed: Optional RNG seed for the initial potentials and the center jitter.

    Returns:
        Per-point score tensor of shape $(N, C)$: the EMA of softmax probabilities over the spheres covering
        each point; points no sphere reaches keep all-zero scores. An empty scene ($N = 0$) returns a
        $(0, 0)$ tensor: the predictor is never called, so the channel count cannot be inferred.
    """
    if pos_key not in data:
        raise KeyError(f"`data` is missing the required key {pos_key!r}.")
    if batch_key not in data:
        raise KeyError(f"`data` is missing the required key {batch_key!r}.")
    if radius <= 0.0:
        raise ValueError(f"`radius` must be > 0, got {radius}.")
    if num_votes <= 0.0:
        raise ValueError(f"`num_votes` must be > 0, got {num_votes}.")
    if potential_size is not None and potential_size <= 0.0:
        raise ValueError(f"`potential_size` must be > 0, got {potential_size}.")
    if jitter is not None and jitter < 0.0:
        raise ValueError(f"`jitter` must be >= 0, got {jitter}.")
    if not 0.0 < inner_ratio <= 1.0:
        raise ValueError(f"`inner_ratio` must be in (0, 1], got {inner_ratio}.")
    if not 0.0 <= ema_smoothing < 1.0:
        raise ValueError(f"`ema_smoothing` must be in [0, 1), got {ema_smoothing}.")
    if sw_batch_size < 1:
        raise ValueError(f"`sw_batch_size` must be >= 1, got {sw_batch_size}.")

    potential_size = potential_size if potential_size is not None else radius / 10.0
    jitter = jitter if jitter is not None else radius / 10.0

    pos = data[pos_key]
    batch = data[batch_key]
    check_batch_alignment(pos, batch, pos_key, batch_key)
    device = pos.device
    n = pos.size(0)
    inner_sq = (inner_ratio * radius) ** 2

    rng = torch.Generator(device=device)
    if seed is not None:
        rng.manual_seed(int(seed))

    output: Optional[Tensor] = None
    for b in torch.unique(batch).tolist():
        idx_b = torch.where(batch == b)[0]
        n_b = int(idx_b.numel())
        pos_b = pos[idx_b]
        data_b = index_select_dict(data, idx_b, n)

        _, cell = voxel_grid_fnv(pos_b, potential_size, return_inverse=True)
        coarse_pos = scatter(pos_b, cell, dim=0, reduce="mean")
        potentials = torch.rand(coarse_pos.size(0), generator=rng, device=device, dtype=pos.dtype) * 1e-3

        scores_b: Optional[Tensor] = None
        with tqdm(desc=f"batch {int(b)}", leave=False, disable=not progress) as pbar:
            while float(potentials.min().item()) < num_votes:
                spheres: List[Dict[str, Any]] = []
                sphere_idx: List[Tensor] = []
                for _ in range(sw_batch_size):
                    if float(potentials.min().item()) >= num_votes:
                        break

                    idx, center = _next_sphere(pos_b, coarse_pos, potentials, rng, radius, jitter)
                    if idx.numel() < 2:
                        continue

                    sphere = index_select_dict(data_b, idx, n_b)
                    sphere[pos_key] = sphere[pos_key] - center
                    if transform is not None:
                        sphere = transform(sphere)
                        if int(sphere[pos_key].size(0)) != int(idx.numel()):
                            raise ValueError(
                                f"`transform` must preserve each sphere's row count; got {int(idx.numel())} -> "
                                f"{int(sphere[pos_key].size(0))} rows."
                            )

                    spheres.append(sphere)
                    sphere_idx.append(idx)

                if not spheres:
                    continue

                packed = collate(spheres, batch_from=pos_key, batch_key=batch_key)
                probs = torch.softmax(predictor(packed).to(device), dim=-1)
                if scores_b is None:
                    scores_b = torch.zeros(n_b, int(probs.size(-1)), device=device, dtype=probs.dtype)

                offset = 0
                for sphere, idx in zip(spheres, sphere_idx):
                    count = int(idx.numel())
                    sphere_probs = probs[offset : offset + count]
                    offset += count
                    inner = sphere[pos_key].square().sum(dim=-1) < inner_sq
                    rows = idx[inner]
                    scores_b[rows] = ema_smoothing * scores_b[rows] + (1.0 - ema_smoothing) * (
                        sphere_probs[inner].to(scores_b.dtype)
                    )

                pbar.update(len(spheres))
                pbar.set_postfix({"min_potential": f"{float(potentials.min().item()):.2f}"})

        if scores_b is None:
            if n_b > 0:
                raise ValueError(
                    f"No sphere with at least 2 points was drawn for batch element {int(b)} (radius={radius}), so "
                    "its scores would silently stay all-zero. Increase `radius` or check the scale of `pos`."
                )
            continue

        if output is None:
            output = torch.zeros(n, int(scores_b.size(1)), device=device, dtype=scores_b.dtype)

        output[idx_b] = scores_b

    if output is None:
        if n > 0:
            raise ValueError(
                f"No sphere with at least 2 points was drawn for any batch element (radius={radius}), so the "
                "class count is unknown. Increase `radius` or check the scale of `pos`."
            )
        return pos.new_zeros((0, 0))
    return output


class PotentialSphereInferer(Inferer):
    r"""Potential-driven sphere voting inferer for large-scale point cloud segmentation.

    Covers the scene with radius-defined spheres centered where a coarse potential grid is lowest and blends
    each sphere's softmax predictions into the running per-point scores by an exponential moving average,
    until every region has been covered about `num_votes` times.

    All parameters are forwarded verbatim to `potential_sphere_inference`.

    Example:
        ```{.python notest}
        from torch_pointcloud.inferers import PotentialSphereInferer

        inferer = PotentialSphereInferer(radius=1.5, num_votes=10.0)
        probs = inferer(room, predictor=lambda d: model(d["x"], d["pos"], d["batch"]))
        ```
    """

    def __init__(
        self,
        radius: float,
        num_votes: float = 10.0,
        potential_size: Optional[float] = None,
        jitter: Optional[float] = None,
        inner_ratio: float = 0.7,
        ema_smoothing: float = 0.95,
        sw_batch_size: int = 1,
        transform: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
        pos_key: str = DataKeys.POS,
        batch_key: str = DataKeys.BATCH,
        progress: bool = False,
        seed: Optional[int] = None,
    ) -> None:
        self.radius = radius
        self.num_votes = num_votes
        self.potential_size = potential_size
        self.jitter = jitter
        self.inner_ratio = inner_ratio
        self.ema_smoothing = ema_smoothing
        self.sw_batch_size = sw_batch_size
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
        return potential_sphere_inference(
            data,
            predictor=predictor,
            radius=self.radius,
            num_votes=self.num_votes,
            potential_size=self.potential_size,
            jitter=self.jitter,
            inner_ratio=self.inner_ratio,
            ema_smoothing=self.ema_smoothing,
            sw_batch_size=self.sw_batch_size,
            transform=self.transform,
            pos_key=self.pos_key,
            batch_key=self.batch_key,
            progress=self.progress,
            seed=self.seed,
        )
