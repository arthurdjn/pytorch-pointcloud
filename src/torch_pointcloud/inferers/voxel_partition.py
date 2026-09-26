r"""K-pass voxel-partition inferer with per-point scatter-back aggregation."""

import math
from typing import Any, Callable, Dict, List, Literal, Optional

import torch
from torch import Tensor
from tqdm import tqdm

from torch_pointcloud.ops.voxelization import voxel_grid_fnv
from torch_pointcloud.utils.data import DataKeys, collate

from ._utils import apply_transform, check_batch_alignment, index_select_dict
from .inferer import Inferer


class VoxelPartitionInferer(Inferer):
    r"""$K$-pass voxel-partition inferer with scatter-back aggregation.

    Partitions each batch element's points into FNV voxel buckets at `voxel_size` and runs the
    predictor on $K = \max_v c_v$ sub-clouds per element, where sub-cloud $i$ picks the
    $(i \bmod c_v)$-th point of every bucket. Per-sub-cloud logits are scatter-summed to
    original-point indices and divided by per-point participation counts; each point is picked
    $\lfloor K / c_v \rfloor$ or $\lfloor K / c_v \rfloor + 1$ times across the $K$ passes, so
    every original point gets at least one prediction.

    For test-time augmentation, wrap in `TTAInferer`: each TTA pass triggers a fresh $K$-pass
    voxel partition under that augmentation.

    Predictions are scatter-summed in float64 for stable averaging across passes; the returned
    tensor is cast back to the predictor's output dtype. An empty scene ($N = 0$) returns a
    $(0, 0)$ tensor: the predictor is never called, so the channel count cannot be inferred.

    Args:
        voxel_size: Side length of the FNV voxel partition (in the units of `pos`).
        transform: Optional per-sub-cloud callable applied after slicing each sub-cloud out of
            `data`. Typical use: the per-sub-cloud part of the model's preprocessing (centering,
            grid coordinates, feature stacking). If it changes the row count (pad, voxelize, ...)
            it must record a source-to-predictor index map under `inverse_key` so the inferer can
            gather predictions back to the sub-cloud's points.
        sw_batch_size: Number of sub-clouds packed into one predictor call via `collate`.
            `>1` amortises FPS / radius costs on the GPU.
        softmax: If `True`, softmax each predictor output before scatter-summing.
        aggregate: `"mean"` divides each point's accumulated predictions by the number of sub-clouds it
            appeared in; `"sum"` returns the plain sum. The argmax is the same within one call, but under
            `TTAInferer` the counts differ between views (each view is partitioned on its own augmented
            positions), so `"sum"` reproduces the reference protocols that add un-normalized probabilities
            over views and fragments.
        pos_key: Dict key for the position tensor.
        batch_key: Dict key for the per-point batch index.
        inverse_key: Dict key under which a row-altering `transform` records a source-to-predictor long
            index map of shape $(N_\text{sub},)$ with values in $[0, N_\text{predictor})$. Any scene-level
            value at this key is dropped before `transform` runs, and the map is popped before the
            predictor is called. Leave `None` when the transform preserves row count.
        seed: RNG seed for the per-pass index shuffle. `None` draws from the global generator, so
            `torch.manual_seed` seeds the inferer together with the transforms. An int is offset by the
            number of calls the instance has made: repeated calls (e.g. `TTAInferer` views) draw
            different shuffles, and a fresh instance replays the same sequence.
        progress: If `True`, show a `tqdm` progress bar per batch element.

    Example:
        ```python
        from torch_pointcloud.inferers import VoxelPartitionInferer

        inferer = VoxelPartitionInferer(voxel_size=0.04, sw_batch_size=4, transform=model.transforms)
        logits = inferer(room, predictor=lambda d: model(d["x"], d["pos"], d["batch"]))
        ```
    """

    def __init__(
        self,
        voxel_size: float,
        transform: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
        sw_batch_size: int = 1,
        softmax: bool = False,
        aggregate: Literal["mean", "sum"] = "mean",
        pos_key: str = DataKeys.POS,
        batch_key: str = DataKeys.BATCH,
        inverse_key: Optional[str] = None,
        seed: Optional[int] = None,
        progress: bool = False,
    ) -> None:
        if voxel_size <= 0.0:
            raise ValueError(f"`voxel_size` must be > 0, got {voxel_size}.")
        if sw_batch_size < 1:
            raise ValueError(f"`sw_batch_size` must be >= 1, got {sw_batch_size}.")
        if aggregate not in ("mean", "sum"):
            raise ValueError(f"`aggregate` must be 'mean' or 'sum', got {aggregate!r}.")

        self.voxel_size = voxel_size
        self.transform = transform
        self.sw_batch_size = sw_batch_size
        self.softmax = softmax
        self.aggregate = aggregate
        self.pos_key = pos_key
        self.batch_key = batch_key
        self.inverse_key = inverse_key
        self.seed = seed
        self.progress = progress
        self._num_calls = 0

    @torch.no_grad()
    def forward(
        self,
        data: Dict[str, Any],
        predictor: Callable[[Dict[str, Any]], Tensor],
    ) -> Tensor:
        if self.pos_key not in data:
            raise KeyError(f"`data` is missing the required key {self.pos_key!r}.")
        if self.batch_key not in data:
            raise KeyError(f"`data` is missing the required key {self.batch_key!r}.")

        pos = data[self.pos_key]
        batch = data[self.batch_key]
        check_batch_alignment(pos, batch, self.pos_key, self.batch_key)
        n = pos.size(0)

        seed = None if self.seed is None else self.seed + self._num_calls
        self._num_calls += 1
        rng = None if seed is None else torch.Generator().manual_seed(int(seed))

        logits_sum: Optional[Tensor] = None
        counts: Optional[Tensor] = None
        out_dtype: Optional[torch.dtype] = None

        for b in torch.unique(batch).tolist():
            idx_b = torch.where(batch == b)[0]
            n_b = int(idx_b.numel())
            data_b = index_select_dict(data, idx_b, n)

            _, inverse, count = voxel_grid_fnv(pos[idx_b], self.voxel_size, return_inverse=True, return_counts=True)
            idx_sort = torch.argsort(inverse, stable=True)
            starts = torch.cumsum(count, dim=0) - count
            k = int(count.max())
            v = int(count.numel())

            sub_indices: List[Tensor] = [
                idx_sort[starts + (i % count)][torch.randperm(v, generator=rng)] for i in range(k)
            ]

            for start in tqdm(
                range(0, k, self.sw_batch_size),
                total=math.ceil(k / self.sw_batch_size),
                desc=f"batch {int(b)}",
                leave=False,
                disable=not self.progress,
            ):
                chunk = sub_indices[start : start + self.sw_batch_size]
                samples: List[Dict[str, Any]] = []
                inverse_maps: List[Optional[Tensor]] = []
                for idx in chunk:
                    sample, inverse_map = apply_transform(
                        index_select_dict(data_b, idx, n_b),
                        self.transform,
                        pos_key=self.pos_key,
                        inverse_key=self.inverse_key,
                    )
                    samples.append(sample)
                    inverse_maps.append(inverse_map)

                packed = collate(samples, batch_from=self.pos_key, batch_key=self.batch_key)
                packed_orig = idx_b[torch.cat(chunk)]

                logits = predictor(packed)
                if self.softmax:
                    logits = torch.softmax(logits, dim=-1)

                sizes = [int(sample[self.pos_key].size(0)) for sample in samples]
                logits = torch.cat(
                    [
                        part if inverse_map is None else part[inverse_map.to(part.device)]
                        for part, inverse_map in zip(logits.split(sizes), inverse_maps)
                    ]
                )

                if logits_sum is None:
                    out_dtype = logits.dtype
                    logits_sum = torch.zeros(n, int(logits.size(-1)), dtype=torch.float64, device=logits.device)
                    counts = torch.zeros(n, dtype=torch.long, device=logits.device)

                assert counts is not None
                packed_orig = packed_orig.to(logits.device)
                logits_sum.index_add_(0, packed_orig, logits.double())
                counts.index_add_(0, packed_orig, torch.ones_like(packed_orig))

        if logits_sum is None or counts is None or out_dtype is None:
            return pos.new_zeros((0, 0))
        if self.aggregate == "sum":
            return logits_sum.to(out_dtype)
        return (logits_sum / counts.clamp_min(1).unsqueeze(-1)).to(out_dtype)
