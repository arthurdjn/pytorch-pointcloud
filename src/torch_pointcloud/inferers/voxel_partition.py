r"""K-pass voxel-partition inferer with per-point scatter-back aggregation."""

from typing import Any, Callable, Dict, List, Optional

import torch
from torch import Tensor

from torch_pointcloud.utils.data import DataKeys, collate
from torch_pointcloud.utils.ops import voxel_grid_fnv

from ._utils import index_select_dict
from .inferer import Inferer


class VoxelPartitionInferer(Inferer):
    r"""$K$-pass voxel-partition inferer with scatter-back aggregation.

    Partitions input points into FNV voxel buckets at `voxel_size` and runs the predictor on
    $K = \max_v c_v$ sub-clouds, where sub-cloud $i$ picks the $(i \bmod c_v)$-th point of every
    bucket. Per-sub-cloud logits are scatter-summed to original-point indices and divided by
    per-point participation counts; each point is picked $\lfloor K / c_v \rfloor$ or
    $\lfloor K / c_v \rfloor + 1$ times across the $K$ passes, so every original point gets at
    least one prediction.

    For test-time augmentation, wrap in `TTAInferer`: each TTA pass triggers a fresh $K$-pass
    voxel partition under that augmentation.

    Args:
        voxel_size: Side length of the FNV voxel partition (in the units of `pos`).
        transform: Optional per-sub-cloud callable applied after slicing each sub-cloud out of
            `data`. Typical use: the model's registered preprocessing transform.
        sub_batch_size: Number of sub-clouds packed into one predictor call via `collate`.
            `>1` amortises FPS / radius costs on the GPU.
        softmax: If `True`, softmax each predictor output before scatter-summing.
        pos_key: Dict key for the position tensor.
        seed: Optional RNG seed for the per-pass index shuffle.

    Example:
        ```python
        from torch_pointcloud.inferers import VoxelPartitionInferer

        inferer = VoxelPartitionInferer(voxel_size=0.04, sub_batch_size=4, transform=model.transforms)
        logits = inferer(room, predictor=lambda d: model(d["x"], d["pos"], d["batch"]))
        ```
    """

    def __init__(
        self,
        voxel_size: float,
        transform: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
        sub_batch_size: int = 1,
        softmax: bool = False,
        pos_key: str = DataKeys.POS,
        seed: Optional[int] = None,
    ) -> None:
        if voxel_size <= 0.0:
            raise ValueError(f"`voxel_size` must be > 0, got {voxel_size}.")
        if sub_batch_size < 1:
            raise ValueError(f"`sub_batch_size` must be >= 1, got {sub_batch_size}.")
        self.voxel_size = voxel_size
        self.transform = transform
        self.sub_batch_size = sub_batch_size
        self.softmax = softmax
        self.pos_key = pos_key
        self.seed = seed

    @torch.no_grad()
    def forward(
        self,
        data: Dict[str, Any],
        predictor: Callable[[Dict[str, Any]], Tensor],
    ) -> Tensor:
        if self.pos_key not in data:
            raise KeyError(f"`data` is missing the required key {self.pos_key!r}.")

        pos = data[self.pos_key]
        n = pos.size(0)

        _, inverse, count = voxel_grid_fnv(pos, self.voxel_size, return_inverse=True, return_counts=True)
        idx_sort = torch.argsort(inverse, stable=True)
        starts = torch.cumsum(count, dim=0) - count
        k = int(count.max())
        v = int(count.numel())

        rng = torch.Generator() if self.seed is None else torch.Generator().manual_seed(int(self.seed))
        sub_indices: List[Tensor] = [idx_sort[starts + (i % count)][torch.randperm(v, generator=rng)] for i in range(k)]

        logits_sum: Optional[Tensor] = None
        counts: Optional[Tensor] = None

        for start in range(0, k, self.sub_batch_size):
            chunk = sub_indices[start : start + self.sub_batch_size]
            samples = [index_select_dict(data, idx, n) for idx in chunk]
            if self.transform is not None:
                samples = [self.transform(s) for s in samples]
            packed = collate(samples)
            packed_orig = torch.cat(chunk)

            logits = predictor(packed)
            if self.softmax:
                logits = torch.softmax(logits, dim=-1)
            if logits_sum is None:
                logits_sum = torch.zeros(n, int(logits.size(-1)), dtype=torch.float64, device=logits.device)
                counts = torch.zeros(n, dtype=torch.long, device=logits.device)
            assert counts is not None
            packed_orig = packed_orig.to(logits.device)
            logits_sum.index_add_(0, packed_orig, logits.double())
            counts.index_add_(0, packed_orig, torch.ones_like(packed_orig))

        if logits_sum is None or counts is None:
            return pos.new_zeros((0, 0))
        return logits_sum / counts.clamp_min(1).unsqueeze(-1)
