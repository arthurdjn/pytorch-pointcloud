"""Transforms that voxelize the points or pad them to a voxel grid."""

from typing import TYPE_CHECKING, Any, Dict, Literal, Optional, Sequence, get_args

import torch
from torch_geometric.nn.pool import voxel_grid
from torch_geometric.nn.pool.consecutive import consecutive_cluster

from torch_pointcloud.utils.conversion import ensure_tuple, ensure_tuple_size
from torch_pointcloud.utils.data import DataKeys
from torch_pointcloud.utils.imports import _TORCH_SCATTER_GITHUB_URL, optional_import
from torch_pointcloud.utils.ops import first_permutation, voxel_grid_fnv
from torch_pointcloud.utils.random import Randomizable
from torch_pointcloud.utils.types import KeyCollection, ValueCollection
from torch_pointcloud.utils.voxelization import hard_voxelize

from . import functional as F
from .base import DictTransform

VoxelMethod = Literal["fnv", "pyg"]
"""Allowed values for `Voxelize.method` (voxel-id hashing scheme)."""

VoxelReduce = Literal["mean", "min", "max", "sum", "first"]
"""Allowed values for `Voxelize.reduce` (per-key per-voxel reduction)."""

VoxelPosReduce = Literal["mean", "min", "max", "sum", "first", "grid"]
"""Allowed values for `Voxelize.pos_reduce` (per-voxel reduction for positions; `"grid"` keeps integer voxel coords)."""

if TYPE_CHECKING:
    from torch_scatter import scatter

scatter, _ = optional_import("torch_scatter", name="scatter", url=_TORCH_SCATTER_GITHUB_URL)

__all__ = [
    "DivisiblePad",
    "HardVoxelize",
    "VoxelMethod",
    "VoxelPosReduce",
    "VoxelReduce",
    "Voxelize",
]


class DivisiblePad(DictTransform, Randomizable):
    r"""Pad per-point tensors so each batch is divisible by `num_samples`.

    Thin dict wrapper around `divisible_pad`; see its docstring for the full
    behavior of each `pad_fill` strategy (`"cycle"`, `"replicate"`, `"random"`).
    The tensor at `ref_key` defines the packed count $n$ and the device. If a
    batch index tensor lives at `batch_key`, padding is done per-batch; otherwise
    a single zero batch is synthesized. Every tensor in the dict whose first dim
    equals $n$ (positions, features, labels, ...) is re-indexed by the same
    gather map, so per-point correspondence is preserved.

    When `dst_inverse_key` is set, the transform also records a source-to-padded
    index map under that dict key: a 1-D long tensor of length $n$ with values
    in $[0, n_\text{padded})$ giving the canonical padded row for each source
    row. If the key already holds a prior inverse map (from an earlier
    invertible transform), the new map composes with it via gather, so the
    stored tensor always maps from the outermost source space to the current
    predictor space. Consumers such as `SlidingWindowInferer` read this key
    once and gather predictions back to the source rows.

    === "Object"

        ![DivisiblePad on an object](../../assets/transforms/divisible_pad.png)

    === "Scene"

        ![DivisiblePad on a room](../../assets/transforms/divisible_pad_scene.png)

    Args:
        num_samples: Target chunk size $k$ for divisibility.
        pad_fill: Fill strategy passed through to `divisible_pad`.
        ref_key: Key whose tensor defines $n$ and the device.
        batch_key: Key for an optional batch index tensor. When present in the
            data, padding runs per-batch; otherwise a single zero batch is
            synthesized for the whole scene.
        seed: Seed for the random padding (only used when `pad_fill="random"`);
            `None` draws from the global generator (see `Randomizable`).
        dst_inverse_key: Key for the source-to-padded row map (see the module docs on sampling keys); composes
            with any prior value at the same key. `None` (the default) disables it.
        allow_missing_keys: If `True`, return the data unchanged when `ref_key`
            is missing instead of raising.

    Example:
        ```python
        from torch_pointcloud.transforms import DivisiblePad

        # Pad a 5000-point block to 8192 (= 2 * 4096) before sliding-window
        # sub-chunking. Random fill duplicates points uniformly at random.
        transform = DivisiblePad(num_samples=4096, pad_fill="random")
        ```
    """

    def __init__(
        self,
        num_samples: int,
        pad_fill: "F.PadFill" = "cycle",
        ref_key: str = DataKeys.POS,
        batch_key: str = DataKeys.BATCH,
        seed: Optional[int] = None,
        dst_inverse_key: Optional[str] = None,
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys=ref_key, allow_missing_keys=allow_missing_keys)
        self.num_samples = num_samples
        self.pad_fill = pad_fill
        self.ref_key = ref_key
        self.batch_key = batch_key
        self.set_random_state(seed)
        self.dst_inverse_key = dst_inverse_key

    def transform(self, data: Dict[str, Any]) -> Dict[str, Any]:
        d = dict(data)
        if self.ref_key not in d:
            if self.allow_missing_keys:
                return d
            raise KeyError(f"`DivisiblePad` requires {self.ref_key!r} in data.")
        ref = d[self.ref_key]
        if not torch.is_tensor(ref):
            raise TypeError(f"Expected tensor at {self.ref_key!r}, got {type(ref).__name__}.")
        n = int(ref.size(0))
        if n == 0:
            return d
        if self.batch_key in d and torch.is_tensor(d[self.batch_key]):
            batch = d[self.batch_key]
        else:
            batch = torch.zeros(n, dtype=torch.long, device=ref.device)
        indices, inverse_indices, padded_batch = F.divisible_pad(
            batch,
            k=self.num_samples,
            mode="all",
            pad_fill=self.pad_fill,
            return_inverse=True,
            generator=self.R,
        )
        prior = d.get(self.dst_inverse_key) if self.dst_inverse_key is not None else None
        for key, value in d.items():
            if key == self.dst_inverse_key:
                continue
            if torch.is_tensor(value) and value.ndim > 0 and value.size(0) == n:
                d[key] = value[indices]
        d[self.batch_key] = padded_batch
        if self.dst_inverse_key is not None:
            d[self.dst_inverse_key] = inverse_indices if prior is None else inverse_indices[prior]
        return d


class HardVoxelize(DictTransform):
    r"""Hard-voxelize a single scene into the per-voxel point stack consumed by voxel detectors.

    Moves the `transform_points_to_voxels` step of voxel detectors (PointPillars, SECOND) out of the
    model and into the data pipeline, mirroring how `BuildOctree` produces an octree for OctFormer.
    The model then receives already-voxelized input and focuses on the network math.

    Reads `pos_key` (and optionally `feat_key`), runs
    `hard_voxelize` on the single sample (the
    batch index is all zeros), and adds three keys while keeping `pos` / `x`:

    - `voxel_key`: the per-voxel point stack.
    - `pos_voxel_key`: integer voxel grid indices $(z, y, x)$ (the single-sample batch column is dropped;
      the per-voxel scene index is synthesized at collation).
    - `num_points_key`: the per-voxel point counts.

    === "Object"

        ![HardVoxelize on an object](../../assets/transforms/hard_voxelize.png)

    === "Scene"

        ![HardVoxelize on a room](../../assets/transforms/hard_voxelize_scene.png)

    Args:
        pos_key: Key holding the point positions $(N, 3)$.
        voxel_size: Voxel size $(v_x, v_y, v_z)$.
        point_cloud_range: Range $(x_\min, y_\min, z_\min, x_\max, y_\max, z_\max)$.
        max_num_points: Maximum number of points kept per voxel.
        max_num_voxels: Maximum number of voxels kept per scene.
        feat_key: Optional key holding extra point features $(N, C)$ concatenated after $xyz$.
        voxel_key: Output key for the per-voxel point stack.
        pos_voxel_key: Output key for the integer voxel grid indices.
        num_points_key: Output key for the per-voxel point counts.
        allow_missing_keys: Unused (`pos_key` is always required); kept for interface parity.

    Shape:
        - `voxel_key`: $(V, \text{max\_num\_points}, 3 + C)$.
        - `pos_voxel_key`: $(V, 3)$ with columns $(z, y, x)$.
        - `num_points_key`: $(V,)$.

    Example:
        ```python
        import torch
        import torch_pointcloud.transforms as T

        data = {"pos": torch.rand(1000, 3) * 50.0, "x": torch.rand(1000, 1)}
        transform = T.HardVoxelize(
            pos_key="pos",
            feat_key="x",
            voxel_size=(0.16, 0.16, 4.0),
            point_cloud_range=(0.0, -39.68, -3.0, 69.12, 39.68, 1.0),
            max_num_points=32,
            max_num_voxels=40000,
        )
        data = transform(data)
        print(data["voxel"].shape, data["pos_voxel"].shape, data["voxel_num_points"].shape)
        ```
    """

    def __init__(
        self,
        pos_key: str,
        voxel_size: Sequence[float],
        point_cloud_range: Sequence[float],
        max_num_points: int,
        max_num_voxels: int,
        feat_key: Optional[str] = None,
        voxel_key: str = DataKeys.VOXEL,
        pos_voxel_key: str = DataKeys.POS_VOXEL,
        num_points_key: str = DataKeys.VOXEL_NUM_POINTS,
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys=pos_key, allow_missing_keys=allow_missing_keys)
        self.pos_key = pos_key
        self.voxel_size = voxel_size
        self.point_cloud_range = point_cloud_range
        self.max_num_points = max_num_points
        self.max_num_voxels = max_num_voxels
        self.feat_key = feat_key
        self.voxel_key = voxel_key
        self.pos_voxel_key = pos_voxel_key
        self.num_points_key = num_points_key

    def transform(self, data: Dict[str, Any]) -> Dict[str, Any]:
        d = dict(data)
        pos = d[self.pos_key]
        feat = d.get(self.feat_key) if self.feat_key is not None else None
        points = pos if feat is None else torch.cat([pos, feat], dim=1)
        batch = pos.new_zeros(points.shape[0], dtype=torch.long)
        voxels, voxel_indices, num_points = hard_voxelize(
            points,
            batch,
            self.voxel_size,
            self.point_cloud_range,
            self.max_num_points,
            self.max_num_voxels,
        )
        d[self.voxel_key] = voxels
        d[self.pos_voxel_key] = voxel_indices[:, 1:]
        d[self.num_points_key] = num_points
        return d


class Voxelize(DictTransform, Randomizable):
    r"""Voxelize a point cloud by grid-binning and per-voxel reduction.

    Sub-samples a point cloud to one representative point per occupied voxel,
    and optionally records a source-to-voxel index map for full-resolution
    back-projection.

    Operates on a single sample (pre-collate). With `dst_inverse_key` set, the stored
    tensor has shape $(N_\text{full},)$ with values in $[0, N_\text{voxel})$:
    for each original point $i$, the voxel it belongs to. Downstream code can
    recover full-resolution predictions with `preds_full = preds_voxel[inverse]`.

    If the key already holds a prior inverse map (e.g. from an earlier
    invertible transform), the new map composes with it via gather, so the
    stored tensor always maps from the outermost source space to the current
    predictor space.

    === "Object"

        ![Voxelize on an object](../../assets/transforms/voxelize.png)

    === "Scene"

        ![Voxelize on a room](../../assets/transforms/voxelize_scene.png)

    Args:
        pos_key: Key holding the positions to sub-sample.
        pos_reduce: How to reduce positions per voxel (`mean`/`min`/`max`/`sum`/`first`/`grid`).
        size: Voxel edge length in the same units as the positions. Must be positive.
        method: Voxel-id hashing scheme (`fnv` matches FNV-1a-based reference pipelines; `pyg` is the default).
        reduce: Per-key reduction for `keys`. `None` (the default) resolves per key to `mean` for
            floating-point tensors and `first` for integer tensors (e.g. `segment`). Integer keys keep
            their dtype: non-`first` reductions compute in float and cast back. The `first`
            representative is the first point of each voxel in input order, deterministic across
            devices (unless `random_sample=True`).
        keys: Additional per-point keys to sub-sample (e.g. `color`, `segment`).
        dst_inverse_key: Key for the source-to-voxel row map (see the module docs on sampling keys); composes
            with any prior value at the same key. `None` (the default) disables it.
        dst_pos_grid_key: When set, also store the integer voxel-grid coordinates under this key. Useful when a
            model needs both real-valued positions (e.g. for rotary position embedding) and integer grid
            coordinates (for serialization / sparse-conv stems); with `pos_reduce="grid"` it holds the same grid
            as `pos_key`.
        random_sample: If `True`, the per-voxel representative used by `reduce="first"`
            (and the `pos`/`grid_pos` derivations) is chosen *randomly* within each
            voxel on every call. Per-voxel random sampling is a meaningful
            training augmentation; leave
            `False` (default) for deterministic validation.
        seed: Seed for the random voxel representative (`random_sample`);
            `None` draws from the global generator (see `Randomizable`).
        allow_missing_keys: If `True`, return the data unchanged when `pos_key` is missing and skip absent `keys`.

    Raises:
        ValueError: If `size` is not positive, or `pos_reduce` / `method` / any `reduce` entry is
            not one of its allowed values.
    """

    def __init__(
        self,
        pos_key: str,
        pos_reduce: VoxelPosReduce,
        size: float,
        method: VoxelMethod = "pyg",
        reduce: Optional[ValueCollection[VoxelReduce]] = None,
        keys: Optional[KeyCollection] = None,
        dst_inverse_key: Optional[str] = None,
        dst_pos_grid_key: Optional[str] = None,
        random_sample: bool = False,
        seed: Optional[int] = None,
        allow_missing_keys: bool = False,
    ) -> None:
        if size <= 0:
            raise ValueError(f"size must be positive; got {size}.")
        if pos_reduce not in get_args(VoxelPosReduce):
            raise ValueError(f"Invalid pos_reduce: {pos_reduce!r}. Expected one of {get_args(VoxelPosReduce)}.")
        if method not in get_args(VoxelMethod):
            raise ValueError(f"Invalid method: {method!r}. Expected one of {get_args(VoxelMethod)}.")
        invalid = set(ensure_tuple(reduce)) - set(get_args(VoxelReduce)) - {None}
        if invalid:
            raise ValueError(f"Invalid reduce(s): {invalid}. Expected one of {get_args(VoxelReduce)}.")

        super().__init__(keys, allow_missing_keys)

        self.pos_key = pos_key
        self.pos_reduce = pos_reduce
        self.size = size
        self.reduce = ensure_tuple_size(reduce, len(self.keys))
        self.method = method
        self.dst_inverse_key = dst_inverse_key
        self.dst_pos_grid_key = dst_pos_grid_key
        self.random_sample = random_sample
        self.set_random_state(seed)

    def _random_perm(self, cluster: torch.Tensor, num_clusters: int) -> torch.Tensor:
        """Pick one random representative-index per cluster (replaces the deterministic perm)."""
        sort_idx = torch.argsort(cluster, stable=True)
        counts = torch.bincount(cluster, minlength=num_clusters)
        idx_ptr = torch.cat([counts.new_zeros(1), counts.cumsum(0)[:-1]])
        rand = torch.rand(num_clusters, device=cluster.device, generator=self.R)
        offsets = torch.minimum((rand * counts.float()).long(), counts - 1)
        return sort_idx[idx_ptr + offsets]

    def _reduce(
        self,
        tensor: torch.Tensor,
        reduce: str,
        cluster: torch.Tensor,
        perm: torch.Tensor,
    ) -> torch.Tensor:
        if reduce == "first":
            return tensor[perm]

        # Integer min/max/sum scatter natively; the float32 round trip below corrupts values above 2^24.
        if not tensor.is_floating_point() and tensor.dtype != torch.bool and reduce != "mean":
            return scatter(tensor, cluster, dim=0, reduce=reduce)

        # Mean (and any bool reduction) needs float input; cast back so the key keeps its dtype.
        out = scatter(tensor.float(), cluster, dim=0, reduce=reduce)
        return out if tensor.is_floating_point() else out.to(tensor.dtype)

    def transform(self, data: Dict[str, Any]) -> Dict[str, Any]:
        data = dict(data)
        if self.pos_key not in data:
            if self.allow_missing_keys:
                return data
            raise KeyError(f"`Voxelize` requires {self.pos_key!r} in data.")
        pos = data[self.pos_key]

        if pos.shape[0] == 0:
            if self.dst_inverse_key is not None:
                data[self.dst_inverse_key] = torch.empty(0, dtype=torch.long, device=pos.device)
            if self.dst_pos_grid_key is not None:
                data[self.dst_pos_grid_key] = torch.empty(0, pos.shape[-1], dtype=torch.long, device=pos.device)
            return data

        start = torch.floor(pos.min(dim=0).values / self.size) * self.size

        if self.method == "fnv":
            # This method is supported only for debugging and reproducibility against FNV-hash-based
            # grid subsampling. This method might be removed in the future (?)
            cluster = voxel_grid_fnv(pos, size=self.size, start=start)
        else:
            cluster = voxel_grid(pos, size=self.size, start=start)

        cluster, _ = consecutive_cluster(cluster)
        num_clusters = int(cluster.max().item()) + 1

        if self.random_sample:
            perm = self._random_perm(cluster, num_clusters=num_clusters)
        else:
            perm = first_permutation(cluster, num_clusters=num_clusters)

        pos_grid = torch.floor((pos[perm] - start) / self.size).long()
        pos_grid = pos_grid - pos_grid.min(dim=0).values
        data[self.pos_key] = (
            pos_grid if self.pos_reduce == "grid" else self._reduce(pos, self.pos_reduce, cluster, perm)
        )
        if self.dst_pos_grid_key is not None:
            data[self.dst_pos_grid_key] = pos_grid

        for key, reduce in self.iter_keys(data, self.reduce):
            tensor = data[key]
            if reduce is None:
                reduce = "mean" if tensor.is_floating_point() else "first"
            data[key] = self._reduce(tensor, reduce, cluster, perm)

        if self.dst_inverse_key is not None:
            prior = data.get(self.dst_inverse_key)
            data[self.dst_inverse_key] = cluster if prior is None else cluster[prior]

        return data
