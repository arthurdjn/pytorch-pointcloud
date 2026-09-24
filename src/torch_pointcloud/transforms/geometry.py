"""Transforms that move points or derive geometric quantities."""

import math
from typing import Any, Dict, Literal, Optional, Sequence, get_args

import torch
from torch import Tensor

from torch_pointcloud.utils.cluster import knn
from torch_pointcloud.utils.conversion import ensure_tuple_size
from torch_pointcloud.utils.types import KeyCollection, ValueCollection

from .base import DictTransform

ShiftMethod = Literal["bbox", "centroid", "min"]


__all__ = [
    "AlignAxis",
    "AxisMinOffset",
    "BBoxCenter",
    "EstimateNormals",
    "Quantize",
    "Shift",
    "ShiftMethod",
]


def estimate_normals(
    pos: Tensor,
    k: int = 16,
    batch: Optional[Tensor] = None,
    orient_to_centroid: bool = False,
) -> Tensor:
    r"""Estimate per-point unit surface normals by local PCA.

    For each point the normal is the eigenvector of the smallest eigenvalue of the covariance of its $k$
    nearest neighbors, i.e. the direction of least variance (the local tangent-plane normal).

    PCA gives no canonical orientation. By default the sign is the arbitrary-but-deterministic sign returned by
    `torch.linalg.eigh`. With `orient_to_centroid`, each normal is flipped to point towards its cloud's
    centroid, which approximates the inward-facing orientation of meshes scanned from inside a room (S3DIS,
    ScanNet) and matters when the consuming model was trained on oriented normals.

    Args:
        pos: Point coordinates of shape $(N, 3)$.
        k: Number of nearest neighbors (the point itself included) used per local PCA. Must not exceed the
            number of points in the smallest cloud.
        batch: Optional $(N,)$ batch index so neighbors never cross cloud boundaries.
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


class EstimateNormals(DictTransform):
    r"""Estimate per-point surface normals from coordinates via local PCA.

    Computes unit normals (see `torch_pointcloud.transforms.functional.estimate_normals`) for clouds that
    ship without them (e.g. S3DIS). Each normal is the least-variance direction of a point's $k$ nearest
    neighbors. With `orient_to_centroid`, normals are flipped to face the cloud centroid.

    See Also:
        `torch_pointcloud.transforms.functional.estimate_normals`

    === "Object"

        ![EstimateNormals on an object](../../assets/transforms/estimate_normals.png)

    === "Scene"

        ![EstimateNormals on a room](../../assets/transforms/estimate_normals_scene.png)

    Args:
        keys: Coordinate keys to estimate normals from.
        normal_key: Keys under which to store the normals (one per coordinate key). Defaults to `normal`.
        k: Number of nearest neighbors (the point itself included) per local PCA.
        orient_to_centroid: If `True`, flip each normal to point towards its cloud's centroid (approximates
            the inward-facing normals of meshes scanned from inside a room).
        batch_key: Optional key holding a per-point batch index so neighbors stay within a cloud.
        allow_missing_keys: If `True`, silently skip absent keys.
    """

    def __init__(
        self,
        keys: KeyCollection,
        normal_key: KeyCollection = "normal",
        k: int = 16,
        orient_to_centroid: bool = False,
        batch_key: Optional[str] = None,
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.normal_key = ensure_tuple_size(normal_key, len(self.keys))
        self.k = k
        self.orient_to_centroid = orient_to_centroid
        self.batch_key = batch_key

    def transform(self, data: Dict[str, Any]) -> Dict[str, Any]:
        d = dict(data)
        batch = d.get(self.batch_key) if self.batch_key is not None else None
        for key, normal_key in self.iter_keys(d, self.normal_key):
            d[normal_key] = estimate_normals(
                d[key],
                k=self.k,
                batch=batch,
                orient_to_centroid=self.orient_to_centroid,
            )
        return d


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


class Shift(DictTransform):
    r"""Shift dictionary tensor entries by subtracting a computed offset.

    Each key is offset independently. The offset is determined by `method`:

    | Method       | Offset                                           |
    | ------------ | ------------------------------------------------ |
    | `"bbox"`     | Midrange: `(min + max) / 2`                      |
    | `"centroid"` | Mean across the reduced dimension                |
    | `"min"`      | Per-axis minimum (shifts to the positive octant) |

    On empty inputs (size $0$ along `dim`) the tensor is returned unchanged.

    === "method=centroid"

        ![Shift method=centroid on an object](../../assets/transforms/shift_centroid.png)

    === "method=bbox"

        ![Shift method=bbox on an object](../../assets/transforms/shift_bbox.png)

    === "method=min"

        ![Shift method=min on an object](../../assets/transforms/shift_min.png)

    === "axes"

        ![Shift restricted to a subset of axes, on an object](../../assets/transforms/shift_axes.png)

    Args:
        keys: The keys to shift.
        method: `"bbox"` (midrange), `"centroid"` (mean), or `"min"` (shift to origin).
        dim: The dimension to reduce over.
        axes: Which axes (last-dim indices) to shift. `None` (default) shifts every
            axis; pass e.g. `axes=[0, 1]` to recenter only XY. Axes outside this list are
            left unchanged - this is the composable knob for mixed-method shifts.
        dst_keys: The keys to store the shifted data in.
        allow_missing_keys: If `True`, skip missing keys silently.

    Example:
        XY shifted by the bbox midpoint, Z shifted by its minimum (equivalent
        to the old `CenterShift(apply_z=True)`):

        ```python
        from torch_pointcloud.transforms import Compose, Shift

        center_shift = Compose([
            Shift(keys="pos", method="bbox", axes=[0, 1]),  # XY: bbox midrange
            Shift(keys="pos", method="min",  axes=[2]),     # Z:  min
        ])
        ```

        Without the Z step (equivalent to the old `CenterShift(apply_z=False)`),
        a single `Shift` suffices:

        ```python
        Shift(keys="pos", method="bbox", axes=[0, 1])
        ```
    """

    def __init__(
        self,
        keys: KeyCollection,
        method: ValueCollection[ShiftMethod],
        dim: int = 0,
        axes: Optional[Sequence[int]] = None,
        dst_keys: Optional[KeyCollection] = None,
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.dim = dim
        self.axes = tuple(axes) if axes is not None else None
        self.dst_keys = ensure_tuple_size(dst_keys or self.keys, len(self.keys))
        self.method = ensure_tuple_size(method, len(self.keys))
        valid = get_args(ShiftMethod)
        if not set(self.method).issubset(valid):
            raise ValueError(f"Invalid method: {method!r}. Expected one of {valid}.")

    def transform(self, data: Dict[str, Any]) -> Dict[str, Any]:
        data = dict(data)
        for key, dst_key, method in self.iter_keys(data, self.dst_keys, self.method):
            x = data[key]
            if not torch.is_tensor(x):
                raise TypeError(f"Expected a tensor, got {type(x).__name__!r}.")
            data[dst_key] = shift(x, method=method, dim=self.dim, axes=self.axes)
        return data


class AlignAxis(DictTransform):
    """Shift dictionary tensor entries so that the minimum along a chosen axis is zero.

    Empty inputs (`N=0`) are returned unchanged.

    === "Object"

        ![AlignAxis on an object](../../assets/transforms/align_axis.png)

    === "Scene"

        ![AlignAxis on a room](../../assets/transforms/align_axis_scene.png)

    Args:
        keys: The keys to align.
        dim: The coordinate axis to align.
        inplace: Whether to modify the tensor in place. Non-contiguous inputs are
            materialized to contiguous via `.contiguous()` before the in-place op,
            so the caller's original tensor may not be mutated in that case.
        allow_missing_keys: If `True`, the transform will not raise an error if the keys are not present in the data.
    """

    def __init__(
        self,
        keys: KeyCollection,
        dim: int = -1,
        inplace: bool = False,
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.dim = dim
        self.inplace = inplace

    def transform(self, data: dict) -> dict:
        data = dict(data)

        for key in self.iter_keys(data):
            x = data[key]
            if not torch.is_tensor(x):
                raise TypeError(f"Expected a tensor, got {type(x).__name__!r}.")

            if x.shape[0] == 0:
                data[key] = x
                continue

            if self.inplace:
                x = x.contiguous()
            else:
                x = x.clone()

            x[:, self.dim] -= x[:, self.dim].min()
            data[key] = x

        return data


class BBoxCenter(DictTransform):
    r"""Derive the center of an axis-aligned bbox stored as a flat tensor.

    Reads a bbox at each source key, laid out as a $(2D,)$ vector
    $[\,\min_0, \ldots, \min_{D-1},\, \max_0, \ldots, \max_{D-1}\,]$, and writes
    the per-axis midpoint $(\min + \max) / 2$ (shape $(D,)$) at the matching
    destination key.

    === "Object"

        ![BBoxCenter on an object](../../assets/transforms/bbox_center.png)

    === "Scene"

        ![BBoxCenter on a room](../../assets/transforms/bbox_center_scene.png)

    Args:
        keys: Source keys holding flat bbox tensors of shape $(2D,)$.
        dst_keys: Destination keys for the centers. Defaults to overwriting
            the source keys.
        allow_missing_keys: If `True`, silently skip absent source keys.

    Example:
        ```python
        from torch_pointcloud.transforms import BBoxCenter

        data = {"block_bbox": torch.tensor([0.0, 0.0, 0.0, 1.5, 1.5, 2.8])}
        BBoxCenter(keys="block_bbox", dst_keys="block_center")(data)
        # data["block_center"] == tensor([0.75, 0.75, 1.40])
        ```
    """

    def __init__(
        self,
        keys: KeyCollection,
        dst_keys: Optional[KeyCollection] = None,
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.dst_keys = ensure_tuple_size(dst_keys or self.keys, len(self.keys))

    def transform(self, data: Dict[str, Any]) -> Dict[str, Any]:
        data = dict(data)
        for key, dst_key in self.iter_keys(data, self.dst_keys):
            bbox = data[key]
            if bbox.numel() % 2 != 0:
                raise ValueError(f"`{key}` must have an even number of elements (got {bbox.numel()}).")
            n_dim = bbox.numel() // 2
            data[dst_key] = (bbox[:n_dim] + bbox[n_dim:]) / 2.0
        return data


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


class Quantize(DictTransform):
    r"""Integer voxel-grid coordinates of every point, keeping the cloud at full resolution.

    Stores $\lfloor p / s \rfloor$ (shifted so the per-axis minimum is $0$) for each point of `keys` under
    `dst_keys`. Unlike `Voxelize`, no reduction happens: points sharing a voxel keep their own rows and get equal
    coordinates. This is how a voxel-partition evaluation feeds sparse models with every raw point (each
    sub-cloud holds one point per voxel, so its rows are exactly the voxels), and how test-time views recompute
    grid coordinates after rotating or scaling the positions.

    === "Object"

        ![Quantize on an object](../../assets/transforms/quantize.png)

    === "Scene"

        ![Quantize on a room](../../assets/transforms/quantize_scene.png)

    Args:
        keys: Keys holding point positions of shape $(N, D)$.
        size: Voxel side length in the units of the positions.
        dst_keys: Keys under which the grid coordinates are stored. Defaults to `keys` (in-place overwrite).
        allow_missing_keys: If `True`, missing keys are skipped.

    Example:
        ```python
        import torch
        from torch_pointcloud.transforms import Quantize

        transform = Quantize(keys="pos", size=0.02, dst_keys="pos_grid")
        data = transform({"pos": torch.tensor([[0.0, 0.0, 0.0], [0.03, 0.0, 0.0], [0.05, 0.0, 0.0]])})
        data["pos_grid"]  # tensor([[0, 0, 0], [1, 0, 0], [2, 0, 0]])
        ```
    """

    def __init__(
        self,
        keys: KeyCollection,
        size: float,
        dst_keys: Optional[KeyCollection] = None,
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        if size <= 0.0:
            raise ValueError(f"`size` must be > 0, got {size}.")

        self.size = size
        self.dst_keys = ensure_tuple_size(dst_keys, len(self.keys)) if dst_keys is not None else self.keys

    def transform(self, data: Dict[str, Any]) -> Dict[str, Any]:
        data = dict(data)
        for key, dst_key in self.iter_keys(data, self.dst_keys):
            data[dst_key] = quantize(data[key], self.size)
        return data


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
        x: Input tensor of shape $(N, D)$.
        axis: Axis index in the last dimension.
        quantile: Optional quantile $q \in [0, 1]$ for the floor reference. When
            `None`, the strict per-axis minimum is used (equivalent to $q = 0$).

    Returns:
        Tensor of shape $(N, 1)$ with the same dtype as `x`. Returns an empty
        $(0, 1)$ tensor when `x` is empty.
    """
    col = x[:, axis]
    if col.numel() == 0:
        return col.unsqueeze(-1).to(x.dtype)
    if quantile is None:
        ref = col.min()
    else:
        ref = torch.quantile(col.float(), quantile).to(col.dtype)
    return (col - ref).unsqueeze(-1).to(x.dtype)


class AxisMinOffset(DictTransform):
    r"""Per-point offset from a floor reference along a chosen coordinate axis.

    For each point and a given axis $a$ along tensor dimension $d$, computes:

    $$
    o_i = p_{i,a} - r
    $$

    where the floor reference $r$ is either the strict minimum $\min_j p_{j,a}$
    (default) or, when `quantile` is set, the empirical quantile
    $Q_q(p_{\cdot,a})$. A small positive quantile gives an outlier-robust floor
    estimate: `quantile=0.0099` reproduces VoteNet's `np.percentile(z, 0.99)`
    height feature.

    The result has the same shape as the input with the coordinate dimension
    reduced to size 1 (e.g. $(N, 3) \to (N, 1)$ or $(B, N, 3) \to (B, N, 1)$).
    For batched inputs, the minimum is computed per-sample.

    === "Object"

        ![AxisMinOffset on an object](../../assets/transforms/axis_min_offset.png)

    === "Scene"

        ![AxisMinOffset on a room](../../assets/transforms/axis_min_offset_scene.png)

    Args:
        keys: Keys holding point positions of shape $(N, D)$.
        axis: Coordinate axis $a$ along which to compute the offset.
        quantile: Optional quantile $q \in [0, 1]$ for the floor reference. When
            `None`, the strict per-axis minimum is used.
        dst_keys: Keys under which the offset tensors are stored. Defaults to `keys`
            (in-place overwrite).
        allow_missing_keys: If True, skip missing keys instead of raising.

    Example:
        Let's say you have a point cloud with positions $(N, 3)$ in XYZ order
        and you want to compute the offset from the minimum along the z-axis,
        i.e. computing the height above the local floor.

        ```python
        from torch_pointcloud.transforms import AxisMinOffset

        data = {
            "pos": torch.randn(10, 3),
        }
        transform = AxisMinOffset(keys="pos", dst_keys="pos_offset", axis=2)
        data = transform(data)
        ```

        Now, the data dictionary will contain the key `pos_offset` with the shape $(N, 1)$.
    """

    def __init__(
        self,
        keys: KeyCollection,
        axis: ValueCollection[int],
        quantile: Optional[float] = None,
        dst_keys: KeyCollection | None = None,
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.dst_keys = ensure_tuple_size(dst_keys or self.keys, len(self.keys))
        self.axis = ensure_tuple_size(axis, len(self.keys))
        self.quantile = quantile

    def transform(self, data: Dict[str, Any]) -> Dict[str, Any]:
        data = dict(data)
        for key, dst_key, axis in self.iter_keys(data, self.dst_keys, self.axis):
            data[dst_key] = axis_min_offset(data[key], axis=axis, quantile=self.quantile)
        return data


def rotation_matrix(angle: float, axis: int = 2, device: Optional[torch.device] = None) -> Tensor:
    r"""$3 \times 3$ rotation matrix for `angle` radians around an axis-aligned axis.

    Args:
        angle: Rotation angle in **radians**.
        axis: Axis index to rotate around (0=X, 1=Y, 2=Z).
        device: Output device. Defaults to CPU.

    Returns:
        Rotation matrix of shape $(3, 3)$.

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
