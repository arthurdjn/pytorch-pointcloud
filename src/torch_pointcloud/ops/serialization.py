"""Space-filling curve serialization of voxel coordinates using Z-order and Hilbert encodings."""

from typing import TYPE_CHECKING, Literal, Sequence, Tuple

import torch
from torch import Tensor

from torch_pointcloud.ops.hilbert import encode as hilbert_encode
from torch_pointcloud.utils.imports import _OCNN_GITHUB_URL, optional_import
from torch_pointcloud.utils.types import OptTensor

if TYPE_CHECKING:
    from ocnn.octree import xyz2key as octree_encode

octree_encode, _ = optional_import("ocnn.octree", "xyz2key", url=_OCNN_GITHUB_URL)


MAX_DEPTH = 16
MAX_CODE_BITS = 63

SerializationOrder = Literal["z", "z-trans", "hilbert", "hilbert-trans"]
SERIALIZATION_ORDERS: Tuple[SerializationOrder, ...] = ("z", "z-trans", "hilbert", "hilbert-trans")


def _z_order_encode(pos_grid: Tensor, depth: int) -> Tensor:
    x, y, z = pos_grid[:, 0].long(), pos_grid[:, 1].long(), pos_grid[:, 2].long()
    return octree_encode(x, y, z, b=None, depth=depth)


def _hilbert_encode(pos_grid: Tensor, depth: int) -> Tensor:
    return hilbert_encode(pos_grid, num_dims=3, num_bits=depth)


@torch.no_grad()
def serialize_pos(
    pos_grid: Tensor,
    batch: OptTensor,
    depth: int,
    order: SerializationOrder,
) -> Tensor:
    r"""Encode / serialize grid coordinates into a code depending on the serialization order.
    The code can be used to sort the grid coordinates or to index them, and was introduced in the paper
    :arxiv: [Point Transformer V3: Simpler, Faster, Stronger](https://arxiv.org/pdf/2312.10035)
    by Xiaoyang Wu, Li Jiang, Peng-Shuai Wang, Zhijian Liu, Xihui Liu, Yu Qiao, Wanli Ouyang, Tong He, Hengshuang Zhao.

    Note:
        To get the code's order and inverse, you can use `torch.argsort` twice:
        ```pycon
        >>> code = serialize_pos(pos_grid, batch, depth, order)  # doctest: +SKIP
        >>> order = torch.argsort(code)  # doctest: +SKIP
        >>> inverse = torch.argsort(order)  # doctest: +SKIP

        ```

    Args:
        pos_grid: A int tensor of shape $(N, 3)$ containing the grid coordinates. Every coordinate must lie
            in $[0, 2^{\text{depth}})$ per axis: the encoders keep only the low `depth` bits, so out-of-range values
            (e.g. a negative coordinate) silently wrap around to a valid code. Grids produced by `Voxelize` or
            `Quantize` are shifted by the per-axis minimum and satisfy this.
        batch: A int tensor of contiguous values from 0 to $B - 1$ of shape $(N)$ containing the batch $B$ indices.
        depth: The depth of the serialization cube.
        order: The serialization order. Available orders are:
            - "z": Z-order curve.
            - "z-trans": Z-order curve transposed.
            - "hilbert": Hilbert curve.
            - "hilbert-trans": Hilbert curve transposed.

    Returns:
        A int tensor of shape $(N)$ containing the serialized grid coordinates.

    Examples:
        ```pycon
        >>> pos = torch.randn(10, 3)
        >>> grid_size = 0.1
        >>> pos_grid = torch.div(pos - pos.min(0).values, grid_size, rounding_mode="trunc")
        >>> batch = torch.zeros(10, dtype=torch.long)
        >>> code = serialize_pos(pos_grid, batch, depth=5, order="z")  # doctest: +SKIP

        ```
    """
    if order not in SERIALIZATION_ORDERS:
        expected_orders = ", ".join(SERIALIZATION_ORDERS)
        raise ValueError(f"Unsupported serialization order: {order}. Expected one of: {expected_orders}")

    # ocnn's key tables stop at depth 16; hilbert supports up to 21 and validates itself.
    if order in ("z", "z-trans") and depth > MAX_DEPTH:
        raise ValueError(
            f"Serialization depth {depth} exceeds the z-order maximum of {MAX_DEPTH} (grid extents above "
            f"2**{MAX_DEPTH} cells per axis). Increase the grid size or use a hilbert order."
        )

    if order == "z":
        serialized_code = _z_order_encode(pos_grid, depth=depth)
    elif order == "z-trans":
        serialized_code = _z_order_encode(pos_grid[:, [1, 0, 2]], depth=depth)
    elif order == "hilbert":
        serialized_code = _hilbert_encode(pos_grid, depth=depth)
    elif order == "hilbert-trans":
        serialized_code = _hilbert_encode(pos_grid[:, [1, 0, 2]], depth=depth)

    if batch is not None:
        max_batch = int(batch.max().item()) if batch.numel() else 0
        if depth * 3 + max_batch.bit_length() > MAX_CODE_BITS:
            raise ValueError(
                f"Batch index {max_batch} needs {max_batch.bit_length()} bits above the {depth * 3} coordinate "
                f"bits, exceeding the {MAX_CODE_BITS}-bit code capacity. Reduce `depth` or the batch size."
            )

        serialized_code = batch << depth * 3 | serialized_code

    return serialized_code


def serialize(
    pos_grid: Tensor,
    batch: Tensor,
    orders: Sequence[SerializationOrder],
    shuffle: bool = False,
) -> Tuple[Tensor, Tensor, Tensor]:
    r"""Encode voxel-grid coordinates along one or more space-filling curves and sort the points by each code.

    Args:
        pos_grid: Non-negative integer grid coordinates of shape $(N, 3)$.
        batch: Per-point batch index of shape $(N,)$.
        orders: The $L$ space-filling curves to encode along, one code row per order.
        shuffle: Whether to permute `orders` before encoding, so consecutive blocks pick different curves.

    Returns:
        The serialization codes, the permutation sorting the points by code, and its inverse.

    Raises:
        ValueError: If `pos_grid` holds a negative coordinate, which would silently wrap around to a valid code.

    Shape:
        - pos_grid: $(N, 3)$
        - batch: $(N,)$
        - output: three tensors of shape $(L, N)$.

    Example:
        ```pycon
        >>> pos_grid = torch.tensor([[0, 0, 0], [3, 1, 2], [1, 1, 1]])
        >>> batch = torch.zeros(3, dtype=torch.long)
        >>> code, order, inverse = serialize(pos_grid, batch, orders=["hilbert", "hilbert-trans"])
        >>> order.shape
        torch.Size([2, 3])
        >>> torch.equal(order.gather(1, inverse), torch.arange(3).expand(2, 3))
        True

        ```
    """
    if shuffle:
        perm = torch.randperm(len(orders))
        orders = [orders[i] for i in perm]

    if pos_grid.numel() and bool(pos_grid.min() < 0):
        raise ValueError(
            "Grid coordinates must be non-negative for serialization: negative values silently wrap around to "
            "valid codes. Shift by the per-axis minimum, as `Voxelize` does."
        )

    # An all-zero grid (single-voxel scene) has bit_length 0, which the encoders reject.
    depth = max(int(pos_grid.max()).bit_length(), 1) if pos_grid.numel() else 1
    serialized_code = torch.stack([serialize_pos(pos_grid, batch, depth=depth, order=order) for order in orders])
    serialized_order = torch.argsort(serialized_code, dim=1)
    serialized_inverse = torch.argsort(serialized_order, dim=1)
    return serialized_code, serialized_order, serialized_inverse
