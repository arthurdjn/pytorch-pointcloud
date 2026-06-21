from typing import TYPE_CHECKING, Literal, Tuple

import torch
from torch import Tensor

from torch_pointcloud.utils.imports import optional_import
from torch_pointcloud.utils.types import OptTensor

from .hilbert import encode as hilbert_encode

if TYPE_CHECKING:
    from ocnn.octree import xyz2key as octree_encode

octree_encode, _ = optional_import("ocnn.octree", "xyz2key")


MAX_DEPTH = 16
MAX_CODE_BITS = 63

SerializationOrder = Literal["z", "z-trans", "hilbert", "hilbert-trans"]
SERIALIZATION_ORDERS: Tuple[SerializationOrder, ...] = ("z", "z-trans", "hilbert", "hilbert-trans")


def _z_order_encode(grid_coord: Tensor, depth: int) -> Tensor:
    x, y, z = grid_coord[:, 0].long(), grid_coord[:, 1].long(), grid_coord[:, 2].long()
    return octree_encode(x, y, z, b=None, depth=depth)


def _hilbert_encode(grid_coord: Tensor, depth: int) -> Tensor:
    return hilbert_encode(grid_coord, num_dims=3, num_bits=depth)


@torch.no_grad()
def serialize_coords(
    grid_coords: Tensor,
    batch_idx: OptTensor,
    depth: int,
    order: SerializationOrder,
) -> Tensor:
    """Encode / serialize grid coordinates into a code depending on the serialization order.
    The code can be used to sort the grid coordinates or to index them, and was introduced in the paper
    :arxiv: [Point Transformer V3: Simpler, Faster, Stronger](https://arxiv.org/pdf/2312.10035)
    by Xiaoyang Wu, Li Jiang, Peng-Shuai Wang, Zhijian Liu, Xihui Liu, Yu Qiao, Wanli Ouyang, Tong He, Hengshuang Zhao.

    Note:
        To get the code's order and inverse, you can use `torch.argsort` twice:
        >>> code = serialize_coords(grid_coords, batch_idx, depth, order)  # doctest: +SKIP
        >>> order = torch.argsort(code)  # doctest: +SKIP
        >>> inverse = torch.argsort(order)  # doctest: +SKIP

    Args:
        grid_coords: A int tensor of shape $(N, 3)$ containing the grid coordinates.
        batch_idx: A int tensor of contiguous values from 0 to $B - 1$ of shape $(N)$ containing the batch $B$ indices.
        depth: The depth of the serialization cube.
        order: The serialization order. Available orders are:
            - "z": Z-order curve.
            - "z-trans": Z-order curve transposed.
            - "hilbert": Hilbert curve.
            - "hilbert-trans": Hilbert curve transposed.

    Returns:
        A int tensor of shape $(N)$ containing the serialized grid coordinates.

    Examples:
        >>> coords = torch.randn(10, 3)
        >>> grid_size = 0.1
        >>> grid_coords = torch.div(coords - coords.min(0).values, grid_size, rounding_mode="trunc")
        >>> batch_idx = torch.zeros(10, dtype=torch.long)
        >>> code = serialize_coords(grid_coords, batch_idx, depth=5, order="z")  # doctest: +SKIP
    """
    if order not in SERIALIZATION_ORDERS:
        expected_orders = ", ".join(SERIALIZATION_ORDERS)
        raise ValueError(f"Unsupported serialization order: {order}. Expected one of: {expected_orders}")

    if order == "z":
        serialized_code = _z_order_encode(grid_coords, depth=depth)
    elif order == "z-trans":
        serialized_code = _z_order_encode(grid_coords[:, [1, 0, 2]], depth=depth)
    elif order == "hilbert":
        serialized_code = _hilbert_encode(grid_coords, depth=depth)
    elif order == "hilbert-trans":
        serialized_code = _hilbert_encode(grid_coords[:, [1, 0, 2]], depth=depth)

    if batch_idx is not None:
        serialized_code = batch_idx << depth * 3 | serialized_code

    return serialized_code
