from typing import Literal, Tuple

import torch
from ocnn.octree import xyz2key as octree_encode
from torch import Tensor

from .hilbert import encode as hilbert_encode

MAX_DEPTH = 16
MAX_CODE_BITS = 63


SerializationOrder = Literal["z", "z-trans", "hilbert", "hilbert-trans"]
SERIALIZATION_ORDERS: Tuple[SerializationOrder, ...] = ("z", "z-trans", "hilbert", "hilbert-trans")


def _z_order_encode(grid_coord: Tensor, depth: int) -> Tensor:
    x, y, z = grid_coord[:, 0].long(), grid_coord[:, 1].long(), grid_coord[:, 2].long()
    # we block the support to batch, maintain batched code in Point class
    code = octree_encode(x, y, z, b=None, depth=depth)
    return code


def _hilbert_encode(grid_coord: Tensor, depth: int) -> Tensor:
    return hilbert_encode(grid_coord, num_dims=3, num_bits=depth)


@torch.inference_mode()
def serialize_grid_coords(
    grid_coords: Tensor,
    batch_idx: Tensor,
    depth: int,
    order: SerializationOrder,
) -> Tuple[Tensor, Tensor, Tensor]:
    if order not in SERIALIZATION_ORDERS:
        expected_orders = ", ".join(SERIALIZATION_ORDERS)
        raise ValueError(f"Unsupported serialization order: {order}. Expected one of: {expected_orders}")

    # Adaptive measure the depth of serialization cube (length = 2 ^ depth)
    # depth = int(grid_coords.max()).bit_length()

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

    serialized_order = torch.argsort(serialized_code)
    serialized_inverse = torch.zeros_like(serialized_order).scatter_(
        dim=0,
        index=serialized_order,
        src=torch.arange(0, len(serialized_code), device=serialized_order.device),
    )

    return serialized_code, serialized_order, serialized_inverse
