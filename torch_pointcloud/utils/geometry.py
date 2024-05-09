import torch
from torch import Tensor


def axis_aligned_bounding_box(xyz: Tensor) -> Tensor:
    """Compute the axis aligned bounding box of a set of points,
    parameterized by (cx,cy,cz) and (dx,dy,dz) where (cx,cy,cz) is the center point of the box,
    and dx is the x-axis length of the box.

    Args:
        xyz: Points of shape (N,3), in XYZ order.

    Returns:
        The axis aligned bounding box of shape (6,).
    """
    x_min, y_min, z_min, *_ = torch.min(xyz, dim=0).values
    x_max, y_max, z_max, *_ = torch.max(xyz, dim=0).values
    cx, cy, cz = (x_min + x_max) / 2.0, (y_min + y_max) / 2.0, (z_min + z_max) / 2.0
    dx, dy, dz = x_max - x_min, y_max - y_min, z_max - z_min
    return torch.tensor([cx, cy, cz, dx, dy, dz], device=xyz.device)
