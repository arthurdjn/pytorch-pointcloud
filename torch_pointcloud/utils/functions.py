import math

import torch
from torch import Tensor


def cross_product_matrix(k: Tensor) -> Tensor:
    m = [
        [0, -k[2], k[1]],
        [k[2], 0, -k[0]],
        [-k[1], k[0], 0],
    ]

    return torch.tensor(m, device=k.device)


def rodrigues_rotation_matrix(axis: Tensor, theta_degrees: float) -> Tensor:
    axis = axis.detach().clone().float()
    axis = axis / axis.norm()
    K = cross_product_matrix(axis)
    t = torch.tensor([theta_degrees / 180.0 * math.pi], device=axis.device)
    R = torch.eye(3, device=axis.device) + torch.sin(t) * K + (1 - torch.cos(t)) * K.mm(K)
    return R
