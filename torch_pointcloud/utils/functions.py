import math
from typing import Literal

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


def spherical_lloyd(
    radius: float,
    num_cells: int,
    dimension: int = 3,
    position: Literal["none", "center", "verticals"] = "none",
    approximation: Literal["discretization", "monte-carlo"] = "discretization",
    approx_n: int = 5000,
    max_iter: int = 500,
    momentum: float = 0.9,
) -> torch.Tensor:
    """
    Creation of kernel points via Lloyd's algorithm on a sphere.

    Args:
        radius: Radius of the kernels.
        num_cells: Number of cells (kernel points) in the Voronoi diagram.
        dimension: Dimension of the space.
        position: Fix position of certain kernel points.
        approximation: Approximation method for Lloyd's algorithm.
        approx_n: Number of points used for approximation.
        max_iter: Maximum number of iterations for the algorithm.
        momentum: Momentum of the low pass filter smoothing kernel point positions.

    Returns:
        Tensor of kernel points with shape [num_cells, dimension], and tensor of maximum moves per iteration.
    """
    if dimension < 2 or dimension > 4:
        raise ValueError("Unsupported dimension. Expected 2, 3 or 4.")
    if position not in ["none", "center", "verticals"]:
        raise ValueError('Unsupported position. Expected "none", "center" or "verticals".')
    if approximation not in ["discretization", "monte-carlo"]:
        raise ValueError('Unsupported approximation method. Expected "discretization" or "monte-carlo".')

    # Radius used for optimization (points are rescaled afterwards)
    radius0 = 1.0

    # Random kernel points (Uniform distribution in a sphere)
    kernel_points = torch.zeros((0, dimension))
    while kernel_points.shape[0] < num_cells:
        new_points = (torch.rand(num_cells, dimension) * 2 - 1) * radius0
        d2 = torch.sum(new_points**2, dim=1)
        valid_points = new_points[(d2 < radius0**2) & (d2 > (0.9 * radius0) ** 2)]
        kernel_points = torch.cat((kernel_points, valid_points), dim=0)

    kernel_points = kernel_points[:num_cells, :].reshape((num_cells, -1))

    # Optional fixing
    if position == "center":
        kernel_points[0, :] *= 0
    elif position == "verticals":
        kernel_points[:3, :] *= 0
        kernel_points[1, -1] += 2 * radius0 / 3
        kernel_points[2, -1] -= 2 * radius0 / 3

    # Initialize discretization if this method is chosen
    if approximation == "monte-carlo":
        X = torch.zeros((0, dimension))
    elif approximation == "discretization":
        side_n = int(approx_n ** (1.0 / dimension))
        dl = 2 * radius0 / side_n
        coords = torch.arange(-radius0 + dl / 2, radius0, dl)
        mesh = torch.meshgrid([coords] * dimension, indexing="ij")
        X = torch.stack([m.ravel() for m in mesh], dim=1)
    else:
        raise ValueError(f'Wrong approximation method chosen: "{approximation}"')

    # Only points inside the sphere are used
    d2 = torch.sum(X**2, dim=1)
    X = X[d2 < radius0 * radius0, :]

    max_moves = torch.zeros((0,))

    for _ in range(max_iter):
        if approximation == "monte-carlo":
            X = torch.empty(approx_n, dimension).uniform_(-radius0, radius0)
            d2 = torch.sum(X**2, dim=1)
            X = X[d2 < radius0**2, :]

        # Get the distances matrix [n_approx, K, dim]
        diff = X.unsqueeze(1) - kernel_points.unsqueeze(0)
        dist2 = torch.sum(diff**2, dim=2)

        # Compute cell centers
        cell_idxs = torch.argmin(dist2, dim=1)
        center_list = []
        for c in range(num_cells):
            mask_idxs = cell_idxs == c
            num_c = torch.sum(mask_idxs)
            center_list.append(torch.sum(X[mask_idxs, :], dim=0) / num_c if num_c > 0 else kernel_points[c])

        # Update kernel points with low pass filter to smooth monte carlo
        centers = torch.stack(center_list, dim=0)
        moves = (1 - momentum) * (centers - kernel_points)
        kernel_points += moves

        # Check moves for convergence
        max_moves = torch.cat((max_moves, torch.max(torch.norm(moves, dim=1)).unsqueeze(0)), dim=0)

        # Optional fixing
        if position == "center":
            kernel_points[0, :] *= 0
        elif position == "verticals":
            kernel_points[0, :] *= 0
            kernel_points[:3, :-1] *= 0

    return kernel_points * radius
