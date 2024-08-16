import math
from typing import Literal

import torch
from torch import Tensor


def cross_product_matrix(k: Tensor) -> Tensor:
    r"""Constructs a skew-symmetric matrix (also known as a cross-product matrix) 
    for a given 3D vector $k = [k1, k2, k3]$. The function returns a 
    3x3 skew-symmetric matrix `M(k)` of the form:

    $$
    M(k) = \begin{bmatrix}
    0 & -k_3 & k_2 \\
    k_3 & 0 & -k_1 \\
    -k_2 & k_1 & 0
    \end{bmatrix}
    $$

    Args:
        k: A tensor of shape `[3]` representing the 3D vector.

    Returns:
        A 3x3 skew-symmetric matrix corresponding to the cross-product operation.

    Example:
        >>> k = torch.tensor([1.0, 2.0, 3.0])
        >>> v = torch.tensor([4.0, 5.0, 6.0])
        >>> m = cross_product_matrix(k)
        >>> cross_product = torch.matmul(m, v)
    """

    m = [
        [0, -k[2], k[1]],
        [k[2], 0, -k[0]],
        [-k[1], k[0], 0],
    ]

    return torch.tensor(m, device=k.device)


def cross_product_matrices(k: Tensor) -> Tensor:
    r"""Constructs a batch of skew-symmetric matrices (also known as cross-product matrices) 
    for a batch of 3D vectors.

    For each 3D vector \( k = [k_1, k_2, k_3] \), the corresponding skew-symmetric matrix \( K \) is:

    $$
    K = \begin{bmatrix}
    0 & -k_3 & k_2 \\
    k_3 & 0 & -k_1 \\
    -k_2 & k_1 & 0
    \end{bmatrix}
    $$

    Args:
        k: A tensor of shape `[N, 3]` representing N 3D vectors.

    Returns:
        A tensor of shape [N, 3, 3] where each [3, 3] matrix is the skew-symmetric 
        matrix corresponding to the cross-product operation for each vector.
    """
    K = torch.zeros(k.shape[0], 3, 3, device=k.device)
    K[:, 0, 1] = -k[:, 2]
    K[:, 0, 2] = k[:, 1]
    K[:, 1, 0] = k[:, 2]
    K[:, 1, 2] = -k[:, 0]
    K[:, 2, 0] = -k[:, 1]
    K[:, 2, 1] = k[:, 0]

    return K


def rodrigues_rotation_matrix(axis: Tensor, theta_degrees: float) -> Tensor:
    r"""Computes a 3D rotation matrix using Rodrigues' rotation formula.

    This function rotates a vector in 3D space around a specified axis by a
    given angle. The rotation matrix is computed using the following formula:

    $$
    R = I + \sin(\theta)K + (1 - \cos(\theta))K^2
    $$

    Where:
    - \( I \) is the identity matrix.
    - \( K \) is the skew-symmetric matrix (cross-product matrix) derived from the axis of rotation.
    - \( \theta \) is the rotation angle in radians, converted from degrees.

    Args:
        axis: A 3D vector representing the axis of rotation.
        theta_degrees: The angle of rotation in degrees.

    Returns:
        A 3x3 rotation matrix that rotates a vector around the specified axis by the specified angle.
    """
    axis = axis.detach().clone().float()
    axis = axis / axis.norm()
    K = cross_product_matrix(axis)
    t = torch.tensor([theta_degrees / 180.0 * math.pi], device=axis.device)
    R = torch.eye(3, device=axis.device) + torch.sin(t) * K + (1 - torch.cos(t)) * K.mm(K)
    return R


def rodrigues_rotation_matrices(axes: Tensor, theta_degrees: Tensor) -> Tensor:
    r"""Computes a batch of 3D rotation matrices using Rodrigues' rotation formula.

    Rodrigues' rotation formula for rotating a vector by an angle \( \theta \) around
    an axis \( k \) is given by:

    $$
    R = I + \sin(\theta)K + (1 - \cos(\theta))K^2
    $$

    Where:
    - \( I \) is the identity matrix.
    - \( K \) is the skew-symmetric matrix (cross-product matrix) of the axis \( k \).
    - \( \theta \) is the angle of rotation in radians.

    Args:
        axes: A tensor of shape $[N, 3]$ representing the axes of rotation.
        theta_degrees: A tensor of shape $[N,]$ representing the angles of rotation in degrees.

    Returns:
        A tensor of shape $[N, 3, 3]$ containing the rotation matrices for each axis-angle pair.
    """
    axes = axes / axes.norm(dim=1, keepdim=True)  # Normalize the axes
    theta_radians = theta_degrees * math.pi / 180.0  # Convert angles to radians

    # Create batch of cross-product matrices for the axes
    K = cross_product_matrices(axes)

    # Rodrigues' formula: R = I + sin(theta) * K + (1 - cos(theta)) * K^2
    eye = torch.eye(3, device=axes.device).unsqueeze(0)  # Identity matrix of shape [1, 3, 3]
    sin_theta = torch.sin(theta_radians).unsqueeze(1).unsqueeze(2)  # [N, 1, 1]
    cos_theta = torch.cos(theta_radians).unsqueeze(1).unsqueeze(2)  # [N, 1, 1]

    # Compute the rotation matrix using tensor operations
    R = eye + sin_theta * K + (1 - cos_theta) * K @ K

    return R


def spherical_lloyd(
    radius: float,
    num_cells: int,
    dim: int = 3,
    position: Literal["none", "center", "vertical"] = "none",
    approximation: Literal["discretization", "monte-carlo"] = "discretization",
    approx_n: int = 5000,
    max_iter: int = 500,
    momentum: float = 0.9,
) -> Tensor:
    r"""Generate kernel points using Lloyd's algorithm on a sphere.

    Args:
        radius (float): Radius of the sphere.
        num_cells (int): Number of kernel points (Voronoi cells).
        dim (int, optional): Dimensionality of the space. Defaults to 3.
        position (str, optional): Fix the position of specific kernel points. Defaults to 'none'.
            Options:
                - 'none': No kernel points are fixed. All points move freely during optimization.
                - 'center': The first kernel point is fixed at the center of the sphere.
                - 'vertical': (3D only) The first three kernel points are fixed along the z-axis:
                    - The first point is fixed at the center.
                    - The second point is placed above the center along the positive z-axis.
                    - The third point is placed below the center along the negative z-axis.
        approximation (str, optional): Approximation method for Lloyd's algorithm. Defaults to 'discretization'.
            Options:
                - 'discretization': Approximates the Voronoi cells using a regular grid of points within the sphere.
                - 'monte-carlo': Approximates the Voronoi cells by randomly sampling points within the sphere.
        approx_n (int, optional): Number of points used for approximation. Defaults to 5000.
        max_iter (int, optional): Maximum number of iterations. Defaults to 500.
        momentum (float, optional): Momentum factor for smoothing kernel point positions. Defaults to 0.9.

    Returns:
        torch.Tensor: Tensor of shape [num_cells, dimension] with the final kernel points on the sphere.
    """

    if not (2 <= dim <= 4):
        raise ValueError("Unsupported dimension. Expected 2, 3, or 4.")
    if position not in ["none", "center", "vertical"]:
        raise ValueError('Unsupported position. Expected "none", "center", or "vertical".')
    if approximation not in ["discretization", "monte-carlo"]:
        raise ValueError('Unsupported approximation. Expected "discretization" or "monte-carlo".')

    # Constants
    radius0 = 1.0  # Initial radius for optimization

    # Generate random kernel points inside the sphere
    kernel_points = torch.empty((0, dim))
    while kernel_points.size(0) < num_cells:
        new_points = (torch.rand(num_cells, dim) * 2 - 1) * radius0
        valid_points = new_points[(new_points.norm(dim=1) < radius0) & (new_points.norm(dim=1) > (0.9 * radius0))]
        kernel_points = torch.cat((kernel_points, valid_points), dim=0)

    kernel_points = kernel_points[:num_cells]

    # Optional fixing of kernel positions
    if position == "center":
        kernel_points[0] = torch.zeros(dim)
    elif position == "vertical" and dim == 3:
        kernel_points[:3, :] = torch.zeros(3, dim)
        kernel_points[1, -1] += 2 * radius0 / 3
        kernel_points[2, -1] -= 2 * radius0 / 3

    # Initialize the approximation points
    if approximation == "discretization":
        side_n = int(approx_n ** (1.0 / dim))
        coords = torch.linspace(-radius0, radius0, side_n, device=kernel_points.device)
        mesh = torch.meshgrid([coords] * dim, indexing="ij")
        X = torch.stack([m.flatten() for m in mesh], dim=-1)
    else:
        X = torch.empty((0, dim), device=kernel_points.device)

    # Only keep points inside the sphere
    X = X[X.norm(dim=1) < radius0]

    # Tensor for tracking the maximum moves
    max_moves = torch.empty((0,), device=kernel_points.device)

    # Lloyd's algorithm iterations
    for _ in range(max_iter):
        if approximation == "monte-carlo":
            X = torch.empty(approx_n, dim, device=kernel_points.device).uniform_(-radius0, radius0)
            X = X[X.norm(dim=1) < radius0]

        # Compute distances and assign points to nearest kernel
        diff = X.unsqueeze(1) - kernel_points.unsqueeze(0)
        dist2 = torch.sum(diff**2, dim=-1)
        nearest_kernel_idx = torch.argmin(dist2, dim=-1)

        # Calculate new kernel positions (cell centers)
        new_kernel_point_list = []
        for c in range(num_cells):
            points_in_cell = X[nearest_kernel_idx == c]
            new_kernel_point_list.append(points_in_cell.mean(dim=0) if points_in_cell.size(0) > 0 else kernel_points[c])

        new_kernel_points = torch.stack(new_kernel_point_list)

        # Update kernel points with momentum smoothing
        moves = (1 - momentum) * (new_kernel_points - kernel_points)
        kernel_points += moves

        # Track the maximum move for each iteration
        max_move_per_iter = torch.max(torch.norm(moves, dim=1))
        max_moves = torch.cat((max_moves, max_move_per_iter.unsqueeze(0)))

        # Optional fixing of kernel positions
        if position == "center":
            kernel_points[0] = torch.zeros(dim)
        elif position == "vertical" and dim == 3:
            kernel_points[0] = torch.zeros(dim)
            kernel_points[1:, :-1] = torch.zeros_like(kernel_points[1:, :-1])

    return kernel_points * radius


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
