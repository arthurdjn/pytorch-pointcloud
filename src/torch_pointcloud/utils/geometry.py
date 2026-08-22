"""Geometric operations: bounding boxes, rotations, point transforms, vertex normals, and spherical sampling."""

import math
import warnings
from typing import Literal, Optional, Tuple, Union, overload

import torch
from torch import Tensor

from .ops import safe_divide


def axis_aligned_bounding_box(coords: Tensor) -> Tensor:
    r"""Compute the axis aligned bounding box of a set of points,
    parameterized by $(c_x, c_y, c_z)$ and $(d_x, d_y, d_z)$ where $(c_x, c_y, c_z)$ is the center point of the box,
    and $d_x$ is the x-axis length of the box.

    Args:
        coords: Points of shape $(N, 3)$, in XYZ order.

    Returns:
        The axis aligned bounding box of shape $(6,)$.
    """
    x_min, y_min, z_min, *_ = torch.min(coords, dim=0).values
    x_max, y_max, z_max, *_ = torch.max(coords, dim=0).values
    cx, cy, cz = (x_min + x_max) / 2.0, (y_min + y_max) / 2.0, (z_min + z_max) / 2.0
    dx, dy, dz = x_max - x_min, y_max - y_min, z_max - z_min
    return torch.tensor([cx, cy, cz, dx, dy, dz], device=coords.device)


def transform_points(points: Tensor, transform: Tensor) -> Tensor:
    r"""Transform points using a $4 \times 4$ transformation matrix.

    This function applies a $4 \times 4$ transformation matrix to a set of points.
    The transformation matrix is assumed to be in the form:

    $$
    \begin{bmatrix}
    R & t \newline
    0 & 1
    \end{bmatrix}
    $$

    Where $R$ is a $3 \times 3$ rotation matrix and $t$ is a $3 \times 1$ translation vector.

    Args:
        points: Points of shape $(N, 3)$, in XYZ order.
        transform: A $4 \times 4$ transformation matrix.

    Returns:
        The transformed points of shape $(N, 3)$.

    Examples:
        ```pycon
        >>> points = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        >>> transform = torch.tensor([[1.0, 0.0, 0.0, 1.0], [0.0, 1.0, 0.0, 2.0], [0.0, 0.0, 1.0, 3.0], [0.0, 0.0, 0.0, 1.0]])
        >>> transform_points(points, transform)
        tensor([[2., 4., 6.],
                [5., 7., 9.]])

        ```
    """
    N, _ = points.shape

    # Convert to homogeneous coordinates (N, 4)
    homogeneous = torch.cat([points, torch.ones(N, 1, device=points.device, dtype=points.dtype)], dim=1)

    # Apply transformation (N, 4)
    transformed = torch.matmul(homogeneous, transform.T)

    # Return to 3D coordinates (N, 3)
    return transformed[:, :3]


def rotate(x: Tensor, R: Tensor) -> Tensor:
    r"""Apply a rotation matrix to a 3D tensor.

    Args:
        x: Input tensor of shape $(\ldots, 3)$.
        R: Rotation matrix of shape $(3, 3)$.

    Returns:
        Rotated tensor with the same shape as `x`.
    """
    return x @ R.to(x.dtype).to(x.device).T


def cross_product_matrix(k: Tensor) -> Tensor:
    r"""Constructs a skew-symmetric matrix (also known as a cross-product matrix)
    for a given 3D vector $k = [k_1, k_2, k_3]$. The function returns a
    $3 \times 3$ skew-symmetric matrix $M(k)$ of the form:

    $$
    M(k) = \begin{bmatrix}
    0 & -k_3 & k_2 \newline
    k_3 & 0 & -k_1 \newline
    -k_2 & k_1 & 0
    \end{bmatrix}
    $$

    Args:
        k: A tensor of shape $(3,)$ representing the 3D vector.

    Returns:
        A $3 \times 3$ skew-symmetric matrix corresponding to the cross-product operation.

    Examples:
        ```pycon
        >>> k = torch.tensor([1.0, 2.0, 3.0])
        >>> v = torch.tensor([4.0, 5.0, 6.0])
        >>> m = cross_product_matrix(k)
        >>> cross_product = torch.matmul(m, v)

        ```
    """

    m = [
        [0, -k[2], k[1]],
        [k[2], 0, -k[0]],
        [-k[1], k[0], 0],
    ]

    return torch.tensor(m, device=k.device)


def rodrigues_rotation_matrix(axis: Tensor, theta: float) -> Tensor:
    r"""Computes a 3D rotation matrix using Rodrigues' rotation formula.

    This function rotates a vector in 3D space around a specified axis by a
    given angle **in radians**. The rotation matrix is computed using:

    $$
    R = I + \sin(\theta)K + (1 - \cos(\theta))K^2
    $$

    Where:

    - $I$ is the identity matrix.
    - $K$ is the skew-symmetric matrix (cross-product matrix) derived from the axis of rotation.
    - $\theta$ is the rotation angle **in radians**.

    Args:
        axis: A 3D vector representing the axis of rotation.
        theta: The angle of rotation **in radians**.  Pass `math.radians(deg)`
            to convert from degrees.

    Returns:
        A $3 \times 3$ rotation matrix that rotates a vector around the specified axis by the specified angle.

    Raises:
        Warning: If $\theta > 2\pi$, which likely indicates degrees were passed
            instead of radians.
    """
    if abs(theta) > 2 * math.pi:
        warnings.warn(
            f"rodrigues_rotation_matrix received theta={theta:.4f} which is "
            f"outside [-2π, 2π]. Did you pass degrees instead of radians? "
            f"Use math.radians({theta:.4f}) to convert.",
            stacklevel=2,
        )

    axis = axis.detach().clone().float()
    axis = axis / axis.norm()
    K = cross_product_matrix(axis)
    t = torch.tensor([theta], device=axis.device)
    R = torch.eye(3, device=axis.device) + torch.sin(t) * K + (1 - torch.cos(t)) * K.mm(K)
    return R


def vertex_normals(vertices: Tensor, face: Tensor) -> Tensor:
    """Compute the vertex normal of a mesh.

    Args:
        vertices: The vertices of the mesh. Shape: $(V, 3)$.
        face: The face of the mesh. Shape: $(F, 3)$.

    Returns:
        The vertex normal of the mesh. Shape: $(V, 3)$.

    Examples:
        ```pycon
        >>> vertices = torch.tensor([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]])
        >>> face = torch.tensor([[0, 1, 2]])
        >>> vertex_normals(vertices, face)
        tensor([[0., 0., 1.],
                [0., 0., 1.],
                [0., 0., 1.]])

        ```
    """
    v01 = vertices[face[:, 1]] - vertices[face[:, 0]]  # (F, 3)
    v02 = vertices[face[:, 2]] - vertices[face[:, 0]]  # (F, 3)
    face_normals = torch.cross(v01, v02, dim=1)  # (F, 3)
    areas = face_normals.norm(dim=1, keepdim=True) * 0.5  # (F, 1)
    face_normals = safe_divide(face_normals, areas * 2, default=0.0)

    weighted_normals = face_normals * areas  # (F, 3)

    vertex_normals = torch.zeros_like(vertices)  # (V, 3)
    vertex_normals.index_add_(0, face[:, 0], weighted_normals)
    vertex_normals.index_add_(0, face[:, 1], weighted_normals)
    vertex_normals.index_add_(0, face[:, 2], weighted_normals)

    return torch.nn.functional.normalize(vertex_normals, dim=1)


def random_spherical_points(
    radius: float,
    num_points: int,
    bounds: Union[float, Tuple[float, float]] = 1.0,
) -> Tensor:
    """Generate random points inside a sphere or a spherical shell based on radius limits.

    Args:
        radius: The radius of the sphere.
        num_points: The number of points to generate.
        bounds: A float or tuple of floats defining the inner and outer bounds of the sphere.
            If a single float is provided, it is treated as the outer limit, with the inner limit as 0.
            Defaults to 1.0.

    Returns:
        Generated points of shape (num_points, dimension).
    """

    if isinstance(bounds, (int, float)):
        bounds = (0.0, float(bounds))

    inner_bound, outer_bound = bounds

    points = torch.zeros((0, 3))  # Initialize an empty tensor for points
    while points.shape[0] < num_points:
        # Generate random points in the bounding cube
        new_points = (torch.rand(num_points, 3) * 2 * radius) - radius
        d2 = torch.sum(new_points**2, dim=1)

        # Filter points that fall within the spherical shell or full sphere as per the given range
        valid_points = new_points[(d2 < (outer_bound * radius) ** 2) & (d2 > (inner_bound * radius) ** 2)]
        points = torch.cat((points, valid_points), dim=0)

    points = points[:num_points]  # Ensure the exact number of points

    return points


@overload
def spherical_points_gradient(
    radius: float,
    num_points: int,
    fixed_position: Literal["none", "center", "vertical"] = "center",
    ratio: float = 0.66,
    max_steps: int = 10_000,
    step_size: float = 1e-2,
    step_decay: float = 0.9995,
    convergence_threshold: float = 1e-5,
    max_step_size: Optional[float] = None,
    return_grad_norms: Literal[False] = False,
) -> Tensor: ...


@overload
def spherical_points_gradient(
    radius: float,
    num_points: int,
    fixed_position: Literal["none", "center", "vertical"] = "center",
    ratio: float = 0.66,
    max_steps: int = 10_000,
    step_size: float = 1e-2,
    step_decay: float = 0.9995,
    convergence_threshold: float = 1e-5,
    max_step_size: Optional[float] = None,
    return_grad_norms: Literal[True] = True,
) -> Tuple[Tensor, Tensor]: ...


def spherical_points_gradient(
    radius: float,
    num_points: int,
    fixed_position: Literal["none", "center", "vertical"] = "center",
    ratio: float = 0.66,
    max_steps: int = 10_000,
    step_size: float = 1e-2,
    step_decay: float = 0.9995,
    convergence_threshold: float = 1e-5,
    max_step_size: Optional[float] = None,
    return_grad_norms: bool = False,
) -> Union[Tensor, Tuple[Tensor, Tensor]]:
    r"""Creation of kernel points via optimization of potentials for a single kernel.

    Args:
        radius: Radius of the kernel.
        num_points: Number of points composing the kernel.
        fixed_position: Fix position of certain kernel points ('none', 'center', or 'vertical').
        ratio: Ratio of the radius where you want the kernel points to be placed.
        max_steps: Maximum number of optimization steps.
        step_size: Step size for moving points based on gradient norms.
        step_decay: Decay factor for reducing the step size over time.
        convergence_threshold: Threshold for stopping the optimization when gradient norm changes are small.
        max_step_size: Maximum distance a point can move in a single step.
        return_grad_norms: Whether to also return the gradient norms recorded during the optimization.

    Returns:
        Optimized kernel points of shape $(\text{num\_points}, 3)$, or a tuple of the kernel points and
            the recorded gradient norms when `return_grad_norms=True`.
    """

    def compute_gradients(points: Tensor) -> Tensor:
        A = points.unsqueeze(1)
        B = points.unsqueeze(0)
        inter_d2 = torch.sum((A - B) ** 2, dim=-1)
        inter_grads = (A - B) / (inter_d2.unsqueeze(-1) ** (3 / 2) + 1e-6)
        inter_grads = torch.sum(inter_grads, dim=0)
        circle_grads = 10 * points
        return inter_grads + circle_grads

    # Parameters
    if fixed_position not in ["none", "center", "vertical"]:
        raise ValueError('Unsupported fixed position. Expected "none", "center", or "vertical".')

    if max_step_size is None:
        max_step_size = 0.05 * radius

    # Initialize kernel points (bounds: [0, 1/√2])
    kernel_points = random_spherical_points(radius, num_points, bounds=(0, 0.7071067811865476))

    # Apply fixed positions if required
    if fixed_position == "center":
        kernel_points[0, :] = 0  # Fix the first point to the center
    elif fixed_position == "vertical":
        kernel_points[:3, :] = 0  # Fix the first three points
        kernel_points[1, -1] += 2 * radius / 3  # Move second point up vertically
        kernel_points[2, -1] -= 2 * radius / 3  # Move third point down vertically

    # Kernel optimization
    saved_grad_norms = torch.zeros(max_steps)
    prev_grad_norms = torch.zeros(num_points)
    step = 0

    while step < max_steps:
        grads = compute_gradients(kernel_points)

        if fixed_position == "vertical":
            grads[1:3, :-1] = 0

        grad_norms = torch.sqrt(torch.sum(grads**2, dim=-1))
        saved_grad_norms[step] = torch.max(grad_norms)

        # Check for stopping conditions
        if (
            fixed_position == "center"
            and torch.max(torch.abs(prev_grad_norms[1:] - grad_norms[1:])) < convergence_threshold
        ):
            break
        elif (
            fixed_position == "vertical"
            and torch.max(torch.abs(prev_grad_norms[3:] - grad_norms[3:])) < convergence_threshold
        ):
            break
        elif torch.max(torch.abs(prev_grad_norms - grad_norms)) < convergence_threshold:
            break

        prev_grad_norms = grad_norms

        # Move points
        moving_dists = torch.minimum(step_size * grad_norms, torch.tensor(max_step_size))
        if fixed_position == "center" or fixed_position == "vertical":
            moving_dists[0] = 0  # Do not move the first point if fixed

        kernel_points -= moving_dists.unsqueeze(-1) * grads / (grad_norms.unsqueeze(-1) + 1e-6)
        step_size *= step_decay
        step += 1

    # Rescale kernel points
    r = torch.sqrt(torch.sum(kernel_points**2, dim=-1))
    kernel_points *= radius * ratio / torch.mean(r[1:])

    if return_grad_norms:
        return kernel_points, saved_grad_norms[:step]
    return kernel_points


def spherical_points_lloyd(
    radius: float,
    num_points: int,
    fixed_position: Literal["none", "center", "vertical"] = "none",
    approximation: Literal["discretization", "monte-carlo"] = "discretization",
    approx_n: int = 5000,
    max_iter: int = 500,
    momentum: float = 0.9,
) -> Tensor:
    r"""Generate kernel points using Lloyd's algorithm on a sphere.

    Args:
        radius: Radius of the sphere.
        num_points: Number of kernel points (Voronoi cells).
        fixed_position (str, optional): Fix the position of specific kernel points. Defaults to 'none'.
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
        Tensor: Tensor of shape [num_points, dimension] with the final kernel points on the sphere.
    """

    if fixed_position not in ["none", "center", "vertical"]:
        raise ValueError('Unsupported fixed position. Expected "none", "center", or "vertical".')
    if approximation not in ["discretization", "monte-carlo"]:
        raise ValueError('Unsupported approximation. Expected "discretization" or "monte-carlo".')

    # Constants
    radius0 = 1.0  # Initial radius for optimization

    # Generate random kernel points inside the sphere
    kernel_points = torch.empty((0, 3))
    while kernel_points.size(0) < num_points:
        new_points = (torch.rand(num_points, 3) * 2 - 1) * radius0
        valid_points = new_points[(new_points.norm(dim=1) < radius0) & (new_points.norm(dim=1) > (0.9 * radius0))]
        kernel_points = torch.cat((kernel_points, valid_points), dim=0)

    kernel_points = kernel_points[:num_points]

    # Optional fixing of kernel positions
    if fixed_position == "center":
        kernel_points[0] = torch.zeros(3)
    elif fixed_position == "vertical":
        kernel_points[:3, :] = torch.zeros(3, 3)
        kernel_points[1, -1] += 2 * radius0 / 3
        kernel_points[2, -1] -= 2 * radius0 / 3

    # Initialize the approximation points
    if approximation == "discretization":
        side_n = int(approx_n ** (1.0 / 3))
        coords = torch.linspace(-radius0, radius0, side_n, device=kernel_points.device)
        mesh = torch.meshgrid([coords] * 3, indexing="ij")
        X = torch.stack([m.flatten() for m in mesh], dim=-1)
    else:
        X = torch.empty((0, 3), device=kernel_points.device)

    # Only keep points inside the sphere
    X = X[X.norm(dim=1) < radius0]

    # Tensor for tracking the maximum moves
    max_moves = torch.empty((0,), device=kernel_points.device)

    # Lloyd's algorithm iterations
    for _ in range(max_iter):
        if approximation == "monte-carlo":
            X = torch.empty(approx_n, 3, device=kernel_points.device).uniform_(-radius0, radius0)
            X = X[X.norm(dim=1) < radius0]

        # Compute distances and assign points to nearest kernel
        diff = X.unsqueeze(1) - kernel_points.unsqueeze(0)
        dist2 = torch.sum(diff**2, dim=-1)
        nearest_kernel_idx = torch.argmin(dist2, dim=-1)

        # Calculate new kernel positions (cell centers)
        new_kernel_point_list = []
        for c in range(num_points):
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
        if fixed_position == "center":
            kernel_points[0] = torch.zeros(3)
        elif fixed_position == "vertical":
            kernel_points[0] = torch.zeros(3)
            kernel_points[1:3, :-1] = torch.zeros_like(kernel_points[1:3, :-1])

    return kernel_points * radius
