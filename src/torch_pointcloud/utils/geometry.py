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
    r"""Transform points using a 4x4 transformation matrix.

    This function applies a 4x4 transformation matrix to a set of points.
    The transformation matrix is assumed to be in the form:

    $$
    \begin{bmatrix}
    R & t \newline
    0 & 1
    \end{bmatrix}
    $$

    Where $R$ is a 3x3 rotation matrix and $t$ is a 3x1 translation vector.

    Args:
        points: Points of shape $(N, 3)$, in XYZ order.
        transform: A 4x4 transformation matrix.

    Returns:
        The transformed points of shape $(N, 3)$.

    Examples:
        >>> points = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        >>> transform = torch.tensor([[1.0, 0.0, 0.0, 1.0], [0.0, 1.0, 0.0, 2.0], [0.0, 0.0, 1.0, 3.0], [0.0, 0.0, 0.0, 1.0]])
        >>> transform_points(points, transform)
        tensor([[2.0, 4.0, 6.0],
                [5.0, 7.0, 9.0]])
    """
    N, _ = points.shape

    # Convert to homogeneous coordinates (N, 4)
    homogeneous = torch.cat([points, torch.ones(N, 1, device=points.device, dtype=points.dtype)], dim=1)

    # Apply transformation (N, 4)
    transformed = torch.matmul(homogeneous, transform.T)

    # Return to 3D coordinates (N, 3)
    return transformed[:, :3]


def cross_product_matrix(k: Tensor) -> Tensor:
    r"""Constructs a skew-symmetric matrix (also known as a cross-product matrix)
    for a given 3D vector $k = [k1, k2, k3]$. The function returns a
    3x3 skew-symmetric matrix $M(k)$ of the form:

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
        A 3x3 skew-symmetric matrix corresponding to the cross-product operation.

    Examples:
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


def rodrigues_rotation_matrix(axis: Tensor, theta: float) -> Tensor:
    r"""Computes a 3D rotation matrix using Rodrigues' rotation formula.

    This function rotates a vector in 3D space around a specified axis by a
    given angle. The rotation matrix is computed using the following formula:

    $$
    R = I + \sin(\theta)K + (1 - \cos(\theta))K^2
    $$

    Where:

    - $I$ is the identity matrix.
    - $K$ is the skew-symmetric matrix (cross-product matrix) derived from the axis of rotation.
    - $\theta$ is the rotation angle in radians.

    Args:
        axis: A 3D vector representing the axis of rotation.
        theta: The angle of rotation in radians.

    Returns:
        A 3x3 rotation matrix that rotates a vector around the specified axis by the specified angle.
    """
    axis = axis.detach().clone().float()
    axis = axis / axis.norm()
    K = cross_product_matrix(axis)
    t = torch.tensor([theta], device=axis.device)
    R = torch.eye(3, device=axis.device) + torch.sin(t) * K + (1 - torch.cos(t)) * K.mm(K)
    return R


def vertex_normals(vertices: Tensor, faces: Tensor) -> Tensor:
    """Compute the vertex normals of a mesh.

    Args:
        vertices: The vertices of the mesh. Shape: $(V, 3)$.
        faces: The faces of the mesh. Shape: $(F, 3)$.

    Returns:
        The vertex normals of the mesh. Shape: $(V, 3)$.

    Examples:
        >>> vertices = torch.tensor([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]])
        >>> faces = torch.tensor([[0, 1, 2]])
        >>> vertex_normals(vertices, faces)
        tensor([[0., 0., 1.],
                [0., 0., 1.],
                [0., 0., 1.]])
    """
    v01 = vertices[faces[:, 1]] - vertices[faces[:, 0]]  # (F, 3)
    v02 = vertices[faces[:, 2]] - vertices[faces[:, 0]]  # (F, 3)
    face_normals = torch.cross(v01, v02, dim=1)  # (F, 3)
    areas = face_normals.norm(dim=1, keepdim=True) * 0.5  # (F, 1)
    face_normals = safe_divide(face_normals, areas * 2, default=0.0)

    weighted_normals = face_normals * areas  # (F, 3)

    vertex_normals = torch.zeros_like(vertices)  # (V, 3)
    vertex_normals.index_add_(0, faces[:, 0], weighted_normals)
    vertex_normals.index_add_(0, faces[:, 1], weighted_normals)
    vertex_normals.index_add_(0, faces[:, 2], weighted_normals)

    return torch.nn.functional.normalize(vertex_normals, dim=1)
