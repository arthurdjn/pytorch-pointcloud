from typing import Literal, Optional, Tuple, Union, overload

import torch
from torch import Tensor


@overload
def random_sample(
    tensor: Tensor,
    num_samples: int,
    return_indices: Literal[True] = True,
    seed: Optional[int] = None,
) -> Tensor: ...


@overload
def random_sample(
    tensor: Tensor,
    num_samples: int,
    return_indices: bool = False,
    seed: Optional[int] = None,
) -> Tensor: ...


def random_sample(
    tensor: Tensor,
    num_samples: int,
    return_indices: bool = False,
    seed: Optional[int] = None,
) -> Union[Tensor, Tuple[Tensor, Tensor]]:
    """Randomly sample a fixed number of values from a tensor.

    Note:
        The data is sampled uniformly from the tensor.

    Args:
        tensor: The input tensor.
        num_samples: The number of values to sample.
        return_indices: Whether to return the indices of the sampled values.
        seed: The seed for the random number generator.

    Returns:
        If `return_indices` is `True`, the function returns a tuple of the sampled values and their indices.
        Otherwise, it returns the sampled values.
    """
    rng = None
    if seed is not None:
        rng = torch.Generator(device=tensor.device)
        rng.manual_seed(seed)

    indices = torch.randint(0, tensor.size(0), (num_samples,), generator=rng)

    if return_indices:
        return tensor[indices], indices
    return tensor[indices]


@overload
def random_sample_vertices(
    vertices: Tensor,
    faces: Tensor,
    num_samples: int,
    return_normals: Literal[True] = True,
    return_indices: Literal[True] = True,
    seed: Optional[int] = None,
) -> Tuple[Tensor, Tensor, Tensor]: ...


@overload
def random_sample_vertices(
    vertices: Tensor,
    faces: Tensor,
    num_samples: int,
    return_normals: Literal[True] = True,
    return_indices: bool = False,
    seed: Optional[int] = None,
) -> Tuple[Tensor, Tensor]: ...


@overload
def random_sample_vertices(
    vertices: Tensor,
    faces: Tensor,
    num_samples: int,
    return_normals: bool = False,
    return_indices: bool = False,
    seed: Optional[int] = None,
) -> Tensor: ...


def random_sample_vertices(
    vertices: Tensor,
    faces: Tensor,
    num_samples: int,
    return_normals: bool = False,
    return_indices: bool = False,
    seed: Optional[int] = None,
) -> Union[Tensor, Tuple[Tensor, Tensor], Tuple[Tensor, Tensor, Tensor]]:
    """Randomly sample a fixed number of vertices from a 3D mesh (vertices, faces),
    using:

    Note:
        The data is sampled uniformly from the mesh.

    Args:
        vertices: The input tensor.
        faces: The input tensor.
        num_samples: The number of vertices to sample.
        return_normals: Whether to return the normals of the sampled vertices.
        return_indices: Whether to return the indices of the sampled vertices.
        seed: The seed for the random number generator.

    Returns:
        If `return_indices` is `True`, the function returns a tuple of the sampled vertices and their indices.
        Otherwise, it returns the sampled vertices.
        If `return_normals` is `True`, the function returns a tuple of the sampled vertices and their normals.
        Otherwise, it returns the sampled vertices.
    """
    rng = torch.Generator()
    if seed is not None:
        rng.manual_seed(seed)

    pos_max = vertices.abs().max()
    vertices = vertices / pos_max

    v01 = vertices[faces[:, 1]] - vertices[faces[:, 0]]
    v02 = vertices[faces[:, 2]] - vertices[faces[:, 0]]
    areas = v01.cross(v02, dim=1)
    areas = areas.norm(p=2, dim=1).abs() / 2

    probs = areas / areas.sum()
    samples = torch.multinomial(probs, num_samples, replacement=True, generator=rng)
    faces = faces[samples]

    frac = torch.rand(num_samples, 2, device=vertices.device, generator=rng)
    mask = frac.sum(dim=-1) > 1
    frac[mask] = 1 - frac[mask]

    v01 = vertices[faces[:, 1]] - vertices[faces[:, 0]]
    v02 = vertices[faces[:, 2]] - vertices[faces[:, 0]]

    if return_normals:
        normals = torch.nn.functional.normalize(v01.cross(v02, dim=1), p=2)

    vertices = vertices[faces[:, 0]]
    vertices += frac[:, :1] * v01
    vertices += frac[:, 1:] * v02
    vertices = vertices * pos_max

    if return_indices and return_normals:
        return vertices, normals, faces[:, 0]
    elif return_normals:
        return vertices, normals
    elif return_indices:
        return vertices, faces[:, 0]
    return vertices


def normalize_scale(points: Tensor, eps: float = 1e-8) -> Tensor:
    r"""Normalize the scale of a 3D tensor as follows:

    $$
    \mathbf{x} = \frac{\mathbf{x} - \mathbf{\mu}}{\max(\sqrt{\sum_{i=1}^3 x_i^2}, \epsilon)}
    $$

    Note:
        The data is normalized to have a unit scale.

    Args:
        points: The input tensor.
        eps: The epsilon value to avoid division by zero.

    Returns:
        The normalized tensor.
    """
    points -= points.mean(dim=-2, keepdim=True)
    points = points / (points.abs().max() + eps)
    return points
