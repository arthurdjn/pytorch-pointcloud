from unittest.mock import Mock, patch, sentinel

from torch_pointcloud.transforms import NormalizeScale, RandomSample, RandomSampleVertices


@patch("torch_pointcloud.transforms.functional.random_sample")
def test_random_sample_transform(mock_fn: Mock) -> None:
    """Test that RandomSample transform calls the functional API correctly."""
    tensor = sentinel.tensor
    num_samples = sentinel.num_samples
    return_indices = sentinel.return_indices
    seed = sentinel.seed
    transform = RandomSample(num_samples=num_samples, return_indices=return_indices, seed=seed)

    result = transform(tensor)

    mock_fn.assert_called_once_with(
        tensor,
        num_samples=num_samples,
        return_indices=return_indices,
        seed=seed,
    )

    assert result is mock_fn.return_value


@patch("torch_pointcloud.transforms.functional.random_sample_vertices")
def test_random_sample_vertices_transform(mock_fn: Mock) -> None:
    """Test that RandomSampleVertices transform calls the functional API correctly."""
    vertices = sentinel.vertices
    faces = sentinel.faces
    num_samples = sentinel.num_samples
    return_normals = sentinel.return_normals
    return_indices = sentinel.return_indices
    seed = sentinel.seed

    transform = RandomSampleVertices(
        num_samples=num_samples,
        return_normals=return_normals,
        return_indices=return_indices,
        seed=seed,
    )

    result = transform(vertices, faces)

    mock_fn.assert_called_once_with(
        vertices,
        faces,
        num_samples=num_samples,
        return_normals=return_normals,
        return_indices=return_indices,
        seed=seed,
    )

    assert result is mock_fn.return_value


@patch("torch_pointcloud.transforms.functional.normalize_scale")
def test_normalize_scale_transform(mock_fn: Mock) -> None:
    """Test that NormalizeScale transform calls the functional API correctly."""
    tensor = sentinel.tensor
    eps = sentinel.eps
    transform = NormalizeScale(eps=eps)

    result = transform(tensor)

    mock_fn.assert_called_once_with(tensor, eps=eps)
    assert result is mock_fn.return_value
