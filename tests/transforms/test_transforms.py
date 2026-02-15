from unittest.mock import MagicMock, Mock, patch, sentinel

import torch

from torch_pointcloud.transforms import NormalizeScale, RandomSample, RandomSampleFaceVertices
from torch_pointcloud.transforms.transforms import (
    Abs,
    ApplyMask,
    BoundingBox,
    Compose,
    InboxMask,
    RemoveNearOrigin,
    SampleFarthestPoints,
)


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


@patch("torch_pointcloud.transforms.functional.random_sample_face_vertices")
def test_random_sample_face_vertices_transform(mock_fn: Mock) -> None:
    """Test that RandomSampleFaceVertices transform calls the functional API correctly."""
    vertices = sentinel.vertices
    faces = sentinel.faces
    num_samples = sentinel.num_samples
    return_normals = sentinel.return_normals
    seed = sentinel.seed

    transform = RandomSampleFaceVertices(num_samples=num_samples, return_normals=return_normals, seed=seed)
    result = transform(vertices, faces)

    mock_fn.assert_called_once_with(vertices, faces, num_samples=num_samples, return_normals=return_normals, seed=seed)

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


def test_compose_applies_transforms_in_order() -> None:
    """Test that Compose applies transforms sequentially."""
    t1 = MagicMock()
    t2 = MagicMock()
    t1.return_value = sentinel.after_t1
    t2.return_value = sentinel.after_t2

    compose = Compose([t1, t2])
    result = compose(sentinel.data)

    t1.assert_called_once_with(sentinel.data)
    t2.assert_called_once_with(sentinel.after_t1)
    assert result is sentinel.after_t2


def test_compose_single_transform() -> None:
    """Test Compose with a single transform."""
    t1 = MagicMock()
    t1.return_value = sentinel.result

    compose = Compose([t1])
    result = compose(sentinel.data)

    t1.assert_called_once_with(sentinel.data)
    assert result is sentinel.result


def test_compose_with_list_input() -> None:
    """Test Compose applies transforms element-wise when data is a list."""
    t1 = MagicMock(side_effect=lambda x: x * 2)
    compose = Compose([t1])

    data = [torch.tensor([1.0]), torch.tensor([2.0])]
    _ = compose(data)

    assert t1.call_count == 2


def test_compose_repr() -> None:
    """Test Compose repr contains child transforms."""
    t1 = Abs()
    t2 = NormalizeScale(eps=1e-6)
    compose = Compose([t1, t2])
    repr_str = repr(compose)
    assert "Compose" in repr_str
    assert "Abs" in repr_str
    assert "NormalizeScale" in repr_str


@patch("torch_pointcloud.transforms.functional.sample_farthest_points")
def test_sample_farthest_points_transform_num_samples(mock_fn: Mock) -> None:
    """Test that SampleFarthestPoints delegates to functional API with num_samples."""
    pos = sentinel.pos
    num_samples = 10

    transform = SampleFarthestPoints(num_samples=num_samples)
    result = transform(pos)

    mock_fn.assert_called_once_with(pos, num_samples=num_samples, ratio=None, random_start=False)
    assert result is mock_fn.return_value


@patch("torch_pointcloud.transforms.functional.sample_farthest_points")
def test_sample_farthest_points_transform_ratio(mock_fn: Mock) -> None:
    """Test that SampleFarthestPoints delegates to functional API with ratio."""
    pos = sentinel.pos
    ratio = 0.5

    transform = SampleFarthestPoints(ratio=ratio)
    result = transform(pos)

    mock_fn.assert_called_once_with(pos, num_samples=None, ratio=ratio, random_start=False)
    assert result is mock_fn.return_value


@patch("torch_pointcloud.transforms.functional.sample_farthest_points")
def test_sample_farthest_points_transform_random_start(mock_fn: Mock) -> None:
    """Test that SampleFarthestPoints delegates to functional API with random_start."""
    pos = sentinel.pos

    transform = SampleFarthestPoints(num_samples=5, random_start=True)
    result = transform(pos)

    mock_fn.assert_called_once_with(pos, num_samples=5, ratio=None, random_start=True)
    assert result is mock_fn.return_value


@patch("torch_pointcloud.transforms.functional.remove_near_origin")
def test_remove_near_origin_transform(mock_fn: Mock) -> None:
    """Test that RemoveNearOrigin delegates to functional API."""
    pos = sentinel.pos
    radius = 0.01

    transform = RemoveNearOrigin(radius=radius)
    result = transform(pos)

    mock_fn.assert_called_once_with(pos, radius=radius, return_mask=False)
    assert result is mock_fn.return_value


@patch("torch_pointcloud.transforms.functional.remove_near_origin")
def test_remove_near_origin_transform_with_mask(mock_fn: Mock) -> None:
    """Test that RemoveNearOrigin delegates to functional API with return_mask."""
    pos = sentinel.pos
    radius = sentinel.radius

    mock_fn.return_value = (sentinel.filtered, sentinel.mask)

    transform = RemoveNearOrigin(radius=radius)
    result = transform(pos, return_mask=True)

    mock_fn.assert_called_once_with(pos, radius=radius, return_mask=True)
    assert result is mock_fn.return_value


@patch("torch_pointcloud.transforms.functional.remove_near_origin")
def test_remove_near_origin_transform_default_radius(mock_fn: Mock) -> None:
    """Test that RemoveNearOrigin uses default radius of 1e-3."""
    pos = sentinel.pos

    transform = RemoveNearOrigin()
    _ = transform(pos)

    mock_fn.assert_called_once_with(pos, radius=1e-3, return_mask=False)


@patch("torch_pointcloud.transforms.functional.abs")
def test_abs_transform_default(mock_fn: Mock) -> None:
    """Test that Abs transform delegates to functional API with default inplace=False."""
    tensor = sentinel.tensor

    transform = Abs()
    result = transform(tensor)

    mock_fn.assert_called_once_with(tensor, inplace=False)
    assert result is mock_fn.return_value


@patch("torch_pointcloud.transforms.functional.abs")
def test_abs_transform_inplace(mock_fn: Mock) -> None:
    """Test that Abs transform delegates to functional API with inplace=True."""
    tensor = sentinel.tensor

    transform = Abs(inplace=True)
    result = transform(tensor)

    mock_fn.assert_called_once_with(tensor, inplace=True)
    assert result is mock_fn.return_value


@patch("torch_pointcloud.transforms.functional.bounding_box")
def test_bounding_box_transform_default_dim(mock_fn: Mock) -> None:
    """Test that BoundingBox transform delegates with default dim=-1."""
    tensor = sentinel.tensor

    transform = BoundingBox()
    result = transform(tensor)

    mock_fn.assert_called_once_with(tensor, dim=-1)
    assert result is mock_fn.return_value


@patch("torch_pointcloud.transforms.functional.bounding_box")
def test_bounding_box_transform_custom_dim(mock_fn: Mock) -> None:
    """Test that BoundingBox transform delegates with custom dim=0."""
    tensor = sentinel.tensor

    transform = BoundingBox(dim=0)
    result = transform(tensor)

    mock_fn.assert_called_once_with(tensor, dim=0)
    assert result is mock_fn.return_value


@patch("torch_pointcloud.transforms.functional.inbox_mask")
def test_inbox_mask_transform_default_dim(mock_fn: Mock) -> None:
    """Test that InboxMask transform delegates with default dim=-1."""
    tensor = sentinel.tensor
    bbox = sentinel.bbox

    transform = InboxMask()
    result = transform(tensor, bbox)

    mock_fn.assert_called_once_with(tensor, bbox, dim=-1)
    assert result is mock_fn.return_value


@patch("torch_pointcloud.transforms.functional.inbox_mask")
def test_inbox_mask_transform_custom_dim(mock_fn: Mock) -> None:
    """Test that InboxMask transform delegates with custom dim=0."""
    tensor = sentinel.tensor
    bbox = sentinel.bbox

    transform = InboxMask(dim=0)
    result = transform(tensor, bbox)

    mock_fn.assert_called_once_with(tensor, bbox, dim=0)
    assert result is mock_fn.return_value


@patch("torch_pointcloud.transforms.functional.apply_mask")
def test_apply_mask_transform(mock_fn: Mock) -> None:
    """Test that ApplyMask transform delegates to functional API."""
    tensor = sentinel.tensor
    mask = sentinel.mask

    transform = ApplyMask(mask=mask)
    result = transform(tensor)

    mock_fn.assert_called_once_with(tensor, mask)
    assert result is mock_fn.return_value
