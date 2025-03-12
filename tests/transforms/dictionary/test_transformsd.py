from unittest.mock import Mock, patch, sentinel

from torch_pointcloud.transforms.dictionary import NormalizeScaled, RandomSampled, RandomSampleVerticesd


@patch("torch_pointcloud.transforms.dictionary.transforms.F.random_sampled")
def test_random_sample_dict_transform(mock_fn: Mock) -> None:
    """Test that RandomSampled transform calls the functional API correctly."""
    data = sentinel.data
    num_samples = sentinel.num_samples
    keys = (sentinel.key,)
    allow_missing_keys = sentinel.allow_missing_keys
    seed = sentinel.seed

    transform = RandomSampled(
        num_samples=num_samples,
        keys=keys,
        allow_missing_keys=allow_missing_keys,
        seed=seed,
    )

    result = transform(data)

    mock_fn.assert_called_once_with(
        data,
        keys=keys,
        num_samples=num_samples,
        seed=seed,
        allow_missing_keys=allow_missing_keys,
    )
    assert result is mock_fn.return_value


@patch("torch_pointcloud.transforms.dictionary.transforms.F.random_sample_verticesd")
def test_random_sample_vertices_dict_transform(mock_fn: Mock) -> None:
    """Test that RandomSampleVerticesd transform calls the functional API correctly."""
    data = sentinel.data
    num_samples = sentinel.num_samples
    keys = (sentinel.key,)
    face_keys = (sentinel.face_key,)
    include_normals = sentinel.include_normals
    allow_missing_keys = sentinel.allow_missing_keys
    seed = sentinel.seed

    transform = RandomSampleVerticesd(
        num_samples=num_samples,
        keys=keys,
        face_keys=face_keys,
        include_normals=include_normals,
        allow_missing_keys=allow_missing_keys,
        seed=seed,
    )

    result = transform(data)

    mock_fn.assert_called_once_with(
        data,
        keys=keys,
        face_keys=face_keys,
        num_samples=num_samples,
        include_normals=include_normals,
        seed=seed,
        allow_missing_keys=allow_missing_keys,
    )
    assert result is mock_fn.return_value


@patch("torch_pointcloud.transforms.dictionary.transforms.F.normalize_scaled")
def test_normalize_scale_dict_transform(mock_fn: Mock) -> None:
    """Test that NormalizeScaled transform calls the functional API correctly."""
    data = sentinel.data
    keys = (sentinel.key,)
    allow_missing_keys = sentinel.allow_missing_keys

    transform = NormalizeScaled(
        keys=keys,
        allow_missing_keys=allow_missing_keys,
    )

    result = transform(data)

    mock_fn.assert_called_once_with(
        data,
        keys=keys,
        allow_missing_keys=allow_missing_keys,
    )
    assert result is mock_fn.return_value
