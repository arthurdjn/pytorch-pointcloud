from unittest.mock import MagicMock, Mock, patch, sentinel

from torch_pointcloud.transforms.dictionary.functional import (
    normalize_scaled,
    random_sample_face_verticesd,
    random_sampled,
)


@patch("torch_pointcloud.transforms.dictionary.functional.F.random_sample")
def test_random_sampled(mock_fn: Mock) -> None:
    """Test that random_sampled correctly applies random_sample to specified keys."""
    data = {
        "points": MagicMock(),
        "normals": MagicMock(),
        "other": MagicMock(),
    }
    num_samples = sentinel.num_samples
    allow_missing_keys = sentinel.allow_missing_keys
    seed = sentinel.seed
    sampled_tensor = sentinel.sampled_tensor
    sampled_indices = sentinel.indices
    mock_fn.return_value = (sampled_tensor, sampled_indices)

    keys = ["points", "normals"]

    result = random_sampled(
        data,
        keys,
        num_samples=num_samples,
        seed=seed,
        allow_missing_keys=allow_missing_keys,
    )

    mock_fn.assert_called_once_with(
        data["points"],
        num_samples,
        seed=seed,
        return_indices=True,
    )

    assert result["points"] is sampled_tensor
    assert result["normals"] is data["normals"][sampled_indices]
    # Check that non-specified keys are unchanged
    assert result["other"] is data["other"]


@patch("torch_pointcloud.transforms.dictionary.functional.F.random_sample_face_vertices")
def test_random_sample_face_verticesd(mock_fn: Mock) -> None:
    """Test that random_sample_face_verticesd correctly processes mesh data."""
    data = {
        "vertices": MagicMock(),
        "colors": MagicMock(),
        "faces": MagicMock(),
        "other": MagicMock(),
    }
    num_samples = sentinel.num_samples
    include_normals = sentinel.include_normals
    allow_missing_keys = sentinel.allow_missing_keys
    seed = sentinel.seed
    sampled_vertices = sentinel.sampled_vertices
    sampled_normals = sentinel.sampled_normals
    sampled_indices = sentinel.sampled_indices
    mock_fn.return_value = (sampled_vertices, sampled_normals, sampled_indices)

    result = random_sample_face_verticesd(
        data,
        keys=["vertices", "colors"],
        face_keys=["faces"],
        num_samples=num_samples,
        include_normals=include_normals,
        seed=seed,
        allow_missing_keys=allow_missing_keys,
    )

    mock_fn.assert_called_once_with(
        data["vertices"],
        data["faces"],
        num_samples,
        return_normals=include_normals,
        return_indices=True,
        seed=seed,
    )

    assert result["vertices"] is sampled_vertices
    assert result["normals"] is sampled_normals
    assert result["colors"] is data["colors"][sampled_indices]
    assert result["other"] is data["other"]


@patch("torch_pointcloud.transforms.dictionary.functional.F.normalize_scale")
def test_normalize_scaled(mock_fn: Mock) -> None:
    """Test that normalize_scaled correctly normalizes the scale of specified keys."""
    data = {
        "points": MagicMock(),
        "other": MagicMock(),
    }
    keys = ["points"]
    allow_missing_keys = sentinel.allow_missing_keys
    normalized_tensor = sentinel.normalized_tensor
    mock_fn.return_value = normalized_tensor

    result = normalize_scaled(data, keys, allow_missing_keys=allow_missing_keys)

    mock_fn.assert_called_once_with(data["points"])

    assert result["points"] is normalized_tensor
    assert result["other"] is data["other"]
