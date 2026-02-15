from unittest.mock import Mock, patch, sentinel

from torch_pointcloud.transforms.dictionary import (
    Absd,
    ApplyMaskd,
    BoundingBoxd,
    InboxMaskd,
    NormalizeScaled,
    RandomSampled,
    RandomSampleFaceVerticesd,
    RemoveNearOrigind,
    SampleFarthestPointsd,
)


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


@patch("torch_pointcloud.transforms.dictionary.transforms.F.random_sample_face_verticesd")
def test_random_sample_face_vertices_dict_transform(mock_fn: Mock) -> None:
    """Test that RandomSampleFaceVerticesd transform calls the functional API correctly."""
    data = sentinel.data
    num_samples = sentinel.num_samples
    keys = (sentinel.key,)
    face_keys = (sentinel.face_key,)
    include_normals = sentinel.include_normals
    normals_key = sentinel.normals_key
    allow_missing_keys = sentinel.allow_missing_keys
    seed = sentinel.seed

    transform = RandomSampleFaceVerticesd(
        num_samples=num_samples,
        keys=keys,
        face_keys=face_keys,
        include_normals=include_normals,
        normals_key=normals_key,
        seed=seed,
        allow_missing_keys=allow_missing_keys,
    )

    result = transform(data)

    mock_fn.assert_called_once_with(
        data,
        keys=keys,
        face_keys=face_keys,
        num_samples=num_samples,
        include_normals=include_normals,
        normals_key=normals_key,
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


@patch("torch_pointcloud.transforms.dictionary.transforms.F.sample_farthest_pointsd")
def test_sample_farthest_pointsd_transform(mock_fn: Mock) -> None:
    """Test that SampleFarthestPointsd delegates to functional API correctly."""
    data = sentinel.data
    pos_key = "pos"
    keys = ("labels",)
    num_samples = 10
    allow_missing_keys = sentinel.allow_missing_keys

    transform = SampleFarthestPointsd(
        pos_key=pos_key,
        keys=keys,
        num_samples=num_samples,
        allow_missing_keys=allow_missing_keys,
    )

    result = transform(data)

    mock_fn.assert_called_once_with(
        data,
        pos_key=pos_key,
        keys=keys,
        num_samples=num_samples,
        ratio=None,
        random_start=False,
        allow_missing_keys=allow_missing_keys,
    )
    assert result is mock_fn.return_value


@patch("torch_pointcloud.transforms.dictionary.transforms.F.sample_farthest_pointsd")
def test_sample_farthest_pointsd_transform_with_ratio(mock_fn: Mock) -> None:
    """Test SampleFarthestPointsd with ratio parameter."""
    data = sentinel.data

    transform = SampleFarthestPointsd(pos_key="pos", ratio=0.5)
    result = transform(data)

    mock_fn.assert_called_once_with(
        data,
        pos_key="pos",
        keys=(),
        num_samples=None,
        ratio=0.5,
        random_start=False,
        allow_missing_keys=False,
    )
    assert result is mock_fn.return_value


@patch("torch_pointcloud.transforms.dictionary.transforms.F.sample_farthest_pointsd")
def test_sample_farthest_pointsd_transform_random_start(mock_fn: Mock) -> None:
    """Test SampleFarthestPointsd with random_start parameter."""
    data = sentinel.data

    transform = SampleFarthestPointsd(pos_key="pos", num_samples=5, random_start=True)
    result = transform(data)

    mock_fn.assert_called_once_with(
        data,
        pos_key="pos",
        keys=(),
        num_samples=5,
        ratio=None,
        random_start=True,
        allow_missing_keys=False,
    )
    assert result is mock_fn.return_value


@patch("torch_pointcloud.transforms.dictionary.transforms.F.remove_near_origind")
def test_remove_near_origind_transform(mock_fn: Mock) -> None:
    """Test that RemoveNearOrigind delegates to functional API correctly."""
    data = sentinel.data
    pos_key = "pos"
    keys = ("labels",)
    radius = 0.01
    allow_missing_keys = sentinel.allow_missing_keys

    transform = RemoveNearOrigind(
        pos_key=pos_key,
        keys=keys,
        radius=radius,
        allow_missing_keys=allow_missing_keys,
    )

    result = transform(data)

    mock_fn.assert_called_once_with(
        data,
        pos_key=pos_key,
        keys=keys,
        radius=radius,
        allow_missing_keys=allow_missing_keys,
    )
    assert result is mock_fn.return_value


@patch("torch_pointcloud.transforms.dictionary.transforms.F.remove_near_origind")
def test_remove_near_origind_transform_defaults(mock_fn: Mock) -> None:
    """Test RemoveNearOrigind with default parameters."""
    data = sentinel.data

    transform = RemoveNearOrigind(pos_key="pos")
    result = transform(data)

    mock_fn.assert_called_once_with(
        data,
        pos_key="pos",
        keys=(),
        radius=1e-3,
        allow_missing_keys=False,
    )
    assert result is mock_fn.return_value


@patch("torch_pointcloud.transforms.dictionary.transforms.F.absd")
def test_absd_transform(mock_fn: Mock) -> None:
    """Test that Absd delegates to functional API correctly."""
    data = sentinel.data
    keys = ("pos",)
    allow_missing_keys = sentinel.allow_missing_keys

    transform = Absd(keys=keys, allow_missing_keys=allow_missing_keys)
    result = transform(data)

    mock_fn.assert_called_once_with(
        data,
        keys=keys,
        inplace=False,
        allow_missing_keys=allow_missing_keys,
    )
    assert result is mock_fn.return_value


@patch("torch_pointcloud.transforms.dictionary.transforms.F.absd")
def test_absd_transform_inplace(mock_fn: Mock) -> None:
    """Test Absd with inplace=True."""
    data = sentinel.data
    keys = ("pos",)

    transform = Absd(keys=keys, inplace=True)
    result = transform(data)

    mock_fn.assert_called_once_with(
        data,
        keys=keys,
        inplace=True,
        allow_missing_keys=False,
    )
    assert result is mock_fn.return_value


@patch("torch_pointcloud.transforms.dictionary.transforms.F.bounding_boxd")
def test_bounding_boxd_transform(mock_fn: Mock) -> None:
    """Test that BoundingBoxd delegates to functional API correctly."""
    data = sentinel.data
    keys = ("pos",)
    allow_missing_keys = sentinel.allow_missing_keys

    transform = BoundingBoxd(keys=keys, allow_missing_keys=allow_missing_keys)
    result = transform(data)

    mock_fn.assert_called_once_with(
        data,
        keys=keys,
        dst_keys=None,
        dim=-1,
        allow_missing_keys=allow_missing_keys,
    )
    assert result is mock_fn.return_value


@patch("torch_pointcloud.transforms.dictionary.transforms.F.bounding_boxd")
def test_bounding_boxd_transform_with_dst_keys(mock_fn: Mock) -> None:
    """Test BoundingBoxd with dst_keys parameter."""
    data = sentinel.data
    keys = ("pos",)
    dst_keys = ("bbox",)

    transform = BoundingBoxd(keys=keys, dst_keys=dst_keys, dim=0)
    result = transform(data)

    mock_fn.assert_called_once_with(
        data,
        keys=keys,
        dst_keys=dst_keys,
        dim=0,
        allow_missing_keys=False,
    )
    assert result is mock_fn.return_value


@patch("torch_pointcloud.transforms.dictionary.transforms.F.inbox_maskd")
def test_inbox_maskd_transform(mock_fn: Mock) -> None:
    """Test that InboxMaskd delegates to functional API correctly."""
    data = sentinel.data
    keys = ("pos",)
    bbox_key = "bbox"
    allow_missing_keys = sentinel.allow_missing_keys

    transform = InboxMaskd(keys=keys, bbox_key=bbox_key, allow_missing_keys=allow_missing_keys)
    result = transform(data)

    mock_fn.assert_called_once_with(
        data,
        keys=keys,
        bbox_key=bbox_key,
        dst_keys=None,
        dim=-1,
        allow_missing_keys=allow_missing_keys,
    )
    assert result is mock_fn.return_value


@patch("torch_pointcloud.transforms.dictionary.transforms.F.inbox_maskd")
def test_inbox_maskd_transform_with_dst_keys(mock_fn: Mock) -> None:
    """Test InboxMaskd with dst_keys and custom dim."""
    data = sentinel.data
    keys = ("pos",)
    dst_keys = ("mask",)

    transform = InboxMaskd(keys=keys, bbox_key="bbox", dst_keys=dst_keys, dim=0)
    result = transform(data)

    mock_fn.assert_called_once_with(
        data,
        keys=keys,
        bbox_key="bbox",
        dst_keys=dst_keys,
        dim=0,
        allow_missing_keys=False,
    )
    assert result is mock_fn.return_value


@patch("torch_pointcloud.transforms.dictionary.transforms.F.apply_maskd")
def test_apply_maskd_transform(mock_fn: Mock) -> None:
    """Test that ApplyMaskd delegates to functional API correctly."""
    data = sentinel.data
    keys = ("pos",)
    mask_key = "mask"
    allow_missing_keys = sentinel.allow_missing_keys

    transform = ApplyMaskd(keys=keys, mask_key=mask_key, allow_missing_keys=allow_missing_keys)
    result = transform(data)

    mock_fn.assert_called_once_with(
        data,
        keys=keys,
        mask_key=mask_key,
        dst_keys=None,
        allow_missing_keys=allow_missing_keys,
    )
    assert result is mock_fn.return_value


@patch("torch_pointcloud.transforms.dictionary.transforms.F.apply_maskd")
def test_apply_maskd_transform_with_dst_keys(mock_fn: Mock) -> None:
    """Test ApplyMaskd with dst_keys parameter."""
    data = sentinel.data
    keys = ("pos",)
    dst_keys = ("filtered",)

    transform = ApplyMaskd(keys=keys, mask_key="mask", dst_keys=dst_keys)
    result = transform(data)

    mock_fn.assert_called_once_with(
        data,
        keys=keys,
        mask_key="mask",
        dst_keys=dst_keys,
        allow_missing_keys=False,
    )
    assert result is mock_fn.return_value
