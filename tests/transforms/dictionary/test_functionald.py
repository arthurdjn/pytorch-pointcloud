from unittest.mock import MagicMock, Mock, patch, sentinel

import pytest
import torch

import torch_pointcloud.transforms.dictionary.functional as F


@patch("torch_pointcloud.transforms.dictionary.functional.F.random_sample")
def test_random_sampled(mock_fn: Mock) -> None:
    """Test that random_sampled correctly applies random_sample to specified keys."""
    data = {
        "points": MagicMock(),
        "normal": MagicMock(),
        "other": MagicMock(),
    }
    num_samples = sentinel.num_samples
    allow_missing_keys = sentinel.allow_missing_keys
    generator = sentinel.generator
    sampled_tensor = sentinel.sampled_tensor
    sampled_indices = sentinel.indices
    mock_fn.return_value = (sampled_tensor, sampled_indices)

    keys = ["points", "normal"]

    result = F.random_sampled(
        data,
        keys,
        num_samples=num_samples,
        generator=generator,
        allow_missing_keys=allow_missing_keys,
    )

    mock_fn.assert_called_once_with(
        data["points"],
        num_samples,
        generator=generator,
        return_indices=True,
    )

    assert result["points"] is sampled_tensor
    assert result["normal"] is data["normal"][sampled_indices]
    # Check that non-specified keys are unchanged
    assert result["other"] is data["other"]


def test_random_sampled_does_not_mutate_original() -> None:
    """Test that random_sampled does not modify the original dictionary."""
    pos = torch.randn(10, 3)
    labels = torch.arange(10)
    data = {"pos": pos, "label": labels, "other": "keep"}
    original_data = dict(data)

    F.random_sampled(data, keys=["pos", "label"], num_samples=3)

    assert data["pos"] is original_data["pos"]
    assert data["label"] is original_data["label"]
    assert data["other"] is original_data["other"]


@patch("torch_pointcloud.transforms.dictionary.functional.F.random_sample_face_vertices")
def test_random_sample_face_verticesd(mock_fn: Mock) -> None:
    """Test that random_sample_face_verticesd correctly processes mesh data."""
    data = {
        "vertices": MagicMock(),
        "color": MagicMock(),
        "face": MagicMock(),
        "other": MagicMock(),
    }
    num_samples = sentinel.num_samples
    allow_missing_keys = sentinel.allow_missing_keys
    generator = sentinel.generator
    sampled_vertices = sentinel.sampled_vertices
    sampled_normals = sentinel.sampled_normals
    mock_fn.return_value = (sampled_vertices, sampled_normals)

    result = F.random_sample_face_verticesd(
        data,
        keys=["vertices"],
        face_key="face",
        num_samples=num_samples,
        normal_key="normal",
        generator=generator,
        allow_missing_keys=allow_missing_keys,
    )

    mock_fn.assert_called_once_with(
        data["vertices"],
        data["face"],
        num_samples,
        return_normals=True,
        generator=generator,
    )

    assert result["vertices"] is sampled_vertices
    assert result["normal"] is sampled_normals
    assert result["other"] is data["other"]
    assert result["color"] is data["color"]  # Unchanged


def test_random_sample_face_verticesd_does_not_mutate_original() -> None:
    """Test that random_sample_face_verticesd does not modify the original dictionary."""
    vertices = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    face = torch.tensor([[0, 1, 2], [0, 2, 3]])
    data = {"vertices": vertices, "face": face, "other": "keep"}
    original_data = dict(data)

    F.random_sample_face_verticesd(
        data,
        keys=["vertices"],
        face_key="face",
        num_samples=5,
        normal_key=None,
    )

    assert data["vertices"] is original_data["vertices"]
    assert data["face"] is original_data["face"]
    assert data["other"] is original_data["other"]


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

    result = F.normalize_scaled(data, keys=keys, allow_missing_keys=allow_missing_keys)

    mock_fn.assert_called_once_with(data["points"], eps=1e-8, method="centroid")

    assert result["points"] is normalized_tensor
    assert result["other"] is data["other"]


def test_normalize_scaled_does_not_mutate_original() -> None:
    """Test that normalize_scaled does not modify the original dictionary."""
    pos = torch.randn(5, 3)
    data = {"pos": pos, "other": "keep"}
    original_data = dict(data)

    F.normalize_scaled(data, keys=["pos"])

    assert data["pos"] is original_data["pos"]
    assert data["other"] is original_data["other"]


@patch("torch_pointcloud.transforms.dictionary.functional.F.sample_farthest_points")
def test_sample_farthest_pointsd_basic(mock_fps: Mock) -> None:
    """Test that sample_farthest_pointsd samples the pos key and applies indices to extra keys."""
    indices = torch.tensor([0, 3, 7])
    mock_fps.return_value = indices

    pos = torch.randn(10, 3)
    labels = torch.arange(10)
    data = {"pos": pos, "label": labels, "other": sentinel.other}

    result = F.sample_farthest_pointsd(data, pos_key="pos", keys=["label"], num_samples=3)

    mock_fps.assert_called_once_with(pos, num_samples=3, ratio=None, random_start=False)
    assert torch.equal(result["pos"], pos[indices])
    assert torch.equal(result["label"], labels[indices])
    assert result["other"] is sentinel.other


@patch("torch_pointcloud.transforms.dictionary.functional.F.sample_farthest_points")
def test_sample_farthest_pointsd_pos_only(mock_fps: Mock) -> None:
    """Test sample_farthest_pointsd with no extra keys."""
    indices = torch.tensor([0, 1])
    mock_fps.return_value = indices

    pos = torch.randn(5, 3)
    data = {"pos": pos}

    result = F.sample_farthest_pointsd(data, pos_key="pos", num_samples=2)

    mock_fps.assert_called_once_with(pos, num_samples=2, ratio=None, random_start=False)
    assert torch.equal(result["pos"], pos[indices])


@patch("torch_pointcloud.transforms.dictionary.functional.F.sample_farthest_points")
def test_sample_farthest_pointsd_with_ratio(mock_fps: Mock) -> None:
    """Test sample_farthest_pointsd with ratio parameter."""
    indices = torch.tensor([0, 2, 4])
    mock_fps.return_value = indices

    pos = torch.randn(10, 3)
    data = {"pos": pos}

    _ = F.sample_farthest_pointsd(data, pos_key="pos", ratio=0.3)

    mock_fps.assert_called_once_with(pos, num_samples=None, ratio=0.3, random_start=False)


@patch("torch_pointcloud.transforms.dictionary.functional.F.sample_farthest_points")
def test_sample_farthest_pointsd_missing_key_raises(mock_fps: Mock) -> None:
    """Test sample_farthest_pointsd raises KeyError for missing keys when allow_missing_keys=False."""
    indices = torch.tensor([0])
    mock_fps.return_value = indices

    pos = torch.randn(5, 3)
    data = {"pos": pos}

    with pytest.raises(KeyError, match="missing"):
        F.sample_farthest_pointsd(data, pos_key="pos", keys=["missing"], num_samples=1)


@patch("torch_pointcloud.transforms.dictionary.functional.F.sample_farthest_points")
def test_sample_farthest_pointsd_allow_missing_keys(mock_fps: Mock) -> None:
    """Test sample_farthest_pointsd skips missing keys when allow_missing_keys=True."""
    indices = torch.tensor([0, 1])
    mock_fps.return_value = indices

    pos = torch.randn(5, 3)
    data = {"pos": pos}

    result = F.sample_farthest_pointsd(data, pos_key="pos", keys=["missing"], num_samples=2, allow_missing_keys=True)
    assert torch.equal(result["pos"], pos[indices])
    assert "missing" not in result


@patch("torch_pointcloud.transforms.dictionary.functional.F.sample_farthest_points")
def test_sample_farthest_pointsd_does_not_mutate_original(mock_fps: Mock) -> None:
    """Test that sample_farthest_pointsd does not modify the original dictionary."""
    indices = torch.tensor([0, 2])
    mock_fps.return_value = indices

    pos = torch.randn(5, 3)
    data = {"pos": pos, "other": "keep"}
    original_data = dict(data)

    F.sample_farthest_pointsd(data, pos_key="pos", num_samples=2)

    assert data["pos"] is original_data["pos"]
    assert data["other"] is original_data["other"]


@patch("torch_pointcloud.transforms.dictionary.functional.F.remove_near_origin")
def test_remove_near_origind_basic(mock_fn: Mock) -> None:
    """Test that remove_near_origind applies mask from pos_key to all specified keys."""
    mask = torch.tensor([False, True, True, False])
    filtered_pos = sentinel.filtered_pos
    mock_fn.return_value = (filtered_pos, mask)

    pos = torch.randn(4, 3)
    labels = torch.tensor([0, 1, 2, 3])
    data = {"pos": pos, "label": labels, "other": sentinel.other}

    result = F.remove_near_origind(data, pos_key="pos", keys=["label"], radius=0.01)

    mock_fn.assert_called_once_with(pos, radius=0.01, return_mask=True)
    assert torch.equal(result["pos"], pos[mask])
    assert torch.equal(result["label"], labels[mask])
    assert result["other"] is sentinel.other


@patch("torch_pointcloud.transforms.dictionary.functional.F.remove_near_origin")
def test_remove_near_origind_pos_only(mock_fn: Mock) -> None:
    """Test remove_near_origind with no extra keys."""
    mask = torch.tensor([True, False, True])
    filtered_pos = sentinel.filtered_pos
    mock_fn.return_value = (filtered_pos, mask)

    pos = torch.randn(3, 3)
    data = {"pos": pos}

    result = F.remove_near_origind(data, pos_key="pos")

    mock_fn.assert_called_once_with(pos, radius=1e-3, return_mask=True)
    assert torch.equal(result["pos"], pos[mask])


@patch("torch_pointcloud.transforms.dictionary.functional.F.remove_near_origin")
def test_remove_near_origind_missing_key_raises(mock_fn: Mock) -> None:
    """Test remove_near_origind raises KeyError for missing keys."""
    mask = torch.tensor([True])
    mock_fn.return_value = (sentinel.filtered, mask)

    data = {"pos": torch.randn(1, 3)}

    with pytest.raises(KeyError, match="missing"):
        F.remove_near_origind(data, pos_key="pos", keys=["missing"])


@patch("torch_pointcloud.transforms.dictionary.functional.F.remove_near_origin")
def test_remove_near_origind_allow_missing_keys(mock_fn: Mock) -> None:
    """Test remove_near_origind skips missing keys when allow_missing_keys=True."""
    mask = torch.tensor([True, False])
    mock_fn.return_value = (sentinel.filtered, mask)

    pos = torch.randn(2, 3)
    data = {"pos": pos}

    result = F.remove_near_origind(data, pos_key="pos", keys=["missing"], allow_missing_keys=True)
    assert torch.equal(result["pos"], pos[mask])


@patch("torch_pointcloud.transforms.dictionary.functional.F.remove_near_origin")
def test_remove_near_origind_does_not_mutate_original(mock_fn: Mock) -> None:
    """Test that remove_near_origind does not modify the original dictionary."""
    mask = torch.tensor([True, False, True])
    mock_fn.return_value = (sentinel.filtered, mask)

    pos = torch.randn(3, 3)
    data = {"pos": pos, "other": "keep"}
    original_data = dict(data)

    F.remove_near_origind(data, pos_key="pos")

    assert data["pos"] is original_data["pos"]
    assert data["other"] is original_data["other"]


@patch("torch_pointcloud.transforms.dictionary.functional.F.abs")
def test_absd_basic(mock_fn: Mock) -> None:
    """Test that absd applies abs to specified keys."""
    abs_result = sentinel.abs_result
    mock_fn.return_value = abs_result

    data = {"pos": MagicMock(), "other": sentinel.other}
    result = F.absd(data, keys=["pos"])

    mock_fn.assert_called_once_with(data["pos"], inplace=False)
    assert result["pos"] is abs_result
    assert result["other"] is sentinel.other


@patch("torch_pointcloud.transforms.dictionary.functional.F.abs")
def test_absd_multiple_keys(mock_fn: Mock) -> None:
    """Test that absd applies abs to multiple keys."""
    mock_fn.side_effect = [sentinel.abs_a, sentinel.abs_b]

    data = {"a": MagicMock(), "b": MagicMock(), "c": sentinel.c}
    result = F.absd(data, keys=["a", "b"])

    assert mock_fn.call_count == 2
    assert result["a"] is sentinel.abs_a
    assert result["b"] is sentinel.abs_b
    assert result["c"] is sentinel.c


@patch("torch_pointcloud.transforms.dictionary.functional.F.abs")
def test_absd_inplace(mock_fn: Mock) -> None:
    """Test that absd passes inplace parameter."""
    mock_fn.return_value = sentinel.result

    data = {"pos": MagicMock()}
    F.absd(data, keys=["pos"], inplace=True)

    mock_fn.assert_called_once_with(data["pos"], inplace=True)


@patch("torch_pointcloud.transforms.dictionary.functional.F.abs")
def test_absd_missing_key_raises(mock_fn: Mock) -> None:
    """Test that absd raises KeyError for missing keys."""
    data = {"pos": MagicMock()}

    with pytest.raises(KeyError, match="missing"):
        F.absd(data, keys=["missing"])


@patch("torch_pointcloud.transforms.dictionary.functional.F.abs")
def test_absd_allow_missing_keys(mock_fn: Mock) -> None:
    """Test absd skips missing keys when allow_missing_keys=True."""
    data = {"pos": MagicMock()}
    result = F.absd(data, keys=["missing"], allow_missing_keys=True)

    mock_fn.assert_not_called()
    assert result["pos"] is data["pos"]


def test_absd_does_not_mutate_original() -> None:
    """Test that absd does not modify the original dictionary."""
    original_tensor = torch.tensor([-1.0, 2.0, -3.0])
    data = {"pos": original_tensor, "other": "keep"}
    original_data = dict(data)

    result = F.absd(data, keys=["pos"])

    assert data["pos"] is original_data["pos"]
    assert data["other"] is original_data["other"]
    assert torch.equal(result["pos"], torch.tensor([1.0, 2.0, 3.0]))


@patch("torch_pointcloud.transforms.dictionary.functional.F.bounding_box")
def test_bounding_boxd_basic(mock_fn: Mock) -> None:
    """Test that bounding_boxd computes bounding box and stores in data."""
    bbox_result = (0.0, 0.0, 0.0, 1.0, 1.0, 1.0)
    mock_fn.return_value = bbox_result

    data = {"pos": MagicMock(), "other": sentinel.other}
    result = F.bounding_boxd(data, keys=["pos"])

    mock_fn.assert_called_once_with(data["pos"], dim=0)
    assert result["pos"] == bbox_result
    assert result["other"] is sentinel.other


@patch("torch_pointcloud.transforms.dictionary.functional.F.bounding_box")
def test_bounding_boxd_with_dst_keys(mock_fn: Mock) -> None:
    """Test that bounding_boxd stores result in dst_keys."""
    bbox_result = (0.0, 1.0)
    mock_fn.return_value = bbox_result

    data = {"pos": MagicMock(), "other": sentinel.other}
    result = F.bounding_boxd(data, keys=["pos"], dst_keys=["bbox"])

    mock_fn.assert_called_once_with(data["pos"], dim=0)
    assert result["bbox"] == bbox_result
    assert result["pos"] is data["pos"]  # original key untouched when dst_keys given
    assert result["other"] is sentinel.other


@patch("torch_pointcloud.transforms.dictionary.functional.F.bounding_box")
def test_bounding_boxd_custom_dim(mock_fn: Mock) -> None:
    """Test bounding_boxd with a custom dim."""
    mock_fn.return_value = sentinel.bbox

    data = {"pos": MagicMock()}
    F.bounding_boxd(data, keys=["pos"], dim=0)

    mock_fn.assert_called_once_with(data["pos"], dim=0)


@patch("torch_pointcloud.transforms.dictionary.functional.F.bounding_box")
def test_bounding_boxd_missing_key_raises(mock_fn: Mock) -> None:
    """Test that bounding_boxd raises KeyError for missing keys."""
    data = {"pos": MagicMock()}

    with pytest.raises(KeyError, match="missing"):
        F.bounding_boxd(data, keys=["missing"])


@patch("torch_pointcloud.transforms.dictionary.functional.F.bounding_box")
def test_bounding_boxd_allow_missing_keys(mock_fn: Mock) -> None:
    """Test bounding_boxd skips missing keys when allow_missing_keys=True."""
    data = {"pos": MagicMock()}
    _ = F.bounding_boxd(data, keys=["missing"], allow_missing_keys=True)

    mock_fn.assert_not_called()


@patch("torch_pointcloud.transforms.dictionary.functional.F.bounding_box")
def test_bounding_boxd_does_not_mutate_original(mock_fn: Mock) -> None:
    """Test that bounding_boxd does not modify the original dictionary."""
    mock_fn.return_value = (0.0, 1.0)

    data = {"pos": MagicMock(), "other": "keep"}
    original_data = dict(data)

    F.bounding_boxd(data, keys=["pos"], dst_keys=["bbox"])

    assert data["pos"] is original_data["pos"]
    assert data["other"] is original_data["other"]
    assert "bbox" not in data  # new key should only appear in the result


@patch("torch_pointcloud.transforms.dictionary.functional.F.inbox_mask")
def test_inbox_maskd_basic(mock_fn: Mock) -> None:
    """Test that inbox_maskd creates a mask and stores it in the data."""
    mask_result = sentinel.mask
    mock_fn.return_value = mask_result

    data = {"pos": MagicMock(), "other": sentinel.other}
    result = F.inbox_maskd(data, keys=["pos"], bbox=sentinel.bbox)

    mock_fn.assert_called_once_with(data["pos"], sentinel.bbox, dim=-1)
    assert result["pos"] is mask_result
    assert result["other"] is sentinel.other


@patch("torch_pointcloud.transforms.dictionary.functional.F.inbox_mask")
def test_inbox_maskd_with_dst_keys(mock_fn: Mock) -> None:
    """Test that inbox_maskd stores result in dst_keys."""
    mask_result = sentinel.mask
    mock_fn.return_value = mask_result

    data = {"pos": MagicMock()}
    result = F.inbox_maskd(data, keys=["pos"], bbox=sentinel.bbox, dst_keys=["mask"])

    mock_fn.assert_called_once_with(data["pos"], sentinel.bbox, dim=-1)
    assert result["mask"] is mask_result
    assert result["pos"] is data["pos"]  # original untouched


@patch("torch_pointcloud.transforms.dictionary.functional.F.inbox_mask")
def test_inbox_maskd_missing_key_raises(mock_fn: Mock) -> None:
    """Test that inbox_maskd raises KeyError for missing source keys."""
    data: dict = {}

    with pytest.raises(KeyError, match="missing"):
        F.inbox_maskd(data, keys=["pos"], bbox=(0.0, 1.0))


@patch("torch_pointcloud.transforms.dictionary.functional.F.inbox_mask")
def test_inbox_maskd_allow_missing_keys(mock_fn: Mock) -> None:
    """Test inbox_maskd skips missing source keys when allow_missing_keys=True."""
    data: dict = {}
    _ = F.inbox_maskd(data, keys=["missing"], bbox=(0.0, 1.0), allow_missing_keys=True)

    mock_fn.assert_not_called()


@patch("torch_pointcloud.transforms.dictionary.functional.F.inbox_mask")
def test_inbox_maskd_custom_dim(mock_fn: Mock) -> None:
    """Test inbox_maskd with a custom dim."""
    mock_fn.return_value = sentinel.mask

    bbox = (0.0, 1.0)
    data = {"pos": MagicMock()}
    F.inbox_maskd(data, keys=["pos"], bbox=bbox, dim=0)

    mock_fn.assert_called_once_with(data["pos"], bbox, dim=0)


@patch("torch_pointcloud.transforms.dictionary.functional.F.inbox_mask")
def test_inbox_maskd_does_not_mutate_original(mock_fn: Mock) -> None:
    """Test that inbox_maskd does not modify the original dictionary."""
    mock_fn.return_value = sentinel.mask

    bbox = (0.0, 0.0, 1.0, 1.0)
    data = {"pos": MagicMock(), "other": "keep"}
    original_data = dict(data)

    F.inbox_maskd(data, keys=["pos"], bbox=bbox, dst_keys=["mask"])

    assert data["pos"] is original_data["pos"]
    assert data["other"] is original_data["other"]
    assert "mask" not in data


@patch("torch_pointcloud.transforms.dictionary.functional.F.apply_mask")
def test_apply_maskd_basic(mock_fn: Mock) -> None:
    """Test that apply_maskd applies a mask to specified keys."""
    masked_result = sentinel.masked
    mock_fn.return_value = masked_result

    mask = sentinel.mask
    data = {"pos": MagicMock(), "mask": mask, "other": sentinel.other}
    result = F.apply_maskd(data, keys=["pos"], mask_key="mask")

    mock_fn.assert_called_once_with(data["pos"], mask)
    assert result["pos"] is masked_result
    assert result["other"] is sentinel.other


@patch("torch_pointcloud.transforms.dictionary.functional.F.apply_mask")
def test_apply_maskd_with_dst_keys(mock_fn: Mock) -> None:
    """Test that apply_maskd stores result in dst_keys."""
    masked_result = sentinel.masked
    mock_fn.return_value = masked_result

    mask = sentinel.mask
    data = {"pos": MagicMock(), "mask": mask}
    result = F.apply_maskd(data, keys=["pos"], mask_key="mask", dst_keys=["filtered_pos"])

    mock_fn.assert_called_once_with(data["pos"], mask)
    assert result["filtered_pos"] is masked_result
    assert result["pos"] is data["pos"]  # original untouched


@patch("torch_pointcloud.transforms.dictionary.functional.F.apply_mask")
def test_apply_maskd_multiple_keys(mock_fn: Mock) -> None:
    """Test apply_maskd applies mask to multiple keys."""
    mock_fn.side_effect = [sentinel.masked_a, sentinel.masked_b]

    mask = sentinel.mask
    data = {"a": MagicMock(), "b": MagicMock(), "mask": mask}
    result = F.apply_maskd(data, keys=["a", "b"], mask_key="mask")

    assert mock_fn.call_count == 2
    assert result["a"] is sentinel.masked_a
    assert result["b"] is sentinel.masked_b


@patch("torch_pointcloud.transforms.dictionary.functional.F.apply_mask")
def test_apply_maskd_missing_mask_key_raises(mock_fn: Mock) -> None:
    """Test that apply_maskd raises KeyError when mask_key is missing."""
    data = {"pos": MagicMock()}

    with pytest.raises(KeyError, match="missing"):
        F.apply_maskd(data, keys=["pos"], mask_key="mask")


@patch("torch_pointcloud.transforms.dictionary.functional.F.apply_mask")
def test_apply_maskd_missing_key_raises(mock_fn: Mock) -> None:
    """Test that apply_maskd raises KeyError for missing source keys."""
    data = {"mask": sentinel.mask}

    with pytest.raises(KeyError, match="missing"):
        F.apply_maskd(data, keys=["pos"], mask_key="mask")


@patch("torch_pointcloud.transforms.dictionary.functional.F.apply_mask")
def test_apply_maskd_allow_missing_keys(mock_fn: Mock) -> None:
    """Test apply_maskd skips missing source keys when allow_missing_keys=True."""
    data = {"mask": sentinel.mask}
    _ = F.apply_maskd(data, keys=["missing"], mask_key="mask", allow_missing_keys=True)

    mock_fn.assert_not_called()


@patch("torch_pointcloud.transforms.dictionary.functional.F.apply_mask")
def test_apply_maskd_does_not_mutate_original(mock_fn: Mock) -> None:
    """Test that apply_maskd does not modify the original dictionary."""
    mock_fn.return_value = sentinel.masked

    mask = sentinel.mask
    data = {"pos": MagicMock(), "mask": mask, "other": "keep"}
    original_data = dict(data)

    F.apply_maskd(data, keys=["pos"], mask_key="mask", dst_keys=["filtered"])

    assert data["pos"] is original_data["pos"]
    assert data["other"] is original_data["other"]
    assert "filtered" not in data
