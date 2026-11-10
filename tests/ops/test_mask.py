import pytest
import torch

from torch_pointcloud.ops.mask import mask_iou, mask_nms, mask_to_instance


def test_mask_iou_matches_dense_counts() -> None:
    masks = torch.rand(40, 300, generator=torch.Generator().manual_seed(0)) < 0.1
    masks[3] = False
    dense = masks.float()
    intersection = dense @ dense.T
    sizes = dense.sum(dim=1)
    expected = intersection / (sizes[:, None] + sizes[None, :] - intersection).clamp(min=1)
    mask_index = masks.nonzero().t()
    shuffled = mask_index[:, torch.randperm(mask_index.size(1), generator=torch.Generator().manual_seed(1))]
    torch.testing.assert_close(mask_iou(mask_index, 40), expected)
    torch.testing.assert_close(mask_iou(shuffled, 40), expected)


def test_mask_iou_without_masks() -> None:
    assert mask_iou(torch.zeros(2, 0, dtype=torch.long), 0).shape == (0, 0)


def test_mask_nms_keeps_the_best_of_each_overlapping_group() -> None:
    masks = torch.tensor(
        [
            [True, True, True, False, False],
            [True, True, False, False, False],
            [False, False, False, True, True],
            [False, False, False, False, True],
        ]
    )
    keep = mask_nms(masks.nonzero().t(), torch.tensor([0.5, 0.9, 0.7, 0.8]), iou_threshold=0.4)
    assert keep.tolist() == [1, 3]


@pytest.mark.parametrize("threshold", [pytest.param(0.0, id="any-overlap"), pytest.param(1.0, id="never")])
def test_mask_nms_threshold_bounds(threshold: float) -> None:
    masks = torch.tensor([[True, True, False], [False, True, True]])
    keep = mask_nms(masks.nonzero().t(), torch.tensor([0.2, 0.6]), iou_threshold=threshold)
    assert keep.tolist() == ([1] if threshold == 0.0 else [1, 0])


def test_mask_nms_without_masks() -> None:
    assert mask_nms(torch.zeros(2, 0, dtype=torch.long), torch.zeros(0), iou_threshold=0.5).tolist() == []


def test_mask_to_instance_gives_shared_rows_to_the_best_mask() -> None:
    masks = torch.tensor([[True, True, False, False], [False, True, True, False], [False, True, False, False]])
    out = mask_to_instance(masks.nonzero().t(), torch.tensor([0.4, 0.9, 0.9]), num_rows=4)
    assert out.tolist() == [0, 2, 1, -1]


def test_mask_to_instance_without_masks() -> None:
    out = mask_to_instance(torch.zeros(2, 0, dtype=torch.long), torch.zeros(0), num_rows=3)
    assert out.tolist() == [-1, -1, -1]
