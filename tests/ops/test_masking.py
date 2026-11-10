import pytest
import torch

from torch_pointcloud.ops.masking import random_mask


def test_random_mask_hides_the_same_number_of_tokens_per_row() -> None:
    torch.manual_seed(0)

    mask = random_mask(4, 10, 0.65)

    assert mask.shape == (4, 10)
    assert mask.dtype == torch.bool
    assert mask.sum(dim=1).tolist() == [6, 6, 6, 6]
    assert not torch.equal(mask[0], mask[1])


@pytest.mark.parametrize(
    "ratio,expected",
    [
        pytest.param(0.0, 0, id="none"),
        pytest.param(1.0, 8, id="all"),
    ],
)
def test_random_mask_bounds(ratio: float, expected: int) -> None:
    assert int(random_mask(2, 8, ratio).sum(dim=1).max()) == expected


def test_random_mask_invalid_ratio_raises() -> None:
    with pytest.raises(ValueError, match="ratio"):
        random_mask(2, 8, 1.5)
