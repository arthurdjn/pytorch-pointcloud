import pytest
import torch

from torch_pointcloud.losses.focal import gaussian_focal_loss


@pytest.mark.parametrize(
    "dtype",
    [pytest.param(torch.float16, id="float16"), pytest.param(torch.bfloat16, id="bfloat16")],
)
def test_gaussian_focal_loss_half_precision_saturated_prediction_is_finite(dtype: torch.dtype) -> None:
    pred = torch.tensor([[1.0, 0.0], [0.5, 1.0]])
    target = torch.tensor([[1.0, 0.0], [0.3, 0.0]])

    loss = gaussian_focal_loss(pred.to(dtype), target.to(dtype))

    assert torch.isfinite(loss)
    torch.testing.assert_close(loss.float(), gaussian_focal_loss(pred, target), rtol=1e-2, atol=1e-3)
