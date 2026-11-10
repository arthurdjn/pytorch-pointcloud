import pytest
import torch

from torch_pointcloud.ops.math import safe_divide


def test_safe_divide_default_replaces_zero_denominators() -> None:
    a = torch.tensor([1.0, 2.0, 3.0])
    b = torch.tensor([1.0, 0.0, 2.0])
    assert torch.equal(safe_divide(a, b, default=0.0), torch.tensor([1.0, 0.0, 1.5]))
    assert safe_divide(a, b)[1].isnan()


def test_safe_divide_tensor_default() -> None:
    a = torch.tensor([1, 2, 3])
    b = torch.tensor([1, 0, 1])
    out = safe_divide(a, b, default=torch.tensor([7, 8, 9]))
    assert out.dtype == torch.float32
    assert torch.equal(out, torch.tensor([1.0, 8.0, 3.0]))


@pytest.mark.parametrize(
    "dtype",
    [
        pytest.param(torch.float16, id="float16"),
        pytest.param(torch.float32, id="float32"),
        pytest.param(torch.float64, id="float64"),
    ],
)
def test_safe_divide_keeps_the_input_dtype(dtype: torch.dtype) -> None:
    a = torch.ones(3, dtype=dtype)
    b = torch.tensor([1, 0, 2], dtype=dtype)
    assert safe_divide(a, b).dtype == dtype
    assert safe_divide(a, b, default=0.0).dtype == dtype
