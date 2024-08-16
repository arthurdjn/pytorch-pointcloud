import random
from typing import Any, List

import numpy as np
import pytest
import torch
from torch import Tensor

from torch_pointcloud.utils.utils import aslist, default_vector, is_tensor, set_seed, to_tensor


@pytest.mark.parametrize(
    "value, expected",
    [
        (None, []),
        ([1, 2, 3], [1, 2, 3]),
        ((1, 2, 3), [1, 2, 3]),
        (torch.tensor([1, 2, 3]), [1, 2, 3]),
        (np.array([1, 2, 3]), [1, 2, 3]),
        (42, [42]),
    ],
)
def test_aslist(value: Any, expected: List[Any]) -> None:
    result = aslist(value)
    assert result == expected, f"Expected {expected}, but got {result}"


@pytest.mark.parametrize(
    "value, expected", [(torch.tensor([1, 2, 3]), True), (np.array([1, 2, 3]), False), ([1, 2, 3], False), (42, False)]
)
def test_is_tensor(value: Any, expected: bool) -> None:
    result = is_tensor(value)
    assert result == expected, f"Expected {expected}, but got {result}"


@pytest.mark.parametrize(
    "value, expected",
    [
        (torch.tensor([1, 2, 3]), torch.tensor([1, 2, 3])),
        ([1, 2, 3], torch.tensor([1, 2, 3])),
        ((1, 2, 3), torch.tensor([1, 2, 3])),
        (np.array([1, 2, 3]), torch.tensor([1, 2, 3])),
        (42, torch.tensor(42)),
    ],
)
def test_to_tensor(value: Any, expected: Tensor) -> None:
    result = to_tensor(value)
    assert torch.equal(result, expected), f"Expected {expected}, but got {result}"


@pytest.mark.parametrize(
    "vector, size, default_value, expected",
    [
        (None, 3, 0, torch.tensor([0, 0, 0])),
        (2, 3, 0, torch.tensor([2, 2, 2])),
        ([1, 2, 3], 3, 0, torch.tensor([1, 2, 3])),
        (torch.tensor([1]), 3, 0, torch.tensor([1, 1, 1])),
        (torch.tensor([1, 2, 3]), 3, 0, torch.tensor([1, 2, 3])),
    ],
)
def test_default_vector(vector: Any, size: int, default_value: int, expected: Tensor) -> None:
    result = default_vector(vector, size, default_value)
    assert torch.equal(result, expected), f"Expected {expected}, but got {result}"


def test_set_seed() -> None:
    seed = 42

    # Set the seed
    set_seed(seed)

    # Generate random numbers
    random_val = random.randint(0, 100)
    np_val = np.random.randint(0, 100)
    torch_val = torch.randint(0, 100, (1,)).item()

    # Reset the seed and generate the same random numbers to verify reproducibility
    set_seed(seed)
    random_val2 = random.randint(0, 100)
    np_val2 = np.random.randint(0, 100)
    torch_val2 = torch.randint(0, 100, (1,)).item()

    # Check that the random numbers generated are the same after setting the seed
    assert random_val == random_val2, f"Expected {random_val}, but got {random_val2}"
    assert np_val == np_val2, f"Expected {np_val}, but got {np_val2}"
    assert torch_val == torch_val2, f"Expected {torch_val}, but got {torch_val2}"
