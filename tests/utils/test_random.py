import random
from unittest.mock import patch

import numpy as np
import torch

from torch_pointcloud.utils.random import max_seed_value, min_seed_value, seed_everything


def test_seed_everything() -> None:
    """Test the seed everything function is setting the seed to all its dependencies."""
    test_seed = 42

    with (
        patch("random.seed") as mock_random_seed,
        patch("numpy.random.seed") as mock_np_seed,
        patch("torch.manual_seed") as mock_torch_seed,
        patch("torch.cuda.manual_seed_all") as mock_torch_cuda_seed,
    ):
        returned_seed = seed_everything(test_seed)
        assert returned_seed == test_seed

        # Verify each seeding function was called exactly once with the correct seed
        mock_random_seed.assert_called_once_with(test_seed)
        mock_np_seed.assert_called_once_with(test_seed)
        mock_torch_seed.assert_called_once_with(test_seed)
        mock_torch_cuda_seed.assert_called_once_with(test_seed)


def test_seed_everything_reproduces_draws() -> None:
    """Reseeding with the same seed replays the same Python, NumPy and torch draws."""
    seed_everything(123)
    python_draw = random.random()
    numpy_draw = np.random.rand(4)
    torch_draw = torch.rand(4)

    seed_everything(123)
    assert random.random() == python_draw
    assert np.array_equal(np.random.rand(4), numpy_draw)
    assert torch.equal(torch.rand(4), torch_draw)


def test_seed_everything_different_seeds_give_different_draws() -> None:
    seed_everything(123)
    torch_draw = torch.rand(64)
    seed_everything(456)
    assert not torch.equal(torch.rand(64), torch_draw)


def test_seed_everything_none_picks_valid_seed() -> None:
    seed = seed_everything(None)
    assert min_seed_value <= seed <= max_seed_value
