from unittest.mock import patch

from torch_pointcloud.utils.random import seed_everything


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
