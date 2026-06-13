import math

import pytest
import torch

from torch_pointcloud.utils.diffusion import DDIMScheduler


def test_linear_beta_schedule_alphas() -> None:
    scheduler = DDIMScheduler(num_train_timesteps=10, beta_start=0.1, beta_end=0.1)
    expected = torch.tensor([0.9**k for k in range(1, 11)])
    assert torch.allclose(scheduler.alphas_cumprod, expected, atol=1e-6)


def test_set_timesteps_evenly_spaced_descending() -> None:
    scheduler = DDIMScheduler(num_train_timesteps=1000)
    scheduler.set_timesteps(100)
    assert scheduler.timesteps.shape == (100,)
    assert scheduler.timesteps[0] == 990
    assert scheduler.timesteps[-1] == 0
    assert torch.all(scheduler.timesteps[:-1] > scheduler.timesteps[1:])


def test_add_noise_matches_closed_form() -> None:
    scheduler = DDIMScheduler(num_train_timesteps=100)
    x0 = torch.randn(32, 4)
    noise = torch.randn(32, 4)
    timesteps = torch.full((32,), 50, dtype=torch.long)
    noisy = scheduler.add_noise(x0, noise, timesteps)
    alpha = scheduler.alphas_cumprod[50]
    expected = math.sqrt(alpha) * x0 + math.sqrt(1 - alpha) * noise
    assert torch.allclose(noisy, expected, atol=1e-6)


def test_velocity_inverts_to_noise_and_sample() -> None:
    """`v = sqrt(a) eps - sqrt(1-a) x0` together with `x_t` recovers both `eps` and `x0`."""
    scheduler = DDIMScheduler(num_train_timesteps=100, prediction_type="v_prediction")
    x0 = torch.randn(8, 3)
    noise = torch.randn(8, 3)
    timesteps = torch.full((8,), 30, dtype=torch.long)
    noisy = scheduler.add_noise(x0, noise, timesteps)
    velocity = scheduler.get_velocity(x0, noise, timesteps)
    alpha = scheduler.alphas_cumprod[30]
    x0_rec = alpha**0.5 * noisy - (1 - alpha) ** 0.5 * velocity
    assert torch.allclose(x0_rec, x0, atol=1e-5)


@pytest.mark.parametrize("prediction_type", ["epsilon", "v_prediction"])
def test_deterministic_step_recovers_clean_sample(prediction_type: str) -> None:
    """With an oracle model and `eta = 0`, DDIM steps from $x_T$ land close to $x_0$."""
    scheduler = DDIMScheduler(num_train_timesteps=1000, prediction_type=prediction_type)
    scheduler.set_timesteps(50)
    torch.manual_seed(0)
    x0 = torch.randn(16, 2)
    noise = torch.randn(16, 2)
    sample = scheduler.add_noise(x0, noise, torch.full((16,), 990, dtype=torch.long))
    for t in scheduler.timesteps:
        alpha = scheduler.alphas_cumprod[t]
        if prediction_type == "epsilon":
            pred = (sample - alpha**0.5 * x0) / (1 - alpha) ** 0.5
        else:
            eps = (sample - alpha**0.5 * x0) / (1 - alpha) ** 0.5
            pred = alpha**0.5 * eps - (1 - alpha) ** 0.5 * x0
        sample = scheduler.step(pred, int(t), sample, eta=0.0)
    assert torch.allclose(sample, x0, atol=1e-3)


def test_step_requires_set_timesteps() -> None:
    scheduler = DDIMScheduler()
    with pytest.raises(RuntimeError, match="set_timesteps"):
        scheduler.step(torch.zeros(1, 1), 0, torch.zeros(1, 1))


def test_unknown_prediction_type_raises() -> None:
    with pytest.raises(ValueError, match="prediction type"):
        DDIMScheduler(prediction_type="nope")
