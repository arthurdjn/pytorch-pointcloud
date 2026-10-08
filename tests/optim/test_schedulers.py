import math
from typing import List

import pytest
import torch

from torch_pointcloud.optim import CosineWarmupLR, PolyLR


def _optimizer(*lrs: float) -> torch.optim.Optimizer:
    return torch.optim.SGD([{"params": [torch.zeros(1, requires_grad=True)], "lr": lr} for lr in lrs], lr=lrs[0])


def _curve(scheduler: torch.optim.lr_scheduler.LRScheduler, steps: int, group: int = 0) -> List[float]:
    values = []
    for _ in range(steps):
        values.append(float(scheduler.get_last_lr()[group]))
        scheduler.step()
    return values


def test_cosine_warmup_linear_warmup_then_cosine_then_floor() -> None:
    sched = CosineWarmupLR(_optimizer(1.0), total_steps=10, warmup_steps=2, warmup_lr_init=0.1, lr_min=0.01)
    values = _curve(sched, 13)
    assert values[0] == pytest.approx(0.1)
    assert values[1] == pytest.approx(0.55)  # halfway through the warm-up
    # the cosine counts from step 0 over `total_steps` (timm's default), so step 5 is the midpoint
    assert values[5] == pytest.approx(0.01 + 0.5 * (1.0 - 0.01) * (1 + math.cos(math.pi * 0.5)))
    assert values[2] == pytest.approx(0.01 + 0.5 * 0.99 * (1 + math.cos(math.pi * 0.2)))
    assert values[10] == values[11] == values[12] == pytest.approx(0.01)


def test_cosine_warmup_prefix_counts_from_the_end_of_warmup() -> None:
    sched = CosineWarmupLR(_optimizer(1.0), total_steps=12, warmup_steps=2, lr_min=0.0, warmup_prefix=True)
    values = _curve(sched, 14)
    assert values[2] == pytest.approx(1.0)  # cosine starts at the base rate right after the warm-up
    assert values[7] == pytest.approx(0.5)  # midpoint of the 10 remaining steps
    assert values[12] == pytest.approx(0.0)


def test_cosine_warmup_without_warmup_is_a_plain_cosine() -> None:
    sched = CosineWarmupLR(_optimizer(2.0), total_steps=4)
    assert _curve(sched, 5) == pytest.approx(
        [2.0, 2.0 * 0.5 * (1 + math.cos(math.pi / 4)), 1.0, 2.0 * 0.5 * (1 + math.cos(3 * math.pi / 4)), 0.0]
    )


def test_cosine_warmup_scales_every_param_group() -> None:
    sched = CosineWarmupLR(_optimizer(1.0, 0.1), total_steps=4, warmup_steps=2)
    assert _curve(sched, 4, group=1) == pytest.approx(
        [0.0, 0.05, 0.1 * 0.5 * (1 + math.cos(math.pi / 2)), 0.1 * 0.5 * (1 + math.cos(3 * math.pi / 4))]
    )


def test_poly_lr_matches_closed_form() -> None:
    sched = PolyLR(_optimizer(1.0), total_steps=5, power=0.9, lr_min=0.1)
    expected = [0.1 + 0.9 * (1 - t / 5) ** 0.9 for t in range(5)] + [0.1]
    assert _curve(sched, 6) == pytest.approx(expected)


def test_poly_lr_pointcept_variant_keeps_last_step_nonzero() -> None:
    n = 10
    sched = PolyLR(_optimizer(1.0), total_steps=n + 1, power=0.9)
    values = _curve(sched, n)
    assert values == pytest.approx([(1 - s / (n + 1)) ** 0.9 for s in range(n)])
    assert values[-1] > 0


def test_schedulers_validate_arguments() -> None:
    with pytest.raises(ValueError, match="total_steps"):
        CosineWarmupLR(_optimizer(1.0), total_steps=0)
    with pytest.raises(ValueError, match="warmup_steps"):
        PolyLR(_optimizer(1.0), total_steps=5, warmup_steps=5)


def test_scheduler_state_dict_resumes_the_curve() -> None:
    sched = CosineWarmupLR(_optimizer(1.0), total_steps=8, warmup_steps=2)
    reference = _curve(sched, 8)
    sched = CosineWarmupLR(_optimizer(1.0), total_steps=8, warmup_steps=2)
    first_half = _curve(sched, 4)
    state = sched.state_dict()
    opt = _optimizer(1.0)
    resumed = CosineWarmupLR(opt, total_steps=8, warmup_steps=2)
    resumed.load_state_dict(state)
    for group in opt.param_groups:
        group["lr"] = resumed.get_last_lr()[0]
    assert first_half + _curve(resumed, 4) == pytest.approx(reference)
