import pytest
import torch.nn as nn

from torch_pointcloud.optim import BNMomentumScheduler, bn_momentum, set_bn_momentum


def test_bn_momentum_votenet_schedule() -> None:
    assert [bn_momentum(e) for e in (0, 19, 20, 39, 40, 60, 160, 400)] == pytest.approx(
        [0.5, 0.5, 0.25, 0.25, 0.125, 0.0625, 0.5 * 0.5**8, 0.001]
    )


def test_bn_momentum_pointnet_schedule_floors_at_clip() -> None:
    values = [bn_momentum(e, init=0.1, clip=0.01) for e in range(0, 200, 20)]
    assert values[:4] == pytest.approx([0.1, 0.05, 0.025, 0.0125])
    assert all(v == 0.01 for v in values[4:])


def test_set_bn_momentum_touches_every_batchnorm_flavor() -> None:
    bn1, bn2, bn3 = nn.BatchNorm1d(2), nn.BatchNorm2d(2), nn.BatchNorm3d(2)
    model = nn.Sequential(bn1, nn.Sequential(bn2, bn3), nn.LayerNorm(2))
    set_bn_momentum(model, 0.03)
    assert bn1.momentum == 0.03 and bn2.momentum == 0.03 and bn3.momentum == 0.03


def test_bn_momentum_scheduler_step_sets_and_returns() -> None:
    bn = nn.BatchNorm1d(2)
    model = nn.Sequential(nn.Linear(2, 2), bn)
    scheduler = BNMomentumScheduler(model, init=0.5, decay_rate=0.5, decay_step=20, clip=0.001)
    assert scheduler.step(0) == 0.5 and bn.momentum == 0.5
    assert scheduler.step(45) == 0.125 and bn.momentum == 0.125
    assert scheduler.step(1000) == 0.001 and bn.momentum == 0.001
