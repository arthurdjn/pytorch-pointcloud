import pytest
import torch
from torch import Tensor

from torch_pointcloud.layers.curvenet_blocks import (
    AttentionGate,
    CurveAggregation,
    CurveGrouping,
    CurveInterventionConv,
    CurveNetFeaturePropagation,
    CurveWalk,
    LocalPointFeatureAggregation,
    RadiusMaxPool,
    crossover_suppression,
    hard_softmax,
)
from torch_pointcloud.utils.imports import _PYG_LIB_AVAILABLE

_REQUIRES_PYG_LIB = pytest.mark.skipif(not _PYG_LIB_AVAILABLE, reason="pyg-lib is not installed")


def _cloud(num_points: int = 64, batch_size: int = 2, channels: int = 16) -> tuple[Tensor, Tensor, Tensor]:
    torch.manual_seed(0)
    pos = torch.rand(num_points * batch_size, 3)
    x = torch.randn(num_points * batch_size, channels)
    batch = torch.arange(batch_size).repeat_interleave(num_points)
    return x, pos, batch


def _ring_table(num_points: int, k: int) -> Tensor:
    offsets = torch.arange(1, k + 1)
    return (torch.arange(num_points).unsqueeze(1) + offsets) % num_points


def test_hard_softmax_is_one_hot_with_softmax_gradient() -> None:
    logits = torch.tensor([[0.1, 2.0, -1.0], [3.0, 0.0, 1.0]], requires_grad=True)
    out = hard_softmax(logits)
    assert out.tolist() == [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]]
    weights = torch.randn(2, 3)
    (out * weights).sum().backward()
    soft = logits.detach().softmax(dim=-1)
    expected = soft * (weights - (soft * weights).sum(dim=-1, keepdim=True))
    assert logits.grad is not None and torch.allclose(logits.grad, expected)


def test_crossover_suppression_values() -> None:
    direction = torch.tensor([[1.0, 0.0], [0.0, 0.0]])
    candidates = torch.tensor([[[2.0, 0.0], [-1.0, 0.0], [0.0, 3.0], [-1.0, 1.0]], [[1.0, 0.0]] * 4])
    factor = crossover_suppression(direction, candidates)
    expected = torch.tensor([[1.0, 0.0, 1.0, 1.0 - 2**-0.5], [1.0, 1.0, 1.0, 1.0]])
    assert torch.allclose(factor, expected, atol=1e-6)


def test_lpfa_geometry_only_and_with_features() -> None:
    x, pos, _ = _cloud(num_points=32, batch_size=1)
    table = _ring_table(32, 8)
    embed = LocalPointFeatureAggregation(0, 24, num_layers=1, aggr="max").eval()
    assert embed(None, pos, table).shape == (32, 24)
    block = LocalPointFeatureAggregation(16, 16, num_layers=2).eval()
    assert block(x, pos, table).shape == (32, 16)


def test_lpfa_rejects_unknown_aggr() -> None:
    with pytest.raises(ValueError):
        LocalPointFeatureAggregation(0, 8, aggr="sum")  # type: ignore[arg-type]


def test_curve_walk_visits_table_neighbors() -> None:
    torch.manual_seed(0)
    num_points, k = 20, 3
    x = torch.randn(num_points, 8)
    table = _ring_table(num_points, k)
    walk = CurveWalk(8, curve_length=4).eval()
    start = torch.tensor([[0, 5, 10]])
    curves = walk(x, table, start)
    assert curves.shape == (1, 3, 4, 8)
    # Every visited feature is a row of `x` reachable in `step + 1` hops along the ring.
    for i, origin in enumerate(start[0].tolist()):
        reachable = {origin}
        for step in range(4):
            reachable = {int(j) for r in reachable for j in table[r]}
            match = (curves[0, i, step].unsqueeze(0) - x).abs().amax(dim=1) < 1e-6
            assert int(match.nonzero()[0]) in reachable


def test_curve_grouping_starts_per_sample() -> None:
    x, _, batch = _cloud(num_points=30, batch_size=2, channels=8)
    table = torch.cat([_ring_table(30, 4), _ring_table(30, 4) + 30])
    grouping = CurveGrouping(8, num_curves=5, curve_length=3).eval()
    assert grouping(x, batch, table).shape == (2, 5, 3, 8)


def test_curve_aggregation_shape() -> None:
    x, _, batch = _cloud(num_points=30, batch_size=2, channels=8)
    aggregation = CurveAggregation(8).eval()
    assert aggregation(x, batch, torch.randn(2, 5, 3, 8)).shape == (60, 8)


def test_attention_gate_in_unit_interval() -> None:
    gate = AttentionGate(16, 8, 4).eval()
    out = gate(torch.randn(10, 16), torch.randn(10, 8))
    assert out.shape == (10, 1) and bool(((out >= 0) & (out <= 1)).all())


def test_curve_intervention_conv_requires_paired_arguments() -> None:
    with pytest.raises(ValueError, match="num_points"):
        CurveInterventionConv(8, 8, num_neighbors=4, num_points=8)
    with pytest.raises(ValueError, match="num_curves"):
        CurveInterventionConv(8, 8, num_neighbors=4, num_curves=4)


@_REQUIRES_PYG_LIB
def test_radius_max_pool_takes_ball_maximum() -> None:
    x, pos, batch = _cloud(num_points=40, batch_size=2, channels=4)
    pool = RadiusMaxPool(num_points=8, radius=0.3, max_num_neighbors=40)
    out, pos_out, batch_out = pool(x, pos, batch)
    assert out.shape == (16, 4) and pos_out.shape == (16, 3)
    assert torch.equal(batch_out, torch.arange(2).repeat_interleave(8))
    for i in range(16):
        inside = (batch == batch_out[i]) & ((pos - pos_out[i]).norm(dim=1) <= 0.3)
        assert torch.allclose(out[i], x[inside].amax(dim=0))


@_REQUIRES_PYG_LIB
@pytest.mark.parametrize(
    "num_points, radius, num_curves, curve_length, out_points",
    [
        pytest.param(None, None, None, None, 64, id="plain"),
        pytest.param(None, None, 10, 4, 64, id="curves"),
        pytest.param(16, 0.5, 10, 4, 16, id="pool-curves"),
    ],
)
def test_curve_intervention_conv_forward(
    num_points: int | None,
    radius: float | None,
    num_curves: int | None,
    curve_length: int | None,
    out_points: int,
) -> None:
    x, pos, batch = _cloud(num_points=64, batch_size=2, channels=16)
    block = CurveInterventionConv(
        16,
        32,
        num_neighbors=8,
        num_points=num_points,
        radius=radius,
        num_curves=num_curves,
        curve_length=curve_length,
    ).eval()
    out, pos_out, batch_out = block(x, pos, batch)
    assert out.shape == (2 * out_points, 32)
    assert pos_out.shape == (2 * out_points, 3) and batch_out.shape == (2 * out_points,)


@_REQUIRES_PYG_LIB
def test_curvenet_feature_propagation_forward() -> None:
    x, pos, batch = _cloud(num_points=8, batch_size=2, channels=32)
    x_skip, pos_skip, batch_skip = _cloud(num_points=32, batch_size=2, channels=16)
    fp = CurveNetFeaturePropagation(32, 16, channels=[24, 24], gate_channels=8).eval()
    out, pos_out, batch_out = fp(x, pos, batch, x_skip, pos_skip, batch_skip)
    assert out.shape == (64, 24) and pos_out is pos_skip and batch_out is batch_skip
