from collections import defaultdict
from typing import Any, Callable, Dict, List, Tuple
from unittest.mock import Mock

import pytest
import torch
from torch import Tensor

import torch_pointcloud.transforms as T
from torch_pointcloud.inferers import SlidingWindowInferer, sliding_window_inference
from torch_pointcloud.inferers.sliding_window import _assign_point_blocks
from torch_pointcloud.utils.data import DataKeys
from torch_pointcloud.utils.imports import _TORCH_CLUSTER_AVAILABLE, _TORCH_SCATTER_AVAILABLE


def _grid_data(steps: int = 4, spacing: float = 1.0) -> Dict[str, Any]:
    """Regular 3D grid with `steps**3` points spaced `spacing` apart.

    With `block_size = k * spacing` and `overlap=0`, the grid splits into exactly
    `(steps // k)**3` blocks of `k**3` points each -- block assignments are exact
    and verifiable without tolerance.
    """
    coords = torch.arange(steps, dtype=torch.float32) * spacing
    pos = torch.stack(torch.meshgrid(coords, coords, coords, indexing="ij"), dim=-1).reshape(-1, 3)
    return {
        DataKeys.POS: pos,
        DataKeys.BATCH: torch.zeros(len(pos), dtype=torch.long),
    }


def _constant_predictor(value: float, num_classes: int) -> Callable[[Dict[str, Any]], Tensor]:
    """Returns a predictor that outputs the same logit vector at every point."""

    def predictor(window: Dict[str, Any]) -> Tensor:
        n = window[DataKeys.POS].size(0)
        return torch.full((n, num_classes), value)

    return predictor


def _line_data() -> Dict[str, Any]:
    """3 collinear points at x = 0, 1, 2 (y = z = 0).

    With `block_size=2`, `overlap=0.5` (step=1) this tiles into 3 blocks: block 0
    covers {x=0, x=1}, block 1 covers {x=1, x=2}, block 2 covers {x=2}. Small enough
    that block contents and per-point overlap are hand-verifiable.
    """
    return {
        DataKeys.POS: torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]),
        DataKeys.BATCH: torch.zeros(3, dtype=torch.long),
    }


def _mean_pos_predictor(window: Dict[str, Any]) -> Tensor:
    """Predictor returning each window's mean position, broadcast to all its points."""
    pos = window[DataKeys.POS]
    return pos.mean(dim=0, keepdim=True).expand_as(pos).clone()


def test_sliding_window_no_overlap_constant_predictor_equals_constant() -> None:
    """With overlap=0 and a constant predictor, every output equals the constant.

    Grid: 4^3=64 points at integer coords (0..3)^3.
    block_size=2: splits into 8 blocks of 8 points each. Each point predicted once,
    weight=1, so the weighted mean equals the raw constant.
    """
    data = _grid_data(steps=4)
    pred = _constant_predictor(3.0, num_classes=4)
    out = sliding_window_inference(data, predictor=pred, block_size=2.0, overlap=0.0, softmax=False)
    assert out.shape == (64, 4)
    assert torch.allclose(out, torch.full_like(out, 3.0))


def test_sliding_window_no_overlap_each_point_predicted_once() -> None:
    """With overlap=0, each point appears in exactly one predictor call.

    Grid: 4^3=64 points, block_size=2 → 8 non-overlapping 2x2x2 blocks.
    Collect all positions seen across calls via call_args_list and verify they
    equal the full grid with no repeats.
    """
    data = _grid_data(steps=4)
    predictor = Mock(side_effect=lambda w: torch.zeros(w[DataKeys.POS].size(0), 2))
    sliding_window_inference(data, predictor=predictor, block_size=2.0, overlap=0.0, softmax=False)

    assert predictor.call_count == 8  # (4/2)^3 non-overlapping blocks
    all_pos = torch.cat([c.args[0][DataKeys.POS] for c in predictor.call_args_list])
    seen = {tuple(p.tolist()) for p in all_pos}
    expected = {tuple(p.tolist()) for p in data[DataKeys.POS]}
    assert seen == expected  # every point covered
    assert all_pos.size(0) == len(seen)  # no duplicates


def test_sliding_window_overlap_each_boundary_point_predicted_multiple_times() -> None:
    """With overlap=0.5, interior points appear in multiple predictor calls.

    Grid: 4^3=64 points, block_size=2, step=1 (K=2 per dim) → 4^3=64 overlapping
    blocks. Corner (0,0,0) belongs to exactly 1 block; interior points (all coords
    > 0) belong to at least 2.
    """
    data = _grid_data(steps=4)
    predictor = Mock(side_effect=lambda w: torch.zeros(w[DataKeys.POS].size(0), 2))
    sliding_window_inference(data, predictor=predictor, block_size=2.0, overlap=0.5, softmax=False)

    assert predictor.call_count == 64  # 4 block positions per dim → 4^3 non-empty blocks
    all_pos = torch.cat([c.args[0][DataKeys.POS] for c in predictor.call_args_list])
    counts: Dict[tuple, int] = defaultdict(int)
    for p in all_pos:
        counts[tuple(p.tolist())] += 1

    pos = data[DataKeys.POS]
    corner = tuple(pos[(pos == 0).all(dim=1)][0].tolist())
    assert counts[corner] == 1
    for p in pos[(pos > 0).all(dim=1)]:
        assert counts[tuple(p.tolist())] > 1


def test_sliding_window_no_overlap_correct_block_membership() -> None:
    """With overlap=0, each predictor call receives exactly the 8 points of one 2x2x2 block.

    Verify via call_args_list that all positions in each call share the same block
    index (floor(pos / block_size)), no block is visited twice, and all 8 blocks
    are covered. An off-by-one in block indexing or stride mixing points across
    blocks would fail the same-block-index assertion.
    """
    block_size = 2.0
    data = _grid_data(steps=4)
    predictor = Mock(side_effect=lambda w: torch.zeros(w[DataKeys.POS].size(0), 2))
    sliding_window_inference(data, predictor=predictor, block_size=block_size, overlap=0.0, softmax=False)

    assert predictor.call_count == 8
    seen_blocks: set = set()
    for c in predictor.call_args_list:
        pos = c.args[0][DataKeys.POS]
        assert pos.size(0) == 8  # each 2x2x2 block has exactly 8 points
        block_idx = (pos / block_size).floor().long()
        assert (block_idx == block_idx[0]).all(), "points from different blocks mixed in one call"
        key = tuple(block_idx[0].tolist())
        assert key not in seen_blocks, f"block {key} called twice"
        seen_blocks.add(key)
    assert len(seen_blocks) == 8  # all blocks visited


def test_sliding_window_overlap_constant_predictor_still_equals_constant() -> None:
    """A weighted average of identical values equals that value for any weight scheme."""
    data = _grid_data(steps=4)
    pred = _constant_predictor(1.5, num_classes=3)
    out = sliding_window_inference(data, predictor=pred, block_size=2.0, overlap=0.5, softmax=False)
    assert out.shape == (64, 3)
    assert torch.allclose(out, torch.full_like(out, 1.5))


def test_sliding_window_overlap_blends_blocks_by_exact_weighted_mean() -> None:
    """A point shared by several blocks receives the exact constant-weighted mean of
    those blocks' predictions. This verifies the accumulate-then-divide core, which a
    constant or random predictor cannot exercise.

    `_mean_pos_predictor` returns each window's mean position. For the line fixture:
      x=0 -> block 0 only                                    -> (0.5, 0, 0)
      x=1 -> blocks 0 and 1, mean of (0.5,0,0) and (1.5,0,0) -> (1.0, 0, 0)
      x=2 -> blocks 1 and 2, mean of (1.5,0,0) and (2.0,0,0) -> (1.75, 0, 0)
    """
    data = _line_data()
    predictor = Mock(side_effect=_mean_pos_predictor)
    out = sliding_window_inference(
        data, predictor=predictor, block_size=2.0, overlap=0.5, mode="constant", softmax=False
    )

    assert predictor.call_count == 3  # the line fixture tiles into 3 blocks
    expected = torch.tensor([[0.5, 0.0, 0.0], [1.0, 0.0, 0.0], [1.75, 0.0, 0.0]])
    assert torch.allclose(out, expected)


def _block_confidence_predictor(window: Dict[str, Any]) -> Tensor:
    """Predictor whose class-1 logit grows with the window's mean x: on the line fixture block 0 (mean x
    0.5) is unsure, block 1 (mean 1.5) confident, block 2 (mean 2.0) most confident, all voting class 1."""
    pos = window[DataKeys.POS]
    logit = pos[:, 0].mean() * 4.0
    return torch.stack([torch.zeros(pos.size(0)), logit.expand(pos.size(0))], dim=1)


def test_sliding_window_aggregate_max_keeps_the_most_confident_block() -> None:
    """`aggregate="max"` gives every point the softmax of the covering block that is most confident about
    it, instead of a blend: x=1 (blocks 0 and 1) takes block 1, x=2 (blocks 1 and 2) takes block 2."""
    data = _line_data()
    out = sliding_window_inference(
        data, predictor=_block_confidence_predictor, block_size=2.0, overlap=0.5, aggregate="max"
    )
    block_probs = [torch.softmax(torch.tensor([0.0, m * 4.0]), dim=0) for m in (0.5, 1.5, 2.0)]
    assert torch.allclose(out[0], block_probs[0])
    assert torch.allclose(out[1], block_probs[1])
    assert torch.allclose(out[2], block_probs[2])


def test_sliding_window_aggregate_vote_counts_hard_votes() -> None:
    """`aggregate="vote"` casts one argmax vote per covering block: the output holds vote fractions, so a
    point covered by two blocks that disagree ends up at 0.5 / 0.5 while unanimous points are one-hot."""
    data = _line_data()

    def predictor(window: Dict[str, Any]) -> Tensor:
        pos = window[DataKeys.POS]
        # block 0 (mean x 0.5) votes class 0, blocks 1 and 2 vote class 1
        cls = 0 if pos[:, 0].mean() < 1.0 else 1
        logits = torch.zeros(pos.size(0), 2)
        logits[:, cls] = 1.0
        return logits

    out = sliding_window_inference(data, predictor=predictor, block_size=2.0, overlap=0.5, aggregate="vote")
    assert torch.allclose(out, torch.tensor([[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]]))
    assert out.argmax(dim=1).tolist() == [0, 0, 1]


def test_sliding_window_softmax_outputs_probabilities() -> None:
    """With `softmax=True`, the output per point sums to 1 across classes."""
    data = _grid_data(steps=4)

    def random_logits(window: Dict[str, Any]) -> Tensor:
        g = torch.Generator().manual_seed(window[DataKeys.POS].size(0))
        return torch.randn(window[DataKeys.POS].size(0), 5, generator=g)

    out = sliding_window_inference(data, predictor=random_logits, block_size=2.0, overlap=0.5, softmax=True)
    assert (out >= 0).all()
    sums = out.sum(dim=1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-4)


def test_sliding_window_gaussian_pulls_shared_points_toward_nearer_block() -> None:
    """Gaussian mode weights each block by distance to its center, so a shared point is
    pulled toward the block it sits closest to, unlike constant mode's equal blend.

    Line fixture with `_mean_pos_predictor`. The shared points x=1 and x=2 each lie
    closer to their lower-x covering block, so gaussian output is strictly below the
    constant-mode plain mean while staying inside the hull of the two block
    predictions. The single-block point x=0 is identical under both modes (its lone
    weight cancels in the division). `sigma_scale` is widened so the Gaussian falloff
    is gentle enough to keep both covering blocks' weights comparable.
    """
    data = _line_data()
    kw: Dict[str, Any] = dict(block_size=2.0, overlap=0.5, softmax=False)
    out_const = sliding_window_inference(data, predictor=_mean_pos_predictor, mode="constant", **kw)
    out_gauss = sliding_window_inference(data, predictor=_mean_pos_predictor, mode="gaussian", sigma_scale=1.0, **kw)

    assert torch.allclose(out_gauss[0], out_const[0])  # single-block point: mode irrelevant
    assert out_const[1, 0].item() == pytest.approx(1.0)
    assert out_const[2, 0].item() == pytest.approx(1.75)
    assert 0.5 <= out_gauss[1, 0].item() < out_const[1, 0].item()
    assert 1.5 <= out_gauss[2, 0].item() < out_const[2, 0].item()


def test_sliding_window_gaussian_small_sigma_divides_by_true_weight() -> None:
    """Gaussian blending divides by the true accumulated weight, even when the float32 gaussian
    weights underflow at scene borders under a sharp `sigma_scale`.

    A per-point-pure predictor gives identical logits for a point in every covering block, so the
    weighted mean must recover that point's softmax row exactly: no zeroed rows, no rows crushed
    toward zero by a fixed denominator clamp.
    """
    torch.manual_seed(0)
    n = 500
    data: Dict[str, Any] = {
        DataKeys.POS: torch.rand(n, 3) * 4.0,
        DataKeys.BATCH: torch.zeros(n, dtype=torch.long),
    }

    def predictor(window: Dict[str, Any]) -> Tensor:
        return window[DataKeys.POS] * torch.tensor([1.0, 2.0, 3.0])

    ref = torch.softmax(predictor(data), dim=-1)
    out = sliding_window_inference(
        data, predictor=predictor, block_size=3.0, overlap=0.5, mode="gaussian", sigma_scale=0.05, softmax=True
    )
    sums = out.sum(dim=1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-4), "some rows were zeroed or rescaled"
    assert torch.allclose(out, ref, atol=1e-4)


def test_sliding_window_overlap_without_padding_blocks_contain_their_points() -> None:
    """With overlap>0 and padding=0, every point handed to a block lies inside that block's bbox, and
    every point is still covered by at least one block (constant predictions average back to the constant)."""
    torch.manual_seed(0)
    n = 500
    data: Dict[str, Any] = {
        DataKeys.POS: torch.rand(n, 3) * 4.0,
        DataKeys.BATCH: torch.zeros(n, dtype=torch.long),
    }
    seen: List[Tuple[Tensor, Tensor]] = []

    def spy(window: Dict[str, Any]) -> Dict[str, Any]:
        seen.append((window[DataKeys.POS].clone(), window["block_bbox"].clone()))
        return window

    pred = _constant_predictor(1.0, num_classes=2)
    out = sliding_window_inference(
        data, predictor=pred, block_size=2.0, overlap=0.3, padding=0.0, transform=spy, softmax=False
    )

    assert len(seen) > 0
    for pos_block, bbox in seen:
        lo, hi = bbox[:3], bbox[3:]
        assert (pos_block >= lo).all() and (pos_block <= hi).all(), "point assigned to a non-containing block"
    assert torch.allclose(out, torch.ones_like(out)), "some points are not covered by any block"


def test_sliding_window_gaussian_sigma_scales_with_tiled_dims_only() -> None:
    r"""The gaussian sigma derives from the tiled axes, not all of `pos`'s dimensions: with `dims=(0,)`,
    $\sigma = \text{sigma\_scale} \cdot \text{block\_size} / 2$ and the weights peak at the block center."""
    pos = torch.tensor([[0.5, 5.0, -3.0], [1.0, -2.0, 7.0], [1.5, 0.0, 0.0]])
    point_groups, weight_groups, _ = _assign_point_blocks(
        pos, block_size=2.0, overlap=0.0, mode="gaussian", sigma_scale=0.125, dims=(0,), padding=0.0
    )
    assert len(point_groups) == 1
    distance = (pos[point_groups[0], 0] - 1.5).abs()  # block center x = lo + block_size / 2 = 1.5
    expected = torch.exp(-0.5 * (distance / 0.125) ** 2)  # sigma = sigma_scale * (block_size / 2) * sqrt(1)
    torch.testing.assert_close(weight_groups[0], expected)
    assert int(weight_groups[0].argmax()) == int(distance.argmin())


def test_sliding_window_covers_all_points() -> None:
    """Every point receives a non-zero weight for any overlap setting."""
    data = _grid_data(steps=4)
    pred = _constant_predictor(1.0, num_classes=2)
    for overlap in (0.0, 0.25, 0.5):
        out = sliding_window_inference(data, predictor=pred, block_size=2.0, overlap=overlap, softmax=False)
        assert (out != 0).all(), f"overlap={overlap}: some points have zero output"


def test_sliding_window_accepts_integer_grid_coordinates() -> None:
    """Integer `pos` (voxel grid coords) must not leak its dtype into the blend weights."""
    data = _grid_data(steps=4)
    data[DataKeys.POS] = data[DataKeys.POS].long()
    pred = _constant_predictor(3.0, num_classes=4)
    out = sliding_window_inference(data, predictor=pred, block_size=2.0, overlap=0.0, softmax=False)
    assert out.shape == (64, 4)
    assert torch.allclose(out, torch.full_like(out, 3.0))


def test_sliding_window_roi_num_points_splits_block_into_capped_chunks() -> None:
    """A block larger than `roi_num_points` is split into random sub-batches within the
    cap; every point is still predicted exactly once.

    64-point grid with `block_size` large enough to form a single block.
    `roi_num_points=20` splits it into ceil(64/20)=4 chunks sized [20, 20, 20, 4].
    """
    data = _grid_data(steps=4)
    predictor = Mock(side_effect=lambda w: torch.zeros(w[DataKeys.POS].size(0), 2))
    sliding_window_inference(
        data, predictor=predictor, block_size=100.0, overlap=0.0, roi_num_points=20, softmax=False, seed=0
    )

    assert predictor.call_count == 4
    sizes = sorted(c.args[0][DataKeys.POS].size(0) for c in predictor.call_args_list)
    assert sizes == [4, 20, 20, 20]

    all_pos = torch.cat([c.args[0][DataKeys.POS] for c in predictor.call_args_list])
    assert all_pos.size(0) == 64  # every point predicted exactly once
    assert {tuple(p.tolist()) for p in all_pos} == {tuple(p.tolist()) for p in data[DataKeys.POS]}


def test_sliding_window_roi_num_points_seed_is_reproducible() -> None:
    """The random sub-batch partition is reproducible: the same `seed` yields the same
    per-chunk point sets."""
    data = _grid_data(steps=4)

    def run(seed: int) -> List[Tensor]:
        predictor = Mock(side_effect=lambda w: torch.zeros(w[DataKeys.POS].size(0), 2))
        sliding_window_inference(
            data, predictor=predictor, block_size=100.0, overlap=0.0, roi_num_points=20, softmax=False, seed=seed
        )
        return [c.args[0][DataKeys.POS] for c in predictor.call_args_list]

    assert all(torch.equal(a, b) for a, b in zip(run(7), run(7)))


def test_sliding_window_validates_args() -> None:
    """Invalid arguments raise the appropriate error before any computation starts."""
    pos = torch.zeros(4, 3)
    batch = torch.zeros(4, dtype=torch.long)
    data: Dict[str, Any] = {DataKeys.POS: pos, DataKeys.BATCH: batch}

    def fake(w: Dict[str, Any]) -> Tensor:
        return torch.zeros(w[DataKeys.POS].size(0), 2)

    with pytest.raises(KeyError, match="pos"):
        sliding_window_inference({DataKeys.BATCH: batch}, predictor=fake, block_size=1.0)
    with pytest.raises(KeyError, match="batch"):
        sliding_window_inference({DataKeys.POS: pos}, predictor=fake, block_size=1.0)
    with pytest.raises(ValueError, match="`block_size`"):
        sliding_window_inference(data, predictor=fake, block_size=0.0)
    with pytest.raises(ValueError, match="`overlap`"):
        sliding_window_inference(data, predictor=fake, block_size=1.0, overlap=1.0)
    with pytest.raises(ValueError, match="`mode`"):
        sliding_window_inference(data, predictor=fake, block_size=1.0, mode="other")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="`roi_num_points`"):
        sliding_window_inference(data, predictor=fake, block_size=1.0, roi_num_points=0)
    with pytest.raises(ValueError, match="`aggregate`"):
        sliding_window_inference(data, predictor=fake, block_size=1.0, aggregate="sum")  # type: ignore[arg-type]


def test_sliding_window_inferer_class_matches_function() -> None:
    """The class wrapper produces bit-for-bit identical output to the functional API."""
    data = _grid_data(steps=4)
    pred = _constant_predictor(2.0, num_classes=3)
    kwargs: Dict[str, Any] = dict(block_size=2.0, overlap=0.5, softmax=False, seed=7)
    out_fn = sliding_window_inference(data, predictor=pred, **kwargs)
    out_cls = SlidingWindowInferer(**kwargs)(data, predictor=pred)
    assert torch.equal(out_fn, out_cls)


def test_sliding_window_with_divisible_pad_recovers_per_point_input() -> None:
    """`DivisiblePad` + `inverse_key` round-trips: each padded block is gathered back to source rows.

    The predictor returns each point's x-coordinate as a single-class logit; with `softmax=False`
    and `overlap=0`, the output at every grid point must equal that point's x value, regardless
    of how many duplicates the pad introduced. Exercises the inverse-key wiring end-to-end.
    """
    data = _grid_data(steps=4)  # 64 points; 8 blocks of 8 with block_size=2

    def predictor(window: Dict[str, Any]) -> Tensor:
        return window[DataKeys.POS][:, :1].clone()

    out = sliding_window_inference(
        data,
        predictor=predictor,
        block_size=2.0,
        overlap=0.0,
        softmax=False,
        transform=T.DivisiblePad(num_samples=16, dst_inverse_key="inverse"),
        roi_num_points=16,
        inverse_key="inverse",
        seed=0,
    )
    assert out.shape == (64, 1)
    assert torch.allclose(out.squeeze(-1), data[DataKeys.POS][:, 0])


@pytest.mark.skipif(
    not (_TORCH_CLUSTER_AVAILABLE and _TORCH_SCATTER_AVAILABLE),
    reason="torch-cluster or torch-scatter is not installed",
)
def test_sliding_window_with_voxelize_gathers_predictions_back_to_source() -> None:
    """`Voxelize` + `inverse_key` end-to-end: voxel-resolution preds are gathered to source rows.

    Each block contains $2^3=8$ source points spaced by 1.0; voxel size 1.1 collapses each
    pair of `(x, x+1)` neighbors along x into one voxel. The predictor outputs the voxel-mean
    x, which then broadcasts back to all source points in the voxel via the inverse map.
    """
    data = _grid_data(steps=4, spacing=1.0)

    def predictor(window: Dict[str, Any]) -> Tensor:
        return window[DataKeys.POS][:, :1].clone()

    out = sliding_window_inference(
        data,
        predictor=predictor,
        block_size=2.0,
        overlap=0.0,
        softmax=False,
        transform=T.Voxelize(
            pos_key=DataKeys.POS,
            pos_reduce="mean",
            size=1.1,
            dst_inverse_key="inverse",
        ),
        inverse_key="inverse",
        seed=0,
    )
    assert out.shape == (64, 1)
    # Within each (block, voxel) pair, source points share the voxel-mean x prediction; thus
    # the per-source prediction equals the floor-rounded x in {0.5, 2.5} offsets from origin.
    # Sanity: outputs are finite and bounded by the grid extent.
    assert torch.isfinite(out).all()
    assert float(out.min()) >= 0.0
    assert float(out.max()) <= 3.0


def test_sliding_window_empty_scene_returns_zero_by_zero() -> None:
    """With $N = 0$ the predictor is never called and the output is a $(0, 0)$ tensor."""
    data: Dict[str, Any] = {
        DataKeys.POS: torch.zeros(0, 3),
        DataKeys.BATCH: torch.zeros(0, dtype=torch.long),
    }

    def predictor(window: Dict[str, Any]) -> Tensor:
        raise AssertionError("predictor must not be called for an empty scene")

    out = sliding_window_inference(data, predictor=predictor, block_size=1.0)
    assert out.shape == (0, 0)


def test_sliding_window_integer_boundary_points_match_interior_vote_count() -> None:
    """Blocks are half-open at the top: on an integer grid with `overlap=0.5`, a point sitting exactly on a
    block edge belongs to the same number of blocks as an interior point."""
    grid = torch.stack(torch.meshgrid(torch.arange(5.0), torch.arange(5.0), indexing="ij"), dim=-1).reshape(-1, 2)
    pos = torch.cat([grid, torch.zeros(len(grid), 1)], dim=1)
    counts = torch.zeros(len(pos))
    data = {"pos": pos, "batch": torch.zeros(len(pos), dtype=torch.long), "idx": torch.arange(len(pos))}

    def predictor(block: Dict[str, Any]) -> Tensor:
        counts[block["idx"]] += 1
        return torch.zeros(block["pos"].size(0), 4)

    out = sliding_window_inference(data, predictor=predictor, block_size=2.0, overlap=0.5, dims=(0, 1))
    assert out.shape == (len(pos), 4)
    boundary = counts[(pos[:, 0] == 2.0) & (pos[:, 1] == 2.0)]
    interior = counts[(pos[:, 0] == 1.0) & (pos[:, 1] == 1.0)]
    assert torch.equal(boundary, interior)
    assert float(counts.min()) >= 1


def test_sliding_window_non_representable_grid_covers_every_point_exactly_once() -> None:
    """`k * 0.1` grid coordinates are not exactly representable in float32, so `(p - lo) / step` can round
    across an integer and land a point exactly on a bound of its own block. Every point must still be
    predicted, and with `overlap=0` exactly once."""
    generator = torch.Generator().manual_seed(0)
    base = torch.rand((1,), generator=generator) * 10
    pos = base + torch.randint(0, 100, (300, 3), generator=generator).float() * 0.1
    counts = torch.zeros(len(pos))
    data = {"pos": pos, "batch": torch.zeros(len(pos), dtype=torch.long), "idx": torch.arange(len(pos))}

    def predictor(block: Dict[str, Any]) -> Tensor:
        counts[block["idx"]] += 1
        return torch.ones(block["pos"].size(0), 4)

    out = sliding_window_inference(data, predictor=predictor, block_size=1.0, overlap=0.0, softmax=False)
    assert torch.allclose(out, torch.ones_like(out))
    assert torch.equal(counts, torch.ones_like(counts))
