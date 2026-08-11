from typing import Any, Callable, Dict

import pytest
import torch
from torch import Tensor

from torch_pointcloud.inferers import SimpleInferer, TTAInferer
from torch_pointcloud.transforms import Compose, RandomFlip, RandomRotate
from torch_pointcloud.utils.data import DataKeys


def _toy_data(n: int = 64, *, seed: int = 0) -> Dict[str, Any]:
    g = torch.Generator().manual_seed(seed)
    return {
        DataKeys.POS: torch.randn(n, 3, generator=g),
        DataKeys.BATCH: torch.zeros(n, dtype=torch.long),
    }


def _pos_logits(num_classes: int) -> Callable[[Dict[str, Any]], Tensor]:
    def predictor(window: Dict[str, Any]) -> Tensor:
        pos = window[DataKeys.POS]
        first = pos[:, :num_classes] if pos.size(1) >= num_classes else pos[:, :1].expand(-1, num_classes)
        return first.contiguous()

    return predictor


def test_tta_single_pass_with_identity_compose_equals_base() -> None:
    """A single TTA pass with a no-op augmentation is equivalent to the base inferer called directly."""
    data = _toy_data()
    base = SimpleInferer()
    predictor = _pos_logits(3)

    inferer = TTAInferer(base=base, transforms=Compose([]), num_passes=1, aggregate="mean")
    out_tta = inferer(data, predictor=predictor)
    out_base = base(data, predictor=predictor)
    assert torch.allclose(out_tta, out_base)


def test_tta_mean_of_identical_passes_equals_one_pass() -> None:
    """With a no-op rotation (angle range collapsed to 0), four passes produce
    four identical logits whose mean equals one pass."""
    data = _toy_data(seed=1)
    base = SimpleInferer()
    predictor = _pos_logits(4)

    noop_rot = Compose([RandomRotate(keys=DataKeys.POS, angle_range=(0.0, 0.0), axis=2, p=1.0)])
    inferer = TTAInferer(base=base, transforms=noop_rot, num_passes=4, aggregate="mean")
    out_tta = inferer(data, predictor=predictor)
    out_base = base(data, predictor=predictor)
    assert torch.allclose(out_tta, out_base, atol=1e-5)


def test_tta_enumerated_sequence_uses_each_view_once() -> None:
    """With a sequence of Composes, `num_passes` is overridden by the sequence length and the aggregate
    equals the mean of the per-view predictions (each view is deterministic, so they can be replayed)."""
    data = _toy_data(seed=2)
    base = SimpleInferer()
    predictor = _pos_logits(3)

    views = [
        Compose([RandomRotate(keys=DataKeys.POS, angle_range=(0.0, 0.0), axis=2, p=1.0)]),
        Compose([RandomRotate(keys=DataKeys.POS, angle_range=(90.0, 90.0), axis=2, p=1.0)]),
        Compose([RandomRotate(keys=DataKeys.POS, angle_range=(180.0, 180.0), axis=2, p=1.0)]),
    ]
    inferer = TTAInferer(base=base, transforms=views, aggregate="mean")
    assert inferer.num_passes == 3
    out = inferer(data, predictor=predictor)
    assert out.shape == (data[DataKeys.POS].size(0), 3)
    assert torch.isfinite(out).all()

    per_view = torch.stack([base(view(dict(data)), predictor=predictor) for view in views])
    assert torch.allclose(out, per_view.mean(dim=0), atol=1e-6)


def test_tta_flip_changes_predictions_but_preserves_shape() -> None:
    """An always-flip pass alters the per-point predictions but keeps shape."""
    data = _toy_data(seed=3)
    base = SimpleInferer()
    predictor = _pos_logits(3)

    aug = Compose([RandomFlip(keys=DataKeys.POS, axes=[0], p=1.0)])
    inferer = TTAInferer(base=base, transforms=aug, num_passes=1, aggregate="mean")
    out_tta = inferer(data, predictor=predictor)
    out_base = base(data, predictor=predictor)
    assert out_tta.shape == out_base.shape
    assert not torch.allclose(out_tta, out_base)


def test_tta_ema_aggregation_returns_probabilities() -> None:
    """EMA aggregation over softmaxed per-pass outputs always yields a valid probability distribution."""
    data = _toy_data(seed=4)
    base = SimpleInferer()

    def random_logits(window: Dict[str, Any]) -> Tensor:
        return torch.randn(window[DataKeys.POS].size(0), 5)

    aug = Compose([RandomRotate(keys=DataKeys.POS, angle_range=(0.0, 0.0), axis=2, p=1.0)])
    inferer = TTAInferer(base=base, transforms=aug, num_passes=3, aggregate="ema", ema_smoothing=0.5)
    out = inferer(data, predictor=random_logits)
    assert (out >= 0).all()
    sums = out.sum(dim=1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)


def test_tta_invalid_num_passes_raises() -> None:
    """`num_passes` must be a positive integer when `transforms` is a single callable."""
    base = SimpleInferer()
    with pytest.raises(ValueError, match="num_passes"):
        TTAInferer(base=base, transforms=Compose([]), num_passes=None)
    with pytest.raises(ValueError, match="num_passes"):
        TTAInferer(base=base, transforms=Compose([]), num_passes=0)


def test_tta_invalid_aggregate_raises() -> None:
    """`aggregate` only accepts "mean" or "ema"; unknown values raise ValueError."""
    base = SimpleInferer()
    with pytest.raises(ValueError, match="aggregate"):
        TTAInferer(base=base, transforms=Compose([]), num_passes=1, aggregate="vote")  # type: ignore[arg-type]


def test_tta_empty_sequence_raises() -> None:
    """An empty augmentation sequence has no passes to run and raises ValueError."""
    base = SimpleInferer()
    with pytest.raises(ValueError, match="at least one"):
        TTAInferer(base=base, transforms=[])


def test_tta_empty_scene_passes_through_base_output() -> None:
    """With $N = 0$ the aggregate of the base inferer's empty outputs is returned unchanged in shape."""
    data: Dict[str, Any] = {
        DataKeys.POS: torch.zeros(0, 3),
        DataKeys.BATCH: torch.zeros(0, dtype=torch.long),
    }

    def predictor(window: Dict[str, Any]) -> Tensor:
        return torch.zeros(0, 5)

    def identity(sample: Dict[str, Any]) -> Dict[str, Any]:
        return dict(sample)

    out = TTAInferer(base=SimpleInferer(), transforms=identity, num_passes=2)(data, predictor=predictor)
    assert out.shape == (0, 5)
