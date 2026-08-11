from typing import Any, Callable, Dict, Tuple

import pytest
import torch
from torch import Tensor

from torch_pointcloud.inferers import KNNWindowInferer, knn_window_inference
from torch_pointcloud.utils.data import DataKeys


def _identity_predictor(num_classes: int) -> Tuple[Callable[[Dict[str, Any]], Tensor], Tensor]:
    """Predictor returning the same logit vector at every point. Easy to verify
    that a weighted-mean window inference collapses to that constant."""
    target = torch.randn(num_classes)

    def predictor(window: Dict[str, Any]) -> Tensor:
        pos_w = window[DataKeys.POS]
        return target.to(pos_w.device).expand(pos_w.size(0), num_classes).contiguous()

    return predictor, target


def test_knn_window_inference_constant_weighted_mean_constant_predictor() -> None:
    """With a constant predictor and equal weights, the weighted-mean output equals the predictor value at every point."""
    torch.manual_seed(0)
    n = 1024
    data: Dict[str, Any] = {
        DataKeys.POS: torch.randn(n, 3) * 5,
        DataKeys.BATCH: torch.zeros(n, dtype=torch.long),
    }
    pred, target = _identity_predictor(num_classes=5)
    out = knn_window_inference(
        data,
        predictor=pred,
        roi_num_points=128,
        sw_batch_size=4,
        overlap=0.3,
        mode="constant",
        aggregate="weighted_mean",
        seed=42,
    )
    assert out.shape == (n, 5)
    assert torch.allclose(out, target.expand(n, 5), atol=1e-4, rtol=1e-4)


def test_knn_window_inference_gaussian_runs_and_preserves_direction() -> None:
    """Gaussian weighting runs end-to-end and preserves the sign / argmax of a
    constant predictor. Edge-of-window points get exponentially small Gaussian
    weights so the magnitude can scale down near the periphery; we only assert
    sign and argmax preservation, which is what matters for argmax-based metrics.
    """
    torch.manual_seed(0)
    n = 1024
    data: Dict[str, Any] = {
        DataKeys.POS: torch.randn(n, 3) * 5,
        DataKeys.BATCH: torch.zeros(n, dtype=torch.long),
    }
    pred, target = _identity_predictor(num_classes=4)
    out = knn_window_inference(
        data,
        predictor=pred,
        roi_num_points=128,
        sw_batch_size=2,
        mode="gaussian",
        sigma_scale=0.125,
        aggregate="weighted_mean",
        overlap=0.3,
        seed=7,
    )
    assert (torch.sign(out) == torch.sign(target).expand_as(out)).all()
    assert (out.argmax(dim=1) == int(target.argmax())).all()


def test_knn_window_inference_ema_returns_probabilities() -> None:
    """EMA aggregation accumulates softmax values, so all outputs are non-negative and finite."""
    torch.manual_seed(0)
    n = 512
    data: Dict[str, Any] = {
        DataKeys.POS: torch.randn(n, 3) * 5,
        DataKeys.BATCH: torch.zeros(n, dtype=torch.long),
    }

    def predictor(window: Dict[str, Any]) -> Tensor:
        pos_w = window[DataKeys.POS]
        return torch.randn(pos_w.size(0), 6, device=pos_w.device)

    out = knn_window_inference(
        data,
        predictor=predictor,
        roi_num_points=64,
        sw_batch_size=1,
        overlap=0.95,
        mode="constant",
        aggregate="ema",
        ema_smoothing=0.5,
        seed=0,
    )
    assert (out >= 0).all()
    assert torch.isfinite(out).all()


def test_knn_window_inference_propagates_extra_per_point_keys() -> None:
    """Per-point keys in `data` (color, intensity, ...) are sliced to the active
    window automatically so the predictor can use them."""
    torch.manual_seed(0)
    n = 256
    data: Dict[str, Any] = {
        DataKeys.POS: torch.randn(n, 3) * 3,
        DataKeys.COLOR: torch.randn(n, 3),
        DataKeys.BATCH: torch.zeros(n, dtype=torch.long),
        "scene_name": "synthetic",
    }

    def predictor(window: Dict[str, Any]) -> Tensor:
        pos = window[DataKeys.POS]
        color = window[DataKeys.COLOR]
        assert window["scene_name"] == "synthetic"
        return (pos.sum(dim=1, keepdim=True) + color.sum(dim=1, keepdim=True)).expand(-1, 3).contiguous()

    out = knn_window_inference(data, predictor=predictor, roi_num_points=64, sw_batch_size=2, overlap=0.3, seed=1)
    assert out.shape == (n, 3)
    assert torch.isfinite(out).all()


def test_knn_window_inference_validates_args() -> None:
    """Invalid arguments raise the appropriate error before any computation starts."""
    pos = torch.zeros(4, 3)
    batch = torch.zeros(4, dtype=torch.long)

    def fake(_window: Dict[str, Any]) -> Tensor:
        return torch.zeros(4, 2)

    data: Dict[str, Any] = {DataKeys.POS: pos, DataKeys.BATCH: batch}

    with pytest.raises(KeyError, match="pos"):
        knn_window_inference({DataKeys.BATCH: batch}, predictor=fake)
    with pytest.raises(KeyError, match="batch"):
        knn_window_inference({DataKeys.POS: pos}, predictor=fake)
    with pytest.raises(ValueError, match="`overlap`"):
        knn_window_inference(data, predictor=fake, overlap=1.5)
    with pytest.raises(ValueError, match="`overlap`"):
        knn_window_inference(data, predictor=fake, overlap=0.0)
    with pytest.raises(ValueError, match="`mode`"):
        knn_window_inference(data, predictor=fake, mode="other")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="`aggregate`"):
        knn_window_inference(data, predictor=fake, aggregate="vote")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="`sw_batch_size`"):
        knn_window_inference(data, predictor=fake, sw_batch_size=0)
    with pytest.raises(ValueError, match="`ema_smoothing`"):
        knn_window_inference(data, predictor=fake, aggregate="ema", ema_smoothing=1.5)


def test_knn_window_overlap_zero_raises_before_predicting() -> None:
    """A zero coverage threshold would end the loop before the first window, so it is rejected."""
    torch.manual_seed(0)
    n = 100
    data: Dict[str, Any] = {
        DataKeys.POS: torch.randn(n, 3),
        DataKeys.BATCH: torch.zeros(n, dtype=torch.long),
    }

    def predictor(window: Dict[str, Any]) -> Tensor:
        return torch.zeros(window[DataKeys.POS].size(0), 2)

    with pytest.raises(ValueError, match="`overlap`"):
        KNNWindowInferer(roi_num_points=32, overlap=0.0)(data, predictor=predictor)


def test_knn_window_gaussian_small_sigma_divides_by_true_weight() -> None:
    """The gaussian weighted mean divides by the true accumulated weight, so a constant predictor
    is recovered exactly at every point even when edge-of-window float32 gaussian weights underflow
    under a sharp `sigma_scale`."""
    torch.manual_seed(0)
    n = 1024
    data: Dict[str, Any] = {
        DataKeys.POS: torch.randn(n, 3) * 5,
        DataKeys.BATCH: torch.zeros(n, dtype=torch.long),
    }
    pred, target = _identity_predictor(num_classes=4)
    out = knn_window_inference(
        data,
        predictor=pred,
        roi_num_points=128,
        sw_batch_size=2,
        mode="gaussian",
        sigma_scale=0.05,
        aggregate="weighted_mean",
        overlap=0.3,
        seed=7,
    )
    assert torch.allclose(out, target.expand(n, 4), atol=1e-4, rtol=1e-4)


def test_knn_window_gaussian_with_ema_raises() -> None:
    """Gaussian distance weights do not apply to EMA updates, so the combination is rejected."""
    data: Dict[str, Any] = {
        DataKeys.POS: torch.zeros(4, 3),
        DataKeys.BATCH: torch.zeros(4, dtype=torch.long),
    }

    def fake(_window: Dict[str, Any]) -> Tensor:
        return torch.zeros(4, 2)

    with pytest.raises(ValueError, match="gaussian"):
        knn_window_inference(data, predictor=fake, mode="gaussian", aggregate="ema")
    with pytest.raises(ValueError, match="gaussian"):
        KNNWindowInferer(mode="gaussian", aggregate="ema")(data, predictor=fake)


def test_knn_window_inferer_class_matches_function() -> None:
    """The `KNNWindowInferer` class is a thin wrapper around `knn_window_inference`.
    Same seed and parameters must give bit-for-bit identical outputs."""
    torch.manual_seed(0)
    n = 256
    data: Dict[str, Any] = {
        DataKeys.POS: torch.randn(n, 3) * 3,
        DataKeys.BATCH: torch.zeros(n, dtype=torch.long),
    }

    def predictor(window: Dict[str, Any]) -> Tensor:
        pos = window[DataKeys.POS]
        return torch.ones(pos.size(0), 3, device=pos.device) * pos.mean()

    kwargs: Dict[str, Any] = dict(
        roi_num_points=64, sw_batch_size=2, overlap=0.3, seed=5, mode="constant", aggregate="weighted_mean"
    )
    out_fn = knn_window_inference(data, predictor=predictor, **kwargs)
    out_cls = KNNWindowInferer(**kwargs)(data, predictor=predictor)
    assert torch.equal(out_fn, out_cls)


def test_knn_window_softmax_averages_probabilities() -> None:
    """`softmax=True` converts each window's logits to probabilities before the weighted mean,
    so output rows sum to 1 regardless of the logit scale."""
    torch.manual_seed(0)
    n = 256
    data: Dict[str, Any] = {
        DataKeys.POS: torch.randn(n, 3) * 3,
        DataKeys.BATCH: torch.zeros(n, dtype=torch.long),
    }

    def predictor(window: Dict[str, Any]) -> Tensor:
        return torch.randn(window[DataKeys.POS].size(0), 4) * 10.0

    out = knn_window_inference(data, predictor=predictor, roi_num_points=64, overlap=0.3, softmax=True, seed=1)
    sums = out.sum(dim=1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)


def test_knn_window_empty_scene_returns_zero_by_zero() -> None:
    """With $N = 0$ the predictor is never called and the output is a $(0, 0)$ tensor."""
    data: Dict[str, Any] = {
        DataKeys.POS: torch.zeros(0, 3),
        DataKeys.BATCH: torch.zeros(0, dtype=torch.long),
    }

    def predictor(window: Dict[str, Any]) -> Tensor:
        raise AssertionError("predictor must not be called for an empty scene")

    out = knn_window_inference(data, predictor=predictor, roi_num_points=8)
    assert out.shape == (0, 0)
