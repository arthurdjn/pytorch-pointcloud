from typing import Any, Dict, List

import torch
from torch import Tensor

from torch_pointcloud.inferers import SimpleInferer
from torch_pointcloud.utils.data import DataKeys


def test_simple_inferer_calls_predictor_once_and_returns_its_output() -> None:
    """The predictor is called exactly once on the full data dict and the result is returned unchanged."""
    n = 32
    data: Dict[str, Any] = {
        DataKeys.POS: torch.randn(n, 3),
        DataKeys.BATCH: torch.zeros(n, dtype=torch.long),
    }
    expected = torch.randn(n, 5)
    calls: List[Dict[str, Any]] = []

    def predictor(d: Dict[str, Any]) -> Tensor:
        calls.append(d)
        return expected

    out = SimpleInferer()(data, predictor=predictor)
    assert len(calls) == 1
    assert calls[0] is data
    assert out is expected


def test_simple_inferer_softmax_returns_probabilities() -> None:
    """`softmax=True` softmaxes the predictor output; the default returns it untouched."""
    n = 8
    data: Dict[str, Any] = {
        DataKeys.POS: torch.randn(n, 3),
        DataKeys.BATCH: torch.zeros(n, dtype=torch.long),
    }
    logits = torch.randn(n, 5)
    out = SimpleInferer(softmax=True)(data, predictor=lambda d: logits)
    torch.testing.assert_close(out, torch.softmax(logits, dim=-1))
