from typing import Any, Dict, List
from unittest.mock import Mock

import pytest
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


def _merge_duplicates(data: Dict[str, Any]) -> Dict[str, Any]:
    pos, inverse = torch.unique(data[DataKeys.POS], dim=0, return_inverse=True)
    return {**data, DataKeys.POS: pos, DataKeys.INVERSE: inverse}


def test_simple_inferer_transform_gathers_predictions_back_through_the_inverse() -> None:
    """A row-altering `transform` runs before the predictor and its inverse map reads the output back per point."""
    pos = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    data: Dict[str, Any] = {
        DataKeys.POS: pos,
        DataKeys.BATCH: torch.zeros(5, dtype=torch.long),
        DataKeys.INVERSE: torch.zeros(5, dtype=torch.long),
    }
    calls: List[Dict[str, Any]] = []

    def predictor(d: Dict[str, Any]) -> Tensor:
        calls.append(d)
        return d[DataKeys.POS][:, :1] * 10.0

    out = SimpleInferer(transform=_merge_duplicates, inverse_key=DataKeys.INVERSE)(data, predictor=predictor)
    assert calls[0][DataKeys.POS].shape == (3, 3)
    assert DataKeys.INVERSE not in calls[0]
    torch.testing.assert_close(out, pos[:, :1] * 10.0)


def test_simple_inferer_transform_runs_on_every_cloud_alone() -> None:
    """Two identical clouds are transformed one by one, predicted in one call and read back in input order."""
    cloud = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    data: Dict[str, Any] = {DataKeys.POS: torch.cat([cloud, cloud]), DataKeys.BATCH: torch.tensor([1, 0, 1, 0, 1, 0])}
    calls: List[Dict[str, Any]] = []

    def predictor(d: Dict[str, Any]) -> Tensor:
        calls.append(d)
        return torch.stack([d[DataKeys.BATCH].float(), d[DataKeys.POS][:, 0]], dim=1)

    out = SimpleInferer(transform=_merge_duplicates, inverse_key=DataKeys.INVERSE)(data, predictor=predictor)
    assert len(calls) == 1
    assert calls[0][DataKeys.BATCH].tolist() == [0, 0, 1, 1]
    torch.testing.assert_close(out[:, 0], torch.tensor([1.0, 0.0, 1.0, 0.0, 1.0, 0.0]))
    torch.testing.assert_close(out[:, 1], data[DataKeys.POS][:, 0])


def test_simple_inferer_rejects_a_row_altering_transform_without_inverse_key() -> None:
    data: Dict[str, Any] = {DataKeys.POS: torch.zeros(4, 3), DataKeys.BATCH: torch.zeros(4, dtype=torch.long)}
    with pytest.raises(ValueError, match="without recording an index map"):
        SimpleInferer(transform=_merge_duplicates)(data, predictor=lambda d: d[DataKeys.POS])


def test_simple_inferer_transform_returns_an_empty_tensor_on_an_empty_batch() -> None:
    data: Dict[str, Any] = {DataKeys.POS: torch.zeros(0, 3), DataKeys.BATCH: torch.zeros(0, dtype=torch.long)}
    predictor = Mock()
    out = SimpleInferer(transform=_merge_duplicates, inverse_key=DataKeys.INVERSE)(data, predictor=predictor)
    assert out.shape == (0, 0)
    predictor.assert_not_called()
