from typing import Any, Callable, Dict, List, Tuple

import pytest
import torch
from torch import Tensor

from torch_pointcloud.inferers import Inferer


def test_inferer_is_abstract_and_cannot_be_instantiated() -> None:
    """Inferer cannot be instantiated directly -- it is an abstract base class."""
    with pytest.raises(TypeError):
        Inferer()  # type: ignore[abstract]


def test_subclass_without_forward_cannot_be_instantiated() -> None:
    """A subclass that omits `forward` stays abstract and raises TypeError on instantiation."""

    class Incomplete(Inferer):
        pass

    with pytest.raises(TypeError):
        Incomplete()  # type: ignore[abstract]


def test_call_delegates_to_forward_with_same_args_and_returns_its_value() -> None:
    """Calling an inferer instance invokes `forward` with the same arguments and returns its output."""
    calls: List[Tuple[Dict[str, Any], Callable[[Dict[str, Any]], Tensor]]] = []

    class Spy(Inferer):
        def forward(self, data: Dict[str, Any], predictor: Callable[[Dict[str, Any]], Tensor]) -> Tensor:
            calls.append((data, predictor))
            return predictor(data)

    def predictor(data: Dict[str, Any]) -> Tensor:
        return data["x"] * 2

    data: Dict[str, Any] = {"x": torch.arange(4).float()}
    out = Spy()(data, predictor=predictor)

    assert len(calls) == 1
    assert calls[0][0] is data
    assert calls[0][1] is predictor
    assert torch.equal(out, torch.arange(4).float() * 2)
