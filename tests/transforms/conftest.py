from typing import Any, Dict

import pytest
import torch


@pytest.fixture
def sample_scene() -> Dict[str, Any]:
    """100-point single-scene dict with all standard Pointcept-style keys."""
    g = torch.Generator().manual_seed(0)
    return {
        "pos": torch.randn(100, 3, generator=g),
        "color": (torch.rand(100, 3, generator=g) * 255).to(torch.uint8),
        "normal": torch.nn.functional.normalize(torch.randn(100, 3, generator=g), dim=-1),
        "segment": torch.randint(0, 10, (100,), generator=g),
    }


@pytest.fixture
def empty_scene() -> Dict[str, Any]:
    """Empty single-scene dict (N=0) with all standard keys."""
    return {
        "pos": torch.empty(0, 3),
        "color": torch.empty(0, 3, dtype=torch.uint8),
        "normal": torch.empty(0, 3),
        "segment": torch.empty(0, dtype=torch.long),
    }


@pytest.fixture
def single_point_scene() -> Dict[str, Any]:
    """Single-point (N=1) dict with all standard keys."""
    return {
        "pos": torch.tensor([[1.0, 2.0, 3.0]]),
        "color": torch.tensor([[128, 64, 32]], dtype=torch.uint8),
        "normal": torch.tensor([[0.0, 0.0, 1.0]]),
        "segment": torch.tensor([5], dtype=torch.long),
    }
