"""Tests for inferer shared utilities."""

from typing import Any, Dict

import pytest
import torch

from torch_pointcloud.inferers._utils import (
    check_batch_alignment,
    gaussian_weights,
    index_select_dict,
    split_chunks,
)


def test_split_chunks_no_cap_returns_single_in_order_chunk() -> None:
    """`max_size=None` yields one chunk holding every index in order."""
    chunks = split_chunks(10, None, torch.Generator())
    assert len(chunks) == 1
    assert torch.equal(chunks[0], torch.arange(10))


def test_split_chunks_below_cap_returns_single_chunk() -> None:
    """A count within the cap is not split."""
    chunks = split_chunks(5, 20, torch.Generator())
    assert len(chunks) == 1
    assert torch.equal(chunks[0], torch.arange(5))


def test_split_chunks_above_cap_partitions_within_cap() -> None:
    """A count above the cap splits into ceil(n / max_size) chunks, each within the
    cap, together covering every index exactly once."""
    chunks = split_chunks(64, 20, torch.Generator().manual_seed(0))
    assert len(chunks) == 4  # ceil(64 / 20)
    assert sorted(c.numel() for c in chunks) == [4, 20, 20, 20]
    assert all(c.numel() <= 20 for c in chunks)
    assert torch.equal(torch.cat(chunks).sort().values, torch.arange(64))


def test_split_chunks_same_seed_is_reproducible() -> None:
    """The same generator seed produces the same partition."""
    chunks_a = split_chunks(64, 20, torch.Generator().manual_seed(7))
    chunks_b = split_chunks(64, 20, torch.Generator().manual_seed(7))
    assert all(torch.equal(a, b) for a, b in zip(chunks_a, chunks_b))


def test_index_select_dict_slices_matching_tensors_only() -> None:
    """Tensors whose first dim equals `n_points` are indexed; tensors with a different
    leading dim and non-tensor entries pass through unchanged."""
    data: Dict[str, Any] = {
        "pos": torch.arange(12).reshape(4, 3),
        "label": torch.tensor([10, 11]),  # first dim 2 != 4: passthrough
        "name": "scene",  # non-tensor: passthrough
    }
    out = index_select_dict(data, torch.tensor([0, 2]), n_points=4)
    assert torch.equal(out["pos"], torch.tensor([[0, 1, 2], [6, 7, 8]]))
    assert torch.equal(out["label"], data["label"])
    assert out["name"] == "scene"


def test_gaussian_weights_matches_closed_form() -> None:
    """Weights equal exp(-d^2 / 2 sigma^2) for a scalar sigma."""
    d = torch.tensor([0.0, 1.0, 3.0])
    sigma = 1.5
    expected = torch.exp(-0.5 * (d / sigma) ** 2)
    assert torch.allclose(gaussian_weights(d, sigma=sigma), expected)


def test_gaussian_weights_accepts_per_row_tensor_sigma() -> None:
    """A tensor sigma broadcastable against `distances` applies a per-row radius:
    a wider sigma yields a larger weight at the same distance."""
    d = torch.ones(2, 2)
    sigma = torch.tensor([[1.0], [4.0]])  # row 0 narrow, row 1 wide
    w = gaussian_weights(d, sigma=sigma)
    assert w.shape == d.shape
    assert (w[1] > w[0]).all()


def test_gaussian_weights_zero_sigma_stays_finite() -> None:
    """A zero sigma is floored by `eps`, so weights stay finite instead of dividing
    by zero."""
    w = gaussian_weights(torch.tensor([0.0, 1.0]), sigma=0.0)
    assert torch.isfinite(w).all()


def test_check_batch_alignment_accepts_a_matching_index() -> None:
    """A batch index with one row per point passes."""
    check_batch_alignment(torch.zeros(6, 3), torch.zeros(6, dtype=torch.long), "pos", "batch")


def test_check_batch_alignment_rejects_a_shorter_index() -> None:
    """A batch index over voxels rather than points raises instead of leaving the tail unpredicted."""
    with pytest.raises(ValueError, match="has 4 rows but `data\\['pos'\\]` has 6"):
        check_batch_alignment(torch.zeros(6, 3), torch.zeros(4, dtype=torch.long), "pos", "batch")
