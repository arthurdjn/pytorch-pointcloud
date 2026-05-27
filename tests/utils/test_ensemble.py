import pytest
import torch

from torch_pointcloud.utils.ensemble import MeanEnsemble, VoteEnsemble, mean_ensemble, vote_ensemble


def test_mean_ensemble_averages_outputs() -> None:
    outputs = [
        torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        torch.tensor([[0.0, 1.0], [1.0, 0.0]]),
        torch.tensor([[0.4, 0.6], [0.6, 0.4]]),
    ]
    out = mean_ensemble(outputs)
    expected = torch.tensor(
        [
            [(1.0 + 0.0 + 0.4) / 3.0, (0.0 + 1.0 + 0.6) / 3.0],
            [(0.0 + 1.0 + 0.6) / 3.0, (1.0 + 0.0 + 0.4) / 3.0],
        ]
    )
    assert torch.allclose(out, expected)


def test_mean_ensemble_single_output_is_identity() -> None:
    x = torch.randn(8, 5)
    assert torch.allclose(mean_ensemble([x]), x)


def test_vote_ensemble_counts_argmax_per_class() -> None:
    """Three votes [argmax=0, argmax=0, argmax=1] across 3 classes."""
    outputs = [
        torch.tensor([[1.0, 0.0, 0.0]]),
        torch.tensor([[0.9, 0.1, 0.0]]),
        torch.tensor([[0.0, 1.0, 0.0]]),
    ]
    out = vote_ensemble(outputs, num_classes=3)
    assert torch.equal(out, torch.tensor([[2.0, 1.0, 0.0]]))


def test_vote_ensemble_argmax_yields_majority_label() -> None:
    outputs = [
        torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        torch.tensor([[0.0, 1.0], [1.0, 0.0]]),
    ]
    votes = vote_ensemble(outputs, num_classes=2)
    assert torch.equal(votes.argmax(dim=-1), torch.tensor([0, 1]))


def test_mean_class_matches_functional() -> None:
    outputs = [torch.randn(16, 4) for _ in range(5)]
    assert torch.allclose(MeanEnsemble()(outputs), mean_ensemble(outputs))


def test_vote_class_matches_functional() -> None:
    outputs = [torch.randn(16, 4) for _ in range(5)]
    assert torch.equal(VoteEnsemble(num_classes=4)(outputs), vote_ensemble(outputs, num_classes=4))


def test_mean_and_vote_share_argmax_when_unanimous() -> None:
    """When every vote agrees, mean and vote argmax must match."""
    template = torch.tensor([[0.1, 0.7, 0.2]])
    outputs = [template.clone() for _ in range(4)]
    mean_pred = mean_ensemble(outputs).argmax(dim=-1)
    vote_pred = vote_ensemble(outputs, num_classes=3).argmax(dim=-1)
    assert torch.equal(mean_pred, vote_pred)


def test_empty_inputs_raise() -> None:
    with pytest.raises(ValueError, match="at least one"):
        mean_ensemble([])
    with pytest.raises(ValueError, match="at least one"):
        vote_ensemble([], num_classes=3)
