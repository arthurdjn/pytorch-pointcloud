import torch

from torch_pointcloud.layers import LogOptimalTransport


def test_log_optimal_transport_is_doubly_stochastic_on_valid_entries() -> None:
    torch.manual_seed(0)
    ot = LogOptimalTransport(num_iterations=50)
    scores = torch.randn(2, 5, 6)
    row_masks = torch.ones(2, 5, dtype=torch.bool)
    col_masks = torch.ones(2, 6, dtype=torch.bool)
    col_masks[1, -1] = False
    out = ot(scores, row_masks, col_masks)
    assert out.shape == (2, 6, 7)
    probs = out.exp()
    # Every valid point row sums to one over the valid columns and the dustbin.
    assert torch.allclose(probs[0, :5].sum(dim=1), torch.ones(5), atol=1e-3)
    assert torch.allclose(probs[1, :5, :5].sum(dim=1) + probs[1, :5, 6], torch.ones(5), atol=1e-3)
    assert torch.all(probs[1, :, 5] == 0)


def test_log_optimal_transport_without_masks_uses_every_entry() -> None:
    torch.manual_seed(0)
    ot = LogOptimalTransport(num_iterations=50)
    scores = torch.randn(1, 4, 3)
    assert torch.allclose(
        ot(scores), ot(scores, torch.ones(1, 4, dtype=torch.bool), torch.ones(1, 3, dtype=torch.bool))
    )
    assert "num_iterations=50" in repr(ot)
