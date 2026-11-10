import torch

from torch_pointcloud.layers.attention_gate import AttentionGate


def test_attention_gate_in_unit_interval() -> None:
    gate = AttentionGate(16, 8, 4).eval()
    out = gate(torch.randn(10, 16), torch.randn(10, 8))
    assert out.shape == (10, 1) and bool(((out >= 0) & (out <= 1)).all())


def test_attention_gate_scores_sum_of_projections() -> None:
    torch.manual_seed(0)
    gate = AttentionGate(6, 3, 4, norm=None)
    g, x = torch.randn(5, 6), torch.randn(5, 3)
    joint = torch.relu(gate.gate.lins[0](g) + gate.skip.lins[0](x))
    expected = torch.sigmoid(gate.psi.lins[0](joint))
    assert torch.allclose(gate(g, x), expected, atol=1e-6)
    assert not torch.allclose(gate(g, x), gate(g, torch.zeros_like(x)))
    assert not torch.allclose(gate(g, x), gate(torch.zeros_like(g), x))
