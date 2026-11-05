import torch

from torch_pointcloud.layers.attention_gate import AttentionGate


def test_attention_gate_in_unit_interval() -> None:
    gate = AttentionGate(16, 8, 4).eval()
    out = gate(torch.randn(10, 16), torch.randn(10, 8))
    assert out.shape == (10, 1) and bool(((out >= 0) & (out <= 1)).all())
