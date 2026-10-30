import torch

from torch_pointcloud.ops.math import hard_softmax


def test_hard_softmax_is_one_hot_with_softmax_gradient() -> None:
    logits = torch.tensor([[0.1, 2.0, -1.0], [3.0, 0.0, 1.0]], requires_grad=True)
    out = hard_softmax(logits)
    assert out.tolist() == [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]]
    weights = torch.randn(2, 3)
    (out * weights).sum().backward()
    soft = logits.detach().softmax(dim=-1)
    expected = soft * (weights - (soft * weights).sum(dim=-1, keepdim=True))
    assert logits.grad is not None and torch.allclose(logits.grad, expected)
