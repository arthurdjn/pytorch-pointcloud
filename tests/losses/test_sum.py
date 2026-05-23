import torch
from torch import nn

from torch_pointcloud.losses import LovaszLoss, SumLoss


def test_sum_loss_adds_components() -> None:
    logits = torch.randn(8, 5)
    target = torch.randint(0, 5, (8,))
    ce, lovasz = nn.CrossEntropyLoss(), LovaszLoss()
    expected = ce(logits, target) + lovasz(logits, target)
    assert torch.allclose(SumLoss([ce, lovasz])(logits, target), expected)
