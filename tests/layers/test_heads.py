import torch
import torch.nn as nn

from torch_pointcloud.layers.heads import create_cls_head, create_seg_head


def test_create_cls_head() -> None:
    head = create_cls_head(num_features=64, num_classes=10)
    assert isinstance(head, nn.Linear)
    assert head.in_features == 64
    assert head.out_features == 10

    x = torch.randn(8, 64)
    assert head(x).shape == (8, 10)


def test_create_cls_head_zero_classes_is_identity() -> None:
    head = create_cls_head(num_features=64, num_classes=0)
    assert isinstance(head, nn.Identity)


def test_create_seg_head() -> None:
    head = create_seg_head(dims=[64, 32, 16], num_classes=10, act="relu", norm="batch_norm1d", dropout=0.0)
    x = torch.randn(8, 64)
    out = head(x)
    assert out.shape == (8, 10)


def test_create_seg_head_zero_classes_is_identity() -> None:
    head = create_seg_head(dims=[64, 32], num_classes=0)
    assert isinstance(head, nn.Identity)


def test_create_seg_head_empty_dims_is_identity() -> None:
    head = create_seg_head(dims=[], num_classes=10)
    assert isinstance(head, nn.Identity)
