import pytest
import torch
from torch import nn

from torch_pointcloud.layers.tnet import DynamicTNet, TNet
from torch_pointcloud.losses import TNetOrthogonalityRegularizer, tnet_orthogonality_regularizer
from torch_pointcloud.utils.imports import _PYG_LIB_AVAILABLE


def _tnet(k: int, track_running_stats: bool = True) -> TNet:
    torch.manual_seed(0)
    return TNet(local_channels=[16, 32], global_channels=[16], k=k, track_running_stats=track_running_stats)


def _batch(num_scenes: int = 2, points_per_scene: int = 10) -> torch.Tensor:
    return torch.arange(num_scenes).repeat_interleave(points_per_scene)


def test_tnet_regularizer_is_zero_for_identity_transforms() -> None:
    """A freshly built T-Net predicts the identity (zero-initialized head), whose residual is exactly zero."""
    tnet = _tnet(k=8)
    tnet(torch.randn(20, 8), _batch())
    assert tnet.running_transform is not None and tnet.running_transform.shape == (2, 8, 8)
    assert tnet_orthogonality_regularizer(tnet).item() == 0.0


def test_tnet_regularizer_matches_manual_frobenius_norm_and_backpropagates() -> None:
    tnet = _tnet(k=8)
    with torch.no_grad():
        tnet.transform.weight.normal_(std=0.1)
    tnet(torch.randn(20, 8), _batch())
    transform = tnet.running_transform
    assert transform is not None and transform.shape == (2, 8, 8)
    expected = torch.norm(torch.eye(8) - transform @ transform.transpose(1, 2), dim=(1, 2)).mean()
    loss = tnet_orthogonality_regularizer(tnet)
    assert torch.allclose(loss, expected)
    loss.backward()
    assert tnet.transform.weight.grad is not None and tnet.transform.weight.grad.abs().sum() > 0


def test_tnet_regularizer_sums_over_the_tnets_of_the_given_module() -> None:
    stnet, ftnet = _tnet(k=3), _tnet(k=8)
    model = nn.ModuleList([stnet, ftnet])
    for layer in (stnet, ftnet):
        with torch.no_grad():
            layer.transform.weight.normal_(std=0.1)
        layer(torch.randn(20, layer.k), _batch())

    both = tnet_orthogonality_regularizer(model)
    assert torch.allclose(both, tnet_orthogonality_regularizer(stnet) + tnet_orthogonality_regularizer(ftnet))
    assert both > tnet_orthogonality_regularizer(ftnet)  # a sub-module restricts it, e.g. to the feature transform


def test_tnet_regularizer_requires_a_forward_pass() -> None:
    with pytest.raises(RuntimeError, match="forward pass"):
        tnet_orthogonality_regularizer(_tnet(k=8))


def test_tnet_regularizer_without_tracking_keeps_no_transform() -> None:
    tnet = _tnet(k=8, track_running_stats=False)
    tnet(torch.randn(20, 8), _batch())
    assert tnet.running_transform is None


@pytest.mark.skipif(not _PYG_LIB_AVAILABLE, reason="DynamicTNet's kNN needs pyg-lib")
def test_tnet_regularizer_module_and_dynamic_tnet() -> None:
    torch.manual_seed(0)
    tnet = DynamicTNet(edge_channels=[16], local_channels=[32], global_channels=[16], k=6, num_neighbors=4)
    with torch.no_grad():
        tnet.transform.weight.normal_(std=0.1)
    tnet(torch.randn(24, 6), _batch(points_per_scene=12))
    criterion = TNetOrthogonalityRegularizer()
    assert torch.allclose(criterion(tnet), tnet_orthogonality_regularizer(tnet))
