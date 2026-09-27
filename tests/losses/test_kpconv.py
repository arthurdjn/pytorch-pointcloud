from typing import Tuple

import pytest
import torch
from torch import Tensor

from torch_pointcloud.layers.kpconv_blocks import KPConv
from torch_pointcloud.losses import KPConvDeformRegularizer, kpconv_deform_regularizer


def _conv(deformable: bool) -> KPConv:
    torch.manual_seed(0)
    return KPConv(
        spatial_dim=3, in_channels=4, out_channels=8, kernel_size=15, kp_radius=1.0, kp_sigma=1.0, deformable=deformable
    )


def _graph(num_points: int = 32, k: int = 4) -> Tuple[Tensor, Tensor, Tensor]:
    torch.manual_seed(0)
    pos = torch.rand(num_points, 3)
    x = torch.randn(num_points, 4)
    target = torch.arange(num_points).repeat_interleave(k)
    source = torch.randint(0, num_points, (num_points * k,))
    return x, pos, torch.stack([source, target])


def test_kpconv_deform_regularizer_is_zero_without_deformable_layers() -> None:
    conv = _conv(deformable=False)
    x, pos, edge_index = _graph()
    conv(x, pos, edge_index)
    loss = kpconv_deform_regularizer(conv)
    assert loss.shape == () and loss.item() == 0.0


def test_kpconv_deform_regularizer_requires_a_forward_pass() -> None:
    with pytest.raises(RuntimeError, match="forward pass"):
        kpconv_deform_regularizer(_conv(deformable=True))


def test_kpconv_deform_regularizer_matches_manual_terms_and_backpropagates() -> None:
    conv = _conv(deformable=True)
    x, pos, edge_index = _graph()
    conv(x, pos, edge_index)
    assert conv.running_min_d2 is not None and conv.running_deformed_kernel is not None
    assert conv.running_min_d2.shape == (32, 15) and conv.running_deformed_kernel.shape == (32, 15, 3)

    fitting = conv.running_min_d2.mean()
    kp = conv.running_deformed_kernel
    d = torch.cdist(kp, kp)
    overlap = torch.clamp_max(d - 1.0, 0.0) ** 2
    overlap = overlap - torch.diag_embed(torch.diagonal(overlap, dim1=-2, dim2=-1))
    expected = 2 * fitting + overlap.sum(-1).mean()
    loss = kpconv_deform_regularizer(conv)
    assert torch.allclose(loss, expected, atol=1e-6)

    loss.backward()
    assert conv.offset_conv is not None
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in conv.offset_conv.parameters())


def test_kpconv_running_min_d2_is_the_closest_neighbor_per_kernel_point() -> None:
    conv = _conv(deformable=True)
    x, pos, edge_index = _graph(num_points=8, k=3)
    with torch.no_grad():
        conv(x, pos, edge_index)
        source, target = edge_index
        assert conv.running_deformed_kernel is not None and conv.running_min_d2 is not None
        kernel_points = conv.running_deformed_kernel  # (N, K, 3)
        rel = pos[source] - pos[target]  # (E, 3)
        d2 = ((rel[:, None, :] - kernel_points[target]) ** 2).sum(-1)  # (E, K)
        expected = torch.full((8, 15), float("inf")).scatter_reduce(
            0, target[:, None].expand(-1, 15), d2, reduce="amin"
        )
    assert torch.allclose(conv.running_min_d2, expected, atol=1e-6)


def test_kpconv_deform_regularizer_module_matches_function() -> None:
    conv = _conv(deformable=True)
    x, pos, edge_index = _graph()
    conv(x, pos, edge_index)
    criterion = KPConvDeformRegularizer(fitting_power=0.5, repulse_extent=1.2)
    assert torch.equal(criterion(conv), kpconv_deform_regularizer(conv, fitting_power=0.5, repulse_extent=1.2))
    assert "repulse_extent=1.2" in repr(criterion)
