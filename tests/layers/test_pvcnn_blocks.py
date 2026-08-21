"""Tests for the PVCNN voxelization block."""

import torch

from torch_pointcloud.layers.pvcnn_blocks import Voxelization


def test_voxelization_single_point_cloud() -> None:
    vox = Voxelization(resolution=4, normalize=True)
    features, coords = vox(torch.randn(1, 2), torch.randn(1, 3), torch.zeros(1, dtype=torch.long))
    assert features.shape == (1, 2, 4, 4, 4)
    assert coords.shape == (1, 3)


def test_voxelization_batch_shapes() -> None:
    vox = Voxelization(resolution=4, normalize=True)
    batch = torch.repeat_interleave(torch.arange(2), 16)
    features, coords = vox(torch.randn(32, 2), torch.randn(32, 3), batch)
    assert features.shape == (2, 2, 4, 4, 4)
