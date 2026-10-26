from typing import Dict

import pytest
import torch
import torch.nn as nn
from torch import Tensor

from torch_pointcloud.models.cylinder3d import Cylinder3DSegmentation
from torch_pointcloud.utils.imports import _CUDA_AVAILABLE, _SPCONV_AVAILABLE

CUDA_SPCONV = pytest.mark.skipif(
    not (_SPCONV_AVAILABLE and _CUDA_AVAILABLE),
    reason="the sparse backbone needs spconv and CUDA",
)

POINT_CLOUD_RANGE = (0.0, -3.14159265359, -4.0, 50.0, 3.14159265359, 2.0)


@pytest.fixture
def data() -> Dict[str, Tensor]:
    torch.manual_seed(0)
    lengths = torch.tensor([3000, 2000])
    num_points = int(lengths.sum())
    pos = (torch.rand(num_points, 3) * 2 - 1) * torch.tensor([40.0, 40.0, 3.0])
    return dict(
        x=torch.rand(num_points, 1).cuda(),
        pos=pos.cuda(),
        batch=torch.repeat_interleave(torch.arange(len(lengths)), lengths).cuda(),
    )


@pytest.fixture
def model() -> Cylinder3DSegmentation:
    return (
        Cylinder3DSegmentation(
            1,
            5,
            point_cloud_range=POINT_CLOUD_RANGE,
            grid_size=(96, 72, 16),
            vfe_channels=(16, 32),
            vfe_out_channels=8,
            channels=(8, 16, 32),
            strides=((2, 2, 2), (2, 2, 1)),
        )
        .cuda()
        .eval()
    )


@CUDA_SPCONV
def test_cylinder3d_forward_returns_one_row_per_point(model: Cylinder3DSegmentation, data: Dict[str, Tensor]) -> None:
    with torch.no_grad():
        logits = model(data["x"], data["pos"], data["batch"])
    assert logits.shape == (data["pos"].shape[0], 5)
    assert torch.isfinite(logits).all()


@CUDA_SPCONV
def test_cylinder3d_scenes_of_a_batch_are_independent(model: Cylinder3DSegmentation, data: Dict[str, Tensor]) -> None:
    mask = data["batch"] == 1
    with torch.no_grad():
        batched = model(data["x"], data["pos"], data["batch"])
        alone = model(data["x"][mask], data["pos"][mask], torch.zeros(int(mask.sum()), dtype=torch.long).cuda())
    assert torch.allclose(batched[mask], alone, atol=1e-4)


@CUDA_SPCONV
def test_cylinder3d_clamps_points_outside_the_range(model: Cylinder3DSegmentation, data: Dict[str, Tensor]) -> None:
    pos = data["pos"].clone()
    pos[:10] *= 100.0
    with torch.no_grad():
        logits = model(data["x"], pos, data["batch"])
    assert logits.shape == (pos.shape[0], 5)
    assert torch.isfinite(logits).all()


@CUDA_SPCONV
def test_cylinder3d_forward_head_returns_one_row_per_voxel(
    model: Cylinder3DSegmentation, data: Dict[str, Tensor]
) -> None:
    with torch.no_grad():
        sparse_x, inverse, skips = model.forward_features(
            data["x"],
            data["pos"],
            data["batch"],
            return_intermediates=True,
        )
        sparse_x = model.forward_decoder(sparse_x, skips)
        features = model.forward_head(sparse_x, pre_logits=True)
        logits = model.forward_head(sparse_x)
    assert features.shape == (sparse_x.features.shape[0], model.num_features)
    assert torch.equal(features, sparse_x.features)
    assert logits.shape == (sparse_x.features.shape[0], 5)
    assert int(inverse.max()) + 1 == logits.shape[0]


@CUDA_SPCONV
def test_cylinder3d_reset_classifier(model: Cylinder3DSegmentation, data: Dict[str, Tensor]) -> None:
    model.reset_classifier(num_classes=0)
    assert isinstance(model.head, nn.Identity)
    with torch.no_grad():
        features = model.cuda()(data["x"], data["pos"], data["batch"])
    assert features.shape == (data["pos"].shape[0], model.num_features)

    model.reset_classifier(num_classes=7)
    with torch.no_grad():
        logits = model.cuda()(data["x"], data["pos"], data["batch"])
    assert logits.shape == (data["pos"].shape[0], 7)


def test_cylinder3d_rejects_a_stride_count_that_does_not_match_the_stages() -> None:
    with pytest.raises(ValueError, match="one stride per encoder stage"):
        Cylinder3DSegmentation(
            1,
            5,
            point_cloud_range=POINT_CLOUD_RANGE,
            grid_size=(96, 72, 16),
            channels=(8, 16, 32),
            strides=((2, 2, 2),),
        )


@pytest.mark.skipif(not _SPCONV_AVAILABLE, reason="spconv is not installed")
def test_cylinder3d_grid_puts_the_last_cell_at_the_upper_bound() -> None:
    model = Cylinder3DSegmentation(
        1,
        5,
        point_cloud_range=(0.0, 0.0, 0.0, 4.0, 4.0, 4.0),
        grid_size=(5, 9, 2),
        channels=(8, 16, 32),
        strides=((2, 2, 2), (2, 2, 1)),
    )
    assert torch.equal(model.vfe.voxel_size, torch.tensor([1.0, 0.5, 4.0]))
