import os
from pathlib import Path
from typing import Dict

import pytest
import torch
from torch import Tensor

from torch_pointcloud.config import DATA_DIR, MODELS_DIR
from torch_pointcloud.models.sphereformer import SphereFormerSegmentation
from torch_pointcloud.utils.imports import (
    _CUDA_AVAILABLE,
    _SPCONV_AVAILABLE,
    _TORCH_SCATTER_AVAILABLE,
)

# See: https://docs.pytest.org/en/stable/how-to/skipping.html#summary
pytestmark = [
    pytest.mark.skipif(not _CUDA_AVAILABLE, reason="CUDA is not available"),
    pytest.mark.skipif(not _SPCONV_AVAILABLE, reason="spconv is not installed"),
    pytest.mark.skipif(not _TORCH_SCATTER_AVAILABLE, reason="torch_scatter is not installed"),
]
pytest.importorskip("sptr")

_WEIGHTS = Path(MODELS_DIR, "sphereformer", "sphereformer-dvlab.semantickitti.pt")
_SEMANTICKITTI = Path(DATA_DIR, "SemanticKITTI", "raw", "sequences", "08", "velodyne")


@pytest.fixture
def data() -> Dict[str, Tensor]:
    torch.manual_seed(42)
    lengths = torch.tensor([400, 600])
    num_points = int(lengths.sum())
    pos = (torch.rand(num_points, 3) * 2 - 1) * torch.tensor([20.0, 20.0, 3.0])
    pos_grid = ((pos - pos.min(0).values) / 0.05).floor().long()
    x = torch.cat([pos, torch.rand(num_points, 1)], dim=1)
    batch = torch.repeat_interleave(torch.arange(len(lengths)), lengths)
    return dict(
        x=x.cuda(),
        pos=pos.cuda(),
        pos_grid=pos_grid.cuda(),
        batch=batch.cuda(),
    )


@pytest.fixture
def model_seg() -> SphereFormerSegmentation:
    # The sptr kernel requires head_dim == 16, and each window-attention branch needs at least one head
    # (num_heads >= 2), so base_channels must be >= 32.
    return SphereFormerSegmentation(
        in_channels=4,
        num_classes=10,
        base_channels=32,
        layers=(32, 64, 128),
        block_reps=1,
        head_dim=16,
        spatial_padding=64,
    ).cuda()


def test_sphereformer_segmentation_forward(model_seg: SphereFormerSegmentation, data: Dict[str, Tensor]) -> None:
    model_seg.eval()
    logits = model_seg(data["x"], data["pos_grid"], data["batch"], data["pos"])
    assert logits.shape == (data["pos_grid"].shape[0], model_seg.num_classes)
    assert torch.isfinite(logits).all()


def test_sphereformer_segmentation_reset_classifier(
    model_seg: SphereFormerSegmentation, data: Dict[str, Tensor]
) -> None:
    model_seg.reset_classifier(num_classes=7)
    model_seg.cuda().eval()
    logits = model_seg(data["x"], data["pos_grid"], data["batch"], data["pos"])
    assert logits.shape == (data["pos_grid"].shape[0], 7)


@pytest.mark.skipif(not _WEIGHTS.exists(), reason=f"pretrained weights not found at {_WEIGHTS}")
@pytest.mark.skipif(not _SEMANTICKITTI.is_dir(), reason="SemanticKITTI not available")
def test_sphereformer_semantickitti_smoke() -> None:
    """Load the pretrained checkpoint and run a single SemanticKITTI scan end-to-end."""
    from torch_pointcloud.datasets import SemanticKITTI
    from torch_pointcloud.models import create_model
    from torch_pointcloud.utils.data import DataKeys, collate

    model, info = create_model(
        "sphereformer-dvlab.semantickitti", task="segmentation", pretrained=True, return_info=True
    )
    model = model.cuda().eval()

    dataset = SemanticKITTI(root=os.fspath(Path(DATA_DIR)), split="val", transform=info["transforms"])
    sample = collate([dataset[0]])

    x = sample[DataKeys.X].cuda()
    pos = sample[DataKeys.POS].cuda()
    pos_grid = sample[DataKeys.POS_GRID].cuda()
    batch = sample[DataKeys.BATCH].cuda()

    with torch.inference_mode():
        logits = model(x, pos_grid, batch, pos)

    assert logits.shape == (x.shape[0], model.num_classes)
    assert torch.isfinite(logits).all()
