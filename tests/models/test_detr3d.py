from pathlib import Path
from typing import Any, Dict

import pytest
import torch
from torch import Tensor

from torch_pointcloud.config import MODELS_DIR
from torch_pointcloud.models import create_model, list_models
from torch_pointcloud.models.detr3d import DETR3DDetection, DETR3DOutput
from torch_pointcloud.utils.imports import _TORCH_CLUSTER_AVAILABLE, _TORCH_SCATTER_AVAILABLE

pytestmark = [
    pytest.mark.skipif(not _TORCH_CLUSTER_AVAILABLE, reason="torch-cluster is not installed"),
    pytest.mark.skipif(not _TORCH_SCATTER_AVAILABLE, reason="torch-scatter is not installed"),
]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _small_detr3d(**overrides: Any) -> DETR3DDetection:
    kwargs: Dict[str, Any] = dict(
        in_channels=0,
        num_classes=5,
        num_angle_bin=1,
        num_queries=16,
        preenc_npoints=128,
        encoder_type="vanilla",
        encoder_embed_dim=32,
        encoder_num_heads=2,
        encoder_feedforward_channels=64,
        encoder_depth=2,
        decoder_embed_dim=32,
        decoder_num_heads=2,
        decoder_feedforward_channels=64,
        decoder_depth=2,
    )
    kwargs.update(overrides)
    return DETR3DDetection(**kwargs)


def _make_inputs(n_per_scene: int = 1024, batch_size: int = 2) -> Dict[str, Tensor]:
    """Two scenes of `n_per_scene` points each (>= the tokenizer's `preenc_npoints`)."""
    torch.manual_seed(0)
    n = n_per_scene * batch_size
    pos = torch.rand(n, 3) * 4.0
    batch = torch.arange(batch_size).repeat_interleave(n_per_scene)
    return {"pos": pos.to(DEVICE), "batch": batch.to(DEVICE)}


def _assert_output_shapes(out: DETR3DOutput, batch_size: int, num_queries: int, num_classes: int, nb: int) -> None:
    assert out["sem_cls_logits"].shape == (batch_size, num_queries, num_classes + 1)
    assert out["center_unnormalized"].shape == (batch_size, num_queries, 3)
    assert out["size_unnormalized"].shape == (batch_size, num_queries, 3)
    assert out["angle_logits"].shape == (batch_size, num_queries, nb)
    assert out["angle_continuous"].shape == (batch_size, num_queries)
    assert out["objectness_prob"].shape == (batch_size, num_queries)
    assert out["sem_cls_prob"].shape == (batch_size, num_queries, num_classes)


def test_detr3d_vanilla_forward_shapes() -> None:
    model = _small_detr3d().to(DEVICE).eval()
    data = _make_inputs()
    with torch.no_grad():
        out = model(None, data["pos"], data["batch"])
    _assert_output_shapes(out, batch_size=2, num_queries=16, num_classes=5, nb=1)
    assert torch.isfinite(out["center_unnormalized"]).all()


def test_detr3d_masked_forward_shapes() -> None:
    # 3DETR-m halves the token count via one interim downsampling; 12 oriented heading bins.
    model = _small_detr3d(encoder_type="masked", num_angle_bin=12).to(DEVICE).eval()
    data = _make_inputs()
    with torch.no_grad():
        out = model(None, data["pos"], data["batch"])
    _assert_output_shapes(out, batch_size=2, num_queries=16, num_classes=5, nb=12)


def test_detr3d_decode_packed_detections() -> None:
    model = _small_detr3d().to(DEVICE).eval()
    data = _make_inputs()
    with torch.no_grad():
        out = model(None, data["pos"], data["batch"])
        det = model.decode(out)
    for key in ("boxes", "scores", "labels", "batch"):
        assert key in det
    n = det["boxes"].shape[0]
    assert det["boxes"].shape == (n, 7)
    assert det["scores"].shape == (n,)
    assert det["labels"].shape == (n,)
    assert det["batch"].shape == (n,)
    assert set(det["batch"].tolist()) <= {0, 1}
    assert det["labels"].max() < model.num_classes if n else True


def test_detr3d_reset_classifier() -> None:
    model = _small_detr3d().eval()
    model.reset_classifier(num_classes=9)
    assert model.num_classes == 9
    model = model.to(DEVICE)
    data = _make_inputs()
    with torch.no_grad():
        out = model(None, data["pos"], data["batch"])
    assert out["sem_cls_logits"].shape[-1] == 10
    assert out["sem_cls_prob"].shape[-1] == 9


def test_detr3d_bad_encoder_type() -> None:
    with pytest.raises(ValueError, match="encoder_type"):
        _small_detr3d(encoder_type="bogus")


def test_detr3d_registered_variants() -> None:
    names = list_models("3detr*", task="detection")
    assert "3detr-fair-m.scannet" in names
    assert "3detr-fair.scannet" in names
    assert "3detr-fair.sunrgbd" in names


def test_detr3d_create_model_no_pretrained() -> None:
    model = create_model("3detr-fair.sunrgbd", task="detection")
    assert isinstance(model, DETR3DDetection)
    assert model.num_classes == 10
    assert model.num_angle_bin == 12
    assert model.num_queries == 128
    assert model.encoder_type == "vanilla"


def test_detr3d_masked_variant_config() -> None:
    model = create_model("3detr-fair-m.scannet", task="detection")
    assert isinstance(model, DETR3DDetection)
    assert model.encoder_type == "masked"
    assert model.num_classes == 18
    assert model.num_angle_bin == 1


@pytest.mark.skipif(
    not Path(MODELS_DIR, "3detr", "3detr-fair-m.scannet.pt").exists(),
    reason="3detr-fair-m.scannet pretrained weights not available",
)
def test_detr3d_pretrained_smoke() -> None:
    model = create_model("3detr-fair-m.scannet", task="detection", pretrained=True).to(DEVICE).eval()
    assert isinstance(model, DETR3DDetection)
    torch.manual_seed(0)
    pos = (torch.rand(40000, 3, device=DEVICE) * 4.0).contiguous()
    batch = torch.zeros(40000, dtype=torch.long, device=DEVICE)
    with torch.no_grad():
        out = model(None, pos, batch)
        det = model.decode(out)
    assert out["center_unnormalized"].shape == (1, 256, 3)
    assert torch.isfinite(out["sem_cls_logits"]).all()
    assert det["boxes"].shape[1] == 7
