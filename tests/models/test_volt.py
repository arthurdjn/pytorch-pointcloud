from typing import Dict, List

import pytest
import torch
from torch import Tensor

from torch_pointcloud.models.volt import VoltSegmentation, checkpoint_filter_fn


@pytest.fixture
def data() -> Dict[str, Tensor]:
    torch.manual_seed(42)
    pos_grid = torch.cat([torch.unique(torch.randint(0, 24, (600, 3)), dim=0) for _ in range(2)])
    lengths = [pos_grid.size(0) // 2, pos_grid.size(0) - pos_grid.size(0) // 2]
    batch = torch.repeat_interleave(torch.arange(2), torch.tensor(lengths))
    return dict(x=torch.randn(pos_grid.size(0), 6), pos_grid=pos_grid, batch=batch)


@pytest.fixture
def model() -> VoltSegmentation:
    torch.manual_seed(0)
    return VoltSegmentation(
        in_channels=6,
        num_classes=10,
        patch_size=4,
        embed_dim=48,
        depth=2,
        num_heads=2,
        decoder_channels=16,
        rope_frequencies=(4, 4, 4),
        use_flash_attn=False,
    ).eval()


def test_volt_segmentation_forward(model: VoltSegmentation, data: Dict[str, Tensor]) -> None:
    logits = model(data["x"], data["pos_grid"], data["batch"])
    assert logits.shape == (data["x"].size(0), 10)


def test_volt_segmentation_composes_features_decoder_head(model: VoltSegmentation, data: Dict[str, Tensor]) -> None:
    tokens, inverse, offset = model.forward_features(data["x"], data["pos_grid"], data["batch"])
    features = model.forward_decoder(tokens, inverse, offset)

    assert tokens.shape == (int(inverse.max()) + 1, 48)
    assert features.shape == (data["x"].size(0), model.num_features)
    torch.testing.assert_close(model.forward_head(features), model(data["x"], data["pos_grid"], data["batch"]))
    torch.testing.assert_close(model.forward_head(features, pre_logits=True), features)


def test_volt_segmentation_samples_are_independent(model: VoltSegmentation, data: Dict[str, Tensor]) -> None:
    logits = model(data["x"], data["pos_grid"], data["batch"])
    first = data["batch"] == 0

    alone = model(data["x"][first], data["pos_grid"][first], data["batch"][first])

    torch.testing.assert_close(logits[first], alone)


def test_volt_segmentation_headless_returns_features(data: Dict[str, Tensor]) -> None:
    model = VoltSegmentation(
        in_channels=6,
        num_classes=0,
        embed_dim=48,
        depth=1,
        num_heads=2,
        decoder_channels=16,
        rope_frequencies=(4, 4, 4),
        use_flash_attn=False,
    ).eval()
    out = model(data["x"], data["pos_grid"], data["batch"])
    assert out.shape == (data["x"].size(0), model.num_features)


def test_volt_segmentation_rejects_mismatched_rope_frequencies() -> None:
    with pytest.raises(ValueError, match="rope_frequencies"):
        VoltSegmentation(6, 10, embed_dim=48, depth=1, num_heads=2, rope_frequencies=(4, 4, 2), use_flash_attn=False)


def test_volt_segmentation_reset_classifier(model: VoltSegmentation, data: Dict[str, Tensor]) -> None:
    model.reset_classifier(num_classes=3)
    assert model(data["x"], data["pos_grid"], data["batch"]).shape == (data["x"].size(0), 3)


def _released_keys(depth: int) -> List[str]:
    keys = ["backbone.tokenizer.proj.weight", "backbone.tokenizer.proj.bias"]
    for i in range(depth):
        for layer in ("norm1", "norm2", "attn.qkv", "attn.proj", "attn.q_norm", "attn.k_norm", "mlp.fc1", "mlp.fc2"):
            keys += [f"backbone.blocks.{i}.{layer}.weight", f"backbone.blocks.{i}.{layer}.bias"]
    for norm in ("pre.0", "pre.3", "post.0"):
        for param in ("weight", "bias", "running_mean", "running_var", "num_batches_tracked"):
            keys.append(f"backbone.decoder.{norm}.{param}")
    keys += ["backbone.decoder.pre.2.weight", "backbone.decoder.unembed.proj.weight", "backbone.decoder.unembed.bias"]
    for head in ("ScanNet", "ScanNet200"):
        keys += [f"seg_heads.{head}.weight", f"seg_heads.{head}.bias", f"distill_heads.{head}.weight"]
    return keys


def test_checkpoint_filter_fn_maps_a_released_checkpoint(model: VoltSegmentation) -> None:
    released = {key: torch.zeros(1) for key in _released_keys(depth=2)}
    released["seg_heads.ScanNet200.weight"] = torch.ones(1)

    state_dict = checkpoint_filter_fn(released, model, head="ScanNet200")

    assert set(state_dict) == set(model.state_dict())
    assert torch.equal(state_dict["head.weight"], torch.ones(1))
