from typing import Dict

import pytest
import torch
from torch import Tensor, nn

from torch_pointcloud.models.volt import VoltSegmentation


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


def test_volt_segmentation_returns_every_block_output(model: VoltSegmentation, data: Dict[str, Tensor]) -> None:
    tokens, inverse, offset, intermediates = model.forward_features(
        data["x"],
        data["pos_grid"],
        data["batch"],
        return_intermediates=True,
    )

    assert len(intermediates) == 2
    assert all(block.shape == tokens.shape for block in intermediates)
    torch.testing.assert_close(intermediates[-1], tokens)
    torch.testing.assert_close(tokens, model.forward_features(data["x"], data["pos_grid"], data["batch"])[0])


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


def test_volt_segmentation_defaults_to_the_authors_batch_norm() -> None:
    model = VoltSegmentation(
        6, 10, embed_dim=48, depth=1, num_heads=2, rope_frequencies=(4, 4, 4), use_flash_attn=False
    )

    norms = [module for module in model.decoder.modules() if isinstance(module, nn.BatchNorm1d)]

    assert len(norms) == 3
    assert all(norm.eps == 1e-3 and norm.momentum == 0.01 for norm in norms)


def test_volt_segmentation_rejects_mismatched_rope_frequencies() -> None:
    with pytest.raises(ValueError, match="rope_frequencies"):
        VoltSegmentation(6, 10, embed_dim=48, depth=1, num_heads=2, rope_frequencies=(4, 4, 2), use_flash_attn=False)


def test_volt_segmentation_reset_classifier(model: VoltSegmentation, data: Dict[str, Tensor]) -> None:
    model.reset_classifier(num_classes=3)
    assert model(data["x"], data["pos_grid"], data["batch"]).shape == (data["x"].size(0), 3)
