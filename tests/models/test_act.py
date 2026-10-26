from typing import Any, Dict

import pytest
import torch

from torch_pointcloud.models.act import ACTDiscreteVAE, ACTPretraining
from torch_pointcloud.ops.cluster import group
from torch_pointcloud.utils.imports import _PYG_LIB_AVAILABLE

pytestmark = [
    pytest.mark.skipif(not _PYG_LIB_AVAILABLE, reason="pyg-lib is not installed"),
]

SMALL_DVAE: Dict[str, Any] = dict(
    image_dim=32,
    image_depth=3,
    image_num_heads=2,
    num_prompt_tokens=4,
    num_groups=16,
    group_size=8,
    encoder_dims=24,
    num_tokens=16,
    tokens_dims=24,
    decoder_dims=24,
)


def _cloud() -> tuple[torch.Tensor, torch.Tensor]:
    pos = torch.randn(512, 3)
    batch = torch.cat([torch.zeros(256), torch.ones(256)]).long()
    return pos, batch


def test_act_discrete_vae_reconstructs_every_patch() -> None:
    model = ACTDiscreteVAE(in_channels=0, **SMALL_DVAE).eval()
    pos, batch = _cloud()

    out = model(None, pos, batch)

    assert out["logits"].shape == (2, 16, 16)
    assert out["fine"].shape == out["neighborhood"].shape == (2, 16, 8, 3)


def test_act_discrete_vae_freezes_the_image_transformer_only() -> None:
    model = ACTDiscreteVAE(in_channels=0, **SMALL_DVAE)

    assert not any(p.requires_grad for p in model.image_blocks.parameters())
    assert not any(p.requires_grad for p in model.image_norm.parameters())
    assert model.deep_prompt_token.requires_grad and model.deep_prompt_token.shape == (2, 4, 32)
    assert model.image_proj_in.weight.requires_grad


def test_act_discrete_vae_group_features_keep_the_patch_tokens() -> None:
    model = ACTDiscreteVAE(in_channels=0, **SMALL_DVAE).eval()
    neighborhood, center = group(*_cloud(), 16, 8)

    features = model.group_features(neighborhood, center)

    assert features.shape == (2, 16, 24)


def test_act_pretraining_regresses_the_frozen_teacher() -> None:
    model = ACTPretraining(
        in_channels=0,
        image_dim=32,
        image_depth=2,
        image_num_heads=2,
        num_prompt_tokens=4,
        embed_dim=48,
        encoder_depth=2,
        decoder_depth=1,
        num_heads=2,
        decoder_num_heads=2,
        num_groups=16,
        group_size=8,
        num_tokens=16,
        tokens_dims=24,
    )
    pos, batch = _cloud()

    predictions, targets = model(None, pos, batch)
    (1 - torch.nn.functional.cosine_similarity(predictions, targets, dim=-1)).mean().backward()

    num_masked = int(0.8 * 16)
    assert predictions.shape == targets.shape == (2 * num_masked, 24)
    assert all(p.grad is None for p in model.teacher.parameters())
    assert model.get_parameter("blocks.blocks.0.attn.qkv.weight").grad is not None


def test_act_pretraining_invalid_mask_ratio_raises() -> None:
    with pytest.raises(ValueError, match="mask_ratio"):
        ACTPretraining(
            in_channels=0, image_dim=32, image_depth=2, image_num_heads=2, num_prompt_tokens=4, mask_ratio=0.0
        )
