import pytest
import torch
from torch_geometric.nn import MLP

from torch_pointcloud.models import model_info
from torch_pointcloud.models.recon import (
    ReConClassification,
    ReConEncoderBlock,
    ReConPartSegmentation,
    ReConPretraining,
)
from torch_pointcloud.utils.data import DataKeys
from torch_pointcloud.utils.imports import _PYG_LIB_AVAILABLE


def test_recon_encoder_block_local_tokens_ignore_the_queries() -> None:
    block = ReConEncoderBlock(32, num_heads=4, num_queries=3).eval()
    x = torch.randn(2, 3 + 10, 32)
    other_queries = torch.cat([torch.randn(2, 3, 32), x[:, 3:]], dim=1)

    out, other = block(x), block(other_queries)

    assert out.shape == x.shape
    assert torch.allclose(out[:, 3:], other[:, 3:], atol=1e-6)
    assert not torch.allclose(out[:, :3], other[:, :3])


@pytest.mark.skipif(not _PYG_LIB_AVAILABLE, reason="pyg-lib is not installed")
def test_recon_classification_pools_the_queries_and_the_tokens() -> None:
    model = ReConClassification(
        in_channels=0, num_classes=10, embed_dim=48, depth=2, num_heads=2, num_groups=16, group_size=8
    ).eval()
    pos = torch.randn(512, 3)
    batch = torch.cat([torch.zeros(256), torch.ones(256)]).long()

    out = model(None, pos, batch)

    assert out.shape == (2, 10)
    assert model.num_features == 4 * 48
    assert model.forward_head(model.forward_features(None, pos, batch), pre_logits=True).shape == (2, 4 * 48)


@pytest.mark.skipif(not _PYG_LIB_AVAILABLE, reason="pyg-lib is not installed")
def test_recon_classification_accepts_features() -> None:
    model = ReConClassification(
        in_channels=3, num_classes=10, embed_dim=48, depth=2, num_heads=2, num_groups=16, group_size=8
    ).eval()
    pos = torch.randn(512, 3)
    batch = torch.cat([torch.zeros(256), torch.ones(256)]).long()

    out_a = model(torch.randn(512, 3), pos, batch)
    out_b = model(torch.randn(512, 3), pos, batch)

    assert model.encoder.local_mlp.channel_list[0] == 6
    assert not torch.allclose(out_a, out_b)


def test_recon_classification_reset_classifier_rejects_global_pool() -> None:
    model = ReConClassification(in_channels=0, num_classes=10, embed_dim=48, depth=2, num_heads=2)

    model.reset_classifier(0)
    assert isinstance(model.head, torch.nn.Identity)
    model.reset_classifier(5)
    assert isinstance(model.head, MLP)
    with pytest.raises(ValueError, match="global_pool"):
        model.reset_classifier(5, global_pool="mean")


@pytest.mark.skipif(not _PYG_LIB_AVAILABLE, reason="pyg-lib is not installed")
def test_recon_part_segmentation_accepts_a_ragged_batch() -> None:
    model = ReConPartSegmentation(
        in_channels=0, num_classes=5, num_categories=16, embed_dim=48, num_heads=2, num_groups=8, group_size=4
    ).eval()
    pos = torch.randn(300, 3)
    batch = torch.cat([torch.zeros(100), torch.ones(200)]).long()
    category = torch.nn.functional.one_hot(torch.tensor([3, 7]), 16).float()

    out = model(None, pos, batch, category)

    assert out.shape == (300, 5)
    assert model.num_features == 1024 + 3 * 48 * 2 + 48 + 64
    assert isinstance(model.head, MLP)
    assert model.head.channel_list[0] == model.num_features


def test_recon_part_segmentation_validates_fetch_idx_against_depth() -> None:
    with pytest.raises(ValueError, match="fetch_idx"):
        ReConPartSegmentation(in_channels=0, num_classes=5, depth=6, embed_dim=48, num_heads=2)


@pytest.mark.skipif(not _PYG_LIB_AVAILABLE, reason="pyg-lib is not installed")
def test_recon_part_segmentation_transform_runs_without_labels() -> None:
    transform = model_info("recon.shapenetpart.zekun-qi")["transform"]
    assert transform is not None
    sample = {DataKeys.POS: torch.randn(3000, 3), DataKeys.CATEGORY: torch.tensor(4)}

    out = transform(sample)

    assert out[DataKeys.POS].shape == (2048, 3)
    assert out[DataKeys.CATEGORY].shape == (1, 16)


def test_recon_encoder_block_detached_tokens_get_no_gradient_from_the_queries() -> None:
    block = ReConEncoderBlock(32, num_heads=4, num_queries=3, detach_tokens=True)
    tokens = torch.randn(2, 10, 32, requires_grad=True)
    x = torch.cat([torch.randn(2, 3, 32), tokens], dim=1)

    block(x)[:, :3].sum().backward()

    assert tokens.grad is None or torch.all(tokens.grad == 0)


@pytest.mark.skipif(not _PYG_LIB_AVAILABLE, reason="pyg-lib is not installed")
def test_recon_pretraining_returns_the_reconstruction_and_the_cross_modal_queries() -> None:
    model = ReConPretraining(
        in_channels=0,
        image_dim=24,
        text_dim=16,
        embed_dim=48,
        encoder_depth=2,
        decoder_depth=1,
        num_heads=2,
        decoder_num_heads=2,
        num_groups=16,
        group_size=8,
    )
    pos = torch.randn(512, 3)
    batch = torch.cat([torch.zeros(256), torch.ones(256)]).long()

    pred, target, pred_batch, target_batch, image, text = model(None, pos, batch)

    num_masked = int(0.6 * 16)
    assert pred.shape == target.shape == (2 * num_masked * 8, 3)
    assert torch.equal(pred_batch, target_batch)
    assert image.shape == (2, 24)
    assert text.shape == (2, 16)
