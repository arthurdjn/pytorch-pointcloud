from typing import List, Tuple

import pytest
import torch
from torch import Tensor, nn

from torch_pointcloud.layers import PositionEmbeddingSine
from torch_pointcloud.models import pcp_mae
from torch_pointcloud.models.pcp_mae import PCPMAEClassification, PCPMAEEncoder, PCPMAEPretraining
from torch_pointcloud.ops.cluster import group
from torch_pointcloud.utils.imports import _PYG_LIB_AVAILABLE


def test_pcp_mae_classification_embeds_the_centers_with_a_sine_embedding() -> None:
    model = PCPMAEClassification(in_channels=0, num_classes=10, embed_dim=48, depth=2, num_heads=2)

    assert isinstance(model.pos_embed[0], PositionEmbeddingSine)
    assert model.pos_embed[1].channel_list == [48, 384, 48]


@pytest.mark.skipif(not _PYG_LIB_AVAILABLE, reason="pyg-lib is not installed")
def test_pcp_mae_classification_basic() -> None:
    model = PCPMAEClassification(
        in_channels=0, num_classes=10, embed_dim=48, depth=2, num_heads=2, num_groups=16, group_size=8
    ).eval()
    pos = torch.randn(512, 3)
    batch = torch.cat([torch.zeros(256), torch.ones(256)]).long()

    assert model(None, pos, batch).shape == (2, 10)
    model.reset_classifier(0)
    assert model(None, pos, batch).shape == (2, 2 * 48)


def test_pcp_mae_classification_reset_classifier_rejects_global_pool() -> None:
    model = PCPMAEClassification(in_channels=0, num_classes=10, embed_dim=48, depth=2, num_heads=2)

    with pytest.raises(ValueError, match="global_pool"):
        model.reset_classifier(5, global_pool="mean")


class RecordShapes(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.shapes: List[Tuple[int, ...]] = []

    def forward(self, x: Tensor) -> Tensor:
        self.shapes.append(tuple(x.shape))
        return x


def test_pcp_mae_encoder_visible_tokens_ignore_the_masked_ones() -> None:
    encoder = PCPMAEEncoder(embed_dim=32, depth=2, num_heads=4).eval()
    x, pos = torch.randn(2, 4, 32), torch.randn(2, 4, 32)

    out, _ = encoder(x, pos, torch.randn(2, 3, 32), torch.randn(2, 3, 32))
    other, _ = encoder(x, pos, torch.randn(2, 3, 32), torch.randn(2, 3, 32))

    torch.testing.assert_close(out, other)


def test_pcp_mae_encoder_drops_the_paths_of_each_group_separately() -> None:
    encoder = PCPMAEEncoder(embed_dim=32, depth=1, num_heads=4, drop_path=0.1)
    recorder = RecordShapes()
    encoder.blocks[0].drop_path = recorder

    encoder(torch.randn(2, 4, 32), torch.randn(2, 4, 32), torch.randn(2, 3, 32), torch.randn(2, 3, 32))

    assert recorder.shapes == [(2, 4, 32), (2, 3, 32), (2, 4, 32), (2, 3, 32)]


@pytest.mark.skipif(not _PYG_LIB_AVAILABLE, reason="pyg-lib is not installed")
def test_pcp_mae_pretraining_predicts_the_center_embedding_of_the_masked_groups(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = PCPMAEPretraining(
        in_channels=0,
        embed_dim=48,
        encoder_depth=2,
        decoder_depth=1,
        num_heads=2,
        decoder_num_heads=2,
        num_groups=16,
        group_size=8,
        pos_embed_channels=(48,),
    ).eval()
    mask = torch.zeros(2, 16, dtype=torch.bool)
    mask[:, :9] = True
    monkeypatch.setattr(pcp_mae, "random_mask", lambda *args, **kwargs: mask)
    pos = torch.randn(512, 3)
    batch = torch.cat([torch.zeros(256), torch.ones(256)]).long()

    pred, target, pred_batch, _, center_pred, center_target = model(None, pos, batch)

    _, center = group(pos, batch, 16, 8)
    assert pred.shape == target.shape == (2 * 9 * 8, 3)
    assert pred_batch.max() == 2 * 9 - 1
    assert center_pred.shape == center_target.shape == (2 * 9, 48)
    assert torch.allclose(center_target, PositionEmbeddingSine(48)(center[mask]))
    assert center_pred.requires_grad and not center_target.requires_grad and not target.requires_grad


def test_pcp_mae_pretraining_invalid_mask_ratio_raises() -> None:
    with pytest.raises(ValueError, match="mask_ratio"):
        PCPMAEPretraining(in_channels=0, embed_dim=48, num_heads=2, mask_ratio=1.0)
