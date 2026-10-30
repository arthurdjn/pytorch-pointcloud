import pytest
import torch

from torch_pointcloud.models import model_info
from torch_pointcloud.models.point2vec import (
    Point2VecClassification,
    Point2VecPartSegmentation,
    Point2VecPretraining,
    checkpoint_filter_fn,
)
from torch_pointcloud.utils.data import DataKeys
from torch_pointcloud.utils.imports import _PYG_LIB_AVAILABLE


@pytest.mark.skipif(not _PYG_LIB_AVAILABLE, reason="pyg-lib is not installed")
def test_point2vec_classification_pools_max_and_mean() -> None:
    model = Point2VecClassification(
        in_channels=0, num_classes=10, embed_dim=48, depth=2, num_heads=2, num_groups=16, group_size=8
    ).eval()
    pos = torch.randn(512, 3)
    batch = torch.cat([torch.zeros(256), torch.ones(256)]).long()

    tokens = model.forward_features(None, pos, batch)
    pooled = model.forward_head(tokens, pre_logits=True)

    assert model(None, pos, batch).shape == (2, 10)
    assert torch.equal(pooled, torch.cat([tokens.max(1)[0], tokens.mean(1)], dim=-1))


def test_point2vec_classification_layers_before_batch_norm_have_no_bias() -> None:
    model = Point2VecClassification(in_channels=0, num_classes=10, embed_dim=48, depth=2, num_heads=2)

    parameters = dict(model.named_parameters())
    assert "encoder.local_mlp.lins.0.bias" not in parameters
    assert "head.lins.0.bias" not in parameters
    assert "head.lins.2.bias" in parameters
    assert "blocks.blocks.0.attn.qkv.bias" in parameters


@pytest.mark.skipif(not _PYG_LIB_AVAILABLE, reason="pyg-lib is not installed")
def test_point2vec_part_segmentation_accepts_a_ragged_batch() -> None:
    model = Point2VecPartSegmentation(
        in_channels=0,
        num_classes=5,
        num_categories=16,
        embed_dim=48,
        num_heads=2,
        num_groups=8,
        group_size=4,
        propagation_channels=(32, 32),
    ).eval()
    pos = torch.randn(300, 3)
    batch = torch.cat([torch.zeros(100), torch.ones(200)]).long()
    category = torch.nn.functional.one_hot(torch.tensor([3, 7]), 16).float()

    out = model(None, pos, batch, category)

    assert out.shape == (300, 5)
    assert model.num_features == 32 + 2 * 48 + 64


def test_point2vec_checkpoint_filter_fn_maps_the_lightning_layout() -> None:
    model = Point2VecPartSegmentation(
        in_channels=0, num_classes=5, num_categories=16, embed_dim=48, num_heads=2, propagation_channels=(32, 32)
    )
    upsampling = torch.randn(32, 3 + 48, 1)
    state = {
        "tokenizer.embedding.first_conv.0.weight": torch.randn(128, 3, 1),
        "encoder.blocks.4.mlp.fc2.bias": torch.randn(48),
        "encoder.norm.weight": torch.randn(48),
        "upsampling.mlp_convs.0.weight": upsampling,
        "seg_head.7.bias": torch.randn(5),
    }

    out = checkpoint_filter_fn(state, model)

    assert set(out) == {
        "encoder.local_mlp.lins.0.weight",
        "blocks.blocks.4.mlp.lins.1.bias",
        "propagation_0.mlp.lins.0.weight",
        "head.lins.2.bias",
    }
    assert out["encoder.local_mlp.lins.0.weight"].shape == (128, 3)
    assert torch.equal(out["propagation_0.mlp.lins.0.weight"][:, :48], upsampling[:, 3:, 0])
    assert torch.equal(out["propagation_0.mlp.lins.0.weight"][:, 48:], upsampling[:, :3, 0])
    assert set(out) <= set(model.state_dict())


def test_point2vec_checkpoint_filter_fn_keeps_the_final_norm_of_a_classifier() -> None:
    model = Point2VecClassification(in_channels=0, num_classes=10, embed_dim=48, depth=2, num_heads=2)

    out = checkpoint_filter_fn({"encoder.norm.weight": torch.randn(48)}, model)

    assert set(out) == {"norm_f.weight"}


@pytest.mark.skipif(not _PYG_LIB_AVAILABLE, reason="pyg-lib is not installed")
def test_point2vec_part_segmentation_transform_runs_without_labels() -> None:
    transform = model_info("point2vec.shapenetpart.karim-knaebel")["transform"]
    assert transform is not None
    sample = {DataKeys.POS: torch.randn(3000, 3), DataKeys.CATEGORY: torch.tensor(4)}

    out = transform(sample)

    assert out[DataKeys.POS].shape == (2048, 3)


@pytest.mark.skipif(not _PYG_LIB_AVAILABLE, reason="pyg-lib is not installed")
def test_point2vec_pretraining_regresses_the_frozen_teacher() -> None:
    model = Point2VecPretraining(
        in_channels=0,
        target_layers=(0, 1),
        embed_dim=48,
        encoder_depth=2,
        decoder_depth=1,
        num_heads=2,
        num_groups=16,
        group_size=8,
    )
    pos = torch.randn(512, 3)
    batch = torch.cat([torch.zeros(256), torch.ones(256)]).long()

    predictions, targets = model(None, pos, batch)
    torch.nn.functional.smooth_l1_loss(predictions, targets, beta=2).backward()

    num_masked = int(0.65 * 16)
    assert predictions.shape == targets.shape == (2 * num_masked, 48)
    assert not targets.requires_grad
    assert all(p.grad is None for p in model.teacher.parameters())
    assert model.get_parameter("blocks.blocks.0.attn.qkv.weight").grad is not None
    assert torch.equal(
        model.get_parameter("teacher.blocks.0.attn.qkv.weight"), model.get_parameter("blocks.blocks.0.attn.qkv.weight")
    )


def test_point2vec_pretraining_update_teacher_moves_toward_the_student_blocks() -> None:
    model = Point2VecPretraining(
        in_channels=0, target_layers=(0, 1), embed_dim=48, encoder_depth=2, decoder_depth=1, num_heads=2
    )
    student = model.get_parameter("blocks.blocks.0.attn.qkv.weight")
    teacher = model.get_parameter("teacher.blocks.0.attn.qkv.weight")
    with torch.no_grad():
        student.add_(1.0)
    before = teacher.clone()

    model.update_teacher(decay=0.75)

    assert torch.allclose(teacher, 0.75 * before + 0.25 * student)
    assert not teacher.requires_grad


def test_point2vec_pretraining_validates_the_target_layers() -> None:
    with pytest.raises(ValueError, match="target_layers"):
        Point2VecPretraining(in_channels=0, target_layers=(2,), embed_dim=48, encoder_depth=2, num_heads=2)
