from typing import Any, Dict

import pytest
import torch
from torch import Tensor

from torch_pointcloud.losses import GeoTransformerLoss
from torch_pointcloud.models import create_model
from torch_pointcloud.models.geotransformer import (
    GeometricStructureEmbedding,
    GeometricTransformer,
    GeoTransformerRegistration,
    KPConvFPN,
    LogOptimalTransport,
    sinusoidal_embedding,
)
from torch_pointcloud.utils.data import DataKeys
from torch_pointcloud.utils.imports import _PYG_LIB_AVAILABLE

pytestmark = pytest.mark.skipif(not _PYG_LIB_AVAILABLE, reason="pyg-lib is not installed")


def _create_model(**overrides: Any) -> GeoTransformerRegistration:
    kwargs: Dict[str, Any] = dict(
        in_channels=1,
        stem_channels=8,
        encoder_channels=[16, 32, 64],
        decoder_channels=[32, 16],
        voxel_sizes=[0.1, 0.2],
        radii=[0.125, 0.25, 0.5],
        kp_sigma=[0.1, 0.2, 0.4],
        max_num_neighbors=[16, 16, 16],
        num_groups=4,
        embed_dim=32,
        coarse_channels=32,
        num_heads=4,
        blocks=["self", "cross"],
        sigma_d=0.2,
        num_points_in_patch=16,
        matching_radius=0.05,
        num_correspondences=32,
        num_targets=16,
        num_sinkhorn_iterations=10,
    )
    kwargs.update(overrides)
    return GeoTransformerRegistration(**kwargs)


def _pair(num_src: int = 600, num_ref: int = 700, seed: int = 0) -> Dict[str, Tensor]:
    torch.manual_seed(seed)
    pos = torch.rand(num_src, 3) * 2.0
    ref_pos = torch.rand(num_ref, 3) * 2.0
    transform = torch.eye(4)
    transform[:3, 3] = torch.tensor([0.1, -0.2, 0.05])
    return {
        DataKeys.POS: pos,
        DataKeys.BATCH: torch.zeros(num_src, dtype=torch.long),
        DataKeys.REF_POS: ref_pos,
        DataKeys.BATCH_REF_POS: torch.zeros(num_ref, dtype=torch.long),
        DataKeys.TRANSFORM: transform.unsqueeze(0),
    }


def test_sinusoidal_embedding_interleaves_sin_and_cos() -> None:
    indices = torch.tensor([0.0, 1.5])
    embedding = sinusoidal_embedding(indices, dim=4)
    assert embedding.shape == (2, 4)
    assert torch.allclose(embedding[0], torch.tensor([0.0, 1.0, 0.0, 1.0]))
    assert embedding[1, 0].item() == pytest.approx(torch.sin(torch.tensor(1.5)).item())
    with pytest.raises(ValueError, match="even"):
        sinusoidal_embedding(indices, dim=3)


def test_geometric_structure_embedding_is_rigid_invariant() -> None:
    torch.manual_seed(0)
    embedding = GeometricStructureEmbedding(16, sigma_d=0.2, sigma_a=15.0, angle_k=3)
    points = torch.rand(1, 20, 3)
    angle = torch.tensor(0.7)
    rotation = torch.tensor(
        [[torch.cos(angle), -torch.sin(angle), 0.0], [torch.sin(angle), torch.cos(angle), 0.0], [0.0, 0.0, 1.0]]
    )
    moved = points @ rotation.T + torch.tensor([0.1, -0.2, 0.3])
    # The distances come from the expanded squared-distance formula, whose float32 cancellation bounds the match.
    assert torch.allclose(embedding(points), embedding(moved), atol=1e-3)
    assert embedding(points).shape == (1, 20, 20, 16)


def test_geometric_transformer_shapes() -> None:
    transformer = GeometricTransformer(24, 32, 16, 4, ["self", "cross"], 0.2, 15.0, 3)
    ref, src = torch.rand(1, 10, 3), torch.rand(1, 12, 3)
    ref_feats, src_feats = transformer(ref, src, torch.randn(1, 10, 24), torch.randn(1, 12, 24))
    assert ref_feats.shape == (1, 10, 16)
    assert src_feats.shape == (1, 12, 16)


def test_log_optimal_transport_is_doubly_stochastic_on_valid_entries() -> None:
    torch.manual_seed(0)
    ot = LogOptimalTransport(num_iterations=50)
    scores = torch.randn(2, 5, 6)
    row_masks = torch.ones(2, 5, dtype=torch.bool)
    col_masks = torch.ones(2, 6, dtype=torch.bool)
    col_masks[1, -1] = False
    out = ot(scores, row_masks, col_masks)
    assert out.shape == (2, 6, 7)
    probs = out.exp()
    # Every valid point row sums to one over the valid columns and the dustbin.
    assert torch.allclose(probs[0, :5].sum(dim=1), torch.ones(5), atol=1e-3)
    assert torch.allclose(probs[1, :5, :5].sum(dim=1) + probs[1, :5, 6], torch.ones(5), atol=1e-3)
    assert torch.all(probs[1, :, 5] == 0)


def test_kpconv_fpn_rejects_inconsistent_configs() -> None:
    with pytest.raises(ValueError, match="voxel_sizes"):
        KPConvFPN(
            1,
            stem_channels=8,
            encoder_channels=[8, 16],
            decoder_channels=[8],
            voxel_sizes=[],
            radii=[0.1, 0.2],
            kp_sigma=[0.1, 0.2],
            max_num_neighbors=[8, 8],
        )
    with pytest.raises(ValueError, match="decoders"):
        KPConvFPN(
            1,
            stem_channels=8,
            encoder_channels=[8, 16],
            decoder_channels=[8, 8],
            voxel_sizes=[0.1],
            radii=[0.1, 0.2],
            kp_sigma=[0.1, 0.2],
            max_num_neighbors=[8, 8],
        )


def test_geotransformer_forward_decode_shapes() -> None:
    model = _create_model().eval()
    data = _pair()
    with torch.no_grad():
        fine, coarse = model.forward_features(
            None, data[DataKeys.POS], data[DataKeys.BATCH], None, data[DataKeys.REF_POS], data[DataKeys.BATCH_REF_POS]
        )
        output = model.forward_head(fine, coarse)
        registration = model.decode(output)
    assert fine["x"].shape[1] == model.num_features == 16
    assert set(fine["batch"].tolist()) == {0, 1}
    num_corr = output["node_corr_batch"].numel()
    assert 0 < num_corr <= 32
    assert output["matching_scores"].shape == (num_corr, 17, 17)
    assert output["ref_patch_points"].shape == (num_corr, 16, 3)
    assert output["src_feats_c"].shape[1] == 32
    assert torch.allclose(output["src_feats_c"].norm(dim=1), torch.ones(output["src_feats_c"].size(0)), atol=1e-4)
    assert "gt_node_corr_indices" not in output
    assert registration["transform"].shape == (1, 4, 4)
    assert torch.allclose(registration["transform"][0, 3], torch.tensor([0.0, 0.0, 0.0, 1.0]))
    rotation = registration["transform"][0, :3, :3]
    assert torch.allclose(rotation @ rotation.T, torch.eye(3), atol=1e-4)
    assert registration["src_points"].shape == registration["ref_points"].shape
    assert registration["scores"].shape[0] == registration["batch"].shape[0] == registration["src_points"].shape[0]


def test_geotransformer_batch_matches_single_pairs() -> None:
    model = _create_model().eval()
    first, second = _pair(seed=0), _pair(600, 650, seed=1)
    with torch.no_grad():
        out_first = model(
            None,
            first[DataKeys.POS],
            first[DataKeys.BATCH],
            None,
            first[DataKeys.REF_POS],
            first[DataKeys.BATCH_REF_POS],
        )
        out_second = model(
            None,
            second[DataKeys.POS],
            second[DataKeys.BATCH],
            None,
            second[DataKeys.REF_POS],
            second[DataKeys.BATCH_REF_POS],
        )
        out = model(
            None,
            torch.cat([first[DataKeys.POS], second[DataKeys.POS]]),
            torch.cat([first[DataKeys.BATCH], second[DataKeys.BATCH] + 1]),
            None,
            torch.cat([first[DataKeys.REF_POS], second[DataKeys.REF_POS]]),
            torch.cat([first[DataKeys.BATCH_REF_POS], second[DataKeys.BATCH_REF_POS] + 1]),
        )
    assert torch.allclose(out["matching_scores"][out["node_corr_batch"] == 0], out_first["matching_scores"], atol=1e-4)
    assert torch.allclose(out["matching_scores"][out["node_corr_batch"] == 1], out_second["matching_scores"], atol=1e-4)
    transforms = model.decode(out)["transform"]
    assert transforms.shape == (2, 4, 4)
    assert torch.allclose(transforms[1], model.decode(out_second)["transform"][0], atol=1e-4)


def test_geotransformer_training_uses_ground_truth_targets() -> None:
    model = _create_model().train()
    data = _pair()
    with pytest.raises(ValueError, match="transform"):
        model(
            None, data[DataKeys.POS], data[DataKeys.BATCH], None, data[DataKeys.REF_POS], data[DataKeys.BATCH_REF_POS]
        )

    output = model(
        None,
        data[DataKeys.POS],
        data[DataKeys.BATCH],
        None,
        data[DataKeys.REF_POS],
        data[DataKeys.BATCH_REF_POS],
        transform=data[DataKeys.TRANSFORM],
    )
    assert output["gt_node_corr_indices"].shape[1] == 2
    assert output["node_corr_batch"].numel() <= 16
    losses = GeoTransformerLoss()(output, data)
    losses["loss"].backward()
    assert torch.isfinite(losses["loss"])
    assert model.transformer.in_proj.weight.grad is not None


def test_geotransformer_x_none_uses_constant_feature() -> None:
    model = _create_model().eval()
    data = _pair()
    ones = torch.ones(data[DataKeys.POS].size(0), 1)
    ref_ones = torch.ones(data[DataKeys.REF_POS].size(0), 1)
    with torch.no_grad():
        a = model(
            None, data[DataKeys.POS], data[DataKeys.BATCH], None, data[DataKeys.REF_POS], data[DataKeys.BATCH_REF_POS]
        )
        b = model(
            ones,
            data[DataKeys.POS],
            data[DataKeys.BATCH],
            ref_ones,
            data[DataKeys.REF_POS],
            data[DataKeys.BATCH_REF_POS],
        )
    assert torch.equal(a["matching_scores"], b["matching_scores"])


@pytest.mark.parametrize(
    "name",
    ["geotransformer.3dmatch.zheng-qin", "geotransformer.kitti.zheng-qin", "geotransformer.modelnet40.zheng-qin"],
)
def test_registered_geotransformer_builds(name: str) -> None:
    model = create_model(name, task="registration")
    assert isinstance(model, GeoTransformerRegistration)
    assert model.num_features == 256
