import pytest
import torch

from torch_pointcloud.losses import GeoTransformerLoss
from torch_pointcloud.utils.data import DataKeys


def _output(batch_size: int = 2, num_nodes: int = 6, num_corr: int = 4, k: int = 5) -> dict[str, torch.Tensor]:
    torch.manual_seed(0)
    feats = torch.nn.functional.normalize(torch.randn(batch_size * num_nodes, 16), dim=1)
    node_batch = torch.arange(batch_size).repeat_interleave(num_nodes)
    gt_indices = torch.stack([torch.arange(num_nodes), torch.arange(num_nodes)], dim=1).repeat(batch_size, 1)
    patch_points = torch.rand(batch_size * num_corr, k, 3)
    return {
        "ref_feats_c": feats,
        "src_feats_c": feats.clone(),
        "ref_batch_c": node_batch,
        "src_batch_c": node_batch.clone(),
        "gt_node_corr_indices": gt_indices,
        "gt_node_corr_overlaps": torch.full((batch_size * num_nodes,), 0.6),
        "gt_node_corr_batch": node_batch.clone(),
        "node_corr_batch": torch.arange(batch_size).repeat_interleave(num_corr),
        "ref_patch_points": patch_points,
        "src_patch_points": patch_points.clone(),
        "ref_patch_masks": torch.ones(batch_size * num_corr, k, dtype=torch.bool),
        "src_patch_masks": torch.ones(batch_size * num_corr, k, dtype=torch.bool),
        "matching_scores": torch.log_softmax(torch.randn(batch_size * num_corr, k + 1, k + 1), dim=-1),
    }


def test_geotransformer_loss_terms_and_identity_transform() -> None:
    output = _output()
    data = {DataKeys.TRANSFORM: torch.eye(4).repeat(2, 1, 1)}
    losses = GeoTransformerLoss(positive_radius=0.05)(output, data)
    assert set(losses) == {"loss", "coarse_loss", "fine_loss"}
    assert torch.isfinite(losses["loss"])
    assert losses["loss"].item() == pytest.approx(losses["coarse_loss"].item() + losses["fine_loss"].item(), rel=1e-5)
    assert not losses["coarse_loss"].requires_grad and not losses["fine_loss"].requires_grad


def test_geotransformer_fine_loss_rewards_matching_diagonal() -> None:
    output = _output(batch_size=1)
    scores = torch.full((4, 6, 6), -10.0)
    scores[:, torch.arange(5), torch.arange(5)] = 0.0
    scores[:, 5, 5] = 0.0
    output["matching_scores"] = scores
    loss = GeoTransformerLoss(positive_radius=0.05).fine_loss(output, torch.eye(4).unsqueeze(0))  # type: ignore[arg-type]
    assert loss.item() == pytest.approx(0.0, abs=1e-6)


def test_geotransformer_fine_loss_without_patches_is_zero() -> None:
    output = _output(batch_size=1)
    for key in ("ref_patch_points", "src_patch_points", "ref_patch_masks", "src_patch_masks", "matching_scores"):
        output[key] = output[key][:0]
    output["node_corr_batch"] = output["node_corr_batch"][:0]
    output["matching_scores"].requires_grad_(True)
    loss = GeoTransformerLoss(positive_radius=0.05).fine_loss(output, torch.eye(4).unsqueeze(0))  # type: ignore[arg-type]
    assert loss.item() == 0.0 and loss.requires_grad


def test_geotransformer_coarse_loss_reads_source_then_reference_correspondences() -> None:
    """`gt_node_corr_indices` pairs a source superpoint with a reference one, in that order."""
    output = _output(batch_size=1, num_nodes=2)
    ref_feats = torch.nn.functional.normalize(torch.tensor([[1.0, 0.0], [0.0, 1.0]]), dim=1)
    output["ref_feats_c"] = ref_feats
    output["src_feats_c"] = ref_feats.flip(0)  # source superpoint i is reference superpoint 1 - i
    output["gt_node_corr_overlaps"] = torch.ones(2)
    criterion = GeoTransformerLoss(positive_radius=0.05)

    output["gt_node_corr_indices"] = torch.tensor([[0, 1], [1, 0]])
    matched = criterion.coarse_loss(output)  # type: ignore[arg-type]
    output["gt_node_corr_indices"] = torch.tensor([[0, 0], [1, 1]])
    swapped = criterion.coarse_loss(output)  # type: ignore[arg-type]
    assert matched < swapped
