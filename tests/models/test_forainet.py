from typing import Any, Dict, Tuple

import pytest
import torch
from torch import Tensor

from torch_pointcloud.models.forainet import ForAINetInstanceSegmentation, ForAINetOutput, ForAINetUNet
from torch_pointcloud.utils.conversion import convert_to_spconv_tensor
from torch_pointcloud.utils.imports import _CUDA_AVAILABLE, _PYG_LIB_AVAILABLE, _SPCONV_AVAILABLE

_REQUIRES_SPCONV = pytest.mark.skipif(not _SPCONV_AVAILABLE, reason="spconv is not installed")
_REQUIRES_CUDA = pytest.mark.skipif(not _CUDA_AVAILABLE, reason="CUDA is not available")
_REQUIRES_PYG_LIB = pytest.mark.skipif(not _PYG_LIB_AVAILABLE, reason="pyg-lib is not installed")


def _model(**kwargs: Any) -> ForAINetInstanceSegmentation:
    torch.manual_seed(0)
    hparams: Dict[str, Any] = dict(
        channels=(8, 16, 24),
        strides=(1, 2, 2),
        decoder_channels=(16, 8, 8),
        decoder_strides=(2, 2, 1),
        num_blocks=1,
        scorer_channels=(16,),
        scorer_decoder_channels=(8,),
    )
    hparams.update(kwargs)
    return ForAINetInstanceSegmentation(4, 3, stuff_classes=[0], **hparams).cuda().eval()


def _voxels() -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """Two scenes filling blocks of 8 x 6 x 4 and 10 x 6 x 4 voxels, with positions at the voxel centres."""
    grids, batches = [], []
    for scene, length in enumerate((8, 10)):
        grids.append(torch.cartesian_prod(torch.arange(length), torch.arange(6), torch.arange(4)))
        batches.append(torch.full((length * 24,), scene))
    pos_grid, batch = torch.cat(grids), torch.cat(batches)
    pos = (pos_grid.float() + 0.5) * 0.2
    x = torch.cat([pos, pos[:, 2:]], dim=1)
    return x.cuda(), pos.cuda(), pos_grid.cuda(), batch.cuda()


def test_unet_rejects_unbalanced_strides() -> None:
    with pytest.raises(ValueError, match="upsample"):
        ForAINetUNet(4, (8, 16), (1, 2), (8, 8), (1, 1))
    with pytest.raises(ValueError, match="one block per encoder block"):
        ForAINetUNet(4, (8, 16), (1, 2), (8,), (2,))


@_REQUIRES_SPCONV
@_REQUIRES_CUDA
def test_unet_returns_the_input_voxels_in_order() -> None:
    x, _, pos_grid, batch = _voxels()
    unet = ForAINetUNet(4, (8, 16, 24), (1, 2, 2), (16, 8, 8), (2, 2, 1), num_blocks=1).cuda().eval()
    sparse_x = convert_to_spconv_tensor(x, pos_grid, batch, padding=1)
    with torch.no_grad():
        out = unet(sparse_x)
    assert torch.equal(out.indices, sparse_x.indices)
    assert out.features.shape == (len(x), 8)


@_REQUIRES_SPCONV
@_REQUIRES_CUDA
def test_forward_predicts_every_voxel() -> None:
    model = _model()
    x, pos, pos_grid, batch = _voxels()
    with torch.no_grad():
        out = model(x, pos, pos_grid, batch)
    assert model.num_features == 8
    assert out["semantic_logits"].shape == (len(x), 3)
    assert out["offset"].shape == (len(x), 3)
    assert out["embedding"].shape == (len(x), 5)
    assert out["features"].shape == (len(x), 8)
    assert out["pos"] is pos and out["batch"] is batch


def _output(model: ForAINetInstanceSegmentation) -> ForAINetOutput:
    """Two scenes whose instance voxels fill the far end of their block, their embeddings split in two along y."""
    x, pos, pos_grid, batch = _voxels()
    with torch.no_grad():
        out = model(x, pos, pos_grid, batch)
    blob = pos[:, 0] > 1.0
    logits = torch.zeros_like(out["semantic_logits"])
    logits[:, 1] = blob.float()
    embedding = torch.zeros_like(out["embedding"])
    embedding[:, 0] = 5.0 * (pos[:, 1] > 0.6).float()
    return {
        "semantic_logits": logits,
        "offset": torch.zeros_like(pos),
        "embedding": embedding,
        "features": out["features"],
        "pos": pos,
        "pos_grid": pos_grid,
        "batch": batch,
    }


@_REQUIRES_SPCONV
@_REQUIRES_CUDA
@_REQUIRES_PYG_LIB
def test_decode_proposes_regions_and_modes_of_each_scene() -> None:
    model = _model(cluster_radius=0.5, min_cluster_size=5)
    output = _output(model)
    instances = model.decode(output)
    mask_index, batch = instances["mask_index"], instances["batch"]
    thing = output["semantic_logits"][:, 1] > 0
    masks = torch.zeros(len(batch), len(thing), dtype=torch.bool, device=thing.device)
    masks[mask_index[0], mask_index[1]] = True

    # every scene: one grown region over its instance voxels, which the mean shift splits by their embedding
    assert masks.shape == (6, len(thing))
    assert torch.equal(mask_index, masks.nonzero().t())
    assert batch.tolist() == [0, 1, 0, 0, 1, 1]
    assert not (masks & ~thing).any()
    for mask, scene in zip(masks, batch.tolist()):
        assert (output["batch"][mask] == scene).all()
    assert torch.equal(masks[:2].any(dim=0), thing)
    assert torch.equal(masks[2:].any(dim=0), thing)
    assert ((instances["scores"] > 0) & (instances["scores"] < 1)).all()
    assert (instances["labels"] == 0).all()
    assert instances["semantic_logits"] is output["semantic_logits"]


@_REQUIRES_SPCONV
@_REQUIRES_CUDA
@_REQUIRES_PYG_LIB
def test_decode_skips_the_mean_shift_of_scenes_with_few_instance_voxels() -> None:
    model = _model(cluster_radius=0.5, min_cluster_size=5, min_mean_shift_size=10_000)
    instances = model.decode(_output(model))
    assert instances["scores"].shape == (2,)
    assert instances["mask_index"][0].unique().tolist() == [0, 1]


@_REQUIRES_SPCONV
@_REQUIRES_CUDA
@_REQUIRES_PYG_LIB
def test_decode_without_instance_voxels() -> None:
    model = _model()
    output = _output(model)
    output["semantic_logits"] = torch.zeros_like(output["semantic_logits"])
    output["semantic_logits"][:, 0] = 1.0
    instances = model.decode(output)
    assert instances["mask_index"].shape == (2, 0)
    assert instances["scores"].shape == instances["batch"].shape == (0,)


@_REQUIRES_SPCONV
@_REQUIRES_CUDA
def test_score_is_invariant_to_the_proposal_order() -> None:
    model = _model()
    x, pos, pos_grid, batch = _voxels()
    with torch.no_grad():
        features = model(x, pos, pos_grid, batch)["features"]
        masks = torch.stack([batch == 0, batch == 1, pos[:, 0] > 1.0])
        scores = model.score(features, pos_grid, masks.nonzero().t(), 3)
        flipped = model.score(features, pos_grid, masks.flip(0).nonzero().t(), 3)
    torch.testing.assert_close(scores, flipped.flip(0), atol=1e-4, rtol=1e-4)
