from typing import Any, Dict, List

import pytest
import torch
from torch import Tensor

from torch_pointcloud.models import model_info
from torch_pointcloud.models.spformer import SPFormerInstanceSegmentation, SPFormerOutput, SPFormerQueryDecoder
from torch_pointcloud.utils.data import DataKeys, collate, select_inputs
from torch_pointcloud.utils.imports import _CUDA_AVAILABLE, _SPCONV_AVAILABLE

_REQUIRES_SPCONV = pytest.mark.skipif(not _SPCONV_AVAILABLE, reason="spconv is not installed")
_REQUIRES_CUDA = pytest.mark.skipif(not _CUDA_AVAILABLE, reason="CUDA is not available")


def _decoder() -> SPFormerQueryDecoder:
    torch.manual_seed(0)
    return SPFormerQueryDecoder(8, 3, num_queries=5, embed_dim=16, num_layers=2, num_heads=2, mlp_dim=32).eval()


def _model(**kwargs: Any) -> SPFormerInstanceSegmentation:
    torch.manual_seed(0)
    hparams: Dict[str, Any] = dict(
        channels=(8, 16),
        depths=1,
        spatial_padding=8,
        num_queries=4,
        embed_dim=16,
        num_layers=1,
        num_heads=2,
        mlp_dim=32,
        num_instances=5,
    )
    hparams.update(kwargs)
    return SPFormerInstanceSegmentation(6, 3, **hparams).eval()


def _scenes() -> List[Dict[str, Tensor]]:
    """Two voxelized scenes sharing the grid corner (7, 7, 7), so they get one spatial shape alone or collated."""
    torch.manual_seed(0)
    scenes = []
    for num_voxels, num_points in ((40, 70), (60, 90)):
        flat = torch.cat([torch.randperm(8**3 - 1)[: num_voxels - 1], torch.tensor([8**3 - 1])])
        scene = {
            DataKeys.X: torch.randn(num_voxels, 6),
            DataKeys.POS_GRID: torch.stack([flat // 64, flat // 8 % 8, flat % 8], dim=1),
            DataKeys.SUPERPOINT: torch.randint(0, 9, (num_points,)),
            DataKeys.INVERSE: torch.randint(0, num_voxels, (num_points,)),
        }
        scenes.append(scene)
    return scenes


def _inputs(scenes: List[Dict[str, Tensor]]) -> List[Tensor]:
    data = collate(scenes, batch_from=DataKeys.POS_GRID, cat_keys=[DataKeys.INVERSE])
    input_keys = model_info("spformer.scannet.jiahao-sun", task="instance-segmentation")["input_keys"]
    return [value.cuda() for value in select_inputs(data, input_keys)]


def test_query_decoder_predicts_after_every_layer() -> None:
    decoder = _decoder()
    x = torch.randn(11, 8)
    batch = torch.tensor([0] * 7 + [1] * 4)
    predictions = decoder(x, batch)
    assert len(predictions) == 3
    for prediction in predictions:
        assert prediction["class_logits"].shape == (2, 5, 4)
        assert prediction["scores"].shape == (2, 5)
        assert prediction["mask_logits"].shape == (5, 11)


def test_query_decoder_reset_classifier() -> None:
    decoder = _decoder()
    score_head = decoder.score_head
    decoder.reset_classifier(6)
    assert decoder.num_classes == 6
    assert decoder(torch.randn(4, 8), torch.zeros(4, dtype=torch.long))[-1]["class_logits"].shape == (1, 5, 7)
    assert decoder.score_head is score_head


def test_query_decoder_scenes_are_independent() -> None:
    """A scene's predictions do not depend on the other scenes of the batch."""
    decoder = _decoder()
    x = torch.randn(11, 8)
    batch = torch.tensor([0] * 7 + [1] * 4)
    with torch.no_grad():
        together = decoder(x, batch)[-1]
        alone = decoder(x[7:], torch.zeros(4, dtype=torch.long))[-1]
    torch.testing.assert_close(together["class_logits"][1:], alone["class_logits"])
    torch.testing.assert_close(together["scores"][1:], alone["scores"])
    torch.testing.assert_close(together["mask_logits"][:, 7:], alone["mask_logits"])


@_REQUIRES_SPCONV
def test_decode_keeps_the_best_pairs_of_every_scene() -> None:
    """Hand-built output: 2 scenes of 3 and 2 superpoints, 4 queries, 3 classes."""
    model = _model()
    class_logits = torch.full((2, 4, 4), -10.0)
    class_logits[0, 0, 1] = 10.0  # scene 0, query 0: class 1
    class_logits[1, 2, 2] = 10.0  # scene 1, query 2: class 2
    mask_logits = torch.full((4, 5), -10.0)
    mask_logits[0, :2] = 10.0  # scene 0, query 0 covers its superpoints 0 and 1
    mask_logits[2, 4] = 10.0  # scene 1, query 2 covers its superpoint 1 (packed index 4)
    output: SPFormerOutput = {
        "class_logits": class_logits,
        "scores": torch.full((2, 4), 0.5),
        "mask_logits": mask_logits,
        "superpoint": torch.tensor([0, 0, 1, 2, 3, 4, 4]),
        "superpoint_batch": torch.tensor([0, 0, 0, 1, 1]),
        "aux_outputs": [],
    }
    instances = model.decode(output)
    assert instances["masks"].shape == (10, 7)
    assert instances["batch"].tolist() == [0] * 5 + [1] * 5
    assert instances["labels"][[0, 5]].tolist() == [1, 2]
    torch.testing.assert_close(instances["scores"][[0, 5]], torch.full((2,), 0.5), atol=1e-3, rtol=0)
    assert instances["masks"][0].tolist() == [True, True, True, False, False, False, False]
    assert instances["masks"][5].tolist() == [False, False, False, False, False, True, True]
    assert not instances["masks"][:5, 4:].any() and not instances["masks"][5:, :4].any()


@_REQUIRES_SPCONV
def test_reset_classifier_rebuilds_the_class_head_only() -> None:
    model = _model()
    mask_proj = model.head.mask_proj
    model.reset_classifier(7)
    assert model.num_classes == model.head.num_classes == 7
    prediction = model.head(torch.randn(3, model.num_features), torch.zeros(3, dtype=torch.long))[-1]
    assert prediction["class_logits"].shape == (1, 4, 8)
    assert model.head.mask_proj is mask_proj


@_REQUIRES_SPCONV
@_REQUIRES_CUDA
def test_forward_and_decode() -> None:
    model = _model().cuda()
    points = [70, 90]
    with torch.no_grad():
        output = model(*_inputs(_scenes()))
        instances = model.decode(output)

    num_superpoints = output["superpoint_batch"].shape[0]
    assert output["class_logits"].shape == (2, 4, 4)
    assert output["mask_logits"].shape == (4, num_superpoints)
    assert output["superpoint"].shape == (sum(points),)
    assert len(output["aux_outputs"]) == 1
    assert instances["masks"].shape == (10, sum(points))
    assert not instances["masks"][instances["batch"] == 0, points[0] :].any()


@_REQUIRES_SPCONV
@_REQUIRES_CUDA
def test_forward_scenes_are_independent() -> None:
    """Collated through the registered input keys, a scene predicts what it predicts alone."""
    model = _model().cuda()
    scenes = _scenes()
    with torch.no_grad():
        together = model(*_inputs(scenes))
        alone = model(*_inputs(scenes[1:]))
    num_superpoints = int((together["superpoint_batch"] == 0).sum())
    torch.testing.assert_close(together["class_logits"][1:], alone["class_logits"], atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(together["scores"][1:], alone["scores"], atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(together["mask_logits"][:, num_superpoints:], alone["mask_logits"], atol=1e-4, rtol=1e-4)
