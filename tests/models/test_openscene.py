from typing import Any, Dict

import pytest
import torch
from torch import Tensor, nn

from torch_pointcloud.models import model_info
from torch_pointcloud.models.openscene import OpenSceneOpenVocabularySegmentation
from torch_pointcloud.utils.data import DataKeys
from torch_pointcloud.utils.imports import _CUDA_AVAILABLE, _PYG_LIB_AVAILABLE, _SPCONV_AVAILABLE

_REQUIRES_CUDA = pytest.mark.skipif(not _CUDA_AVAILABLE, reason="CUDA is not available")
_REQUIRES_SPCONV = pytest.mark.skipif(not _SPCONV_AVAILABLE, reason="spconv is not installed")
_REQUIRES_PYG_LIB = pytest.mark.skipif(not _PYG_LIB_AVAILABLE, reason="pyg-lib is not installed")

REGISTERED_MODELS = [
    pytest.param("openscene-openseg.scannet20.songyou-peng", id="openseg-scannet20"),
    pytest.param("openscene-lseg.scannet20.songyou-peng", id="lseg-scannet20"),
    pytest.param("openscene-openseg.nuscenes.songyou-peng", id="openseg-nuscenes"),
]


@pytest.fixture
def data() -> Dict[str, Tensor]:
    torch.manual_seed(42)
    lengths = torch.tensor([256, 512])
    flat = torch.randperm(64**3)[: int(lengths.sum())]  # unique voxels, so submanifold convs are well defined
    pos_grid = torch.stack([flat // 64**2, (flat // 64) % 64, flat % 64], dim=1)
    x = torch.randn(int(lengths.sum()), 3)
    batch = torch.repeat_interleave(torch.arange(len(lengths)), lengths)
    return dict(
        x=x.cuda(),
        pos_grid=pos_grid.cuda(),
        batch=batch.cuda(),
    )


@pytest.fixture
def model() -> OpenSceneOpenVocabularySegmentation:
    return OpenSceneOpenVocabularySegmentation(
        in_channels=3,
        embed_dim=32,
        stem_channels=8,
        encoder_channels=(8, 16, 32, 64),
        decoder_channels=(32, 32, 16, 16),
        encoder_depths=(1, 1, 1, 1),
        decoder_depths=(1, 1, 1, 1),
        stem_kernel_size=5,
        kernel_size=3,
        spatial_padding=64,
    )


@_REQUIRES_CUDA
@_REQUIRES_SPCONV
def test_openscene_forward_returns_unit_embeddings(
    model: OpenSceneOpenVocabularySegmentation, data: Dict[str, Tensor]
) -> None:
    model.cuda()
    embeddings = model(data["x"], data["pos_grid"], data["batch"])
    assert embeddings.shape == (data["pos_grid"].shape[0], model.embed_dim)
    assert torch.allclose(embeddings.norm(dim=1), torch.ones(embeddings.shape[0], device="cuda"), atol=1e-5)


@_REQUIRES_CUDA
@_REQUIRES_SPCONV
def test_openscene_forward_without_features_feeds_ones(
    model: OpenSceneOpenVocabularySegmentation, data: Dict[str, Tensor]
) -> None:
    model.cuda()
    model.eval()
    ones = torch.ones(data["pos_grid"].shape[0], model.in_channels, device="cuda")
    with torch.no_grad():
        from_none = model(None, data["pos_grid"], data["batch"])
        from_ones = model(ones, data["pos_grid"], data["batch"])
    assert torch.equal(from_none, from_ones)


@_REQUIRES_SPCONV
def test_openscene_stride_two_convolutions_keep_the_stage_width(model: OpenSceneOpenVocabularySegmentation) -> None:
    state = model.state_dict()
    widths = [model.stem_channels, *model.encoder_channels[:-1]]
    for s, (width, channels) in enumerate(zip(widths, model.encoder_channels)):
        assert state[f"encoder.down.{s}.0.weight"].shape == (width, 2, 2, 2, width)  # (out, 2, 2, 2, in)
        assert state[f"encoder.enc.{s}.block0.conv1.weight"].shape == (channels, 3, 3, 3, width)


@_REQUIRES_CUDA
@_REQUIRES_SPCONV
def test_openscene_forward_features_decoder_head(
    model: OpenSceneOpenVocabularySegmentation, data: Dict[str, Tensor]
) -> None:
    model.cuda()
    sparse_x, skips = model.forward_features(data["x"], data["pos_grid"], data["batch"], return_intermediates=True)
    assert len(skips) == 4
    assert [skip["x"].shape[1] for skip in skips] == [8, 8, 16, 32]
    sparse_x = model.forward_decoder(sparse_x, skips)
    embeddings = model.forward_head(sparse_x)
    assert embeddings.shape == (data["pos_grid"].shape[0], model.embed_dim)


@_REQUIRES_CUDA
@_REQUIRES_SPCONV
def test_openscene_forward_head_pre_logits(model: OpenSceneOpenVocabularySegmentation, data: Dict[str, Tensor]) -> None:
    model.cuda()
    sparse_x, skips = model.forward_features(data["x"], data["pos_grid"], data["batch"], return_intermediates=True)
    sparse_x = model.forward_decoder(sparse_x, skips)
    feats = model.forward_head(sparse_x, pre_logits=True)
    assert torch.equal(feats, sparse_x.features)
    assert feats.shape[1] == model.num_features == model.decoder_channels[-1]


@_REQUIRES_SPCONV
def test_openscene_head_projects_into_the_text_space(model: OpenSceneOpenVocabularySegmentation) -> None:
    assert isinstance(model.head, nn.Linear)
    assert model.head.bias is None
    assert (model.head.in_features, model.head.out_features) == (model.num_features, model.embed_dim)


@pytest.mark.parametrize(
    "kwargs",
    [
        pytest.param(dict(decoder_channels=(32, 32, 16)), id="decoder_channels"),
        pytest.param(dict(encoder_depths=(1, 1, 1)), id="encoder_depths"),
        pytest.param(dict(decoder_depths=(1, 1, 1)), id="decoder_depths"),
    ],
)
def test_openscene_rejects_mismatched_stage_counts(kwargs: Dict[str, Any]) -> None:
    with pytest.raises(ValueError, match="must have the same length"):
        OpenSceneOpenVocabularySegmentation(in_channels=3, embed_dim=32, **kwargs)


@_REQUIRES_PYG_LIB
@pytest.mark.parametrize("name", REGISTERED_MODELS)
def test_openscene_registered_transform_runs_without_labels(name: str) -> None:
    torch.manual_seed(0)
    pos = torch.rand(2000, 3) * 4.0
    transform = model_info(name, task="open-vocabulary-segmentation")["transform"]
    assert transform is not None
    data = transform({DataKeys.POS: pos})
    num_voxels = data[DataKeys.POS_GRID].shape[0]
    assert DataKeys.SEGMENT not in data and DataKeys.ORIGIN_SEGMENT not in data
    assert torch.equal(data[DataKeys.ORIGIN_POS], pos)
    assert data[DataKeys.INVERSE].shape == (2000,)
    assert int(data[DataKeys.INVERSE].max()) == num_voxels - 1
