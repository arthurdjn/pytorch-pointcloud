from typing import Any, Dict

import pytest
import torch
from torch import Tensor, nn

from torch_pointcloud.models.openscene import OpenSceneSegmentation
from torch_pointcloud.utils.imports import _CUDA_AVAILABLE, _SPCONV_AVAILABLE

# See: https://docs.pytest.org/en/stable/how-to/skipping.html#summary
pytestmark = [
    pytest.mark.skipif(not _CUDA_AVAILABLE, reason="CUDA is not available"),
    pytest.mark.skipif(not _SPCONV_AVAILABLE, reason="spconv is not installed"),
]
spconv = pytest.importorskip("spconv.pytorch")


@pytest.fixture
def data() -> Dict[str, Tensor]:
    torch.manual_seed(42)
    lengths = torch.tensor([256, 512])
    pos_grid = torch.randint(0, 64, (int(lengths.sum()), 3))
    x = torch.randn(int(lengths.sum()), 3)
    batch = torch.repeat_interleave(torch.arange(len(lengths)), lengths)
    return dict(
        x=x.cuda(),
        pos_grid=pos_grid.cuda(),
        batch=batch.cuda(),
    )


@pytest.fixture
def model() -> OpenSceneSegmentation:
    return OpenSceneSegmentation(
        in_channels=3,
        embed_dim=32,
        text_model="openai/clip-vit-base-patch32",
        stem_channels=8,
        encoder_channels=(8, 16, 32, 64),
        decoder_channels=(32, 32, 16, 16),
        encoder_depths=(1, 1, 1, 1),
        decoder_depths=(1, 1, 1, 1),
        stem_kernel_size=5,
        kernel_size=3,
        spatial_padding=64,
    ).cuda()


def test_openscene_forward(model: OpenSceneSegmentation, data: Dict[str, Tensor]) -> None:
    embeddings = model(data["x"], data["pos_grid"], data["batch"])
    assert embeddings.shape == (data["pos_grid"].shape[0], model.embed_dim)
    assert model.text_model == "openai/clip-vit-base-patch32"


def test_openscene_forward_without_features_feeds_ones(model: OpenSceneSegmentation, data: Dict[str, Tensor]) -> None:
    model.eval()
    ones = torch.ones(data["pos_grid"].shape[0], model.in_channels, device="cuda")
    with torch.no_grad():
        from_none = model(None, data["pos_grid"], data["batch"])
        from_ones = model(ones, data["pos_grid"], data["batch"])
    assert torch.equal(from_none, from_ones)


def test_openscene_forward_features_decoder_head(model: OpenSceneSegmentation, data: Dict[str, Tensor]) -> None:
    sparse_x, skips = model.forward_features(data["x"], data["pos_grid"], data["batch"], return_intermediates=True)
    assert len(skips) == 4
    assert [skip["x"].shape[1] for skip in skips] == [8, 8, 16, 32]
    sparse_x = model.forward_decoder(sparse_x, skips)
    embeddings = model.forward_head(sparse_x)
    assert embeddings.shape == (data["pos_grid"].shape[0], model.embed_dim)


def test_openscene_forward_head_pre_logits(model: OpenSceneSegmentation, data: Dict[str, Tensor]) -> None:
    sparse_x, skips = model.forward_features(data["x"], data["pos_grid"], data["batch"], return_intermediates=True)
    sparse_x = model.forward_decoder(sparse_x, skips)
    feats = model.forward_head(sparse_x, pre_logits=True)
    assert torch.equal(feats, sparse_x.features)
    assert feats.shape[1] == model.num_features == model.decoder_channels[-1]


def test_openscene_head_projects_into_the_text_space(model: OpenSceneSegmentation) -> None:
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
        OpenSceneSegmentation(in_channels=3, embed_dim=32, text_model="openai/clip-vit-base-patch32", **kwargs)
