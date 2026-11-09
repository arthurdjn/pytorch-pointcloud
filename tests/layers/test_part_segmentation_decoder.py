import pytest
import torch

from torch_pointcloud.layers import PartSegmentationDecoder
from torch_pointcloud.utils.imports import _PYG_LIB_AVAILABLE

pytestmark = pytest.mark.skipif(not _PYG_LIB_AVAILABLE, reason="pyg-lib is not installed")


def _inputs() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    x = torch.randn(2, 8, 24)
    center = torch.randn(2, 8, 3)
    pos = torch.randn(300, 3)
    batch = torch.cat([torch.zeros(200), torch.ones(100)]).long()
    category = torch.nn.functional.one_hot(torch.tensor([3, 7]), 16).float()
    return x, center, pos, batch, category


def test_part_segmentation_decoder_ragged_batch() -> None:
    decoder = PartSegmentationDecoder(in_channels=24, num_categories=16, propagation_channels=(32, 16)).eval()
    x, center, pos, batch, category = _inputs()
    out = decoder(x, center, pos, batch, category)
    assert out.shape == (300, decoder.out_channels)
    assert decoder.out_channels == 16 + 2 * 24 + 64
    first = decoder(x[:1], center[:1], pos[:200], batch[:200], category[:1])
    assert torch.allclose(out[:200], first, atol=1e-6)


def test_part_segmentation_decoder_global_descriptor_is_shared_by_the_shape() -> None:
    decoder = PartSegmentationDecoder(in_channels=24, num_categories=16, propagation_channels=(32, 16)).eval()
    x, center, pos, batch, category = _inputs()
    out = decoder(x, center, pos, batch, category)
    x_global = out[:, 16:]
    assert torch.equal(x_global[:, :24], x.amax(dim=1)[batch])
    assert torch.allclose(x_global[:, 24:48], x.mean(dim=1)[batch])
    assert torch.equal(x_global[:200], x_global[:1].expand(200, -1))
    assert not torch.equal(x_global[0], x_global[-1])


@pytest.mark.parametrize(
    "category",
    [
        pytest.param(torch.nn.functional.one_hot(torch.tensor([[3, 7]]), 16).float(), id="extra-dim"),
        pytest.param(torch.nn.functional.one_hot(torch.tensor([3, 7, 1]), 16).float(), id="more-rows-than-shapes"),
        pytest.param(torch.tensor([3, 7]), id="index"),
    ],
)
def test_part_segmentation_decoder_rejects_a_misshaped_category(category: torch.Tensor) -> None:
    decoder = PartSegmentationDecoder(in_channels=24, num_categories=16, propagation_channels=(32, 16)).eval()
    x, center, pos, batch, _ = _inputs()
    with pytest.raises(ValueError, match="category"):
        decoder(x, center, pos, batch, category)


def test_part_segmentation_decoder_bias_and_eps() -> None:
    biased = PartSegmentationDecoder(in_channels=24, num_categories=16, propagation_channels=(32, 16))
    unbiased = PartSegmentationDecoder(in_channels=24, num_categories=16, propagation_channels=(32, 16), bias=False)
    assert set(biased.state_dict()) - set(unbiased.state_dict()) == {
        "propagation.mlp.lins.0.bias",
        "propagation.mlp.lins.1.bias",
    }

    loose = PartSegmentationDecoder(in_channels=24, num_categories=16, propagation_channels=(32, 16), eps=1e-2)
    loose.load_state_dict(biased.state_dict())
    biased.eval()
    loose.eval()
    x, center, pos, batch, category = _inputs()
    pos[:8], pos[200:208] = center[0], center[1]
    with torch.no_grad():
        out = biased(x, center, pos, batch, category)
        out_loose = loose(x, center, pos, batch, category)
    assert not torch.allclose(out[:, :16], out_loose[:, :16])
    assert torch.equal(out[:, 16:], out_loose[:, 16:])
